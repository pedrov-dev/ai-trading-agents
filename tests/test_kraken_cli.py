import json
from datetime import UTC, datetime
from pathlib import Path

from agent.signals import TradeIntent
from execution.kraken_cli import (
    CommandRunResult,
    KrakenCLIConfig,
    KrakenCLIExecutor,
    _build_private_order_payload,
    _default_cli_executable,
)
from execution.kraken_cli import (
    main as kraken_cli_main,
)
from execution.orders import OrderStatus


def _make_intent(*, side: str = "buy") -> TradeIntent:
    return TradeIntent(
        symbol_id="btc_usd",
        side=side,
        notional_usd=250.0,
        quantity=0.00367647,
        current_price=68_000.0,
        score=0.91,
        rationale=("ETF approval momentum", "Risk checks cleared"),
        generated_at=datetime(2026, 4, 3, 12, 0, tzinfo=UTC),
    )


def test_default_cli_executable_prefers_venv_script_next_to_sys_executable(
    tmp_path,
    monkeypatch,
) -> None:
    venv_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True, exist_ok=True)
    venv_python.write_text("", encoding="utf-8")
    local_cli = venv_python.with_name("kraken-cli.exe")
    local_cli.write_text("", encoding="utf-8")

    resolved_python = tmp_path / "base-python" / "python.exe"
    resolved_python.parent.mkdir(parents=True, exist_ok=True)
    resolved_python.write_text("", encoding="utf-8")

    monkeypatch.setattr("execution.kraken_cli.sys.executable", str(venv_python))

    original_resolve = Path.resolve

    def fake_resolve(self: Path, *args: object, **kwargs: object) -> Path:
        if self == venv_python:
            return resolved_python
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", fake_resolve, raising=False)

    assert _default_cli_executable() == str(local_cli)


def test_config_from_env_prefers_local_console_script_for_generic_kraken_cli_value(
    tmp_path,
    monkeypatch,
) -> None:
    venv_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True, exist_ok=True)
    venv_python.write_text("", encoding="utf-8")
    local_cli = venv_python.with_name("kraken-cli.exe")
    local_cli.write_text("", encoding="utf-8")

    monkeypatch.setattr("execution.kraken_cli.sys.executable", str(venv_python))

    config = KrakenCLIConfig.from_env({"KRAKEN_CLI_EXECUTABLE": "kraken-cli"})

    assert config.executable == str(local_cli)


def test_build_command_translates_trade_intent_to_kraken_cli_args(tmp_path) -> None:
    executor = KrakenCLIExecutor(
        config=KrakenCLIConfig(
            executable="kraken-cli",
            dry_run=True,
            audit_log_path=tmp_path / "orders.jsonl",
        )
    )

    command = executor.build_command(_make_intent())

    assert command[:2] == ("kraken-cli", "add-order")
    assert command[command.index("--pair") + 1] == "XBT/USD"
    assert command[command.index("--side") + 1] == "buy"
    assert command[command.index("--volume") + 1] == "0.00367647"
    assert "--validate" in command


def test_submit_trade_intent_dry_run_records_request_and_simulated_fill(tmp_path) -> None:
    audit_path = tmp_path / "orders.jsonl"
    executor = KrakenCLIExecutor(
        config=KrakenCLIConfig(
            executable="kraken-cli",
            dry_run=True,
            audit_log_path=audit_path,
        )
    )

    result = executor.submit_trade_intent(_make_intent())

    assert result.status == OrderStatus.SIMULATED
    assert result.fill is not None
    assert result.retry_count == 0
    assert result.attempts[0].status == OrderStatus.SIMULATED

    entries = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert [entry["event"] for entry in entries] == ["order_requested", "order_simulated"]


def test_submit_trade_intent_validate_only_hits_kraken_without_live_fill(tmp_path) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_runner(command: tuple[str, ...], timeout_seconds: int) -> CommandRunResult:
        assert timeout_seconds == 15
        calls.append(command)
        return CommandRunResult(exit_code=0, stdout='{"result": "validated"}', stderr="")

    executor = KrakenCLIExecutor(
        config=KrakenCLIConfig(
            executable="kraken-cli",
            dry_run=False,
            live_enabled=True,
            validate_only=True,
            timeout_seconds=15,
            audit_log_path=tmp_path / "orders.jsonl",
        ),
        runner=fake_runner,
    )

    result = executor.submit_trade_intent(_make_intent())

    assert len(calls) == 1
    assert "--validate" in calls[0]
    assert result.status == OrderStatus.VALIDATED
    assert result.fill is None
    assert result.is_successful is True


def test_submit_trade_intent_retries_transient_cli_failures(tmp_path) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_runner(command: tuple[str, ...], timeout_seconds: int) -> CommandRunResult:
        assert timeout_seconds == 15
        calls.append(command)
        return CommandRunResult(
            exit_code=75,
            stdout="",
            stderr="temporary network timeout from kraken",
        )

    executor = KrakenCLIExecutor(
        config=KrakenCLIConfig(
            executable="kraken-cli",
            dry_run=False,
            live_enabled=True,
            max_retries=2,
            timeout_seconds=15,
            audit_log_path=tmp_path / "orders.jsonl",
        ),
        runner=fake_runner,
    )

    result = executor.submit_trade_intent(_make_intent(side="sell"))

    assert result.status == OrderStatus.FAILED
    assert result.failure is not None
    assert result.failure.retryable is True
    assert result.retry_count == 2
    assert len(calls) == 3
    assert len(result.attempts) == 3
    assert all(attempt.retryable for attempt in result.attempts)


def test_kraken_cli_main_validate_mode_requires_credentials(capsys, monkeypatch) -> None:
    monkeypatch.delenv("KRAKEN_API_KEY", raising=False)
    monkeypatch.delenv("KRAKEN_API_SECRET", raising=False)

    exit_code = kraken_cli_main(
        [
            "add-order",
            "--pair",
            "XBT/USD",
            "--side",
            "buy",
            "--type",
            "market",
            "--volume",
            "0.01",
            "--clordid",
            "demo-validate-1",
            "--validate",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "KRAKEN_API_KEY" in captured.err
    assert "KRAKEN_API_SECRET" in captured.err


def test_kraken_cli_main_live_submit_requires_api_credentials(capsys, monkeypatch) -> None:
    monkeypatch.delenv("KRAKEN_API_KEY", raising=False)
    monkeypatch.delenv("KRAKEN_API_SECRET", raising=False)

    exit_code = kraken_cli_main(
        [
            "add-order",
            "--pair",
            "XBT/USD",
            "--side",
            "buy",
            "--type",
            "market",
            "--volume",
            "0.01",
            "--clordid",
            "demo-live-1",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "KRAKEN_API_KEY" in captured.err
    assert "KRAKEN_API_SECRET" in captured.err


def test_build_private_order_payload_normalizes_slash_pair_for_private_api() -> None:
    args = type(
        "Args",
        (),
        {
            "pair": "XBT/USD",
            "side": "buy",
            "order_type": "market",
            "volume": "0.01",
            "validate": True,
            "price": None,
        },
    )()

    payload = _build_private_order_payload(args)

    assert payload["pair"] == "XBTUSD"


def test_executor_default_runner_receives_runtime_env(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_command(
        command: tuple[str, ...],
        timeout_seconds: int,
        *,
        env: dict[str, str] | None = None,
    ) -> CommandRunResult:
        captured["command"] = command
        captured["timeout_seconds"] = timeout_seconds
        captured["env"] = env
        return CommandRunResult(exit_code=0, stdout='{"result": "validated"}', stderr="")

    monkeypatch.setattr("execution.kraken_cli._run_command", fake_run_command)

    executor = KrakenCLIExecutor(
        config=KrakenCLIConfig(
            executable="kraken-cli",
            dry_run=False,
            live_enabled=True,
            validate_only=True,
            timeout_seconds=15,
            audit_log_path=tmp_path / "orders.jsonl",
        ),
        env={
            "KRAKEN_API_KEY": "demo-key",
            "KRAKEN_API_SECRET": "demo-secret",
        },
    )

    result = executor.submit_trade_intent(_make_intent())

    assert result.status == OrderStatus.VALIDATED
    assert captured["timeout_seconds"] == 15
    assert captured["env"] is not None
    assert captured["env"]["KRAKEN_API_KEY"] == "demo-key"
    assert captured["env"]["KRAKEN_API_SECRET"] == "demo-secret"
