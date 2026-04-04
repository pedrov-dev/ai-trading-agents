import json
from datetime import UTC, datetime

from agent.signals import TradeIntent
from execution.kraken_cli import CommandRunResult, KrakenCLIConfig, KrakenCLIExecutor
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
