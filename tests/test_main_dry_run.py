import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent.portfolio import LocalPortfolioStateProvider
from agent.risk import RiskCheckResult, RiskViolation
from agent.signals import TradeIntent
from detection.event_detection_postgres import PostgresEventDetectionRepository
from execution.kraken_cli import CommandRunResult, KrakenCLIConfig
from identity.erc8004_registry import SepoliaContractsConfig
from ingestion.prices_config import PRICE_SYMBOLS
from ingestion.prices_ingestion import PriceQuote
from ingestion.rss_config import FeedSource
from main import (
    build_local_demo_app,
    build_runtime_preflight,
    reset_runtime_state,
    validate_runtime_requirements,
)
from storage.raw_postgres import PostgresIngestionRunsRepository, PostgresRawEventsRepository
from tests.storage_fakes import (
    StubEventDetectionRepository,
    StubIngestionRunsRepository,
    StubObjectStore,
    StubRawEventsRepository,
)

BTC_SYMBOL = next(symbol for symbol in PRICE_SYMBOLS if symbol.symbol_id == "btc_usd")


def _storage_overrides(tmp_path: Path) -> dict[str, object]:
    return {
        "runs_repository": StubIngestionRunsRepository(),
        "raw_events_repository": StubRawEventsRepository(),
        "event_repository": StubEventDetectionRepository(),
        "object_store": StubObjectStore(tmp_path / "artifacts" / "raw_payloads"),
    }


def test_local_portfolio_provider_keeps_equity_stable_when_opening_short() -> None:
    provider = LocalPortfolioStateProvider(starting_equity=10_000.0)

    provider.record_fill(
        symbol_id="btc_usd",
        side="sell",
        quantity=0.01,
        price=50_000.0,
    )
    snapshot = provider.get_portfolio_snapshot()

    assert snapshot.cash_usd == 10_500.0
    assert snapshot.total_equity == 10_000.0
    assert snapshot.positions[0].side == "short"


def test_local_portfolio_provider_realizes_pnl_when_closing_a_long() -> None:
    provider = LocalPortfolioStateProvider(starting_equity=10_000.0)

    provider.record_fill(
        symbol_id="btc_usd",
        side="buy",
        quantity=0.01,
        price=50_000.0,
    )
    provider.record_fill(
        symbol_id="btc_usd",
        side="sell",
        quantity=0.01,
        price=52_000.0,
    )
    snapshot = provider.get_portfolio_snapshot()

    assert snapshot.cash_usd == 10_020.0
    assert snapshot.total_equity == 10_020.0
    assert snapshot.realized_pnl_today == 20.0
    assert snapshot.open_position_count() == 0


def test_local_portfolio_provider_tracks_partial_close_losses() -> None:
    provider = LocalPortfolioStateProvider(starting_equity=10_000.0)

    provider.record_fill(
        symbol_id="btc_usd",
        side="buy",
        quantity=0.02,
        price=50_000.0,
    )
    provider.record_fill(
        symbol_id="btc_usd",
        side="sell",
        quantity=0.01,
        price=49_000.0,
    )
    snapshot = provider.get_portfolio_snapshot()

    assert snapshot.cash_usd == 9_490.0
    assert snapshot.total_equity == 9_990.0
    assert snapshot.realized_pnl_today == -10.0
    assert snapshot.open_position_count() == 1
    assert snapshot.positions[0].side == "long"
    assert snapshot.positions[0].quantity == 0.01
    assert snapshot.consecutive_losses == 1
    assert snapshot.last_loss_at is not None


def test_kraken_paper_app_closes_open_position_on_take_profit(tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_runner(command: tuple[str, ...], timeout_seconds: int) -> CommandRunResult:
        assert timeout_seconds == 15
        calls.append(command)
        return CommandRunResult(
            exit_code=0,
            stdout='{"status": "validated", "validated": true}',
            stderr="",
        )

    app = build_local_demo_app(
        base_dir=tmp_path,
        trading_mode="paper",
        execution_runner=fake_runner,
        env={
            "KRAKEN_API_KEY": "demo-key",
            "KRAKEN_API_SECRET": "demo-secret",
        },
        execution_config=KrakenCLIConfig(
            executable="kraken-cli",
            dry_run=False,
            live_enabled=True,
            validate_only=True,
            audit_log_path=tmp_path / "artifacts" / "orders_audit.jsonl",
        ),
        **_storage_overrides(tmp_path),
    )
    app._portfolio_provider.record_fill(
        symbol_id="btc_usd",
        side="buy",
        quantity=0.01,
        price=50_000.0,
        filled_at=datetime(2026, 4, 3, 12, 0, tzinfo=UTC),
    )
    app._latest_quotes = [
        PriceQuote(
            symbol_id="btc_usd",
            current=51_500.0,
            open=50_100.0,
            high=51_600.0,
            low=49_900.0,
            prev_close=50_050.0,
            timestamp=1712100000,
            asset_class="spot",
        )
    ]

    result = app.execute_trade_cycle(classification_count=0)

    assert len(result.trade_intents) == 1
    assert result.trade_intents[0].side == "sell"
    assert len(result.execution_results) == 1
    assert result.portfolio.open_position_count() == 0
    assert result.portfolio.realized_pnl_today == 15.0
    assert "--side" in calls[0]
    assert "sell" in calls[0]


def test_kraken_paper_app_runs_end_to_end_and_writes_demo_artifacts(tmp_path: Path) -> None:
    def fake_parse_feed(_url: str) -> dict[str, object]:
        return {
            "entries": [
                {
                    "title": "SEC approved the bitcoin ETF after exchange review",
                    "link": "https://example.test/bitcoin-etf-approved",
                    "published": "2026-04-03T12:00:00+00:00",
                }
            ]
        }

    def fake_http_get(url: str, params: dict[str, str]) -> dict[str, object]:
        if url.endswith("/Ticker"):
            assert params == {"pair": BTC_SYMBOL.ticker}
            return {
                "error": [],
                "result": {
                    BTC_SYMBOL.ticker: {
                        "c": ["68000.0", "1"],
                        "o": "66000.0",
                        "h": ["68500.0", "68500.0"],
                        "l": ["65500.0", "65500.0"],
                        "p": ["67000.0", "67000.0"],
                    }
                },
            }
        raise AssertionError(f"Unexpected URL called: {url}")

    calls: list[tuple[str, ...]] = []

    def fake_runner(command: tuple[str, ...], timeout_seconds: int) -> CommandRunResult:
        assert timeout_seconds == 15
        calls.append(command)
        return CommandRunResult(
            exit_code=0,
            stdout='{"status": "validated", "validated": true}',
            stderr="",
        )

    app = build_local_demo_app(
        base_dir=tmp_path,
        feed_groups={
            "market_news": [FeedSource(source_id="demo_feed", url="https://example.test/rss")]
        },
        symbols=[BTC_SYMBOL],
        parse_feed=fake_parse_feed,
        http_get=fake_http_get,
        env={
            "KRAKEN_API_KEY": "demo-key",
            "KRAKEN_API_SECRET": "demo-secret",
        },
        trading_mode="paper",
        execution_runner=fake_runner,
        execution_config=KrakenCLIConfig(
            executable="kraken-cli",
            dry_run=False,
            live_enabled=True,
            validate_only=True,
            audit_log_path=tmp_path / "artifacts" / "orders_audit.jsonl",
        ),
        **_storage_overrides(tmp_path),
    )

    result = app.run_cycle(feed_group="market_news")

    assert result.rss_result.inserted_count == 1
    assert result.prices_result.inserted_count == 1
    assert result.classification_count >= 1
    assert len(result.detected_events) >= 1
    assert len(result.trade_intents) == 1
    assert len(result.execution_results) == 1
    assert all(item.status.value == "validated" for item in result.execution_results)
    assert len(calls) == 1
    assert all("--validate" in call for call in calls)
    assert result.artifact_count >= 4
    assert result.checkpoint_count >= 4
    assert result.audit_summary.total_events >= 2
    assert result.execution_results[0].fill is not None
    assert result.portfolio.open_position_count() == 1
    assert {position.symbol_id for position in result.portfolio.positions} == {"btc_usd"}
    assert {position.exit_horizon_label for position in result.portfolio.positions} == {"5m"}

    assert (tmp_path / "artifacts" / "orders_audit.jsonl").exists()
    assert (tmp_path / "artifacts" / "trading_journal.jsonl").exists()
    assert (tmp_path / "artifacts" / "validation_artifacts.jsonl").exists()
    assert (tmp_path / "artifacts" / "validation_checkpoints.jsonl").exists()
    assert (tmp_path / "artifacts" / "run_summary.json").exists()


def test_kraken_paper_app_records_signal_outcomes_by_horizon_on_exit(tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_runner(command: tuple[str, ...], timeout_seconds: int) -> CommandRunResult:
        assert timeout_seconds == 15
        calls.append(command)
        return CommandRunResult(
            exit_code=0,
            stdout='{"status": "validated", "validated": true}',
            stderr="",
        )

    app = build_local_demo_app(
        base_dir=tmp_path,
        trading_mode="paper",
        execution_runner=fake_runner,
        env={
            "KRAKEN_API_KEY": "demo-key",
            "KRAKEN_API_SECRET": "demo-secret",
        },
        execution_config=KrakenCLIConfig(
            executable="kraken-cli",
            dry_run=False,
            live_enabled=True,
            validate_only=True,
            audit_log_path=tmp_path / "artifacts" / "orders_audit.jsonl",
        ),
        **_storage_overrides(tmp_path),
    )
    app._portfolio_provider.record_fill(
        symbol_id="btc_usd",
        side="buy",
        quantity=0.01,
        price=50_000.0,
        filled_at=datetime(2026, 4, 3, 12, 0, tzinfo=UTC),
        position_id="pos-5m",
        source_signal_id="signal-123",
        raw_event_id="evt-123",
        event_type="ETF_APPROVAL",
        exit_horizon_label="5m",
        max_hold_minutes=5,
        exit_due_at=datetime(2026, 4, 3, 12, 5, tzinfo=UTC),
    )
    app._latest_quotes = [
        PriceQuote(
            symbol_id="btc_usd",
            current=50_500.0,
            open=50_100.0,
            high=50_600.0,
            low=49_900.0,
            prev_close=50_050.0,
            timestamp=1712100000,
            asset_class="spot",
        )
    ]

    result = app.execute_trade_cycle(classification_count=0)
    signal_outcomes = [
        artifact for artifact in result.artifacts if artifact.kind.value == "signal_outcome"
    ]

    assert len(result.trade_intents) == 1
    assert len(result.execution_results) == 1
    assert len(signal_outcomes) == 1
    assert signal_outcomes[0].payload["exit_horizon_label"] == "5m"
    assert signal_outcomes[0].refs["raw_event_id"] == "evt-123"
    assert signal_outcomes[0].refs["signal_id"] == "signal-123"
    assert result.portfolio.open_position_count() == 0
    assert any("sell" in command for command in calls)


def test_run_summary_includes_confidence_calibration_for_resolved_trades(tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_runner(command: tuple[str, ...], timeout_seconds: int) -> CommandRunResult:
        assert timeout_seconds == 15
        calls.append(command)
        return CommandRunResult(
            exit_code=0,
            stdout='{"status": "validated", "validated": true}',
            stderr="",
        )

    app = build_local_demo_app(
        base_dir=tmp_path,
        trading_mode="paper",
        execution_runner=fake_runner,
        env={
            "KRAKEN_API_KEY": "demo-key",
            "KRAKEN_API_SECRET": "demo-secret",
        },
        execution_config=KrakenCLIConfig(
            executable="kraken-cli",
            dry_run=False,
            live_enabled=True,
            validate_only=True,
            audit_log_path=tmp_path / "artifacts" / "orders_audit.jsonl",
        ),
        **_storage_overrides(tmp_path),
    )
    app._portfolio_provider.record_fill(
        symbol_id="btc_usd",
        side="buy",
        quantity=0.01,
        price=50_000.0,
        filled_at=datetime(2026, 4, 3, 12, 0, tzinfo=UTC),
        position_id="pos-30m",
        source_signal_id="signal-456",
        raw_event_id="evt-456",
        event_type="ETF_APPROVAL",
        exit_horizon_label="30m",
        max_hold_minutes=30,
        exit_due_at=datetime(2026, 4, 3, 12, 30, tzinfo=UTC),
        confidence_score=0.92,
        expected_move="up",
    )
    app._latest_quotes = [
        PriceQuote(
            symbol_id="btc_usd",
            current=51_500.0,
            open=50_100.0,
            high=51_600.0,
            low=49_900.0,
            prev_close=50_050.0,
            timestamp=1712100000,
            asset_class="spot",
        )
    ]

    result = app.execute_trade_cycle(classification_count=0)
    summary = json.loads((tmp_path / "artifacts" / "run_summary.json").read_text(encoding="utf-8"))
    journal_rows = [
        json.loads(line)
        for line in (tmp_path / "artifacts" / "trading_journal.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    assert len(result.trade_intents) == 1
    assert len(result.execution_results) == 1
    assert result.calibration_summary.resolved_prediction_count == 1
    assert result.calibration_summary.hit_rate == 1.0
    assert summary["calibration_summary"]["resolved_prediction_count"] == 1
    assert summary["calibration_summary"]["brier_score"] >= 0.0
    assert journal_rows[-1]["expected_move"] == "up"
    assert journal_rows[-1]["actual_move"] == "up"
    assert journal_rows[-1]["prediction_correct"] is True
    assert any("sell" in command for command in calls)


def test_run_cycle_writes_incremental_action_log_and_summary_state(tmp_path: Path) -> None:
    def fake_parse_feed(_url: str) -> dict[str, object]:
        return {
            "entries": [
                {
                    "title": "SEC approved the bitcoin ETF after exchange review",
                    "link": "https://example.test/bitcoin-etf-approved",
                    "published": "2026-04-03T12:00:00+00:00",
                }
            ]
        }

    def fake_http_get(url: str, params: dict[str, str]) -> dict[str, object]:
        if url.endswith("/Ticker"):
            assert params == {"pair": BTC_SYMBOL.ticker}
            return {
                "error": [],
                "result": {
                    BTC_SYMBOL.ticker: {
                        "c": ["68000.0", "1"],
                        "o": "66000.0",
                        "h": ["68500.0", "68500.0"],
                        "l": ["65500.0", "65500.0"],
                        "p": ["67000.0", "67000.0"],
                    }
                },
            }
        raise AssertionError(f"Unexpected URL called: {url}")

    def fake_runner(command: tuple[str, ...], timeout_seconds: int) -> CommandRunResult:
        assert timeout_seconds == 15
        return CommandRunResult(
            exit_code=0,
            stdout='{"status": "validated", "validated": true}',
            stderr="",
        )

    app = build_local_demo_app(
        base_dir=tmp_path,
        feed_groups={
            "market_news": [
                FeedSource(source_id="demo_feed", url="https://example.test/rss")
            ]
        },
        symbols=[BTC_SYMBOL],
        parse_feed=fake_parse_feed,
        http_get=fake_http_get,
        env={
            "KRAKEN_API_KEY": "demo-key",
            "KRAKEN_API_SECRET": "demo-secret",
        },
        trading_mode="paper",
        execution_runner=fake_runner,
        execution_config=KrakenCLIConfig(
            executable="kraken-cli",
            dry_run=False,
            live_enabled=True,
            validate_only=True,
            audit_log_path=tmp_path / "artifacts" / "orders_audit.jsonl",
        ),
        **_storage_overrides(tmp_path),
    )

    result = app.run_cycle(feed_group="market_news")

    activity_log_path = tmp_path / "artifacts" / "activity_log.jsonl"
    summary_path = tmp_path / "artifacts" / "run_summary.json"

    assert activity_log_path.exists()
    activity_records = [
        json.loads(line)
        for line in activity_log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert activity_records[0]["action"] == "cycle_started"
    assert activity_records[-1]["action"] == "cycle_completed"
    assert any(record["action"] == "artifact_recorded" for record in activity_records)
    assert all("affects" in record for record in activity_records)
    assert all("summary" in record for record in activity_records)
    assert all("payload" not in record for record in activity_records)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert summary["counts"]["artifacts"] == result.artifact_count
    assert summary["counts"]["checkpoints"] == result.checkpoint_count
    assert summary["last_action"]["action"] == "cycle_completed"
    assert "artifacts" not in summary
    assert "checkpoints" not in summary
    assert "recent_events" not in summary["audit_summary"]
    assert "rationale" not in summary["trade_intents"][0]
    assert "request" not in summary["execution_results"][0]
    assert "benchmark_summary" in summary["pnl_snapshot"]
    assert summary["pnl_snapshot"]["position_pnl"]["btc_usd"]["benchmark_comparisons"]["momentum"][
        "label"
    ] == "Momentum baseline"


def test_local_demo_app_restores_portfolio_from_trade_journal(tmp_path: Path) -> None:
    def fake_parse_feed(_url: str) -> dict[str, object]:
        return {
            "entries": [
                {
                    "title": "SEC approved the bitcoin ETF after exchange review",
                    "link": "https://example.test/bitcoin-etf-approved",
                    "published": "2026-04-03T12:00:00+00:00",
                }
            ]
        }

    def fake_http_get(url: str, params: dict[str, str]) -> dict[str, object]:
        if url.endswith("/Ticker"):
            assert params == {"pair": BTC_SYMBOL.ticker}
            return {
                "error": [],
                "result": {
                    BTC_SYMBOL.ticker: {
                        "c": ["68000.0", "1"],
                        "o": "66000.0",
                        "h": ["68500.0", "68500.0"],
                        "l": ["65500.0", "65500.0"],
                        "p": ["67000.0", "67000.0"],
                    }
                },
            }
        raise AssertionError(f"Unexpected URL called: {url}")

    def fake_runner(command: tuple[str, ...], timeout_seconds: int) -> CommandRunResult:
        assert timeout_seconds == 15
        return CommandRunResult(
            exit_code=0,
            stdout='{"status": "validated", "validated": true}',
            stderr="",
        )

    first_app = build_local_demo_app(
        base_dir=tmp_path,
        feed_groups={
            "market_news": [
                FeedSource(source_id="demo_feed", url="https://example.test/rss")
            ]
        },
        symbols=[BTC_SYMBOL],
        parse_feed=fake_parse_feed,
        http_get=fake_http_get,
        env={
            "KRAKEN_API_KEY": "demo-key",
            "KRAKEN_API_SECRET": "demo-secret",
        },
        trading_mode="paper",
        execution_runner=fake_runner,
        execution_config=KrakenCLIConfig(
            executable="kraken-cli",
            dry_run=False,
            live_enabled=True,
            validate_only=True,
            audit_log_path=tmp_path / "artifacts" / "orders_audit.jsonl",
        ),
        **_storage_overrides(tmp_path),
    )

    first_result = first_app.run_cycle(feed_group="market_news")
    restored_app = build_local_demo_app(
        base_dir=tmp_path,
        env={
            "KRAKEN_API_KEY": "demo-key",
            "KRAKEN_API_SECRET": "demo-secret",
        },
        trading_mode="paper",
        execution_runner=fake_runner,
        execution_config=KrakenCLIConfig(
            executable="kraken-cli",
            dry_run=False,
            live_enabled=True,
            validate_only=True,
            audit_log_path=tmp_path / "artifacts" / "orders_audit.jsonl",
        ),
        **_storage_overrides(tmp_path),
    )

    restored_portfolio = restored_app._portfolio_provider.get_portfolio_snapshot()

    assert first_result.portfolio.open_position_count() == 1
    assert restored_portfolio.open_position_count() == 1
    assert {position.symbol_id for position in restored_portfolio.positions} == {"btc_usd"}
    assert {position.exit_horizon_label for position in restored_portfolio.positions} == {"5m"}
    assert restored_portfolio.cash_usd == first_result.portfolio.cash_usd
    assert restored_portfolio.total_equity == first_result.portfolio.total_equity


def test_local_demo_app_uses_env_agent_profile(tmp_path: Path) -> None:
    app = build_local_demo_app(
        base_dir=tmp_path,
        env={
            "AGENT_DISPLAY_NAME": "Sepolia Momentum Bot",
            "AGENT_STRATEGY_NAME": "momentum_breakout_v2",
            "AGENT_OWNER": "pedrov-dev",
            "AGENT_URI": "https://example.test/agent.json",
            "AGENT_CAPABILITIES": "trading,eip712-signing,checkpoints",
        },
        **_storage_overrides(tmp_path),
    )

    assert app.identity.display_name == "Sepolia Momentum Bot"
    assert app.identity.strategy_name == "momentum_breakout_v2"
    assert app.identity.owner == "pedrov-dev"
    assert app.identity.metadata["agent_uri"] == "https://example.test/agent.json"


def test_local_demo_app_persists_agent_id_env_file(tmp_path: Path) -> None:
    app = build_local_demo_app(
        base_dir=tmp_path,
        identity_layer="erc8004",
        **_storage_overrides(tmp_path),
    )
    env_path = tmp_path / ".runtime.env"

    app.persist_agent_id(agent_id=77, env_path=env_path)

    contents = env_path.read_text(encoding="utf-8")
    assert "AGENT_ID=77" in contents
    assert "IDENTITY_LAYER=erc8004" in contents


def test_shared_contract_status_includes_balance_and_claim_state() -> None:
    app = build_local_demo_app(base_dir=Path("."), **_storage_overrides(Path(".")))
    app._shared_contract_config = SepoliaContractsConfig(agent_id=42)

    class _VaultClient:
        def has_claimed(self, agent_id: int) -> bool:
            assert agent_id == 42
            return True

        def get_balance(self, agent_id: int) -> int:
            assert agent_id == 42
            return 50_000_000_000_000_000

    app._vault_client = _VaultClient()
    app._identity = app._identity.__class__(
        agent_id="42",
        display_name=app.identity.display_name,
        strategy_name=app.identity.strategy_name,
        owner=app.identity.owner,
        exchange="sepolia",
        wallet_address="0x0000000000000000000000000000000000000042",
        metadata=app.identity.metadata,
        registered_at=app.identity.registered_at,
    )

    status = app.shared_contract_status()

    assert status["enabled"] is True
    assert status["agent_id"] == "42"
    assert status["has_claimed_allocation"] is True
    assert status["vault_balance_wei"] == 50_000_000_000_000_000


def test_build_local_demo_app_requires_external_storage_by_default(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Postgres storage is required"):
        build_local_demo_app(
            base_dir=tmp_path,
            env={
                "KRAKEN_API_KEY": "demo-key",
                "KRAKEN_API_SECRET": "demo-secret",
            },
        )


def test_build_local_demo_app_defaults_to_kraken_paper_mode(tmp_path: Path) -> None:
    app = build_local_demo_app(
        base_dir=tmp_path,
        env={
            "KRAKEN_CLI_TIMEOUT_SECONDS": "21",
            "KRAKEN_API_KEY": "demo-key",
            "KRAKEN_API_SECRET": "demo-secret",
        },
        **_storage_overrides(tmp_path),
    )

    assert app._executor._config.dry_run is False
    assert app._executor._config.live_enabled is True
    assert app._executor._config.validate_only is True
    assert app._executor._config.timeout_seconds == 21
    assert app._executor._config.audit_log_path == tmp_path / "artifacts" / "orders_audit.jsonl"


def test_build_local_demo_app_uses_postgres_repositories_when_database_configured(
    tmp_path: Path,
) -> None:
    app = build_local_demo_app(
        base_dir=tmp_path,
        env={
            "POSTGRES_ENABLED": "true",
            "DATABASE_URL": "postgresql://demo:demo@localhost:5432/ai_trading",
            "CF_R2_BUCKET": "demo-bucket",
            "CF_R2_ENDPOINT": "https://example.r2.cloudflarestorage.com",
            "CF_R2_ACCESS_KEY": "demo-access",
            "CF_R2_SECRET_KEY": "demo-secret-key",
            "KRAKEN_API_KEY": "demo-key",
            "KRAKEN_API_SECRET": "demo-secret",
        },
    )

    assert isinstance(app._runs_repository, PostgresIngestionRunsRepository)
    assert isinstance(app._raw_events_repository, PostgresRawEventsRepository)
    assert isinstance(app._event_repository, PostgresEventDetectionRepository)


def test_build_local_demo_app_supports_kraken_live_mode(tmp_path: Path) -> None:
    app = build_local_demo_app(
        base_dir=tmp_path,
        env={"KRAKEN_CLI_ALLOW_LIVE_SUBMIT": "true"},
        trading_mode="live",
        **_storage_overrides(tmp_path),
    )

    assert app._executor._config.dry_run is False
    assert app._executor._config.live_enabled is True
    assert app._executor._config.validate_only is False
    assert app._executor._config.audit_log_path == tmp_path / "artifacts" / "orders_audit.jsonl"


def test_validate_runtime_requirements_blocks_live_submit_without_opt_in(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="KRAKEN_CLI_ALLOW_LIVE_SUBMIT"):
        validate_runtime_requirements(
            trading_mode="live",
            identity_layer="none",
            base_dir=tmp_path,
            env={
                "POSTGRES_ENABLED": "true",
                "DATABASE_URL": "postgresql://demo:demo@localhost:5432/ai_trading",
                "CF_R2_BUCKET": "demo-bucket",
                "CF_R2_ENDPOINT": "https://example.r2.cloudflarestorage.com",
                "CF_R2_ACCESS_KEY": "demo-access",
                "CF_R2_SECRET_KEY": "demo-secret-key",
                "KRAKEN_API_KEY": "demo-key",
                "KRAKEN_API_SECRET": "demo-secret",
            },
        )


def test_validate_runtime_requirements_blocks_paper_without_kraken_credentials(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="KRAKEN_API_KEY"):
        validate_runtime_requirements(
            trading_mode="paper",
            identity_layer="none",
            base_dir=tmp_path,
            env={
                "POSTGRES_ENABLED": "true",
                "DATABASE_URL": "postgresql://demo:demo@localhost:5432/ai_trading",
                "CF_R2_BUCKET": "demo-bucket",
                "CF_R2_ENDPOINT": "https://example.r2.cloudflarestorage.com",
                "CF_R2_ACCESS_KEY": "demo-access",
                "CF_R2_SECRET_KEY": "demo-secret-key",
            },
        )


def test_validate_runtime_requirements_blocks_missing_erc8004_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="PRIVATE_KEY"):
        validate_runtime_requirements(
            trading_mode="paper",
            identity_layer="erc8004",
            base_dir=tmp_path,
            env={
                "POSTGRES_ENABLED": "true",
                "DATABASE_URL": "postgresql://demo:demo@localhost:5432/ai_trading",
                "CF_R2_BUCKET": "demo-bucket",
                "CF_R2_ENDPOINT": "https://example.r2.cloudflarestorage.com",
                "CF_R2_ACCESS_KEY": "demo-access",
                "CF_R2_SECRET_KEY": "demo-secret-key",
                "SEPOLIA_RPC_URL": "https://ethereum-sepolia-rpc.publicnode.com",
                "KRAKEN_API_KEY": "demo-key",
                "KRAKEN_API_SECRET": "demo-secret",
            },
            require_transaction_keys=True,
        )


def test_build_runtime_preflight_reports_storage_blockers_when_not_configured(
    tmp_path: Path,
) -> None:
    report = build_runtime_preflight(
        trading_mode="paper",
        identity_layer="none",
        base_dir=tmp_path,
        env={
            "KRAKEN_API_KEY": "demo-key",
            "KRAKEN_API_SECRET": "demo-secret",
        },
    )

    assert report["status"] == "error"
    assert any("Postgres storage is required" in issue for issue in report["issues"])
    assert any("R2" in issue or "object storage" in issue for issue in report["issues"])


def test_build_runtime_preflight_reports_live_mode_blockers(tmp_path: Path) -> None:
    report = build_runtime_preflight(
        trading_mode="live",
        identity_layer="none",
        base_dir=tmp_path,
        env={
            "POSTGRES_ENABLED": "true",
            "DATABASE_URL": "postgresql://demo:demo@localhost:5432/ai_trading",
            "CF_R2_BUCKET": "demo-bucket",
            "CF_R2_ENDPOINT": "https://example.r2.cloudflarestorage.com",
            "CF_R2_ACCESS_KEY": "demo-access",
            "CF_R2_SECRET_KEY": "demo-secret-key",
        },
    )

    assert report["status"] == "error"
    assert any("KRAKEN_CLI_ALLOW_LIVE_SUBMIT" in issue for issue in report["issues"])


def test_reset_runtime_state_clears_local_artifacts_and_runtime_env(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)
    for path in (
        artifacts_dir / "orders_audit.jsonl",
        artifacts_dir / "trading_journal.jsonl",
        artifacts_dir / "validation_artifacts.jsonl",
        artifacts_dir / "validation_checkpoints.jsonl",
        artifacts_dir / "activity_log.jsonl",
        artifacts_dir / "run_summary.json",
    ):
        path.write_text('{"status":"stale"}\n', encoding="utf-8")

    raw_path = artifacts_dir / "raw_payloads" / "raw" / "source_type=prices"
    raw_path.mkdir(parents=True)
    (raw_path / "sample.json.gz").write_text("stale", encoding="utf-8")
    runtime_env_path = tmp_path / ".runtime.env"
    runtime_env_path.write_text("AGENT_ID=77\n", encoding="utf-8")

    report = reset_runtime_state(
        base_dir=tmp_path,
        env={},
        reset_postgres=False,
        reset_object_store=False,
    )

    assert report["local_artifacts_removed"] >= 6
    assert report["runtime_env_removed"] is True
    assert not runtime_env_path.exists()
    assert not any(artifacts_dir.iterdir())


def test_execute_trade_cycle_skips_execution_when_runtime_risk_recheck_fails(
    tmp_path: Path,
) -> None:
    app = build_local_demo_app(base_dir=tmp_path, **_storage_overrides(tmp_path))

    class _AlwaysIntentStrategy:
        def generate_trade_intents(self, **_: object) -> list[TradeIntent]:
            return [
                TradeIntent(
                    symbol_id="btc_usd",
                    side="buy",
                    notional_usd=250.0,
                    quantity=0.00367647,
                    current_price=68000.0,
                    score=0.91,
                    rationale=("Strong ETF approval signal",),
                    generated_at=datetime(2026, 4, 3, tzinfo=UTC),
                )
            ]

        def reassess_trade_intent(
            self,
            *,
            intent: TradeIntent,
            portfolio: object,
        ) -> RiskCheckResult:
            del intent, portfolio
            return RiskCheckResult(
                approved=False,
                allowed_notional=0.0,
                violations=(
                    RiskViolation(
                        code="runtime_circuit_breaker",
                        message="Execution blocked after a runtime risk re-check.",
                    ),
                ),
                notes=("Blocked at execution time.",),
            )

    app._strategy = _AlwaysIntentStrategy()

    result = app.execute_trade_cycle()
    risk_artifacts = [
        artifact
        for artifact in result.artifacts
        if artifact.kind.value == "pre_trade_risk_check"
    ]

    assert len(result.trade_intents) == 1
    assert result.execution_results == ()
    assert len(risk_artifacts) == 1
    assert risk_artifacts[0].payload["approved"] is False
    assert risk_artifacts[0].payload["violations"][0]["code"] == "runtime_circuit_breaker"
