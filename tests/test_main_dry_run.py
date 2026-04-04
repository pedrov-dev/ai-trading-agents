from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent.portfolio import LocalPortfolioStateProvider
from agent.risk import RiskCheckResult, RiskViolation
from agent.signals import TradeIntent
from identity.erc8004_registry import SepoliaContractsConfig
from ingestion.prices_config import PRICE_SYMBOLS
from ingestion.rss_config import FeedSource
from main import build_local_demo_app, validate_runtime_requirements

BTC_SYMBOL = next(symbol for symbol in PRICE_SYMBOLS if symbol.symbol_id == "btc_usd")


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

    app = build_local_demo_app(
        base_dir=tmp_path,
        feed_groups={
            "market_news": [FeedSource(source_id="demo_feed", url="https://example.test/rss")]
        },
        symbols=[BTC_SYMBOL],
        parse_feed=fake_parse_feed,
        http_get=fake_http_get,
        env={},
        trading_mode="paper",
    )

    result = app.run_cycle(feed_group="market_news")

    assert result.rss_result.inserted_count == 1
    assert result.prices_result.inserted_count == 1
    assert result.classification_count >= 1
    assert len(result.detected_events) >= 1
    assert len(result.trade_intents) == 1
    assert len(result.execution_results) == 1
    assert result.execution_results[0].status.value == "validated"
    assert result.artifact_count >= 4
    assert result.checkpoint_count >= 4
    assert result.audit_summary.total_events >= 2
    assert result.portfolio.open_position_count() == 0

    assert (tmp_path / "artifacts" / "orders_audit.jsonl").exists()
    assert (tmp_path / "artifacts" / "validation_artifacts.jsonl").exists()
    assert (tmp_path / "artifacts" / "validation_checkpoints.jsonl").exists()
    assert (tmp_path / "artifacts" / "run_summary.json").exists()


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
    )

    assert app.identity.display_name == "Sepolia Momentum Bot"
    assert app.identity.strategy_name == "momentum_breakout_v2"
    assert app.identity.owner == "pedrov-dev"
    assert app.identity.metadata["agent_uri"] == "https://example.test/agent.json"


def test_local_demo_app_persists_agent_id_env_file(tmp_path: Path) -> None:
    app = build_local_demo_app(base_dir=tmp_path)
    env_path = tmp_path / ".runtime.env"

    app.persist_agent_id(agent_id=77, env_path=env_path)

    contents = env_path.read_text(encoding="utf-8")
    assert "AGENT_ID=77" in contents


def test_shared_contract_status_includes_balance_and_claim_state() -> None:
    app = build_local_demo_app(base_dir=Path("."))
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


def test_build_local_demo_app_defaults_to_kraken_paper_mode(tmp_path: Path) -> None:
    app = build_local_demo_app(
        base_dir=tmp_path,
        env={"KRAKEN_CLI_TIMEOUT_SECONDS": "21"},
    )

    assert app._executor._config.dry_run is False
    assert app._executor._config.live_enabled is True
    assert app._executor._config.validate_only is True
    assert app._executor._config.timeout_seconds == 21
    assert app._executor._config.audit_log_path == tmp_path / "artifacts" / "orders_audit.jsonl"


def test_build_local_demo_app_supports_kraken_live_mode(tmp_path: Path) -> None:
    app = build_local_demo_app(
        base_dir=tmp_path,
        env={"KRAKEN_CLI_ALLOW_LIVE_SUBMIT": "true"},
        trading_mode="live",
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
            env={},
        )


def test_validate_runtime_requirements_blocks_missing_erc8004_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="PRIVATE_KEY"):
        validate_runtime_requirements(
            trading_mode="paper",
            identity_layer="erc8004",
            base_dir=tmp_path,
            env={"SEPOLIA_RPC_URL": "https://ethereum-sepolia-rpc.publicnode.com"},
            require_transaction_keys=True,
        )


def test_execute_trade_cycle_skips_execution_when_runtime_risk_recheck_fails(
    tmp_path: Path,
) -> None:
    app = build_local_demo_app(base_dir=tmp_path)

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
