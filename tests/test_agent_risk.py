from datetime import UTC, datetime, timedelta

from agent.portfolio import PortfolioSnapshot, Position
from agent.risk import RiskConfig, RiskManager
from agent.signals import Signal


def _make_signal(*, side: str = "buy", score: float = 0.9) -> Signal:
    return Signal(
        raw_event_id="evt-1",
        event_type="ETF_APPROVAL",
        symbol_id="btc_usd",
        side=side,
        confidence=0.95,
        score=score,
        current_price=68000.0,
        generated_at=datetime(2026, 4, 3, tzinfo=UTC),
        rationale=("Bullish event bias", "Price confirmation supports entry"),
    )


def test_risk_manager_caps_position_size_and_approves_when_limits_are_clear() -> None:
    portfolio = PortfolioSnapshot(total_equity=10_000.0, cash_usd=10_000.0)
    manager = RiskManager(RiskConfig(max_position_fraction=0.05))

    notional = manager.size_for_signal(signal=_make_signal(score=0.95), portfolio=portfolio)
    result = manager.evaluate(
        signal=_make_signal(score=0.95),
        portfolio=portfolio,
        proposed_notional=notional,
    )

    assert notional <= 500.0
    assert result.approved is True
    assert result.allowed_notional <= 500.0


def test_risk_manager_blocks_after_max_daily_loss() -> None:
    portfolio = PortfolioSnapshot(
        total_equity=10_000.0,
        cash_usd=9_200.0,
        realized_pnl_today=-600.0,
    )
    manager = RiskManager(RiskConfig(max_daily_loss_fraction=0.05))

    result = manager.evaluate(signal=_make_signal(), portfolio=portfolio, proposed_notional=300.0)

    assert result.approved is False
    assert any(violation.code == "max_daily_loss" for violation in result.violations)


def test_risk_manager_blocks_when_max_concurrent_positions_is_reached() -> None:
    portfolio = PortfolioSnapshot(
        total_equity=10_000.0,
        cash_usd=4_000.0,
        positions=(
            Position(symbol_id="btc_usd", side="long", quantity=0.1, entry_price=64000.0),
            Position(symbol_id="eth_usd", side="long", quantity=1.0, entry_price=3000.0),
        ),
    )
    manager = RiskManager(RiskConfig(max_concurrent_positions=2))

    result = manager.evaluate(signal=_make_signal(), portfolio=portfolio, proposed_notional=300.0)

    assert result.approved is False
    assert any(violation.code == "max_concurrent_positions" for violation in result.violations)


def test_risk_manager_enforces_cooldown_after_losses() -> None:
    portfolio = PortfolioSnapshot(
        total_equity=10_000.0,
        cash_usd=9_500.0,
        consecutive_losses=2,
        last_loss_at=datetime.now(UTC) - timedelta(minutes=10),
    )
    manager = RiskManager(RiskConfig(cooldown_minutes_after_loss=30))

    result = manager.evaluate(signal=_make_signal(), portfolio=portfolio, proposed_notional=250.0)

    assert result.approved is False
    assert any(violation.code == "cooldown_after_loss" for violation in result.violations)
