from datetime import UTC, datetime

from agent.portfolio import PortfolioSnapshot
from agent.risk import RiskConfig
from agent.strategy import SimpleEventDrivenStrategy, StrategyConfig
from detection.event_detection import DetectedEvent
from ingestion.prices_ingestion import PriceQuote


def test_strategy_generates_trade_intent_for_high_confidence_event() -> None:
    strategy = SimpleEventDrivenStrategy(risk_config=RiskConfig(max_position_fraction=0.05))
    portfolio = PortfolioSnapshot(total_equity=10_000.0, cash_usd=10_000.0)
    events = [
        DetectedEvent(
            raw_event_id="evt-1",
            event_type="ETF_APPROVAL",
            rule_name="etf_approval",
            confidence=0.95,
            matched_text="bitcoin etf approval",
            detected_at=datetime(2026, 4, 3, tzinfo=UTC),
        )
    ]
    prices = [
        PriceQuote(
            symbol_id="btc_usd",
            current=68000.0,
            open=66000.0,
            high=68500.0,
            low=65500.0,
            prev_close=65800.0,
            timestamp=1712100000,
            asset_class="spot",
        )
    ]

    intents = strategy.generate_trade_intents(
        detected_events=events,
        price_quotes=prices,
        portfolio=portfolio,
    )

    assert len(intents) == 1
    assert intents[0].side == "buy"
    assert intents[0].notional_usd > 0
    assert any("ETF_APPROVAL" in reason for reason in intents[0].rationale)


def test_strategy_skips_low_score_or_risk_blocked_setups() -> None:
    strategy = SimpleEventDrivenStrategy(
        config=StrategyConfig(min_signal_score=0.75),
        risk_config=RiskConfig(max_daily_loss_fraction=0.02),
    )
    portfolio = PortfolioSnapshot(
        total_equity=10_000.0,
        cash_usd=9_000.0,
        realized_pnl_today=-300.0,
    )
    events = [
        DetectedEvent(
            raw_event_id="evt-2",
            event_type="PROTOCOL_UPGRADE",
            rule_name="protocol_upgrade",
            confidence=0.55,
            matched_text="mainnet upgrade",
            detected_at=datetime(2026, 4, 3, tzinfo=UTC),
        )
    ]
    prices = [
        PriceQuote(
            symbol_id="btc_usd",
            current=68000.0,
            open=67950.0,
            high=68200.0,
            low=67800.0,
            prev_close=67700.0,
            timestamp=1712100000,
            asset_class="spot",
        )
    ]

    intents = strategy.generate_trade_intents(
        detected_events=events,
        price_quotes=prices,
        portfolio=portfolio,
    )

    assert intents == []
