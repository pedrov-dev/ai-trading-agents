from datetime import UTC, datetime

from agent.portfolio import PortfolioSnapshot, Position
from agent.risk import RiskConfig
from agent.strategy import ExitConfig, SimpleEventDrivenStrategy, StrategyConfig
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


def test_strategy_generates_exit_intent_when_take_profit_is_hit() -> None:
    strategy = SimpleEventDrivenStrategy(
        exit_config=ExitConfig(
            profit_target_fraction=0.02,
            stop_loss_fraction=0.05,
            max_hold_minutes=240,
        )
    )
    portfolio = PortfolioSnapshot(
        total_equity=10_000.0,
        cash_usd=9_500.0,
        positions=(
            Position(
                symbol_id="btc_usd",
                side="long",
                quantity=0.01,
                entry_price=50_000.0,
                opened_at=datetime(2026, 4, 3, 12, 0, tzinfo=UTC),
            ),
        ),
    )
    prices = [
        PriceQuote(
            symbol_id="btc_usd",
            current=51_500.0,
            open=50_500.0,
            high=51_700.0,
            low=50_200.0,
            prev_close=50_400.0,
            timestamp=1712100000,
            asset_class="spot",
        )
    ]

    intents = strategy.evaluate_position_exits(
        portfolio=portfolio,
        price_quotes=prices,
        detected_events=[],
        now=datetime(2026, 4, 3, 12, 30, tzinfo=UTC),
    )

    assert len(intents) == 1
    assert intents[0].symbol_id == "btc_usd"
    assert intents[0].side == "sell"
    assert intents[0].quantity == 0.01
    assert any("profit" in reason.lower() for reason in intents[0].rationale)


def test_strategy_generates_exit_intent_for_opposite_event_on_open_position() -> None:
    strategy = SimpleEventDrivenStrategy(exit_config=ExitConfig())
    portfolio = PortfolioSnapshot(
        total_equity=10_000.0,
        cash_usd=9_500.0,
        positions=(
            Position(
                symbol_id="btc_usd",
                side="long",
                quantity=0.01,
                entry_price=50_000.0,
                opened_at=datetime(2026, 4, 3, 12, 0, tzinfo=UTC),
            ),
        ),
    )
    prices = [
        PriceQuote(
            symbol_id="btc_usd",
            current=49_900.0,
            open=50_100.0,
            high=50_200.0,
            low=49_800.0,
            prev_close=50_150.0,
            timestamp=1712100000,
            asset_class="spot",
        )
    ]
    events = [
        DetectedEvent(
            raw_event_id="evt-3",
            event_type="SECURITY_INCIDENT",
            rule_name="security_incident",
            confidence=0.92,
            matched_text="bitcoin wallet exploit puts btc under pressure",
            detected_at=datetime(2026, 4, 3, 12, 20, tzinfo=UTC),
        )
    ]

    intents = strategy.evaluate_position_exits(
        portfolio=portfolio,
        price_quotes=prices,
        detected_events=events,
        now=datetime(2026, 4, 3, 12, 25, tzinfo=UTC),
    )

    assert len(intents) == 1
    assert intents[0].side == "sell"
    assert any("opposite" in reason.lower() for reason in intents[0].rationale)
