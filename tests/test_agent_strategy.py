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

    assert len(intents) == 4
    assert all(intent.side == "buy" for intent in intents)
    assert sum(intent.notional_usd for intent in intents) > 0
    assert [intent.exit_horizon_label for intent in intents] == ["5m", "30m", "4h", "24h"]
    assert all(intent.event_type == "ETF_APPROVAL" for intent in intents)
    assert all(intent.event_group == "etf_news" for intent in intents)
    assert all(any("ETF_APPROVAL" in reason for reason in intent.rationale) for intent in intents)
    assert all(intent.expected_move_fraction is not None for intent in intents)
    assert all(intent.stop_distance_fraction == 0.02 for intent in intents)
    assert all(
        intent.risk_reward_ratio is not None and intent.risk_reward_ratio > 1.5
        for intent in intents
    )
    assert all(
        any("expected move" in reason.lower() for reason in intent.rationale)
        for intent in intents
    )


def test_strategy_reassess_trade_intent_blocks_entries_below_rr_threshold() -> None:
    strategy = SimpleEventDrivenStrategy(
        risk_config=RiskConfig(min_risk_reward_ratio=1.5),
        exit_config=ExitConfig(profit_target_fraction=0.03, stop_loss_fraction=0.02),
    )
    portfolio = PortfolioSnapshot(total_equity=10_000.0, cash_usd=10_000.0)
    intent = strategy.generate_trade_intents(
        detected_events=[
            DetectedEvent(
                raw_event_id="evt-rr-low",
                event_type="PROTOCOL_UPGRADE",
                rule_name="protocol_upgrade",
                confidence=0.84,
                matched_text="upgrade delivered but follow-through is muted",
                detected_at=datetime(2026, 4, 3, 12, 0, tzinfo=UTC),
            )
        ],
        price_quotes=[
            PriceQuote(
                symbol_id="btc_usd",
                current=68_000.0,
                open=67_900.0,
                high=68_100.0,
                low=67_800.0,
                prev_close=67_850.0,
                timestamp=1712100000,
                asset_class="spot",
            )
        ],
        portfolio=portfolio,
    )[0]

    result = strategy.reassess_trade_intent(intent=intent, portfolio=portfolio)

    assert result.approved is False
    assert result.risk_reward_ratio is not None
    assert result.risk_reward_ratio <= 1.5
    assert any(violation.code == "risk_reward_below_threshold" for violation in result.violations)
    assert any("stop distance" in note.lower() for note in result.notes)


def test_strategy_splits_high_confidence_event_into_multiple_exit_horizons() -> None:
    strategy = SimpleEventDrivenStrategy(risk_config=RiskConfig(max_position_fraction=0.05))
    portfolio = PortfolioSnapshot(total_equity=10_000.0, cash_usd=10_000.0)
    events = [
        DetectedEvent(
            raw_event_id="evt-horizons",
            event_type="ETF_APPROVAL",
            rule_name="etf_approval",
            confidence=0.95,
            matched_text="bitcoin etf approval",
            detected_at=datetime(2026, 4, 3, 12, 0, tzinfo=UTC),
        )
    ]
    prices = [
        PriceQuote(
            symbol_id="btc_usd",
            current=68_000.0,
            open=66_000.0,
            high=68_500.0,
            low=65_500.0,
            prev_close=65_800.0,
            timestamp=1712100000,
            asset_class="spot",
        )
    ]

    intents = strategy.generate_trade_intents(
        detected_events=events,
        price_quotes=prices,
        portfolio=portfolio,
    )

    assert len(intents) == 4
    assert [intent.exit_horizon_label for intent in intents] == ["5m", "30m", "4h", "24h"]
    assert all(intent.signal_id is not None for intent in intents)
    assert all(intent.raw_event_id == "evt-horizons" for intent in intents)
    assert round(sum(intent.notional_usd for intent in intents), 2) > 0


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


def test_strategy_defaults_to_no_trade_when_confidence_is_below_threshold() -> None:
    strategy = SimpleEventDrivenStrategy(
        config=StrategyConfig(min_signal_score=0.7, min_confidence_score=0.7),
        risk_config=RiskConfig(max_position_fraction=0.05),
    )
    portfolio = PortfolioSnapshot(total_equity=10_000.0, cash_usd=10_000.0)
    events = [
        DetectedEvent(
            raw_event_id="evt-no-trade",
            event_type="ETF_APPROVAL",
            rule_name="etf_approval",
            confidence=0.62,
            matched_text="bitcoin etf approval confirmed",
            detected_at=datetime(2026, 4, 3, 12, 15, tzinfo=UTC),
        )
    ]
    prices = [
        PriceQuote(
            symbol_id="btc_usd",
            current=68_000.0,
            open=66_000.0,
            high=68_500.0,
            low=65_500.0,
            prev_close=65_800.0,
            timestamp=1712100000,
            asset_class="spot",
        )
    ]

    intents = strategy.generate_trade_intents(
        detected_events=events,
        price_quotes=prices,
        portfolio=portfolio,
    )
    no_trade_decisions = strategy.consume_no_trade_decisions()

    assert intents == []
    assert len(no_trade_decisions) == 1
    assert no_trade_decisions[0].reason_code == "confidence_below_threshold"
    assert no_trade_decisions[0].confidence_score == 0.62
    assert no_trade_decisions[0].threshold == 0.7
    assert "below threshold" in no_trade_decisions[0].reason.lower()


def test_strategy_decays_repeated_same_thesis_within_cooldown() -> None:
    strategy = SimpleEventDrivenStrategy(
        config=StrategyConfig(
            min_signal_score=0.7,
            thesis_cooldown_hours=6,
            thesis_repeat_penalty=0.12,
        )
    )
    portfolio = PortfolioSnapshot(total_equity=10_000.0, cash_usd=10_000.0)
    prices = [
        PriceQuote(
            symbol_id="btc_usd",
            current=68_000.0,
            open=66_000.0,
            high=68_500.0,
            low=65_500.0,
            prev_close=65_800.0,
            timestamp=1712100000,
            asset_class="spot",
        )
    ]

    first_intents = strategy.generate_trade_intents(
        detected_events=[
            DetectedEvent(
                raw_event_id="evt-cooldown",
                event_type="ETF_APPROVAL",
                rule_name="etf_approval",
                confidence=0.82,
                matched_text="bitcoin etf approval",
                detected_at=datetime(2026, 4, 3, 12, 0, tzinfo=UTC),
            )
        ],
        price_quotes=prices,
        portfolio=portfolio,
    )
    repeated_intents = strategy.generate_trade_intents(
        detected_events=[
            DetectedEvent(
                raw_event_id="evt-cooldown",
                event_type="ETF_APPROVAL",
                rule_name="etf_approval",
                confidence=0.82,
                matched_text="bitcoin etf approval",
                detected_at=datetime(2026, 4, 3, 13, 0, tzinfo=UTC),
            )
        ],
        price_quotes=prices,
        portfolio=portfolio,
    )

    assert len(first_intents) == 4
    assert len(repeated_intents) == 4
    assert first_intents[0].confidence_score is not None
    assert repeated_intents[0].confidence_score is not None
    assert repeated_intents[0].confidence_score < first_intents[0].confidence_score
    assert any("cooldown" in reason.lower() for reason in repeated_intents[0].rationale)


def test_strategy_blocks_repeated_same_thesis_until_new_event_arrives() -> None:
    strategy = SimpleEventDrivenStrategy(
        config=StrategyConfig(
            min_signal_score=0.7,
            thesis_cooldown_hours=6,
            thesis_repeat_penalty=0.16,
        )
    )
    portfolio = PortfolioSnapshot(total_equity=10_000.0, cash_usd=10_000.0)
    prices = [
        PriceQuote(
            symbol_id="btc_usd",
            current=68_000.0,
            open=66_000.0,
            high=68_500.0,
            low=65_500.0,
            prev_close=65_800.0,
            timestamp=1712100000,
            asset_class="spot",
        )
    ]

    base_time = datetime(2026, 4, 3, 12, 0, tzinfo=UTC)
    for hour_offset in range(4):
        strategy.generate_trade_intents(
            detected_events=[
                DetectedEvent(
                    raw_event_id="evt-repeat",
                    event_type="ETF_APPROVAL",
                    rule_name="etf_approval",
                    confidence=0.82,
                    matched_text="bitcoin etf approval",
                    detected_at=base_time.replace(hour=12 + hour_offset),
                )
            ],
            price_quotes=prices,
            portfolio=portfolio,
        )

    blocked_intents = strategy.generate_trade_intents(
        detected_events=[
            DetectedEvent(
                raw_event_id="evt-repeat",
                event_type="ETF_APPROVAL",
                rule_name="etf_approval",
                confidence=0.82,
                matched_text="bitcoin etf approval",
                detected_at=datetime(2026, 4, 3, 16, 30, tzinfo=UTC),
            )
        ],
        price_quotes=prices,
        portfolio=portfolio,
    )
    refreshed_intents = strategy.generate_trade_intents(
        detected_events=[
            DetectedEvent(
                raw_event_id="evt-fresh",
                event_type="ETF_APPROVAL",
                rule_name="etf_approval",
                confidence=0.82,
                matched_text="bitcoin etf approval on fresh headline",
                detected_at=datetime(2026, 4, 3, 17, 0, tzinfo=UTC),
            )
        ],
        price_quotes=prices,
        portfolio=portfolio,
    )

    assert blocked_intents == []
    assert len(refreshed_intents) == 4
    assert refreshed_intents[0].confidence_score is not None
    assert refreshed_intents[0].confidence_score >= 0.7


def test_strategy_expires_cooldown_after_six_hours() -> None:
    strategy = SimpleEventDrivenStrategy(
        config=StrategyConfig(
            min_signal_score=0.7,
            thesis_cooldown_hours=6,
            thesis_repeat_penalty=0.12,
        )
    )
    portfolio = PortfolioSnapshot(total_equity=10_000.0, cash_usd=10_000.0)
    prices = [
        PriceQuote(
            symbol_id="btc_usd",
            current=68_000.0,
            open=66_000.0,
            high=68_500.0,
            low=65_500.0,
            prev_close=65_800.0,
            timestamp=1712100000,
            asset_class="spot",
        )
    ]

    first_intents = strategy.generate_trade_intents(
        detected_events=[
            DetectedEvent(
                raw_event_id="evt-expire",
                event_type="ETF_APPROVAL",
                rule_name="etf_approval",
                confidence=0.82,
                matched_text="bitcoin etf approval",
                detected_at=datetime(2026, 4, 3, 12, 0, tzinfo=UTC),
            )
        ],
        price_quotes=prices,
        portfolio=portfolio,
    )
    expired_window_intents = strategy.generate_trade_intents(
        detected_events=[
            DetectedEvent(
                raw_event_id="evt-expire",
                event_type="ETF_APPROVAL",
                rule_name="etf_approval",
                confidence=0.82,
                matched_text="bitcoin etf approval",
                detected_at=datetime(2026, 4, 3, 19, 30, tzinfo=UTC),
            )
        ],
        price_quotes=prices,
        portfolio=portfolio,
    )

    assert len(first_intents) == 4
    assert len(expired_window_intents) == 4
    assert first_intents[0].confidence_score is not None
    assert expired_window_intents[0].confidence_score is not None
    assert expired_window_intents[0].confidence_score == first_intents[0].confidence_score


def test_strategy_decays_similar_macro_thesis_across_different_headlines() -> None:
    strategy = SimpleEventDrivenStrategy(
        config=StrategyConfig(
            min_signal_score=0.7,
            thesis_cooldown_hours=6,
            thesis_repeat_penalty=0.12,
        )
    )
    portfolio = PortfolioSnapshot(total_equity=10_000.0, cash_usd=10_000.0)
    prices = [
        PriceQuote(
            symbol_id="btc_usd",
            current=64_500.0,
            open=66_000.0,
            high=66_500.0,
            low=64_000.0,
            prev_close=65_800.0,
            timestamp=1712100000,
            asset_class="spot",
        )
    ]

    first_intents = strategy.generate_trade_intents(
        detected_events=[
            DetectedEvent(
                raw_event_id="evt-macro-1",
                event_type="REGULATORY_NEWS",
                rule_name="macro_headwinds",
                confidence=0.83,
                matched_text="BTC bearish because macro headwinds and tight liquidity",
                detected_at=datetime(2026, 4, 3, 12, 0, tzinfo=UTC),
            )
        ],
        price_quotes=prices,
        portfolio=portfolio,
    )
    repeated_intents = strategy.generate_trade_intents(
        detected_events=[
            DetectedEvent(
                raw_event_id="evt-macro-2",
                event_type="REGULATORY_NEWS",
                rule_name="rates_usd_pressure",
                confidence=0.83,
                matched_text=(
                    "BTC bearish because rates and USD strength are "
                    "pressuring risk assets"
                ),
                detected_at=datetime(2026, 4, 3, 13, 0, tzinfo=UTC),
            )
        ],
        price_quotes=prices,
        portfolio=portfolio,
    )

    assert len(first_intents) == 4
    assert len(repeated_intents) == 4
    assert first_intents[0].confidence_score is not None
    assert repeated_intents[0].confidence_score is not None
    assert repeated_intents[0].confidence_score < first_intents[0].confidence_score
    assert any("similar thesis" in reason.lower() for reason in repeated_intents[0].rationale)


def test_strategy_skips_trade_when_volatility_is_not_meaningful() -> None:
    strategy = SimpleEventDrivenStrategy(
        config=StrategyConfig(
            min_signal_score=0.7,
            min_confidence_score=0.7,
            min_meaningful_volatility_filter=0.8,
        ),
        risk_config=RiskConfig(max_position_fraction=0.05),
    )
    portfolio = PortfolioSnapshot(total_equity=10_000.0, cash_usd=10_000.0)
    events = [
        DetectedEvent(
            raw_event_id="evt-low-vol-skip",
            event_type="ETF_APPROVAL",
            rule_name="etf_approval",
            confidence=0.9,
            matched_text="bitcoin etf approval",
            detected_at=datetime(2026, 4, 3, 12, 15, tzinfo=UTC),
        )
    ]
    prices = [
        PriceQuote(
            symbol_id="btc_usd",
            current=68_000.0,
            open=67_900.0,
            high=68_200.0,
            low=67_850.0,
            prev_close=67_880.0,
            timestamp=1712100000,
            asset_class="spot",
            atr=250.0,
            realized_volatility=0.019,
            volatility_filter=0.19,
        )
    ]

    intents = strategy.generate_trade_intents(
        detected_events=events,
        price_quotes=prices,
        portfolio=portfolio,
    )
    no_trade_decisions = strategy.consume_no_trade_decisions()

    assert intents == []
    assert len(no_trade_decisions) == 1
    assert no_trade_decisions[0].reason_code == "volatility_not_meaningful"
    assert "volatility filter" in no_trade_decisions[0].reason.lower()


def test_strategy_skips_news_only_setup_below_confirmation_threshold() -> None:
    strategy = SimpleEventDrivenStrategy(
        config=StrategyConfig(
            min_signal_score=0.5,
            min_confidence_score=0.7,
            entry_confirmation_threshold=0.65,
            reduced_size_confirmation_threshold=0.5,
        ),
        risk_config=RiskConfig(max_position_fraction=0.05),
    )
    portfolio = PortfolioSnapshot(total_equity=10_000.0, cash_usd=10_000.0)
    events = [
        DetectedEvent(
            raw_event_id="evt-news-only",
            event_type="ETF_APPROVAL",
            rule_name="etf_approval",
            confidence=0.92,
            matched_text="bitcoin etf approval headline crosses",
            detected_at=datetime(2026, 4, 4, 12, 15, tzinfo=UTC),
        )
    ]
    prices = [
        PriceQuote(
            symbol_id="btc_usd",
            current=68_000.0,
            open=67_980.0,
            high=68_050.0,
            low=67_900.0,
            prev_close=67_960.0,
            timestamp=1712189700,
            asset_class="spot",
        )
    ]

    intents = strategy.generate_trade_intents(
        detected_events=events,
        price_quotes=prices,
        portfolio=portfolio,
    )
    no_trade_decisions = strategy.consume_no_trade_decisions()

    assert intents == []
    assert len(no_trade_decisions) == 1
    assert no_trade_decisions[0].reason_code == "confirmation_below_threshold"
    assert "needs at least" in no_trade_decisions[0].reason.lower()


def test_strategy_scales_position_size_by_confirmation_strength() -> None:
    strategy = SimpleEventDrivenStrategy(
        config=StrategyConfig(
            min_signal_score=0.5,
            min_confidence_score=0.7,
            entry_confirmation_threshold=0.65,
            reduced_size_confirmation_threshold=0.5,
        ),
        risk_config=RiskConfig(max_position_fraction=0.05),
    )
    portfolio = PortfolioSnapshot(total_equity=10_000.0, cash_usd=10_000.0)
    events = [
        DetectedEvent(
            raw_event_id="evt-strong",
            event_type="SECURITY_INCIDENT",
            rule_name="security_incident",
            confidence=0.94,
            matched_text="bitcoin drops after exchange security incident",
            detected_at=datetime(2026, 4, 4, 13, 0, tzinfo=UTC),
        ),
        DetectedEvent(
            raw_event_id="evt-medium",
            event_type="TECHNICAL_BREAKOUT",
            rule_name="technical_breakout",
            confidence=0.88,
            matched_text="solana breakout extends but turnover stays mixed",
            detected_at=datetime(2026, 4, 4, 13, 5, tzinfo=UTC),
        ),
    ]
    prices = [
        PriceQuote(
            symbol_id="btc_usd",
            current=63_500.0,
            open=66_500.0,
            high=66_700.0,
            low=63_200.0,
            prev_close=66_200.0,
            timestamp=1712190000,
            asset_class="spot",
            session_volume=28_000.0,
            volume_ratio=2.4,
        ),
        PriceQuote(
            symbol_id="sol_usd",
            current=124.0,
            open=120.0,
            high=126.0,
            low=119.0,
            prev_close=118.5,
            timestamp=1712190300,
            asset_class="spot",
            session_volume=9_000.0,
            volume_ratio=1.0,
        ),
    ]

    intents = strategy.generate_trade_intents(
        detected_events=events,
        price_quotes=prices,
        portfolio=portfolio,
    )

    by_symbol = {
        symbol_id: [intent for intent in intents if intent.symbol_id == symbol_id]
        for symbol_id in {intent.symbol_id for intent in intents}
    }

    assert {intent.symbol_id for intent in intents} == {"btc_usd", "sol_usd"}
    assert round(sum(intent.notional_usd for intent in by_symbol["btc_usd"]), 2) > round(
        sum(intent.notional_usd for intent in by_symbol["sol_usd"]),
        2,
    )
    assert all(
        "reduced-size mode" in " ".join(intent.rationale).lower()
        for intent in by_symbol["sol_usd"]
    )
    assert all(intent.side == "sell" for intent in by_symbol["btc_usd"])
    assert all(intent.side == "buy" for intent in by_symbol["sol_usd"])


def test_strategy_keeps_opposite_direction_thesis_separate() -> None:
    strategy = SimpleEventDrivenStrategy(
        config=StrategyConfig(
            min_signal_score=0.7,
            thesis_cooldown_hours=6,
            thesis_repeat_penalty=0.12,
        )
    )
    portfolio = PortfolioSnapshot(total_equity=10_000.0, cash_usd=10_000.0)
    prices = [
        PriceQuote(
            symbol_id="btc_usd",
            current=68_000.0,
            open=66_000.0,
            high=68_500.0,
            low=65_500.0,
            prev_close=65_800.0,
            timestamp=1712100000,
            asset_class="spot",
        )
    ]

    bullish_intents = strategy.generate_trade_intents(
        detected_events=[
            DetectedEvent(
                raw_event_id="evt-bullish",
                event_type="ETF_APPROVAL",
                rule_name="etf_approval",
                confidence=0.86,
                matched_text="bitcoin etf approval boosts institutional demand",
                detected_at=datetime(2026, 4, 3, 12, 0, tzinfo=UTC),
            )
        ],
        price_quotes=prices,
        portfolio=portfolio,
    )
    bearish_intents = strategy.generate_trade_intents(
        detected_events=[
            DetectedEvent(
                raw_event_id="evt-bearish",
                event_type="SECURITY_INCIDENT",
                rule_name="exchange_hack",
                confidence=0.9,
                matched_text="bitcoin turns bearish after a major exchange security incident",
                detected_at=datetime(2026, 4, 3, 13, 0, tzinfo=UTC),
            )
        ],
        price_quotes=prices,
        portfolio=portfolio,
    )

    assert len(bullish_intents) == 4
    assert len(bearish_intents) == 4
    assert all(intent.side == "buy" for intent in bullish_intents)
    assert all(intent.side == "sell" for intent in bearish_intents)
    assert not any("similar thesis" in reason.lower() for reason in bearish_intents[0].rationale)


def test_strategy_ranks_entry_opportunities_and_executes_only_top_n_signals() -> None:
    strategy = SimpleEventDrivenStrategy(
        config=StrategyConfig(
            min_signal_score=0.6,
            max_ranked_signals_per_cycle=1,
            confidence_weight=0.1,
            novelty_weight=0.15,
            risk_reward_weight=0.15,
            diversification_weight=0.6,
        ),
        risk_config=RiskConfig(max_position_fraction=0.05),
    )
    portfolio = PortfolioSnapshot(
        total_equity=10_000.0,
        cash_usd=9_000.0,
        positions=(
            Position(
                symbol_id="btc_usd",
                side="long",
                quantity=0.01,
                entry_price=68_000.0,
                event_type="ETF_APPROVAL",
            ),
            Position(
                symbol_id="eth_usd",
                side="long",
                quantity=0.6,
                entry_price=3_400.0,
                event_type="TECHNICAL_BREAKOUT",
            ),
        ),
    )
    events = [
        DetectedEvent(
            raw_event_id="evt-sol-breakout",
            event_type="TECHNICAL_BREAKOUT",
            rule_name="sol_breakout",
            confidence=0.9,
            matched_text="solana breakout extends bullish momentum",
            detected_at=datetime(2026, 4, 3, 12, 0, tzinfo=UTC),
        ),
        DetectedEvent(
            raw_event_id="evt-xrp-regulatory",
            event_type="REGULATORY_ACTION",
            rule_name="xrp_sec_pressure",
            confidence=0.78,
            matched_text="xrp faces fresh sec enforcement pressure",
            detected_at=datetime(2026, 4, 3, 12, 5, tzinfo=UTC),
        ),
    ]
    prices = [
        PriceQuote(
            symbol_id="sol_usd",
            current=124.0,
            open=120.0,
            high=126.0,
            low=118.0,
            prev_close=119.0,
            timestamp=1712100000,
            asset_class="spot",
        ),
        PriceQuote(
            symbol_id="xrp_usd",
            current=0.49,
            open=0.55,
            high=0.56,
            low=0.48,
            prev_close=0.54,
            timestamp=1712100000,
            asset_class="spot",
        ),
    ]

    intents = strategy.generate_trade_intents(
        detected_events=events,
        price_quotes=prices,
        portfolio=portfolio,
    )

    assert len(intents) == 4
    assert {intent.symbol_id for intent in intents} == {"xrp_usd"}
    assert all(intent.side == "sell" for intent in intents)


def test_strategy_adds_ranking_breakdown_to_selected_intent_rationale() -> None:
    strategy = SimpleEventDrivenStrategy(
        config=StrategyConfig(
            min_signal_score=0.6,
            confidence_weight=0.35,
            novelty_weight=0.2,
            risk_reward_weight=0.25,
            diversification_weight=0.2,
        )
    )
    portfolio = PortfolioSnapshot(total_equity=10_000.0, cash_usd=10_000.0)
    prices = [
        PriceQuote(
            symbol_id="btc_usd",
            current=68_000.0,
            open=66_000.0,
            high=68_500.0,
            low=65_500.0,
            prev_close=65_800.0,
            timestamp=1712100000,
            asset_class="spot",
        )
    ]

    intents = strategy.generate_trade_intents(
        detected_events=[
            DetectedEvent(
                raw_event_id="evt-ranked",
                event_type="ETF_APPROVAL",
                rule_name="etf_approval",
                confidence=0.9,
                matched_text="bitcoin etf approval sees strong follow-through",
                detected_at=datetime(2026, 4, 3, 12, 0, tzinfo=UTC),
            )
        ],
        price_quotes=prices,
        portfolio=portfolio,
    )

    ranking_text = " ".join(intents[0].rationale).lower()
    assert "ranking breakdown" in ranking_text
    assert "confidence=" in ranking_text
    assert "novelty=" in ranking_text
    assert "risk/reward=" in ranking_text
    assert "diversification=" in ranking_text


def test_strategy_respects_opportunity_budget_and_searches_beyond_btc() -> None:
    strategy = SimpleEventDrivenStrategy(
        config=StrategyConfig(
            min_signal_score=0.6,
            max_ranked_signals_per_cycle=3,
            confidence_weight=0.7,
            novelty_weight=0.2,
            risk_reward_weight=0.1,
            diversification_weight=0.0,
        ),
        risk_config=RiskConfig(
            max_position_fraction=0.05,
            max_concurrent_positions=3,
            max_positions_per_asset=1,
        ),
    )
    portfolio = PortfolioSnapshot(
        total_equity=10_000.0,
        cash_usd=8_000.0,
        positions=(
            Position(
                symbol_id="btc_usd",
                side="long",
                quantity=0.01,
                entry_price=68_000.0,
                event_type="ETF_APPROVAL",
            ),
            Position(
                symbol_id="eth_usd",
                side="long",
                quantity=0.5,
                entry_price=3_300.0,
                event_type="TECHNICAL_BREAKOUT",
            ),
        ),
    )
    events = [
        DetectedEvent(
            raw_event_id="evt-btc-repeat",
            event_type="ETF_APPROVAL",
            rule_name="btc_repeat",
            confidence=0.97,
            matched_text="bitcoin etf approval keeps driving upside momentum",
            detected_at=datetime(2026, 4, 3, 12, 0, tzinfo=UTC),
        ),
        DetectedEvent(
            raw_event_id="evt-sol-budget",
            event_type="TECHNICAL_BREAKOUT",
            rule_name="sol_breakout",
            confidence=0.9,
            matched_text="solana breakout extends bullish momentum",
            detected_at=datetime(2026, 4, 3, 12, 2, tzinfo=UTC),
        ),
        DetectedEvent(
            raw_event_id="evt-xrp-budget",
            event_type="REGULATORY_ACTION",
            rule_name="xrp_regulatory",
            confidence=0.74,
            matched_text="xrp faces fresh sec enforcement pressure",
            detected_at=datetime(2026, 4, 3, 12, 5, tzinfo=UTC),
        ),
    ]
    prices = [
        PriceQuote(
            symbol_id="btc_usd",
            current=68_200.0,
            open=67_000.0,
            high=68_400.0,
            low=66_900.0,
            prev_close=66_800.0,
            timestamp=1712100000,
            asset_class="spot",
        ),
        PriceQuote(
            symbol_id="sol_usd",
            current=124.0,
            open=120.0,
            high=126.0,
            low=118.0,
            prev_close=119.0,
            timestamp=1712100000,
            asset_class="spot",
        ),
        PriceQuote(
            symbol_id="xrp_usd",
            current=0.49,
            open=0.55,
            high=0.56,
            low=0.48,
            prev_close=0.54,
            timestamp=1712100000,
            asset_class="spot",
        ),
    ]

    intents = strategy.generate_trade_intents(
        detected_events=events,
        price_quotes=prices,
        portfolio=portfolio,
    )

    assert len(intents) == 1
    assert [intent.symbol_id for intent in intents] == ["sol_usd"]
    assert [intent.exit_horizon_label for intent in intents] == ["5m"]


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
    assert intents[0].event_group is None
    assert intents[0].quantity == 0.01
    assert any("profit" in reason.lower() for reason in intents[0].rationale)


def test_strategy_generates_exit_intent_when_horizon_time_is_reached() -> None:
    strategy = SimpleEventDrivenStrategy(
        exit_config=ExitConfig(
            profit_target_fraction=0.5,
            stop_loss_fraction=0.5,
        )
    )
    portfolio = PortfolioSnapshot(
        total_equity=10_000.0,
        cash_usd=9_500.0,
        positions=(
            Position(
                position_id="pos-5m",
                symbol_id="btc_usd",
                side="long",
                quantity=0.01,
                entry_price=50_000.0,
                opened_at=datetime(2026, 4, 3, 12, 0, tzinfo=UTC),
                source_signal_id="signal-123",
                raw_event_id="evt-5m",
                event_type="ETF_APPROVAL",
                exit_horizon_label="5m",
                max_hold_minutes=5,
                exit_due_at=datetime(2026, 4, 3, 12, 5, tzinfo=UTC),
            ),
        ),
    )
    prices = [
        PriceQuote(
            symbol_id="btc_usd",
            current=50_100.0,
            open=50_000.0,
            high=50_200.0,
            low=49_900.0,
            prev_close=49_950.0,
            timestamp=1712100000,
            asset_class="spot",
        )
    ]

    intents = strategy.evaluate_position_exits(
        portfolio=portfolio,
        price_quotes=prices,
        detected_events=[],
        now=datetime(2026, 4, 3, 12, 6, tzinfo=UTC),
    )

    assert len(intents) == 1
    assert intents[0].side == "sell"
    assert intents[0].position_id == "pos-5m"
    assert intents[0].exit_horizon_label == "5m"
    assert any("5m" in reason or "horizon" in reason.lower() for reason in intents[0].rationale)


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
