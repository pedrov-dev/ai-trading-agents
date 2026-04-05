from datetime import UTC, datetime

from agent.signals import build_signal, build_trade_intent
from detection.event_detection import DetectedEvent
from ingestion.prices_ingestion import PriceQuote


def test_build_signal_scores_bullish_event_with_price_confirmation() -> None:
    event = DetectedEvent(
        raw_event_id="evt-1",
        event_type="ETF_APPROVAL",
        rule_name="etf_approval",
        confidence=0.95,
        matched_text="bitcoin etf approval",
        detected_at=datetime(2026, 4, 3, tzinfo=UTC),
    )
    quote = PriceQuote(
        symbol_id="btc_usd",
        current=68000.0,
        open=66000.0,
        high=68500.0,
        low=65500.0,
        prev_close=65800.0,
        timestamp=1712100000,
        asset_class="spot",
    )

    signal = build_signal(event=event, quote=quote)

    assert signal.side == "buy"
    assert signal.symbol_id == "btc_usd"
    assert signal.score >= 0.8
    assert signal.event_group == "etf_news"
    assert signal.signal_id is not None
    assert any("bullish" in reason.lower() for reason in signal.rationale)


def test_build_signal_scores_bearish_event_with_negative_momentum() -> None:
    event = DetectedEvent(
        raw_event_id="evt-2",
        event_type="SECURITY_INCIDENT",
        rule_name="security_incident",
        confidence=0.9,
        matched_text="exchange hack",
        detected_at=datetime(2026, 4, 3, tzinfo=UTC),
    )
    quote = PriceQuote(
        symbol_id="eth_usd",
        current=3100.0,
        open=3300.0,
        high=3325.0,
        low=3050.0,
        prev_close=3280.0,
        timestamp=1712100000,
        asset_class="spot",
    )

    signal = build_signal(event=event, quote=quote)

    assert signal.side == "sell"
    assert signal.symbol_id == "eth_usd"
    assert signal.score >= 0.8
    assert signal.event_group == "exchange_hacks"
    assert any("price confirmation" in reason.lower() for reason in signal.rationale)


def test_build_trade_intent_preserves_event_attribution() -> None:
    event = DetectedEvent(
        raw_event_id="evt-3",
        event_type="MACRO_NEWS",
        rule_name="macro_news",
        confidence=0.82,
        matched_text="fed rate cut",
        detected_at=datetime(2026, 4, 3, tzinfo=UTC),
    )
    quote = PriceQuote(
        symbol_id="btc_usd",
        current=68_500.0,
        open=67_900.0,
        high=68_900.0,
        low=67_500.0,
        prev_close=67_700.0,
        timestamp=1712100000,
        asset_class="spot",
    )

    signal = build_signal(event=event, quote=quote)
    intent = build_trade_intent(signal=signal, notional_usd=500.0)

    assert intent.event_type == "MACRO_NEWS"
    assert intent.event_group == "macro_news"
    assert intent.signal_id == signal.signal_id
    assert intent.notional_usd == 500.0


def test_build_signal_weakens_when_volatility_is_low() -> None:
    event = DetectedEvent(
        raw_event_id="evt-low-vol",
        event_type="ETF_APPROVAL",
        rule_name="etf_approval",
        confidence=0.88,
        matched_text="bitcoin etf approval",
        detected_at=datetime(2026, 4, 3, tzinfo=UTC),
    )
    neutral_quote = PriceQuote(
        symbol_id="btc_usd",
        current=68_000.0,
        open=67_200.0,
        high=68_500.0,
        low=66_900.0,
        prev_close=67_000.0,
        timestamp=1712100000,
        asset_class="spot",
        atr=1_000.0,
        realized_volatility=0.0147,
        volatility_filter=1.0,
    )
    low_vol_quote = PriceQuote(
        symbol_id="btc_usd",
        current=68_000.0,
        open=67_200.0,
        high=68_500.0,
        low=66_900.0,
        prev_close=67_000.0,
        timestamp=1712100000,
        asset_class="spot",
        atr=320.0,
        realized_volatility=0.0201,
        volatility_filter=0.24,
    )

    neutral_signal = build_signal(event=event, quote=neutral_quote)
    low_vol_signal = build_signal(event=event, quote=low_vol_quote)

    assert low_vol_signal.confidence < neutral_signal.confidence
    assert low_vol_signal.score < neutral_signal.score
    assert any("low volatility" in reason.lower() for reason in low_vol_signal.rationale)


def test_build_signal_strengthens_when_volatility_is_high() -> None:
    event = DetectedEvent(
        raw_event_id="evt-high-vol",
        event_type="ETF_APPROVAL",
        rule_name="etf_approval",
        confidence=0.88,
        matched_text="bitcoin etf approval",
        detected_at=datetime(2026, 4, 3, tzinfo=UTC),
    )
    neutral_quote = PriceQuote(
        symbol_id="btc_usd",
        current=68_000.0,
        open=67_200.0,
        high=68_500.0,
        low=66_900.0,
        prev_close=67_000.0,
        timestamp=1712100000,
        asset_class="spot",
        atr=1_000.0,
        realized_volatility=0.0147,
        volatility_filter=1.0,
    )
    high_vol_quote = PriceQuote(
        symbol_id="btc_usd",
        current=68_000.0,
        open=67_200.0,
        high=69_400.0,
        low=66_200.0,
        prev_close=67_000.0,
        timestamp=1712100000,
        asset_class="spot",
        atr=2_100.0,
        realized_volatility=0.0154,
        volatility_filter=2.0,
    )

    neutral_signal = build_signal(event=event, quote=neutral_quote)
    high_vol_signal = build_signal(event=event, quote=high_vol_quote)

    assert high_vol_signal.confidence > neutral_signal.confidence
    assert high_vol_signal.score > neutral_signal.score
    assert any("high volatility" in reason.lower() for reason in high_vol_signal.rationale)


def test_build_signal_penalizes_repeated_low_novelty_narrative() -> None:
    fresh_event = DetectedEvent(
        raw_event_id="evt-fresh",
        event_type="ETF_APPROVAL",
        rule_name="etf_approval",
        confidence=0.88,
        matched_text="bitcoin etf approval rumor",
        detected_at=datetime(2026, 4, 3, tzinfo=UTC),
        novelty_score=1.0,
        repeat_count=0,
        narrative_key="etf-approval-rumor",
    )
    repeated_event = DetectedEvent(
        raw_event_id="evt-repeat",
        event_type="ETF_APPROVAL",
        rule_name="etf_approval",
        confidence=0.88,
        matched_text="bitcoin etf approval rumor again",
        detected_at=datetime(2026, 4, 3, 1, tzinfo=UTC),
        novelty_score=0.2,
        repeat_count=4,
        narrative_key="etf-approval-rumor",
    )
    quote = PriceQuote(
        symbol_id="btc_usd",
        current=68_000.0,
        open=67_200.0,
        high=68_500.0,
        low=66_900.0,
        prev_close=67_000.0,
        timestamp=1712100000,
        asset_class="spot",
    )

    fresh_signal = build_signal(event=fresh_event, quote=quote)
    repeated_signal = build_signal(event=repeated_event, quote=quote)

    assert repeated_signal.score < fresh_signal.score
    assert repeated_signal.confidence < fresh_signal.confidence
    assert any("repeated narrative" in reason.lower() for reason in repeated_signal.rationale)


def test_build_signal_uses_weighted_confirmation_without_volume() -> None:
    event = DetectedEvent(
        raw_event_id="evt-weighted-no-volume",
        event_type="ETF_APPROVAL",
        rule_name="etf_approval",
        confidence=0.91,
        matched_text="bitcoin etf approval keeps sentiment bullish",
        detected_at=datetime(2026, 4, 4, 9, 0, tzinfo=UTC),
    )
    quote = PriceQuote(
        symbol_id="btc_usd",
        current=68_000.0,
        open=66_000.0,
        high=68_400.0,
        low=65_800.0,
        prev_close=65_500.0,
        timestamp=1712182800,
        asset_class="spot",
    )

    signal = build_signal(event=event, quote=quote)

    assert signal.news_confirmed is True
    assert signal.price_confirmed is True
    assert signal.volume_confirmed is False
    assert signal.confirmation_count == 2
    assert signal.confirmation_score == 0.75
    assert signal.score >= 0.75
    assert any(
        "volume confirmation is unavailable" in reason.lower()
        for reason in signal.rationale
    )


def test_build_signal_strengthens_bearish_setup_with_volume_spike() -> None:
    event = DetectedEvent(
        raw_event_id="evt-bearish-volume",
        event_type="SECURITY_INCIDENT",
        rule_name="security_incident",
        confidence=0.94,
        matched_text="bitcoin turns bearish after a major exchange hack",
        detected_at=datetime(2026, 4, 4, 10, 0, tzinfo=UTC),
    )
    no_volume_quote = PriceQuote(
        symbol_id="btc_usd",
        current=63_500.0,
        open=66_500.0,
        high=66_700.0,
        low=63_200.0,
        prev_close=66_200.0,
        timestamp=1712186400,
        asset_class="spot",
        session_volume=12_500.0,
        volume_ratio=1.0,
    )
    spike_quote = PriceQuote(
        symbol_id="btc_usd",
        current=63_500.0,
        open=66_500.0,
        high=66_700.0,
        low=63_200.0,
        prev_close=66_200.0,
        timestamp=1712186400,
        asset_class="spot",
        session_volume=28_000.0,
        volume_ratio=2.4,
    )

    base_signal = build_signal(event=event, quote=no_volume_quote)
    confirmed_signal = build_signal(event=event, quote=spike_quote)

    assert confirmed_signal.news_confirmed is True
    assert confirmed_signal.price_confirmed is True
    assert confirmed_signal.volume_confirmed is True
    assert confirmed_signal.confirmation_count == 3
    assert confirmed_signal.confirmation_score == 1.0
    assert confirmed_signal.score > base_signal.score
    assert any("price breakdown" in reason.lower() for reason in confirmed_signal.rationale)
    assert any("volume spike" in reason.lower() for reason in confirmed_signal.rationale)
