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
