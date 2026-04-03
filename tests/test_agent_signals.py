from datetime import UTC, datetime

from agent.signals import build_signal
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
    assert any("price confirmation" in reason.lower() for reason in signal.rationale)
