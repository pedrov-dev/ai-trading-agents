from datetime import UTC, datetime

from agent.news_signal import (
    build_thesis_fingerprint,
    extract_thesis_tokens,
    infer_trade_side,
    select_quote_for_event,
)
from detection.event_detection import DetectedEvent
from ingestion.prices_ingestion import PriceQuote


def test_infer_trade_side_maps_supported_event_biases() -> None:
    assert infer_trade_side("ETF_APPROVAL") == "buy"
    assert infer_trade_side("SECURITY_INCIDENT") == "sell"
    assert infer_trade_side("UNKNOWN_EVENT") is None


def test_select_quote_for_event_prefers_matching_symbol_keywords() -> None:
    event = DetectedEvent(
        raw_event_id="evt-ton-news",
        event_type="TOKEN_LISTING",
        rule_name="telegram_listing",
        confidence=0.82,
        matched_text="Telegram sentiment boosts Toncoin after a new listing.",
        detected_at=datetime(2026, 4, 5, tzinfo=UTC),
    )
    quotes = [
        PriceQuote(
            symbol_id="btc_usd",
            current=68_000.0,
            open=67_500.0,
            high=68_300.0,
            low=67_100.0,
            prev_close=67_200.0,
            timestamp=1712275200,
            asset_class="spot",
        ),
        PriceQuote(
            symbol_id="ton_usd",
            current=5.4,
            open=5.1,
            high=5.5,
            low=5.0,
            prev_close=5.0,
            timestamp=1712275200,
            asset_class="spot",
        ),
    ]

    quote = select_quote_for_event(event=event, price_quotes=quotes)

    assert quote is not None
    assert quote.symbol_id == "ton_usd"


def test_extract_thesis_tokens_normalizes_reasoning_themes() -> None:
    event = DetectedEvent(
        raw_event_id="evt-thesis",
        event_type="ETF_APPROVAL",
        rule_name="etf_approval",
        confidence=0.9,
        matched_text="Bitcoin approval improves macro liquidity backdrop.",
        detected_at=datetime(2026, 4, 5, tzinfo=UTC),
    )

    tokens = extract_thesis_tokens(event=event, symbol_id="btc_usd")
    fingerprint = build_thesis_fingerprint(
        symbol_id="btc_usd",
        side="buy",
        event_type=event.event_type,
        event_group="etf_news",
        thesis_tokens=tokens,
    )

    assert "approval" in tokens
    assert "macro" in tokens
    assert fingerprint.startswith("thesis-")
