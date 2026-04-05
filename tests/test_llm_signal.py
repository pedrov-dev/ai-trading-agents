from datetime import UTC, datetime

from agent.llm_signal import generate_llm_signal
from detection.event_detection import DetectedEvent
from ingestion.prices_ingestion import PriceQuote


def test_generate_llm_signal_is_currently_a_no_op() -> None:
    event = DetectedEvent(
        raw_event_id="evt-llm",
        event_type="ETF_APPROVAL",
        rule_name="etf_approval",
        confidence=0.9,
        matched_text="bitcoin etf approval",
        detected_at=datetime(2026, 4, 5, tzinfo=UTC),
    )
    quote = PriceQuote(
        symbol_id="btc_usd",
        current=68_000.0,
        open=66_000.0,
        high=68_400.0,
        low=65_800.0,
        prev_close=65_500.0,
        timestamp=1712275200,
        asset_class="spot",
    )

    result = generate_llm_signal(event=event, quote=quote)

    assert result.enabled is False
    assert result.confidence_delta == 0.0
    assert any("not enabled" in reason.lower() for reason in result.rationale)
