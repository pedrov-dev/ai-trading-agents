from datetime import UTC, datetime

from agent.event_signal import (
    apply_event_score_adjustments,
    event_novelty_adjustment,
    signal_time_decay_factor,
)
from detection.event_detection import DetectedEvent


def test_event_novelty_adjustment_penalizes_repeated_narratives() -> None:
    event = DetectedEvent(
        raw_event_id="evt-repeat",
        event_type="ETF_APPROVAL",
        rule_name="etf_approval",
        confidence=0.88,
        matched_text="bitcoin etf approval rumor again",
        detected_at=datetime(2026, 4, 5, tzinfo=UTC),
        novelty_score=0.2,
        repeat_count=4,
        narrative_key="etf-approval-rumor",
    )

    novelty_score, rationale = event_novelty_adjustment(event)

    assert novelty_score == 0.2
    assert rationale is not None
    assert "repeated narrative" in rationale.lower()


def test_signal_time_decay_factor_weakens_stale_signals() -> None:
    factor, rationale = signal_time_decay_factor(
        age_minutes=240,
        enabled=True,
        half_life_minutes=60,
        floor=0.35,
    )

    assert factor < 1.0
    assert rationale is not None
    assert "time decay" in rationale.lower()


def test_apply_event_score_adjustments_handles_shocks_and_breakouts() -> None:
    shock_score, shock_rationale = apply_event_score_adjustments(
        event_type="SECURITY_INCIDENT",
        score=0.6,
        price_confirmed=False,
        volume_confirmed=False,
        volume_unavailable=True,
    )
    breakout_score, breakout_rationale = apply_event_score_adjustments(
        event_type="TECHNICAL_BREAKOUT",
        score=0.8,
        price_confirmed=True,
        volume_confirmed=False,
        volume_unavailable=False,
    )

    assert shock_score > 0.6
    assert any("shock-event severity" in item.lower() for item in shock_rationale)
    assert breakout_score < 0.8
    assert any("confirming volume spike" in item.lower() for item in breakout_rationale)
