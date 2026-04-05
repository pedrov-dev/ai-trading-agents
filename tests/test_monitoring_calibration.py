from datetime import UTC, datetime

from monitoring.calibration import build_calibration_summary
from monitoring.trade_journal import TradeJournalEntry

_DEF_TIME = datetime(2026, 4, 3, 12, 0, tzinfo=UTC)


def test_build_calibration_summary_groups_hits_by_confidence_bucket() -> None:
    entries = (
        TradeJournalEntry(
            entry_id="resolved-1",
            recorded_at=_DEF_TIME,
            symbol_id="btc_usd",
            side="sell",
            event_type="full_exit",
            quantity=0.01,
            price=52_000.0,
            confidence_score=0.9,
            expected_move="up",
            actual_move="up",
            prediction_correct=True,
        ),
        TradeJournalEntry(
            entry_id="resolved-2",
            recorded_at=_DEF_TIME.replace(minute=5),
            symbol_id="eth_usd",
            side="buy",
            event_type="full_exit",
            quantity=0.1,
            price=2_950.0,
            confidence_score=0.9,
            expected_move="down",
            actual_move="up",
            prediction_correct=False,
        ),
        TradeJournalEntry(
            entry_id="resolved-3",
            recorded_at=_DEF_TIME.replace(minute=10),
            symbol_id="sol_usd",
            side="sell",
            event_type="full_exit",
            quantity=1.0,
            price=180.0,
            confidence_score=0.6,
            expected_move="up",
            actual_move="up",
            prediction_correct=True,
        ),
        TradeJournalEntry(
            entry_id="resolved-4",
            recorded_at=_DEF_TIME.replace(minute=15),
            symbol_id="xrp_usd",
            side="buy",
            event_type="full_exit",
            quantity=100.0,
            price=0.52,
            confidence_score=0.6,
            expected_move="down",
            actual_move="up",
            prediction_correct=False,
        ),
        TradeJournalEntry(
            entry_id="open-1",
            recorded_at=_DEF_TIME.replace(minute=20),
            symbol_id="ada_usd",
            side="buy",
            event_type="entry",
            quantity=50.0,
            price=0.75,
            confidence_score=0.8,
            expected_move="up",
            position_side="long",
            position_quantity=50.0,
            position_entry_price=0.75,
            position_id="pos-open",
        ),
    )

    summary = build_calibration_summary(entries)

    assert summary.resolved_prediction_count == 4
    assert summary.unresolved_prediction_count == 1
    assert summary.hit_rate == 0.5
    assert summary.brier_score == 0.335
    assert [bucket.bucket_label for bucket in summary.buckets] == ["0.6-0.7", "0.9-1.0"]
    assert summary.buckets[0].hit_rate == 0.5
    assert summary.buckets[1].hit_rate == 0.5
    assert summary.buckets[1].average_confidence == 0.9


def test_build_calibration_summary_returns_empty_metrics_without_resolved_predictions() -> None:
    entries = (
        TradeJournalEntry(
            entry_id="open-1",
            recorded_at=_DEF_TIME,
            symbol_id="btc_usd",
            side="buy",
            event_type="entry",
            quantity=0.01,
            price=50_000.0,
            confidence_score=0.82,
            expected_move="up",
            position_side="long",
            position_quantity=0.01,
            position_entry_price=50_000.0,
            position_id="pos-open",
        ),
    )

    summary = build_calibration_summary(entries)

    assert summary.resolved_prediction_count == 0
    assert summary.unresolved_prediction_count == 1
    assert summary.hit_rate == 0.0
    assert summary.brier_score == 0.0
    assert summary.buckets == ()
