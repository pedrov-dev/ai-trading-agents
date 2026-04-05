"""Confidence calibration analytics derived from resolved trade outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from monitoring.trade_journal import TradeJournalEntry


@dataclass(frozen=True)
class ConfidenceBucket:
    """One confidence band used for calibration reporting."""

    bucket_label: str
    bucket_start: float
    bucket_end: float
    average_confidence: float
    hit_rate: float
    sample_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "bucket_label": self.bucket_label,
            "bucket_start": self.bucket_start,
            "bucket_end": self.bucket_end,
            "average_confidence": self.average_confidence,
            "hit_rate": self.hit_rate,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True)
class CalibrationSummary:
    """Aggregate confidence-calibration metrics for resolved predictions."""

    resolved_prediction_count: int
    unresolved_prediction_count: int
    hit_rate: float
    brier_score: float
    buckets: tuple[ConfidenceBucket, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolved_prediction_count": self.resolved_prediction_count,
            "unresolved_prediction_count": self.unresolved_prediction_count,
            "hit_rate": self.hit_rate,
            "brier_score": self.brier_score,
            "buckets": [bucket.to_dict() for bucket in self.buckets],
        }


def build_calibration_summary(
    entries: tuple[TradeJournalEntry, ...] | list[TradeJournalEntry],
) -> CalibrationSummary:
    """Compute directional confidence calibration from trade-journal entries."""
    resolved_entries = [
        entry
        for entry in entries
        if entry.actual_move is not None
        and entry.prediction_correct is not None
        and entry.confidence_score is not None
        and entry.expected_move is not None
    ]
    latest_open_entries_by_position: dict[str, TradeJournalEntry] = {}
    for entry in entries:
        position_key = entry.position_id or entry.symbol_id
        if (
            entry.position_side is None
            or entry.position_quantity <= 0
            or entry.confidence_score is None
            or entry.expected_move is None
        ):
            latest_open_entries_by_position.pop(position_key, None)
            continue
        latest_open_entries_by_position[position_key] = entry

    if not resolved_entries:
        return CalibrationSummary(
            resolved_prediction_count=0,
            unresolved_prediction_count=len(latest_open_entries_by_position),
            hit_rate=0.0,
            brier_score=0.0,
            buckets=(),
        )

    outcomes = [1.0 if entry.prediction_correct else 0.0 for entry in resolved_entries]
    confidences = [
        entry.confidence_score if entry.confidence_score is not None else 0.0
        for entry in resolved_entries
    ]
    brier_score = round(
        sum(
            (confidence - outcome) ** 2
            for confidence, outcome in zip(confidences, outcomes, strict=True)
        )
        / len(resolved_entries),
        4,
    )
    hit_rate = round(sum(outcomes) / len(outcomes), 4)

    bucket_rows: list[ConfidenceBucket] = []
    for bucket_index in range(10):
        bucket_start = round(bucket_index / 10, 1)
        bucket_end = round(bucket_start + 0.1, 1)
        bucket_entries = [
            entry
            for entry in resolved_entries
            if _bucket_index(
                entry.confidence_score if entry.confidence_score is not None else 0.0
            )
            == bucket_index
        ]
        if not bucket_entries:
            continue
        bucket_hit_rate = round(
            sum(1.0 if entry.prediction_correct else 0.0 for entry in bucket_entries)
            / len(bucket_entries),
            4,
        )
        average_confidence = round(
            sum(
                entry.confidence_score if entry.confidence_score is not None else 0.0
                for entry in bucket_entries
            )
            / len(bucket_entries),
            4,
        )
        bucket_rows.append(
            ConfidenceBucket(
                bucket_label=f"{bucket_start:.1f}-{bucket_end:.1f}",
                bucket_start=bucket_start,
                bucket_end=bucket_end,
                average_confidence=average_confidence,
                hit_rate=bucket_hit_rate,
                sample_count=len(bucket_entries),
            )
        )

    return CalibrationSummary(
        resolved_prediction_count=len(resolved_entries),
        unresolved_prediction_count=len(latest_open_entries_by_position),
        hit_rate=hit_rate,
        brier_score=brier_score,
        buckets=tuple(bucket_rows),
    )


def _bucket_index(confidence_score: float) -> int:
    bounded = min(max(confidence_score, 0.0), 1.0)
    if bounded == 1.0:
        return 9
    return int(bounded * 10)
