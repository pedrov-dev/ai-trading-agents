from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from detection.event_detection import EventDetector
from detection.event_detection_postgres import EventDetectionRepository
from storage.raw_ingestion import RawEventsRepository


@dataclass(frozen=True)
class RawEventForClassification:
    id: str
    source_type: str
    source_id: str
    dedup_hash: str
    payload_preview: dict[str, Any]


class EventDetectionService:
    """Run event detection on raw events and store matches."""

    def __init__(
        self,
        *,
        detector: EventDetector,
        raw_events_repository: RawEventsRepository,
        event_detection_repository: EventDetectionRepository,
    ) -> None:
        self._detector = detector
        self._raw_events_repository = raw_events_repository
        self._event_detection_repository = event_detection_repository

    def classify_pending_events(
        self,
        *,
        source_type: str = "rss",
        batch_size: int = 100,
    ) -> int:
        pending = self._raw_events_repository.fetch_raw_events_for_classification(
            source_type=source_type,
            status="ok",
            limit=batch_size,
        )

        processed = 0
        for row in pending:
            raw_event = RawEventForClassification(
                id=str(row["id"]),
                source_type=str(row["source_type"]),
                source_id=str(row["source_id"]),
                dedup_hash=str(row["dedup_hash"]),
                payload_preview=row["payload_preview"] or {},
            )

            matches = self._detector.detect(
                source_type=raw_event.source_type,
                payload_preview=raw_event.payload_preview,
            )

            if matches:
                for match in matches:
                    self._event_detection_repository.insert_detected_event(
                        raw_event_id=raw_event.id,
                        event_type=match.event_type,
                        rule_name=match.rule_name,
                        confidence=match.confidence,
                        detected_at=match.detected_at or datetime.now(UTC),
                        metadata={
                            "matched_text": match.matched_text,
                        },
                    )
                self._raw_events_repository.transition_raw_event_status(
                    source_type=raw_event.source_type,
                    dedup_hash=raw_event.dedup_hash,
                    status="classified",
                )
            else:
                self._raw_events_repository.transition_raw_event_status(
                    source_type=raw_event.source_type,
                    dedup_hash=raw_event.dedup_hash,
                    status="no_event",
                )

            processed += 1

        return processed
