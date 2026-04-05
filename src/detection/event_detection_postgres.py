from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any, Protocol

from detection.event_detection import DetectedEvent


class EventDetectionRepository(Protocol):
    """Persist detected events from raw text."""

    def insert_detected_event(
        self,
        *,
        raw_event_id: str,
        event_type: str,
        rule_name: str,
        confidence: float,
        detected_at: datetime,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        ...

    def list_detected_events(self) -> tuple[DetectedEvent, ...]:
        ...


class PostgresEventDetectionRepository:
    """DB-API based repository for detected events."""

    def __init__(self, connection_factory: Callable[[], Any]) -> None:
        self._connection_factory = connection_factory

    def list_detected_events(self) -> tuple[DetectedEvent, ...]:
        query = """
        SELECT raw_event_id::text,
               event_type,
               rule_name,
               confidence,
               detected_at,
               metadata
        FROM detected_events
        ORDER BY detected_at ASC
        """
        with self._connection_factory() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()

        events: list[DetectedEvent] = []
        for row in rows:
            metadata = row[5] if isinstance(row[5], dict) else {}
            matched_text = metadata.get("matched_text") if metadata else None
            novelty_score = metadata.get("novelty_score") if metadata else None
            repeat_count = metadata.get("repeat_count") if metadata else 0
            narrative_key = metadata.get("narrative_key") if metadata else None
            events.append(
                DetectedEvent(
                    raw_event_id=str(row[0]),
                    event_type=str(row[1]),
                    rule_name=str(row[2]),
                    confidence=float(row[3]),
                    detected_at=row[4],
                    matched_text=str(matched_text) if matched_text is not None else None,
                    novelty_score=float(novelty_score) if novelty_score is not None else None,
                    repeat_count=int(repeat_count or 0),
                    narrative_key=str(narrative_key) if narrative_key is not None else None,
                )
            )
        return tuple(events)

    def insert_detected_event(
        self,
        *,
        raw_event_id: str,
        event_type: str,
        rule_name: str,
        confidence: float,
        detected_at: datetime,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        query = """
        INSERT INTO detected_events (
            raw_event_id,
            event_type,
            rule_name,
            confidence,
            detected_at,
            metadata
        )
        VALUES (%s, %s, %s, %s, %s, %s::jsonb)
        """

        metadata_json = None
        if metadata is not None:
            import json

            metadata_json = json.dumps(metadata, separators=(",", ":"), sort_keys=True)

        with self._connection_factory() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    query,
                    (
                        raw_event_id,
                        event_type,
                        rule_name,
                        confidence,
                        detected_at,
                        metadata_json,
                    ),
                )
            connection.commit()
