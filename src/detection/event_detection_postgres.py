from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any, Protocol


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


class PostgresEventDetectionRepository:
    """DB-API based repository for detected events."""

    def __init__(self, connection_factory: Callable[[], Any]) -> None:
        self._connection_factory = connection_factory

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
