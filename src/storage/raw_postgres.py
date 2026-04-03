"""Postgres repositories for ingestion runs and raw events."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any


class PostgresIngestionRunsRepository:
    """DB-API based repository for ingestion_runs lifecycle operations."""

    def __init__(self, connection_factory: Callable[[], Any]) -> None:
        self._connection_factory = connection_factory

    def start_run(self, source_type: str, started_at: datetime) -> str:
        query = """
        INSERT INTO ingestion_runs (source_type, started_at, status)
        VALUES (%s, %s, %s)
        RETURNING id::text
        """
        with self._connection_factory() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (source_type, started_at, "running"))
                row = cursor.fetchone()
            connection.commit()

        if not row:
            raise RuntimeError("Failed to create ingestion run")
        return str(row[0])

    def finish_run(
        self,
        run_id: str,
        *,
        finished_at: datetime,
        status: str,
        fetched_count: int,
        inserted_count: int,
        duplicate_count: int,
        error_summary: str | None,
    ) -> None:
        query = """
        UPDATE ingestion_runs
        SET finished_at = %s,
            status = %s,
            fetched_count = %s,
            inserted_count = %s,
            duplicate_count = %s,
            error_summary = %s
        WHERE id = %s::uuid
        """
        with self._connection_factory() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    query,
                    (
                        finished_at,
                        status,
                        fetched_count,
                        inserted_count,
                        duplicate_count,
                        error_summary,
                        run_id,
                    ),
                )
                rowcount = cursor.rowcount
                if rowcount != 1:
                    raise RuntimeError(
                        "Failed to finalize ingestion run "
                        f"{run_id}: expected 1 row updated, got {rowcount}"
                    )
            connection.commit()


class PostgresRawEventsRepository:
    """DB-API based repository for idempotent raw_events inserts."""

    def __init__(self, connection_factory: Callable[[], Any]) -> None:
        self._connection_factory = connection_factory

    def insert_raw_event(
        self,
        *,
        source_type: str,
        source_id: str,
        observed_at: datetime,
        event_time: datetime | None,
        dedup_hash: str,
        payload_preview: dict[str, Any],
        object_key: str,
        ingest_run_id: str,
        status: str,
    ) -> bool:
        query = """
        INSERT INTO raw_events (
            source_type,
            source_id,
            observed_at,
            event_time,
            dedup_hash,
            payload_preview,
            object_key,
            ingest_run_id,
            status
        )
        VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s::uuid, %s)
        ON CONFLICT (source_type, dedup_hash) DO NOTHING
        RETURNING id::text
        """

        payload_json = _dumps_json(payload_preview)

        with self._connection_factory() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    query,
                    (
                        source_type,
                        source_id,
                        observed_at,
                        event_time,
                        dedup_hash,
                        payload_json,
                        object_key,
                        ingest_run_id,
                        status,
                    ),
                )
                row = cursor.fetchone()
            connection.commit()

        if row is None:
            return False
        return True

    def transition_raw_event_status(
        self,
        *,
        source_type: str,
        dedup_hash: str,
        status: str,
    ) -> None:
        query = """
        UPDATE raw_events
        SET status = %s
        WHERE source_type = %s
          AND dedup_hash = %s
        """
        with self._connection_factory() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (status, source_type, dedup_hash))
                rowcount = cursor.rowcount
                if rowcount != 1:
                    raise RuntimeError(
                        "Failed to transition raw_event status "
                        f"for source_type={source_type} dedup_hash={dedup_hash}: "
                        f"expected 1 row updated, got {rowcount}"
                    )
            connection.commit()

    def transition_pending_raw_events_for_run(
        self,
        *,
        ingest_run_id: str,
        status: str,
    ) -> int:
        query = """
        UPDATE raw_events
        SET status = %s
        WHERE ingest_run_id = %s::uuid
          AND status = 'pending'
        """
        with self._connection_factory() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (status, ingest_run_id))
                rowcount = cursor.rowcount
            connection.commit()
        return int(rowcount)

    def fetch_raw_events_for_classification(
        self,
        *,
        source_type: str,
        status: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        query = """
        SELECT id::text AS id,
               source_type,
               source_id,
               dedup_hash,
               payload_preview
        FROM raw_events
        WHERE source_type = %s
          AND status = %s
        ORDER BY observed_at ASC
        LIMIT %s
        """

        with self._connection_factory() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (source_type, status, limit))
                rows = cursor.fetchall()

        events: list[dict[str, Any]] = []
        for row in rows:
            events.append(
                {
                    "id": str(row[0]),
                    "source_type": row[1],
                    "source_id": row[2],
                    "dedup_hash": row[3],
                    "payload_preview": row[4],
                }
            )
        return events


def _dumps_json(value: dict[str, Any]) -> str:
    import json

    return json.dumps(value, separators=(",", ":"), sort_keys=True)
