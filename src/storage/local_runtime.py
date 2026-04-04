"""Local in-memory/file-backed adapters for demo and dry-run execution."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from detection.event_detection import DetectedEvent


class InMemoryIngestionRunsRepository:
    """Keep ingestion run state in memory for local dry runs."""

    def __init__(self) -> None:
        self._runs: dict[str, dict[str, Any]] = {}

    def start_run(self, source_type: str, started_at: datetime) -> str:
        run_id = f"run-{uuid4().hex[:12]}"
        self._runs[run_id] = {
            "run_id": run_id,
            "source_type": source_type,
            "started_at": started_at,
            "status": "running",
        }
        return run_id

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
        if run_id not in self._runs:
            raise RuntimeError(f"Unknown ingestion run id: {run_id}")
        self._runs[run_id].update(
            {
                "finished_at": finished_at,
                "status": status,
                "fetched_count": fetched_count,
                "inserted_count": inserted_count,
                "duplicate_count": duplicate_count,
                "error_summary": error_summary,
            }
        )


class InMemoryRawEventsRepository:
    """Track raw events in memory while preserving the production repository contract."""

    def __init__(self) -> None:
        self._rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        self._ordered_keys: list[tuple[str, str]] = []

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
        key = (source_type, dedup_hash)
        if key in self._rows_by_key:
            return False

        row = {
            "id": f"raw-{uuid4().hex[:12]}",
            "source_type": source_type,
            "source_id": source_id,
            "observed_at": observed_at,
            "event_time": event_time,
            "dedup_hash": dedup_hash,
            "payload_preview": dict(payload_preview),
            "object_key": object_key,
            "ingest_run_id": ingest_run_id,
            "status": status,
        }
        self._rows_by_key[key] = row
        self._ordered_keys.append(key)
        return True

    def transition_raw_event_status(
        self,
        *,
        source_type: str,
        dedup_hash: str,
        status: str,
    ) -> None:
        row = self._rows_by_key.get((source_type, dedup_hash))
        if row is None:
            raise RuntimeError(
                "Failed to transition raw_event status for "
                f"source_type={source_type} dedup_hash={dedup_hash}"
            )
        row["status"] = status

    def transition_pending_raw_events_for_run(
        self,
        *,
        ingest_run_id: str,
        status: str,
    ) -> int:
        updated = 0
        for row in self._rows_by_key.values():
            if row["ingest_run_id"] == ingest_run_id and row["status"] == "pending":
                row["status"] = status
                updated += 1
        return updated

    def fetch_raw_events_for_classification(
        self,
        *,
        source_type: str,
        status: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for key in self._ordered_keys:
            row = self._rows_by_key[key]
            if row["source_type"] != source_type or row["status"] != status:
                continue
            rows.append(
                {
                    "id": row["id"],
                    "source_type": row["source_type"],
                    "source_id": row["source_id"],
                    "dedup_hash": row["dedup_hash"],
                    "payload_preview": dict(row["payload_preview"]),
                }
            )
            if len(rows) >= limit:
                break
        return rows


class LocalFileObjectStore:
    """Persist raw payloads as gzip-compressed JSON files for local inspection."""

    def __init__(self, base_path: str | Path) -> None:
        self._base_path = Path(base_path)

    def put_json_gzip(
        self,
        *,
        key: str,
        payload: dict[str, Any],
        metadata: dict[str, str],
    ) -> str:
        target_path = self._base_path / key
        target_path.parent.mkdir(parents=True, exist_ok=True)
        document = {"payload": payload, "metadata": metadata}
        with gzip.open(target_path, "wt", encoding="utf-8") as handle:
            json.dump(document, handle, sort_keys=True)
        return key


class InMemoryEventDetectionRepository:
    """Collect detected events in memory for the strategy and demo summary."""

    def __init__(self) -> None:
        self._events: list[DetectedEvent] = []

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
        matched_text = None
        if metadata is not None:
            candidate = metadata.get("matched_text")
            matched_text = str(candidate) if candidate is not None else None

        self._events.append(
            DetectedEvent(
                raw_event_id=raw_event_id,
                event_type=event_type,
                rule_name=rule_name,
                confidence=confidence,
                matched_text=matched_text,
                detected_at=detected_at,
            )
        )

    def list_detected_events(self) -> tuple[DetectedEvent, ...]:
        return tuple(self._events)


class JsonlFileStore:
    """Append JSON-serializable records to a JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, record: Mapping[str, Any]) -> None:
        self.append_many((record,))

    def append_many(self, records: Iterable[Mapping[str, Any]]) -> int:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with self._path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(dict(record), sort_keys=True) + "\n")
                written += 1
        return written

    def write_json(self, payload: Mapping[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )
