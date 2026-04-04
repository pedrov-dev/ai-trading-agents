"""Readable audit summaries for judges and demo walkthroughs."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuditEvent:
    """One parsed audit-log event."""

    event: str
    status: str
    recorded_at: datetime
    symbol_id: str | None = None
    client_order_id: str | None = None
    message: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "status": self.status,
            "recorded_at": self.recorded_at.isoformat(),
            "symbol_id": self.symbol_id,
            "client_order_id": self.client_order_id,
            "message": self.message,
            "payload": self.payload,
        }


@dataclass(frozen=True)
class AuditSummary:
    """Aggregate audit-log view used in demos and judge reviews."""

    total_events: int
    failure_count: int
    fill_count: int
    status_counts: dict[str, int]
    event_counts: dict[str, int]
    symbol_counts: dict[str, int]
    recent_events: tuple[AuditEvent, ...]
    last_recorded_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_events": self.total_events,
            "failure_count": self.failure_count,
            "fill_count": self.fill_count,
            "status_counts": self.status_counts,
            "event_counts": self.event_counts,
            "symbol_counts": self.symbol_counts,
            "recent_events": [event.to_dict() for event in self.recent_events],
            "last_recorded_at": self.last_recorded_at.isoformat()
            if self.last_recorded_at
            else None,
        }


def load_audit_events(path: str | Path) -> tuple[AuditEvent, ...]:
    """Load Kraken-style JSONL audit events from disk."""
    audit_path = Path(path)
    if not audit_path.exists():
        return ()

    events: list[AuditEvent] = []
    with audit_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                continue

            request = payload.get("request", {})
            symbol_id = request.get("symbol_id") if isinstance(request, dict) else None
            client_order_id = (
                request.get("client_order_id") if isinstance(request, dict) else None
            )
            event = str(payload.get("event", "unknown_event"))
            status = str(payload.get("status", "unknown"))
            recorded_at = _parse_datetime(payload.get("recorded_at"))
            message = f"{event} for {symbol_id or 'unknown_symbol'} ({status})"
            events.append(
                AuditEvent(
                    event=event,
                    status=status,
                    recorded_at=recorded_at,
                    symbol_id=symbol_id,
                    client_order_id=client_order_id,
                    message=message,
                    payload=payload,
                )
            )

    return tuple(sorted(events, key=lambda item: item.recorded_at))


def build_audit_summary(
    events: tuple[AuditEvent, ...] | list[AuditEvent],
    *,
    recent_event_limit: int = 5,
) -> AuditSummary:
    """Summarize audit events into judge-friendly counts and recent activity."""
    ordered = tuple(sorted(events, key=lambda item: item.recorded_at))
    status_counts = Counter(event.status for event in ordered)
    event_counts = Counter(event.event for event in ordered)
    symbol_counts = Counter(event.symbol_id for event in ordered if event.symbol_id)
    failure_count = sum(1 for event in ordered if event.status == "failed")
    fill_count = sum(
        1
        for event in ordered
        if event.status in {"filled", "simulated"}
        or (isinstance(event.payload.get("fill"), dict) and event.payload.get("fill"))
    )

    return AuditSummary(
        total_events=len(ordered),
        failure_count=failure_count,
        fill_count=fill_count,
        status_counts=dict(status_counts),
        event_counts=dict(event_counts),
        symbol_counts=dict(symbol_counts),
        recent_events=ordered[-recent_event_limit:],
        last_recorded_at=ordered[-1].recorded_at if ordered else None,
    )


def build_audit_summary_from_file(
    path: str | Path,
    *,
    recent_event_limit: int = 5,
) -> AuditSummary:
    """Load and summarize a Kraken JSONL audit log in one call."""
    return build_audit_summary(
        load_audit_events(path),
        recent_event_limit=recent_event_limit,
    )


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    return datetime.now(UTC)
