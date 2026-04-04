"""Persistent trade and position journaling for local runtime continuity."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from agent.portfolio import (
    LocalPortfolioStateProvider,
    PortfolioSnapshot,
    Position,
)
from execution.orders import ExecutionResult
from storage.local_runtime import JsonlFileStore

_CLOSE_EVENT_TYPES = {"partial_exit", "full_exit", "reverse"}


@dataclass(frozen=True)
class TradeJournalEntry:
    """One durable portfolio-affecting fill record."""

    entry_id: str
    recorded_at: datetime
    symbol_id: str
    side: str
    event_type: str
    quantity: float
    price: float
    realized_pnl_usd: float = 0.0
    position_side: str | None = None
    position_quantity: float = 0.0
    position_entry_price: float | None = None
    intent_id: str | None = None
    client_order_id: str | None = None
    notes: tuple[str, ...] = ()

    @classmethod
    def from_execution_result(
        cls,
        *,
        execution_result: ExecutionResult,
        before_portfolio: PortfolioSnapshot,
        after_portfolio: PortfolioSnapshot,
        notes: tuple[str, ...] = (),
    ) -> TradeJournalEntry:
        """Build a journal row from one successful execution result."""
        fill = execution_result.fill
        if fill is None:
            raise ValueError("A fill is required to build a trade journal entry.")

        before_position = before_portfolio.position_for_symbol(
            execution_result.request.symbol_id
        )
        after_position = after_portfolio.position_for_symbol(
            execution_result.request.symbol_id
        )
        realized_change = round(
            after_portfolio.realized_pnl_today - before_portfolio.realized_pnl_today,
            2,
        )
        return cls(
            entry_id=(
                f"{execution_result.request.client_order_id}:"
                f"{fill.filled_at.isoformat()}"
            ),
            recorded_at=fill.filled_at,
            symbol_id=execution_result.request.symbol_id,
            side=str(execution_result.request.side),
            event_type=_derive_event_type(
                before_position=before_position,
                after_position=after_position,
            ),
            quantity=round(fill.filled_quantity, 8),
            price=round(fill.average_price, 8),
            realized_pnl_usd=realized_change,
            position_side=after_position.side if after_position is not None else None,
            position_quantity=(
                round(after_position.quantity, 8) if after_position is not None else 0.0
            ),
            position_entry_price=(
                round(after_position.entry_price, 8)
                if after_position is not None
                else None
            ),
            intent_id=execution_result.request.intent_id,
            client_order_id=execution_result.request.client_order_id,
            notes=tuple(str(item) for item in notes),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TradeJournalEntry:
        """Rehydrate a journal entry from JSON-loaded data."""
        position_entry_price = payload.get("position_entry_price")
        position_side = payload.get("position_side")
        intent_id = payload.get("intent_id")
        client_order_id = payload.get("client_order_id")
        return cls(
            entry_id=str(payload.get("entry_id", "journal-entry")),
            recorded_at=_parse_datetime(payload.get("recorded_at")),
            symbol_id=str(payload.get("symbol_id", "unknown_symbol")),
            side=str(payload.get("side", "buy")),
            event_type=str(payload.get("event_type", "entry")),
            quantity=float(payload.get("quantity", 0.0)),
            price=float(payload.get("price", 0.0)),
            realized_pnl_usd=float(payload.get("realized_pnl_usd", 0.0)),
            position_side=str(position_side) if position_side is not None else None,
            position_quantity=float(payload.get("position_quantity", 0.0)),
            position_entry_price=(
                float(position_entry_price)
                if position_entry_price is not None
                else None
            ),
            intent_id=str(intent_id) if intent_id is not None else None,
            client_order_id=(
                str(client_order_id) if client_order_id is not None else None
            ),
            notes=tuple(str(item) for item in payload.get("notes", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable journal row."""
        return {
            "entry_id": self.entry_id,
            "recorded_at": self.recorded_at.isoformat(),
            "symbol_id": self.symbol_id,
            "side": self.side,
            "event_type": self.event_type,
            "quantity": self.quantity,
            "price": self.price,
            "realized_pnl_usd": self.realized_pnl_usd,
            "position_side": self.position_side,
            "position_quantity": self.position_quantity,
            "position_entry_price": self.position_entry_price,
            "intent_id": self.intent_id,
            "client_order_id": self.client_order_id,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class TradeJournalSummary:
    """Aggregate view of the persistent trade journal."""

    total_entries: int
    closed_trade_count: int
    open_position_count: int
    realized_pnl_usd: float
    win_count: int
    loss_count: int
    event_counts: dict[str, int]
    symbol_counts: dict[str, int]
    open_positions: dict[str, dict[str, Any]] = field(default_factory=dict)
    recent_entries: tuple[TradeJournalEntry, ...] = ()
    last_recorded_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_entries": self.total_entries,
            "closed_trade_count": self.closed_trade_count,
            "open_position_count": self.open_position_count,
            "realized_pnl_usd": self.realized_pnl_usd,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "event_counts": self.event_counts,
            "symbol_counts": self.symbol_counts,
            "open_positions": self.open_positions,
            "recent_entries": [entry.to_dict() for entry in self.recent_entries],
            "last_recorded_at": self.last_recorded_at.isoformat()
            if self.last_recorded_at
            else None,
        }


class LocalTradeJournal:
    """Append-only local journal for fills that modify portfolio state."""

    def __init__(self, path: str | Path) -> None:
        self._store = JsonlFileStore(path)

    @property
    def path(self) -> Path:
        return self._store.path

    def record_entry(self, entry: TradeJournalEntry) -> None:
        self._store.append(entry.to_dict())

    def record_entries(self, entries: tuple[TradeJournalEntry, ...]) -> int:
        return self._store.append_many(entry.to_dict() for entry in entries)

    def load_entries(self) -> tuple[TradeJournalEntry, ...]:
        return load_trade_journal_entries(self.path)

    def replay_into(
        self,
        portfolio_provider: LocalPortfolioStateProvider,
    ) -> PortfolioSnapshot:
        """Replay journaled fills to restore local state after a restart."""
        snapshot = portfolio_provider.get_portfolio_snapshot()
        for entry in self.load_entries():
            if entry.side not in {"buy", "sell"}:
                continue
            side: Literal["buy", "sell"] = (
                "buy" if entry.side == "buy" else "sell"
            )
            snapshot = portfolio_provider.record_fill(
                symbol_id=entry.symbol_id,
                side=side,
                quantity=entry.quantity,
                price=entry.price,
                filled_at=entry.recorded_at,
            )
        return snapshot

    def build_summary(self, *, recent_entry_limit: int = 5) -> TradeJournalSummary:
        return build_trade_journal_summary_from_file(
            self.path,
            recent_entry_limit=recent_entry_limit,
        )


def load_trade_journal_entries(path: str | Path) -> tuple[TradeJournalEntry, ...]:
    """Load journal entries from disk."""
    journal_path = Path(path)
    if not journal_path.exists():
        return ()

    entries: list[TradeJournalEntry] = []
    with journal_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                continue
            entries.append(TradeJournalEntry.from_dict(payload))

    return tuple(sorted(entries, key=lambda item: item.recorded_at))


def build_trade_journal_summary(
    entries: tuple[TradeJournalEntry, ...] | list[TradeJournalEntry],
    *,
    recent_entry_limit: int = 5,
) -> TradeJournalSummary:
    """Summarize trade journal rows for judge-friendly reporting."""
    ordered = tuple(sorted(entries, key=lambda item: item.recorded_at))
    event_counts = Counter(entry.event_type for entry in ordered)
    symbol_counts = Counter(entry.symbol_id for entry in ordered)
    close_entries = [entry for entry in ordered if entry.event_type in _CLOSE_EVENT_TYPES]
    open_positions: dict[str, dict[str, Any]] = {}

    for entry in ordered:
        if entry.position_side is None or entry.position_quantity <= 0:
            open_positions.pop(entry.symbol_id, None)
            continue
        open_positions[entry.symbol_id] = {
            "side": entry.position_side,
            "quantity": entry.position_quantity,
            "entry_price": entry.position_entry_price,
            "last_event_type": entry.event_type,
            "recorded_at": entry.recorded_at.isoformat(),
        }

    return TradeJournalSummary(
        total_entries=len(ordered),
        closed_trade_count=len(close_entries),
        open_position_count=len(open_positions),
        realized_pnl_usd=round(sum(entry.realized_pnl_usd for entry in ordered), 2),
        win_count=sum(1 for entry in close_entries if entry.realized_pnl_usd > 0),
        loss_count=sum(1 for entry in close_entries if entry.realized_pnl_usd < 0),
        event_counts=dict(event_counts),
        symbol_counts=dict(symbol_counts),
        open_positions=open_positions,
        recent_entries=ordered[-recent_entry_limit:],
        last_recorded_at=ordered[-1].recorded_at if ordered else None,
    )


def build_trade_journal_summary_from_file(
    path: str | Path,
    *,
    recent_entry_limit: int = 5,
) -> TradeJournalSummary:
    """Load and summarize the journal in one call."""
    return build_trade_journal_summary(
        load_trade_journal_entries(path),
        recent_entry_limit=recent_entry_limit,
    )


def _derive_event_type(
    *,
    before_position: Position | None,
    after_position: Position | None,
) -> str:
    if before_position is None and after_position is not None:
        return "entry"
    if before_position is not None and after_position is None:
        return "full_exit"
    if before_position is None or after_position is None:
        return "entry"
    if before_position.side != after_position.side:
        return "reverse"
    if after_position.quantity > before_position.quantity:
        return "scale_in"
    if after_position.quantity < before_position.quantity:
        return "partial_exit"
    return "rebalance"


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    return datetime.now(UTC)
