"""Persistent trade and position journaling for local runtime continuity."""

from __future__ import annotations

import json
import math
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
from agent.signals import MoveDirection
from detection.event_types import event_performance_group
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
    confidence_score: float | None = None
    expected_move: MoveDirection | None = None
    actual_move: MoveDirection | None = None
    prediction_correct: bool | None = None
    realized_pnl_usd: float = 0.0
    position_side: str | None = None
    position_quantity: float = 0.0
    position_entry_price: float | None = None
    position_id: str | None = None
    signal_id: str | None = None
    raw_event_id: str | None = None
    source_event_type: str | None = None
    source_event_group: str | None = None
    exit_horizon_label: str | None = None
    max_hold_minutes: int | None = None
    exit_due_at: datetime | None = None
    intent_id: str | None = None
    client_order_id: str | None = None
    realized_return_fraction: float | None = None
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
            execution_result.request.symbol_id,
            position_id=execution_result.request.position_id,
        )
        after_position = after_portfolio.position_for_symbol(
            execution_result.request.symbol_id,
            position_id=execution_result.request.position_id,
        )
        realized_change = round(
            after_portfolio.realized_pnl_today - before_portfolio.realized_pnl_today,
            2,
        )
        journal_event_type = _derive_event_type(
            before_position=before_position,
            after_position=after_position,
        )
        expected_move = (
            before_position.expected_move
            if before_position is not None and before_position.expected_move is not None
            else ("up" if before_position is not None and before_position.side == "long" else "down")
        )
        confidence_score = (
            before_position.confidence_score
            if before_position is not None and before_position.confidence_score is not None
            else round(execution_result.request.score, 4)
        )
        actual_move = (
            _derive_actual_move(before_position=before_position, exit_price=fill.average_price)
            if journal_event_type in _CLOSE_EVENT_TYPES
            else None
        )
        prediction_correct = expected_move == actual_move if actual_move is not None else None
        return cls(
            entry_id=(
                f"{execution_result.request.client_order_id}:"
                f"{fill.filled_at.isoformat()}"
            ),
            recorded_at=fill.filled_at,
            symbol_id=execution_result.request.symbol_id,
            side=str(execution_result.request.side),
            event_type=journal_event_type,
            quantity=round(fill.filled_quantity, 8),
            price=round(fill.average_price, 8),
            confidence_score=confidence_score,
            expected_move=expected_move,
            actual_move=actual_move,
            prediction_correct=prediction_correct,
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
            position_id=(
                execution_result.request.position_id
                or (after_position.position_id if after_position is not None else None)
                or (before_position.position_id if before_position is not None else None)
            ),
            signal_id=execution_result.request.signal_id,
            raw_event_id=execution_result.request.raw_event_id,
            source_event_type=execution_result.request.event_type,
            source_event_group=event_performance_group(execution_result.request.event_type),
            exit_horizon_label=execution_result.request.exit_horizon_label,
            max_hold_minutes=execution_result.request.max_hold_minutes,
            exit_due_at=execution_result.request.exit_due_at,
            intent_id=execution_result.request.intent_id,
            client_order_id=execution_result.request.client_order_id,
            realized_return_fraction=_calculate_realized_return_fraction(
                before_position=before_position,
                filled_quantity=fill.filled_quantity,
                realized_pnl_usd=realized_change,
                journal_event_type=_derive_event_type(
                    before_position=before_position,
                    after_position=after_position,
                ),
            ),
            notes=tuple(str(item) for item in notes),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TradeJournalEntry:
        """Rehydrate a journal entry from JSON-loaded data."""
        position_entry_price = payload.get("position_entry_price")
        position_side = payload.get("position_side")
        position_id = payload.get("position_id")
        signal_id = payload.get("signal_id")
        raw_event_id = payload.get("raw_event_id")
        source_event_type = payload.get("source_event_type")
        source_event_group = payload.get("source_event_group")
        exit_horizon_label = payload.get("exit_horizon_label")
        max_hold_minutes = payload.get("max_hold_minutes")
        exit_due_at = payload.get("exit_due_at")
        intent_id = payload.get("intent_id")
        client_order_id = payload.get("client_order_id")
        realized_return_fraction = payload.get("realized_return_fraction")
        resolved_source_event_type = (
            str(source_event_type) if source_event_type is not None else None
        )
        resolved_source_event_group = (
            str(source_event_group)
            if source_event_group is not None
            else event_performance_group(resolved_source_event_type)
        )
        return cls(
            entry_id=str(payload.get("entry_id", "journal-entry")),
            recorded_at=_parse_datetime(payload.get("recorded_at")),
            symbol_id=str(payload.get("symbol_id", "unknown_symbol")),
            side=str(payload.get("side", "buy")),
            event_type=str(payload.get("event_type", "entry")),
            quantity=float(payload.get("quantity", 0.0)),
            price=float(payload.get("price", 0.0)),
            confidence_score=(
                float(payload["confidence_score"])
                if payload.get("confidence_score") is not None
                else None
            ),
            expected_move=cast_move_direction(payload.get("expected_move")),
            actual_move=cast_move_direction(payload.get("actual_move")),
            prediction_correct=(
                bool(payload["prediction_correct"])
                if payload.get("prediction_correct") is not None
                else None
            ),
            realized_pnl_usd=float(payload.get("realized_pnl_usd", 0.0)),
            position_side=str(position_side) if position_side is not None else None,
            position_quantity=float(payload.get("position_quantity", 0.0)),
            position_entry_price=(
                float(position_entry_price)
                if position_entry_price is not None
                else None
            ),
            position_id=str(position_id) if position_id is not None else None,
            signal_id=str(signal_id) if signal_id is not None else None,
            raw_event_id=str(raw_event_id) if raw_event_id is not None else None,
            source_event_type=resolved_source_event_type,
            source_event_group=resolved_source_event_group,
            exit_horizon_label=(
                str(exit_horizon_label) if exit_horizon_label is not None else None
            ),
            max_hold_minutes=(
                int(max_hold_minutes) if max_hold_minutes is not None else None
            ),
            exit_due_at=(
                _parse_datetime(exit_due_at) if exit_due_at is not None else None
            ),
            intent_id=str(intent_id) if intent_id is not None else None,
            client_order_id=(
                str(client_order_id) if client_order_id is not None else None
            ),
            realized_return_fraction=(
                float(realized_return_fraction)
                if realized_return_fraction is not None
                else None
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
            "confidence_score": self.confidence_score,
            "expected_move": self.expected_move,
            "actual_move": self.actual_move,
            "prediction_correct": self.prediction_correct,
            "realized_pnl_usd": self.realized_pnl_usd,
            "position_side": self.position_side,
            "position_quantity": self.position_quantity,
            "position_entry_price": self.position_entry_price,
            "position_id": self.position_id,
            "signal_id": self.signal_id,
            "raw_event_id": self.raw_event_id,
            "source_event_type": self.source_event_type,
            "source_event_group": self.source_event_group,
            "exit_horizon_label": self.exit_horizon_label,
            "max_hold_minutes": self.max_hold_minutes,
            "exit_due_at": self.exit_due_at.isoformat() if self.exit_due_at else None,
            "intent_id": self.intent_id,
            "client_order_id": self.client_order_id,
            "realized_return_fraction": self.realized_return_fraction,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class EventTypePerformance:
    """Aggregate realized performance metrics for one signal event bucket."""

    avg_return: float
    hit_rate: float
    sharpe: float
    trade_count: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "avg_return": self.avg_return,
            "hit_rate": self.hit_rate,
            "sharpe": self.sharpe,
            "trade_count": self.trade_count,
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
    source_event_counts: dict[str, int] = field(default_factory=dict)
    event_performance: dict[str, EventTypePerformance] = field(default_factory=dict)
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
            "source_event_counts": self.source_event_counts,
            "event_performance": {
                event_type: metrics.to_dict()
                for event_type, metrics in self.event_performance.items()
            },
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
                position_id=entry.position_id,
                source_signal_id=entry.signal_id,
                raw_event_id=entry.raw_event_id,
                event_type=entry.source_event_type,
                exit_horizon_label=entry.exit_horizon_label,
                max_hold_minutes=entry.max_hold_minutes,
                exit_due_at=entry.exit_due_at,
                confidence_score=entry.confidence_score,
                expected_move=entry.expected_move,
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
    source_event_counts = Counter(
        entry.source_event_group for entry in ordered if entry.source_event_group
    )
    close_entries = [entry for entry in ordered if entry.event_type in _CLOSE_EVENT_TYPES]
    open_positions: dict[str, dict[str, Any]] = {}

    for entry in ordered:
        position_key = entry.position_id or entry.symbol_id
        if entry.position_side is None or entry.position_quantity <= 0:
            open_positions.pop(position_key, None)
            continue
        open_positions[position_key] = {
            "symbol_id": entry.symbol_id,
            "side": entry.position_side,
            "quantity": entry.position_quantity,
            "entry_price": entry.position_entry_price,
            "signal_id": entry.signal_id,
            "raw_event_id": entry.raw_event_id,
            "source_event_type": entry.source_event_type,
            "source_event_group": entry.source_event_group,
            "exit_horizon_label": entry.exit_horizon_label,
            "max_hold_minutes": entry.max_hold_minutes,
            "exit_due_at": entry.exit_due_at.isoformat() if entry.exit_due_at else None,
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
        source_event_counts=dict(source_event_counts),
        event_performance=_build_event_performance(close_entries),
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


def _build_event_performance(
    close_entries: list[TradeJournalEntry],
) -> dict[str, EventTypePerformance]:
    grouped_returns: dict[str, list[float]] = {}

    for entry in close_entries:
        if entry.source_event_group is None or entry.realized_return_fraction is None:
            continue
        grouped_returns.setdefault(entry.source_event_group, []).append(
            entry.realized_return_fraction
        )

    return {
        event_type: EventTypePerformance(
            avg_return=round(sum(returns) / len(returns), 6),
            hit_rate=round(sum(1 for item in returns if item > 0) / len(returns), 4),
            sharpe=round(_calculate_sharpe_ratio(returns), 6),
            trade_count=len(returns),
        )
        for event_type, returns in sorted(grouped_returns.items())
    }


def _calculate_realized_return_fraction(
    *,
    before_position: Position | None,
    filled_quantity: float,
    realized_pnl_usd: float,
    journal_event_type: str,
) -> float | None:
    if journal_event_type not in _CLOSE_EVENT_TYPES:
        return None
    if before_position is None or before_position.entry_price <= 0:
        return None

    closed_quantity = min(abs(before_position.quantity), abs(filled_quantity))
    entry_notional = closed_quantity * before_position.entry_price
    if entry_notional <= 0:
        return None

    return round(realized_pnl_usd / entry_notional, 6)


def _calculate_sharpe_ratio(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0

    mean_return = sum(returns) / len(returns)
    variance = sum((item - mean_return) ** 2 for item in returns) / (len(returns) - 1)
    standard_deviation = math.sqrt(variance)
    if standard_deviation == 0:
        return 0.0

    return mean_return / standard_deviation * math.sqrt(len(returns))


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    return datetime.now(UTC)


def _derive_actual_move(
    *,
    before_position: Position | None,
    exit_price: float,
) -> MoveDirection | None:
    if before_position is None:
        return None
    if exit_price > before_position.entry_price:
        return "up"
    if exit_price < before_position.entry_price:
        return "down"
    return "flat"


def cast_move_direction(value: Any) -> MoveDirection | None:
    if value in {"up", "down", "flat"}:
        return value
    return None
