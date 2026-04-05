"""Persistent trade and position journaling for local runtime continuity."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast

from agent.portfolio import (
    LocalPortfolioStateProvider,
    PortfolioSnapshot,
    Position,
)
from agent.signals import MoveDirection, RejectedTradeCandidate
from detection.event_types import event_performance_group
from execution.orders import ExecutionResult
from monitoring.drawdown import EquityPoint, build_drawdown_snapshot
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
    signal_family: str | None = None
    signal_version: str | None = None
    model_version: str | None = None
    feature_set: str | None = None
    asset: str | None = None
    direction: str | None = None
    confidence: float | None = None
    raw_event_id: str | None = None
    source_event_type: str | None = None
    source_event_group: str | None = None
    exit_horizon_label: str | None = None
    max_hold_minutes: int | None = None
    exit_due_at: datetime | None = None
    intent_id: str | None = None
    client_order_id: str | None = None
    realized_return_fraction: float | None = None
    selection_rank: int | None = None
    selection_composite_score: float | None = None
    rejected_alternatives: tuple[RejectedTradeCandidate, ...] = ()
    timing_label: str | None = None
    asset_selection_label: str | None = None
    best_alternative_symbol: str | None = None
    best_alternative_return_fraction: float | None = None
    learning_reason_codes: tuple[str, ...] = ()
    heuristic_version: str | None = None
    notes: tuple[str, ...] = ()

    @classmethod
    def from_execution_result(
        cls,
        *,
        execution_result: ExecutionResult,
        before_portfolio: PortfolioSnapshot,
        after_portfolio: PortfolioSnapshot,
        notes: tuple[str, ...] = (),
        timing_label: str | None = None,
        asset_selection_label: str | None = None,
        best_alternative_symbol: str | None = None,
        best_alternative_return_fraction: float | None = None,
        learning_reason_codes: tuple[str, ...] = (),
        heuristic_version: str | None = None,
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
            else (
                "up"
                if before_position is not None and before_position.side == "long"
                else "down"
            )
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
            signal_family=execution_result.request.signal_family,
            signal_version=execution_result.request.signal_version,
            model_version=execution_result.request.model_version,
            feature_set=execution_result.request.feature_set,
            asset=(
                execution_result.request.asset
                or execution_result.request.symbol_id.split("_", maxsplit=1)[0].upper()
            ),
            direction=(
                execution_result.request.direction
                or ("long" if execution_result.request.side == "buy" else "short")
            ),
            confidence=(
                execution_result.request.confidence
                if execution_result.request.confidence is not None
                else confidence_score
            ),
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
            selection_rank=(
                before_position.selection_rank
                if before_position is not None
                else after_position.selection_rank
                if after_position is not None
                else None
            ),
            selection_composite_score=(
                before_position.selection_composite_score
                if before_position is not None
                else after_position.selection_composite_score
                if after_position is not None
                else None
            ),
            rejected_alternatives=(
                before_position.rejected_alternatives
                if before_position is not None
                else after_position.rejected_alternatives
                if after_position is not None
                else ()
            ),
            timing_label=str(timing_label) if timing_label is not None else None,
            asset_selection_label=(
                str(asset_selection_label) if asset_selection_label is not None else None
            ),
            best_alternative_symbol=(
                str(best_alternative_symbol) if best_alternative_symbol is not None else None
            ),
            best_alternative_return_fraction=(
                round(float(best_alternative_return_fraction), 6)
                if best_alternative_return_fraction is not None
                else None
            ),
            learning_reason_codes=tuple(str(item) for item in learning_reason_codes),
            heuristic_version=str(heuristic_version) if heuristic_version is not None else None,
            notes=tuple(str(item) for item in notes),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TradeJournalEntry:
        """Rehydrate a journal entry from JSON-loaded data."""
        position_entry_price = payload.get("position_entry_price")
        position_side = payload.get("position_side")
        position_id = payload.get("position_id")
        audit_metadata = payload.get("audit_metadata")
        audit_payload = (
            cast(dict[str, Any], audit_metadata)
            if isinstance(audit_metadata, dict)
            else {}
        )
        raw_signal_id = payload.get("signal_id")
        signal_instance_id = audit_payload.get(
            "signal_instance_id",
            payload.get("signal_instance_id"),
        )
        signal_family = audit_payload.get("signal_family", payload.get("signal_family"))
        signal_version = payload.get("signal_version")
        model_version = audit_payload.get("model_version", payload.get("model_version"))
        feature_set = audit_payload.get("feature_set", payload.get("feature_set"))
        asset = payload.get("asset")
        direction = payload.get("direction")
        confidence = payload.get("confidence")
        signal_id = signal_instance_id if signal_instance_id is not None else raw_signal_id
        raw_event_id = audit_payload.get("raw_event_id", payload.get("raw_event_id"))
        source_event_type = payload.get("source_event_type")
        source_event_group = payload.get("source_event_group")
        exit_horizon_label = payload.get("exit_horizon_label")
        max_hold_minutes = payload.get("max_hold_minutes")
        exit_due_at = payload.get("exit_due_at")
        intent_id = payload.get("intent_id")
        client_order_id = payload.get("client_order_id")
        realized_return_fraction = payload.get("realized_return_fraction")
        timing_label = payload.get("timing_label")
        asset_selection_label = payload.get("asset_selection_label")
        best_alternative_symbol = payload.get("best_alternative_symbol")
        best_alternative_return_fraction = payload.get("best_alternative_return_fraction")
        learning_reason_codes = payload.get("learning_reason_codes")
        heuristic_version = payload.get("heuristic_version")
        selection_rank = payload.get("selection_rank")
        selection_composite_score = payload.get("selection_composite_score")
        raw_rejected_alternatives = payload.get("rejected_alternatives", ())
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
            signal_family=(
                str(signal_family)
                if signal_family is not None
                else str(raw_signal_id)
                if signal_instance_id is not None and raw_signal_id is not None
                else None
            ),
            signal_version=str(signal_version) if signal_version is not None else None,
            model_version=str(model_version) if model_version is not None else None,
            feature_set=str(feature_set) if feature_set is not None else None,
            asset=(
                str(asset)
                if asset is not None
                else str(payload.get("symbol_id", "unknown_symbol"))
                .split("_", maxsplit=1)[0]
                .upper()
            ),
            direction=(
                str(direction)
                if direction is not None
                else "long"
                if str(payload.get("side", "buy")).lower() == "buy"
                else "short"
            ),
            confidence=(
                float(confidence)
                if confidence is not None
                else float(payload["confidence_score"])
                if payload.get("confidence_score") is not None
                else None
            ),
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
            selection_rank=int(selection_rank) if selection_rank is not None else None,
            selection_composite_score=(
                float(selection_composite_score)
                if selection_composite_score is not None
                else None
            ),
            rejected_alternatives=tuple(
                RejectedTradeCandidate(
                    symbol_id=str(item.get("symbol_id", "unknown_symbol")),
                    side=cast(
                        Literal["buy", "sell"],
                        "sell" if str(item.get("side", "buy")).lower() == "sell" else "buy",
                    ),
                    reference_price=float(item.get("reference_price", 0.0)),
                    score=float(item.get("score", 0.0)),
                    confidence_score=float(item.get("confidence_score", 0.0)),
                    composite_score=float(item.get("composite_score", 0.0)),
                    event_type=(
                        str(item.get("event_type"))
                        if item.get("event_type") is not None
                        else None
                    ),
                    event_group=(
                        str(item.get("event_group"))
                        if item.get("event_group") is not None
                        else None
                    ),
                )
                for item in raw_rejected_alternatives
                if isinstance(item, dict) and float(item.get("reference_price", 0.0)) > 0
            ),
            timing_label=str(timing_label) if timing_label is not None else None,
            asset_selection_label=(
                str(asset_selection_label) if asset_selection_label is not None else None
            ),
            best_alternative_symbol=(
                str(best_alternative_symbol) if best_alternative_symbol is not None else None
            ),
            best_alternative_return_fraction=(
                float(best_alternative_return_fraction)
                if best_alternative_return_fraction is not None
                else None
            ),
            learning_reason_codes=(
                tuple(str(item) for item in learning_reason_codes)
                if learning_reason_codes is not None
                else ()
            ),
            heuristic_version=str(heuristic_version) if heuristic_version is not None else None,
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
            "signal_id": self.signal_family or self.signal_id,
            "signal_version": self.signal_version,
            "heuristic_version": self.heuristic_version,
            "asset": self.asset,
            "direction": self.direction,
            "confidence": self.confidence,
            "audit_metadata": {
                key: value
                for key, value in {
                    "signal_instance_id": self.signal_id,
                    "signal_family": self.signal_family,
                    "model_version": self.model_version,
                    "feature_set": self.feature_set,
                    "raw_event_id": self.raw_event_id,
                }.items()
                if value is not None
            },
            "source_event_type": self.source_event_type,
            "source_event_group": self.source_event_group,
            "exit_horizon_label": self.exit_horizon_label,
            "max_hold_minutes": self.max_hold_minutes,
            "exit_due_at": self.exit_due_at.isoformat() if self.exit_due_at else None,
            "intent_id": self.intent_id,
            "client_order_id": self.client_order_id,
            "realized_return_fraction": self.realized_return_fraction,
            "selection_rank": self.selection_rank,
            "selection_composite_score": self.selection_composite_score,
            "rejected_alternatives": [
                item.to_dict() for item in self.rejected_alternatives
            ],
            "timing_label": self.timing_label,
            "asset_selection_label": self.asset_selection_label,
            "best_alternative_symbol": self.best_alternative_symbol,
            "best_alternative_return_fraction": self.best_alternative_return_fraction,
            "learning_reason_codes": list(self.learning_reason_codes),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class EventTypePerformance:
    """Aggregate realized performance metrics for one reporting bucket."""

    avg_return: float
    hit_rate: float
    sharpe: float
    trade_count: int
    realized_pnl_usd: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_fraction: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.hit_rate

    @property
    def trades(self) -> int:
        return self.trade_count

    def to_dict(self) -> dict[str, float | int]:
        return {
            "avg_return": self.avg_return,
            "hit_rate": self.hit_rate,
            "win_rate": self.win_rate,
            "sharpe": self.sharpe,
            "trade_count": self.trade_count,
            "trades": self.trades,
            "realized_pnl_usd": self.realized_pnl_usd,
            "profit_factor": self.profit_factor,
            "max_drawdown_fraction": self.max_drawdown_fraction,
            "drawdown": self.max_drawdown_fraction,
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
    hit_rate: float = 0.0
    avg_return: float = 0.0
    sharpe: float = 0.0
    trade_frequency_per_hour: float = 0.0
    trade_frequency_per_day: float = 0.0
    event_counts: dict[str, int] = field(default_factory=dict)
    symbol_counts: dict[str, int] = field(default_factory=dict)
    source_event_counts: dict[str, int] = field(default_factory=dict)
    signal_counts: dict[str, int] = field(default_factory=dict)
    signal_version_counts: dict[str, int] = field(default_factory=dict)
    event_performance: dict[str, EventTypePerformance] = field(default_factory=dict)
    asset_performance: dict[str, EventTypePerformance] = field(default_factory=dict)
    signal_performance: dict[str, EventTypePerformance] = field(default_factory=dict)
    signal_version_performance: dict[str, EventTypePerformance] = field(default_factory=dict)
    heuristic_version_performance: dict[str, EventTypePerformance] = field(default_factory=dict)
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
            "hit_rate": self.hit_rate,
            "avg_return": self.avg_return,
            "sharpe": self.sharpe,
            "trade_frequency_per_hour": self.trade_frequency_per_hour,
            "trade_frequency_per_day": self.trade_frequency_per_day,
            "event_counts": self.event_counts,
            "symbol_counts": self.symbol_counts,
            "source_event_counts": self.source_event_counts,
            "signal_counts": self.signal_counts,
            "signal_version_counts": self.signal_version_counts,
            "event_performance": {
                event_type: metrics.to_dict()
                for event_type, metrics in self.event_performance.items()
            },
            "asset_performance": {
                symbol_id: metrics.to_dict()
                for symbol_id, metrics in self.asset_performance.items()
            },
            "signal_performance": {
                signal_id: metrics.to_dict()
                for signal_id, metrics in self.signal_performance.items()
            },
            "signal_version_performance": {
                version_key: metrics.to_dict()
                for version_key, metrics in self.signal_version_performance.items()
            },
            "heuristic_version_performance": {
                version_key: metrics.to_dict()
                for version_key, metrics in self.heuristic_version_performance.items()
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
                selection_rank=entry.selection_rank,
                selection_composite_score=entry.selection_composite_score,
                rejected_alternatives=entry.rejected_alternatives,
                heuristic_version=entry.heuristic_version,
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
    signal_counts = Counter(
        key for entry in ordered if (key := _signal_bucket_key(entry)) is not None
    )
    signal_version_counts = Counter(
        key for entry in ordered if (key := _signal_version_bucket_key(entry)) is not None
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
            "signal_id": entry.signal_family or entry.signal_id,
            "signal_version": entry.signal_version,
            "heuristic_version": entry.heuristic_version,
            "asset": entry.asset,
            "direction": entry.direction,
            "confidence": entry.confidence,
            "audit_metadata": {
                key: value
                for key, value in {
                    "signal_instance_id": entry.signal_id,
                    "signal_family": entry.signal_family,
                    "model_version": entry.model_version,
                    "feature_set": entry.feature_set,
                    "raw_event_id": entry.raw_event_id,
                }.items()
                if value is not None
            },
            "source_event_type": entry.source_event_type,
            "source_event_group": entry.source_event_group,
            "exit_horizon_label": entry.exit_horizon_label,
            "max_hold_minutes": entry.max_hold_minutes,
            "exit_due_at": entry.exit_due_at.isoformat() if entry.exit_due_at else None,
            "last_event_type": entry.event_type,
            "recorded_at": entry.recorded_at.isoformat(),
        }

    win_count = sum(1 for entry in close_entries if entry.realized_pnl_usd > 0)
    loss_count = sum(1 for entry in close_entries if entry.realized_pnl_usd < 0)
    returns = [
        entry.realized_return_fraction
        for entry in close_entries
        if entry.realized_return_fraction is not None
    ]
    hit_rate = round(win_count / len(close_entries), 4) if close_entries else 0.0
    avg_return = round(sum(returns) / len(returns), 6) if returns else 0.0
    sharpe = round(_calculate_sharpe_ratio(returns), 6) if returns else 0.0
    trade_frequency_per_hour, trade_frequency_per_day = _calculate_trade_frequency(close_entries)

    return TradeJournalSummary(
        total_entries=len(ordered),
        closed_trade_count=len(close_entries),
        open_position_count=len(open_positions),
        realized_pnl_usd=round(sum(entry.realized_pnl_usd for entry in ordered), 2),
        win_count=win_count,
        loss_count=loss_count,
        hit_rate=hit_rate,
        avg_return=avg_return,
        sharpe=sharpe,
        trade_frequency_per_hour=trade_frequency_per_hour,
        trade_frequency_per_day=trade_frequency_per_day,
        event_counts=dict(event_counts),
        symbol_counts=dict(symbol_counts),
        source_event_counts=dict(source_event_counts),
        signal_counts=dict(signal_counts),
        signal_version_counts=dict(signal_version_counts),
        event_performance=_build_event_performance(close_entries),
        asset_performance=_build_asset_performance(close_entries),
        signal_performance=_build_signal_performance(close_entries),
        signal_version_performance=_build_signal_version_performance(close_entries),
        heuristic_version_performance=_build_heuristic_version_performance(close_entries),
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
    return _build_grouped_performance(
        close_entries,
        key_resolver=lambda entry: entry.source_event_group,
    )


def _build_asset_performance(
    close_entries: list[TradeJournalEntry],
) -> dict[str, EventTypePerformance]:
    return _build_grouped_performance(
        close_entries,
        key_resolver=lambda entry: entry.symbol_id,
    )


def _build_signal_performance(
    close_entries: list[TradeJournalEntry],
) -> dict[str, EventTypePerformance]:
    return _build_grouped_performance(
        close_entries,
        key_resolver=_signal_bucket_key,
    )


def _build_signal_version_performance(
    close_entries: list[TradeJournalEntry],
) -> dict[str, EventTypePerformance]:
    return _build_grouped_performance(
        close_entries,
        key_resolver=_signal_version_bucket_key,
    )


def _build_heuristic_version_performance(
    close_entries: list[TradeJournalEntry],
) -> dict[str, EventTypePerformance]:
    return _build_grouped_performance(
        close_entries,
        key_resolver=lambda entry: entry.heuristic_version,
    )


def _build_grouped_performance(
    close_entries: list[TradeJournalEntry],
    *,
    key_resolver: Any,
) -> dict[str, EventTypePerformance]:
    grouped_entries: dict[str, list[TradeJournalEntry]] = {}

    for entry in close_entries:
        key = key_resolver(entry)
        if key is None:
            continue
        grouped_entries.setdefault(str(key), []).append(entry)

    return {
        key: _calculate_group_metrics(entries)
        for key, entries in sorted(grouped_entries.items())
    }


def _calculate_group_metrics(entries: list[TradeJournalEntry]) -> EventTypePerformance:
    returns = [
        entry.realized_return_fraction
        for entry in entries
        if entry.realized_return_fraction is not None
    ]
    trade_count = len(entries)
    win_rate = (
        round(sum(1 for entry in entries if entry.realized_pnl_usd > 0) / trade_count, 4)
        if trade_count
        else 0.0
    )
    realized_pnl_usd = round(sum(entry.realized_pnl_usd for entry in entries), 2)
    return EventTypePerformance(
        avg_return=round(sum(returns) / len(returns), 6) if returns else 0.0,
        hit_rate=win_rate,
        sharpe=round(_calculate_sharpe_ratio(returns), 6) if returns else 0.0,
        trade_count=trade_count,
        realized_pnl_usd=realized_pnl_usd,
        profit_factor=_calculate_profit_factor(entries),
        max_drawdown_fraction=_calculate_max_drawdown_fraction(returns),
    )


def _signal_bucket_key(entry: TradeJournalEntry) -> str | None:
    return entry.signal_family or entry.signal_id


def _signal_version_bucket_key(entry: TradeJournalEntry) -> str | None:
    signal_key = _signal_bucket_key(entry)
    if signal_key is None or entry.signal_version is None:
        return None
    return f"{signal_key}:{entry.signal_version}"


def _calculate_profit_factor(entries: list[TradeJournalEntry]) -> float:
    gross_profit = sum(entry.realized_pnl_usd for entry in entries if entry.realized_pnl_usd > 0)
    gross_loss = abs(
        sum(entry.realized_pnl_usd for entry in entries if entry.realized_pnl_usd < 0)
    )
    if gross_profit <= 0 or gross_loss <= 0:
        return 0.0
    return round(gross_profit / gross_loss, 6)


def _calculate_max_drawdown_fraction(returns: list[float]) -> float:
    if not returns:
        return 0.0

    equity = 1.0
    history: list[EquityPoint] = []
    for index, trade_return in enumerate(returns):
        equity *= 1.0 + trade_return
        history.append(
            EquityPoint(
                recorded_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=index),
                equity=round(max(equity, 0.000001), 6),
            )
        )
    return round(build_drawdown_snapshot(history).max_drawdown_fraction, 6)


def _calculate_trade_frequency(
    close_entries: list[TradeJournalEntry],
) -> tuple[float, float]:
    trade_count = len(close_entries)
    if trade_count == 0:
        return 0.0, 0.0
    if trade_count == 1:
        return 1.0, 24.0

    observed_seconds = max(
        (close_entries[-1].recorded_at - close_entries[0].recorded_at).total_seconds(),
        60.0,
    )
    trades_per_hour = round(trade_count / (observed_seconds / 3600.0), 4)
    return trades_per_hour, round(trades_per_hour * 24.0, 4)


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
