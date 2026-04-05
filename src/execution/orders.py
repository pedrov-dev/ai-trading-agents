"""Immutable order lifecycle models for the Kraken execution layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from agent.signals import TradeIntent, TradeSide


class ExecutionMode(StrEnum):
    """Supported execution modes for the MVP execution layer."""

    DRY_RUN = "dry_run"
    LIVE = "live"


class OrderStatus(StrEnum):
    """Lifecycle states tracked for each order request and execution attempt."""

    REQUESTED = "requested"
    SIMULATED = "simulated"
    VALIDATED = "validated"
    SUBMITTED = "submitted"
    FILLED = "filled"
    RETRYING = "retrying"
    FAILED = "failed"


@dataclass(frozen=True)
class OrderRequest:
    """Auditable exchange-ready order request derived from a trade intent."""

    intent_id: str
    client_order_id: str
    symbol_id: str
    side: TradeSide
    order_type: str
    notional_usd: float
    quantity: float
    current_price: float
    score: float
    rationale: tuple[str, ...]
    generated_at: datetime
    signal_id: str | None = None
    signal_family: str | None = None
    signal_version: str | None = None
    model_version: str | None = None
    feature_set: str | None = None
    asset: str | None = None
    direction: str | None = None
    confidence: float | None = None
    heuristic_version: str | None = None
    raw_event_id: str | None = None
    event_type: str | None = None
    event_group: str | None = None
    exit_horizon_label: str | None = None
    max_hold_minutes: int | None = None
    exit_due_at: datetime | None = None
    position_id: str | None = None
    requested_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    execution_mode: ExecutionMode = ExecutionMode.DRY_RUN
    status: OrderStatus = OrderStatus.REQUESTED

    @classmethod
    def from_trade_intent(
        cls,
        intent: TradeIntent,
        *,
        execution_mode: ExecutionMode = ExecutionMode.DRY_RUN,
        requested_at: datetime | None = None,
        intent_id: str | None = None,
        client_order_id: str | None = None,
        order_type: str = "market",
    ) -> OrderRequest:
        """Build a stable order request from one strategy-produced trade intent."""
        resolved_requested_at = requested_at or datetime.now(UTC)
        intent_suffix = (intent.position_id or intent.exit_horizon_label or "core").replace(
            ":", "-"
        )
        resolved_intent_id = intent_id or (
            f"intent-{intent.symbol_id}-{intent_suffix}-{resolved_requested_at.strftime('%Y%m%d%H%M%S')}"
        )
        resolved_client_order_id = client_order_id or (
            f"{resolved_intent_id}-{resolved_requested_at.strftime('%H%M%S')}"
        )
        return cls(
            intent_id=resolved_intent_id,
            client_order_id=resolved_client_order_id,
            symbol_id=intent.symbol_id,
            side=intent.side,
            order_type=order_type,
            notional_usd=round(intent.notional_usd, 2),
            quantity=round(intent.quantity, 8),
            current_price=intent.current_price,
            score=intent.score,
            rationale=intent.rationale,
            generated_at=intent.generated_at,
            signal_id=intent.signal_id,
            signal_family=intent.signal_family,
            signal_version=intent.signal_version,
            model_version=intent.model_version,
            feature_set=intent.feature_set,
            asset=intent.asset,
            direction=intent.direction,
            confidence=intent.confidence,
            heuristic_version=intent.heuristic_version,
            raw_event_id=intent.raw_event_id,
            event_type=intent.event_type,
            event_group=intent.event_group,
            exit_horizon_label=intent.exit_horizon_label,
            max_hold_minutes=intent.max_hold_minutes,
            exit_due_at=intent.exit_due_at,
            position_id=intent.position_id,
            requested_at=resolved_requested_at,
            execution_mode=execution_mode,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the order request."""
        return {
            "intent_id": self.intent_id,
            "client_order_id": self.client_order_id,
            "symbol_id": self.symbol_id,
            "side": self.side,
            "order_type": self.order_type,
            "notional_usd": self.notional_usd,
            "quantity": self.quantity,
            "current_price": self.current_price,
            "score": self.score,
            "rationale": list(self.rationale),
            "generated_at": self.generated_at.isoformat(),
            "signal_id": self.signal_family or self.signal_id,
            "signal_version": self.signal_version,
            "heuristic_version": self.heuristic_version,
            "asset": self.asset,
            "direction": self.direction,
            "confidence": self.confidence,
            "event_type": self.event_type,
            "event_group": self.event_group,
            "exit_horizon_label": self.exit_horizon_label,
            "max_hold_minutes": self.max_hold_minutes,
            "exit_due_at": self.exit_due_at.isoformat() if self.exit_due_at else None,
            "position_id": self.position_id,
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
            "requested_at": self.requested_at.isoformat(),
            "execution_mode": self.execution_mode.value,
            "status": self.status.value,
        }


@dataclass(frozen=True)
class OrderAttempt:
    """One execution attempt, including the CLI command and its outcome."""

    attempt_number: int
    command: tuple[str, ...]
    status: OrderStatus
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    response: dict[str, Any] | None = None
    retryable: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable execution-attempt record."""
        return {
            "attempt_number": self.attempt_number,
            "command": list(self.command),
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "response": self.response,
            "retryable": self.retryable,
        }


@dataclass(frozen=True)
class OrderFill:
    """Captured fill or simulated fill information for an executed order."""

    status: OrderStatus
    filled_quantity: float
    average_price: float
    filled_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable fill record."""
        return {
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "average_price": self.average_price,
            "filled_at": self.filled_at.isoformat(),
        }


@dataclass(frozen=True)
class OrderFailure:
    """Terminal failure details for an order that could not be completed."""

    code: str
    message: str
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    retryable: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable failure record."""
        return {
            "code": self.code,
            "message": self.message,
            "occurred_at": self.occurred_at.isoformat(),
            "retryable": self.retryable,
        }


@dataclass(frozen=True)
class ExecutionResult:
    """Final result of submitting one order request through the Kraken adapter."""

    request: OrderRequest
    status: OrderStatus
    attempts: tuple[OrderAttempt, ...] = ()
    fill: OrderFill | None = None
    failure: OrderFailure | None = None
    completed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def retry_count(self) -> int:
        """Return how many retries were required after the first attempt."""
        return max(len(self.attempts) - 1, 0)

    @property
    def is_successful(self) -> bool:
        """Whether the order completed without a terminal failure."""
        return self.status in {
            OrderStatus.SIMULATED,
            OrderStatus.VALIDATED,
            OrderStatus.SUBMITTED,
            OrderStatus.FILLED,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable execution result for auditing."""
        return {
            "status": self.status.value,
            "completed_at": self.completed_at.isoformat(),
            "retry_count": self.retry_count,
            "request": self.request.to_dict(),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "fill": self.fill.to_dict() if self.fill else None,
            "failure": self.failure.to_dict() if self.failure else None,
        }
