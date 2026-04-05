"""ERC-8004-style validation artifacts for trading decisions and outcomes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from agent.portfolio import PortfolioSnapshot
from agent.risk import RiskCheckResult
from agent.signals import NoTradeDecision, TradeIntent
from execution.orders import ExecutionResult


class ArtifactKind(StrEnum):
    """Supported artifact categories for the hackathon MVP."""

    TRADE_INTENT = "trade_intent"
    NO_TRADE_DECISION = "no_trade_decision"
    PRE_TRADE_RISK_CHECK = "pre_trade_risk_check"
    EXECUTION_RESULT = "execution_result"
    SIGNAL_OUTCOME = "signal_outcome"
    PERFORMANCE_CHECKPOINT = "performance_checkpoint"


class ArtifactStatus(StrEnum):
    """Outcome state associated with a validation artifact."""

    RECORDED = "recorded"
    PASSED = "passed"
    FAILED = "failed"


@dataclass(frozen=True)
class ArtifactEvidence:
    """One measurable fact attached to a validation artifact."""

    name: str
    value: Any
    unit: str | None = None
    passed: bool | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable evidence item."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "passed": self.passed,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ValidationArtifact:
    """Canonical, JSON-serializable record for one verifiable trading milestone."""

    artifact_id: str
    kind: ArtifactKind
    status: ArtifactStatus
    subject_id: str
    payload: dict[str, Any]
    agent_id: str | None = None
    evidence: tuple[ArtifactEvidence, ...] = ()
    refs: dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    schema_version: str = "erc8004-mvp/v1"

    @classmethod
    def from_trade_intent(
        cls,
        intent: TradeIntent,
        *,
        agent_id: str | None = None,
    ) -> ValidationArtifact:
        """Create a trade-intent artifact from a strategy-produced intent."""
        subject_id = (
            f"{intent.signal_id or 'signal'}:{intent.symbol_id}:{intent.side}:"
            f"{intent.exit_horizon_label or 'core'}:{intent.generated_at.isoformat()}"
        )
        resolved_signal_id = intent.signal_family or intent.signal_id
        payload = {
            "symbol_id": intent.symbol_id,
            "side": intent.side,
            "direction": intent.direction,
            "asset": intent.asset,
            "notional_usd": round(intent.notional_usd, 2),
            "quantity": round(intent.quantity, 8),
            "current_price": intent.current_price,
            "score": intent.score,
            "confidence": intent.confidence,
            "confidence_score": intent.confidence_score,
            "expected_move": intent.expected_move,
            "expected_move_fraction": intent.expected_move_fraction,
            "stop_distance_fraction": intent.stop_distance_fraction,
            "risk_reward_ratio": intent.risk_reward_ratio,
            "rationale": list(intent.rationale),
            "generated_at": intent.generated_at.isoformat(),
            "signal_id": resolved_signal_id,
            "signal_version": intent.signal_version,
            "heuristic_version": intent.heuristic_version,
            "event_type": intent.event_type,
            "exit_horizon_label": intent.exit_horizon_label,
            "max_hold_minutes": intent.max_hold_minutes,
            "exit_due_at": intent.exit_due_at.isoformat() if intent.exit_due_at else None,
            "position_id": intent.position_id,
            "selection_rank": intent.selection_rank,
            "selection_composite_score": intent.selection_composite_score,
            "rejected_alternatives": [
                candidate.to_dict() for candidate in intent.rejected_alternatives
            ],
            "audit_metadata": {
                key: value
                for key, value in {
                    "signal_instance_id": intent.signal_id,
                    "signal_family": intent.signal_family,
                    "model_version": intent.model_version,
                    "feature_set": intent.feature_set,
                    "raw_event_id": intent.raw_event_id,
                }.items()
                if value is not None
            },
        }
        evidence = (
            ArtifactEvidence(name="score", value=intent.score, unit="normalized"),
            ArtifactEvidence(
                name="confidence_score",
                value=intent.confidence_score,
                unit="normalized",
            ),
            ArtifactEvidence(name="notional_usd", value=round(intent.notional_usd, 2), unit="usd"),
            ArtifactEvidence(name="quantity", value=round(intent.quantity, 8), unit="asset"),
            ArtifactEvidence(
                name="risk_reward_ratio",
                value=intent.risk_reward_ratio,
                unit="ratio",
                passed=(intent.risk_reward_ratio is None or intent.risk_reward_ratio > 1.5),
            ),
        )
        artifact_id = _stable_id(
            "trade-intent",
            agent_id or "unknown",
            subject_id,
            str(payload["notional_usd"]),
            str(payload["score"]),
        )
        return cls(
            artifact_id=artifact_id,
            kind=ArtifactKind.TRADE_INTENT,
            status=ArtifactStatus.RECORDED,
            subject_id=subject_id,
            payload=payload,
            agent_id=agent_id,
            evidence=evidence,
            refs={
                "symbol_id": intent.symbol_id,
                **({"signal_id": intent.signal_id} if intent.signal_id else {}),
                **({"signal_family": intent.signal_family} if intent.signal_family else {}),
                **({"signal_version": intent.signal_version} if intent.signal_version else {}),
                **({"raw_event_id": intent.raw_event_id} if intent.raw_event_id else {}),
                **({"event_type": intent.event_type} if intent.event_type else {}),
                **(
                    {"exit_horizon_label": intent.exit_horizon_label}
                    if intent.exit_horizon_label
                    else {}
                ),
                **({"position_id": intent.position_id} if intent.position_id else {}),
            },
            created_at=intent.generated_at,
        )

    @classmethod
    def from_no_trade_decision(
        cls,
        decision: NoTradeDecision,
        *,
        agent_id: str | None = None,
    ) -> ValidationArtifact:
        """Create an explicit abstention artifact when the strategy defaults to no trade."""
        subject_id = (
            f"{decision.signal_id or 'no-trade'}:{decision.symbol_id}:"
            f"{decision.reason_code}:{decision.detected_at.isoformat()}"
        )
        payload = {
            "symbol_id": decision.symbol_id,
            "event_type": decision.event_type,
            "event_group": decision.event_group,
            "signal_id": decision.signal_family or decision.signal_id,
            "signal_version": decision.signal_version,
            "heuristic_version": None,
            "asset": decision.asset,
            "direction": decision.direction,
            "confidence": decision.confidence,
            "reason_code": decision.reason_code,
            "reason": decision.reason,
            "confidence_score": decision.confidence_score,
            "threshold": decision.threshold,
            "score": decision.score,
            "rationale": list(decision.rationale),
            "detected_at": decision.detected_at.isoformat(),
            "audit_metadata": {
                key: value
                for key, value in {
                    "signal_instance_id": decision.signal_id,
                    "signal_family": decision.signal_family,
                    "model_version": decision.model_version,
                    "feature_set": decision.feature_set,
                    "raw_event_id": decision.raw_event_id,
                }.items()
                if value is not None
            },
        }
        evidence = (
            ArtifactEvidence(name="decision", value="no_trade", passed=True),
            ArtifactEvidence(
                name="confidence_score",
                value=decision.confidence_score,
                unit="normalized",
                passed=True,
            ),
            ArtifactEvidence(
                name="threshold",
                value=decision.threshold,
                unit="normalized",
                passed=True,
            ),
        )
        artifact_id = _stable_id(
            "no-trade-decision",
            agent_id or "unknown",
            subject_id,
            decision.reason_code,
            str(decision.confidence_score),
        )
        return cls(
            artifact_id=artifact_id,
            kind=ArtifactKind.NO_TRADE_DECISION,
            status=ArtifactStatus.RECORDED,
            subject_id=subject_id,
            payload=payload,
            agent_id=agent_id,
            evidence=evidence,
            refs={
                "symbol_id": decision.symbol_id,
                **({"signal_id": decision.signal_id} if decision.signal_id else {}),
                **({"raw_event_id": decision.raw_event_id} if decision.raw_event_id else {}),
                **({"event_type": decision.event_type} if decision.event_type else {}),
            },
            created_at=decision.detected_at,
        )

    @classmethod
    def from_risk_check(
        cls,
        result: RiskCheckResult,
        *,
        agent_id: str | None = None,
        subject_id: str,
        proposed_notional: float | None = None,
        checked_at: datetime | None = None,
    ) -> ValidationArtifact:
        """Create a pre-trade risk artifact from an objective risk evaluation."""
        violation_codes = [violation.code for violation in result.violations]
        payload = {
            "approved": result.approved,
            "allowed_notional": round(result.allowed_notional, 2),
            "proposed_notional": (
                round(proposed_notional, 2) if proposed_notional is not None else None
            ),
            "expected_move_fraction": result.expected_move_fraction,
            "stop_distance_fraction": result.stop_distance_fraction,
            "risk_reward_ratio": result.risk_reward_ratio,
            "violations": [
                {"code": violation.code, "message": violation.message}
                for violation in result.violations
            ],
            "notes": list(result.notes),
        }
        evidence = (
            ArtifactEvidence(name="approved", value=result.approved, passed=result.approved),
            ArtifactEvidence(
                name="allowed_notional",
                value=round(result.allowed_notional, 2),
                unit="usd",
                passed=result.approved,
            ),
            ArtifactEvidence(
                name="risk_reward_ratio",
                value=result.risk_reward_ratio,
                unit="ratio",
                passed=result.approved,
            ),
            ArtifactEvidence(name="violation_count", value=len(result.violations), unit="count"),
        )
        artifact_id = _stable_id(
            "risk-check",
            agent_id or "unknown",
            subject_id,
            str(result.approved),
            ",".join(violation_codes),
        )
        return cls(
            artifact_id=artifact_id,
            kind=ArtifactKind.PRE_TRADE_RISK_CHECK,
            status=ArtifactStatus.PASSED if result.approved else ArtifactStatus.FAILED,
            subject_id=subject_id,
            payload=payload,
            agent_id=agent_id,
            evidence=evidence,
            refs={"subject_id": subject_id},
            created_at=checked_at or datetime.now(UTC),
        )

    @classmethod
    def from_execution_result(
        cls,
        result: ExecutionResult,
        *,
        agent_id: str | None = None,
    ) -> ValidationArtifact:
        """Create an execution-result artifact from the Kraken execution layer."""
        payload = result.to_dict()
        success = result.is_successful
        fill_quantity = result.fill.filled_quantity if result.fill is not None else 0.0
        evidence = (
            ArtifactEvidence(name="is_successful", value=success, passed=success),
            ArtifactEvidence(name="retry_count", value=result.retry_count, unit="count"),
            ArtifactEvidence(name="filled_quantity", value=fill_quantity, unit="asset"),
        )
        artifact_id = _stable_id(
            "execution-result",
            agent_id or "unknown",
            result.request.intent_id,
            result.status.value,
            str(result.retry_count),
        )
        return cls(
            artifact_id=artifact_id,
            kind=ArtifactKind.EXECUTION_RESULT,
            status=ArtifactStatus.PASSED if success else ArtifactStatus.FAILED,
            subject_id=result.request.intent_id,
            payload=payload,
            agent_id=agent_id,
            evidence=evidence,
            refs={
                "intent_id": result.request.intent_id,
                "client_order_id": result.request.client_order_id,
                "symbol_id": result.request.symbol_id,
                **({"signal_id": result.request.signal_id} if result.request.signal_id else {}),
                **(
                    {"raw_event_id": result.request.raw_event_id}
                    if result.request.raw_event_id
                    else {}
                ),
                **(
                    {"event_type": result.request.event_type}
                    if result.request.event_type
                    else {}
                ),
                **(
                    {"exit_horizon_label": result.request.exit_horizon_label}
                    if result.request.exit_horizon_label
                    else {}
                ),
                **(
                    {"position_id": result.request.position_id}
                    if result.request.position_id
                    else {}
                ),
            },
            created_at=result.completed_at,
        )

    @classmethod
    def from_signal_outcome(
        cls,
        execution_result: ExecutionResult,
        *,
        symbol_id: str | None = None,
        side: str | None = None,
        entry_price: float,
        opened_at: datetime,
        exit_horizon_label: str | None = None,
        raw_event_id: str | None = None,
        signal_id: str | None = None,
        event_type: str | None = None,
        realized_pnl_usd: float = 0.0,
        agent_id: str | None = None,
    ) -> ValidationArtifact:
        """Create a realized signal-outcome artifact for horizon-by-horizon evaluation."""
        fill = execution_result.fill
        if fill is None:
            raise ValueError("A fill is required to build a signal outcome artifact.")

        resolved_symbol = symbol_id or execution_result.request.symbol_id
        resolved_side = side or ("long" if execution_result.request.side == "buy" else "short")
        resolved_horizon = exit_horizon_label or execution_result.request.exit_horizon_label
        resolved_signal_instance_id = signal_id or execution_result.request.signal_id
        resolved_signal_id = execution_result.request.signal_family or resolved_signal_instance_id
        resolved_raw_event_id = raw_event_id or execution_result.request.raw_event_id
        resolved_event_type = event_type or execution_result.request.event_type
        realized_return_fraction = 0.0
        if entry_price > 0:
            realized_return_fraction = (
                (fill.average_price - entry_price) / entry_price
                if resolved_side == "long"
                else (entry_price - fill.average_price) / entry_price
            )
        elapsed_minutes = max((fill.filled_at - opened_at).total_seconds() / 60.0, 0.0)
        realized_pnl_value = round(realized_pnl_usd, 2)
        realized_return_value = round(realized_return_fraction, 6)
        payload = {
            "symbol_id": resolved_symbol,
            "side": resolved_side,
            "entry_price": round(entry_price, 8),
            "exit_price": round(fill.average_price, 8),
            "opened_at": opened_at.isoformat(),
            "closed_at": fill.filled_at.isoformat(),
            "elapsed_minutes": round(elapsed_minutes, 2),
            "realized_pnl_usd": realized_pnl_value,
            "realized_return_fraction": realized_return_value,
            "exit_horizon_label": resolved_horizon,
            "signal_id": resolved_signal_id,
            "signal_version": execution_result.request.signal_version,
            "heuristic_version": execution_result.request.heuristic_version,
            "asset": execution_result.request.asset,
            "direction": execution_result.request.direction,
            "confidence": execution_result.request.confidence,
            "event_type": resolved_event_type,
            "intent_id": execution_result.request.intent_id,
            "position_id": execution_result.request.position_id,
            "audit_metadata": {
                key: value
                for key, value in {
                    "signal_instance_id": resolved_signal_instance_id,
                    "signal_family": execution_result.request.signal_family,
                    "model_version": execution_result.request.model_version,
                    "feature_set": execution_result.request.feature_set,
                    "raw_event_id": resolved_raw_event_id,
                }.items()
                if value is not None
            },
        }
        evidence = (
            ArtifactEvidence(
                name="realized_pnl_usd",
                value=realized_pnl_value,
                unit="usd",
                passed=realized_pnl_value >= 0,
            ),
            ArtifactEvidence(
                name="realized_return_fraction",
                value=realized_return_value,
                unit="fraction",
                passed=realized_return_value >= 0,
            ),
            ArtifactEvidence(
                name="elapsed_minutes",
                value=round(elapsed_minutes, 2),
                unit="minutes",
            ),
        )
        subject_id = (
            f"{resolved_signal_id or execution_result.request.intent_id}:"
            f"{resolved_horizon or 'outcome'}:{fill.filled_at.isoformat()}"
        )
        artifact_id = _stable_id(
            "signal-outcome",
            agent_id or "unknown",
            subject_id,
            str(payload["realized_pnl_usd"]),
        )
        return cls(
            artifact_id=artifact_id,
            kind=ArtifactKind.SIGNAL_OUTCOME,
            status=(
                ArtifactStatus.PASSED
                if realized_return_value >= 0
                else ArtifactStatus.FAILED
            ),
            subject_id=subject_id,
            payload=payload,
            agent_id=agent_id,
            evidence=evidence,
            refs={
                "symbol_id": resolved_symbol,
                "intent_id": execution_result.request.intent_id,
                **(
                    {"signal_id": resolved_signal_instance_id}
                    if resolved_signal_instance_id
                    else {}
                ),
                **(
                    {"signal_family": execution_result.request.signal_family}
                    if execution_result.request.signal_family
                    else {}
                ),
                **(
                    {"signal_version": execution_result.request.signal_version}
                    if execution_result.request.signal_version
                    else {}
                ),
                **({"raw_event_id": resolved_raw_event_id} if resolved_raw_event_id else {}),
                **({"event_type": resolved_event_type} if resolved_event_type else {}),
                **({"exit_horizon_label": resolved_horizon} if resolved_horizon else {}),
                **(
                    {"position_id": execution_result.request.position_id}
                    if execution_result.request.position_id
                    else {}
                ),
            },
            created_at=fill.filled_at,
        )

    @classmethod
    def from_performance_checkpoint(
        cls,
        portfolio: PortfolioSnapshot,
        *,
        agent_id: str | None = None,
        checkpoint_name: str = "pnl_snapshot",
    ) -> ValidationArtifact:
        """Create a portfolio-performance checkpoint artifact."""
        pnl_value = round(portfolio.realized_pnl_today, 2)
        passed = pnl_value >= 0
        payload = {
            "checkpoint_name": checkpoint_name,
            "total_equity": round(portfolio.total_equity, 2),
            "cash_usd": round(portfolio.cash_usd, 2),
            "open_position_count": portfolio.open_position_count(),
            "realized_pnl_today": pnl_value,
            "consecutive_losses": portfolio.consecutive_losses,
            "as_of": portfolio.as_of.isoformat(),
        }
        evidence = (
            ArtifactEvidence(
                name="realized_pnl_today",
                value=pnl_value,
                unit="usd",
                passed=passed,
            ),
            ArtifactEvidence(
                name="open_position_count",
                value=portfolio.open_position_count(),
            ),
            ArtifactEvidence(
                name="total_equity",
                value=round(portfolio.total_equity, 2),
                unit="usd",
            ),
        )
        subject_id = f"{checkpoint_name}:{portfolio.as_of.isoformat()}"
        artifact_id = _stable_id(
            "performance-checkpoint",
            agent_id or "unknown",
            checkpoint_name,
            portfolio.as_of.isoformat(),
            str(pnl_value),
        )
        return cls(
            artifact_id=artifact_id,
            kind=ArtifactKind.PERFORMANCE_CHECKPOINT,
            status=ArtifactStatus.PASSED if passed else ArtifactStatus.FAILED,
            subject_id=subject_id,
            payload=payload,
            agent_id=agent_id,
            evidence=evidence,
            refs={"checkpoint_name": checkpoint_name},
            created_at=portfolio.as_of,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable artifact payload."""
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind.value,
            "status": self.status.value,
            "subject_id": self.subject_id,
            "agent_id": self.agent_id,
            "payload": self.payload,
            "evidence": [item.to_dict() for item in self.evidence],
            "refs": dict(self.refs),
            "created_at": self.created_at.isoformat(),
            "schema_version": self.schema_version,
        }


def _stable_id(prefix: str, *parts: str) -> str:
    joined = "|".join(parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"
