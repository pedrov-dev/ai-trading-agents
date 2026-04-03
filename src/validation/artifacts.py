"""ERC-8004-style validation artifacts for trading decisions and outcomes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from agent.portfolio import PortfolioSnapshot
from agent.risk import RiskCheckResult
from agent.signals import TradeIntent
from execution.orders import ExecutionResult


class ArtifactKind(StrEnum):
    """Supported artifact categories for the hackathon MVP."""

    TRADE_INTENT = "trade_intent"
    PRE_TRADE_RISK_CHECK = "pre_trade_risk_check"
    EXECUTION_RESULT = "execution_result"
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
        subject_id = f"{intent.symbol_id}:{intent.side}:{intent.generated_at.isoformat()}"
        payload = {
            "symbol_id": intent.symbol_id,
            "side": intent.side,
            "notional_usd": round(intent.notional_usd, 2),
            "quantity": round(intent.quantity, 8),
            "current_price": intent.current_price,
            "score": intent.score,
            "rationale": list(intent.rationale),
            "generated_at": intent.generated_at.isoformat(),
        }
        evidence = (
            ArtifactEvidence(name="score", value=intent.score, unit="normalized"),
            ArtifactEvidence(name="notional_usd", value=round(intent.notional_usd, 2), unit="usd"),
            ArtifactEvidence(name="quantity", value=round(intent.quantity, 8), unit="asset"),
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
            refs={"symbol_id": intent.symbol_id},
            created_at=intent.generated_at,
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
            },
            created_at=result.completed_at,
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
