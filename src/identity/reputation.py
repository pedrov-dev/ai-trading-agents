"""Transparent reputation scoring derived from objective validation artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from validation.artifacts import ArtifactKind, ValidationArtifact


@dataclass(frozen=True)
class ReputationEvent:
    """One scored reputation change caused by a measurable outcome."""

    source: str
    delta: float
    reason: str
    recorded_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    refs: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable reputation event."""
        return {
            "source": self.source,
            "delta": self.delta,
            "reason": self.reason,
            "recorded_at": self.recorded_at.isoformat(),
            "refs": dict(self.refs),
        }


@dataclass(frozen=True)
class ReputationSnapshot:
    """Current objective reputation state for a registered agent."""

    agent_id: str
    score: float = 50.0
    successful_validations: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    win_count: int = 0
    loss_count: int = 0
    cumulative_pnl: float = 0.0
    events: tuple[ReputationEvent, ...] = ()
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable reputation snapshot."""
        return {
            "agent_id": self.agent_id,
            "score": self.score,
            "successful_validations": self.successful_validations,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "cumulative_pnl": self.cumulative_pnl,
            "events": [event.to_dict() for event in self.events],
            "last_updated": self.last_updated.isoformat(),
        }


class ReputationEngine:
    """Apply simple, explainable reputation adjustments from objective outcomes."""

    def __init__(
        self,
        *,
        starting_score: float = 50.0,
        min_score: float = 0.0,
        max_score: float = 100.0,
    ) -> None:
        self._starting_score = starting_score
        self._min_score = min_score
        self._max_score = max_score

    def initialize(self, agent_id: str) -> ReputationSnapshot:
        """Create the default reputation state for a new agent."""
        return ReputationSnapshot(agent_id=agent_id, score=self._starting_score)

    def apply_artifacts(
        self,
        snapshot: ReputationSnapshot,
        artifacts: tuple[ValidationArtifact, ...] | list[ValidationArtifact],
    ) -> ReputationSnapshot:
        """Apply a sequence of validation artifacts to a reputation snapshot."""
        updated = snapshot
        for artifact in artifacts:
            updated = self.apply_artifact(updated, artifact)
        return updated

    def apply_artifact(
        self,
        snapshot: ReputationSnapshot,
        artifact: ValidationArtifact,
    ) -> ReputationSnapshot:
        """Apply one artifact-derived reputation change."""
        delta = 0.0
        successful_validations = snapshot.successful_validations
        successful_executions = snapshot.successful_executions
        failed_executions = snapshot.failed_executions
        win_count = snapshot.win_count
        loss_count = snapshot.loss_count
        cumulative_pnl = snapshot.cumulative_pnl
        reason = f"Processed {artifact.kind.value} artifact."

        if artifact.kind == ArtifactKind.PRE_TRADE_RISK_CHECK:
            approved = bool(artifact.payload.get("approved", False))
            delta = 1.0 if approved else -1.0
            if approved:
                successful_validations += 1
            reason = "Risk validation approved." if approved else "Risk validation rejected trade."
        elif artifact.kind == ArtifactKind.EXECUTION_RESULT:
            status = str(artifact.payload.get("status", "failed"))
            retry_count = int(artifact.payload.get("retry_count", 0))
            if status in {"simulated", "submitted", "filled"}:
                delta = max(2.0, 5.0 - (retry_count * 0.5))
                successful_executions += 1
                reason = f"Execution completed successfully with status={status}."
            else:
                delta = -6.0
                failed_executions += 1
                reason = f"Execution failed with status={status}."
        elif artifact.kind == ArtifactKind.SIGNAL_OUTCOME:
            pnl_value = float(artifact.payload.get("realized_pnl_usd", 0.0))
            cumulative_pnl = round(cumulative_pnl + pnl_value, 2)
            if pnl_value > 0:
                delta = min(pnl_value / 25.0, 4.0)
                win_count += 1
                reason = (
                    "Positive signal outcome recorded for horizon "
                    f"{artifact.payload.get('exit_horizon_label', 'scheduled')}."
                )
            elif pnl_value < 0:
                delta = -min(abs(pnl_value) / 25.0, 4.0)
                loss_count += 1
                reason = (
                    "Negative signal outcome recorded for horizon "
                    f"{artifact.payload.get('exit_horizon_label', 'scheduled')}."
                )
            else:
                reason = "Flat signal outcome recorded."
        elif artifact.kind == ArtifactKind.PERFORMANCE_CHECKPOINT:
            pnl_value = float(artifact.payload.get("realized_pnl_today", 0.0))
            cumulative_pnl = round(cumulative_pnl + pnl_value, 2)
            if pnl_value > 0:
                delta = min(pnl_value / 50.0, 5.0)
                win_count += 1
                reason = f"Positive realized PnL recorded: ${pnl_value:.2f}."
            elif pnl_value < 0:
                delta = -min(abs(pnl_value) / 50.0, 5.0)
                loss_count += 1
                reason = f"Negative realized PnL recorded: ${pnl_value:.2f}."
            else:
                reason = "Flat realized PnL recorded."
        elif artifact.kind == ArtifactKind.TRADE_INTENT:
            score = float(artifact.payload.get("score", 0.0))
            delta = 0.5 if score >= 0.8 else 0.0
            reason = "High-conviction trade intent recorded."

        event = ReputationEvent(
            source=artifact.kind.value,
            delta=round(delta, 2),
            reason=reason,
            recorded_at=artifact.created_at,
            refs={
                "artifact_id": artifact.artifact_id,
                "subject_id": artifact.subject_id,
            },
        )
        next_score = _clamp(snapshot.score + delta, self._min_score, self._max_score)
        return replace(
            snapshot,
            score=round(next_score, 2),
            successful_validations=successful_validations,
            successful_executions=successful_executions,
            failed_executions=failed_executions,
            win_count=win_count,
            loss_count=loss_count,
            cumulative_pnl=cumulative_pnl,
            events=snapshot.events + (event,),
            last_updated=artifact.created_at,
        )


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))
