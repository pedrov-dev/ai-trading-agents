"""Objective checkpoint records derived from validation artifacts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from validation.artifacts import ArtifactKind, ValidationArtifact


class CheckpointType(StrEnum):
    """High-level checkpoint categories aligned with the trading lifecycle."""

    TRADE_INTENT = "trade_intent"
    NO_TRADE_DECISION = "no_trade_decision"
    PRE_TRADE_RISK = "pre_trade_risk"
    EXECUTION_OUTCOME = "execution_outcome"
    SIGNAL_OUTCOME = "signal_outcome"
    PERFORMANCE = "performance"


class CheckpointStatus(StrEnum):
    """Pass/fail state for a measured checkpoint."""

    PASSED = "passed"
    FAILED = "failed"


@dataclass(frozen=True)
class ValidationCheckpoint:
    """A measurable checkpoint that judges can inspect directly."""

    checkpoint_id: str
    checkpoint_type: CheckpointType
    artifact_id: str
    subject_id: str
    metric_name: str
    metric_value: bool | float | int | str
    status: CheckpointStatus
    passed: bool
    agent_id: str | None = None
    threshold: bool | float | int | str | None = None
    notes: tuple[str, ...] = ()
    recorded_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable checkpoint payload."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_type": self.checkpoint_type.value,
            "artifact_id": self.artifact_id,
            "subject_id": self.subject_id,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "status": self.status.value,
            "passed": self.passed,
            "agent_id": self.agent_id,
            "threshold": self.threshold,
            "notes": list(self.notes),
            "recorded_at": self.recorded_at.isoformat(),
        }


def build_checkpoint_from_artifact(artifact: ValidationArtifact) -> ValidationCheckpoint:
    """Derive one objective checkpoint from a validation artifact."""
    metric_name: str
    metric_value: bool | float | int | str
    threshold: bool | float | int | str | None
    checkpoint_type: CheckpointType
    notes: tuple[str, ...]

    if artifact.kind == ArtifactKind.TRADE_INTENT:
        metric_name = "notional_usd"
        metric_value = float(artifact.payload.get("notional_usd", 0.0))
        threshold = 0.0
        passed = metric_value > 0
        checkpoint_type = CheckpointType.TRADE_INTENT
        notes = ("Trade intent recorded for downstream execution.",)
    elif artifact.kind == ArtifactKind.NO_TRADE_DECISION:
        metric_name = "reason_code"
        metric_value = str(artifact.payload.get("reason_code", "no_trade"))
        threshold = artifact.payload.get("threshold")
        passed = True
        checkpoint_type = CheckpointType.NO_TRADE_DECISION
        notes = (str(artifact.payload.get("reason", "Strategy defaulted to no trade.")),)
    elif artifact.kind == ArtifactKind.PRE_TRADE_RISK_CHECK:
        metric_name = "approved"
        metric_value = bool(artifact.payload.get("approved", False))
        threshold = True
        passed = bool(metric_value)
        checkpoint_type = CheckpointType.PRE_TRADE_RISK
        notes = tuple(str(note) for note in artifact.payload.get("notes", []))
    elif artifact.kind == ArtifactKind.EXECUTION_RESULT:
        metric_name = "execution_status"
        metric_value = str(artifact.payload.get("status", "failed"))
        threshold = "simulated|submitted|filled"
        passed = metric_value in {"simulated", "submitted", "filled"}
        checkpoint_type = CheckpointType.EXECUTION_OUTCOME
        notes = (f"Execution result captured with status={metric_value}.",)
    elif artifact.kind == ArtifactKind.SIGNAL_OUTCOME:
        metric_name = "realized_return_fraction"
        metric_value = float(artifact.payload.get("realized_return_fraction", 0.0))
        threshold = 0.0
        passed = metric_value >= 0
        checkpoint_type = CheckpointType.SIGNAL_OUTCOME
        notes = (
            "Signal horizon outcome recorded for "
            f"{artifact.payload.get('exit_horizon_label', 'scheduled')}.",
        )
    elif artifact.kind == ArtifactKind.PERFORMANCE_CHECKPOINT:
        metric_name = "realized_pnl_today"
        metric_value = float(artifact.payload.get("realized_pnl_today", 0.0))
        threshold = 0.0
        passed = metric_value >= 0
        checkpoint_type = CheckpointType.PERFORMANCE
        notes = (f"Performance snapshot {artifact.payload.get('checkpoint_name', 'pnl_snapshot')}",)
    else:
        raise ValueError(f"Unsupported artifact kind: {artifact.kind}")

    checkpoint_id = _stable_checkpoint_id(artifact.artifact_id, checkpoint_type.value, metric_name)
    return ValidationCheckpoint(
        checkpoint_id=checkpoint_id,
        checkpoint_type=checkpoint_type,
        artifact_id=artifact.artifact_id,
        subject_id=artifact.subject_id,
        metric_name=metric_name,
        metric_value=metric_value,
        status=CheckpointStatus.PASSED if passed else CheckpointStatus.FAILED,
        passed=passed,
        agent_id=artifact.agent_id,
        threshold=threshold,
        notes=notes,
        recorded_at=artifact.created_at,
    )


def build_checkpoints(
    artifacts: tuple[ValidationArtifact, ...] | list[ValidationArtifact],
) -> tuple[ValidationCheckpoint, ...]:
    """Build checkpoints for a collection of artifacts."""
    return tuple(build_checkpoint_from_artifact(artifact) for artifact in artifacts)


def _stable_checkpoint_id(artifact_id: str, checkpoint_type: str, metric_name: str) -> str:
    hash_value = hashlib.sha256(
        f"{artifact_id}|{checkpoint_type}|{metric_name}".encode()
    ).hexdigest()[:12]
    return f"checkpoint-{hash_value}"
