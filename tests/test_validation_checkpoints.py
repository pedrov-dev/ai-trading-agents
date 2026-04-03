from agent.portfolio import PortfolioSnapshot
from agent.risk import RiskCheckResult
from execution.orders import OrderStatus
from tests.test_validation_artifacts import _make_execution_result
from validation.artifacts import ValidationArtifact
from validation.checkpoints import (
    CheckpointStatus,
    CheckpointType,
    build_checkpoint_from_artifact,
)


def test_build_checkpoint_marks_approved_risk_check_as_passed() -> None:
    artifact = ValidationArtifact.from_risk_check(
        RiskCheckResult(approved=True, allowed_notional=250.0),
        agent_id="agent-123",
        subject_id="intent-123",
        proposed_notional=250.0,
    )

    checkpoint = build_checkpoint_from_artifact(artifact)

    assert checkpoint.checkpoint_type == CheckpointType.PRE_TRADE_RISK
    assert checkpoint.status == CheckpointStatus.PASSED
    assert checkpoint.passed is True
    assert checkpoint.metric_name == "approved"


def test_build_checkpoint_marks_failed_execution_and_negative_pnl_as_failed() -> None:
    execution_checkpoint = build_checkpoint_from_artifact(
        ValidationArtifact.from_execution_result(
            _make_execution_result(status=OrderStatus.FAILED),
            agent_id="agent-123",
        )
    )
    performance_checkpoint = build_checkpoint_from_artifact(
        ValidationArtifact.from_performance_checkpoint(
            PortfolioSnapshot(
                total_equity=9_850.0,
                cash_usd=8_500.0,
                realized_pnl_today=-150.0,
            ),
            agent_id="agent-123",
        )
    )

    assert execution_checkpoint.checkpoint_type == CheckpointType.EXECUTION_OUTCOME
    assert execution_checkpoint.status == CheckpointStatus.FAILED
    assert execution_checkpoint.passed is False
    assert performance_checkpoint.checkpoint_type == CheckpointType.PERFORMANCE
    assert performance_checkpoint.status == CheckpointStatus.FAILED
    assert performance_checkpoint.metric_value == -150.0
