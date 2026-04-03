from agent.portfolio import PortfolioSnapshot
from agent.risk import RiskCheckResult
from execution.orders import OrderStatus
from identity.reputation import ReputationEngine, ReputationSnapshot
from tests.test_validation_artifacts import _make_execution_result
from validation.artifacts import ValidationArtifact


def test_reputation_engine_rewards_verified_success_and_positive_pnl() -> None:
    engine = ReputationEngine()
    snapshot = ReputationSnapshot(agent_id="agent-123")

    updated = engine.apply_artifacts(
        snapshot,
        (
            ValidationArtifact.from_risk_check(
                RiskCheckResult(approved=True, allowed_notional=250.0),
                agent_id="agent-123",
                subject_id="intent-123",
                proposed_notional=250.0,
            ),
            ValidationArtifact.from_execution_result(
                _make_execution_result(status=OrderStatus.FILLED),
                agent_id="agent-123",
            ),
            ValidationArtifact.from_performance_checkpoint(
                PortfolioSnapshot(
                    total_equity=10_250.0,
                    cash_usd=9_000.0,
                    realized_pnl_today=125.0,
                ),
                agent_id="agent-123",
            ),
        ),
    )

    assert updated.score > snapshot.score
    assert updated.successful_validations >= 1
    assert updated.successful_executions == 1
    assert updated.win_count == 1
    assert updated.cumulative_pnl == 125.0


def test_reputation_engine_penalizes_failed_execution_and_losses() -> None:
    engine = ReputationEngine()
    snapshot = ReputationSnapshot(agent_id="agent-123")

    updated = engine.apply_artifacts(
        snapshot,
        (
            ValidationArtifact.from_execution_result(
                _make_execution_result(status=OrderStatus.FAILED),
                agent_id="agent-123",
            ),
            ValidationArtifact.from_performance_checkpoint(
                PortfolioSnapshot(
                    total_equity=9_850.0,
                    cash_usd=8_500.0,
                    realized_pnl_today=-150.0,
                ),
                agent_id="agent-123",
            ),
        ),
    )

    assert updated.score < snapshot.score
    assert updated.failed_executions == 1
    assert updated.loss_count == 1
    assert updated.cumulative_pnl == -150.0
