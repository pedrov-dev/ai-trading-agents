from datetime import UTC, datetime

from agent.portfolio import PortfolioSnapshot
from agent.risk import RiskCheckResult, RiskViolation
from agent.signals import NoTradeDecision, TradeIntent
from execution.orders import (
    ExecutionMode,
    ExecutionResult,
    OrderAttempt,
    OrderFailure,
    OrderFill,
    OrderRequest,
    OrderStatus,
)
from validation.artifacts import ArtifactKind, ArtifactStatus, ValidationArtifact

_DEF_TIME = datetime(2026, 4, 3, 12, 0, tzinfo=UTC)


def _make_intent() -> TradeIntent:
    return TradeIntent(
        symbol_id="btc_usd",
        side="buy",
        notional_usd=250.0,
        quantity=0.00367647,
        current_price=68_000.0,
        score=0.91,
        rationale=("ETF approval momentum", "Risk checks cleared"),
        generated_at=_DEF_TIME,
    )


def _make_execution_result(*, status: OrderStatus = OrderStatus.FILLED) -> ExecutionResult:
    request = OrderRequest.from_trade_intent(
        _make_intent(),
        execution_mode=ExecutionMode.DRY_RUN,
        requested_at=datetime(2026, 4, 3, 12, 1, tzinfo=UTC),
        intent_id="intent-123",
        client_order_id="client-123",
    )
    attempt = OrderAttempt(
        attempt_number=1,
        command=("kraken-cli", "add-order", "--validate"),
        status=status,
        started_at=datetime(2026, 4, 3, 12, 1, tzinfo=UTC),
        finished_at=datetime(2026, 4, 3, 12, 2, tzinfo=UTC),
        exit_code=0 if status != OrderStatus.FAILED else 1,
        stdout='{"status": "ok"}',
    )
    fill = None
    failure = None
    if status == OrderStatus.FILLED:
        fill = OrderFill(
            status=OrderStatus.FILLED,
            filled_quantity=request.quantity,
            average_price=request.current_price,
            filled_at=datetime(2026, 4, 3, 12, 2, tzinfo=UTC),
        )
    if status == OrderStatus.FAILED:
        failure = OrderFailure(code="cli_exit_1", message="temporary failure")
    return ExecutionResult(
        request=request,
        status=status,
        attempts=(attempt,),
        fill=fill,
        failure=failure,
        completed_at=datetime(2026, 4, 3, 12, 2, tzinfo=UTC),
    )


def test_trade_intent_artifact_is_serializable_and_stable() -> None:
    intent = _make_intent()

    artifact = ValidationArtifact.from_trade_intent(intent, agent_id="agent-123")
    duplicate = ValidationArtifact.from_trade_intent(intent, agent_id="agent-123")

    payload = artifact.to_dict()

    assert artifact.kind == ArtifactKind.TRADE_INTENT
    assert artifact.status == ArtifactStatus.RECORDED
    assert artifact.artifact_id == duplicate.artifact_id
    assert payload["agent_id"] == "agent-123"
    assert payload["payload"]["symbol_id"] == "btc_usd"
    assert payload["payload"]["confidence_score"] == 0.91
    assert payload["payload"]["expected_move"] == "up"
    assert payload["payload"]["rationale"] == ["ETF approval momentum", "Risk checks cleared"]


def test_signal_outcome_artifact_preserves_event_signal_and_horizon_lineage() -> None:
    execution_result = _make_execution_result()
    closed_position = PortfolioSnapshot(
        total_equity=10_000.0,
        cash_usd=9_750.0,
    )
    del closed_position
    outcome_artifact = ValidationArtifact.from_signal_outcome(
        execution_result,
        symbol_id="btc_usd",
        side="long",
        entry_price=67_000.0,
        opened_at=datetime(2026, 4, 3, 12, 0, tzinfo=UTC),
        exit_horizon_label="30m",
        raw_event_id="evt-123",
        signal_id="signal-123",
        event_type="ETF_APPROVAL",
        realized_pnl_usd=3.68,
    )

    assert outcome_artifact.kind == ArtifactKind.SIGNAL_OUTCOME
    assert outcome_artifact.payload["exit_horizon_label"] == "30m"
    assert outcome_artifact.refs["signal_id"] == "signal-123"
    assert outcome_artifact.refs["raw_event_id"] == "evt-123"
    assert any(item.name == "realized_return_fraction" for item in outcome_artifact.evidence)


def test_no_trade_artifact_captures_default_skip_reason() -> None:
    decision = NoTradeDecision(
        symbol_id="btc_usd",
        reason_code="confidence_below_threshold",
        reason="Confidence 0.62 is below the default threshold 0.70.",
        confidence_score=0.62,
        threshold=0.7,
        score=0.7374,
        event_type="ETF_APPROVAL",
        raw_event_id="evt-no-trade",
        detected_at=_DEF_TIME,
        rationale=("Defaulted to no trade because conviction stayed below threshold.",),
    )

    artifact = ValidationArtifact.from_no_trade_decision(decision, agent_id="agent-123")

    assert artifact.kind == ArtifactKind.NO_TRADE_DECISION
    assert artifact.status == ArtifactStatus.RECORDED
    assert artifact.payload["reason_code"] == "confidence_below_threshold"
    assert artifact.payload["threshold"] == 0.7
    assert artifact.payload["confidence_score"] == 0.62
    assert any(item.name == "confidence_score" and item.value == 0.62 for item in artifact.evidence)


def test_risk_execution_and_performance_artifacts_capture_objective_evidence() -> None:
    risk_result = RiskCheckResult(
        approved=False,
        allowed_notional=0.0,
        violations=(
            RiskViolation(code="max_daily_loss", message="Daily loss limit reached."),
        ),
        notes=("Trading paused for the rest of the session.",),
    )
    risk_artifact = ValidationArtifact.from_risk_check(
        risk_result,
        agent_id="agent-123",
        subject_id="intent-123",
        proposed_notional=300.0,
    )
    execution_artifact = ValidationArtifact.from_execution_result(
        _make_execution_result(),
        agent_id="agent-123",
    )
    performance_artifact = ValidationArtifact.from_performance_checkpoint(
        PortfolioSnapshot(
            total_equity=10_250.0,
            cash_usd=9_000.0,
            realized_pnl_today=125.0,
        ),
        agent_id="agent-123",
        checkpoint_name="daily-close",
    )

    assert risk_artifact.kind == ArtifactKind.PRE_TRADE_RISK_CHECK
    assert any(
        item.name == "approved" and item.value is False
        for item in risk_artifact.evidence
    )
    assert execution_artifact.kind == ArtifactKind.EXECUTION_RESULT
    assert any(
        item.name == "retry_count" and item.value == 0
        for item in execution_artifact.evidence
    )
    assert performance_artifact.kind == ArtifactKind.PERFORMANCE_CHECKPOINT
    assert performance_artifact.payload["checkpoint_name"] == "daily-close"
    assert any(
        item.name == "realized_pnl_today" and item.value == 125.0
        for item in performance_artifact.evidence
    )
