from datetime import UTC, datetime

from agent.signals import TradeIntent
from execution.orders import ExecutionMode, OrderRequest, OrderStatus


def _make_intent() -> TradeIntent:
    return TradeIntent(
        symbol_id="btc_usd",
        side="buy",
        notional_usd=250.0,
        quantity=0.00367647,
        current_price=68_000.0,
        score=0.91,
        rationale=("ETF approval momentum", "Risk checks cleared"),
        generated_at=datetime(2026, 4, 3, 12, 0, tzinfo=UTC),
    )


def test_order_request_from_trade_intent_is_json_serializable() -> None:
    request = OrderRequest.from_trade_intent(
        _make_intent(),
        execution_mode=ExecutionMode.DRY_RUN,
        requested_at=datetime(2026, 4, 3, 12, 1, tzinfo=UTC),
        intent_id="intent-123",
    )

    payload = request.to_dict()

    assert request.status == OrderStatus.REQUESTED
    assert request.intent_id == "intent-123"
    assert request.client_order_id.startswith("intent-123-")
    assert payload["execution_mode"] == "dry_run"
    assert payload["requested_at"] == "2026-04-03T12:01:00+00:00"
    assert payload["rationale"] == ["ETF approval momentum", "Risk checks cleared"]
