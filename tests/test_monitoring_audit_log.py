import json
from datetime import UTC, datetime

from monitoring.audit_log import build_audit_summary_from_file

_DEF_TIME = datetime(2026, 4, 3, 12, 0, tzinfo=UTC)


def test_build_audit_summary_from_file_counts_statuses_and_symbols(tmp_path) -> None:
    audit_path = tmp_path / "orders_audit.jsonl"
    entries = [
        {
            "event": "order_requested",
            "recorded_at": _DEF_TIME.isoformat(),
            "status": "requested",
            "request": {"symbol_id": "btc_usd", "client_order_id": "ord-1"},
        },
        {
            "event": "order_simulated",
            "recorded_at": _DEF_TIME.replace(minute=1).isoformat(),
            "status": "simulated",
            "request": {"symbol_id": "btc_usd", "client_order_id": "ord-1"},
            "fill": {"filled_quantity": 0.01, "average_price": 70000.0},
        },
        {
            "event": "order_failed",
            "recorded_at": _DEF_TIME.replace(minute=2).isoformat(),
            "status": "failed",
            "request": {"symbol_id": "eth_usd", "client_order_id": "ord-2"},
            "attempt": {"stderr": "temporary failure"},
        },
    ]
    audit_path.write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )

    summary = build_audit_summary_from_file(audit_path)

    assert summary.total_events == 3
    assert summary.failure_count == 1
    assert summary.fill_count == 1
    assert summary.status_counts["requested"] == 1
    assert summary.status_counts["simulated"] == 1
    assert summary.symbol_counts["btc_usd"] == 2
    assert summary.symbol_counts["eth_usd"] == 1
    assert summary.recent_events[-1].message.startswith("order_failed")
