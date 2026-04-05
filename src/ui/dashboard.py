# ruff: noqa: E402
"""Streamlit dashboard for portfolio, performance, and runtime controls."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

st: Any | None
try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - handled at runtime.
    st = None

try:
    import streamlit_autorefresh as _streamlit_autorefresh  # type: ignore[import-untyped]
except ModuleNotFoundError:  # pragma: no cover - handled at runtime.
    _streamlit_autorefresh = None

st_autorefresh = (
    cast(Callable[..., Any], _streamlit_autorefresh.st_autorefresh)
    if _streamlit_autorefresh is not None
    else None
)

from ingestion.rss_config import RSS_FEED_GROUPS
from main import (
    ROOT_DIR,
    RuntimeCycleResult,
    TradingApplication,
    build_local_demo_app,
    build_runtime_preflight,
)


def load_json_file(path: str | Path) -> dict[str, Any] | None:
    """Load one JSON file if it exists and is valid."""
    file_path = Path(path)
    if not file_path.exists():
        return None
    try:
        loaded = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL records while skipping malformed lines."""
    file_path = Path(path)
    if not file_path.exists():
        return []

    records: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def build_equity_history(
    artifact_records: Sequence[Mapping[str, Any]],
    *,
    latest_payload: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build a simple equity series for charting from persisted artifacts."""
    history: list[dict[str, Any]] = []
    for record in artifact_records:
        if record.get("kind") != "performance_checkpoint":
            continue
        payload = record.get("payload")
        if not isinstance(payload, Mapping):
            continue
        timestamp = str(payload.get("as_of") or record.get("created_at") or "")
        history.append(
            {
                "timestamp": timestamp,
                "equity": _as_float(payload.get("total_equity")),
                "cash_usd": _as_float(payload.get("cash_usd")),
                "realized_pnl_today": _as_float(payload.get("realized_pnl_today")),
            }
        )

    if history:
        return sorted(history, key=lambda row: row["timestamp"])

    if latest_payload is None:
        return []

    portfolio = latest_payload.get("portfolio")
    pnl_snapshot = latest_payload.get("pnl_snapshot")
    if not isinstance(portfolio, Mapping):
        return []

    return [
        {
            "timestamp": str(portfolio.get("as_of") or "latest"),
            "equity": _as_float(portfolio.get("total_equity")),
            "cash_usd": _as_float(portfolio.get("cash_usd")),
            "realized_pnl_today": _as_float(portfolio.get("realized_pnl_today")),
            "net_pnl_usd": _as_float(
                pnl_snapshot.get("net_pnl_usd") if isinstance(pnl_snapshot, Mapping) else 0.0
            ),
        }
    ]


def build_position_rows(payload: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Flatten per-position PnL details for a simple table."""
    if payload is None:
        return []
    pnl_snapshot = payload.get("pnl_snapshot")
    if not isinstance(pnl_snapshot, Mapping):
        return []
    position_pnl = pnl_snapshot.get("position_pnl")
    if not isinstance(position_pnl, Mapping):
        return []

    rows: list[dict[str, Any]] = []
    for symbol_id, details in position_pnl.items():
        if not isinstance(details, Mapping):
            continue
        rows.append(
            {
                "symbol_id": str(details.get("symbol_id") or symbol_id),
                "side": str(details.get("side") or "-"),
                "quantity": _as_float(details.get("quantity")),
                "entry_price": _as_float(details.get("entry_price")),
                "current_price": _as_float(details.get("current_price")),
                "market_value_usd": _as_float(details.get("market_value_usd")),
                "unrealized_pnl_usd": _as_float(details.get("unrealized_pnl_usd")),
                "unrealized_return_pct": round(
                    _as_float(details.get("unrealized_return_fraction")) * 100,
                    2,
                ),
            }
        )

    return sorted(rows, key=lambda row: abs(row["market_value_usd"]), reverse=True)


def build_trade_intent_rows(payload: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Flatten recent trade intents for display."""
    if payload is None:
        return []
    trade_intents = payload.get("trade_intents")
    if not isinstance(trade_intents, Sequence):
        return []

    rows: list[dict[str, Any]] = []
    for item in trade_intents:
        if not isinstance(item, Mapping):
            continue
        rows.append(
            {
                "symbol_id": str(item.get("symbol_id") or "-"),
                "side": str(item.get("side") or "-"),
                "score": _as_float(item.get("score")),
                "notional_usd": _as_float(item.get("notional_usd")),
                "quantity": _as_float(item.get("quantity")),
                "generated_at": str(item.get("generated_at") or ""),
            }
        )
    return rows


def build_execution_rows(payload: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Flatten execution outcomes for a simple table."""
    if payload is None:
        return []
    execution_results = payload.get("execution_results")
    if not isinstance(execution_results, Sequence):
        return []

    rows: list[dict[str, Any]] = []
    for item in execution_results:
        if not isinstance(item, Mapping):
            continue
        request_payload = item.get("request")
        fill_payload = item.get("fill")
        request: Mapping[str, Any] = (
            cast(Mapping[str, Any], request_payload)
            if isinstance(request_payload, Mapping)
            else {}
        )
        fill: Mapping[str, Any] = (
            cast(Mapping[str, Any], fill_payload)
            if isinstance(fill_payload, Mapping)
            else {}
        )
        rows.append(
            {
                "symbol_id": str(
                    request.get("symbol_id") or item.get("symbol_id") or "-"
                ),
                "side": str(request.get("side") or item.get("side") or "-"),
                "status": str(item.get("status") or "unknown"),
                "filled_quantity": _as_float(
                    fill.get("filled_quantity") or item.get("filled_quantity")
                ),
                "average_price": _as_float(
                    fill.get("average_price") or item.get("average_price")
                ),
                "client_order_id": str(
                    request.get("client_order_id") or item.get("client_order_id") or "-"
                ),
                "completed_at": str(item.get("completed_at") or ""),
            }
        )
    return rows


def build_activity_rows(activity_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Flatten live activity updates into a concise table for the dashboard."""
    rows: list[dict[str, Any]] = []
    for record in activity_records:
        affects = record.get("affects")
        affects_text = ""
        if isinstance(affects, Sequence) and not isinstance(affects, (str, bytes)):
            affects_text = ", ".join(str(item) for item in affects)
        rows.append(
            {
                "timestamp": str(record.get("timestamp") or ""),
                "stage": str(record.get("stage") or "runtime"),
                "action": str(record.get("action") or "-"),
                "status": str(record.get("status") or "-"),
                "summary": str(record.get("summary") or ""),
                "affects": affects_text,
            }
        )
    return rows


def main() -> None:
    render_dashboard()


def render_dashboard() -> None:
    streamlit = cast(Any, _ensure_streamlit())
    streamlit.set_page_config(page_title="AI Trading Dashboard", layout="wide")
    _initialize_session_state(streamlit)

    streamlit.title("📈 AI Trading Agents Dashboard")
    streamlit.caption(
        "Super basic local UI for portfolio/performance visibility and runtime controls."
    )
    _apply_auto_refresh(streamlit)

    with streamlit.sidebar:
        streamlit.header("Controls")
        base_dir_value = streamlit.text_input("Base directory", value=str(ROOT_DIR))
        base_dir = Path(base_dir_value).expanduser()
        feed_group = streamlit.selectbox(
            "Feed group",
            options=sorted(RSS_FEED_GROUPS.keys()),
            index=0,
        )
        trading_mode = streamlit.selectbox("Trading mode", options=["paper", "live"], index=0)
        identity_layer = streamlit.selectbox(
            "Identity layer",
            options=["none", "erc8004"],
            index=0,
        )
        streamlit.divider()
        rss_interval = streamlit.number_input(
            "RSS interval (sec)",
            min_value=10,
            value=120,
            step=10,
        )
        prices_interval = streamlit.number_input(
            "Prices interval (sec)",
            min_value=10,
            value=60,
            step=10,
        )
        detection_interval = streamlit.number_input(
            "Detection interval (sec)",
            min_value=10,
            value=60,
            step=10,
        )
        execution_interval = streamlit.number_input(
            "Execution interval (sec)",
            min_value=10,
            value=60,
            step=10,
        )
        allow_live_actions = streamlit.checkbox(
            "I understand live mode can submit real orders",
            value=False,
        )

        preflight_clicked = streamlit.button("Run preflight", width="stretch")
        run_cycle_clicked = streamlit.button(
            "Run one-shot cycle",
            width="stretch",
        )
        start_scheduler_clicked = streamlit.button(
            "Start scheduler",
            width="stretch",
        )
        stop_scheduler_clicked = streamlit.button("Stop scheduler", width="stretch")

    artifact_paths = _artifact_paths(base_dir)
    _sync_summary_payload(streamlit, artifact_paths["summary"])

    current_app = _get_or_create_app(
        streamlit,
        base_dir=base_dir,
        trading_mode=trading_mode,
        identity_layer=identity_layer,
    )
    execution_summary = current_app.execution_mode_summary()

    if preflight_clicked:
        streamlit.session_state["last_action_result"] = build_runtime_preflight(
            trading_mode=trading_mode,
            identity_layer=identity_layer,
            base_dir=base_dir,
        )
        streamlit.session_state["status_message"] = "Preflight completed."

    if run_cycle_clicked:
        if execution_summary.get("will_submit_real_orders") and not allow_live_actions:
            streamlit.session_state["status_message"] = (
                "Live mode is armed. Tick the confirmation box before running it from the UI."
            )
        else:
            result = current_app.run_cycle(feed_group=feed_group)
            payload = _payload_from_result(result, current_app)
            streamlit.session_state["latest_result"] = result
            streamlit.session_state["latest_payload"] = payload
            streamlit.session_state["last_action_result"] = payload
            streamlit.session_state["status_message"] = (
                "Trading cycle completed and dashboard data refreshed."
            )

    if start_scheduler_clicked:
        if streamlit.session_state.get("scheduler_running"):
            streamlit.session_state["status_message"] = "Scheduler is already running."
        elif execution_summary.get("will_submit_real_orders") and not allow_live_actions:
            streamlit.session_state["status_message"] = (
                "Live mode is armed. Tick the confirmation box before starting the scheduler."
            )
        else:
            app = _get_or_create_app(
                streamlit,
                base_dir=base_dir,
                trading_mode=trading_mode,
                identity_layer=identity_layer,
                force_rebuild=True,
            )
            scheduler = app.wire_scheduler(
                feed_group=feed_group,
                rss_interval_seconds=int(rss_interval),
                prices_interval_seconds=int(prices_interval),
                detection_interval_seconds=int(detection_interval),
                execution_interval_seconds=int(execution_interval),
            )
            scheduler.start()
            streamlit.session_state["app"] = app
            streamlit.session_state["scheduler"] = scheduler
            streamlit.session_state["scheduler_running"] = True
            streamlit.session_state["status_message"] = "Scheduler started."

    if stop_scheduler_clicked:
        _stop_scheduler(streamlit)
        streamlit.session_state["status_message"] = "Scheduler stopped."

    _render_status_banner(streamlit, execution_summary)

    status_message = streamlit.session_state.get("status_message")
    if status_message:
        streamlit.info(status_message)

    latest_payload = cast(
        dict[str, Any] | None,
        streamlit.session_state.get("latest_payload")
        or load_json_file(artifact_paths["summary"]),
    )
    artifact_records = load_jsonl_records(artifact_paths["artifacts"])
    history_rows = build_equity_history(artifact_records, latest_payload=latest_payload)
    position_rows = build_position_rows(latest_payload)
    trade_intent_rows = build_trade_intent_rows(latest_payload)
    execution_rows = build_execution_rows(latest_payload)

    _render_kpis(streamlit, latest_payload)

    streamlit.subheader("Performance history")
    if history_rows:
        streamlit.line_chart(
            {
                "equity_usd": [row["equity"] for row in history_rows],
                "cash_usd": [row["cash_usd"] for row in history_rows],
            }
        )
        with streamlit.expander("History rows", expanded=False):
            streamlit.dataframe(history_rows, width='stretch')
    else:
        streamlit.write(
            "No performance history yet. Run a cycle or wait for the next auto-refresh."
        )

    left_column, right_column = streamlit.columns(2)
    with left_column:
        streamlit.subheader("Open positions")
        if position_rows:
            streamlit.dataframe(position_rows, width='stretch')
        else:
            streamlit.write("No open positions recorded yet.")

        streamlit.subheader("Recent trade intents")
        if trade_intent_rows:
            streamlit.dataframe(trade_intent_rows, width='stretch')
        else:
            streamlit.write("No trade intents available yet.")

    with right_column:
        streamlit.subheader("Execution results")
        if execution_rows:
            streamlit.dataframe(execution_rows, width='stretch')
        else:
            streamlit.write("No execution results available yet.")

        activity_records = load_jsonl_records(artifact_paths["activity"])
        activity_rows = build_activity_rows(activity_records)
        streamlit.subheader("Live activity log")
        if activity_rows:
            streamlit.dataframe(activity_rows[-15:], width='stretch')
        else:
            streamlit.write("No incremental actions written yet.")

        audit_records = load_jsonl_records(artifact_paths["audit"])
        streamlit.subheader("Audit log")
        if audit_records:
            streamlit.dataframe(audit_records[-10:], width='stretch')
        else:
            streamlit.write("No audit events written yet.")

    streamlit.subheader("Runtime details")
    runtime_details = {
        "execution_config": execution_summary,
        "base_dir": str(base_dir),
        "scheduler_running": bool(streamlit.session_state.get("scheduler_running")),
        "artifact_paths": {key: str(value) for key, value in artifact_paths.items()},
    }
    streamlit.json(runtime_details)

    last_action_result = streamlit.session_state.get("last_action_result")
    if last_action_result is not None:
        with streamlit.expander("Last action result", expanded=False):
            streamlit.json(last_action_result)

    if latest_payload is not None:
        with streamlit.expander("Latest run summary JSON", expanded=False):
            streamlit.json(latest_payload)


def _apply_auto_refresh(streamlit: Any, *, interval_ms: int = 30_000) -> None:
    if st_autorefresh is not None:
        st_autorefresh(interval=interval_ms, key="dashboard_auto_refresh")
        streamlit.caption(f"Auto-refreshing every {interval_ms // 1000} seconds.")
        return

    streamlit.caption(
        f"Auto-refresh requested every {interval_ms // 1000} seconds, but the optional "
        "`streamlit-autorefresh` package is not installed."
    )


def _initialize_session_state(streamlit: Any) -> None:
    defaults = {
        "app": None,
        "app_config": None,
        "latest_result": None,
        "latest_payload": None,
        "last_action_result": None,
        "scheduler": None,
        "scheduler_running": False,
        "status_message": None,
    }
    for key, value in defaults.items():
        streamlit.session_state.setdefault(key, value)


def _artifact_paths(base_dir: Path) -> dict[str, Path]:
    artifacts_dir = base_dir / "artifacts"
    return {
        "summary": artifacts_dir / "run_summary.json",
        "artifacts": artifacts_dir / "validation_artifacts.jsonl",
        "audit": artifacts_dir / "orders_audit.jsonl",
        "activity": artifacts_dir / "activity_log.jsonl",
        "checkpoints": artifacts_dir / "validation_checkpoints.jsonl",
    }


def _sync_summary_payload(streamlit: Any, summary_path: Path) -> None:
    summary_payload = load_json_file(summary_path)
    if summary_payload is not None:
        streamlit.session_state["latest_payload"] = summary_payload


def _get_or_create_app(
    streamlit: Any,
    *,
    base_dir: Path,
    trading_mode: str,
    identity_layer: str,
    force_rebuild: bool = False,
) -> TradingApplication:
    requested_config = {
        "base_dir": str(base_dir),
        "trading_mode": trading_mode,
        "identity_layer": identity_layer,
    }
    config_changed = streamlit.session_state.get("app_config") != requested_config
    if config_changed and streamlit.session_state.get("scheduler_running"):
        _stop_scheduler(streamlit)

    if force_rebuild or config_changed or streamlit.session_state.get("app") is None:
        app = build_local_demo_app(
            base_dir=base_dir,
            trading_mode=trading_mode,
            identity_layer=identity_layer,
        )
        streamlit.session_state["app"] = app
        streamlit.session_state["app_config"] = requested_config
    return cast(TradingApplication, streamlit.session_state["app"])


def _payload_from_result(result: RuntimeCycleResult, app: TradingApplication) -> dict[str, Any]:
    payload = result.to_summary_dict()
    payload["trading_mode"] = getattr(app, "_trading_mode", "paper")
    payload["identity_layer"] = getattr(app, "_identity_layer", "none")
    payload["runtime_mode"] = getattr(app, "_runtime_mode", "local")
    payload["execution_config"] = app.execution_mode_summary()
    return payload


def _stop_scheduler(streamlit: Any) -> None:
    scheduler = streamlit.session_state.get("scheduler")
    if scheduler is not None:
        scheduler.shutdown(wait=True)
    streamlit.session_state["scheduler"] = None
    streamlit.session_state["scheduler_running"] = False


def _render_status_banner(streamlit: Any, execution_summary: Mapping[str, Any]) -> None:
    if execution_summary.get("will_submit_real_orders"):
        streamlit.error("Live order submission is enabled for this configuration.")
    elif execution_summary.get("live_connected_paper_trading"):
        streamlit.success("Paper mode is active: Kraken validation only, no real orders.")
    else:
        streamlit.warning("Review the execution config before running the app.")


def _render_kpis(streamlit: Any, latest_payload: Mapping[str, Any] | None) -> None:
    streamlit.subheader("Portfolio & performance")
    if latest_payload is None:
        streamlit.write("No run data yet. Use the controls to run or refresh the dashboard.")
        return

    portfolio = latest_payload.get("portfolio")
    pnl_snapshot = latest_payload.get("pnl_snapshot")
    drawdown_snapshot = latest_payload.get("drawdown_snapshot")
    journal_summary = latest_payload.get("journal_summary")

    portfolio_map = portfolio if isinstance(portfolio, Mapping) else {}
    pnl_map = pnl_snapshot if isinstance(pnl_snapshot, Mapping) else {}
    drawdown_map = drawdown_snapshot if isinstance(drawdown_snapshot, Mapping) else {}
    journal_map = journal_summary if isinstance(journal_summary, Mapping) else {}

    first_row = streamlit.columns(4)
    first_row[0].metric("Total equity", _format_usd(portfolio_map.get("total_equity")))
    first_row[1].metric("Cash", _format_usd(portfolio_map.get("cash_usd")))
    first_row[2].metric("Net PnL", _format_usd(pnl_map.get("net_pnl_usd")))
    first_row[3].metric("Unrealized PnL", _format_usd(pnl_map.get("unrealized_pnl_usd")))

    second_row = streamlit.columns(4)
    second_row[0].metric(
        "Open positions",
        str(int(_as_float(portfolio_map.get("open_position_count")))),
    )
    second_row[1].metric("Win rate", _format_pct(pnl_map.get("win_rate"), scale=100))
    second_row[2].metric(
        "Max drawdown",
        _format_pct(drawdown_map.get("max_drawdown_fraction"), scale=100),
    )
    second_row[3].metric(
        "Closed trades",
        str(int(_as_float(journal_map.get("closed_trade_count")))),
    )


def _ensure_streamlit() -> Any:
    if st is None:
        raise RuntimeError(
            "Streamlit is not installed. Run `pip install -r requirements-dev.txt` first."
        )
    return st


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_pct(value: Any, *, scale: float = 1.0) -> str:
    return f"{_as_float(value) * scale:.2f}%"


def _format_usd(value: Any) -> str:
    return f"${_as_float(value):,.2f}"


if __name__ == "__main__":
    main()
