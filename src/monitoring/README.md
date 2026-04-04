# Monitoring

Helpers for audit logs, drawdown computation, and PnL reporting.

## Key modules

- `audit_log.py`: parse Kraken-style JSONL audit events and produce summaries.
- `drawdown.py`: drawdown snapshots and metrics.
- `pnl.py`: mark-to-market PnL computations and snapshots.

## Public API

- `load_audit_events(path)`, `build_audit_summary(...)`, `build_audit_summary_from_file(path)`.
- `build_drawdown_snapshot(history)`, `build_pnl_snapshot(portfolio=..., price_quotes=...)`.

## Integration

- Inputs: audit JSONL, `PortfolioSnapshot`, `PriceQuote` (from ingestion). Outputs: artifacts/dashboards/alerts.

## Usage

```python
summary = build_audit_summary_from_file('artifacts/orders_audit.jsonl')
pnl = build_pnl_snapshot(portfolio=portfolio, price_quotes=quotes)
drawdown = build_drawdown_snapshot(history)
```

## Notes

- Run periodically; expected chronological equity history for accurate drawdown metrics.
