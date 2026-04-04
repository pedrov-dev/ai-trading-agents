# Top-level entrypoints

Kraken paper/live harness and lightweight scheduler for running ingestion, detection, strategy, and execution flows.

## Top-level scripts

- `main.py`: orchestrates ingest → detect → strategy → execute → validate flows.
- `info_scheduler.py`: thin scheduler wrapper exposing jobs for recurring ingestion/detection/execution tasks.

## How to run

Typical invocations:

```powershell
python -m src.main --base-dir ./artifacts --feed-group market_news --trading-mode paper
python -m src.main --serve --trading-mode paper --rss-interval-seconds 120 --prices-interval-seconds 60
python -m src.main --trading-mode paper --identity-layer erc8004 --full-flow
```

## Integration

- Orchestrates ingestion, detection, strategy, execution, identity, monitoring, storage, and validation layers.

## Notes

- Honors `TRADING_MODE`, `IDENTITY_LAYER`, `AGENT_*`, and `KRAKEN_AUDIT_LOG_PATH`; persists runtime artifacts under the provided `--base-dir`.
