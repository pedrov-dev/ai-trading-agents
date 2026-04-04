# AI Trading Agents

Local-first event-driven crypto trading MVP for paper/dry-run execution on Kraken-style flows.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
python -m pytest
python src/main.py --base-dir .
```

## What the local dry run does

- ingests RSS articles and Kraken price quotes
- classifies crypto-relevant events
- generates explainable trade intents
- runs conservative risk-aware paper execution
- writes audit logs, validation artifacts, checkpoints, and a run summary

## Output files

After a run, inspect:

- `artifacts/orders_audit.jsonl`
- `artifacts/validation_artifacts.jsonl`
- `artifacts/validation_checkpoints.jsonl`
- `artifacts/run_summary.json`

## Notes

- Default mode is **safe dry run**.
- No live trading is enabled by default.
- For optional external-backed mode, copy values from `.env.example`.
