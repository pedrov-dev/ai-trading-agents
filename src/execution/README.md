# Execution

Kraken CLI adapter and immutable order models for paper/live order submission.

## Key modules

- `kraken_cli.py`: CLI adapter for building and submitting orders.
- `orders.py`: order models and execution result types.
- `__init__.py`: package exports.

## Public API

- `KrakenCLIExecutor(config=None)` — `build_command`, `submit_order`, `submit_trade_intent`.
- `KrakenCLIConfig.from_env` for configuration.
- `OrderRequest.from_trade_intent`, models `ExecutionResult`, `OrderAttempt`, `OrderFill`, `OrderFailure`.

## Integration

- Runs the repo-installed `kraken-cli` subprocess; writes audit events to `artifacts/orders_audit.jsonl`; honors `dry_run`/`live_enabled` flags.
- `pip install -r requirements-dev.txt` now installs the local console script automatically via `-e .`.
- If `KRAKEN_API_KEY` and `KRAKEN_API_SECRET` are set, `kraken-cli add-order --validate` can call Kraken’s private `AddOrder` validation endpoint.

## Usage

```python
from execution import KrakenCLIExecutor
exec = KrakenCLIExecutor()
result = exec.submit_order(order_request)
print(result.to_dict())
```

## Notes

- Defaults to simulated/dry-run.
- For a live-connected **paper-only** run, set `KRAKEN_EXECUTION_DRY_RUN=false`, `KRAKEN_LIVE_ENABLED=true`, and keep `KRAKEN_VALIDATE_ONLY=true` so the CLI sends `--validate` without placing a real order.
- Set `KRAKEN_VALIDATE_ONLY=false` only if you intentionally want real submission with exchange credentials.
