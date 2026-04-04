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

- Runs `kraken-cli` subprocess; writes audit events to `artifacts/orders_audit.jsonl`; honors `dry_run`/`live_enabled` flags.

## Usage

```python
from execution import KrakenCLIExecutor
exec = KrakenCLIExecutor()
result = exec.submit_order(order_request)
print(result.to_dict())
```

## Notes

- Defaults to simulated/dry-run; enable live with `KRAKEN_LIVE_ENABLED` and provide exchange credentials for production.
