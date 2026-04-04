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

- Runs the repo-installed `kraken-cli` subprocess; writes audit events to `artifacts/orders_audit.jsonl`; supports Kraken `paper` validation mode and Kraken `live` submission mode.
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

- The app-level default is Kraken **paper** mode (`--trading-mode paper`).
- Use `python src/main.py --base-dir . --trading-mode paper` for exchange-backed validation without placing a real order.
- Use `python src/main.py --base-dir . --trading-mode live` only when you intentionally want real submission and have set `KRAKEN_CLI_ALLOW_LIVE_SUBMIT=true` plus exchange credentials.
- `src/main.py` blocks unsafe live submission unless `KRAKEN_CLI_ALLOW_LIVE_SUBMIT=true` is explicitly set.
