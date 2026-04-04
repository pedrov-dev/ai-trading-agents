# Usage Guide

> PowerShell examples below assume you are in the repo root. The app now auto-loads `.env` and `.runtime.env` from `--base-dir`, so saved Kraken/Sepolia settings are picked up automatically.

## 1) One-time setup

1. Create and activate the virtual environment.
2. Install dependencies.
3. Optionally run the test suite.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
kraken-cli --help
python -m pytest
python src/main.py --help
```

---

## 2) Mode A — Local one-shot dry run

Use this for the default, safest demo run.

1. Activate the environment.
2. Run one complete ingest → detect → strategy → execution cycle.
3. Inspect the generated artifacts.

```powershell
.\.venv\Scripts\Activate.ps1
python src/main.py --base-dir .
```

Artifacts written after the run:

- `artifacts/run_summary.json`
- `artifacts/orders_audit.jsonl`
- `artifacts/validation_artifacts.jsonl`
- `artifacts/validation_checkpoints.jsonl`

---

## 3) Mode B — Local scheduler service

Use this to keep the pipeline running continuously.

1. Start the service.
2. Stop it with `Ctrl+C` when you are done.

```powershell
.\.venv\Scripts\Activate.ps1
python src/main.py --base-dir . --serve
```

Optional custom intervals:

```powershell
python src/main.py --base-dir . --serve `
  --rss-interval-seconds 120 `
  --prices-interval-seconds 60 `
  --detection-interval-seconds 60 `
  --execution-interval-seconds 60
```

---

## 4) Mode C — Live-connected Kraken paper trading

Use this to reach Kraken in **paper-only validation mode** without placing a real order.

1. Activate the environment.
2. Put `KRAKEN_API_KEY` and `KRAKEN_API_SECRET` in `.env` if you want the repo-installed `kraken-cli` to call Kraken’s private `AddOrder` validation endpoint.
3. Run the safe paper-mode flag.

```powershell
.\.venv\Scripts\Activate.ps1
python src/main.py --base-dir . --kraken-paper
```

Optional shell overrides if you do not want to use `.env`:

```powershell
$env:KRAKEN_API_KEY = "..."
$env:KRAKEN_API_SECRET = "..."
python src/main.py --base-dir . --kraken-paper
```

> `--kraken-paper` forces `KRAKEN_EXECUTION_DRY_RUN=false`, `KRAKEN_LIVE_ENABLED=true`, and `KRAKEN_VALIDATE_ONLY=true` for the current run.
>
> The output now includes an `execution_config` block so you can confirm `live_connected_paper_trading=true` and `will_submit_real_orders=false` before trusting the run.

---

## 5) Mode D — Shared Sepolia status run

Use this to run the app against the shared ERC-8004 Sepolia setup without sending optional transactions.

1. Open `.env.example` and copy the shared-contract values into your shell or env loader.
2. Set the required Sepolia variables.
3. Run the one-shot Sepolia cycle.

```powershell
.\.venv\Scripts\Activate.ps1
$env:TRADING_RUNTIME_MODE = "sepolia"
$env:SEPOLIA_RPC_URL = "https://ethereum-sepolia-rpc.publicnode.com"
$env:SEPOLIA_CHAIN_ID = "11155111"
# Set the five shared contract addresses from `.env.example`
python src/main.py --runtime-mode sepolia --base-dir .
```

---

## 6) Mode E — Shared Sepolia on-chain actions

Use this when you want to register the agent, claim the allocation, submit intents, or post checkpoints.

1. Complete all variables from **Mode D**.
2. Add the required signing keys and optional agent profile values.
3. Run the specific action you need.

```powershell
.\.venv\Scripts\Activate.ps1
$env:PRIVATE_KEY = "0x..."
$env:AGENT_WALLET_PRIVATE_KEY = "0x..."
$env:AGENT_DISPLAY_NAME = "AI Trading Agent Demo"
$env:AGENT_STRATEGY_NAME = "simple_event_driven"
```

Available commands:

```powershell
python src/main.py --runtime-mode sepolia --base-dir . --register-agent
python src/main.py --runtime-mode sepolia --base-dir . --claim-allocation
python src/main.py --runtime-mode sepolia --base-dir . --submit-onchain
python src/main.py --runtime-mode sepolia --base-dir . --post-checkpoints
python src/main.py --runtime-mode sepolia --base-dir . --full-flow
```

If registration succeeds, the numeric `AGENT_ID` is persisted to `.runtime.env` for reuse.

---

## Quick troubleshooting

- Check all CLI options with `python src/main.py --help`.
- If Sepolia mode reports missing config, export the required variables first.
- `--serve` cannot be combined with the on-chain action flags.
