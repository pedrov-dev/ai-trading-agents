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

## 2) Mode A — Kraken Paper Trading

This is the default and safest mode. Orders go through Kraken's validation path and are **not** submitted live.

1. Activate the environment.
2. Optionally put `KRAKEN_API_KEY` and `KRAKEN_API_SECRET` in `.env` if you want Kraken's private `AddOrder` validation endpoint.
3. Run one complete ingest → detect → strategy → execution cycle.

```powershell
.\.venv\Scripts\Activate.ps1
python src/main.py --base-dir . --trading-mode paper
```

Artifacts written after the run:

- `artifacts/run_summary.json`
- `artifacts/orders_audit.jsonl`
- `artifacts/validation_artifacts.jsonl`
- `artifacts/validation_checkpoints.jsonl`

> Check the `execution_config` block in the output. It should show `trading_mode: paper`, `live_connected_paper_trading: true`, and `will_submit_real_orders: false`.

---

## 3) Mode B — Kraken Live Trading

Use this only when you intentionally want real live order submission.

1. Activate the environment.
2. Set `KRAKEN_API_KEY`, `KRAKEN_API_SECRET`, and `KRAKEN_CLI_ALLOW_LIVE_SUBMIT=true`.
3. Run the live mode explicitly.

```powershell
.\.venv\Scripts\Activate.ps1
$env:KRAKEN_API_KEY = "..."
$env:KRAKEN_API_SECRET = "..."
$env:KRAKEN_CLI_ALLOW_LIVE_SUBMIT = "true"
python src/main.py --base-dir . --trading-mode live
```

> The app blocks live trading unless `KRAKEN_CLI_ALLOW_LIVE_SUBMIT=true` is present.

---

## 4) Optional scheduler service

Use this to keep the pipeline running continuously in either paper or live mode.

```powershell
.\.venv\Scripts\Activate.ps1
python src/main.py --base-dir . --trading-mode paper --serve
```

Optional custom intervals:

```powershell
python src/main.py --base-dir . --trading-mode paper --serve `
  --rss-interval-seconds 120 `
  --prices-interval-seconds 60 `
  --detection-interval-seconds 60 `
  --execution-interval-seconds 60
```

---

## 5) Optional ERC-8004 identity layer

Use this when you want the shared Sepolia identity / reputation / validation flow on top of either Kraken trading mode.

```powershell
.\.venv\Scripts\Activate.ps1
$env:IDENTITY_LAYER = "erc8004"
$env:SEPOLIA_RPC_URL = "https://ethereum-sepolia-rpc.publicnode.com"
$env:SEPOLIA_CHAIN_ID = "11155111"
# Set the shared contract addresses from `.env.example`
python src/main.py --base-dir . --trading-mode paper --identity-layer erc8004
```

---

## 6) Optional ERC-8004 on-chain actions

Use this when you want to register the agent, claim the allocation, submit intents, or post checkpoints.

```powershell
.\.venv\Scripts\Activate.ps1
$env:PRIVATE_KEY = "0x..."
$env:AGENT_WALLET_PRIVATE_KEY = "0x..."
$env:AGENT_DISPLAY_NAME = "AI Trading Agent Demo"
$env:AGENT_STRATEGY_NAME = "simple_event_driven"
```

Available commands:

```powershell
python src/main.py --base-dir . --trading-mode paper --identity-layer erc8004 --register-agent
python src/main.py --base-dir . --trading-mode paper --identity-layer erc8004 --claim-allocation
python src/main.py --base-dir . --trading-mode paper --identity-layer erc8004 --submit-onchain
python src/main.py --base-dir . --trading-mode paper --identity-layer erc8004 --post-checkpoints
python src/main.py --base-dir . --trading-mode paper --identity-layer erc8004 --full-flow
```

If registration succeeds, the numeric `AGENT_ID` is persisted to `.runtime.env` for reuse.

---

## Quick troubleshooting

- Check all CLI options with `python src/main.py --help`.
- If the ERC-8004 layer reports missing config, export the required variables first.
- `--serve` cannot be combined with the on-chain action flags.
