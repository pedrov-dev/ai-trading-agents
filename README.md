# AI Trading Agents

Local-first event-driven crypto trading MVP for **Kraken paper trading** or **Kraken live trading**, with an optional ERC-8004 identity layer.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
kraken-cli --help
python -m pytest
set KRAKEN_API_KEY=...
set KRAKEN_API_SECRET=...
python src/main.py --base-dir . --preflight
python src/main.py --base-dir .
```

For step-by-step run instructions for every supported mode, see `USAGE.md`.

## Basic dashboard UI

A super basic local dashboard is available for demos and judging:

```bash
streamlit run src/ui/dashboard.py
```

It shows portfolio/performance metrics from `artifacts/run_summary.json` and lets you trigger preflight checks, one-shot runs, scheduler start/stop, and ERC-8004 actions from the browser.

## What the Kraken runtime does

- ingests RSS articles and Kraken price quotes
- classifies crypto-relevant events
- generates explainable trade intents
- runs conservative risk-aware Kraken paper validation or live submission
- writes audit logs, validation artifacts, checkpoints, and a run summary

## Output files

After a run, inspect:

- `artifacts/orders_audit.jsonl`
- `artifacts/validation_artifacts.jsonl`
- `artifacts/validation_checkpoints.jsonl`
- `artifacts/run_summary.json`

## Optional ERC-8004 / Sepolia identity layer

The repo also supports the official shared ERC-8004 contracts on **Sepolia** when you want identity, reputation, and validation records on-chain.

1. Copy values from `.env.example`
2. Set `IDENTITY_LAYER=erc8004`
3. Provide `SEPOLIA_RPC_URL`, `PRIVATE_KEY`, and `AGENT_WALLET_PRIVATE_KEY`
4. Reuse the official shared addresses already included in `.env.example`
5. Run `python src/main.py --base-dir . --trading-mode paper --identity-layer erc8004`

Optional on-chain actions are explicit:

```bash
python src/main.py --base-dir . --trading-mode paper --identity-layer erc8004 --register-agent
python src/main.py --base-dir . --trading-mode paper --identity-layer erc8004 --claim-allocation
python src/main.py --base-dir . --trading-mode paper --identity-layer erc8004 --submit-onchain
python src/main.py --base-dir . --trading-mode paper --identity-layer erc8004 --post-checkpoints
python src/main.py --base-dir . --trading-mode paper --identity-layer erc8004 --full-flow
```

Set the real registration profile in your env before using the shared contracts:
`AGENT_DISPLAY_NAME`, `AGENT_STRATEGY_NAME`, `AGENT_OWNER`, `AGENT_DESCRIPTION`, `AGENT_URI`, and `AGENT_CAPABILITIES`.

When a numeric on-chain `agentId` is available, the app also persists it to `.runtime.env` so later Sepolia runs can reuse it more easily.
The run summary now reports shared-contract status, including missing env values, derived wallet addresses, claim state, vault balance, and any on-chain action results.

> Judging reads only from the shared contracts, so do **not** deploy private copies for leaderboard runs.

## Notes

- Default mode is **Kraken paper trading**.
- Paper mode requires real `KRAKEN_API_KEY` and `KRAKEN_API_SECRET` so Kraken can validate the order on the exchange endpoint.
- Live trading is available with `python src/main.py --base-dir . --trading-mode live` plus `KRAKEN_CLI_ALLOW_LIVE_SUBMIT=true` and exchange credentials.
- Optional on-chain identity is enabled with `--identity-layer erc8004`.
- `python src/main.py --base-dir . --preflight` checks Kraken and optional ERC-8004 readiness without running a trading cycle.
- The app auto-loads `.env` / `.runtime.env` from the selected `--base-dir`.
- For optional external-backed mode, copy values from `.env.example`.
