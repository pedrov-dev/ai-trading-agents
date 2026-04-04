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

## Shared Sepolia judging mode

The repo now includes the first implementation pieces for the official ERC-8004 shared contracts on **Sepolia**.

1. Copy values from `.env.example`
2. Set `TRADING_RUNTIME_MODE=sepolia`
3. Provide `SEPOLIA_RPC_URL`, `PRIVATE_KEY`, and `AGENT_WALLET_PRIVATE_KEY`
4. Reuse the official shared addresses already included in `.env.example`
5. Run `python src/main.py --runtime-mode sepolia --base-dir .`

Optional on-chain actions are explicit:

```bash
python src/main.py --runtime-mode sepolia --base-dir . --register-agent
python src/main.py --runtime-mode sepolia --base-dir . --claim-allocation
python src/main.py --runtime-mode sepolia --base-dir . --submit-onchain
python src/main.py --runtime-mode sepolia --base-dir . --post-checkpoints
python src/main.py --runtime-mode sepolia --base-dir . --full-flow
```

Set the real registration profile in your env before using the shared contracts:
`AGENT_DISPLAY_NAME`, `AGENT_STRATEGY_NAME`, `AGENT_OWNER`, `AGENT_DESCRIPTION`, `AGENT_URI`, and `AGENT_CAPABILITIES`.

When a numeric on-chain `agentId` is available, the app also persists it to `.runtime.env` so later Sepolia runs can reuse it more easily.
The run summary now reports shared-contract status, including missing env values, derived wallet addresses, claim state, vault balance, and any on-chain action results.

> Judging reads only from the shared contracts, so do **not** deploy private copies for leaderboard runs.

## Notes

- Default mode is **safe dry run**.
- No live trading is enabled by default.
- `local` mode remains the default for tests and offline development.
- For optional external-backed mode, copy values from `.env.example`.
