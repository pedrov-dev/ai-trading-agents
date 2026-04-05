# Agent

Event-driven decision layer converting detected events + price quotes into sized, risk-checked trade intents.

## Key modules

- `__init__.py`: package exports and public API surface.
- `portfolio.py`: `Position`, `PortfolioSnapshot` models and `LocalPortfolioStateProvider`.
- `risk.py`: conservative sizing and guardrails (`RiskManager`, `RiskConfig`, `RiskCheckResult`).
- `signals.py`: event→signal scoring and builders (`Signal`, `TradeIntent`, `build_signal`, `build_trade_intent`).
- `strategy.py`: orchestration (`SimpleEventDrivenStrategy`, `StrategyConfig`) that ranks signals and produces intents.

## Public API

- `Position`, `PortfolioSnapshot` — portfolio models.
- `LocalPortfolioStateProvider.get_portfolio_snapshot()`, `record_fill(...)`, `set_realized_pnl(...)` — demo state management.
- `RiskManager.size_for_signal(signal, portfolio) -> float`, `evaluate(...)->RiskCheckResult` — sizing & checks.
- `build_signal(event, quote) -> Signal`, `build_trade_intent(signal, notional_usd, rationale_suffix=()) -> TradeIntent`.
- `SimpleEventDrivenStrategy.generate_trade_intents(detected_events, price_quotes, portfolio) -> list[TradeIntent]`, `reassess_trade_intent(intent, portfolio, now=None) -> RiskCheckResult`.

## Integration

- Consumes `DetectedEvent` (detection) and `PriceQuote` (ingestion); reads `PortfolioSnapshot` from a provider; emits `TradeIntent` for the execution layer.

## Usage

```python
from agent import SimpleEventDrivenStrategy, LocalPortfolioStateProvider
strategy = SimpleEventDrivenStrategy()
intents = strategy.generate_trade_intents(detected_events, price_quotes, LocalPortfolioStateProvider().get_portfolio_snapshot())
```

## Notes

- `StrategyConfig` now includes both **signal freshness decay** (`signal_time_decay_enabled`, `signal_decay_half_life_minutes`, `signal_decay_floor`) and a **thesis-level cooldown** (`thesis_cooldown_enabled`, `thesis_cooldown_hours`, `thesis_repeat_penalty`) so stale setups fade over time and repeated `symbol + thesis + side` replays are eventually blocked until a fresh event arrives.
- Tune `StrategyConfig`/`RiskConfig` for deployment; `LocalPortfolioStateProvider` is in-memory and non-persistent; `build_signal` raises for unsupported event types.
