"""Simple explainable strategy that turns events into trade intents."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from agent.portfolio import PortfolioSnapshot
from agent.risk import RiskCheckResult, RiskConfig, RiskManager
from agent.signals import (
    Signal,
    TradeIntent,
    build_signal,
    build_trade_intent,
    select_quote_for_event,
)
from detection.event_detection import DetectedEvent
from ingestion.prices_ingestion import PriceQuote


@dataclass(frozen=True)
class StrategyConfig:
    """Thresholds for the MVP event-driven strategy."""

    min_signal_score: float = 0.7
    max_intents_per_cycle: int = 2


class SimpleEventDrivenStrategy:
    """Deterministic strategy that favors strong, explainable event setups."""

    def __init__(
        self,
        *,
        config: StrategyConfig | None = None,
        risk_config: RiskConfig | None = None,
    ) -> None:
        self._config = config or StrategyConfig()
        self._risk_manager = RiskManager(risk_config or RiskConfig())

    def generate_trade_intents(
        self,
        *,
        detected_events: list[DetectedEvent],
        price_quotes: list[PriceQuote],
        portfolio: PortfolioSnapshot,
    ) -> list[TradeIntent]:
        ranked_signals: dict[str, Signal] = {}

        for event in detected_events:
            quote = select_quote_for_event(event=event, price_quotes=price_quotes)
            if quote is None:
                continue

            try:
                signal = build_signal(event=event, quote=quote)
            except ValueError:
                continue

            if signal.score < self._config.min_signal_score:
                continue

            current_best = ranked_signals.get(signal.symbol_id)
            if current_best is None or signal.score > current_best.score:
                ranked_signals[signal.symbol_id] = signal

        intents: list[TradeIntent] = []

        for signal in sorted(
            ranked_signals.values(),
            key=lambda item: item.score,
            reverse=True,
        ):
            proposed_notional = self._risk_manager.size_for_signal(
                signal=signal,
                portfolio=portfolio,
            )
            risk_result = self._risk_manager.evaluate(
                signal=signal,
                portfolio=portfolio,
                proposed_notional=proposed_notional,
            )
            if not risk_result.approved:
                continue

            intents.append(
                build_trade_intent(
                    signal=signal,
                    notional_usd=risk_result.allowed_notional,
                    rationale_suffix=risk_result.notes
                    + (f"Event {signal.event_type} cleared all configured risk checks.",),
                )
            )

            if len(intents) >= self._config.max_intents_per_cycle:
                break

        return intents

    def reassess_trade_intent(
        self,
        *,
        intent: TradeIntent,
        portfolio: PortfolioSnapshot,
        now: datetime | None = None,
    ) -> RiskCheckResult:
        """Re-run the same conservative risk guardrails immediately before execution."""
        signal = Signal(
            raw_event_id=f"runtime-recheck:{intent.symbol_id}",
            event_type="RUNTIME_RECHECK",
            symbol_id=intent.symbol_id,
            side=intent.side,
            confidence=intent.score,
            score=intent.score,
            current_price=intent.current_price,
            generated_at=intent.generated_at,
            rationale=intent.rationale,
        )
        return self._risk_manager.evaluate(
            signal=signal,
            portfolio=portfolio,
            proposed_notional=intent.notional_usd,
            now=now,
        )
