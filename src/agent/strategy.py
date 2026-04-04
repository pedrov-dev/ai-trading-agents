"""Simple explainable strategy that turns events into trade intents."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from agent.portfolio import PortfolioSnapshot, Position
from agent.risk import RiskCheckResult, RiskConfig, RiskManager
from agent.signals import (
    Signal,
    TradeIntent,
    build_signal,
    build_trade_intent,
    infer_trade_side,
    select_quote_for_event,
)
from detection.event_detection import DetectedEvent
from ingestion.prices_ingestion import PriceQuote


@dataclass(frozen=True)
class StrategyConfig:
    """Thresholds for the MVP event-driven strategy."""

    min_signal_score: float = 0.7
    max_intents_per_cycle: int = 2


@dataclass(frozen=True)
class ExitConfig:
    """Basic close-out rules for autonomous paper-trading exits."""

    profit_target_fraction: float = 0.03
    stop_loss_fraction: float = 0.02
    max_hold_minutes: int = 180
    close_on_opposite_event: bool = True


class SimpleEventDrivenStrategy:
    """Deterministic strategy that favors strong, explainable event setups."""

    def __init__(
        self,
        *,
        config: StrategyConfig | None = None,
        risk_config: RiskConfig | None = None,
        exit_config: ExitConfig | None = None,
    ) -> None:
        self._config = config or StrategyConfig()
        self._risk_manager = RiskManager(risk_config or RiskConfig())
        self._exit_config = exit_config or ExitConfig()

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
            if portfolio.has_open_position(signal.symbol_id):
                continue

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

    def evaluate_position_exits(
        self,
        *,
        portfolio: PortfolioSnapshot,
        price_quotes: list[PriceQuote],
        detected_events: list[DetectedEvent],
        now: datetime | None = None,
    ) -> list[TradeIntent]:
        """Generate close-out intents for open positions that hit exit conditions."""
        checked_at = now or datetime.now(UTC)
        quote_by_symbol = {quote.symbol_id: quote for quote in price_quotes}
        opposite_event_symbols = (
            _symbols_with_opposite_events(
                portfolio=portfolio,
                detected_events=detected_events,
                price_quotes=price_quotes,
            )
            if self._exit_config.close_on_opposite_event
            else set()
        )
        intents: list[TradeIntent] = []

        for position in portfolio.positions:
            quote = quote_by_symbol.get(position.symbol_id)
            if quote is None or quote.current <= 0:
                continue

            return_fraction = _position_return_fraction(
                position=position,
                current_price=quote.current,
            )
            hold_minutes = max(
                (checked_at - position.opened_at).total_seconds() / 60.0,
                0.0,
            )
            rationale: list[str] = []

            if return_fraction >= self._exit_config.profit_target_fraction:
                rationale.append(
                    f"Profit target hit at {return_fraction * 100:.2f}% unrealized return."
                )
            if return_fraction <= -self._exit_config.stop_loss_fraction:
                rationale.append(
                    f"Stop loss hit at {return_fraction * 100:.2f}% unrealized return."
                )
            if (
                self._exit_config.max_hold_minutes > 0
                and hold_minutes >= self._exit_config.max_hold_minutes
            ):
                rationale.append(
                    f"Max hold time exceeded at {hold_minutes:.1f} minutes."
                )
            if position.symbol_id in opposite_event_symbols:
                rationale.append(
                    "Opposite event signal detected for an already-open position."
                )

            if not rationale:
                continue

            rationale.append(
                "Exit sized to close "
                f"{position.quantity:.8f} {position.symbol_id} at "
                f"${quote.current:,.2f}."
            )
            intents.append(
                TradeIntent(
                    symbol_id=position.symbol_id,
                    side="sell" if position.side == "long" else "buy",
                    notional_usd=round(abs(position.quantity) * quote.current, 2),
                    quantity=round(position.quantity, 8),
                    current_price=quote.current,
                    score=1.0,
                    rationale=tuple(rationale),
                    generated_at=checked_at,
                )
            )

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
        existing_position = portfolio.position_for_symbol(intent.symbol_id)
        reduce_only = existing_position is not None and (
            (existing_position.side == "long" and intent.side == "sell")
            or (existing_position.side == "short" and intent.side == "buy")
        )
        return self._risk_manager.evaluate(
            signal=signal,
            portfolio=portfolio,
            proposed_notional=intent.notional_usd,
            now=now,
            reduce_only=reduce_only,
        )


def _position_return_fraction(*, position: Position, current_price: float) -> float:
    if position.entry_price <= 0:
        return 0.0
    if position.side == "short":
        return (position.entry_price - current_price) / position.entry_price
    return (current_price - position.entry_price) / position.entry_price


def _symbols_with_opposite_events(
    *,
    portfolio: PortfolioSnapshot,
    detected_events: list[DetectedEvent],
    price_quotes: list[PriceQuote],
) -> set[str]:
    symbols: set[str] = set()
    for event in detected_events:
        event_side = infer_trade_side(event.event_type)
        if event_side is None:
            continue
        quote = select_quote_for_event(event=event, price_quotes=price_quotes)
        if quote is None:
            continue
        position = portfolio.position_for_symbol(quote.symbol_id)
        if position is None:
            continue
        if (position.side == "long" and event_side == "sell") or (
            position.side == "short" and event_side == "buy"
        ):
            symbols.add(quote.symbol_id)
    return symbols
