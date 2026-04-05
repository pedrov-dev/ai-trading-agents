"""Simple explainable strategy that turns events into trade intents."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

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
from detection.event_types import event_performance_group
from ingestion.prices_ingestion import PriceQuote


@dataclass(frozen=True)
class StrategyConfig:
    """Thresholds and ranking weights for the event-driven strategy."""

    min_signal_score: float = 0.7
    max_intents_per_cycle: int = 2
    max_ranked_signals_per_cycle: int | None = None
    confidence_weight: float = 0.35
    novelty_weight: float = 0.2
    risk_reward_weight: float = 0.25
    diversification_weight: float = 0.2
    thesis_cooldown_enabled: bool = True
    thesis_cooldown_hours: int = 6
    thesis_repeat_penalty: float = 0.12
    thesis_similarity_threshold: float = 0.75


@dataclass(frozen=True)
class _ThesisCooldownState:
    """Recent history for one thesis key used to suppress repeated replays."""

    last_seen_at: datetime
    last_raw_event_id: str
    thesis_fingerprint: str | None = None
    thesis_tokens: tuple[str, ...] = ()
    event_key: str | None = None
    repeat_count: int = 0


@dataclass(frozen=True)
class RankedSignal:
    """Signal plus the multi-factor ranking breakdown used for top-N selection."""

    signal: Signal
    composite_score: float
    confidence_score: float
    novelty_score: float
    risk_reward_score: float
    diversification_score: float
    proposed_notional: float
    risk_result: RiskCheckResult

    def ranking_rationale(self) -> tuple[str, ...]:
        return (
            "Ranking breakdown: "
            f"confidence={self.confidence_score:.2f}, "
            f"novelty={self.novelty_score:.2f}, "
            f"risk/reward={self.risk_reward_score:.2f}, "
            f"diversification={self.diversification_score:.2f}, "
            f"composite={self.composite_score:.2f}.",
        )


@dataclass(frozen=True)
class ExitHorizon:
    """One explicit time window used to evaluate and close a signal tranche."""

    label: str
    hold_minutes: int
    weight: float = 0.25


_DEFAULT_EXIT_HORIZONS: tuple[ExitHorizon, ...] = (
    ExitHorizon(label="5m", hold_minutes=5, weight=0.25),
    ExitHorizon(label="30m", hold_minutes=30, weight=0.25),
    ExitHorizon(label="4h", hold_minutes=240, weight=0.25),
    ExitHorizon(label="24h", hold_minutes=1440, weight=0.25),
)


@dataclass(frozen=True)
class ExitConfig:
    """Basic close-out rules for autonomous paper-trading exits."""

    profit_target_fraction: float = 0.03
    stop_loss_fraction: float = 0.02
    max_hold_minutes: int = 180
    close_on_opposite_event: bool = True
    target_horizons: tuple[ExitHorizon, ...] = _DEFAULT_EXIT_HORIZONS


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
        self._risk_config = risk_config or RiskConfig()
        self._risk_manager = RiskManager(self._risk_config)
        self._exit_config = exit_config or ExitConfig()
        self._thesis_cooldowns: dict[tuple[str, str], list[_ThesisCooldownState]] = {}

    def generate_trade_intents(
        self,
        *,
        detected_events: list[DetectedEvent],
        price_quotes: list[PriceQuote],
        portfolio: PortfolioSnapshot,
    ) -> list[TradeIntent]:
        ranked_signals: dict[str, RankedSignal] = {}

        for event in detected_events:
            quote = select_quote_for_event(event=event, price_quotes=price_quotes)
            if quote is None:
                continue

            try:
                base_signal = build_signal(event=event, quote=quote)
            except ValueError:
                continue

            recent_theses = self._recent_thesis_states(signal=base_signal)
            signal = self._apply_thesis_cooldown(signal=base_signal)
            if signal.score < self._config.min_signal_score:
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

            ranked_signal = self._rank_signal(
                signal=signal,
                quote=quote,
                portfolio=portfolio,
                proposed_notional=proposed_notional,
                risk_result=risk_result,
                recent_theses=recent_theses,
            )
            current_best = ranked_signals.get(signal.symbol_id)
            if current_best is None or (
                ranked_signal.composite_score,
                ranked_signal.signal.score,
            ) > (
                current_best.composite_score,
                current_best.signal.score,
            ):
                ranked_signals[signal.symbol_id] = ranked_signal

        intents: list[TradeIntent] = []
        planned_intents_by_symbol: dict[str, int] = {}
        selected_new_symbols: set[str] = set()
        selected_signal_count = 0
        max_ranked_signals = self._max_ranked_signals_per_cycle()

        for ranked_signal in sorted(
            ranked_signals.values(),
            key=lambda item: (item.composite_score, item.signal.score),
            reverse=True,
        ):
            signal = ranked_signal.signal
            if not self._can_allocate_symbol(
                symbol_id=signal.symbol_id,
                portfolio=portfolio,
                selected_new_symbols=selected_new_symbols,
            ):
                continue

            available_intent_slots = self._available_intent_slots_for_symbol(
                symbol_id=signal.symbol_id,
                portfolio=portfolio,
                planned_intents_by_symbol=planned_intents_by_symbol,
            )
            if available_intent_slots == 0:
                continue

            rationale_suffix = ranked_signal.risk_result.notes + (
                "Signal selected in the top "
                f"{max_ranked_signals} ranked opportunities for this cycle.",
                f"Event {signal.event_type} cleared all configured risk checks.",
            )
            if available_intent_slots is not None and available_intent_slots < len(
                self._exit_config.target_horizons
            ):
                rationale_suffix += (
                    "Opportunity budget capped "
                    f"{signal.symbol_id} to {available_intent_slots} active position "
                    "slot(s) this cycle.",
                )

            symbol_intents = _build_horizon_trade_intents(
                signal=signal,
                allowed_notional=ranked_signal.risk_result.allowed_notional,
                rationale_suffix=rationale_suffix,
                exit_config=self._exit_config,
                max_intents=available_intent_slots,
            )
            if not symbol_intents:
                continue

            intents.extend(symbol_intents)
            planned_intents_by_symbol[signal.symbol_id] = (
                planned_intents_by_symbol.get(signal.symbol_id, 0) + len(symbol_intents)
            )
            if not portfolio.has_open_position(signal.symbol_id):
                selected_new_symbols.add(signal.symbol_id)
            selected_signal_count += 1

            if selected_signal_count >= max_ranked_signals:
                break

        return intents

    def _max_ranked_signals_per_cycle(self) -> int:
        configured_limit = self._config.max_ranked_signals_per_cycle
        if configured_limit is None:
            configured_limit = self._config.max_intents_per_cycle
        return max(1, configured_limit)

    def _can_allocate_symbol(
        self,
        *,
        symbol_id: str,
        portfolio: PortfolioSnapshot,
        selected_new_symbols: set[str],
    ) -> bool:
        if portfolio.has_open_position(symbol_id):
            return True
        if self._risk_config.max_concurrent_positions <= 0:
            return False
        return (
            portfolio.open_symbol_count() + len(selected_new_symbols)
            < self._risk_config.max_concurrent_positions
        )

    def _available_intent_slots_for_symbol(
        self,
        *,
        symbol_id: str,
        portfolio: PortfolioSnapshot,
        planned_intents_by_symbol: dict[str, int],
    ) -> int | None:
        max_positions_per_asset = self._risk_config.max_positions_per_asset
        if max_positions_per_asset is None:
            return None
        if max_positions_per_asset <= 0:
            return 0
        current_positions = len(portfolio.positions_for_symbol(symbol_id))
        planned_positions = planned_intents_by_symbol.get(symbol_id, 0)
        return max(0, max_positions_per_asset - current_positions - planned_positions)

    def _recent_thesis_states(self, *, signal: Signal) -> tuple[_ThesisCooldownState, ...]:
        if not self._config.thesis_cooldown_enabled or self._config.thesis_cooldown_hours <= 0:
            return ()

        history_key = (signal.symbol_id, signal.side)
        cooldown_window = timedelta(hours=self._config.thesis_cooldown_hours)
        return tuple(
            state
            for state in self._thesis_cooldowns.get(history_key, [])
            if signal.generated_at < state.last_seen_at + cooldown_window
        )

    def _best_thesis_match(
        self,
        *,
        signal: Signal,
        recent_theses: tuple[_ThesisCooldownState, ...],
    ) -> tuple[float, _ThesisCooldownState | None]:
        best_similarity = 0.0
        best_state: _ThesisCooldownState | None = None

        for state in recent_theses:
            similarity = _thesis_similarity(signal=signal, state=state)
            if similarity > best_similarity:
                best_similarity = similarity
                best_state = state

        return best_similarity, best_state

    def _rank_signal(
        self,
        *,
        signal: Signal,
        quote: PriceQuote,
        portfolio: PortfolioSnapshot,
        proposed_notional: float,
        risk_result: RiskCheckResult,
        recent_theses: tuple[_ThesisCooldownState, ...],
    ) -> RankedSignal:
        confidence_score = _clamp_score((signal.score * 0.7) + (signal.confidence * 0.3))
        best_similarity, best_state = self._best_thesis_match(
            signal=signal,
            recent_theses=recent_theses,
        )
        repeat_count = 0 if best_state is None else best_state.repeat_count + 1
        novelty_score = _clamp_score(
            1.0 - min(1.0, best_similarity * (0.85 + (0.05 * repeat_count)))
        )
        risk_reward_score = self._risk_reward_score(
            signal=signal,
            quote=quote,
            proposed_notional=proposed_notional,
            risk_result=risk_result,
        )
        diversification_score = self._diversification_score(
            signal=signal,
            portfolio=portfolio,
        )

        weighted_components = (
            (self._config.confidence_weight, confidence_score),
            (self._config.novelty_weight, novelty_score),
            (self._config.risk_reward_weight, risk_reward_score),
            (self._config.diversification_weight, diversification_score),
        )
        total_weight = sum(max(weight, 0.0) for weight, _ in weighted_components) or 1.0
        composite_score = round(
            sum(max(weight, 0.0) * score for weight, score in weighted_components)
            / total_weight,
            4,
        )
        ranked_signal = RankedSignal(
            signal=replace(signal),
            composite_score=composite_score,
            confidence_score=confidence_score,
            novelty_score=novelty_score,
            risk_reward_score=risk_reward_score,
            diversification_score=diversification_score,
            proposed_notional=proposed_notional,
            risk_result=risk_result,
        )
        ranked_signal = replace(
            ranked_signal,
            signal=replace(
                signal,
                rationale=signal.rationale + ranked_signal.ranking_rationale(),
            ),
        )
        return ranked_signal

    def _risk_reward_score(
        self,
        *,
        signal: Signal,
        quote: PriceQuote,
        proposed_notional: float,
        risk_result: RiskCheckResult,
    ) -> float:
        aligned_move = 0.0
        if quote.open > 0:
            session_move = (quote.current - quote.open) / quote.open
            aligned_move = (
                max(session_move, 0.0)
                if signal.side == "buy"
                else max(-session_move, 0.0)
            )
        aligned_score = _clamp_score(aligned_move / 0.03)
        configured_rr = self._exit_config.profit_target_fraction / max(
            self._exit_config.stop_loss_fraction,
            0.0001,
        )
        reward_risk_component = _clamp_score(configured_rr / 2.0)
        sizing_efficiency = _clamp_score(
            risk_result.allowed_notional / proposed_notional if proposed_notional > 0 else 0.0
        )
        return _clamp_score(
            (signal.score * 0.45)
            + (aligned_score * 0.3)
            + (reward_risk_component * 0.15)
            + (sizing_efficiency * 0.1)
        )

    def _diversification_score(
        self,
        *,
        signal: Signal,
        portfolio: PortfolioSnapshot,
    ) -> float:
        if not portfolio.positions:
            return 1.0

        total_open_notional = max(portfolio.total_open_notional(), 0.0)
        target_side = "long" if signal.side == "buy" else "short"
        same_side_notional = sum(
            position.notional_usd
            for position in portfolio.positions
            if position.side == target_side
        )
        side_diversity = 1.0
        if total_open_notional > 0:
            side_diversity = 1.0 - (same_side_notional / total_open_notional)

        same_theme_count = sum(
            1
            for position in portfolio.positions
            if position.event_type is not None
            and event_performance_group(position.event_type) == signal.event_group
        )
        theme_diversity = 1.0 - (same_theme_count / max(len(portfolio.positions), 1))
        return _clamp_score((0.2 * 1.0) + (0.5 * side_diversity) + (0.3 * theme_diversity))

    def _apply_thesis_cooldown(self, *, signal: Signal) -> Signal:
        """Decay repeated or highly similar theses so the strategy seeks fresh setups."""
        if (
            not self._config.thesis_cooldown_enabled
            or self._config.thesis_cooldown_hours <= 0
            or self._config.thesis_repeat_penalty <= 0
        ):
            return signal

        history_key = (signal.symbol_id, signal.side)
        similarity_threshold = min(max(self._config.thesis_similarity_threshold, 0.0), 1.0)
        recent_states = list(self._recent_thesis_states(signal=signal))

        repeat_count = 0
        adjusted_signal = signal
        best_similarity, best_state = self._best_thesis_match(
            signal=signal,
            recent_theses=tuple(recent_states),
        )

        if best_state is not None and best_similarity >= similarity_threshold:
            repeat_count = best_state.repeat_count + 1
            exact_replay = signal.raw_event_id == best_state.last_raw_event_id
            penalty_scale = 1.0 if exact_replay else 0.2
            penalty = min(
                self._config.thesis_repeat_penalty
                * repeat_count
                * best_similarity
                * penalty_scale,
                max(signal.score - 0.01, 0.0),
            )
            adjusted_score = round(max(signal.score - penalty, 0.0), 4)
            adjusted_confidence = round(
                max(0.0, min(signal.confidence, adjusted_score)),
                4,
            )
            similarity_label = "repeated thesis" if exact_replay else "similar thesis"
            adjusted_signal = replace(
                signal,
                confidence=adjusted_confidence,
                score=adjusted_score,
                rationale=signal.rationale
                + (
                    f"Thesis cooldown active for a {similarity_label} on "
                    f"{signal.symbol_id}; similarity={best_similarity:.2f} "
                    f"inside the {self._config.thesis_cooldown_hours}h "
                    f"window reduced confidence by {penalty:.2f}.",
                ),
            )

        recent_states.append(
            _ThesisCooldownState(
                last_seen_at=signal.generated_at,
                last_raw_event_id=signal.raw_event_id,
                thesis_fingerprint=signal.thesis_fingerprint,
                thesis_tokens=signal.thesis_tokens,
                event_key=signal.event_group or signal.event_type,
                repeat_count=repeat_count,
            )
        )
        self._thesis_cooldowns[history_key] = recent_states[-12:]
        return adjusted_signal

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
            if position.exit_due_at is not None and checked_at >= position.exit_due_at:
                rationale.append(
                    "Exit horizon "
                    f"{position.exit_horizon_label or 'scheduled'} reached after "
                    f"{hold_minutes:.1f} minutes."
                )
            elif (
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
                    confidence_score=position.confidence_score,
                    expected_move=position.expected_move,
                    generated_at=checked_at,
                    signal_id=position.source_signal_id,
                    raw_event_id=position.raw_event_id,
                    event_type=position.event_type,
                    exit_horizon_label=position.exit_horizon_label,
                    max_hold_minutes=position.max_hold_minutes,
                    exit_due_at=position.exit_due_at,
                    position_id=position.position_id,
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
            signal_id=intent.signal_id or f"runtime-recheck:{intent.symbol_id}",
            raw_event_id=intent.raw_event_id or f"runtime-recheck:{intent.symbol_id}",
            event_type=intent.event_type or "RUNTIME_RECHECK",
            symbol_id=intent.symbol_id,
            side=intent.side,
            confidence=intent.score,
            score=intent.score,
            current_price=intent.current_price,
            generated_at=intent.generated_at,
            rationale=intent.rationale,
        )
        existing_position = portfolio.position_for_symbol(
            intent.symbol_id,
            position_id=intent.position_id,
        )
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


def _thesis_similarity(*, signal: Signal, state: _ThesisCooldownState) -> float:
    if signal.raw_event_id == state.last_raw_event_id:
        return 1.0
    if signal.thesis_fingerprint and signal.thesis_fingerprint == state.thesis_fingerprint:
        return 1.0

    signal_event_key = signal.event_group or signal.event_type
    same_event_key = bool(
        signal_event_key and state.event_key and signal_event_key == state.event_key
    )
    token_similarity = _jaccard_similarity(signal.thesis_tokens, state.thesis_tokens)

    if same_event_key and token_similarity > 0.0:
        return round(min(1.0, 0.6 + (0.4 * token_similarity)), 4)
    if same_event_key:
        return 0.0
    return round(token_similarity * 0.35, 4)


def _jaccard_similarity(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _clamp_score(value: float) -> float:
    return round(min(max(value, 0.0), 1.0), 4)


def _build_horizon_trade_intents(
    *,
    signal: Signal,
    allowed_notional: float,
    rationale_suffix: tuple[str, ...],
    exit_config: ExitConfig,
    max_intents: int | None = None,
) -> list[TradeIntent]:
    if max_intents is not None and max_intents <= 0:
        return []

    horizons = tuple(horizon for horizon in exit_config.target_horizons if horizon.weight > 0)
    if max_intents is not None:
        horizons = horizons[:max_intents]
    if not horizons:
        return [
            build_trade_intent(
                signal=signal,
                notional_usd=allowed_notional,
                rationale_suffix=rationale_suffix,
                max_hold_minutes=exit_config.max_hold_minutes,
                position_id=f"{signal.signal_id}:core",
            )
        ]

    intents: list[TradeIntent] = []
    remaining_notional = round(allowed_notional, 2)
    total_weight = sum(horizon.weight for horizon in horizons)

    for index, horizon in enumerate(horizons):
        if index == len(horizons) - 1:
            tranche_notional = remaining_notional
        else:
            tranche_notional = round(
                allowed_notional * (horizon.weight / total_weight),
                2,
            )
            remaining_notional = round(remaining_notional - tranche_notional, 2)

        if tranche_notional <= 0:
            continue

        intents.append(
            build_trade_intent(
                signal=signal,
                notional_usd=tranche_notional,
                rationale_suffix=rationale_suffix
                + (
                    f"Exit tranche assigned to the {horizon.label} evaluation window.",
                ),
                exit_horizon_label=horizon.label,
                max_hold_minutes=horizon.hold_minutes,
                position_id=f"{signal.signal_id}:{horizon.label}",
            )
        )

    return intents


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
