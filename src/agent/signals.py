"""Explainable signal scoring and trade-intent models."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from agent.event_signal import (
    apply_event_score_adjustments as _apply_event_score_adjustments,
)
from agent.event_signal import (
    clamp_signal_score as _clamp_signal_score,
)
from agent.event_signal import (
    event_novelty_adjustment as _event_novelty_adjustment,
)
from agent.event_signal import (
    resolve_signal_age_minutes as _resolve_signal_age_minutes,
)
from agent.event_signal import (
    signal_time_decay_factor as _signal_time_decay_factor,
)
from agent.event_signal import (
    volatility_adjustment as _volatility_adjustment,
)
from agent.momentum_signal import (
    DEFAULT_PRICE_CONFIRMATION_THRESHOLD as _DEFAULT_PRICE_CONFIRMATION_THRESHOLD,
)
from agent.momentum_signal import MOMENTUM_SIGNAL_VERSION as _MOMENTUM_SIGNAL_VERSION
from agent.momentum_signal import (
    price_confirmation_state as _price_confirmation_state,
)
from agent.momentum_signal import (
    price_momentum as _price_momentum,
)
from agent.news_signal import (
    NEWS_SIGNAL_PRICE_ONLY_VERSION as _NEWS_SIGNAL_PRICE_ONLY_VERSION,
)
from agent.news_signal import NEWS_SIGNAL_VERSION as _NEWS_SIGNAL_VERSION
from agent.news_signal import (
    build_thesis_fingerprint as _build_thesis_fingerprint,
)
from agent.news_signal import (
    event_bias_for as _event_bias_for,
)
from agent.news_signal import (
    extract_thesis_tokens as _extract_thesis_tokens,
)
from agent.news_signal import (
    infer_trade_side,
    select_quote_for_event,
)
from agent.volume_breakout_signal import (
    DEFAULT_VOLUME_SPIKE_THRESHOLD as _DEFAULT_VOLUME_SPIKE_THRESHOLD,
)
from agent.volume_breakout_signal import (
    VOLUME_BREAKOUT_SIGNAL_VERSION as _VOLUME_BREAKOUT_SIGNAL_VERSION,
)
from agent.volume_breakout_signal import (
    volume_confirmation_state as _volume_confirmation_state,
)
from detection.event_detection import DetectedEvent
from detection.event_types import event_performance_group
from ingestion.prices_ingestion import PriceQuote

TradeSide = Literal["buy", "sell"]
MoveDirection = Literal["up", "down", "flat"]
SignalDirection = Literal["long", "short"]

_NEWS_SIGNAL_WEIGHT = 0.4
_PRICE_CONFIRMATION_WEIGHT = 0.35
_VOLUME_CONFIRMATION_WEIGHT = 0.25
_NEWS_PRICE_SYNERGY_BONUS = 0.05
_PRICE_VOLUME_SYNERGY_BONUS = 0.10
_ALL_CONFIRMATIONS_SYNERGY_BONUS = 0.15


@dataclass(frozen=True)
class RejectedTradeCandidate:
    """Counterfactual candidate that lost the rank/selection contest this cycle."""

    symbol_id: str
    side: TradeSide
    reference_price: float
    score: float
    confidence_score: float
    composite_score: float
    event_type: str | None = None
    event_group: str | None = None

    def __post_init__(self) -> None:
        if self.reference_price <= 0:
            raise ValueError("RejectedTradeCandidate reference_price must be positive.")
        object.__setattr__(self, "reference_price", round(self.reference_price, 8))
        object.__setattr__(self, "score", round(min(max(self.score, 0.0), 1.0), 4))
        object.__setattr__(
            self,
            "confidence_score",
            round(min(max(self.confidence_score, 0.0), 1.0), 4),
        )
        object.__setattr__(
            self,
            "composite_score",
            round(min(max(self.composite_score, 0.0), 1.0), 4),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol_id": self.symbol_id,
            "side": self.side,
            "reference_price": self.reference_price,
            "score": self.score,
            "confidence_score": self.confidence_score,
            "composite_score": self.composite_score,
            "event_type": self.event_type,
            "event_group": self.event_group,
        }


@dataclass(frozen=True)
class Signal:
    """Scored trading opportunity derived from a detected event and price quote."""

    raw_event_id: str
    event_type: str
    symbol_id: str
    side: TradeSide
    confidence: float
    score: float
    current_price: float
    generated_at: datetime
    rationale: tuple[str, ...]
    event_group: str | None = None
    signal_id: str | None = None
    signal_family: str | None = None
    signal_version: str | None = None
    model_version: str | None = None
    feature_set: str | None = None
    asset: str | None = None
    direction: SignalDirection | None = None
    thesis_fingerprint: str | None = None
    thesis_tokens: tuple[str, ...] = ()
    event_novelty_score: float = 1.0
    event_repeat_count: int = 0
    narrative_key: str | None = None
    news_confirmed: bool = True
    price_confirmed: bool = False
    volume_confirmed: bool = False
    volume_unavailable: bool = False
    confirmation_count: int = 1
    confirmation_score: float = 0.0


@dataclass(frozen=True)
class TradeIntent:
    """Exchange-agnostic trade request ready for later execution wiring."""

    symbol_id: str
    side: TradeSide
    notional_usd: float
    quantity: float
    current_price: float
    score: float
    rationale: tuple[str, ...]
    confidence_score: float | None = None
    expected_move: MoveDirection | None = None
    expected_move_fraction: float | None = None
    stop_distance_fraction: float | None = None
    risk_reward_ratio: float | None = None
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    signal_id: str | None = None
    signal_family: str | None = None
    signal_version: str | None = None
    model_version: str | None = None
    feature_set: str | None = None
    asset: str | None = None
    direction: SignalDirection | None = None
    confidence: float | None = None
    raw_event_id: str | None = None
    event_type: str | None = None
    event_group: str | None = None
    exit_horizon_label: str | None = None
    max_hold_minutes: int | None = None
    exit_due_at: datetime | None = None
    position_id: str | None = None
    selection_rank: int | None = None
    selection_composite_score: float | None = None
    rejected_alternatives: tuple[RejectedTradeCandidate, ...] = ()
    heuristic_version: str | None = None

    def __post_init__(self) -> None:
        resolved_confidence_score = (
            self.score if self.confidence_score is None else self.confidence_score
        )
        resolved_confidence = (
            self.confidence
            if self.confidence is not None
            else resolved_confidence_score
        )
        if not 0.0 <= resolved_confidence_score <= 1.0:
            raise ValueError("Trade intent confidence_score must be between 0.0 and 1.0.")
        if not 0.0 <= resolved_confidence <= 1.0:
            raise ValueError("Trade intent confidence must be between 0.0 and 1.0.")

        resolved_expected_move = self.expected_move or ("up" if self.side == "buy" else "down")
        if resolved_expected_move not in {"up", "down", "flat"}:
            raise ValueError("Trade intent expected_move must be one of: up, down, flat.")
        resolved_direction = self.direction or _resolve_direction(self.side)
        if resolved_direction not in {"long", "short"}:
            raise ValueError("Trade intent direction must be either 'long' or 'short'.")
        if self.expected_move_fraction is not None and self.expected_move_fraction < 0:
            raise ValueError("Trade intent expected_move_fraction must be non-negative.")
        if self.stop_distance_fraction is not None and self.stop_distance_fraction < 0:
            raise ValueError("Trade intent stop_distance_fraction must be non-negative.")
        if self.risk_reward_ratio is not None and self.risk_reward_ratio < 0:
            raise ValueError("Trade intent risk_reward_ratio must be non-negative.")
        if self.selection_rank is not None and self.selection_rank <= 0:
            raise ValueError("Trade intent selection_rank must be positive when set.")

        object.__setattr__(self, "confidence_score", round(resolved_confidence_score, 4))
        object.__setattr__(self, "confidence", round(resolved_confidence, 4))
        object.__setattr__(self, "expected_move", resolved_expected_move)
        object.__setattr__(self, "direction", resolved_direction)
        object.__setattr__(self, "asset", _infer_asset(self.asset or self.symbol_id))
        if self.expected_move_fraction is not None:
            object.__setattr__(
                self,
                "expected_move_fraction",
                round(self.expected_move_fraction, 4),
            )
        if self.stop_distance_fraction is not None:
            object.__setattr__(
                self,
                "stop_distance_fraction",
                round(self.stop_distance_fraction, 4),
            )
        if self.risk_reward_ratio is not None:
            object.__setattr__(self, "risk_reward_ratio", round(self.risk_reward_ratio, 4))
        if self.selection_composite_score is not None:
            object.__setattr__(
                self,
                "selection_composite_score",
                round(min(max(self.selection_composite_score, 0.0), 1.0), 4),
            )
        object.__setattr__(
            self,
            "rejected_alternatives",
            tuple(self.rejected_alternatives),
        )


@dataclass(frozen=True)
class NoTradeDecision:
    """Explicit strategy abstention recorded when a setup should be skipped."""

    symbol_id: str
    reason_code: str
    reason: str
    confidence_score: float
    threshold: float | None
    score: float
    event_type: str | None = None
    raw_event_id: str | None = None
    signal_id: str | None = None
    signal_family: str | None = None
    signal_version: str | None = None
    model_version: str | None = None
    feature_set: str | None = None
    asset: str | None = None
    direction: SignalDirection | None = None
    confidence: float | None = None
    event_group: str | None = None
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    rationale: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("No-trade confidence_score must be between 0.0 and 1.0.")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("No-trade score must be between 0.0 and 1.0.")
        if self.threshold is not None and not 0.0 <= self.threshold <= 1.0:
            raise ValueError("No-trade threshold must be between 0.0 and 1.0 when set.")
        if self.direction is not None and self.direction not in {"long", "short"}:
            raise ValueError("No-trade direction must be either 'long' or 'short' when set.")

        object.__setattr__(self, "confidence_score", round(self.confidence_score, 4))
        object.__setattr__(
            self,
            "confidence",
            round(self.confidence if self.confidence is not None else self.confidence_score, 4),
        )
        object.__setattr__(self, "score", round(self.score, 4))
        object.__setattr__(self, "asset", _infer_asset(self.asset or self.symbol_id))
        if self.threshold is not None:
            object.__setattr__(self, "threshold", round(self.threshold, 4))

def build_signal(
    *,
    event: DetectedEvent,
    quote: PriceQuote,
    price_confirmation_threshold: float = _DEFAULT_PRICE_CONFIRMATION_THRESHOLD,
    volume_spike_threshold: float = _DEFAULT_VOLUME_SPIKE_THRESHOLD,
    technical_breakout_volume_penalty: float = 0.15,
    evaluation_time: datetime | None = None,
    signal_time_decay_enabled: bool = True,
    signal_decay_half_life_minutes: float = 360.0,
    signal_decay_floor: float = 0.35,
) -> Signal:
    side = infer_trade_side(event.event_type)
    if side is None:
        raise ValueError(f"Unsupported event type for trading: {event.event_type}")

    generated_at = event.detected_at or datetime.now(UTC)
    base_confidence = min(max(event.confidence, 0.0), 1.0)
    event_bias = abs(_event_bias_for(event.event_type))
    price_move = _price_momentum(quote)
    price_confirmed, price_rationale = _price_confirmation_state(
        side=side,
        price_move=price_move,
        threshold=price_confirmation_threshold,
    )
    volume_confirmed, volume_unavailable, volume_rationale = _volume_confirmation_state(
        quote=quote,
        threshold=volume_spike_threshold,
    )

    volatility_multiplier, volatility_rationale = _volatility_adjustment(quote)
    event_novelty_score, novelty_rationale = _event_novelty_adjustment(event)
    novelty_multiplier = 0.65 + (event_novelty_score * 0.35)
    signal_age_minutes = _resolve_signal_age_minutes(
        generated_at=generated_at,
        quote=quote,
        evaluation_time=evaluation_time,
    )
    time_decay_factor, time_decay_rationale = _signal_time_decay_factor(
        age_minutes=signal_age_minutes,
        enabled=signal_time_decay_enabled,
        half_life_minutes=signal_decay_half_life_minutes,
        floor=signal_decay_floor,
    )
    effective_confidence = min(
        max(
            base_confidence
            * volatility_multiplier
            * novelty_multiplier
            * time_decay_factor,
            0.0,
        ),
        1.0,
    )

    news_confirmed = True
    confirmation_count = sum((news_confirmed, price_confirmed, volume_confirmed))
    confirmation_score = _weighted_confirmation_score(
        news_confirmed=news_confirmed,
        price_confirmed=price_confirmed,
        volume_confirmed=volume_confirmed,
    )
    synergy_bonus = _confirmation_synergy_bonus(
        news_confirmed=news_confirmed,
        price_confirmed=price_confirmed,
        volume_confirmed=volume_confirmed,
    )
    quality_adjustment = round(
        ((effective_confidence - 0.5) * 0.12)
        + ((event_novelty_score - 0.5) * 0.06)
        + (event_bias * 0.01),
        4,
    )
    score = _clamp_signal_score(confirmation_score + synergy_bonus + quality_adjustment)

    bias_label = "bullish" if side == "buy" else "bearish"
    rationale_items = [
        f"{bias_label.title()} event bias from {event.event_type}",
        f"Event confidence contributed {base_confidence:.2f} to the opportunity score.",
        price_rationale,
        (
            f"Weighted confirmation score reached {confirmation_score:.2f} from "
            f"{confirmation_count} active signal(s) "
            f"(news={_NEWS_SIGNAL_WEIGHT:.2f}, price={_PRICE_CONFIRMATION_WEIGHT:.2f}, "
            f"volume={_VOLUME_CONFIRMATION_WEIGHT:.2f})."
        ),
    ]
    if synergy_bonus > 0:
        rationale_items.append(
            f"Signal synergy bonus added {synergy_bonus:.2f} after multiple confirmations aligned."
        )
    if volume_rationale is not None:
        rationale_items.append(volume_rationale)
    score, event_specific_rationale = _apply_event_score_adjustments(
        event_type=event.event_type,
        score=score,
        price_confirmed=price_confirmed,
        volume_confirmed=volume_confirmed,
        volume_unavailable=volume_unavailable,
        technical_breakout_volume_penalty=technical_breakout_volume_penalty,
    )
    rationale_items.extend(event_specific_rationale)
    if novelty_rationale is not None:
        rationale_items.append(novelty_rationale)
    if volatility_rationale is not None:
        rationale_items.append(volatility_rationale)
    if time_decay_rationale is not None:
        rationale_items.append(time_decay_rationale)
    rationale = tuple(rationale_items)

    event_group = event_performance_group(event.event_type)
    signal_family = _resolve_signal_family(
        event_type=event.event_type,
        event_group=event_group,
    )
    feature_set = _resolve_feature_set(volume_unavailable=volume_unavailable)
    signal_version = _resolve_signal_version(
        signal_family=signal_family,
        feature_set=feature_set,
    )
    thesis_tokens = _extract_thesis_tokens(event=event, symbol_id=quote.symbol_id)
    return Signal(
        signal_id=_build_signal_id(
            raw_event_id=event.raw_event_id,
            event_type=event.event_type,
            symbol_id=quote.symbol_id,
            generated_at=generated_at,
        ),
        signal_family=signal_family,
        signal_version=signal_version,
        model_version="rule-based",
        feature_set=feature_set,
        asset=_infer_asset(quote.symbol_id),
        direction=_resolve_direction(side),
        raw_event_id=event.raw_event_id,
        event_type=event.event_type,
        symbol_id=quote.symbol_id,
        side=side,
        confidence=round(effective_confidence, 4),
        score=score,
        current_price=quote.current,
        generated_at=generated_at,
        rationale=rationale,
        event_group=event_group,
        thesis_fingerprint=_build_thesis_fingerprint(
            symbol_id=quote.symbol_id,
            side=side,
            event_type=event.event_type,
            event_group=event_group,
            thesis_tokens=thesis_tokens,
        ),
        thesis_tokens=thesis_tokens,
        event_novelty_score=event_novelty_score,
        event_repeat_count=max(event.repeat_count, 0),
        narrative_key=event.narrative_key,
        news_confirmed=news_confirmed,
        price_confirmed=price_confirmed,
        volume_confirmed=volume_confirmed,
        volume_unavailable=volume_unavailable,
        confirmation_count=confirmation_count,
        confirmation_score=confirmation_score,
    )


def build_trade_intent(
    *,
    signal: Signal,
    notional_usd: float,
    rationale_suffix: tuple[str, ...] = (),
    exit_horizon_label: str | None = None,
    max_hold_minutes: int | None = None,
    position_id: str | None = None,
    expected_move_fraction: float | None = None,
    stop_distance_fraction: float | None = None,
    risk_reward_ratio: float | None = None,
    selection_rank: int | None = None,
    selection_composite_score: float | None = None,
    rejected_alternatives: tuple[RejectedTradeCandidate, ...] = (),
    heuristic_version: str | None = None,
) -> TradeIntent:
    quantity = 0.0
    if signal.current_price > 0:
        quantity = round(notional_usd / signal.current_price, 8)

    horizon_due_at = None
    if max_hold_minutes is not None and max_hold_minutes > 0:
        horizon_due_at = signal.generated_at + timedelta(minutes=max_hold_minutes)

    rationale = signal.rationale + rationale_suffix + (
        f"Target notional set to ${notional_usd:,.2f}.",
    )
    resolved_signal_id = signal.signal_id or _build_signal_id(
        raw_event_id=signal.raw_event_id,
        event_type=signal.event_type,
        symbol_id=signal.symbol_id,
        generated_at=signal.generated_at,
    )

    return TradeIntent(
        symbol_id=signal.symbol_id,
        side=signal.side,
        notional_usd=round(notional_usd, 2),
        quantity=quantity,
        current_price=signal.current_price,
        score=signal.score,
        rationale=rationale,
        confidence_score=signal.score,
        expected_move="up" if signal.side == "buy" else "down",
        expected_move_fraction=expected_move_fraction,
        stop_distance_fraction=stop_distance_fraction,
        risk_reward_ratio=risk_reward_ratio,
        generated_at=signal.generated_at,
        signal_id=resolved_signal_id,
        signal_family=signal.signal_family,
        signal_version=signal.signal_version,
        model_version=signal.model_version,
        feature_set=signal.feature_set,
        asset=signal.asset,
        direction=signal.direction,
        confidence=signal.confidence,
        raw_event_id=signal.raw_event_id,
        event_type=signal.event_type,
        event_group=signal.event_group,
        exit_horizon_label=exit_horizon_label,
        max_hold_minutes=max_hold_minutes,
        exit_due_at=horizon_due_at,
        position_id=position_id,
        selection_rank=selection_rank,
        selection_composite_score=selection_composite_score,
        rejected_alternatives=rejected_alternatives,
        heuristic_version=heuristic_version,
    )


def _weighted_confirmation_score(
    *,
    news_confirmed: bool,
    price_confirmed: bool,
    volume_confirmed: bool,
) -> float:
    score = 0.0
    if news_confirmed:
        score += _NEWS_SIGNAL_WEIGHT
    if price_confirmed:
        score += _PRICE_CONFIRMATION_WEIGHT
    if volume_confirmed:
        score += _VOLUME_CONFIRMATION_WEIGHT
    return round(min(score, 1.0), 4)


def _confirmation_synergy_bonus(
    *,
    news_confirmed: bool,
    price_confirmed: bool,
    volume_confirmed: bool,
) -> float:
    if news_confirmed and price_confirmed and volume_confirmed:
        return _ALL_CONFIRMATIONS_SYNERGY_BONUS
    if price_confirmed and volume_confirmed:
        return _PRICE_VOLUME_SYNERGY_BONUS
    if news_confirmed and price_confirmed:
        return _NEWS_PRICE_SYNERGY_BONUS
    return 0.0


def _resolve_signal_family(*, event_type: str, event_group: str | None) -> str:
    normalized_type = event_type.lower()
    normalized_group = (event_group or "").lower()
    if "news" in normalized_group or any(
        token in normalized_type for token in ("etf", "news", "fed", "sec", "cpi")
    ):
        return "news_sentiment"
    if any(token in normalized_type for token in ("volume", "breakout")):
        return "volume_breakout"
    if any(token in normalized_type for token in ("momentum", "trend")):
        return "momentum"
    return normalized_group or "event_signal"


def _resolve_feature_set(*, volume_unavailable: bool) -> str:
    components = ["news", "price"]
    if not volume_unavailable:
        components.append("volume")
    return "+".join(components)


def _resolve_signal_version(*, signal_family: str, feature_set: str) -> str:
    if signal_family == "news_sentiment":
        if "volume" in feature_set:
            return _NEWS_SIGNAL_VERSION
        return _NEWS_SIGNAL_PRICE_ONLY_VERSION
    if signal_family == "volume_breakout":
        return _VOLUME_BREAKOUT_SIGNAL_VERSION
    if signal_family == "momentum":
        return _MOMENTUM_SIGNAL_VERSION
    return "v1"


def _infer_asset(value: str) -> str:
    candidate = str(value).split("_", maxsplit=1)[0].strip().upper()
    return candidate or str(value).upper()


def _resolve_direction(side: TradeSide) -> SignalDirection:
    return "long" if side == "buy" else "short"


def _build_signal_id(
    *,
    raw_event_id: str,
    event_type: str,
    symbol_id: str,
    generated_at: datetime,
) -> str:
    digest = hashlib.sha256(
        f"{raw_event_id}|{event_type}|{symbol_id}|{generated_at.isoformat()}".encode()
    ).hexdigest()[:12]
    return f"signal-{digest}"


__all__ = [
    "MoveDirection",
    "NoTradeDecision",
    "RejectedTradeCandidate",
    "Signal",
    "TradeIntent",
    "TradeSide",
    "build_signal",
    "build_trade_intent",
    "infer_trade_side",
    "select_quote_for_event",
]
