"""Explainable signal scoring and trade-intent models."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Literal

from detection.event_detection import DetectedEvent
from detection.event_types import event_performance_group
from ingestion.prices_ingestion import PriceQuote

TradeSide = Literal["buy", "sell"]

_EVENT_BIAS: dict[str, float] = {
    "ETF_APPROVAL": 0.95,
    "TOKEN_LISTING": 0.75,
    "PROTOCOL_UPGRADE": 0.55,
    "ETF_DELAY": -0.7,
    "REGULATORY_ACTION": -0.45,
    "STABLECOIN_DEPEG": -0.95,
    "SECURITY_INCIDENT": -0.95,
    "NETWORK_OUTAGE": -0.65,
    "ETF_NEWS": 0.8,
    "REGULATORY_NEWS": -0.35,
    "EXCHANGE_HACK": -0.95,
    "EXCHANGE_HACKS": -0.95,
    "MACRO_NEWS": 0.25,
    "WHALE_ACTIVITY": 0.65,
    "TECHNICAL_BREAKOUT": 0.7,
    "TECHNICAL_BREAKOUTS": 0.7,
}

_SYMBOL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "btc_usd": ("btc", "bitcoin", "xbt"),
    "eth_usd": ("eth", "ether", "ethereum"),
    "sol_usd": ("sol", "solana"),
    "xrp_usd": ("xrp", "ripple"),
}

_PREFERRED_SYMBOLS: tuple[str, ...] = ("btc_usd", "eth_usd", "sol_usd", "xrp_usd")


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
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    signal_id: str | None = None
    raw_event_id: str | None = None
    event_type: str | None = None
    event_group: str | None = None
    exit_horizon_label: str | None = None
    max_hold_minutes: int | None = None
    exit_due_at: datetime | None = None
    position_id: str | None = None


def infer_trade_side(event_type: str) -> TradeSide | None:
    bias = _EVENT_BIAS.get(event_type, 0.0)
    if bias > 0:
        return "buy"
    if bias < 0:
        return "sell"
    return None


def select_quote_for_event(
    *,
    event: DetectedEvent,
    price_quotes: list[PriceQuote],
) -> PriceQuote | None:
    if not price_quotes:
        return None

    quote_by_symbol = {quote.symbol_id: quote for quote in price_quotes}
    event_text = f"{event.event_type} {event.rule_name} {event.matched_text or ''}".lower()

    for symbol_id, keywords in _SYMBOL_KEYWORDS.items():
        if any(keyword in event_text for keyword in keywords) and symbol_id in quote_by_symbol:
            return quote_by_symbol[symbol_id]

    for symbol_id in _PREFERRED_SYMBOLS:
        if symbol_id in quote_by_symbol:
            return quote_by_symbol[symbol_id]

    return price_quotes[0]


def build_signal(*, event: DetectedEvent, quote: PriceQuote) -> Signal:
    side = infer_trade_side(event.event_type)
    if side is None:
        raise ValueError(f"Unsupported event type for trading: {event.event_type}")

    event_bias = abs(_EVENT_BIAS[event.event_type])
    price_move = _price_momentum(quote)
    aligned_move = max(price_move, 0.0) if side == "buy" else max(-price_move, 0.0)
    price_bonus = min(aligned_move * 3.0, 0.12)
    score = min(1.0, round((event.confidence * 0.72) + (event_bias * 0.18) + price_bonus, 4))

    bias_label = "bullish" if side == "buy" else "bearish"
    rationale = (
        f"{bias_label.title()} event bias from {event.event_type}",
        f"Event confidence contributed {event.confidence:.2f} to the opportunity score.",
        f"Price confirmation move: {price_move * 100:.2f}% from the session open.",
    )

    generated_at = event.detected_at or datetime.now(UTC)
    return Signal(
        signal_id=_build_signal_id(
            raw_event_id=event.raw_event_id,
            event_type=event.event_type,
            symbol_id=quote.symbol_id,
            generated_at=generated_at,
        ),
        raw_event_id=event.raw_event_id,
        event_type=event.event_type,
        symbol_id=quote.symbol_id,
        side=side,
        confidence=event.confidence,
        score=score,
        current_price=quote.current,
        generated_at=generated_at,
        rationale=rationale,
        event_group=event_performance_group(event.event_type),
    )


def build_trade_intent(
    *,
    signal: Signal,
    notional_usd: float,
    rationale_suffix: tuple[str, ...] = (),
    exit_horizon_label: str | None = None,
    max_hold_minutes: int | None = None,
    position_id: str | None = None,
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
        generated_at=signal.generated_at,
        signal_id=resolved_signal_id,
        raw_event_id=signal.raw_event_id,
        event_type=signal.event_type,
        event_group=signal.event_group,
        exit_horizon_label=exit_horizon_label,
        max_hold_minutes=max_hold_minutes,
        exit_due_at=horizon_due_at,
        position_id=position_id,
    )


def _price_momentum(quote: PriceQuote) -> float:
    if quote.open <= 0:
        return 0.0
    return (quote.current - quote.open) / quote.open


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
