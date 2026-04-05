"""Explainable signal scoring and trade-intent models."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Literal

from detection.event_detection import DetectedEvent
from detection.event_types import event_performance_group
from ingestion.prices_ingestion import PriceQuote

TradeSide = Literal["buy", "sell"]
MoveDirection = Literal["up", "down", "flat"]

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

_REASONING_STOPWORDS: set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "asset",
    "assets",
    "at",
    "because",
    "for",
    "from",
    "into",
    "is",
    "of",
    "on",
    "risk",
    "the",
    "to",
    "with",
}

_REASONING_THEME_ALIASES: dict[str, str] = {
    "macro": "macro",
    "macroeconomic": "macro",
    "liquidity": "macro",
    "headwind": "macro",
    "headwinds": "macro",
    "tight": "macro",
    "tightening": "macro",
    "pressure": "macro",
    "pressuring": "macro",
    "pressured": "macro",
    "rate": "macro",
    "rates": "macro",
    "yield": "macro",
    "yields": "macro",
    "usd": "macro",
    "dollar": "macro",
    "dxy": "macro",
    "strength": "macro",
    "weakness": "macro",
    "fed": "macro",
    "federal": "macro",
    "inflation": "macro",
    "cpi": "macro",
    "ppi": "macro",
    "hawkish": "macro",
    "dovish": "macro",
    "approval": "approval",
    "approved": "approval",
    "delay": "delay",
    "delayed": "delay",
    "listing": "listing",
    "listed": "listing",
    "regulation": "regulation",
    "regulatory": "regulation",
    "sec": "regulation",
    "hack": "security",
    "hacked": "security",
    "breach": "security",
    "exploit": "security",
    "breakout": "technical",
    "breakouts": "technical",
    "whale": "flow",
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
    thesis_fingerprint: str | None = None
    thesis_tokens: tuple[str, ...] = ()


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
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    signal_id: str | None = None
    raw_event_id: str | None = None
    event_type: str | None = None
    event_group: str | None = None
    exit_horizon_label: str | None = None
    max_hold_minutes: int | None = None
    exit_due_at: datetime | None = None
    position_id: str | None = None

    def __post_init__(self) -> None:
        resolved_confidence = self.score if self.confidence_score is None else self.confidence_score
        if not 0.0 <= resolved_confidence <= 1.0:
            raise ValueError("Trade intent confidence_score must be between 0.0 and 1.0.")

        resolved_expected_move = self.expected_move or ("up" if self.side == "buy" else "down")
        if resolved_expected_move not in {"up", "down", "flat"}:
            raise ValueError("Trade intent expected_move must be one of: up, down, flat.")

        object.__setattr__(self, "confidence_score", round(resolved_confidence, 4))
        object.__setattr__(self, "expected_move", resolved_expected_move)


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

        object.__setattr__(self, "confidence_score", round(self.confidence_score, 4))
        object.__setattr__(self, "score", round(self.score, 4))
        if self.threshold is not None:
            object.__setattr__(self, "threshold", round(self.threshold, 4))


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
    event_group = event_performance_group(event.event_type)
    thesis_tokens = _extract_thesis_tokens(event=event, symbol_id=quote.symbol_id)
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
        event_group=event_group,
        thesis_fingerprint=_build_thesis_fingerprint(
            symbol_id=quote.symbol_id,
            side=side,
            event_type=event.event_type,
            event_group=event_group,
            thesis_tokens=thesis_tokens,
        ),
        thesis_tokens=thesis_tokens,
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
        confidence_score=signal.score,
        expected_move="up" if signal.side == "buy" else "down",
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


def _extract_thesis_tokens(*, event: DetectedEvent, symbol_id: str) -> tuple[str, ...]:
    event_text = f"{event.rule_name} {event.matched_text or ''}".lower()
    raw_tokens = re.findall(r"[a-z0-9]+", event_text)
    symbol_keywords = set(_SYMBOL_KEYWORDS.get(symbol_id, ()))
    normalized_tokens: list[str] = []

    for token in raw_tokens:
        if token in symbol_keywords or token in {"bullish", "bearish", "up", "down"}:
            continue
        if len(token) <= 2 or token in _REASONING_STOPWORDS:
            continue

        normalized_token = _REASONING_THEME_ALIASES.get(token, token)
        if normalized_token in _REASONING_STOPWORDS:
            continue
        normalized_tokens.append(normalized_token)

    ordered_tokens = tuple(sorted(dict.fromkeys(normalized_tokens)))
    if ordered_tokens:
        return ordered_tokens

    fallback = event_performance_group(event.event_type) or event.event_type.lower()
    return (fallback,)


def _build_thesis_fingerprint(
    *,
    symbol_id: str,
    side: TradeSide,
    event_type: str,
    event_group: str | None,
    thesis_tokens: tuple[str, ...],
) -> str:
    reasoning_signature = "|".join(thesis_tokens) if thesis_tokens else "generic"
    digest = hashlib.sha256(
        f"{symbol_id}|{side}|{event_type}|{event_group or ''}|{reasoning_signature}".encode()
    ).hexdigest()[:12]
    return f"thesis-{digest}"


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
