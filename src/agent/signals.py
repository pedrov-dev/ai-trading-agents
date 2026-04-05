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

_NEWS_SIGNAL_WEIGHT = 0.4
_PRICE_CONFIRMATION_WEIGHT = 0.35
_VOLUME_CONFIRMATION_WEIGHT = 0.25
_NEWS_PRICE_SYNERGY_BONUS = 0.05
_PRICE_VOLUME_SYNERGY_BONUS = 0.10
_ALL_CONFIRMATIONS_SYNERGY_BONUS = 0.15
_DEFAULT_PRICE_CONFIRMATION_THRESHOLD = 0.001
_DEFAULT_VOLUME_SPIKE_THRESHOLD = 1.5
_TECHNICAL_BREAKOUT_EVENT_TYPES: frozenset[str] = frozenset(
    {"TECHNICAL_BREAKOUT", "TECHNICAL_BREAKOUTS"}
)
_SHOCK_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "SECURITY_INCIDENT",
        "EXCHANGE_HACK",
        "EXCHANGE_HACKS",
        "STABLECOIN_DEPEG",
        "NETWORK_OUTAGE",
    }
)

_SYMBOL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "btc_usd": ("btc", "bitcoin", "xbt"),
    "eth_usd": ("eth", "ether", "ethereum"),
    "sol_usd": ("sol", "solana"),
    "xrp_usd": ("xrp", "ripple"),
    "bnb_usd": ("bnb", "binance coin", "binance"),
    "doge_usd": ("doge", "dogecoin"),
    "ada_usd": ("ada", "cardano"),
    "avax_usd": ("avax", "avalanche"),
    "link_usd": ("link", "chainlink"),
    "ton_usd": ("toncoin", "telegram", "the open network", "ton network"),
    "matic_usd": ("matic", "polygon", "polygon pos"),
    "dot_usd": ("dot", "polkadot"),
    "ltc_usd": ("ltc", "litecoin"),
    "bch_usd": ("bch", "bitcoin cash"),
    "uni_usd": ("uniswap",),
    "aave_usd": ("aave",),
    "arb_usd": ("arbitrum",),
    "op_usd": ("optimism", "op mainnet"),
    "render_usd": ("render", "rndr"),
    "inj_usd": ("injective",),
    "near_usd": ("near protocol", "near"),
    "atom_usd": ("cosmos", "atom"),
    "apt_usd": ("aptos",),
    "sui_usd": ("sui",),
}

_PREFERRED_SYMBOLS: tuple[str, ...] = (
    "btc_usd",
    "eth_usd",
    "sol_usd",
    "xrp_usd",
    "bnb_usd",
    "doge_usd",
    "ada_usd",
    "avax_usd",
    "link_usd",
    "ton_usd",
    "matic_usd",
    "dot_usd",
    "ltc_usd",
    "bch_usd",
    "uni_usd",
    "aave_usd",
    "arb_usd",
    "op_usd",
    "render_usd",
    "inj_usd",
    "near_usd",
    "atom_usd",
    "apt_usd",
    "sui_usd",
)

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
        if self.expected_move_fraction is not None and self.expected_move_fraction < 0:
            raise ValueError("Trade intent expected_move_fraction must be non-negative.")
        if self.stop_distance_fraction is not None and self.stop_distance_fraction < 0:
            raise ValueError("Trade intent stop_distance_fraction must be non-negative.")
        if self.risk_reward_ratio is not None and self.risk_reward_ratio < 0:
            raise ValueError("Trade intent risk_reward_ratio must be non-negative.")

        object.__setattr__(self, "confidence_score", round(resolved_confidence, 4))
        object.__setattr__(self, "expected_move", resolved_expected_move)
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
        matches_symbol = any(
            _event_mentions_keyword(event_text, keyword) for keyword in keywords
        )
        if matches_symbol and symbol_id in quote_by_symbol:
            return quote_by_symbol[symbol_id]

    for symbol_id in _PREFERRED_SYMBOLS:
        if symbol_id in quote_by_symbol:
            return quote_by_symbol[symbol_id]

    return price_quotes[0]


def _event_mentions_keyword(event_text: str, keyword: str) -> bool:
    normalized_keyword = keyword.strip().lower()
    if not normalized_keyword:
        return False
    if " " in normalized_keyword or "-" in normalized_keyword or "/" in normalized_keyword:
        return normalized_keyword in event_text
    pattern = rf"(?<![a-z0-9]){re.escape(normalized_keyword)}(?![a-z0-9])"
    return re.search(pattern, event_text) is not None


def build_signal(
    *,
    event: DetectedEvent,
    quote: PriceQuote,
    price_confirmation_threshold: float = _DEFAULT_PRICE_CONFIRMATION_THRESHOLD,
    volume_spike_threshold: float = _DEFAULT_VOLUME_SPIKE_THRESHOLD,
    technical_breakout_volume_penalty: float = 0.15,
) -> Signal:
    side = infer_trade_side(event.event_type)
    if side is None:
        raise ValueError(f"Unsupported event type for trading: {event.event_type}")

    base_confidence = min(max(event.confidence, 0.0), 1.0)
    event_bias = abs(_EVENT_BIAS[event.event_type])
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
    effective_confidence = min(
        max(base_confidence * volatility_multiplier * novelty_multiplier, 0.0),
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
    if event.event_type in _SHOCK_EVENT_TYPES and not price_confirmed:
        score = _clamp_signal_score(score + 0.25)
        rationale_items.append(
            "Shock-event severity kept the setup actionable even before "
            "full price confirmation arrived."
        )
    if (
        event.event_type in _TECHNICAL_BREAKOUT_EVENT_TYPES
        and not volume_confirmed
        and not volume_unavailable
    ):
        score = _clamp_signal_score(score - technical_breakout_volume_penalty)
        rationale_items.append(
            "Technical breakout lacked a confirming volume spike, so the "
            "setup was downgraded into reduced-size mode."
        )
    if novelty_rationale is not None:
        rationale_items.append(novelty_rationale)
    if volatility_rationale is not None:
        rationale_items.append(volatility_rationale)
    rationale = tuple(rationale_items)

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
        raw_event_id=signal.raw_event_id,
        event_type=signal.event_type,
        event_group=signal.event_group,
        exit_horizon_label=exit_horizon_label,
        max_hold_minutes=max_hold_minutes,
        exit_due_at=horizon_due_at,
        position_id=position_id,
    )


def _event_novelty_adjustment(event: DetectedEvent) -> tuple[float, str | None]:
    if event.novelty_score is None:
        return 1.0, None

    novelty_score = min(max(event.novelty_score, 0.0), 1.0)
    repeat_count = max(event.repeat_count, 0)
    if novelty_score >= 0.95 and repeat_count == 0:
        return (
            novelty_score,
            "Fresh narrative detected; first occurrence kept the novelty score high.",
        )

    return (
        novelty_score,
        "Repeated narrative lowered novelty to "
        f"{novelty_score:.2f} after {repeat_count} prior occurrence(s).",
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


def _price_confirmation_state(
    *,
    side: TradeSide,
    price_move: float,
    threshold: float,
) -> tuple[bool, str]:
    abs_move_percent = abs(price_move) * 100
    threshold = max(threshold, 0.0)

    if side == "buy":
        if price_move >= threshold:
            return (
                True,
                "Price confirmation via breakout supported the bullish "
                f"thesis with a {price_move * 100:.2f}% move from the session open.",
            )
        if price_move <= -threshold:
            return (
                False,
                "Price moved against the bullish thesis by "
                f"{abs_move_percent:.2f}%, so confirmation is not active yet.",
            )
        return (
            False,
            "Price confirmation is still pending for the bullish thesis; "
            f"the move from the session open is only {price_move * 100:.2f}%.",
        )

    if price_move <= -threshold:
        return (
            True,
            "Price confirmation from the price breakdown supported the "
            f"bearish thesis with a {abs_move_percent:.2f}% move from the session open.",
        )
    if price_move >= threshold:
        return (
            False,
            "Price moved against the bearish thesis by "
            f"{price_move * 100:.2f}%, so confirmation is not active yet.",
        )
    return (
        False,
        "Price confirmation is still pending for the bearish thesis; the "
        f"move from the session open is only {price_move * 100:.2f}%.",
    )


def _volume_confirmation_state(
    *,
    quote: PriceQuote,
    threshold: float,
) -> tuple[bool, bool, str | None]:
    volume_ratio = quote.volume_ratio
    if volume_ratio is None:
        return (
            False,
            True,
            "Volume confirmation is unavailable for this quote, so the "
            "setup stays below max confidence.",
        )

    session_volume_label = (
        f" on {quote.session_volume:,.2f} units traded"
        if quote.session_volume is not None and quote.session_volume > 0
        else ""
    )
    if volume_ratio >= max(threshold, 0.0):
        return (
            True,
            False,
            "Volume spike confirmed at "
            f"{volume_ratio:.2f}x the recent baseline{session_volume_label}.",
        )

    return (
        False,
        False,
        "Volume stayed muted at "
        f"{volume_ratio:.2f}x the recent baseline{session_volume_label}; "
        f"it needs {max(threshold, 0.0):.2f}x for confirmation.",
    )


def _clamp_signal_score(value: float) -> float:
    return round(min(max(value, 0.0), 1.0), 4)


def _price_momentum(quote: PriceQuote) -> float:
    if quote.open <= 0:
        return 0.0
    return (quote.current - quote.open) / quote.open


def _volatility_adjustment(quote: PriceQuote) -> tuple[float, str | None]:
    atr = quote.atr
    realized_volatility = quote.realized_volatility
    volatility_filter = quote.volatility_filter

    if (
        volatility_filter is None
        and atr is not None
        and realized_volatility is not None
        and quote.current > 0
        and realized_volatility > 0
    ):
        volatility_filter = (atr / quote.current) / max(realized_volatility, 0.0001)

    if volatility_filter is None or volatility_filter <= 0:
        return 1.0, None

    bounded_filter = min(max(volatility_filter, 0.0), 3.0)
    atr_label = f"ATR={atr:,.2f}" if atr is not None else "ATR=n/a"
    rv_label = (
        f"realized_volatility={realized_volatility:.4f}"
        if realized_volatility is not None
        else "realized_volatility=n/a"
    )
    if bounded_filter < 1.0:
        multiplier = max(0.65, 1.0 - ((1.0 - bounded_filter) * 0.35))
        return (
            round(multiplier, 4),
            "Low volatility weakened the signal: "
            f"volatility_filter={bounded_filter:.2f}, {atr_label}, {rv_label}.",
        )
    if bounded_filter > 1.0:
        multiplier = min(1.2, 1.0 + ((bounded_filter - 1.0) * 0.2))
        return (
            round(multiplier, 4),
            "High volatility strengthened the signal: "
            f"volatility_filter={bounded_filter:.2f}, {atr_label}, {rv_label}.",
        )
    return (
        1.0,
        "Volatility stayed neutral for this setup: "
        f"volatility_filter={bounded_filter:.2f}, {atr_label}, {rv_label}.",
    )


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
