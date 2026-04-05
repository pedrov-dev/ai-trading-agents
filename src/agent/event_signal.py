"""Event-level quality modifiers for signal generation."""

from __future__ import annotations

from datetime import UTC, datetime

from detection.event_detection import DetectedEvent
from ingestion.prices_ingestion import PriceQuote

TECHNICAL_BREAKOUT_EVENT_TYPES: frozenset[str] = frozenset(
    {"TECHNICAL_BREAKOUT", "TECHNICAL_BREAKOUTS"}
)
SHOCK_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "SECURITY_INCIDENT",
        "EXCHANGE_HACK",
        "EXCHANGE_HACKS",
        "STABLECOIN_DEPEG",
        "NETWORK_OUTAGE",
    }
)


def event_novelty_adjustment(event: DetectedEvent) -> tuple[float, str | None]:
    """Return a novelty multiplier input and explanation."""

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


def resolve_signal_age_minutes(
    *,
    generated_at: datetime,
    quote: PriceQuote,
    evaluation_time: datetime | None,
) -> float:
    """Resolve how stale a signal is at evaluation time."""

    resolved_evaluation_time = evaluation_time
    if resolved_evaluation_time is None and quote.timestamp > 0:
        try:
            resolved_evaluation_time = datetime.fromtimestamp(quote.timestamp, tz=UTC)
        except (OverflowError, OSError, ValueError):
            resolved_evaluation_time = None

    if resolved_evaluation_time is None:
        return 0.0

    return max((resolved_evaluation_time - generated_at).total_seconds() / 60.0, 0.0)


def signal_time_decay_factor(
    *,
    age_minutes: float,
    enabled: bool,
    half_life_minutes: float,
    floor: float,
) -> tuple[float, str | None]:
    """Return the freshness decay multiplier applied to confidence."""

    if not enabled or age_minutes <= 0 or half_life_minutes <= 0:
        return 1.0, None

    bounded_floor = min(max(floor, 0.0), 1.0)
    decay_factor = max(bounded_floor, 0.5 ** (age_minutes / half_life_minutes))
    rounded_factor = round(decay_factor, 4)
    if rounded_factor >= 0.9999:
        return 1.0, None

    return (
        rounded_factor,
        "Signal time decay reduced confidence with age: "
        f"{age_minutes:.1f} stale minute(s) -> factor {rounded_factor:.2f}.",
    )


def volatility_adjustment(quote: PriceQuote) -> tuple[float, str | None]:
    """Return a volatility-based confidence adjustment."""

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


def apply_event_score_adjustments(
    *,
    event_type: str,
    score: float,
    price_confirmed: bool,
    volume_confirmed: bool,
    volume_unavailable: bool,
    technical_breakout_volume_penalty: float = 0.15,
) -> tuple[float, tuple[str, ...]]:
    """Apply event-type-specific score overrides and rationale."""

    adjusted_score = clamp_signal_score(score)
    rationale_items: list[str] = []

    if event_type in SHOCK_EVENT_TYPES and not price_confirmed:
        adjusted_score = clamp_signal_score(adjusted_score + 0.25)
        rationale_items.append(
            "Shock-event severity kept the setup actionable even before "
            "full price confirmation arrived."
        )

    if (
        event_type in TECHNICAL_BREAKOUT_EVENT_TYPES
        and not volume_confirmed
        and not volume_unavailable
    ):
        adjusted_score = clamp_signal_score(
            adjusted_score - technical_breakout_volume_penalty
        )
        rationale_items.append(
            "Technical breakout lacked a confirming volume spike, so the "
            "setup was downgraded into reduced-size mode."
        )

    return adjusted_score, tuple(rationale_items)


def clamp_signal_score(value: float) -> float:
    """Clamp a signal score to the 0..1 range."""

    return round(min(max(value, 0.0), 1.0), 4)


__all__ = [
    "SHOCK_EVENT_TYPES",
    "TECHNICAL_BREAKOUT_EVENT_TYPES",
    "apply_event_score_adjustments",
    "clamp_signal_score",
    "event_novelty_adjustment",
    "resolve_signal_age_minutes",
    "signal_time_decay_factor",
    "volatility_adjustment",
]
