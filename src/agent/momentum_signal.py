"""Price-momentum confirmation helpers for signals."""

from __future__ import annotations

from typing import Literal

from ingestion.prices_ingestion import PriceQuote

TradeSide = Literal["buy", "sell"]
DEFAULT_PRICE_CONFIRMATION_THRESHOLD = 0.001


def price_momentum(quote: PriceQuote) -> float:
    """Return the move from session open as a fraction."""

    if quote.open <= 0:
        return 0.0
    return (quote.current - quote.open) / quote.open


def price_confirmation_state(
    *,
    side: TradeSide,
    price_move: float,
    threshold: float = DEFAULT_PRICE_CONFIRMATION_THRESHOLD,
) -> tuple[bool, str]:
    """Determine whether price action confirms the thesis."""

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


__all__ = [
    "DEFAULT_PRICE_CONFIRMATION_THRESHOLD",
    "price_confirmation_state",
    "price_momentum",
]
