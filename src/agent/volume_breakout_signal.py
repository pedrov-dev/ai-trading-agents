"""Volume breakout confirmation helpers for signals."""

from __future__ import annotations

from ingestion.prices_ingestion import PriceQuote

DEFAULT_VOLUME_SPIKE_THRESHOLD = 1.5
VOLUME_BREAKOUT_SIGNAL_VERSION = "v1"


def volume_confirmation_state(
    *,
    quote: PriceQuote,
    threshold: float = DEFAULT_VOLUME_SPIKE_THRESHOLD,
) -> tuple[bool, bool, str | None]:
    """Determine whether volume confirms the setup."""

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


__all__ = [
    "DEFAULT_VOLUME_SPIKE_THRESHOLD",
    "VOLUME_BREAKOUT_SIGNAL_VERSION",
    "volume_confirmation_state",
]
