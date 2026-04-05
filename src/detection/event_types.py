from __future__ import annotations

from typing import Final

REQUESTED_EVENT_GROUPS: Final[tuple[str, ...]] = (
    "regulatory_news",
    "exchange_hacks",
    "etf_news",
    "macro_news",
    "whale_activity",
    "technical_breakouts",
)

_EVENT_TYPE_GROUPS: Final[dict[str, str]] = {
    "ETF_APPROVAL": "etf_news",
    "ETF_DELAY": "etf_news",
    "ETF_NEWS": "etf_news",
    "REGULATORY_ACTION": "regulatory_news",
    "REGULATORY_NEWS": "regulatory_news",
    "SECURITY_INCIDENT": "exchange_hacks",
    "EXCHANGE_HACK": "exchange_hacks",
    "EXCHANGE_HACKS": "exchange_hacks",
    "MACRO_NEWS": "macro_news",
    "WHALE_ACTIVITY": "whale_activity",
    "TECHNICAL_BREAKOUT": "technical_breakouts",
    "TECHNICAL_BREAKOUTS": "technical_breakouts",
}


def event_performance_group(event_type: str | None) -> str | None:
    """Map an event type to the reporting bucket used for performance pruning."""

    if event_type is None:
        return None

    normalized = event_type.strip().upper()
    if not normalized:
        return None

    if normalized in _EVENT_TYPE_GROUPS:
        return _EVENT_TYPE_GROUPS[normalized]

    return normalized.lower()
