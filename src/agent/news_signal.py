"""News- and event-driven helpers for signal generation."""

from __future__ import annotations

import hashlib
import re
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


def event_bias_for(event_type: str) -> float:
    """Return the directional bias associated with a supported event type."""

    return _EVENT_BIAS.get(event_type, 0.0)


def infer_trade_side(event_type: str) -> TradeSide | None:
    """Map the event bias to a trade side."""

    bias = event_bias_for(event_type)
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
    """Select the quote most relevant to the event narrative."""

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


def extract_thesis_tokens(*, event: DetectedEvent, symbol_id: str) -> tuple[str, ...]:
    """Extract stable narrative tokens used to deduplicate similar theses."""

    event_text = f"{event.rule_name} {event.matched_text or ''}".lower()
    raw_tokens = re.findall(r"[a-z0-9]+", event_text)
    symbol_keywords = set(_SYMBOL_KEYWORDS.get(symbol_id, ()))
    normalized_tokens: list[str] = []

    for token in raw_tokens:
        if token in symbol_keywords or token in {"bullish", "bearish", "up", "down"}:
            continue
        if len(token) <= 2 or token in _REASONING_STOPWORDS:
            continue

        normalized_token = _REASONING_THEME_ALIASES.get(token) or token
        if normalized_token in _REASONING_STOPWORDS:
            continue
        normalized_tokens.append(normalized_token)

    ordered_tokens = tuple(sorted(dict.fromkeys(normalized_tokens)))
    if ordered_tokens:
        return ordered_tokens

    fallback = event_performance_group(event.event_type) or event.event_type.lower()
    return (fallback,)


def build_thesis_fingerprint(
    *,
    symbol_id: str,
    side: TradeSide,
    event_type: str,
    event_group: str | None,
    thesis_tokens: tuple[str, ...],
) -> str:
    """Build a stable fingerprint for the thesis behind a signal."""

    reasoning_signature = "|".join(thesis_tokens) if thesis_tokens else "generic"
    digest = hashlib.sha256(
        f"{symbol_id}|{side}|{event_type}|{event_group or ''}|{reasoning_signature}".encode()
    ).hexdigest()[:12]
    return f"thesis-{digest}"


def _event_mentions_keyword(event_text: str, keyword: str) -> bool:
    normalized_keyword = keyword.strip().lower()
    if not normalized_keyword:
        return False
    if " " in normalized_keyword or "-" in normalized_keyword or "/" in normalized_keyword:
        return normalized_keyword in event_text
    pattern = rf"(?<![a-z0-9]){re.escape(normalized_keyword)}(?![a-z0-9])"
    return re.search(pattern, event_text) is not None


__all__ = [
    "TradeSide",
    "build_thesis_fingerprint",
    "event_bias_for",
    "extract_thesis_tokens",
    "infer_trade_side",
    "select_quote_for_event",
]
