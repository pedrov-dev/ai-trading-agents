"""Crypto market symbols and polling configuration for price ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PriceSymbol:
    """Symbol metadata for crypto price ingestion."""

    symbol_id: str
    ticker: str
    asset_class: Literal["spot"]
    base_asset: str
    quote_asset: str = "USD"


def _spot(*, symbol_id: str, ticker: str, base_asset: str) -> PriceSymbol:
    return PriceSymbol(
        symbol_id=symbol_id,
        ticker=ticker,
        asset_class="spot",
        base_asset=base_asset,
    )


KRAKEN_BASE_URL = "https://api.kraken.com/0/public"

# Crypto trades 24/7, so one-minute polling remains a good cadence for the core universe.
CORE_POLLING_FREQUENCY_SECONDS = 60
SECONDARY_POLLING_FREQUENCY_SECONDS = 60 * 60

# Backwards-compatible alias used by the existing scheduler wiring.
POLLING_FREQUENCY_SECONDS = CORE_POLLING_FREQUENCY_SECONDS

# Tier A: high-liquidity, news-sensitive names we want available every minute.
TIER_A_PRICE_SYMBOLS: list[PriceSymbol] = [
    _spot(symbol_id="btc_usd", ticker="XBTUSD", base_asset="BTC"),
    _spot(symbol_id="eth_usd", ticker="ETHUSD", base_asset="ETH"),
    _spot(symbol_id="sol_usd", ticker="SOLUSD", base_asset="SOL"),
    _spot(symbol_id="xrp_usd", ticker="XRPUSD", base_asset="XRP"),
    _spot(symbol_id="bnb_usd", ticker="BNBUSD", base_asset="BNB"),
    _spot(symbol_id="doge_usd", ticker="DOGEUSD", base_asset="DOGE"),
    _spot(symbol_id="ada_usd", ticker="ADAUSD", base_asset="ADA"),
    _spot(symbol_id="avax_usd", ticker="AVAXUSD", base_asset="AVAX"),
    _spot(symbol_id="link_usd", ticker="LINKUSD", base_asset="LINK"),
    _spot(symbol_id="ton_usd", ticker="TONUSD", base_asset="TON"),
]

# Tier B: narrative-driven names for hourly or event-triggered fetches.
TIER_B_PRICE_SYMBOLS: list[PriceSymbol] = [
    _spot(symbol_id="matic_usd", ticker="MATICUSD", base_asset="MATIC"),
    _spot(symbol_id="dot_usd", ticker="DOTUSD", base_asset="DOT"),
    _spot(symbol_id="ltc_usd", ticker="LTCUSD", base_asset="LTC"),
    _spot(symbol_id="bch_usd", ticker="BCHUSD", base_asset="BCH"),
    _spot(symbol_id="uni_usd", ticker="UNIUSD", base_asset="UNI"),
    _spot(symbol_id="aave_usd", ticker="AAVEUSD", base_asset="AAVE"),
    _spot(symbol_id="arb_usd", ticker="ARBUSD", base_asset="ARB"),
    _spot(symbol_id="op_usd", ticker="OPUSD", base_asset="OP"),
    _spot(symbol_id="render_usd", ticker="RENDERUSD", base_asset="RENDER"),
    _spot(symbol_id="inj_usd", ticker="INJUSD", base_asset="INJ"),
    _spot(symbol_id="near_usd", ticker="NEARUSD", base_asset="NEAR"),
    _spot(symbol_id="atom_usd", ticker="ATOMUSD", base_asset="ATOM"),
    _spot(symbol_id="apt_usd", ticker="APTUSD", base_asset="APT"),
    _spot(symbol_id="sui_usd", ticker="SUIUSD", base_asset="SUI"),
]

# Existing runtime behavior: the app polls Tier A every 60s.
PRICE_SYMBOLS: list[PriceSymbol] = list(TIER_A_PRICE_SYMBOLS)
SECONDARY_PRICE_SYMBOLS: list[PriceSymbol] = list(TIER_B_PRICE_SYMBOLS)
ALL_PRICE_SYMBOLS: list[PriceSymbol] = [*PRICE_SYMBOLS, *SECONDARY_PRICE_SYMBOLS]

