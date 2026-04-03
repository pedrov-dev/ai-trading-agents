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


KRAKEN_BASE_URL = "https://api.kraken.com/0/public"

# Crypto trades 24/7, so one-minute polling remains a good MVP cadence.
POLLING_FREQUENCY_SECONDS = 60

# Focus on liquid Kraken USD spot pairs for the hackathon MVP.
PRICE_SYMBOLS: list[PriceSymbol] = [
    PriceSymbol(symbol_id="btc_usd", ticker="XBTUSD", asset_class="spot", base_asset="BTC"),
    PriceSymbol(symbol_id="eth_usd", ticker="ETHUSD", asset_class="spot", base_asset="ETH"),
    PriceSymbol(symbol_id="sol_usd", ticker="SOLUSD", asset_class="spot", base_asset="SOL"),
    PriceSymbol(symbol_id="xrp_usd", ticker="XRPUSD", asset_class="spot", base_asset="XRP"),
]

