"""Crypto market prices ingestion using Kraken public endpoints."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
from statistics import fmean, pstdev
from typing import TYPE_CHECKING, Any

import requests

from info_scheduler import InfoScheduler
from ingestion.prices_config import KRAKEN_BASE_URL, POLLING_FREQUENCY_SECONDS, PriceSymbol

if TYPE_CHECKING:
    from storage.raw_ingestion import IngestionRunResult, PricesRawIngestionPipeline


@dataclass(frozen=True)
class PriceQuote:
    """Current quote data for one symbol."""

    symbol_id: str
    current: float
    open: float
    high: float
    low: float
    prev_close: float
    timestamp: int
    asset_class: str
    atr: float | None = None
    realized_volatility: float | None = None
    volatility_filter: float | None = None

    @property
    def dedup_hash(self) -> str:
        return sha256(f"prices:{self.symbol_id}:{self.timestamp}".encode()).hexdigest()


@dataclass(frozen=True)
class HistoricalBar:
    """Daily OHLCV bar for one symbol and timestamp."""

    symbol_id: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class _VolatilityMetrics:
    """Pre-computed ATR and realized volatility for a single symbol."""

    atr: float | None = None
    realized_volatility: float | None = None
    volatility_filter: float | None = None


def _default_http_get(url: str, params: dict[str, str]) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object from Kraken")

    errors = payload.get("error", [])
    if isinstance(errors, list) and errors:
        raise ValueError(f"Kraken returned API errors: {errors}")

    return payload


class PricesIngestionService:
    """Fetches current and historical prices for Kraken spot pairs."""

    def __init__(
        self,
        symbols: list[PriceSymbol],
        http_get: Callable[[str, dict[str, str]], dict[str, Any]] | None = None,
        atr_lookback_bars: int = 14,
        realized_volatility_lookback_bars: int = 20,
        enrich_volatility_metrics: bool = False,
    ) -> None:
        self._symbols = symbols
        self._http_get = http_get or _default_http_get
        self._atr_lookback_bars = max(2, atr_lookback_bars)
        self._realized_volatility_lookback_bars = max(2, realized_volatility_lookback_bars)
        self._enrich_volatility_metrics = enrich_volatility_metrics

    def fetch_current_prices(self) -> list[PriceQuote]:
        quotes: list[PriceQuote] = []
        observed_at = int(datetime.now(UTC).timestamp())
        volatility_metrics = (
            self._build_volatility_metrics() if self._enrich_volatility_metrics else {}
        )

        for symbol in self._symbols:
            payload = self._http_get(
                f"{KRAKEN_BASE_URL}/Ticker",
                self._quote_params(symbol),
            )
            ticker_data = self._extract_result_entry(payload, symbol)
            metrics = volatility_metrics.get(symbol.symbol_id, _VolatilityMetrics())
            quotes.append(
                PriceQuote(
                    symbol_id=symbol.symbol_id,
                    current=self._first_float(ticker_data.get("c")),
                    open=self._first_float(ticker_data.get("o")),
                    high=self._first_float(ticker_data.get("h")),
                    low=self._first_float(ticker_data.get("l")),
                    prev_close=self._first_float(ticker_data.get("p")),
                    timestamp=observed_at,
                    asset_class=symbol.asset_class,
                    atr=metrics.atr,
                    realized_volatility=metrics.realized_volatility,
                    volatility_filter=metrics.volatility_filter,
                )
            )
        return quotes

    def fetch_historical_prices(self, start: date, end: date) -> list[HistoricalBar]:
        start_unix = self._to_unix(start)
        end_unix = self._to_unix(end)
        bars: list[HistoricalBar] = []

        for symbol in self._symbols:
            payload = self._http_get(
                f"{KRAKEN_BASE_URL}/{self._candle_endpoint(symbol)}",
                self._candle_params(symbol, start_unix=start_unix, end_unix=end_unix),
            )
            for row in self._extract_ohlc_rows(payload, symbol):
                ts = int(row[0])
                if ts < start_unix or ts > end_unix + 86_400:
                    continue
                bars.append(
                    HistoricalBar(
                        symbol_id=symbol.symbol_id,
                        timestamp=ts,
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        volume=float(row[6]),
                    )
                )

        return bars

    def _build_volatility_metrics(self) -> dict[str, _VolatilityMetrics]:
        lookback_days = max(
            self._atr_lookback_bars,
            self._realized_volatility_lookback_bars,
        ) + 2
        end_date = datetime.now(UTC).date()
        start_date = end_date - timedelta(days=lookback_days)

        try:
            bars = self.fetch_historical_prices(start=start_date, end=end_date)
        except Exception:
            return {}

        bars_by_symbol: dict[str, list[HistoricalBar]] = {}
        for bar in bars:
            bars_by_symbol.setdefault(bar.symbol_id, []).append(bar)

        return {
            symbol_id: self._calculate_volatility_metrics(symbol_bars)
            for symbol_id, symbol_bars in bars_by_symbol.items()
        }

    def _calculate_volatility_metrics(
        self,
        bars: list[HistoricalBar],
    ) -> _VolatilityMetrics:
        ordered_bars = sorted(bars, key=lambda bar: bar.timestamp)
        if len(ordered_bars) < 2:
            return _VolatilityMetrics()

        atr_bars = ordered_bars[-self._atr_lookback_bars :]
        true_ranges: list[float] = []
        previous_close: float | None = None
        for bar in atr_bars:
            true_range = max(bar.high - bar.low, 0.0)
            if previous_close is not None:
                true_range = max(
                    true_range,
                    abs(bar.high - previous_close),
                    abs(bar.low - previous_close),
                )
            true_ranges.append(true_range)
            previous_close = bar.close
        atr = round(fmean(true_ranges), 4) if true_ranges else None

        rv_bars = ordered_bars[-self._realized_volatility_lookback_bars :]
        close_returns: list[float] = []
        previous_close = None
        for bar in rv_bars:
            if previous_close is not None and previous_close > 0:
                close_returns.append((bar.close - previous_close) / previous_close)
            previous_close = bar.close
        realized_volatility = round(pstdev(close_returns), 6) if len(close_returns) >= 2 else None

        latest_close = ordered_bars[-1].close
        volatility_filter = None
        if atr is not None and realized_volatility is not None and latest_close > 0:
            normalized_atr = atr / latest_close
            volatility_filter = round(
                normalized_atr / max(realized_volatility, 0.0001),
                4,
            )

        return _VolatilityMetrics(
            atr=atr,
            realized_volatility=realized_volatility,
            volatility_filter=volatility_filter,
        )

    @staticmethod
    def _to_unix(value: date) -> int:
        return int(datetime(value.year, value.month, value.day, tzinfo=UTC).timestamp())

    @staticmethod
    def _first_float(value: Any) -> float:
        if isinstance(value, list) and value:
            return float(value[0])
        if isinstance(value, (int, float, str)):
            return float(value)
        return 0.0

    @staticmethod
    def _extract_result_entry(payload: dict[str, Any], symbol: PriceSymbol) -> dict[str, Any]:
        result = payload.get("result", {})
        if not isinstance(result, dict) or not result:
            raise ValueError(f"Expected Kraken result payload for pair={symbol.ticker}")

        exact_match = result.get(symbol.ticker)
        if isinstance(exact_match, dict):
            return exact_match

        for candidate in result.values():
            if isinstance(candidate, dict):
                return candidate

        raise ValueError(f"No ticker data found for pair={symbol.ticker}")

    @staticmethod
    def _extract_ohlc_rows(payload: dict[str, Any], symbol: PriceSymbol) -> list[list[Any]]:
        result = payload.get("result", {})
        if not isinstance(result, dict):
            return []

        exact_match = result.get(symbol.ticker)
        if isinstance(exact_match, list):
            return [row for row in exact_match if isinstance(row, list)]

        for key, candidate in result.items():
            if key == "last":
                continue
            if isinstance(candidate, list):
                return [row for row in candidate if isinstance(row, list)]

        return []

    def _quote_params(self, symbol: PriceSymbol) -> dict[str, str]:
        return {"pair": symbol.ticker}

    @staticmethod
    def _candle_endpoint(_symbol: PriceSymbol) -> str:
        return "OHLC"

    def _candle_params(
        self, symbol: PriceSymbol, *, start_unix: int, end_unix: int
    ) -> dict[str, str]:
        del end_unix
        return {
            "pair": symbol.ticker,
            "interval": "1440",
            "since": str(start_unix),
        }

    def fetch_and_persist_current_prices(
        self, pipeline: PricesRawIngestionPipeline
    ) -> IngestionRunResult:
        """Fetch current prices for all symbols and persist raw records via the pipeline."""
        quotes = self.fetch_current_prices()
        return pipeline.persist_quotes(quotes=quotes)


def wire_prices_fetch_job(
    scheduler: InfoScheduler,
    prices_service: PricesIngestionService,
    pipeline: PricesRawIngestionPipeline,
    interval_seconds: int = POLLING_FREQUENCY_SECONDS,
) -> None:
    """Wire periodic price ingestion into the scheduler with raw persistence."""

    def _fetch_current_prices() -> None:
        prices_service.fetch_and_persist_current_prices(pipeline=pipeline)

    scheduler.register_prices_job(
        _fetch_current_prices,
        interval_seconds=interval_seconds,
    )
