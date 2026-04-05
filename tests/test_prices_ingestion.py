from datetime import UTC, date, datetime, timedelta

from ingestion.prices_config import PRICE_SYMBOLS
from ingestion.prices_ingestion import PricesIngestionService

BTC_SYMBOL = next(symbol for symbol in PRICE_SYMBOLS if symbol.symbol_id == "btc_usd")


def test_fetch_current_prices_parses_kraken_ticker_payload() -> None:
    def fake_http_get(url: str, params: dict[str, str]) -> dict[str, object]:
        assert url.endswith("/Ticker")
        assert params == {"pair": BTC_SYMBOL.ticker}
        return {
            "error": [],
            "result": {
                BTC_SYMBOL.ticker: {
                    "c": ["64000.1", "1"],
                    "o": "63000.0",
                    "h": ["65000.0", "65000.0"],
                    "l": ["62000.0", "62000.0"],
                    "p": ["63500.0", "63500.0"],
                }
            },
        }

    service = PricesIngestionService(symbols=[BTC_SYMBOL], http_get=fake_http_get)

    quotes = service.fetch_current_prices()

    assert len(quotes) == 1
    assert quotes[0].symbol_id == "btc_usd"
    assert quotes[0].asset_class == "spot"
    assert quotes[0].current == 64000.1


def test_fetch_historical_prices_parses_kraken_ohlc_payload() -> None:
    def fake_http_get(url: str, params: dict[str, str]) -> dict[str, object]:
        assert url.endswith("/OHLC")
        assert params["pair"] == BTC_SYMBOL.ticker
        return {
            "error": [],
            "result": {
                BTC_SYMBOL.ticker: [
                    [1711843200, "62000", "64500", "61500", "64000", "63000", "123.4", 1000],
                    [1711929600, "64000", "65000", "63500", "64800", "64400", "150.2", 1200],
                ],
                "last": 1711929600,
            },
        }

    service = PricesIngestionService(symbols=[BTC_SYMBOL], http_get=fake_http_get)

    bars = service.fetch_historical_prices(date(2024, 3, 31), date(2024, 4, 2))

    assert len(bars) == 2
    assert bars[0].symbol_id == "btc_usd"
    assert bars[1].close == 64800.0
    assert bars[1].volume == 150.2


def test_fetch_current_prices_can_enrich_volatility_metrics() -> None:
    today = date.today()
    recent_days = [today - timedelta(days=offset) for offset in range(5, 0, -1)]
    recent_rows = [
        [
            int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp()),
            open_price,
            high_price,
            low_price,
            close_price,
            vwap,
            volume,
            trades,
        ]
        for day, open_price, high_price, low_price, close_price, vwap, volume, trades in [
            (recent_days[0], "61200", "62600", "60500", "62000", "61800", "80", 700),
            (recent_days[1], "62000", "63200", "61500", "62800", "62500", "82", 720),
            (recent_days[2], "62800", "63800", "62100", "63400", "63100", "85", 740),
            (recent_days[3], "63400", "64500", "62800", "64000", "63800", "88", 760),
            (recent_days[4], "64000", "65000", "63500", "64800", "64400", "90", 780),
        ]
    ]

    def fake_http_get(url: str, params: dict[str, str]) -> dict[str, object]:
        if url.endswith("/Ticker"):
            return {
                "error": [],
                "result": {
                    BTC_SYMBOL.ticker: {
                        "c": ["64000.1", "1"],
                        "o": "63000.0",
                        "h": ["65000.0", "65000.0"],
                        "l": ["62000.0", "62000.0"],
                        "p": ["63500.0", "63500.0"],
                    }
                },
            }
        assert url.endswith("/OHLC")
        return {
            "error": [],
            "result": {
                BTC_SYMBOL.ticker: recent_rows,
                "last": recent_rows[-1][0],
            },
        }

    service = PricesIngestionService(
        symbols=[BTC_SYMBOL],
        http_get=fake_http_get,
        enrich_volatility_metrics=True,
    )

    quote = service.fetch_current_prices()[0]

    assert quote.atr is not None
    assert quote.realized_volatility is not None
    assert quote.volatility_filter is not None
    assert quote.volatility_filter > 0
