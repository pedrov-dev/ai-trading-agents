from datetime import date

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
