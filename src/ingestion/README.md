# Ingestion

Collectors for market prices and crypto RSS feeds; normalize and persist raw records for downstream processing.

## Key modules

- `prices_config.py` / `prices_ingestion.py`: price symbols, Kraken fetcher, current/historical fetch + scheduler wiring.
- `rss_config.py` / `rss_ingestion.py`: feed groups, parser, dedupe, article normalization & persistence.

## Public API

- `PricesIngestionService`, `PriceQuote`, `HistoricalBar`, `wire_prices_fetch_job`, `PRICE_SYMBOLS`.
- `RSSIngestionService`, `FeedArticle`, `fetch_and_persist_group`, `RSS_FEED_GROUPS`.

## Integration

- Persists via the raw ingestion pipeline (storage layer); schedule using `InfoScheduler` or configured polling frequency.

## Usage

```python
from ingestion.prices_ingestion import PricesIngestionService
service = PricesIngestionService(symbols=PRICE_SYMBOLS)
service.fetch_and_persist_current_prices(pipeline)
```

## Notes

- `PRICE_SYMBOLS` now represents the Tier A core universe polled every 60 seconds.
- `SECONDARY_PRICE_SYMBOLS` exposes the Tier B event-driven universe intended for hourly or trigger-based fetches.
- Configure storage backend and network endpoints in the storage layer; Kraken and RSS endpoints are public by default.
