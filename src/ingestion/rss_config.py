"""Crypto-focused RSS feed source definitions and polling frequencies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeedSource:
    """RSS source metadata."""

    source_id: str
    url: str


RSS_FEED_GROUPS: dict[str, list[FeedSource]] = {
    "market_news": [
        FeedSource(source_id="coindesk", url="https://www.coindesk.com/arc/outboundfeeds/rss/"),
        FeedSource(source_id="cointelegraph", url="https://cointelegraph.com/rss"),
        FeedSource(source_id="decrypt", url="https://decrypt.co/feed"),
    ],
    "exchange_ops": [
        FeedSource(source_id="kraken_blog", url="https://blog.kraken.com/feed"),
        FeedSource(source_id="kraken_status", url="https://status.kraken.com/history.atom"),
    ],
    "policy": [
        FeedSource(source_id="sec_press", url="https://www.sec.gov/rss/news/press.xml"),
        FeedSource(source_id="sec_litigation", url="https://www.sec.gov/rss/litigation/litreleases.xml"),
        FeedSource(source_id="cftc_press", url="https://www.cftc.gov/PressRoom/PressReleases/rss.xml"),
    ],
    "protocols": [
        FeedSource(source_id="ethereum_blog", url="https://blog.ethereum.org/feed.xml"),
        FeedSource(source_id="solana_blog", url="https://solana.com/rss.xml"),
        FeedSource(source_id="chainalysis_blog", url="https://www.chainalysis.com/blog/feed/"),
    ],
}


POLLING_FREQUENCY_SECONDS: dict[str, int] = {
    "market_news": 120,
    "exchange_ops": 120,
    "policy": 300,
    "protocols": 300,
}
