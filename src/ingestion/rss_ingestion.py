"""Crypto RSS ingestion and basic processing for the information layer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from ingestion.rss_config import FeedSource

if TYPE_CHECKING:
    from storage.raw_ingestion import IngestionRunResult, RSSRawIngestionPipeline


@dataclass(frozen=True)
class FeedArticle:
    """Normalized RSS article representation."""

    title: str
    url: str
    published: str
    source: str

    @property
    def dedup_hash(self) -> str:
        return sha256(f"{self.title}{self.source}".encode()).hexdigest()


def _default_parse_feed(url: str) -> dict[str, Any]:
    """Parse RSS URL with feedparser on demand to keep tests deterministic."""
    import feedparser

    parsed = feedparser.parse(url)
    entries = []
    for item in parsed.entries:
        entries.append(
            {
                "title": str(getattr(item, "title", "")),
                "link": str(getattr(item, "link", "")),
                "published": str(getattr(item, "published", "")),
            }
        )
    return {"entries": entries}


class RSSIngestionService:
    """Fetches grouped RSS sources and runs basic in-memory deduplication."""

    def __init__(
        self,
        feed_groups: dict[str, list[FeedSource]],
        parse_feed: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        self._feed_groups = feed_groups
        self._parse_feed = parse_feed or _default_parse_feed

    @staticmethod
    def _entry_to_article(entry: dict[str, Any], source: FeedSource) -> FeedArticle:
        """Normalize a parser entry dictionary into a FeedArticle."""
        return FeedArticle(
            title=str(entry.get("title", "")),
            url=str(entry.get("link", "")),
            published=str(entry.get("published", "")),
            source=source.source_id,
        )

    def fetch_group(self, group_name: str) -> list[FeedArticle]:
        articles: list[FeedArticle] = []
        for source in self._feed_groups[group_name]:
            parsed = self._parse_feed(source.url)
            entries = parsed.get("entries", [])
            for entry in entries:
                articles.append(self._entry_to_article(entry, source))
        return articles

    def fetch_from_each_group(self) -> dict[str, list[FeedArticle]]:
        return {group_name: self.fetch_group(group_name) for group_name in self._feed_groups}

    def deduplicate(self, articles: list[FeedArticle]) -> list[FeedArticle]:
        seen_hashes: set[str] = set()
        unique: list[FeedArticle] = []
        for article in articles:
            if article.dedup_hash in seen_hashes:
                continue
            seen_hashes.add(article.dedup_hash)
            unique.append(article)
        return unique

    def fetch_and_persist_group(
        self,
        group_name: str,
        pipeline: RSSRawIngestionPipeline,
    ) -> IngestionRunResult:
        """Fetch one group and persist append-only raw records via the shared pipeline."""

        articles = self.fetch_group(group_name)
        return pipeline.persist_articles(source_group=group_name, articles=articles)
