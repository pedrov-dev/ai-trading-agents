"""Raw ingestion pipeline abstractions for RSS and crypto price data."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from ingestion.prices_ingestion import PriceQuote
from ingestion.rss_ingestion import FeedArticle


class IngestionRunsRepository(Protocol):
    """Persists lifecycle state for each ingestion run."""

    def start_run(self, source_type: str, started_at: datetime) -> str:
        """Start a run and return its run identifier."""
        ...

    def finish_run(
        self,
        run_id: str,
        *,
        finished_at: datetime,
        status: str,
        fetched_count: int,
        inserted_count: int,
        duplicate_count: int,
        error_summary: str | None,
    ) -> None:
        """Finish a run with counters and terminal status."""
        ...


class RawEventsRepository(Protocol):
    """Persists deduplicated metadata rows for raw payload objects."""

    def insert_raw_event(
        self,
        *,
        source_type: str,
        source_id: str,
        observed_at: datetime,
        event_time: datetime | None,
        dedup_hash: str,
        payload_preview: dict[str, Any],
        object_key: str,
        ingest_run_id: str,
        status: str,
    ) -> bool:
        """Insert one raw event row. Return False when row already exists."""
        ...

    def transition_raw_event_status(
        self,
        *,
        source_type: str,
        dedup_hash: str,
        status: str,
    ) -> None:
        """Transition one existing raw event row status by unique source/hash key."""
        ...

    def transition_pending_raw_events_for_run(
        self,
        *,
        ingest_run_id: str,
        status: str,
    ) -> int:
        """Transition pending raw_event rows for one run and return affected row count."""
        ...

    def fetch_raw_events_for_classification(
        self,
        *,
        source_type: str,
        status: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Fetch raw events with a given status (e.g., 'ok') for classification."""
        ...


class RawObjectStore(Protocol):
    """Stores full raw payloads in S3-compatible object storage."""

    def put_json_gzip(
        self,
        *,
        key: str,
        payload: dict[str, Any],
        metadata: dict[str, str],
    ) -> str:
        """Write compressed JSON payload and return the written object key."""
        ...


@dataclass
class _RawRecord:
    """Normalized item ready to be stored via the object store and event repository."""

    source_id: str
    dedup_hash: str
    event_time: datetime | None
    payload: dict[str, Any]
    payload_preview: dict[str, Any]


@dataclass(frozen=True)
class IngestionRunResult:
    """Summary counters for a completed ingestion run."""

    run_id: str
    status: str
    fetched_count: int
    inserted_count: int
    duplicate_count: int

    @property
    def dedup_rate(self) -> float:
        """Fraction of fetched items that were duplicates; 0.0 when fetched_count is zero."""
        if self.fetched_count == 0:
            return 0.0
        return self.duplicate_count / self.fetched_count


class _RawIngestionPipeline:
    """Shared append-only persist loop used by all source-specific pipelines."""

    def __init__(
        self,
        *,
        source_type: str,
        runs_repository: IngestionRunsRepository,
        raw_events_repository: RawEventsRepository,
        object_store: RawObjectStore,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self._source_type = source_type
        self._runs_repository = runs_repository
        self._raw_events_repository = raw_events_repository
        self._object_store = object_store
        self._now_provider = now_provider or (lambda: datetime.now(UTC))

    def _run(self, records: list[_RawRecord]) -> IngestionRunResult:
        started_at = self._now_provider()
        run_id = self._runs_repository.start_run(
            source_type=self._source_type, started_at=started_at
        )

        fetched_count = len(records)
        inserted_count = 0
        duplicate_count = 0

        try:
            for record in records:
                observed_at = self._now_provider()
                object_key = _build_object_key(
                    source_type=self._source_type,
                    source_id=record.source_id,
                    observed_at=observed_at,
                    event_id=record.dedup_hash,
                )
                was_inserted = self._raw_events_repository.insert_raw_event(
                    source_type=self._source_type,
                    source_id=record.source_id,
                    observed_at=observed_at,
                    event_time=record.event_time,
                    dedup_hash=record.dedup_hash,
                    payload_preview=record.payload_preview,
                    object_key=object_key,
                    ingest_run_id=run_id,
                    status="pending",
                )

                if not was_inserted:
                    duplicate_count += 1
                    continue

                full_payload = {**record.payload, "observed_at": observed_at.isoformat()}
                try:
                    self._object_store.put_json_gzip(
                        key=object_key,
                        payload=full_payload,
                        metadata={
                            "source_type": self._source_type,
                            "source_id": record.source_id,
                            "dedup_hash": record.dedup_hash,
                            "observed_at": observed_at.isoformat(),
                        },
                    )
                except Exception as upload_exc:
                    try:
                        self._raw_events_repository.transition_raw_event_status(
                            source_type=self._source_type,
                            dedup_hash=record.dedup_hash,
                            status="failed",
                        )
                    except Exception as status_exc:
                        upload_exc.add_note(
                            "Failed to mark raw_event as failed "
                            f"for source_type={self._source_type} "
                            f"dedup_hash={record.dedup_hash}: {status_exc!r}"
                        )
                    raise

                self._raw_events_repository.transition_raw_event_status(
                    source_type=self._source_type,
                    dedup_hash=record.dedup_hash,
                    status="ok",
                )
                inserted_count += 1

        except Exception as exc:
            try:
                self._raw_events_repository.transition_pending_raw_events_for_run(
                    ingest_run_id=run_id,
                    status="failed",
                )
            except Exception as reconcile_exc:
                exc.add_note(
                    "Failed to reconcile pending raw_events "
                    f"for run_id={run_id}: {reconcile_exc!r}"
                )

            try:
                self._runs_repository.finish_run(
                    run_id,
                    finished_at=self._now_provider(),
                    status="error",
                    fetched_count=fetched_count,
                    inserted_count=inserted_count,
                    duplicate_count=duplicate_count,
                    error_summary=str(exc),
                )
            except Exception as finish_exc:
                # Keep the original ingest failure as the primary error.
                exc.add_note(
                    f"Run finalization also failed for run_id={run_id}: {finish_exc!r}"
                )
            raise

        self._runs_repository.finish_run(
            run_id,
            finished_at=self._now_provider(),
            status="ok",
            fetched_count=fetched_count,
            inserted_count=inserted_count,
            duplicate_count=duplicate_count,
            error_summary=None,
        )
        return IngestionRunResult(
            run_id=run_id,
            status="ok",
            fetched_count=fetched_count,
            inserted_count=inserted_count,
            duplicate_count=duplicate_count,
        )


class RSSRawIngestionPipeline(_RawIngestionPipeline):
    """Persist RSS events as append-only metadata rows plus full object payloads."""

    def __init__(
        self,
        *,
        runs_repository: IngestionRunsRepository,
        raw_events_repository: RawEventsRepository,
        object_store: RawObjectStore,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(
            source_type="rss",
            runs_repository=runs_repository,
            raw_events_repository=raw_events_repository,
            object_store=object_store,
            now_provider=now_provider,
        )

    def persist_articles(
        self, *, source_group: str, articles: list[FeedArticle]
    ) -> IngestionRunResult:
        return self._run([_rss_to_record(source_group, article) for article in articles])


class PricesRawIngestionPipeline(_RawIngestionPipeline):
    """Persist price quotes as append-only metadata rows plus full object payloads."""

    def __init__(
        self,
        *,
        runs_repository: IngestionRunsRepository,
        raw_events_repository: RawEventsRepository,
        object_store: RawObjectStore,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(
            source_type="prices",
            runs_repository=runs_repository,
            raw_events_repository=raw_events_repository,
            object_store=object_store,
            now_provider=now_provider,
        )

    def persist_quotes(self, *, quotes: list[PriceQuote]) -> IngestionRunResult:
        return self._run([_quote_to_record(quote) for quote in quotes])


def _rss_to_record(source_group: str, article: FeedArticle) -> _RawRecord:
    event_time = _parse_event_time(article.published)
    return _RawRecord(
        source_id=article.source,
        dedup_hash=article.dedup_hash,
        event_time=event_time,
        payload={
            "source_type": "rss",
            "source_group": source_group,
            "source_id": article.source,
            "title": article.title,
            "url": article.url,
            "published": article.published,
            "event_time": event_time.isoformat() if event_time else None,
            "dedup_hash": article.dedup_hash,
        },
        payload_preview={
            "title": article.title,
            "url": article.url,
            "published": article.published,
            "source_group": source_group,
        },
    )


def _quote_to_record(quote: PriceQuote) -> _RawRecord:
    event_time = datetime.fromtimestamp(quote.timestamp, tz=UTC) if quote.timestamp else None
    return _RawRecord(
        source_id=quote.symbol_id,
        dedup_hash=quote.dedup_hash,
        event_time=event_time,
        payload={
            "source_type": "prices",
            "symbol_id": quote.symbol_id,
            "asset_class": quote.asset_class,
            "current": quote.current,
            "open": quote.open,
            "high": quote.high,
            "low": quote.low,
            "prev_close": quote.prev_close,
            "timestamp": quote.timestamp,
            "dedup_hash": quote.dedup_hash,
        },
        payload_preview={
            "symbol_id": quote.symbol_id,
            "asset_class": quote.asset_class,
            "current": quote.current,
            "timestamp": quote.timestamp,
        },
    )


def _parse_event_time(value: str) -> datetime | None:
    cleaned = value.strip()
    if not cleaned:
        return None
    try:
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1] + "+00:00"
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _build_object_key(
    *, source_type: str, source_id: str, observed_at: datetime, event_id: str
) -> str:
    date_part = observed_at.strftime("%Y-%m-%d")
    hour_part = observed_at.strftime("%H")
    return (
        f"raw/source_type={source_type}/"
        f"date={date_part}/hour={hour_part}/"
        f"source_id={source_id}/{event_id}.json.gz"
    )
