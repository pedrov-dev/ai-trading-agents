"""Lightweight APScheduler wrapper for recurring crypto-ingestion jobs."""

from __future__ import annotations

from collections.abc import Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.base import STATE_STOPPED


class InfoScheduler:
    """Thin wrapper around APScheduler with stable job identifiers."""

    def __init__(self) -> None:
        self._scheduler = BackgroundScheduler()
        self._scheduler.start(paused=True)

    def _register_interval_job(
        self,
        func: Callable[[], None],
        interval_seconds: int,
        *,
        job_id: str,
    ) -> None:
        self._scheduler.add_job(
            func,
            "interval",
            seconds=interval_seconds,
            id=job_id,
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )

    def register_rss_job(self, func: Callable[[], None], *, interval_seconds: int) -> None:
        self._register_interval_job(func, interval_seconds, job_id="rss_ingestion")

    def register_prices_job(self, func: Callable[[], None], *, interval_seconds: int) -> None:
        self._register_interval_job(func, interval_seconds, job_id="prices_ingestion")

    def register_secondary_prices_job(
        self,
        func: Callable[[], None],
        *,
        interval_seconds: int,
    ) -> None:
        self._register_interval_job(
            func,
            interval_seconds,
            job_id="secondary_prices_ingestion",
        )

    def register_event_detection_job(
        self,
        func: Callable[[], None],
        *,
        interval_seconds: int,
    ) -> None:
        self._register_interval_job(func, interval_seconds, job_id="event_detection")

    def register_execution_job(self, func: Callable[[], None], *, interval_seconds: int) -> None:
        self._register_interval_job(func, interval_seconds, job_id="trade_execution")

    def start(self) -> None:
        self._scheduler.resume()

    def shutdown(self, *, wait: bool = True) -> None:
        if self._scheduler.state != STATE_STOPPED:
            self._scheduler.shutdown(wait=wait)
