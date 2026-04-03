from unittest.mock import MagicMock

from info_scheduler import InfoScheduler


def test_scheduler_registers_named_jobs() -> None:
    scheduler = InfoScheduler()
    job = MagicMock()

    scheduler.register_rss_job(job, interval_seconds=60)
    scheduler.register_prices_job(job, interval_seconds=60)
    scheduler.register_event_detection_job(job, interval_seconds=60)

    jobs = {scheduled_job.id for scheduled_job in scheduler._scheduler.get_jobs()}

    assert {"rss_ingestion", "prices_ingestion", "event_detection"}.issubset(jobs)

    scheduler.shutdown()
