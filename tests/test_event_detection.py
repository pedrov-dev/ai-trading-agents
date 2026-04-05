from datetime import UTC, datetime

from detection.event_detection import RuleBasedEventDetector
from detection.event_detection_service import EventDetectionService
from detection.event_types import event_performance_group
from tests.storage_fakes import StubEventDetectionRepository, StubRawEventsRepository


def test_rule_based_detector_matches_crypto_specific_events() -> None:
    detector = RuleBasedEventDetector()

    matches = detector.detect(
        source_type="rss",
        payload_preview={
            "title": "SEC delays spot Bitcoin ETF decision after exchange filing",
            "summary": "Regulators requested more time to review the crypto ETF proposal.",
        },
    )

    event_types = {match.event_type for match in matches}

    assert "ETF_DELAY" in event_types
    assert "REGULATORY_ACTION" in event_types
    assert "EARNINGS_RELEASE" not in event_types


def test_rule_based_detector_matches_new_event_tracking_categories() -> None:
    detector = RuleBasedEventDetector()

    macro_matches = detector.detect(
        source_type="rss",
        payload_preview={
            "title": "Fed rate cut hopes rise after CPI cooled more than expected",
            "summary": "Macro traders rotated back into crypto after the inflation report.",
        },
    )
    whale_matches = detector.detect(
        source_type="rss",
        payload_preview={
            "title": "Whale accumulation spotted as large wallet bought more BTC",
            "summary": "On-chain trackers flagged a mega transfer into cold storage.",
        },
    )
    breakout_matches = detector.detect(
        source_type="rss",
        payload_preview={
            "title": "Bitcoin technical breakout above resistance triggers trend followers",
            "summary": "Analysts cited a clean breakout above the prior range high.",
        },
    )

    assert {match.event_type for match in macro_matches} >= {"MACRO_NEWS"}
    assert {match.event_type for match in whale_matches} >= {"WHALE_ACTIVITY"}
    assert {match.event_type for match in breakout_matches} >= {"TECHNICAL_BREAKOUT"}


def test_event_performance_group_maps_requested_buckets() -> None:
    assert event_performance_group("ETF_APPROVAL") == "etf_news"
    assert event_performance_group("REGULATORY_ACTION") == "regulatory_news"
    assert event_performance_group("SECURITY_INCIDENT") == "exchange_hacks"
    assert event_performance_group("MACRO_NEWS") == "macro_news"
    assert event_performance_group("WHALE_ACTIVITY") == "whale_activity"
    assert event_performance_group("TECHNICAL_BREAKOUT") == "technical_breakouts"
    assert event_performance_group("TOKEN_LISTING") == "token_listing"


def test_event_detection_service_scores_first_occurrence_higher_than_repeated_narrative() -> None:
    raw_events = StubRawEventsRepository()
    detected_events = StubEventDetectionRepository()
    service = EventDetectionService(
        detector=RuleBasedEventDetector(),
        raw_events_repository=raw_events,
        event_detection_repository=detected_events,
    )

    observed_at = datetime(2026, 4, 4, 12, 0, tzinfo=UTC)
    raw_events.insert_raw_event(
        source_type="rss",
        source_id="story-1",
        observed_at=observed_at,
        event_time=observed_at,
        dedup_hash="hash-1",
        payload_preview={
            "title": "ETF approval rumor lifts bitcoin sentiment",
            "summary": "Analysts say the bitcoin ETF approval narrative is back in focus.",
        },
        object_key="raw/story-1.json.gz",
        ingest_run_id="run-1",
        status="ok",
    )
    service.classify_pending_events(source_type="rss")

    raw_events.insert_raw_event(
        source_type="rss",
        source_id="story-2",
        observed_at=observed_at.replace(hour=13),
        event_time=observed_at.replace(hour=13),
        dedup_hash="hash-2",
        payload_preview={
            "title": "Bitcoin ETF approval rumor resurfaces after another filing",
            "summary": "The same ETF approval narrative appeared again across crypto desks.",
        },
        object_key="raw/story-2.json.gz",
        ingest_run_id="run-2",
        status="ok",
    )
    service.classify_pending_events(source_type="rss")

    etf_events = [
        event
        for event in detected_events.list_detected_events()
        if event.event_type == "ETF_APPROVAL"
    ]

    assert len(etf_events) == 2
    assert etf_events[0].novelty_score is not None
    assert etf_events[0].novelty_score >= 0.95
    assert etf_events[0].repeat_count == 0
    assert etf_events[0].narrative_key is not None
    assert etf_events[1].novelty_score is not None
    assert etf_events[1].novelty_score < etf_events[0].novelty_score
    assert etf_events[1].repeat_count >= 1
    assert etf_events[1].narrative_key == etf_events[0].narrative_key
