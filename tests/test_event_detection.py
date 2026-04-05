from detection.event_detection import RuleBasedEventDetector
from detection.event_types import event_performance_group


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
