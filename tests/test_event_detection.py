from detection.event_detection import RuleBasedEventDetector


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
