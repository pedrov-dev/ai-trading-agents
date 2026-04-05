from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from info_scheduler import InfoScheduler


@dataclass(frozen=True)
class DetectedEvent:
    raw_event_id: str
    event_type: str
    rule_name: str
    confidence: float
    matched_text: str | None = None
    detected_at: datetime | None = None


class EventDetector(Protocol):
    """Pluggable interface for raw text event classification."""

    def detect(
        self,
        *,
        source_type: str,
        payload_preview: dict[str, Any],
    ) -> list[DetectedEvent]:
        ...


@dataclass(frozen=True)
class EventRule:
    name: str
    event_type: str
    keywords: tuple[str, ...]
    patterns: tuple[str, ...] = ()
    confidence: float = 1.0

    def matches(self, text: str) -> tuple[bool, str | None]:
        normalized = text.lower().strip()

        for keyword in self.keywords:
            if keyword in normalized:
                return True, keyword

        for pattern in self.patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return True, pattern

        return False, None


class RuleBasedEventDetector:
    """Rule-based event detection tuned for crypto news and market incidents."""

    def __init__(self, rules: list[EventRule] | None = None) -> None:
        self._rules = rules if rules is not None else self._default_rules()

    def detect(self, *, source_type: str, payload_preview: dict[str, Any]) -> list[DetectedEvent]:
        del source_type
        text = self._normalize_payload_text(payload_preview)
        if not text:
            return []

        detected: list[DetectedEvent] = []

        for rule in self._rules:
            matched, matched_text = rule.matches(text)
            if not matched:
                continue

            detected.append(
                DetectedEvent(
                    raw_event_id="",
                    event_type=rule.event_type,
                    rule_name=rule.name,
                    confidence=rule.confidence,
                    matched_text=matched_text,
                    detected_at=datetime.now(UTC),
                )
            )

        return detected

    @staticmethod
    def _normalize_payload_text(payload_preview: dict[str, Any]) -> str:
        texts: list[str] = []

        for key in ("title", "headline", "description", "text", "summary"):
            value = payload_preview.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())

        return " ".join(texts)

    @staticmethod
    def _default_rules() -> list[EventRule]:
        return [
            EventRule(
                name="etf_approval",
                event_type="ETF_APPROVAL",
                keywords=("etf approval", "approved the bitcoin etf", "approved the ether etf"),
                patterns=(r"\b(sec|regulator).*(approve|approval).*(bitcoin|ether|crypto).*(etf)\b",),
                confidence=0.95,
            ),
            EventRule(
                name="etf_delay",
                event_type="ETF_DELAY",
                keywords=("etf delay", "delays spot bitcoin etf", "postpones etf decision"),
                patterns=(r"\b(sec|regulator).*(delay|postpone|extend).*(bitcoin|ether|crypto).*(etf)\b",),
                confidence=0.9,
            ),
            EventRule(
                name="regulatory_action",
                event_type="REGULATORY_ACTION",
                keywords=("sec delays", "sec lawsuit", "cftc action", "regulatory action"),
                patterns=(r"\b(sec|cftc|doj|regulator).*(delay|approve|charge|sue|settlement|fine|penalty)\b",),
                confidence=0.85,
            ),
            EventRule(
                name="token_listing",
                event_type="TOKEN_LISTING",
                keywords=("lists on kraken", "token listing", "added trading support"),
                patterns=(r"\b(listed on|launches trading for|adds support for)\b",),
                confidence=0.8,
            ),
            EventRule(
                name="stablecoin_depeg",
                event_type="STABLECOIN_DEPEG",
                keywords=("depeg", "loses peg", "below $1"),
                patterns=(r"\b(usdt|usdc|dai|stablecoin).*(depeg|loses? peg|below \$?1)\b",),
                confidence=0.95,
            ),
            EventRule(
                name="security_incident",
                event_type="SECURITY_INCIDENT",
                keywords=("hack", "exploit", "drained", "private key leak"),
                patterns=(r"\b(exploit|breach|drain(?:ed)?|stolen funds?|validator compromise)\b",),
                confidence=0.95,
            ),
            EventRule(
                name="network_outage",
                event_type="NETWORK_OUTAGE",
                keywords=("network outage", "halted block production", "degraded performance"),
                patterns=(r"\b(outage|halt(?:ed)?|degraded|congestion)\b.*\b(network|chain|validator)\b",),
                confidence=0.8,
            ),
            EventRule(
                name="protocol_upgrade",
                event_type="PROTOCOL_UPGRADE",
                keywords=("mainnet upgrade", "hard fork", "testnet upgrade"),
                patterns=(r"\b(upgrade|hard fork|soft fork|mainnet launch)\b",),
                confidence=0.75,
            ),
            EventRule(
                name="macro_news",
                event_type="MACRO_NEWS",
                keywords=(
                    "fed rate cut",
                    "fed pause",
                    "inflation cooled",
                    "cpi cooled",
                    "jobs report",
                    "stimulus package",
                ),
                patterns=(
                    r"\b(fed|fomc|ecb|boj|cpi|inflation|jobs report|payrolls|rate cut|rate hike)\b",
                ),
                confidence=0.68,
            ),
            EventRule(
                name="whale_activity",
                event_type="WHALE_ACTIVITY",
                keywords=(
                    "whale accumulation",
                    "whale moved",
                    "large wallet",
                    "mega transfer",
                    "on-chain whale",
                ),
                patterns=(r"\b(whale|wallet|on-chain).*(bought|accumulated|moved|transferred|withdrew)\b",),
                confidence=0.78,
            ),
            EventRule(
                name="technical_breakout",
                event_type="TECHNICAL_BREAKOUT",
                keywords=(
                    "technical breakout",
                    "breaks resistance",
                    "breakout above",
                    "breakdown below",
                    "trendline breakout",
                ),
                patterns=(r"\b(breakout|breakdown|breaks resistance|breaks support|trendline)\b",),
                confidence=0.72,
            ),
        ]


def wire_event_detection_job(
    scheduler: InfoScheduler,
    event_detection_job: Callable[[], None],
    interval_seconds: int = 60,
) -> None:
    """Add periodic event detection job to the scheduler."""

    scheduler.register_event_detection_job(
        event_detection_job,
        interval_seconds=interval_seconds,
    )
