"""Placeholder for future LLM-backed signal generation."""

from __future__ import annotations

from dataclasses import dataclass

from detection.event_detection import DetectedEvent
from ingestion.prices_ingestion import PriceQuote


@dataclass(frozen=True)
class LLMSignalResult:
    """Neutral result returned until the LLM generator is implemented."""

    enabled: bool = False
    confidence_delta: float = 0.0
    rationale: tuple[str, ...] = (
        "LLM signal generator is not enabled yet.",
    )


def generate_llm_signal(
    *,
    event: DetectedEvent,
    quote: PriceQuote,
) -> LLMSignalResult:
    """Return a no-op result for the future LLM signal pathway."""

    del event, quote
    return LLMSignalResult()


__all__ = ["LLMSignalResult", "generate_llm_signal"]
