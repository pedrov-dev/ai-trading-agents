"""Post-trade review and bounded heuristic-refinement helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from agent.signals import RejectedTradeCandidate, TradeSide
from agent.strategy import StrategyConfig
from ingestion.prices_ingestion import PriceQuote
from monitoring.calibration import CalibrationSummary
from monitoring.trade_journal import TradeJournalEntry

TimingLabel = Literal["too_early", "on_time", "too_late"]
AssetSelectionLabel = Literal["correct_asset", "wrong_asset", "unknown"]


@dataclass(frozen=True)
class PostTradeReview:
    """Structured hindsight review for one closed trade."""

    symbol_id: str
    reviewed_at: datetime
    timing_label: TimingLabel
    asset_selection_label: AssetSelectionLabel
    realized_return_fraction: float | None = None
    best_alternative_symbol: str | None = None
    best_alternative_return_fraction: float | None = None
    reason_codes: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol_id": self.symbol_id,
            "reviewed_at": self.reviewed_at.isoformat(),
            "timing_label": self.timing_label,
            "asset_selection_label": self.asset_selection_label,
            "realized_return_fraction": self.realized_return_fraction,
            "best_alternative_symbol": self.best_alternative_symbol,
            "best_alternative_return_fraction": self.best_alternative_return_fraction,
            "reason_codes": list(self.reason_codes),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PostTradeReview:
        return cls(
            symbol_id=str(payload.get("symbol_id", "unknown_symbol")),
            reviewed_at=datetime.fromisoformat(
                str(payload.get("reviewed_at", datetime.now(UTC).isoformat()))
            ),
            timing_label=cast(
                TimingLabel,
                str(payload.get("timing_label", "on_time")),
            ),
            asset_selection_label=cast(
                AssetSelectionLabel,
                str(payload.get("asset_selection_label", "unknown")),
            ),
            realized_return_fraction=(
                float(payload["realized_return_fraction"])
                if payload.get("realized_return_fraction") is not None
                else None
            ),
            best_alternative_symbol=(
                str(payload["best_alternative_symbol"])
                if payload.get("best_alternative_symbol")
                else None
            ),
            best_alternative_return_fraction=(
                float(payload["best_alternative_return_fraction"])
                if payload.get("best_alternative_return_fraction") is not None
                else None
            ),
            reason_codes=tuple(str(item) for item in payload.get("reason_codes", ())),
            notes=tuple(str(item) for item in payload.get("notes", ())),
        )


@dataclass(frozen=True)
class HeuristicAdjustment:
    """One bounded strategy-config change applied by the learning loop."""

    field_name: str
    old_value: float
    new_value: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class HeuristicLearningState:
    """Persisted heuristic state so the next run can resume from the latest config."""

    heuristic_version: str
    current_config: StrategyConfig
    last_post_trade_reviews: tuple[PostTradeReview, ...] = ()
    last_applied_adjustments: tuple[HeuristicAdjustment, ...] = ()
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "heuristic_version": self.heuristic_version,
            "current_config": asdict(self.current_config),
            "last_post_trade_reviews": [
                review.to_dict() for review in self.last_post_trade_reviews
            ],
            "last_applied_adjustments": [
                adjustment.to_dict() for adjustment in self.last_applied_adjustments
            ],
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HeuristicLearningState:
        allowed_fields = {item.name for item in fields(StrategyConfig)}
        raw_config = payload.get("current_config", {})
        config_map = {
            key: value
            for key, value in dict(raw_config).items()
            if key in allowed_fields
        }
        return cls(
            heuristic_version=str(payload.get("heuristic_version", "strategy-v1")),
            current_config=StrategyConfig(**config_map),
            last_post_trade_reviews=tuple(
                PostTradeReview.from_dict(item)
                for item in payload.get("last_post_trade_reviews", ())
                if isinstance(item, dict)
            ),
            last_applied_adjustments=tuple(
                HeuristicAdjustment(
                    field_name=str(item.get("field_name", "unknown")),
                    old_value=float(item.get("old_value", 0.0)),
                    new_value=float(item.get("new_value", 0.0)),
                    reason=str(item.get("reason", "")),
                )
                for item in payload.get("last_applied_adjustments", ())
                if isinstance(item, dict)
            ),
            updated_at=datetime.fromisoformat(
                str(payload.get("updated_at", datetime.now(UTC).isoformat()))
            ),
        )


def evaluate_post_trade_review(
    *,
    journal_entry: TradeJournalEntry,
    opened_at: datetime | None = None,
    latest_quotes: tuple[PriceQuote, ...] | list[PriceQuote] = (),
    rejected_alternatives: tuple[RejectedTradeCandidate, ...]
    | list[RejectedTradeCandidate] = (),
    alternative_outperformance_gap: float = 0.015,
) -> PostTradeReview:
    """Classify a closed trade as early/late/right-asset using simple bounded proxies."""
    realized_return = journal_entry.realized_return_fraction or 0.0
    reason_codes: list[str] = []
    notes: list[str] = []

    elapsed_minutes: float | None = None
    if opened_at is not None:
        elapsed_minutes = max(
            (journal_entry.recorded_at - opened_at).total_seconds() / 60.0,
            0.0,
        )

    hold_ratio: float | None = None
    if elapsed_minutes is not None and journal_entry.max_hold_minutes:
        hold_ratio = elapsed_minutes / max(journal_entry.max_hold_minutes, 1)

    timing_label: TimingLabel = "on_time"
    if hold_ratio is not None and realized_return < 0 and hold_ratio < 0.35:
        timing_label = "too_early"
        reason_codes.append("early_reversal")
        notes.append(
            "Trade reversed after only "
            f"{elapsed_minutes:.1f} minutes; require more confirmation "
            "before entry."
        )
    elif hold_ratio is not None and realized_return <= 0 and hold_ratio >= 0.75:
        timing_label = "too_late"
        reason_codes.append("stale_follow_through")
        notes.append(
            "Setup needed "
            f"{elapsed_minutes:.1f} minutes and still faded; consider "
            "entering earlier or skipping stale signals."
        )
    else:
        notes.append("Trade timing stayed within the expected execution window.")

    asset_selection_label: AssetSelectionLabel = (
        "unknown" if not rejected_alternatives else "correct_asset"
    )
    best_alternative_symbol: str | None = None
    best_alternative_return: float | None = None
    quote_by_symbol = {quote.symbol_id: quote for quote in latest_quotes}

    for alternative in rejected_alternatives:
        quote = quote_by_symbol.get(alternative.symbol_id)
        if quote is None:
            continue
        alternative_return = _realized_return_fraction(
            side=alternative.side,
            entry_price=alternative.reference_price,
            exit_price=quote.current,
        )
        if best_alternative_return is None or alternative_return > best_alternative_return:
            best_alternative_symbol = alternative.symbol_id
            best_alternative_return = alternative_return

    if (
        best_alternative_symbol is not None
        and best_alternative_return is not None
        and best_alternative_return > realized_return + alternative_outperformance_gap
    ):
        asset_selection_label = "wrong_asset"
        reason_codes.append("better_rejected_alternative")
        notes.append(
            "Rejected alternative "
            f"{best_alternative_symbol} outperformed the chosen trade by "
            f"{(best_alternative_return - realized_return) * 100:.2f}% over the same window."
        )
    elif rejected_alternatives:
        notes.append("No rejected alternative materially outperformed the chosen trade.")

    return PostTradeReview(
        symbol_id=journal_entry.symbol_id,
        reviewed_at=journal_entry.recorded_at,
        timing_label=timing_label,
        asset_selection_label=asset_selection_label,
        realized_return_fraction=round(realized_return, 6),
        best_alternative_symbol=best_alternative_symbol,
        best_alternative_return_fraction=(
            round(best_alternative_return, 6)
            if best_alternative_return is not None
            else None
        ),
        reason_codes=tuple(reason_codes),
        notes=tuple(notes),
    )


def refine_strategy_config(
    *,
    current_config: StrategyConfig,
    calibration_summary: CalibrationSummary,
    timing_labels: tuple[str, ...] | list[str],
    asset_selection_labels: tuple[str, ...] | list[str],
    max_threshold_step: float = 0.02,
    max_weight_step: float = 0.05,
) -> tuple[StrategyConfig, tuple[HeuristicAdjustment, ...]]:
    """Convert recent post-trade labels into bounded StrategyConfig updates."""
    pending_updates: dict[str, float] = {}
    reasons: dict[str, list[str]] = {}

    def current_value(field_name: str) -> float:
        raw_value = pending_updates.get(field_name)
        if raw_value is not None:
            return float(raw_value)
        config_value = getattr(current_config, field_name)
        if isinstance(config_value, bool):
            return 1.0 if config_value else 0.0
        if isinstance(config_value, (int, float)):
            return float(config_value)
        return 0.0

    def set_value(
        field_name: str,
        proposed: float,
        *,
        minimum: float = 0.0,
        maximum: float = 1.0,
        reason: str,
    ) -> None:
        clamped = round(min(max(proposed, minimum), maximum), 4)
        if round(current_value(field_name), 4) == clamped:
            return
        pending_updates[field_name] = clamped
        reasons.setdefault(field_name, []).append(reason)

    early_count = sum(1 for item in timing_labels if item == "too_early")
    late_count = sum(1 for item in timing_labels if item == "too_late")
    timing_delta = min(abs(early_count - late_count) * 0.01, max_threshold_step)
    if early_count > late_count and timing_delta > 0:
        set_value(
            "min_confidence_score",
            current_value("min_confidence_score") + timing_delta,
            minimum=0.5,
            maximum=0.95,
            reason="Recent trades were too early, so entry confidence is being tightened.",
        )
        set_value(
            "entry_confirmation_threshold",
            current_value("entry_confirmation_threshold") + timing_delta,
            minimum=0.4,
            maximum=0.95,
            reason=(
                "Recent trades were too early, so weighted confirmation "
                "now requires more proof."
            ),
        )
        set_value(
            "reduced_size_confirmation_threshold",
            current_value("reduced_size_confirmation_threshold") + min(timing_delta, 0.01),
            minimum=0.35,
            maximum=0.9,
            reason="Reduced-size mode was tightened after early reversals.",
        )
    elif late_count > early_count and timing_delta > 0:
        set_value(
            "min_confidence_score",
            current_value("min_confidence_score") - timing_delta,
            minimum=0.5,
            maximum=0.95,
            reason=(
                "Recent trades were too late, so the confidence floor is "
                "being loosened slightly."
            ),
        )
        set_value(
            "entry_confirmation_threshold",
            current_value("entry_confirmation_threshold") - timing_delta,
            minimum=0.4,
            maximum=0.95,
            reason="Recent trades were too late, so the strategy can enter slightly sooner.",
        )
        set_value(
            "reduced_size_confirmation_threshold",
            current_value("reduced_size_confirmation_threshold") - min(timing_delta, 0.01),
            minimum=0.35,
            maximum=0.9,
            reason="Reduced-size mode was loosened after consistently late entries.",
        )

    if calibration_summary.resolved_prediction_count >= 10:
        hit_rate_gap = current_value("min_confidence_score") - calibration_summary.hit_rate
        if hit_rate_gap > 0.12:
            set_value(
                "min_confidence_score",
                current_value("min_confidence_score") + 0.01,
                minimum=0.5,
                maximum=0.95,
                reason="Low hit rate confirmed the need for stricter confidence gating.",
            )
        elif hit_rate_gap < -0.1:
            set_value(
                "min_confidence_score",
                current_value("min_confidence_score") - 0.01,
                minimum=0.5,
                maximum=0.95,
                reason=(
                    "The calibration summary shows the strategy is outperforming "
                    "the current confidence floor."
                ),
            )

    wrong_asset_count = sum(1 for item in asset_selection_labels if item == "wrong_asset")
    if wrong_asset_count > 0:
        weight_delta = min(0.02 * wrong_asset_count, max_weight_step)
        set_value(
            "diversification_weight",
            current_value("diversification_weight") + weight_delta,
            minimum=0.05,
            maximum=0.6,
            reason="Wrong-asset reviews increased the emphasis on diversification during ranking.",
        )
        set_value(
            "confidence_weight",
            current_value("confidence_weight") - min(weight_delta / 2, max_weight_step),
            minimum=0.05,
            maximum=0.6,
            reason="Wrong-asset reviews reduced the dominance of pure confidence ranking.",
        )
        set_value(
            "novelty_weight",
            current_value("novelty_weight") + min(weight_delta / 2, max_weight_step),
            minimum=0.05,
            maximum=0.5,
            reason="Wrong-asset reviews increased the value of fresh, differentiated setups.",
        )

    if pending_updates:
        entry_threshold = pending_updates.get(
            "entry_confirmation_threshold",
            current_config.entry_confirmation_threshold,
        )
        reduced_threshold = pending_updates.get(
            "reduced_size_confirmation_threshold",
            current_config.reduced_size_confirmation_threshold,
        )
        if reduced_threshold > entry_threshold:
            pending_updates["reduced_size_confirmation_threshold"] = round(
                entry_threshold,
                4,
            )
            reasons.setdefault("reduced_size_confirmation_threshold", []).append(
                "Reduced-size confirmation cannot exceed the strong-entry threshold."
            )

    updated_config = (
        replace(current_config, **cast(dict[str, Any], pending_updates))
        if pending_updates
        else current_config
    )
    adjustments = tuple(
        HeuristicAdjustment(
            field_name=field_name,
            old_value=float(getattr(current_config, field_name)),
            new_value=float(value),
            reason=" ".join(reasons.get(field_name, ())),
        )
        for field_name, value in sorted(pending_updates.items())
    )
    return updated_config, adjustments


def load_learning_state(path: str | Path) -> HeuristicLearningState | None:
    state_path = Path(path)
    if not state_path.exists():
        return None
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return HeuristicLearningState.from_dict(payload)


def persist_learning_state(path: str | Path, state: HeuristicLearningState) -> None:
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = state_path.with_suffix(f"{state_path.suffix}.tmp")
    temp_path.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(state_path)


def _realized_return_fraction(*, side: TradeSide, entry_price: float, exit_price: float) -> float:
    if entry_price <= 0:
        return 0.0
    if side == "sell":
        return (entry_price - exit_price) / entry_price
    return (exit_price - entry_price) / entry_price
