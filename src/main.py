"""Kraken-backed trading entrypoint with paper/live modes and optional ERC-8004 wiring."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import threading
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from agent.news_signal import select_quote_for_event
from agent.portfolio import LocalPortfolioStateProvider, PortfolioSnapshot
from agent.risk import RiskConfig
from agent.signals import NoTradeDecision, TradeIntent
from agent.strategy import SimpleEventDrivenStrategy, StrategyConfig
from detection.event_detection import DetectedEvent, RuleBasedEventDetector
from detection.event_detection_postgres import (
    EventDetectionRepository,
    PostgresEventDetectionRepository,
)
from detection.event_detection_service import EventDetectionService
from execution.kraken_cli import (
    DEFAULT_AUDIT_LOG_PATH,
    CommandRunner,
    KrakenCLIConfig,
    KrakenCLIExecutor,
)
from execution.orders import ExecutionResult
from identity.erc8004_registry import (
    AgentIdentity,
    HackathonVaultClient,
    IdentityRegistry,
    LocalERC8004Registry,
    OnChainERC8004Registry,
    OnChainTransactionResult,
    ReputationRegistryClient,
    RiskRouterClient,
    RiskRouterIntent,
    SepoliaContractsConfig,
    ValidationRegistryClient,
)
from identity.reputation import ReputationEngine, ReputationSnapshot
from info_scheduler import InfoScheduler
from ingestion.prices_config import (
    PRICE_SYMBOLS,
    SECONDARY_POLLING_FREQUENCY_SECONDS,
    SECONDARY_PRICE_SYMBOLS,
    PriceSymbol,
)
from ingestion.prices_ingestion import PriceQuote, PricesIngestionService
from ingestion.rss_config import RSS_FEED_GROUPS, FeedSource
from ingestion.rss_ingestion import RSSIngestionService
from monitoring.audit_log import AuditSummary, build_audit_summary_from_file
from monitoring.calibration import CalibrationSummary, build_calibration_summary
from monitoring.drawdown import DrawdownSnapshot, EquityPoint, build_drawdown_snapshot
from monitoring.learning import (
    HeuristicLearningState,
    PostTradeReview,
    evaluate_post_trade_review,
    load_learning_state,
    persist_learning_state,
    refine_strategy_config,
)
from monitoring.pnl import PnLSnapshot, build_pnl_snapshot
from monitoring.trade_journal import (
    LocalTradeJournal,
    TradeJournalEntry,
    TradeJournalSummary,
)
from storage.local_runtime import (
    JsonlFileStore,
    LocalEventDetectionRepository,
    LocalFileObjectStore,
    LocalInMemoryIngestionRunsRepository,
    LocalInMemoryRawEventsRepository,
)
from storage.object_storage import ObjectStorageConfig, S3CompatibleObjectStore
from storage.raw_ingestion import (
    IngestionRunResult,
    IngestionRunsRepository,
    PricesRawIngestionPipeline,
    RawEventsRepository,
    RawObjectStore,
    RSSRawIngestionPipeline,
)
from storage.raw_postgres import (
    PostgresIngestionRunsRepository,
    PostgresRawEventsRepository,
    postgres_connection_factory_from_env,
    probe_postgres_connection,
)
from validation.artifacts import ArtifactKind, ValidationArtifact
from validation.checkpoints import ValidationCheckpoint, build_checkpoints

ROOT_DIR = Path(__file__).resolve().parents[1]
RUNTIME_ENV_FILE_NAMES = (".env", ".env.params", ".env.secrets", ".runtime.env")


@dataclass(frozen=True)
class RuntimePaths:
    """Filesystem locations used by the Kraken runtime flow."""

    base_dir: Path
    artifacts_dir: Path
    raw_payload_dir: Path
    audit_log_path: Path
    journal_log_path: Path
    artifacts_log_path: Path
    checkpoints_log_path: Path
    activity_log_path: Path
    summary_path: Path
    heuristic_state_path: Path
    heuristic_refinements_path: Path

    @classmethod
    def from_base_dir(cls, base_dir: str | Path) -> RuntimePaths:
        resolved = Path(base_dir)
        artifacts_dir = resolved / "artifacts"
        return cls(
            base_dir=resolved,
            artifacts_dir=artifacts_dir,
            raw_payload_dir=artifacts_dir / "raw_payloads",
            audit_log_path=artifacts_dir / DEFAULT_AUDIT_LOG_PATH.name,
            journal_log_path=artifacts_dir / "trading_journal.jsonl",
            artifacts_log_path=artifacts_dir / "validation_artifacts.jsonl",
            checkpoints_log_path=artifacts_dir / "validation_checkpoints.jsonl",
            activity_log_path=artifacts_dir / "activity_log.jsonl",
            summary_path=artifacts_dir / "run_summary.json",
            heuristic_state_path=artifacts_dir / "heuristic_state.json",
            heuristic_refinements_path=artifacts_dir / "heuristic_refinements.jsonl",
        )


@dataclass(frozen=True)
class TierPromotionConfig:
    """Thresholds for temporarily promoting Tier B assets into the core poll loop."""

    volume_ratio_threshold: float = 2.0
    volatility_filter_threshold: float = 1.75
    news_score_threshold: float = 0.85
    correlation_break_threshold: float = 0.05
    promotion_duration_hours: int = 24

    @property
    def promotion_duration(self) -> timedelta:
        return timedelta(hours=max(self.promotion_duration_hours, 1))

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> TierPromotionConfig:
        resolved_env = dict(env or {})
        return cls(
            volume_ratio_threshold=float(
                resolved_env.get("TIER_PROMOTION_VOLUME_RATIO_THRESHOLD", "2.0")
            ),
            volatility_filter_threshold=float(
                resolved_env.get("TIER_PROMOTION_VOLATILITY_THRESHOLD", "1.75")
            ),
            news_score_threshold=float(
                resolved_env.get("TIER_PROMOTION_NEWS_SCORE_THRESHOLD", "0.85")
            ),
            correlation_break_threshold=float(
                resolved_env.get("TIER_PROMOTION_CORRELATION_BREAK_THRESHOLD", "0.05")
            ),
            promotion_duration_hours=int(
                resolved_env.get("TIER_PROMOTION_DURATION_HOURS", "24")
            ),
        )


@dataclass(frozen=True)
class RuntimeCycleResult:
    """Outcome of one Kraken runtime cycle across ingestion, detection, and execution."""

    rss_result: IngestionRunResult
    prices_result: IngestionRunResult
    classification_count: int
    detected_events: tuple[DetectedEvent, ...]
    trade_intents: tuple[TradeIntent, ...]
    no_trade_decisions: tuple[NoTradeDecision, ...]
    execution_results: tuple[ExecutionResult, ...]
    artifacts: tuple[ValidationArtifact, ...]
    checkpoints: tuple[ValidationCheckpoint, ...]
    portfolio: PortfolioSnapshot
    pnl_snapshot: PnLSnapshot
    drawdown_snapshot: DrawdownSnapshot
    audit_summary: AuditSummary
    journal_summary: TradeJournalSummary
    calibration_summary: CalibrationSummary
    reputation: ReputationSnapshot

    @property
    def artifact_count(self) -> int:
        return len(self.artifacts)

    @property
    def checkpoint_count(self) -> int:
        return len(self.checkpoints)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rss_result": {
                "run_id": self.rss_result.run_id,
                "status": self.rss_result.status,
                "fetched_count": self.rss_result.fetched_count,
                "inserted_count": self.rss_result.inserted_count,
                "duplicate_count": self.rss_result.duplicate_count,
            },
            "prices_result": {
                "run_id": self.prices_result.run_id,
                "status": self.prices_result.status,
                "fetched_count": self.prices_result.fetched_count,
                "inserted_count": self.prices_result.inserted_count,
                "duplicate_count": self.prices_result.duplicate_count,
            },
            "classification_count": self.classification_count,
            "detected_events": [
                {
                    "raw_event_id": event.raw_event_id,
                    "event_type": event.event_type,
                    "rule_name": event.rule_name,
                    "confidence": event.confidence,
                    "matched_text": event.matched_text,
                    "novelty_score": event.novelty_score,
                    "repeat_count": event.repeat_count,
                    "narrative_key": event.narrative_key,
                    "detected_at": event.detected_at.isoformat()
                    if event.detected_at
                    else None,
                }
                for event in self.detected_events
            ],
            "trade_intents": [
                {
                    "symbol_id": intent.symbol_id,
                    "side": intent.side,
                    "notional_usd": intent.notional_usd,
                    "quantity": intent.quantity,
                    "score": intent.score,
                    "confidence": intent.confidence,
                    "confidence_score": intent.confidence_score,
                    "expected_move": intent.expected_move,
                    "expected_move_fraction": intent.expected_move_fraction,
                    "stop_distance_fraction": intent.stop_distance_fraction,
                    "risk_reward_ratio": intent.risk_reward_ratio,
                    "generated_at": intent.generated_at.isoformat(),
                    "signal_id": intent.signal_family or intent.signal_id,
                    "signal_version": intent.signal_version,
                    "heuristic_version": intent.heuristic_version,
                    "asset": intent.asset,
                    "direction": intent.direction,
                    "event_type": intent.event_type,
                    "event_group": intent.event_group,
                    "exit_horizon_label": intent.exit_horizon_label,
                    "position_id": intent.position_id,
                    "selection_rank": intent.selection_rank,
                    "selection_composite_score": intent.selection_composite_score,
                    "audit_metadata": {
                        key: value
                        for key, value in {
                            "signal_instance_id": intent.signal_id,
                            "signal_family": intent.signal_family,
                            "model_version": intent.model_version,
                            "feature_set": intent.feature_set,
                            "raw_event_id": intent.raw_event_id,
                        }.items()
                        if value is not None
                    },
                    "rationale": list(intent.rationale),
                }
                for intent in self.trade_intents
            ],
            "no_trade_decisions": [
                {
                    "symbol_id": decision.symbol_id,
                    "reason_code": decision.reason_code,
                    "reason": decision.reason,
                    "confidence": decision.confidence,
                    "confidence_score": decision.confidence_score,
                    "threshold": decision.threshold,
                    "score": decision.score,
                    "event_type": decision.event_type,
                    "event_group": decision.event_group,
                    "signal_id": decision.signal_family or decision.signal_id,
                    "signal_version": decision.signal_version,
                    "asset": decision.asset,
                    "direction": decision.direction,
                    "detected_at": decision.detected_at.isoformat(),
                    "audit_metadata": {
                        key: value
                        for key, value in {
                            "signal_instance_id": decision.signal_id,
                            "signal_family": decision.signal_family,
                            "model_version": decision.model_version,
                            "feature_set": decision.feature_set,
                            "raw_event_id": decision.raw_event_id,
                        }.items()
                        if value is not None
                    },
                    "rationale": list(decision.rationale),
                }
                for decision in self.no_trade_decisions
            ],
            "execution_results": [result.to_dict() for result in self.execution_results],
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "checkpoints": [checkpoint.to_dict() for checkpoint in self.checkpoints],
            "signal_discovery": _build_signal_discovery_summary(self.artifacts),
            "portfolio": {
                "total_equity": self.portfolio.total_equity,
                "cash_usd": self.portfolio.cash_usd,
                "open_position_count": self.portfolio.open_position_count(),
                "realized_pnl_today": self.portfolio.realized_pnl_today,
                "as_of": self.portfolio.as_of.isoformat(),
            },
            "pnl_snapshot": self.pnl_snapshot.to_dict(),
            "drawdown_snapshot": self.drawdown_snapshot.to_dict(),
            "audit_summary": self.audit_summary.to_dict(),
            "journal_summary": self.journal_summary.to_dict(),
            "calibration_summary": self.calibration_summary.to_dict(),
            "reputation": self.reputation.to_dict(),
        }

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "status": "completed",
            "updated_at": self.portfolio.as_of.isoformat(),
            "rss_result": {
                "run_id": self.rss_result.run_id,
                "status": self.rss_result.status,
                "fetched_count": self.rss_result.fetched_count,
                "inserted_count": self.rss_result.inserted_count,
                "duplicate_count": self.rss_result.duplicate_count,
            },
            "prices_result": {
                "run_id": self.prices_result.run_id,
                "status": self.prices_result.status,
                "fetched_count": self.prices_result.fetched_count,
                "inserted_count": self.prices_result.inserted_count,
                "duplicate_count": self.prices_result.duplicate_count,
            },
            "classification_count": self.classification_count,
            "counts": {
                "detected_events": len(self.detected_events),
                "trade_intents": len(self.trade_intents),
                "no_trade_decisions": len(self.no_trade_decisions),
                "executions": len(self.execution_results),
                "artifacts": self.artifact_count,
                "checkpoints": self.checkpoint_count,
            },
            "detected_events": [
                {
                    "raw_event_id": event.raw_event_id,
                    "event_type": event.event_type,
                    "rule_name": event.rule_name,
                    "confidence": event.confidence,
                    "novelty_score": event.novelty_score,
                    "repeat_count": event.repeat_count,
                    "narrative_key": event.narrative_key,
                    "detected_at": event.detected_at.isoformat()
                    if event.detected_at
                    else None,
                }
                for event in self.detected_events[-10:]
            ],
            "trade_intents": [
                {
                    "symbol_id": intent.symbol_id,
                    "side": intent.side,
                    "notional_usd": round(intent.notional_usd, 2),
                    "quantity": round(intent.quantity, 8),
                    "score": intent.score,
                    "confidence": intent.confidence,
                    "confidence_score": intent.confidence_score,
                    "expected_move": intent.expected_move,
                    "expected_move_fraction": intent.expected_move_fraction,
                    "stop_distance_fraction": intent.stop_distance_fraction,
                    "risk_reward_ratio": intent.risk_reward_ratio,
                    "generated_at": intent.generated_at.isoformat(),
                    "signal_id": intent.signal_family or intent.signal_id,
                    "signal_version": intent.signal_version,
                    "heuristic_version": intent.heuristic_version,
                    "asset": intent.asset,
                    "direction": intent.direction,
                    "event_type": intent.event_type,
                    "event_group": intent.event_group,
                    "exit_horizon_label": intent.exit_horizon_label,
                    "position_id": intent.position_id,
                    "selection_rank": intent.selection_rank,
                    "selection_composite_score": intent.selection_composite_score,
                    "audit_metadata": {
                        key: value
                        for key, value in {
                            "signal_instance_id": intent.signal_id,
                            "signal_family": intent.signal_family,
                            "model_version": intent.model_version,
                            "feature_set": intent.feature_set,
                            "raw_event_id": intent.raw_event_id,
                        }.items()
                        if value is not None
                    },
                }
                for intent in self.trade_intents[-10:]
            ],
            "no_trade_decisions": [
                {
                    "symbol_id": decision.symbol_id,
                    "reason_code": decision.reason_code,
                    "reason": decision.reason,
                    "confidence": decision.confidence,
                    "confidence_score": decision.confidence_score,
                    "threshold": decision.threshold,
                    "score": decision.score,
                    "event_type": decision.event_type,
                    "signal_id": decision.signal_family or decision.signal_id,
                    "signal_version": decision.signal_version,
                    "asset": decision.asset,
                    "direction": decision.direction,
                    "detected_at": decision.detected_at.isoformat(),
                    "audit_metadata": {
                        key: value
                        for key, value in {
                            "signal_instance_id": decision.signal_id,
                            "signal_family": decision.signal_family,
                            "model_version": decision.model_version,
                            "feature_set": decision.feature_set,
                            "raw_event_id": decision.raw_event_id,
                        }.items()
                        if value is not None
                    },
                }
                for decision in self.no_trade_decisions[-10:]
            ],
            "execution_results": [
                {
                    "symbol_id": result.request.symbol_id,
                    "side": result.request.side,
                    "status": result.status.value,
                    "filled_quantity": (
                        round(result.fill.filled_quantity, 8)
                        if result.fill is not None
                        else 0.0
                    ),
                    "average_price": (
                        round(result.fill.average_price, 8)
                        if result.fill is not None
                        else 0.0
                    ),
                    "client_order_id": result.request.client_order_id,
                    "signal_id": result.request.signal_id,
                    "raw_event_id": result.request.raw_event_id,
                    "event_type": result.request.event_type,
                    "event_group": result.request.event_group,
                    "exit_horizon_label": result.request.exit_horizon_label,
                    "position_id": result.request.position_id,
                    "completed_at": result.completed_at.isoformat(),
                }
                for result in self.execution_results[-10:]
            ],
            "portfolio": {
                "total_equity": self.portfolio.total_equity,
                "cash_usd": self.portfolio.cash_usd,
                "open_position_count": self.portfolio.open_position_count(),
                "realized_pnl_today": self.portfolio.realized_pnl_today,
                "as_of": self.portfolio.as_of.isoformat(),
            },
            "pnl_snapshot": self.pnl_snapshot.to_dict(),
            "drawdown_snapshot": self.drawdown_snapshot.to_dict(),
            "audit_summary": {
                "total_events": self.audit_summary.total_events,
                "failure_count": self.audit_summary.failure_count,
                "fill_count": self.audit_summary.fill_count,
                "status_counts": dict(self.audit_summary.status_counts),
                "event_counts": dict(self.audit_summary.event_counts),
                "symbol_counts": dict(self.audit_summary.symbol_counts),
                "last_recorded_at": self.audit_summary.last_recorded_at.isoformat()
                if self.audit_summary.last_recorded_at
                else None,
            },
            "journal_summary": self.journal_summary.to_dict(),
                "calibration_summary": self.calibration_summary.to_dict(),
                "reputation": self.reputation.to_dict(),
                "signal_discovery": _build_signal_discovery_summary(self.artifacts),
                "last_action": {
                "action": "cycle_completed",
                "status": "completed",
                "occurred_at": self.portfolio.as_of.isoformat(),
                "affects": [
                    "run_summary",
                    "portfolio",
                    "pnl_snapshot",
                    "drawdown_snapshot",
                ],
            },
        }


class LocalArtifactLedger:
    """Persist action-scoped runtime updates to JSONL logs and a live summary file."""

    def __init__(self, paths: RuntimePaths) -> None:
        self._artifacts_store = JsonlFileStore(paths.artifacts_log_path)
        self._checkpoints_store = JsonlFileStore(paths.checkpoints_log_path)
        self._activity_store = JsonlFileStore(paths.activity_log_path)
        self._summary_store = JsonlFileStore(paths.summary_path)
        self._recorded_artifact_ids: set[str] = set()
        self._recorded_checkpoint_ids: set[str] = set()
        self._summary_state: dict[str, Any] = {
            "status": "idle",
            "updated_at": None,
            "counts": {
                "actions": 0,
                "detected_events": 0,
                "trade_intents": 0,
                "no_trade_decisions": 0,
                "executions": 0,
                "artifacts": 0,
                "checkpoints": 0,
            },
            "last_action": None,
        }

    def start_cycle(
        self,
        *,
        rss_result: IngestionRunResult | None,
        prices_result: IngestionRunResult | None,
        classification_count: int,
        detected_events: tuple[DetectedEvent, ...],
    ) -> None:
        started_at = datetime.now(UTC)
        self._summary_state = {
            "status": "running",
            "updated_at": started_at.isoformat(),
            "rss_result": self._ingestion_result_to_dict(rss_result, fallback_run_id="manual-rss"),
            "prices_result": self._ingestion_result_to_dict(
                prices_result,
                fallback_run_id="manual-prices",
            ),
            "classification_count": classification_count,
            "counts": {
                "actions": 0,
                "detected_events": len(detected_events),
                "trade_intents": 0,
                "no_trade_decisions": 0,
                "executions": 0,
                "artifacts": 0,
                "checkpoints": 0,
            },
            "detected_events": [
                {
                    "raw_event_id": event.raw_event_id,
                    "event_type": event.event_type,
                    "rule_name": event.rule_name,
                    "confidence": event.confidence,
                    "novelty_score": event.novelty_score,
                    "repeat_count": event.repeat_count,
                    "narrative_key": event.narrative_key,
                    "detected_at": event.detected_at.isoformat() if event.detected_at else None,
                }
                for event in detected_events[-10:]
            ],
            "trade_intents": [],
            "no_trade_decisions": [],
            "execution_results": [],
            "last_action": None,
        }
        self.record_action(
            action="cycle_started",
            status="running",
            affects=("run_summary", "runtime_cycle"),
            details={
                "classification_count": classification_count,
                "detected_event_count": len(detected_events),
            },
            occurred_at=started_at,
        )

    def record_artifact(self, artifact: ValidationArtifact) -> ValidationCheckpoint:
        artifact_delta = 0
        if artifact.artifact_id not in self._recorded_artifact_ids:
            self._artifacts_store.append(artifact.to_dict())
            self._recorded_artifact_ids.add(artifact.artifact_id)
            artifact_delta = 1

        checkpoint = build_checkpoints((artifact,))[0]
        checkpoint_delta = 0
        if checkpoint.checkpoint_id not in self._recorded_checkpoint_ids:
            self._checkpoints_store.append(checkpoint.to_dict())
            self._recorded_checkpoint_ids.add(checkpoint.checkpoint_id)
            checkpoint_delta = 1

        if artifact_delta or checkpoint_delta:
            self.record_action(
                action="artifact_recorded",
                status=artifact.status.value,
                affects=(
                    "validation_artifacts",
                    "validation_checkpoints",
                    artifact.kind.value,
                    artifact.subject_id,
                ),
                details={
                    "artifact_id": artifact.artifact_id,
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "artifact_kind": artifact.kind.value,
                    "subject_id": artifact.subject_id,
                },
                occurred_at=artifact.created_at,
                count_updates={
                    "artifacts": artifact_delta,
                    "checkpoints": checkpoint_delta,
                },
            )
        return checkpoint

    def record_action(
        self,
        *,
        action: str,
        affects: tuple[str, ...] | list[str],
        details: Mapping[str, Any] | None = None,
        status: str = "recorded",
        occurred_at: datetime | None = None,
        summary_updates: Mapping[str, Any] | None = None,
        count_updates: Mapping[str, int] | None = None,
    ) -> dict[str, Any]:
        timestamp = (occurred_at or datetime.now(UTC)).isoformat()
        affects_list = [str(item) for item in affects if str(item)]
        detail_map = dict(details or {})
        record = {
            "timestamp": timestamp,
            "stage": self._stage_for_action(action),
            "action": action,
            "status": status,
            "summary": self._summarize_action(
                action=action,
                status=status,
                affects=affects_list,
                details=detail_map,
            ),
            "affects": affects_list,
            "details": detail_map,
        }
        if summary_updates:
            for key, value in summary_updates.items():
                if key == "counts" and isinstance(value, Mapping):
                    existing_counts = self._summary_state.setdefault("counts", {})
                    existing_counts.update(dict(value))
                    continue
                self._summary_state[key] = value

        counts = self._summary_state.setdefault("counts", {})
        for key, delta in (count_updates or {}).items():
            counts[key] = int(counts.get(key, 0)) + int(delta)
        counts["actions"] = int(counts.get("actions", 0)) + 1

        last_action = {
            "action": action,
            "status": status,
            "occurred_at": timestamp,
            "affects": record["affects"],
        }
        self._summary_state["updated_at"] = timestamp
        self._summary_state["last_action"] = last_action

        self._activity_store.append(record)
        self._summary_store.write_json(self._summary_state)
        return record

    @staticmethod
    def _stage_for_action(action: str) -> str:
        stage_map = {
            "cycle_started": "cycle",
            "cycle_completed": "cycle",
            "events_detected": "detection",
            "trade_intents_generated": "strategy",
            "no_trade_decisions_recorded": "strategy",
            "execution_skipped": "execution",
            "execution_recorded": "execution",
            "portfolio_updated": "portfolio",
            "artifact_recorded": "validation",
            "checkpoint_recorded": "validation",
            "performance_snapshot_updated": "monitoring",
        }
        return stage_map.get(action, "runtime")

    @staticmethod
    def _summarize_action(
        *,
        action: str,
        status: str,
        affects: list[str],
        details: Mapping[str, Any],
    ) -> str:
        if action == "cycle_started":
            return "Cycle started and live summary tracking is active."
        if action == "events_detected":
            return (
                f"Detected {details.get('count', 0)} new market events for strategy review."
            )
        if action == "trade_intents_generated":
            intent_type = details.get("intent_type", "trade")
            return f"Generated {details.get('count', 0)} {intent_type} intents."
        if action == "no_trade_decisions_recorded":
            return (
                "Defaulted to no trade for "
                f"{details.get('count', 0)} setup(s) below the configured gate."
            )
        if action == "execution_skipped":
            return "Execution was blocked by the runtime risk re-check."
        if action == "execution_recorded":
            symbol_id = details.get("client_order_id") or "order"
            return f"Recorded execution outcome for {symbol_id} ({status})."
        if action == "portfolio_updated":
            return "Portfolio state changed after a successful fill."
        if action == "artifact_recorded":
            return f"Saved {details.get('artifact_kind', 'validation')} evidence and checkpoint."
        if action == "checkpoint_recorded":
            return "Saved a standalone validation checkpoint update."
        if action == "performance_snapshot_updated":
            return "Updated headline portfolio, PnL, and drawdown metrics."
        if action == "cycle_completed":
            return "Cycle finished and the summary snapshot is finalized."
        if affects:
            return f"Recorded {action} affecting {', '.join(affects[:3])}."
        return f"Recorded {action} ({status})."

    def record(
        self,
        *,
        artifacts: tuple[ValidationArtifact, ...],
        checkpoints: tuple[ValidationCheckpoint, ...],
        summary: Mapping[str, Any],
    ) -> None:
        for artifact in artifacts:
            self.record_artifact(artifact)
        for checkpoint in checkpoints:
            if checkpoint.checkpoint_id in self._recorded_checkpoint_ids:
                continue
            self._checkpoints_store.append(checkpoint.to_dict())
            self._recorded_checkpoint_ids.add(checkpoint.checkpoint_id)
            self.record_action(
                action="checkpoint_recorded",
                status=checkpoint.status.value,
                affects=("validation_checkpoints", checkpoint.checkpoint_type.value),
                details={
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "subject_id": checkpoint.subject_id,
                },
                occurred_at=checkpoint.recorded_at,
                count_updates={"checkpoints": 1},
            )

        completed_at = str(
            summary.get("updated_at")
            or self._summary_state.get("updated_at")
            or datetime.now(UTC).isoformat()
        )
        self.record_action(
            action="cycle_completed",
            status=str(summary.get("status") or "completed"),
            affects=("run_summary", "portfolio", "pnl_snapshot", "drawdown_snapshot"),
            details={
                "artifact_count": int(
                    self._summary_state.get("counts", {}).get("artifacts", 0)
                ),
                "checkpoint_count": int(
                    self._summary_state.get("counts", {}).get("checkpoints", 0)
                ),
            },
            occurred_at=datetime.fromisoformat(completed_at),
            summary_updates=summary,
        )

    @staticmethod
    def _ingestion_result_to_dict(
        result: IngestionRunResult | None,
        *,
        fallback_run_id: str,
    ) -> dict[str, Any]:
        resolved = result or IngestionRunResult(
            run_id=fallback_run_id,
            status="ok",
            fetched_count=0,
            inserted_count=0,
            duplicate_count=0,
        )
        return {
            "run_id": resolved.run_id,
            "status": resolved.status,
            "fetched_count": resolved.fetched_count,
            "inserted_count": resolved.inserted_count,
            "duplicate_count": resolved.duplicate_count,
        }


class TradingApplication:
    """Local demo harness for the event-driven Kraken paper/live trading workflow."""

    def __init__(
        self,
        *,
        paths: RuntimePaths,
        feed_groups: dict[str, list[FeedSource]],
        symbols: list[PriceSymbol],
        secondary_symbols: list[PriceSymbol] | None = None,
        tier_promotion_config: TierPromotionConfig | None = None,
        parse_feed: Any | None = None,
        http_get: Any | None = None,
        scheduler: InfoScheduler | None = None,
        identity_registry: IdentityRegistry | None = None,
        execution_config: KrakenCLIConfig | None = None,
        risk_config: RiskConfig | None = None,
        execution_runner: CommandRunner | None = None,
        runtime_env: Mapping[str, str] | None = None,
        runtime_mode: str = "local",
        trading_mode: str = "paper",
        identity_layer: str = "none",
        agent_display_name: str = "AI Trading Agent Demo",
        agent_strategy_name: str = "simple_event_driven",
        agent_owner: str | None = None,
        agent_metadata: Mapping[str, str] | None = None,
        runs_repository: IngestionRunsRepository | None = None,
        raw_events_repository: RawEventsRepository | None = None,
        event_repository: EventDetectionRepository | None = None,
        object_store: RawObjectStore | None = None,
    ) -> None:
        self.paths = paths
        self._scheduler = scheduler or InfoScheduler()
        resolved_runtime_env = {
            key: str(value) for key, value in (runtime_env or {}).items()
        }
        has_storage_overrides = _storage_overrides_provided(
            runs_repository=runs_repository,
            raw_events_repository=raw_events_repository,
            event_repository=event_repository,
            object_store=object_store,
        )
        if has_storage_overrides:
            self._storage_backend = "explicit_override"
            self._runs_repository = cast(IngestionRunsRepository, runs_repository)
            self._raw_events_repository = cast(RawEventsRepository, raw_events_repository)
            self._event_repository = cast(EventDetectionRepository, event_repository)
            self._object_store = cast(RawObjectStore, object_store)
        elif _local_storage_requested(resolved_runtime_env):
            (
                self._storage_backend,
                self._runs_repository,
                self._raw_events_repository,
                self._event_repository,
                self._object_store,
            ) = _build_local_storage(paths)
        else:
            (
                self._storage_backend,
                self._runs_repository,
                self._raw_events_repository,
                self._event_repository,
            ) = _require_postgres_storage(resolved_runtime_env)
            self._object_store = _require_object_store(
                resolved_runtime_env,
                env_path=paths.base_dir / ".env",
            )
        self._rss_pipeline = RSSRawIngestionPipeline(
            runs_repository=self._runs_repository,
            raw_events_repository=self._raw_events_repository,
            object_store=self._object_store,
        )
        self._prices_pipeline = PricesRawIngestionPipeline(
            runs_repository=self._runs_repository,
            raw_events_repository=self._raw_events_repository,
            object_store=self._object_store,
        )
        self._rss_service = RSSIngestionService(
            feed_groups=feed_groups,
            parse_feed=parse_feed,
        )
        self._base_price_symbols = list(symbols)
        self._secondary_symbols = list(secondary_symbols or [])
        self._secondary_symbol_map = {
            symbol.symbol_id: symbol for symbol in self._secondary_symbols
        }
        self._price_http_get = http_get
        self._tier_promotion_config = (
            tier_promotion_config or TierPromotionConfig.from_env(resolved_runtime_env)
        )
        self._promoted_secondary_symbols: dict[str, dict[str, Any]] = {}
        self._sync_price_services()
        self._detection_service = EventDetectionService(
            detector=RuleBasedEventDetector(),
            raw_events_repository=self._raw_events_repository,
            event_detection_repository=self._event_repository,
        )
        self._risk_config = risk_config or RiskConfig(
            max_concurrent_positions=3,
            max_positions_per_asset=1,
        )
        saved_learning_state = load_learning_state(paths.heuristic_state_path)
        self._strategy = SimpleEventDrivenStrategy(
            config=(saved_learning_state.current_config if saved_learning_state else None),
            risk_config=self._risk_config,
        )
        if saved_learning_state is not None:
            self._strategy.set_heuristic_version(saved_learning_state.heuristic_version)
        self._learning_state = saved_learning_state or HeuristicLearningState(
            heuristic_version=self._strategy.heuristic_version,
            current_config=self._strategy.config,
        )
        self._last_post_trade_reviews: list[PostTradeReview] = []
        self._heuristic_refinement_store = JsonlFileStore(paths.heuristic_refinements_path)
        resolved_execution_config = execution_config or KrakenCLIConfig(
            dry_run=False,
            live_enabled=True,
            validate_only=True,
            audit_log_path=paths.audit_log_path,
        )
        self._executor = KrakenCLIExecutor(
            config=resolved_execution_config,
            runner=execution_runner,
            env=runtime_env,
        )
        self._portfolio_provider = LocalPortfolioStateProvider(
            starting_equity=10_000.0,
            starting_cash_usd=10_000.0,
        )
        self._trade_journal = LocalTradeJournal(paths.journal_log_path)
        self._trade_journal.replay_into(self._portfolio_provider)
        self._runtime_mode = runtime_mode
        self._trading_mode = trading_mode
        self._identity_layer = identity_layer
        registry = identity_registry or LocalERC8004Registry()
        self._identity_registry = registry
        self._agent_display_name = agent_display_name
        self._agent_strategy_name = agent_strategy_name
        self._agent_owner = agent_owner or os.environ.get("USERNAME", "local-demo")
        self._agent_metadata = {
            key: str(value) for key, value in (agent_metadata or {}).items()
        }
        self._shared_contract_config = (
            registry.config if isinstance(registry, OnChainERC8004Registry) else None
        )
        self._vault_client = (
            HackathonVaultClient(config=self._shared_contract_config)
            if self._shared_contract_config is not None
            else None
        )
        self._risk_router_client = (
            RiskRouterClient(config=self._shared_contract_config)
            if self._shared_contract_config is not None
            else None
        )
        self._validation_registry_client = (
            ValidationRegistryClient(config=self._shared_contract_config)
            if self._shared_contract_config is not None
            else None
        )
        self._reputation_registry_client = (
            ReputationRegistryClient(config=self._shared_contract_config)
            if self._shared_contract_config is not None
            else None
        )
        identity_metadata = dict(self._agent_metadata)
        identity_metadata["mode"] = runtime_mode
        if isinstance(registry, OnChainERC8004Registry):
            identity_metadata.update(
                {
                    "chain_id": str(registry.config.chain_id),
                    "agent_registry_address": registry.config.agent_registry_address,
                }
            )
            if registry.config.agent_wallet_address is not None:
                identity_metadata.setdefault(
                    "agent_wallet_address",
                    registry.config.agent_wallet_address,
                )
        self._identity_metadata = identity_metadata
        self._identity = self._initialize_identity(registry)
        self._reputation_engine = ReputationEngine()
        self._reputation = self._reputation_engine.initialize(self._identity.agent_id)
        self._artifact_ledger = LocalArtifactLedger(paths)
        persist_learning_state(self.paths.heuristic_state_path, self._learning_state)
        snapshot = self._portfolio_provider.get_portfolio_snapshot()
        self._equity_history: list[EquityPoint] = [
            EquityPoint(recorded_at=snapshot.as_of, equity=snapshot.total_equity)
        ]
        self._latest_quotes: list[PriceQuote] = []
        self._pending_rss_result: IngestionRunResult | None = None
        self._pending_prices_result: IngestionRunResult | None = None
        self._pending_classification_count = 0
        self._handled_event_keys: set[tuple[str, str, str]] = set()
        if self._shared_contract_config is not None and self._identity.agent_id.isdigit():
            self.persist_agent_id(self._identity.agent_id)

    @property
    def identity(self) -> AgentIdentity:
        return self._identity

    def execution_mode_summary(self) -> dict[str, Any]:
        config = self._executor._config
        live_connected_paper = (
            config.live_enabled and not config.dry_run and config.validate_only
        )
        tier_status = self.dynamic_tier_status()
        return {
            "trading_mode": self._trading_mode,
            "identity_layer": self._identity_layer,
            "kraken_cli_executable": config.executable,
            "kraken_dry_run": config.dry_run,
            "kraken_live_enabled": config.live_enabled,
            "kraken_validate_only": config.validate_only,
            "live_connected_paper_trading": live_connected_paper,
            "will_submit_real_orders": (
                config.live_enabled and not config.dry_run and not config.validate_only
            ),
            "storage_backend": self._storage_backend,
            "object_store_backend": (
                "r2"
                if isinstance(self._object_store, S3CompatibleObjectStore)
                else "local"
                if self._storage_backend == "local"
                else "explicit_override"
                if self._storage_backend == "explicit_override"
                else "unknown"
            ),
            "opportunity_budget": {
                "max_positions": self._risk_config.max_concurrent_positions,
                "max_per_asset": self._risk_config.max_positions_per_asset,
            },
            "core_universe_size": len(tier_status["active_core_symbol_ids"]),
            "secondary_universe_size": len(tier_status["secondary_symbol_ids"]),
            "dynamic_tier_status": tier_status,
        }

    def _build_prices_service(self, *, symbols: list[PriceSymbol]) -> PricesIngestionService:
        return PricesIngestionService(
            symbols=symbols,
            http_get=self._price_http_get,
            enrich_volatility_metrics=True,
        )

    def _active_core_symbols(
        self,
        *,
        now: datetime | None = None,
    ) -> list[PriceSymbol]:
        self._prune_expired_tier_promotions(now=now)
        promoted_ids = set(self._promoted_secondary_symbols)
        promoted_symbols = [
            symbol for symbol in self._secondary_symbols if symbol.symbol_id in promoted_ids
        ]
        return [*self._base_price_symbols, *promoted_symbols]

    def _available_secondary_symbols(
        self,
        *,
        now: datetime | None = None,
    ) -> list[PriceSymbol]:
        self._prune_expired_tier_promotions(now=now)
        promoted_ids = set(self._promoted_secondary_symbols)
        return [
            symbol
            for symbol in self._secondary_symbols
            if symbol.symbol_id not in promoted_ids
        ]

    def _sync_price_services(self, *, now: datetime | None = None) -> None:
        active_core_symbols = self._active_core_symbols(now=now)
        available_secondary_symbols = self._available_secondary_symbols(now=now)
        self._prices_service = self._build_prices_service(symbols=active_core_symbols)
        self._secondary_prices_service = (
            self._build_prices_service(symbols=available_secondary_symbols)
            if available_secondary_symbols
            else None
        )

    def _prune_expired_tier_promotions(self, *, now: datetime | None = None) -> bool:
        checked_at = now or datetime.now(UTC)
        expired_symbol_ids = [
            symbol_id
            for symbol_id, details in self._promoted_secondary_symbols.items()
            if cast(datetime, details["expires_at"]) <= checked_at
        ]
        for symbol_id in expired_symbol_ids:
            del self._promoted_secondary_symbols[symbol_id]
        return bool(expired_symbol_ids)

    @staticmethod
    def _price_return_fraction(quote: PriceQuote | None) -> float:
        if quote is None or quote.open <= 0:
            return 0.0
        return (quote.current - quote.open) / quote.open

    def _correlation_break_score(
        self,
        *,
        quote: PriceQuote,
        benchmark_quote: PriceQuote | None,
    ) -> float:
        benchmark_return = self._price_return_fraction(benchmark_quote)
        symbol_return = self._price_return_fraction(quote)
        divergence = abs(symbol_return - benchmark_return)
        if symbol_return * benchmark_return < 0:
            divergence += 0.02
        return round(max(divergence, 0.0), 4)

    def _promotion_lookup_quotes(
        self,
        price_quotes: list[PriceQuote] | None = None,
    ) -> list[PriceQuote]:
        quote_lookup = {
            quote.symbol_id: quote for quote in (price_quotes or self._latest_quotes)
        }
        for symbol in [*self._base_price_symbols, *self._secondary_symbols]:
            if symbol.symbol_id not in quote_lookup:
                quote_lookup[symbol.symbol_id] = PriceQuote(
                    symbol_id=symbol.symbol_id,
                    current=0.0,
                    open=0.0,
                    high=0.0,
                    low=0.0,
                    prev_close=0.0,
                    timestamp=0,
                    asset_class=symbol.asset_class,
                )
        return list(quote_lookup.values())

    def _news_scores_by_symbol(
        self,
        detected_events: list[DetectedEvent] | tuple[DetectedEvent, ...],
        *,
        price_quotes: list[PriceQuote] | None = None,
    ) -> dict[str, float]:
        if not detected_events:
            return {}

        lookup_quotes = self._promotion_lookup_quotes(price_quotes)
        cumulative_scores: dict[str, float] = {}
        event_counts: dict[str, int] = {}

        for event in detected_events:
            quote = select_quote_for_event(event=event, price_quotes=lookup_quotes)
            if quote is None or quote.symbol_id not in self._secondary_symbol_map:
                continue
            novelty = event.novelty_score if event.novelty_score is not None else 1.0
            bounded_novelty = min(max(novelty, 0.0), 1.0)
            weighted_confidence = max(event.confidence, 0.0) * (0.7 + (0.3 * bounded_novelty))
            cumulative_scores[quote.symbol_id] = (
                cumulative_scores.get(quote.symbol_id, 0.0) + weighted_confidence
            )
            event_counts[quote.symbol_id] = event_counts.get(quote.symbol_id, 0) + 1

        return {
            symbol_id: round(
                min(
                    1.5,
                    (score / max(event_counts[symbol_id], 1))
                    + (0.15 * max(event_counts[symbol_id] - 1, 0)),
                ),
                4,
            )
            for symbol_id, score in cumulative_scores.items()
        }

    def refresh_dynamic_tier_promotions(
        self,
        *,
        detected_events: list[DetectedEvent] | tuple[DetectedEvent, ...] | None = None,
        price_quotes: list[PriceQuote] | None = None,
        now: datetime | None = None,
    ) -> tuple[str, ...]:
        checked_at = now or datetime.now(UTC)
        did_prune = self._prune_expired_tier_promotions(now=checked_at)
        actual_quotes = list(price_quotes or self._latest_quotes)
        if actual_quotes:
            self._latest_quotes = self._merge_quotes(self._latest_quotes, actual_quotes)
        lookup_quotes = self._promotion_lookup_quotes(actual_quotes)
        quote_by_symbol = {quote.symbol_id: quote for quote in lookup_quotes}
        benchmark_quote = quote_by_symbol.get("btc_usd") or quote_by_symbol.get("eth_usd")
        news_scores = self._news_scores_by_symbol(
            detected_events or (),
            price_quotes=lookup_quotes,
        )
        newly_promoted: list[str] = []

        for symbol in self._secondary_symbols:
            quote = quote_by_symbol.get(symbol.symbol_id)
            if quote is None:
                continue

            volume_ratio = quote.volume_ratio or 0.0
            volatility_filter = quote.volatility_filter or 0.0
            news_score = news_scores.get(symbol.symbol_id, 0.0)
            correlation_break = self._correlation_break_score(
                quote=quote,
                benchmark_quote=benchmark_quote,
            )
            reasons: list[str] = []

            if volume_ratio >= self._tier_promotion_config.volume_ratio_threshold:
                reasons.append(f"volume_spike:{volume_ratio:.2f}x")
            if (
                volatility_filter
                >= self._tier_promotion_config.volatility_filter_threshold
            ):
                reasons.append(f"volatility_spike:{volatility_filter:.2f}")
            if news_score >= self._tier_promotion_config.news_score_threshold:
                reasons.append(f"news_spike:{news_score:.2f}")
            if (
                correlation_break
                >= self._tier_promotion_config.correlation_break_threshold
            ):
                reasons.append(f"correlation_break:{correlation_break:.4f}")

            if not reasons:
                continue

            existing = self._promoted_secondary_symbols.get(symbol.symbol_id)
            if existing is None:
                newly_promoted.append(symbol.symbol_id)
            self._promoted_secondary_symbols[symbol.symbol_id] = {
                "symbol_id": symbol.symbol_id,
                "promoted_at": (
                    cast(datetime, existing["promoted_at"])
                    if existing is not None
                    else checked_at
                ),
                "expires_at": checked_at + self._tier_promotion_config.promotion_duration,
                "reasons": tuple(reasons),
                "metrics": {
                    "volume_ratio": round(volume_ratio, 4),
                    "volatility_filter": round(volatility_filter, 4),
                    "news_score": round(news_score, 4),
                    "correlation_break": round(correlation_break, 4),
                },
            }

        if newly_promoted or did_prune:
            self._sync_price_services(now=checked_at)

        return tuple(sorted(newly_promoted))

    def dynamic_tier_status(self, *, now: datetime | None = None) -> dict[str, Any]:
        checked_at = now or datetime.now(UTC)
        if self._prune_expired_tier_promotions(now=checked_at):
            self._sync_price_services(now=checked_at)

        promotions: dict[str, Any] = {}
        for symbol_id, details in sorted(self._promoted_secondary_symbols.items()):
            promotions[symbol_id] = {
                "promoted_at": cast(datetime, details["promoted_at"]).isoformat(),
                "expires_at": cast(datetime, details["expires_at"]).isoformat(),
                "reasons": list(cast(tuple[str, ...], details["reasons"])),
                "metrics": dict(cast(dict[str, float], details["metrics"])),
            }

        return {
            "enabled": bool(self._secondary_symbols),
            "promotion_window_hours": self._tier_promotion_config.promotion_duration_hours,
            "thresholds": {
                "volume_ratio": self._tier_promotion_config.volume_ratio_threshold,
                "volatility_filter": self._tier_promotion_config.volatility_filter_threshold,
                "news_score": self._tier_promotion_config.news_score_threshold,
                "correlation_break": self._tier_promotion_config.correlation_break_threshold,
            },
            "active_core_symbol_ids": tuple(
                symbol.symbol_id for symbol in self._active_core_symbols(now=checked_at)
            ),
            "secondary_symbol_ids": tuple(
                symbol.symbol_id
                for symbol in self._available_secondary_symbols(now=checked_at)
            ),
            "promotions": promotions,
        }

    def _initialize_identity(self, registry: IdentityRegistry) -> AgentIdentity:
        if isinstance(registry, OnChainERC8004Registry):
            preconfigured_agent_id = registry.config.agent_id
            if preconfigured_agent_id is not None:
                try:
                    existing = registry.get(str(preconfigured_agent_id))
                except Exception as exc:
                    existing = None
                    self._identity_metadata.setdefault("registry_lookup_error", str(exc))
                if existing is not None:
                    return existing

            pending_agent_id = (
                str(preconfigured_agent_id)
                if preconfigured_agent_id is not None
                else "pending-registration"
            )
            return AgentIdentity(
                agent_id=pending_agent_id,
                display_name=self._agent_display_name,
                strategy_name=self._agent_strategy_name,
                owner=self._agent_owner,
                exchange="sepolia",
                wallet_address=registry.config.agent_wallet_address,
                metadata=dict(self._identity_metadata)
                | {"registration_status": "pending"},
            )

        return registry.register(
            display_name=self._agent_display_name,
            strategy_name=self._agent_strategy_name,
            owner=self._agent_owner,
            exchange="kraken",
            metadata=self._identity_metadata,
        )

    def ensure_onchain_identity(self) -> AgentIdentity:
        if not isinstance(self._identity_registry, OnChainERC8004Registry):
            return self._identity
        if self._identity.agent_id.isdigit():
            self.persist_agent_id(self._identity.agent_id)
            return self._identity

        identity = self._identity_registry.register(
            display_name=self._agent_display_name,
            strategy_name=self._agent_strategy_name,
            owner=self._agent_owner,
            exchange="sepolia",
            wallet_address=(
                self._shared_contract_config.agent_wallet_address
                if self._shared_contract_config is not None
                else self._identity.wallet_address
            ),
            metadata=self._identity_metadata,
        )
        self._identity = identity
        self.persist_agent_id(identity.agent_id)
        return self._identity

    def persist_agent_id(
        self,
        agent_id: int | str | None = None,
        *,
        env_path: str | Path | None = None,
    ) -> Path:
        resolved_agent_id = str(agent_id or self._identity.agent_id).strip()
        if not resolved_agent_id.isdigit():
            raise ValueError("Only numeric on-chain `agentId` values can be persisted.")

        target_path = (
            Path(env_path) if env_path is not None else self.paths.base_dir / ".runtime.env"
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)

        existing_lines = (
            target_path.read_text(encoding="utf-8").splitlines()
            if target_path.exists()
            else []
        )
        updated_lines: list[str] = []
        saw_agent_id = False
        saw_identity_layer = False

        for line in existing_lines:
            stripped = line.strip()
            if stripped.startswith("AGENT_ID="):
                updated_lines.append(f"AGENT_ID={resolved_agent_id}")
                saw_agent_id = True
            elif stripped.startswith("IDENTITY_LAYER=") and self._identity_layer == "erc8004":
                updated_lines.append("IDENTITY_LAYER=erc8004")
                saw_identity_layer = True
            else:
                updated_lines.append(line)

        if not saw_agent_id:
            updated_lines.append(f"AGENT_ID={resolved_agent_id}")
        if self._identity_layer == "erc8004" and not saw_identity_layer:
            updated_lines.append("IDENTITY_LAYER=erc8004")

        target_path.write_text("\n".join(updated_lines).rstrip() + "\n", encoding="utf-8")
        os.environ["AGENT_ID"] = resolved_agent_id
        return target_path

    def shared_contract_status(self) -> dict[str, Any]:
        if self._shared_contract_config is None:
            return {
                "enabled": False,
                "runtime_mode": self._runtime_mode,
            }

        status: dict[str, Any] = {
            "enabled": True,
            "runtime_mode": self._runtime_mode,
            "chain_id": self._shared_contract_config.chain_id,
            "agent_id": self._identity.agent_id,
            "agent_display_name": self._agent_display_name,
            "agent_strategy_name": self._agent_strategy_name,
            "agent_registry_address": self._shared_contract_config.agent_registry_address,
            "hackathon_vault_address": self._shared_contract_config.hackathon_vault_address,
            "risk_router_address": self._shared_contract_config.risk_router_address,
            "validation_registry_address": self._shared_contract_config.validation_registry_address,
            "reputation_registry_address": self._shared_contract_config.reputation_registry_address,
            "transaction_ready": self._shared_contract_config.is_ready_for_transactions,
            "missing_required_values": list(
                self._shared_contract_config.missing_required_values()
            ),
            "operator_wallet_address": self._shared_contract_config.operator_wallet_address,
            "agent_wallet_address": self._shared_contract_config.agent_wallet_address,
            "registration_status": self._identity.metadata.get(
                "registration_status",
                "registered" if self._identity.agent_id.isdigit() else "pending",
            ),
            "needs_registration": (
                self._identity.metadata.get("registration_status") == "pending"
                or not self._identity.agent_id.isdigit()
            ),
            "agent_id_env_path": str(self.paths.base_dir / ".runtime.env"),
            "has_claimed_allocation": None,
            "vault_balance_wei": None,
        }

        resolved_agent_id = self._try_resolved_agent_id_for_chain()
        if resolved_agent_id is None or self._vault_client is None:
            return status

        try:
            status["has_claimed_allocation"] = self._vault_client.has_claimed(
                resolved_agent_id
            )
            status["vault_balance_wei"] = self._vault_client.get_balance(resolved_agent_id)
        except Exception as exc:
            status["vault_error"] = str(exc)

        return status

    def claim_sandbox_allocation(self) -> OnChainTransactionResult:
        if self._vault_client is None:
            raise RuntimeError("Shared Sepolia mode is not enabled for this app instance.")
        self.ensure_onchain_identity()
        return self._vault_client.claim_allocation(self._resolved_agent_id_for_chain())

    def submit_trade_intent_onchain(
        self,
        intent: TradeIntent,
        *,
        max_slippage_bps: int = 100,
        ttl_seconds: int = 300,
    ) -> OnChainTransactionResult:
        if self._risk_router_client is None:
            raise RuntimeError("Shared Sepolia mode is not enabled for this app instance.")
        self.ensure_onchain_identity()
        if self.identity.wallet_address is None:
            raise RuntimeError(
                "The registered agent is missing a wallet address for RiskRouter signing."
            )

        agent_id = self._resolved_agent_id_for_chain()
        nonce = self._risk_router_client.get_intent_nonce(agent_id)
        deadline = int(intent.generated_at.timestamp()) + ttl_seconds
        router_intent = RiskRouterIntent.from_trade_intent(
            agent_id=agent_id,
            agent_wallet=self.identity.wallet_address,
            trade_intent=intent,
            nonce=nonce,
            deadline=deadline,
            max_slippage_bps=max_slippage_bps,
        )
        simulation = self._risk_router_client.simulate_trade_intent(router_intent)
        if not simulation.approved:
            return OnChainTransactionResult(
                tx_hash="not-submitted",
                status="rejected",
                details={
                    "agent_id": agent_id,
                    "pair": router_intent.pair,
                    "reason": simulation.reason,
                },
            )
        return self._risk_router_client.submit_trade_intent(router_intent)

    def post_checkpoint_onchain(
        self,
        checkpoint: ValidationCheckpoint,
        *,
        score: int = 85,
    ) -> OnChainTransactionResult:
        if self._validation_registry_client is None:
            raise RuntimeError("Shared Sepolia mode is not enabled for this app instance.")
        self.ensure_onchain_identity()
        return self._validation_registry_client.post_checkpoint(
            checkpoint,
            agent_id=self._resolved_agent_id_for_chain(),
            score=score,
        )

    def get_onchain_reputation_score(self) -> int | None:
        if self._reputation_registry_client is None:
            return None
        self.ensure_onchain_identity()
        return self._reputation_registry_client.get_average_score(
            self._resolved_agent_id_for_chain()
        )

    def run_shared_contract_actions(
        self,
        *,
        trade_intents: tuple[TradeIntent, ...],
        checkpoints: tuple[ValidationCheckpoint, ...],
        register_agent: bool = False,
        claim_allocation: bool = False,
        submit_trade_intents: bool = False,
        post_checkpoints: bool = False,
    ) -> dict[str, Any]:
        summary: dict[str, Any] = self.shared_contract_status()
        summary.update(
            {
                "registration": None,
                "claim_allocation": None,
                "trade_submissions": [],
                "checkpoint_posts": [],
                "reputation_score": None,
            }
        )

        if self._shared_contract_config is None:
            return summary

        if register_agent:
            summary["registration"] = self.ensure_onchain_identity().to_dict()

        persisted_path = None
        resolved_agent_id = self._try_resolved_agent_id_for_chain()
        if resolved_agent_id is not None:
            persisted_path = self.persist_agent_id(resolved_agent_id)
            summary["persisted_agent_env_path"] = str(persisted_path)

        if claim_allocation:
            summary["claim_allocation"] = self.claim_sandbox_allocation().__dict__

        if submit_trade_intents:
            summary["trade_submissions"] = [
                self.submit_trade_intent_onchain(intent).__dict__
                for intent in trade_intents
            ]

        if post_checkpoints:
            summary["checkpoint_posts"] = [
                self.post_checkpoint_onchain(checkpoint).__dict__
                for checkpoint in checkpoints
            ]

        try:
            summary["reputation_score"] = self.get_onchain_reputation_score()
        except Exception as exc:
            summary["reputation_error"] = str(exc)

        return summary

    def _try_resolved_agent_id_for_chain(self) -> int | None:
        if self._identity.agent_id.isdigit():
            return int(self._identity.agent_id)
        return None

    def _resolved_agent_id_for_chain(self) -> int:
        resolved = self._try_resolved_agent_id_for_chain()
        if resolved is not None:
            return resolved
        raise RuntimeError(
            "The current identity does not have a numeric on-chain `agentId` yet."
        )

    def wire_scheduler(
        self,
        *,
        feed_group: str = "market_news",
        rss_interval_seconds: int = 120,
        prices_interval_seconds: int = 60,
        secondary_prices_interval_seconds: int = SECONDARY_POLLING_FREQUENCY_SECONDS,
        detection_interval_seconds: int = 60,
        execution_interval_seconds: int = 60,
    ) -> InfoScheduler:
        def _run_rss_job() -> None:
            self.ingest_rss_group(feed_group=feed_group)

        def _run_prices_job() -> None:
            self.ingest_prices()

        def _run_secondary_prices_job() -> None:
            self.ingest_secondary_prices()

        def _run_detection_job() -> None:
            self.classify_events()

        def _run_execution_job() -> None:
            self.execute_trade_cycle()

        self._scheduler.register_rss_job(
            _run_rss_job,
            interval_seconds=rss_interval_seconds,
        )
        self._scheduler.register_prices_job(
            _run_prices_job,
            interval_seconds=prices_interval_seconds,
        )
        if (
            self._secondary_prices_service is not None
            and secondary_prices_interval_seconds > 0
        ):
            self._scheduler.register_secondary_prices_job(
                _run_secondary_prices_job,
                interval_seconds=secondary_prices_interval_seconds,
            )
        self._scheduler.register_event_detection_job(
            _run_detection_job,
            interval_seconds=detection_interval_seconds,
        )
        self._scheduler.register_execution_job(
            _run_execution_job,
            interval_seconds=execution_interval_seconds,
        )
        return self._scheduler

    def ingest_rss_group(self, *, feed_group: str = "market_news") -> IngestionRunResult:
        articles = self._rss_service.fetch_group(feed_group)
        unique_articles = self._rss_service.deduplicate(articles)
        result = self._rss_pipeline.persist_articles(
            source_group=feed_group,
            articles=unique_articles,
        )
        return self._remember_rss_result(result)

    @staticmethod
    def _merge_quotes(
        existing_quotes: list[PriceQuote],
        new_quotes: list[PriceQuote],
    ) -> list[PriceQuote]:
        merged = {quote.symbol_id: quote for quote in existing_quotes}
        for quote in new_quotes:
            merged[quote.symbol_id] = quote
        return list(merged.values())

    @staticmethod
    def _combine_ingestion_results(
        primary: IngestionRunResult,
        secondary: IngestionRunResult,
    ) -> IngestionRunResult:
        return IngestionRunResult(
            run_id=f"{primary.run_id}+{secondary.run_id}",
            status="ok" if primary.status == "ok" and secondary.status == "ok" else "error",
            fetched_count=primary.fetched_count + secondary.fetched_count,
            inserted_count=primary.inserted_count + secondary.inserted_count,
            duplicate_count=primary.duplicate_count + secondary.duplicate_count,
        )

    @classmethod
    def _accumulate_ingestion_result(
        cls,
        current: IngestionRunResult | None,
        latest: IngestionRunResult | None,
    ) -> IngestionRunResult | None:
        if latest is None:
            return current
        if current is None:
            return latest
        return cls._combine_ingestion_results(current, latest)

    def _remember_rss_result(self, result: IngestionRunResult) -> IngestionRunResult:
        self._pending_rss_result = self._accumulate_ingestion_result(
            self._pending_rss_result,
            result,
        )
        return result

    def _remember_prices_result(
        self,
        result: IngestionRunResult | None,
    ) -> IngestionRunResult | None:
        if result is None:
            return None
        self._pending_prices_result = self._accumulate_ingestion_result(
            self._pending_prices_result,
            result,
        )
        return result

    def _resolve_pending_cycle_inputs(
        self,
        *,
        rss_result: IngestionRunResult | None,
        prices_result: IngestionRunResult | None,
        classification_count: int | None,
    ) -> tuple[IngestionRunResult | None, IngestionRunResult | None, int]:
        resolved_rss_result = rss_result or self._pending_rss_result
        resolved_prices_result = prices_result or self._pending_prices_result
        resolved_classification_count = (
            self._pending_classification_count
            if classification_count is None
            else classification_count
        )
        self._pending_rss_result = None
        self._pending_prices_result = None
        self._pending_classification_count = 0
        return (
            resolved_rss_result,
            resolved_prices_result,
            resolved_classification_count,
        )

    def _ingest_quotes_with_service(
        self,
        service: PricesIngestionService,
    ) -> IngestionRunResult:
        quotes = service.fetch_current_prices()
        self._latest_quotes = self._merge_quotes(self._latest_quotes, quotes)
        return self._prices_pipeline.persist_quotes(quotes=quotes)

    def ingest_prices(self, *, include_secondary: bool = False) -> IngestionRunResult:
        result = self._ingest_quotes_with_service(self._prices_service)
        if include_secondary:
            secondary_result = self.ingest_secondary_prices(record_pending=False)
            if secondary_result is not None:
                result = self._combine_ingestion_results(result, secondary_result)
        self._remember_prices_result(result)
        return result

    def ingest_secondary_prices(
        self,
        *,
        record_pending: bool = True,
    ) -> IngestionRunResult | None:
        if self._secondary_prices_service is None:
            return None
        result = self._ingest_quotes_with_service(self._secondary_prices_service)
        self.refresh_dynamic_tier_promotions(price_quotes=list(self._latest_quotes))
        if record_pending:
            self._remember_prices_result(result)
        return result

    def classify_events(self) -> int:
        classified_count = self._detection_service.classify_pending_events(
            source_type="rss",
            batch_size=100,
        )
        self._pending_classification_count += max(0, classified_count)
        return classified_count

    def run_cycle(self, *, feed_group: str = "market_news") -> RuntimeCycleResult:
        rss_result = self.ingest_rss_group(feed_group=feed_group)
        prices_result = self.ingest_prices(include_secondary=True)
        classification_count = self.classify_events()
        return self.execute_trade_cycle(
            rss_result=rss_result,
            prices_result=prices_result,
            classification_count=classification_count,
        )

    def execute_trade_cycle(
        self,
        *,
        rss_result: IngestionRunResult | None = None,
        prices_result: IngestionRunResult | None = None,
        classification_count: int | None = None,
    ) -> RuntimeCycleResult:
        self._last_post_trade_reviews = []
        rss_result, prices_result, classification_count = self._resolve_pending_cycle_inputs(
            rss_result=rss_result,
            prices_result=prices_result,
            classification_count=classification_count,
        )
        current_portfolio = self._portfolio_provider.get_portfolio_snapshot()
        detected_events = self._new_detected_events()
        promoted_symbols = self.refresh_dynamic_tier_promotions(
            detected_events=list(detected_events),
            price_quotes=list(self._latest_quotes),
        )
        self._artifact_ledger.start_cycle(
            rss_result=rss_result,
            prices_result=prices_result,
            classification_count=classification_count,
            detected_events=detected_events,
        )
        if detected_events:
            self._artifact_ledger.record_action(
                action="events_detected",
                status="recorded",
                affects=("event_detection", "strategy"),
                details={
                    "count": len(detected_events),
                    "event_types": sorted({event.event_type for event in detected_events}),
                },
            )
        if promoted_symbols:
            self._artifact_ledger.record_action(
                action="tier_promotions_updated",
                status="active",
                affects=("prices_ingestion", "strategy"),
                details={
                    "symbols": list(promoted_symbols),
                    "promotion_window_hours": (
                        self._tier_promotion_config.promotion_duration_hours
                    ),
                },
            )

        evaluate_position_exits = getattr(self._strategy, "evaluate_position_exits", None)
        exit_intents = (
            tuple(
                cast(Any, evaluate_position_exits)(
                    portfolio=current_portfolio,
                    price_quotes=list(self._latest_quotes),
                    detected_events=list(detected_events),
                )
            )
            if callable(evaluate_position_exits)
            else ()
        )
        if exit_intents:
            self._artifact_ledger.record_action(
                action="trade_intents_generated",
                status="recorded",
                affects=("strategy", "execution_queue", "portfolio"),
                details={
                    "intent_type": "exit",
                    "count": len(exit_intents),
                    "symbols": sorted({intent.symbol_id for intent in exit_intents}),
                },
                count_updates={"trade_intents": len(exit_intents)},
            )
        exit_results, exit_artifacts = self._execute_intents(
            trade_intents=exit_intents,
            current_portfolio=current_portfolio,
        )

        current_portfolio = self._portfolio_provider.get_portfolio_snapshot()
        entry_intents = tuple(
            self._strategy.generate_trade_intents(
                detected_events=list(detected_events),
                price_quotes=list(self._latest_quotes),
                portfolio=current_portfolio,
            )
        )
        if entry_intents:
            self._artifact_ledger.record_action(
                action="trade_intents_generated",
                status="recorded",
                affects=("strategy", "execution_queue", "portfolio"),
                details={
                    "intent_type": "entry",
                    "count": len(entry_intents),
                    "symbols": sorted({intent.symbol_id for intent in entry_intents}),
                },
                count_updates={"trade_intents": len(entry_intents)},
            )
        consume_no_trade_decisions = getattr(self._strategy, "consume_no_trade_decisions", None)
        no_trade_decisions = (
            tuple(cast(Any, consume_no_trade_decisions)())
            if callable(consume_no_trade_decisions)
            else ()
        )
        no_trade_artifacts = self._record_no_trade_decisions(decisions=no_trade_decisions)
        if no_trade_decisions:
            self._artifact_ledger.record_action(
                action="no_trade_decisions_recorded",
                status="recorded",
                affects=("strategy", "validation_artifacts", "signal_gating"),
                details={
                    "count": len(no_trade_decisions),
                    "reasons": sorted({decision.reason_code for decision in no_trade_decisions}),
                    "symbols": sorted({decision.symbol_id for decision in no_trade_decisions}),
                },
                summary_updates={
                    "no_trade_decisions": [
                        {
                            "symbol_id": decision.symbol_id,
                            "reason_code": decision.reason_code,
                            "reason": decision.reason,
                            "confidence": decision.confidence,
                            "confidence_score": decision.confidence_score,
                            "threshold": decision.threshold,
                            "score": decision.score,
                            "event_type": decision.event_type,
                            "signal_id": decision.signal_family or decision.signal_id,
                            "signal_version": decision.signal_version,
                            "asset": decision.asset,
                            "direction": decision.direction,
                            "detected_at": decision.detected_at.isoformat(),
                            "audit_metadata": {
                                key: value
                                for key, value in {
                                    "signal_instance_id": decision.signal_id,
                                    "signal_family": decision.signal_family,
                                    "model_version": decision.model_version,
                                    "feature_set": decision.feature_set,
                                    "raw_event_id": decision.raw_event_id,
                                }.items()
                                if value is not None
                            },
                        }
                        for decision in no_trade_decisions[-10:]
                    ]
                },
                count_updates={"no_trade_decisions": len(no_trade_decisions)},
            )
        entry_results, entry_artifacts = self._execute_intents(
            trade_intents=entry_intents,
            current_portfolio=current_portfolio,
        )

        trade_intents = exit_intents + entry_intents
        execution_results = [*exit_results, *entry_results]
        artifacts = [*exit_artifacts, *no_trade_artifacts, *entry_artifacts]

        portfolio = self._portfolio_provider.get_portfolio_snapshot()
        performance_artifact = ValidationArtifact.from_performance_checkpoint(
            portfolio,
            agent_id=self._identity.agent_id,
            checkpoint_name="local_dry_run_cycle",
        )
        artifacts.append(performance_artifact)
        self._artifact_ledger.record_artifact(performance_artifact)

        checkpoints = build_checkpoints(artifacts)
        self._reputation = self._reputation_engine.apply_artifacts(
            self._reputation,
            artifacts,
        )
        pnl_snapshot = build_pnl_snapshot(
            portfolio=portfolio,
            price_quotes=self._latest_quotes,
        )
        self._equity_history.append(
            EquityPoint(recorded_at=portfolio.as_of, equity=portfolio.total_equity)
        )
        drawdown_snapshot = build_drawdown_snapshot(self._equity_history)
        audit_summary = build_audit_summary_from_file(self.paths.audit_log_path)
        journal_summary = self._trade_journal.build_summary()
        calibration_summary = build_calibration_summary(self._trade_journal.load_entries())
        learning_state = self._apply_post_trade_learning(
            calibration_summary=calibration_summary,
        )
        self._artifact_ledger.record_action(
            action="performance_snapshot_updated",
            status="recorded",
            affects=(
                "portfolio",
                "pnl_snapshot",
                "drawdown_snapshot",
                "journal_summary",
                "calibration_summary",
                "heuristic_state",
            ),
            details={
                "total_equity": round(portfolio.total_equity, 2),
                "cash_usd": round(portfolio.cash_usd, 2),
                "open_position_count": portfolio.open_position_count(),
                "realized_pnl_today": round(portfolio.realized_pnl_today, 2),
                "signal_outcome_count": _build_signal_discovery_summary(
                    artifacts
                )["total_outcomes"],
                "post_trade_review_count": len(self._last_post_trade_reviews),
                "heuristic_adjustment_count": len(
                    learning_state.last_applied_adjustments
                ),
            },
            occurred_at=portfolio.as_of,
            summary_updates={
                "portfolio": {
                    "total_equity": portfolio.total_equity,
                    "cash_usd": portfolio.cash_usd,
                    "open_position_count": portfolio.open_position_count(),
                    "realized_pnl_today": portfolio.realized_pnl_today,
                    "as_of": portfolio.as_of.isoformat(),
                },
                "pnl_snapshot": pnl_snapshot.to_dict(),
                "drawdown_snapshot": drawdown_snapshot.to_dict(),
                "audit_summary": audit_summary.to_dict(),
                "journal_summary": journal_summary.to_dict(),
                "calibration_summary": calibration_summary.to_dict(),
                "post_trade_reviews": [
                    review.to_dict() for review in self._last_post_trade_reviews[-10:]
                ],
                "heuristic_state": learning_state.to_dict(),
                "reputation": self._reputation.to_dict(),
                "signal_discovery": _build_signal_discovery_summary(artifacts),
            },
        )

        result = RuntimeCycleResult(
            rss_result=rss_result
            or IngestionRunResult(
                run_id="manual-rss",
                status="ok",
                fetched_count=0,
                inserted_count=0,
                duplicate_count=0,
            ),
            prices_result=prices_result
            or IngestionRunResult(
                run_id="manual-prices",
                status="ok",
                fetched_count=0,
                inserted_count=0,
                duplicate_count=0,
            ),
            classification_count=classification_count,
            detected_events=detected_events,
            trade_intents=trade_intents,
            no_trade_decisions=no_trade_decisions,
            execution_results=tuple(execution_results),
            artifacts=tuple(artifacts),
            checkpoints=checkpoints,
            portfolio=portfolio,
            pnl_snapshot=pnl_snapshot,
            drawdown_snapshot=drawdown_snapshot,
            audit_summary=audit_summary,
            journal_summary=journal_summary,
            calibration_summary=calibration_summary,
            reputation=self._reputation,
        )
        self._artifact_ledger.record(
            artifacts=result.artifacts,
            checkpoints=result.checkpoints,
            summary=result.to_summary_dict(),
        )
        return result

    def _apply_post_trade_learning(
        self,
        *,
        calibration_summary: CalibrationSummary,
    ) -> HeuristicLearningState:
        current_config = getattr(self._strategy, "config", None)
        if not isinstance(current_config, StrategyConfig):
            current_config = self._learning_state.current_config

        updated_config, adjustments = refine_strategy_config(
            current_config=current_config,
            calibration_summary=calibration_summary,
            timing_labels=tuple(
                review.timing_label for review in self._last_post_trade_reviews
            ),
            asset_selection_labels=tuple(
                review.asset_selection_label for review in self._last_post_trade_reviews
            ),
        )
        heuristic_version = str(
            getattr(self._strategy, "heuristic_version", self._learning_state.heuristic_version)
        )
        apply_refined_config = getattr(self._strategy, "apply_refined_config", None)
        if adjustments and callable(apply_refined_config):
            heuristic_version = str(apply_refined_config(updated_config))
            effective_config = getattr(self._strategy, "config", updated_config)
        else:
            effective_config = updated_config

        if not isinstance(effective_config, StrategyConfig):
            effective_config = updated_config

        learning_state = HeuristicLearningState(
            heuristic_version=heuristic_version,
            current_config=effective_config,
            last_post_trade_reviews=tuple(self._last_post_trade_reviews[-10:]),
            last_applied_adjustments=adjustments,
        )
        persist_learning_state(self.paths.heuristic_state_path, learning_state)
        self._learning_state = learning_state

        if self._last_post_trade_reviews or adjustments:
            self._heuristic_refinement_store.append(
                {
                    "recorded_at": learning_state.updated_at.isoformat(),
                    "heuristic_version": learning_state.heuristic_version,
                    "post_trade_reviews": [
                        review.to_dict() for review in learning_state.last_post_trade_reviews
                    ],
                    "applied_adjustments": [
                        adjustment.to_dict()
                        for adjustment in learning_state.last_applied_adjustments
                    ],
                }
            )
        return learning_state

    def _record_no_trade_decisions(
        self,
        *,
        decisions: tuple[NoTradeDecision, ...],
    ) -> list[ValidationArtifact]:
        artifacts: list[ValidationArtifact] = []
        for decision in decisions:
            artifact = ValidationArtifact.from_no_trade_decision(
                decision,
                agent_id=self._identity.agent_id,
            )
            self._artifact_ledger.record_artifact(artifact)
            artifacts.append(artifact)
        return artifacts

    def _execute_intents(
        self,
        *,
        trade_intents: tuple[TradeIntent, ...],
        current_portfolio: PortfolioSnapshot,
    ) -> tuple[list[ExecutionResult], list[ValidationArtifact]]:
        execution_results: list[ExecutionResult] = []
        artifacts: list[ValidationArtifact] = []
        working_portfolio = current_portfolio

        for intent in trade_intents:
            subject_id = (
                f"{intent.signal_id or intent.symbol_id}:{intent.side}:"
                f"{intent.exit_horizon_label or 'core'}:{intent.generated_at.isoformat()}"
            )
            trade_artifact = ValidationArtifact.from_trade_intent(
                intent,
                agent_id=self._identity.agent_id,
            )
            self._artifact_ledger.record_artifact(trade_artifact)
            risk_result = self._strategy.reassess_trade_intent(
                intent=intent,
                portfolio=working_portfolio,
            )
            risk_artifact = ValidationArtifact.from_risk_check(
                risk_result,
                agent_id=self._identity.agent_id,
                subject_id=subject_id,
                proposed_notional=intent.notional_usd,
                checked_at=intent.generated_at,
            )
            self._artifact_ledger.record_artifact(risk_artifact)
            artifacts.extend((trade_artifact, risk_artifact))
            if not risk_result.approved:
                self._artifact_ledger.record_action(
                    action="execution_skipped",
                    status="blocked",
                    affects=("execution_queue", "portfolio", intent.symbol_id),
                    details={
                        "subject_id": subject_id,
                        "allowed_notional": round(risk_result.allowed_notional, 2),
                        "violations": [violation.code for violation in risk_result.violations],
                    },
                    occurred_at=intent.generated_at,
                )
                continue

            execution_result = self._executor.submit_trade_intent(intent)
            execution_results.append(execution_result)
            self._artifact_ledger.record_action(
                action="execution_recorded",
                status=execution_result.status.value,
                affects=("execution_results", "orders_audit", intent.symbol_id),
                details={
                    "client_order_id": execution_result.request.client_order_id,
                    "side": intent.side,
                    "filled_quantity": (
                        execution_result.fill.filled_quantity
                        if execution_result.fill is not None
                        else 0.0
                    ),
                },
                occurred_at=execution_result.completed_at,
                count_updates={"executions": 1},
            )
            if execution_result.fill is not None and execution_result.is_successful:
                portfolio_before_fill = working_portfolio
                closed_position = portfolio_before_fill.position_for_symbol(
                    intent.symbol_id,
                    position_id=intent.position_id,
                )
                self._portfolio_provider.record_fill(
                    symbol_id=intent.symbol_id,
                    side=intent.side,
                    quantity=execution_result.fill.filled_quantity,
                    price=execution_result.fill.average_price,
                    filled_at=execution_result.fill.filled_at,
                    position_id=intent.position_id,
                    source_signal_id=intent.signal_id,
                    raw_event_id=intent.raw_event_id,
                    event_type=intent.event_type,
                    exit_horizon_label=intent.exit_horizon_label,
                    max_hold_minutes=intent.max_hold_minutes,
                    exit_due_at=intent.exit_due_at,
                    confidence_score=intent.confidence_score,
                    expected_move=intent.expected_move,
                    selection_rank=intent.selection_rank,
                    selection_composite_score=intent.selection_composite_score,
                    rejected_alternatives=intent.rejected_alternatives,
                    heuristic_version=(
                        intent.heuristic_version or self._strategy.heuristic_version
                    ),
                )
                working_portfolio = self._portfolio_provider.get_portfolio_snapshot()
                journal_entry = TradeJournalEntry.from_execution_result(
                    execution_result=execution_result,
                    before_portfolio=portfolio_before_fill,
                    after_portfolio=working_portfolio,
                    notes=intent.rationale,
                    heuristic_version=(
                        intent.heuristic_version
                        or (closed_position.heuristic_version if closed_position else None)
                        or self._strategy.heuristic_version
                    ),
                )
                if (
                    closed_position is not None
                    and journal_entry.event_type in {"partial_exit", "full_exit", "reverse"}
                ):
                    review = evaluate_post_trade_review(
                        journal_entry=journal_entry,
                        opened_at=closed_position.opened_at,
                        latest_quotes=self._latest_quotes,
                        rejected_alternatives=(
                            closed_position.rejected_alternatives
                            or intent.rejected_alternatives
                        ),
                    )
                    self._last_post_trade_reviews.append(review)
                    journal_entry = replace(
                        journal_entry,
                        timing_label=review.timing_label,
                        asset_selection_label=review.asset_selection_label,
                        best_alternative_symbol=review.best_alternative_symbol,
                        best_alternative_return_fraction=(
                            review.best_alternative_return_fraction
                        ),
                        learning_reason_codes=review.reason_codes,
                        notes=journal_entry.notes + review.notes,
                    )
                self._trade_journal.record_entry(journal_entry)
                if (
                    closed_position is not None
                    and journal_entry.event_type in {"partial_exit", "full_exit", "reverse"}
                ):
                    signal_outcome_artifact = ValidationArtifact.from_signal_outcome(
                        execution_result,
                        symbol_id=closed_position.symbol_id,
                        side=closed_position.side,
                        entry_price=closed_position.entry_price,
                        opened_at=closed_position.opened_at,
                        exit_horizon_label=(
                            closed_position.exit_horizon_label or intent.exit_horizon_label
                        ),
                        raw_event_id=closed_position.raw_event_id or intent.raw_event_id,
                        signal_id=(
                            closed_position.source_signal_id or intent.signal_id
                        ),
                        event_type=closed_position.event_type or intent.event_type,
                        realized_pnl_usd=journal_entry.realized_pnl_usd,
                        agent_id=self._identity.agent_id,
                    )
                    self._artifact_ledger.record_artifact(signal_outcome_artifact)
                    artifacts.append(signal_outcome_artifact)
                self._artifact_ledger.record_action(
                    action="portfolio_updated",
                    status="recorded",
                    affects=("portfolio", "trade_journal", intent.symbol_id),
                    details={
                        "symbol_id": intent.symbol_id,
                        "side": intent.side,
                        "cash_usd": round(working_portfolio.cash_usd, 2),
                        "total_equity": round(working_portfolio.total_equity, 2),
                        "open_position_count": working_portfolio.open_position_count(),
                    },
                    occurred_at=execution_result.fill.filled_at,
                    summary_updates={
                        "portfolio": {
                            "total_equity": working_portfolio.total_equity,
                            "cash_usd": working_portfolio.cash_usd,
                            "open_position_count": working_portfolio.open_position_count(),
                            "realized_pnl_today": working_portfolio.realized_pnl_today,
                            "as_of": working_portfolio.as_of.isoformat(),
                        }
                    },
                )
            execution_artifact = ValidationArtifact.from_execution_result(
                execution_result,
                agent_id=self._identity.agent_id,
            )
            self._artifact_ledger.record_artifact(execution_artifact)
            artifacts.append(execution_artifact)

        return execution_results, artifacts

    def _new_detected_events(self) -> tuple[DetectedEvent, ...]:
        unseen: list[DetectedEvent] = []
        for event in self._event_repository.list_detected_events():
            event_key = (event.raw_event_id, event.event_type, event.rule_name)
            if event_key in self._handled_event_keys:
                continue
            self._handled_event_keys.add(event_key)
            unseen.append(event)
        return tuple(unseen)


def _build_signal_discovery_summary(
    artifacts: tuple[ValidationArtifact, ...] | list[ValidationArtifact],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total_outcomes": 0,
        "by_horizon": {},
        "no_trade_count": 0,
        "no_trade_by_reason": {},
        "recent_no_trade_decisions": [],
    }
    for artifact in artifacts:
        if artifact.kind == ArtifactKind.NO_TRADE_DECISION:
            payload = artifact.payload
            reason_code = str(payload.get("reason_code") or "unknown")
            summary["no_trade_count"] = int(summary.get("no_trade_count", 0)) + 1
            no_trade_by_reason = cast(dict[str, int], summary["no_trade_by_reason"])
            no_trade_by_reason[reason_code] = int(no_trade_by_reason.get(reason_code, 0)) + 1
            cast(list[dict[str, Any]], summary["recent_no_trade_decisions"]).append(
                {
                    "symbol_id": str(payload.get("symbol_id") or "-"),
                    "event_type": str(payload.get("event_type") or "-"),
                    "reason_code": reason_code,
                    "confidence_score": float(payload.get("confidence_score") or 0.0),
                    "threshold": payload.get("threshold"),
                    "detected_at": str(
                        payload.get("detected_at") or artifact.created_at.isoformat()
                    ),
                }
            )
            continue
        if artifact.kind != ArtifactKind.SIGNAL_OUTCOME:
            continue
        payload = artifact.payload
        horizon = str(payload.get("exit_horizon_label") or "unlabeled")
        bucket = cast(
            dict[str, Any],
            summary["by_horizon"].setdefault(
                horizon,
                {
                    "sample_count": 0,
                    "win_count": 0,
                    "loss_count": 0,
                    "flat_count": 0,
                    "avg_realized_pnl_usd": 0.0,
                    "avg_return_fraction": 0.0,
                    "event_types": {},
                },
            ),
        )
        realized_pnl = float(payload.get("realized_pnl_usd", 0.0))
        return_fraction = float(payload.get("realized_return_fraction", 0.0))
        event_type = str(payload.get("event_type") or "unknown")
        bucket["sample_count"] += 1
        summary["total_outcomes"] += 1
        bucket["avg_realized_pnl_usd"] += realized_pnl
        bucket["avg_return_fraction"] += return_fraction
        bucket["event_types"][event_type] = int(bucket["event_types"].get(event_type, 0)) + 1
        if realized_pnl > 0:
            bucket["win_count"] += 1
        elif realized_pnl < 0:
            bucket["loss_count"] += 1
        else:
            bucket["flat_count"] += 1

    for bucket in cast(dict[str, dict[str, Any]], summary["by_horizon"]).values():
        sample_count = int(bucket.get("sample_count", 0))
        if sample_count <= 0:
            continue
        bucket["avg_realized_pnl_usd"] = round(
            float(bucket["avg_realized_pnl_usd"]) / sample_count,
            2,
        )
        bucket["avg_return_fraction"] = round(
            float(bucket["avg_return_fraction"]) / sample_count,
            6,
        )
        bucket["win_rate"] = round(float(bucket["win_count"]) / sample_count, 4)

    return summary


def build_identity_registry(
    *,
    runtime_mode: str = "local",
    env: Mapping[str, str] | None = None,
) -> IdentityRegistry:
    if runtime_mode == "local":
        return LocalERC8004Registry()
    if runtime_mode == "sepolia":
        return OnChainERC8004Registry(config=SepoliaContractsConfig.from_env(env))
    raise ValueError(f"Unsupported runtime mode: {runtime_mode}")


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    parsed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        cleaned_key = key.strip()
        cleaned_value = value.strip()
        if (
            len(cleaned_value) >= 2
            and cleaned_value[0] == cleaned_value[-1]
            and cleaned_value[0] in {"\"", "'"}
        ):
            cleaned_value = cleaned_value[1:-1]
        if cleaned_key:
            parsed[cleaned_key] = cleaned_value
    return parsed


def _normalize_trading_mode(trading_mode: str | None) -> str:
    resolved = str(trading_mode or "paper").strip().lower().replace("_", "-")
    aliases = {
        "paper": "paper",
        "live": "live",
    }
    if resolved not in aliases:
        raise ValueError(
            "Unsupported trading mode: "
            f"{trading_mode!r}. Use `paper` or `live`."
        )
    return aliases[resolved]


def _normalize_identity_layer(identity_layer: str | None) -> str:
    resolved = str(identity_layer or "none").strip().lower().replace("_", "-")
    aliases = {
        "none": "none",
        "off": "none",
        "local": "none",
        "erc8004": "erc8004",
        "sepolia": "erc8004",
    }
    if resolved not in aliases:
        raise ValueError(
            "Unsupported identity layer: "
            f"{identity_layer!r}. Use `none` or `erc8004`."
        )
    return aliases[resolved]


def _resolve_identity_runtime(
    *,
    identity_layer: str | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    env_map = env if env is not None else os.environ
    candidate = identity_layer
    if candidate is None:
        env_layer = str(env_map.get("IDENTITY_LAYER", "")).strip()
        if env_layer:
            candidate = env_layer
    resolved_layer = _normalize_identity_layer(candidate)
    resolved_runtime_mode = "sepolia" if resolved_layer == "erc8004" else "local"
    return resolved_layer, resolved_runtime_mode


def _resolve_runtime_env(
    *,
    base_dir: str | Path,
    env: Mapping[str, str] | None = None,
    trading_mode: str | None = None,
) -> dict[str, str]:
    resolved_base_dir = Path(base_dir)
    merged: dict[str, str] = {}
    for env_file_name in RUNTIME_ENV_FILE_NAMES:
        merged.update(_read_env_file(resolved_base_dir / env_file_name))

    source = env if env is not None else os.environ
    merged.update({key: str(value) for key, value in source.items()})

    resolved_trading_mode = _normalize_trading_mode(
        trading_mode or merged.get("TRADING_MODE") or "paper"
    )
    merged["TRADING_MODE"] = resolved_trading_mode
    merged.setdefault("KRAKEN_EXECUTION_DRY_RUN", "false")
    merged.setdefault("KRAKEN_LIVE_ENABLED", "true")
    merged.setdefault(
        "KRAKEN_VALIDATE_ONLY",
        "true" if resolved_trading_mode == "paper" else "false",
    )
    return merged


def _load_agent_profile(env: Mapping[str, str] | None, runtime_mode: str) -> dict[str, Any]:
    env_map = env if env is not None else os.environ
    display_name = str(env_map.get("AGENT_DISPLAY_NAME", "AI Trading Agent Demo")).strip()
    strategy_name = str(
        env_map.get("AGENT_STRATEGY_NAME", "simple_event_driven")
    ).strip()
    owner = str(
        env_map.get("AGENT_OWNER", os.environ.get("USERNAME", "local-demo"))
    ).strip()

    metadata: dict[str, str] = {"mode": runtime_mode}
    if env_map.get("AGENT_URI"):
        metadata["agent_uri"] = str(env_map["AGENT_URI"]).strip()
    if env_map.get("AGENT_CAPABILITIES"):
        metadata["capabilities"] = str(env_map["AGENT_CAPABILITIES"]).strip()
    if env_map.get("AGENT_DESCRIPTION"):
        metadata["description"] = str(env_map["AGENT_DESCRIPTION"]).strip()

    return {
        "agent_display_name": display_name or "AI Trading Agent Demo",
        "agent_strategy_name": strategy_name or "simple_event_driven",
        "agent_owner": owner or os.environ.get("USERNAME", "local-demo"),
        "agent_metadata": metadata,
    }


def _storage_overrides_provided(
    *,
    runs_repository: IngestionRunsRepository | None,
    raw_events_repository: RawEventsRepository | None,
    event_repository: EventDetectionRepository | None,
    object_store: RawObjectStore | None,
) -> bool:
    components = {
        "runs_repository": runs_repository,
        "raw_events_repository": raw_events_repository,
        "event_repository": event_repository,
        "object_store": object_store,
    }
    provided = [name for name, value in components.items() if value is not None]
    if not provided:
        return False

    missing = [name for name, value in components.items() if value is None]
    if missing:
        raise ValueError(
            "Storage overrides must include runs_repository, raw_events_repository, "
            "event_repository, and object_store together."
        )
    return True


def _local_storage_requested(env: Mapping[str, str] | None) -> bool:
    env_map = env if env is not None else {}
    storage_backend = str(env_map.get("STORAGE_BACKEND", "")).strip().lower()
    return storage_backend in {"local", "file", "memory"}


def _build_local_storage(
    paths: RuntimePaths,
) -> tuple[
    str,
    LocalInMemoryIngestionRunsRepository,
    LocalInMemoryRawEventsRepository,
    LocalEventDetectionRepository,
    LocalFileObjectStore,
]:
    paths.raw_payload_dir.mkdir(parents=True, exist_ok=True)
    return (
        "local",
        LocalInMemoryIngestionRunsRepository(),
        LocalInMemoryRawEventsRepository(),
        LocalEventDetectionRepository(),
        LocalFileObjectStore(paths.raw_payload_dir),
    )


def _require_postgres_storage(
    env: Mapping[str, str] | None,
) -> tuple[
    str,
    PostgresIngestionRunsRepository,
    PostgresRawEventsRepository,
    EventDetectionRepository,
]:
    resolved_env = dict(env or {})
    connection_factory = postgres_connection_factory_from_env(resolved_env)
    if connection_factory is None:
        raise ValueError(
            "Postgres storage is required. Set POSTGRES_ENABLED=true and configure "
            "DATABASE_URL or POSTGRES_* before starting the trading runtime."
        )

    return (
        "postgres",
        PostgresIngestionRunsRepository(connection_factory),
        PostgresRawEventsRepository(connection_factory),
        PostgresEventDetectionRepository(connection_factory),
    )


def _require_object_store(
    env: Mapping[str, str] | None,
    *,
    env_path: Path | None = None,
) -> RawObjectStore:
    resolved_env = dict(env or {})
    try:
        config = ObjectStorageConfig.from_env(
            env=resolved_env,
            env_path=env_path or ROOT_DIR,
        )
    except ValueError as exc:
        raise ValueError(
            "Cloudflare R2 / object storage is required for raw payload persistence. "
            "Configure CF_R2_BUCKET, CF_R2_ENDPOINT, CF_R2_ACCESS_KEY, and "
            "CF_R2_SECRET_KEY."
        ) from exc
    return S3CompatibleObjectStore.from_config(config)


def _build_execution_config(
    *,
    paths: RuntimePaths,
    env: Mapping[str, str] | None = None,
) -> KrakenCLIConfig:
    config = KrakenCLIConfig.from_env(env)
    if env is not None and env.get("KRAKEN_AUDIT_LOG_PATH"):
        return config
    return replace(config, audit_log_path=paths.audit_log_path)


def build_local_demo_app(
    *,
    base_dir: str | Path = ROOT_DIR,
    feed_groups: dict[str, list[FeedSource]] | None = None,
    symbols: list[PriceSymbol] | None = None,
    secondary_symbols: list[PriceSymbol] | None = None,
    tier_promotion_config: TierPromotionConfig | None = None,
    parse_feed: Any | None = None,
    http_get: Any | None = None,
    trading_mode: str = "paper",
    identity_layer: str | None = None,
    env: Mapping[str, str] | None = None,
    execution_config: KrakenCLIConfig | None = None,
    risk_config: RiskConfig | None = None,
    max_positions: int = 3,
    max_per_asset: int = 1,
    execution_runner: CommandRunner | None = None,
    runs_repository: IngestionRunsRepository | None = None,
    raw_events_repository: RawEventsRepository | None = None,
    event_repository: EventDetectionRepository | None = None,
    object_store: RawObjectStore | None = None,
) -> TradingApplication:
    resolved_trading_mode = _normalize_trading_mode(trading_mode)
    runtime_env = _resolve_runtime_env(
        base_dir=base_dir,
        env=env,
        trading_mode=resolved_trading_mode,
    )
    resolved_identity_layer, resolved_runtime_mode = _resolve_identity_runtime(
        identity_layer=identity_layer,
        env=runtime_env,
    )
    runtime_env["IDENTITY_LAYER"] = resolved_identity_layer
    paths = RuntimePaths.from_base_dir(base_dir)
    agent_profile = _load_agent_profile(runtime_env, resolved_runtime_mode)
    resolved_risk_config = risk_config or RiskConfig(
        max_concurrent_positions=max_positions,
        max_positions_per_asset=max_per_asset,
    )
    resolved_symbols = symbols or PRICE_SYMBOLS
    resolved_secondary_symbols = (
        secondary_symbols
        if secondary_symbols is not None
        else SECONDARY_PRICE_SYMBOLS
        if symbols is None
        else None
    )
    resolved_tier_promotion_config = (
        tier_promotion_config or TierPromotionConfig.from_env(runtime_env)
    )
    return TradingApplication(
        paths=paths,
        feed_groups=feed_groups or RSS_FEED_GROUPS,
        symbols=resolved_symbols,
        secondary_symbols=resolved_secondary_symbols,
        tier_promotion_config=resolved_tier_promotion_config,
        parse_feed=parse_feed,
        http_get=http_get,
        identity_registry=build_identity_registry(
            runtime_mode=resolved_runtime_mode,
            env=runtime_env,
        ),
        execution_config=execution_config or _build_execution_config(paths=paths, env=runtime_env),
        risk_config=resolved_risk_config,
        execution_runner=execution_runner,
        runtime_env=runtime_env,
        runtime_mode=resolved_runtime_mode,
        trading_mode=resolved_trading_mode,
        identity_layer=resolved_identity_layer,
        runs_repository=runs_repository,
        raw_events_repository=raw_events_repository,
        event_repository=event_repository,
        object_store=object_store,
        **agent_profile,
    )


def build_runtime_preflight(
    *,
    trading_mode: str = "paper",
    identity_layer: str | None = None,
    base_dir: str | Path = ROOT_DIR,
    env: Mapping[str, str] | None = None,
    require_transaction_keys: bool = False,
) -> dict[str, Any]:
    resolved_trading_mode = _normalize_trading_mode(trading_mode)
    env_map = _resolve_runtime_env(
        base_dir=base_dir,
        env=env,
        trading_mode=resolved_trading_mode,
    )
    resolved_identity_layer, resolved_runtime_mode = _resolve_identity_runtime(
        identity_layer=identity_layer,
        env=env_map,
    )

    paths = RuntimePaths.from_base_dir(base_dir)
    issues: list[str] = []
    try:
        paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
        paths.raw_payload_dir.mkdir(parents=True, exist_ok=True)
        probe_path = paths.artifacts_dir / ".write_probe"
        probe_path.write_text("ok", encoding="utf-8")
        probe_path.unlink(missing_ok=True)
        artifacts_writable = True
    except OSError:
        artifacts_writable = False
        issues.append(f"Artifacts directory is not writable: {paths.artifacts_dir}")

    execution_config = KrakenCLIConfig.from_env(env_map)
    local_storage_mode = _local_storage_requested(env_map)
    kraken_credentials_present = bool(str(env_map.get("KRAKEN_API_KEY", "")).strip()) and bool(
        str(env_map.get("KRAKEN_API_SECRET", "")).strip()
    )
    live_submit_enabled = str(env_map.get("KRAKEN_CLI_ALLOW_LIVE_SUBMIT", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    will_submit_real_orders = (
        execution_config.live_enabled
        and not execution_config.dry_run
        and not execution_config.validate_only
    )
    live_connected_paper = (
        execution_config.live_enabled
        and not execution_config.dry_run
        and execution_config.validate_only
    )
    if will_submit_real_orders and not live_submit_enabled:
        issues.append("Kraken live trading requires `KRAKEN_CLI_ALLOW_LIVE_SUBMIT=true`.")

    postgres_connection_factory = (
        None
        if local_storage_mode
        else postgres_connection_factory_from_env(env_map)
    )
    postgres_configured = postgres_connection_factory is not None
    postgres_reachable = None
    postgres_error = None
    if not local_storage_mode and postgres_connection_factory is None:
        issues.append(
            "Postgres storage is required. Set POSTGRES_ENABLED=true and configure "
            "DATABASE_URL or POSTGRES_* before starting the trading runtime."
        )

    object_store_configured = False
    object_store_error = None
    if local_storage_mode:
        object_store_configured = True
    else:
        try:
            ObjectStorageConfig.from_env(env=env_map, env_path=paths.base_dir)
            object_store_configured = True
        except ValueError as exc:
            object_store_error = str(exc)
            issues.append(
                "Cloudflare R2 / object storage is required for raw payload persistence. "
                "Configure CF_R2_BUCKET, CF_R2_ENDPOINT, CF_R2_ACCESS_KEY, and "
                "CF_R2_SECRET_KEY."
            )

    if (
        not execution_config.dry_run
        and execution_config.live_enabled
        and not kraken_credentials_present
    ):
        issues.append(
            "Kraken paper/live trading requires `KRAKEN_API_KEY` and `KRAKEN_API_SECRET`."
        )
    if require_transaction_keys and resolved_identity_layer != "erc8004":
        issues.append("On-chain identity actions require `--identity-layer erc8004`.")

    sepolia_missing: list[str] = []
    if resolved_runtime_mode == "sepolia":
        config = SepoliaContractsConfig.from_env(env_map)
        sepolia_missing = list(config.missing_required_values())
        if not require_transaction_keys:
            sepolia_missing = [
                key
                for key in sepolia_missing
                if key not in {"PRIVATE_KEY", "AGENT_WALLET_PRIVATE_KEY"}
            ]
        if sepolia_missing:
            issues.append(
                "Missing required Sepolia configuration: " + ", ".join(sorted(sepolia_missing))
            )
        invalid_keys: list[str] = []
        if config.private_key and config.operator_wallet_address is None:
            invalid_keys.append("PRIVATE_KEY")
        if config.agent_wallet_private_key and config.agent_wallet_address is None:
            invalid_keys.append("AGENT_WALLET_PRIVATE_KEY")
        if invalid_keys:
            issues.append("Unable to derive wallet addresses from: " + ", ".join(invalid_keys))

    if postgres_connection_factory is not None:
        postgres_reachable, postgres_error = probe_postgres_connection(
            postgres_connection_factory
        )
        if not postgres_reachable and postgres_error:
            issues.append(f"Postgres is configured but not reachable: {postgres_error}")

    return {
        "status": "ready" if not issues else "error",
        "trading_mode": resolved_trading_mode,
        "identity_layer": resolved_identity_layer,
        "runtime_mode": resolved_runtime_mode,
        "checks": {
            "artifacts_writable": artifacts_writable,
            "kraken_credentials_present": kraken_credentials_present,
            "live_submit_enabled": live_submit_enabled,
            "live_connected_paper_trading": live_connected_paper,
            "will_submit_real_orders": will_submit_real_orders,
            "sepolia_missing_required_values": sepolia_missing,
            "postgres_configured": postgres_configured,
            "postgres_reachable": postgres_reachable,
            "object_store_configured": object_store_configured,
        },
        "execution_config": {
            "kraken_cli_executable": execution_config.executable,
            "kraken_dry_run": execution_config.dry_run,
            "kraken_live_enabled": execution_config.live_enabled,
            "kraken_validate_only": execution_config.validate_only,
            "storage_backend": (
                "local"
                if local_storage_mode
                else "postgres"
                if postgres_configured
                else "unconfigured"
            ),
            "object_store_backend": (
                "local"
                if local_storage_mode
                else "r2"
                if object_store_configured
                else "unconfigured"
            ),
            "postgres_error": postgres_error,
            "object_store_error": object_store_error,
        },
        "issues": issues,
    }


def validate_runtime_requirements(
    *,
    trading_mode: str = "paper",
    identity_layer: str | None = None,
    base_dir: str | Path,
    env: Mapping[str, str] | None = None,
    require_transaction_keys: bool = False,
) -> None:
    report = build_runtime_preflight(
        trading_mode=trading_mode,
        identity_layer=identity_layer,
        base_dir=base_dir,
        env=env,
        require_transaction_keys=require_transaction_keys,
    )
    if report["issues"]:
        raise ValueError(str(report["issues"][0]))


def _reset_local_runtime_artifacts(paths: RuntimePaths) -> tuple[int, tuple[str, ...]]:
    removed_paths: list[str] = []
    removable_files = (
        paths.audit_log_path,
        paths.journal_log_path,
        paths.artifacts_log_path,
        paths.checkpoints_log_path,
        paths.activity_log_path,
        paths.summary_path,
    )
    for file_path in removable_files:
        if file_path.exists():
            file_path.unlink()
            removed_paths.append(str(file_path))

    if paths.raw_payload_dir.exists():
        shutil.rmtree(paths.raw_payload_dir)
        removed_paths.append(str(paths.raw_payload_dir))

    return len(removed_paths), tuple(removed_paths)


def _reset_postgres_runtime_tables(connection_factory: Any) -> None:
    query = """
    TRUNCATE TABLE
        detected_events,
        raw_events,
        ingestion_runs
    RESTART IDENTITY CASCADE
    """
    with connection_factory() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
        connection.commit()


def reset_runtime_state(
    *,
    base_dir: str | Path = ROOT_DIR,
    env: Mapping[str, str] | None = None,
    reset_postgres: bool = True,
    reset_object_store: bool = True,
    clear_runtime_env: bool = True,
) -> dict[str, Any]:
    resolved_env = _resolve_runtime_env(
        base_dir=base_dir,
        env=env,
        trading_mode=(str((env or {}).get("TRADING_MODE", "")).strip() or None),
    )
    paths = RuntimePaths.from_base_dir(base_dir)

    local_artifacts_removed, local_removed_paths = _reset_local_runtime_artifacts(paths)

    runtime_env_path = Path(base_dir) / ".runtime.env"
    runtime_env_removed = False
    if clear_runtime_env and runtime_env_path.exists():
        runtime_env_path.unlink()
        runtime_env_removed = True

    issues: list[str] = []
    local_storage_mode = _local_storage_requested(resolved_env)
    postgres_reset = False
    postgres_error = None
    if reset_postgres and not local_storage_mode:
        connection_factory = postgres_connection_factory_from_env(resolved_env)
        if connection_factory is None:
            postgres_error = (
                "Postgres storage is required to reset the runtime state. Set "
                "POSTGRES_ENABLED=true and configure DATABASE_URL or POSTGRES_*."
            )
            issues.append(postgres_error)
        else:
            reachable, error = probe_postgres_connection(connection_factory)
            if not reachable:
                postgres_error = f"Unable to reset Postgres runtime state: {error}"
                issues.append(postgres_error)
            else:
                _reset_postgres_runtime_tables(connection_factory)
                postgres_reset = True

    deleted_object_keys = 0
    object_store_reset = False
    object_store_error = None
    if reset_object_store and not local_storage_mode:
        try:
            config = ObjectStorageConfig.from_env(
                env=resolved_env,
                env_path=Path(base_dir),
            )
            object_store = S3CompatibleObjectStore.from_config(config)
            deleted_object_keys += object_store.delete_prefix(prefix="raw/")
            deleted_object_keys += object_store.delete_prefix(prefix="healthcheck/")
            object_store_reset = True
        except Exception as exc:
            object_store_error = f"Unable to reset object storage state: {exc}"
            issues.append(object_store_error)

    return {
        "status": "reset" if not issues else "partial",
        "base_dir": str(Path(base_dir)),
        "local_artifacts_removed": local_artifacts_removed,
        "local_removed_paths": list(local_removed_paths),
        "runtime_env_removed": runtime_env_removed,
        "postgres_reset": postgres_reset,
        "postgres_error": postgres_error,
        "object_store_reset": object_store_reset,
        "object_store_deleted_keys": deleted_object_keys,
        "object_store_error": object_store_error,
        "issues": issues,
    }


def run_scheduler_service(
    app: TradingApplication,
    *,
    feed_group: str = "market_news",
    rss_interval_seconds: int = 120,
    prices_interval_seconds: int = 60,
    secondary_prices_interval_seconds: int = SECONDARY_POLLING_FREQUENCY_SECONDS,
    detection_interval_seconds: int = 60,
    execution_interval_seconds: int = 60,
) -> None:
    scheduler = app.wire_scheduler(
        feed_group=feed_group,
        rss_interval_seconds=rss_interval_seconds,
        prices_interval_seconds=prices_interval_seconds,
        secondary_prices_interval_seconds=secondary_prices_interval_seconds,
        detection_interval_seconds=detection_interval_seconds,
        execution_interval_seconds=execution_interval_seconds,
    )
    stop_event = threading.Event()
    handled_signals = [signal.SIGINT]
    if hasattr(signal, "SIGTERM"):
        handled_signals.append(signal.SIGTERM)

    previous_handlers: dict[signal.Signals, Any] = {}

    def _request_shutdown(signum: int, _frame: Any) -> None:
        print(
            json.dumps(
                {
                    "event": "shutdown_requested",
                    "signal": signum,
                    "runtime_mode": getattr(app, "_runtime_mode", "unknown"),
                },
                sort_keys=True,
            )
        )
        stop_event.set()

    for signum in handled_signals:
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, _request_shutdown)

    scheduler.start()
    print(
        json.dumps(
            {
                "event": "scheduler_started",
                "feed_group": feed_group,
                "rss_interval_seconds": rss_interval_seconds,
                "prices_interval_seconds": prices_interval_seconds,
                "secondary_prices_interval_seconds": secondary_prices_interval_seconds,
                "detection_interval_seconds": detection_interval_seconds,
                "execution_interval_seconds": execution_interval_seconds,
            },
            sort_keys=True,
        )
    )

    try:
        while not stop_event.wait(0.5):
            continue
    finally:
        scheduler.shutdown(wait=True)
        for signum, previous_handler in previous_handlers.items():
            signal.signal(signum, previous_handler)


def _build_cli_parser(
    *,
    default_env: Mapping[str, str] | None = None,
) -> argparse.ArgumentParser:
    resolved_defaults = dict(default_env or {})
    parser = argparse.ArgumentParser(
        description=(
            "Run the trading demo in Kraken paper or Kraken live mode, with optional "
            "ERC-8004 identity wiring."
        )
    )
    parser.add_argument(
        "--base-dir",
        default=str(ROOT_DIR),
        help="Directory for demo artifacts.",
    )
    parser.add_argument(
        "--feed-group",
        default="market_news",
        help="RSS feed group to ingest during the one-shot Kraken cycle.",
    )
    parser.add_argument(
        "--trading-mode",
        choices=("paper", "live"),
        default=resolved_defaults.get("TRADING_MODE", "paper"),
        help="Kraken trading mode: `paper` validates orders, `live` can submit real orders.",
    )
    parser.add_argument(
        "--identity-layer",
        choices=("none", "erc8004"),
        default=("erc8004" if resolved_defaults.get("IDENTITY_LAYER") == "erc8004" else "none"),
        help="Optional identity layer for shared ERC-8004 / Sepolia actions.",
    )
    parser.add_argument(
        "--register-agent",
        action="store_true",
        help=(
            "Register the agent on the shared `AgentRegistry` if no numeric "
            "`AGENT_ID` exists yet."
        ),
    )
    parser.add_argument(
        "--claim-allocation",
        action="store_true",
        help="Call `HackathonVault.claimAllocation(agentId)` after the local run completes.",
    )
    parser.add_argument(
        "--submit-onchain",
        action="store_true",
        help="Simulate and submit generated trade intents through the shared `RiskRouter`.",
    )
    parser.add_argument(
        "--post-checkpoints",
        action="store_true",
        help="Post generated checkpoints to the shared `ValidationRegistry`.",
    )
    parser.add_argument(
        "--full-flow",
        action="store_true",
        help=(
            "Run the full shared-contract path: register, claim, submit intents, "
            "and post checkpoints."
        ),
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run the recurring scheduler instead of a one-shot cycle.",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=int(resolved_defaults.get("OPPORTUNITY_MAX_POSITIONS", "3")),
        help="Opportunity budget: maximum concurrent symbols the agent may hold.",
    )
    parser.add_argument(
        "--max-per-asset",
        type=int,
        default=int(resolved_defaults.get("OPPORTUNITY_MAX_PER_ASSET", "1")),
        help="Opportunity budget: maximum active position slots allowed per asset.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate Kraken and optional ERC-8004 configuration without running a cycle.",
    )
    parser.add_argument(
        "--reset-storage",
        action="store_true",
        help=(
            "Delete local runtime artifacts and clear the configured Postgres/R2 trading "
            "state for a blank-slate restart."
        ),
    )
    parser.add_argument(
        "--rss-interval-seconds",
        type=int,
        default=int(resolved_defaults.get("RSS_INTERVAL_SECONDS", "120")),
        help="Scheduler interval for RSS ingestion jobs.",
    )
    parser.add_argument(
        "--prices-interval-seconds",
        type=int,
        default=int(resolved_defaults.get("PRICES_INTERVAL_SECONDS", "60")),
        help="Scheduler interval for core Tier A price ingestion jobs.",
    )
    parser.add_argument(
        "--secondary-prices-interval-seconds",
        type=int,
        default=int(
            resolved_defaults.get(
                "SECONDARY_PRICES_INTERVAL_SECONDS",
                str(SECONDARY_POLLING_FREQUENCY_SECONDS),
            )
        ),
        help="Scheduler interval for Tier B event-driven price ingestion jobs.",
    )
    parser.add_argument(
        "--detection-interval-seconds",
        type=int,
        default=int(resolved_defaults.get("DETECTION_INTERVAL_SECONDS", "60")),
        help="Scheduler interval for event detection jobs.",
    )
    parser.add_argument(
        "--execution-interval-seconds",
        type=int,
        default=int(resolved_defaults.get("EXECUTION_INTERVAL_SECONDS", "60")),
        help="Scheduler interval for trade execution jobs.",
    )
    return parser


def main() -> int:
    base_dir_parser = argparse.ArgumentParser(add_help=False)
    base_dir_parser.add_argument("--base-dir", default=str(ROOT_DIR))
    bootstrap_args, _ = base_dir_parser.parse_known_args()
    default_env = _resolve_runtime_env(
        base_dir=bootstrap_args.base_dir,
        env=os.environ,
        trading_mode=(os.environ.get("TRADING_MODE") or None),
    )
    parser = _build_cli_parser(default_env=default_env)
    args = parser.parse_args()

    try:
        if args.full_flow:
            args.register_agent = True
            args.claim_allocation = True
            args.submit_onchain = True
            args.post_checkpoints = True

        require_transaction_keys = any(
            (
                args.register_agent,
                args.claim_allocation,
                args.submit_onchain,
                args.post_checkpoints,
            )
        )
        resolved_trading_mode = args.trading_mode
        resolved_env = _resolve_runtime_env(
            base_dir=args.base_dir,
            env=os.environ,
            trading_mode=resolved_trading_mode,
        )
        if args.reset_storage:
            report = reset_runtime_state(
                base_dir=args.base_dir,
                env=resolved_env,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["status"] == "reset" else 2
        if args.preflight:
            report = build_runtime_preflight(
                trading_mode=resolved_trading_mode,
                identity_layer=args.identity_layer,
                base_dir=args.base_dir,
                env=resolved_env,
                require_transaction_keys=require_transaction_keys,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["status"] == "ready" else 2
        validate_runtime_requirements(
            trading_mode=resolved_trading_mode,
            identity_layer=args.identity_layer,
            base_dir=args.base_dir,
            env=resolved_env,
            require_transaction_keys=require_transaction_keys,
        )
        app = build_local_demo_app(
            base_dir=args.base_dir,
            trading_mode=resolved_trading_mode,
            identity_layer=args.identity_layer,
            env=resolved_env,
            max_positions=args.max_positions,
            max_per_asset=args.max_per_asset,
        )

        if args.serve:
            if require_transaction_keys:
                raise ValueError(
                    "`--serve` currently supports the recurring ingest/detect/execute "
                    "loop only. Run one-shot commands for on-chain actions."
                )
            run_scheduler_service(
                app,
                feed_group=args.feed_group,
                rss_interval_seconds=args.rss_interval_seconds,
                prices_interval_seconds=args.prices_interval_seconds,
                secondary_prices_interval_seconds=args.secondary_prices_interval_seconds,
                detection_interval_seconds=args.detection_interval_seconds,
                execution_interval_seconds=args.execution_interval_seconds,
            )
            return 0

        result = app.run_cycle(feed_group=args.feed_group)
        payload = result.to_dict()
        payload["trading_mode"] = getattr(app, "_trading_mode", resolved_trading_mode)
        payload["identity_layer"] = getattr(app, "_identity_layer", args.identity_layer)
        payload["runtime_mode"] = getattr(app, "_runtime_mode", "local")
        payload["execution_config"] = app.execution_mode_summary()
        if getattr(app, "_identity_layer", "none") == "erc8004":
            payload["shared_contracts"] = app.run_shared_contract_actions(
                trade_intents=result.trade_intents,
                checkpoints=result.checkpoints,
                register_agent=args.register_agent,
                claim_allocation=args.claim_allocation,
                submit_trade_intents=args.submit_onchain,
                post_checkpoints=args.post_checkpoints,
            )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except ValueError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True))
        return 2
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
