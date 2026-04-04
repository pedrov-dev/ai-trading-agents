"""Local-first dry-run entrypoint for the hackathon trading agent."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.portfolio import LocalPortfolioStateProvider, PortfolioSnapshot
from agent.risk import RiskCheckResult
from agent.signals import TradeIntent
from agent.strategy import SimpleEventDrivenStrategy
from detection.event_detection import DetectedEvent, RuleBasedEventDetector
from detection.event_detection_service import EventDetectionService
from execution.kraken_cli import (
    DEFAULT_AUDIT_LOG_PATH,
    KrakenCLIConfig,
    KrakenCLIExecutor,
)
from execution.orders import ExecutionResult
from identity.erc8004_registry import (
    AgentIdentity,
    IdentityRegistry,
    LocalERC8004Registry,
    OnChainERC8004Registry,
    SepoliaContractsConfig,
)
from identity.reputation import ReputationEngine, ReputationSnapshot
from info_scheduler import InfoScheduler
from ingestion.prices_config import PRICE_SYMBOLS, PriceSymbol
from ingestion.prices_ingestion import PriceQuote, PricesIngestionService
from ingestion.rss_config import RSS_FEED_GROUPS, FeedSource
from ingestion.rss_ingestion import RSSIngestionService
from monitoring.audit_log import AuditSummary, build_audit_summary_from_file
from monitoring.drawdown import DrawdownSnapshot, EquityPoint, build_drawdown_snapshot
from monitoring.pnl import PnLSnapshot, build_pnl_snapshot
from storage.local_runtime import (
    InMemoryEventDetectionRepository,
    InMemoryIngestionRunsRepository,
    InMemoryRawEventsRepository,
    JsonlFileStore,
    LocalFileObjectStore,
)
from storage.raw_ingestion import (
    IngestionRunResult,
    PricesRawIngestionPipeline,
    RSSRawIngestionPipeline,
)
from validation.artifacts import ValidationArtifact
from validation.checkpoints import ValidationCheckpoint, build_checkpoints

ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DryRunRuntimePaths:
    """Filesystem locations used by the local dry-run flow."""

    base_dir: Path
    artifacts_dir: Path
    raw_payload_dir: Path
    audit_log_path: Path
    artifacts_log_path: Path
    checkpoints_log_path: Path
    summary_path: Path

    @classmethod
    def from_base_dir(cls, base_dir: str | Path) -> DryRunRuntimePaths:
        resolved = Path(base_dir)
        artifacts_dir = resolved / "artifacts"
        return cls(
            base_dir=resolved,
            artifacts_dir=artifacts_dir,
            raw_payload_dir=artifacts_dir / "raw_payloads",
            audit_log_path=artifacts_dir / DEFAULT_AUDIT_LOG_PATH.name,
            artifacts_log_path=artifacts_dir / "validation_artifacts.jsonl",
            checkpoints_log_path=artifacts_dir / "validation_checkpoints.jsonl",
            summary_path=artifacts_dir / "run_summary.json",
        )


@dataclass(frozen=True)
class DryRunCycleResult:
    """Outcome of one local dry-run cycle across ingestion, detection, and execution."""

    rss_result: IngestionRunResult
    prices_result: IngestionRunResult
    classification_count: int
    detected_events: tuple[DetectedEvent, ...]
    trade_intents: tuple[TradeIntent, ...]
    execution_results: tuple[ExecutionResult, ...]
    artifacts: tuple[ValidationArtifact, ...]
    checkpoints: tuple[ValidationCheckpoint, ...]
    portfolio: PortfolioSnapshot
    pnl_snapshot: PnLSnapshot
    drawdown_snapshot: DrawdownSnapshot
    audit_summary: AuditSummary
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
                    "generated_at": intent.generated_at.isoformat(),
                    "rationale": list(intent.rationale),
                }
                for intent in self.trade_intents
            ],
            "execution_results": [result.to_dict() for result in self.execution_results],
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "checkpoints": [checkpoint.to_dict() for checkpoint in self.checkpoints],
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
            "reputation": self.reputation.to_dict(),
        }


class LocalArtifactLedger:
    """Persist dry-run outputs as simple JSONL and JSON summary files."""

    def __init__(self, paths: DryRunRuntimePaths) -> None:
        self._artifacts_store = JsonlFileStore(paths.artifacts_log_path)
        self._checkpoints_store = JsonlFileStore(paths.checkpoints_log_path)
        self._summary_store = JsonlFileStore(paths.summary_path)

    def record(
        self,
        *,
        artifacts: tuple[ValidationArtifact, ...],
        checkpoints: tuple[ValidationCheckpoint, ...],
        summary: Mapping[str, Any],
    ) -> None:
        self._artifacts_store.append_many(artifact.to_dict() for artifact in artifacts)
        self._checkpoints_store.append_many(
            checkpoint.to_dict() for checkpoint in checkpoints
        )
        self._summary_store.write_json(summary)


class DryRunApplication:
    """Local demo harness for the event-driven paper-trading workflow."""

    def __init__(
        self,
        *,
        paths: DryRunRuntimePaths,
        feed_groups: dict[str, list[FeedSource]],
        symbols: list[PriceSymbol],
        parse_feed: Any | None = None,
        http_get: Any | None = None,
        scheduler: InfoScheduler | None = None,
        identity_registry: IdentityRegistry | None = None,
        runtime_mode: str = "local",
    ) -> None:
        self.paths = paths
        self._scheduler = scheduler or InfoScheduler()
        self._runs_repository = InMemoryIngestionRunsRepository()
        self._raw_events_repository = InMemoryRawEventsRepository()
        self._object_store = LocalFileObjectStore(paths.raw_payload_dir)
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
        self._prices_service = PricesIngestionService(
            symbols=symbols,
            http_get=http_get,
        )
        self._event_repository = InMemoryEventDetectionRepository()
        self._detection_service = EventDetectionService(
            detector=RuleBasedEventDetector(),
            raw_events_repository=self._raw_events_repository,
            event_detection_repository=self._event_repository,
        )
        self._strategy = SimpleEventDrivenStrategy()
        self._executor = KrakenCLIExecutor(
            config=KrakenCLIConfig(
                dry_run=True,
                live_enabled=False,
                audit_log_path=paths.audit_log_path,
            )
        )
        self._portfolio_provider = LocalPortfolioStateProvider(
            starting_equity=10_000.0,
            starting_cash_usd=10_000.0,
        )
        self._runtime_mode = runtime_mode
        registry = identity_registry or LocalERC8004Registry()
        identity_metadata = {"mode": runtime_mode}
        if isinstance(registry, OnChainERC8004Registry):
            identity_metadata.update(
                {
                    "chain_id": str(registry.config.chain_id),
                    "agent_registry_address": registry.config.agent_registry_address,
                }
            )
        self._identity = registry.register(
            display_name="AI Trading Agent Demo",
            strategy_name="simple_event_driven",
            owner=os.environ.get("USERNAME", "local-demo"),
            exchange="sepolia" if runtime_mode == "sepolia" else "kraken",
            metadata=identity_metadata,
        )
        self._reputation_engine = ReputationEngine()
        self._reputation = self._reputation_engine.initialize(self._identity.agent_id)
        self._artifact_ledger = LocalArtifactLedger(paths)
        snapshot = self._portfolio_provider.get_portfolio_snapshot()
        self._equity_history: list[EquityPoint] = [
            EquityPoint(recorded_at=snapshot.as_of, equity=snapshot.total_equity)
        ]
        self._latest_quotes: list[PriceQuote] = []
        self._handled_event_keys: set[tuple[str, str, str]] = set()

    @property
    def identity(self) -> AgentIdentity:
        return self._identity

    def wire_scheduler(
        self,
        *,
        feed_group: str = "market_news",
        rss_interval_seconds: int = 120,
        prices_interval_seconds: int = 60,
        detection_interval_seconds: int = 60,
        execution_interval_seconds: int = 60,
    ) -> InfoScheduler:
        def _run_rss_job() -> None:
            self.ingest_rss_group(feed_group=feed_group)

        def _run_prices_job() -> None:
            self.ingest_prices()

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
        return self._rss_pipeline.persist_articles(
            source_group=feed_group,
            articles=unique_articles,
        )

    def ingest_prices(self) -> IngestionRunResult:
        quotes = self._prices_service.fetch_current_prices()
        self._latest_quotes = quotes
        return self._prices_pipeline.persist_quotes(quotes=quotes)

    def classify_events(self) -> int:
        return self._detection_service.classify_pending_events(
            source_type="rss",
            batch_size=100,
        )

    def run_cycle(self, *, feed_group: str = "market_news") -> DryRunCycleResult:
        rss_result = self.ingest_rss_group(feed_group=feed_group)
        prices_result = self.ingest_prices()
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
        classification_count: int = 0,
    ) -> DryRunCycleResult:
        current_portfolio = self._portfolio_provider.get_portfolio_snapshot()
        detected_events = self._new_detected_events()
        trade_intents = tuple(
            self._strategy.generate_trade_intents(
                detected_events=list(detected_events),
                price_quotes=list(self._latest_quotes),
                portfolio=current_portfolio,
            )
        )

        execution_results: list[ExecutionResult] = []
        artifacts: list[ValidationArtifact] = []

        for intent in trade_intents:
            subject_id = (
                f"{intent.symbol_id}:{intent.side}:{intent.generated_at.isoformat()}"
            )
            trade_artifact = ValidationArtifact.from_trade_intent(
                intent,
                agent_id=self._identity.agent_id,
            )
            risk_artifact = ValidationArtifact.from_risk_check(
                RiskCheckResult(
                    approved=True,
                    allowed_notional=intent.notional_usd,
                    notes=("Trade approved by the strategy risk manager.",),
                ),
                agent_id=self._identity.agent_id,
                subject_id=subject_id,
                proposed_notional=intent.notional_usd,
                checked_at=intent.generated_at,
            )
            execution_result = self._executor.submit_trade_intent(intent)
            execution_results.append(execution_result)
            if execution_result.fill is not None and execution_result.is_successful:
                self._portfolio_provider.record_fill(
                    symbol_id=intent.symbol_id,
                    side=intent.side,
                    quantity=execution_result.fill.filled_quantity,
                    price=execution_result.fill.average_price,
                    filled_at=execution_result.fill.filled_at,
                )
            execution_artifact = ValidationArtifact.from_execution_result(
                execution_result,
                agent_id=self._identity.agent_id,
            )
            artifacts.extend((trade_artifact, risk_artifact, execution_artifact))

        portfolio = self._portfolio_provider.get_portfolio_snapshot()
        performance_artifact = ValidationArtifact.from_performance_checkpoint(
            portfolio,
            agent_id=self._identity.agent_id,
            checkpoint_name="local_dry_run_cycle",
        )
        artifacts.append(performance_artifact)

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

        result = DryRunCycleResult(
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
            execution_results=tuple(execution_results),
            artifacts=tuple(artifacts),
            checkpoints=checkpoints,
            portfolio=portfolio,
            pnl_snapshot=pnl_snapshot,
            drawdown_snapshot=drawdown_snapshot,
            audit_summary=audit_summary,
            reputation=self._reputation,
        )
        self._artifact_ledger.record(
            artifacts=result.artifacts,
            checkpoints=result.checkpoints,
            summary=result.to_dict(),
        )
        return result

    def _new_detected_events(self) -> tuple[DetectedEvent, ...]:
        unseen: list[DetectedEvent] = []
        for event in self._event_repository.list_detected_events():
            event_key = (event.raw_event_id, event.event_type, event.rule_name)
            if event_key in self._handled_event_keys:
                continue
            self._handled_event_keys.add(event_key)
            unseen.append(event)
        return tuple(unseen)


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


def build_local_demo_app(
    *,
    base_dir: str | Path = ROOT_DIR,
    feed_groups: dict[str, list[FeedSource]] | None = None,
    symbols: list[PriceSymbol] | None = None,
    parse_feed: Any | None = None,
    http_get: Any | None = None,
    runtime_mode: str = "local",
    env: Mapping[str, str] | None = None,
) -> DryRunApplication:
    return DryRunApplication(
        paths=DryRunRuntimePaths.from_base_dir(base_dir),
        feed_groups=feed_groups or RSS_FEED_GROUPS,
        symbols=symbols or PRICE_SYMBOLS,
        parse_feed=parse_feed,
        http_get=http_get,
        identity_registry=build_identity_registry(runtime_mode=runtime_mode, env=env),
        runtime_mode=runtime_mode,
    )


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the trading demo in local dry-run or shared Sepolia mode."
    )
    parser.add_argument(
        "--base-dir",
        default=str(ROOT_DIR),
        help="Directory for demo artifacts.",
    )
    parser.add_argument(
        "--feed-group",
        default="market_news",
        help="RSS feed group to ingest during the one-shot dry run.",
    )
    parser.add_argument(
        "--runtime-mode",
        choices=("local", "sepolia"),
        default=os.environ.get("TRADING_RUNTIME_MODE", "local"),
        help="Execution mode: keep local dry-run defaults or use shared Sepolia contracts.",
    )
    return parser


def main() -> int:
    parser = _build_cli_parser()
    args = parser.parse_args()
    app = build_local_demo_app(
        base_dir=args.base_dir,
        runtime_mode=args.runtime_mode,
        env=os.environ,
    )
    result = app.run_cycle(feed_group=args.feed_group)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
