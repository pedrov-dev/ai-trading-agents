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
        if self._shared_contract_config is not None and self._identity.agent_id.isdigit():
            self.persist_agent_id(self._identity.agent_id)

    @property
    def identity(self) -> AgentIdentity:
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
        saw_runtime_mode = False

        for line in existing_lines:
            stripped = line.strip()
            if stripped.startswith("AGENT_ID="):
                updated_lines.append(f"AGENT_ID={resolved_agent_id}")
                saw_agent_id = True
            elif stripped.startswith("TRADING_RUNTIME_MODE=") and self._runtime_mode == "sepolia":
                updated_lines.append("TRADING_RUNTIME_MODE=sepolia")
                saw_runtime_mode = True
            else:
                updated_lines.append(line)

        if not saw_agent_id:
            updated_lines.append(f"AGENT_ID={resolved_agent_id}")
        if self._runtime_mode == "sepolia" and not saw_runtime_mode:
            updated_lines.append("TRADING_RUNTIME_MODE=sepolia")

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
            "agent_registry_address": self._shared_contract_config.agent_registry_address,
            "hackathon_vault_address": self._shared_contract_config.hackathon_vault_address,
            "risk_router_address": self._shared_contract_config.risk_router_address,
            "validation_registry_address": self._shared_contract_config.validation_registry_address,
            "reputation_registry_address": self._shared_contract_config.reputation_registry_address,
            "transaction_ready": self._shared_contract_config.is_ready_for_transactions,
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
        return self._validation_registry_client.post_checkpoint(
            checkpoint,
            agent_id=self._resolved_agent_id_for_chain(),
            score=score,
        )

    def get_onchain_reputation_score(self) -> int | None:
        if self._reputation_registry_client is None:
            return None
        return self._reputation_registry_client.get_average_score(
            self._resolved_agent_id_for_chain()
        )

    def run_shared_contract_actions(
        self,
        *,
        trade_intents: tuple[TradeIntent, ...],
        checkpoints: tuple[ValidationCheckpoint, ...],
        claim_allocation: bool = False,
        submit_trade_intents: bool = False,
        post_checkpoints: bool = False,
    ) -> dict[str, Any]:
        summary: dict[str, Any] = self.shared_contract_status()
        summary.update(
            {
                "claim_allocation": None,
                "trade_submissions": [],
                "checkpoint_posts": [],
                "reputation_score": None,
            }
        )

        if self._shared_contract_config is None:
            return summary

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
    payload = result.to_dict()
    payload["runtime_mode"] = args.runtime_mode
    if args.runtime_mode == "sepolia":
        payload["shared_contracts"] = app.run_shared_contract_actions(
            trade_intents=result.trade_intents,
            checkpoints=result.checkpoints,
            claim_allocation=args.claim_allocation,
            submit_trade_intents=args.submit_onchain,
            post_checkpoints=args.post_checkpoints,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
