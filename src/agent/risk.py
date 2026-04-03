"""Conservative sizing and guardrails for trade intents."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from agent.portfolio import PortfolioSnapshot
from agent.signals import Signal


@dataclass(frozen=True)
class RiskConfig:
    """Risk guardrails for the MVP trading agent."""

    max_position_fraction: float = 0.05
    max_daily_loss_fraction: float = 0.03
    max_concurrent_positions: int = 3
    cooldown_minutes_after_loss: int = 30
    min_notional_usd: float = 50.0


@dataclass(frozen=True)
class RiskViolation:
    """One guardrail that prevented a trade from being approved."""

    code: str
    message: str


@dataclass(frozen=True)
class RiskCheckResult:
    """Result of running all pre-trade checks."""

    approved: bool
    allowed_notional: float
    violations: tuple[RiskViolation, ...] = ()
    notes: tuple[str, ...] = ()


class RiskManager:
    """Conservative position sizing plus hard stop guardrails."""

    def __init__(self, config: RiskConfig | None = None) -> None:
        self._config = config or RiskConfig()

    def size_for_signal(self, *, signal: Signal, portfolio: PortfolioSnapshot) -> float:
        normalized_score = min(max(signal.score, 0.0), 1.0)
        target_fraction = min(
            self._config.max_position_fraction,
            max(self._config.max_position_fraction * normalized_score, 0.01),
        )
        position_cap = portfolio.total_equity * self._config.max_position_fraction
        notional = portfolio.total_equity * target_fraction

        if signal.side == "buy":
            notional = min(notional, max(portfolio.cash_usd, 0.0))

        return round(min(notional, position_cap), 2)

    def evaluate(
        self,
        *,
        signal: Signal,
        portfolio: PortfolioSnapshot,
        proposed_notional: float,
        now: datetime | None = None,
    ) -> RiskCheckResult:
        del signal
        checked_at = now or datetime.now(UTC)
        max_position_notional = portfolio.total_equity * self._config.max_position_fraction
        allowed_notional = round(max(0.0, min(proposed_notional, max_position_notional)), 2)

        violations: list[RiskViolation] = []
        notes: list[str] = []

        if portfolio.total_equity <= 0:
            violations.append(
                RiskViolation(
                    code="invalid_equity",
                    message="Portfolio equity must be positive before taking a new trade.",
                )
            )

        daily_loss_limit = portfolio.total_equity * self._config.max_daily_loss_fraction
        if portfolio.realized_pnl_today <= -daily_loss_limit:
            violations.append(
                RiskViolation(
                    code="max_daily_loss",
                    message="Daily loss limit reached; block new entries until the next session.",
                )
            )

        if portfolio.open_position_count() >= self._config.max_concurrent_positions:
            violations.append(
                RiskViolation(
                    code="max_concurrent_positions",
                    message="Maximum concurrent positions already open.",
                )
            )

        if portfolio.consecutive_losses > 0 and portfolio.last_loss_at is not None:
            cooldown_until = portfolio.last_loss_at + timedelta(
                minutes=self._config.cooldown_minutes_after_loss
            )
            if checked_at < cooldown_until:
                violations.append(
                    RiskViolation(
                        code="cooldown_after_loss",
                        message="Cooldown after recent losses is still active.",
                    )
                )

        if allowed_notional <= 0:
            violations.append(
                RiskViolation(
                    code="position_size_zero",
                    message="Computed trade size is zero after applying limits.",
                )
            )
        elif allowed_notional < self._config.min_notional_usd:
            violations.append(
                RiskViolation(
                    code="min_notional",
                    message="Computed trade size is below the minimum actionable notional.",
                )
            )

        if allowed_notional < proposed_notional:
            notes.append("Position size was capped by the max position-size guardrail.")

        return RiskCheckResult(
            approved=not violations,
            allowed_notional=allowed_notional,
            violations=tuple(violations),
            notes=tuple(notes),
        )
