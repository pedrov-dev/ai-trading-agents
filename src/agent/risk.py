"""Conservative sizing and guardrails for trade intents."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from agent.portfolio import PortfolioSnapshot
from agent.signals import Signal


def _normalize_fraction(value: float | None) -> float | None:
    if value is None:
        return None
    return round(max(value, 0.0), 4)


def _format_fraction(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


@dataclass(frozen=True)
class RiskConfig:
    """Risk guardrails for the MVP trading agent."""

    max_position_fraction: float = 0.05
    max_daily_loss_fraction: float = 0.03
    max_concurrent_positions: int = 3
    max_positions_per_asset: int | None = None
    cooldown_minutes_after_loss: int = 30
    min_notional_usd: float = 50.0
    min_risk_reward_ratio: float = 1.5


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
    expected_move_fraction: float | None = None
    stop_distance_fraction: float | None = None
    risk_reward_ratio: float | None = None


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
        reduce_only: bool = False,
        expected_move_fraction: float | None = None,
        stop_distance_fraction: float | None = None,
        risk_reward_ratio: float | None = None,
    ) -> RiskCheckResult:
        checked_at = now or datetime.now(UTC)
        max_position_notional = portfolio.total_equity * self._config.max_position_fraction
        capped_notional = min(proposed_notional, max_position_notional)
        allowed_notional = round(
            max(0.0, proposed_notional if reduce_only else capped_notional),
            2,
        )

        resolved_expected_move = _normalize_fraction(expected_move_fraction)
        resolved_stop_distance = _normalize_fraction(stop_distance_fraction)
        resolved_risk_reward = risk_reward_ratio
        if resolved_risk_reward is None and resolved_expected_move is not None:
            resolved_risk_reward = resolved_expected_move / max(
                resolved_stop_distance or 0.0,
                0.0001,
            )
        if resolved_risk_reward is not None:
            resolved_risk_reward = round(max(resolved_risk_reward, 0.0), 4)

        violations: list[RiskViolation] = []
        notes: list[str] = []

        if not reduce_only and portfolio.total_equity <= 0:
            violations.append(
                RiskViolation(
                    code="invalid_equity",
                    message="Portfolio equity must be positive before taking a new trade.",
                )
            )

        if not reduce_only:
            daily_loss_limit = portfolio.total_equity * self._config.max_daily_loss_fraction
            if portfolio.realized_pnl_today <= -daily_loss_limit:
                violations.append(
                    RiskViolation(
                        code="max_daily_loss",
                        message=(
                            "Daily loss limit reached; block new entries until "
                            "the next session."
                        ),
                    )
                )

            if portfolio.open_symbol_count() >= self._config.max_concurrent_positions:
                violations.append(
                    RiskViolation(
                        code="max_concurrent_positions",
                        message="Maximum concurrent symbols already open.",
                    )
                )

            max_positions_per_asset = self._config.max_positions_per_asset
            if max_positions_per_asset is not None and max_positions_per_asset > 0:
                open_positions_for_symbol = len(portfolio.positions_for_symbol(signal.symbol_id))
                if open_positions_for_symbol >= max_positions_per_asset:
                    violations.append(
                        RiskViolation(
                            code="max_positions_per_asset",
                            message=(
                                "Opportunity budget already allocated to "
                                f"{signal.symbol_id}."
                            ),
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

            if resolved_risk_reward is not None:
                notes.append(
                    "Estimated expected move "
                    f"{_format_fraction(resolved_expected_move)}, stop distance "
                    f"{_format_fraction(resolved_stop_distance)}, risk/reward ratio "
                    f"{resolved_risk_reward:.2f}x."
                )
                if resolved_risk_reward <= self._config.min_risk_reward_ratio:
                    violations.append(
                        RiskViolation(
                            code="risk_reward_below_threshold",
                            message=(
                                "Estimated risk/reward ratio "
                                f"{resolved_risk_reward:.2f}x must be greater than "
                                f"{self._config.min_risk_reward_ratio:.2f}x before "
                                "execution."
                            ),
                        )
                    )

        if allowed_notional <= 0:
            violations.append(
                RiskViolation(
                    code="position_size_zero",
                    message="Computed trade size is zero after applying limits.",
                )
            )
        elif not reduce_only and allowed_notional < self._config.min_notional_usd:
            violations.append(
                RiskViolation(
                    code="min_notional",
                    message="Computed trade size is below the minimum actionable notional.",
                )
            )

        if not reduce_only and allowed_notional < proposed_notional:
            notes.append("Position size was capped by the max position-size guardrail.")

        if reduce_only:
            notes.append("Reduce-only exit bypassed the entry guardrails.")

        return RiskCheckResult(
            approved=not violations,
            allowed_notional=allowed_notional,
            violations=tuple(violations),
            notes=tuple(notes),
            expected_move_fraction=resolved_expected_move,
            stop_distance_fraction=resolved_stop_distance,
            risk_reward_ratio=resolved_risk_reward,
        )
