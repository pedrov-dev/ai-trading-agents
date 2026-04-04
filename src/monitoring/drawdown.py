"""Drawdown and recovery metrics for judge-facing reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class EquityPoint:
    """One observed point on the portfolio equity curve."""

    recorded_at: datetime
    equity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "recorded_at": self.recorded_at.isoformat(),
            "equity": self.equity,
        }


@dataclass(frozen=True)
class DrawdownSnapshot:
    """Current and worst observed drawdown from an equity history."""

    current_equity: float
    high_water_mark_equity: float
    low_water_mark_equity: float
    current_drawdown_fraction: float
    max_drawdown_fraction: float
    observation_count: int
    peak_at: datetime | None = None
    trough_at: datetime | None = None
    is_recovered: bool = False
    as_of: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_equity": self.current_equity,
            "high_water_mark_equity": self.high_water_mark_equity,
            "low_water_mark_equity": self.low_water_mark_equity,
            "current_drawdown_fraction": self.current_drawdown_fraction,
            "max_drawdown_fraction": self.max_drawdown_fraction,
            "observation_count": self.observation_count,
            "peak_at": self.peak_at.isoformat() if self.peak_at else None,
            "trough_at": self.trough_at.isoformat() if self.trough_at else None,
            "is_recovered": self.is_recovered,
            "as_of": self.as_of.isoformat(),
        }


def build_drawdown_snapshot(
    history: tuple[EquityPoint, ...] | list[EquityPoint],
) -> DrawdownSnapshot:
    """Compute current and maximum drawdown from a chronological equity history."""
    if not history:
        raise ValueError("At least one equity observation is required to compute drawdown.")

    ordered = sorted(history, key=lambda point: point.recorded_at)
    peak_equity = ordered[0].equity
    low_water_mark = ordered[0].equity
    peak_at = ordered[0].recorded_at
    trough_at = ordered[0].recorded_at
    max_drawdown = 0.0

    for point in ordered:
        if point.equity > peak_equity:
            peak_equity = point.equity
            peak_at = point.recorded_at

        if point.equity < low_water_mark:
            low_water_mark = point.equity
            trough_at = point.recorded_at

        drawdown = 0.0
        if peak_equity > 0:
            drawdown = (peak_equity - point.equity) / peak_equity
        max_drawdown = max(max_drawdown, drawdown)

    current_equity = ordered[-1].equity
    current_drawdown = 0.0
    if peak_equity > 0:
        current_drawdown = (peak_equity - current_equity) / peak_equity

    return DrawdownSnapshot(
        current_equity=current_equity,
        high_water_mark_equity=peak_equity,
        low_water_mark_equity=low_water_mark,
        current_drawdown_fraction=current_drawdown,
        max_drawdown_fraction=max_drawdown,
        observation_count=len(ordered),
        peak_at=peak_at,
        trough_at=trough_at,
        is_recovered=current_equity >= peak_equity,
        as_of=ordered[-1].recorded_at,
    )
