"""Portfolio state models for conservative trading decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, Protocol

PositionSide = Literal["long", "short"]


@dataclass(frozen=True)
class Position:
    """One open position tracked by the decision layer."""

    symbol_id: str
    side: PositionSide
    quantity: float
    entry_price: float
    opened_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def notional_usd(self) -> float:
        return abs(self.quantity) * self.entry_price


@dataclass(frozen=True)
class PortfolioSnapshot:
    """Current portfolio and daily risk state used before creating new intents."""

    total_equity: float
    cash_usd: float
    positions: tuple[Position, ...] = ()
    realized_pnl_today: float = 0.0
    consecutive_losses: int = 0
    last_loss_at: datetime | None = None
    as_of: datetime = field(default_factory=lambda: datetime.now(UTC))

    def open_position_count(self) -> int:
        return len(self.positions)

    def has_open_position(self, symbol_id: str) -> bool:
        return any(position.symbol_id == symbol_id for position in self.positions)

    def position_for_symbol(self, symbol_id: str) -> Position | None:
        for position in self.positions:
            if position.symbol_id == symbol_id:
                return position
        return None

    def total_open_notional(self) -> float:
        return sum(position.notional_usd for position in self.positions)


class PortfolioStateProvider(Protocol):
    """Abstract source of portfolio state for the strategy layer."""

    def get_portfolio_snapshot(self) -> PortfolioSnapshot:
        ...
