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


class LocalPortfolioStateProvider:
    """Mutable local portfolio state used for demo and paper-trading flows."""

    def __init__(
        self,
        *,
        starting_equity: float = 10_000.0,
        starting_cash_usd: float | None = None,
    ) -> None:
        resolved_cash = (
            starting_equity if starting_cash_usd is None else starting_cash_usd
        )
        self._snapshot = PortfolioSnapshot(
            total_equity=round(starting_equity, 2),
            cash_usd=round(resolved_cash, 2),
        )

    def get_portfolio_snapshot(self) -> PortfolioSnapshot:
        return self._snapshot

    def record_fill(
        self,
        *,
        symbol_id: str,
        side: Literal["buy", "sell"],
        quantity: float,
        price: float,
        filled_at: datetime | None = None,
    ) -> PortfolioSnapshot:
        if quantity <= 0 or price <= 0:
            return self._snapshot

        observed_at = filled_at or datetime.now(UTC)
        notional = round(quantity * price, 2)
        next_cash = (
            self._snapshot.cash_usd - notional
            if side == "buy"
            else self._snapshot.cash_usd + notional
        )

        existing = self._snapshot.position_for_symbol(symbol_id)
        remaining_positions = tuple(
            position
            for position in self._snapshot.positions
            if position.symbol_id != symbol_id
        )
        realized_change = 0.0
        updated_position: Position | None = None

        if existing is None:
            updated_position = Position(
                symbol_id=symbol_id,
                side="long" if side == "buy" else "short",
                quantity=round(quantity, 8),
                entry_price=round(price, 8),
                opened_at=observed_at,
            )
        else:
            existing_signed_quantity = _signed_quantity(existing)
            fill_signed_quantity = quantity if side == "buy" else -quantity
            new_signed_quantity = existing_signed_quantity + fill_signed_quantity

            if existing_signed_quantity * fill_signed_quantity > 0:
                weighted_entry = (
                    (abs(existing_signed_quantity) * existing.entry_price)
                    + (abs(fill_signed_quantity) * price)
                ) / abs(new_signed_quantity)
                updated_position = Position(
                    symbol_id=symbol_id,
                    side=existing.side,
                    quantity=round(abs(new_signed_quantity), 8),
                    entry_price=round(weighted_entry, 8),
                    opened_at=existing.opened_at,
                )
            else:
                closed_quantity = min(
                    abs(existing_signed_quantity),
                    abs(fill_signed_quantity),
                )
                realized_change = round(
                    _calculate_realized_pnl(
                        position=existing,
                        exit_price=price,
                        closed_quantity=closed_quantity,
                    ),
                    2,
                )

                if abs(new_signed_quantity) > 0:
                    if existing_signed_quantity * new_signed_quantity > 0:
                        updated_position = Position(
                            symbol_id=symbol_id,
                            side=existing.side,
                            quantity=round(abs(new_signed_quantity), 8),
                            entry_price=round(existing.entry_price, 8),
                            opened_at=existing.opened_at,
                        )
                    else:
                        updated_position = Position(
                            symbol_id=symbol_id,
                            side="long" if new_signed_quantity > 0 else "short",
                            quantity=round(abs(new_signed_quantity), 8),
                            entry_price=round(price, 8),
                            opened_at=observed_at,
                        )

        updated_positions = remaining_positions + ((updated_position,) if updated_position else ())
        realized_total = round(self._snapshot.realized_pnl_today + realized_change, 2)
        total_equity = round(self._snapshot.total_equity + realized_change, 2)
        consecutive_losses = self._snapshot.consecutive_losses
        last_loss_at = self._snapshot.last_loss_at

        if realized_change < 0:
            consecutive_losses += 1
            last_loss_at = observed_at
        elif realized_change > 0:
            consecutive_losses = 0
            last_loss_at = None

        self._snapshot = PortfolioSnapshot(
            total_equity=total_equity,
            cash_usd=round(next_cash, 2),
            positions=updated_positions,
            realized_pnl_today=realized_total,
            consecutive_losses=consecutive_losses,
            last_loss_at=last_loss_at,
            as_of=observed_at,
        )
        return self._snapshot

    def set_realized_pnl(
        self,
        pnl_usd: float,
        *,
        as_of: datetime | None = None,
    ) -> PortfolioSnapshot:
        observed_at = as_of or datetime.now(UTC)
        baseline_equity = self._snapshot.total_equity - self._snapshot.realized_pnl_today
        self._snapshot = PortfolioSnapshot(
            total_equity=round(baseline_equity + pnl_usd, 2),
            cash_usd=self._snapshot.cash_usd,
            positions=self._snapshot.positions,
            realized_pnl_today=round(pnl_usd, 2),
            consecutive_losses=self._snapshot.consecutive_losses,
            last_loss_at=self._snapshot.last_loss_at,
            as_of=observed_at,
        )
        return self._snapshot


def _signed_quantity(position: Position) -> float:
    return position.quantity if position.side == "long" else -position.quantity


def _calculate_realized_pnl(
    *,
    position: Position,
    exit_price: float,
    closed_quantity: float,
) -> float:
    quantity = abs(closed_quantity)
    if position.side == "short":
        return (position.entry_price - exit_price) * quantity
    return (exit_price - position.entry_price) * quantity
