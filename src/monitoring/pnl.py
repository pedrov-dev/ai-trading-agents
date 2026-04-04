"""Judge-friendly PnL, win-rate, and exposure summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from agent.portfolio import PortfolioSnapshot, Position
from ingestion.prices_ingestion import PriceQuote


@dataclass(frozen=True)
class PositionPnL:
    """Per-position mark-to-market summary."""

    symbol_id: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    entry_notional_usd: float
    market_value_usd: float
    unrealized_pnl_usd: float
    unrealized_return_fraction: float
    as_of: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol_id": self.symbol_id,
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "entry_notional_usd": self.entry_notional_usd,
            "market_value_usd": self.market_value_usd,
            "unrealized_pnl_usd": self.unrealized_pnl_usd,
            "unrealized_return_fraction": self.unrealized_return_fraction,
            "as_of": self.as_of.isoformat(),
        }


@dataclass(frozen=True)
class ExposureMetrics:
    """Current portfolio exposure split by long and short positions."""

    gross_exposure_usd: float
    net_exposure_usd: float
    long_exposure_usd: float
    short_exposure_usd: float
    cash_ratio: float

    def to_dict(self) -> dict[str, float]:
        return {
            "gross_exposure_usd": self.gross_exposure_usd,
            "net_exposure_usd": self.net_exposure_usd,
            "long_exposure_usd": self.long_exposure_usd,
            "short_exposure_usd": self.short_exposure_usd,
            "cash_ratio": self.cash_ratio,
        }


@dataclass(frozen=True)
class PnLSnapshot:
    """Headline performance numbers for judges and demos."""

    realized_pnl_usd: float
    unrealized_pnl_usd: float
    net_pnl_usd: float
    exposure: ExposureMetrics
    open_position_count: int
    winning_positions: int
    losing_positions: int
    win_rate: float
    position_pnl: dict[str, PositionPnL] = field(default_factory=dict)
    as_of: datetime = field(default_factory=lambda: datetime.now(UTC))
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "realized_pnl_usd": self.realized_pnl_usd,
            "unrealized_pnl_usd": self.unrealized_pnl_usd,
            "net_pnl_usd": self.net_pnl_usd,
            "exposure": self.exposure.to_dict(),
            "open_position_count": self.open_position_count,
            "winning_positions": self.winning_positions,
            "losing_positions": self.losing_positions,
            "win_rate": self.win_rate,
            "position_pnl": {
                symbol_id: item.to_dict() for symbol_id, item in self.position_pnl.items()
            },
            "as_of": self.as_of.isoformat(),
            "notes": list(self.notes),
        }


def build_pnl_snapshot(
    *,
    portfolio: PortfolioSnapshot,
    price_quotes: list[PriceQuote] | tuple[PriceQuote, ...],
    as_of: datetime | None = None,
) -> PnLSnapshot:
    """Build a mark-to-market PnL snapshot from the latest portfolio and quotes."""
    observed_at = as_of or portfolio.as_of
    quote_by_symbol = {quote.symbol_id: quote for quote in price_quotes}

    position_pnl: dict[str, PositionPnL] = {}
    notes: list[str] = []
    long_exposure_usd = 0.0
    short_exposure_usd = 0.0
    unrealized_total = 0.0
    winning_positions = 0
    losing_positions = 0

    for position in portfolio.positions:
        price_quote = quote_by_symbol.get(position.symbol_id)
        current_price = price_quote.current if price_quote is not None else position.entry_price
        if price_quote is None:
            notes.append(
                "No live quote available for "
                f"{position.symbol_id}; using entry price for mark-to-market."
            )

        entry_notional = round(abs(position.quantity) * position.entry_price, 2)
        current_notional = round(abs(position.quantity) * current_price, 2)
        unrealized_pnl = round(_calculate_unrealized_pnl(position, current_price), 2)
        unrealized_return_fraction = 0.0
        if entry_notional > 0:
            unrealized_return_fraction = round(unrealized_pnl / entry_notional, 6)

        if position.side == "long":
            long_exposure_usd += current_notional
        else:
            short_exposure_usd += current_notional

        if unrealized_pnl > 0:
            winning_positions += 1
        elif unrealized_pnl < 0:
            losing_positions += 1

        unrealized_total += unrealized_pnl
        position_pnl[position.symbol_id] = PositionPnL(
            symbol_id=position.symbol_id,
            side=position.side,
            quantity=position.quantity,
            entry_price=position.entry_price,
            current_price=current_price,
            entry_notional_usd=entry_notional,
            market_value_usd=current_notional,
            unrealized_pnl_usd=unrealized_pnl,
            unrealized_return_fraction=unrealized_return_fraction,
            as_of=observed_at,
        )

    gross_exposure_usd = round(long_exposure_usd + short_exposure_usd, 2)
    net_exposure_usd = round(long_exposure_usd - short_exposure_usd, 2)
    cash_ratio = 0.0
    if portfolio.total_equity > 0:
        cash_ratio = round(portfolio.cash_usd / portfolio.total_equity, 6)

    open_position_count = portfolio.open_position_count()
    measured_positions = winning_positions + losing_positions
    win_rate = round(winning_positions / measured_positions, 4) if measured_positions > 0 else 0.0

    realized_pnl_usd = round(portfolio.realized_pnl_today, 2)
    unrealized_pnl_usd = round(unrealized_total, 2)
    net_pnl_usd = round(realized_pnl_usd + unrealized_pnl_usd, 2)

    exposure = ExposureMetrics(
        gross_exposure_usd=round(gross_exposure_usd, 2),
        net_exposure_usd=round(net_exposure_usd, 2),
        long_exposure_usd=round(long_exposure_usd, 2),
        short_exposure_usd=round(short_exposure_usd, 2),
        cash_ratio=cash_ratio,
    )
    return PnLSnapshot(
        realized_pnl_usd=realized_pnl_usd,
        unrealized_pnl_usd=unrealized_pnl_usd,
        net_pnl_usd=net_pnl_usd,
        exposure=exposure,
        open_position_count=open_position_count,
        winning_positions=winning_positions,
        losing_positions=losing_positions,
        win_rate=win_rate,
        position_pnl=position_pnl,
        as_of=observed_at,
        notes=tuple(notes),
    )


def _calculate_unrealized_pnl(position: Position, current_price: float) -> float:
    quantity = abs(position.quantity)
    if position.side == "short":
        return (position.entry_price - current_price) * quantity
    return (current_price - position.entry_price) * quantity
