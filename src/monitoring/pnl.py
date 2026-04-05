"""Judge-friendly PnL, win-rate, and exposure summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from random import Random
from typing import Any

from agent.portfolio import PortfolioSnapshot, Position
from ingestion.prices_ingestion import PriceQuote


@dataclass(frozen=True)
class BenchmarkComparison:
    """Comparison between a signal return and a naive benchmark return."""

    benchmark_id: str
    label: str
    symbol_id: str
    side: str
    reference_price: float
    current_price: float
    return_fraction: float
    excess_return_fraction: float
    beat_signal: bool
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "label": self.label,
            "symbol_id": self.symbol_id,
            "side": self.side,
            "reference_price": self.reference_price,
            "current_price": self.current_price,
            "return_fraction": self.return_fraction,
            "excess_return_fraction": self.excess_return_fraction,
            "beat_signal": self.beat_signal,
            "notes": list(self.notes),
        }


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
    benchmark_comparisons: dict[str, BenchmarkComparison] = field(default_factory=dict)

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
            "benchmark_comparisons": {
                benchmark_id: item.to_dict()
                for benchmark_id, item in self.benchmark_comparisons.items()
            },
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
    benchmark_summary: dict[str, BenchmarkComparison] = field(default_factory=dict)
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
            "benchmark_summary": {
                benchmark_id: item.to_dict()
                for benchmark_id, item in self.benchmark_summary.items()
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
    btc_quote = quote_by_symbol.get("btc_usd")

    position_pnl: dict[str, PositionPnL] = {}
    notes: list[str] = []
    long_exposure_usd = 0.0
    short_exposure_usd = 0.0
    unrealized_total = 0.0
    winning_positions = 0
    losing_positions = 0
    weighted_signal_return = 0.0
    weighted_benchmark_returns: dict[str, float] = {}
    weighted_benchmark_samples: dict[str, BenchmarkComparison] = {}
    total_position_weight = 0.0

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
        benchmark_comparisons = (
            _build_benchmark_comparisons(
                position=position,
                quote=price_quote,
                btc_quote=btc_quote,
                signal_return_fraction=unrealized_return_fraction,
            )
            if price_quote is not None
            else {}
        )
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
            benchmark_comparisons=benchmark_comparisons,
        )
        if entry_notional > 0 and benchmark_comparisons:
            total_position_weight += entry_notional
            weighted_signal_return += unrealized_return_fraction * entry_notional
            for benchmark_id, comparison in benchmark_comparisons.items():
                weighted_benchmark_returns[benchmark_id] = (
                    weighted_benchmark_returns.get(benchmark_id, 0.0)
                    + (comparison.return_fraction * entry_notional)
                )
                weighted_benchmark_samples.setdefault(benchmark_id, comparison)

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
    benchmark_summary: dict[str, BenchmarkComparison] = {}
    if total_position_weight > 0:
        weighted_signal_fraction = weighted_signal_return / total_position_weight
        for benchmark_id, total_return in weighted_benchmark_returns.items():
            sample = weighted_benchmark_samples[benchmark_id]
            benchmark_return = round(total_return / total_position_weight, 6)
            excess_return = round(weighted_signal_fraction - benchmark_return, 6)
            benchmark_summary[benchmark_id] = BenchmarkComparison(
                benchmark_id=benchmark_id,
                label=sample.label,
                symbol_id="portfolio",
                side="weighted",
                reference_price=0.0,
                current_price=0.0,
                return_fraction=benchmark_return,
                excess_return_fraction=excess_return,
                beat_signal=excess_return >= 0,
                notes=(
                    "Weighted by entry notional across currently open positions.",
                ),
            )

    if portfolio.positions:
        notes.append(
            "Benchmark excess returns compare signal return against naive baselines "
            "over the same quote snapshot."
        )
        if btc_quote is None:
            notes.append("BTC quote unavailable; skipped buy-and-hold BTC comparisons.")
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
        benchmark_summary=benchmark_summary,
        as_of=observed_at,
        notes=tuple(notes),
    )


def _calculate_unrealized_pnl(position: Position, current_price: float) -> float:
    quantity = abs(position.quantity)
    if position.side == "short":
        return (position.entry_price - current_price) * quantity
    return (current_price - position.entry_price) * quantity


def _build_benchmark_comparisons(
    *,
    position: Position,
    quote: PriceQuote,
    btc_quote: PriceQuote | None,
    signal_return_fraction: float,
) -> dict[str, BenchmarkComparison]:
    comparisons: dict[str, BenchmarkComparison] = {}

    if btc_quote is not None:
        btc_return = _directional_return(
            entry_price=btc_quote.open,
            current_price=btc_quote.current,
            side="buy",
        )
        comparisons["buy_and_hold_btc"] = _create_benchmark_comparison(
            benchmark_id="buy_and_hold_btc",
            label="Buy & hold BTC",
            symbol_id="btc_usd",
            side="buy",
            reference_price=btc_quote.open,
            current_price=btc_quote.current,
            benchmark_return_fraction=btc_return,
            signal_return_fraction=signal_return_fraction,
            notes=("Tracks BTC from the session open to the latest quote.",),
        )

    random_side = _random_entry_side(position.symbol_id, quote.timestamp)
    random_return = _directional_return(
        entry_price=quote.open,
        current_price=quote.current,
        side=random_side,
    )
    comparisons["random_entry"] = _create_benchmark_comparison(
        benchmark_id="random_entry",
        label="Random entry",
        symbol_id=position.symbol_id,
        side=random_side,
        reference_price=quote.open,
        current_price=quote.current,
        benchmark_return_fraction=random_return,
        signal_return_fraction=signal_return_fraction,
        notes=("Uses a deterministic seeded side choice per symbol and quote timestamp.",),
    )

    momentum_side = _momentum_side(quote)
    momentum_return = _directional_return(
        entry_price=quote.open,
        current_price=quote.current,
        side=momentum_side,
    )
    comparisons["momentum"] = _create_benchmark_comparison(
        benchmark_id="momentum",
        label="Momentum baseline",
        symbol_id=position.symbol_id,
        side=momentum_side,
        reference_price=quote.open,
        current_price=quote.current,
        benchmark_return_fraction=momentum_return,
        signal_return_fraction=signal_return_fraction,
        notes=("Follows the intraday move direction from session open to current quote.",),
    )

    volatility_side, breakout_entry = _volatility_breakout_setup(quote)
    volatility_return = _directional_return(
        entry_price=breakout_entry,
        current_price=quote.current,
        side=volatility_side,
    )
    comparisons["volatility_breakout"] = _create_benchmark_comparison(
        benchmark_id="volatility_breakout",
        label="Volatility breakout baseline",
        symbol_id=position.symbol_id,
        side=volatility_side,
        reference_price=breakout_entry,
        current_price=quote.current,
        benchmark_return_fraction=volatility_return,
        signal_return_fraction=signal_return_fraction,
        notes=("Triggers after price clears half of the observed session range from the open.",),
    )
    return comparisons


def _create_benchmark_comparison(
    *,
    benchmark_id: str,
    label: str,
    symbol_id: str,
    side: str,
    reference_price: float,
    current_price: float,
    benchmark_return_fraction: float,
    signal_return_fraction: float,
    notes: tuple[str, ...],
) -> BenchmarkComparison:
    rounded_benchmark_return = round(benchmark_return_fraction, 6)
    excess_return = round(signal_return_fraction - rounded_benchmark_return, 6)
    return BenchmarkComparison(
        benchmark_id=benchmark_id,
        label=label,
        symbol_id=symbol_id,
        side=side,
        reference_price=round(reference_price, 8),
        current_price=round(current_price, 8),
        return_fraction=rounded_benchmark_return,
        excess_return_fraction=excess_return,
        beat_signal=excess_return >= 0,
        notes=notes,
    )


def _directional_return(*, entry_price: float, current_price: float, side: str) -> float:
    if entry_price <= 0 or current_price <= 0 or side == "flat":
        return 0.0
    if side == "sell":
        return (entry_price - current_price) / entry_price
    return (current_price - entry_price) / entry_price


def _random_entry_side(symbol_id: str, timestamp: int) -> str:
    seed = int.from_bytes(
        sha256(f"{symbol_id}:{timestamp}:random-entry".encode()).digest()[:8],
        byteorder="big",
    )
    return Random(seed).choice(("buy", "sell"))


def _momentum_side(quote: PriceQuote) -> str:
    if quote.current > quote.open:
        return "buy"
    if quote.current < quote.open:
        return "sell"
    return "flat"


def _volatility_breakout_setup(quote: PriceQuote) -> tuple[str, float]:
    if quote.open <= 0:
        return ("flat", 0.0)
    session_range = max(quote.high - quote.low, 0.0)
    if session_range <= 0:
        return ("flat", quote.open)

    upper_breakout = quote.open + (session_range * 0.5)
    lower_breakout = max(quote.open - (session_range * 0.5), 0.0)

    if quote.current >= upper_breakout and upper_breakout > 0:
        return ("buy", upper_breakout)
    if quote.current <= lower_breakout and lower_breakout > 0:
        return ("sell", lower_breakout)
    return ("flat", quote.current)
