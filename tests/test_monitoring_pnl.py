from datetime import UTC, datetime

import pytest

from agent.portfolio import PortfolioSnapshot, Position
from ingestion.prices_ingestion import PriceQuote
from monitoring.pnl import build_pnl_snapshot

_DEF_TIME = datetime(2026, 4, 3, 12, 0, tzinfo=UTC)


def test_build_pnl_snapshot_reports_realized_unrealized_and_exposure() -> None:
    portfolio = PortfolioSnapshot(
        total_equity=10_250.0,
        cash_usd=9_000.0,
        realized_pnl_today=125.0,
        positions=(
            Position(symbol_id="btc_usd", side="long", quantity=0.005, entry_price=68_000.0),
            Position(symbol_id="eth_usd", side="short", quantity=0.1, entry_price=3_000.0),
        ),
        as_of=_DEF_TIME,
    )
    quotes = [
        PriceQuote(
            symbol_id="btc_usd",
            current=72_000.0,
            open=69_000.0,
            high=72_500.0,
            low=68_500.0,
            prev_close=68_800.0,
            timestamp=1,
            asset_class="spot",
        ),
        PriceQuote(
            symbol_id="eth_usd",
            current=3_200.0,
            open=3_050.0,
            high=3_250.0,
            low=2_980.0,
            prev_close=3_010.0,
            timestamp=1,
            asset_class="spot",
        ),
    ]

    snapshot = build_pnl_snapshot(portfolio=portfolio, price_quotes=quotes)

    assert snapshot.open_position_count == 2
    assert snapshot.realized_pnl_usd == 125.0
    assert snapshot.unrealized_pnl_usd == pytest.approx(0.0)
    assert snapshot.net_pnl_usd == pytest.approx(125.0)
    assert snapshot.win_rate == pytest.approx(0.5)
    assert snapshot.exposure.gross_exposure_usd == pytest.approx(680.0)
    assert snapshot.exposure.net_exposure_usd == pytest.approx(40.0)
    assert snapshot.position_pnl["btc_usd"].unrealized_pnl_usd == pytest.approx(20.0)
    assert snapshot.position_pnl["eth_usd"].unrealized_pnl_usd == pytest.approx(-20.0)
