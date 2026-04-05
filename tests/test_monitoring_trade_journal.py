import json
from datetime import UTC, datetime

import pytest

from monitoring.trade_journal import (
    LocalTradeJournal,
    TradeJournalEntry,
    build_trade_journal_summary,
    build_trade_journal_summary_from_file,
)

_DEF_TIME = datetime(2026, 4, 3, 12, 0, tzinfo=UTC)


def test_local_trade_journal_appends_and_summarizes_position_lifecycle(tmp_path) -> None:
    journal_path = tmp_path / "trading_journal.jsonl"
    journal = LocalTradeJournal(journal_path)
    journal.record_entries(
        (
            TradeJournalEntry(
                entry_id="journal-1",
                recorded_at=_DEF_TIME,
                symbol_id="btc_usd",
                side="buy",
                event_type="entry",
                quantity=0.01,
                price=50_000.0,
                confidence_score=0.91,
                expected_move="up",
                realized_pnl_usd=0.0,
                position_side="long",
                position_quantity=0.01,
                position_entry_price=50_000.0,
                source_event_type="ETF_APPROVAL",
                source_event_group="etf_news",
                notes=("Opened long position.",),
            ),
            TradeJournalEntry(
                entry_id="journal-2",
                recorded_at=_DEF_TIME.replace(minute=15),
                symbol_id="btc_usd",
                side="sell",
                event_type="full_exit",
                quantity=0.01,
                price=52_000.0,
                confidence_score=0.91,
                expected_move="up",
                actual_move="up",
                prediction_correct=True,
                realized_pnl_usd=20.0,
                position_side=None,
                position_quantity=0.0,
                position_entry_price=None,
                source_event_type="ETF_APPROVAL",
                source_event_group="etf_news",
                realized_return_fraction=0.04,
                notes=("Take profit exit.",),
            ),
        )
    )

    lines = journal_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event_type"] == "entry"
    assert json.loads(lines[0])["source_event_type"] == "ETF_APPROVAL"
    assert json.loads(lines[1])["actual_move"] == "up"

    summary = build_trade_journal_summary_from_file(journal_path)

    assert summary.total_entries == 2
    assert summary.event_counts["entry"] == 1
    assert summary.event_counts["full_exit"] == 1
    assert summary.symbol_counts["btc_usd"] == 2
    assert summary.open_position_count == 0
    assert summary.closed_trade_count == 1
    assert summary.realized_pnl_usd == 20.0
    assert summary.win_count == 1
    assert summary.loss_count == 0
    assert summary.source_event_counts["etf_news"] == 2
    assert summary.event_performance["etf_news"].avg_return == 0.04
    assert summary.event_performance["etf_news"].hit_rate == 1.0
    assert summary.event_performance["etf_news"].sharpe == 0.0


def test_trade_journal_summarizes_event_type_performance_metrics() -> None:
    entries = (
        TradeJournalEntry(
            entry_id="close-1",
            recorded_at=_DEF_TIME,
            symbol_id="btc_usd",
            side="sell",
            event_type="full_exit",
            quantity=0.01,
            price=52_000.0,
            realized_pnl_usd=25.0,
            source_event_type="ETF_APPROVAL",
            source_event_group="etf_news",
            realized_return_fraction=0.05,
        ),
        TradeJournalEntry(
            entry_id="close-2",
            recorded_at=_DEF_TIME.replace(minute=5),
            symbol_id="eth_usd",
            side="sell",
            event_type="full_exit",
            quantity=0.5,
            price=3_100.0,
            realized_pnl_usd=45.0,
            source_event_type="ETF_DELAY",
            source_event_group="etf_news",
            realized_return_fraction=0.03,
        ),
        TradeJournalEntry(
            entry_id="close-3",
            recorded_at=_DEF_TIME.replace(minute=10),
            symbol_id="btc_usd",
            side="sell",
            event_type="full_exit",
            quantity=0.01,
            price=49_000.0,
            realized_pnl_usd=-10.0,
            source_event_type="MACRO_NEWS",
            source_event_group="macro_news",
            realized_return_fraction=-0.02,
        ),
    )

    summary = build_trade_journal_summary(entries, recent_entry_limit=3)

    etf_metrics = summary.event_performance["etf_news"]
    macro_metrics = summary.event_performance["macro_news"]
    btc_metrics = summary.asset_performance["btc_usd"]
    eth_metrics = summary.asset_performance["eth_usd"]

    assert summary.hit_rate == pytest.approx(0.6667, rel=1e-4)
    assert summary.avg_return == pytest.approx(0.02)
    assert summary.sharpe > 0
    assert summary.trade_frequency_per_hour == pytest.approx(18.0)

    assert etf_metrics.trade_count == 2
    assert etf_metrics.avg_return == 0.04
    assert etf_metrics.hit_rate == 1.0
    assert etf_metrics.sharpe > 0
    assert macro_metrics.trade_count == 1
    assert macro_metrics.avg_return == -0.02
    assert macro_metrics.hit_rate == 0.0
    assert macro_metrics.sharpe == 0.0

    assert btc_metrics.trade_count == 2
    assert btc_metrics.avg_return == pytest.approx(0.015)
    assert btc_metrics.hit_rate == pytest.approx(0.5)
    assert btc_metrics.realized_pnl_usd == pytest.approx(15.0)
    assert eth_metrics.trade_count == 1
    assert eth_metrics.avg_return == pytest.approx(0.03)
    assert eth_metrics.hit_rate == pytest.approx(1.0)
    assert eth_metrics.realized_pnl_usd == pytest.approx(45.0)


def test_trade_journal_tracks_metrics_by_signal_family_and_version() -> None:
    entries = (
        TradeJournalEntry(
            entry_id="signal-close-1",
            recorded_at=_DEF_TIME,
            symbol_id="btc_usd",
            side="sell",
            event_type="full_exit",
            quantity=0.01,
            price=52_000.0,
            realized_pnl_usd=20.0,
            realized_return_fraction=0.04,
            signal_id="signal-001",
            signal_family="news_sentiment",
            signal_version="v1",
            heuristic_version="v3",
            model_version="gpt-5.3",
            feature_set="news+price",
            asset="BTC",
            direction="long",
            confidence=0.72,
        ),
        TradeJournalEntry(
            entry_id="signal-close-2",
            recorded_at=_DEF_TIME.replace(minute=5),
            symbol_id="btc_usd",
            side="sell",
            event_type="full_exit",
            quantity=0.01,
            price=49_500.0,
            realized_pnl_usd=-10.0,
            realized_return_fraction=-0.02,
            signal_id="signal-002",
            signal_family="news_sentiment",
            signal_version="v1",
            heuristic_version="v3",
            model_version="gpt-5.3",
            feature_set="news+price",
            asset="BTC",
            direction="long",
            confidence=0.68,
        ),
        TradeJournalEntry(
            entry_id="signal-close-3",
            recorded_at=_DEF_TIME.replace(minute=10),
            symbol_id="eth_usd",
            side="sell",
            event_type="full_exit",
            quantity=0.5,
            price=3_200.0,
            realized_pnl_usd=30.0,
            realized_return_fraction=0.06,
            signal_id="signal-003",
            signal_family="news_sentiment",
            signal_version="v2",
            heuristic_version="v3",
            model_version="gpt-5.3",
            feature_set="news+price+volume",
            asset="ETH",
            direction="long",
            confidence=0.81,
        ),
    )

    summary = build_trade_journal_summary(entries, recent_entry_limit=3)

    signal_metrics = summary.signal_performance["news_sentiment"]
    version_v1_metrics = summary.signal_version_performance["news_sentiment:v1"]
    version_v2_metrics = summary.signal_version_performance["news_sentiment:v2"]

    assert signal_metrics.trade_count == 3
    assert signal_metrics.win_rate == pytest.approx(0.6667, rel=1e-4)
    assert signal_metrics.profit_factor == pytest.approx(5.0)
    assert signal_metrics.max_drawdown_fraction == pytest.approx(0.02)

    assert version_v1_metrics.trade_count == 2
    assert version_v1_metrics.win_rate == pytest.approx(0.5)
    assert version_v1_metrics.profit_factor == pytest.approx(2.0)
    assert version_v1_metrics.max_drawdown_fraction == pytest.approx(0.02)

    assert version_v2_metrics.trade_count == 1
    assert version_v2_metrics.win_rate == pytest.approx(1.0)
    assert version_v2_metrics.profit_factor == 0.0
    assert version_v2_metrics.max_drawdown_fraction == pytest.approx(0.0)
