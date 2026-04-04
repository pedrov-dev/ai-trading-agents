import json
from datetime import UTC, datetime

from monitoring.trade_journal import (
    LocalTradeJournal,
    TradeJournalEntry,
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
                realized_pnl_usd=0.0,
                position_side="long",
                position_quantity=0.01,
                position_entry_price=50_000.0,
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
                realized_pnl_usd=20.0,
                position_side=None,
                position_quantity=0.0,
                position_entry_price=None,
                notes=("Take profit exit.",),
            ),
        )
    )

    lines = journal_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event_type"] == "entry"

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
