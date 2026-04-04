from pathlib import Path

from agent.portfolio import LocalPortfolioStateProvider
from ingestion.prices_config import PRICE_SYMBOLS
from ingestion.rss_config import FeedSource
from main import build_local_demo_app

BTC_SYMBOL = next(symbol for symbol in PRICE_SYMBOLS if symbol.symbol_id == "btc_usd")


def test_local_portfolio_provider_keeps_equity_stable_when_opening_short() -> None:
    provider = LocalPortfolioStateProvider(starting_equity=10_000.0)

    provider.record_fill(
        symbol_id="btc_usd",
        side="sell",
        quantity=0.01,
        price=50_000.0,
    )
    snapshot = provider.get_portfolio_snapshot()

    assert snapshot.cash_usd == 10_500.0
    assert snapshot.total_equity == 10_000.0
    assert snapshot.positions[0].side == "short"


def test_local_demo_app_runs_end_to_end_and_writes_demo_artifacts(tmp_path: Path) -> None:
    def fake_parse_feed(_url: str) -> dict[str, object]:
        return {
            "entries": [
                {
                    "title": "SEC approved the bitcoin ETF after exchange review",
                    "link": "https://example.test/bitcoin-etf-approved",
                    "published": "2026-04-03T12:00:00+00:00",
                }
            ]
        }

    def fake_http_get(url: str, params: dict[str, str]) -> dict[str, object]:
        if url.endswith("/Ticker"):
            assert params == {"pair": BTC_SYMBOL.ticker}
            return {
                "error": [],
                "result": {
                    BTC_SYMBOL.ticker: {
                        "c": ["68000.0", "1"],
                        "o": "66000.0",
                        "h": ["68500.0", "68500.0"],
                        "l": ["65500.0", "65500.0"],
                        "p": ["67000.0", "67000.0"],
                    }
                },
            }
        raise AssertionError(f"Unexpected URL called: {url}")

    app = build_local_demo_app(
        base_dir=tmp_path,
        feed_groups={
            "market_news": [FeedSource(source_id="demo_feed", url="https://example.test/rss")]
        },
        symbols=[BTC_SYMBOL],
        parse_feed=fake_parse_feed,
        http_get=fake_http_get,
    )

    result = app.run_cycle(feed_group="market_news")

    assert result.rss_result.inserted_count == 1
    assert result.prices_result.inserted_count == 1
    assert result.classification_count >= 1
    assert len(result.detected_events) >= 1
    assert len(result.trade_intents) == 1
    assert len(result.execution_results) == 1
    assert result.execution_results[0].status.value == "simulated"
    assert result.artifact_count >= 4
    assert result.checkpoint_count >= 4
    assert result.audit_summary.total_events >= 2
    assert result.portfolio.open_position_count() == 1

    assert (tmp_path / "artifacts" / "orders_audit.jsonl").exists()
    assert (tmp_path / "artifacts" / "validation_artifacts.jsonl").exists()
    assert (tmp_path / "artifacts" / "validation_checkpoints.jsonl").exists()
    assert (tmp_path / "artifacts" / "run_summary.json").exists()
