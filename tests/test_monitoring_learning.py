from datetime import UTC, datetime, timedelta

from agent.signals import RejectedTradeCandidate
from agent.strategy import StrategyConfig
from ingestion.prices_ingestion import PriceQuote
from monitoring.calibration import CalibrationSummary
from monitoring.learning import evaluate_post_trade_review, refine_strategy_config
from monitoring.trade_journal import TradeJournalEntry

_DEF_TIME = datetime(2026, 4, 3, 12, 0, tzinfo=UTC)


def test_evaluate_post_trade_review_flags_timing_and_wrong_asset() -> None:
    entry = TradeJournalEntry(
        entry_id="close-1",
        recorded_at=_DEF_TIME + timedelta(minutes=10),
        symbol_id="btc_usd",
        side="sell",
        event_type="full_exit",
        quantity=0.01,
        price=49_000.0,
        confidence_score=0.92,
        expected_move="up",
        actual_move="down",
        prediction_correct=False,
        realized_pnl_usd=-10.0,
        source_event_type="ETF_APPROVAL",
        source_event_group="etf_news",
        max_hold_minutes=120,
        exit_due_at=_DEF_TIME + timedelta(minutes=120),
        realized_return_fraction=-0.02,
    )

    review = evaluate_post_trade_review(
        journal_entry=entry,
        opened_at=_DEF_TIME,
        latest_quotes=(
            PriceQuote(
                symbol_id="eth_usd",
                current=2_200.0,
                open=2_000.0,
                high=2_220.0,
                low=1_980.0,
                prev_close=1_990.0,
                timestamp=1712100000,
                asset_class="spot",
            ),
        ),
        rejected_alternatives=(
            RejectedTradeCandidate(
                symbol_id="eth_usd",
                side="buy",
                reference_price=2_000.0,
                score=0.84,
                confidence_score=0.8,
                composite_score=0.86,
                event_type="PROTOCOL_UPGRADE",
                event_group="protocol_news",
            ),
        ),
    )

    assert review.timing_label == "too_early"
    assert review.asset_selection_label == "wrong_asset"
    assert any("eth_usd" in note.lower() for note in review.notes)


def test_refine_strategy_config_tightens_and_rebalances_from_reviews() -> None:
    current_config = StrategyConfig(
        min_confidence_score=0.7,
        entry_confirmation_threshold=0.65,
        reduced_size_confirmation_threshold=0.5,
        confidence_weight=0.35,
        novelty_weight=0.2,
        risk_reward_weight=0.25,
        diversification_weight=0.2,
    )
    calibration_summary = CalibrationSummary(
        resolved_prediction_count=12,
        unresolved_prediction_count=0,
        hit_rate=0.33,
        brier_score=0.42,
        buckets=(),
    )

    updated_config, adjustments = refine_strategy_config(
        current_config=current_config,
        calibration_summary=calibration_summary,
        timing_labels=("too_early", "too_early", "too_late"),
        asset_selection_labels=("wrong_asset",),
    )

    assert updated_config.min_confidence_score > current_config.min_confidence_score
    assert (
        updated_config.entry_confirmation_threshold
        > current_config.entry_confirmation_threshold
    )
    assert updated_config.diversification_weight > current_config.diversification_weight
    assert updated_config.confidence_weight < current_config.confidence_weight
    assert any(item.field_name == "min_confidence_score" for item in adjustments)
