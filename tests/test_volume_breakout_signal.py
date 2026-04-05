from agent.volume_breakout_signal import volume_confirmation_state
from ingestion.prices_ingestion import PriceQuote


def test_volume_confirmation_state_handles_missing_volume_data() -> None:
    quote = PriceQuote(
        symbol_id="btc_usd",
        current=68_000.0,
        open=66_000.0,
        high=68_400.0,
        low=65_800.0,
        prev_close=65_500.0,
        timestamp=1712275200,
        asset_class="spot",
    )

    confirmed, unavailable, reason = volume_confirmation_state(quote=quote, threshold=1.5)

    assert confirmed is False
    assert unavailable is True
    assert reason is not None
    assert "unavailable" in reason.lower()


def test_volume_confirmation_state_flags_spikes() -> None:
    quote = PriceQuote(
        symbol_id="btc_usd",
        current=68_000.0,
        open=66_000.0,
        high=68_400.0,
        low=65_800.0,
        prev_close=65_500.0,
        timestamp=1712275200,
        asset_class="spot",
        session_volume=20_000.0,
        volume_ratio=2.1,
    )

    confirmed, unavailable, reason = volume_confirmation_state(quote=quote, threshold=1.5)

    assert confirmed is True
    assert unavailable is False
    assert reason is not None
    assert "volume spike" in reason.lower()
