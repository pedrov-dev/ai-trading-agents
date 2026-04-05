from agent.momentum_signal import price_confirmation_state, price_momentum
from ingestion.prices_ingestion import PriceQuote


def test_price_momentum_uses_session_open() -> None:
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

    move = price_momentum(quote)

    assert round(move, 4) == round((68_000.0 - 66_000.0) / 66_000.0, 4)


def test_price_confirmation_state_supports_buy_and_sell_setups() -> None:
    buy_confirmed, buy_reason = price_confirmation_state(
        side="buy",
        price_move=0.02,
        threshold=0.001,
    )
    sell_confirmed, sell_reason = price_confirmation_state(
        side="sell",
        price_move=-0.03,
        threshold=0.001,
    )

    assert buy_confirmed is True
    assert "bullish thesis" in buy_reason.lower()
    assert sell_confirmed is True
    assert "bearish thesis" in sell_reason.lower()
