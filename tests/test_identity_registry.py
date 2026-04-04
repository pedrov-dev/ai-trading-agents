from datetime import UTC, datetime

from agent.signals import TradeIntent
from identity.erc8004_registry import (
    LocalERC8004Registry,
    RiskRouterIntent,
    SepoliaContractsConfig,
)


def test_local_registry_registers_and_retrieves_agent_identity() -> None:
    registry = LocalERC8004Registry()

    identity = registry.register(
        display_name="Atlas",
        strategy_name="Event Driven BTC",
        owner="pedrov-dev",
        wallet_address="0xabc123",
        metadata={"environment": "paper", "exchange": "kraken"},
    )

    payload = identity.to_dict()

    assert identity.agent_id.startswith("agent-")
    assert registry.get(identity.agent_id) == identity
    assert len(registry.list_identities()) == 1
    assert payload["display_name"] == "Atlas"
    assert payload["exchange"] == "kraken"
    assert payload["metadata"]["environment"] == "paper"


def test_sepolia_contracts_config_loads_shared_addresses_from_env() -> None:
    config = SepoliaContractsConfig.from_env(
        {
            "SEPOLIA_RPC_URL": "https://ethereum-sepolia-rpc.publicnode.com",
            "CHAIN_ID": "11155111",
            "AGENT_REGISTRY_ADDRESS": "0x97b07dDc405B0c28B17559aFFE63BdB3632d0ca3",
            "HACKATHON_VAULT_ADDRESS": "0x0E7CD8ef9743FEcf94f9103033a044caBD45fC90",
            "RISK_ROUTER_ADDRESS": "0xd6A6952545FF6E6E6681c2d15C59f9EB8F40FdBC",
            "REPUTATION_REGISTRY_ADDRESS": "0x423a9904e39537a9997fbaF0f220d79D7d545763",
            "VALIDATION_REGISTRY_ADDRESS": "0x92bF63E5C7Ac6980f237a7164Ab413BE226187F1",
            "PRIVATE_KEY": "0xoperator",
            "AGENT_WALLET_PRIVATE_KEY": "0xagent",
            "AGENT_ID": "42",
        }
    )

    assert config.chain_id == 11155111
    assert config.agent_id == 42
    assert config.agent_registry_address == "0x97b07dDc405B0c28B17559aFFE63BdB3632d0ca3"
    assert config.validation_registry_address == "0x92bF63E5C7Ac6980f237a7164Ab413BE226187F1"
    assert config.has_private_keys is True


def test_risk_router_intent_maps_trade_intent_to_onchain_payload() -> None:
    trade_intent = TradeIntent(
        symbol_id="btc_usd",
        side="buy",
        notional_usd=125.5,
        quantity=0.00184,
        current_price=68_000.0,
        score=0.92,
        rationale=("ETF approval momentum",),
        generated_at=datetime(2026, 4, 3, 12, 0, tzinfo=UTC),
    )

    intent = RiskRouterIntent.from_trade_intent(
        agent_id=42,
        agent_wallet="0x0000000000000000000000000000000000000042",
        trade_intent=trade_intent,
        nonce=7,
        deadline=1_800_000_000,
        max_slippage_bps=125,
    )

    assert intent.pair == "XBTUSD"
    assert intent.action == "BUY"
    assert intent.amount_usd_scaled == 12550
    assert intent.as_tuple() == (
        42,
        "0x0000000000000000000000000000000000000042",
        "XBTUSD",
        "BUY",
        12550,
        125,
        7,
        1_800_000_000,
    )
