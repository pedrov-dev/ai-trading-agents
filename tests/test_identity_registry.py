from identity.erc8004_registry import LocalERC8004Registry


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
