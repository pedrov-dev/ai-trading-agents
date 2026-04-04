from datetime import UTC, datetime

from agent.signals import TradeIntent
from identity.erc8004_registry import (
    HackathonVaultClient,
    LocalERC8004Registry,
    OnChainTransactionResult,
    RiskRouterClient,
    RiskRouterIntent,
    SepoliaContractsConfig,
    ValidationRegistryClient,
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
            "PRIVATE_KEY": (
                "0x59c6995e998f97a5a0044966f094538c5f12027b2f9d1fb0f7c8f6a4038f6f9d"
            ),
            "AGENT_WALLET_PRIVATE_KEY": (
                "0x8b3a350cf5c34c9194ca3a545d6f3ac0f723f0f2bf1f6ec8abca9015ae049631"
            ),
            "AGENT_ID": "42",
        }
    )

    assert config.chain_id == 11155111
    assert config.agent_id == 42
    assert config.agent_registry_address == "0x97b07dDc405B0c28B17559aFFE63BdB3632d0ca3"
    assert config.validation_registry_address == "0x92bF63E5C7Ac6980f237a7164Ab413BE226187F1"
    assert config.has_private_keys is True
    assert config.operator_wallet_address is not None
    assert config.agent_wallet_address is not None
    assert config.operator_wallet_address.startswith("0x")
    assert config.agent_wallet_address.startswith("0x")


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


def test_hackathon_vault_client_claims_and_reads_balance() -> None:
    class _Call:
        def __init__(self, result: object) -> None:
            self._result = result

        def call(self) -> object:
            return self._result

    class _VaultFunctions:
        def __init__(self) -> None:
            self.claimed_agent_ids: list[int] = []

        def claimAllocation(self, agent_id: int) -> dict[str, object]:
            self.claimed_agent_ids.append(agent_id)
            return {"method": "claimAllocation", "agent_id": agent_id}

        def getBalance(self, agent_id: int) -> _Call:
            assert agent_id == 42
            return _Call(50_000_000_000_000_000)

    class _VaultContract:
        def __init__(self) -> None:
            self.functions = _VaultFunctions()

    contract = _VaultContract()
    client = HackathonVaultClient(
        config=SepoliaContractsConfig(agent_id=42),
        contract=contract,
        transaction_sender=lambda call: OnChainTransactionResult(
            tx_hash="0xvault",
            details={"method": str(call["method"]), "agent_id": int(call["agent_id"])}
        ),
    )

    result = client.claim_allocation()

    assert result.tx_hash == "0xvault"
    assert result.details["agent_id"] == 42
    assert client.get_balance() == 50_000_000_000_000_000
    assert contract.functions.claimed_agent_ids == [42]


def test_risk_router_client_simulates_and_submits_trade_intent() -> None:
    class _Call:
        def __init__(self, result: object) -> None:
            self._result = result

        def call(self) -> object:
            return self._result

    class _RouterFunctions:
        def __init__(self) -> None:
            self.simulated: list[tuple[object, ...]] = []
            self.submitted: list[tuple[tuple[object, ...], bytes]] = []

        def getIntentNonce(self, agent_id: int) -> _Call:
            assert agent_id == 42
            return _Call(9)

        def simulateIntent(self, intent_tuple: tuple[object, ...]) -> _Call:
            self.simulated.append(intent_tuple)
            return _Call((True, "ok"))

        def submitTradeIntent(
            self,
            intent_tuple: tuple[object, ...],
            signature: bytes,
        ) -> dict[str, object]:
            self.submitted.append((intent_tuple, signature))
            return {"method": "submitTradeIntent", "signature": signature.hex()}

    class _RouterContract:
        def __init__(self) -> None:
            self.functions = _RouterFunctions()

    contract = _RouterContract()
    intent = RiskRouterIntent(
        agent_id=42,
        agent_wallet="0x0000000000000000000000000000000000000042",
        pair="XBTUSD",
        action="BUY",
        amount_usd_scaled=12550,
        max_slippage_bps=125,
        nonce=9,
        deadline=1_800_000_000,
    )
    client = RiskRouterClient(
        config=SepoliaContractsConfig(agent_id=42),
        contract=contract,
        signer=lambda _: bytes.fromhex("12" * 65),
        transaction_sender=lambda _: OnChainTransactionResult(tx_hash="0xrouter"),
    )

    simulation = client.simulate_trade_intent(intent)
    result = client.submit_trade_intent(intent)

    assert simulation.approved is True
    assert simulation.reason == "ok"
    assert result.tx_hash == "0xrouter"
    assert contract.functions.simulated == [intent.as_tuple()]
    assert contract.functions.submitted[0][0] == intent.as_tuple()
    assert contract.functions.submitted[0][1] == bytes.fromhex("12" * 65)


def test_validation_registry_client_builds_checkpoint_hash() -> None:
    checkpoint_payload = {
        "checkpoint_id": "checkpoint-123",
        "artifact_id": "artifact-123",
        "subject_id": "btc_usd:buy",
        "metric_name": "approved",
        "metric_value": True,
        "agent_id": "42",
        "recorded_at": "2026-04-03T12:00:00+00:00",
    }

    checkpoint_hash = ValidationRegistryClient.build_checkpoint_hash(checkpoint_payload)

    assert checkpoint_hash.startswith("0x")
    assert len(checkpoint_hash) == 66
