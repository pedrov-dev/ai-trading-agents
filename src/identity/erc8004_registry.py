"""ERC-8004 identity helpers for local dry-runs and shared Sepolia judging mode."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

_SHARED_AGENT_REGISTRY_ABI = [
    (
        "function register(address agentWallet, string name, string description, "
        "string[] capabilities, string agentURI) external returns (uint256 "
        "agentId)"
    ),
    "function isRegistered(uint256 agentId) external view returns (bool)",
    (
        "function getAgent(uint256 agentId) external view returns (tuple(address "
        "operatorWallet, address agentWallet, string name, string description, "
        "string[] capabilities, uint256 registeredAt, bool active))"
    ),
]

_SYMBOL_TO_ROUTER_PAIR: dict[str, str] = {
    "btc_usd": "XBTUSD",
    "eth_usd": "ETHUSD",
    "sol_usd": "SOLUSD",
    "xrp_usd": "XRPUSD",
}

_DEFAULT_CAPABILITIES: tuple[str, ...] = ("trading", "eip712-signing", "checkpoints")


@dataclass(frozen=True)
class AgentIdentity:
    """Registered identity metadata for one autonomous trading agent."""

    agent_id: str
    display_name: str
    strategy_name: str
    owner: str
    exchange: str = "kraken"
    wallet_address: str | None = None
    public_key: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable identity record."""
        return {
            "agent_id": self.agent_id,
            "display_name": self.display_name,
            "strategy_name": self.strategy_name,
            "owner": self.owner,
            "exchange": self.exchange,
            "wallet_address": self.wallet_address,
            "public_key": self.public_key,
            "metadata": dict(self.metadata),
            "registered_at": self.registered_at.isoformat(),
        }


@dataclass(frozen=True)
class IdentityRegistration:
    """Registry wrapper for a registered identity plus registry metadata."""

    identity: AgentIdentity
    registry_name: str = "erc8004-local"
    registered_uri: str = "local://erc8004/agents"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable registration payload."""
        return {
            "registry_name": self.registry_name,
            "registered_uri": self.registered_uri,
            "identity": self.identity.to_dict(),
        }


@dataclass(frozen=True)
class SepoliaContractsConfig:
    """Minimal config required to talk to the shared ERC-8004 Sepolia contracts."""

    rpc_url: str = "https://ethereum-sepolia-rpc.publicnode.com"
    chain_id: int = 11155111
    agent_registry_address: str = "0x97b07dDc405B0c28B17559aFFE63BdB3632d0ca3"
    hackathon_vault_address: str = "0x0E7CD8ef9743FEcf94f9103033a044caBD45fC90"
    risk_router_address: str = "0xd6A6952545FF6E6E6681c2d15C59f9EB8F40FdBC"
    reputation_registry_address: str = "0x423a9904e39537a9997fbaF0f220d79D7d545763"
    validation_registry_address: str = "0x92bF63E5C7Ac6980f237a7164Ab413BE226187F1"
    private_key: str | None = None
    agent_wallet_private_key: str | None = None
    agent_id: int | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> SepoliaContractsConfig:
        env_map = env or os.environ
        chain_id_raw = str(
            env_map.get("SEPOLIA_CHAIN_ID", env_map.get("CHAIN_ID", cls.chain_id))
        ).strip()
        agent_id_raw = str(env_map.get("AGENT_ID", "")).strip()

        return cls(
            rpc_url=str(env_map.get("SEPOLIA_RPC_URL", cls.rpc_url)).strip() or cls.rpc_url,
            chain_id=int(chain_id_raw or cls.chain_id),
            agent_registry_address=str(
                env_map.get("AGENT_REGISTRY_ADDRESS", cls.agent_registry_address)
            ).strip()
            or cls.agent_registry_address,
            hackathon_vault_address=str(
                env_map.get("HACKATHON_VAULT_ADDRESS", cls.hackathon_vault_address)
            ).strip()
            or cls.hackathon_vault_address,
            risk_router_address=str(
                env_map.get("RISK_ROUTER_ADDRESS", cls.risk_router_address)
            ).strip()
            or cls.risk_router_address,
            reputation_registry_address=str(
                env_map.get(
                    "REPUTATION_REGISTRY_ADDRESS",
                    cls.reputation_registry_address,
                )
            ).strip()
            or cls.reputation_registry_address,
            validation_registry_address=str(
                env_map.get(
                    "VALIDATION_REGISTRY_ADDRESS",
                    cls.validation_registry_address,
                )
            ).strip()
            or cls.validation_registry_address,
            private_key=_optional_env_value(env_map, "PRIVATE_KEY"),
            agent_wallet_private_key=_optional_env_value(
                env_map,
                "AGENT_WALLET_PRIVATE_KEY",
            ),
            agent_id=int(agent_id_raw) if agent_id_raw.isdigit() else None,
        )

    @property
    def has_private_keys(self) -> bool:
        return bool(self.private_key and self.agent_wallet_private_key)

    @property
    def has_minimum_network_config(self) -> bool:
        return all(
            (
                self.rpc_url,
                self.agent_registry_address,
                self.hackathon_vault_address,
                self.risk_router_address,
                self.reputation_registry_address,
                self.validation_registry_address,
            )
        )

    @property
    def is_ready_for_transactions(self) -> bool:
        return self.has_minimum_network_config and self.has_private_keys

    def missing_required_values(self) -> tuple[str, ...]:
        missing: list[str] = []
        if not self.rpc_url:
            missing.append("SEPOLIA_RPC_URL")
        if not self.agent_registry_address:
            missing.append("AGENT_REGISTRY_ADDRESS")
        if not self.hackathon_vault_address:
            missing.append("HACKATHON_VAULT_ADDRESS")
        if not self.risk_router_address:
            missing.append("RISK_ROUTER_ADDRESS")
        if not self.reputation_registry_address:
            missing.append("REPUTATION_REGISTRY_ADDRESS")
        if not self.validation_registry_address:
            missing.append("VALIDATION_REGISTRY_ADDRESS")
        if not self.private_key:
            missing.append("PRIVATE_KEY")
        if not self.agent_wallet_private_key:
            missing.append("AGENT_WALLET_PRIVATE_KEY")
        return tuple(missing)


@dataclass(frozen=True)
class RiskRouterIntent:
    """On-chain payload shape expected by the shared RiskRouter contract."""

    agent_id: int
    agent_wallet: str
    pair: str
    action: str
    amount_usd_scaled: int
    max_slippage_bps: int
    nonce: int
    deadline: int

    @classmethod
    def from_trade_intent(
        cls,
        *,
        agent_id: int,
        agent_wallet: str,
        trade_intent: Any,
        nonce: int,
        deadline: int,
        max_slippage_bps: int = 100,
    ) -> RiskRouterIntent:
        symbol_id = str(getattr(trade_intent, "symbol_id", "")).lower()
        pair = _SYMBOL_TO_ROUTER_PAIR.get(symbol_id, symbol_id.replace("_", "").upper())
        action = str(getattr(trade_intent, "side", "buy")).upper()
        notional_usd = float(getattr(trade_intent, "notional_usd", 0.0))

        return cls(
            agent_id=agent_id,
            agent_wallet=agent_wallet,
            pair=pair,
            action=action,
            amount_usd_scaled=int(round(notional_usd * 100)),
            max_slippage_bps=max_slippage_bps,
            nonce=nonce,
            deadline=deadline,
        )

    def as_tuple(self) -> tuple[int, str, str, str, int, int, int, int]:
        return (
            self.agent_id,
            self.agent_wallet,
            self.pair,
            self.action,
            self.amount_usd_scaled,
            self.max_slippage_bps,
            self.nonce,
            self.deadline,
        )

    def to_eip712_structured_data(
        self,
        *,
        chain_id: int,
        verifying_contract: str,
    ) -> dict[str, Any]:
        return {
            "domain": {
                "name": "RiskRouter",
                "version": "1",
                "chainId": chain_id,
                "verifyingContract": verifying_contract,
            },
            "primaryType": "TradeIntent",
            "types": {
                "TradeIntent": [
                    {"name": "agentId", "type": "uint256"},
                    {"name": "agentWallet", "type": "address"},
                    {"name": "pair", "type": "string"},
                    {"name": "action", "type": "string"},
                    {"name": "amountUsdScaled", "type": "uint256"},
                    {"name": "maxSlippageBps", "type": "uint256"},
                    {"name": "nonce", "type": "uint256"},
                    {"name": "deadline", "type": "uint256"},
                ]
            },
            "message": {
                "agentId": self.agent_id,
                "agentWallet": self.agent_wallet,
                "pair": self.pair,
                "action": self.action,
                "amountUsdScaled": self.amount_usd_scaled,
                "maxSlippageBps": self.max_slippage_bps,
                "nonce": self.nonce,
                "deadline": self.deadline,
            },
        }


class IdentityRegistry(Protocol):
    """Abstract registry contract for agent identity registration."""

    def register(
        self,
        *,
        display_name: str,
        strategy_name: str,
        owner: str,
        exchange: str = "kraken",
        wallet_address: str | None = None,
        public_key: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> AgentIdentity:
        ...

    def get(self, agent_id: str) -> AgentIdentity | None:
        ...

    def list_identities(self) -> tuple[AgentIdentity, ...]:
        ...


class LocalERC8004Registry:
    """In-memory registry that keeps agent identities stable and auditable."""

    def __init__(self) -> None:
        self._identities: dict[str, AgentIdentity] = {}

    def register(
        self,
        *,
        display_name: str,
        strategy_name: str,
        owner: str,
        exchange: str = "kraken",
        wallet_address: str | None = None,
        public_key: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> AgentIdentity:
        """Register and return a stable local identity for an agent."""
        normalized_metadata = {key: str(value) for key, value in (metadata or {}).items()}
        resolved_exchange = normalized_metadata.get("exchange", exchange)
        agent_id = _stable_agent_id(display_name, strategy_name, owner, wallet_address)

        existing = self._identities.get(agent_id)
        if existing is not None:
            return existing

        identity = AgentIdentity(
            agent_id=agent_id,
            display_name=display_name,
            strategy_name=strategy_name,
            owner=owner,
            exchange=resolved_exchange,
            wallet_address=wallet_address,
            public_key=public_key,
            metadata=normalized_metadata,
        )
        self._identities[agent_id] = identity
        return identity

    def get(self, agent_id: str) -> AgentIdentity | None:
        """Return one identity by its stable agent id."""
        return self._identities.get(agent_id)

    def list_identities(self) -> tuple[AgentIdentity, ...]:
        """Return all registered identities in insertion order."""
        return tuple(self._identities.values())

    def registration_for(self, agent_id: str) -> IdentityRegistration | None:
        """Return a registration wrapper for the given identity if present."""
        identity = self.get(agent_id)
        if identity is None:
            return None
        return IdentityRegistration(identity=identity)


class OnChainERC8004Registry:
    """Shared Sepolia registry wrapper used for judging-compliant agent registration."""

    def __init__(
        self,
        *,
        config: SepoliaContractsConfig,
        contract: Any | None = None,
    ) -> None:
        self._config = config
        self._contract = contract
        self._web3: Any | None = None
        self._identities: dict[str, AgentIdentity] = {}

    @property
    def config(self) -> SepoliaContractsConfig:
        return self._config

    def register(
        self,
        *,
        display_name: str,
        strategy_name: str,
        owner: str,
        exchange: str = "sepolia",
        wallet_address: str | None = None,
        public_key: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> AgentIdentity:
        if self._config.agent_id is not None:
            existing = self.get(str(self._config.agent_id))
            if existing is not None:
                return existing

        if not self._config.is_ready_for_transactions:
            missing = ", ".join(self._config.missing_required_values())
            raise RuntimeError(
                "Sepolia shared-contract mode requires additional configuration: "
                f"{missing}"
            )

        tx_hash, onchain_agent_id, resolved_wallet = self._register_onchain(
            display_name=display_name,
            strategy_name=strategy_name,
            owner=owner,
            wallet_address=wallet_address,
            metadata=metadata,
        )
        identity = AgentIdentity(
            agent_id=str(onchain_agent_id),
            display_name=display_name,
            strategy_name=strategy_name,
            owner=owner,
            exchange=exchange,
            wallet_address=resolved_wallet,
            public_key=public_key,
            metadata={
                key: str(value) for key, value in (metadata or {}).items()
            }
            | {
                "agent_registry_address": self._config.agent_registry_address,
                "registration_tx_hash": tx_hash,
            },
        )
        self._identities[identity.agent_id] = identity
        return identity

    def get(self, agent_id: str) -> AgentIdentity | None:
        cached = self._identities.get(agent_id)
        if cached is not None:
            return cached

        onchain_agent_id = _coerce_onchain_agent_id(agent_id)
        if onchain_agent_id is None:
            return None

        contract = self._resolve_contract()
        if contract is None:
            return None

        registered = bool(contract.functions.isRegistered(onchain_agent_id).call())
        if not registered:
            return None

        details = contract.functions.getAgent(onchain_agent_id).call()
        operator_wallet, agent_wallet, name, description, capabilities, registered_at, active = (
            details
        )
        identity = AgentIdentity(
            agent_id=str(onchain_agent_id),
            display_name=str(name),
            strategy_name=str(description),
            owner=str(operator_wallet),
            exchange="sepolia",
            wallet_address=str(agent_wallet),
            metadata={
                "active": str(bool(active)).lower(),
                "capabilities": json.dumps(list(capabilities)),
                "registered_at_unix": str(registered_at),
                "agent_registry_address": self._config.agent_registry_address,
            },
            registered_at=datetime.fromtimestamp(int(registered_at), tz=UTC),
        )
        self._identities[identity.agent_id] = identity
        return identity

    def list_identities(self) -> tuple[AgentIdentity, ...]:
        return tuple(self._identities.values())

    def _resolve_contract(self) -> Any | None:
        if self._contract is not None:
            return self._contract
        if not self._config.has_minimum_network_config:
            return None

        try:
            from web3 import Web3
        except ImportError as exc:
            raise RuntimeError(
                "Install `web3` to enable Sepolia shared-contract mode."
            ) from exc

        web3 = Web3(Web3.HTTPProvider(self._config.rpc_url))
        if not web3.is_connected():
            raise RuntimeError(
                f"Unable to connect to Sepolia RPC at {self._config.rpc_url}."
            )

        self._web3 = web3
        self._contract = web3.eth.contract(
            address=Web3.to_checksum_address(self._config.agent_registry_address),
            abi=_SHARED_AGENT_REGISTRY_ABI,
        )
        return self._contract

    def _register_onchain(
        self,
        *,
        display_name: str,
        strategy_name: str,
        owner: str,
        wallet_address: str | None,
        metadata: Mapping[str, str] | None,
    ) -> tuple[str, int, str]:
        contract = self._resolve_contract()
        if contract is None or self._web3 is None or self._config.private_key is None:
            raise RuntimeError("Sepolia registry contract is not available for registration.")

        account = self._web3.eth.account.from_key(self._config.private_key)
        resolved_wallet = wallet_address or account.address
        capabilities = _capabilities_from_metadata(metadata)
        agent_uri = (metadata or {}).get(
            "agent_uri",
            "https://github.com/pedrov-dev/ai-trading-agents",
        )
        description = f"{strategy_name} | operator={owner}"

        transaction = contract.functions.register(
            resolved_wallet,
            display_name,
            description,
            capabilities,
            agent_uri,
        ).build_transaction(
            {
                "from": account.address,
                "nonce": self._web3.eth.get_transaction_count(account.address),
                "chainId": self._config.chain_id,
            }
        )
        signed = self._web3.eth.account.sign_transaction(
            transaction,
            private_key=self._config.private_key,
        )
        raw_transaction = getattr(signed, "raw_transaction", None)
        if raw_transaction is None:
            raw_transaction = signed.rawTransaction
        tx_hash = self._web3.eth.send_raw_transaction(raw_transaction)
        receipt = self._web3.eth.wait_for_transaction_receipt(tx_hash)
        logs = contract.events.AgentRegistered().process_receipt(receipt)
        if not logs:
            raise RuntimeError("Unable to parse the `AgentRegistered` event from Sepolia.")

        onchain_agent_id = int(logs[0]["args"]["agentId"])
        return tx_hash.hex(), onchain_agent_id, resolved_wallet


def _optional_env_value(env: Mapping[str, str], key: str) -> str | None:
    value = str(env.get(key, "")).strip()
    return value or None


def _capabilities_from_metadata(metadata: Mapping[str, str] | None) -> list[str]:
    if not metadata:
        return list(_DEFAULT_CAPABILITIES)

    raw_capabilities = metadata.get("capabilities", "")
    if not raw_capabilities:
        return list(_DEFAULT_CAPABILITIES)

    capabilities = [item.strip() for item in raw_capabilities.split(",") if item.strip()]
    return capabilities or list(_DEFAULT_CAPABILITIES)


def _coerce_onchain_agent_id(agent_id: str | int | None) -> int | None:
    if agent_id is None:
        return None
    if isinstance(agent_id, int):
        return agent_id

    candidate = agent_id.strip()
    if candidate.isdigit():
        return int(candidate)
    return None


def _stable_agent_id(
    display_name: str,
    strategy_name: str,
    owner: str,
    wallet_address: str | None,
) -> str:
    digest = hashlib.sha256(
        f"{display_name}|{strategy_name}|{owner}|{wallet_address or ''}".encode()
    ).hexdigest()[:12]
    return f"agent-{digest}"
