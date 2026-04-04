"""ERC-8004 identity helpers for local dry-runs and shared Sepolia judging mode."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
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

_SHARED_HACKATHON_VAULT_ABI = [
    "function claimAllocation(uint256 agentId) external",
    "function getBalance(uint256 agentId) external view returns (uint256)",
    "function hasClaimed(uint256 agentId) external view returns (bool)",
]

_SHARED_RISK_ROUTER_ABI = [
    (
        "function submitTradeIntent((uint256 agentId, address agentWallet, string "
        "pair, string action, uint256 amountUsdScaled, uint256 "
        "maxSlippageBps, uint256 nonce, uint256 deadline) intent, bytes "
        "signature) external"
    ),
    (
        "function simulateIntent((uint256 agentId, address agentWallet, string "
        "pair, string action, uint256 amountUsdScaled, uint256 "
        "maxSlippageBps, uint256 nonce, uint256 deadline) intent) external "
        "view returns (bool valid, string reason)"
    ),
    "function getIntentNonce(uint256 agentId) external view returns (uint256)",
]

_SHARED_VALIDATION_REGISTRY_ABI = [
    (
        "function postEIP712Attestation(uint256 agentId, bytes32 checkpointHash, "
        "uint8 score, string notes) external"
    ),
    (
        "function getAverageValidationScore(uint256 agentId) external view returns "
        "(uint256)"
    ),
]

_SHARED_REPUTATION_REGISTRY_ABI = [
    (
        "function submitFeedback(uint256 agentId, uint8 score, bytes32 outcomeRef, "
        "string comment, uint8 feedbackType) external"
    ),
    "function getAverageScore(uint256 agentId) external view returns (uint256)",
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
    def operator_wallet_address(self) -> str | None:
        return _derive_account_address(self.private_key)

    @property
    def agent_wallet_address(self) -> str | None:
        return _derive_account_address(self.agent_wallet_private_key)

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


@dataclass(frozen=True)
class OnChainTransactionResult:
    """Common transaction receipt summary for shared Sepolia contract calls."""

    tx_hash: str
    status: str = "submitted"
    block_number: int | None = None
    gas_used: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskRouterSimulation:
    """Dry-run validation result returned by the shared RiskRouter."""

    approved: bool
    reason: str = ""


class _SharedSepoliaClientBase:
    """Common web3 setup and transaction-sending helpers for shared contracts."""

    def __init__(
        self,
        *,
        config: SepoliaContractsConfig,
        contract_address: str,
        abi: list[str],
        contract: Any | None = None,
        transaction_sender: Callable[[Any], OnChainTransactionResult] | None = None,
    ) -> None:
        self._config = config
        self._contract_address = contract_address
        self._abi = abi
        self._contract = contract
        self._web3: Any | None = None
        self._transaction_sender = transaction_sender

    def _resolve_contract(self) -> Any:
        if self._contract is not None:
            return self._contract
        if not self._contract_address:
            raise RuntimeError("Missing shared-contract address for the selected client.")

        try:
            from web3 import Web3
        except ImportError as exc:
            raise RuntimeError(
                "Install `web3` to enable shared Sepolia contract interactions."
            ) from exc

        if self._web3 is None:
            web3 = Web3(Web3.HTTPProvider(self._config.rpc_url))
            if not web3.is_connected():
                raise RuntimeError(
                    f"Unable to connect to Sepolia RPC at {self._config.rpc_url}."
                )
            self._web3 = web3

        self._contract = self._web3.eth.contract(
            address=Web3.to_checksum_address(self._contract_address),
            abi=self._abi,
        )
        return self._contract

    def _send_transaction(self, contract_call: Any) -> OnChainTransactionResult:
        if self._transaction_sender is not None:
            return self._transaction_sender(contract_call)

        contract = self._resolve_contract()
        del contract
        if self._web3 is None or self._config.private_key is None:
            raise RuntimeError(
                "Sepolia transaction signing requires a configured `PRIVATE_KEY`."
            )

        account = self._web3.eth.account.from_key(self._config.private_key)
        transaction = contract_call.build_transaction(
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

        return OnChainTransactionResult(
            tx_hash=_tx_hash_as_hex(tx_hash),
            status="confirmed" if int(getattr(receipt, "status", 1)) == 1 else "reverted",
            block_number=getattr(receipt, "blockNumber", None),
            gas_used=getattr(receipt, "gasUsed", None),
        )


class HackathonVaultClient(_SharedSepoliaClientBase):
    """Client wrapper for claiming and inspecting shared hackathon sandbox capital."""

    def __init__(
        self,
        *,
        config: SepoliaContractsConfig,
        contract: Any | None = None,
        transaction_sender: Callable[[Any], OnChainTransactionResult] | None = None,
    ) -> None:
        super().__init__(
            config=config,
            contract_address=config.hackathon_vault_address,
            abi=_SHARED_HACKATHON_VAULT_ABI,
            contract=contract,
            transaction_sender=transaction_sender,
        )

    def claim_allocation(self, agent_id: int | None = None) -> OnChainTransactionResult:
        resolved_agent_id = _resolve_required_agent_id(agent_id, self._config.agent_id)
        contract = self._resolve_contract()
        result = self._send_transaction(contract.functions.claimAllocation(resolved_agent_id))
        details = dict(result.details)
        details["agent_id"] = resolved_agent_id
        try:
            details["balance_wei"] = self.get_balance(resolved_agent_id)
        except Exception:
            pass
        return OnChainTransactionResult(
            tx_hash=result.tx_hash,
            status=result.status,
            block_number=result.block_number,
            gas_used=result.gas_used,
            details=details,
        )

    def get_balance(self, agent_id: int | None = None) -> int:
        resolved_agent_id = _resolve_required_agent_id(agent_id, self._config.agent_id)
        contract = self._resolve_contract()
        return int(contract.functions.getBalance(resolved_agent_id).call())

    def has_claimed(self, agent_id: int | None = None) -> bool:
        resolved_agent_id = _resolve_required_agent_id(agent_id, self._config.agent_id)
        contract = self._resolve_contract()
        return bool(contract.functions.hasClaimed(resolved_agent_id).call())


class RiskRouterClient(_SharedSepoliaClientBase):
    """Client wrapper for simulating, signing, and submitting trade intents."""

    def __init__(
        self,
        *,
        config: SepoliaContractsConfig,
        contract: Any | None = None,
        signer: Callable[[RiskRouterIntent], bytes] | None = None,
        transaction_sender: Callable[[Any], OnChainTransactionResult] | None = None,
    ) -> None:
        super().__init__(
            config=config,
            contract_address=config.risk_router_address,
            abi=_SHARED_RISK_ROUTER_ABI,
            contract=contract,
            transaction_sender=transaction_sender,
        )
        self._signer = signer

    def get_intent_nonce(self, agent_id: int | None = None) -> int:
        resolved_agent_id = _resolve_required_agent_id(agent_id, self._config.agent_id)
        contract = self._resolve_contract()
        return int(contract.functions.getIntentNonce(resolved_agent_id).call())

    def simulate_trade_intent(self, intent: RiskRouterIntent) -> RiskRouterSimulation:
        contract = self._resolve_contract()
        approved, reason = contract.functions.simulateIntent(intent.as_tuple()).call()
        return RiskRouterSimulation(approved=bool(approved), reason=str(reason))

    def sign_trade_intent(self, intent: RiskRouterIntent) -> bytes:
        if self._signer is not None:
            return self._signer(intent)
        if self._config.agent_wallet_private_key is None:
            raise RuntimeError(
                "RiskRouter signing requires `AGENT_WALLET_PRIVATE_KEY` in the environment."
            )

        try:
            from eth_account import Account
        except ImportError as exc:
            raise RuntimeError(
                "Install `eth-account` to sign shared RiskRouter intents."
            ) from exc

        structured_data = intent.to_eip712_structured_data(
            chain_id=self._config.chain_id,
            verifying_contract=self._config.risk_router_address,
        )
        signed = Account.sign_typed_data(
            self._config.agent_wallet_private_key,
            full_message=structured_data,
        )
        signature = getattr(signed, "signature", b"")
        if isinstance(signature, bytes):
            return signature
        hex_signature = str(signature)
        if hex_signature.startswith("0x"):
            hex_signature = hex_signature[2:]
        return bytes.fromhex(hex_signature)

    def submit_trade_intent(
        self,
        intent: RiskRouterIntent,
        *,
        signature: bytes | None = None,
    ) -> OnChainTransactionResult:
        contract = self._resolve_contract()
        resolved_signature = signature or self.sign_trade_intent(intent)
        result = self._send_transaction(
            contract.functions.submitTradeIntent(intent.as_tuple(), resolved_signature)
        )
        details = dict(result.details)
        details.update(
            {
                "agent_id": intent.agent_id,
                "pair": intent.pair,
                "action": intent.action,
                "amount_usd_scaled": intent.amount_usd_scaled,
            }
        )
        return OnChainTransactionResult(
            tx_hash=result.tx_hash,
            status=result.status,
            block_number=result.block_number,
            gas_used=result.gas_used,
            details=details,
        )


class ValidationRegistryClient(_SharedSepoliaClientBase):
    """Client wrapper for posting and reading checkpoint attestations."""

    def __init__(
        self,
        *,
        config: SepoliaContractsConfig,
        contract: Any | None = None,
        transaction_sender: Callable[[Any], OnChainTransactionResult] | None = None,
    ) -> None:
        super().__init__(
            config=config,
            contract_address=config.validation_registry_address,
            abi=_SHARED_VALIDATION_REGISTRY_ABI,
            contract=contract,
            transaction_sender=transaction_sender,
        )

    @staticmethod
    def build_checkpoint_hash(checkpoint: Mapping[str, Any] | Any) -> str:
        if hasattr(checkpoint, "to_dict"):
            payload = checkpoint.to_dict()
        else:
            payload = dict(checkpoint)
        canonical_json = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        try:
            from web3 import Web3

            return _tx_hash_as_hex(Web3.keccak(text=canonical_json))
        except ImportError:
            return f"0x{hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()}"

    def post_checkpoint(
        self,
        checkpoint: Mapping[str, Any] | Any,
        *,
        agent_id: int | None = None,
        score: int = 85,
        notes: str | None = None,
    ) -> OnChainTransactionResult:
        resolved_agent_id = _resolve_required_agent_id(
            agent_id if agent_id is not None else getattr(checkpoint, "agent_id", None),
            self._config.agent_id,
        )
        contract = self._resolve_contract()
        checkpoint_hash = self.build_checkpoint_hash(checkpoint)
        note_text = notes or _checkpoint_note_text(checkpoint)
        bounded_score = max(0, min(int(score), 100))
        result = self._send_transaction(
            contract.functions.postEIP712Attestation(
                resolved_agent_id,
                checkpoint_hash,
                bounded_score,
                note_text,
            )
        )
        details = dict(result.details)
        details.update(
            {
                "agent_id": resolved_agent_id,
                "checkpoint_hash": checkpoint_hash,
                "score": bounded_score,
            }
        )
        return OnChainTransactionResult(
            tx_hash=result.tx_hash,
            status=result.status,
            block_number=result.block_number,
            gas_used=result.gas_used,
            details=details,
        )

    def get_average_validation_score(self, agent_id: int | None = None) -> int:
        resolved_agent_id = _resolve_required_agent_id(agent_id, self._config.agent_id)
        contract = self._resolve_contract()
        return int(contract.functions.getAverageValidationScore(resolved_agent_id).call())


class ReputationRegistryClient(_SharedSepoliaClientBase):
    """Read helper for leaderboard-facing reputation scores."""

    def __init__(
        self,
        *,
        config: SepoliaContractsConfig,
        contract: Any | None = None,
        transaction_sender: Callable[[Any], OnChainTransactionResult] | None = None,
    ) -> None:
        super().__init__(
            config=config,
            contract_address=config.reputation_registry_address,
            abi=_SHARED_REPUTATION_REGISTRY_ABI,
            contract=contract,
            transaction_sender=transaction_sender,
        )

    def get_average_score(self, agent_id: int | None = None) -> int:
        resolved_agent_id = _resolve_required_agent_id(agent_id, self._config.agent_id)
        contract = self._resolve_contract()
        return int(contract.functions.getAverageScore(resolved_agent_id).call())


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
        resolved_wallet = (
            wallet_address
            or self._config.agent_wallet_address
            or self._config.operator_wallet_address
            or account.address
        )
        capabilities = _capabilities_from_metadata(metadata)
        agent_uri = (metadata or {}).get(
            "agent_uri",
            "https://github.com/pedrov-dev/ai-trading-agents",
        )
        description = (metadata or {}).get(
            "description",
            f"{strategy_name} | operator={owner}",
        )

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


def _resolve_required_agent_id(
    agent_id: str | int | None,
    fallback_agent_id: int | None,
) -> int:
    resolved = _coerce_onchain_agent_id(agent_id)
    if resolved is not None:
        return resolved
    if fallback_agent_id is not None:
        return fallback_agent_id
    raise RuntimeError(
        "No numeric `agentId` is available yet. Register on `AgentRegistry` "
        "first or set `AGENT_ID`."
    )


def _checkpoint_note_text(checkpoint: Mapping[str, Any] | Any) -> str:
    if hasattr(checkpoint, "notes"):
        notes = tuple(str(note) for note in getattr(checkpoint, "notes", ()))
        if notes:
            return " | ".join(notes)
    if isinstance(checkpoint, Mapping):
        metric_name = str(checkpoint.get("metric_name", "checkpoint"))
        metric_value = checkpoint.get("metric_value", "")
        return f"{metric_name}={metric_value}"
    return "Local validation checkpoint submitted from the trading agent."


def _tx_hash_as_hex(tx_hash: Any) -> str:
    if isinstance(tx_hash, bytes):
        return f"0x{tx_hash.hex()}"
    if hasattr(tx_hash, "hex"):
        value = tx_hash.hex()
        return value if str(value).startswith("0x") else f"0x{value}"
    value = str(tx_hash)
    return value if value.startswith("0x") else f"0x{value}"


def _derive_account_address(private_key: str | None) -> str | None:
    if not private_key:
        return None
    try:
        from eth_account import Account

        return str(Account.from_key(private_key).address)
    except Exception:
        digest = hashlib.sha256(private_key.encode("utf-8")).hexdigest()[-40:]
        return f"0x{digest}"


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
