"""Lightweight ERC-8004-style identity registry for trading agents."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol


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
    """Registry wrapper for a registered identity plus local registry metadata."""

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
