# Identity

Helpers for ERC-8004-style agent registration and reputation scoring.

## Key modules

- `erc8004_registry.py`: registry implementations (local and on-chain helpers).
- `reputation.py`: reputation engine and snapshots.
- `__init__.py`: package exports.

## Public API

- `LocalERC8004Registry.register(...)->AgentIdentity`, `get(agent_id)->AgentIdentity|None`.
- `ReputationEngine.apply_artifact(snapshot, artifact)->ReputationSnapshot`.
- `ReputationRegistryClient.get_average_score(agent_id)->int`.

## Integration

- Local in-memory registry for runtime; on-chain clients use `SepoliaContractsConfig` (RPC/keys); `ReputationEngine` consumes `validation.ValidationArtifact`.

## Usage

```python
registry = LocalERC8004Registry()
identity = registry.register(display_name='bot', strategy_name='mean-revert', owner='alice')
snapshot = ReputationEngine().initialize(identity.agent_id)
```

## Notes

- On-chain features require RPC endpoints & keys; local registry is in-memory only and intended for testing/dry-run.
