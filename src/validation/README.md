# Validation

Utilities to build validation artifacts and checkpoints from intents, executions, and performance snapshots.

## Key modules

- `artifacts.py`: `ValidationArtifact`, builders from trade intents, risk checks, execution results.
- `checkpoints.py`: `ValidationCheckpoint`, helpers to assemble checkpoints from artifacts.

## Public API

- `ValidationArtifact.from_trade_intent|from_risk_check|from_execution_result`, `ValidationArtifact.to_dict()`.
- `build_checkpoint_from_artifact`, `build_checkpoints`, `ValidationCheckpoint`.

## Integration

- Run after ingestion/execution; artifacts are persisted via the storage layer and consumed by identity/reputation components.

## Usage

```python
from validation import ValidationArtifact, build_checkpoint_from_artifact
artifact = ValidationArtifact.from_trade_intent(intent, agent_id='agent-1')
checkpoint = build_checkpoint_from_artifact(artifact)
```

## Notes

- Artifacts are JSON-serializable and intended for long-term storage in the artifacts directory or object store.
