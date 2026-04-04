# Storage

Backends and adapters for raw payloads, artifacts, and runtime state.

## Key modules

- `local_runtime.py`: local file/in-memory stores and JSONL logging utilities.
- `object_storage.py`: S3-compatible adapter (boto3) for object persistence.
- `raw_postgres.py`: Postgres repositories for raw events and run state.
- `raw_ingestion.py`: ingestion pipeline orchestrating stores.

## Public API

- `put_json_gzip(key, payload, metadata)`, `fetch_document(key)` for object stores.
- `start_run` / `finish_run` for run lifecycle; `insert_raw_event`, `fetch_raw_events_for_classification` for raw events.

## Integration

- Depends on Postgres (DB connection factory) and S3-compatible stores; called by ingestion and scheduler components.

## Usage

```python
obj = S3CompatibleObjectStore.from_config(ObjectStorageConfig.from_env())
obj.put_json_gzip(key="raw/.../id.json.gz", payload=payload_dict, metadata={"source_type":"rss"})
data = obj.fetch_document(key="raw/.../id.json.gz")
```

## Notes

- Configure credentials via env vars; local stores are for dry-run; object key conventions follow `raw/source_type=.../date=.../hour=...`.
