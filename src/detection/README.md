# Detection

Rule-based event detection that classifies raw payload previews (news / price previews) and persists detected events.

## Key modules

- `event_detection.py`: detectors & rule implementations.
- `event_detection_service.py`: classification service for pending raw payloads.
- `event_detection_postgres.py`: Postgres repository adapter for detected events.
- `__init__.py`: package wiring and exports.

## Public API

- `RuleBasedEventDetector.detect(source_type, payload_preview) -> list[DetectedEvent]`
- `EventDetectionService.classify_pending_events(source_type='rss', batch_size=100) -> int`
- `PostgresEventDetectionRepository.insert_detected_event(...)`

## Integration

- Integrates with `storage.raw_ingestion.RawEventsRepository`, Postgres via `connection_factory()`, and optionally scheduled via `InfoScheduler`.

## Usage

```python
detector = RuleBasedEventDetector()
repo = PostgresEventDetectionRepository(conn_factory)
EventDetectionService(detector=detector, raw_events_repository=raw_repo, event_detection_repository=repo).classify_pending_events()
```

## Notes

- Requires a `detected_events` table (stores `metadata` as `jsonb`); `connection_factory()` must supply DB connections; payload previews are expected on raw events.
