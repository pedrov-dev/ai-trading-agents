"""Local JSONL helpers used for runtime summaries and journals."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


class JsonlFileStore:
    """Append JSON-serializable records to a JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, record: Mapping[str, Any]) -> None:
        self.append_many((record,))

    def append_many(self, records: Iterable[Mapping[str, Any]]) -> int:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with self._path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(dict(record), sort_keys=True) + "\n")
                written += 1
        return written

    def write_json(self, payload: Mapping[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._path.with_suffix(f"{self._path.suffix}.tmp")
        temp_path.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_path.replace(self._path)
