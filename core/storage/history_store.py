"""SQLite-backed persistence for compact VisionForge analysis history.

Only metadata and summary fields are stored. Images, videos, model weights, and
other large artifacts remain in the filesystem.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    completed_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    filename TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_analysis_history_completed_at
    ON analysis_history(completed_at DESC, id DESC);
CREATE INDEX IF NOT EXISTS idx_analysis_history_kind
    ON analysis_history(kind);
"""


class HistoryStore:
    """Small, thread-safe-per-connection SQLite history store."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path), timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    @staticmethod
    def _validated_record(record: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(record, dict):
            raise TypeError("History record must be a dictionary")
        # Ensure JSON serialization fails here, with a useful message, not later.
        try:
            json.dumps(record, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"History record is not JSON-serializable: {exc}") from exc
        completed_at = record.get("timestamp") or record.get("completed_at")
        if not isinstance(completed_at, str) or not completed_at.strip():
            raise ValueError("History record must contain a non-empty timestamp")
        kind = record.get("kind", "unknown")
        filename = record.get("filename", "unknown")
        return {**record, "timestamp": completed_at, "kind": str(kind), "filename": str(filename)}

    def save(self, record: dict[str, Any]) -> int:
        """Persist one record and return its database row ID."""
        payload = self._validated_record(record)
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO analysis_history(completed_at, kind, filename, payload_json) VALUES (?, ?, ?, ?)",
                (payload["timestamp"], payload["kind"], payload["filename"],
                 json.dumps(payload, ensure_ascii=False, allow_nan=False)),
            )
            return int(cursor.lastrowid)

    def save_many(self, records: Iterable[dict[str, Any]]) -> int:
        """Persist a batch atomically; return the number of saved records."""
        prepared = [self._validated_record(record) for record in records]
        rows = [
            (r["timestamp"], r["kind"], r["filename"], json.dumps(r, ensure_ascii=False, allow_nan=False))
            for r in prepared
        ]
        with self._connect() as connection:
            connection.executemany(
                "INSERT INTO analysis_history(completed_at, kind, filename, payload_json) VALUES (?, ?, ?, ?)", rows
            )
        return len(rows)

    def load_all(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Return records oldest-to-newest, suitable for the existing history UI."""
        if limit is not None and (not isinstance(limit, int) or limit < 1):
            raise ValueError("limit must be a positive integer or None")
        query = "SELECT payload_json FROM analysis_history ORDER BY id DESC"
        params: tuple[Any, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (limit,)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        records: list[dict[str, Any]] = []
        for row in reversed(rows):
            try:
                record = json.loads(row["payload_json"])
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(record, dict):
                records.append(record)
        return records

    def count(self, kind: str | None = None) -> int:
        with self._connect() as connection:
            if kind is None:
                row = connection.execute("SELECT COUNT(*) AS n FROM analysis_history").fetchone()
            else:
                row = connection.execute(
                    "SELECT COUNT(*) AS n FROM analysis_history WHERE kind = ?", (kind,)
                ).fetchone()
        return int(row["n"])

    def clear(self) -> None:
        """Delete history records. Intended for an explicit user-requested reset."""
        with self._connect() as connection:
            connection.execute("DELETE FROM analysis_history")
            connection.execute("DELETE FROM sqlite_sequence WHERE name = 'analysis_history'")
