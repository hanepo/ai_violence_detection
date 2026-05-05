from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    threat_score REAL NOT NULL,
    cv REAL NOT NULL,
    ca REAL NOT NULL,
    video_path TEXT,
    metadata_json TEXT
);
"""


@dataclass
class EventRecord:
    event_id: str
    threat_score: float
    cv: float
    ca: float
    video_path: str | None = None
    metadata_json: str | None = None


class EventDatabase:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute(SCHEMA_SQL)
        self.conn.commit()

    def insert_event(self, record: EventRecord) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """
            INSERT INTO events (event_id, timestamp_utc, threat_score, cv, ca, video_path, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.event_id,
                now,
                record.threat_score,
                record.cv,
                record.ca,
                record.video_path,
                record.metadata_json,
            ),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def list_recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        cur = self.conn.execute(
            """
            SELECT event_id, timestamp_utc, threat_score, cv, ca, video_path, metadata_json
            FROM events
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
        items: list[dict[str, Any]] = []
        for r in rows:
            metadata_raw = r[6]
            metadata: dict[str, Any] = {}
            if isinstance(metadata_raw, str) and metadata_raw:
                try:
                    loaded = json.loads(metadata_raw)
                    if isinstance(loaded, dict):
                        metadata = loaded
                except Exception:
                    metadata = {}

            items.append(
                {
                    "event_id": r[0],
                    "timestamp_utc": r[1],
                    "threat_score": r[2],
                    "cv": r[3],
                    "ca": r[4],
                    "video_path": r[5],
                    "metadata_json": metadata_raw,
                    "metadata": metadata,
                    "alert_timestamp_utc": metadata.get("alert_timestamp_utc"),
                    "clip_duration_seconds": metadata.get("clip_duration_seconds"),
                }
            )

        return items

    def get_stats(self) -> dict[str, Any]:
        total = self.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        high_risk = self.conn.execute("SELECT COUNT(*) FROM events WHERE threat_score >= 0.8").fetchone()[0]
        avg_score_row = self.conn.execute("SELECT AVG(threat_score) FROM events").fetchone()[0]
        avg_score = float(avg_score_row) if avg_score_row is not None else 0.0
        return {
            "total_events": int(total),
            "high_risk_events": int(high_risk),
            "avg_threat_score": avg_score,
        }
