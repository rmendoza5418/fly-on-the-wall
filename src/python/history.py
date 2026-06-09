"""
history.py — SQLite-backed session history for Fly on the Wall.

Stores metadata for every completed recording so the app can surface
past sessions without relying on scattered temp files.

Schema:
    sessions(id, recorded_at, duration_secs, audio_path, transcript_path,
             summary_path, word_count, action_item_count, device_label)
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

DB_PATH = Path.home() / ".fotw" / "history.db"


def _ensure_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            recorded_at      TEXT    NOT NULL,
            duration_secs    REAL,
            audio_path       TEXT,
            transcript_path  TEXT,
            summary_path     TEXT,
            action_items_path TEXT,
            word_count       INTEGER,
            action_item_count INTEGER,
            device_label     TEXT,
            model_used       TEXT
        )
    """)
    conn.commit()


@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    _ensure_db(conn)
    try:
        yield conn
    finally:
        conn.close()


def log_session(
    recorded_at: datetime,
    duration_secs: float,
    audio_path: str,
    transcript_path: str,
    summary_path: str,
    action_items_path: str | None,
    word_count: int,
    action_item_count: int,
    device_label: str | None,
    model_used: str,
) -> int:
    """Insert a completed session and return the new row ID."""
    with _db() as conn:
        cur = conn.execute(
            """
            INSERT INTO sessions
                (recorded_at, duration_secs, audio_path, transcript_path,
                 summary_path, action_items_path, word_count, action_item_count,
                 device_label, model_used)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                recorded_at.isoformat(),
                round(duration_secs, 2),
                audio_path,
                transcript_path,
                summary_path,
                action_items_path,
                word_count,
                action_item_count,
                device_label,
                model_used,
            ),
        )
        conn.commit()
        return cur.lastrowid


def get_recent(limit: int = 20) -> list[dict]:
    """Return the N most recent sessions, newest first."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY recorded_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_session(session_id: int) -> dict | None:
    with _db() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
    return dict(row) if row else None


def delete_session(session_id: int, delete_files: bool = False) -> None:
    """Remove a session from history. Optionally delete associated files."""
    if delete_files:
        session = get_session(session_id)
        if session:
            for key in ("audio_path", "transcript_path", "summary_path", "action_items_path"):
                p = session.get(key)
                if p:
                    try:
                        Path(p).unlink(missing_ok=True)
                    except OSError:
                        pass

    with _db() as conn:
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()


def total_recording_time() -> float:
    """Sum of all session durations in seconds."""
    with _db() as conn:
        row = conn.execute("SELECT SUM(duration_secs) FROM sessions").fetchone()
    return row[0] or 0.0
