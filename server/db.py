"""SQLite persistence and the task state machine.

Task lifecycle::

    pending --claim--> processing --result--> refining --done--> ready
        ^                  |                                     |
        +---- lease expired+          any state --explicit failure--> failed
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    token TEXT UNIQUE NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    status TEXT NOT NULL DEFAULT 'pending',
    original_filename TEXT NOT NULL,
    audio_ext TEXT NOT NULL,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    duration_sec REAL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    claimed_by TEXT,
    lease_until TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    transcript_text TEXT,
    segments_json TEXT,
    worker_meta TEXT,
    summary_json TEXT,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status, created_at);
CREATE INDEX IF NOT EXISTS idx_tasks_user ON tasks(user_id, created_at DESC);
"""

# Statuses visible to clients and workers.
PENDING = "pending"
PROCESSING = "processing"
REFINING = "refining"
READY = "ready"
FAILED = "failed"

TASK_COLUMNS = (
    "id, user_id, status, original_filename, audio_ext, size_bytes, duration_sec, "
    "created_at, updated_at, claimed_by, lease_until, attempts, transcript_text, "
    "segments_json, worker_meta, summary_json, error"
)


def utcnow() -> str:
    # Microsecond precision keeps lexicographic comparisons of ISO strings exact.
    return datetime.now(timezone.utc).isoformat()


class Database:
    """Thread-safe wrapper around a single SQLite connection (WAL mode)."""

    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------- users

    def create_user(self, username: str, password_hash: str, token: str) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO users (username, password_hash, token, created_at) VALUES (?, ?, ?, ?)",
                (username, password_hash, token, utcnow()),
            )
            return int(cur.lastrowid)

    def get_user_by_username(self, username: str) -> sqlite3.Row | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()
        return row

    def get_user_by_token(self, token: str) -> sqlite3.Row | None:
        if not token:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM users WHERE token = ?", (token,)
            ).fetchone()
        return row

    # ------------------------------------------------------------- tasks

    def create_task(
        self,
        user_id: int,
        original_filename: str,
        audio_ext: str,
        size_bytes: int,
        duration_sec: float | None,
    ) -> int:
        now = utcnow()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO tasks (user_id, status, original_filename, audio_ext, size_bytes,"
                " duration_sec, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, PENDING, original_filename, audio_ext, size_bytes, duration_sec, now, now),
            )
            return int(cur.lastrowid)

    def get_task(self, task_id: int) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(
                f"SELECT {TASK_COLUMNS} FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()

    def list_tasks(self, user_id: int, limit: int = 50) -> list[sqlite3.Row]:
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {TASK_COLUMNS} FROM tasks WHERE user_id = ?"
                " ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return list(rows)

    def claim_task(self, worker_id: str, lease_seconds: int) -> sqlite3.Row | None:
        """Atomically hand the oldest runnable task to a worker.

        ``processing`` tasks whose lease has expired are reclaimable, so a
        crashed worker cannot stall the queue forever.
        """
        now = datetime.now(timezone.utc)
        lease_until = (now + timedelta(seconds=lease_seconds)).isoformat()
        now_iso = now.isoformat()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE tasks SET status = ?, claimed_by = ?, lease_until = ?,"
                " attempts = attempts + 1, updated_at = ?"
                " WHERE id = ("
                "   SELECT id FROM tasks"
                "   WHERE status = ? OR (status = ? AND lease_until < ?)"
                "   ORDER BY created_at LIMIT 1)"
                " RETURNING " + TASK_COLUMNS,
                (PROCESSING, worker_id, lease_until, now_iso, PENDING, PROCESSING, now_iso),
            )
            row = cur.fetchone()
            self._conn.commit()
        return row

    def renew_lease(self, task_id: int, worker_id: str, lease_seconds: int) -> bool:
        lease_until = (
            datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)
        ).isoformat()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE tasks SET lease_until = ?, updated_at = ?"
                " WHERE id = ? AND status = ? AND claimed_by = ?",
                (lease_until, utcnow(), task_id, PROCESSING, worker_id),
            )
            return cur.rowcount > 0

    def store_result(
        self,
        task_id: int,
        transcript_text: str,
        segments: list[dict],
        duration_sec: float | None,
        worker_meta: dict | None,
    ) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE tasks SET status = ?, transcript_text = ?, segments_json = ?,"
                " duration_sec = COALESCE(?, duration_sec), worker_meta = ?, updated_at = ?"
                " WHERE id = ? AND status = ?",
                (
                    REFINING,
                    transcript_text,
                    json_dumps(segments),
                    duration_sec,
                    json_dumps(worker_meta) if worker_meta else None,
                    utcnow(),
                    task_id,
                    PROCESSING,
                ),
            )
            return cur.rowcount > 0

    def update_size(self, task_id: int, size_bytes: int) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE tasks SET size_bytes = ?, updated_at = ? WHERE id = ?",
                (size_bytes, utcnow(), task_id),
            )

    def delete_task(self, task_id: int, user_id: int) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM tasks WHERE id = ? AND user_id = ?", (task_id, user_id)
            )
            return cur.rowcount > 0

    def delete_finished(self, user_id: int) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM tasks WHERE user_id = ? AND status IN ('ready', 'failed')",
                (user_id,),
            )
            return cur.rowcount

    def delete_all(self, user_id: int) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM tasks WHERE user_id = ?", (user_id,)
            )
            return cur.rowcount

    def mark_ready(self, task_id: int, summary: dict) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE tasks SET status = ?, summary_json = ?, error = NULL, updated_at = ? WHERE id = ?",
                (READY, json_dumps(summary), utcnow(), task_id),
            )

    def mark_failed(self, task_id: int, error: str, from_statuses: tuple[str, ...] = (PENDING, PROCESSING)) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute(
                f"UPDATE tasks SET status = ?, error = ?, updated_at = ?"
                f" WHERE id = ? AND status IN ({','.join('?' * len(from_statuses))})",
                (FAILED, error[:2000], utcnow(), task_id, *from_statuses),
            )
            return cur.rowcount > 0


def json_dumps(obj) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)
