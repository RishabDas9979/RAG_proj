"""Append-only, hash-chained audit log backed by SQLite.

Each entry stores the hash of the previous entry, so editing or removing any row breaks
the chain from that point on. SQLite triggers also reject UPDATE and DELETE.

Limits: someone with direct database access can drop the triggers and rewrite the *whole*
chain consistently. To catch that, store ``head()`` somewhere else (a different system or a
signed receipt) and pass it to ``verify(expected_head=...)``.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Callable

GENESIS = "0" * 64

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL, actor TEXT NOT NULL, role TEXT NOT NULL, action TEXT NOT NULL,
    resource TEXT NOT NULL, outcome TEXT NOT NULL, details TEXT NOT NULL,
    prev_hash TEXT NOT NULL, hash TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS entries_no_update BEFORE UPDATE ON entries
BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS entries_no_delete BEFORE DELETE ON entries
BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
"""

_FIELDS = ("ts", "actor", "role", "action", "resource", "outcome", "details")


def _digest(prev_hash: str, record: dict) -> str:
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256((prev_hash + canonical).encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, path: str = ":memory:", clock: Callable[[], str] | None = None):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._lock = threading.Lock()
        self._clock = clock or (lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def now(self) -> str:
        return self._clock()

    def append(self, actor: str, role: str, action: str, resource: str, outcome: str,
               details: dict | None = None) -> int:
        with self._lock:
            row = self._conn.execute("SELECT hash FROM entries ORDER BY seq DESC LIMIT 1").fetchone()
            prev = row["hash"] if row else GENESIS
            record = {
                "ts": self._clock(), "actor": actor, "role": role, "action": action,
                "resource": resource, "outcome": outcome,
                "details": json.dumps(details or {}, sort_keys=True, separators=(",", ":")),
            }
            cur = self._conn.execute(
                "INSERT INTO entries (ts, actor, role, action, resource, outcome, details, prev_hash, hash) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (*[record[f] for f in _FIELDS], prev, _digest(prev, record)),
            )
            self._conn.commit()
            return cur.lastrowid

    def entries(self, limit: int | None = None) -> list[dict]:
        sql = "SELECT * FROM entries ORDER BY seq"
        rows = self._conn.execute(sql + (f" LIMIT {int(limit)}" if limit else "")).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["details"] = json.loads(d["details"])
            out.append(d)
        return out

    def head(self) -> tuple[int, str]:
        row = self._conn.execute("SELECT seq, hash FROM entries ORDER BY seq DESC LIMIT 1").fetchone()
        return (row["seq"], row["hash"]) if row else (0, GENESIS)

    def verify(self, expected_head: tuple[int, str] | None = None) -> tuple[bool, int | None]:
        """Return (ok, first_bad_seq). Recomputes every hash and checks every link."""
        prev, expected_seq = GENESIS, 0
        for r in self._conn.execute("SELECT * FROM entries ORDER BY seq"):
            expected_seq += 1
            record = {f: r[f] for f in _FIELDS}
            if r["seq"] != expected_seq or r["prev_hash"] != prev or r["hash"] != _digest(prev, record):
                return False, r["seq"]
            prev = r["hash"]
        if expected_head is not None and expected_head != (expected_seq, prev):
            return False, expected_seq  # truncated or replaced since the head was recorded
        return True, None

    def close(self) -> None:
        self._conn.close()

    def __del__(self):
        try:
            self._conn.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
