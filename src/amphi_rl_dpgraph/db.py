
# Thin SQLite connection factory for the PHI exposure context store.
# connect_db applies WAL mode and NORMAL sync for write throughput, and sets
# row_factory to sqlite3.Row for column-name access. open_context is a
# convenience wrapper. get_cross_modal_remask_count queries the remask_events
# audit table and returns 0 safely if the table does not yet exist.

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DBConfig:
    db_path: str
    timeout: float = 10.0
    check_same_thread: bool = False


def connect_db(cfg: DBConfig) -> sqlite3.Connection:
    Path(cfg.db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        cfg.db_path,
        timeout=cfg.timeout,
        check_same_thread=cfg.check_same_thread,
    )
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.row_factory = sqlite3.Row
    return conn


def open_context(db_path: str) -> sqlite3.Connection:
    return connect_db(DBConfig(db_path=db_path))


def get_cross_modal_remask_count(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute(
            """
            SELECT COUNT(*) AS n
            FROM remask_events
            WHERE trigger_reason = 'cross_modal_link_bonus'
            """
        ).fetchone()
        if row is not None:
            return int(row["n"])
    except Exception:
        pass
    return 0
