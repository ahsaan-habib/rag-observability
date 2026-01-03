"""One row per request. Langfuse holds the full traces; this holds the numbers
we aggregate, so metrics don't depend on the trace backend being up."""
from __future__ import annotations

import os
import sqlite3
import time
from contextlib import closing

DB = os.environ.get("OBS_DB", "obs.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id            INTEGER PRIMARY KEY,
    ts            REAL NOT NULL,
    trace_id      TEXT,
    outcome       TEXT NOT NULL,          -- answer | refusal | error
    refusal_reason TEXT,
    latency_ms    REAL,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    cost_usd      REAL,
    fully_cited   INTEGER,                -- answers only: every claim carries [n]
    model         TEXT,
    prompt        TEXT,
    release       TEXT
);
CREATE INDEX IF NOT EXISTS requests_ts ON requests(ts);
"""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def record(conn: sqlite3.Connection, **row) -> None:
    row.setdefault("ts", time.time())
    cols = ",".join(row)
    with closing(conn.cursor()) as cur:
        cur.execute(f"INSERT INTO requests ({cols}) VALUES ({','.join('?' * len(row))})",
                    list(row.values()))
    conn.commit()
