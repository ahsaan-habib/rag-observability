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
    release       TEXT,
    embed_fp      TEXT                    -- embedding model fingerprint at request time
);
CREATE INDEX IF NOT EXISTS requests_ts ON requests(ts);

-- without a marker on the chart, a step change names nothing
CREATE TABLE IF NOT EXISTS deploys (
    id      INTEGER PRIMARY KEY,
    ts      REAL NOT NULL,
    release TEXT NOT NULL,
    note    TEXT
);
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


def mark_deploy(conn: sqlite3.Connection, release: str, note: str = "") -> None:
    conn.execute("INSERT INTO deploys (ts, release, note) VALUES (?, ?, ?)", (time.time(), release, note))
    conn.commit()


def deploys(conn: sqlite3.Connection, since: float = 0) -> list[dict]:
    return [dict(r) for r in conn.execute("SELECT * FROM deploys WHERE ts >= ? ORDER BY ts", (since,))]
