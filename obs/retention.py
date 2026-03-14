"""Trace retention. Traces hold whatever users typed — that's user data, so
retention is a decision, not a default.

Policy: keep every error and refusal. Keep successful-request traces in full
for 14 days, then keep a deterministic 20% sample (hash of the trace id, so a
re-run prunes the same ones). The sqlite numbers are tiny and kept as-is.

    obs retention --dry-run
"""
from __future__ import annotations

import hashlib
import os
import time

import httpx

KEEP_DAYS = int(os.environ.get("OBS_KEEP_FULL_DAYS", "14"))
SAMPLE = float(os.environ.get("OBS_SAMPLE_AFTER", "0.20"))


def _keep(trace_id: str) -> bool:
    h = int(hashlib.sha1(trace_id.encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < SAMPLE


def prune(conn, dry_run: bool = False) -> int:
    cutoff = time.time() - KEEP_DAYS * 86400
    rows = conn.execute(
        "SELECT id, trace_id FROM requests WHERE outcome = 'answer' AND ts < ? AND trace_pruned = 0"
        " AND trace_id IS NOT NULL", (cutoff,)).fetchall()
    doomed = [r for r in rows if not _keep(r["trace_id"])]
    if dry_run:
        return len(doomed)
    http = httpx.Client(base_url=os.environ["LANGFUSE_HOST"], timeout=30,
                        auth=(os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"]))
    n = 0
    for r in doomed:
        resp = http.delete(f"/api/public/traces/{r['trace_id']}")
        if resp.status_code in (200, 204, 404):
            conn.execute("UPDATE requests SET trace_pruned = 1 WHERE id = ?", (r["id"],))
            n += 1
    conn.commit()
    return n
