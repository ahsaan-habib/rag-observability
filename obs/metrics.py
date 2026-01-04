"""Four metrics, and never the mean.

The average of p50 and p99 describes nobody's experience, and it moves the
wrong way: making 95% of requests faster and 5% catastrophically slower
improves it.
"""
from __future__ import annotations

import sqlite3
import time


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def window(conn: sqlite3.Connection, since: float, until: float | None = None) -> dict:
    until = until or time.time()
    rows = conn.execute("SELECT * FROM requests WHERE ts >= ? AND ts < ?", (since, until)).fetchall()
    n = len(rows)
    if not n:
        return {"requests": 0}
    lat = sorted(r["latency_ms"] for r in rows if r["latency_ms"] is not None)
    answers = [r for r in rows if r["outcome"] == "answer"]
    errors = sum(r["outcome"] == "error" for r in rows)
    refusals = sum(r["outcome"] == "refusal" for r in rows)
    costs = sorted(r["cost_usd"] or 0.0 for r in rows)
    return {
        "requests": n,
        "latency_p50_ms": round(percentile(lat, 0.50), 1),
        "latency_p95_ms": round(percentile(lat, 0.95), 1),
        "latency_p99_ms": round(percentile(lat, 0.99), 1),
        # median cost, not mean, for the same reason as latency
        "cost_per_request_usd": round(percentile(costs, 0.50), 6),
        "cost_total_usd": round(sum(costs), 4),
        "citation_coverage": round(sum(r["fully_cited"] or 0 for r in answers) / len(answers), 4)
        if answers else None,
        # an error is a bug; a refusal is the gate doing its job. never one number.
        "error_rate": round(errors / n, 4),
        "refusal_rate": round(refusals / n, 4),
    }


def daily(conn: sqlite3.Connection, days: int = 14) -> list[dict]:
    now = time.time()
    start = now - days * 86400
    out = []
    for d in range(days):
        a, b = start + d * 86400, start + (d + 1) * 86400
        out.append({"day_start": a, **window(conn, a, b)})
    return out
