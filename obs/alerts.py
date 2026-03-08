"""Compare the last hour against the previous 7 days and notify on breaches.

    obs alerts            # run from cron every 5 min
    obs weekly            # the things that are reviewed, not paged on

What pages and what doesn't:
  error rate above baseline   page now     unambiguous, rare, actionable
  cost per request doubling   page now     cheap to check, expensive to miss
  p95 regression              page if sustained across 3 windows (spiky otherwise)
  citation coverage / recall  weekly review + CI gate — too noisy per hour;
                              paging on it taught me to ignore the pager
  refusal rate climbing       weekly review — often correct behaviour
Notifications go to OBS_WEBHOOK_URL (any endpoint that accepts a JSON POST,
e.g. a self-hosted ntfy topic) or stdout.
"""
from __future__ import annotations

import os
import time

import httpx

from . import metrics, store

WEBHOOK = os.environ.get("OBS_WEBHOOK_URL")


def check(conn, now: float | None = None) -> list[str]:
    now = now or time.time()
    recent = metrics.window(conn, now - 3600, now)
    base = metrics.window(conn, now - 8 * 86400, now - 3600)
    if recent.get("requests", 0) < 10 or base.get("requests", 0) < 50:
        return []
    alerts = []
    if recent["error_rate"] > max(0.02, 2 * base["error_rate"]):
        alerts.append(f"error rate {recent['error_rate']:.1%} (baseline {base['error_rate']:.1%})")
    if recent["cost_per_request_usd"] > 2 * base["cost_per_request_usd"] > 0:
        alerts.append(f"cost/request ${recent['cost_per_request_usd']:.5f} "
                      f"(baseline ${base['cost_per_request_usd']:.5f})")
    # sustained: each of the last three 20-minute windows breaches
    slices = [metrics.window(conn, now - (k + 1) * 1200, now - k * 1200) for k in range(3)]
    if all(w.get("requests", 0) >= 3 and w["latency_p95_ms"] > 1.5 * base["latency_p95_ms"] for w in slices):
        alerts.append(f"p95 {recent['latency_p95_ms']:.0f} ms for 1h (baseline {base['latency_p95_ms']:.0f} ms)")
    return alerts


def weekly(conn, now: float | None = None) -> str:
    now = now or time.time()
    this = metrics.window(conn, now - 7 * 86400, now)
    prev = metrics.window(conn, now - 14 * 86400, now - 7 * 86400)
    lines = ["weekly review (this week vs last):"]
    for k in ("requests", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
              "cost_per_request_usd", "citation_coverage", "error_rate", "refusal_rate"):
        lines.append(f"  {k:<22} {this.get(k)!s:>10}   {prev.get(k)!s:>10}")
    evals = store.eval_runs(conn, now - 14 * 86400)
    if evals:
        e = evals[-1]
        lines.append(f"  last golden run ({e['dataset_version']}): recall {e['context_recall']:.3f}, "
                     f"faithfulness {e['faithfulness']:.3f}, refusal {e['refusal_correctness']:.3f}")
    return "\n".join(lines)


def notify(alerts: list[str]) -> None:
    text = "rag alerts:\n" + "\n".join(f"- {a}" for a in alerts)
    if WEBHOOK:
        httpx.post(WEBHOOK, json={"text": text}, timeout=10)
    else:
        print(text)


def run() -> int:
    alerts = check(store.connect())
    if alerts:
        notify(alerts)
    return len(alerts)
