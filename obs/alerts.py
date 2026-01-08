"""Compare the last hour against the previous 7 days and notify on breaches.

    obs alerts            # run from cron every 5 min
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
    if recent["latency_p95_ms"] > 1.5 * base["latency_p95_ms"]:
        alerts.append(f"p95 {recent['latency_p95_ms']:.0f} ms (baseline {base['latency_p95_ms']:.0f} ms)")
    if (recent["citation_coverage"] or 1) < (base["citation_coverage"] or 0) - 0.05:
        alerts.append(f"citation coverage {recent['citation_coverage']:.1%} "
                      f"(baseline {base['citation_coverage']:.1%})")
    return alerts


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
