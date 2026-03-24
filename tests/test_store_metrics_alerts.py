import sqlite3
import time

from obs import alerts, metrics, store
from obs.cost import _pricing, request_cost


def test_connect_migrates_an_old_database(tmp_path, monkeypatch):
    db = tmp_path / "old.db"
    old = sqlite3.connect(db)
    old.execute("CREATE TABLE requests (id INTEGER PRIMARY KEY, ts REAL NOT NULL, trace_id TEXT, "
                "outcome TEXT NOT NULL, refusal_reason TEXT, latency_ms REAL, input_tokens INTEGER, "
                "output_tokens INTEGER, cost_usd REAL, fully_cited INTEGER, model TEXT, prompt TEXT, release TEXT)")
    old.commit()
    monkeypatch.setattr(store, "DB", str(db))
    conn = store.connect()
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(requests)")}
    assert {"embed_fp", "trace_pruned"} <= cols
    store.connect()        # idempotent


def test_percentile_interpolates():
    assert metrics.percentile([], 0.5) == 0.0
    assert metrics.percentile([10, 20, 30, 40], 0.5) == 25.0
    assert metrics.percentile([10, 20, 30, 40], 1.0) == 40


def test_window_keeps_errors_and_refusals_apart(conn):
    now = time.time()
    for i, (outcome, cited) in enumerate([("answer", 1), ("answer", 0), ("refusal", None), ("error", None)]):
        store.record(conn, ts=now - 10, outcome=outcome, latency_ms=100 * (i + 1), cost_usd=0.001,
                     fully_cited=cited)
    w = metrics.window(conn, now - 60)
    assert w["requests"] == 4 and w["error_rate"] == 0.25 and w["refusal_rate"] == 0.25
    assert w["citation_coverage"] == 0.5 and w["latency_p50_ms"] == 250.0 and w["cost_total_usd"] == 0.004
    assert metrics.window(conn, now - 5) == {"requests": 0}
    assert len(metrics.daily(conn, 3)) == 3


def seed(conn, now, base_n, recent_n, recent_outcome="answer", recent_latency=100):
    for i in range(base_n):
        store.record(conn, ts=now - 2 * 86400 - i, outcome="answer", latency_ms=100, cost_usd=0.001)
    for i in range(recent_n):
        store.record(conn, ts=now - 60 * (i + 1) * 3, outcome=recent_outcome, latency_ms=recent_latency,
                     cost_usd=0.001)


def test_alerts_quiet_on_too_little_data_and_on_normal_traffic(conn):
    now = time.time()
    seed(conn, now, 10, 5, "error")
    assert alerts.check(conn, now) == []
    store.record(conn, ts=now - 1, outcome="answer", latency_ms=100, cost_usd=0.001)


def test_error_rate_and_sustained_latency_page(conn):
    now = time.time()
    seed(conn, now, 100, 20, "error", recent_latency=400)
    got = alerts.check(conn, now)
    assert any(a.startswith("error rate") for a in got) and any(a.startswith("p95") for a in got)


def test_weekly_mentions_last_golden_run(conn):
    store.record_eval(conn, {"dataset_version": "v3", "metrics": {
        "context_recall": 0.9, "context_precision": 0.8, "faithfulness": 0.88, "refusal_correctness": 0.9}})
    text = alerts.weekly(conn)
    assert "last golden run (v3)" in text and "faithfulness 0.880" in text


def test_notify_posts_to_webhook(monkeypatch, capsys):
    sent = []
    monkeypatch.setattr(alerts, "WEBHOOK", None)
    alerts.notify(["x"])
    assert "- x" in capsys.readouterr().out
    monkeypatch.setattr(alerts, "WEBHOOK", "http://hook")
    monkeypatch.setattr(alerts.httpx, "post", lambda url, json, timeout: sent.append((url, json)))
    alerts.notify(["y"])
    assert sent == [("http://hook", {"text": "rag alerts:\n- y"})]


def test_cost_charges_wall_time(tmp_path, monkeypatch):
    (tmp_path / "pricing.yaml").write_text("compute_usd_per_hour: 3.6\nmodels:\n  m:\n    input_per_m: 1.0\n"
                                           "    output_per_m: 2.0\n")
    monkeypatch.chdir(tmp_path)
    _pricing.cache_clear()
    try:
        assert request_cost("m", 1_000_000, 500_000, 1000) == 2.001
        assert request_cost("unknown", 10, 10, 0) == 0.0
    finally:
        _pricing.cache_clear()
