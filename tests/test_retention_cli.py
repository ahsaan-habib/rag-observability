import json
import sys
import time

from obs import cli, retention, store


def test_prune_keeps_errors_refusals_and_a_sample(conn, monkeypatch):
    old = time.time() - 30 * 86400
    for i in range(50):
        store.record(conn, ts=old, outcome="answer", trace_id=f"t{i}")
    store.record(conn, ts=old, outcome="error", trace_id="err")
    store.record(conn, ts=time.time(), outcome="answer", trace_id="fresh")
    doomed = retention.prune(conn, dry_run=True)
    kept = sum(retention._keep(f"t{i}") for i in range(50))
    assert doomed == 50 - kept and 0 < kept < 50

    deleted = []

    class FakeClient:
        def __init__(self, **kw):
            assert kw["auth"] == ("pk", "sk")

        def delete(self, path):
            deleted.append(path)
            return type("R", (), {"status_code": 404 if path.endswith("t0") else 204})()

    monkeypatch.setattr(retention.httpx, "Client", FakeClient)
    monkeypatch.setenv("LANGFUSE_HOST", "http://lf")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    assert retention.prune(conn) == doomed
    assert not any(p.endswith(("/err", "/fresh")) for p in deleted)
    assert retention.prune(conn, dry_run=True) == 0     # marked, not pruned twice


def test_cli_deploy_eval_import_report(conn, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(store, "connect", lambda: conn)
    res = tmp_path / "nightly.json"
    res.write_text(json.dumps({"dataset_version": "v3", "metrics": {
        "context_recall": 0.9, "context_precision": 0.8, "faithfulness": 0.88, "refusal_correctness": 0.9}}))
    for argv in (["deploy", "v0.3.1", "--note", "bump"], ["eval-import", str(res)], ["report", "--hours", "1"],
                 ["weekly"], ["retention", "--dry-run"]):
        monkeypatch.setattr(sys, "argv", ["obs", *argv])
        cli.main()
    out = capsys.readouterr().out
    assert "marked deploy v0.3.1" in out and "recorded eval run v3" in out and "would prune 0 traces" in out
    assert store.deploys(conn)[0]["release"] == "v0.3.1" and store.eval_runs(conn)[0]["faithfulness"] == 0.88
