from __future__ import annotations

import argparse
import json
import time


def cmd_deploy(args) -> None:
    from . import store

    store.mark_deploy(store.connect(), args.release, args.note)
    print(f"marked deploy {args.release}")


def cmd_report(args) -> None:
    from . import metrics, store

    conn = store.connect()
    print(json.dumps(metrics.window(conn, time.time() - args.hours * 3600), indent=2))


def cmd_alerts(args) -> None:
    from .alerts import run

    raise SystemExit(1 if run() else 0)


def cmd_eval_import(args) -> None:
    import os

    from . import store

    results = json.load(open(args.results))
    store.record_eval(store.connect(), results, os.environ.get("OBS_RELEASE", ""))
    print(f"recorded eval run {results.get('dataset_version')}: {results['metrics']}")


def cmd_weekly(args) -> None:
    from . import store
    from .alerts import weekly

    print(weekly(store.connect()))


def main() -> None:
    p = argparse.ArgumentParser(prog="obs")
    sub = p.add_subparsers(required=True)

    d = sub.add_parser("deploy", help="record a deploy marker (call it from your deploy script)")
    d.add_argument("release")
    d.add_argument("--note", default="")
    d.set_defaults(func=cmd_deploy)

    r = sub.add_parser("report", help="metrics for the last N hours")
    r.add_argument("--hours", type=float, default=24)
    r.set_defaults(func=cmd_report)

    e = sub.add_parser("eval-import", help="store a rag-eval-gate results.json as a point in time")
    e.add_argument("results")
    e.set_defaults(func=cmd_eval_import)

    a = sub.add_parser("alerts", help="check the last hour against the baseline, notify on breach")
    a.set_defaults(func=cmd_alerts)

    w = sub.add_parser("weekly", help="the review list: quality and refusal trends")
    w.set_defaults(func=cmd_weekly)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
