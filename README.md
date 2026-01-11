# rag-observability

Tracing, quality-over-time metrics and alerting for
[rag-grounded](https://github.com/ahsaan-habib/rag-grounded).

An AI pipeline degrades without producing any signal a normal monitor
watches for. Off-topic retrieval, a changed embedding model, a prompt edit
that dropped the citation rule: every one of those is a `200 OK` in normal
time. This layer exists to make "it got worse" visible and localisable.

```
             ┌──────────── rag-grounded (on_step hook) ────────────┐
 request ──▶ │ retrieve ─▶ rerank ─▶ generate ─▶ gate              │ ──▶ answer / refusal
             └────┬───────────┬──────────┬─────────┬───────────────┘
                  ▼           ▼          ▼         ▼
            Langfuse trace: spans with the *decision* (chunk ids, bm25 vs dense tops,
            overlap, rerank kept/scores, exact prompt, tokens)          ── what did it see?
                  │
            obs.db (sqlite): one row per request — outcome, latency, tokens,
            cost, fully_cited, prompt version, embed fingerprint        ── is it getting worse?
                  │
        /metrics · dashboard (deploy lines) · obs alerts · nightly golden-set points
```

## What's recorded

**Per span** (Langfuse): retrieve → `candidates`, `bm25_top`, `dense_top`,
`overlap`; rerank → `kept`, `scores`, whether rerank overruled the fused #1;
generate → exact messages sent, response, token usage; gate → unsupported
claims. Enough to reconstruct any request without re-running it.

**Per request** (sqlite): outcome (`answer` / `refusal` / `error`), refusal
reason, latency, tokens, cost, whether every claim carries a citation, model,
prompt version, release, embedding-model fingerprint.

## Metrics

| | |
|---|---|
| latency p50 / p95 / p99 | never the mean |
| cost per request | median; compute time × `compute_usd_per_hour` for local models, token prices for hosted ones (`pricing.yaml`) |
| citation coverage | share of answers where every claim is cited |
| error rate, refusal rate | separate — an error is a bug, a refusal is the gate working |
| golden-set recall / precision / faithfulness / refusal | from nightly `rag-eval-gate` runs (`obs eval-import`) |

Deploy markers (`obs deploy <release>`) are drawn on every chart; a step
change with no marker names nothing.

## Run it

```bash
docker compose up -d                 # langfuse on :3000, create a project, copy keys
cp .env.example .env && $EDITOR .env
pip install -e .
uvicorn obs.api:app --port 8001      # POST /ask, GET /metrics, dashboard at /
obs deploy v0.3.1 --note "bump sentence-transformers"
obs report --hours 24
obs alerts                           # cron it, see ops/crontab.example
```

## Costs of this layer

Storage grows with full prompts on every request; traces contain whatever
users typed, so they need the same retention and access rules as any user
data; and instrumentation is code that rots silently when the thing it wraps
is refactored.
