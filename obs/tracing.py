"""Langfuse tracing for rag-grounded.

rag-grounded calls `on_step(name, payload, duration_ms)` after every stage.
We turn each call into a span on the current request's trace. The trace for
the current request lives in a contextvar so concurrent requests don't mix.
"""
from __future__ import annotations

import contextvars
import os
import time
from datetime import datetime, timedelta, timezone

from langfuse import Langfuse

from rag_grounded.answer.schemas import Result
from rag_grounded.pipeline import RAGPipeline

from rag_grounded.answer.claims import cited_numbers, split_claims
from rag_grounded.embed import fingerprint

from . import store
from .cost import request_cost
from .retriever import TracedHybridRetriever, last_retrieval

_current = contextvars.ContextVar("langfuse_trace", default=None)

RELEASE = os.environ.get("OBS_RELEASE", "dev")


class Tracer:
    def __init__(self, client: Langfuse | None = None):
        self.lf = client or Langfuse(release=RELEASE)
        self.db = store.connect()
        # Two embedding models give valid vectors of the same dimension, so a
        # silent model change throws nothing. Put the fingerprint on every
        # request so it at least shows up as a step at a deploy.
        self.embed_fp = fingerprint()

    # rag-grounded hook
    def on_step(self, name: str, payload: dict, duration_ms: float) -> None:
        trace = _current.get()
        if trace is None:
            return
        end = datetime.now(timezone.utc)
        start = end - timedelta(milliseconds=duration_ms)
        payload = dict(payload)
        if name == "retrieve":
            payload.update(last_retrieval.get())
        elif name == "rerank":
            ranked = payload.get("ranked", [])
            kept = [cid for cid, _ in ranked]
            fused_top = last_retrieval.get().get("fused_top")
            payload.update(kept=kept, scores=[s for _, s in ranked],
                           # did rerank overrule retrieval's first choice?
                           fused_top_kept=fused_top in kept,
                           fused_top_rank=kept.index(fused_top) + 1 if fused_top in kept else None)
        if name == "generate":
            trace.generation(
                name="generate", start_time=start, end_time=end,
                input=payload.get("messages"), output=payload.get("response"),
                usage={"input": payload.get("input_tokens", 0),
                       "output": payload.get("output_tokens", 0), "unit": "TOKENS"},
            )
        else:
            trace.span(name=name, start_time=start, end_time=end, metadata=payload)

    def ask(self, pipe: RAGPipeline, query: str, user_id: str | None = None) -> Result:
        trace = self.lf.trace(name="rag.ask", input=query, user_id=user_id,
                              metadata={"model": pipe.llm.model,
                                        "prompt": f"{pipe.prompt.id}@v{pipe.prompt.version}",
                                        "embed_fp": self.embed_fp})
        token = _current.set(trace)
        t0 = time.perf_counter()
        base = {"trace_id": trace.id, "model": pipe.llm.model, "release": RELEASE,
                "prompt": f"{pipe.prompt.id}@v{pipe.prompt.version}", "embed_fp": self.embed_fp}
        try:
            res = pipe.ask(query)
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            trace.update(output={"error": repr(e)}, level="ERROR", tags=["error"])
            store.record(self.db, outcome="error", latency_ms=ms,
                         cost_usd=request_cost(pipe.llm.model, 0, 0, ms), **base)
            raise
        finally:
            _current.reset(token)

        t = res.trace
        out = res.output
        cost = request_cost(pipe.llm.model, t.input_tokens, t.output_tokens, t.timings_ms["total"])
        fully_cited = None
        if out.kind == "answer":
            claims = split_claims(out.text)
            fully_cited = int(all(cited_numbers(c) for c in claims)) if claims else 1
        trace.update(output=out.model_dump(), tags=[out.kind],
                     metadata={"cost_usd": cost, "timings_ms": t.timings_ms})
        store.record(self.db, outcome=out.kind, refusal_reason=getattr(out, "reason", None),
                     latency_ms=t.timings_ms["total"], input_tokens=t.input_tokens,
                     output_tokens=t.output_tokens, cost_usd=cost, fully_cited=fully_cited, **base)
        return res

    def flush(self) -> None:
        self.lf.flush()


def traced_pipeline(tracer: Tracer, **kw) -> RAGPipeline:
    kw.setdefault("retriever", TracedHybridRetriever())
    return RAGPipeline(on_step=tracer.on_step, **kw)
