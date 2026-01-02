"""Langfuse tracing for rag-grounded.

rag-grounded calls `on_step(name, payload, duration_ms)` after every stage.
We turn each call into a span on the current request's trace. The trace for
the current request lives in a contextvar so concurrent requests don't mix.
"""
from __future__ import annotations

import contextvars
import os
from datetime import datetime, timedelta, timezone

from langfuse import Langfuse

from rag_grounded.answer.schemas import Result
from rag_grounded.pipeline import RAGPipeline

from .retriever import TracedHybridRetriever, last_retrieval

_current = contextvars.ContextVar("langfuse_trace", default=None)

RELEASE = os.environ.get("OBS_RELEASE", "dev")


class Tracer:
    def __init__(self, client: Langfuse | None = None):
        self.lf = client or Langfuse(release=RELEASE)

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
                                        "prompt": f"{pipe.prompt.id}@v{pipe.prompt.version}"})
        token = _current.set(trace)
        try:
            res = pipe.ask(query)
        except Exception as e:
            trace.update(output={"error": repr(e)}, level="ERROR", tags=["error"])
            raise
        finally:
            _current.reset(token)
        trace.update(output=res.output.model_dump(), tags=[res.output.kind])
        return res

    def flush(self) -> None:
        self.lf.flush()


def traced_pipeline(tracer: Tracer, **kw) -> RAGPipeline:
    kw.setdefault("retriever", TracedHybridRetriever())
    return RAGPipeline(on_step=tracer.on_step, **kw)
