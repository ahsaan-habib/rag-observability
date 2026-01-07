"""rag-grounded behind a traced API.

    uvicorn obs.api:app --port 8001
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from . import metrics, store
from .tracing import Tracer, traced_pipeline

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    tracer = Tracer()
    state["tracer"], state["pipe"] = tracer, traced_pipeline(tracer)
    yield
    tracer.flush()   # async-flushed during normal operation; drain on shutdown


app = FastAPI(title="rag-observability", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    user_id: str | None = None


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    res = state["tracer"].ask(state["pipe"], req.question, user_id=req.user_id)
    return {"output": res.output.model_dump(), "timings_ms": res.trace.timings_ms}


@app.get("/metrics")
def get_metrics(hours: float = 24) -> dict:
    return metrics.window(state["tracer"].db, time.time() - hours * 3600)


@app.get("/metrics/daily")
def get_daily(days: int = 14) -> dict:
    db = state["tracer"].db
    return {"days": metrics.daily(db, days), "deploys": store.deploys(db, time.time() - days * 86400)}
