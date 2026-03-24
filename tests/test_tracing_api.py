import pytest
from fastapi.testclient import TestClient

from rag_grounded.ingest.chunker import Chunk
from rag_grounded.llm.ollama import Generation
from rag_grounded.pipeline import RAGPipeline

from conftest import FakeLangfuse
from obs import api, metrics, store, tracing
from obs.retriever import TracedHybridRetriever

CHUNKS = {"c1": Chunk("c1", "laravel/eloquent.md", "Removing Global Scopes",
                      "Call withoutGlobalScope on the query builder to remove a global scope.")}


class Vectors:
    def search(self, q, k):
        return [("c1", 0.8)]

    def get(self, ids):
        return [CHUNKS[i] for i in ids if i in CHUNKS]


class BM25:
    def search(self, q, k):
        return [("c1", 7.5)]


class Reranker:
    def rank(self, q, chunks):
        return [(c, 4.0) for c in chunks]


class LLM:
    model = "fake"

    def __init__(self, text="Call withoutGlobalScope on the query builder to remove a global scope [1].", fail=False):
        self.text, self.fail = text, fail

    def chat(self, messages, **kw):
        if self.fail:
            raise RuntimeError("model down")
        return Generation(self.text, 100, 20, "fake")


@pytest.fixture
def tracer(conn, monkeypatch):
    monkeypatch.setattr(store, "connect", lambda: conn)
    return tracing.Tracer(client=FakeLangfuse())


def make_pipe(tracer, llm):
    return tracing.traced_pipeline(tracer, retriever=TracedHybridRetriever(vectors=Vectors(), bm25=BM25()),
                                   reranker=Reranker(), llm=llm)


def test_every_span_has_its_required_fields(tracer):
    res = tracer.ask(make_pipe(tracer, LLM()), "drop a scope?", user_id="u1")
    assert res.output.kind == "answer"
    t = tracer.lf.traces[0]
    assert {s["name"] for s in t.spans} == {"retrieve", "rerank", "gate"} and t.generations[0]["usage"]["input"] == 100
    assert not any("instrumentation-gap" in u.get("tags", []) for u in t.updates)
    retrieve = next(s for s in t.spans if s["name"] == "retrieve")["metadata"]
    assert retrieve["overlap"] == 1 and retrieve["fused_top"] == "c1"
    rerank = next(s for s in t.spans if s["name"] == "rerank")["metadata"]
    assert rerank["fused_top_kept"] is True and rerank["fused_top_rank"] == 1
    row = dict(tracer.db.execute("SELECT * FROM requests").fetchone())
    assert row["outcome"] == "answer" and row["fully_cited"] == 1 and row["trace_id"] == "trace-1"
    assert row["embed_fp"].startswith("BAAI/bge-small")


def test_missing_field_is_flagged_on_the_trace(tracer):
    pipe = RAGPipeline(retriever=TracedHybridRetriever(vectors=Vectors(), bm25=BM25()),
                       reranker=Reranker(), llm=LLM(), on_step=tracer.on_step)
    real = pipe.retriever.retrieve
    pipe.retriever.retrieve = lambda q: (real(q), tracing.last_retrieval.set({}))[0]   # a refactor "lost" it
    tracer.ask(pipe, "q")
    assert any("instrumentation-gap" in u.get("tags", []) for u in tracer.lf.traces[0].updates)


def test_errors_are_recorded_then_raised(tracer):
    with pytest.raises(RuntimeError):
        tracer.ask(make_pipe(tracer, LLM(fail=True)), "q")
    assert metrics.window(tracer.db, 0)["error_rate"] == 1.0


def test_api_ask_metrics_dashboard(tracer, monkeypatch):
    monkeypatch.setattr(api, "Tracer", lambda: tracer)
    monkeypatch.setattr(api, "traced_pipeline", lambda t: make_pipe(t, LLM()))
    with TestClient(api.app) as c:
        assert c.post("/ask", json={"question": "q"}).json()["output"]["kind"] == "answer"
        assert c.get("/metrics?hours=1").json()["requests"] == 1
        daily = c.get("/metrics/daily?days=2").json()
        assert len(daily["days"]) == 2 and daily["deploys"] == []
        assert "quality over time" in c.get("/").text
    assert tracer.lf.flushed
