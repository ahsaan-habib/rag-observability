"""Offline by default: fake embedder, temp sqlite, a recording stand-in for
the Langfuse client. Nothing here needs Langfuse or Ollama running."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("OBS_DB", os.path.join(tempfile.mkdtemp(prefix="obs-test-"), "obs.db"))

import fake_st  # noqa: E402

fake_st.install()

import pytest  # noqa: E402

from obs import store  # noqa: E402


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB", str(tmp_path / "obs.db"))
    return store.connect()


class FakeSpanHost:
    def __init__(self):
        self.id = "trace-1"
        self.spans, self.generations, self.updates = [], [], []

    def span(self, **kw):
        self.spans.append(kw)

    def generation(self, **kw):
        self.generations.append(kw)

    def update(self, **kw):
        self.updates.append(kw)


class FakeLangfuse:
    def __init__(self):
        self.traces, self.flushed = [], False

    def trace(self, **kw):
        t = FakeSpanHost()
        t.kw = kw
        self.traces.append(t)
        return t

    def flush(self):
        self.flushed = True
