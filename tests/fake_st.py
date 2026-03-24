"""A stand-in for sentence-transformers so the tests run in seconds, offline,
without torch or a model download.

Bag-of-words hashed into 256 dims, then normalised: texts that share words
score close, unrelated texts score near zero. Crude, deterministic, and enough
to exercise every code path that only needs "similar" vs "not similar".
"""
from __future__ import annotations

import re
import sys
import types
import zlib

import numpy as np

DIM = 256
_WORD = re.compile(r"[a-z0-9]+")


def _vec(text: str) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    for w in _WORD.findall(text.lower()):
        v[zlib.crc32(w.encode()) % DIM] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


class SentenceTransformer:
    def __init__(self, name: str = "fake", *a, **kw):
        self.name = name

    def encode(self, texts, batch_size: int = 32, normalize_embeddings: bool = True,
               show_progress_bar: bool = False, **kw):
        if isinstance(texts, str):
            return _vec(texts)
        return np.array([_vec(t) for t in texts]) if texts else np.zeros((0, DIM), dtype=np.float32)


class CrossEncoder:
    def __init__(self, name: str = "fake", *a, **kw):
        self.name = name

    def predict(self, pairs, **kw):
        # word overlap scaled into logit-ish territory: 0 overlap -> -5
        out = []
        for q, d in pairs:
            qs, ds = set(_WORD.findall(q.lower())), set(_WORD.findall(d.lower()))
            out.append(10.0 * len(qs & ds) / max(len(qs), 1) - 5.0)
        return np.array(out, dtype=np.float32)


def install() -> None:
    """Make `import sentence_transformers` resolve to this module."""
    if "sentence_transformers" not in sys.modules:
        mod = types.ModuleType("sentence_transformers")
        mod.SentenceTransformer = SentenceTransformer
        mod.CrossEncoder = CrossEncoder
        sys.modules["sentence_transformers"] = mod
