"""HybridRetriever that remembers what each half returned.

Record the DECISION, not just the duration. When quality drops the question
is always "what did it see", never "how long did it take".
"""
from __future__ import annotations

import contextvars

from rag_grounded.retrieval.fuse import reciprocal_rank_fusion
from rag_grounded.retrieval.hybrid import HybridRetriever

last_retrieval: contextvars.ContextVar[dict] = contextvars.ContextVar("last_retrieval", default={})


class TracedHybridRetriever(HybridRetriever):
    def retrieve(self, query: str) -> list[str]:
        bm25 = self.bm25.search(query, self.per_retriever)
        dense = self.vectors.search(query, self.per_retriever)
        bm25_ids, dense_ids = [c for c, _ in bm25], [c for c, _ in dense]
        fused = reciprocal_rank_fusion([bm25_ids, dense_ids])[: self.candidates]
        last_retrieval.set({
            "bm25_top": bm25_ids[:3],
            "bm25_scores": [round(s, 3) for _, s in bm25[:3]],
            "dense_top": dense_ids[:3],
            "dense_scores": [round(s, 3) for _, s in dense[:3]],
            "overlap": len(set(bm25_ids) & set(dense_ids)),
            "fused_top": fused[0] if fused else None,
        })
        return fused
