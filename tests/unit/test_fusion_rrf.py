from __future__ import annotations

import pytest

from core.query_engine.fusion import RRFusion
from core.types import RetrievalResult


def _result(chunk_id: str, score: float, text: str | None = None) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=text or chunk_id,
        metadata={"source": chunk_id},
    )


def test_rrf_fusion_merges_dense_and_sparse_rankings_deterministically() -> None:
    dense = [_result("chunk-1", 0.9), _result("chunk-2", 0.8), _result("chunk-3", 0.7)]
    sparse = [_result("chunk-2", 2.0), _result("chunk-4", 1.5), _result("chunk-1", 1.0)]

    fused = RRFusion(k=60).fuse(dense, sparse)

    assert [item.chunk_id for item in fused] == ["chunk-2", "chunk-1", "chunk-4", "chunk-3"]
    assert fused[0].score > fused[1].score > fused[2].score > fused[3].score


def test_rrf_fusion_respects_top_k_cutoff() -> None:
    dense = [_result("a", 0.9), _result("b", 0.8)]
    sparse = [_result("b", 1.0), _result("c", 0.7)]

    fused = RRFusion(k=10).fuse(dense, sparse, top_k=2)

    assert [item.chunk_id for item in fused] == ["b", "a"]


def test_rrf_fusion_prefers_first_payload_for_duplicate_chunk() -> None:
    dense = [_result("chunk-1", 0.9, text="dense text")]
    sparse = [_result("chunk-1", 1.0, text="sparse text")]

    fused = RRFusion().fuse(dense, sparse)

    assert len(fused) == 1
    assert fused[0].text == "dense text"
    assert fused[0].metadata == {"source": "chunk-1"}


def test_rrf_fusion_requires_positive_k() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        RRFusion(k=0)
