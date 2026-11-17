"""ColBERT 残差量化、流式反量化、MaxSim、RRF 与 LanceDB 三表测试。"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.rag import colbert_store as cs  # noqa: E402


class FakeRuntime:
    """低成本确定性编码器：dim=8，避免测试加载 587MB ONNX。"""

    dim = 8

    def _vec(self, text: str, n: int = 12) -> np.ndarray:
        seed = int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        v = rng.normal(size=(max(1, n), self.dim)).astype(np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

    def encode_documents(self, texts, batch_size=1):
        return [self._vec(t, max(1, min(20, len(t) // 10))) for t in texts]

    def encode_document(self, text):
        return self._vec(text, 16)

    def encode_query(self, text):
        return self._vec(text, 4)


def test_quantize_roundtrip_4bit_and_2bit():
    rng = np.random.default_rng(7)
    vecs = rng.normal(size=(24, 64)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    for bits in (4, 2):
        center, blob, scale, offset = cs.quantize_vectors(vecs, bits)
        assert len(blob) == (len(vecs) * 64 * bits + 7) // 8
        rec = np.concatenate(list(cs.iter_residual_vectors(
            blob, len(vecs), 64, bits, scale, offset, center)))
        assert rec.shape == vecs.shape
        assert np.isfinite(rec).all()
        assert float(np.max(np.abs(rec - vecs))) <= scale + 0.02


def test_streaming_maxsim_matches_direct_matrix():
    rng = np.random.default_rng(11)
    doc = rng.normal(size=(37, 64)).astype(np.float32)
    doc /= np.linalg.norm(doc, axis=1, keepdims=True)
    q = rng.normal(size=(5, 64)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    center, blob, scale, offset = cs.quantize_vectors(doc, 4)
    rec = np.concatenate(list(cs.iter_residual_vectors(
        blob, len(doc), 64, 4, scale, offset, center)))
    direct = float((q @ rec.T).max(axis=1).sum())
    streamed = cs.maxsim_stream(q, blob, len(doc), 64, 4, scale, offset, center,
                                batch_tokens=7)
    assert abs(direct - streamed) < 1e-4


def test_rrf_fuse_prefers_documents_in_both_lists():
    fused = cs.rrf_fuse([["a", "b", "c"], ["b", "c"]], limit=10)
    ids = [doc_id for doc_id, _ in fused]
    assert ids[0] == "b"
    assert ("a" in ids) and ("c" in ids)
    assert len(fused) == 3


def test_store_index_search_list_delete(tmp_path):
    runtime = FakeRuntime()
    store = cs.ColBertDocStore(db_dir=str(tmp_path / "idx"),
                               runtime=runtime, dim=8)
    store._chunk_encoder = runtime
    text = (
        "# 产品手册\n\n"
        "## 认证方式\n\n"
        + ("系统支持 API Key 和 OAuth2 两种认证方式。" * 12)
        + "\n\n## 限流\n\n"
        + ("默认每秒 10 次请求，超过会返回 429。" * 14)
    )
    n = store.index_text("产品手册", text, source="/docs/manual.md",
                         max_chars=500, min_chars=300)
    assert n >= 1
    stats = store.stats()
    assert stats["documents"] == n == stats["vectors"] == stats["residuals"]

    hits = store.search("OAuth2 认证", top_k=3, coarse_k=10)
    assert hits
    assert any("OAuth2" in h["content"] for h in hits)

    listed = store.list_documents()
    # 列表按文档去重：同一文档的多个切片只出现一次。
    assert len(listed) == 1
    assert listed[0]["title"] == "产品手册"
    assert listed[0]["source"] == "/docs/manual.md"

    removed = store.delete(source="/docs/manual.md")
    assert removed == n
    assert store.stats() == {"documents": 0, "vectors": 0, "residuals": 0}


def test_store_reindex_overwrites_same_source(tmp_path):
    runtime = FakeRuntime()
    store = cs.ColBertDocStore(db_dir=str(tmp_path / "idx"),
                               runtime=runtime, dim=8)
    store._chunk_encoder = runtime
    text = "## 认证\n\n" + ("OAuth2 认证说明。" * 30)
    n1 = store.index_text("手册", text, source="/docs/a.md",
                          max_chars=500, min_chars=300)
    n2 = store.index_text("手册", text, source="/docs/a.md",
                          max_chars=500, min_chars=300)
    assert n1 == n2
    assert store.stats()["documents"] == n1, "同源重建不能翻倍"


def test_store_stats_without_loading_model(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("stats/list 不应加载 ColBERT 模型")

    monkeypatch.setattr(cs, "get_runtime", boom)
    store = cs.ColBertDocStore(db_dir=str(tmp_path / "idx"), runtime=None)
    assert store.stats()["documents"] == 0
    assert store.list_documents() == []
