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
    assert store.stats() == {"documents": 0, "vectors": 0, "residuals": 0,
                             "meta": 0}


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


class _ChannelRuntime:
    """中心向量与 BM25 刻意指向不同切片，用于验证并集候选与相关度门限。"""

    dim = 4

    def __init__(self):
        self.seen_queries = []

    def encode_query(self, text):
        self.seen_queries.append(text)
        return np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    def encode_document(self, text):
        # "alpha" 切片中心与查询同向（向量通道第 1）；"beta" 切片中心正交。
        if "beta" in text:
            return np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        return np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)


def test_search_reranks_union_of_channels(tmp_path):
    """精排候选必须是两通道并集：BM25 头部不能挤掉中心向量头部。"""
    store = cs.ColBertDocStore(db_dir=str(tmp_path / "idx"),
                               runtime=_ChannelRuntime(), dim=4)
    store.add_chunks("Doc A", "/a.md", ["alpha"])
    store.add_chunks("Doc B", "/b.md", ["beta"])

    # 查询词 "beta"：BM25 只命中 B，中心向量只命中 A（coarse_k=1）。
    # 旧实现按 RRF 截断到 1 条时只剩 BM25 的 B；并集后 A（MaxSim 最高）
    # 仍能进入精排并拿到第 1。
    hits = store.search("beta", top_k=1, coarse_k=1)
    assert hits and hits[0]["source"] == "/a.md"
    assert hits[0]["score_per_token"] == pytest.approx(1.0)


def test_search_min_score_filters_irrelevant(tmp_path):
    store = cs.ColBertDocStore(db_dir=str(tmp_path / "idx"),
                               runtime=_ChannelRuntime(), dim=4)
    store.add_chunks("Doc A", "/a.md", ["alpha"])
    store.add_chunks("Doc B", "/b.md", ["beta"])

    # beta 切片 MaxSim=0 → 每 token 平均分 0，低于门限被丢弃；alpha 保留。
    hits = store.search("beta", top_k=2, coarse_k=2, min_score=0.5)
    assert [h["source"] for h in hits] == ["/a.md"]
    # 门限关闭（0）时两条都返回。
    assert len(store.search("beta", top_k=2, coarse_k=2)) == 2


def test_search_cleans_query_tokens_before_encoding(tmp_path):
    """查询进 ColBERT 前只做轻量清理（去标点/停用词，保留疑问词与形容词）。"""
    runtime = _ChannelRuntime()
    store = cs.ColBertDocStore(db_dir=str(tmp_path / "idx"),
                               runtime=runtime, dim=4)
    store.add_chunks("Doc A", "/a.md", ["alpha"])
    store.search("alpha 的 是？", top_k=1, coarse_k=1)
    assert runtime.seen_queries[-1] == "alpha"



def test_store_stats_without_loading_model(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("stats/list 不应加载 ColBERT 模型")

    monkeypatch.setattr(cs, "get_runtime", boom)
    store = cs.ColBertDocStore(db_dir=str(tmp_path / "idx"), runtime=None)
    assert store.stats()["documents"] == 0
    assert store.list_documents() == []


def test_doc_meta_upsert_get_list_and_delete_cascade(tmp_path):
    """doc_meta 第四表：upsert/模糊匹配/list，删除时与三表连带清理。"""
    runtime = FakeRuntime()
    store = cs.ColBertDocStore(db_dir=str(tmp_path / "idx"),
                               runtime=runtime, dim=8)
    store._chunk_encoder = runtime
    text = "## 认证\n\n" + ("OAuth2 认证说明。" * 30)
    store.index_text("手册", text, source="/docs/a.md",
                     max_chars=500, min_chars=300)
    store.upsert_doc_meta(
        "docs/a.md", "# 目录\n\n手册",
        chapters='[{"level": 2, "title": "认证"}]',
        md_path="/cache/a.md", source="/docs/a.md")
    assert store.list_doc_names() == ["docs/a.md"]
    assert store.stats()["meta"] == 1

    meta = store.get_doc_meta("docs/a.md")
    assert meta["digest"].startswith("# 目录")
    assert meta["md_path"] == "/cache/a.md"
    assert "认证" in meta["chapters"]
    # 模糊匹配语义对齐 VectorStore.get_document_digest：basename / 双向子串
    assert store.get_doc_meta("a.md")["doc_name"] == "docs/a.md"
    assert store.get_doc_meta("a")["doc_name"] == "docs/a.md"
    assert store.get_doc_meta("不存在") is None

    # 重复 upsert 覆盖不翻倍
    store.upsert_doc_meta("docs/a.md", "# 目录 v2", source="/docs/a.md")
    assert store.stats()["meta"] == 1
    assert store.get_doc_meta("docs/a.md")["digest"] == "# 目录 v2"

    # delete(source=) 连带清理：三表 + doc_meta 一起删
    store.delete(source="/docs/a.md")
    assert store.list_doc_names() == []
    assert store.stats() == {"documents": 0, "vectors": 0, "residuals": 0,
                             "meta": 0}
