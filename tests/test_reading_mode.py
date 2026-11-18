"""检索阅读模式：RAGEngine 旁路切片/入库/检索与 rag_tools 气泡分流。"""
from __future__ import annotations

import hashlib
import os
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.config import AgentConfig  # noqa: E402
from agent.rag import colbert_store as cs  # noqa: E402
from agent.rag.engine import RAGEngine  # noqa: E402
from agent.tools.rag_tool import RagSearchTool  # noqa: E402


class FakeRuntime:
    dim = 8

    def _vec(self, text: str, n: int = 12) -> np.ndarray:
        seed = int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        v = rng.normal(size=(max(1, n), self.dim)).astype(np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

    def encode_documents(self, texts, batch_size=1):
        return [self._vec(t, 12) for t in texts]

    def encode_document(self, text):
        return self._vec(text, 16)

    def encode_query(self, text):
        return self._vec(text, 4)


def _engine(tmp_path) -> RAGEngine:
    eng = RAGEngine(
        workspace=str(tmp_path),
        knowledge_base=str(tmp_path / "kb"),
        reading_mode=True,
        reading_quant_bits=4,
        reading_coarse_k=10,
        reading_top_n=3,
        reading_chunk_max=500,
        reading_chunk_min=300,
    )
    store = cs.ColBertDocStore(
        db_dir=str(tmp_path / ".agent" / "colbert_docs.lancedb"),
        runtime=FakeRuntime(), dim=8)
    eng._reading_store = store
    # 语义切片编码器用测试 Fake 注入，避免加载真实 model2vec。
    eng._reading_chunk_encoder = store.runtime
    return eng


def test_reading_config_roundtrip_and_clamp(tmp_path):
    cfg = AgentConfig()
    assert cfg.rag_reading_coarse_k == 50          # 粗筛候选默认 50
    assert abs(cfg.rag_reading_min_score - 0.61) < 1e-9
    cfg.rag_reading_mode_enabled = True
    cfg.rag_reading_quant_bits = 2
    cfg.rag_reading_coarse_k = 100
    cfg.rag_reading_top_n = 5
    cfg.rag_reading_min_score = 0.72
    cfg.rag_reading_use_ann = True
    cfg.rag_reading_model_dir = "/models/colbert"
    cfg.rag_reading_chunk_max = 9000
    cfg.rag_reading_chunk_min = 8500
    cfg.rag_reading_semantic_threshold = 0.42
    path = str(tmp_path / "config.json")
    cfg.save(path)
    loaded = AgentConfig.load(path)
    assert loaded.rag_reading_mode_enabled is True
    assert loaded.rag_reading_quant_bits == 2
    assert loaded.rag_reading_coarse_k == 100
    assert loaded.rag_reading_top_n == 5
    assert abs(loaded.rag_reading_min_score - 0.72) < 1e-9
    assert loaded.rag_reading_use_ann is True
    assert loaded.rag_reading_model_dir == "/models/colbert"
    assert loaded.rag_reading_chunk_max == 9000
    assert loaded.rag_reading_chunk_min == 8500
    assert abs(loaded.rag_reading_semantic_threshold - 0.42) < 1e-9


def test_reading_min_score_and_coarse_clamped():
    """手改 config.json 越界值必须回退到安全区间。"""
    cfg = AgentConfig()
    cfg.rag_reading_min_score = 1.5
    cfg.rag_reading_coarse_k = 999
    cfg._migrate()
    assert abs(cfg.rag_reading_min_score - 0.95) < 1e-9
    assert cfg.rag_reading_coarse_k == 50
    cfg.rag_reading_min_score = -1
    cfg._migrate()
    assert cfg.rag_reading_min_score == 0.0


def test_reading_chunks_add_and_search_returns_source_window(tmp_path):
    eng = _engine(tmp_path)
    text = (
        "# 认证设计\n\n## OAuth2 授权码\n\n"
        + ("客户端先用 authorization code 换 access_token，"
           "兑换时必须出示 client_secret。" * 20)
        + "\n\n## PKCE\n\n"
        + ("公共客户端生成 code_verifier，授权时提交 challenge，"
           "兑换时提交 verifier。" * 20)
    )
    chunks = eng._reading_chunks(text, "/docs/auth.md", "auth.md")
    assert chunks and all(c["source"] == "/docs/auth.md" for c in chunks)
    added = eng._reading_add_streaming(iter(chunks))
    assert added == len(chunks) == eng._get_reading_store().stats()["documents"]

    hits = eng._reading_search("公共客户端怎么防止授权码被截获？", top_k=3)
    assert hits
    assert hits[0]["source"] == "/docs/auth.md"
    assert isinstance(hits[0]["chunk_index"], int)
    assert hits[0]["reading_mode"] is True

    window = eng.get_chunk_window("/docs/auth.md", hits[0]["chunk_index"])
    assert window["current"]["text"]
    assert set(window) == {"current", "prev", "next"}

    status = eng.status()
    assert status["reading_mode"] is True
    assert status["chunks_stored"] == len(chunks)
    assert status["sources"] == ["/docs/auth.md"]


def test_reading_search_applies_min_score(tmp_path):
    """相关度门限生效时返回空，交给工具层给「未召回」提示。"""
    eng = _engine(tmp_path)
    text = ("# 认证\n\n" + ("客户端先用 authorization code 换 access_token。" * 20))
    eng._reading_add_streaming(iter(eng._reading_chunks(
        text, "/docs/auth.md", "auth.md")))
    assert eng._reading_search("认证流程", top_k=3)
    eng._reading_min_score = 2.0        # 高于任何可能的 MaxSim/token
    assert eng._reading_search("认证流程", top_k=3) == []


def test_reading_cleanup_deletes_colbert_chunks(tmp_path):
    """删除知识库文件时，普通 store 与 ColBERT 三表都必须清理。"""
    eng = object.__new__(RAGEngine)
    eng._reading_mode = True
    eng._knowledge_base = str(tmp_path)
    eng._rag_dir = str(tmp_path / "rag")
    eng._markdown_dir = str(tmp_path / "md")
    eng._cache_md_path = lambda source: str(tmp_path / "missing.md")

    class NormalStore:
        def __init__(self):
            self.deleted = []

        def delete_by_source(self, source):
            self.deleted.append(source)

        def delete_document(self, name):
            pass

        def delete_document_by_md_path(self, path):
            pass

    class ReadingStore:
        def __init__(self):
            self.deleted = []

        def delete(self, source="", doc_id="", title=""):
            self.deleted.append(source)
            return 1

    normal = NormalStore()
    reading = ReadingStore()
    eng._get_store = lambda: normal
    eng._get_reading_store = lambda: reading

    source = str(tmp_path / "old.md")
    old = {source: {"mtime": 1, "size": 10, "chunk_count": 3}}
    new = dict(old)
    stats = {"files_deleted": 0}
    deleted = eng._cleanup_deleted_files(
        [], stats, threading.Lock(), old, new)
    assert deleted == {source}
    assert normal.deleted == [source]
    assert reading.deleted == [source]
    assert stats["files_deleted"] == 1


def test_rag_search_tool_reading_mode_injects_top_chunk_only():
    class FakeEngine:
        def search(self, query, top_k=5, source=None):
            return [
                {"source": "/docs/a.md", "chunk_index": 0,
                 "text": "第一片正文" * 100, "score": 9.0},
                {"source": "/docs/a.md", "chunk_index": 1,
                 "text": "第二片正文" * 100, "score": 8.0},
                {"source": "/docs/a.md", "chunk_index": 2,
                 "text": "第三片正文" * 100, "score": 7.0},
            ]

        def search_formatted(self, *args, **kwargs):
            return "should not be used"

    cfg = AgentConfig()
    cfg.rag_reading_mode_enabled = True
    cfg.rag_reading_top_n = 3
    tool = RagSearchTool(FakeEngine())
    tool.bind(cfg, None)
    result = tool.run(query="认证流程")
    assert result.success
    assert len(result.attachments) == 3
    assert result.attachments[0]["used"] is True
    assert all(not a["used"] for a in result.attachments[1:])
    assert "已注入上下文" in result.output
    assert "第一片正文" in result.output       # 最高分切片给模型总结
    assert "第二片正文" not in result.output   # 其余只出气泡
    assert "第三片正文" not in result.output


def test_rag_search_tool_follows_engine_reading_mode_over_config():
    """config 与引擎短暂不一致时，按引擎实际管线分流，避免长切片漏进上下文。"""
    class FakeEngine:
        reading_mode = True

        def search(self, query, top_k=5, source=None):
            return [{"source": "/docs/a.md", "chunk_index": 0,
                     "text": "长切片正文" * 100, "score": 9.0}]

        def search_formatted(self, *args, **kwargs):
            return "should not be used"

    cfg = AgentConfig()
    cfg.rag_reading_mode_enabled = False  # 与引擎不一致
    tool = RagSearchTool(FakeEngine())
    tool.bind(cfg, None)
    result = tool.run(query="认证流程")
    assert result.success
    assert len(result.attachments) == 1
    assert result.attachments[0]["used"] is True
    assert "长切片正文" in result.output


def test_doc_search_tool_reading_mode_bubbles_and_blocks_mutation(monkeypatch):
    """doc_search_tool 与知识库共用三表：阅读模式下只出气泡，禁止绕过同步改索引。"""
    from agent.tools import doc_search as ds

    class FakeStore:
        def search(self, query, top_k=5, coarse_k=20, include_content=True,
                   max_content_chars=1500, use_ann=False, min_score=0.0):
            return [{"doc_id": "abc:00000", "title": "手册 > 认证",
                     "source": "/docs/a.md", "chunk_index": 0,
                     "content": "长切片正文" * 100, "score": 9.5,
                     "token_count": 12}]

        def index_text(self, *args, **kwargs):
            raise AssertionError("reading mode must not index")

        def delete(self, *args, **kwargs):
            raise AssertionError("reading mode must not delete")

    store = FakeStore()
    monkeypatch.setattr(ds, "_get_store", lambda config: store)
    cfg = AgentConfig()
    cfg.rag_reading_mode_enabled = True
    tool = ds.DocSearchTool()
    tool.bind(cfg, None)

    r = tool.run(op="search", query="怎么认证")
    assert r.success
    assert len(r.attachments) == 1 and r.attachments[0]["used"] is True
    assert "长切片正文" in r.output            # 最高分切片进上下文
    assert "已注入上下文" in r.output

    r = tool.run(op="index", path="/docs/a.md")
    assert not r.success and "rag_tools(op=\"ingest\")" in r.error
    r = tool.run(op="delete", doc_id="abc:00000")
    assert not r.success
