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
        # 阅读模式由检索模式决定：极速检索 = 阅读模式。
        rag_mode="fast",
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
    eng._get_reading_store = lambda force=False: reading

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


def test_rag_search_tool_reading_mode_injects_all_chunks():
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
    cfg.rag_mode = "fast"
    cfg.rag_reading_mode_enabled = True
    cfg.rag_reading_top_n = 3
    tool = RagSearchTool(FakeEngine())
    tool.bind(cfg, None)
    result = tool.run(query="认证流程")
    assert result.success
    assert len(result.attachments) == 3
    assert all(a["used"] for a in result.attachments)
    assert "已注入上下文" in result.output
    # 所有切片都抽取核心句后进上下文，不再只给 top1
    assert "第一片正文" in result.output
    assert "第二片正文" in result.output
    assert "第三片正文" in result.output


def test_rag_search_tool_reading_mode_compacts_all_chunks():
    """长切片也必须全部压缩后注入，不能只给 top1。"""
    filler = "".join(
        f"第{i}句是普通背景说明，不涉及关键结论。" for i in range(80))
    chunks = [
        {"source": "/docs/a.md", "chunk_index": 0, "score": 9.0,
         "text": filler + "关键结论：一线城市住宿上限 500 元。"},
        {"source": "/docs/b.md", "chunk_index": 0, "score": 8.0,
         "text": filler + "关键结论：发票需在 7 个工作日内提交。"},
        {"source": "/docs/c.md", "chunk_index": 0, "score": 7.0,
         "text": filler + "关键结论：设备补贴每月 200 元。"},
    ]

    class FakeEngine:
        def search(self, query, top_k=5, source=None):
            return chunks

        def search_formatted(self, *args, **kwargs):
            return "should not be used"

    cfg = AgentConfig()
    cfg.rag_mode = "fast"
    cfg.rag_reading_top_n = 3
    tool = RagSearchTool(FakeEngine())
    tool.bind(cfg, None)
    res = tool.run(query="制度总结")

    assert res.success
    assert len(res.attachments) == 3
    assert all(a["used"] for a in res.attachments)
    for fact in ("500 元", "7 个工作日", "200 元"):
        assert fact in res.output
    assert len(res.output) < sum(len(c["text"]) for c in chunks), \
        "长切片应先压缩再注入"


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
    """doc_search_tool 与知识库共用三表：阅读模式下所有切片压缩注入，禁止改索引。"""
    from agent.tools import doc_search as ds

    filler = "".join(f"第{i}句是普通背景说明。" for i in range(60))

    class FakeStore:
        def search(self, query, top_k=5, coarse_k=20, include_content=True,
                   max_content_chars=1500, use_ann=False, min_score=0.0):
            return [
                {"doc_id": "abc:00000", "title": "手册 > 认证",
                 "source": "/docs/a.md", "chunk_index": 0,
                 "content": filler + "认证需提前 1 个工作日申请。",
                 "score": 9.5, "token_count": 12},
                {"doc_id": "abc:00001", "title": "手册 > 令牌",
                 "source": "/docs/b.md", "chunk_index": 0,
                 "content": filler + "令牌有效期为 7 天。",
                 "score": 8.5, "token_count": 12},
            ]

        def index_text(self, *args, **kwargs):
            raise AssertionError("reading mode must not index")

        def delete(self, *args, **kwargs):
            raise AssertionError("reading mode must not delete")

    store = FakeStore()
    monkeypatch.setattr(ds, "_get_store", lambda config: store)
    cfg = AgentConfig()
    cfg.rag_mode = "fast"
    cfg.rag_reading_mode_enabled = True
    tool = ds.DocSearchTool()
    tool.bind(cfg, None)

    r = tool.run(op="search", query="怎么认证")
    assert r.success
    assert len(r.attachments) == 2
    assert all(a["used"] for a in r.attachments)
    assert "1 个工作日" in r.output       # 所有切片的核心句都注入
    assert "7 天" in r.output
    assert "已注入上下文" in r.output

    r = tool.run(op="index", path="/docs/a.md")
    assert not r.success and "rag_tools(op=\"ingest\")" in r.error
    r = tool.run(op="delete", doc_id="abc:00000")
    assert not r.success


class _FakeNormalDocuments:
    """VectorStore documents 表的最小 Fake（digest/outline/list 语义）。"""

    def __init__(self):
        self.upserts = []

    def upsert_document(self, doc_name, digest, markdown="",
                        chapters="", md_path=""):
        self.upserts.append({"doc_name": doc_name, "digest": digest,
                             "chapters": chapters, "md_path": md_path})

    def list_documents(self):
        return sorted({u["doc_name"] for u in self.upserts})

    def get_document_digest(self, name):
        for u in self.upserts:
            if name and (name in u["doc_name"] or u["doc_name"] in name):
                return {"doc_name": u["doc_name"], "digest": u["digest"],
                        "chapters": u["chapters"], "md_path": u["md_path"]}
        return None


def test_build_and_store_digest_routes_by_mode(tmp_path):
    """digest 写入按模式分流：阅读模式写 ColBERT doc_meta，不碰普通表。"""
    eng = _engine(tmp_path)
    normal = _FakeNormalDocuments()
    eng._get_store = lambda: normal
    eng.build_and_store_digest(
        "docs/auth.md", "# 认证设计\n\n## OAuth2\n\n" + "正文" * 50,
        md_path="/cache/auth.md", source="/docs/auth.md")
    assert normal.upserts == [], "阅读模式不得写普通 documents 表"
    meta = eng._get_reading_store().get_doc_meta("docs/auth.md")
    assert meta is not None and meta["md_path"] == "/cache/auth.md"
    assert meta["digest"].startswith("# 目录")
    # 读取接口分流：outline/list 只反映本模式的文档
    assert eng.list_documents() == ["docs/auth.md"]
    assert eng.get_document_outline("auth")["doc_name"] == "docs/auth.md"
    # 全书摘要门控：文档不在库时直接拦截（未生成摘要时返回空串）
    assert eng.get_book_summary("不存在") == ""
    assert eng.get_book_summary("auth") == ""


def test_build_and_store_digest_normal_mode_writes_documents_table(tmp_path):
    """普通模式 digest 写普通 documents 表，不触碰 ColBERT。"""
    eng = object.__new__(RAGEngine)
    eng._reading_mode = False
    normal = _FakeNormalDocuments()
    eng._get_store = lambda: normal
    eng._get_reading_store = lambda force=False: (_ for _ in ()).throw(
        AssertionError("普通模式不应触碰 ColBERT"))
    eng.build_and_store_digest("docs/b.md", "# 标题\n\n## 章\n\n正文")
    assert len(normal.upserts) == 1
    assert normal.upserts[0]["doc_name"] == "docs/b.md"
    assert eng.list_documents() == ["docs/b.md"]
    assert eng.get_document_outline("b")["doc_name"] == "docs/b.md"


def test_migrate_stale_mode_data_clears_other_store(tmp_path):
    """模式切换迁移：重索引前无条件清掉旧模式 store 的向量。"""

    class NormalStore:
        def __init__(self):
            self.deleted = []

        def delete_by_source(self, s):
            self.deleted.append(s)

    class ReadingStore:
        def __init__(self):
            self.deleted = []

        def delete(self, source="", doc_id="", title=""):
            self.deleted.append(source or doc_id or title)
            return 1

    eng = object.__new__(RAGEngine)
    normal = NormalStore()
    reading = ReadingStore()
    eng._get_store = lambda: normal
    eng._reading_store = reading

    # 普通模式重跑「原为阅读模式」的文件 → 清 ColBERT
    eng._reading_mode = False
    eng._migrate_stale_mode_data("/docs/a.md", True)
    assert reading.deleted == ["/docs/a.md"]
    assert normal.deleted == []
    # 阅读模式重跑「原为普通模式」的文件 → 清普通库
    eng._reading_mode = True
    eng._migrate_stale_mode_data("/docs/a.md", False)
    assert normal.deleted == ["/docs/a.md"]
    # 同模式 / 无历史条目：不动
    eng._migrate_stale_mode_data("/docs/a.md", True)
    eng._migrate_stale_mode_data("/docs/a.md", None)
    assert normal.deleted == ["/docs/a.md"]
    assert reading.deleted == ["/docs/a.md"]


def test_delete_document_routes_by_mode_and_dual_deletes(tmp_path):
    """知识遗忘按模式分流，另一模式库的残留（向量+元数据）一并删除。"""

    class NormalStore:
        def __init__(self):
            self.deleted_meta = []
            self.deleted_src = []

        def get_document_digest(self, name):
            return {"doc_name": name, "digest": "", "chapters": "",
                    "md_path": ""}

        def iter_all_chunks(self):
            return []

        def delete_document(self, name):
            self.deleted_meta.append(name)

        def delete_by_source(self, name):
            self.deleted_src.append(name)

    class ReadingStore:
        def __init__(self):
            self.deleted = []
            self.meta_deleted = []

        def get_doc_meta(self, name):
            return {"doc_name": name, "digest": "", "chapters": "",
                    "md_path": ""}

        def delete_matching(self, keyword):
            self.deleted.append(keyword)
            return 1

        def delete_doc_meta(self, name):
            self.meta_deleted.append(name)

    normal = NormalStore()
    reading = ReadingStore()

    # 阅读模式：只走 ColBERT（delete_matching 连带清 doc_meta）
    eng = object.__new__(RAGEngine)
    eng._reading_mode = True
    eng._reading_store = reading
    eng._get_reading_store = lambda force=False: reading
    eng._get_store = lambda: (_ for _ in ()).throw(
        AssertionError("阅读模式删除不应触碰普通库"))
    assert eng.delete_document("docs/a.md") is True
    assert reading.deleted == ["docs/a.md"]

    # 普通模式：删普通库 + 无条件双删 ColBERT 残留
    eng2 = object.__new__(RAGEngine)
    eng2._reading_mode = False
    eng2._reading_store = reading
    eng2._get_store = lambda: normal
    eng2._get_reading_store = lambda force=False: reading
    assert eng2.delete_document("docs/a.md") is True
    assert normal.deleted_meta == ["docs/a.md"]
    assert normal.deleted_src == ["docs/a.md"]
    assert reading.deleted == ["docs/a.md", "docs/a.md"]


def test_get_reading_store_force_semantics(tmp_path):
    """_get_reading_store：普通模式默认 None，force=True 可取 store 用于迁移清理。"""
    eng = object.__new__(RAGEngine)
    eng._reading_mode = False
    eng._reading_store = None
    eng._reading_store_lock = threading.Lock()
    eng._reading_model_dir = ""
    eng._workspace = str(tmp_path)
    assert eng._get_reading_store() is None
    store = eng._get_reading_store(force=True)
    assert store is not None
    assert eng._get_reading_store() is None      # 不污染默认路径
    assert eng._get_reading_store(force=True) is store
