"""RAG 全流程验收（2026-11-06）：工具鲁棒性、检索一致性、生命周期与计数。

用轻量 stub 隔离真实模型/LanceDB，锁住本轮修复的行为，避免回归。
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.agent import Agent  # noqa: E402
from agent.config import AgentConfig  # noqa: E402
from agent.kernel import ContextBudget  # noqa: E402
from agent.kernel.loop import KernelLoop  # noqa: E402
from agent.llm_client import StreamEvent  # noqa: E402
from agent.rag.engine import RAGEngine  # noqa: E402
from agent.tools.rag_tool import (  # noqa: E402
    RagIngestTool, RagOutlineTool, RagSearchTool, RagStatusTool, _as_text,
)
from agent.tools.base import ToolRegistry  # noqa: E402


@pytest.fixture()
def cfg(tmp_path):
    c = AgentConfig()
    c.workspace = str(tmp_path)
    c.knowledge_base = str(tmp_path / "kb")
    return c


class _Noop:
    def is_cancelled(self): return False
    def on_tool_start(self, n, a): pass
    def on_tool_end(self, n, r): pass
    def on_log(self, l): pass
    def on_error(self, m): pass
    def confirm(self, m): return True
    def ask_user(self, p): return None


class _Engine:
    def __init__(self):
        self.calls = []
        self.digest = "# 目录\n第一章"
        self.summary = ""

    def search(self, query, top_k=5, source=None):
        self.calls.append(("search", query, top_k, source))
        return [{"text": "片段", "source": "a.md", "score": 0.9}]

    def search_formatted(self, query, top_k=5, source=None):
        return "从知识库检索到 1 条"

    def list_documents(self):
        return ["a.md", "b.md"]

    def get_document_outline(self, name):
        return {"doc_name": "a.md", "digest": self.digest}

    def get_book_summary(self, name):
        return self.summary

    def status(self):
        return {"knowledge_base": "/kb", "rag_dir": "/rag",
                "chunks_stored": 1, "sources": ["a.md"]}

    def close(self):
        self.calls.append(("close",))

    def ingest(self, force=False, progress_callback=None, log_callback=None):
        self.calls.append(("ingest", force))
        return {"files_found": 1, "files_extracted": 1, "files_skipped": 0,
                "total_chars": 100, "chunks": 2,
                "errors": ["err-1", "err-2", "err-3", "err-4"]}


# ---------------------------------------------------------------------------
# 1. 参数类型归一化（本地小模型乱传参数不应抛异常 / 不应原样重试）
# ---------------------------------------------------------------------------

def test_as_text_normalizes_scalar_and_rejects_containers():
    assert _as_text(None) == ""
    assert _as_text("  abc  ") == "abc"
    assert _as_text(123) == "123"
    assert _as_text(True) == ""
    assert _as_text(["a"]) is None
    assert _as_text({"a": 1}) is None


def test_rag_search_never_crashes_on_bad_argument_types():
    eng = _Engine()
    tool = RagSearchTool(eng)
    tool.bind(SimpleNamespace(rag_search_top_k="not-an-int"), None)  # 非法配置

    assert tool.run(query=123, top_k="3").success          # 数字查询词 + 字符串 top_k
    assert tool.run(query="abc", source=[]).success        # 非法 source → 不限定
    bad = tool.run(query=["abc"])
    assert not bad.success and "合法参数" in bad.error
    assert "不要重复调用" in tool.run(query=None).error     # 空参仍带反循环指引
    assert tool.run(query="abc", top_k=3).success
    assert eng.calls[0][2] == 3


def test_outline_rejects_non_string_book_name():
    assert not RagOutlineTool(_Engine()).run(book_name=["a"]).success


# ---------------------------------------------------------------------------
# 2. outline 收敛指引无条件 + 带参输出体积兜底
# ---------------------------------------------------------------------------

def test_outline_guidance_present_even_without_book_summary():
    eng = _Engine()
    res = RagOutlineTool(eng).run(book_name="a")
    assert res.success
    assert "至多 3 次" in res.output and "无需再 rag_search" in res.output

    eng.summary = "全书讲了 X"
    res2 = RagOutlineTool(eng).run(book_name="a")
    assert "【全书摘要】" in res2.output and "至多 3 次" in res2.output


def test_outline_long_digest_is_capped():
    eng = _Engine()
    eng.digest = "章" * 250_000
    res = RagOutlineTool(eng).run(book_name="a")
    assert res.success and len(res.output) < 210_000
    assert "已截断" in res.output


# ---------------------------------------------------------------------------
# 3. ingest 成功后失效状态缓存 + 错误详情可见
# ---------------------------------------------------------------------------

def test_ingest_notifies_data_changed_and_shows_errors():
    eng = _Engine()
    notified = []
    reg = SimpleNamespace(callbacks=_Noop(),
                          on_rag_data_changed=lambda: notified.append(1))
    tool = RagIngestTool(eng, reg)
    res = tool.run(force="false")   # 字符串 false 不得触发全量重建
    assert res.success and eng.calls == [("ingest", False)]
    assert notified == [1]
    assert "err-1" in res.output and "err-3" in res.output
    assert "err-4" not in res.output and "其余 1 个省略" in res.output


# ---------------------------------------------------------------------------
# 4. shutdown 后工具不得静默复活已释放引擎
# ---------------------------------------------------------------------------

def test_shutdown_clears_rag_tool_engines(cfg):
    reg = ToolRegistry(config=cfg, callbacks=_Noop())
    fake = _Engine()
    reg._rag_engine = fake
    tools = {}
    for name, tool in (("rag_search", RagSearchTool(fake)),
                       ("rag_outline", RagOutlineTool(fake)),
                       ("rag_status", RagStatusTool(fake)),
                       ("rag_ingest", RagIngestTool(fake))):
        tools[name] = tool
        reg._tools[name] = tool
    reg.shutdown()
    # shutdown 会清空 _tools，用保存的引用验证引擎句柄已被摘掉
    for tool in tools.values():
        assert tool._engine is None
    assert ("close",) in fake.calls
    res = tools["rag_search"].run(query="abc")
    assert not res.success and "不要重试" in res.error


# ---------------------------------------------------------------------------
# 5. 格式化输出截断不越界
# ---------------------------------------------------------------------------

def test_search_formatted_never_exceeds_total_cap(cfg, monkeypatch):
    eng = RAGEngine(workspace=cfg.workspace, knowledge_base=cfg.knowledge_base)
    chunks = [{"text": "内容" * 2000, "source": f"f{i}.md", "score": 0.9}
              for i in range(10)]
    monkeypatch.setattr(eng, "search", lambda *a, **k: chunks)
    out = eng.search_formatted("查询", top_k=10, max_total_chars=8000)
    assert len(out) <= 8000 + 40, f"格式化输出 {len(out)} 字超过上限"


# ---------------------------------------------------------------------------
# 6. 全书摘要匹配不得串书
# ---------------------------------------------------------------------------

def test_book_summary_short_keyword_does_not_crossmatch(cfg):
    eng = RAGEngine(workspace=cfg.workspace, knowledge_base=cfg.knowledge_base)
    state = {"books": {
        "薄伽梵歌.epub": {"book_summary": "A 书摘要"},
        "神之所欲.epub": {"book_summary": "B 书摘要"},
        "导论集.epub": {"book_summary": "C 书摘要"},
    }}
    path = os.path.join(cfg.workspace, "rag", "chapter_summaries.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)

    assert eng.get_book_summary("薄伽梵歌.epub") == "A 书摘要"
    # 短词「导论」旧实现会双向子串命中「导论集」，甚至命中任意短名
    assert eng.get_book_summary("导论") == ""
    assert eng.get_book_summary("集") == ""
    # 长词唯一命中仍可用
    assert eng.get_book_summary("薄伽梵歌.epub") == "A 书摘要"


# ---------------------------------------------------------------------------
# 7. rag_search 硬上限跨自动续跑段
# ---------------------------------------------------------------------------

class _LLM:
    def __init__(self):
        self.n = 0

    def chat_stream(self, messages, tools=None, temperature=0.7, **kw):
        self.n += 1
        return iter([StreamEvent(type="content", content="答复"),
                     StreamEvent(type="done", usage={})])


def _kernel(cfg):
    return KernelLoop(
        llm=_LLM(), dispatcher=SimpleNamespace(dispatch=lambda calls: [],
                                               schemas=lambda: []),
        callbacks=_Noop(), config=cfg,
        system_prompt=lambda: "sys", volatile_context=lambda: "",
        session=None, max_turns=3,
    )


def test_rag_search_budget_survives_resume_segments(cfg):
    k = _kernel(cfg)
    k._rag_search_calls = 2
    snap = k.failure_snapshot()
    assert snap["rag_search_calls"] == 2
    k.seed_failures(snap)
    k.run("继续", resume=True)
    assert k._rag_search_calls == 2, "续跑段不得把检索上限清零"


# ---------------------------------------------------------------------------
# 8. 关闭精排仍走混合融合；阈值显式下传
# ---------------------------------------------------------------------------

def test_rerank_disabled_still_uses_hybrid_pipeline(cfg):
    eng = RAGEngine(workspace=cfg.workspace, knowledge_base=cfg.knowledge_base,
                    rerank_enabled=False, min_similarity=0.3)
    store = SimpleNamespace()
    seen = {}

    def _hybrid(query, top_k=5, where=None, min_similarity=None,
                use_reranker=True):
        seen.update(min_similarity=min_similarity, use_reranker=use_reranker)
        return [{"text": "x", "source": "a.md", "score": 0.9}]

    store.count = lambda: 1
    store.search_with_rerank = _hybrid
    store.search = lambda *a, **k: pytest.fail("关闭 rerank 不应退化成纯向量")
    eng._store = store
    assert eng.search("查询", top_k=2)
    assert seen["use_reranker"] is False and seen["min_similarity"] == 0.3


# ---------------------------------------------------------------------------
# 9. 查询缓存键区分阈值（改配置后不得继续命中旧排序）
# ---------------------------------------------------------------------------

def test_query_cache_key_isolates_min_similarity(cfg):
    from agent.rag.vector_store import VectorStore

    # 构造是轻量的（LanceDB 连接/模型都在首次使用时才初始化）
    store = VectorStore(persist_dir=os.path.join(cfg.workspace, "rag", "vectors"))
    store._table = SimpleNamespace(version=1)
    store._query_cache_put("hybrid|s0.500|r1", "q", 5, None, [{"text": "x"}])
    assert store._query_cache_get("hybrid|s0.500|r1", "q", 5, None)
    assert store._query_cache_get("hybrid|s0.300|r1", "q", 5, None) is None
    assert store._query_cache_get("hybrid|s0.500|r0", "q", 5, None) is None


# ---------------------------------------------------------------------------
# 10. close() 等待在途 ingest 退出
# ---------------------------------------------------------------------------

def test_close_waits_for_inflight_ingest(cfg):
    eng = RAGEngine(workspace=cfg.workspace, knowledge_base=cfg.knowledge_base)
    eng._store = None
    eng._ingest_lock.acquire()
    released = []

    def _hold():
        time.sleep(0.4)
        eng._ingest_lock.release()
        released.append(1)

    t = threading.Thread(target=_hold)
    t.start()
    t0 = time.monotonic()
    eng.close()
    elapsed = time.monotonic() - t0
    t.join()
    assert released and elapsed >= 0.3, "close 必须等 in-flight ingest 退出"
