"""doc_search_tool 注册、op 分发与输出格式测试（不加载真实模型）。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.config import AgentConfig  # noqa: E402
from agent.tools import doc_search as ds  # noqa: E402


class FakeStore:
    def __init__(self):
        self.deleted = []

    def index_text(self, title, text, source="", overwrite=True, quant_bits=4):
        return 2

    def index_path(self, path, overwrite=True, quant_bits=4):
        return {"files": 1, "chunks": 2}

    def search(self, query, top_k=5, coarse_k=20, include_content=True,
               max_content_chars=1500):
        return [{
            "doc_id": "abc:00000",
            "title": "手册 > 认证",
            "content": "支持 OAuth2 认证。",
            "score": 9.5,
            "rrf_score": 0.03,
            "token_count": 12,
        }]

    def list_documents(self, limit=50):
        return [{"doc_id": "abc:00000", "title": "手册 > 认证"}]

    def delete(self, doc_id="", source="", title=""):
        self.deleted.append((doc_id, source, title))
        return 2

    def stats(self):
        return {"documents": 2, "vectors": 2, "residuals": 2}


def _tool(monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(ds, "_get_store", lambda config: store)
    cfg = AgentConfig()
    cfg.workspace = os.path.expanduser("~/base-agent-workspace")
    cfg.knowledge_base = os.path.join(cfg.workspace, "kb")
    tool = ds.DocSearchTool()
    tool.bind(cfg, None)
    return tool, store


def test_doc_search_tool_registered_and_lite_available():
    from agent.agent import SYSTEM_PROMPT
    from agent.kernel.loop import _LITE_TOOL_NAMES, _NEEDS_FULL_TOOLS_RE
    from agent.tools import ToolRegistry

    class CB:
        def is_cancelled(self): return False
        def on_content(self, t): pass
        def on_reasoning(self, t): pass
        def on_tool_start(self, n, a): pass
        def on_tool_end(self, n, r): pass
        def on_log(self, l): pass
        def on_usage(self, u): pass
        def on_token_speed(self, t, s): pass
        def on_finished(self): pass
        def confirm(self, m): return True
        def ask_user(self, p): return "ok"

    cfg = AgentConfig()
    cfg.workspace = os.path.expanduser("~/base-agent-workspace")
    cfg.knowledge_base = os.path.join(cfg.workspace, "kb")
    reg = ToolRegistry(config=cfg, callbacks=CB())
    assert "doc_search_tool" in reg.names()
    assert "doc_search" not in reg.names(), "兼容别名不进 schema"
    assert reg.get("doc_search") is not None
    assert "doc_search_tool" in _LITE_TOOL_NAMES
    assert _NEEDS_FULL_TOOLS_RE.search("帮我检索这本长文档")
    assert "doc_search_tool" in SYSTEM_PROMPT


def test_doc_search_tool_index_search_list_delete_stats(monkeypatch):
    tool, store = _tool(monkeypatch)

    r = tool.run(op="index", title="手册", text="正文")
    assert r.success and "Indexed 2 chunk" in r.output

    r = tool.run(op="search", query="怎么认证", top_k=3)
    assert r.success
    assert "OAuth2" in r.output
    assert "abc:00000" in r.output and "score=" in r.output

    r = tool.run(op="list")
    assert r.success and "手册 > 认证" in r.output

    r = tool.run(op="stats")
    assert r.success and "documents" in r.output

    r = tool.run(op="delete", doc_id="abc:00000")
    assert r.success and store.deleted[-1][0] == "abc:00000"

    r = tool.run(op="delete", path="/docs/a.md")
    assert r.success and store.deleted[-1][1] == "/docs/a.md"


def test_doc_search_tool_input_validation_and_confirmation(monkeypatch):
    tool, _ = _tool(monkeypatch)
    r = tool.run(op="index")
    assert not r.success
    r = tool.run(op="search", query="")
    assert not r.success
    r = tool.run(op="bogus")
    assert not r.success
    assert tool.should_confirm({"op": "delete"}) is True
    assert tool.should_confirm({"op": "search"}) is False
