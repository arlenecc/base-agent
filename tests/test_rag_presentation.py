"""检索即展示（Retrieval-as-Presentation）的旁路数据与切片窗口测试。

覆盖：
* ToolResult.attachments 只在 UI 侧流转，绝不进模型消息；
* 切片附件去重 / used 标记；
* RAGEngine.get_chunk_window 的前后翻页取数（含首尾边界与引擎不可用）。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from agent.rag.engine import RAGEngine
from agent.tools.base import ToolResult
from agent.tools.rag_tool import _chunk_attachments


def test_attachments_never_enter_model_message():
    tr = ToolResult(
        success=True, output="注入正文",
        attachments=[{"source": "/kb/a.md", "chunk_index": 0, "text": "原文"}],
    )
    assert tr.to_message() == "注入正文"
    # 失败结果同理：附件不参与错误回执
    tr2 = ToolResult(False, error="boom",
                     attachments=[{"source": "/kb/a.md", "chunk_index": 0}])
    assert tr2.to_message() == "ERROR: boom"


def test_chunk_attachments_dedup_and_used_flag():
    items = [
        {"source": "/kb/a.md", "chunk_index": 0, "score": 0.9, "text": "A0"},
        {"source": "/kb/a.md", "chunk_index": 0, "score": 0.8, "text": "A0-dup"},
        {"source": "/kb/a.md", "chunk_index": 4, "score": 0.7, "text": "A4"},
    ]
    out = _chunk_attachments(items)
    assert [a["chunk_index"] for a in out] == [0, 4]
    assert out[0]["used"] is True
    assert out[1]["used"] is False
    assert out[0]["text"] == "A0"        # 保留先到（排名更高）的文本


class _FakeStore:
    def __init__(self, n=3):
        self.n = n
        self.calls = []

    def get_chunks_by_source_indices(self, source, indices):
        self.calls.append((source, list(indices)))
        return {
            i: {"id": f"s{i}", "text": f"t{i}", "source": source,
                "chunk_index": i}
            for i in indices if 0 <= i < self.n
        }


class _FakeEngine:
    def __init__(self, store=None):
        if store is not None:
            self._store = store


def test_chunk_window_returns_prev_and_next():
    eng = _FakeEngine(_FakeStore(3))
    win = RAGEngine.get_chunk_window(eng, "/kb/a.md", 1)
    assert win["current"]["chunk_index"] == 1
    assert win["prev"]["chunk_index"] == 0
    assert win["next"]["chunk_index"] == 2


def test_chunk_window_boundaries():
    eng = _FakeEngine(_FakeStore(3))
    first = RAGEngine.get_chunk_window(eng, "/kb/a.md", 0)
    assert first["prev"] is None and first["next"]["chunk_index"] == 1
    last = RAGEngine.get_chunk_window(eng, "/kb/a.md", 2)
    assert last["next"] is None and last["prev"]["chunk_index"] == 1


def test_chunk_window_unavailable_returns_empty():
    # 引擎已释放（无 store）/ 非法序号 / 切片不存在 → 空 dict（UI 回退原文）
    assert RAGEngine.get_chunk_window(_FakeEngine(), "/kb/a.md", 1) == {}
    assert RAGEngine.get_chunk_window(_FakeEngine(_FakeStore()), "/kb/a.md",
                                      "x") == {}
    assert RAGEngine.get_chunk_window(_FakeEngine(_FakeStore()), "/kb/a.md",
                                      9) == {}


def test_vector_store_source_indices_roundtrip(tmp_path):
    """真实 LanceDB：按 (source, chunk_index) 精确取片，且不串文档。"""
    from agent.rag.vector_store import VectorStore

    class _EF:
        def __call__(self, input):
            return [[0.25] * 128 for _ in input]

        def embed_query(self, text):
            return [0.25] * 128

        def name(self):
            return "fake-ef"

    store = VectorStore(persist_dir=str(tmp_path / "vectors"),
                        embedding_function=_EF())
    try:
        store.add([{"source": "/kb/a.md", "chunk_index": i, "text": f"t{i}"}
                   for i in range(3)])
        store.add([{"source": "/kb/b.md", "chunk_index": 0, "text": "other"}])
        rows = store.get_chunks_by_source_indices("/kb/a.md", [0, 1, 4, -1])
        assert set(rows) == {0, 1}
        assert rows[1]["text"] == "t1"
        assert rows[1]["id"]                      # 供气泡/溯源使用
        assert all(r["source"] == "/kb/a.md" for r in rows.values())
    finally:
        store.close()