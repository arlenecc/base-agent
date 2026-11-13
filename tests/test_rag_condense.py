"""RAG 检索结果经辅助模型归纳精简后再注入主模型的流程测试。

覆盖：aux 可用时精简并注入、aux 为 None 时回退原始格式化、aux 调用失败时回退、
超限硬截断、空分片。
"""
import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from agent.llm_client import StreamEvent
from agent.tools.rag_tool import RagSearchTool
from agent.rag.postprocess import condense_retrieved_chunks, build_condensed_output


def _legacy_tool(engine, aux_getter):
    """关闭「检索即展示」，走旧的「全量结果精简/格式化注入」路径。

    同时把精简区间放宽到覆盖测试里的小样本，确保真的走到 aux 精简分支
    （默认下限 4000 字会把几十字的样本直接跳过）。
    """
    tool = RagSearchTool(engine, aux_llm_getter=aux_getter)
    tool._config = SimpleNamespace(
        rag_presentation_mode=False,
        rag_condense_min_chars=1,
        rag_condense_max_chars=1_000_000,
        rag_search_top_k=5,
    )
    return tool


class _FakeEngine:
    def __init__(self, chunks, formatted):
        self._chunks = chunks
        self._formatted = formatted
        self.search_called = False
        self.formatted_called = False

    def search(self, query, top_k=3, source=None):
        self.search_called = True
        return self._chunks

    def search_formatted(self, query, top_k=3, source=None,
                         max_chars_per_result=1500, max_total_chars=8000):
        self.formatted_called = True
        return self._formatted


class _FakeAux:
    def __init__(self, text):
        self._text = text

    def chat_stream(self, messages, tools=None, temperature=0.7, **kwargs):
        yield StreamEvent("content", content=self._text)
        yield StreamEvent("done")


def _chunks():
    return [
        {"text": "量子计算利用叠加与纠缠实现并行计算。", "source": "/kb/q1.md",
         "chunk_index": 0, "score": 0.9},
        {"text": "Shor 算法可多项式时间分解大整数。", "source": "/kb/q2.md",
         "chunk_index": 0, "score": 0.8},
    ]


def test_condense_via_aux_injects_summary():
    engine = _FakeEngine(_chunks(), "原始格式化长文本")
    aux = _FakeAux("量子计算：叠加/纠缠实现并行；Shor 算法可分解大整数。")
    tool = _legacy_tool(engine, lambda: aux)
    res = tool.run(query="量子计算", top_k=3)
    assert res.success
    assert "量子计算：叠加/纠缠" in res.output
    assert "q1.md" in res.output and "q2.md" in res.output
    # 走的是 aux 精简路径，而非原始格式化
    assert engine.search_called and not engine.formatted_called


def test_fallback_when_aux_none():
    engine = _FakeEngine(_chunks(), "原始格式化长文本")
    tool = _legacy_tool(engine, lambda: None)
    res = tool.run(query="量子计算", top_k=3)
    assert res.success
    assert res.output == "原始格式化长文本"
    assert engine.formatted_called


def test_fallback_when_aux_raises():
    class _BadAux:
        def chat_stream(self, messages, tools=None, temperature=0.7, **kwargs):
            raise RuntimeError("boom")
            yield  # 使其成为生成器

    engine = _FakeEngine(_chunks(), "原始格式化长文本")
    tool = _legacy_tool(engine, lambda: _BadAux())
    res = tool.run(query="量子计算", top_k=3)
    assert res.success
    assert res.output == "原始格式化长文本"
    assert engine.formatted_called


# ---------------------------------------------------------------------------
# 检索即展示（Retrieval-as-Presentation）
# ---------------------------------------------------------------------------

def test_presentation_injects_only_top_chunk():
    engine = _FakeEngine(_chunks(), "原始格式化长文本")
    tool = RagSearchTool(engine)          # 未 bind → 默认 presentation 开启
    res = tool.run(query="量子计算", top_k=3)
    assert res.success
    # 只有排名最高的切片正文进上下文
    assert "量子计算利用叠加与纠缠" in res.output
    assert "Shor 算法" not in res.output
    assert "知识库检索到 2 个相关切片" in res.output
    assert "相关度: 0.9000" in res.output
    # 两条召回都作为气泡附件；第一条标记为「已参考」
    assert len(res.attachments) == 2
    assert res.attachments[0]["used"] is True
    assert res.attachments[0]["source"] == "/kb/q1.md"
    assert res.attachments[1]["used"] is False
    assert res.attachments[1]["text"] == "Shor 算法可多项式时间分解大整数。"
    # 不走辅助模型精简（只有一片，无需归纳）
    assert not engine.formatted_called


def test_presentation_keeps_graph_items_in_context():
    graph_chunk = {
        "text": "[实体] 量子计算（类型: Concept，度数 2）\n利用叠加与纠缠。",
        "source": "知识图谱", "chunk_index": None, "score": 0.95,
        "kind": "kg_entity",
    }
    engine = _FakeEngine([graph_chunk] + _chunks(), "")
    tool = RagSearchTool(engine)
    res = tool.run(query="量子计算", top_k=3)
    assert "[实体] 量子计算" in res.output          # 图谱实体仍进上下文
    assert "量子计算利用叠加与纠缠" in res.output    # 排名最高的原文切片
    assert "Shor 算法" not in res.output
    # 气泡只包含文档切片，不含图谱条目
    assert [a["source"] for a in res.attachments] == ["/kb/q1.md", "/kb/q2.md"]
    assert res.attachments[0]["used"] is True


def test_presentation_empty_results_falls_back_to_formatted():
    engine = _FakeEngine([], "（知识库中未找到与该查询相关的内容。）")
    tool = RagSearchTool(engine)
    res = tool.run(query="不存在的主题", top_k=3)
    assert res.success
    assert res.output == "（知识库中未找到与该查询相关的内容。）"
    assert res.attachments == []
    assert engine.formatted_called


def test_condense_truncates_over_limit():
    aux = _FakeAux("重" * 5000)
    out = condense_retrieved_chunks("q", _chunks(), aux, max_chars=2000)
    assert len(out) <= 2000 + len("…（已截断）")


def test_condense_empty_chunks_returns_empty():
    aux = _FakeAux("anything")
    assert condense_retrieved_chunks("q", [], aux) == ""


def test_build_condensed_output_includes_sources():
    out = build_condensed_output("q", _chunks(), "摘要内容")
    assert "摘要内容" in out
    assert "q1.md" in out and "q2.md" in out
