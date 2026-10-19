"""RAG 检索结果经辅助模型归纳精简后再注入主模型的流程测试。

覆盖：aux 可用时精简并注入、aux 为 None 时回退原始格式化、aux 调用失败时回退、
超限硬截断、空分片。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from agent.llm_client import StreamEvent
from agent.tools.rag_tool import RagSearchTool
from agent.rag.postprocess import condense_retrieved_chunks, build_condensed_output


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

    def chat_stream(self, messages, tools=None, temperature=0.7):
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
    tool = RagSearchTool(engine, aux_llm_getter=lambda: aux)
    res = tool.run(query="量子计算", top_k=3)
    assert res.success
    assert "量子计算：叠加/纠缠" in res.output
    assert "q1.md" in res.output and "q2.md" in res.output
    # 走的是 aux 精简路径，而非原始格式化
    assert engine.search_called and not engine.formatted_called


def test_fallback_when_aux_none():
    engine = _FakeEngine(_chunks(), "原始格式化长文本")
    tool = RagSearchTool(engine, aux_llm_getter=lambda: None)
    res = tool.run(query="量子计算", top_k=3)
    assert res.success
    assert res.output == "原始格式化长文本"
    assert engine.formatted_called


def test_fallback_when_aux_raises():
    class _BadAux:
        def chat_stream(self, messages, tools=None, temperature=0.7):
            raise RuntimeError("boom")
            yield  # 使其成为生成器

    engine = _FakeEngine(_chunks(), "原始格式化长文本")
    tool = RagSearchTool(engine, aux_llm_getter=lambda: _BadAux())
    res = tool.run(query="量子计算", top_k=3)
    assert res.success
    assert res.output == "原始格式化长文本"
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
