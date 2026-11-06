"""残余缺陷修复验收（2026-11-06）：rag_ingest 破坏性标记、session 预算计入
raw_content 第二视图、失败累计硬上限熔断常量。

注：本文件只覆盖「低风险、可单测」的残余项；code_run/shell_run 沙箱、
temp 模式 aux 引擎复用、_render_raw O(n²) 等架构级/大型重构项不在此单测范围。
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, "src")

from agent.kernel.session import Session  # noqa: E402
from agent.tools.rag_tool import RagIngestTool  # noqa: E402


def test_rag_ingest_is_destructive():
    # rag_ingest 重建知识库是破坏性操作，必须触发人工确认。
    assert RagIngestTool.destructive is True


def test_render_raw_counts_raw_content_second_view():
    # 本地 MLX 后端会把 assistant["raw_content"] 随消息发送（跨轮 KV 前缀），
    # 预算估算必须把它计入，否则 RAW 区会静默超出实际配额。
    rec = SimpleNamespace(
        prompt="用户问题",
        assistant_text="答复",
        raw_content="X" * 300,  # 第二视图：逐 token 原始生成序列
        tool_calls=[],
        tool_results=[],
    )
    rendered = Session._render_raw([rec])
    assert "X" * 300 in rendered
    assert len(rendered) >= 300 + len("用户问题") + len("答复")


def test_failed_total_hard_cap_constant_defined():
    # 失败-成功交替抖动时，连续/同工具熔断被穿插成功清零；失败总数硬上限
    # 是覆盖该绕过场景的绝对兜底。常量必须存在且合理。
    from agent.kernel import loop as _loop  # noqa: E402

    assert hasattr(_loop, "_FAILED_TOTAL_HARD_CAP")
    assert isinstance(_loop._FAILED_TOTAL_HARD_CAP, int)
    assert _loop._FAILED_TOTAL_HARD_CAP > _loop._SAME_TOOL_FAIL_LIMIT
