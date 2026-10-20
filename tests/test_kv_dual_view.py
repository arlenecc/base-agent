"""双视图历史（content / raw_content）的契约测试。

这两条契约是本地 MLX 后端跨轮 KV 复用的前提，且必须保证对 HTTP 后端零影响：
  * 本地后端读 ``raw_content``，把它还原成「生成时的逐 token 序列」；
  * HTTP 后端必须把 ``raw_content`` 剥掉再发往远端。

本文件不加载模型，只验证纯函数与消息形状。
"""
from __future__ import annotations

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.llm_client import _strip_raw_content  # noqa: E402
from agent.local.llm import (  # noqa: E402
    _assistant_prompt_prefix,
    _rebuild_raw_messages,
)


# ---------------------------------------------------------------------------
# 本地后端：raw_content → 忠实于生成的形状
# ---------------------------------------------------------------------------

def test_rebuild_uses_raw_as_content():
    """带 raw_content 的 assistant 消息：content 换成 raw，并给字符串
    reasoning_content（模板暗门：传字符串即跳过自动拆分）。"""
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "正文", "raw_content": "<think>想</think>\n\n正文"},
        {"role": "user", "content": "u2"},
    ]
    out = _rebuild_raw_messages(msgs)
    a = out[2]
    assert a["content"] == "<think>想</think>\n\n正文"
    assert a["reasoning_content"] == ""
    assert "raw_content" not in a
    # 其余消息原样
    assert out[0] == {"role": "system", "content": "sys"}
    assert out[3] == {"role": "user", "content": "u2"}


def test_rebuild_noop_without_raw():
    """没有 raw_content（HTTP 后端历史）→ 原样返回，连对象都不重建。"""
    msgs = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "正文"},
    ]
    assert _rebuild_raw_messages(msgs) is msgs


def test_rebuild_ignores_empty_raw():
    msgs = [{"role": "assistant", "content": "正文", "raw_content": ""}]
    assert _rebuild_raw_messages(msgs) is msgs


# ---------------------------------------------------------------------------
# HTTP 后端：零影响
# ---------------------------------------------------------------------------

def test_strip_raw_content_removes_second_view():
    msgs = [
        {"role": "assistant", "content": "正文", "raw_content": "<think>想</think>\n\n正文"},
    ]
    assert _strip_raw_content(msgs) == [{"role": "assistant", "content": "正文"}]


def test_strip_raw_content_noop_when_absent():
    """绝大多数历史没有该字段 → 原样返回，避免每次请求都重建列表。"""
    msgs = [{"role": "user", "content": "u"}]
    assert _strip_raw_content(msgs) is msgs


# ---------------------------------------------------------------------------
# 模板预置片段（生成序列的头部，重建历史时必须带上）
# ---------------------------------------------------------------------------

def test_assistant_prompt_prefix():
    mark = "<|im_start|>assistant\n"
    # 开启思考：模板预置 <think> 开头
    assert _assistant_prompt_prefix(f"...{mark}<think>\n") == "<think>\n"
    # 关闭思考：模板预置空 think 块
    assert _assistant_prompt_prefix(f"...{mark}<think>\n\n</think>\n\n") == "<think>\n\n</think>\n\n"
    # 没有 assistant 标记 → 空串，不抛异常
    assert _assistant_prompt_prefix("no marker here") == ""


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
