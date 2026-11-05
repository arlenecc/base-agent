"""两个「沉淀态」原子工具（对齐 GenericAgent 的同名原子工具）。

- ``update_working_checkpoint``  —— 短期：本任务后续必须记住的非显然事实。
- ``start_long_term_update``     —— 长期：任务完成/检查点后触发经验蒸馏。

二者都不做重活：只写 Session 的工作记忆并返回极简回执。真正的 L4 蒸馏由内核
在 ``CONSOLIDATE`` / 终止边通过 Agent 的 hook 完成（异步、不阻塞回复）。
"""
from __future__ import annotations

import os
import re
from typing import Any, Optional

from ..config import AgentConfig
from ..kernel.compressor import compact_json
from ..memory_v2 import (
    LONG_TERM_MIN_TURNS, MemoryLayout, long_term_update_payload,
)
from .base import Tool, ToolRegistry, ToolResult
from .memory import _work_memory

# 跨 run 持久化的 key（Session 是进程内的，工作记忆还要能跨会话存活）
_CHECKPOINT_KEY = "__checkpoint__"
_MAX_KEY_INFO = 4_000


def _session(registry: ToolRegistry) -> Optional[Any]:
    """内核把当前 Session 挂在 registry 上，工具通过它读写工作记忆。"""
    return getattr(registry, "session", None)


class UpdateWorkingCheckpointTool(Tool):
    """Short-term working notepad — auto-injected into every subsequent turn.

    对应 GenericAgent 的 ``update_working_checkpoint``：只记「非显然、后续必须
    知道」的状态（已验证的路径、关键 ID、阶段性结论、已排除的方案），不记流水账。
    """

    name = "update_working_checkpoint"
    description = (
        "Short-term working notepad, auto-injected into every later turn. "
        "Call early/mid task ONLY when there is non-obvious state worth keeping: "
        "verified paths/IDs/ports, decisions, approaches already ruled out. "
        "Do NOT store transcripts or facts re-derivable in one tool call. "
        "Empty key_info clears it."
    )
    parameters = {
        "type": "object",
        "properties": {
            "key_info": {
                "type": "string",
                "description": f"Compact notes (<={_MAX_KEY_INFO} chars). Empty string clears the notepad.",
            },
            "persist": {
                "type": "boolean",
                "description": "Also persist across sessions (default false).",
            },
        },
        "required": ["key_info"],
    }

    config: AgentConfig
    registry: ToolRegistry

    def bind(self, config: AgentConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry

    def run(self, key_info: str, persist: bool = False) -> ToolResult:
        text = (key_info or "").strip()[:_MAX_KEY_INFO]
        sess = _session(self.registry)
        if sess is None:
            # 旧实现这里静默返回成功，模型以为 checkpoint 已写入，实际什么都没存
            # （无内核 session 时工具收不到任何回注）——失败必须可见。
            return ToolResult(
                False,
                error="working checkpoint unavailable (no kernel session bound)",
            )
        sess.set_key_info(text)
        if persist:
            try:
                _work_memory(self.config).set(_CHECKPOINT_KEY, text)
            except Exception as e:
                return ToolResult(False, error=f"persist failed: {e}")
        if not text:
            return ToolResult(True, output="working checkpoint cleared")
        hint = ""
        if sess.current_turn <= 2:
            hint = " (tip: skip re-calling unless new non-obvious facts appear)"
        return ToolResult(True, output=f"working checkpoint updated ({len(text)} chars){hint}")


class StartLongTermUpdateTool(Tool):
    """L2/L3 长期记忆的唯一正式写入触发点（改进 6）。

    它自己不写任何记忆文件：只做门控，并把【记忆提纯】prompt + 会话摘要 +
    L0 决策树回执给模型；模型再按决策树用 ``file_append`` 追加到 L2/L3。
    内核检测到该工具（未被门控拒绝时）仍会打上 distill 标记，终止边由
    Agent hook 做既有的知识图谱异步蒸馏。
    """

    name = "start_long_term_update"
    description = (
        "ONLY entry point for writing L2/L3 long-term memory. Call after finishing "
        "a task or reaching an important checkpoint; rejected if the session has "
        f"fewer than {LONG_TERM_MIN_TURNS} turns (short tasks must not touch "
        "long-term memory). Returns the 【记忆提纯】prompt; then classify verified "
        "facts by the L0 decision tree and append them with file_append to "
        "memory/global_mem.txt (L2) or memory/xxx_sop.md (L3). "
        "Never overwrite, never store unverified/volatile data."
    )
    parameters = {
        "type": "object",
        "properties": {
            "focus": {
                "type": "string",
                "description": "What to distill, e.g. 'python env setup' (optional).",
            },
        },
        "required": [],
    }

    config: AgentConfig
    registry: ToolRegistry

    def bind(self, config: AgentConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry

    def run(self, focus: str = "") -> ToolResult:
        sess = _session(self.registry)
        turns = int(getattr(sess, "current_turn", 0) or 0)
        # 改进 6 门控：短任务不许碰长期记忆。用「成功 + REJECTED」而不是失败
        # 返回：内核会把失败计入熔断记账，且本地小模型常对失败原样重试。
        if turns < LONG_TERM_MIN_TURNS:
            return ToolResult(True, output=compact_json({
                "status": "REJECTED",
                "reason": (f"当前仅 {turns} 轮（需 ≥{LONG_TERM_MIN_TURNS} 轮）："
                           "短任务不许碰长期记忆。本次请求已直接拒绝。"),
                "next": "不要重试本工具；直接完成当前任务即可。",
            }, 600))
        transcript = ""
        if sess is not None:
            transcript = "\n".join(sess.all_digest_lines()[-25:])
            ki = sess.get_key_info()
            if ki:
                transcript = f"[checkpoint]\n{ki}\n\n{transcript}"
        if not transcript:
            transcript = "(当前会话尚无有效记录)"

        layout = MemoryLayout(getattr(self.config, "workspace", "") or "")
        payload = long_term_update_payload(transcript, focus or "", layout)
        return ToolResult(True, output=compact_json(payload, 4000))


# ---------------------------------------------------------------------------
# 供 Agent 的 L4 hook 复用：把 checkpoint 从磁盘恢复到 Session
# ---------------------------------------------------------------------------

def restore_checkpoint(config: AgentConfig, session: Any) -> str:
    """把上次持久化的 checkpoint 恢复到 Session（跨会话连续性）。"""
    if session is None:
        return ""
    try:
        text = str(_work_memory(config).get(_CHECKPOINT_KEY) or "")
    except Exception:
        return ""
    if text and not session.get_key_info():
        session.set_key_info(text)
    return text
