"""Kernel primitives: states, outcomes, exit reasons.

Ported from GenericAgent's ``agent_loop.StepOutcome`` / ``BaseHandler`` and
extended with an explicit three-state automaton.

The kernel owns *nothing* but control flow: it asks the LLM (THINK),
dispatches tools (ACT), and reduces information entropy (CONSOLIDATE).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class KernelState(str, Enum):
    """三态自动机的状态。

    THINK       决策态：把折叠后的上下文交给 LLM，取得下一步意图。
    ACT         执行态：分发工具，能力全部在这一态落地。
    CONSOLIDATE 沉淀态：不调 LLM，只做信息减熵（L3 折叠 / L4 蒸馏）。
    """

    THINK = "think"
    ACT = "act"
    CONSOLIDATE = "consolidate"


class ExitReason(str, Enum):
    """自动机的终态原因（不是状态，是终止边的标签）。"""

    DONE = "done"                 # 模型给出最终答复
    INTERRUPT = "interrupt"       # 工具要求中断整轮（保留给未来的阻塞语义）
    MAX_TURNS = "max_turns"       # 超过预算轮次
    CANCELLED = "cancelled"       # 用户取消
    ERROR = "error"               # LLM / 传输异常


@dataclass
class ToolCall:
    """一个已解析的工具调用。"""

    id: str
    name: str
    args: Dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""

    @property
    def compact(self) -> str:
        """给 UI / 日志用的极简描述（不进上下文）。"""
        a = {k: v for k, v in self.args.items() if not k.startswith("_")}
        s = str(a)
        return f"{self.name}({s[:120]}{'...' if len(s) > 120 else ''})"


@dataclass
class StepOutcome:
    """一次工具分发的产物 —— 直接对应 GenericAgent 的同名结构。

    data         回灌给模型的内容（已在 L1 级裁剪过）。
    next_prompt  追加到下一轮 user 消息的提示；**空串表示任务完成**。
    should_exit  强制中断整条自动机。
    """

    data: Any = None
    next_prompt: Optional[str] = None
    should_exit: bool = False

    @property
    def finished(self) -> bool:
        return self.should_exit or not self.next_prompt


@dataclass
class LLMResponse:
    """一次 THINK 的产物。"""

    content: str = ""
    reasoning: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Dict[str, Any] = field(default_factory=dict)
    # 模型按协议输出的 <summary> 物理快照（L2 级摘要的原料）。
    summary: str = ""
    # 模型原始输出（含 think 标签与工具调用 XML），来自 done 事件的
    # raw_content。只有本地 MLX 后端会填；HTTP 后端恒为 ""。
    # 它是双视图历史的第二视图：保存进 TurnRecord 后，MLX 后端用它把历史
    # assistant 消息重建回生成时的逐 token 序列，跨轮 KV 才对得上前缀。
    raw_content: str = ""

    @property
    def has_calls(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class TurnRecord:
    """一轮的完整记录。Session 用它在 L3 阶段折叠历史。

    「一轮」= 一次 THINK + 随后的 ACT。折叠时作为一个整体丢弃，
    因此永远不会产生悬挂的 tool 消息（OpenAI 会因缺少前置
    assistant(tool_calls) 而 400）。
    """

    index: int = 0
    prompt: str = ""                                  # 注入的 user 内容（已压缩）
    assistant_text: str = ""                          # 模型文本（已压缩）
    # 双视图历史的第二视图：模型原始输出（含 think 标签与工具调用 XML）。
    # 只有本地 MLX 后端有值；HTTP 后端为空，历史沿用 assistant_text 旧形状。
    raw_content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)   # {id, name, content}
    summary: str = ""                                 # L2 摘要（本轮一行）
    tokens: int = 0                                   # 估算 token 数

    def digest_line(self) -> str:
        """L2/L3：本轮折叠成的一行摘要。"""
        if self.summary:
            return f"[T{self.index}] {self.summary}"
        if self.tool_calls:
            names = ",".join(tc.name for tc in self.tool_calls[:4])
            tail = "..." if len(self.tool_calls) > 4 else ""
            return f"[T{self.index}] 调用 {names}{tail}"
        if self.assistant_text:
            return f"[T{self.index}] {self.assistant_text[:40]}"
        return f"[T{self.index}] (空轮)"


# ---------------------------------------------------------------------------
# 转移表：内核唯一的控制流真相来源
# ---------------------------------------------------------------------------

# (当前状态, 事件) -> 下一状态。事件由 loop 在每步判定后查表。
TRANSITIONS: Dict[tuple, str] = {
    (KernelState.THINK, "has_calls"): KernelState.ACT,
    (KernelState.THINK, "no_calls"): "exit",
    (KernelState.THINK, "retry"): KernelState.THINK,
    (KernelState.THINK, "error"): "exit",
    (KernelState.ACT, "continue"): KernelState.THINK,
    (KernelState.ACT, "consolidate"): KernelState.CONSOLIDATE,
    (KernelState.ACT, "interrupt"): "exit",
    (KernelState.ACT, "done"): "exit",
    (KernelState.CONSOLIDATE, "done"): KernelState.THINK,
}


def next_state(current: KernelState, event: str) -> Any:
    """查转移表。未知组合一律终止，避免自动机静默卡死。"""
    return TRANSITIONS.get((current, event), "exit")
