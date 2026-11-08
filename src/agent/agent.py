"""The agent: reasoning loop, tool dispatch, streaming, confirmation, skills."""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from .config import AgentConfig
from .llm_client import LLMClient, StreamEvent
from .skills import SkillManager
from .tools import ToolRegistry, ToolResult, parse_args
from .tools import risk as _risk
from .kernel import (
    ContextBudget, ExitReason, KernelHooks, KernelLoop, KernelResult, Session,
    l4_extract_json, l4_normalize,
)
from .kernel.compressor import DISTILL_PROMPT, DISTILL_SYSTEM
from .tools.checkpoint import restore_checkpoint
from .atomic_io import atomic_write_text
from .memory_sop_texts import SOP_KEY_INFO_HINT
from .memory_v2 import (
    MemoryLayout, archive_past_sessions, ensure_l0, is_l23_write,
    is_l3_sop_read, memory_system_block, record_turn,
    run_memory_maintenance, sync_l1,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Callbacks protocol
# ---------------------------------------------------------------------------


class AgentCallbacks:
    """Override these methods to drive the UI. Default impls are no-ops."""

    def is_cancelled(self) -> bool:
        """协作式取消：run() 在循环边界/流式事件/工具执行前轮询。

        默认 False（本地单测无取消概念）。UI 的 BridgeCallbacks 返回
        「终止对话」按钮设置的标志——run() 必须能及时看到它，否则取消只在
        下一个恰好需要确认的工具处生效，其余场景会跑满 max_iterations。
        """
        return False

    def on_content(self, text: str) -> None: ...
    def on_reasoning(self, text: str) -> None: ...
    def on_tool_start(self, name: str, args: Dict[str, Any]) -> None: ...
    def on_tool_end(self, name: str, result) -> None: ...
    def on_log(self, line: str) -> None: ...
    def on_usage(self, usage: Dict[str, int]) -> None: ...
    def on_token_speed(self, total_tokens: int, speed: float) -> None: ...
    def on_skill_suggested(self, skill) -> None: ...
    def on_plan_updated(self, plan: Optional[Dict[str, Any]]) -> None: ...
    def on_error(self, message: str) -> None: ...
    def on_context_shrunk(self, summary: str, reason: str) -> None: ...

    def confirm(self, message: str) -> bool:
        return True

    def ask_user(self, prompt: str) -> Optional[str]:
        return None

    def on_finished(self) -> None: ...
    def on_final_answer(self, text: str) -> None:
        """一轮的【最终答复】（含内核事后构造的兜底文本）。

        与 ``on_content`` 不同：后者是流式片段，前者只在收尾时出现一次且是
        权威结果。UI 用它在本轮没有可见流式正文时补显示，避免「模型其实给了
        兜底答复、气泡却一片空白」。默认 no-op 以兼容非 GUI 调用方。
        """
        ...


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are base-agent, a local autonomous agent. Complete the user's task by reasoning, planning, and calling tools.

## 交互协议
- 每轮正文先输出 <summary>…</summary>（≤30 词真实快照：上一步新信息 + 本次意图），禁止空话。
- 需求未完成就调用工具推进；查资料类问答信息已足时**直接给出最终回答本身就是推进**，禁止把「接下来再查 X」当推进。
- `ask_user` 是弹框打断，非必要不用：仅缺自己无法获取的关键信息才问，一次一个问题；寒暄/问候/闲聊绝不调用，禁止问「你想做什么任务」、禁止问可推断的内容、禁止请求继续或许可。
- 禁止复读：同一句话/同一问题不输出第二遍。

## 任务推进
- 只有明显多步、需要跟踪进度的任务才先 `update_plan(op="create", ...)` 拆 2~7 个可验证步骤，**规划完立即开工**；**≤3 步可完成的小任务（单次读文件 / 单次问答 / 单次命令 / 单次搜索）不要建执行计划，直接执行即可**。
- 工具结果回来先在 <summary> 写清证实/推翻了什么；每步完成用 `update_plan(op="patch", status=..., note=...)` 更新；计划有未完成步骤不许只给文字总结收尾。
- 工具报错是信号：读报错分类 → 换没试过的方法重试 → 至少 3 种方法都失败才标 blocked 并继续其余步骤。绝不重复同样的失败调用、不把报错甩给用户、不停下问要不要继续。

## 全局记忆 L0~L4（核心公理常驻本提示；规则全文见 memory/memory_management_sop.md）
- 写任何 L1/L2/L3 前先 `file_read("memory/memory_management_sop.md")` 读全文并提炼进 key_info。L1 是极简索引：决策前先看；需要细节才 `file_read` L2(global_mem.txt) 或 memory/ 下的 L3 SOP，禁止把 L2/L3 全文塞进上下文。
- 读到 L3 SOP 后立刻用 `update_working_checkpoint(key_info=...)` 提炼「关键前置 + 典型坑 + 指针」，之后每轮随 key_info 回注，不重复读同一份。
- L2/L3 唯一正式写入入口是 `start_long_term_update`（轮数 <5 直接拒绝）：按 L0 决策树归类后用 `file_append` **追加**；严禁 `file_write`/`code_run` 覆盖、严禁写未验证/易变信息。写完内核自动同步 L1 并重新注入。
- 会话全文/摘要由系统自动落到 `memory/raw_sessions/`，无需手动记录。

## 何时调用工具 / 记忆
- 纯寒暄/问候/闲聊：只回一句文字，**绝不调用任何工具**（含 `rag_*` / `memory_*` / `update_plan`），不建计划、不写记忆、不检索知识库；即使知识库已加载也**绝不主动** `rag_outline` / `rag_search`。误调报错也不重试。
- 自我认知/能力类：直接回答，不查知识库、不查记忆。
- 消息里写了具体目标就是任务，直接开做；绝对不要说「你没有给出具体任务」。只缺某个参数时用 `ask_user` 问那一个参数。
- 依赖用户资料/文件的问题才查证：不必建计划、不必固化技能，但该调的工具照调。
- ⚠️ `rag_*` 检索是「被动」能力：仅在用户**显式**提到某文档/书名/具体查询词时才调用；用户只是打招呼或没提资料需求时，即使知识库已加载也**绝不主动** `rag_outline` / `rag_search`。
  - 列文档 → `rag_outline()` 不带参数。
  - 问某本书内容 → 先总后分：① `rag_outline(book_name="书名")` 拿目录/【全书摘要】；② 缺细节才 `rag_search(query=概念词, source="书名")`；③ 组织成有针对性、结构清晰的回答。**不要跳过 outline 直接 search**。收敛（内核强制：本轮 rag_search 超过 3 次会被直接拦截）：有【全书摘要】问整体内容时读摘要直接作答、不要再 search；确需细节只 search 一次前言/导论；章节已有摘要直接用；只涉及个别章节只 search 那几章。不重复已检索过的主题、不逐章扫描；禁止把「接下来再检索 X」写进回复；两轮无新信息立即用已有材料作答。
  - 问句没点名文档 → 先 `rag_outline()` 不带参数列全部：只 1 篇就对它先总后分；多篇时挑文档名/主题与问题最相关的一篇直接执行、回答开头注明「基于《XX》」；确实无法判断才用 ask_user 请用户选择。
  - 问文件内容 → `file_read` 读完给结论；`rag_search` 的 query 用概念词，不要把问句原样当检索词，查完不凭印象回答。
- 涉及用户本人/过往工作：先 `memory_search` 召回长期记忆再回答。
- 只在有真实状态要记时才调记忆工具；一步任务产生的「后续必须知道」事实记进 key_info，否则不记。

## 通用规则
1. 破坏性最小；不可逆/危险操作先确认。路径相对于工作区，仅 `code_run`/`shell_run`/外部 MCP 需确认。
2. 复杂任务先 `skill_semantic_search`，无结果再 `skill_search`；命中就 `skill_load` 照做。
3. 语义检索（memory_search/skill_semantic_search/rag_search）的 query 用用户原话的自然语言，禁止翻译/转拼音/压成光杆关键词。
4. 完成后用简洁纯文本回复。"""

# ---------------------------------------------------------------------------
# Token estimation: approximate token count from text length.
# Used for real-time speed display during reasoning streaming (where we don't
# yet have the API's usage figure) and as a fallback when the API's usage
# omits reasoning tokens entirely.
#
# 直接复用 kernel.budget.estimate_tokens：旧实现在本文件维护了一份常数不同
# （ASCII 4 / CJK 1.5 vs 3.5 / 1.3）的副本——预算记账与 tok/s 显示对同一段
# 文本算出两个数，且两边各自演化容易进一步分叉。保留 _estimate_tokens 作为
# 本模块的导入名，调用点无需改动。
# ---------------------------------------------------------------------------

from .kernel.budget import estimate_tokens as _estimate_tokens


# 系统提示动态注入段的上限。
# 这些段**不在五级压缩的作用范围内**（压缩只作用于 Session 的历史区）：
# 若不加限制会随会话时长单调膨胀，最终挤爆 30k 预算。
_MAX_INJECTED_FACTS = 20        # 当天事实最多注入条数
_MAX_FACTS_CHARS = 2000         # 事实段总字符上限
_MAX_FACT_CHARS = 120           # 单条事实字符上限
_MAX_WM_KEYS = 20               # 工作记忆最多注入的 key 数
_MAX_WM_CHARS = 2000            # 工作记忆段总字符上限（key + value）
_MAX_SKILL_PROMPT_CHARS = 1500  # 命中技能 prompt 的字符上限
_MAX_DAILY_FACTS = 200          # 当天事实桶的持久化条数上限

# ---------------------------------------------------------------------------
# Harness 调优参数
# ---------------------------------------------------------------------------

# 并发执行的工具上限。模型一次返回多个互相独立的只读调用时（多个 file_read /
# rag_search / web_scan），串行会让 wall time 变成 Σ，并发后是 max。
_MAX_PARALLEL_TOOLS = 8

# 自动续跑段数上限（每段 = 一次完整内核 run）。MAX_TURNS 截断或「提前收尾但
# 计划未完成」时自动续跑，替代旧的「停车等用户手动输入继续」。每段仍是完整
# 的内核 run，取消/反思门/失败恢复全部生效；额度用尽才交回用户。
#
# 4 段 × max_iterations(默认 15) = 最多 60 轮，长任务基本能自己跑完。段与段
# 之间通过 seed_failures 传递失败上下文，所以续跑段不是「从零重试」，而是
# 接着上一段的错误继续找方案。
#
# 另加墙钟上限：自动续跑原先只有段数/轮次约束，没有墙钟。本地推理模型
# 4 段 × 15 轮、每轮几十秒，最坏几十分钟不返回任何结果。到点后把已完成的
# 进展交回用户（MAX_TURNS 收尾文本本身就会总结进展），避免无限等待。
_AUTO_CONTINUE_MAX = 4
_AUTO_CONTINUE_WALL_CLOCK = 600.0   # 单次用户输入的全部续跑总时长（秒）

# 后台 aux 任务（事实抽取 / L4 蒸馏 / 记忆维护）让路的空闲阈值：主对话
# 安静这么久才动手，避免插队在两轮对话之间抢 GPU / _gen_lock / 冲 KV 缓存。
_AUX_IDLE_DELAY = 2.0
# 允许并发的工具白名单：纯只读、彼此无顺序依赖。写操作（file_write /
# file_modify）即使 destructive=False（工作区锁定）也不能并发——两个写操作
# 可能落在同一个文件上。
#
# 只放真正耗时的 I/O / 检索型工具：并发的收益来自网络与向量检索的等待，
# 本地瞬时操作并发意义不大，却会改变「取消后剩余调用逐个跳过」的既有语义
# （两个 file_read 同时启动后，中途取消就无法再拦住第二个）。因此 file_read /
# skill_load / rag_status / memory_graph 保持串行。
_PARALLEL_SAFE_TOOLS = frozenset({
    "web_scan",                                   # 网络请求
    "rag_search", "rag_outline",                  # 向量 + BM25 检索
    "skill_search", "skill_semantic_search",      # 技能检索（含 LanceDB 查询）
    "memory_search",                              # 长期记忆语义检索
})



# ---------------------------------------------------------------------------
# 能力层：工具分发
#
# 内核在 ACT 态调用 dispatcher.dispatch(tool_calls)。确认、取消、并发这些
# 「如何执行」的细节属于能力层，不属于控制流，因此放在这里而不是内核里。
# ---------------------------------------------------------------------------


class _ToolDispatcher:
    """把内核的 ToolCall 列表变成 ToolResult 列表（顺序一致）。"""

    def __init__(self, agent: "Agent"):
        self.agent = agent
        self._cache: Optional[List[Dict[str, Any]]] = None

    def schemas(self) -> List[Dict[str, Any]]:
        # 工具集在注册表构造后即固定；每轮重建纯属浪费（且会让
        # 「schema 已缓存」的断言失效）。
        if self._cache is None:
            try:
                self._cache = self.agent.tools.schemas()
            except Exception as e:
                # 旧实现把一次失败缓存成 []，之后整个会话模型都看不到任何
                # 工具，且内核的 schema 告警分支因异常已被吞永远不会触发。
                logger.warning("[kernel] 工具 schema 获取失败（本轮不缓存，下轮重试）: %s", e)
                return []
        return self._cache

    def dispatch(self, tool_calls: List["Any"]) -> List[ToolResult]:
        agent = self.agent
        results: Dict[int, ToolResult] = {}
        parallel_jobs: List[Tuple[int, str, Dict[str, Any]]] = []

        for idx, tc in enumerate(tool_calls):
            name, args = tc.name, tc.args

            # 内核解析参数失败 → 把原始串报回模型，而不是静默当 {}
            if "__parse_error__" in args:
                agent._log(f"[tool] {name} arg-parse error: {args['__parse_error__']}")
                agent.callbacks.on_tool_start(name, {"_raw_arguments": args.get("__raw__", "")})
                res = ToolResult(False, error=str(args["__parse_error__"]))
                agent.callbacks.on_tool_end(name, res)
                results[idx] = res
                continue

            if agent.callbacks.is_cancelled():
                # 必须为每个 tool_call 补一条结果，否则下一轮因悬挂 tool_calls 被 API 400 拒绝。
                agent._log(f"[tool] {name} 已因用户取消而跳过")
                agent.callbacks.on_tool_start(name, args)
                res = ToolResult(False, error="用户取消了对话，工具未执行")
                agent.callbacks.on_tool_end(name, res)
                results[idx] = res
                continue

            tool = agent.tools.get(name)
            if tool is None:
                agent.callbacks.on_tool_start(name, args)
                res = agent.tools.execute(name, args)   # 返回 unknown tool 错误
                agent.callbacks.on_tool_end(name, res)
                results[idx] = res
                continue

            if tool.should_confirm(args):
                confirm_msg = agent._confirm_message(name, args)
                if not agent.callbacks.confirm(confirm_msg):
                    agent._log(f"[tool] {name} DENIED by user")
                    res = ToolResult(False, error=f"User denied {name} ({confirm_msg})")
                    # 注意：**故意不发** on_tool_start——工具从未真正启动。
                    # tests/test_agent.py::test_code_run_aborted_when_user_declines
                    # 明确断言 tool_starts == []（UI 侧据此区分「未执行」与
                    # 「执行失败」）。曾尝试补发 start 使 start/end 配对，
                    # 结果与该契约冲突，已撤回。
                    agent.callbacks.on_tool_end(name, res)
                    results[idx] = res
                    continue
                agent.callbacks.on_tool_start(name, args)
                res = agent._execute_tool(name, args)
                agent._log_tool_result(name, res)
                agent.callbacks.on_tool_end(name, res)
                results[idx] = res
                continue

            if name in _PARALLEL_SAFE_TOOLS:
                parallel_jobs.append((idx, name, args))
                continue

            agent.callbacks.on_tool_start(name, args)
            res = agent._execute_tool(name, args)
            agent._log_tool_result(name, res)
            agent.callbacks.on_tool_end(name, res)
            results[idx] = res

        if parallel_jobs:
            agent._run_parallel_tools(parallel_jobs, results)

        return [
            results.get(i) or ToolResult(False, error="工具未执行（harness 内部错误）")
            for i in range(len(tool_calls))
        ]


# ---------------------------------------------------------------------------
# 经验层：内核 hook → 记忆 / 技能（L4 蒸馏）
# ---------------------------------------------------------------------------


class _AgentKernelHooks(KernelHooks):
    """把内核的 CONSOLIDATE / 终止边接到 base-agent 的经验系统。"""

    def __init__(self, agent: "Agent"):
        self.agent = agent

    def consolidated(self, session: Session, reason: str) -> bool:
        # 折叠已用尽仍超预算 → 用 LLM 摘要做最后兜底（旧 _shrink_context 的等价物）。
        # 返回「是否真的缩小了上下文」：内核据此决定遇 context 错误后要不要重试
        # ——没东西可压时重试同一个超长 prompt 是白费一次调用。
        if reason == "fold-exhausted":
            return bool(self.agent._shrink_session(reason))
        return False

    def before_consolidate(self, session: Session, reason: str) -> None:
        # 折叠丢弃旧轮「之前」：把即将被折掉的旧轮先抽一次事实，避免裁剪丢事实。
        # 与 finished 的批量沉淀共用 _fact_extract_lock（持锁期间跑 LLM），若上一次
        # 沉淀仍在进行则自动跳过，不会堆叠。抽完清零批量计数器，避免紧接着的
        # finished 又重复抽同一段。
        agent = self.agent
        keep = session.raw_turns + 1
        n_discard = max(0, len(session.turns) - keep)
        if n_discard <= 0:
            return
        try:
            agent._extract_turns_async(session.turns[:n_discard])
        except Exception as e:
            agent._log(f"[kernel] pre-trim extraction error: {e}")
        agent._last_extract_turn = session.current_turn

    def failed(self, session: Session, err: str) -> None:
        self.agent.callbacks.on_error(err)

    # 改进 5/6：内核只按标签插入提示/触发同步，路径分类留在这里（Agent 层）。
    def classify_tool_call(self, call: Any) -> str:
        if is_l3_sop_read(call):
            return "sop_read"
        if is_l23_write(call):
            return "l23_write"
        return ""

    def sop_hint(self) -> str:
        return SOP_KEY_INFO_HINT

    def memory_written(self) -> None:
        # L2/L3 变更 → 同步 L1 索引并让 system prompt 缓存失效（下一轮立即生效）
        _ = sync_l1(self.agent._memory_layout)
        self.agent._invalidate_prompt_cache()

    def finished(self, session: Session, text: str) -> None:
        agent = self.agent
        # 批量沉淀：不再每轮都抽。按会话累计轮数差值——每累计 raw_turns 个
        # 真实内核轮做一次事实抽取 + 经验蒸馏（自动续跑段/长任务自然计入）；
        # 内核「裁剪丢弃旧轮前」的 before_consolidate 强制沉淀并同步推进基准。
        # distill_requested 时立即沉淀。批大小与内核折叠窗口对齐——保证裁剪前
        # 总已把最近 raw_turns 轮的内容抽过，不会因折叠丢事实。
        turn_now = session.current_turn
        batch_due = (turn_now - agent._last_extract_turn) >= max(1, session.raw_turns)
        if not (batch_due or agent.kernel.distill_requested):
            return
        agent._last_extract_turn = turn_now
        # 既有能力：异步抽取事实写知识图谱（不阻塞回复）
        try:
            agent._maybe_extract_facts_async(text)
        except Exception as e:
            agent._log(f"[kernel] fact extraction error: {e}")
        # 改进 7/8：任务终止边主引擎空闲 → 后台归档 + 辅助引擎挖掘/整理
        try:
            agent._run_memory_maintenance_async()
        except Exception as e:
            agent._log(f"[kernel] memory maintenance error: {e}")
        # L4 经验蒸馏（模型显式调用 start_long_term_update，或长任务）。
        # 阈值用 session.current_turn（真实轮数）；distill_requested 时立即蒸馏。
        if agent.kernel.distill_requested or session.current_turn >= agent._distill_min_turns:
            try:
                agent._distill_experience_async(session)
            except Exception as e:
                agent._log(f"[kernel] distill error: {e}")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class Agent:
    def __init__(
        self,
        llm: "Any",
        config: AgentConfig,
        callbacks: AgentCallbacks,
        tool_registry: Optional[ToolRegistry] = None,
        skills: Optional[SkillManager] = None,
    ):
        self.llm = llm  # object with chat_stream(messages, tools, temperature)
        self.config = config
        self.callbacks = callbacks
        # 辅助模型（用于事实抽取、RAG 检索结果归纳等轻量子任务），由 UI 在运行时
        # 加载/卸载；初始为 None。RAG 工具通过惰性 getter 读取当前值，aux 未加载时
        # 自动回退到「重排结果原样注入」。
        self.aux_llm = None
        self.tools = tool_registry or ToolRegistry(
            config=config, callbacks=callbacks,
            aux_llm_getter=lambda: getattr(self, "aux_llm", None),
        )
        self.skills = skills or SkillManager(
            path=os.path.join(config.workspace, ".agent", "skills.json"),
            skills_dir=os.path.join(config.workspace, ".agent", "skills"),
        )
        # max_iterations 走 property（见类尾部定义）：读始终取 config 的当前值，
        # 写也落到 config —— 否则外部（UI「应用配置」）改 config.max_iterations
        # 后这里读的还是 __init__ 时的过期快照。
        # 注意：赋值必须发生在 property 已可用的前提下；Python 的 property 是
        # 类属性，实例化后立即生效，所以这里直接写即可。
        self.max_iterations = int(getattr(config, "max_iterations", 15) or 15)
        # max_history / max_context_tokens / context_shrink_ratio 仍从 config 读取
        # 以兼容既有配置与 UI；真正的上下文治理已由内核的五级压缩接管
        # （30k 预算 + L3 折叠），这里只保留字段避免外部读写报错。
        self.max_history = int(getattr(config, "max_history", 50))
        self.max_context_tokens = int(getattr(config, "max_context_tokens", 32000))
        self.context_shrink_ratio = float(getattr(config, "context_shrink_ratio", 0.9))
        self._history: List[Dict[str, Any]] = []  # 兼容字段：历史所有权已在 Session
        self._total_tokens = 0
        self._cached_prompt_turn = -1
        self._cached_prompt = ""
        self._turn_idx = 0  # incremented each run() call; used for prompt cache
        self._tool_call_counter = 0  # stable id generator for missing tool_call ids
        # schema 列表缓存：工具集在 ToolRegistry 构造完成后就不再变化（MCP 工具
        # 也是在 __init__ 里注册的），没必要每轮、每次迭代都重建一遍 dict。
        self._tools_schemas_cache: Optional[List[Dict[str, Any]]] = None

        # ---- 内核：历史交给 Session，能力交给 dispatcher，经验交给 hooks ----
        # Agent 退化为门面：run() 只做「记录请求 → 委托内核 → 回调 UI」。
        self._budget = ContextBudget(
            int(getattr(config, "kernel_max_context_tokens", 30_000) or 30_000)
        )
        self.session = Session(
            budget=self._budget,
            raw_turns=int(getattr(config, "kernel_raw_turns", 8) or 8),
            max_digest_lines=int(getattr(config, "kernel_max_digest_lines", 40) or 40),
        )
        # 工具通过 registry.session 读写工作记忆（update_working_checkpoint 等）
        self.tools.session = self.session
        # rag_ingest 工具由模型自行触发同步时，通知这里失效知识库状态缓存，
        # 避免随后 volatile_context 继续注入 30s 前的旧 chunks 数。
        self.tools.on_rag_data_changed = self.invalidate_rag_status
        # MCP 后台连接就绪后失效内核 schema 缓存：MCP 工具并入注册表后，
        # 下一轮 run 下发的 schema 才包含它们（lite 模式不含 MCP，不受影响）。
        def _on_mcp_tools_changed() -> None:
            self.kernel.invalidate_schemas()
            self._log("[mcp] 后台连接完成，工具已并入（下一轮生效）")
        self.tools.on_mcp_tools_changed = _on_mcp_tools_changed
        self._last_user_text = ""
        self._distill_min_turns = 6
        # L4 蒸馏的「已蒸馏摘要行数」快照：下轮只蒸新增内容（见
        # _distill_experience_async），避免窗口滑动重叠导致语义重复。
        self._last_distill_lines = 0
        # 自动续跑开关（MAX_TURNS 截断 / 提前收尾但计划未完成时自动续跑，
        # 替代「停车等用户手动输入继续」）。测试可置 False 关闭。
        self._auto_continue = True
        self.dispatcher = _ToolDispatcher(self)
        self.kernel = KernelLoop(
            # 传 callable：UI「应用配置」会整体替换 agent.llm
            llm=lambda: self.llm,
            dispatcher=self.dispatcher,
            callbacks=self.callbacks,
            config=self.config,
            system_prompt=self._system_prompt,
            volatile_context=self.volatile_context,
            session=self.session,
            hooks=_AgentKernelHooks(self),
            log=self._log,
            max_turns=int(getattr(config, "max_iterations", 15) or 15),
        )

        # 事实抽取的防堆叠锁。必须用 RLock：_extract_facts_sync 持锁期间会调
        # _get_extract_llm()，后者也要 acquire 同一把锁（普通 Lock 会自锁死，
        # 与 vector_store 的 _documents_lock 是同一类问题）。另外必须在
        # __init__ 里创建——懒初始化是 check-then-act 竞态，两个抽取线程可能
        # 各自建一把锁并各自成功 acquire，防堆叠形同虚设。
        self._fact_extract_lock = threading.RLock()
        # 后台记忆维护自己的防堆叠锁：与事实抽取锁分开，避免多个维护线程
        # 同时排队等锁（每个用户轮次都会在终止边触发一次）。
        self._memory_maint_lock = threading.Lock()
        # 最近一次「用户在等回复」的活动时刻（monotonic）。后台 aux 任务
        # （事实抽取 / L4 蒸馏 / 记忆维护）据此让路：主对话刚结束的几秒内
        # 不抢 GPU/`_gen_lock`，等用户继续提问时后台已经让位，回复首 token
        # 不再被后台任务排到后面（本地后端 aux 与主模型共用同一实例时，
        # 插队还会冲掉跨轮 KV 缓存，见 _get_extract_llm 注释）。
        self._last_activity = time.monotonic()
        # 事实抽取/蒸馏的批量计数：按「会话累计轮数」的差值计——每累计
        # session.raw_turns 个真实内核轮才沉淀一次。旧实现用「每个用户轮 +1」
        # 的计数器，emit_finished 收口到用户轮次末尾后，自动续跑 5 段 × 15 轮
        # 只算 1 次，长任务的事实抽取频率大幅下降；差值对续跑段/长任务自然
        # 累积。内核「裁剪丢弃旧轮前」强制沉淀时同步推进基准（见 before_consolidate）。
        self._last_extract_turn = 0

        # ---- 改进 1~4：L0~L4 记忆体系 ----
        self._memory_layout = MemoryLayout(config.workspace)
        # L0 提炼版常驻 system prompt，全文物化成文件供按需 file_read：
        # 启动即保证文件存在，避免「写记忆前先读全文」变成死链。
        try:
            ensure_l0(self._memory_layout)
        except Exception as e:  # pragma: no cover - defensive
            self._log(f"[memory] L0 materialize error: {e}")
        # 全文/摘要落盘挂在 Session.finish_turn 的统一回调：所有收尾分支都覆盖。
        # 传方法本身（callable），只有当天首轮才会真正取 system prompt。
        # 包一层 try/except + UI 日志：落盘失败用户必须看得见（旧实现静默）。
        def _record_turn(rec) -> None:
            try:
                record_turn(self._memory_layout, rec, self._system_prompt)
            except Exception as e:
                self._log(f"[memory] 会话落盘失败（对话可能未存档）: {e}")
        self.session.on_turn_finished = _record_turn
        # 应用启动时先把「今天之前」的会话归档掉（纯文件 IO，不碰任何 LLM，
        # 避免 UI 构造 Agent 时隐式加载辅助模型权重）。
        try:
            threading.Thread(
                target=lambda: archive_past_sessions(self._memory_layout),
                daemon=True, name="memory-archive-boot",
            ).start()
        except Exception as e:  # pragma: no cover - defensive
            self._log(f"[memory] boot archive error: {e}")

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """清空会话状态（UI 的「清空对话」）。"""
        self._history = []
        self._total_tokens = 0
        self._tool_call_counter = 0
        self.session.reset()
        self.kernel._total_tokens = 0
        self.kernel._last_consolidated_turn = -1
        # 清空对话必须连预算记账与跨段 seed 一起归零：否则新一轮的
        # CONSOLIDATE 会在重算前读到旧账，残留 seed 还会污染下一轮输入。
        for _sec in ("system", "earlier", "raw", "tools"):
            self._budget.set_used(_sec, 0)
        self.kernel._seed_failures = None
        self.kernel._rag_search_calls = 0
        self.kernel._repeat_streak = 0
        self.kernel._last_call_sigs = []
        self.kernel._step_fails = 0
        self.kernel._shrink_guard = 0
        # 会话已重置（current_turn 归零）→ 抽取基准同步归零。
        self._last_extract_turn = 0
        self._last_distill_lines = 0
        self._last_user_text = ""
        # Invalidate cached prompt
        self._cached_prompt_turn = -1
        self._cached_prompt = ""
        # 会话历史清空 → 本地后端的跨轮 KV 缓存对应的 token 序列彻底作废。
        self.invalidate_kv_cache()

    # ------------------------------------------------------------------
    # max_iterations：始终以 config 为唯一真源
    #
    # 旧实现是 __init__ 时的一次性快照（``self.max_iterations = config...``），
    # 之后 UI「应用配置」改 config.max_iterations 完全不生效——只有顺手重建
    # Agent 时才跟着变，表现为「改了最大轮次却没反应」。
    # 改成 property：读取实时取 config，赋值也写回 config，外部两种写法都对。
    # ------------------------------------------------------------------
    @property
    def max_iterations(self) -> int:
        return int(getattr(self.config, "max_iterations", 15) or 15)

    @max_iterations.setter
    def max_iterations(self, value) -> None:
        try:
            n = int(value)
        except (TypeError, ValueError):
            n = 15
        self.config.max_iterations = max(1, n)

    def close(self) -> None:
        """Release the dedicated background LLM client's connection pool.

        ``self.llm`` is owned by the UI (which closes it separately); we only
        own ``_extract_llm`` (a lazily-built independent LLMClient created to
        avoid racing the agent's shared httpx.Client from the extraction thread).

        关闭必须持锁：后台抽取/蒸馏线程可能正在用它，裸关会让它们半路炸掉
        （并留下未归还的连接）。取不到锁说明有后台调用在跑，交给线程自己收尾。
        """
        lock = getattr(self, "_fact_extract_lock", None)
        acquired = bool(lock and lock.acquire(blocking=False))
        try:
            # 取不到锁说明有后台调用在跑 —— 按上面的注释应该「交给线程自己收尾」，
            # 旧实现却无条件 _close_extract_llm()，把后台线程正在流式使用的
            # httpx.Client 关掉，那些线程会半路炸掉并留下未归还的连接。
            if acquired:
                self._close_extract_llm()
        finally:
            if acquired:
                lock.release()

    def invalidate_kv_cache(self) -> None:
        """丢弃本地后端的跨轮 KV 缓存。

        缓存对应的 token 序列是「上一轮 prompt + 上一轮输出」。一旦历史被改写
        （会话重置、L3 折叠把旧轮次换成摘要、本轮被取消只生成一半），下一轮的
        prompt 就不再以它为前缀——此时若还拿旧 KV 续算，轻则白算、重则输出
        错位内容。HTTP 后端没有这个方法，静默跳过。
        """
        for llm in (getattr(self, "llm", None), getattr(self, "aux_llm", None)):
            fn = getattr(llm, "invalidate_kv_cache", None)
            if not callable(fn):
                continue
            try:
                fn()
            except Exception as e:  # 防御：清理失败不该影响上层流程
                logger.debug("Agent: invalidate_kv_cache failed: %s", e)

    def history(self) -> List[Dict[str, Any]]:
        """当前可见历史的扁平视图（折叠区 + 原文窗口）。

        历史的所有权在 ``Session``；这里只是给 UI / 测试一个只读快照。
        """
        return self.session.build_messages(self._system_prompt())

    # ------------------------------------------------------------------
    # Context-size estimation and shrink strategy
    # ------------------------------------------------------------------

    def _tools_schemas(self) -> List[Dict[str, Any]]:
        """Cached tool schemas（供测试观察 schema 是否缓存；内核实际走
        ``_ToolDispatcher.schemas()``）。"""
        if self._tools_schemas_cache is None:
            try:
                self._tools_schemas_cache = self.tools.schemas() if self.tools else []
            except Exception:
                self._tools_schemas_cache = []
        return self._tools_schemas_cache

    @staticmethod
    def _is_context_too_long_error(err_str: str) -> bool:
        """Heuristically detect LLM API errors caused by exceeding the
        context window. Covers OpenAI, Anthropic, and common local server
        (vLLM/llama.cpp/Ollama) error message variants."""
        e = err_str.lower()
        if "context_length" in e or "context length" in e:
            return True
        if "context window" in e or "maximum context" in e:
            return True
        if "prompt is too long" in e or "prompt too long" in e:
            return True
        if "too many tokens" in e or "token limit" in e or "tokens limit" in e:
            return True
        return False

    def _summarize_via_llm(self, prompt: str, system: str = "", llm: Any = None) -> str:
        """Call the LLM (non-interactive, no tools) to produce a summary.

        Uses a low temperature for determinism. ``llm`` lets background threads
        pass the dedicated extraction client (``httpx.Client`` is not
        thread-safe, so they must not share ``self.llm`` with the UI thread).
        ``system`` lets callers swap the persona (摘要器 / 记忆提纯器).
        """
        client = llm or self.llm
        messages = [
            {"role": "system",
             "content": system or "你是精确的对话摘要器，用中文输出要点列表，不加前言和解释。"},
            {"role": "user", "content": prompt},
        ]
        out: List[str] = []
        for ev in client.chat_stream(messages, tools=None, temperature=0.2,
                                     kv_group="aux"):
            if ev.type == "content":
                out.append(ev.content)
            elif ev.type == "done":
                break
        return "".join(out).strip()

    def _append_summary_to_memory(self, summary: str, reason: str) -> None:
        """Append the shrink summary to workspace/memory.md with a timestamp
        header. Creates the file if absent. Uses atomic write (temp + rename)
        so concurrent shrinks or crashes can't corrupt the log."""
        try:
            memory_path = os.path.join(self.config.workspace, "memory.md")
            os.makedirs(self.config.workspace, exist_ok=True)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            entry = (
                f"\n## 上下文摘要 - {now}\n"
                f"- 触发原因: {reason}\n\n"
                f"{summary}\n"
            )
            # Read existing content (if any) and append. Atomic write to avoid
            # corruption if the process is killed mid-write.
            existing = ""
            if os.path.exists(memory_path):
                try:
                    with open(memory_path, "r", encoding="utf-8") as f:
                        existing = f.read()
                except OSError:
                    existing = ""
            # 必须用唯一临时名：固定 ``memory_path + ".tmp"`` 在并发触发
            # （多个后台线程同时收缩上下文）时会互相 truncate 同一个 tmp，
            # 落盘混合内容；写失败还会留下僵尸 .tmp。
            atomic_write_text(memory_path, existing + entry)
        except Exception as e:  # pragma: no cover - defensive
            self._log(f"[context] failed to append memory.md: {e}")

    def _wait_for_idle(self, max_wait: float = 15.0) -> None:
        """后台 aux 任务让路：等主对话空闲后再动 GPU / `_gen_lock`。

        用户点发送 → run() 更新 _last_activity。后台线程（事实抽取 / L4 蒸馏 /
        记忆维护）在动手前先等主对话安静 `_AUX_IDLE_DELAY` 秒；最多等
        max_wait 秒，防止后台任务被高频对话无限饿死。轮询 0.2s，用户取消时
        立即放行（后台任务本身也要响应取消语义）。

        本地后端的 aux 与主模型**共用同一实例**（见 _get_extract_llm）：若
        不让路，事实抽取会插在两轮对话之间抢 `_gen_lock`，用户等回复的首
        token 被排到后台任务后面；GGUF 上 aux 走 legacy 还会 reset 掉刚预热
        好的跨轮 KV 缓存，下一轮退回全量预填充（实测 6~9s）。
        """
        deadline = time.monotonic() + max_wait
        while (time.monotonic() - self._last_activity) < _AUX_IDLE_DELAY:
            if time.monotonic() >= deadline or self._cancelled():
                return
            time.sleep(0.2)

    def _maybe_extract_facts_async(self, assistant_reply: str) -> None:
        """Launch fact extraction in a daemon thread.

        The extraction calls the LLM (200ms-2s) and we don't want to delay
        on_finished() — the user should get their reply immediately.  The
        thread is a daemon so it won't block process exit if the user
        closes the app right after a reply.

        We capture the *recent conversation window* on the main thread before
        spawning the worker (the worker must not read the Session — by the
        time it runs, the main thread may have already started the next turn
        and folded/mutated it).  A multi-turn window (instead of just
        the last exchange) lets the extractor resolve pronouns and project
        references ("这个项目用 X 吗") that a single isolated turn cannot.
        """
        # Collect the last few user/assistant exchanges for context.
        recent: List[str] = []
        for t in reversed(self.session.turns):
            if (t.prompt or "").strip():
                recent.append(f"User: {t.prompt.strip()}")
                if len(recent) >= 6:
                    break
            if (t.assistant_text or "").strip():
                recent.append(f"Assistant: {t.assistant_text.strip()}")
                if len(recent) >= 6:
                    break
        recent.reverse()
        if not recent:
            return
        # 跳过无实质内容的对话：最新一条 user 消息过短（如「read data.txt」、
        # 「write and read back」这类单句指令）说明没有可抽取的持久事实，
        # 抽取只会浪费一次 LLM 调用。这与旧行为保持一致（user/assistant 均
        # 需 >= 20 字符才抽取）。
        last_user = (self._last_user_text or "").strip()
        if len(last_user) < 20:
            return
        if len((assistant_reply or "").strip()) < 20:
            return
        # The final reply is already in _history as the last assistant msg;
        # if not (edge case), append it explicitly.
        conversation = "\n\n".join(recent)

        import threading
        thread = threading.Thread(
            target=self._extract_facts_sync,
            args=(conversation,),
            daemon=True,
            name="fact-extraction",
        )
        thread.start()

    def _extract_turns_async(self, turns: List[Any]) -> None:
        """对给定的轮次集合异步抽取事实（通常用于「裁剪丢弃历史对话前」）。

        与 ``_maybe_extract_facts_async`` 不同：后者取最近窗口，本方法抽取**指定的**
        轮次集合，用于内核 ``before_consolidate`` 把即将被 ``force_fold`` 折掉、
        即将消失于上下文的旧轮先沉淀一遍，避免裁剪丢事实。同样走 daemon 线程 +
        ``_fact_extract_lock``，不会阻塞用户看到回复、也不会与别处沉淀堆叠。
        """
        convs: List[str] = []
        for t in turns:
            if (t.prompt or "").strip():
                convs.append(f"User: {t.prompt.strip()}")
            if (t.assistant_text or "").strip():
                convs.append(f"Assistant: {t.assistant_text.strip()}")
        if len(convs) < 2:
            # 至少要有一对 user/assistant 才有可抽取的事实，否则白费一次 LLM 调用。
            return
        conversation = "\n\n".join(convs)
        import threading
        thread = threading.Thread(
            target=self._extract_facts_sync,
            args=(conversation,),
            daemon=True,
            name="fact-extraction-trim",
        )
        thread.start()

    def _extract_facts_sync(self, conversation: str) -> None:
        # 先让路：主对话可能刚结束、用户正在等下一轮首 token（本地后端 aux
        # 与主模型共用实例，抢锁还会冲 KV 缓存）。
        self._wait_for_idle()
        """Best-effort: use the LLM to extract entities/relations/observations
        from a conversation fragment and persist them to the knowledge
        graph.  Runs on a daemon thread — any failure is logged and swallowed.

        Concurrency: extraction uses a *dedicated* LLM client (its own
        ``httpx.Client``), never the agent's ``self.llm``.  The agent's
        ``httpx.Client`` is NOT thread-safe — if the daemon thread reused it
        while the main agent thread streamed the next turn, the shared
        connection pool would race and fail the request.  A ``_fact_extract_lock``
        further prevents multiple extractions from piling up.
        """
        lock = self._fact_extract_lock
        if not lock.acquire(blocking=False):
            # A previous extraction is still running; skip to avoid piling up.
            return
        try:
            from .graph_memory import extract_facts_via_llm
            extract_llm = self._get_extract_llm()
            if extract_llm is None:
                # 拿不到专用客户端就跳过：绝不与 agent 线程共用 httpx.Client。
                self._log("[memory] 跳过事实抽取（无可用专用客户端）")
                return
            data = extract_facts_via_llm(extract_llm, conversation)

            entities = data.get("entities", [])
            relations = data.get("relations", [])
            if not entities and not relations:
                self._log("[memory] 本轮无可持久化事实（抽取为空或 LLM 返回空图）")
                return

            from .tools.memory import _get_long_memory
            lm = _get_long_memory(self.tools, self.config)
            g = lm.graph
            added = 0
            # 收集本次抽取到的所有 observation（实体观察），同时用于
            # 长期记忆（graph）与短期记忆（当天事实）两份写入。
            all_observations: List[str] = []
            for ent in entities:
                name = ent.get("name", "").strip()
                if not name:
                    continue
                etype = ent.get("type", "")
                obs = [o.strip() for o in ent.get("observations", []) if o and o.strip()]
                if obs:
                    g.add_observations(name, obs)
                    added += len(obs)
                    all_observations.extend(obs)
                elif etype:
                    g.create_entity(name, entity_type=etype)
            for rel in relations:
                src = rel.get("source", "").strip()
                tgt = rel.get("target", "").strip()
                lbl = rel.get("label", "").strip()
                if src and tgt and lbl:
                    g.create_relation(src, tgt, lbl)
            if added:
                self._log(f"[memory] 抽取 {added} 条 observation + {len(relations)} 条 relation 存入长期记忆")
            # 同时追加到短期记忆（当天桶）：一次抽取，两份写入，避免重复抽取。
            if all_observations:
                self._append_daily_facts(all_observations)
        except Exception as e:  # pragma: no cover — best-effort
            self._log(f"[memory] 事实抽取失败: {e}")
        finally:
            lock.release()

    def _get_extract_llm(self):
        """Return a dedicated LLM client for fact extraction.

        Fact extraction runs on a daemon thread concurrently with the agent's
        next turn.  ``LLMClient`` wraps a single ``httpx.Client`` that is NOT
        thread-safe, so extraction must not reuse ``self.llm``.  We lazily
        build one independent ``LLMClient`` (its own connection pool) and
        cache it on the instance.

        For non-``LLMClient`` llm objects (test mocks, scripted LLMs) we return
        ``self.llm`` unchanged — those have no shared socket state.
        """
        from .llm_client import LLMClient
        from .local.llm import LocalLLMClient

        # 辅助模型（若已配置并加载）且主模型是真实客户端时，事实抽取改用它，
        # 避免与主模型抢占推理资源。主模型是测试 mock / 脚本化 LLM 时（非真实
        # 客户端），仍走下方逻辑复用主 llm——否则会用真实辅助模型替换掉 mock，
        # 既偏离测试预期又白白加载一份权重。辅助客户端由 main_window 管理生命
        # 周期（加载/卸载），这里只读用、不缓存、不关闭。
        aux = getattr(self, "aux_llm", None)
        main_is_real = isinstance(self.llm, (LLMClient, LocalLLMClient))
        if aux is not None and main_is_real and isinstance(aux, (LLMClient, LocalLLMClient)):
            return aux

        # 「应用配置」换模型后必须重建：缓存的客户端仍指向旧 base_url/model。
        # 本地模型按对象身份做 key（没有 base_url/model 属性）。
        cur_key = (type(self.llm).__name__,
                   getattr(self.llm, "base_url", None),
                   getattr(self.llm, "model", None))
        cached = getattr(self, "_extract_llm", None)
        if cached is not None and getattr(self, "_extract_llm_key", None) == cur_key:
            return cached

        if not isinstance(self.llm, (LLMClient, LocalLLMClient)):
            # Non-LLMClient (test mock / scripted): reuse self.llm (no shared
            # socket state to race).
            return self.llm
        if isinstance(self.llm, LocalLLMClient):
            # 本地模型：抽取复用主客户端，避免重复加载 GB 级权重（本地推理可共享）。
            self._extract_llm = self.llm
            self._extract_llm_key = cur_key
            return self._extract_llm

        # 懒初始化会被两个线程并发触发：confirm 弹窗的
        # _describe_code_purpose（worker 线程）与抽取守护线程。不加锁会各自
        # 建一个 LLMClient，后写者覆盖 _extract_llm，被覆盖一方的 httpx 连接
        # 池永不关闭（close() 只关属性里最后一个）。
        #
        # 但这里绝不能阻塞等锁：抽取线程会**在整个 LLM 调用期间**持有这把锁
        # （见 _extract_facts_sync），阻塞等待会让 agent 线程卡住一个完整
        # 请求（最长 read timeout）才发出确认弹窗——与「不延迟弹窗」的意图
        # 正好相反。
        #
        # 取不到锁时返回 None（跳过本次抽取），**绝不**退回 self.llm ——
        # 那正是本方法存在的意义：self.llm 的 httpx.Client 正被 agent 线程
        # 流式使用，共用会直接踩连接池竞态。
        if not self._fact_extract_lock.acquire(blocking=False):
            return None
        try:
            # 锁内重查：key 变化时的旧客户端关闭也必须在锁内进行 —— 锁外关
            # 会命中「后台抽取/蒸馏线程持锁正用该客户端做流式请求」的窗口，
            # 把它正在跑的请求直接打断。能拿到锁就说明没有线程持有它整段
            # 调用（RLock 跨线程互斥），此时关闭才是安全的。
            cached = getattr(self, "_extract_llm", None)
            if cached is not None:
                if getattr(self, "_extract_llm_key", None) == cur_key:
                    return cached
                self._close_extract_llm()
                self._extract_llm = None
            try:
                self._extract_llm = LLMClient(
                    base_url=self.llm.base_url,
                    api_key=self.llm.api_key,
                    model=self.llm.model,
                    timeout=self.llm.timeout,
                    max_tokens=self.llm.max_tokens,
                    top_p=self.llm.top_p,
                    min_p=self.llm.min_p,
                    top_k=self.llm.top_k,
                    repetition_penalty=self.llm.repetition_penalty,
                )
                self._extract_llm_key = cur_key
                return self._extract_llm
            except Exception:  # pragma: no cover — best-effort
                pass
        finally:
            self._fact_extract_lock.release()
        return None

    def _close_extract_llm(self) -> None:
        """关闭专用抽取客户端（换模型 / close() 时调用）。"""
        client = getattr(self, "_extract_llm", None)
        self._extract_llm = None
        self._extract_llm_key = None
        if client is None or client is self.llm:
            return
        try:
            client.close()
        except Exception:
            pass

    def _append_daily_facts(self, facts: List[str]) -> None:
        """Append auto-extracted facts to work memory under today's bucket.

        短期记忆只保留当天的自动抽取事实；历史事实已在长期记忆 graph 中
        持久化（供 memory_search 检索），无需在短期记忆里重复保留。对话时
        只把当天事实注入上下文（见 _system_prompt），长期记忆则靠检索。

        Key 约定：``__auto_facts__`` → {"date": "YYYY-MM-DD", "facts": [...]}
        使用保留前缀，避免与模型手动 work_memory set 的 key 冲突。
        """
        import datetime
        from .tools.memory import _work_memory
        try:
            today = datetime.date.today().isoformat()
            wm = _work_memory(self.config)
            existing = wm.get("__auto_facts__") or ""
            # 解析现有当天事实（若跨天则视为过期，重置为空）。
            bucket_date, bucket_facts = self._parse_daily_facts(existing)
            if bucket_date != today:
                bucket_facts = []
            # 去重追加（保持原有顺序，新事实排后）。
            seen = set(bucket_facts)
            for f in facts:
                if f and f not in seen:
                    bucket_facts.append(f)
                    seen.add(f)
            # 桶上限：事实每轮累积，不设上限会同时撑爆 work_memory.json
            # 与注入的系统提示（注入侧另有 _MAX_INJECTED_FACTS 二次兜底）。
            # 超限时丢弃最旧的，保留最近的。
            if len(bucket_facts) > _MAX_DAILY_FACTS:
                bucket_facts = bucket_facts[-_MAX_DAILY_FACTS:]
            wm.set(
                "__auto_facts__",
                json.dumps({"date": today, "facts": bucket_facts}, ensure_ascii=False),
            )
        except Exception as e:  # pragma: no cover — best-effort
            self._log(f"[memory] 短期记忆追加失败: {e}")

    @staticmethod
    def _parse_daily_facts(raw: str):
        """Parse the __auto_facts__ work-memory value → (date, [facts]).

        Returns ("", []) on malformed/empty input so callers treat it as
        "no valid today-bucket" and reset.
        """
        if not raw:
            return "", []
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                return "", []
            date = data.get("date", "")
            facts = data.get("facts", [])
            if not isinstance(date, str) or not isinstance(facts, list):
                return "", []
            return date, [f for f in facts if isinstance(f, str) and f.strip()]
        except (json.JSONDecodeError, TypeError):
            return "", []

    # ------------------------------------------------------------------
    def warm_up(self, client=None) -> None:
        """模型加载完成后预热一次：触发首次推理的编译/初始化，让首句不等冷启动。

        **仅 MLX 需要**：``mlx_lm`` 首次 generate 要编译计算图（常 20-30s），且按
        输入序列形状惰性编译并缓存。旧实现有两个致命问题，导致编译的图首句完全
        复用不上、实测预热跑完首句首 token 仍要 32s（比不预热的 17s 还慢）：
          1. 没传 ``tools``：首句 prompt 含 23 个工具 schema（数千 token），占位
             提问只有 system + 一句短问，序列长度差几千 → 必然按新形状重编译；
          2. 没带易变上下文：kernel 首句把 ``anchor_prompt`` + ``volatile_context``
             一并拼进 user 消息，占位提问完全没有这段。
        这里按 kernel 首句的**同一构造方式**拼占位消息并带上 tools，使 warm_up
        编译的图与首句形状一致。走 ``kv_group="aux"`` 独立缓存，不污染主对话缓存。
        GGUF(llama.cpp)/ONNX 没有图编译，跳过；系统段 prefill 由 ``preload_kv()``
        承担。实测 MLX：本修正 + preload_kv 后首句首 token 32.3s → 1.3s，说明
        preload_kv 并非「白等」，它把 6007 token 的预填充锁在空闲期；去掉它首句
        要全量预填充 6055 token（18s）。两者必须配套。

        网络后端 / 未标注 backend 的 mock 保持原行为。失败不影响主流程。
        """
        client = client or self.llm
        if client is None:
            return
        if getattr(client, "_backend", None) in ("gguf", "onnx"):
            return
        try:
            # 与 kernel/loop.py 的 _ask 首句保持同样的消息形状：
            #   user_text = session.anchor_prompt(extra=...) + "\n\n" + volatile_context()
            # 只把「用户真正输入的那句话」换成占位提问，其余（锚点/工作记忆/知识库
            # 状态/匹配技能/tools schema/system）全部照搬 → prompt 长度与首句几乎相同。
            # 不调用 session.build_messages()：它会改写窗口/折叠计数等会话状态。
            placeholder = (
                "你好，请介绍一下你自己，并说明你作为一个本地自主智能体"
                "都能帮我完成哪些任务？"
            )
            user_text = self.session.anchor_prompt(extra=placeholder)
            vc = self.volatile_context()
            if vc:
                user_text = user_text + "\n\n" + vc
            messages = [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": user_text},
            ]
            tools = self.kernel.schemas() if self.kernel else None
            # 消费完整流（含思考/重试），触发首次编译/图优化。max_tokens 限极小值：
            # 预热只需「跑一次真实形状的前向（prefill+少量 decode）」，不必产出整段
            # 回答——MLX 的图编译与输出长度无关，限短同样能触发。
            for _ in client.chat_stream(messages, tools=tools,
                                        kv_group="aux", max_tokens=4):
                pass
            # 注意：此处**不再** invalidate 主缓存。aux 预热走独立缓存、不写实例级
            # self._mlx_cache，原 invalidate 本就是 no-op；若在此清缓存，会误删
            # worker 后续预填好的系统提示 KV。
        except Exception as e:  # 预热失败不致命
            self._log(f"[warmup] 预热失败（忽略）: {e}")

    # ------------------------------------------------------------------
    def preload_kv(self) -> None:
        """在 worker 线程内预填系统提示 KV，让首句无需全量预填充。

        MLX / GGUF 两个本地后端都有效（网络后端无 KV 缓存概念，忽略）。必须在
        真实生成线程调用——由 AgentWorker.run() 在空闲期触发，缓存与首句生成
        同线程/同上下文，首句 form ① 命中：实测 GGUF 首句 eval 50 token、
        MLX 首句 suffix 48 token（都只付用户消息那段）。失败不致命。
        """
        client = self.llm
        if client is None:
            return
        preload = getattr(client, "preload_system_kv", None)
        if preload is None:
            return
        try:
            system_prompt = self._system_prompt()
            schemas = self.kernel.schemas() if self.kernel else []
            preload(system_prompt, schemas)
        except Exception as e:
            self._log(f"[warmup] 系统提示 KV 预热失败(忽略): {e}")

    def run(self, user_input: str) -> str:
        """Process one user turn. Returns the final assistant content.

        控制流已下沉到 ``kernel.KernelLoop``（三态自动机 + 五级压缩）。
        这里只保留门面职责：记录请求、恢复工作记忆、同步上限、委托内核、回调 UI。
        """
        self._turn_idx += 1
        # Invalidate per-turn cached prompt so it's rebuilt with current memory/skills.
        self._cached_prompt_turn = -1
        self._cached_prompt = ""
        self._last_user_text = user_input
        # 标记「用户在等回复」：后台 aux 任务据此让路（见 _last_activity 注释）
        self._last_activity = time.monotonic()
        self._log(f"USER: {user_input}")

        # skill suggestion (固化) - non-blocking
        # 寒暄 / 问询 / 查询类（简单、无需计划、非多步完成）不值得固化成技能，
        # 跳过计数，避免反复弹出「是否保存为技能」的打扰弹窗。这类输入命中
        # _is_simple_input（短且无多步信号，恰好对应「不涉及计划、非多步」），
        # 真正的多步/执行类任务仍正常累积、达阈值后建议固化。
        if not KernelLoop._is_simple_input(user_input):
            try:
                suggested = self.skills.record_request(user_input)
                if suggested is not None:
                    self.callbacks.on_skill_suggested(suggested)
            except Exception as e:  # pragma: no cover
                self._log(f"[skills] record_request error: {e}")

        # 跨会话连续性：恢复上一次 persist 的工作检查点
        restore_checkpoint(self.config, self.session)

        # 任务边界：上一个计划已全部终态化 → 新请求应当重新规划。
        # 不清的话旧计划会一直通过 anchor 回注，模型沿着过期计划走，也不再调用
        # update_plan —— 表现为「计划面板停在第一次那幅，新任务没人监督」。
        # 若还有未完成步骤则**保留**（用户说「继续」时要能接着干），但必须注入
        # 一次性 fork 提示：旧计划有 open 步骤时模型会静默沿用旧计划——上一任务
        # 的卡住步骤会让**新任务**也顶着旧计划跑，表现为「新任务的计划与输入
        # 无关」。提示强制模型显式判断：新任务 → create 替换；继续 → 接着做。
        if (self.session.get_plan() is not None
                and not self.session.plan_open_steps()):
            self.session.clear_plan()
            self._log("[plan] 上一任务已收尾，清空计划以便重新规划")
        elif self.session.get_plan() is not None:
            open_ids = "; ".join(
                f"{s.get('id')}.{s.get('title')}"
                for s in self.session.plan_open_steps()[:5]
            )
            self.session.working["__plan_boundary__"] = (
                f"上一个任务的计划还有未完成步骤：{open_ids}。请先判断本条消息的意图："
                "若是一个**新任务**（与旧计划的目标无关），立即用 update_plan"
                "(op=\"create\", goal=..., steps=[...]) 建立新计划替换旧计划，"
                "不要沿用无关步骤；若是要求**继续**上一个任务，则从剩余步骤接着执行。"
            )
            self._log("[plan] 上一任务计划未收尾，已注入新任务/继续任务的判断提示")
        # 每轮同步一次面板：即使模型本轮没调 update_plan，用户看到的也是
        # Session 的真实状态（而不是上一轮的残影）。
        # 必须传**深拷贝**：直接把 Session 里的活字典交给 UI 线程，正是
        # session.py 里明确列为已修 bug 的跨线程共享可变状态模式（当前只因
        # plan patch 走深拷贝+整体替换才侥幸安全）。
        try:
            self.callbacks.on_plan_updated(deepcopy(self.session.get_plan()))
        except Exception:  # pragma: no cover - defensive
            pass

        # max_iterations 可能在构造后被 UI 改动 → 每轮同步进内核。
        # property 读的就是 config 的当前值，无需再手工搬运。
        self.kernel.max_turns = max(1, int(self.max_iterations or 15))

        # emit_finished=False：hooks.finished 会做事实抽取 + L4 蒸馏（各一次
        # 后台 LLM 调用）。自动续跑可能有多段，每段都触发会让单次用户输入跑最多
        # 4×2 次冗余后台调用、并向知识图谱重复写入近似实体。统一挪到本方法末尾
        # 只触发一次（见下方 `self._fire_finished_hook`）。
        result: Optional[KernelResult] = None
        try:
            result = self.kernel.run(user_input, emit_finished=False)
            # 首段的失败上下文：续跑段要靠它接着上一段的错误找方案。
            carry: Optional[Dict[str, Any]] = self.kernel.failure_snapshot()

            # ---- 自动续跑：消灭「必须人工输入继续才能往下走」----
            # 两类停车都不该甩回给用户：
            #   1. MAX_TURNS：长任务在轮次上限处被硬截断；
            #   2. DONE 但计划仍有未完成步骤（反思门提醒额度用尽）：模型反复给出
            #      文字总结，而计划面板显示还有 open 步骤——用户看到的正是
            #      「实际已完成却一直显示有未完成项，必须打继续」。
            # 每段续跑都是完整的内核 run（轮次预算/反思门/取消/失败恢复全部生效，
            # _finish_nudge 等记账也随 run 重置）；最多续 _AUTO_CONTINUE_MAX 段，
            # 期间用户点「终止」立即停。额度用尽才真正交回用户。
            segments = 0
            t_run_start = time.monotonic()
            while (segments < _AUTO_CONTINUE_MAX
                   and self._auto_continue
                   and result.reason in (ExitReason.MAX_TURNS, ExitReason.DONE)
                   and not self._cancelled()
                   and self._should_auto_continue(result)
                   # 墙钟上限：段数/轮次约束不了「每轮都很慢」——本地推理模型
                   # 4 段 × 15 轮、每轮几十秒，最坏几十分钟不返回。到点交回用户
                   # （MAX_TURNS 收尾文本本身会总结已完成的进展）。
                   and time.monotonic() - t_run_start < _AUTO_CONTINUE_WALL_CLOCK):
                segments += 1
                # 把上一段的失败上下文带进本段：否则 run() 重置记账后，续跑段既不
                # 知道上一段卡在哪，也拿不到失败感知的反思门预算 —— 表现为「续跑
                # 之后就不再针对报错找办法，而是原地打转然后停住」。
                if carry:
                    self.kernel.seed_failures(carry)
                open_ids = "; ".join(
                    f"{s.get('id')}.{s.get('title')}"
                    for s in self.session.plan_open_steps()[:5]
                )
                # 失败信息必须跟着续跑一起走（两条续跑路径都要）：续跑段最需要的
                # 就是「上一段卡在哪」，否则它只能从头瞎试、然后再次停住。
                fail_txt = ""
                if carry and carry.get("last_fail"):
                    fname, ferr = carry["last_fail"]
                    fail_txt = (
                        f"\n上一段最后一次失败：`{fname}` → {ferr[:200]}。"
                        "请先读完整错误定位原因，再换一种**没试过**的方法"
                        "（换工具 / 换参数 / 用 code_run 写最小诊断脚本验证假设）。"
                        "不要原样重试，也不要只回答文字。"
                    )
                if result.reason is ExitReason.MAX_TURNS:
                    self._log(f"[agent] 达到轮次上限，自动续跑 {segments}/{_AUTO_CONTINUE_MAX}")
                    self.session.working["__truncated__"] = (
                        f"[系统提示] 上一段在达到轮次上限（{self.max_iterations}）后被自动续跑。"
                        "请先用一句话总结当前进展，然后**立即继续执行**计划中的剩余步骤；"
                        f"若全部步骤已完成，用 update_plan 收尾并给出最终答复。{fail_txt}"
                    )
                    result = self.kernel.run(
                        "[System] 任务在上一段达到轮次上限被截断，已自动续跑。"
                        "请继续执行计划中的剩余步骤；不要询问用户是否继续。",
                        resume=True,
                        emit_finished=False,
                    )
                else:
                    self._log(f"[agent] 模型提前收尾但计划未完成（{open_ids}），自动续跑 "
                              f"{segments}/{_AUTO_CONTINUE_MAX}")
                    self.session.working["__plan_boundary__"] = (
                        f"你上一条回复被判为提前收尾，但计划中仍有未完成步骤：{open_ids}。{fail_txt}\n"
                        "若这些步骤的工作**实际已经完成**，立即用 update_plan"
                        "(op=\"patch\", step_id=..., status=\"done\", note=\"结论\") 逐一收尾，"
                        "然后给出最终答复；若尚未完成，继续执行；"
                        "只有至少 3 种不同方法都确认不可行，才标为 blocked/skipped "
                        "并写明原因后再给出最终答复。"
                        "不要再次只给出文字总结就停止。"
                    )
                    result = self.kernel.run(
                        "[System] 上一条回复被判定为提前收尾，已自动续跑。"
                        "请按上面的指示处理计划中的剩余步骤；不要询问用户是否继续。",
                        resume=True,
                        emit_finished=False,
                    )
                carry = self.kernel.failure_snapshot()
            if result.reason is ExitReason.MAX_TURNS:
                if segments:
                    # 续跑额度用尽仍被截断：如实告知用户，等待指示。
                    self._log("[agent] 轮次上限且自动续跑额度用尽，停车等待用户指示。")
                    self.session.working["__truncated__"] = (
                        f"[系统提示] 任务已连续 {segments + 1} 段达到最大轮次上限"
                        f"（每段 {self.max_iterations} 轮），已自动停止。"
                        "请先总结已完成的进展与剩余工作，等待用户指示后再继续。"
                    )
                else:
                    self._log("[agent] max turns reached, stopping.")
                    # 让下一轮知道上一轮是被截断的：否则模型会以为任务已完成。
                    self.session.working["__truncated__"] = (
                        f"[系统提示] 上一轮在达到最大轮次上限（{self.max_iterations}）后被中断，"
                        "任务可能尚未完成。请先总结已完成的进展，再继续执行剩余步骤。"
                    )
        except Exception as e:
            # 兜底：内核内部的异常保护不可能覆盖 session.build_messages /
            # 工具 __str__ / json.dumps 等所有路径。旧实现没有这一层——一旦
            # 抛出，on_finished 与 _fire_finished_hook 都不执行，UI 永远停在
            # 「思考中」，用户既看不到错误也没有输入入口，只能重启。
            self._log(f"[agent] run() 未捕获异常: {type(e).__name__}: {e}")
            try:
                self.callbacks.on_error(f"{type(e).__name__}: {e}")
            except Exception:
                pass
            result = None
        finally:
            # 无论正常结束 / 取消 / 异常，都必须触发 finished——UI 的 _busy
            # 复位全靠它（见 AgentWorker.run 与 _on_finished）。
            text = ""
            if result is not None and (result.text or "").strip():
                text = result.text
            elif result is not None and not (result.text or "").strip():
                # 空回复要按退出原因分流：旧实现一律显示「模型连续多次返回
                # 空响应」——用户点了「终止对话」却被告知模型返回空响应，
                # LLM 报错时也看不到真实错误（真实错误只走了一条 on_error）。
                reason = getattr(result, "reason", None)
                if reason is ExitReason.CANCELLED:
                    text = "（已取消本次对话。）"
                elif reason is ExitReason.ERROR:
                    text = "（本轮因模型/网络异常中断，请查看上方错误信息后重试。）"
                else:
                    text = (
                        "（模型连续多次返回空响应，已停止重试。请检查网络/API 状态后重试，"
                        "或换个说法再问一次。）"
                    )
                self._log("[agent] 空回复：已向用户返回可见提示")
            elif result is None:
                text = "（本轮异常中断，请查看上方错误信息后重试。）"
            # 取消 / 异常 / 空回复轮次不做经验沉淀：否则「（已取消本次对话。）」
            # 会被拿去抽取事实、写知识图谱，甚至固化成 SKILL.md。UI 的 _busy
            # 复位由下面无条件执行的 callbacks.on_finished() 负责，不受影响。
            self._fire_finished_hook(
                text,
                allow_memory=(result is not None
                              and result.reason in (ExitReason.DONE, ExitReason.MAX_TURNS)
                              and not getattr(result, "no_memory", False)
                              and bool((text or "").strip())),
            )
            # 先于 on_finished 发出：两者都是 worker 线程 → UI 线程的 queued
            # 信号，FIFO 保序，UI 收尾时能读到本轮最终答复。内核/agent 事后
            # 构造的兜底文本（问候收尾、工具失败熔断占位、关思考作答、空回复
            # 提示）从未走过 on_content 流式通道，只靠这里补到对话气泡。
            try:
                self.callbacks.on_final_answer(text)
            except Exception as e:
                self._log(f"[agent] on_final_answer 失败(忽略): {e}")
            self.callbacks.on_finished()
        return text

    def _fire_finished_hook(self, text: str, allow_memory: bool = True) -> None:
        """整个用户轮次（含全部自动续跑段）结束后，触发一次 finished 钩子。

        事实抽取与 L4 蒸馏都在这个钩子里，是「一次用户输入一次」的语义；
        直接由内核每段触发会把它们放大成最多 4 次。
        ``allow_memory=False`` 用于取消/异常/空回复轮次：只跳过经验沉淀，
        UI 复位仍由调用方的 ``callbacks.on_finished()`` 完成。
        """
        if not allow_memory:
            self._log("[agent] 取消/异常/空回复轮次跳过经验沉淀")
            return
        hooks = getattr(self.kernel, "hooks", None)
        finished = getattr(hooks, "finished", None)
        if not callable(finished):
            return
        try:
            finished(self.session, text or "")
        except Exception as e:
            self._log(f"[kernel] finished hook error: {e}")

    def _should_auto_continue(self, result: KernelResult) -> bool:
        """判断是否值得自动续跑。"""
        if result.reason is ExitReason.MAX_TURNS:
            return True
        if result.reason is ExitReason.DONE:
            return bool(self.session.plan_open_steps())
        return False

    def _cancelled(self) -> bool:
        try:
            return bool(self.callbacks.is_cancelled())
        except Exception:
            return False

    # ------------------------------------------------------------------
    # 兜底收缩：内核 L3 折叠用尽后仍未落回预算时的最后一道防线
    # ------------------------------------------------------------------
    def _shrink_session(self, reason: str) -> bool:
        """把最老的原文轮交给 LLM 摘要，写进 memory.md 并折叠。

        正常情况下五级压缩（尤其是 L2/L3）已经把历史压成常量，走不到这里。
        保留它是因为单个巨型工具输出（如整本 PDF）仍可能一次性击穿预算。
        """
        session = self.session
        if len(session.turns) <= 1:
            return False

        old = session.turns[:-1]
        transcript = "\n".join(
            f"{t.digest_line()}\n{(t.assistant_text or '')[:600]}"
            + "".join(str(r.get("content", ""))[:400] for r in t.tool_results)
            for t in old
        )
        summary_prompt = (
            "把下面的对话压缩成要点摘要，只保留：\n"
            "1. 用户的核心需求与当前目标\n2. 已做的决定及原因\n"
            "3. 已完成的步骤与结果（含工具关键输出）\n4. 未完成的子任务与下一步\n"
            "5. 具体路径、数值、标识符、配置值、偏好、URL\n6. 错误/警告等需注意的事项\n\n"
            "要求：只输出要点列表，不要前言或解释；不要复述对话；不要编造。\n\n"
            f"对话历史:\n{transcript}"
        )
        # 改进 8：压缩摘要改由辅助引擎承担，绝不占主引擎推理流。
        #  * UI 已加载辅助模型 → 直接用它（它就是「辅助引擎」）；
        #  * 未加载 → 用独立抽取客户端（自己的 httpx 连接池/推理实例）；
        #  * 拿到的仍是本地主引擎本身 → 干脆不调 LLM，直接用已有 L2 摘要行
        #    折叠，宁可摘要粗一点也不让用户在压缩轮卡十几秒。
        # L1/L2/L3 的系统化整理由后台维护线程按「记忆整理 SOP」在辅助模型
        # 空闲时执行（见 _run_memory_maintenance_async）。
        # 注意：这里不 import .local.llm（会拉起 MLX 依赖），用类名判定本地
        # 客户端即可；否则缺依赖的环境会把整段压缩吞成空摘要。
        try:
            aux = getattr(self, "aux_llm", None)
            if aux is not None:
                summary = self._summarize_via_llm(summary_prompt, llm=aux)
            else:
                try:
                    client = self._get_extract_llm()
                except Exception:
                    client = self.llm
                if (client is self.llm
                        and type(self.llm).__name__ == "LocalLLMClient"):
                    summary = ""
                else:
                    summary = self._summarize_via_llm(summary_prompt, llm=client)
        except Exception as e:
            self._log(f"[context] summarize LLM call failed: {e}")
            summary = ""
        if not summary.strip():
            summary = "; ".join(t.digest_line() for t in old) or "(上下文收缩：旧轮已清理)"

        self._append_summary_to_memory(summary.strip(), reason)
        n = session.force_fold(len(old))
        session.extra_digest.append(
            f"[摘要 {datetime.now().strftime('%H:%M')}] {summary.strip()[:400]}"
        )
        self._log(f"[context] 兜底收缩：折叠 {n} 轮（reason: {reason}）")
        try:
            self.callbacks.on_context_shrunk(summary.strip(), reason)
        except Exception:  # pragma: no cover - defensive
            pass
        return True

    # ------------------------------------------------------------------
    # L4：经验蒸馏（把本次任务沉淀成长期记忆 / 可复用技能）
    # ------------------------------------------------------------------
    def _distill_experience_async(self, session: Session) -> None:
        """异步 L4 蒸馏：不阻塞用户看到回复。

        产出三条去路，构成能力进化闭环：
          facts/lessons → 知识图谱（``memory_search`` 可召回）
          skill         → ``workspace/skills/<name>/SKILL.md``（自动进向量索引）
        """
        # 窗口改增量：从上次蒸馏的行数快照往后取（留 2 行衔接上下文）。旧实现
        # 固定取最近 25 行，窗口随会话滑动重叠——同一批经验被反复蒸馏，知识
        # 图谱堆满措辞变体（字面去重挡不住语义重复）。无足够新增则跳过本次。
        lines = session.all_digest_lines()
        start = max(0, min(self._last_distill_lines, len(lines)) - 2)
        new_lines = lines[start:]
        if len(new_lines) < 3:
            return          # 没有足够的新增内容，不值得一次蒸馏 LLM 调用
        transcript = "\n".join(new_lines)
        if not transcript.strip():
            return

        def _target() -> None:
            # 先让路：主对话可能刚结束、用户正在等下一轮首 token。
            self._wait_for_idle()
            # 与事实抽取共用同一把锁：长会话里 run() 可能连着触发多次蒸馏，
            # 没有上限会同时跑 N 个 LLM 调用（也各自占一个连接池）。
            if not self._fact_extract_lock.acquire(blocking=False):
                self._log("[distill] 上一次抽取/蒸馏仍在进行，跳过本次")
                return
            try:
                # httpx.Client 非线程安全 → 后台线程必须走独立的抽取客户端。
                # 拿不到就放弃本次，绝不退回 self.llm。
                llm = self._get_extract_llm()
                if llm is None:
                    self._log("[distill] 跳过（无可用专用客户端）")
                    return
                prompt = DISTILL_PROMPT.format(transcript=transcript[:4000])
                raw = self._summarize_via_llm(prompt, system=DISTILL_SYSTEM, llm=llm)
                data = l4_normalize(l4_extract_json(raw))
                # 蒸馏成功后推进行数快照（下轮只蒸新增），失败不推进——
                # 下次会带上这批内容重试。
                self._last_distill_lines = len(session.all_digest_lines())
                observations = list(data["facts"]) + list(data["lessons"])
                if observations:
                    from .tools.memory import _get_long_memory

                    lm = _get_long_memory(self.tools, self.config)
                    lm.graph.create_entity(
                        "Experience", entity_type="distilled", observations=observations
                    )
                    self._log(f"[distill] 写入 {len(observations)} 条经验到知识图谱")
                sk = data.get("skill") or {}
                if sk.get("name") and sk.get("summary"):
                    self.tools.execute(
                        "skill_save",
                        {
                            "name": sk["name"],
                            "description": sk["summary"],
                            "when_to_use": sk.get("when_to_use", ""),
                            "steps": sk.get("steps") or sk["summary"],
                            "pitfalls": "; ".join(data["lessons"]),
                        },
                    )
                    self._log(f"[distill] 固化技能 {sk['name']}")
            except Exception as e:
                self._log(f"[distill] error: {e}")
            finally:
                self._fact_extract_lock.release()

        threading.Thread(target=_target, daemon=True, name="l4-distill").start()

    # ------------------------------------------------------------------
    # 改进 7/8：后台记忆维护（归档 → 历史挖掘 → 记忆整理）
    # ------------------------------------------------------------------
    def _run_memory_maintenance_async(self) -> None:
        """任务终止边触发：主引擎已空闲，后台线程做归档与辅助引擎提炼。

        * 归档是纯文件 IO，永远执行（今天之前的全文 → 骨架汇总 + zip + 删原文件）；
        * 语义提炼 / 记忆整理只在「辅助模型已加载且未被事实抽取占用」时跑，
          不隐式加载模型权重，也不与蒸馏抢同一把锁；
        * 整流程每天至多跑一次完整版（见 memory_v2.run_memory_maintenance）。
        """
        layout = self._memory_layout

        def _target() -> None:
            # 先让路：主对话可能刚结束、用户正在等下一轮首 token。
            self._wait_for_idle()
            if not self._memory_maint_lock.acquire(blocking=False):
                return  # 已有维护线程在跑，本次直接让位
            try:
                # 关键：不能与事实抽取抢 _fact_extract_lock。
                # 两个线程都是「拿不到锁就跳过」，若维护先抢到，先启动的
                # _maybe_extract_facts_async 会静默 return —— 那是既有的长期
                # 事实抽取能力，绝不能被新增的后台维护挤掉。
                # 因此：先让出 0.2s 让事实抽取先入场，再用带超时的 acquire
                # 等它跑完（后台 daemon 线程，完全不阻塞用户看到回复）。
                time.sleep(0.2)
                acquired = self._fact_extract_lock.acquire(timeout=10)
                if not acquired:
                    # 辅助引擎长时间被占用：只做纯 IO 归档，挖掘/整理留给下次。
                    try:
                        archive_past_sessions(layout)
                    except Exception as e:
                        self._log(f"[memory] archive error: {e}")
                    return
                try:
                    aux = getattr(self, "aux_llm", None)
                    if aux is None:
                        # 辅助模型未加载：只归档，绝不隐式加载权重。
                        run_memory_maintenance(layout, None)
                        return

                    def _call(system: str, prompt: str) -> str:
                        return self._summarize_via_llm(prompt, system=system, llm=aux)

                    result = run_memory_maintenance(layout, _call)
                    if result.get("cleaned") or result.get("mined"):
                        self._invalidate_prompt_cache()
                        self._log(
                            f"[memory] 维护完成: archived={result.get('archived')} "
                            f"mined={result.get('mined') or '-'} "
                            f"cleaned={result.get('cleaned')}")
                finally:
                    self._fact_extract_lock.release()
            except Exception as e:
                self._log(f"[memory] maintenance error: {e}")
            finally:
                self._memory_maint_lock.release()

        threading.Thread(target=_target, daemon=True,
                         name="memory-maintenance").start()

    # ------------------------------------------------------------------
    # Tool execution helpers
    # ------------------------------------------------------------------

    def _log_tool_result(self, name: str, res) -> None:
        self._log(f"[tool] {name} -> success={res.success} "
                  f"out_len={len(res.output)} err={(res.error or '')[:120]}")

    def _execute_tool(self, name: str, args: Dict[str, Any]):
        """在后台线程执行工具，主线程轮询取消标志。

        协作式取消原本只在工具**之间**轮询：单个 shell_run 最长 600s、MCP
        tools/call 300s，用户点「终止对话」后要等它跑完才有反应。这里把执行
        放进线程，主线程每 0.1s 检查一次取消标志，命中就立刻返回合成结果
        （后台线程继续跑，由工具自身的超时/kill 逻辑收尾）。
        """
        from .tools import ToolResult

        box: Dict[str, Any] = {}

        def _target() -> None:
            try:
                box["res"] = self.tools.execute(name, args)
            except Exception as e:  # pragma: no cover - defensive
                box["res"] = ToolResult(False, error=f"{type(e).__name__}: {e}")

        t = threading.Thread(target=_target, daemon=True, name=f"tool-{name}")
        t.start()
        while True:
            t.join(0.1)
            if not t.is_alive():
                break
            if self.callbacks.is_cancelled():
                self._log(f"[tool] {name} 执行中被取消，立即返回（后台仍在运行）")
                return ToolResult(False, error="用户取消了对话，工具未执行完")
        return box.get("res") or ToolResult(False, error="工具无返回值")

    def _exec_tool_blocking(self, name: str, args: Dict[str, Any]):
        """纯执行版：无取消轮询线程，供**已在工作线程中**的调用方使用。

        旧实现让 ThreadPoolExecutor 的 worker 再调 ``_execute_tool``——每个
        并行任务实际占 2 个线程（池 worker + 内部轮询线程），且内部那个以
        10Hz 忙等轮询取消。并行 8 个工具 = 16 个线程，其中 8 个纯忙等。
        取消语义由 dispatcher 的工具间检查 + 池本身的协作保证，不再需要
        （也不应该）在每个 worker 里再起一根轮询线程。
        """
        from .tools import ToolResult

        # 并行段由 _run_parallel_tools 启动；若主线程在等待期间被取消，
        # 后台 daemon 线程在此处尽早退出，减少写入已 reset 会话的副作用。
        if self.callbacks.is_cancelled():
            return ToolResult(False, error="已取消")
        try:
            return self.tools.execute(name, args)
        except Exception as e:  # pragma: no cover - defensive
            return ToolResult(False, error=f"{type(e).__name__}: {e}")

    def _run_parallel_tools(
        self,
        jobs: List[Tuple[int, str, Dict[str, Any]]],
        results: Dict[int, Any],
    ) -> None:
        """并发执行一组互相独立的只读工具，结果按索引回填。

        UI 回调（on_tool_start / on_tool_end）由主线程发出；工具自身在 daemon
        线程里跑。取消 / 总时限到达时立即返回，未完成的填「未完成」错误，后台
        daemon 线程继续跑完自行退出——主线程（UI）不再被最长任务（web_scan /
        rag_search）拖住，进程退出也不被非 daemon worker 阻塞。
        """
        from .tools import ToolResult

        n = len(jobs)
        if n == 0:
            return
        # 终止对话场景下的总等待上限：常规并行检索几秒完成。公式必须覆盖
        # 「单工具自身超时」：web_scan 自身 30s 超时，n 个并发全慢时总耗时
        # ≈30s——旧公式 max(30, 10n) 在 n≤3 时恰好卡在 30s 边缘，后完成的
        # 结果会被砍成「未完成」（用户看到的就是扫了几个页面什么都没回来）。
        # 放宽到 max(60, 15n)：仍是有限等待，且大于任何单工具自身超时。
        allow = max(60.0, 15.0 * n)
        deadline = time.monotonic() + allow
        out: List[Any] = [None] * n
        lock = threading.Lock()
        # 并发上限：_MAX_PARALLEL_TOOLS 定义后一直没被引用，模型一次发 N 个
        # rag_search/web_scan 就会起 N 个线程，各自可能触发 embedding/reranker
        # 首次加载（~130MB / ~2.3GB）互相踩。用信号量把实际并发压到上限内，
        # 其余排队（总等待 allow 已按 n 放宽，排队不会提前超时）。
        sem = threading.Semaphore(_MAX_PARALLEL_TOOLS)

        def _run(idx: int, name: str, args: Dict[str, Any]) -> None:
            try:
                res = self._exec_tool_blocking(name, args)
            finally:
                sem.release()
            with lock:
                out[idx] = res

        def _acquire_until_deadline() -> bool:
            """带超时与取消的信号量排队：旧实现裸 acquire() 无限阻塞——前 8 个
            槽位全部挂住（web_scan 黑洞目标）时，deadline 与「终止对话」在
            槽位释放前都不生效（取消延迟可达数十秒）。超时/取消返回 False，
            该任务不启动线程，由既有兜底填「未完成」结果。
            """
            while True:
                if sem.acquire(blocking=False):
                    return True
                if time.monotonic() >= deadline or self.callbacks.is_cancelled():
                    return False
                time.sleep(0.05)

        threads = []
        for idx, name, args in jobs:
            self.callbacks.on_tool_start(name, args)
            if not _acquire_until_deadline():
                continue     # 不启动线程：out[idx] 保持 None → 兜底「未完成」
            t = threading.Thread(target=_run, args=(idx, name, args),
                                name=f"tool-par-{name}-{idx}", daemon=True)
            t.start()
            threads.append(t)

        done_ev = threading.Event()

        def _waiter() -> None:
            for t in threads:
                t.join()
            done_ev.set()

        threading.Thread(target=_waiter, name="tool-par-waiter",
                        daemon=True).start()
        while not done_ev.wait(0.1):
            if time.monotonic() >= deadline or self.callbacks.is_cancelled():
                break

        with lock:
            for idx in range(n):
                if out[idx] is None:
                    out[idx] = ToolResult(
                        False,
                        error="工具在限定时间内未完成（可能因用户取消或超时）")
        for idx, name, args in jobs:
            res = out[idx]
            self._log_tool_result(name, res)
            self.callbacks.on_tool_end(name, res)
            results[idx] = res

    # ------------------------------------------------------------------
    def _system_prompt(self) -> str:
        """Build the full system prompt: base + memory + matched skill.

        Caches the prompt within a single agent turn (self._turn_idx) so that
        _estimate_history_tokens() and _build_messages() don't recompute it
        twice for the same turn.  Memory and skill match results are stable
        within one LLM call cycle.
        """
        if (hasattr(self, "_cached_prompt_turn")
                and self._cached_prompt_turn == self._turn_idx
                and hasattr(self, "_cached_prompt")):
            return self._cached_prompt

        # 改进 3/4：L0 META-SOP 恒定 + L1 全局索引实时加载，紧跟基础系统提示。
        # L0 是静态文本；L1 只在 L2/L3 变更时由 memory_written() 同步重写，
        # 两次变更之间整段 system prompt 逐字不变 —— 与「KV 缓存只认完整前缀」
        # 不冲突（变更后的那一轮重算一次前缀，之后继续稳定命中）。
        base = SYSTEM_PROMPT + "\n\n" + memory_system_block(self._memory_layout)

        # 注意：工作记忆 / 当天事实 / 匹配技能 / 知识库状态等「每轮易变」内容一律
        # 不进系统提示，改由 volatile_context() 注入用户消息（见 kernel/loop.py）。
        # 原因：本模型 KV 缓存是 ArraysCache（不可裁剪），多轮复用只认「完整前缀
        # 匹配」（form ①）；系统提示哪怕只加一个会变的 token，前缀整体对不上 →
        # 每轮退回全量预填充（十几秒）。系统提示恒定后，多轮可稳定命中缓存，而
        # 模型每轮仍能在用户消息里看到这些即时上下文。

        self._cached_prompt_turn = self._turn_idx
        self._cached_prompt = base
        return base

    def _invalidate_prompt_cache(self) -> None:
        """L1 索引被同步更新后，下一轮必须重新拼 system prompt。"""
        self._cached_prompt_turn = -1
        self._cached_prompt = ""

    # ------------------------------------------------------------------
    def volatile_context(self) -> str:
        """每轮易变的即时上下文：知识库状态 + 工作记忆 scratchpad + 当天自动抽取
        事实 + 按当前用户输入匹配到的技能。

        这些原本拼进系统提示，但本模型 KV 缓存（ArraysCache）不可裁剪，多轮只认
        完整前缀匹配；系统提示哪怕加一个会变的 token 都会让前缀整体对不上 → 每轮
        全量预填充。统一移到用户消息后，系统提示 = 恒定 SYSTEM_PROMPT，多轮稳定
        命中缓存；模型每轮仍能在用户消息里看到这些即时上下文（与 anchor_prompt
        回注的 key_info/plan 同源）。
        """
        parts: List[str] = []
        # 知识库状态：30s TTL 缓存，但 reload / 背景索引都会改它，放进用户消息
        # 就不会再污染系统提示前缀。
        try:
            rag_engine = self.tools.get_rag_engine() if self.tools else None
            if rag_engine is not None:
                status = self._rag_status_cached(rag_engine)
                chunks = status.get("chunks_stored", 0)
                sources = status.get("sources", [])
                if chunks > 0:
                    source_list = ", ".join(
                        os.path.basename(s) for s in sources[:8]
                    )
                    if len(sources) > 8:
                        source_list += f" (and {len(sources) - 8} more)"
                    parts.append(
                        f"# Knowledge base\n{chunks} chunks in {len(sources)} "
                        f"file(s): {source_list}. Answer from these via `rag_search`.")
                else:
                    parts.append(
                        "# Knowledge base\nConfigured but empty. "
                        "Use `rag_ingest` to index.")
        except Exception:  # pragma: no cover
            pass
        # 工作记忆 + 当天事实
        try:
            from .tools.memory import _work_memory
            wm = _work_memory(self.config).list()
            compact_wm = {}
            today_facts: List[str] = []
            wm_used = 0
            for k, v in wm.items():
                if k == "__auto_facts__":
                    _, facts = self._parse_daily_facts(str(v))
                    today_facts = facts
                    continue
                if len(compact_wm) >= _MAX_WM_KEYS or wm_used >= _MAX_WM_CHARS:
                    break
                s = str(v)
                s = s[:200] + "…" if len(s) > 200 else s
                compact_wm[k] = s
                wm_used += len(k) + len(s)
            if compact_wm or today_facts:
                sub = []
                if today_facts:
                    capped = []
                    used = 0
                    for f in today_facts:
                        if len(capped) >= _MAX_INJECTED_FACTS or used >= _MAX_FACTS_CHARS:
                            break
                        item = f if len(f) <= _MAX_FACT_CHARS else f[:_MAX_FACT_CHARS] + "…"
                        capped.append(item)
                        used += len(item)
                    dropped = len(today_facts) - len(capped)
                    fact_block = "\n".join(f"- {f}" for f in capped)
                    if dropped > 0:
                        fact_block += f"\n- (…{dropped} more omitted)"
                    sub.append("# Today's facts (auto-extracted)\n" + fact_block)
                if compact_wm:
                    sub.append("# Work memory (scratchpad)\n" +
                               json.dumps(compact_wm, ensure_ascii=False,
                                          separators=(",", ":")))
                parts.append("\n\n".join(sub))
        except Exception:  # pragma: no cover
            pass
        # 匹配到的技能（按当前用户输入）
        try:
            user_text = self._last_user_text
            matched = self.skills.match(user_text) if user_text else None
            if matched:
                # 技能 prompt 是用户自写的原文，长度不可控；截断并提示用
                # skill_load 读全文，避免长技能把系统提示撑大。
                sp = matched.prompt or ""
                if len(sp) > _MAX_SKILL_PROMPT_CHARS:
                    sp = (sp[:_MAX_SKILL_PROMPT_CHARS] +
                          "\n…（已截断，用 skill_load 读取完整技能内容）")
                parts.append(f"# Active skill: {matched.name}\n{sp}")
        except Exception:  # pragma: no cover
            pass
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    def invalidate_rag_status(self) -> None:
        """丢弃知识库状态缓存（同步知识后调用）。"""
        self._rag_status_cache = None

    def reload_rag(self) -> None:
        """知识库同步完成后刷新引擎句柄并失效状态缓存。"""
        self.invalidate_rag_status()
        if self.tools is not None:
            self.tools.reload_rag()

    def _rag_status_cached(self, engine: Any, ttl: float = 30.0) -> Optional[Dict[str, Any]]:
        cache = getattr(self, "_rag_status_cache", None)
        if isinstance(cache, tuple) and cache[1] is engine and (time.time() - cache[0]) < ttl:
            return cache[2]
        try:
            status = engine.status()
        except Exception:
            return None
        self._rag_status_cache = (time.time(), engine, status)
        return status

    @staticmethod
    def _summarize_code(code: str) -> str:
        """Heuristic fallback: infer the purpose of a code snippet.

        Tries comments/docstrings first (models usually emit a purpose comment),
        then the first non-import statement.  Returns a short "做什么" phrase
        WITHOUT dumping the code.  Used only when LLM summarization is
        unavailable.
        """
        code = code.strip()
        if not code:
            return "（空代码）"
        lines = [ln.rstrip() for ln in code.splitlines()]

        # 1) Extract a purpose comment / docstring near the top.
        for ln in lines:
            s = ln.strip()
            if not s:
                continue
            # Module docstring or inline comment describing intent.
            if s.startswith('"""') or s.startswith("'''"):
                doc = s.strip('"\'').strip()
                if doc and not doc.startswith(("import", "from", "#!")):
                    return doc[:80]
            if s.startswith("#"):
                comment = s.lstrip("#").strip()
                # Skip shebang and import markers.
                if comment and not comment.startswith(("!", "encoding", "coding")):
                    return comment[:80]
            # Stop at the first real statement (imports come before purpose comments
            # in many generated snippets, so scan past leading imports).
            if s.startswith(("import ", "from ")):
                continue
            break

        # 2) Infer from the first executable statement.
        for ln in lines:
            s = ln.strip()
            if not s or s.startswith(("#", "import ", "from ")):
                continue
            if s.startswith("def "):
                m = re.match(r"def\s+(\w+)", s)
                return f"定义函数 {m.group(1)}" if m else "定义函数"
            if s.startswith("class "):
                m = re.match(r"class\s+(\w+)", s)
                return f"定义类 {m.group(1)}" if m else "定义类"
            if s.startswith("pip install"):
                return "安装 Python 包：" + s[len("pip install"):].strip()[:60]
            if s.startswith("npm install"):
                return "安装 Node 包：" + s[len("npm install"):].strip()[:60]
            return "执行代码片段：" + s[:60]
        return "执行一段代码"

    @staticmethod
    def _summarize_command(cmd: str) -> str:
        """Infer the purpose of a shell command (no full command dump)."""
        cmd = cmd.strip()
        if not cmd:
            return "（空命令）"
        # Map common commands to a human-readable purpose.
        parts = cmd.split()
        prog = parts[0] if parts else ""
        known = {
            "pip": "安装/管理 Python 包",
            "npm": "安装/管理 Node 包",
            "brew": "安装/管理 macOS 软件",
            "apt": "安装/管理系统软件包",
            "apt-get": "安装/管理系统软件包",
            "yum": "安装/管理系统软件包",
            "git": "Git 版本控制操作",
            "cd": "切换目录",
            "ls": "列出目录内容",
            "cat": "查看文件内容",
            "mkdir": "创建目录",
            "touch": "创建文件",
            "rm": "删除文件/目录",
            "mv": "移动/重命名文件",
            "cp": "复制文件",
            "curl": "发起网络请求",
            "wget": "下载文件",
            "python": "运行 Python 脚本",
            "python3": "运行 Python 脚本",
            "node": "运行 Node 脚本",
            "grep": "搜索文本内容",
            "find": "查找文件",
            "chmod": "修改文件权限",
            "echo": "输出文本",
            "tar": "打包/解压归档",
            "unzip": "解压 zip 文件",
            "ssh": "SSH 远程连接",
            "docker": "Docker 容器操作",
            "kubectl": "Kubernetes 集群操作",
        }
        if prog in known:
            return known[prog]
        # Generic: show the program name only.
        return f"执行命令 {prog}"

    def _describe_code_purpose(self, code: str) -> str:
        """Return a one-line purpose description for code/command.

        Strategy: heuristic FIRST (zero latency — most generated code carries
        a purpose comment/docstring, and known commands map to a fixed phrase).
        Only when the heuristic yields a generic/empty description do we fall
        back to the LLM (a slow 1-3s call that would otherwise delay the
        confirmation dialog on every run).
        """
        is_cmd = self._looks_like_command(code)
        if is_cmd:
            heuristic = self._summarize_command(code)
            # Command mapping is authoritative for known programs; only fall
            # back to LLM for unknown programs.
            if not heuristic.startswith("执行命令 "):
                return heuristic
        else:
            heuristic = self._summarize_code(code)
            # A useful heuristic is a real purpose phrase (comment/docstring/
            # function name), not the generic "执行代码片段：<first line>" or
            # "执行一段代码" fallback.
            if not heuristic.startswith(("执行代码片段", "执行一段代码")):
                return heuristic

        # 不再回退 LLM：确认弹框在交互关键路径上，兜底描述延迟弹框 1-3s
        # （本地后端 aux 还与主模型共用实例、抢 _gen_lock/冲 KV）。启发式给
        # 不出的就显示通用描述——弹框里始终有完整代码/命令原文可看，描述只是
        # 辅助，不值得为此拖慢每一次确认。
        return heuristic

    @staticmethod
    def _looks_like_command(text: str) -> bool:
        """True if text looks like a shell command rather than code."""
        s = text.strip()
        if not s:
            return False
        # Commands are usually a single line starting with a known program,
        # without Python keywords.
        if "\n" in s:
            return False
        first = s.split()[0] if s.split() else ""
        return first in {
            "pip", "npm", "brew", "apt", "apt-get", "yum", "git", "cd", "ls",
            "cat", "mkdir", "touch", "rm", "mv", "cp", "curl", "wget", "python",
            "python3", "node", "grep", "find", "chmod", "echo", "tar", "unzip",
            "ssh", "docker", "kubectl",
        }

    def _confirm_message(self, name: str, args: Dict[str, Any]) -> str:
        """Build a confirmation message.

        ``code_run`` / ``shell_run`` 必须让用户看到**真实**代码/命令预览与
        风险命中原因——只给启发式「用途」会被恶意/诱导性注释蒙蔽（确认弹窗
        失去核对意义）。其他工具展示脱敏后的参数摘要（避免密钥进弹窗）。
        """
        if name == "code_run":
            code = str(args.get("code", "")).strip()
            timeout = args.get("timeout")
            purpose = self._describe_code_purpose(code)
            reason = _risk.risky_reason_code(code, self.config.workspace)
            preview = self._code_preview(code)
            msg = f"【用途】{purpose}\n\n【实际代码】\n{preview}"
            if reason:
                msg += f"\n\n【风险提示】{reason}"
            if timeout is not None:
                msg += f"\n\n超时：{timeout}s"
            return msg + "\n\n是否继续？"
        if name == "shell_run":
            cmd = str(args.get("command", "")).strip()
            timeout = args.get("timeout")
            purpose = self._describe_code_purpose(cmd)
            reason = _risk.risky_reason_command(cmd, self.config.workspace)
            preview = self._code_preview(cmd)
            msg = f"【用途】{purpose}\n\n【实际命令】\n{preview}"
            if reason:
                msg += f"\n\n【风险提示】{reason}"
            if timeout is not None:
                msg += f"\n超时：{timeout}s"
            return msg + "\n\n是否继续？"

        # 其他工具：展示脱敏后的参数摘要（密钥/环境变量不进弹窗）。
        try:
            arg_sum = {str(k): (f"<redacted:{type(v).__name__}>"
                                if any(s in str(k).lower() for s in
                                       ("key", "token", "secret", "password",
                                        "auth", "credential", "env", "cookie"))
                                else str(v)[:120])
                       for k, v in (args or {}).items()}
            pretty = json.dumps(arg_sum, ensure_ascii=False)
        except Exception:
            pretty = "<args>"
        if len(pretty) > 500:
            pretty = pretty[:500] + "…"
        return f"即将调用工具 `{name}`：\n{pretty}\n\n是否继续？"

    @staticmethod
    def _code_preview(src: str) -> str:
        """真实代码/命令的前若干行预览，避免弹窗无限拉长。"""
        lines = src.splitlines()
        head = lines[:12]
        preview = "\n".join(head)
        if len(lines) > 12:
            preview += "\n…"
        if len(preview) > 1200:
            preview = preview[:1200] + "\n…"
        return preview or "（空）"

    def _log(self, line: str) -> None:
        self.callbacks.on_log(line)
