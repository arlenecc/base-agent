"""The agent: reasoning loop, tool dispatch, streaming, confirmation, skills."""
from __future__ import annotations

import json
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
from .kernel import (
    ContextBudget, ExitReason, KernelHooks, KernelLoop, KernelResult, Session,
    l4_extract_json, l4_normalize,
)
from .kernel.compressor import DISTILL_PROMPT, DISTILL_SYSTEM
from .tools.checkpoint import restore_checkpoint
from .atomic_io import atomic_write_text


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


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are base-agent, a local autonomous agent. Complete the user's task by reasoning, planning, and calling tools.

## 交互协议（每轮必须遵守）
回复正文里先输出一个 <summary>…</summary>（≤30 字的物理快照）：上一步结果产生的新信息 + 本次意图。
它会被压缩进长期工作记忆，必须含真实信息，禁止空话。
**若用户需求未完成，必须调用工具**，不要只用文字描述将要做什么。

## 任务推进：先规划，再步步为营
1. **开始前先规划**：接到非平凡任务时，第一件事是 `update_plan(op="create", goal=..., steps=[...])`，
   把目标拆成 2~7 个**可验证**的步骤（每步能用一个工具调用判定完成与否）。一句话就能完成的任务不必规划。
   **规划完立即开工**：建好计划只是起点，不是交付物——紧接着就必须开始执行第一步（调用工具），
   绝不能把计划复述一遍就停下来等用户说「开始/继续」。
2. **每步先反思再动手**：每次工具结果回来后，先在 `<summary>` 里写清「这一步证实/推翻了什么」，
   再决定下一步：是继续下一步、修正当前步骤的方法，还是回头改计划。不要重复同一个失败的调用。
3. **及时更新计划**：完成一步就 `update_plan(op="patch", step_id=..., status="done", note="结论")`；
   进行中标 `doing`；走不通标 `blocked`/`skipped` 并写明原因（blocked 前至少尝试 3 种不同方法）；发现遗漏用 `append`。
   计划每轮都会回注给你的上下文，也是用户唯一能看到进度的地方——保持它准确。
4. **没做完不许收尾**：计划里还有未完成步骤时，不要只给文字总结就结束。全部步骤终态化后才能输出最终答复。

## 能力分层
- `code_run` 是能力核心（可编程，什么都能做）；`shell_run` 是它对系统命令的延伸。组合二者覆盖任意操作。
- `file_read` / `file_write` / `file_patch` 是高频文件操作的快捷方式。
- `web_scan` / `web_execute_js` 用于浏览器。
- `rag_search` / `rag_outline` / `rag_ingest`：本地知识库（可学习、可进化）。
- `skill_semantic_search` / `skill_search` / `skill_load` / `skill_save`：外置经验与能力。
- `memory_search` / `memory_graph` / `work_memory`：长期经验与短期草稿。
- `update_working_checkpoint`：把非显然、后续必须知道的状态写进工作记忆（每轮自动回注）。
- `update_plan`：任务级「目标 + 路径」的规划与进度更新（每轮自动回注；有计划时未完成步骤不得收尾）。
- `start_long_term_update`：任务完成/检查点时触发经验蒸馏。

Rules:
1. Plan, then call tools. After each result, decide the next step.
2. Prefer the least destructive path. Risky/irreversible actions need confirmation.
3. Missing info → `ask_user`, never guess.
4. Recall prior context with `memory_search` before answering about the user or past work. Use `work_memory` only for multi-step notes (intermediate results, decisions, preferences).
5. Complex task → `skill_semantic_search` first; if it finds nothing, try `skill_search` (they index different directories). Then `skill_load` and follow it. No match → proceed.
6. User names a book/document → `rag_outline` for structure, then `rag_search` (with source) for details.
7. Semantic search queries (`memory_search`/`skill_semantic_search`/`rag_search`): use the user's own natural-language phrasing. Never translate, transliterate to pinyin, or reduce to bare keywords — these are vector searches and mangled queries silently return nothing.
8. On tool ERROR — the error is a SIGNAL, not a stop sign. You MUST recover on your own and keep going: (a) read the FULL error and classify it (bad arguments / path missing / permission / missing dependency / syntax / network); (b) retry with a DIFFERENT method that you have NOT tried yet — another tool, different arguments, or a minimal `code_run` diagnostic script to test your hypothesis first; (c) repeat (b) until the step succeeds. Never repeat an identical failing call, never dump the error back to the user, and NEVER stop to ask "should I continue?" — a failure pauses nothing. Only after ≥3 genuinely different methods have failed may you mark the step `blocked` with the reason in its note, and then you MUST move on to the remaining steps. `ask_user` is only for information you truly cannot obtain yourself — never for permission to continue.
9. Reply concisely in plain text when done.

Paths are relative to the workspace. Only `code_run`, `shell_run` and external MCP tools require confirmation."""

# ---------------------------------------------------------------------------
# Token estimation: approximate token count from text length.
# Used for real-time speed display during reasoning streaming (where we don't
# yet have the API's usage figure) and as a fallback when the API's usage
# omits reasoning tokens entirely.
# ---------------------------------------------------------------------------

# CJK 码点 → None 的删除表，供 str.translate 一次性统计 CJK 字符数（C 层实现）。
# 覆盖统一表意文字、扩展 A、CJK 符号/日文假名、全角字符。
_CJK_DELETE_TABLE = {
    cp: None
    for cp in list(range(0x4E00, 0xA000))      # CJK 统一表意文字
    + list(range(0x3400, 0x4DC0))              # CJK 扩展 A
    + list(range(0x3000, 0x3100))              # CJK 符号 + 日文假名
    + list(range(0xFF00, 0xFFF0))              # 全角字符
}


def _estimate_tokens(text: str) -> int:
    """Rough token-count estimate from text.

    Heuristic: ASCII text averages ~4 chars/token; CJK characters are denser
    (~1.5 chars/token). We split the text into CJK and non-CJK runs and apply
    the appropriate ratio. This is only for live speed display and as a
    fallback when the API underreports -- the exact count is not critical.
    """
    if not text:
        return 0
    # Fast path: if the text is pure ASCII (common for English/code/JSON),
    # skip the per-character CJK check entirely — str.isascii() is O(1)-ish
    # (implemented as a single memcmp on CPython).
    if text.isascii():
        return max(1, len(text) // 4)

    # Mixed/CJK text: count CJK characters via str.translate for speed.
    # Building the translation table is O(1) and translate is O(n) in C,
    # much faster than a Python-level per-character loop for long text
    # (旧实现是纯 Python 逐字符循环，长文本下明显偏慢，与上面的注释不符)。
    # translate() 删除的是 CJK 字符，所以剩下的是非 CJK —— 用总长度相减
    # 得到 CJK 数量（比逐字符 Python 循环快一个数量级）。
    cjk = len(text) - len(text.translate(_CJK_DELETE_TABLE))
    other = len(text) - cjk
    # CJK: ~1.5 chars/token; ASCII/other: ~4 chars/token
    return max(1, int(cjk / 1.5 + other / 4.0))


# 系统提示动态注入段的上限。
# 这些段**不在五级压缩的作用范围内**（压缩只作用于 Session 的历史区）：
# 若不加限制会随会话时长单调膨胀，最终挤爆 30k 预算。
_MAX_INJECTED_FACTS = 20        # 当天事实最多注入条数
_MAX_FACTS_CHARS = 2000         # 事实段总字符上限
_MAX_FACT_CHARS = 120           # 单条事实字符上限
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
_AUTO_CONTINUE_MAX = 4
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
            except Exception:
                self._cache = []
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

    def finished(self, session: Session, text: str) -> None:
        agent = self.agent
        # 既有能力：异步抽取事实写知识图谱（不阻塞回复）
        try:
            agent._maybe_extract_facts_async(text)
        except Exception as e:
            agent._log(f"[kernel] fact extraction error: {e}")
        # 新增能力：L4 经验蒸馏（模型显式调用 start_long_term_update，或长任务）
        if agent.kernel.distill_requested or session.current_turn >= agent._distill_min_turns:
            agent._distill_experience_async(session)

    def failed(self, session: Session, err: str) -> None:
        self.agent.callbacks.on_error(err)


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
            raw_turns=int(getattr(config, "kernel_raw_turns", 2) or 2),
            max_digest_lines=int(getattr(config, "kernel_max_digest_lines", 40) or 40),
        )
        # 工具通过 registry.session 读写工作记忆（update_working_checkpoint 等）
        self.tools.session = self.session
        self._last_user_text = ""
        self._distill_min_turns = 6
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

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """清空会话状态（UI 的「清空对话」）。"""
        self._history = []
        self._total_tokens = 0
        self._tool_call_counter = 0
        self.session.reset()
        self.kernel._total_tokens = 0
        self.kernel._last_consolidated_turn = -1
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
        for ev in client.chat_stream(messages, tools=None, temperature=0.2):
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

    def _extract_facts_sync(self, conversation: str) -> None:
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
        self._log(f"USER: {user_input}")

        # skill suggestion (固化) - non-blocking
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
        result: KernelResult = self.kernel.run(user_input, emit_finished=False)
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
        while (segments < _AUTO_CONTINUE_MAX
               and self._auto_continue
               and result.reason in (ExitReason.MAX_TURNS, ExitReason.DONE)
               and not self._cancelled()
               and self._should_auto_continue(result)):
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
        # 连续空回复把重试额度用尽后，内核会以 DONE 退出但 final_text 为空串；
        # 直接返回 "" 会让用户看到一个完全空白的回复，不知道出了什么。
        if not (result.text or "").strip():
            result.text = (
                "（模型连续多次返回空响应，已停止重试。请检查网络/API 状态后重试，"
                "或换个说法再问一次。）"
            )
            self._log("[agent] 空回复：已向用户返回可见提示")
        self._fire_finished_hook(result.text)
        self.callbacks.on_finished()
        return result.text

    def _fire_finished_hook(self, text: str) -> None:
        """整个用户轮次（含全部自动续跑段）结束后，触发一次 finished 钩子。

        事实抽取与 L4 蒸馏都在这个钩子里，是「一次用户输入一次」的语义；
        直接由内核每段触发会把它们放大成最多 4 次。
        """
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
        try:
            summary = self._summarize_via_llm(summary_prompt)
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
        transcript = "\n".join(session.all_digest_lines()[-25:])
        if not transcript.strip():
            return

        def _target() -> None:
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

    def _run_parallel_tools(
        self,
        jobs: List[Tuple[int, str, Dict[str, Any]]],
        results: Dict[int, Any],
    ) -> None:
        """并发执行一组互相独立的只读工具，结果按索引回填。

        UI 回调（on_tool_start / on_tool_end）全部在主线程发出，只有工具本身
        在线程池里跑，因此界面上的工具顺序不受并发影响。
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from .tools import ToolResult

        workers = min(_MAX_PARALLEL_TOOLS, len(jobs))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tool-par") as ex:
            futures = {}
            for idx, name, args in jobs:
                self.callbacks.on_tool_start(name, args)
                futures[ex.submit(self._execute_tool, name, args)] = (idx, name)
            for fut in as_completed(futures):
                idx, name = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:  # pragma: no cover - defensive
                    res = ToolResult(False, error=f"{type(e).__name__}: {e}")
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

        base = SYSTEM_PROMPT

        try:
            from .tools.memory import _work_memory

            # Work memory: inject only the keys the model explicitly set
            # (its scratchpad), PLUS today's auto-extracted facts.  Long-term
            # memory is NOT injected here — it is recalled on demand via the
            # memory_search tool.
            wm = _work_memory(self.config).list()
            compact_wm = {}
            today_facts: List[str] = []
            for k, v in wm.items():
                if k == "__auto_facts__":
                    # 短期记忆里的自动抽取事实：只注入当天桶（历史已在长期
                    # 记忆 graph 中，靠 memory_search 检索）。
                    _, facts = self._parse_daily_facts(str(v))
                    today_facts = facts
                    continue
                s = str(v)
                compact_wm[k] = s[:200] + "…" if len(s) > 200 else s
            if compact_wm or today_facts:
                parts = []
                if today_facts:
                    # 当天事实用列表形式注入，更易读（而非嵌套 JSON）。
                    # 必须加条数/长度上限：事实每轮累积且系统提示不在上下文
                    # 收缩范围内，放任不管会随会话时长单调膨胀直至挤爆窗口。
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
                    parts.append("# Today's facts (auto-extracted)\n" + fact_block)
                if compact_wm:
                    parts.append("# Work memory (scratchpad)\n" +
                                 json.dumps(compact_wm, ensure_ascii=False, separators=(",", ":")))
                base += "\n\n" + "\n\n".join(parts)
        except Exception:  # pragma: no cover
            pass
        # inject matched skill prompt
        try:
            # 技能匹配的是用户意图：用本轮用户输入（历史所有权已在 Session，
            # 这里不再扫描它）。
            user_text = self._last_user_text
            matched = self.skills.match(user_text) if user_text else None
            if matched:
                # 技能 prompt 是用户自写的原文，长度不可控；截断并提示用
                # skill_load 读全文，避免长技能把系统提示撑大。
                sp = matched.prompt or ""
                if len(sp) > _MAX_SKILL_PROMPT_CHARS:
                    sp = (sp[:_MAX_SKILL_PROMPT_CHARS] +
                          "\n…（已截断，用 skill_load 读取完整技能内容）")
                base += f"\n\n# Active skill: {matched.name}\n{sp}"
        except Exception:  # pragma: no cover
            pass

        # Dynamically inject knowledge base status so the agent knows RAG
        # tools are available and whether the KB has indexed content.  Without
        # this the agent has no way to discover that a local KB exists and
        # will never proactively call rag_search / rag_status.
        try:
            rag_engine = self.tools.get_rag_engine() if self.tools else None
            if rag_engine is not None:
                # status() 会连 LanceDB 并可能触发 lancedb 的惰性导入（冷启动
                # 数秒）。它每轮都被调用一次，缓存起来——知识库状态本就以分钟级
                # 变化，同步后由 reload_rag() 显式失效。
                status = self._rag_status_cached(rag_engine)
                chunks = status.get("chunks_stored", 0)
                sources = status.get("sources", [])
                if chunks > 0:
                    # 仅注入文档名（不含路径），最多 8 个，超出省略。
                    source_list = ", ".join(
                        os.path.basename(s) for s in sources[:8]
                    )
                    if len(sources) > 8:
                        source_list += f" (and {len(sources) - 8} more)"
                    base += (
                        f"\n\n# Knowledge base\n"
                        f"{chunks} chunks in {len(sources)} file(s): {source_list}. "
                        f"Answer from these via `rag_search`."
                    )
                else:
                    base += (
                        f"\n\n# Knowledge base\n"
                        f"Configured but empty. Use `rag_ingest` to index."
                    )
        except Exception:  # pragma: no cover
            pass

        self._cached_prompt_turn = self._turn_idx
        self._cached_prompt = base
        return base

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

        # Fallback: ask the LLM (dedicated client, with a short timeout).
        try:
            llm = self._get_extract_llm()
            if llm is not None and llm is not self.llm:
                prompt = (
                    "用 ≤20 字中文说明以下代码/命令做什么，只输出这一句，"
                    "不要贴代码、不要以「这段代码」开头：\n\n"
                    + code[:2000]
                )
                # chat_stream 产出的是 StreamEvent 数据类，不是 dict。旧代码写
                # `ev.get(...)` 会抛 AttributeError 并被下面的 except 吞掉，
                # 导致 LLM 兜底永远是死代码、只能退回通用描述。这里按属性访问。
                answer_parts = []
                for ev in llm.chat_stream([{"role": "user", "content": prompt}]):
                    if ev.type == "content":
                        answer_parts.append(ev.content)
                    elif ev.type == "done":
                        break
                answer = "".join(answer_parts).strip()
                if answer:
                    return answer[:80]
        except Exception:
            pass
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
        """Build a concise confirmation message (purpose description, no code).

        ``code_run`` / ``shell_run`` carry full code/commands — dumping them
        into the dialog makes it unreadably long and pushes the Yes/No buttons
        off-screen.  We show a short *purpose* description instead.
        """
        if name == "code_run":
            code = str(args.get("code", "")).strip()
            timeout = args.get("timeout")
            purpose = self._describe_code_purpose(code)
            msg = f"即将执行代码：{purpose}"
            if timeout is not None:
                msg += f"\n超时：{timeout}s"
            return msg + "\n\n是否继续？"
        if name == "shell_run":
            cmd = str(args.get("command", "")).strip()
            timeout = args.get("timeout")
            purpose = self._describe_code_purpose(cmd)
            msg = f"即将执行命令：{purpose}"
            if timeout is not None:
                msg += f"\n超时：{timeout}s"
            return msg + "\n\n是否继续？"

        # Other tools: show a compact argument summary (truncated).
        try:
            pretty = json.dumps(args, ensure_ascii=False)
        except Exception:
            pretty = str(args)
        if len(pretty) > 500:
            pretty = pretty[:500] + "…"
        return f"即将调用工具 `{name}`：\n{pretty}\n\n是否继续？"

    def _log(self, line: str) -> None:
        self.callbacks.on_log(line)
