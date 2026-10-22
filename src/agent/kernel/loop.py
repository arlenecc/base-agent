"""极简三态自动机：内核只做控制流。

    THINK ──has_calls──► ACT ──continue──► THINK
      │                   │
      │ no_calls          ├─ consolidate ─► CONSOLIDATE ──► THINK
      ▼                   └─ interrupt/done ─► EXIT
     EXIT

它不拥有工具实现、不拥有记忆、不拥有 RAG —— 那些都是注入的能力：
``dispatcher``（能力）、``session``（历史）、``hooks``（经验沉淀）。
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .budget import ContextBudget, estimate_tokens
from .compressor import has_summary_tag, l0_capture, l1_trim, l2_summary
from .outcome import (
    ExitReason, KernelState, LLMResponse, StepOutcome, ToolCall, next_state,
)
from .session import Session

logger = logging.getLogger(__name__)


# 空响应 / 流中断的重试上限
_MAX_EMPTY_RETRY = 3
_LLM_RETRY_BASE_DELAY = 1.5
_MAX_LLM_RETRIES = 2

# L0 硬上限：工具结果进上下文前先兜住病态长度（工具自身上限更高，例如
# file_read 允许 200K 字符）。L1 之后还会按 RAW 区余额再裁一刀。
_L0_HARD_CAP = 20_000

# 流中断 / 截断标记（对齐 GenericAgent 的 do_no_tool）
_TRUNCATION_MARKS = ("[!!! 流异常中断", "!!!Error:", "max_tokens !!!]")
_CODE_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_]*\n[\s\S]{50,}?```")
_TAG_STRIP_RE = re.compile(r"<(thinking|think|summary)>[\s\S]*?</\1>", re.I)

# ---- 规划门 / 反思门（防止「跑丢」与「半路停」）---------------------------
# 规划门：任务开始后、还没干过实际活、且还没有计划时，每轮提醒先规划；一旦
# 开始干实际活就停止（一句话任务如「读 a.txt」不被强制规划）。多步任务真正
# 要执行前，ACT 态的「规划前置门」还会硬拦一次，强制先把计划写进 Session。
_PLAN_GATE_TURNS = 3  # 历史常量/测试断言用；规划门现已持续化（见 THINK 态）
# 补建提示上限：已开始执行但 plan 仍空时，最多提醒这么多次把计划补上（兜底，
# 防止「首轮误执行」导致全程无 plan 可跟踪）。
_PLAN_NUDGE_MAX = 3
# 多步任务信号：用户输入命中这些模式时视为应强制先规划（硬拦截不豁免）。
_MULTISTEP_HINT = re.compile(
    r"(步骤|计划|规划|先做|先要|然后|接着|依次|分别|逐个|多个|多项|"
    r"第[一二三四五六七八九十\d]+步|^\s*\d+[\.、\)]|一、|二、|首先|其次|最后|"
    r"以下任务|分别做|依次做|分步)",
    re.I | re.M,
)
# 多步**动词**：不含上面那些连接词也显然不是一步能做完的动作。
# 「帮我重构 foo.py」「批量转换这些图片」「把项目部署到测试服」都不命中
# _MULTISTEP_HINT，旧实现的 simple 豁免会把它们整条放行 → 无计划、无监督地
# 跑完，反思门因 plan_open_steps() 恒空而完全失效。
_MULTISTEP_VERB_RE = re.compile(
    r"(重构|重写|批量|全部|所有|每个|一系列|整套|完整|"
    r"部署|迁移|搭建|改造|升级|替换|整合|拆分|合并|"
    r"整理|归档|备份|还原|恢复|排查|定位并|分析并|"
    r"制定|实现|开发|配置|初始化|生成报告|写一份|做一份)",
    re.I,
)
# 多步任务却只在文本里「规划 + 声称开始」、既没把计划写入执行计划也没真正
# 动手就想收尾时，反思门给出的强制纠正提示。
_PLAN_FINISH_HINT = (
    "[System] 你只是在回复文本里写出了计划、并声称「开始执行第一个任务」，"
    "但既没有调用 update_plan 把计划写入执行计划（Session.working['plan'] 为空），"
    "也没有发出任何实际工具调用——这等于什么都没做。\n"
    "请立即：1) 用 update_plan(op='create', goal=..., steps=[...]) 把计划写入执行计划；"
    "2) 紧接着发出第一步的实际工具调用并持续推进，直到计划完成。\n"
    "禁止只在文本里描述计划——计划必须写进执行计划，每一步都必须真正调用工具。"
)
# 模型想收尾但计划里还有未完成步骤时的最大提醒次数。超过后尊重模型的决定，
# 但在回复末尾附上未完成清单——用户必须知道任务其实没做完。
_FINISH_NUDGE_MAX = 2
# 发生过失败时大幅放宽预算：模型在失败后想用文字收尾 = 「停车等用户打继续」，
# 正是要防的行为。一次提醒的成本就是一轮 LLM，而停车的代价是用户被迫手动
# 介入，因此给到 8 次而不是 4 次。每次提醒都会附上最近一次错误**和已经试过
# 且失败的方案清单**，所以第 n 次提醒不是第 1 次的复读。
_FINISH_NUDGE_FAIL_MAX = 8
# 「计划建好了但一步都还没干」时的预算。模型建完计划后最常见的行为就是把
# 计划念一遍然后收尾——此时没有失败可引用，但「本段一次实际工具调用都没发
# 生」本身就是最强的不该停车信号。默认 2 次会让它第 3 次就放行，正是用户
# 看到的「规划完不执行，得手动说继续」。
_FINISH_NUDGE_PLAN_MAX = 6
# 失败调用签名保留数（识别「原样重试」：同名工具 + 相同参数再次失败）
_MAX_FAIL_SIGS = 8
# 把步骤标记 blocked 前至少要有多少次失败尝试（步骤边界 = 上次 update_plan）。
# 3 次 = 要求「至少 3 种不同方法都失败」才允许放弃一步，与系统提示 rule 8
# 和反思门提醒里的措辞保持一致。
_MIN_ATTEMPTS_BEFORE_BLOCK = 3
# 「未充分尝试就标记 blocked」的提醒上限（防无限循环）
_BLOCK_NUDGE_MAX = 2
# 重复调用检测：同名 + 同参数的调用连续出现到第 (N+1) 次时开始提醒；达到
# 上限则直接结束本轮。注意这里管的是**成功**调用——失败调用由 _fail_sigs 负责。
_REPEAT_CALL_WARN = 2      # 第 3 次相同调用 → 提醒换动作
_REPEAT_CALL_LIMIT = 4     # 第 5 次 → 收尾，不再烧轮次
# 失败调用「原样重试」的熔断：与成功调用的 _REPEAT_CALL_LIMIT 对等。
# 模型对同一无效参数反复重试（如 update_working_checkpoint({}) 缺 key_info）
# 只提示不收敛会烧满 max_iterations。连续相同失败签名达到上限 → 强制结束本轮。
# 设为 3：允许「失败→失败→改而回复」的正常模式完整跑完（第三轮才收尾），
# 而对「持续相同失败」的问候场景在第 4 轮熔断（远优于烧满 max_iterations）。
_FAIL_REPEAT_LIMIT = 3      # 连续 3 次相同失败签名（第 4 次出现）→ 收尾

# 「模型在向用户追问必要信息」的信号。只在任务未起步时作为放行依据，
# 因此刻意收得比反思门的禁止清单更窄：只认真正索取信息的句式，不认
# 「请问需要我继续吗」这类停车话术（后者由反思门继续拦）。
_ASK_USER_RE = re.compile(
    r"(请(提供|告诉|告知|说明|补充|指定|确认)|需要(你|您)(提供|告诉|确认|补充)|"
    r"能否(提供|告诉|说明)|麻烦(提供|告诉)|"
    r"(哪个|哪些)(文件|目录|路径|项目|模块|分支|环境)|"
    r"想?要我(先)?(做|处理|修改|改)|希望我(做|先做)|你希望|你想要)",
    re.I,
)

_PLAN_GATE_HINT = (
    "\n[System] 开始执行前，先用 update_plan(op=\"create\", goal=..., steps=[...]) "
    "把目标和路径规划成 2~7 个可验证的步骤，并写入执行计划（Session.working['plan']）；"
    "**计划建好之前不要先发实际工具调用**——先把目标与步骤登记为执行计划，"
    "再**紧接着在同一轮或下一轮开始执行第一步**（计划建好不等于任务开始，"
    "不要停下来问用户是否开始/是否继续）；然后从第一步开始，每完成一步就用 "
    "update_plan(op=\"patch\", step_id=..., status=\"done\", note=\"结论\") 更新进度。"
)


class KernelHooks:
    """经验沉淀回调。默认 no-op —— 内核本身不做任何沉淀。"""

    def consolidated(self, session: Session, reason: str) -> bool:
        """折叠/沉淀回调。返回 True 表示上下文真的变小了。"""
    def finished(self, session: Session, text: str) -> None: ...
    def failed(self, session: Session, err: str) -> None: ...


class KernelResult:
    """一次 run() 的产物。"""

    def __init__(self, text: str = "", reason: ExitReason = ExitReason.DONE,
                 turns: int = 0, distilled: bool = False):
        self.text = text
        self.reason = reason
        self.turns = turns
        self.distilled = distilled

    def __repr__(self) -> str:
        return f"<KernelResult {self.reason.value} turns={self.turns} len={len(self.text)}>"


class KernelLoop:
    """三态自动机主循环。"""

    def __init__(
        self,
        llm: Any,
        dispatcher: Any,                      # dispatch(List[ToolCall]) -> List[ToolResult]
        callbacks: Any,
        config: Any,
        system_prompt: Callable[[], str],
        volatile_context: Callable[[], str],
        session: Optional[Session] = None,
        hooks: Optional[KernelHooks] = None,
        log: Optional[Callable[[str], None]] = None,
        max_turns: int = 0,
    ):
        # llm 允许是「对象」或「返回对象的 callable」。UI 会整体替换
        # agent.llm（切换模型/配置），用 callable 才能拿到最新的客户端。
        self.llm = llm
        self.dispatcher = dispatcher
        self.callbacks = callbacks
        self.config = config
        self.system_prompt = system_prompt
        self.volatile_context = volatile_context
        self.hooks = hooks or KernelHooks()
        self.log = log or (lambda *_a, **_k: None)
        self.session = session or Session(
            budget=ContextBudget(
                int(getattr(config, "kernel_max_context_tokens", 30_000) or 30_000)
            ),
            raw_turns=int(getattr(config, "kernel_raw_turns", 20) or 20),
            max_digest_lines=int(getattr(config, "kernel_max_digest_lines", 40) or 40),
        )
        # 必须与 Session 共用同一个预算对象，否则内核记的账和 Session 折叠时
        # 读的余额是两本账（L1/L3 会各按自己的数裁决）。
        self.budget = self.session.budget
        self.consolidate_every = max(1, int(getattr(config, "kernel_consolidate_every", 4) or 4))
        self.max_tool_chars = int(getattr(config, "kernel_max_tool_chars", 6_000) or 6_000)
        self.max_turns = max(1, int(max_turns or getattr(config, "max_iterations", 15) or 15))

        self.state: KernelState = KernelState.THINK
        self.distill_requested = False
        self._schemas: Optional[List[Dict[str, Any]]] = None
        self._schema_tokens = 0
        self._empty_retry = 0
        self._shrink_guard = 0
        self._reask_guard = 0
        self._turns_used = 0
        self._last_consolidated_turn = -1
        self._total_tokens = 0
        self._finish_nudge = 0
        # ---- 失败恢复记账（每次 run() 重置）----
        self._fail_streak = 0        # 连续失败的工具调用数（任一批全成功即清零）
        self._failed_total = 0       # 本次 run 失败总数；>0 即放宽反思门预算
        self._last_fail: Optional[Tuple[str, str]] = None   # 最近一次 (工具名, 错误)
        self._fail_sigs: List[str] = []          # 近期失败调用签名（原样重试检测）
        self._step_fails = 0         # 上次 update_plan 之后的失败数 ≈ 当前步骤的尝试次数
        self._block_nudge = 0        # 「未充分尝试就 blocked」已提醒次数
        # 本段是否真的动过手（调用过非 update_plan 的工具）。建完计划一次实际
        # 工具调用都没有就想收尾 = 「只规划不执行」，反思门必须多拦几次。
        self._did_work = False
        # 补建提示计数：已开始执行但 plan 仍空时，最多提醒 _PLAN_NUDGE_MAX 次。
        self._plan_nudged = 0
        self._plan_finish_nudge = 0
        # 重复调用检测（成功调用也会空转，见 ACT 态同名注释）；run() 会重置。
        self._repeat_streak = 0
        self._last_call_sigs: List[str] = []
        # 失败调用「原样重试」熔断计数（连续相同失败签名次数）；run() 会重置。
        self._fail_repeat_streak = 0
        # 同一工具连续失败计数（不论参数是否相同）；run() 会重置。用于问候
        # 场景快速收尾（弱模型死磕同一工具只换参数）。
        self._same_tool_fail_streak = 0
        self._same_tool_fail_name: Optional[str] = None
        # 上层（Agent 自动续跑）注入的「上一段失败上下文」，run() 开始时消费一次。
        self._seed_failures: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # 注入的能力：工具 schema
    # ------------------------------------------------------------------
    def set_schemas(self, schemas: List[Dict[str, Any]]) -> None:
        self._schemas = list(schemas or [])
        self._schema_tokens = estimate_tokens(json.dumps(self._schemas, ensure_ascii=False)) if self._schemas else 0
        self.budget.set_used("tools", self._schema_tokens)

    def schemas(self) -> List[Dict[str, Any]]:
        if self._schemas is None:
            try:
                self.set_schemas(self.dispatcher.schemas())
            except Exception as e:
                # 静默降级成「没有任何工具」会让模型在无法调用工具的状态下空跑，
                # 用户只看到反复的文字回复却不知道原因。至少留一条日志。
                self.log(f"[kernel] 工具 schema 获取失败，本轮将无工具可用: {e}")
                logger.warning("kernel: tool schema unavailable: %s", e)
                self.set_schemas([])
        self.budget.set_used("tools", self._schema_tokens)
        return self._schemas or []

    def _reset_retry_guards(self) -> None:
        """一轮被真正提交后，清掉「空回复 / 只吐代码块」的重试计数。

        只应在 ``session.finish_turn`` 之后调用：被回滚的重试轮不算提交，
        提前清零会让重试永远触不到上限（空回复无限循环）。
        """
        self._empty_retry = 0
        self._reask_guard = 0

    def _emit_ui(self, attr: str, *args) -> None:
        """安全调用 UI 回调。

        UI 回调抛异常不能把整轮变成 ExitReason.ERROR：那时用户只看到一句
        「LLM error」，完全看不出是界面侧的问题（on_content / on_reasoning /
        on_token_speed 之前都没包，同文件的 on_token_speed 收尾处却包了）。
        这里统一降级为日志。
        """
        fn = getattr(self.callbacks, attr, None)
        if not callable(fn):
            return
        try:
            fn(*args)
        except Exception as e:
            logger.warning("kernel: UI callback %s failed: %s", attr, e)

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def run(self, user_input: str, resume: bool = False,
            emit_finished: bool = True) -> KernelResult:
        """执行一段内核循环。

        ``emit_finished=False`` 供「自动续跑」使用：续跑段是同一个用户轮次的
        一部分，此时不能触发 ``hooks.finished`` —— 它会做事实抽取与 L4 蒸馏，
        每段都触发意味着单次用户输入最多跑 4 次冗余的后台 LLM 调用，并向知识
        图谱重复写入多条近似实体。只有最后一段才允许触发。
        """
        session = self.session
        self.state = KernelState.THINK
        self._empty_retry = 0
        self._shrink_guard = 0
        self._reask_guard = 0
        self._finish_nudge = 0
        self._fail_streak = 0
        self._failed_total = 0
        self._last_fail = None
        self._fail_sigs = []
        self._step_fails = 0
        self._block_nudge = 0
        self._did_work = False
        self._plan_nudged = 0
        self._plan_finish_nudge = 0
        # 重复调用检测（成功调用同样空转，见 ACT 态的重复调用检测注释）
        self._repeat_streak = 0
        self._last_call_sigs: List[str] = []
        self._fail_repeat_streak = 0
        self._same_tool_fail_streak = 0
        self._same_tool_fail_name = None
        # 续跑段（Agent 自动续跑）跳过「规划前置门」：续跑由系统提示驱动，
        # 其 user_input 含「步骤 / 继续执行」等多步信号，不应被当成用户发起的
        # 多步任务而强制先建计划——否则会把续跑段的正常收尾也卡掉。
        self._resume = resume
        # 续跑段：把上一段的失败上下文带进来（见 seed_failures / failure_snapshot）。
        # 不这么做的话，新段 _failed_total=0 → 反思门退回默认 2 次预算，且提醒里
        # **不再有任何错误信息** —— 模型在续跑段完全不知道上一段卡在哪，只能
        # 瞎试或再次停车。这正是「失败后就停住、等人提醒」在多段续跑下的放大器。
        seed = self._seed_failures
        self._seed_failures = None
        if seed:
            self._failed_total = int(seed.get("failed_total", 0) or 0)
            self._last_fail = seed.get("last_fail") or None
            self._fail_sigs = list(seed.get("fail_sigs") or [])[-_MAX_FAIL_SIGS:]
            self._fail_streak = int(seed.get("fail_streak", 0) or 0)
            # 防死循环计数跨段续上（见 failure_snapshot 注释）
            self._finish_nudge = int(seed.get("finish_nudge", 0) or 0)
            self._plan_finish_nudge = int(seed.get("plan_finish_nudge", 0) or 0)
            self._block_nudge = int(seed.get("block_nudge", 0) or 0)
            self._empty_retry = int(seed.get("empty_retry", 0) or 0)
            self._reask_guard = int(seed.get("reask_guard", 0) or 0)
            self._plan_nudged = int(seed.get("plan_nudged", 0) or 0)
            self._fail_repeat_streak = int(seed.get("fail_repeat_streak", 0) or 0)
            self._same_tool_fail_streak = int(
                seed.get("same_tool_fail_streak", 0) or 0)
            self._same_tool_fail_name = seed.get("same_tool_fail_name")
            self._did_work = bool(seed.get("did_work", False))
        # max_turns 是「本次 run() 的轮次上限」，不是「整个会话的累计轮次」。
        # Session.turn_index 跨 run() 单调递增（它就是历史编号），拿它当循环
        # 上界会让 Agent 在累计 N 轮之后彻底不再调用 LLM。
        self._turns_used = 0
        self.distill_requested = False

        pending = user_input
        final_text = ""
        exit_reason = ExitReason.MAX_TURNS

        while self._turns_used < self.max_turns:
            if self._cancelled():
                self.log("[kernel] 收到取消请求，终止")
                exit_reason = ExitReason.CANCELLED
                break

            # ---- CONSOLIDATE ----
            if self._needs_consolidate():
                self.state = KernelState.CONSOLIDATE
                self._last_consolidated_turn = session.current_turn
                over = self.budget.raw_over
                reason = "预算告罄" if over else f"每 {self.consolidate_every} 轮"
                # 周期性沉淀也必须真的折叠（旧代码这里传 0，force_fold 变成空操作），
                # 否则 turns 只会增长、extra_digest 永远为空。
                #
                # 折叠量必须匹配原文窗口：consolidate_every 默认 4（每 4 轮
                # 触发一次），旧实现恒折 1 轮 → 净增长 +3 轮/4 轮，turns
                # （每条都带 raw_content / prompt / tool_results，推理模型
                # 单轮几十 KB）无界增长，与「L2+L3 历史 O(1)」的设计直接
                # 矛盾。现在一律压到 raw_turns + 1（原文窗口 + 当前轮）。
                keep = session.raw_turns + 1
                folded = session.force_fold(max(1, len(session.turns) - keep))
                if folded:
                    # 历史被改写了（旧轮次摘掉、换成摘要行）。不必整体丢弃 KV
                    # 缓存——_stream_mlx 的「中途分叉」复用逻辑(form ③)会在下一
                    # 轮按 token 比对裁到公共前缀(系统提示 + 幸存早期轮次)，只
                    # 重算被折叠的那一小段；整体丢弃反而让折叠后整轮退回全量
                    # 预填充(十几秒)。这里故意不再调用 _invalidate_kv()。
                    # 注意：Qwen3.5 的线性注意力层用 ArraysCache（不可裁剪），
                    # form ③ 的裁剪会失败、退化为全量重建——折叠/回滚后的那一
                    # 轮仍会慢（llm.py 有 info 级日志），属预期而非回归。
                    pass
                self.log(f"[kernel] CONSOLIDATE ({reason}) folded={folded} {session.stats()}")
                try:
                    self.hooks.consolidated(session, reason)
                except Exception as e:
                    self.log(f"[kernel] consolidate hook error: {e}")
                # 只折一轮就回 THINK：`raw` 的账在 build_messages 里刷新，这里
                # 旧实现用 continue 循环折叠——但 _used["raw"] 是上一轮的旧值，
                # 会连折 3×3=9 轮才罢休（其中 6 轮是白折的）。
                # 真的没东西可折了才交上层兜底收缩。
                if over and folded == 0 and self._shrink_guard < 3:
                    self._shrink_guard += 1
                    try:
                        self.hooks.consolidated(session, "fold-exhausted")
                    except Exception:
                        pass
                self.state = next_state(KernelState.CONSOLIDATE, "done")

            # ---- THINK ----
            self.state = KernelState.THINK
            # 规划门：还没干过实际活、且还没有计划时，持续提醒先规划再动手。
            # 一旦开始干实际活（调过非 update_plan 的工具）就停止——一句话任务
            # （「读一下 a.txt」）不会被强制规划。多步任务真正要执行前，ACT 态
            # 的「规划前置门」还会硬拦一次（见下）。
            # 简单输入（一句问候 / 单个小问题）豁免：不要求先建执行计划。
            # 否则「hi」也会被逼着 create 一个『询问用户需求』的一步计划，
            # 而「询问」根本不是可执行的工具动作——模型随即卡在「立即开始第
            # 1 步」上，表现为反复发无意义的工具调用。
            if (session.get_plan() is None and not self._did_work
                    and not self._is_simple_input(user_input)):
                pending = (pending or "") + _PLAN_GATE_HINT
            user_text = session.anchor_prompt(extra=pending)
            # 每轮易变上下文（工作记忆 / 当天事实 / 匹配技能）从系统提示移到用户
            # 消息：本模型 KV 缓存（ArraysCache）不可裁剪，多轮只认完整前缀匹配，
            # 系统提示每轮漂移会令缓存整体失效。移入用户消息后系统提示恒定，多轮
            # 稳定命中缓存；模型每轮仍可见这些即时上下文（见 agent.volatile_context）。
            vc = self.volatile_context()
            if vc:
                user_text = user_text + "\n\n" + vc
            rec = session.start_turn(user_text)
            # rollback 路径不减计数：重试由各自的 guard 兜底，这里只保证终止性。
            self._turns_used += 1
            messages = session.build_messages(self.system_prompt(), user_text)
            self.schemas()

            try:
                resp = self._ask(messages)
            except Exception as e:
                msg = str(e)
                # 上下文超长是可恢复的：交给上层兜底收缩后重试同一轮。
                # 不重试 = 用户直接看到硬错误（旧实现的被动收缩正是为此存在）。
                if self._is_context_error(msg) and self._shrink_guard < 3:
                    self._shrink_guard += 1
                    self._rollback_turn(session)
                    # 本轮被摘掉 → 缓存末尾对应段从历史消失。同样不必整体丢弃：
                    # 下一轮 form ③ 会按 token 比对裁到新末尾(新 prompt 是旧缓存
                    # 的前缀)，只重算被回滚掉的那一轮。保留缓存优于全量重建。
                    shrunk = False
                    try:
                        shrunk = bool(self.hooks.consolidated(session, "fold-exhausted"))
                    except Exception:
                        pass
                    if shrunk:
                        self.log(f"[kernel] 上下文超长，兜底收缩后重试 ({self._shrink_guard})")
                        # 重置 pending：否则重试循环会在下一轮把规划门提示
                        # **再次**追加到已含提示的 pending 上（最多 3 份相同
                        # 段落重复进 prompt，纯浪费 token）。
                        pending = user_input
                        continue
                    # 没东西可压了 —— 再发一次同样的超长 prompt 没有意义。
                self.log(f"[kernel] LLM error: {e}")
                try:
                    self.hooks.failed(session, str(e))
                except Exception:
                    pass
                exit_reason = ExitReason.ERROR
                break

            if resp is None:                      # 流式期间被取消
                # 这一轮只有 user 没有 assistant，留着会让历史出现两条相邻
                # user 消息；取消即丢弃，回退掉。
                self._rollback_turn(session)
                exit_reason = ExitReason.CANCELLED
                break

            if not resp.has_calls:
                # 反思门：模型想收尾，但计划里还有未完成步骤 —— 不能就这么放它
                # 走。这正是「半路任务跑丢 / 直接就停了」的主因：模型自认为做完
                # 了，实际步骤没走完。提醒有限次，超过后尊重决定但在回复里附上
                # 未完成清单（用户必须知道任务并没完成）。
                open_steps = session.plan_open_steps()
                # 任务尚未起步（既没执行计划、也没干过实际活）时，若模型是在向
                # 用户追问必要信息（目标文件 / 路径 / 范围），必须放行：把提问
                # 当作最终答复收下并结束本轮，用户才回答得了。
                #
                # 旧行为是无差别 rollback + 提醒「禁止问用户是否继续」：模型的
                # 提问被整轮丢弃，用户一个字都看不到，内核又拿同一份上下文重问
                # 一遍——表现为「一遍又一遍提示要用户输入任务，却完全不给输入
                # 机会，然后循环起来」。任务还没开工时强推工具调用只会让它原地
                # 打转，因为缺少的正是用户那句回答。
                #
                # 限定「未起步」是刻意的：一旦计划建好或已经干过活，模型就不能
                # 再用「请问需要我继续吗」当停车借口（那正是反思门要防的行为）。
                if (session.get_plan() is None and not self._did_work
                        and self._looks_like_asking_user(resp.content)):
                    final_text = resp.content
                    session.finish_turn(rec, resp.content, [], [],
                                        l2_summary(resp.content),
                                        raw_content=resp.raw_content)
                    self.log("[kernel] 任务未起步且模型在向用户追问必要信息 → 收下提问，交给用户回答")
                    exit_reason = ExitReason.DONE
                    break
                # 多步任务却连执行计划都没写、也没真正动手就想收尾 —— 这正是
                # 「思考里显示计划 + 声称开始第一个任务，然后就停住、什么也没干」
                # 的失败模式。先拦：必须先把计划写入执行计划并立即发第一步工具。
                if (session.get_plan() is None
                        and (_MULTISTEP_HINT.search(user_input or "")
                             or _MULTISTEP_VERB_RE.search(user_input or ""))
                        and not self._did_work
                        and self._plan_finish_nudge < _FINISH_NUDGE_PLAN_MAX
                        and not self._resume):
                    self._plan_finish_nudge += 1
                    self._rollback_turn(session)
                    pending = _PLAN_FINISH_HINT
                    self.log("[kernel] 反思门：多步任务未写计划也未动手，强制要求先 create 并开工")
                    continue
                # 预算分三档。「停下来问用户是否继续」在两种情况下尤其不该放行：
                #   1. 发生过失败 → 应依据错误信息自动恢复（最宽）；
                #   2. 计划建好了但本段一次实际工具调用都没有 → 只是把计划念了
                #      一遍就收尾，正是「规划完不执行，必须手动说继续」的成因。
                if self._failed_total:
                    nudge_budget = _FINISH_NUDGE_FAIL_MAX
                elif not self._did_work:
                    nudge_budget = _FINISH_NUDGE_PLAN_MAX
                else:
                    nudge_budget = _FINISH_NUDGE_MAX
                if open_steps and self._finish_nudge < nudge_budget:
                    self._finish_nudge += 1
                    todo = "; ".join(
                        f"{s.get('id')}.{s.get('title')}" for s in open_steps[:5]
                    )
                    self._rollback_turn(session)
                    if self._failed_total and self._last_fail:
                        # 失败感知的提醒：把最近一次失败摆到模型眼前，并要求
                        # 「依据失败信息自动解决」而不是停下来等用户发话。
                        # 关键信息有三块：最近的错误、已经试过且失败的方案、
                        # 以及「第 n 次提醒」——模型必须知道自己在原地打转。
                        fname, ferr = self._last_fail
                        tried = self._tried_approaches()
                        tried_txt = f"\n已试过且失败的方案（不要再走）：{tried}。" if tried else ""
                        pending = (
                            f"[System] 计划中还有 {len(open_steps)} 个未完成步骤：{todo}。"
                            f"此前有工具调用失败（共 {self._failed_total} 次，"
                            f"最近：{fname} → {ferr[:200]}）。{tried_txt}\n"
                            f"这是第 {self._finish_nudge} 次提醒。"
                            "禁止停下来、禁止问用户「是否继续」、禁止只输出文字总结——"
                            "你必须**调用工具**往下推进：\n"
                            "1) 读完整错误，判定属于参数错误 / 路径不存在 / 权限 / "
                            "依赖缺失 / 语法错误 / 网络不可达 中的哪一类；\n"
                            "2) 针对该原因换一种**没试过**的方法（换工具、换参数、"
                            "或用 code_run 写最小诊断脚本先验证假设）；\n"
                            "3) 至少尝试 3 种不同方法都确认不可行，才把该步标记 "
                            "blocked 并在 note 写明原因，然后继续后续步骤。\n"
                            "现在就调用工具。"
                        )
                    elif not self._did_work:
                        # 「只规划不执行」专用提醒：计划建好了，但本段一次实际
                        # 工具调用都没有。此时模型只是把计划复述了一遍——必须
                        # 点名让它执行 Now 那一步，而不是泛泛地说「继续」。
                        now = next(
                            (f"{s.get('id')}.{s.get('title')}" for s in open_steps
                             if s.get("status") == "doing"),
                            f"{open_steps[0].get('id')}.{open_steps[0].get('title')}",
                        )
                        pending = (
                            f"[System] 计划已建立，但你还"
                            + ("**一次工具都没调用**" if not self._resume
                               else "在本段没有调用任何工具")
                            + "——只把计划写出来不算开始执行。"
                            f"这是第 {self._finish_nudge} 次提醒。\n"
                            f"现在立即开始执行当前步骤（Now）：{now}。\n"
                            "要求：下一轮必须包含**工具调用**（读文件 / 执行命令 / "
                            "检索 / 写代码等实际动作），不要只描述你打算做什么，"
                            "不要停下来问用户是否开始或是否继续。"
                        )
                    else:
                        pending = (
                            f"[System] 计划中还有 {len(open_steps)} 个未完成步骤：{todo}。"
                            "请继续执行下一步（调用相应工具）。注意：若某步的工作**实际已经"
                            "完成**只是忘了更新状态，用 update_plan(op=\"patch\", "
                            "step_id=..., status=\"done\", note=\"结论\") 收尾即可，"
                            "不要为响应提醒而重复已完成的工作；若某步确实无法完成，"
                            "把它标为 blocked/skipped 并写明原因。"
                        )
                    self.log(f"[kernel] 反思门：还有 {len(open_steps)} 步未完成，提醒继续")
                    continue
                event = self._no_tool_event(resp)
                if event == "retry":
                    self._rollback_turn(session)
                    pending = "[ERROR] 上一轮回复为空或流被截断，请重新生成；如需操作必须显式调用工具。"
                    continue
                if event == "reask":
                    self._reask_guard += 1
                    if self._reask_guard <= 1:
                        # 第一次只是提醒：整轮作废重来（这一轮没有 assistant
                        # 输出，留着只会污染历史）。
                        self._rollback_turn(session)
                        pending = (
                            "[System] 检测到你上轮主要内容是大段代码却未调用工具。"
                            "若这些代码需要执行/写入/分析，请显式调用 code_run / "
                            "file_write / file_patch；"
                            "若只是展示给用户的示例，请在回复中补充自然语言说明。"
                        )
                        continue
                    # 第二次不再纠缠：把这轮的正文当作最终答复收下。
                    # 注意必须在 finish_turn 之后再 break —— 回滚会把 rec 从
                    # session 摘掉，之后再 finish 写的是游离对象，等于丢轮。
                    final_text = resp.content
                    session.finish_turn(rec, resp.content, [], [],
                                        l2_summary(resp.content),
                                        raw_content=resp.raw_content)
                    exit_reason = ExitReason.DONE
                    break
                final_text = resp.content
                session.finish_turn(rec, resp.content, [], [],
                                    l2_summary(resp.content),
                                    raw_content=resp.raw_content)
                exit_reason = ExitReason.DONE
                break

            # ---- ACT ----
            self.state = next_state(KernelState.THINK, "has_calls")
            tool_calls = resp.tool_calls

            # ---- 规划前置门：先于执行 ----
            # 多步任务必须先 update_plan(create) 写入执行计划，再逐步执行；否则
            # 后续没有 plan 可跟踪（anchor_prompt 不回注、UI 面板空、反思门对
            # open_steps 无效）。软提示（规划门）之外这里硬拦一次：从未建过计划、
            # 也还没干过实际活，却要发出非 update_plan 的实际工具调用 → 回退该轮
            # 并强制先规划。本批若已含 update_plan 调用则放行（让 dispatch 先建计划）。
            # 首轮 + 简短无多步信号的指令视为简单任务，放行（读文件/问答这类
            # 一句话任务不该被卡住）。
            has_actual = any(tc.name != "update_plan" for tc in tool_calls)
            has_plan_op = any(tc.name == "update_plan" for tc in tool_calls)
            if (session.get_plan() is None
                    and not self._did_work
                    and has_actual and not has_plan_op
                    and not self._resume):
                simple = (self._turns_used <= 1
                          and len(user_input or "") <= 60
                          and not _MULTISTEP_HINT.search(user_input or "")
                          and not _MULTISTEP_VERB_RE.search(user_input or ""))
                if not simple:
                    self._rollback_turn(session)
                    pending = (
                        "[System] 你准备执行实际工具，但还没有执行计划（Session 中 plan 为空）。"
                        "请先用 update_plan(op='create', goal='一句话目标', steps=[...]) 把目标与 "
                        "2~7 个可验证步骤规划好；创建后工具结果会指示『立即开始第一步』，"
                        "不要停下来问用户是否开始。计划每轮回注，是后续多步任务跟踪的依据，"
                        "必须先于执行写入。"
                    )
                    self.log("[kernel] 规划前置门：拦截首次执行，要求先 update_plan(create)")
                    continue

            try:
                raw_results = self.dispatcher.dispatch(tool_calls)
            except Exception as e:
                self.log(f"[kernel] dispatcher error: {e}")
                raw_results = [None] * len(tool_calls)

            # 失败记账必须在拼装 pending 之前：步骤内失败数要在 update_plan
            # 把它重置之前，被「未充分尝试就 blocked」的守卫读走。
            step_fails_before = self._step_fails
            batch = self._track_failures(tool_calls, raw_results)
            # 建计划本身不算「开工」：只调 update_plan 就收尾 = 只规划不执行。
            if any(tc.name != "update_plan" for tc in tool_calls):
                self._did_work = True

            # ---- 重复调用检测（成功调用也要管）----
            # 失败记账只覆盖「调用失败」，但卡死更常见的形态是**反复调用同一个
            # 成功的工具**：模型不知道怎么推进，就一遍遍写同一个 checkpoint /
            # 用同样的参数重试。每轮都成功 → 反思门的「无工具调用」分支永远不
            # 触发，于是把 max_iterations 全耗在空转上（用户看到的就是一串一模
            # 一样的工具调用）。
            sigs = [self._call_sig(tc) for tc in tool_calls]
            if sigs and sigs == self._last_call_sigs:
                self._repeat_streak += 1
            else:
                self._repeat_streak = 0
            self._last_call_sigs = sigs
            repeat_warn: Optional[str] = None
            if self._repeat_streak >= _REPEAT_CALL_WARN:
                self.log(
                    "[kernel] 连续 %d 次相同的工具调用（%s）→ 判定卡住"
                    % (self._repeat_streak + 1, tool_calls[0].name if tool_calls else "?"))
                if self._repeat_streak >= _REPEAT_CALL_LIMIT:
                    # 提醒已经给过，仍在原地打转 → 收尾，别再烧轮次。
                    self.log("[kernel] 重复调用达到上限，结束本轮")
                    exit_reason = ExitReason.DONE
                    break
                # 只收集、不直接赋值 pending：pending 会在下面被
                # `pending = "\n".join(parts)` 整体覆盖（_to_outcomes 对每个
                # outcome 恒设 next_prompt="\n"，`if not next_prompts` 分支
                # 永不成立），直接赋值等于提醒被静默丢弃、模型从未被告知。
                repeat_warn = (
                    f"[System] 你已经连续 {self._repeat_streak + 1} 次调用**完全相同**的"
                    f"工具与参数（{tool_calls[0].name}）——这不会让任务前进哪怕一步。"
                    "立刻换一个动作：\n"
                    "1) 若这一步其实做不了/不需要工具，就用 update_plan(patch) 把它"
                    "标为 done（写明结论）或 skipped，然后进入下一步；\n"
                    "2) 若任务本身已经可以回答（例如只是问候/闲聊/缺少信息），"
                    "直接用文字回复用户结束本轮；\n"
                    "3) 若确实需要用户补充信息，用 ask_user 工具提问。\n"
                    "不要再次发出与上一轮相同的调用。"
                )

            outcomes = self._to_outcomes(tool_calls, raw_results)
            tool_results = [
                {"id": tc.id, "name": tc.name, "content": str(o.data or "")}
                for tc, o in zip(tool_calls, outcomes)
            ]
            summary = resp.summary or l2_summary(resp.content, [tc.name for tc in tool_calls])
            session.finish_turn(rec, resp.content, tool_calls, tool_results, summary,
                                raw_content=resp.raw_content)
            # 这一轮被真正提交了（不是被回滚的重试轮）→ 重置空回复/重问守卫。
            # 旧实现让它们跨轮累积：长任务后期一次偶发空回复/只吐代码块，就会被
            # 过早判成 "done" 收尾。
            self._reset_retry_guards()
            # 「必须输出 <summary>」的纪律提醒过去看 resp.summary 是否非空，而
            # l2_summary 对任何非空 content 都保证返回非空（最差也给「（空回复）」）
            # → has_summary 恒为 True，这条 TIPS 是死代码，L2 摘要纪律实际失效。
            # 改成直接问「模型有没有真的按协议写 <summary>」。
            has_summary = has_summary_tag(resp.content)

            if any(o.should_exit for o in outcomes):
                final_text = resp.content
                exit_reason = ExitReason.INTERRUPT
                break

            next_prompts = sorted({o.next_prompt for o in outcomes if o.next_prompt})
            if not next_prompts:
                final_text = resp.content
                exit_reason = ExitReason.DONE
                break

            parts = list(next_prompts)
            if not has_summary:
                parts.append("[TIPS] 回复正文必须包含 <summary>…</summary>（≤30 字物理快照）。")
            # 失败恢复提示：计划步骤出错时不许停下等用户说「继续」——依据失败
            # 信息自动换方案，并防住原样重试与「未充分尝试就 blocked」。
            parts.extend(self._recovery_hints(tool_calls, raw_results, batch,
                                              step_fails_before))
            # 补建提示：已开始执行但 plan 仍空 → 提醒补建计划（兜底）。
            # 防止「首轮误执行」导致全程无 plan 可跟踪；最多提醒 _PLAN_NUDGE_MAX 次。
            if (self._did_work and session.get_plan() is None
                    and self._plan_nudged < _PLAN_NUDGE_MAX):
                self._plan_nudged += 1
                parts.append(
                    "[System] 你已经开始执行但 Session 中还没有执行计划（plan 为空）——"
                    "后续多步任务将没有进度可跟踪。请立即用 update_plan(op='create', "
                    "goal=..., steps=[...]) 把目标和已完成/待做步骤登记为计划；"
                    "之后每完成一步用 update_plan(op='patch', step_id=..., "
                    "status='done', note='结论') 更新进度。计划每轮回注，是跟踪推进的依据。"
                )
            # 重复调用提醒放在最后拼入（必须在 pending 赋值之前 append）。
            if repeat_warn:
                parts.append(repeat_warn)
            # 失败调用原样重试熔断：与成功调用的 _REPEAT_CALL_LIMIT 对等。
            # 模型对同一无效参数反复重试（如 update_working_checkpoint({}) 缺
            # key_info）只提示不收敛会烧满 max_iterations。连续相同失败签名
            # 达到上限 → 强制结束本轮（避免空转、用户久等）。
            fr = batch.get("fail_repeat_streak", 0)
            ft = batch.get("same_tool_fail_streak", 0)
            if fr >= _FAIL_REPEAT_LIMIT:
                self.log("[kernel] 连续 %d 次相同失败调用，结束本轮（避免空转）"
                         % (fr + 1))
                if not final_text:
                    final_text = (resp.content
                                  or "（工具调用持续失败，已停止当前回合）")
                exit_reason = ExitReason.DONE
                break
            elif (self._is_greeting(user_input)
                  and not (resp.content or "").strip()):
                # 严格问候（hi/你好/在吗…）却没有任何文字答复：本地弱模型常无视
                # 「寒暄禁止调工具」的指令，先成功写一次 checkpoint（work_memory
                # set）把失败计数 ft 重置为 0，再发 work_memory({}) 空参空转。原
                # 阈值要 ft>=2 才触发，被这次「成功→失败」恰好绕过。问候本身不需要
                # 任何工具，与其烧轮等它「自觉」，不如只要严格问候 + 无文字答复就直接
                # 构造一句礼貌文字回复并结束本轮（不论工具成功/失败）。注意用
                # _is_greeting（严格寒暄）而非 _is_simple_input（任何短输入），否则
                # 「读文件」等短任务会被误杀、丧失自我恢复机会。
                self.log("[kernel] 问候无文字答复，直接文字收尾（避免空转）")
                final_text = resp.content or "你好！有什么可以帮你的吗？"
                exit_reason = ExitReason.DONE
                break
            pending = "\n".join(parts)
            self.state = next_state(KernelState.ACT, "continue")
        else:
            exit_reason = ExitReason.MAX_TURNS

        if exit_reason == ExitReason.MAX_TURNS and not final_text:
            final_text = (
                f"（已达到本轮最大轮次上限 {self.max_turns}，任务可能未完成。"
                f"请先总结已完成的进展，再询问用户是否继续。）"
            )
        # 提醒次数用尽后模型仍要收尾：尊重决定，但必须把「其实没做完」告诉用户，
        # 否则又是一个静默的半途而废。
        if exit_reason is ExitReason.DONE and self._finish_nudge:
            left = session.plan_open_steps()
            if left:
                todo = "; ".join(f"{s.get('id')}.{s.get('title')}" for s in left[:8])
                final_text = (final_text or "") + (
                    f"\n\n---\n⚠ 计划中仍有 {len(left)} 个未完成步骤：{todo}"
                    "（内核已多次要求继续执行仍未完成。若某步其实已做完，用 "
                    "update_plan 补标 done；若确实不可行，标 blocked 并写明原因。）"
                )

        if emit_finished:
            try:
                self.hooks.finished(session, final_text)
            except Exception as e:
                self.log(f"[kernel] finished hook error: {e}")

        return KernelResult(final_text, exit_reason, self._turns_used,
                            self.distill_requested)

    # ------------------------------------------------------------------
    # 转移判定
    # ------------------------------------------------------------------
    def _needs_consolidate(self) -> bool:
        # 仅当原文区真的越过预算才折叠。周期性折叠（每 N 轮）会重写历史 token
        # 序列，破坏多轮 KV 缓存前缀 → 每 N 轮退回全量预填充。预算充裕时保持
        # 历史追加式，KV 缓存稳定命中；预算溢出由 force_fold 成块折叠兜底
        # （两次折叠之间摘要块恒定，缓存可命中）。
        return bool(self.budget.raw_over)

    @staticmethod
    def _rollback_turn(session: Session) -> None:
        """空轮/无效轮不占历史，直接回退。"""
        if session.turns:
            session.turns.pop()
        session.turn_index = max(0, session.turn_index - 1)

    # ------------------------------------------------------------------
    # THINK：一次 LLM 调用
    # ------------------------------------------------------------------
    def _llm(self) -> Any:
        """解析 LLM 客户端（允许运行时被整体替换）。"""
        return self.llm() if callable(self.llm) else self.llm

    def _invalidate_kv(self) -> None:
        """丢弃本地后端的跨轮 KV 缓存（历史被改写时）。

        HTTP 后端没有该接口 → getattr 拿不到就静默跳过，零影响。
        """
        try:
            fn = getattr(self._llm(), "invalidate_kv_cache", None)
            if callable(fn):
                fn()
        except Exception as e:
            self.log(f"[kernel] invalidate_kv failed: {e}")

    def _ask(self, messages: List[Dict[str, Any]]) -> Optional[LLMResponse]:
        content_buf: List[str] = []
        reasoning_buf: List[str] = []
        calls: List[Dict[str, Any]] = []
        usage: Dict[str, Any] = {}
        # 模型原始输出（双视图历史的第二视图），只在 done 事件里出现一次。
        raw_buf: str = ""
        ttfb: Optional[float] = None
        last_emit: Optional[float] = None
        live_est = 0
        attempt = 0
        t0 = time.time()
        cancelled = False
        seen_done = False

        while True:
            content_buf, reasoning_buf, calls, usage = [], [], [], {}
            # raw_buf 也必须随重试重置：旧实现漏了它。当前不会出错（got_any=False
            # 时不可能收到 done 事件），但属于脆弱点——一旦重试路径提前 break，
            # 上一轮的 raw_content 会污染本轮的双视图历史。
            raw_buf = ""
            ttfb = last_emit = None
            live_est = 0
            got_any = False
            seen_done = False
            try:
                for ev in self._llm().chat_stream(
                    messages,
                    tools=self.schemas(),
                    temperature=getattr(self.config, "temperature", 0.7),
                ):
                    got_any = True
                    if self._cancelled():
                        cancelled = True
                        break
                    now = time.time()
                    if ttfb is None:
                        ttfb = now
                    t = getattr(ev, "type", "")
                    if t == "content":
                        content_buf.append(ev.content)
                        self._emit_ui("on_content", ev.content)
                        live_est += estimate_tokens(ev.content)
                    elif t == "reasoning":
                        reasoning_buf.append(ev.content)
                        self._emit_ui("on_reasoning", ev.content)
                        live_est += estimate_tokens(ev.content)
                    elif t == "done":
                        seen_done = True
                        calls = getattr(ev, "tool_calls", None) or []
                        usage = getattr(ev, "usage", None) or {}
                        raw_buf = getattr(ev, "raw_content", "") or ""
                        continue
                    if last_emit is None:
                        last_emit = now
                    elif live_est and now - last_emit >= 0.3:
                        gen = max(now - (ttfb or t0), 1e-6)
                        self._emit_ui("on_token_speed",
                                      self._total_tokens + live_est, live_est / gen)
                        last_emit = now
                break
            except Exception as e:
                # 只有「一个 chunk 都没收到」才重试：已吐过内容的流重试会拼接两段输出。
                if got_any or attempt >= _MAX_LLM_RETRIES:
                    raise
                if self._is_context_error(str(e)) or re.search(r"HTTP 4\d\d", str(e)):
                    raise
                attempt += 1
                delay = _LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                self.log(f"[kernel] LLM 首字节前失败，{delay:.1f}s 后重试 ({attempt}): {e}")
                if self._cancelled():
                    raise
                # 分段休眠并轮询取消：一次性 time.sleep(delay) 最长 3s 完全不
                # 响应「终止对话」，用户点了取消还要干等。
                _slept = 0.0
                while _slept < delay:
                    if self._cancelled():
                        raise
                    time.sleep(min(0.1, delay - _slept))
                    _slept += 0.1

        if cancelled:
            return None

        reasoning_text = "".join(reasoning_buf)
        if not seen_done and ("".join(content_buf) or reasoning_text):
            # 流被截断（网络断开 / 服务端提前结束）——必须留下可见痕迹。
            self.log("[agent] warning: LLM stream ended without 'done' event")

        if usage:
            try:
                self.callbacks.on_usage(usage)
            except Exception:
                pass
            details = usage.get("completion_tokens_details") or {}
            api_reasoning = int(details.get("reasoning_tokens") or 0) if isinstance(details, dict) else 0
            est_reasoning = estimate_tokens(reasoning_text)
            api_total = int(usage.get("total_tokens") or 0)
            # API 的 total 可信（含 reasoning）时才直接用；否则补上估算值，
            # 否则推理密集型模型会显示低得离谱的 tok/s。
            if api_reasoning or not reasoning_text:
                self._total_tokens = getattr(self, "_total_tokens", 0) + api_total
            else:
                self._total_tokens = getattr(self, "_total_tokens", 0) + api_total + est_reasoning
        completion = int(usage.get("completion_tokens") or 0)
        gen_elapsed = max(time.time() - (ttfb or t0), 1e-6)
        try:
            self.callbacks.on_token_speed(self._total_tokens,
                                          completion / gen_elapsed if completion else 0.0)
        except Exception:
            pass

        # 注意：这里**不能**重置 _empty_retry / _reask_guard。它们由
        # _no_tool_event 在「本轮回复不可用」时递增、在重试额度用尽时判 done；
        # 每次流式成功就清零会让重试永远触不到上限（空回复无限循环）。
        # 正确的重置点是「一轮被真正提交」之后（见 _reset_retry_guards）。

        text = "".join(content_buf)
        resp = LLMResponse(
            content=text,
            reasoning=reasoning_text,
            tool_calls=[self._to_tool_call(c) for c in calls],
            usage=usage,
            raw_content=raw_buf,
        )
        resp.summary = l2_summary(text, [tc.name for tc in resp.tool_calls]) if text else ""
        return resp

    def _to_tool_call(self, raw: Dict[str, Any]) -> ToolCall:
        fn = (raw.get("function") or {}) if isinstance(raw, dict) else {}
        args_raw = fn.get("arguments", "")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) and args_raw else {}
        except (json.JSONDecodeError, ValueError) as e:
            # 解析失败必须报回模型，静默当 {} 会变成莫名的「缺少必填参数」。
            args = {"__parse_error__": f"Invalid JSON in tool arguments: {e}",
                    "__raw__": args_raw}
        if not isinstance(args, dict):
            args = {}
        tid = str(raw.get("id") or "")
        if not tid:
            # 部分本地模型不发 id；没有 id 就无法把 tool 结果关联回去。
            # 前缀必须区别于模型常见的 "call_1"，否则同一轮内可能撞 id
            # （两条 tool 消息同 tool_call_id → API 4xx）。
            self._tc_counter = getattr(self, "_tc_counter", 0) + 1
            tid = f"kcall_{self._tc_counter}"
        return ToolCall(
            id=tid,
            name=str(fn.get("name") or ""),
            args=args,
            raw_arguments=args_raw if isinstance(args_raw, str) else "",
        )

    # ------------------------------------------------------------------
    # ACT：工具结果 → StepOutcome（L1 裁剪在这里发生）
    # ------------------------------------------------------------------
    def _to_outcomes(self, tool_calls: List[ToolCall],
                     raw_results: List[Any]) -> List[StepOutcome]:
        n = max(1, len(tool_calls))
        max_chars = self.budget.remaining_for_tool(n, self.max_tool_chars)
        out: List[StepOutcome] = []
        for tc, res in zip(tool_calls, raw_results):
            # ask_user 与普通工具同路径：用户回答作为 tool 结果落进历史，
            # 模型带着答案继续任务。之前把它特判成 INTERRUPT 会让整轮终止、
            # 用户答案被直接丢弃（回答进不了任何后续 LLM 调用）。
            if tc.name == "start_long_term_update":
                self.distill_requested = True
            # L0：先兜住病态长度（工具本身没上限时保护进程内存），
            # L1：再按 RAW 区余额裁到真正要注入的长度。
            out.append(StepOutcome(
                data=l1_trim(l0_capture(self._result_text(res), hard_cap=_L0_HARD_CAP),
                             max_chars),
                next_prompt="\n"))
        return out

    @staticmethod
    def _result_text(res: Any) -> str:
        if res is None:
            return "ERROR: 工具未返回结果"
        to_msg = getattr(res, "to_message", None)
        if callable(to_msg):
            try:
                return to_msg()
            except Exception:
                pass
        return str(res)

    # ------------------------------------------------------------------
    # ACT：失败记账与自动恢复提示
    #
    # 计划步骤失败时，内核不允许「自动停下等用户打继续」：失败信息必须转化为
    # 下一轮的恢复指令（换方案 / 先诊断），同时防住两种浪费——原样重试同一个
    # 失败调用、没试几次就把步骤标记 blocked。
    # ------------------------------------------------------------------
    @staticmethod
    def _result_ok(res: Any) -> Tuple[bool, str]:
        """ToolResult → (是否成功, 错误摘要)。None/异常形态一律算失败。"""
        if res is None:
            return False, "工具未返回结果"
        if getattr(res, "success", True):
            return True, ""
        err = str(getattr(res, "error", "") or "") or "未知错误"
        return False, err[:300]

    @staticmethod
    def _call_sig(tc: ToolCall) -> str:
        """失败调用签名：工具名 + 参数。用于识别「原样重试」。"""
        try:
            args = json.dumps(tc.args, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            args = str(tc.args)
        return f"{tc.name}::{args[:500]}"

    def failure_snapshot(self) -> Dict[str, Any]:
        """导出失败上下文，供上层在「自动续跑」的段之间传递。

        段与段之间内核记账会被 run() 重置，失败信息随之中断——续跑段就不知道
        上一段卡在哪。上层把本快照用 ``seed_failures`` 注入下一段即可续上。
        """
        return {
            "failed_total": self._failed_total,
            "last_fail": self._last_fail,
            "fail_sigs": list(self._fail_sigs)[-_MAX_FAIL_SIGS:],
            "fail_streak": self._fail_streak,
            # 防死循环的各类 nudge 计数也必须跨段传递：run() 每次都会把它们清零，
            # 而 Agent 最多自动续跑 4 段 —— 不传的话「段内最多拦 N 次」会被放大
            # 成 4×N 次，单次用户输入最坏 4×15 轮 LLM 调用且没有墙钟上限，
            # 推理模型下可能几十分钟不返回任何结果。
            "finish_nudge": self._finish_nudge,
            "plan_finish_nudge": self._plan_finish_nudge,
            "block_nudge": self._block_nudge,
            "empty_retry": self._empty_retry,
            "reask_guard": self._reask_guard,
            "plan_nudged": self._plan_nudged,
            "fail_repeat_streak": self._fail_repeat_streak,
            "same_tool_fail_streak": self._same_tool_fail_streak,
            "same_tool_fail_name": self._same_tool_fail_name,
            # 上一段是否已经真正干过活。不传的话续跑段 _did_work=False，
            # 反思门文案会变成「你还一次工具都没调用」（对已经干过活的段是错的，
            # 会误导模型），失败感知预算也会从 6 掉回 2。
            "did_work": self._did_work,
        }

    def seed_failures(self, snapshot: Optional[Dict[str, Any]]) -> None:
        """把上一段的失败上下文带进下一段 run()（自动续跑用）。

        除 ``_failed_total / _last_fail / _fail_sigs / _fail_streak`` 外，还带上
        各类防死循环 nudge 计数与 ``_did_work``：续跑段因此仍享有失败感知的
        反思门预算、提醒里能引用上一段的真实错误，且不会把「段内最多拦 N 次」
        放大成「每段各拦 N 次」。
        """
        self._seed_failures = dict(snapshot) if snapshot else None

    def _tried_approaches(self, limit: int = 5) -> str:
        """把近期失败过的调用压成「已试过且失败」的清单（提醒里点名排除）。

        模型反复停车的一大原因是它不知道自己已经试过什么，于是把同一个思路
        再讲一遍。把签名里已失败的工具/参数摆出来，它才能挑一个**没试过**的路。
        """
        seen: List[str] = []
        for sig in self._fail_sigs:
            name, _, args = sig.partition("::")
            if not name:
                continue
            short = args[:60].replace("\n", " ")
            item = f"{name}({short})" if short else name
            if item not in seen:
                seen.append(item)
        return "； ".join(seen[-limit:])

    def _track_failures(self, tool_calls: List[ToolCall],
                        raw_results: List[Any]) -> Dict[str, Any]:
        """ACT 态失败记账：连续失败、失败签名、当前步骤的失败次数。

        ``update_plan`` 的失败不计入步骤失败数（它是对计划的登记，不是对步骤
        本身的尝试）；但它**成功**时视为步骤边界，重置计数。返回本批摘要
        （含「原样重试」检测结果），供 ``_recovery_hints`` 使用。
        """
        failed: List[Tuple[ToolCall, str]] = []
        repeats: List[Tuple[str, str]] = []
        plan_mutated = False
        batch_fail_sigs: List[str] = []
        # 上一轮最后记录的失败签名：用于判断「连续相同失败签名」（原样重试）。
        # 必须在循环前取，循环内 self._fail_sigs 会被不断 append / 截断。
        prev_fail_sig = self._fail_sigs[-1] if self._fail_sigs else None
        for tc, res in zip(tool_calls, raw_results):
            ok, err = self._result_ok(res)
            if tc.name == "update_plan":
                if ok:
                    plan_mutated = True
                continue
            if ok:
                continue
            failed.append((tc, err))
            self._fail_streak += 1
            self._failed_total += 1
            self._last_fail = (tc.name, err)
            self._step_fails += 1
            sig = self._call_sig(tc)
            batch_fail_sigs.append(sig)
            if sig in self._fail_sigs:
                repeats.append((tc.name, err))
            self._fail_sigs.append(sig)
            del self._fail_sigs[:-_MAX_FAIL_SIGS]
        if not failed:
            self._fail_streak = 0
            self._fail_repeat_streak = 0
            self._same_tool_fail_streak = 0
            self._same_tool_fail_name = None
        else:
            # 原「连续相同失败签名」计数（原样重试 → 必须熔断以免烧满
            # max_iterations）：仅签名完全一致的连续重试才累加；换参数/换
            # 工具探索不算，避免误伤自主任务的多方法尝试。
            if prev_fail_sig is not None and all(
                    s == prev_fail_sig for s in batch_fail_sigs):
                self._fail_repeat_streak += 1
            else:
                self._fail_repeat_streak = 1
            # 「同一工具连续失败」计数（不限参数是否相同）：弱模型在问候等
            # 简单场景常死磕同一个工具、只换参数（work_memory({}) →
            # work_memory({key_info}) 退化），签名每次略变但始终失败。这与
            # 「换工具/换方法探索」不同，应在问候场景尽快收尾。本轮失败若
            # 都来自同一个工具名且与上轮失败工具相同 → 累加；否则清零。
            names = {tc.name for tc, _ in failed}
            if len(names) == 1:
                tname = next(iter(names))
                if self._same_tool_fail_name == tname:
                    self._same_tool_fail_streak += 1
                else:
                    self._same_tool_fail_streak = 1
                    self._same_tool_fail_name = tname
            else:
                self._same_tool_fail_streak = 0
                self._same_tool_fail_name = None
        if plan_mutated:
            self._step_fails = 0
        return {"failed": failed, "repeats": repeats,
                "plan_mutated": plan_mutated,
                "fail_repeat_streak": self._fail_repeat_streak,
                "same_tool_fail_streak": self._same_tool_fail_streak}

    def _recovery_hints(self, tool_calls: List[ToolCall], raw_results: List[Any],
                        batch: Dict[str, Any], step_fails_before: int) -> List[str]:
        """依据本批失败信息生成「自动恢复」提示，拼进下一轮锚点。

        三个层次：
          1. 原样重试 → 禁止重复，必须换方法
          2. 连续失败 ≥2 → 停止同思路，先诊断再换方案
          3. 未充分尝试就把步骤标记 blocked → 恢复 doing 再试
        """
        hints: List[str] = []
        if batch["repeats"]:
            name, err = batch["repeats"][0]
            hints.append(
                f"[System] 检测到与之前完全相同的失败调用：`{name}`"
                f"（错误：{err[:160]}）。原样重试不会改变结果，禁止重复。"
                "请换参数/换工具/换思路；或用 code_run 写诊断脚本定位根因；"
                "确认此路不通后，把对应计划步骤标记 blocked 并继续后续步骤。"
            )
        elif self._fail_streak >= 2:
            name, err = self._last_fail or ("", "")
            tried = self._tried_approaches()
            hints.append(
                f"[System] 已连续失败 {self._fail_streak} 次"
                f"（最近：{name} → {err[:200]}）。"
                f"已试过且失败的方案：{tried or '（无记录）'}。\n"
                "不要停下来问用户是否继续——请依据错误信息自动修复：先读完整错误"
                "定位原因，再换一种**没试过**的方法重试（不同工具/不同参数/先用 "
                "code_run 写最小诊断脚本验证假设）。同一思路只允许试一次。"
            )
        elif batch["failed"]:
            # 首次失败也必须是「换方案」而不是「原样重发 / 停下报告」。
            # 旧实现只在连续失败 ≥2 次时才提示，第一次失败后模型最常见的反应
            # 就是原样重试或把错误念一遍给用户听。
            name, err = batch["failed"][0]
            hints.append(
                f"[System] 工具 `{name}` 调用失败：{err[:200]}。\n"
                "请**先完整读完错误**，判定是参数错误 / 路径不存在 / 权限 / "
                "依赖缺失 / 语法错误 / 网络不可达 中的哪一类，然后换一种方法达成"
                "同一步骤的目标（换工具、换参数，或先用 code_run 写最小诊断脚本"
                "验证假设）。不要原样重发同一个调用，不要停下来向用户报告失败——"
                "失败是给你定位用的信号，不是停车的理由。"
            )
        if self._premature_block(tool_calls, raw_results, step_fails_before):
            hints.append(
                f"[System] 步骤刚被标记 blocked，但本步骤至今只失败了 "
                f"{step_fails_before} 次——尝试不够。请先用 update_plan 把它恢复 "
                "doing，再依据上面的错误信息换**没试过**的方案重试（至少 3 种"
                "不同方法）；只有多种方法都确认不可行后，才标记 blocked 并写明"
                "原因，然后继续后续步骤。不要向用户确认「是否继续」。"
            )
        return hints

    def _premature_block(self, tool_calls: List[ToolCall], raw_results: List[Any],
                         step_fails_before: int) -> bool:
        """本批把某步骤标记为 blocked，且该步骤的失败尝试次数不足。

        命中即消耗一次提醒额度（``_block_nudge``）——守卫只提醒有限次，
        模型坚持放弃时最终尊重它的决定。
        """
        if self._block_nudge >= _BLOCK_NUDGE_MAX:
            return False
        if step_fails_before >= _MIN_ATTEMPTS_BEFORE_BLOCK:
            return False
        for tc, res in zip(tool_calls, raw_results):
            if tc.name != "update_plan":
                continue
            if str(tc.args.get("status", "")).strip().lower() != "blocked":
                continue
            ok, _ = self._result_ok(res)
            if ok:
                self._block_nudge += 1
                return True
        return False

    # ------------------------------------------------------------------
    # 无工具调用时的判定（对齐 GenericAgent 的 do_no_tool）
    # ------------------------------------------------------------------
    @staticmethod
    def _is_simple_input(user_input: str) -> bool:
        """一句话任务判定：短、且不含多步信号。

        与「规划前置门」里的 simple 判定同源（那里还要求首轮）。用于让问候、
        闲聊、单个小问题跳过强制规划——它们本来也不需要执行计划。
        """
        t = (user_input or "").strip()
        if not t:
            return True
        if len(t) > 60:
            return False
        return not (_MULTISTEP_HINT.search(t) or _MULTISTEP_VERB_RE.search(t))

    # 严格问候集合：只认真正的寒暄，与「任何短输入」的 _is_simple_input 区分开。
    # 熔断的「问候快速收尾」只能用它，否则会把「读文件」等短任务也误杀。
    _GREETING_SET = {
        "hi", "hello", "hey", "hiya", "yo", "sup", "howdy",
        "你好", "您好", "您好呀", "嗨", "在吗", "在么", "在不在", "在的",
        "有人吗", "在不在呀",
    }

    @staticmethod
    def _is_greeting(user_input: str) -> bool:
        """严格问候判定（区别于 _is_simple_input 的『任何短输入』）。

        只为真正的寒暄打断强制收尾，不误伤「读文件」等短任务——它们失败两次
        也应给模型自我恢复的机会（换路径 / 问用户），不应被一句话回复截胡。
        """
        t = (user_input or "").strip().lower()
        if not t:
            return True
        t = t.rstrip("。！？!?，,.、；; \t\n")
        return t in KernelLoop._GREETING_SET

    def _looks_like_asking_user(self, content: str) -> bool:
        """回复是否是在向用户追问完成任务所必需的信息。

        判定偏保守（宁可漏判也不要误判）：漏判只是多走一轮提醒，误判会让模型
        学会用「问一句」来逃避执行。两个信号任一命中即可：
          1. 结尾 200 字里有问号；
          2. 结尾命中索取信息的句式（_ASK_USER_RE）。
        """
        c = (content or "").strip()
        if not c:
            return False
        tail = c[-200:]
        if "？" in tail or "?" in tail:
            return True
        return bool(_ASK_USER_RE.search(tail))

    def _no_tool_event(self, resp: LLMResponse) -> str:
        content = (resp.content or "").strip()
        if not content and not (resp.reasoning or "").strip():
            self._empty_retry += 1
            return "retry" if self._empty_retry <= _MAX_EMPTY_RETRY else "done"
        tail = content[-120:]
        if any(m in tail for m in _TRUNCATION_MARKS) or (
            content.endswith("</summary>") and len(content) < 100
        ):
            self._empty_retry += 1
            return "retry" if self._empty_retry <= _MAX_EMPTY_RETRY else "done"
        if self._looks_like_unexecuted_code(content):
            return "reask"
        return "done"

    @staticmethod
    def _looks_like_unexecuted_code(content: str) -> bool:
        """恰好一个大代码块 + 结尾 + 正文几乎没有自然语言 → 疑似该执行却没执行。"""
        blocks = _CODE_BLOCK_RE.findall(content)
        if len(blocks) != 1:
            return False
        m = _CODE_BLOCK_RE.search(content)
        if not m or m.end() < len(content) - 2:
            return False
        residual = content.replace(m.group(0), "")
        residual = _TAG_STRIP_RE.sub("", residual)
        return len(re.sub(r"\s+", "", residual)) <= 30

    # ------------------------------------------------------------------
    def _cancelled(self) -> bool:
        try:
            return bool(self.callbacks.is_cancelled())
        except Exception:
            return False

    @staticmethod
    def _is_context_error(err: str) -> bool:
        e = (err or "").lower()
        return any(k in e for k in (
            "context_length", "context length", "context window",
            "maximum context", "prompt is too long", "prompt too long",
            "too many tokens", "token limit",
        ))
