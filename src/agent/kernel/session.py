"""Session: the owner of history.

The kernel does not keep a ``_history`` list. It hands every turn to the
Session, and the Session decides what the model is allowed to see:

    [system]
      + L3 折叠区（旧轮 → 一行摘要，同类项合并）
      + 最近 K 轮原文（保头保尾，工具输出已过 L1）
      + 本轮 user 消息（= 锚点提示 + next_prompt）

Because turns are dropped as whole units, a ``tool`` message can never be
orphaned from its ``assistant(tool_calls)`` (which makes OpenAI-compatible
servers reject the request with HTTP 400).
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from .budget import ContextBudget, estimate_tokens
from .compressor import compact_json, l3_fold
from .outcome import TurnRecord, ToolCall

# 计划步骤状态。done/blocked/skipped 都是终态；pending/doing 是非终态。
_PLAN_OPEN = ("pending", "doing")
_PLAN_STATUSES = ("pending", "doing", "done", "blocked", "skipped")
# 每步标题/备注的长度上限（计划每轮都要回注，不能无限膨胀）
_MAX_PLAN_STEPS = 20
_MAX_STEP_TITLE = 120
_MAX_STEP_NOTE = 200
_MAX_PLAN_GOAL = 300
# 状态标记（省 token：一个字符表示一步）
_PLAN_MARK = {"done": "✓", "doing": "▶", "pending": " ", "blocked": "✗", "skipped": "⊘"}


def render_plan_dict(plan: Optional[Dict[str, Any]]) -> str:
    """把一份 plan dict 渲染成紧凑多行文本（``Session.render_plan`` 的纯函数版）。

    UI 面板用：``on_plan_updated(plan)`` 信号携带的 plan 来自**工作线程**，
    旧实现转手去读 ``agent.session.render_plan()`` —— 跨线程读共享可变状态，
    patch 工具原地改 step dict 时可能渲染出半新半旧的中间态。用信号携带的
    快照 + 本纯函数，UI 线程不再碰 Session。
    """
    if not isinstance(plan, dict):
        return ""
    lines: List[str] = []
    goal = plan.get("goal") or ""
    if goal:
        lines.append(f"Goal: {goal}")
    cur = ""
    for s in plan.get("steps", []):
        if not isinstance(s, dict):
            continue
        mark = _PLAN_MARK.get(s.get("status", "pending"), " ")
        title = s.get("title", "")
        note = s.get("note", "")
        line = f"- [{mark}] {s.get('id', '?')}. {title}"
        if note:
            line += f"  # {note}"
        lines.append(line)
        if s.get("status") == "doing" and not cur:
            cur = f"{s.get('id', '?')}. {title}"
    if cur:
        lines.append(f"Now: {cur}")
    open_steps = [s for s in plan.get("steps", [])
                  if isinstance(s, dict) and s.get("status") in _PLAN_OPEN]
    if open_steps:
        lines.append(f"Remaining: {len(open_steps)} step(s) not finished")
    return "\n".join(lines)


def normalize_plan(data: Any) -> Optional[Dict[str, Any]]:
    """把模型传来的 plan 归一化为确定结构；非法返回 None。

    计划是「模型自述、内核依赖」的数据 —— 内核的「未完成步骤不得收尾」门
    要读它，因此必须容忍脏输入（steps 不是 list、status 是奇怪的值、
    条目不是 dict），否则一个畸形计划就能让 run() 抛异常。
    """
    if not isinstance(data, dict):
        return None
    goal = str(data.get("goal", "") or "").strip()[:_MAX_PLAN_GOAL]
    raw_steps = data.get("steps")
    # 本地模型（Qwen XML 等文本工具调用）常把 steps 作为**JSON 字符串**传入
    # （双重编码）；宽容解析，否则 update_plan 对本地模型必然返回 None。
    if isinstance(raw_steps, str):
        import json as _json
        try:
            raw_steps = _json.loads(raw_steps)
        except (ValueError, TypeError):
            raw_steps = None
    if not isinstance(raw_steps, list):
        return None
    steps: List[Dict[str, str]] = []
    for i, s in enumerate(raw_steps[:_MAX_PLAN_STEPS], 1):
        if isinstance(s, dict):
            title = str(s.get("title", "") or "").strip()[:_MAX_STEP_TITLE]
            status = str(s.get("status", "") or "").strip().lower()
            note = str(s.get("note", "") or "").strip()[:_MAX_STEP_NOTE]
        elif isinstance(s, str):
            title, status, note = s.strip()[:_MAX_STEP_TITLE], "", ""
        else:
            continue
        if not title:
            continue
        steps.append({
            "id": str(s.get("id", i)) if isinstance(s, dict) else str(i),
            "title": title,
            "status": status if status in _PLAN_STATUSES else "pending",
            "note": note,
        })
    if not steps:
        return None
    return {"goal": goal, "steps": steps}


class Session:
    """历史容器 + 工作记忆 + L3 折叠。"""

    def __init__(
        self,
        budget: ContextBudget,
        raw_turns: int = 2,
        max_digest_lines: int = 40,
    ):
        self.budget = budget
        self.raw_turns = max(0, int(raw_turns))
        self.max_digest_lines = max(5, int(max_digest_lines))
        self.turns: List[TurnRecord] = []
        self.working: Dict[str, Any] = {}     # key_info / checkpoint / 临时事实
        # CONSOLIDATE 态主动折叠后留下的摘要行（这些 TurnRecord 已从 turns 移除）
        self.extra_digest: List[str] = []
        self.turn_index = 0
        self.folded_count = 0                 # 已被折叠的轮数（统计用）
        self._last_fold_at = 0.0

    # ------------------------------------------------------------------
    # 轮次生命周期
    # ------------------------------------------------------------------
    @property
    def current_turn(self) -> int:
        return self.turn_index

    def start_turn(self, prompt: str = "") -> TurnRecord:
        self.turn_index += 1
        rec = TurnRecord(index=self.turn_index, prompt=prompt)
        self.turns.append(rec)
        return rec

    def finish_turn(
        self,
        rec: TurnRecord,
        assistant_text: str = "",
        tool_calls: Optional[List[ToolCall]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
        summary: str = "",
        raw_content: str = "",
    ) -> None:
        rec.assistant_text = assistant_text or ""
        rec.raw_content = raw_content or ""
        rec.tool_calls = list(tool_calls or [])
        rec.tool_results = list(tool_results or [])
        rec.summary = summary
        rec.tokens = estimate_tokens(
            rec.prompt + rec.assistant_text
            + "".join(str(r.get("content", "")) for r in rec.tool_results)
        )
        # 一次性提示到此才真正消费：本轮 turn 已落地，提示不再需要回注。
        # 被 rollback 的轮不会走到这里，提示会继续保留给下一轮（见 anchor_prompt）。
        self.working.pop("__truncated__", None)
        self.working.pop("__plan_boundary__", None)

    # ------------------------------------------------------------------
    # 工作记忆（L4 的落地位置）
    # ------------------------------------------------------------------
    def set_key_info(self, text: str) -> None:
        self.working["key_info"] = text
        self.working["passed_sessions"] = 0

    def get_key_info(self) -> str:
        return str(self.working.get("key_info", "") or "")

    def set_fact(self, key: str, value: Any) -> None:
        facts = self.working.setdefault("facts", {})
        if not isinstance(facts, dict):
            facts = {}
            self.working["facts"] = facts
        facts[str(key)] = value

    def clear_working(self) -> None:
        self.working.pop("key_info", None)
        self.working.pop("facts", None)

    # ------------------------------------------------------------------
    # 计划（L2.5：任务级「目标 + 路径」的一等公民表示）
    #
    # 放在 working 里而不是 Session 的独立字段：它与 key_info 同生命周期
    # （跨 run 存续、reset 清空），且天然获得 anchor_prompt 的每轮回注。
    # ------------------------------------------------------------------
    def set_plan(self, plan: Any) -> bool:
        """写入计划（已由 normalize_plan 校验）。返回是否接受。"""
        norm = normalize_plan(plan)
        if norm is None:
            return False
        self.working["plan"] = norm
        return True

    def get_plan(self) -> Optional[Dict[str, Any]]:
        p = self.working.get("plan")
        return p if isinstance(p, dict) else None

    def clear_plan(self) -> None:
        self.working.pop("plan", None)

    def plan_open_steps(self) -> List[Dict[str, str]]:
        """未完成的步骤（pending / doing）——反思门的判据。"""
        p = self.get_plan()
        if not p:
            return []
        return [s for s in p.get("steps", [])
                if s.get("status") in _PLAN_OPEN]

    def render_plan(self) -> str:
        """把计划渲染成紧凑的多行文本（每轮回注 + UI 面板共用）。

        状态用单字符标记：✓ 完成 / ▶ 进行中 / ✗ 阻塞 / ⊘ 跳过 / 空格 待办。
        实现委托给纯函数 ``render_plan_dict``（UI 面板用信号携带的 plan 快照
        调它，避免跨线程读 Session）。
        """
        return render_plan_dict(self.get_plan())

    # ------------------------------------------------------------------
    # 锚点提示：每轮注入的「工作记忆 + 进度 + 周期性纪律」
    # ------------------------------------------------------------------
    def anchor_prompt(self, extra: str = "", every_n: int = 8) -> str:
        """下一轮 user 消息的正文。对应 GenericAgent 的 ``_get_anchor_prompt``。"""
        parts: List[str] = []
        if extra:
            parts.append(extra.strip())
        parts.append(f"\n### [WORKING MEMORY]")
        parts.append(f"Current turn: {self.turn_index + 1}")
        ki = self.get_key_info()
        if ki:
            parts.append(f"<key_info>{ki}</key_info>")
        # 计划紧随 key_info：这是「防止跑丢」的核心回注。模型每轮都能看到
        # 目标、全量步骤与当前进度，不需要靠记忆维持任务连贯性。
        plan = self.render_plan()
        if plan:
            parts.append(f"### [PLAN]\n{plan}")
        # 上一轮被轮次上限截断：必须让模型看到，否则它会以为任务已完成。
        # 用 peek 而不是 pop：这些提示是「一次性」的，但真正的一次性语义是
        # 「本轮 turn 落地后才失效」。旧实现在 anchor_prompt 里 pop，而该轮
        # 紧接着可能被反思门/前置门 rollback —— 提示随之消失，自动续跑注入的
        # 「上一条被判提前收尾」这条关键指令模型永远看不到（只能收到泛化的
        # 「还有 N 步未完成」）。消费点挪到 finish_turn。
        truncated = self.working.get("__truncated__")
        if truncated:
            parts.append(f"[SYSTEM] {truncated}")
        # 任务边界提示（agent.run 在「计划仍有未完成步骤 + 新的用户请求」时
        # 注入）：让模型在「新任务重建计划 / 继续旧任务」之间显式判断。没有
        # 这个提示时，模型会静默沿用旧计划——表现为「新任务的计划面板显示的
        # 是上一个任务的内容」。
        boundary = self.working.get("__plan_boundary__")
        if boundary:
            parts.append(f"[SYSTEM] {boundary}")
        facts = self.working.get("facts")
        if isinstance(facts, dict) and facts:
            parts.append(f"<facts>{compact_json(facts, 600)}</facts>")
        # 周期性纪律（对应 GA 的 turn%13 / turn%31 / turn%175 注入）
        n = self.turn_index + 1
        if n % 13 == 0:
            parts.append(
                "[TIPS] 已运行多轮：调用 update_working_checkpoint 保存关键上下文；"
                "停止无效重试，无进展就换策略（1) 探测物理边界 2) 回读相关 SOP/技能）。"
            )
        if n % 31 == 0:
            parts.append(
                "[TIPS] 把检查点/关键发现/已试过的方案写入**文件**，不要只放在工作记忆里。"
            )
        if n % (every_n * 2) == 0:
            parts.append(
                "[TIPS] 长任务：如已有可复用流程，用 skill_save 固化；"
                "阶段性成果用 start_long_term_update 蒸馏进长期记忆。"
            )
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # CONSOLIDATE：主动折叠
    # ------------------------------------------------------------------
    def force_fold(self, n: int = 1) -> int:
        """把最老的 n 轮从 ``turns`` 摘掉，摘要行留进 ``extra_digest``。

        整轮摘除（而不是摘单条消息）保证不会出现悬挂的 ``tool`` 消息。
        返回实际折叠的轮数。
        """
        folded = 0
        for _ in range(max(0, int(n))):
            if not self.turns:
                break
            self.extra_digest.append(self.turns.pop(0).digest_line())
            folded += 1
        # extra_digest 只由本方法追加、而 l3_fold 只裁**输出**不裁列表本身 ——
        # 长跑会话里它会随 folded_count 无上限增长，且 digest_lines() /
        # all_digest_lines() 每次都全量拷贝。这里按 max_digest_lines 滚动裁掉
        # 最旧的行（保留一条溢出标记，避免「历史凭空消失」）。
        if len(self.extra_digest) > self.max_digest_lines:
            drop = len(self.extra_digest) - self.max_digest_lines
            del self.extra_digest[:drop]
            self.extra_digest[0] = f"[…更早的 {drop} 轮已折叠省略]"
        if folded:
            # 摘掉后 turn_index 不再等于列表长度，后续 start_turn 仍单调自增，
            # 摘要行的 [T{index}] 保持可追溯。
            self.folded_count += folded
        return folded

    # ------------------------------------------------------------------
    # 组装消息
    # ------------------------------------------------------------------
    def _windows(self) -> tuple:
        """切分 ``(old, raw)`` —— L3 的核心。

        ``raw`` = 最近 ``raw_turns`` 个**已完成**的轮 + 当前进行中的轮。
        当前轮虽还没结果，但它的 user 消息必须在场，否则模型看不到本轮指令。
        ``old`` = 其余全部，折叠成摘要行。
        预算紧张时自动收缩 ``raw``（这就是「折叠」的自动触发点）。
        """
        if not self.turns:
            return [], []
        cur = self.turns[-1]
        completed = self.turns[:-1]
        keep = min(self.raw_turns, len(completed))
        # 收缩阈值必须与越界判定同一口径：``remaining_for_raw()`` 还要再扣掉
        # system/earlier/tools 的超支（overspent）。旧实现只跟 quota.raw 比，
        # 长系统提示 + 25 个工具 schema 很容易让那两个区超配额，此时原文窗
        # 不收缩 → 实际发送量越过 30k，只能等 API 报 context_length 再兜底。
        raw_ceiling = self.budget.quota.raw - self.budget.overspent()
        while keep > 0 and estimate_tokens(
            self._render_raw(completed[-keep:] + [cur])
        ) > raw_ceiling:
            keep -= 1
        old = completed[:-keep] if keep else list(completed)
        raw = (completed[-keep:] if keep else []) + [cur]
        return old, raw

    @staticmethod
    def _render_raw(turns: List[TurnRecord]) -> str:
        """估算原文区大小。

        必须与实际发出的消息同源：``tool_calls[*].function.arguments`` 的 JSON
        也会发出去（工具参数里常带大段代码），漏算会让 RAW 区静默超预算。
        """
        return "".join(
            (t.prompt or "") + (t.assistant_text or "")
            + "".join(
                json.dumps(tc.args, ensure_ascii=False) if isinstance(tc.args, dict)
                else str(tc.args)
                for tc in t.tool_calls
            )
            + "".join(str(r.get("content", "")) for r in t.tool_results)
            for t in turns
        )

    def _turn_messages(self, t: TurnRecord) -> List[Dict[str, Any]]:
        # 进行中的轮（还没产出任何东西）不进消息列表：末尾挂一条空 assistant
        # 会让模型从空回合继续生成。
        if not t.prompt and not t.assistant_text and not t.tool_calls:
            return []
        out: List[Dict[str, Any]] = []
        if t.prompt:
            out.append({"role": "user", "content": t.prompt})
        if not (t.assistant_text or t.tool_calls):
            return out
        assistant: Dict[str, Any] = {"role": "assistant", "content": t.assistant_text or None}
        # 第二视图：模型原始输出。HTTP 后端不认识这个键（安全忽略），本地 MLX
        # 后端会用它重建「生成时的逐 token 序列」去命中跨轮 KV 前缀。
        if t.raw_content:
            assistant["raw_content"] = t.raw_content
        if t.tool_calls:
            assistant["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.args, ensure_ascii=False),
                    },
                }
                for tc in t.tool_calls
            ]
        out.append(assistant)
        # 每个 tool_call 都必须有一条对应结果，否则 API 400。
        if t.tool_calls:
            by_id = {r.get("id"): r for r in t.tool_results if r.get("id")}
            for tc in t.tool_calls:
                r = by_id.get(tc.id)
                out.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "content": str(r.get("content", "")) if r else "(no result)",
                })
        return out

    def build_messages(self, system_prompt: str, user_text: str = "") -> List[Dict[str, Any]]:
        """产出本轮发给模型的完整消息列表（已过 L1/L3）。"""
        old, raw = self._windows()
        self.folded_count = len(self.extra_digest) + len(old)

        digest = l3_fold(
            list(self.extra_digest) + [t.digest_line() for t in old],
            max_lines=self.max_digest_lines,
        )

        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        if digest:
            messages.append({
                "role": "user",
                "content": f"<earlier_context>\n{digest}\n</earlier_context>",
            })
        for t in raw:
            messages.extend(self._turn_messages(t))
        # 当前轮的 prompt 已经在上面由 _turn_messages 发出，这里只补「额外」
        # 的 user 文本（例如 history() 这类不带 turn 的调用）。
        if user_text and (not raw or raw[-1].prompt != user_text):
            messages.append({"role": "user", "content": user_text})

        # 记账
        self.budget.set_used("system", estimate_tokens(system_prompt))
        self.budget.set_used("earlier", estimate_tokens(digest))
        self.budget.set_used("raw", estimate_tokens(
            self._render_raw(raw) + (user_text if (not raw or raw[-1].prompt != user_text) else "")
        ))
        return messages

    # ------------------------------------------------------------------
    # 供工具/UI 观察
    # ------------------------------------------------------------------
    def digest_lines(self) -> List[str]:
        """已折叠（L3）区的摘要行。"""
        old, _raw = self._windows()
        return list(self.extra_digest) + [t.digest_line() for t in old]

    def all_digest_lines(self) -> List[str]:
        """全部摘要行（折叠区 + 原文窗口）—— L4 蒸馏要用完整轨迹。"""
        return list(self.extra_digest) + [t.digest_line() for t in self.turns]

    def stats(self) -> Dict[str, Any]:
        _old, raw = self._windows()
        return {
            "turn": self.turn_index,
            "total_turns": len(self.turns),
            "raw_window": len(raw),
            "folded": self.folded_count,
            "budget": self.budget.snapshot(),
            "key_info_chars": len(self.get_key_info()),
        }

    def reset(self) -> None:
        self.turns.clear()
        self.working.clear()
        self.extra_digest.clear()
        self.turn_index = 0
        self.folded_count = 0
