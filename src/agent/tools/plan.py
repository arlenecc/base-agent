"""计划工具：把「目标 + 路径」变成内核可依赖的一等公民状态。

对应 GenericAgent 的规划能力，但落点不同：计划不写在文件里，而是写进
``Session.working["plan"]`` —— 于是它每轮都由 ``anchor_prompt()`` 自动
回注，内核的反思门也能直接读它判断「任务是否真的做完了」。

四个操作：
  create  新建/整体替换计划（任务开始时调用一次）
  patch   局部更新：改某几步的 status / note
  append  追加步骤（执行中发现遗漏）
  clear   清空（任务收尾或用户要求重新开始）
"""
from __future__ import annotations

import copy
from typing import Any, Optional

from ..config import AgentConfig
from ..kernel.session import Session, normalize_plan
from .base import Tool, ToolRegistry, ToolResult

# patch 允许修改的字段（title 用 replace 语义单独处理，不进这里）
_PATCHABLE = ("status", "note")


def _session(registry: ToolRegistry) -> Optional[Session]:
    sess = getattr(registry, "session", None)
    return sess if isinstance(sess, Session) else None


class UpdatePlanTool(Tool):
    """Create or update the structured execution plan for the current task.

    开始一个任务前先调用 ``create`` 规划目标与路径；每完成一步用 ``patch``
    把对应步骤标为 done 并写下结论；发现遗漏用 ``append``；全部收尾用
    ``clear``。计划每轮自动回注给你的上下文，你也可以随时读它确认进度。
    """

    name = "update_plan"
    description = (
        "Manage the structured execution plan (goal + ordered steps). "
        "Call op='create' ONCE BEFORE starting a non-trivial task — the plan "
        "MUST be written into the execution plan before any actual tool call is "
        "issued, so progress is tracked from step one; "
        "op='patch' to mark a step done/blocked with its conclusion, "
        "op='append' when you discover a missing step, op='clear' when finished. "
        "Mark a step 'blocked' only after at least 2 different methods have "
        "failed (explain why in note) — a single failure is not a reason to "
        "give up on a step. "
        "The plan is re-injected into your context every turn — keep it accurate."
    )
    destructive = False
    # 参数填错时回给模型的示例（见 ToolRegistry.execute 的 usage_hint 处理）。
    # 本地小模型最常见的错误就是漏掉 op 直接调 update_plan({})，然后原样重试
    # 若干次；给一个能照抄的形状，多数情况一次就能自纠。
    usage_hint = (
        'update_plan({"op": "patch", "step_id": "1", "status": "done", '
        '"note": "结论"})\n'
        '  · op 必填：create / patch / append / clear\n'
        '  · create 另需 goal 与 steps（steps 每项形如 {"title": "..."}）\n'
        '  · patch 需要 status 和/或 note；只改某一步时务必带 step_id'
    )
    parameters = {
        "type": "object",
        "properties": {
            "op": {
                "type": "string",
                "enum": ["create", "patch", "append", "clear"],
                "description": "create=新建/替换; patch=更新步骤; append=追加步骤; clear=清空",
            },
            "goal": {
                "type": "string",
                "description": "一句话目标（create 时必填，≤300 字符）",
            },
            "steps": {
                "type": "array",
                "description": (
                    "create/append 的步骤列表。每项可以是字符串，或 "
                    "{title, status?, note?}；status ∈ pending/doing/done/blocked/skipped"
                ),
                "items": {"type": "object"},
            },
            "step_id": {
                "type": "string",
                "description": "patch 的目标步骤 id（不填则更新所有 status 匹配的步骤）",
            },
            "status": {
                "type": "string",
                "enum": ["pending", "doing", "done", "blocked", "skipped"],
                "description": (
                    "patch 的新状态。blocked 仅限多种方法都确认不可行时使用"
                    "（note 里写明原因）；一次失败不足以标记 blocked"
                ),
            },
            "note": {
                "type": "string",
                "description": "patch 的结论/原因（≤200 字符），例如失败原因或关键产出",
            },
        },
        "required": ["op"],
    }

    config: AgentConfig
    registry: ToolRegistry

    def bind(self, config: AgentConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry

    # ------------------------------------------------------------------
    def run(self, op: str, goal: str = "", steps: Any = None,
            step_id: str = "", status: str = "", note: str = "") -> ToolResult:
        # 本地模型的文本工具调用会把 steps 作为 JSON 字符串传入（双重编码）；
        # 统一在此解析成 list，create/append 都能正确消费。
        if isinstance(steps, str):
            _s = steps.strip()
            if _s:
                try:
                    import json as _json
                    steps = _json.loads(_s)
                except (ValueError, TypeError):
                    pass
        sess = _session(self.registry)
        if sess is None:
            return ToolResult(False, error="plan unavailable (no kernel session)")
        op = (op or "").strip().lower()

        if op == "clear":
            sess.clear_plan()
            self._notify(sess)
            return ToolResult(True, output="plan cleared")

        if op == "create":
            plan = normalize_plan({"goal": goal, "steps": steps if steps is not None else []})
            if plan is None:
                return ToolResult(
                    False,
                    error="create needs a goal and at least one non-empty step title",
                )
            # 新建计划时把第一步标为 doing：给模型一个明确的「当前位置」。
            # 仅当模型没有显式设定任何状态时才提升——否则会把模型标好的
            # doing/done 覆盖回 doing，计划面板显示错乱。
            if all(s["status"] == "pending" for s in plan["steps"]):
                plan["steps"][0]["status"] = "doing"
            sess.set_plan(plan)
            self._notify(sess)
            out = f"plan created: {len(plan['steps'])} step(s)"
            raw_n = len(steps) if isinstance(steps, list) else 1
            if raw_n > len(plan["steps"]):
                out += f"\n⚠ 步骤数超过上限 {len(plan['steps'])}，多出的 {raw_n - len(plan['steps'])} 步被丢弃"
            # 建计划不等于开工：模型最常见的失败模式是 create 完就把计划复述
            # 一遍然后收尾，等用户说「开始/继续」。这里直接在工具结果里下达
            # 开工指令，让它在同一条消息里就看到「下一步该干什么」。
            out += "\n▶ 现在立即开始执行第 1 步（调用相应工具），不要停下来询问是否开始。"
            return ToolResult(True, output=f"{out}\n{sess.render_plan()}")

        if op == "append":
            if not steps:
                return ToolResult(False, error="append needs steps")
            cur = sess.get_plan()
            base = cur or {"goal": goal, "steps": []}
            merged = list(base.get("steps", []))
            # 追加的步骤 id 从「现有数字 id 最大值 + 1」起，并跳过已占用 id：
            # 旧实现 id 非数字时回退 len(merged)+1，与既有数字 id 撞号后
            # patch(step_id=...) 会一次改中两步。
            used_ids = {str(s.get("id", "")) for s in merged}
            numeric = []
            for s in merged:
                try:
                    numeric.append(int(s.get("id", 0)))
                except (TypeError, ValueError):
                    continue
            next_id = (max(numeric) + 1) if numeric else 1
            appended = 0
            for s in (steps if isinstance(steps, list) else [steps]):
                if isinstance(s, dict):
                    title = str(s.get("title", "") or "").strip()
                    st = str(s.get("status", "") or "pending")
                    nt = str(s.get("note", "") or "").strip()
                elif isinstance(s, str):
                    title, st, nt = s.strip(), "pending", ""
                else:
                    continue
                if not title:
                    continue
                while str(next_id) in used_ids:
                    next_id += 1
                merged.append({"id": str(next_id), "title": title,
                               "status": st if st in ("pending", "doing", "done",
                                                      "blocked", "skipped") else "pending",
                               "note": nt})
                used_ids.add(str(next_id))
                next_id += 1
                appended += 1
            if not appended:
                return ToolResult(False, error="append produced no valid step")
            plan = normalize_plan({"goal": base.get("goal", "") or goal, "steps": merged})
            if plan is None:
                return ToolResult(False, error="append produced an empty plan")
            sess.set_plan(plan)
            dropped = len(merged) - len(plan["steps"])
            self._notify(sess)
            out = f"plan now has {len(plan['steps'])} step(s)"
            if dropped:
                # normalize_plan 保留 raw_steps[:20]，丢掉的是**最新**追加的步，
                # 旧文案写成「最早 N 步被丢弃」与事实相反，会误导模型去重做旧步。
                out += (f"\n⚠ 步骤数超过上限（20），最新追加的 {dropped} 步被丢弃；"
                        "请改用 update_plan(op='create') 重建一个更聚焦的计划")
            return ToolResult(True, output=f"{out}\n{sess.render_plan()}")

        if op == "patch":
            cur = sess.get_plan()
            if cur is None:
                return ToolResult(False, error="no plan to patch — call op='create' first")
            new_status = (status or "").strip().lower()
            if new_status and new_status not in ("pending", "doing", "done",
                                                 "blocked", "skipped"):
                return ToolResult(False, error=f"invalid status: {status!r}")
            if not new_status and not note:
                return ToolResult(False, error="patch needs status and/or note")
            target = (step_id or "").strip()
            # 深拷贝后再改：旧实现直接原地改 sess.get_plan() 返回的活字典，
            # normalize_plan 一旦拒绝（超长/全空）计划已处于半改状态。
            working = copy.deepcopy(cur)
            steps_list = working.get("steps", [])
            # 先筛目标集合，再统一写入。旧实现「无 step_id 时」会把 note 写到
            # 状态已匹配的步骤上再 break（[done,pending] + patch(status=done)
            # 污染第 1 步、漏掉第 2 步）；只给 note 时更会改写所有步骤。
            if target:
                targets = [s for s in steps_list if str(s.get("id", "")) == target]
            elif new_status:
                targets = [next((s for s in steps_list
                                 if s.get("status") != new_status), None)]
                targets = [s for s in targets if s is not None]
            else:
                # 只给 note：必须配 step_id，否则语义不明（改哪一步？）
                return ToolResult(
                    False,
                    error="patch with only 'note' requires 'step_id'",
                )
            if not targets:
                # 错误必须带上当前计划：模型最常见的失败就是记错 step_id
                #（计划被 create 替换后 id 变了）。只报 "no step matched"
                # 模型只能瞎猜重试；附上计划渲染它一次就能自纠。
                return ToolResult(
                    False,
                    error=(f"no step matched step_id={target!r}; "
                           f"current plan:\n{sess.render_plan()}"),
                )
            for s in targets:
                if new_status:
                    s["status"] = new_status
                if note:
                    s["note"] = note
            touched = len(targets)
            # 把上一步标 done 时，顺带把下一个 pending 步骤标成 doing，
            # 让「Now:」始终指向当前位置（否则模型要自己推断）。
            # 必须从**被标记的步骤之后**开始找 pending：旧实现从头找第一个
            # pending，当被标 done 的步骤前面还有 pending（被跳过的步骤）时，
            # 「Now:」会指回更早的步骤，模型据此反复做同一件事。
            if new_status == "done":
                target_ids = {s.get("id") for s in targets}
                last_idx = max(
                    (i for i, s in enumerate(steps_list) if s.get("id") in target_ids),
                    default=-1,
                )
                for s in steps_list[last_idx + 1:]:
                    if s.get("status") == "pending":
                        s["status"] = "doing"
                        break
            if not sess.set_plan(working):
                return ToolResult(False, error="patched plan failed validation")
            self._notify(sess)
            remaining = len(sess.plan_open_steps())
            return ToolResult(
                True,
                output=(f"patched {touched} step(s); {remaining} open step(s) left\n"
                        f"{sess.render_plan()}"),
            )

        return ToolResult(False, error=f"unknown op: {op!r}")

    # ------------------------------------------------------------------
    def _notify(self, sess: Session) -> None:
        """把计划变更推给 UI（面板实时刷新）。失败不影响计划本身已写入。"""
        try:
            cb = getattr(self.registry, "callbacks", None)
            hook = getattr(cb, "on_plan_updated", None)
            if callable(hook):
                hook(sess.get_plan())
        except Exception:
            pass
