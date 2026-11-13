"""定时任务工具（schedule_tools）：通过对话创建/管理 APScheduler 任务。

支持两类任务：
* ``kind=agent``：自然语言任务，由主模型在常驻对话线程里执行，结果直接流式
  显示在对话框（适合总结、检索、写报告等需要推理的任务）；
* ``kind=tool``：直接调用已注册工具（code_run / shell_run / MCP 自定义工具
  等），执行结果以「定时任务」气泡显示（适合固定脚本/命令）。

触发规则六选一：``cron``（五段 crontab）/ ``interval_minutes``（≥1）/
``run_at``（一次性 ISO 时间）/ ``weekday`` + ``time``（每周几 + HH:MM）/
``date`` + ``time``（指定日期 + HH:MM，一次性）/
``delay_minutes`` 或 ``delay_hours``（从当前时间后推，一次性）。
"""
from __future__ import annotations

import json
from typing import Any, Dict

from .base import Tool, ToolResult

_OP_HELP = (
    "可用 op：create（创建）/ list（查看）/ delete（删除）/ run_now（立即执行一次）"
    "/ pause（暂停）/ resume（恢复）。")
_CREATE_HELP = (
    "create 示例：\n"
    "- 每天 9 点让模型总结知识库新增内容："
    "{op:\"create\", name:\"每日知识总结\", kind:\"agent\", cron:\"0 9 * * *\", "
    "prompt:\"总结知识库昨天新增的文档要点\"}\n"
    "- 每周一 9:00："
    "{op:\"create\", name:\"周一早报\", kind:\"agent\", weekday:\"周一\", "
    "time:\"09:00\", prompt:\"总结上周新增内容\"}\n"
    "- 指定日期 2026-11-20 14:30 执行一次："
    "{op:\"create\", name:\"项目复盘\", kind:\"agent\", date:\"2026-11-20\", "
    "time:\"14:30\", prompt:\"整理项目复盘\"}\n"
    "- 30 分钟后执行一次："
    "{op:\"create\", name:\"稍后提醒\", kind:\"agent\", delay_minutes:30, "
    "prompt:\"提醒我检查构建结果\"}\n"
    "- 2 小时后跑一段代码："
    "{op:\"create\", name:\"延迟巡检\", kind:\"tool\", delay_hours:2, "
    "tool_name:\"code_run\", tool_args:{\"code\":\"print('ok')\"}}\n"
    "- 每 30 分钟跑一段代码："
    "{op:\"create\", name:\"巡检\", kind:\"tool\", interval_minutes:30, "
    "tool_name:\"code_run\", tool_args:{\"code\":\"print('ok')\"}}")


def _job_lines(spec: Dict[str, Any]) -> str:
    from ..scheduler.service import describe_trigger

    kind = spec.get("kind") or "agent"
    target = (f"prompt：{(spec.get('prompt') or '')[:80]}" if kind == "agent"
              else f"工具：{spec.get('tool_name')} {json.dumps(spec.get('tool_args') or {}, ensure_ascii=False)[:120]}")
    status = spec.get("last_status") or "—"
    return (
        f"- 「{spec.get('name')}」 id={spec.get('id')}\n"
        f"  触发：{describe_trigger(spec)}\n"
        f"  类型：{kind}（{target}）\n"
        f"  下次：{spec.get('next_run') or '（已暂停或无后续触发）'}\n"
        f"  上次：{spec.get('last_run_at') or '未执行'}（{status}）"
    )


class ScheduleTool(Tool):
    """创建/查看/删除/立即执行定时任务。"""

    name = "schedule_tools"
    description = (
        "管理定时任务（APScheduler）。用户说「每天/每周X/某月某日/过 N 分钟"
        "或 N 小时后/每隔 N 分钟」要自动做某事时使用。\n"
        "任务类型 kind：\n"
        "- agent：自然语言任务，由主模型执行（总结、检索、写报告等需要推理的任务）；\n"
        "- tool：直接调用已注册工具，适合固定脚本/命令——code_run 参数 "
        "{\"code\":\"...\"}，shell_run 参数 {\"command\":\"...\"}，也可用任意 "
        "MCP/自定义工具名与参数。\n"
        "触发规则六选一：cron（五段，如 \"0 9 * * *\"）、interval_minutes（≥1）、"
        "run_at（一次性 ISO 时间）、weekday+time（每周几 + HH:MM）、"
        "date+time（指定日期 + HH:MM，一次性）、delay_minutes/delay_hours"
        "（从现在起后推，一次性）。\n"
        "创建成功后把任务名、触发规则、下次执行时间告诉用户；任务到点自动执行，"
        "结果会显示在对话里。" + _OP_HELP
    )
    parameters = {
        "type": "object",
        "properties": {
            "op": {
                "type": "string",
                "enum": ["create", "list", "delete", "run_now", "pause", "resume"],
                "description": "操作类型",
            },
            "task_id": {
                "type": "string",
                "description": "delete/run_now/pause/resume 的目标（任务 ID 或任务名）",
            },
            "name": {"type": "string", "description": "任务名（创建时必填，简短中文）"},
            "kind": {
                "type": "string",
                "enum": ["agent", "tool"],
                "description": "任务类型：agent=主模型执行；tool=直接调用工具",
            },
            "cron": {
                "type": "string",
                "description": "五段 crontab（分 时 日 月 周），如 \"0 9 * * *\"",
            },
            "interval_minutes": {
                "type": "integer",
                "description": "每隔多少分钟执行一次（≥1）",
            },
            "run_at": {
                "type": "string",
                "description": "一次性执行时间（ISO，如 \"2026-11-14 09:00\"）",
            },
            "weekday": {
                "type": "string",
                "description": "每周几（周一/星期一/mon/1-7，1=周一），需配合 time",
            },
            "time": {
                "type": "string",
                "description": "执行时间 HH:MM（配合 weekday 或 date）",
            },
            "date": {
                "type": "string",
                "description": "指定日期（YYYY-MM-DD；也支持 MM-DD / M月D日），配合 time，一次性",
            },
            "delay_minutes": {
                "type": "number",
                "description": "从现在起多少分钟后执行（一次性，≥1）",
            },
            "delay_hours": {
                "type": "number",
                "description": "从现在起多少小时后执行（一次性，至少 1 分钟）",
            },
            "prompt": {
                "type": "string",
                "description": "kind=agent 时模型要执行的任务内容",
            },
            "tool_name": {
                "type": "string",
                "description": "kind=tool 时要调用的工具名（code_run/shell_run/MCP 工具等）",
            },
            "tool_args": {
                "type": "object",
                "description": "kind=tool 时传给工具的参数对象",
            },
        },
        "required": ["op"],
    }
    destructive = False

    def bind(self, config: Any, registry: Any) -> None:
        # 不持有 registry：任务执行走 UI 注入的 SchedulerService（统一入口）。
        pass

    def run(self, op: str = "", task_id: str = "", name: str = "",
            kind: str = "agent", cron: str = "", interval_minutes: Any = None,
            run_at: str = "", weekday: str = "", time: str = "",
            date: str = "", delay_minutes: Any = None,
            delay_hours: Any = None, prompt: str = "", tool_name: str = "",
            tool_args: Any = None, **kwargs) -> ToolResult:
        from ..scheduler import get_scheduler
        from ..scheduler.service import SchedulerError

        svc = get_scheduler()
        if svc is None:
            return ToolResult(False, error="定时任务服务未启动（应用未就绪），请稍后再试。")
        if not svc.available:
            return ToolResult(False, error=svc.unavailable_reason)

        op = str(op or "").strip().lower()
        if op not in ("create", "list", "delete", "run_now", "pause", "resume"):
            return ToolResult(False, error=f"未知 op {op!r}。{_OP_HELP}")

        # 模型（本地小模型）常把 tool_args 传成 JSON 字符串；统一归一化。
        if isinstance(tool_args, str):
            raw = tool_args.strip()
            if raw:
                try:
                    tool_args = json.loads(raw)
                except json.JSONDecodeError:
                    return ToolResult(
                        False,
                        error=("tool_args 不是合法 JSON 对象；"
                               "例：{\"code\": \"print('ok')\"}。"))
            else:
                tool_args = {}
        if tool_args is not None and not isinstance(tool_args, dict):
            return ToolResult(False, error="tool_args 必须是 JSON 对象。")

        try:
            if op == "create":
                from ..scheduler.service import describe_trigger

                spec = svc.create_job(
                    name=name, kind=kind, cron=cron,
                    interval_minutes=interval_minutes, run_at=run_at,
                    weekday=weekday, time=time, date=date,
                    delay_minutes=delay_minutes, delay_hours=delay_hours,
                    prompt=prompt, tool_name=tool_name, tool_args=tool_args)
                nxt = spec.get("next_run") or "（等待调度器计算）"
                return ToolResult(success=True, output=(
                    f"✅ 已创建定时任务「{spec['name']}」（{spec['id']}）\n"
                    f"触发：{describe_trigger(spec)}\n"
                    f"下次执行：{nxt}\n"
                    "到点后会自动执行并把结果展示在对话中。"))
            if op == "list":
                jobs = svc.list_jobs()
                if not jobs:
                    return ToolResult(success=True, output="当前没有定时任务。")
                return ToolResult(success=True, output=(
                    f"当前共 {len(jobs)} 个定时任务：\n"
                    + "\n".join(_job_lines(j) for j in jobs)))
            if op == "delete":
                deleted = svc.delete_job(task_id or name)
                return ToolResult(success=True, output=f"✅ 已删除定时任务「{deleted}」。")
            if op == "run_now":
                ran = svc.run_now(task_id or name)
                return ToolResult(success=True, output=(
                    f"✅ 已触发任务「{ran}」立即执行一次（结果稍后显示在对话中）。"))
            if op == "pause":
                spec = svc.pause_job(task_id or name)
                return ToolResult(success=True, output=f"⏸ 已暂停任务「{spec['name']}」。")
            spec = svc.resume_job(task_id or name)
            return ToolResult(success=True, output=(
                f"▶️ 已恢复任务「{spec['name']}」，下次执行：{spec.get('next_run') or '（待计算）'}。"))
        except SchedulerError as e:
            hint = "\n" + _CREATE_HELP if op == "create" else ""
            return ToolResult(False, error=f"{e}{hint}")
        except Exception as e:
            return ToolResult(False, error=f"定时任务操作失败：{type(e).__name__}: {e}")
