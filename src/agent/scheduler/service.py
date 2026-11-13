"""定时任务服务：APScheduler 封装 + 任务定义持久化。

职责边界：
* **只做调度**：cron / interval / 一次性（ISO / 指定日期 / 后推 N 分钟或小时）
  以及每周几 + 时间；不关心任务内容怎么执行。
* **两类执行器**（由 UI 注入）：
    - ``agent_submitter(name, prompt)``：自然语言任务，投递到常驻对话线程，
      由主模型执行（结果按普通回合流式输出到对话框）；
    - ``tool_executor(tool_name, args)``：直接调用已注册工具（code_run /
      shell_run / MCP 自定义工具等），结果经 ``notify`` 展示。
* **持久化**：任务定义存 ``<workspace>/.agent/scheduler_jobs.json``（原子写），
  应用重启后自动恢复；APScheduler 用内存 JobStore，不引入 SQLAlchemy。
* **线程模型**：所有任务在 APScheduler 的 daemon 线程池里跑；``notify`` /
  ``agent_submitter`` 必须线程安全（UI 侧用 Qt 跨线程信号实现）。
"""
from __future__ import annotations

import copy
import json
import logging
import os
import tempfile
import threading
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_STATE_FILENAME = "scheduler_jobs.json"
_MIN_INTERVAL_MINUTES = 1.0
_MAX_RESULT_CHARS = 4000
_MISFIRE_GRACE_SECONDS = 300


class SchedulerError(ValueError):
    """任务参数/状态错误（工具层直接把它转成给模型看的错误文案）。"""


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _local_tz():
    return datetime.now().astimezone().tzinfo


def _make_trigger(spec: Dict[str, Any]):
    """任务定义 → APScheduler trigger（导入延迟到调用时，便于缺依赖时优雅降级）。"""
    trig = spec.get("trigger") or {}
    tz = _local_tz()
    kind = trig.get("type")
    if kind == "cron":
        from apscheduler.triggers.cron import CronTrigger
        return CronTrigger.from_crontab(str(trig.get("cron") or ""), timezone=tz)
    if kind == "interval":
        from apscheduler.triggers.interval import IntervalTrigger
        return IntervalTrigger(minutes=float(trig.get("minutes") or 1), timezone=tz)
    from apscheduler.triggers.date import DateTrigger
    return DateTrigger(run_date=datetime.fromisoformat(str(trig.get("run_at"))),
                       timezone=tz)


# 周几别名 → (APScheduler cron 名, 中文标签)。数字按 1=周一 … 7=周日。
_WEEKDAYS = {
    "1": ("mon", "一"), "mon": ("mon", "一"), "monday": ("mon", "一"),
    "周一": ("mon", "一"), "星期一": ("mon", "一"), "礼拜一": ("mon", "一"),
    "2": ("tue", "二"), "tue": ("tue", "二"), "tuesday": ("tue", "二"),
    "周二": ("tue", "二"), "星期二": ("tue", "二"), "礼拜二": ("tue", "二"),
    "3": ("wed", "三"), "wed": ("wed", "三"), "wednesday": ("wed", "三"),
    "周三": ("wed", "三"), "星期三": ("wed", "三"), "礼拜三": ("wed", "三"),
    "4": ("thu", "四"), "thu": ("thu", "四"), "thursday": ("thu", "四"),
    "周四": ("thu", "四"), "星期四": ("thu", "四"), "礼拜四": ("thu", "四"),
    "5": ("fri", "五"), "fri": ("fri", "五"), "friday": ("fri", "五"),
    "周五": ("fri", "五"), "星期五": ("fri", "五"), "礼拜五": ("fri", "五"),
    "6": ("sat", "六"), "sat": ("sat", "六"), "saturday": ("sat", "六"),
    "周六": ("sat", "六"), "星期六": ("sat", "六"), "礼拜六": ("sat", "六"),
    "7": ("sun", "日"), "sun": ("sun", "日"), "sunday": ("sun", "日"),
    "周日": ("sun", "日"), "周天": ("sun", "日"), "星期日": ("sun", "日"),
    "星期天": ("sun", "日"), "礼拜日": ("sun", "日"), "礼拜天": ("sun", "日"),
    "日": ("sun", "日"), "天": ("sun", "日"),
}


def _parse_weekday(value: Any) -> tuple:
    """解析「每周几」：周一/星期一/礼拜一/mon/1-7 → (cron 名, 中文标签)。"""
    key = "".join(str(value or "").strip().lower().split())
    item = _WEEKDAYS.get(key)
    if item is None:
        raise SchedulerError(
            f"weekday 无法识别（{value!r}）；支持 周一/星期一/礼拜一/mon/1-7。")
    return item


def _parse_hhmm(value: Any) -> tuple:
    """解析 HH:MM（也接受全角冒号）→ (hour, minute)。"""
    import re

    raw = str(value or "").strip().replace("：", ":")
    m = re.match(r"^(\d{1,2}):(\d{1,2})$", raw)
    if not m:
        raise SchedulerError(
            f"time 格式无效（{value!r}）；请用 HH:MM，如 \"09:30\"。")
    hour, minute = int(m.group(1)), int(m.group(2))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise SchedulerError(f"time 超出范围（{value!r}）；小时 0-23、分钟 0-59。")
    return hour, minute


def _parse_date(value: Any):
    """解析指定日期：2026-11-20 / 2026/11/20 / 2026年11月20日 / 11月20日。"""
    raw = str(value or "").strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y年%m月%d日", "%Y%m%d"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    # 不带年份：按今年解释；已过去的月日顺延到下一年。
    import re

    m = (re.match(r"^(\d{1,2})[-/.](\d{1,2})$", raw)
         or re.match(r"^(\d{1,2})月(\d{1,2})日$", raw))
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        today = datetime.now(_local_tz()).date()
        try:
            when_day = datetime(today.year, month, day).date()
            if when_day < today:
                when_day = datetime(today.year + 1, month, day).date()
        except ValueError:
            raise SchedulerError(f"date 日期无效（{value!r}）。")
        return when_day
    raise SchedulerError(
        f"date 格式无效（{value!r}）；请用 YYYY-MM-DD，如 \"2026-11-20\"。")


def _num_text(value: float) -> str:
    return f"{value:g}"


def _build_trigger(cron: str, interval_minutes: Any, run_at: str,
                   weekday: Any = "", time_of_day: Any = "",
                   date_text: Any = "", delay_minutes: Any = None,
                   delay_hours: Any = None) -> Dict[str, Any]:
    """校验并归一化触发规则（六种方式互斥）。"""
    cron = " ".join(str(cron or "").split())
    interval_given = interval_minutes not in (None, "")
    run_at = str(run_at or "").strip()
    weekday = str(weekday or "").strip()
    time_of_day = str(time_of_day or "").strip()
    date_text = str(date_text or "").strip()
    delay_min_given = delay_minutes not in (None, "")
    delay_hour_given = delay_hours not in (None, "")
    provided = [
        bool(cron), interval_given, bool(run_at), bool(weekday),
        bool(date_text), delay_min_given or delay_hour_given,
    ]
    if sum(provided) != 1:
        raise SchedulerError(
            "触发规则必须且只能指定一种：cron（五段 crontab）/ "
            "interval_minutes（≥1）/ run_at（一次性 ISO 时间）/ "
            "weekday + time（每周几 + HH:MM）/ date + time（指定日期 + HH:MM）/ "
            "delay_minutes 或 delay_hours（从现在起后推）。")
    if time_of_day and not (weekday or date_text):
        raise SchedulerError(
            "time 需要与 weekday（每周几）或 date（指定日期）一起使用。")
    tz = _local_tz()
    now = datetime.now(tz)
    if cron:
        from apscheduler.triggers.cron import CronTrigger
        if len(cron.split()) != 5:
            raise SchedulerError(
                f"cron 必须是五段（分 时 日 月 周），收到 {cron!r}；"
                "例：每天 9 点 = \"0 9 * * *\"。")
        try:
            CronTrigger.from_crontab(cron, timezone=tz)
        except Exception as e:
            raise SchedulerError(f"cron 表达式无效（{cron!r}）：{e}") from e
        return {"type": "cron", "cron": cron}
    if interval_given:
        try:
            minutes = float(interval_minutes)
        except (TypeError, ValueError):
            raise SchedulerError("interval_minutes 必须是数字（单位：分钟）。")
        if minutes < _MIN_INTERVAL_MINUTES:
            raise SchedulerError(
                f"interval_minutes 不能小于 {_MIN_INTERVAL_MINUTES:g} 分钟"
                "（过小会让任务连续触发）。")
        if minutes == int(minutes):
            minutes = int(minutes)
        return {"type": "interval", "minutes": minutes}
    if run_at:
        try:
            when = datetime.fromisoformat(run_at)
        except ValueError:
            raise SchedulerError(
                f"run_at 时间格式无效（{run_at!r}）；"
                "请用 ISO 格式，如 \"2026-11-14 09:00\" 或 \"2026-11-14T09:00:00\"。")
        if when.tzinfo is None:
            when = when.replace(tzinfo=tz)
        if when <= now:
            raise SchedulerError("run_at 必须是未来时间（一次性任务不能定在过去）。")
        return {"type": "date", "run_at": when.isoformat(timespec="seconds")}
    if weekday:
        if not time_of_day:
            raise SchedulerError("weekday（每周几）需要同时提供 time（HH:MM）。")
        dow, label = _parse_weekday(weekday)
        hour, minute = _parse_hhmm(time_of_day)
        expr = f"{minute} {hour} * * {dow}"
        from apscheduler.triggers.cron import CronTrigger
        try:
            CronTrigger.from_crontab(expr, timezone=tz)
        except Exception as e:
            raise SchedulerError(f"每周任务规则无效：{e}") from e
        return {
            "type": "cron",
            "cron": expr,
            "desc": f"每周{label} {hour:02d}:{minute:02d}（本地时区）",
        }
    if date_text:
        if not time_of_day:
            raise SchedulerError("date（指定日期）需要同时提供 time（HH:MM）。")
        day = _parse_date(date_text)
        hour, minute = _parse_hhmm(time_of_day)
        when = datetime(day.year, day.month, day.day, hour, minute, tzinfo=tz)
        if when <= now:
            raise SchedulerError("date + time 必须是未来时间（不能定在过去）。")
        return {
            "type": "date",
            "run_at": when.isoformat(timespec="seconds"),
            "desc": f"指定时间 {when:%Y-%m-%d %H:%M}（本地时区）",
        }
    if delay_min_given and delay_hour_given:
        raise SchedulerError("delay_minutes 与 delay_hours 只能二选一。")
    try:
        if delay_min_given:
            amount = float(delay_minutes)
            minutes = amount
            label = f"{_num_text(amount)} 分钟后"
        else:
            amount = float(delay_hours)
            minutes = amount * 60
            label = f"{_num_text(amount)} 小时后"
    except (TypeError, ValueError):
        raise SchedulerError("delay_minutes / delay_hours 必须是数字。")
    if minutes < 1:
        raise SchedulerError("后推时间不能小于 1 分钟。")
    when = now + timedelta(minutes=minutes)
    return {
        "type": "date",
        "run_at": when.isoformat(timespec="seconds"),
        "desc": f"{label}（{when:%Y-%m-%d %H:%M}）",
    }


def describe_trigger(spec: Dict[str, Any]) -> str:
    """任务定义 → 人可读的触发描述。"""
    trig = spec.get("trigger") or {}
    if trig.get("desc"):
        return str(trig["desc"])
    kind = trig.get("type")
    if kind == "cron":
        return f"cron {trig.get('cron')}（本地时区）"
    if kind == "interval":
        return f"每 {trig.get('minutes')} 分钟"
    return f"一次性 {trig.get('run_at')}"


class SchedulerService:
    """APScheduler 调度器 + 任务定义 JSON 持久化。"""

    def __init__(
        self,
        workspace: str,
        *,
        tool_executor: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        agent_submitter: Optional[Callable[[str, str], None]] = None,
        notify: Optional[Callable[[str, str], None]] = None,
        log: Optional[Callable[[str], None]] = None,
        auto_start: bool = True,
    ):
        self._workspace = str(workspace or "")
        self._path = self._state_path(self._workspace)
        self._tool_executor = tool_executor
        self._agent_submitter = agent_submitter
        self._notify = notify
        self._log_fn = log
        self._lock = threading.RLock()
        self._jobs: Dict[str, Dict[str, Any]] = {}
        # 正在执行的 job id（防止同一任务叠跑；线程为 daemon，退出应用不被
        # 长任务阻塞——APScheduler 自带的线程池线程在解释器退出时会被 join）。
        self._running: set = set()
        self._sched = None
        self._start_error = ""
        self._load()
        if auto_start:
            self.start()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    @staticmethod
    def _state_path(workspace: str) -> str:
        return os.path.join(str(workspace or ""), ".agent", _STATE_FILENAME)

    @property
    def available(self) -> bool:
        return self._sched is not None

    @property
    def unavailable_reason(self) -> str:
        if self._sched is not None:
            return ""
        return self._start_error or "定时任务服务未启动"

    def start(self) -> None:
        """启动调度器并从磁盘恢复任务（幂等）。"""
        if self._sched is not None:
            return
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
        except Exception as e:  # 依赖缺失不阻塞主程序
            self._start_error = (
                f"定时任务需要 APScheduler（pip install APScheduler）：{e}")
            self._log(f"[scheduler] ⚠ {self._start_error}")
            return
        sched = BackgroundScheduler(timezone=_local_tz())
        try:
            from apscheduler.events import EVENT_JOB_MISSED
            sched.add_listener(self._on_job_missed, EVENT_JOB_MISSED)
        except Exception:
            pass
        with self._lock:
            self._prune_stale_once_locked()
            for spec in list(self._jobs.values()):
                if not spec.get("enabled", True):
                    continue
                try:
                    self._add_job_locked(sched, spec)
                except Exception as e:
                    self._log(f"[scheduler] ⚠ 恢复任务「{spec.get('name')}」失败: {e}")
        sched.start()
        self._sched = sched
        self._start_error = ""
        n = len(self._jobs)
        if n:
            self._log(f"[scheduler] 已启动，恢复 {n} 个定时任务")

    def shutdown(self, wait: bool = False) -> None:
        sched, self._sched = self._sched, None
        if sched is not None:
            try:
                sched.shutdown(wait=wait)
            except Exception:
                pass
            self._log("[scheduler] 已停止")

    def reconfigure(self, workspace: str) -> None:
        """工作区变化后切换任务文件并重启调度器（路径未变则不动）。"""
        new_path = self._state_path(workspace)
        if os.path.abspath(new_path) == os.path.abspath(self._path):
            return
        self.shutdown(wait=False)
        self._workspace = str(workspace or "")
        self._path = new_path
        with self._lock:
            self._jobs = {}
        self._load()
        self.start()

    # ------------------------------------------------------------------
    # 任务 CRUD
    # ------------------------------------------------------------------

    def create_job(
        self,
        *,
        name: str,
        kind: str = "agent",
        cron: str = "",
        interval_minutes: Any = None,
        run_at: str = "",
        weekday: str = "",
        time: str = "",
        date: str = "",
        delay_minutes: Any = None,
        delay_hours: Any = None,
        prompt: str = "",
        tool_name: str = "",
        tool_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """创建任务并登记到调度器，返回任务描述 dict。"""
        if self._sched is None:
            raise SchedulerError(self.unavailable_reason)
        name = str(name or "").strip()
        if not name:
            raise SchedulerError("任务名 name 不能为空（用一句简短中文描述）。")
        kind = str(kind or "agent").strip().lower()
        if kind not in ("agent", "tool"):
            raise SchedulerError("kind 只能是 agent（自然语言任务）或 tool（直接调用工具）。")
        prompt = str(prompt or "").strip()
        tool_name = str(tool_name or "").strip()
        if kind == "agent" and not prompt:
            raise SchedulerError("kind=agent 时必须提供 prompt（要模型执行的任务内容）。")
        if kind == "tool" and not tool_name:
            raise SchedulerError("kind=tool 时必须提供 tool_name（要调用的工具名）。")
        if tool_args is not None and not isinstance(tool_args, dict):
            raise SchedulerError("tool_args 必须是 JSON 对象（如 {\"code\": \"...\"}）。")
        trigger = _build_trigger(
            cron, interval_minutes, run_at, weekday=weekday, time_of_day=time,
            date_text=date, delay_minutes=delay_minutes,
            delay_hours=delay_hours)
        with self._lock:
            if any(j.get("name") == name for j in self._jobs.values()):
                raise SchedulerError(
                    f"已存在同名任务「{name}」，请先删除或换个名字。")
            job_id = "sched_" + uuid.uuid4().hex[:8]
            spec = {
                "id": job_id,
                "name": name,
                "kind": kind,
                "trigger": trigger,
                "prompt": prompt,
                "tool_name": tool_name,
                "tool_args": dict(tool_args or {}),
                "enabled": True,
                "created_at": _now_iso(),
                "last_run_at": "",
                "last_status": "",
                "last_result": "",
            }
            self._jobs[job_id] = spec
            try:
                self._add_job_locked(self._sched, spec)
            except Exception:
                self._jobs.pop(job_id, None)
                raise
            self._save_locked()
        self._log(f"[scheduler] 已创建任务「{name}」({describe_trigger(spec)})")
        return self.describe(job_id)

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            out = []
            for spec in self._jobs.values():
                item = copy.deepcopy(spec)
                item["next_run"] = self._next_run_locked(item["id"])
                out.append(item)
        return sorted(out, key=lambda s: s.get("created_at") or "")

    def describe(self, job_id_or_name: str) -> Dict[str, Any]:
        with self._lock:
            spec = self._resolve_locked(job_id_or_name)
            item = copy.deepcopy(spec)
            item["next_run"] = self._next_run_locked(spec["id"])
            return item

    def delete_job(self, job_id_or_name: str, silent: bool = False) -> str:
        with self._lock:
            spec = self._resolve_locked(job_id_or_name)
            self._jobs.pop(spec["id"], None)
            if self._sched is not None:
                try:
                    self._sched.remove_job(spec["id"])
                except Exception:
                    pass
            self._save_locked()
        if not silent:
            self._log(f"[scheduler] 已删除任务「{spec.get('name')}」")
        return str(spec.get("name") or "")

    def pause_job(self, job_id_or_name: str) -> Dict[str, Any]:
        with self._lock:
            spec = self._resolve_locked(job_id_or_name)
            if self._sched is not None:
                try:
                    self._sched.pause_job(spec["id"])
                except Exception:
                    pass
            spec["enabled"] = False
            self._save_locked()
        self._log(f"[scheduler] 已暂停任务「{spec.get('name')}」")
        return self.describe(spec["id"])

    def resume_job(self, job_id_or_name: str) -> Dict[str, Any]:
        with self._lock:
            spec = self._resolve_locked(job_id_or_name)
            spec["enabled"] = True
            if self._sched is not None:
                # 暂停中的任务仍在调度器里 → resume；启动时被跳过（或添加
                # 失败）的任务不在调度器里 → 直接补登记，否则 resume 会
                # 「静默成功」但永不触发，直到下次重启才恢复。
                if self._sched.get_job(spec["id"]) is not None:
                    try:
                        self._sched.resume_job(spec["id"])
                    except Exception:
                        pass
                else:
                    try:
                        self._add_job_locked(self._sched, spec)
                    except Exception as e:
                        self._log(f"[scheduler] ⚠ 恢复任务「{spec.get('name')}」"
                                  f"重新登记失败: {e}")
            self._save_locked()
        self._log(f"[scheduler] 已恢复任务「{spec.get('name')}」")
        return self.describe(spec["id"])

    def run_now(self, job_id_or_name: str) -> str:
        """立即在后台线程执行一次（不改变原定触发计划）。"""
        with self._lock:
            spec = self._resolve_locked(job_id_or_name)
            job_id = spec["id"]
            name = str(spec.get("name") or "")
        self._dispatch_job(job_id)
        self._log(f"[scheduler] 立即执行任务「{name}」")
        return name

    # ------------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------------

    def _dispatch_job(self, job_id: str) -> None:
        """把一次执行丢到 daemon 线程（APScheduler 的触发线程立即返回）。

        这样即使应用退出时任务仍在跑（code_run/shell_run 可能数百秒），
        解释器也不会被 APScheduler 线程池的 join 卡住；``_running`` 保证
        同一任务不会叠跑。
        """
        with self._lock:
            if job_id in self._running:
                return
            self._running.add(job_id)
        threading.Thread(target=self._run_and_release, args=(job_id,),
                         name=f"sched-{job_id}", daemon=True).start()

    def _run_and_release(self, job_id: str) -> None:
        try:
            self._execute_job(job_id)
        finally:
            with self._lock:
                self._running.discard(job_id)

    def _execute_job(self, job_id: str) -> None:
        with self._lock:
            spec = copy.deepcopy(self._jobs.get(job_id))
        if not spec:
            return
        started = _now_iso()
        status, text = "ok", ""
        try:
            if spec.get("kind") == "agent":
                if self._agent_submitter is None:
                    raise SchedulerError("主模型执行器不可用（应用未就绪）")
                self._agent_submitter(str(spec["name"]), str(spec.get("prompt") or ""))
                status = "submitted"
            else:
                if self._tool_executor is None:
                    raise SchedulerError("工具执行器不可用（应用未就绪）")
                result = self._tool_executor(
                    str(spec.get("tool_name") or ""),
                    dict(spec.get("tool_args") or {}))
                ok = bool(getattr(result, "success", True))
                text = (result.to_message() if hasattr(result, "to_message")
                        else str(result)).strip()
                status = "ok" if ok else "error"
                if len(text) > _MAX_RESULT_CHARS:
                    text = text[:_MAX_RESULT_CHARS] + "…（已截断）"
                self._notify_result(str(spec["name"]), text or "（无输出）")
        except Exception as e:  # 单次失败不影响后续调度
            status = "error"
            text = f"执行失败：{type(e).__name__}: {e}"
            logger.warning("scheduler job %s failed: %s", job_id, e)
            self._notify_result(str(spec.get("name") or job_id), text)
        finally:
            with self._lock:
                cur = self._jobs.get(job_id)
                if cur is not None:
                    cur["last_run_at"] = started
                    cur["last_status"] = status
                    cur["last_result"] = text[:500]
                    self._save_locked()
        # 一次性任务执行完即移除（APScheduler 的 DateTrigger 不会再触发）
        if (spec.get("trigger") or {}).get("type") == "date":
            try:
                self.delete_job(job_id, silent=True)
            except SchedulerError:
                pass  # 执行期间被用户删除：已经是目标状态

    def _notify_result(self, name: str, text: str) -> None:
        if self._notify is None:
            return
        try:
            self._notify(name, text)
        except Exception as e:
            logger.warning("scheduler notify failed: %s", e)

    def _on_job_missed(self, event) -> None:
        """一次性任务错过执行窗口（应用关闭期间到点）→ 清理，避免堆积。"""
        job_id = str(getattr(event, "job_id", "") or "")
        if not job_id:
            return
        with self._lock:
            spec = self._jobs.get(job_id)
            if not spec or (spec.get("trigger") or {}).get("type") != "date":
                return
            self._jobs.pop(job_id, None)
            self._save_locked()
        self._log(f"[scheduler] 一次性任务「{spec.get('name')}」已错过执行窗口，已清理")

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _add_job_locked(self, sched, spec: Dict[str, Any]) -> None:
        sched.add_job(
            self._dispatch_job,
            _make_trigger(spec),
            args=[spec["id"]],
            id=spec["id"],
            name=spec.get("name") or spec["id"],
            replace_existing=True,
            coalesce=True,
            max_instances=1,
            misfire_grace_time=_MISFIRE_GRACE_SECONDS,
        )

    def _prune_stale_once_locked(self) -> None:
        """启动时清掉已过期的一次性任务（错过执行窗口就不再补跑）。"""
        now = datetime.now(_local_tz())
        stale = []
        for job_id, spec in self._jobs.items():
            trig = spec.get("trigger") or {}
            if trig.get("type") != "date":
                continue
            try:
                when = datetime.fromisoformat(str(trig.get("run_at")))
            except (TypeError, ValueError):
                stale.append(job_id)
                continue
            if when.tzinfo is None:
                when = when.replace(tzinfo=_local_tz())
            if (now - when).total_seconds() > _MISFIRE_GRACE_SECONDS:
                stale.append(job_id)
        for job_id in stale:
            self._jobs.pop(job_id, None)
        if stale:
            self._save_locked()
            self._log(f"[scheduler] 已清理 {len(stale)} 个过期的一次性任务")

    def _next_run_locked(self, job_id: str) -> str:
        if self._sched is None:
            return ""
        try:
            job = self._sched.get_job(job_id)
            nxt = getattr(job, "next_run_time", None) if job else None
            return nxt.isoformat(timespec="seconds") if nxt else ""
        except Exception:
            return ""

    def _resolve_locked(self, job_id_or_name: str) -> Dict[str, Any]:
        key = str(job_id_or_name or "").strip()
        if not key:
            raise SchedulerError("缺少 task_id（任务 ID 或任务名）。")
        spec = self._jobs.get(key)
        if spec is not None:
            return spec
        for j in self._jobs.values():
            if j.get("name") == key:
                return j
        raise SchedulerError(f"未找到任务「{key}」（可先 op=list 查看）。")

    def _save_locked(self) -> None:
        data = {"version": 1, "jobs": list(self._jobs.values())}
        try:
            directory = os.path.dirname(self._path)
            os.makedirs(directory, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".scheduler.", suffix=".tmp",
                                       dir=directory)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                os.replace(tmp, self._path)
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except OSError as e:
            logger.warning("scheduler: 保存任务失败: %s", e)

    def _load(self) -> None:
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        jobs = raw.get("jobs") if isinstance(raw, dict) else raw
        if not isinstance(jobs, list):
            return
        with self._lock:
            for item in jobs:
                if not isinstance(item, dict) or not item.get("id"):
                    continue
                try:
                    _make_trigger(item)  # 校验触发器仍可解析
                except Exception as e:
                    logger.warning("scheduler: 跳过损坏任务 %r: %s",
                                   item.get("name"), e)
                    continue
                self._jobs[str(item["id"])] = item

    def _log(self, line: str) -> None:
        logger.info("%s", line)
        if self._log_fn is not None:
            try:
                self._log_fn(line)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 模块级单例（工具层通过 get_scheduler() 取 UI 启动的服务）
# ---------------------------------------------------------------------------

_scheduler: Optional[SchedulerService] = None


def set_scheduler(service: Optional[SchedulerService]) -> None:
    global _scheduler
    _scheduler = service


def get_scheduler() -> Optional[SchedulerService]:
    return _scheduler
