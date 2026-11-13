"""定时任务：SchedulerService + schedule_tools 的端到端行为测试。

覆盖：触发规则校验、创建/持久化/恢复、删除/暂停/恢复、run_now 的
tool/agent 两类执行、一次性任务执行后清理、工具层 CRUD 与不可用降级。
"""
from __future__ import annotations

import json
import os
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from agent.scheduler import SchedulerService, SchedulerError, set_scheduler
from agent.tools.base import ToolResult
from agent.tools.schedule_tool import ScheduleTool


def _wait_until(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def svc(tmp_path):
    service = SchedulerService(str(tmp_path), auto_start=True)
    yield service
    service.shutdown(wait=False)


def test_create_list_persist_and_reload(tmp_path):
    s1 = SchedulerService(str(tmp_path), auto_start=True)
    spec = s1.create_job(name="早报", kind="agent", cron="0 9 * * *",
                         prompt="总结昨天的文档")
    assert spec["id"].startswith("sched_")
    assert spec["trigger"] == {"type": "cron", "cron": "0 9 * * *"}
    assert s1.describe(spec["id"])["next_run"]
    s1.shutdown(wait=False)

    # 新实例从同一 workspace 恢复
    s2 = SchedulerService(str(tmp_path), auto_start=True)
    try:
        jobs = s2.list_jobs()
        assert [j["name"] for j in jobs] == ["早报"]
        assert jobs[0]["next_run"]
    finally:
        s2.shutdown(wait=False)


def test_trigger_validation(svc):
    with pytest.raises(SchedulerError):
        svc.create_job(name="无触发", prompt="x")
    with pytest.raises(SchedulerError):
        svc.create_job(name="双触发", cron="0 9 * * *", interval_minutes=5,
                       prompt="x")
    with pytest.raises(SchedulerError):
        svc.create_job(name="坏 cron", cron="0 9 * *", prompt="x")
    with pytest.raises(SchedulerError):
        svc.create_job(name="间隔太小", interval_minutes=0.1, prompt="x")
    with pytest.raises(SchedulerError):
        svc.create_job(name="过去时间", run_at="2000-01-01 09:00", prompt="x")
    with pytest.raises(SchedulerError):
        svc.create_job(name="缺 prompt", kind="agent", cron="0 9 * * *")
    with pytest.raises(SchedulerError):
        svc.create_job(name="缺工具", kind="tool", cron="0 9 * * *")


def test_create_weekly_trigger(svc):
    from datetime import datetime

    spec = svc.create_job(name="周一早报", kind="agent", weekday="周一",
                          time="9:05", prompt="x")
    assert spec["trigger"] == {
        "type": "cron", "cron": "5 9 * * mon",
        "desc": "每周一 09:05（本地时区）",
    }
    assert spec["next_run"]
    nxt = datetime.fromisoformat(spec["next_run"])
    assert nxt.weekday() == 0 and (nxt.hour, nxt.minute) == (9, 5)


def test_create_date_time_trigger(svc):
    from datetime import datetime, timedelta

    from agent.scheduler.service import describe_trigger

    target = datetime.now() + timedelta(days=2, minutes=5)
    spec = svc.create_job(
        name="项目复盘", kind="agent", date=target.strftime("%Y-%m-%d"),
        time=target.strftime("%H:%M"), prompt="x")
    trig = spec["trigger"]
    assert trig["type"] == "date"
    when = datetime.fromisoformat(trig["run_at"])
    assert (when.year, when.month, when.day) == (
        target.year, target.month, target.day)
    assert (when.hour, when.minute) == (target.hour, target.minute)
    assert "指定时间" in describe_trigger(spec)

    # 不带年份的月日：未来日期按今年/下一年解释
    target = datetime.now() + timedelta(days=40)
    spec = svc.create_job(
        name="月日任务", kind="agent", date=target.strftime("%m月%d日"),
        time="10:00", prompt="x")
    when = datetime.fromisoformat(spec["trigger"]["run_at"])
    assert (when.year, when.month, when.day) == (
        target.year, target.month, target.day)


def test_create_delay_trigger(svc):
    from datetime import datetime, timedelta

    from agent.scheduler.service import describe_trigger

    before = datetime.now().astimezone()
    spec = svc.create_job(name="30分钟后", kind="agent",
                          delay_minutes=30, prompt="x")
    when = datetime.fromisoformat(spec["trigger"]["run_at"])
    assert before + timedelta(minutes=29, seconds=50) <= when \
        <= before + timedelta(minutes=31)
    assert "30 分钟后" in describe_trigger(spec)

    before = datetime.now().astimezone()
    spec = svc.create_job(name="2小时后", kind="agent",
                          delay_hours=2, prompt="x")
    when = datetime.fromisoformat(spec["trigger"]["run_at"])
    assert before + timedelta(hours=1, minutes=59) <= when \
        <= before + timedelta(hours=2, minutes=1)
    assert "2 小时后" in describe_trigger(spec)


def test_new_trigger_validation(svc):
    cases = [
        {"weekday": "周一"},                          # 缺 time
        {"time": "09:00"},                            # time 无 weekday/date
        {"weekday": "周八", "time": "09:00"},          # 非法周几
        {"weekday": "周一", "time": "25:00"},          # 非法小时
        {"weekday": "周一", "time": "09:00", "cron": "0 9 * * *"},  # 双规则
        {"date": "2026-13-01", "time": "09:00"},       # 非法日期
        {"date": "2000-01-01", "time": "09:00"},       # 过去时间
        {"date": "2026-11-20"},                        # 缺 time
        {"delay_minutes": 30, "delay_hours": 1},       # 双延迟
        {"delay_minutes": 0},                          # 小于 1 分钟
    ]
    for i, kwargs in enumerate(cases):
        with pytest.raises(SchedulerError):
            svc.create_job(name=f"bad{i}", kind="agent", prompt="x", **kwargs)


def test_duplicate_name_rejected(svc):
    svc.create_job(name="同名", kind="agent", cron="0 9 * * *", prompt="x")
    with pytest.raises(SchedulerError):
        svc.create_job(name="同名", kind="agent", cron="0 10 * * *", prompt="y")


def test_pause_resume_and_delete(svc):
    spec = svc.create_job(name="任务A", kind="agent", cron="0 9 * * *",
                          prompt="x")
    paused = svc.pause_job(spec["id"])
    assert paused["enabled"] is False
    assert paused["next_run"] == ""
    resumed = svc.resume_job(spec["id"])
    assert resumed["enabled"] is True
    assert resumed["next_run"]
    svc.delete_job(spec["id"])
    assert svc.list_jobs() == []
    # 任务文件同步删除
    data = json.load(open(os.path.join(str(svc._workspace), ".agent",
                                       "scheduler_jobs.json"), encoding="utf-8"))
    assert data["jobs"] == []


def test_tool_job_run_now_notifies_and_records(tmp_path):
    calls, notes = [], []

    def executor(name, args):
        calls.append((name, args))
        return ToolResult(True, output=f"{name} ok")

    service = SchedulerService(
        str(tmp_path), auto_start=True, tool_executor=executor,
        notify=lambda name, text: notes.append((name, text)))
    try:
        spec = service.create_job(
            name="巡检", kind="tool", interval_minutes=30,
            tool_name="code_run", tool_args={"code": "print(1)"})
        assert spec["next_run"]
        service.run_now(spec["id"])
        assert _wait_until(lambda: bool(notes))
        assert calls == [("code_run", {"code": "print(1)"})]
        assert notes[0][0] == "巡检" and "code_run ok" in notes[0][1]
        assert _wait_until(
            lambda: service.describe(spec["id"])["last_status"] == "ok")
    finally:
        service.shutdown(wait=False)


def test_agent_job_run_now_submits(tmp_path):
    submitted = []
    service = SchedulerService(
        str(tmp_path), auto_start=True,
        agent_submitter=lambda name, prompt: submitted.append((name, prompt)))
    try:
        spec = service.create_job(name="总结", kind="agent",
                                  interval_minutes=10, prompt="总结新增文档")
        service.run_now(spec["id"])
        assert _wait_until(lambda: bool(submitted))
        assert submitted == [("总结", "总结新增文档")]
        assert _wait_until(
            lambda: service.describe(spec["id"])["last_status"] == "submitted")
    finally:
        service.shutdown(wait=False)


def test_one_shot_job_removed_after_run(tmp_path):
    notes = []
    service = SchedulerService(
        str(tmp_path), auto_start=True,
        tool_executor=lambda n, a: ToolResult(True, "done"),
        notify=lambda name, text: notes.append(text))
    try:
        spec = service.create_job(name="一次性", kind="tool",
                                  run_at=time.strftime("%Y-%m-%d %H:%M:%S",
                                                       time.localtime(time.time() + 3600)),
                                  tool_name="code_run", tool_args={})
        assert spec["trigger"]["type"] == "date"
        service.run_now(spec["id"])
        assert _wait_until(lambda: bool(notes))
        assert _wait_until(lambda: service.list_jobs() == [])
    finally:
        service.shutdown(wait=False)


def test_schedule_tool_crud_via_singleton(tmp_path):
    service = SchedulerService(str(tmp_path), auto_start=True,
                               tool_executor=lambda n, a: ToolResult(True, "ok"),
                               notify=lambda name, text: None)
    set_scheduler(service)
    try:
        tool = ScheduleTool()
        res = tool.run(op="create", name="每十分钟", kind="tool",
                       interval_minutes=10, tool_name="code_run",
                       tool_args='{"code": "print(1)"}')
        assert res.success, res.error
        assert "已创建定时任务「每十分钟」" in res.output

        res = tool.run(op="list")
        assert "当前共 1 个定时任务" in res.output
        assert "每十分钟" in res.output

        res = tool.run(op="pause", task_id="每十分钟")
        assert res.success and "已暂停" in res.output
        res = tool.run(op="resume", task_id="每十分钟")
        assert res.success and "已恢复" in res.output

        res = tool.run(op="run_now", task_id="每十分钟")
        assert res.success and "已触发" in res.output

        res = tool.run(op="delete", task_id="每十分钟")
        assert res.success and "已删除" in res.output
        assert tool.run(op="list").output == "当前没有定时任务。"
    finally:
        set_scheduler(None)
        service.shutdown(wait=False)


def test_schedule_tool_weekday_date_delay(tmp_path):
    """工具层支持每周几、指定日期、当前时间后推三种新触发方式。"""
    from datetime import datetime, timedelta

    service = SchedulerService(str(tmp_path), auto_start=True)
    set_scheduler(service)
    try:
        tool = ScheduleTool()
        res = tool.run(op="create", name="周一任务", kind="agent",
                       weekday="星期一", time="09:30", prompt="x")
        assert res.success, res.error
        assert "每周一 09:30" in res.output

        target = datetime.now() + timedelta(days=3)
        res = tool.run(op="create", name="日期任务", kind="agent",
                       date=target.strftime("%Y-%m-%d"), time="14:30",
                       prompt="x")
        assert res.success, res.error
        assert "指定时间" in res.output

        res = tool.run(op="create", name="延迟任务", kind="agent",
                       delay_minutes=45, prompt="x")
        assert res.success, res.error
        assert "45 分钟后" in res.output
        assert len(service.list_jobs()) == 3
    finally:
        set_scheduler(None)
        service.shutdown(wait=False)


def test_schedule_tool_unavailable_and_bad_args(tmp_path):
    set_scheduler(None)
    res = ScheduleTool().run(op="create", name="x", interval_minutes=5,
                             prompt="y")
    assert not res.success and "未启动" in res.error

    service = SchedulerService(str(tmp_path), auto_start=True)
    set_scheduler(service)
    try:
        res = ScheduleTool().run(op="create", name="x", interval_minutes=5,
                                 kind="tool", tool_name="code_run",
                                 tool_args="{bad json}")
        assert not res.success and "tool_args" in res.error
        res = ScheduleTool().run(op="create", name="x", prompt="y")
        assert not res.success and "触发规则" in res.error
        res = ScheduleTool().run(op="delete", task_id="不存在")
        assert not res.success and "未找到任务" in res.error
    finally:
        set_scheduler(None)
        service.shutdown(wait=False)


def test_one_shot_fires_automatically(tmp_path):
    """真实 APScheduler 触发：一次性任务到点自动执行并自行清理。"""
    from datetime import datetime, timedelta

    notes = []
    service = SchedulerService(
        str(tmp_path), auto_start=True,
        tool_executor=lambda n, a: ToolResult(True, "自动执行完成"),
        notify=lambda name, text: notes.append((name, text)))
    try:
        run_at = (datetime.now() + timedelta(seconds=2)).strftime(
            "%Y-%m-%d %H:%M:%S")
        service.create_job(name="到点执行", kind="tool", run_at=run_at,
                           tool_name="code_run", tool_args={})
        assert _wait_until(lambda: bool(notes), timeout=12)
        assert notes[0][0] == "到点执行"
        assert "自动执行完成" in notes[0][1]
        assert _wait_until(lambda: service.list_jobs() == [], timeout=5)
    finally:
        service.shutdown(wait=False)


def test_resume_reschedules_job_not_loaded(tmp_path):
    """暂停中的任务启动时不进调度器；resume 必须补登记，否则永不触发。"""
    s1 = SchedulerService(str(tmp_path), auto_start=True)
    spec = s1.create_job(name="暂停任务", kind="agent", cron="0 9 * * *",
                         prompt="x")
    s1.pause_job(spec["id"])
    s1.shutdown(wait=False)

    s2 = SchedulerService(str(tmp_path), auto_start=True)
    try:
        assert s2.describe(spec["id"])["next_run"] == ""
        resumed = s2.resume_job(spec["id"])
        assert resumed["enabled"] is True
        assert resumed["next_run"]
    finally:
        s2.shutdown(wait=False)


def test_agent_creates_job_through_schedule_tool(config, recording_callbacks, tmp_path):
    """完整链路：模型 tool_call → 内核分发 → schedule_tools → 任务落库。"""
    import json as _json

    from agent.agent import Agent
    from agent.llm_client import StreamEvent

    submitted = []
    service = SchedulerService(
        str(tmp_path), auto_start=True,
        agent_submitter=lambda n, p: submitted.append((n, p)))
    set_scheduler(service)
    tool = ScheduleTool()

    class _Reg:
        session = None

        def get(self, name):
            return tool if name == "schedule_tools" else None

        def execute(self, name, args):
            return tool.run(**args)

        def schemas(self):
            return [tool.schema()]

        def names(self):
            return ["schedule_tools"]

    class _LLM:
        def __init__(self):
            self.n = 0

        def chat_stream(self, messages, tools=None, temperature=0.7, **kw):
            self.n += 1
            if self.n == 1:
                tc = [{"id": "c1", "function": {
                    "name": "schedule_tools",
                    "arguments": _json.dumps({
                        "op": "create", "name": "每日总结",
                        "kind": "agent", "cron": "0 9 * * *",
                        "prompt": "总结知识库新增内容"}),
                }}]
                return iter([StreamEvent(type="done", tool_calls=tc,
                                         usage={"total_tokens": 1})])
            return iter([StreamEvent(type="content", content="已创建任务"),
                         StreamEvent(type="done")])

    try:
        agent = Agent(llm=_LLM(), config=config,
                      callbacks=recording_callbacks, tool_registry=_Reg())
        agent.run("每天早上九点总结知识库")
        jobs = service.list_jobs()
        assert len(jobs) == 1
        assert jobs[0]["name"] == "每日总结"
        assert jobs[0]["trigger"] == {"type": "cron", "cron": "0 9 * * *"}
        assert recording_callbacks.tool_ends
        assert "已创建定时任务「每日总结」" in recording_callbacks.tool_ends[0][1].output
        # agent 任务真正执行时会投递到对话线程
        service.run_now(jobs[0]["id"])
        assert _wait_until(lambda: bool(submitted))
        assert submitted[0][0] == "每日总结"
    finally:
        set_scheduler(None)
        service.shutdown(wait=False)


def test_missed_once_job_is_cleaned(tmp_path):
    """一次性任务错过窗口（misfire）后从任务表清理；周期任务保留。"""
    service = SchedulerService(str(tmp_path), auto_start=True)
    try:
        once = service.create_job(name="错过的一次性", kind="agent",
                                  run_at=time.strftime(
                                      "%Y-%m-%d %H:%M:%S",
                                      time.localtime(time.time() + 3600)),
                                  prompt="x")
        cron = service.create_job(name="周期任务", kind="agent",
                                  cron="0 9 * * *", prompt="x")

        class _Evt:
            job_id = once["id"]

        service._on_job_missed(_Evt())
        names = [j["name"] for j in service.list_jobs()]
        assert names == ["周期任务"]
        # 周期任务错过一次只是跳过本次，不删除
        class _Evt2:
            job_id = cron["id"]

        service._on_job_missed(_Evt2())
        assert [j["name"] for j in service.list_jobs()] == ["周期任务"]
    finally:
        service.shutdown(wait=False)


def test_schedule_tools_registered(config):
    """schedule_tools 必须进入默认工具集（模型才能通过对话建任务）。"""
    from agent.agent import AgentCallbacks
    from agent.tools import ToolRegistry

    reg = ToolRegistry(config=config, callbacks=AgentCallbacks())
    try:
        assert "schedule_tools" in reg.names()
        schema = next(s for s in reg.schemas()
                      if s["function"]["name"] == "schedule_tools")
        ops = schema["function"]["parameters"]["properties"]["op"]["enum"]
        assert "create" in ops and "run_now" in ops
    finally:
        reg.shutdown()
