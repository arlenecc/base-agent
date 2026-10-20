"""计划与反思能力：规划门、反思门、update_plan 工具、anchor 回注。

对应「先规划 → 每步反思 → 防跑丢/防半路停」的闭环。
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.agent import Agent                      # noqa: E402
from agent.config import AgentConfig               # noqa: E402
from agent.kernel import normalize_plan            # noqa: E402
from agent.kernel.loop import _PLAN_GATE_TURNS     # noqa: E402
from agent.kernel.session import Session           # noqa: E402
from agent.kernel.budget import ContextBudget      # noqa: E402
from agent.llm_client import StreamEvent           # noqa: E402


def _evt(t, content="", tool_calls=None, usage=None):
    return StreamEvent(type=t, content=content, tool_calls=tool_calls or [], usage=usage or {})


def _tc(name, args, cid="c1"):
    return [{"id": cid, "function": {"name": name, "arguments": json.dumps(args)}}]


class MockLLM:
    def __init__(self, scripts):
        self._scripts = list(scripts)
        self.calls = []

    def chat_stream(self, messages, tools=None, temperature=0.7):
        self.calls.append({"messages": messages, "tools": tools})
        if not self._scripts:
            raise AssertionError("MockLLM ran out of scripted responses")
        return iter(self._scripts.pop(0))


class Noop:
    def is_cancelled(self): return False
    def on_content(self, t): pass
    def on_reasoning(self, t): pass
    def on_tool_start(self, n, a): pass
    def on_tool_end(self, n, r): pass
    def on_log(self, l): pass
    def on_usage(self, u): pass
    def on_token_speed(self, t, s): pass
    def on_finished(self): pass
    def on_plan_updated(self, plan): pass
    def confirm(self, m): return True
    def ask_user(self, p): return "ok"


@pytest.fixture()
def cfg(tmp_path):
    c = AgentConfig()
    c.workspace = str(tmp_path)
    c.knowledge_base = str(tmp_path / "kb")
    return c


@pytest.fixture()
def session():
    return Session(budget=ContextBudget(30_000), raw_turns=2)


# ---------------------------------------------------------------------------
# normalize_plan：脏输入必须被归一，不能让内核抛异常
# ---------------------------------------------------------------------------

def test_normalize_plan_accepts_wellformed():
    p = normalize_plan({"goal": "部署", "steps": [
        {"title": "读配置", "status": "done"},
        {"title": "改端口"},
        "写文档",                      # 字符串形式
    ]})
    assert p is not None
    assert p["goal"] == "部署"
    assert [s["status"] for s in p["steps"]] == ["done", "pending", "pending"]
    assert p["steps"][2]["title"] == "写文档"
    # id 自动编号
    assert [s["id"] for s in p["steps"]] == ["1", "2", "3"]


def test_normalize_plan_rejects_garbage():
    assert normalize_plan(None) is None
    assert normalize_plan({"goal": "x"}) is None                 # 无 steps
    assert normalize_plan({"goal": "x", "steps": []}) is None    # 空 steps
    assert normalize_plan({"goal": "x", "steps": "nope"}) is None
    # 全部无标题 → 空 steps → 拒绝
    assert normalize_plan({"goal": "x", "steps": [{"note": "no title"}]}) is None
    # 非法 status 归一为 pending
    p = normalize_plan({"goal": "g", "steps": [{"title": "a", "status": "WEIRD"}]})
    assert p["steps"][0]["status"] == "pending"


def test_normalize_plan_truncates_long_fields():
    p = normalize_plan({"goal": "G" * 500,
                        "steps": [{"title": "T" * 500, "note": "N" * 500}]})
    assert len(p["goal"]) == 300
    assert len(p["steps"][0]["title"]) == 120
    assert len(p["steps"][0]["note"]) == 200


# ---------------------------------------------------------------------------
# Session：计划渲染与 open 步骤
# ---------------------------------------------------------------------------

def test_session_plan_roundtrip_and_render(session):
    assert session.get_plan() is None
    assert session.plan_open_steps() == []
    assert session.render_plan() == ""

    assert session.set_plan({"goal": "修 bug", "steps": [
        {"title": "复现", "status": "done"},
        {"title": "定位", "status": "doing", "note": "在 auth.py"},
        {"title": "验证"},
    ]}) is True
    text = session.render_plan()
    assert "Goal: 修 bug" in text
    assert "- [✓] 1. 复现" in text
    assert "- [▶] 2. 定位  # 在 auth.py" in text
    assert "- [ ] 3. 验证" in text
    assert "Now: 2. 定位" in text
    assert "Remaining: 2 step(s)" in text
    assert [s["id"] for s in session.plan_open_steps()] == ["2", "3"]

    session.clear_plan()
    assert session.get_plan() is None


def test_anchor_prompt_injects_plan(session):
    session.set_plan({"goal": "上线", "steps": [{"title": "打包", "status": "doing"}]})
    anchor = session.anchor_prompt(extra="用户请求")
    assert "### [PLAN]" in anchor
    assert "Goal: 上线" in anchor
    assert "打包" in anchor


def test_reset_clears_plan(session):
    session.set_plan({"goal": "g", "steps": [{"title": "t"}]})
    session.reset()
    assert session.get_plan() is None


# ---------------------------------------------------------------------------
# 内核：规划门（未规划先提醒）与反思门（未完成不得收尾）
# ---------------------------------------------------------------------------

def test_plan_gate_reminds_before_acting(cfg):
    """前几轮没有计划时，锚点里必须出现规划提示。"""
    mock = MockLLM([
        [_evt("content", "好的"), _evt("done", usage={"total_tokens": 1})],
    ])
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    agent.kernel.run("帮我部署一下")

    first = mock.calls[0]["messages"]
    anchor = next((str(m.get("content") or "") for m in first
                   if m.get("role") == "user"), "")
    assert "update_plan" in anchor, "首轮应提醒先规划"


def test_plan_gate_skips_simple_input(cfg):
    """一句话任务（问候 / 读个文件）不该被逼着建执行计划。

    曾经无条件注入规划提示，于是「hi」也会被逼着 create 一个『询问用户需求』
    的一步计划，而「询问」根本不是可执行的工具动作——模型随即卡在「立即开始
    第 1 步」上反复发无意义调用。
    """
    mock = MockLLM([
        [_evt("content", "好的"), _evt("done", usage={"total_tokens": 1})],
    ])
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    agent.kernel.run("读文件")

    first = mock.calls[0]["messages"]
    anchor = next((str(m.get("content") or "") for m in first
                   if m.get("role") == "user"), "")
    assert "开始执行前" not in anchor, "简单输入不应被注入规划提示"


def test_plan_gate_stops_after_work_starts(cfg):
    """多步任务：首轮提醒先规划，之后不再**新**注入。

    提示一旦注入就成为该轮 user 消息的一部分，会在 raw 窗口里多留几轮——
    所以判据是「最后一轮不再有」，而不是「第二轮就没有」。
    """
    n = 8
    create = [{"id": "c1", "function": {"name": "update_plan",
               "arguments": json.dumps(
                   {"op": "create", "goal": "改代码", "steps": ["读文件", "改代码"]})}}]
    mock = MockLLM(
        [[_evt("done", tool_calls=create, usage={"total_tokens": 1})]]
        + [[_evt("done", tool_calls=_tc("file_read", {"path": "a.txt"}),
                 usage={"total_tokens": 1})] for _ in range(n)]
        + [[_evt("content", "读完"), _evt("done", usage={"total_tokens": 1})]]
    )
    with open(os.path.join(cfg.workspace, "a.txt"), "w") as f:
        f.write("A")
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    agent.max_iterations = 20
    agent.kernel.run("先读文件，然后帮我改代码")

    hinted = [
        i for i, c in enumerate(mock.calls, 1)
        if any("开始执行前" in str(m.get("content") or "") for m in c["messages"])
    ]
    assert hinted, "多步任务首轮应提醒先规划"
    assert hinted[-1] < n, f"最后一轮仍被注入新的规划提示: {hinted}"


def test_reflection_gate_blocks_premature_finish(cfg):
    """有计划且步骤未完成时，模型想收尾会被拦回来继续。"""
    mock = MockLLM([
        # 第 1 轮：建立计划
        [_evt("done", tool_calls=_tc("update_plan", {
            "op": "create", "goal": "改端口",
            "steps": [{"title": "读配置"}, {"title": "改端口"}, {"title": "重启验证"}],
        }), usage={"total_tokens": 1})],
        # 第 2 轮：什么工具都不调就想收尾 → 应被反思门拦下
        [_evt("content", "已经读完了"), _evt("done", usage={"total_tokens": 1})],
        # 第 3 轮：被拦后继续执行
        [_evt("done", tool_calls=_tc("file_read", {"path": "a.txt"}), usage={"total_tokens": 1})],
        # 第 4 轮：标记完成
        [_evt("done", tool_calls=_tc("update_plan", {
            "op": "patch", "step_id": "2", "status": "done", "note": "已改",
        }), usage={"total_tokens": 1})],
        # 第 5 轮：再次想收尾（还有 1 步未完成）→ 再拦一次
        [_evt("content", "done"), _evt("done", usage={"total_tokens": 1})],
        # 第 6 轮：真正收尾
        [_evt("content", "全部完成"), _evt("done", usage={"total_tokens": 1})],
    ])
    with open(os.path.join(cfg.workspace, "a.txt"), "w") as f:
        f.write("port = 8080")
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    agent.max_iterations = 15
    out = agent.kernel.run("把端口改成 9090")

    assert agent.session.get_plan() is not None
    # 计划被建立、步骤被 patch，说明流程真的走过去了
    assert agent.session.plan_open_steps() != []
    assert "全部完成" in out.text
    # 至少 5 次 LLM 调用：被反思门拦了两次
    assert len(mock.calls) >= 5


def test_reflection_gate_gives_up_after_max_and_warns_user(cfg):
    """提醒次数用尽后尊重模型，但回复必须附上未完成清单。

    本用例是「只建了计划、一次实际工具都没调用」→ 走 PLAN 档预算（6 次）。
    """
    from agent.kernel.loop import _FINISH_NUDGE_PLAN_MAX

    give_up = [_evt("content", "我做完了"), _evt("done", usage={"total_tokens": 1})]
    mock = MockLLM([
        [_evt("done", tool_calls=_tc("update_plan", {
            "op": "create", "goal": "g", "steps": [{"title": "a"}, {"title": "b"}],
        }), usage={"total_tokens": 1})],
        *([give_up] * _FINISH_NUDGE_PLAN_MAX),
        [_evt("content", "就这样吧"), _evt("done", usage={"total_tokens": 1})],
    ])
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    agent.max_iterations = 30
    out = agent.kernel.run("做任务")

    assert "未完成步骤" in out.text
    assert "a" in out.text and "b" in out.text


def test_plan_created_but_not_started_is_pushed_to_execute(cfg):
    """回归：建完计划就只输出文字 → 必须被拦并要求「立即执行 Now 那一步」。

    修复前无失败时反思门只有 2 次预算，模型建完计划把计划复述一遍就放行，
    用户看到的正是不执行、必须手动说「继续/开始」。
    """
    from agent.kernel.loop import _FINISH_NUDGE_MAX, _FINISH_NUDGE_PLAN_MAX

    assert _FINISH_NUDGE_PLAN_MAX > _FINISH_NUDGE_MAX, "计划未开工时预算必须更宽"
    mock = MockLLM([
        [_evt("done", tool_calls=_tc("update_plan", {
            "op": "create", "goal": "写脚本", "steps": [{"title": "写脚本"}],
        }), usage={"total_tokens": 1})],
        # 连续 3 次只把计划念一遍 → 旧实现第 3 次就放行
        [_evt("content", "计划已创建：1.写脚本"), _evt("done", usage={"total_tokens": 1})],
        [_evt("content", "接下来我会开始执行"), _evt("done", usage={"total_tokens": 1})],
        [_evt("content", "马上开始第一步"), _evt("done", usage={"total_tokens": 1})],
        # 真正开工
        [_evt("done", tool_calls=_tc("file_write", {"path": "a.py", "content": "x"}),
              usage={"total_tokens": 1})],
        [_evt("done", tool_calls=_tc("update_plan", {"op": "patch", "step_id": "1",
                                                     "status": "done", "note": "ok"}),
              usage={"total_tokens": 1})],
        [_evt("content", "完成"), _evt("done", usage={"total_tokens": 1})],
    ])
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    agent.max_iterations = 30
    agent._auto_continue = False
    out = agent.kernel.run("写个脚本")

    # 第 3 次纯文字仍被拦（没有被放行）
    nudge = _last_user_prompt(mock, 3)
    assert "一次工具都没调用" in nudge, "提醒必须点破「只规划没执行」"
    assert "立即开始执行" in nudge and "写脚本" in nudge, "提醒必须点名 Now 那一步"
    assert out.text == "完成"


def _last_user_prompt(mock, call_idx):
    """第 call_idx 次 LLM 调用的最后一条 user 消息（= 本轮锚点）。"""
    msgs = mock.calls[call_idx]["messages"]
    users = [str(m.get("content") or "") for m in msgs if m.get("role") == "user"]
    return users[-1] if users else ""


def test_finish_allowed_when_all_steps_terminal(cfg):
    """计划全部终态化后，模型可以正常收尾，不被拦。"""
    mock = MockLLM([
        [_evt("done", tool_calls=_tc("update_plan", {
            "op": "create", "goal": "g", "steps": [{"title": "a"}],
        }), usage={"total_tokens": 1})],
        [_evt("done", tool_calls=_tc("update_plan", {
            "op": "patch", "step_id": "1", "status": "done", "note": "ok",
        }), usage={"total_tokens": 1})],
        [_evt("content", "完成"), _evt("done", usage={"total_tokens": 1})],
    ])
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    out = agent.kernel.run("做任务")

    assert out.text == "完成"
    assert "未完成步骤" not in out.text
    assert agent.session.plan_open_steps() == []


def test_no_plan_means_no_gate(cfg):
    """没有计划的普通任务不受反思门影响（回归：不能把简单任务卡住）。"""
    mock = MockLLM([[_evt("content", "42"), _evt("done", usage={"total_tokens": 1})]])
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    out = agent.kernel.run("6 乘 7 等于几")
    assert out.text == "42"
    assert len(mock.calls) == 1


# ---------------------------------------------------------------------------
# update_plan 工具
# ---------------------------------------------------------------------------

def _registry(cfg, session, callbacks=None):
    from agent.tools import ToolRegistry
    reg = ToolRegistry(config=cfg, callbacks=callbacks or Noop())
    reg.session = session
    return reg


def test_update_plan_tool_lifecycle(cfg, session):
    reg = _registry(cfg, session)

    # create
    res = reg.execute("update_plan", {
        "op": "create", "goal": "修 bug",
        "steps": [{"title": "复现"}, {"title": "修复"}, {"title": "验证"}],
    })
    assert res.success, res.error
    plan = session.get_plan()
    assert plan["goal"] == "修 bug"
    # 第一步自动标 doing
    assert plan["steps"][0]["status"] == "doing"
    assert "Now: 1. 复现" in res.output

    # patch：标完成后自动推进下一步
    res = reg.execute("update_plan", {"op": "patch", "step_id": "1",
                                      "status": "done", "note": "稳定复现"})
    assert res.success, res.error
    plan = session.get_plan()
    assert plan["steps"][0]["status"] == "done"
    assert plan["steps"][0]["note"] == "稳定复现"
    assert plan["steps"][1]["status"] == "doing", "完成后应自动推进下一步"

    # append
    res = reg.execute("update_plan", {"op": "append", "steps": [{"title": "补回归测试"}]})
    assert res.success, res.error
    assert len(session.get_plan()["steps"]) == 4

    # patch 不存在的 id
    res = reg.execute("update_plan", {"op": "patch", "step_id": "99", "status": "done"})
    assert not res.success

    # clear
    res = reg.execute("update_plan", {"op": "clear"})
    assert res.success and session.get_plan() is None

    reg.shutdown()


def test_update_plan_create_accepts_steps_without_goal(cfg, session):
    """goal 选填：漏写时仍应建成计划（steps 才是路径的主体）。"""
    reg = _registry(cfg, session)
    ok = reg.execute("update_plan", {"op": "create", "steps": [{"title": "a"}]})
    assert ok.success, ok.error
    assert session.get_plan()["steps"][0]["title"] == "a"
    reg.shutdown()


def test_update_plan_rejects_bad_input(cfg, session):
    reg = _registry(cfg, session)
    # 有 goal 没 steps → 空计划 → 拒绝
    assert not reg.execute("update_plan", {"op": "create", "goal": "g"}).success
    # 没有任何计划时 patch 无从下手
    assert not reg.execute("update_plan", {"op": "patch", "status": "done"}).success
    # 未知 op
    assert not reg.execute("update_plan", {"op": "bogus"}).success
    # 非法 status
    assert not reg.execute("update_plan", {"op": "patch",
                                          "status": "WEIRD"}).success
    reg.shutdown()


def test_update_plan_notifies_callback(cfg, session):
    seen = []
    cb = Noop()
    cb.on_plan_updated = lambda p: seen.append(p)
    reg = _registry(cfg, session, callbacks=cb)
    reg.execute("update_plan", {"op": "create", "goal": "g",
                                "steps": [{"title": "a"}]})
    assert seen and seen[-1]["goal"] == "g"
    reg.shutdown()


def test_update_plan_is_registered(cfg):
    from agent.tools import ToolRegistry
    reg = ToolRegistry(config=cfg, callbacks=Noop())
    assert "update_plan" in reg.names()
    reg.shutdown()


# ---------------------------------------------------------------------------
# 任务边界：计划跨 run 的复位与保留
# ---------------------------------------------------------------------------

def _plan_call(op, **kw):
    return _tc("update_plan", dict(op=op, **kw))


def test_plan_resets_when_previous_task_finished(cfg):
    """上一个计划全部终态化后，新请求必须清空计划让它重新规划。

    不清的后果：旧计划一直通过 anchor 回注，模型沿着过期计划走、也不再调用
    update_plan —— 表现为「计划面板停在第一次那幅，新任务没人监督」。
    """
    mock = MockLLM([
        [_evt("done", tool_calls=_plan_call("create", goal="任务A",
                                             steps=[{"title": "a"}]), usage={"total_tokens": 1})],
        [_evt("done", tool_calls=_plan_call("patch", step_id="1", status="done",
                                            note="ok"), usage={"total_tokens": 1})],
        [_evt("content", "A完成"), _evt("done", usage={"total_tokens": 1})],
        # 第二个任务：新请求
        [_evt("content", "B完成"), _evt("done", usage={"total_tokens": 1})],
    ])
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    agent.kernel.run("任务A")
    assert agent.session.get_plan() is not None

    agent.kernel.run("任务B")          # 直接调内核不走门面
    assert agent.session.get_plan() is not None, "内核不管任务边界"

    # 走门面（真实入口）：第二个 run 开始时应清掉已完成的计划
    mock2 = MockLLM([[_evt("content", "ok"), _evt("done", usage={"total_tokens": 1})]])
    agent.llm = mock2
    agent.run("任务C")
    assert agent.session.get_plan() is None, "已完成的任务计划应被清空"


def test_plan_survives_when_previous_task_unfinished(cfg):
    """有未完成步骤时保留计划 —— 用户说「继续」要能接着干。

    同时验证：run() 开始时注入一次性「新任务/继续任务」的 fork 提示
    （`__plan_boundary__`），模型不再静默沿用旧计划。
    """
    mock = MockLLM([
        [_evt("done", tool_calls=_plan_call("create", goal="任务A",
                                             steps=[{"title": "a"}, {"title": "b"}]),
          usage={"total_tokens": 1})],
        [_evt("done", tool_calls=_plan_call("patch", step_id="1", status="done",
                                            note="ok"), usage={"total_tokens": 1})],
        # 连续三次只想文字收尾 → 反思门提醒用尽后放行（step2 仍未完成）
        [_evt("content", "先这样"), _evt("done", usage={"total_tokens": 1})],
        [_evt("content", "还是先这样"), _evt("done", usage={"total_tokens": 1})],
        [_evt("content", "算了"), _evt("done", usage={"total_tokens": 1})],
    ])
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    agent.max_iterations = 15
    agent._auto_continue = False     # 本测试专测「停车等用户」的边界状态
    agent.run("任务A")
    assert [s["id"] for s in agent.session.plan_open_steps()] == ["2"]

    # 「继续」：计划必须还在，且锚点里出现 fork 提示（新任务 → create 替换）
    mock2 = MockLLM([
        [_evt("done", tool_calls=_plan_call("patch", step_id="2", status="done",
                                            note="补做"), usage={"total_tokens": 1})],
        [_evt("content", "完成"), _evt("done", usage={"total_tokens": 1})],
    ])
    agent.llm = mock2
    agent.run("继续")
    assert agent.session.get_plan() is not None
    assert agent.session.get_plan()["goal"] == "任务A"
    assert agent.session.plan_open_steps() == []
    anchor = next(
        (str(m.get("content") or "") for m in reversed(mock2.calls[0]["messages"])
         if m.get("role") == "user"), "")
    assert "新任务" in anchor and "create" in anchor, (
        "计划有 open 步骤时，run() 必须注入新任务/继续任务的 fork 提示"
    )


def test_auto_continue_resumes_after_max_turns(cfg):
    """轮次上限截断时自动续跑，不再要求用户手动输入「继续」。"""
    tc = [{"id": "c1", "function": {"name": "file_write",
                                    "arguments": json.dumps({"path": "x.txt", "content": "v"})}}]
    mock = MockLLM([
        # 第一段：3 轮工具调用耗尽 max_iterations=3 → MAX_TURNS
        *([[_evt("done", tool_calls=tc, usage={"total_tokens": 1})]] * 3),
        # 续跑段：完成任务
        [_evt("content", "任务完成"), _evt("done", usage={"total_tokens": 1})],
    ])
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    agent.max_iterations = 3
    # 简单任务指令（无多步信号）以绕过「规划前置门」；本测试只验证
    # MAX_TURNS 截断后自动续跑，不验证规划行为。
    out = agent.run("写 x.txt")
    assert out == "任务完成"
    assert len(mock.calls) >= 4, "MAX_TURNS 后应自动续跑而不是停车"


def test_auto_continue_finishes_premature_stop(cfg):
    """「提前收尾但计划未完成」时自动续跑：模型在续跑段把步骤补上收尾。"""
    mock = MockLLM([
        [_evt("done", tool_calls=_plan_call("create", goal="g",
                                            steps=[{"title": "a"}]),
          usage={"total_tokens": 1})],
        # 先真正动过手（否则走「只规划不执行」的 6 次预算，不会触发续跑）
        [_evt("done", tool_calls=_tc("file_write", {"path": "a.txt", "content": "x"}),
              usage={"total_tokens": 1})],
        # 想直接文字收尾 → 反思门提醒 2 次仍不改（plan 有 open 步骤）
        [_evt("content", "做完了"), _evt("done", usage={"total_tokens": 1})],
        [_evt("content", "真的做完了"), _evt("done", usage={"total_tokens": 1})],
        [_evt("content", "就这样"), _evt("done", usage={"total_tokens": 1})],
        # 自动续跑段：按提示把步骤补成 done 后正常收尾
        [_evt("done", tool_calls=_plan_call("patch", step_id="1", status="done",
                                            note="补记"), usage={"total_tokens": 1})],
        [_evt("content", "全部完成"), _evt("done", usage={"total_tokens": 1})],
    ])
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    agent.max_iterations = 15
    out = agent.run("做任务")
    assert out == "全部完成"
    assert agent.session.plan_open_steps() == []
    assert len(mock.calls) >= 6


def test_run_start_notifies_plan_to_ui(cfg):
    """每轮开始都把当前计划同步给 UI —— 面板不能是上一轮的残影。"""
    seen = []
    cb = Noop()
    cb.on_plan_updated = lambda p: seen.append(p)
    mock = MockLLM([[_evt("content", "ok"), _evt("done", usage={"total_tokens": 1})]])
    agent = Agent(llm=mock, config=cfg, callbacks=cb)
    agent.run("你好")
    assert seen, "run() 开始时应至少同步一次计划状态"
    assert seen[-1] is None      # 没有计划 → 面板清空


# ---------------------------------------------------------------------------
# 规划前置门：先于执行（先写执行计划，再逐步执行）
# ---------------------------------------------------------------------------

def test_plan_before_act_gate_blocks_first_execution_without_plan(cfg):
    """多步任务指令下，首次实际工具调用必须先有 plan，否则被回退并要求先规划。"""
    mock = MockLLM([
        # 第 1 轮：直接发实际工具（无 plan）→ 应被规划前置门拦下
        [_evt("done", tool_calls=_tc("file_read", {"path": "a.txt"}), usage={"total_tokens": 1})],
        # 第 2 轮：被拦后先建计划
        [_evt("done", tool_calls=_tc("update_plan", {
            "op": "create", "goal": "读并概括",
            "steps": [{"title": "读 a.txt"}, {"title": "写概括"}],
        }), usage={"total_tokens": 1})],
        # 第 3 轮：真正开工
        [_evt("done", tool_calls=_tc("file_read", {"path": "a.txt"}), usage={"total_tokens": 1})],
        [_evt("content", "完成"), _evt("done", usage={"total_tokens": 1})],
    ])
    with open(os.path.join(cfg.workspace, "a.txt"), "w") as f:
        f.write("A")
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    agent.max_iterations = 20
    agent.kernel.run("依次读 a.txt 并概括")     # 含多步信号「依次」→ 不豁免

    # 被拦后会多消耗一轮（先规划再执行），所以至少 3 次 LLM 调用
    assert len(mock.calls) >= 3
    # 第 2 轮（被拦后那轮）锚点必须点名「先规划再执行」
    nudge = _last_user_prompt(mock, 1)
    assert "执行计划" in nudge, "规划前置门必须要求先写入执行计划"
    assert "先于执行" in nudge or "update_plan" in nudge
    # 最终计划确实被建立
    assert agent.session.get_plan() is not None


def test_simple_task_is_exempt_from_plan_gate(cfg):
    """一条短的一句话指令（无多步信号）不应被规划前置门卡住。"""
    mock = MockLLM([
        [_evt("done", tool_calls=_tc("file_read", {"path": "a.txt"}), usage={"total_tokens": 1})],
        [_evt("content", "读完了"), _evt("done", usage={"total_tokens": 1})],
    ])
    with open(os.path.join(cfg.workspace, "a.txt"), "w") as f:
        f.write("A")
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    out = agent.kernel.run("读 a.txt")
    # 没有被拦：一轮就执行完（无需先绕去建计划）
    assert len(mock.calls) == 2
    assert agent.session.get_plan() is None     # 简单任务本就不必有计划
    assert "读完了" in out.text


def test_plan_nudge_after_accidental_first_execution(cfg):
    """首轮误执行（plan 仍空）后，后续轮必须收到「补建计划」提示。"""
    mock = MockLLM([
        # 第 1 轮：短指令下误发实际工具（简单任务豁免 → 放行，did_work=True, plan 空）
        [_evt("done", tool_calls=_tc("file_write", {"path": "a.txt", "content": "x"}),
              usage={"total_tokens": 1})],
        # 第 2 轮：纯文字收尾 → 但上一轮已开工却无 plan，应收到补建提示
        [_evt("content", "做完了"), _evt("done", usage={"total_tokens": 1})],
    ])
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    out = agent.kernel.run("写个文件")
    nudge = _last_user_prompt(mock, 1)
    assert "已经开始执行但" in nudge and "执行计划" in nudge, (
        "首轮误执行后必须提醒补建计划，否则全程无 plan 可跟踪"
    )
    assert agent.session.get_plan() is None


def test_multistep_task_cannot_end_with_only_text_plan(cfg):
    """多步任务：模型只在文本里写计划 + 声称开始、不发工具也不写计划就想收尾，
    必须被反思门拦下，强制先把计划写入执行计划并立即开工（不能只在文本里规划）。"""
    mock = MockLLM([
        # 第 1 轮：纯文本「计划 + 开始第一个任务」——没调工具、没写计划
        [_evt("content", "计划：1) 读 a.txt 2) 概括。现在开始执行第一个任务。")],
        # 第 2 轮：被拦后，正确做法——先把计划写入执行计划
        [_evt("done", tool_calls=_tc("update_plan", {
            "op": "create", "goal": "读并概括",
            "steps": [{"title": "读 a.txt"}, {"title": "概括"}],
        }), usage={"total_tokens": 1})],
        # 第 3 轮：真正开工
        [_evt("done", tool_calls=_tc("file_read", {"path": "a.txt"}), usage={"total_tokens": 1})],
        [_evt("content", "完成"), _evt("done", usage={"total_tokens": 1})],
    ])
    with open(os.path.join(cfg.workspace, "a.txt"), "w") as f:
        f.write("A")
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    agent.max_iterations = 20
    out = agent.kernel.run("依次读 a.txt 并概括")     # 含多步信号「依次」
    # 被拦后会多消耗一轮（先写计划再执行），所以至少 3 次 LLM 调用
    assert len(mock.calls) >= 3
    # 第 2 轮（被拦后那轮）锚点必须点名「写入执行计划」并调用 update_plan
    nudge = _last_user_prompt(mock, 1)
    assert "执行计划" in nudge, "必须要求把计划写入执行计划，而非只在文本里规划"
    assert "update_plan" in nudge
    # 最终计划确实被建立（不再只是文本里的计划）
    assert agent.session.get_plan() is not None

