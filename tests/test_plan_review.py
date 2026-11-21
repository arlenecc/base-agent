"""计划评审 + 收尾拦截（决策模型两个判定点）单元测试。

覆盖：plan_review 两个判定函数的各分支（通过/拒绝/模型不可用/打分失败）、
update_plan(op='create') 的评审拦截与重试上限、Agent._decision_gate 的
强制续跑判定与跳过条件。
"""
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.config import AgentConfig
from agent.kernel.outcome import ExitReason
from agent.kernel.session import ContextBudget, Session
from agent.plan_review import (
    COMPLETION_MAX_FORCE, PLAN_REVIEW_MAX_REJECTS,
    check_plan_completion, review_plan_steps,
)

# ---------------------------------------------------------------------------
# 桩
# ---------------------------------------------------------------------------

class _FakeModel:
    """可编程决策模型：score(query, texts) → 按预设函数逐条返回。"""

    def __init__(self, fn):
        self._fn = fn
        self.calls = []

    def score(self, query, texts):
        self.calls.append((query, list(texts)))
        return [float(self._fn(query, t)) for t in texts]


class _BoomModel:
    def score(self, query, texts):
        raise RuntimeError("model boom")


STEPS = [
    {"id": "1", "title": "读取登录模块源码定位问题", "status": "pending"},
    {"id": "2", "title": "修复验证并回归测试登录", "status": "pending"},
]


# ---------------------------------------------------------------------------
# plan_review：计划评审
# ---------------------------------------------------------------------------

def test_review_passes_without_model():
    ok, fb = review_plan_steps(None, "修复登录", STEPS)
    assert ok and fb == ""


def test_review_passes_high_relevance():
    m = _FakeModel(lambda q, t: 0.8)
    ok, fb = review_plan_steps(m, "修复登录", STEPS)
    assert ok and fb == ""
    assert m.calls and m.calls[0][0] == "修复登录"
    assert m.calls[0][1] == [s["title"] for s in STEPS]


def test_review_rejects_low_relevance_with_feedback():
    m = _FakeModel(lambda q, t: 0.2)
    ok, fb = review_plan_steps(m, "修复登录", STEPS)
    assert not ok
    assert "平均相关性" in fb and "重新规划" in fb


def test_review_passes_on_model_error():
    ok, fb = review_plan_steps(_BoomModel(), "修复登录", STEPS)
    assert ok and fb == ""


# ---------------------------------------------------------------------------
# plan_review：收尾拦截
# ---------------------------------------------------------------------------

def test_completion_passes_without_model_or_request():
    assert check_plan_completion(None, "请求", ["证据"])[0]
    assert check_plan_completion(_FakeModel(lambda q, t: 0.9), "", ["证据"])[0]
    assert check_plan_completion(_FakeModel(lambda q, t: 0.9), "请求", [""])[0]


def test_completion_passes_when_relevant():
    m = _FakeModel(lambda q, t: 0.9)
    ok, fb = check_plan_completion(m, "修复登录", ["已定位并修复登录问题"])
    assert ok and fb == ""


def test_completion_rejects_insufficient_evidence():
    m = _FakeModel(lambda q, t: 0.2)
    ok, fb = check_plan_completion(m, "修复登录", ["随便答了答"])
    assert not ok
    assert "尚未真正完成" in fb and "不要在没有依据的情况下直接回答" in fb


def test_completion_passes_on_model_error():
    ok, fb = check_plan_completion(_BoomModel(), "修复登录", ["证据"])
    assert ok and fb == ""


# ---------------------------------------------------------------------------
# update_plan(op='create')：评审拦截 + 重试上限
# ---------------------------------------------------------------------------

def _registry(tmp_path):
    from agent.tools.base import ToolRegistry
    cfg = AgentConfig()
    cfg.workspace = str(tmp_path)
    cfg.knowledge_base = ""
    reg = ToolRegistry(config=cfg, callbacks=SimpleNamespace())
    reg.session = Session(budget=ContextBudget(30_000))
    return reg


def _create_args(goal="修复登录超时问题"):
    return {"op": "create", "goal": goal,
            "steps": [{"title": s["title"]} for s in STEPS]}


def test_create_review_rejected_then_accepted(tmp_path, monkeypatch):
    """低相关计划被拒绝并要求重新规划；重试超上限后放行。"""
    reg = _registry(tmp_path)
    from agent import plan_review as pr

    m = _FakeModel(lambda q, t: 0.2)
    monkeypatch.setattr(pr, "load_plan_model", lambda cfg: m)
    res = reg.execute("update_plan", _create_args())
    assert not res.success
    assert "重新规划" in (res.error or "")
    # 第 2 次仍拒绝（上限 2），第 3 次放行
    assert not reg.execute("update_plan", _create_args()).success
    res3 = reg.execute("update_plan", _create_args())
    assert res3.success
    assert reg.session.get_plan() is not None
    assert PLAN_REVIEW_MAX_REJECTS == 2


def test_create_review_skipped_when_model_unavailable(tmp_path, monkeypatch):
    """决策模型不可用 → 拦截点静默跳过，按原方式建计划。"""
    reg = _registry(tmp_path)
    from agent import plan_review as pr
    monkeypatch.setattr(pr, "load_plan_model", lambda cfg: None)
    res = reg.execute("update_plan", _create_args())
    assert res.success


def test_create_review_skipped_single_step(tmp_path, monkeypatch):
    """单步计划（平凡任务）不做评审。"""
    reg = _registry(tmp_path)
    from agent import plan_review as pr
    m = _FakeModel(lambda q, t: 0.1)
    monkeypatch.setattr(pr, "load_plan_model", lambda cfg: m)
    res = reg.execute("update_plan", {"op": "create", "goal": "读文件",
                                      "steps": [{"title": "读取文件"}]})
    assert res.success
    assert m.calls == []  # 单步计划未触发评审打分


def test_review_counter_resets_on_new_goal(tmp_path, monkeypatch):
    """换目标（新任务）→ 拒绝计数重置，新计划重新获得完整重试额度。"""
    reg = _registry(tmp_path)
    from agent import plan_review as pr
    m = _FakeModel(lambda q, t: 0.2)
    monkeypatch.setattr(pr, "load_plan_model", lambda cfg: m)
    for _ in range(PLAN_REVIEW_MAX_REJECTS + 1):
        reg.execute("update_plan", _create_args("任务A"))
    # 新目标 B：首次 create 又会被评审拒绝（计数已重置）
    res = reg.execute("update_plan", _create_args("完全不同的任务B"))
    assert not res.success


# ---------------------------------------------------------------------------
# Agent._decision_gate：收尾拦截
# ---------------------------------------------------------------------------

def _agent_with(model, plan=None):
    from agent.agent import Agent
    a = object.__new__(Agent)
    a.session = Session(budget=ContextBudget(30_000))
    a.config = AgentConfig()
    a._plan_model_cache = model
    a._plan_model_loaded = True
    a._current_request = "修复登录超时问题"
    a._gate_continues = 0
    a._gate_feedback = ""
    a._log = lambda *args, **kw: None
    if plan is not None:
        assert a.session.set_plan(plan)
    return a


_DONE_PLAN = {"goal": "修复登录超时问题", "steps": [
    {"id": "1", "title": "定位登录超时根因", "status": "done",
     "note": "已定位为 token 过期校验 bug"},
    {"id": "2", "title": "修复并回归验证登录", "status": "done",
     "note": "已修复并通过回归"},
]}


def test_gate_forces_continue_on_irrelevant_evidence():
    a = _agent_with(_FakeModel(lambda q, t: 0.2), plan=_DONE_PLAN)
    result = SimpleNamespace(text="完成了", reason=ExitReason.DONE)
    assert a._should_auto_continue(result) is True
    assert a._gate_continues == 1
    assert a._gate_feedback  # 续跑反馈已备好（经 __plan_boundary__ 注入）


def test_gate_passes_on_relevant_evidence():
    a = _agent_with(_FakeModel(lambda q, t: 0.9), plan=_DONE_PLAN)
    result = SimpleNamespace(text="完成了", reason=ExitReason.DONE)
    assert a._should_auto_continue(result) is False


def test_gate_uses_answer_when_no_plan():
    """无计划直答：以答复文本为证据；相关性低 → 拦截。"""
    a = _agent_with(_FakeModel(lambda q, t: 0.2))
    result = SimpleNamespace(text="大概是这样吧", reason=ExitReason.DONE)
    assert a._should_auto_continue(result) is True


def test_gate_skipped_without_model():
    a = _agent_with(None, plan=_DONE_PLAN)
    result = SimpleNamespace(text="完成了", reason=ExitReason.DONE)
    assert a._should_auto_continue(result) is False


def test_gate_skipped_after_max_force():
    a = _agent_with(_FakeModel(lambda q, t: 0.2), plan=_DONE_PLAN)
    a._gate_continues = COMPLETION_MAX_FORCE
    result = SimpleNamespace(text="完成了", reason=ExitReason.DONE)
    assert a._should_auto_continue(result) is False


def test_gate_open_steps_take_priority():
    """仍有 open steps → 原有续跑路径优先，不打分。"""
    m = _FakeModel(lambda q, t: 0.9)
    plan = {"goal": "修复登录", "steps": [
        {"id": "1", "title": "第一步", "status": "done", "note": ""},
        {"id": "2", "title": "第二步未完成", "status": "doing", "note": ""},
    ]}
    a = _agent_with(m, plan=plan)
    result = SimpleNamespace(text="部分完成", reason=ExitReason.DONE)
    assert a._should_auto_continue(result) is True
    assert m.calls == []  # 未走决策模型打分
