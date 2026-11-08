"""失败恢复：计划步骤失败后必须自动换方案继续，而不是停下等用户打「继续」。

覆盖四层机制：
  1. 反思门失败感知：步骤失败后模型想文字收尾 → 提醒预算放宽 + 引用错误信息
  2. 原样重试检测：同名工具 + 相同参数再次失败 → 禁止重复
  3. 连续失败提示：停止同思路，先诊断再换方案
  4. 防过早 blocked：失败尝试不够就标记 blocked → 恢复 doing 再试
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.agent import Agent                      # noqa: E402
from agent.config import AgentConfig               # noqa: E402
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


def _last_user_prompt(mock, call_idx):
    """第 call_idx 次 LLM 调用的最后一条 user 消息（= 本轮锚点）。"""
    msgs = mock.calls[call_idx]["messages"]
    users = [str(m.get("content") or "") for m in msgs if m.get("role") == "user"]
    return users[-1] if users else ""


def _plan(op, **kw):
    return _tc("update_plan", dict(op=op, **kw))


# ---------------------------------------------------------------------------
# 1. 失败感知的反思门：步骤失败后想收尾 → 依据失败信息自动恢复
# ---------------------------------------------------------------------------

def test_failed_step_is_not_a_reason_to_stop(cfg):
    """步骤失败后模型想放弃 → 被拦回，且提醒里必须带失败信息与恢复指令。"""
    mock = MockLLM([
        # 1: 规划
        [_evt("done", tool_calls=_plan("create", goal="读文件并总结",
                                       steps=[{"title": "读 a.txt"}, {"title": "写总结"}]),
          usage={"total_tokens": 1})],
        # 2: 第一步就失败（文件不存在）
        [_evt("done", tool_calls=_tc("file_read", {"path": "a.txt"}),
              usage={"total_tokens": 1})],
        # 3: 模型想放弃 → 必须被失败感知的反思门拦回
        [_evt("content", "a.txt 不存在，这个任务我做不了"), _evt("done", usage={"total_tokens": 1})],
        # 4: 被拦后依据失败信息自救：先创建文件再读
        [_evt("done", tool_calls=_tc("file_write", {"path": "a.txt", "content": "hello"}),
              usage={"total_tokens": 1})],
        [_evt("done", tool_calls=_tc("file_read", {"path": "a.txt"}),
              usage={"total_tokens": 1})],
        [_evt("done", tool_calls=_plan("patch", step_id="1", status="done", note="已读"),
              usage={"total_tokens": 1})],
        [_evt("done", tool_calls=_plan("patch", step_id="2", status="done", note="已写"),
              usage={"total_tokens": 1})],
        [_evt("content", "完成"), _evt("done", usage={"total_tokens": 1})],
    ])
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    out = agent.kernel.run("读 a.txt 并总结")

    assert out.text == "完成"
    assert "未完成步骤" not in out.text
    assert agent.session.plan_open_steps() == []
    # 第 4 次调用的锚点（被拦后的提醒）必须：引用失败细节 + 要求自动恢复、
    # 明确不许停下来问用户 + 列出已试过且失败的方案（避免原地打转）。
    nudge = _last_user_prompt(mock, 3)
    assert "a.txt" in nudge, "提醒应引用失败细节"
    assert ("禁止停下来" in nudge or "自动解决" in nudge), "提醒应要求自动恢复而非停下"
    assert "已试过且失败" in nudge, "提醒应列出已失败方案，逼模型换没试过的路"


def test_failure_widens_nudge_budget(cfg):
    """有失败发生时反思门预算放宽（8 次 vs 默认 2 次）——不会早早收尾。"""
    from agent.kernel.loop import _FINISH_NUDGE_FAIL_MAX, _FINISH_NUDGE_MAX

    give_up = [_evt("content", "放弃"), _evt("done", usage={"total_tokens": 1})]
    mock = MockLLM([
        [_evt("done", tool_calls=_plan("create", goal="g",
                                       steps=[{"title": "a"}, {"title": "b"}]),
          usage={"total_tokens": 1})],
        # 一次失败
        [_evt("done", tool_calls=_tc("file_read", {"path": "missing.txt"}),
              usage={"total_tokens": 1})],
        # 之后连续 N 次只想文字收尾：前 _FINISH_NUDGE_FAIL_MAX 次应被拦，
        # 第 N+1 次才被放行
        *([give_up] * _FINISH_NUDGE_FAIL_MAX),
        [_evt("content", "最终答复"), _evt("done", usage={"total_tokens": 1})],
    ])
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    agent.max_iterations = 30
    out = agent.kernel.run("做任务")

    # 1 create + 1 fail + (FAIL_MAX 次被拦 + 1 次放行)
    assert len(mock.calls) == 2 + _FINISH_NUDGE_FAIL_MAX + 1
    assert _FINISH_NUDGE_FAIL_MAX > _FINISH_NUDGE_MAX, "失败时预算必须更宽"
    # 放行时仍必须告知用户任务未完成
    assert "未完成步骤" in out.text


def test_no_failure_keeps_default_budget(cfg):
    """无失败、且已经动过手时预算仍是默认 2 次（回归：不能把普通任务卡死）。

    注意「建完计划一次工具都没调用」走的是更宽的 PLAN 档（见 test_plan.py 的
    test_plan_created_but_not_started_is_pushed_to_execute）——那里多拦几次是对的，
    本用例验证的是**已经干过活**之后不要过度纠缠。
    """
    mock = MockLLM([
        [_evt("done", tool_calls=_plan("create", goal="g",
                                       steps=[{"title": "a"}, {"title": "b"}]),
          usage={"total_tokens": 1})],
        # 真正动过手 → 走默认档
        [_evt("done", tool_calls=_tc("file_write", {"path": "a.txt", "content": "x"}),
              usage={"total_tokens": 1})],
        [_evt("content", "我做完了"), _evt("done", usage={"total_tokens": 1})],
        [_evt("content", "真的做完了"), _evt("done", usage={"total_tokens": 1})],
        [_evt("content", "就这样吧"), _evt("done", usage={"total_tokens": 1})],
    ])
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    agent.max_iterations = 15
    out = agent.kernel.run("做任务")

    assert len(mock.calls) == 5      # 1 create + 1 work + (2 次被拦 + 1 次放行)
    assert "未完成步骤" in out.text


# ---------------------------------------------------------------------------
# 2. 原样重试检测
# ---------------------------------------------------------------------------

def test_identical_failed_call_is_called_out(cfg):
    """同名工具 + 相同参数再次失败 → 明确禁止原样重试。"""
    mock = MockLLM([
        [_evt("done", tool_calls=_tc("file_read", {"path": "x.txt"}),
              usage={"total_tokens": 1})],
        [_evt("done", tool_calls=_tc("file_read", {"path": "x.txt"}),
              usage={"total_tokens": 1})],
        [_evt("content", "ok"), _evt("done", usage={"total_tokens": 1})],
    ])
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    agent.kernel.run("读 x.txt")

    hint = _last_user_prompt(mock, 2)
    assert "完全相同" in hint, "应识别出原样重试"
    assert "禁止重复" in hint
    assert "file_read" in hint


def test_different_args_after_failure_is_not_repeat(cfg):
    """换了参数的同类调用不算原样重试，不应被点名。"""
    mock = MockLLM([
        [_evt("done", tool_calls=_tc("file_read", {"path": "x.txt"}),
              usage={"total_tokens": 1})],
        [_evt("done", tool_calls=_tc("file_read", {"path": "y.txt"}),
              usage={"total_tokens": 1})],
        [_evt("content", "ok"), _evt("done", usage={"total_tokens": 1})],
    ])
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    agent.kernel.run("读文件")

    hint = _last_user_prompt(mock, 2)
    assert "禁止重复" not in hint


# ---------------------------------------------------------------------------
# 3. 连续失败提示
# ---------------------------------------------------------------------------

def test_consecutive_failures_push_diagnosis_not_stopping(cfg):
    """连续失败 → 提示先诊断再换方案，且明确「不要停下来问用户是否继续」。"""
    mock = MockLLM([
        [_evt("done", tool_calls=_tc("file_read", {"path": "x.txt"}),
              usage={"total_tokens": 1})],
        [_evt("done", tool_calls=_tc("file_read", {"path": "y.txt"}),
              usage={"total_tokens": 1})],
        [_evt("content", "ok"), _evt("done", usage={"total_tokens": 1})],
    ])
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    agent.kernel.run("读文件")

    hint = _last_user_prompt(mock, 2)
    assert "连续失败 2 次" in hint
    assert "不要停下来问用户是否继续" in hint
    assert "诊断" in hint


def test_success_resets_fail_streak(cfg):
    """一次成功后连续失败计数清零，不应再出现「连续失败 N 次」提示。"""
    with open(os.path.join(cfg.workspace, "ok.txt"), "w") as f:
        f.write("hi")
    mock = MockLLM([
        [_evt("done", tool_calls=_tc("file_read", {"path": "x.txt"}),
              usage={"total_tokens": 1})],
        [_evt("done", tool_calls=_tc("file_read", {"path": "ok.txt"}),
              usage={"total_tokens": 1})],          # 成功 → 清零
        [_evt("done", tool_calls=_tc("file_read", {"path": "y.txt"}),
              usage={"total_tokens": 1})],          # 又失败一次（streak=1）
        [_evt("content", "ok"), _evt("done", usage={"total_tokens": 1})],
    ])
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    agent.kernel.run("读文件")

    hint = _last_user_prompt(mock, 3)
    assert "连续失败" not in hint


# ---------------------------------------------------------------------------
# 4. 防过早 blocked
# ---------------------------------------------------------------------------

def test_premature_block_nudges_more_attempts(cfg):
    """只失败一次就把步骤标记 blocked → 守卫要求恢复 doing 再试。"""
    mock = MockLLM([
        [_evt("done", tool_calls=_plan("create", goal="g", steps=[{"title": "a"}]),
              usage={"total_tokens": 1})],
        # 一次失败
        [_evt("done", tool_calls=_tc("file_read", {"path": "x.txt"}),
              usage={"total_tokens": 1})],
        # 立刻标记 blocked
        [_evt("done", tool_calls=_plan("patch", step_id="1", status="blocked",
                                       note="文件不存在"), usage={"total_tokens": 1})],
        # 被守卫拦回：恢复 doing 并自救
        [_evt("done", tool_calls=_plan("patch", step_id="1", status="doing"),
              usage={"total_tokens": 1})],
        [_evt("done", tool_calls=_tc("file_write", {"path": "x.txt", "content": "hi"}),
              usage={"total_tokens": 1})],
        [_evt("done", tool_calls=_plan("patch", step_id="1", status="done", note="ok"),
              usage={"total_tokens": 1})],
        [_evt("content", "完成"), _evt("done", usage={"total_tokens": 1})],
    ])
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    out = agent.kernel.run("做任务")

    guard = _last_user_prompt(mock, 3)
    assert "尝试不够" in guard
    assert "恢复" in guard and "doing" in guard
    assert out.text == "完成"
    assert agent.session.plan_open_steps() == []


def test_block_without_any_failure_is_challenged(cfg):
    """一步都没失败就标记 blocked（逃避执行）→ 同样被守卫拦回。"""
    mock = MockLLM([
        [_evt("done", tool_calls=_plan("create", goal="g", steps=[{"title": "a"}]),
              usage={"total_tokens": 1})],
        [_evt("done", tool_calls=_plan("patch", step_id="1", status="blocked",
                                       note="太难了"), usage={"total_tokens": 1})],
        [_evt("done", tool_calls=_plan("patch", step_id="1", status="doing"),
              usage={"total_tokens": 1})],
        [_evt("done", tool_calls=_tc("file_write", {"path": "a.txt", "content": "x"}),
              usage={"total_tokens": 1})],
        [_evt("done", tool_calls=_plan("patch", step_id="1", status="done", note="ok"),
              usage={"total_tokens": 1})],
        [_evt("content", "完成"), _evt("done", usage={"total_tokens": 1})],
    ])
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    out = agent.kernel.run("做任务")

    guard = _last_user_prompt(mock, 2)
    assert "尝试不够" in guard
    assert out.text == "完成"


def test_block_allowed_after_enough_failures(cfg):
    """失败次数足够（≥ _MIN_ATTEMPTS_BEFORE_BLOCK）后才标记 blocked → 守卫放行。"""
    from agent.kernel.loop import _MIN_ATTEMPTS_BEFORE_BLOCK

    script = [
        [_evt("done", tool_calls=_plan("create", goal="g", steps=[{"title": "a"}]),
              usage={"total_tokens": 1})],
    ]
    # 先攒够「充分尝试」的失败次数
    for i in range(_MIN_ATTEMPTS_BEFORE_BLOCK):
        script.append([_evt("done", tool_calls=_tc("file_read", {"path": f"f{i}.txt"}),
                            usage={"total_tokens": 1})])
    script.append(
        [_evt("done", tool_calls=_plan("patch", step_id="1", status="blocked",
                                       note="多种方法都失败"), usage={"total_tokens": 1})])
    script.append([_evt("content", "此路不通"), _evt("done", usage={"total_tokens": 1})])

    mock = MockLLM(script)
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    out = agent.kernel.run("做任务")

    # 守卫提醒落在 patch 之后那一轮的锚点上（= 最后一次调用）
    guard = _last_user_prompt(mock, len(script) - 1)
    assert "尝试不够" not in guard, "充分尝试后标记 blocked 不应被拦"
    assert out.text == "此路不通"
    # 阻塞决策已放行（守卫未拦）；任务完成后执行计划自动关闭，不再持久化
    assert agent.session.get_plan() is None


def test_block_nudge_is_bounded(cfg):
    """守卫提醒有上限：模型坚持放弃时最终尊重决定，不能无限循环。"""
    mock = MockLLM([
        [_evt("done", tool_calls=_plan("create", goal="g", steps=[{"title": "a"}]),
              usage={"total_tokens": 1})],
        # 之后每次都标记 blocked（一步都没试）——守卫最多提醒 2 次
        [_evt("done", tool_calls=_plan("patch", step_id="1", status="blocked", note="x"),
              usage={"total_tokens": 1})],
        [_evt("done", tool_calls=_plan("patch", step_id="1", status="blocked", note="x"),
              usage={"total_tokens": 1})],
        [_evt("done", tool_calls=_plan("patch", step_id="1", status="blocked", note="x"),
              usage={"total_tokens": 1})],
        [_evt("content", "放弃"), _evt("done", usage={"total_tokens": 1})],
    ])
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    out = agent.kernel.run("做任务")

    # 第 5 次调用（最后一次 blocked 之后）不应再带守卫提醒
    assert "尝试不够" not in _last_user_prompt(mock, 4)
    assert out.text == "放弃"


# ---------------------------------------------------------------------------
# 5. 首次失败就必须换方案（不是原样重发 / 不是停下向用户报告）
# ---------------------------------------------------------------------------

def test_first_failure_demands_a_new_approach(cfg):
    """第一次失败也要给出「换方法」指令，而不是等连续失败 2 次才提示。

    回归：旧实现只在 `_fail_streak >= 2` 时才提示，首次失败后模型最常见的
    反应是原样重发、或把错误念一遍给用户听然后停下。
    """
    mock = MockLLM([
        [_evt("done", tool_calls=_tc("file_read", {"path": "x.txt"}),
              usage={"total_tokens": 1})],
        [_evt("content", "ok"), _evt("done", usage={"total_tokens": 1})],
    ])
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    agent.kernel.run("读文件")

    hint = _last_user_prompt(mock, 1)
    assert "File does not exist: x.txt" in hint, "必须带上真实错误信息"
    assert "换一种方法" in hint or "换工具" in hint, "必须要求换方案"
    assert "不要原样重发" in hint, "必须禁止原样重试"
    assert "不要停下来向用户报告失败" in hint, "必须禁止停车"


# ---------------------------------------------------------------------------
# 6. 失败上下文跨「自动续跑段」保留
# ---------------------------------------------------------------------------

def test_failure_snapshot_carries_into_next_segment(cfg):
    """续跑段必须知道上一段卡在哪：否则它既拿不到错误信息，也拿不到失败感知
    的反思门预算 —— 表现为「续跑之后不再针对报错找办法，而是原地打转后停住」。
    """
    mock = MockLLM([
        [_evt("done", tool_calls=_tc("file_read", {"path": "x.txt"}),
              usage={"total_tokens": 1})],
        [_evt("content", "ok"), _evt("done", usage={"total_tokens": 1})],
    ])
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    k = agent.kernel
    k.run("读文件")

    snap = k.failure_snapshot()
    assert snap["failed_total"] == 1
    assert snap["last_fail"] is not None and snap["last_fail"][0] == "file_read"
    assert snap["fail_sigs"], "失败签名必须一起带走（原样重试检测靠它）"

    # 续跑段：本段没有任何新失败，但 seeded 的失败计数必须还在
    mock2 = MockLLM([
        [_evt("content", "继续"), _evt("done", usage={"total_tokens": 1})],
    ])
    agent.llm = mock2
    k.seed_failures(snap)
    k.run("继续")
    assert k._failed_total >= 1, "续跑段丢失了上一段的失败上下文"
    assert k._last_fail is not None and k._last_fail[0] == "file_read"


def test_auto_continue_segment_sees_the_previous_error(cfg):
    """端到端：第一段失败后被轮次上限截断，续跑段的锚点必须带上该错误。"""
    mock = MockLLM([
        # --- 第一段（max_iterations=3）---
        [_evt("done", tool_calls=_plan("create", goal="读完并总结",
                                       steps=[{"title": "读文件"}, {"title": "写总结"}]),
              usage={"total_tokens": 1})],
        [_evt("done", tool_calls=_tc("file_read", {"path": "x.txt"}),
              usage={"total_tokens": 1})],          # 失败
        [_evt("content", "我卡住了"), _evt("done", usage={"total_tokens": 1})],
        # --- 续跑段：锚点应含上一段的错误 ---
        [_evt("done", tool_calls=_tc("file_write", {"path": "x.txt", "content": "hi"}),
              usage={"total_tokens": 1})],
        [_evt("done", tool_calls=_plan("patch", step_id="1", status="done", note="已读"),
              usage={"total_tokens": 1})],
        [_evt("done", tool_calls=_plan("patch", step_id="2", status="done", note="已写"),
              usage={"total_tokens": 1})],
        # --- 第三段：收尾 ---
        # 计划全关闭后触发「成果核验反思」：专家把关最多打回 3 次，模型每次
        # 都坚持交卷（不调工具完善），连拦 3 次后放行收尾。
        [_evt("content", "全部完成"), _evt("done", usage={"total_tokens": 1})],
        [_evt("content", "全部完成"), _evt("done", usage={"total_tokens": 1})],
        [_evt("content", "全部完成"), _evt("done", usage={"total_tokens": 1})],
        [_evt("content", "全部完成"), _evt("done", usage={"total_tokens": 1})],
    ])
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    agent.max_iterations = 3
    out = agent.run("读完并总结")

    assert out == "全部完成"
    seg2_first = _last_user_prompt(mock, 3)
    assert "file_read" in seg2_first, (
        "续跑段的锚点必须引用上一段失败的工具与错误"
    )
    assert "File does not exist: x.txt" in seg2_first
    assert agent.session.plan_open_steps() == []
