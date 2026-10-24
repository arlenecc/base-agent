"""内核测试：三态自动机、五级压缩、Session、原子工具与能力进化闭环。

对应 docs/KERNEL_REFACTOR.md：
内核只控制流；历史交给 Session，能力交给 code_run/shell_run（+ rag/skill/mcp），
经验交给 memory（图谱）与 skills（SKILL.md）。
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.agent import Agent                      # noqa: E402
from agent.config import AgentConfig               # noqa: E402
from agent.kernel import (                         # noqa: E402
    ContextBudget, KernelState, Session, estimate_tokens,
    l0_capture, l1_trim, l2_summary, l3_fold,
)
from agent.kernel.outcome import ExitReason, StepOutcome, next_state  # noqa: E402
from agent.llm_client import StreamEvent           # noqa: E402


def _evt(t, content="", tool_calls=None, usage=None):
    return StreamEvent(type=t, content=content, tool_calls=tool_calls or [], usage=usage or {})


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
    """最小 AgentCallbacks 实现。"""

    def is_cancelled(self):
        return False

    def on_content(self, t):
        pass

    def on_reasoning(self, t):
        pass

    def on_tool_start(self, n, a):
        pass

    def on_tool_end(self, n, r):
        pass

    def on_log(self, l):
        pass

    def on_usage(self, u):
        pass

    def on_token_speed(self, t, s):
        pass

    def on_finished(self):
        pass

    def confirm(self, m):
        return True

    def ask_user(self, p):
        return "ok"


@pytest.fixture()
def cfg(tmp_path):
    c = AgentConfig()
    c.workspace = str(tmp_path)
    c.knowledge_base = str(tmp_path / "kb")
    return c


# ---------------------------------------------------------------------------
# 三态自动机
# ---------------------------------------------------------------------------


def test_transition_table():
    assert next_state(KernelState.THINK, "has_calls") == KernelState.ACT
    assert next_state(KernelState.THINK, "no_calls") == "exit"
    assert next_state(KernelState.ACT, "consolidate") == KernelState.CONSOLIDATE
    assert next_state(KernelState.ACT, "interrupt") == "exit"
    assert next_state(KernelState.CONSOLIDATE, "done") == KernelState.THINK
    # 未知组合一律终止，避免自动机静默卡死
    assert next_state(KernelState.ACT, "???") == "exit"


def test_step_outcome_finished_semantics():
    assert StepOutcome(data="x", next_prompt="\n").finished is False
    assert StepOutcome(data="x", next_prompt="").finished is True
    assert StepOutcome(data="x", should_exit=True).finished is True


def test_kernel_state_walks_think_act_and_returns_done(cfg):
    tc = [{"id": "c1", "function": {"name": "file_read",
                                    "arguments": json.dumps({"path": "a.txt"})}}]
    with open(os.path.join(cfg.workspace, "a.txt"), "w") as f:
        f.write("hello")
    mock = MockLLM([
        [_evt("done", tool_calls=tc, usage={"total_tokens": 1})],
        [_evt("content", "got it"), _evt("done", usage={"total_tokens": 1})],
    ])
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    result = agent.kernel.run("read a.txt")

    assert result.reason is ExitReason.DONE
    assert result.turns == 2
    assert agent.kernel.state == KernelState.THINK


def test_kernel_ask_user_answer_flows_back_to_model(cfg):
    """ask_user 是普通工具：用户答案作为 tool 结果回传给模型，模型继续任务。

    曾把它特判成 INTERRUPT —— 整轮在提问处终止、答案被丢弃，
    模型永远等不到用户的回答（「缺信息就问，别瞎猜」的协议因此失效）。
    """
    tc = [{"id": "c1", "function": {"name": "ask_user",
                                    "arguments": json.dumps({"prompt": "which one?"})}}]
    mock = MockLLM([
        [_evt("done", tool_calls=tc, usage={"total_tokens": 1})],
        [_evt("content", "好的，用方案B"), _evt("done", usage={"total_tokens": 1})],
    ])
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    result = agent.kernel.run("ask")

    assert result.reason is ExitReason.DONE
    assert result.text == "好的，用方案B"
    # 第二轮调用必须能看到用户答案（ask_user 的返回被回灌；Noop 返回 "ok"）
    second_messages = mock.calls[1]["messages"]
    assert any(m.get("role") == "tool" and "ok" in m.get("content", "")
               for m in second_messages)


def test_kernel_retries_empty_response_then_gives_up(cfg):
    """空回复重试最多 3 次，之后必须终止而不是无限循环。"""
    scripts = [[_evt("content", "   "), _evt("done", usage={})] for _ in range(6)]
    mock = MockLLM(scripts)
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    agent.kernel.max_turns = 50
    result = agent.kernel.run("hi")
    assert result.reason is ExitReason.DONE
    assert len(mock.calls) <= 4


# ---------------------------------------------------------------------------
# 五级压缩
# ---------------------------------------------------------------------------


def test_l0_capture_caps_every_type():
    assert len(l0_capture("x" * 10_000, hard_cap=1_000)) <= 1_200
    assert len(l0_capture({"a": "y" * 10_000}, hard_cap=1_000)) <= 1_200
    assert l0_capture(None) == ""


def test_l1_trim_keeps_head_and_tail_and_denoises():
    body = "H" * 3000 + "\n" + "<script>x</script>\n" + "T" * 3000
    out = l1_trim(body, 1_000)
    assert len(out) <= 1_100
    assert out.startswith("H") and out.rstrip().endswith("T")
    assert "<script>" not in out


def test_l2_summary_never_empty():
    assert l2_summary("<summary>读了配置，下一步改端口</summary>")
    assert l2_summary("", ["file_read"]) == "调用 file_read"
    assert l2_summary("") == "（无文本输出）"


def test_l3_fold_bounded():
    out = l3_fold([f"[T{i}] step" for i in range(500)], max_lines=10)
    assert len(out.split("\n")) <= 10


def test_l4_distill_roundtrip_is_safe():
    from agent.kernel import l4_extract_json, l4_normalize

    data = l4_extract_json('{"facts":["a"],"lessons":["b"],'
                           '"skill":{"name":"Do It!","summary":"s"}}')
    norm = l4_normalize(data)
    assert norm["skill"]["name"] == "do-it"
    assert norm["facts"] == ["a"] and norm["lessons"] == ["b"]
    # 脏输入不能抛
    assert l4_normalize(l4_extract_json("not json at all"))["facts"] == []


# ---------------------------------------------------------------------------
# Session：历史恒定 + 无悬挂 tool 消息
# ---------------------------------------------------------------------------


def test_session_folds_and_never_dangles(cfg):
    from agent.kernel import ToolCall

    s = Session(budget=ContextBudget(30_000), raw_turns=2, max_digest_lines=20)
    for i in range(1, 8):
        rec = s.start_turn(f"prompt {i}")
        s.finish_turn(
            rec, f"reply {i}",
            [ToolCall(id=f"c{i}", name="file_read", args={"path": "a"})],
            [{"id": f"c{i}", "name": "file_read", "content": "x" * 500}],
            f"摘要{i}",
        )
    rec = s.start_turn("prompt 8")
    msgs = s.build_messages("SYS", "prompt 8")

    # 原文窗口恒定：只保留最近 raw_turns 个已完成轮
    assistants = [m for m in msgs if m.get("role") == "assistant"]
    assert len(assistants) == 2
    # 折叠区存在
    assert any("<earlier_context>" in str(m.get("content")) for m in msgs)
    # 无悬挂 tool 消息
    for i, m in enumerate(msgs):
        if m.get("role") == "tool":
            assert msgs[i - 1].get("role") == "assistant" and msgs[i - 1].get("tool_calls")


def test_session_force_fold_moves_turns_to_digest(cfg):
    from agent.kernel import ToolCall

    s = Session(budget=ContextBudget(30_000), raw_turns=3)
    for i in range(4):
        rec = s.start_turn(f"p{i}")
        s.finish_turn(rec, f"r{i}", [], [], f"摘要{i}")
    assert len(s.turns) == 4
    assert s.force_fold(2) == 2
    assert len(s.turns) == 2
    assert len(s.extra_digest) == 2


def test_session_anchor_injects_key_info_and_truncation(cfg):
    s = Session(budget=ContextBudget(30_000))
    s.set_key_info("端口 8080 已占用")
    s.working["__truncated__"] = "上一轮被截断"
    anchor = s.anchor_prompt()
    assert "端口 8080 已占用" in anchor
    assert "上一轮被截断" in anchor
    # anchor_prompt 只 peek 不消费：真正的一次性语义是「本轮 turn 落地后失效」。
    # 旧实现在这里 pop，而该轮随后可能被反思门/前置门 rollback —— 提示随之消失，
    # 自动续跑注入的「上一条被判提前收尾」模型永远看不到。
    assert s.working.get("__truncated__") == "上一轮被截断"


def test_one_shot_hints_survive_rollback(cfg):
    """一次性提示必须活过 rollback，只在 turn 真正落地后才消费。"""
    s = Session(budget=ContextBudget(30_000))
    s.working["__plan_boundary__"] = "上一条被判提前收尾，已自动续跑"

    # 第一轮：anchor 播报后该轮被回滚（提示不得被吃掉）
    user_text = s.anchor_prompt()
    assert "上一条被判提前收尾" in user_text
    rec = s.start_turn(user_text)
    s.rollback_turn(rec) if hasattr(s, "rollback_turn") else None
    assert s.working.get("__plan_boundary__"), "rollback 后提示被吃掉了"

    # 第二轮：仍然播报（模型终于看到它）
    again = s.anchor_prompt()
    assert "上一条被判提前收尾" in again

    # 本轮被真正提交 → 提示到此消费
    rec2 = s.start_turn(again)
    s.finish_turn(rec2, "好的，继续执行", [], [], "继续执行")
    assert "__plan_boundary__" not in s.working


def test_kernel_and_session_share_one_budget(cfg):
    """内核与 Session 必须共用同一本账，否则 L1/L3 会各按各的数裁决。"""
    agent = Agent(llm=MockLLM([]), config=cfg, callbacks=Noop())
    assert agent.kernel.budget is agent.session.budget


def test_max_turns_is_per_run_not_cumulative(cfg):
    """max_turns 是「本次 run()」的上限，不是会话累计轮次。

    Session.turn_index 是历史编号（跨 run 单调递增）；拿它当循环上界会让
    Agent 在累计 N 轮之后彻底不再调用 LLM。
    """
    scripts = [[_evt("content", "ok"), _evt("done", usage={"total_tokens": 1})]
               for _ in range(4)]
    mock = MockLLM(scripts)
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    agent.max_iterations = 2          # 每次 run 最多 2 轮

    for i in range(4):
        assert agent.run(f"第{i}次") == "ok", f"第 {i} 次 run 未能产出回复"

    # 每次 run 只用了 1 轮，4 次都应该真的调用了 LLM
    assert len(mock.calls) == 4
    # 会话累计轮次确实在增长（说明上限不是靠重置历史实现的）
    assert agent.session.current_turn == 4


def test_reask_keeps_the_final_turn(cfg):
    """模型连续只吐代码块时，最终那一轮必须留在 Session 里。

    回滚会把 rec 从 session 摘掉，之后再 finish_turn 写的是游离对象 → 丢轮，
    导致 L4 蒸馏的 transcript 缺最后一条结论。
    """
    block = "```python\nprint('a')\n" + "\n".join(f"x{i}={i}" for i in range(20)) + "\n```"
    mock = MockLLM([
        [_evt("content", block), _evt("done", usage={"total_tokens": 1})],
        [_evt("content", block), _evt("done", usage={"total_tokens": 1})],
    ])
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    agent.max_iterations = 5
    out = agent.run("write something")

    assert out == block
    rec = next((t for t in agent.session.turns if t.assistant_text), None)
    assert rec is not None, "最终答复轮丢失"
    assert rec.assistant_text == block
    # 第一次 reask 被回滚（不占历史），最终这一轮必须留下并可被 L4 追溯
    assert len(agent.session.turns) == 1
    assert agent.session.all_digest_lines()


def test_periodic_consolidate_really_folds(cfg):
    """每 N 轮的周期性沉淀必须真的折叠，不能是空操作。"""
    with open(os.path.join(cfg.workspace, "a.txt"), "w") as f:
        f.write("A" * 50)
    tc = [{"id": "c1", "function": {"name": "file_read",
                                    "arguments": json.dumps({"path": "a.txt"})}}]
    scripts = [[_evt("done", tool_calls=tc, usage={"total_tokens": 1})] for _ in range(6)]
    scripts.append([_evt("content", "done"), _evt("done", usage={"total_tokens": 1})])
    mock = MockLLM(scripts)
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    agent.max_iterations = 10
    agent._distill_min_turns = 999
    agent.run("loop")

    # 7 轮里应该发生过折叠：turns 被折走、摘要行留下
    assert agent.session.extra_digest, "周期性 CONSOLIDATE 没有折叠任何轮"
    assert len(agent.session.turns) < 7


def test_prompt_stays_within_30k_over_long_run(cfg):
    """长任务下每一轮 prompt 的估算 token 数都不得超过 30k 硬上限。"""
    with open(os.path.join(cfg.workspace, "big.txt"), "w") as f:
        f.write("x" * 40_000)
    tc = [{"id": "c1", "function": {"name": "file_read",
                                    "arguments": json.dumps({"path": "big.txt"})}}]
    scripts = [[_evt("done", tool_calls=tc, usage={"total_tokens": 5})] for _ in range(10)]
    scripts.append([_evt("content", "done"), _evt("done", usage={"total_tokens": 5})])
    mock = MockLLM(scripts)
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    agent.max_iterations = 20
    agent._distill_min_turns = 999
    agent.run("read big")

    worst = 0
    for call in mock.calls:
        total = sum(
            estimate_tokens(json.dumps(m, ensure_ascii=False, default=str))
            for m in call["messages"]
        )
        worst = max(worst, total)
    assert worst <= 30_000, f"prompt 峰值 {worst} 超过 30k 硬上限"


def test_budget_quotas_sum_to_total():
    b = ContextBudget(30_000)
    q = b.quota
    assert q.system + q.earlier + q.raw + q.tools + q.reserve <= 30_001
    assert q.raw == 12_000
    b.set_used("raw", 20_000)
    assert b.raw_over
    assert "budget" in b.snapshot()


def test_estimate_tokens_is_monotonic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 40) >= estimate_tokens("a" * 20)
    assert estimate_tokens("你好" * 20) > 0


# ---------------------------------------------------------------------------
# 9 个原子工具 + 能力工具
# ---------------------------------------------------------------------------


def test_atomic_and_capability_tools_are_registered(cfg):
    from agent.tools import ToolRegistry

    reg = ToolRegistry(config=cfg, callbacks=Noop())
    names = set(reg.names())

    # GenericAgent 的 9 个原子工具
    for t in ("code_run", "file_read", "file_write", "file_patch", "web_scan",
              "update_working_checkpoint", "ask_user", "start_long_term_update"):
        assert t in names, f"缺少原子工具 {t}"

    # base-agent 的能力工具（全部保留）
    for t in ("shell_run", "work_memory", "memory_graph", "memory_search",
              "skill_search", "skill_semantic_search", "skill_load", "skill_save",
              "rag_search", "rag_outline", "rag_status", "rag_ingest"):
        assert t in names, f"缺少能力工具 {t}"
    reg.shutdown()


def test_alias_tool_delegates_to_target(cfg):
    from agent.tools import ToolRegistry

    reg = ToolRegistry(config=cfg, callbacks=Noop())
    with open(os.path.join(cfg.workspace, "p.txt"), "w") as f:
        f.write("old content")
    # file_patch 是 file_modify 的别名（对齐 GenericAgent 命名）
    res = reg.execute("file_patch", {"path": "p.txt", "old": "old", "new": "new"})
    assert res.success
    with open(os.path.join(cfg.workspace, "p.txt")) as f:
        assert f.read() == "new content"
    reg.shutdown()


def test_update_working_checkpoint_writes_session(cfg):
    from agent.tools import ToolRegistry

    reg = ToolRegistry(config=cfg, callbacks=Noop())
    reg.session = Session(budget=ContextBudget(30_000))
    res = reg.execute("update_working_checkpoint", {"key_info": "端口 8080 已占用"})
    assert res.success
    assert "8080" in reg.session.get_key_info()
    res = reg.execute("update_working_checkpoint", {"key_info": ""})
    assert res.success and reg.session.get_key_info() == ""
    reg.shutdown()


def test_start_long_term_update_marks_distill_and_returns_transcript(cfg):
    from agent.tools import ToolRegistry

    reg = ToolRegistry(config=cfg, callbacks=Noop())
    s = Session(budget=ContextBudget(30_000))
    rec = s.start_turn("部署到 staging")
    s.finish_turn(rec, "已部署", [], [], "部署完成")
    reg.session = s

    res = reg.execute("start_long_term_update", {"focus": "部署流程"})
    assert res.success
    assert "DISTILL_REQUESTED" in res.output
    assert "部署完成" in res.output
    reg.shutdown()


def test_skill_save_freezes_reusable_procedure(cfg):
    """能力进化闭环的写入端：写出 SKILL.md，供下次 skill_* 检索召回。"""
    from agent.tools import ToolRegistry

    reg = ToolRegistry(config=cfg, callbacks=Noop())
    res = reg.execute("skill_save", {
        "name": "Extract PDF Tables!",
        "description": "从 PDF 抽取表格为 CSV",
        "when_to_use": "用户要求导出 PDF 表格时",
        "steps": "1. pdftotext\n2. 解析\n3. 写 CSV",
        "pitfalls": "扫描件需先 OCR",
        "tags": "pdf,table",
    })
    assert res.success, res.error
    path = os.path.join(cfg.workspace, "skills", "extract-pdf-tables", "SKILL.md")
    assert os.path.isfile(path)
    body = open(path, encoding="utf-8").read()
    assert "name: extract-pdf-tables" in body
    assert "扫描件需先 OCR" in body
    reg.shutdown()


def test_skill_save_rejects_path_traversal(cfg):
    from agent.tools import ToolRegistry

    reg = ToolRegistry(config=cfg, callbacks=Noop())
    res = reg.execute("skill_save", {
        "name": "../../evil", "description": "d", "steps": "s",
    })
    assert res.success  # 名字被清洗成合法 slug
    assert not os.path.exists(os.path.join(cfg.workspace, "evil"))
    assert os.path.isdir(os.path.join(cfg.workspace, "skills", "evil"))
    reg.shutdown()


# ---------------------------------------------------------------------------
# rollback 后任务原文不得丢失（2026-10-24 用户反馈：模型自问「你没有给出具体任务」）
# ---------------------------------------------------------------------------

def _carry_task_unit():
    from agent.kernel.loop import _carry_task_text
    # 提示必须自带任务原文
    out = _carry_task_text("从 URL 获取今天的招聘信息并整理成列表", "[System] 请先建计划")
    assert "从 URL 获取今天的招聘信息并整理成列表" in out
    assert "[System] 请先建计划" in out
    # 无任务时原样返回提示；无提示时返回任务
    assert _carry_task_text("", "hint") == "hint"
    assert _carry_task_text("task", "") == "task"
    # 提示里已含任务时不重复拼接（幂等）
    assert _carry_task_text("task", "task\n\nhint") == "task\n\nhint"


def test_kernel_rollback_keeps_user_task_in_context():
    """规划前置门 rollback 后，下一轮 messages 必须仍含原始用户消息。

    旧实现：rollback 摘掉含用户消息的轮次 + pending 被提示整体覆盖 →
    模型再也看不到任务，开始自问「你没有给出具体任务内容」并复读。
    """
    from agent.agent import Agent
    from agent.config import AgentConfig

    MSG = "https://example.com/jobs 从这里获取今天所有的招聘信息，并整理成列表"
    # 第一轮：只在文本里写计划、不发工具 → 触发规划前置门（rollback + 提示）
    # 第二轮：正常发一个工具调用收尾。
    llm = MockLLM([
        [_evt("content", "我的计划：1) 读取网页 2) 解析 3) 整理。现在开始第一步。")],
        [_evt("done", tool_calls=[{
            "id": "c1", "type": "function",
            "function": {"name": "ask_user", "arguments": "{}"},
        }])],
    ])
    agent = Agent(llm=llm, config=AgentConfig(), callbacks=Noop())
    try:
        agent.run(MSG)
    except Exception:
        pass  # 工具缺参会报错，不影响断言（我们只看 messages）

    assert len(llm.calls) >= 2, "应当至少发生了两轮 LLM 调用"
    for i, call in enumerate(llm.calls):
        users = [m for m in call["messages"] if m.get("role") == "user"]
        joined = "\n".join(str(m.get("content") or "") for m in users)
        assert "example.com/jobs" in joined, (
            f"第 {i + 1} 轮的 user 消息里丢失了原始任务（URL 不见了）——"
            "rollback 覆盖 pending 的回归")
