"""任务未起步时「模型在正文里追问用户」必须放行。

回归场景：用户输入是个多步任务（命中多步信号），但信息不足以开工，模型于是
在正文里问「请问要重构哪个文件？」。旧实现把它当成「规划完不执行」，整轮
rollback 掉再提醒「禁止问用户、必须调用工具」——提问被丢弃、用户一个字看不到，
内核拿同一份上下文重问，最多循环 _FINISH_NUDGE_PLAN_MAX 次，表现为「一遍遍
提示要用户输入任务，却完全不给输入机会」。

正确行为：任务还没起步时缺少的正是用户那句回答，应当把提问收下并结束本轮。
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.agent import Agent  # noqa: E402
from agent.config import AgentConfig  # noqa: E402
from agent.kernel.loop import KernelLoop  # noqa: E402
from agent.kernel.outcome import ExitReason  # noqa: E402
from agent.llm_client import StreamEvent  # noqa: E402


def _evt(t, content="", tool_calls=None, usage=None):
    return StreamEvent(type=t, content=content, tool_calls=tool_calls or [],
                       usage=usage or {})


class MockLLM:
    """按脚本依次返回流式事件；脚本耗尽即报错（用来断言「没有多余的轮次」）。"""

    def __init__(self, scripts):
        self._scripts = list(scripts)
        self.calls = []

    def chat_stream(self, messages, tools=None, temperature=0.7):
        self.calls.append({"messages": messages, "tools": tools})
        if not self._scripts:
            raise AssertionError("MockLLM ran out of scripted responses")
        return iter(self._scripts.pop(0))


class Noop:
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
# 判定函数
# ---------------------------------------------------------------------------

def test_looks_like_asking_user(cfg):
    k = object.__new__(KernelLoop)
    # 索取信息
    assert k._looks_like_asking_user("请问要重构哪个文件？")
    assert k._looks_like_asking_user("需要你提供一下目标目录")
    assert k._looks_like_asking_user("能否告诉我是哪个模块")
    # 非提问：陈述 / 已经开工的表达
    assert not k._looks_like_asking_user("")
    assert not k._looks_like_asking_user("我已经读完文件，接下来执行第二步。")
    assert not k._looks_like_asking_user("任务已完成。")


# ---------------------------------------------------------------------------
# 端到端：未起步 + 追问 → 收下提问，不再循环
# ---------------------------------------------------------------------------

def test_asking_user_before_any_work_exits_immediately(cfg):
    """脚本只给一轮：若内核还去 nudge 重问，MockLLM 会因脚本耗尽而失败。"""
    question = "这个任务需要先明确范围：请问要重构哪个文件？"
    mock = MockLLM([[_evt("content", question), _evt("done", usage={})]])
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())

    # "重构" 命中多步动词信号 → 旧逻辑会走「未写计划也未动手」的强制纠正分支
    result = agent.kernel.run("帮我重构一下")

    assert result.reason is ExitReason.DONE
    assert len(mock.calls) == 1, "追问应被收下，不应再消耗提醒轮次"
    assert question in result.text


def test_plan_created_but_not_started_still_nudges(cfg):
    """反例：计划已存在（说明任务已起步）→ 不许用「问一句」停车，必须继续。

    这里模型建好计划后只在正文里复述、不调工具，反思门应当拦下并重试。
    """
    tc = [{"id": "c1", "function": {"name": "update_plan",
                                    "arguments": '{"op":"create","goal":"重构","steps":["读代码","改代码"]}'}}]
    # 给足脚本：计划建好后模型只在正文里复述，反思门会连着提醒
    # _FINISH_NUDGE_PLAN_MAX 轮才放行，脚本不够会先耗尽而报 ERROR。
    mock = MockLLM(
        [[_evt("done", tool_calls=tc, usage={})]]
        + [[_evt("content", "已完成"), _evt("done", usage={})] for _ in range(8)]
    )
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    result = agent.kernel.run("帮我重构一下")

    assert result.reason is ExitReason.DONE
    # 计划已建 → 提问放行分支不应生效（否则只会调用 1 次就退出）
    assert agent.session.get_plan() is not None
    assert len(mock.calls) >= 2


# ---------------------------------------------------------------------------
# 简单输入豁免 / 重复成功调用检测
# ---------------------------------------------------------------------------

def test_simple_input_skips_plan_gate(cfg):
    """「hi」这类问候不该被逼着建执行计划。"""
    assert KernelLoop._is_simple_input("hi") is True
    assert KernelLoop._is_simple_input("你好") is True
    assert KernelLoop._is_simple_input("1+1 等于多少？") is True
    # 多步信号 → 不豁免
    assert KernelLoop._is_simple_input("先读 a.txt 然后改 b.py 并跑测试") is False
    assert KernelLoop._is_simple_input("帮我重构这个项目") is False


def test_repeated_successful_call_stops_the_loop(cfg):
    """反复调用**同一个成功**的工具（参数完全相同）必须被拦住。

    每轮都成功 → 反思门的「无工具调用」分支永不触发，旧实现会把
    max_iterations 全耗在空转上（日志里就是一串一模一样的 checkpoint 调用）。
    """
    tc = [{"id": "c1", "function": {"name": "update_working_checkpoint",
                                    "arguments": '{"key_info": "计划已建立", "persist": "false"}'}}]
    # 脚本给满，若内核不拦，会一直调用到脚本耗尽
    scripts = [[_evt("done", tool_calls=tc, usage={})] for _ in range(12)]
    mock = MockLLM(scripts)
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    result = agent.kernel.run("hi")

    assert result.reason is ExitReason.DONE
    # 第 3 次相同调用开始提醒，第 5 次强制收尾 → 总调用数应当远小于脚本数
    assert len(mock.calls) <= 6, f"重复调用未被拦住，共调用 {len(mock.calls)} 次"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
