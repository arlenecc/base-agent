"""上下文精简全流程的端到端串联验收。

与 test_memory_v2.py 的单点测试互补：这里验证「环节之间的衔接」——
多轮落盘 → L0/L1 注入 → 读 SOP → key_info 回注 → 长期记忆门控 →
L2/L3 写入 → L1 同步 → 归档 → 挖掘 → 整理 → 兜底压缩，
以及新增后台维护不会挤掉既有的事实抽取能力。
"""
from __future__ import annotations

import json
import os
import sys
import time
import zipfile
from datetime import datetime

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.agent import Agent, _AgentKernelHooks  # noqa: E402
from agent.config import AgentConfig  # noqa: E402
from agent.llm_client import StreamEvent  # noqa: E402
from agent.memory_sop_texts import SOP_KEY_INFO_HINT  # noqa: E402
from agent.memory_v2 import (  # noqa: E402
    MemoryLayout, archive_past_sessions, cleanup_memory,
    mine_history_insight, record_turn, sync_l1,
)
from agent.kernel.outcome import TurnRecord  # noqa: E402
from agent.tools.base import ToolResult  # noqa: E402


@pytest.fixture()
def cfg(tmp_path):
    c = AgentConfig()
    c.workspace = str(tmp_path)
    c.knowledge_base = str(tmp_path / "kb")
    return c


@pytest.fixture()
def layout(cfg):
    lay = MemoryLayout(cfg.workspace)
    lay.ensure()
    return lay


def _evt(t, content="", tool_calls=None):
    return StreamEvent(type=t, content=content,
                       tool_calls=tool_calls or [], usage={"total_tokens": 1})


class _MockLLM:
    def __init__(self, scripts):
        self._scripts = list(scripts)
        self.calls = []

    def chat_stream(self, messages, tools=None, temperature=0.7, **kwargs):
        self.calls.append({"messages": messages})
        if not self._scripts:
            raise AssertionError("MockLLM ran out of scripted responses")
        return iter(self._scripts.pop(0))


class _Noop:
    def is_cancelled(self): return False
    def on_content(self, t): pass
    def on_reasoning(self, t): pass
    def on_tool_start(self, n, a): pass
    def on_tool_end(self, n, r): pass
    def on_log(self, l): pass
    def on_usage(self, u): pass
    def on_token_speed(self, t, s): pass
    def on_finished(self): pass
    def confirm(self, m): return True
    def ask_user(self, p): return "ok"
    def on_skill_suggested(self, s): pass
    def on_plan_updated(self, p): pass
    def on_error(self, m): pass
    def on_context_shrunk(self, s, r): pass


class _StubDispatcher:
    """记录派发、按工具名返回成功结果，不让真实 registry 参与。"""

    def __init__(self, outputs=None):
        self.calls = []
        self.outputs = outputs or {}

    def schemas(self):
        return []

    def dispatch(self, calls):
        self.calls.extend((tc.name, dict(tc.args)) for tc in calls)
        return [ToolResult(True, output=self.outputs.get(tc.name, "ok"))
                for tc in calls]


def _tool(name, **args):
    return [{"id": f"c{name}", "function": {"name": name,
                                            "arguments": json.dumps(args)}}]


# ---------------------------------------------------------------------------
# 串联 1：多轮对话 → 全文/摘要按轮追加 + L0/L1 常驻 system prompt
# ---------------------------------------------------------------------------

def test_pipeline_multiturn_records_and_injects(cfg, layout):
    scripts = [
        [_evt("content", "第一轮答复内容"), _evt("done")],
        [_evt("content", "第二轮答复内容"), _evt("done")],
        [_evt("done", tool_calls=_tool("file_read", path="memory/ops_sop.md"))],
        [_evt("content", "第三轮答复内容"), _evt("done")],
    ]
    mock = _MockLLM(scripts)
    agent = Agent(llm=mock, config=cfg, callbacks=_Noop())
    stub = _StubDispatcher({"file_read": "SOP 关键前置与典型坑"})
    agent.kernel.dispatcher = stub

    agent.run("第一轮问题")
    agent.run("第二轮问题")
    agent.run("按 ops_sop 做事")

    day = datetime.now().strftime("%Y-%m-%d")
    full = open(layout.full_path(day), encoding="utf-8").read()
    digest = open(layout.digest_path(day), encoding="utf-8").read()
    assert full.count("## [T") >= 3, "每轮全文都要追加"
    assert digest.count("[T") >= 3, "每轮摘要都要追加"
    assert "第一轮问题" in full and "第三轮答复内容" in full
    # 每轮 system prompt 都带 L0 + L1，且同一 run 内恒定
    for call in mock.calls:
        sp = call["messages"][0]["content"]
        assert "核心公理" in sp and "# [Global Memory Insight]" in sp
    assert mock.calls[0]["messages"][0]["content"] == \
        mock.calls[1]["messages"][0]["content"]
    # 读到 L3 SOP 的那一轮触发了内核提示
    assert any(SOP_KEY_INFO_HINT[:20] in str(m.get("content") or "")
               for m in mock.calls[-1]["messages"])


# ---------------------------------------------------------------------------
# 串联 2：读 SOP → 提炼 key_info → 之后每轮随 anchor 回注
# ---------------------------------------------------------------------------

def test_pipeline_sop_to_key_info_every_turn(cfg):
    scripts = [
        [_evt("done", tool_calls=_tool("file_read", path="memory/ops_sop.md"))],
        [_evt("done", tool_calls=_tool("update_working_checkpoint",
                                       key_info="SOP要点: 先验证再写入; 坑: 别覆盖"))],
        [_evt("content", "按要点完成任务"), _evt("done")],
        [_evt("content", "兜底"), _evt("done")],
    ]
    mock = _MockLLM(scripts)
    agent = Agent(llm=mock, config=cfg, callbacks=_Noop())
    stub = _StubDispatcher({"file_read": "SOP 正文"})
    original = stub.dispatch

    def _dispatch(calls):
        for tc in calls:
            if tc.name == "update_working_checkpoint":
                # 真实工具会写 Session；stub 需要复刻，否则验证不了每轮回注
                agent.session.set_key_info(tc.args.get("key_info", ""))
        return original(calls)

    stub.dispatch = _dispatch
    agent.kernel.dispatcher = stub
    agent.kernel.run("按 SOP 做事")

    # 第 2 轮被强制要求提炼
    assert any(SOP_KEY_INFO_HINT[:20] in str(m.get("content") or "")
               for m in mock.calls[1]["messages"])
    # 第 3 轮 user 消息里带 key_info（每轮自动回注，不用再读 SOP）
    last_user = mock.calls[2]["messages"][-1]["content"]
    assert "SOP要点" in last_user and "别覆盖" in last_user
    assert agent.session.get_key_info() == \
        "SOP要点: 先验证再写入; 坑: 别覆盖"


# ---------------------------------------------------------------------------
# 串联 3：门控 → 提纯回执 → L2/L3 追加 → L1 同步 → 下一轮注入
# ---------------------------------------------------------------------------

def test_pipeline_long_term_write_syncs_l1(cfg, layout):
    # 预置 5 轮以通过门控
    scripts = []
    for i in range(5):
        # 参数必须不同：连续 5 次完全相同的调用会被内核的重复调用熔断拦下
        scripts.append([_evt("done", tool_calls=_tool("file_read", path=f"a{i}.txt"))])
    scripts.append([_evt("done", tool_calls=_tool("start_long_term_update",
                                                  focus="环境配置"))])
    scripts.append([_evt("done", tool_calls=_tool(
        "file_append", path="memory/global_mem.txt",
        content="## [SERVER]\nip=1.2.3.4\n"))])
    scripts.append([_evt("content", "已整理长期记忆"), _evt("done")])
    scripts.append([_evt("content", "兜底"), _evt("done")])

    mock = _MockLLM(scripts)
    agent = Agent(llm=mock, config=cfg, callbacks=_Noop())
    stub = _StubDispatcher()
    agent.kernel.dispatcher = stub
    # 该输入被内核判为多步任务，规划前置门会拦下首个执行工具要求先建计划。
    # 预置计划（既有保护机制正常工作），让测试聚焦记忆链路本身。
    agent.session.set_plan({
        "goal": "整理环境事实",
        "steps": [{"id": "s1", "title": "写入 L2 事实库", "status": "doing"}],
    })

    # 让 file_append 真的落盘（stub 不写文件），以便 sync_l1 读到新 section
    original_dispatch = stub.dispatch

    def _dispatch(calls):
        stub.calls.extend((tc.name, dict(tc.args)) for tc in calls)
        results = []
        for tc in calls:
            if tc.name == "start_long_term_update":
                # 走真实工具：门控 + 【记忆提纯】回执都参与串联
                results.append(agent.tools.execute("start_long_term_update", tc.args))
                continue
            if tc.name == "file_append":
                with open(layout.l2_path, "a", encoding="utf-8") as f:
                    f.write(tc.args.get("content", ""))
            results.append(ToolResult(True, output="ok"))
        return results

    stub.dispatch = _dispatch
    agent.kernel.run("整理环境事实")

    names = [n for n, _ in stub.calls]
    assert "start_long_term_update" in names and "file_append" in names
    # 提纯回执确实回灌给了模型
    assert any("MEMORY_PURIFY_REQUESTED" in str(m.get("content") or "")
               for call in mock.calls for m in call["messages"])
    text = open(layout.l1_path, encoding="utf-8").read()
    assert "SERVER" in text, "L2 写入后 L1 索引必须同步"
    assert len(text.splitlines()) <= 30
    # 同步后的索引进入后续轮次的 system prompt
    assert any("SERVER" in c["messages"][0]["content"] for c in mock.calls[-2:])


# ---------------------------------------------------------------------------
# 串联 4：往日归档 → 历史挖掘（保留旧活动）→ 记忆整理 → 三份产物齐全
# ---------------------------------------------------------------------------

def test_pipeline_archive_mine_cleanup(layout):
    y = datetime(2026, 11, 4, 9, 0, 0)
    record_turn(layout, TurnRecord(index=1, prompt="继续做知识库项目",
                                   assistant_text="好的", summary="知识库继续"),
                "SYS_MARK", now=y)
    with open(layout.l1_path, "w", encoding="utf-8") as f:
        f.write("L2: 现空\nL3: ops_sop(代理池)\n[RULES]\n1. 禁无条件杀python\n")

    assert archive_past_sessions(layout, today="2026-11-05") == 1
    assert os.path.isfile(layout.all_histories_path)
    assert os.path.isfile(layout.zip_path("2026-11-04"))

    def _llm(system, prompt):
        if "历史重点挖掘" in system:
            return json.dumps({
                "ongoing": ["[持续] 知识库 | 1104_0900-1104_0900 | 继续做知识库项目 | 频次1"],
                "gone": [],
                "emotions": [],
            })
        return json.dumps({
            "l1_patches": [{"old": "ops_sop(代理池)", "new": "ops_sop"}],
            "report": "删掉名字翻译",
        })

    assert mine_history_insight(layout, _llm)
    # 第二次扫描：旧活动必须仍在（活动知识层是持久状态，不是一次性报告）
    with open(layout.all_histories_path, "a", encoding="utf-8") as f:
        f.write("\n===== session 1105_1000-1105_1030 =====\n"
                "### USER\n新的一天开始\n### AGENT\n好\n")
    assert mine_history_insight(layout, _llm)
    insight = open(layout.history_insight_path, encoding="utf-8").read()
    assert "知识库" in insight
    assert "PROCESSED:" in insight and "1105_1000-1105_1030" in insight

    assert cleanup_memory(layout, _llm) >= 1
    l1 = open(layout.l1_path, encoding="utf-8").read()
    assert "ops_sop\n" in l1 and "1. 禁无条件杀python" in l1
    assert os.path.isfile(layout.cleanup_report_path)


# ---------------------------------------------------------------------------
# 串联 5：后台维护不能挤掉既有的事实抽取（不破坏其他功能）
# ---------------------------------------------------------------------------

def test_pipeline_maintenance_does_not_starve_fact_extraction(cfg):
    agent = Agent(llm=_MockLLM([]), config=cfg, callbacks=_Noop())
    agent.aux_llm = None  # 未加载辅助模型：维护只做归档
    # 让 _maybe_extract_facts_async 通过前置过滤
    agent._last_user_text = "请帮我把知识库项目的部署配置和环境事实整理一下"
    reply = "这是关于部署配置与环境事实的详细答复，包含具体路径与验证结论，内容足够长以通过事实抽取过滤。"
    for i in range(2):
        rec = agent.session.start_turn(f"问题{i}")
        agent.session.finish_turn(rec, reply, [], [], f"摘要{i}")

    events = []
    real_lock = agent._fact_extract_lock

    def _fake_extract(conversation):
        # 复刻 _extract_facts_sync 的「拿不到锁就跳过」语义
        if not real_lock.acquire(blocking=False):
            events.append("extract-skipped")
            return
        try:
            time.sleep(0.1)
            events.append("extract-done")
        finally:
            real_lock.release()

    agent._extract_facts_sync = _fake_extract
    hooks = _AgentKernelHooks(agent)
    hooks.finished(agent.session, reply)

    # 等待后台线程（维护让位 sleep 0.2s + acquire timeout）
    for _ in range(30):
        if "extract-done" in events:
            break
        time.sleep(0.1)
    assert "extract-done" in events, "事实抽取不得被后台维护抢锁挤掉"
    assert "extract-skipped" not in events
