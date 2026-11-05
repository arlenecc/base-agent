"""L0~L4 记忆体系改进（2026-11-05）的行为测试。

覆盖：全文/摘要按天落盘、L0+L1 注入、L1 同步、SOP 只读一次、长期记忆门控、
L2/L3 写入后同步 L1、往日归档与挖掘、辅助引擎压缩与词级整理。
"""
from __future__ import annotations

import json
import os
import sys
import zipfile
from datetime import datetime
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.agent import Agent  # noqa: E402
from agent.config import AgentConfig  # noqa: E402
from agent.kernel.outcome import TurnRecord  # noqa: E402
from agent.kernel import ContextBudget, Session  # noqa: E402
from agent.llm_client import StreamEvent  # noqa: E402
from agent.memory_sop_texts import (  # noqa: E402
    MEMORY_MANAGEMENT_SOP, SOP_KEY_INFO_HINT,
)
from agent.memory_v2 import (  # noqa: E402
    MemoryLayout, archive_past_sessions, cleanup_memory, is_l23_write,
    is_l3_sop_read, memory_system_block, mine_history_insight, record_turn,
    sync_l1,
)
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


def _rec(idx=1, summary="第1步完成"):
    return TurnRecord(
        index=idx, prompt="用户问：主要思想是什么", assistant_text="助手正文",
        raw_content="<thinking>...</thinking>助手完整原文", summary=summary,
    )


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


# ---------------------------------------------------------------------------
# 改进 1 + 2：全文与摘要按当天日期追加
# ---------------------------------------------------------------------------

def test_record_turn_appends_full_and_digest_by_day(layout):
    now = datetime(2026, 11, 5, 10, 23, 1)
    record_turn(layout, _rec(1), "SYS_PROMPT_MARK", now=now)
    record_turn(layout, _rec(2, "第2步完成"), "SYS_PROMPT_MARK", now=now)

    full = open(layout.full_path("2026-11-05"), encoding="utf-8").read()
    digest = open(layout.digest_path("2026-11-05"), encoding="utf-8").read()
    # 全文：两轮都在（追加而不是覆盖），且含 system prompt 段与完整原文
    assert full.count("## [T") == 2
    assert "SYS_PROMPT_MARK" in full          # 改进 7 要剥离的段
    assert "助手完整原文" in full              # 全文（assistant-response）
    assert "用户问" in full
    # 摘要：每轮一行 [Tn]，不含全文
    assert "[T1] 10:23:01" in digest and "[T2] 10:23:01" in digest
    assert "第1步完成" in digest and "第2步完成" in digest
    assert "助手完整原文" not in digest


def test_session_finish_turn_triggers_recorder(layout):
    """落盘挂在 finish_turn 统一回调：所有收尾分支都不漏轮。"""
    s = Session(budget=ContextBudget(30_000))
    seen = []
    s.on_turn_finished = seen.append
    r1 = s.start_turn("问题1")
    s.finish_turn(r1, "答1", [], [], "摘要1")
    r2 = s.start_turn("问题2")
    s.finish_turn(r2, "答2", [], [], "摘要2")
    assert [r.index for r in seen] == [1, 2]
    # 回调抛异常不能影响主流程
    s.on_turn_finished = lambda rec: (_ for _ in ()).throw(RuntimeError("disk full"))
    r3 = s.start_turn("问题3")
    s.finish_turn(r3, "答3", [], [], "摘要3")
    assert r3.summary == "摘要3"
    # reset 不能把回调清掉（UI 清空对话后仍要继续落盘）
    s.reset()
    assert s.on_turn_finished is not None


def test_agent_run_persists_turn(cfg):
    """端到端：一轮真实对话后当天全文/摘要文件存在。"""
    mock = _MockLLM([[
        _evt("content", "你好，有什么可以帮你的吗？"), _evt("done"),
    ]])
    agent = Agent(llm=mock, config=cfg, callbacks=_Noop())
    agent.kernel.run("你好")
    day = datetime.now().strftime("%Y-%m-%d")
    lay = MemoryLayout(cfg.workspace)
    assert os.path.isfile(lay.full_path(day))
    assert os.path.isfile(lay.digest_path(day))
    assert "你好" in open(lay.full_path(day), encoding="utf-8").read()


# ---------------------------------------------------------------------------
# 改进 3 + 4：L0 恒定 + L1 实时索引进入 system prompt
# ---------------------------------------------------------------------------

def test_system_prompt_injects_l0_and_l1(cfg, layout):
    agent = Agent(llm=_MockLLM([]), config=cfg, callbacks=_Noop())
    prompt = agent._system_prompt()
    assert "核心公理" in prompt and "No Execution, No Memory" in prompt
    assert "# [Global Memory Insight]" in prompt
    assert prompt.index("核心公理") < prompt.index("# [Global Memory Insight]")
    # L1 文件真实落盘且可维护
    assert os.path.isfile(layout.l1_path)
    assert "[RULES]" in open(layout.l1_path, encoding="utf-8").read()
    # L0 恒定、L1 是独立的一段（两次构建一致 → KV 前缀稳定）
    assert agent._system_prompt() == prompt


def test_memory_system_block_reads_updated_l1(layout):
    with open(layout.l1_path, "w", encoding="utf-8") as f:
        f.write("# [Global Memory Insight]\nL2: 现空\nCUSTOM_RULE_MARK\n")
    assert "CUSTOM_RULE_MARK" in memory_system_block(layout)


def test_sync_l1_only_patches_index_lines(layout):
    with open(layout.l1_path, "w", encoding="utf-8") as f:
        f.write("# [Global Memory Insight]\nL2: 现空\nL3: 现空\nL4: raw_sessions/\n"
                "[RULES]\n1. 手写红线不可丢\n")
    with open(layout.l2_path, "w", encoding="utf-8") as f:
        f.write("## [SERVER]\nip=1.2.3.4\n\n## [PROXY]\nport=8080\n")
    for name in ("aaa_sop.md", "bbb_sop.md", "tool.py"):
        with open(os.path.join(layout.memory_dir, name), "w", encoding="utf-8") as f:
            f.write("x")

    changed = sync_l1(layout)
    assert changed
    text = open(layout.l1_path, encoding="utf-8").read()
    assert "SERVER" in text and "PROXY" in text
    assert "aaa_sop.md" in text and "tool.py" in text
    assert "1. 手写红线不可丢" in text  # RULES 原样保留
    assert len(text.splitlines()) <= 30


def test_sync_l1_caps_at_30_lines(layout):
    with open(layout.l1_path, "w", encoding="utf-8") as f:
        f.write("L2: 现空\n" + "\n".join(f"rule {i}" for i in range(27)))
    for i in range(20):
        with open(os.path.join(layout.memory_dir, f"s{i}_sop.md"), "w",
                  encoding="utf-8") as f:
            f.write("x")
    sync_l1(layout)
    assert len(open(layout.l1_path, encoding="utf-8").read().splitlines()) <= 30


# ---------------------------------------------------------------------------
# 改进 5：SOP 读取后内核强制「只读一次 + 提炼进 key_info」
# ---------------------------------------------------------------------------

def _call(name, **args):
    return SimpleNamespace(name=name, args=args, id="c1")


def test_classify_memory_tool_calls():
    assert is_l3_sop_read(_call("file_read", path="memory/foo_sop.md"))
    assert not is_l3_sop_read(_call("file_read", path="memory/history_insight.md"))
    assert not is_l3_sop_read(_call("rag_search", path="memory/foo_sop.md"))
    assert not is_l3_sop_read(_call("file_read", path="other/foo_sop.md"))
    assert is_l23_write(_call("file_append", path="memory/global_mem.txt"))
    assert is_l23_write(_call("file_append", path="memory/new_sop.md"))
    assert not is_l23_write(_call("file_write", path="notes.md"))


def test_sop_read_forces_key_info_hint(cfg):
    tc = [{"id": "c1",
           "function": {"name": "file_read",
                        "arguments": json.dumps({"path": "memory/foo_sop.md"})}}]
    mock = _MockLLM([
        [_evt("done", tool_calls=tc)],
        [_evt("content", "按 SOP 完成"), _evt("done")],
        [_evt("content", "兜底"), _evt("done")],
    ])
    agent = Agent(llm=mock, config=cfg, callbacks=_Noop())

    class _Stub:
        def dispatch(self, calls):
            return [ToolResult(True, output="SOP 正文：关键前置 + 典型坑")
                    for _ in calls]

    agent.kernel.dispatcher = _Stub()
    agent.kernel.run("按 SOP 做事")
    last_user = mock.calls[-1]["messages"][-1]["content"]
    assert SOP_KEY_INFO_HINT[:20] in last_user
    assert "update_working_checkpoint" in last_user


def test_l23_write_syncs_l1_and_invalidates_prompt_cache(cfg, layout):
    with open(layout.l2_path, "w", encoding="utf-8") as f:
        f.write("## [NEW_SECTION]\nfact=1\n")
    tc = [{"id": "c1",
           "function": {"name": "file_append",
                        "arguments": json.dumps({"path": "memory/global_mem.txt",
                                                 "content": "fact=2"})}}]
    mock = _MockLLM([
        [_evt("done", tool_calls=tc)],
        [_evt("content", "已记录"), _evt("done")],
        [_evt("content", "兜底"), _evt("done")],
    ])
    agent = Agent(llm=mock, config=cfg, callbacks=_Noop())

    class _Stub:
        def dispatch(self, calls):
            return [ToolResult(True, output="ok") for _ in calls]

    agent.kernel.dispatcher = _Stub()
    agent.kernel.run("记录事实")
    assert "NEW_SECTION" in open(layout.l1_path, encoding="utf-8").read()
    # 缓存失效后下一轮重新拼的 system prompt 必须带上新索引（改进 4：同步并加载）
    assert "NEW_SECTION" in mock.calls[-1]["messages"][0]["content"]


# ---------------------------------------------------------------------------
# 改进 6：file_append 只追加、不覆盖
# ---------------------------------------------------------------------------

def test_file_append_never_overwrites(cfg):
    from agent.tools import ToolRegistry

    p = os.path.join(cfg.workspace, "memory", "global_mem.txt")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write("## [OLD]\nkeep me\n")
    reg = ToolRegistry(config=cfg, callbacks=_Noop())
    res = reg.execute("file_append",
                      {"path": "memory/global_mem.txt", "content": "## [NEW]\nfact\n"})
    assert res.success
    text = open(p, encoding="utf-8").read()
    assert "keep me" in text and "## [NEW]" in text
    reg.shutdown()


# ---------------------------------------------------------------------------
# 改进 7：往日归档、骨架汇总、zip、删原文件、历史挖掘
# ---------------------------------------------------------------------------

def test_archive_past_sessions_strips_and_zips(layout):
    yesterday = datetime(2026, 11, 4, 10, 0, 0)
    record_turn(layout, _rec(1), "SECRET_SYS_MARK", now=yesterday)
    r2 = _rec(2, "第2步完成")
    r2.raw_content = "FULL_RESP_MARK 完整原文"
    record_turn(layout, r2, "SECRET_SYS_MARK",
                now=datetime(2026, 11, 4, 10, 10, 0))
    with open(layout.digest_path("2026-11-04"), "a", encoding="utf-8") as f:
        f.write("[T9] 手工摘要\n")

    n = archive_past_sessions(layout, today="2026-11-05")
    assert n == 1
    histories = open(layout.all_histories_path, encoding="utf-8").read()
    assert "用户问" in histories and "第2步完成" in histories
    # 剥离 system prompt 段与 assistant-response 段
    assert "SECRET_SYS_MARK" not in histories
    assert "FULL_RESP_MARK" not in histories
    assert "===== session 1104_1000-1104_1010 =====" in histories
    # zip 内文件名与 session 标识一致
    with zipfile.ZipFile(layout.zip_path("2026-11-04")) as zf:
        names = zf.namelist()
    assert "1104_1000-1104_1010.md" in names
    assert "1104_1000-1104_1010-digest.md" in names
    # 原文件已删除
    assert not os.path.exists(layout.full_path("2026-11-04"))
    assert not os.path.exists(layout.digest_path("2026-11-04"))
    # 增量：再跑不重复追加
    before = len(histories)
    assert archive_past_sessions(layout, today="2026-11-05") == 0
    assert len(open(layout.all_histories_path, encoding="utf-8").read()) == before


def test_archive_keeps_today(layout):
    record_turn(layout, _rec(1), "S", now=datetime(2026, 11, 5, 9, 0, 0))
    assert archive_past_sessions(layout, today="2026-11-05") == 0
    assert os.path.isfile(layout.full_path("2026-11-05"))


def test_mine_history_insight_is_incremental(layout):
    with open(layout.all_histories_path, "w", encoding="utf-8") as f:
        f.write("===== session 1104_1000-1104_1010 =====\n"
                "### USER\n你怎么又搞错了\n### AGENT\n抱歉\n\n"
                "===== session 1104_1100-1104_1130 =====\n"
                "### USER\n继续做知识库\n### AGENT\n好\n")
    calls = []

    def _llm(system, prompt):
        calls.append(prompt)
        return json.dumps({
            "ongoing": ["[持续活动] 知识库 | 1104_1100-1104_1130 | 继续做知识库 | 频次1"],
            "gone": ["[已消失] 旧项目 | 1104_1000-1104_1010 | 你怎么又搞错了 | 频次1"],
            "emotions": ["[情绪/责备] 1104_1000-1104_1010 | 你怎么又搞错了"],
        })

    cursor = mine_history_insight(layout, _llm)
    assert cursor == "1104_1100-1104_1130"
    text = open(layout.history_insight_path, encoding="utf-8").read()
    assert "## 活动知识层 / 持续活动" in text
    assert "## 活动知识层 / 已消失事项" in text   # 两个独立区域
    assert "## 情绪事件列表" in text
    assert f"LAST_SESSION: {cursor}" in text
    assert len(calls) == 1
    # 无新 session → 不再调用辅助模型、不重复追加情绪事件
    assert mine_history_insight(layout, _llm) == ""
    assert len(calls) == 1
    assert open(layout.history_insight_path, encoding="utf-8").read().count(
        "[情绪/责备]") == 1


def test_mine_history_insight_handles_year_rollover(layout):
    """跨年不能靠 session 标识的字典序：0101… < 1231… 也必须当新 session。"""
    with open(layout.all_histories_path, "w", encoding="utf-8") as f:
        f.write("===== session 1231_2000-1231_2100 =====\n### USER\n去年的事\n### AGENT\n好\n\n"
                "===== session 0101_0900-0101_0930 =====\n### USER\n新年的事\n### AGENT\n好\n")
    with open(layout.history_insight_path, "w", encoding="utf-8") as f:
        f.write("## 活动知识层 / 持续活动\n（暂无）\n\n"
                "## 活动知识层 / 已消失事项\n（暂无）\n\n"
                "## 情绪事件列表\n（暂无）\n\n"
                "## 增量标记\nLAST_SESSION: 1231_2000-1231_2100\n"
                "PROCESSED: 1231_2000-1231_2100\n")
    prompts = []

    def _llm(system, prompt):
        prompts.append(prompt)
        return json.dumps({"ongoing": ["[持续] 新年活动 | 0101_0900-0101_0930"],
                          "gone": [], "emotions": []})

    cursor = mine_history_insight(layout, _llm)
    assert cursor == "0101_0900-0101_0930"
    assert len(prompts) == 1
    assert "新年的事" in prompts[0] and "去年的事" not in prompts[0]


def test_mine_history_insight_keeps_unparsed_llm_output(layout):
    """辅助模型输出非 JSON 时不能丢原文，也不能卡在同一批反复重试。"""
    with open(layout.all_histories_path, "w", encoding="utf-8") as f:
        f.write("===== session 1104_1000-1104_1010 =====\n"
                "### USER\n重要但语气复杂的一段话\n### AGENT\n好\n")
    calls = []

    def _llm(system, prompt):
        calls.append(prompt)
        return "这不是 JSON，但含重要发现：用户对此很失望"

    assert mine_history_insight(layout, _llm)
    text = open(layout.history_insight_path, encoding="utf-8").read()
    assert "未解析发现" in text and "用户对此很失望" in text
    assert "PROCESSED:" in text
    # 增量已推进：无新 session 时不再重复调用辅助模型
    assert mine_history_insight(layout, _llm) == ""
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# 改进 8：压缩走辅助引擎、整理只允许词级 patch
# ---------------------------------------------------------------------------

def test_shrink_session_uses_aux_not_main(cfg):
    main_calls = []
    aux_calls = []

    class _LLM:
        def __init__(self, sink): self.sink = sink
        def chat_stream(self, messages, tools=None, temperature=0.7, **kwargs):
            self.sink.append(messages)
            return iter([StreamEvent(type="content", content="压缩摘要要点"),
                         StreamEvent(type="done")])

    agent = Agent(llm=_LLM(main_calls), config=cfg, callbacks=_Noop())
    agent.aux_llm = _LLM(aux_calls)
    s = agent.session
    for i in range(3):
        r = s.start_turn(f"问题{i}")
        s.finish_turn(r, f"回答{i}", [], [], f"摘要{i}")
    assert agent._shrink_session("fold-exhausted")
    assert aux_calls and not main_calls, "压缩不得占用主引擎"


def test_cleanup_memory_applies_only_word_patches(layout):
    with open(layout.l1_path, "w", encoding="utf-8") as f:
        f.write("L2: 现空\nL3: foo_sop(代理池)\n[RULES]\n1. 手写红线不可丢\n")

    def _llm(system, prompt):
        assert "记忆整理 SOP" in system
        return json.dumps({
            "l1_patches": [
                {"old": "foo_sop(代理池)", "new": "foo_sop"},     # 名字翻译 → 删
                {"old": "不存在的一行", "new": "X"},               # 找不到 → 跳过
                {"old": "1. 手写红线不可丢", "new": "OVERWRITE"},   # 红线不动
            ],
            "report": "删除了名字翻译；未动 RULES",
        })

    applied = cleanup_memory(layout, _llm)
    text = open(layout.l1_path, encoding="utf-8").read()
    assert "foo_sop\n" in text and "代理池" not in text
    assert "不存在的一行" not in text
    # RULES 是人工维护的红线区：自动整理不得改写
    assert "1. 手写红线不可丢" in text and "OVERWRITE" not in text
    assert os.path.isfile(layout.cleanup_report_path)
    assert applied == 1


def test_cleanup_memory_rejects_long_and_line_blowing_patches(layout):
    with open(layout.l1_path, "w", encoding="utf-8") as f:
        f.write("L2: 现空\n" + "\n".join(f"rule {i}" for i in range(29)))

    def _llm(system, prompt):
        return json.dumps({"l1_patches": [
            {"old": "L2: 现空", "new": "X" * 300},
            {"old": "rule 1", "new": "a\nb\nc"},
        ], "report": "none"})

    assert cleanup_memory(layout, _llm) == 0
    assert open(layout.l1_path, encoding="utf-8").read().startswith("L2: 现空")
