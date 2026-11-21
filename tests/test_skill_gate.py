"""技能加载前判定（决策模型）单元 + 工具集成测试。

覆盖：多候选只允许最匹配者、单候选不合适禁止加载、全部不合适禁止加载、
模型不可用放行、无检索记录放行、旧回合记录不误伤、自动注入判定，以及
skill_search → skill_load 的工具级闭环。
"""
import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.config import AgentConfig
from agent.kernel.session import ContextBudget, Session
from agent.skill_gate import (
    SKILL_SUITABILITY_THRESHOLD, judge_skill_injection, judge_skill_load,
    record_candidates,
)


class _Model:
    """按技能文本前缀给分；记录打分输入便于断言请求文本。"""

    def __init__(self, mapping):
        self._m = mapping
        self.calls = []

    def score(self, query, texts):
        self.calls.append((query, list(texts)))
        out = []
        for t in texts:
            for k, v in self._m.items():
                if t.startswith(k):
                    out.append(v)
                    break
            else:
                out.append(0.01)
        return out


class _Boom:
    def score(self, query, texts):
        raise RuntimeError("boom")


def _reg(tmp_path, prompt="帮我提取 PDF 里的表格"):
    reg = SimpleNamespace()
    cfg = AgentConfig()
    cfg.workspace = str(tmp_path)
    reg.session = Session(budget=ContextBudget(30_000))
    reg.session.start_turn(prompt)
    return reg, cfg


CAND_A = {"name": "extract-pdf-tables", "path": "/skills/pdf",
          "description": "从 PDF 提取表格并导出 Excel", "tags": "pdf,表格"}
CAND_B = {"name": "deploy-web", "path": "/skills/deploy",
          "description": "部署 Web 应用到云端", "tags": "部署,云端"}


def _patch_model(monkeypatch, model):
    from agent import plan_review
    monkeypatch.setattr(plan_review, "load_plan_model", lambda cfg: model)


# ---------------------------------------------------------------------------
# judge_skill_load
# ---------------------------------------------------------------------------

def test_multi_candidate_only_best_allowed(tmp_path, monkeypatch):
    reg, cfg = _reg(tmp_path)
    _patch_model(monkeypatch, _Model({"extract-pdf-tables": 0.9,
                                      "deploy-web": 0.1}))
    record_candidates(reg, "提取 PDF 表格", [CAND_A, CAND_B])
    ok, msg = judge_skill_load(reg, cfg, "/skills/deploy")
    assert not ok and "extract-pdf-tables" in msg and "/skills/pdf" in msg
    ok, _ = judge_skill_load(reg, cfg, "/skills/pdf")
    assert ok


def test_single_candidate_irrelevant_blocked(tmp_path, monkeypatch):
    reg, cfg = _reg(tmp_path)
    _patch_model(monkeypatch, _Model({"deploy-web": 0.1}))
    record_candidates(reg, "提取 PDF 表格", [CAND_B])
    ok, msg = judge_skill_load(reg, cfg, "/skills/deploy")
    assert not ok and "禁止加载" in msg


def test_single_candidate_relevant_allowed(tmp_path, monkeypatch):
    reg, cfg = _reg(tmp_path)
    _patch_model(monkeypatch, _Model({"extract-pdf-tables": 0.9}))
    record_candidates(reg, "提取 PDF 表格", [CAND_A])
    assert judge_skill_load(reg, cfg, "/skills/pdf")[0]


def test_all_candidates_irrelevant_blocked(tmp_path, monkeypatch):
    reg, cfg = _reg(tmp_path)
    _patch_model(monkeypatch, _Model({"extract-pdf-tables": 0.2,
                                      "deploy-web": 0.1}))
    record_candidates(reg, "提取 PDF 表格", [CAND_A, CAND_B])
    ok, msg = judge_skill_load(reg, cfg, "/skills/pdf")
    assert not ok and "均不匹配" in msg


def test_requested_not_in_candidates_blocked(tmp_path, monkeypatch):
    """检索后只能加载检索返回的技能；加载候选之外的技能被拒绝。"""
    reg, cfg = _reg(tmp_path)
    _patch_model(monkeypatch, _Model({"extract-pdf-tables": 0.9}))
    record_candidates(reg, "提取 PDF 表格", [CAND_A])
    ok, msg = judge_skill_load(reg, cfg, "/skills/unknown")
    assert not ok and "不在本回合检索结果中" in msg


def test_empty_search_result_blocks_load(tmp_path, monkeypatch):
    """本回合检索无结果 → 加载任何技能都无依据，禁止。"""
    reg, cfg = _reg(tmp_path)
    _patch_model(monkeypatch, _Model({"deploy-web": 0.9}))
    record_candidates(reg, "不存在的技能", [])
    ok, msg = judge_skill_load(reg, cfg, "/skills/deploy")
    assert not ok and "没有返回任何结果" in msg


def test_model_unavailable_allows(tmp_path, monkeypatch):
    reg, cfg = _reg(tmp_path)
    _patch_model(monkeypatch, None)
    record_candidates(reg, "q", [CAND_B])
    assert judge_skill_load(reg, cfg, "/skills/deploy")[0]


def test_no_search_record_allows(tmp_path, monkeypatch):
    reg, cfg = _reg(tmp_path)
    _patch_model(monkeypatch, _Model({"deploy-web": 0.1}))
    assert judge_skill_load(reg, cfg, "/skills/deploy")[0]


def test_cleared_state_allows(tmp_path, monkeypatch):
    """新用户回合会清空检索记录 → 不再拦截（由 Agent.run 调用 clear_state）。"""
    from agent.skill_gate import clear_state
    reg, cfg = _reg(tmp_path)
    _patch_model(monkeypatch, _Model({"deploy-web": 0.1}))
    record_candidates(reg, "上一回合的检索", [CAND_B])
    reg.session.start_turn("新的任务：写诗")
    clear_state(reg)
    assert judge_skill_load(reg, cfg, "/skills/deploy")[0]


def test_scoring_error_allows(tmp_path, monkeypatch):
    reg, cfg = _reg(tmp_path)
    _patch_model(monkeypatch, _Boom())
    record_candidates(reg, "q", [CAND_B])
    assert judge_skill_load(reg, cfg, "/skills/deploy")[0]


def test_request_includes_conversation_state(tmp_path, monkeypatch):
    reg, cfg = _reg(tmp_path)
    model = _Model({"extract-pdf-tables": 0.9})
    _patch_model(monkeypatch, model)
    rec = reg.session.turns[-1]
    reg.session.finish_turn(rec, "已确认需要表格", summary="用户上传了季度报表 PDF")
    reg.session.start_turn("那就提取其中的表格")
    record_candidates(reg, "提取表格", [CAND_A])
    assert judge_skill_load(reg, cfg, "/skills/pdf")[0]
    request = model.calls[0][0]
    assert "提取其中的表格" in request          # 当前用户问句
    assert "对话进展" in request and "季度报表" in request  # 对话状态


# ---------------------------------------------------------------------------
# judge_skill_injection（Active skill 注入前）
# ---------------------------------------------------------------------------

def test_injection_blocked_when_irrelevant(tmp_path, monkeypatch):
    reg, cfg = _reg(tmp_path)
    _patch_model(monkeypatch, _Model({"deploy-web": 0.1}))
    ok, msg = judge_skill_injection(reg, cfg, "deploy-web", "部署 Web 应用到云端")
    assert not ok and "禁止使用" in msg


def test_injection_allowed_when_relevant(tmp_path, monkeypatch):
    reg, cfg = _reg(tmp_path, prompt="把这个 Web 应用部署到云端")
    _patch_model(monkeypatch, _Model({"deploy-web": 0.9}))
    assert judge_skill_injection(reg, cfg, "deploy-web", "部署 Web 应用到云端")[0]


def test_injection_skipped_without_model(tmp_path, monkeypatch):
    reg, cfg = _reg(tmp_path)
    _patch_model(monkeypatch, None)
    assert judge_skill_injection(reg, cfg, "deploy-web", "部署")[0]


# ---------------------------------------------------------------------------
# 工具级：skill_search → skill_load 闭环
# ---------------------------------------------------------------------------

def _write_skill(root, slug, name, desc, keywords, body):
    d = os.path.join(root, ".agent", "skills", slug)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "skill.json"), "w", encoding="utf-8") as f:
        json.dump({"name": name, "description": desc, "keywords": keywords,
                   "tags": keywords, "version": "1.0", "entry": "prompt.md"},
                  f, ensure_ascii=False)
    with open(os.path.join(d, "prompt.md"), "w", encoding="utf-8") as f:
        f.write(body)
    return d


def _tool_reg(tmp_path):
    from agent.tools.base import ToolRegistry
    cfg = AgentConfig()
    cfg.workspace = str(tmp_path)
    cfg.knowledge_base = ""
    reg = ToolRegistry(config=cfg, callbacks=SimpleNamespace())
    reg.session = Session(budget=ContextBudget(30_000))
    reg.session.start_turn("帮我提取 PDF 里的表格并导出 Excel")
    return reg, cfg


def test_tool_search_then_load_gate_closed_loop(tmp_path, monkeypatch):
    import agent.skill_gate as sg
    pdf_dir = _write_skill(tmp_path, "extract-pdf-tables", "extract-pdf-tables",
                           "从 PDF 提取表格并导出 Excel",
                           ["pdf", "表格", "excel"], "按步骤提取表格。")
    dep_dir = _write_skill(tmp_path, "deploy-web", "deploy-web",
                           "部署 Web 应用到云端", ["部署", "云端"], "部署步骤。")
    reg, cfg = _tool_reg(tmp_path)
    _patch_model(monkeypatch, _Model({"extract-pdf-tables": 0.9,
                                      "deploy-web": 0.1}))
    # 检索（关键词）→ 记录候候选
    res = reg.execute("skill_tools", {"op": "search", "query": "PDF 表格 部署"})
    assert res.success
    state = getattr(reg, sg._STATE_ATTR, None)
    assert state and len(state["candidates"]) >= 2
    # 加载非最匹配技能 → 拒绝
    bad = reg.execute("skill_tools", {"op": "load", "path": dep_dir})
    assert not bad.success and "extract-pdf-tables" in (bad.error or "")
    assert not os.path.exists(os.path.join(dep_dir, "__loaded__"))
    # 加载最匹配技能 → 放行并读到内容
    good = reg.execute("skill_tools", {"op": "load", "path": pdf_dir})
    assert good.success and "提取表格" in good.output
    reg.shutdown()


def test_tool_load_without_search_allowed(tmp_path, monkeypatch):
    """没有检索记录（例如模型记得旧技能路径）→ 不拦截。"""
    pdf_dir = _write_skill(tmp_path, "extract-pdf-tables", "extract-pdf-tables",
                           "从 PDF 提取表格", ["pdf"], "正文")
    reg, cfg = _tool_reg(tmp_path)
    _patch_model(monkeypatch, _Model({"extract-pdf-tables": 0.1}))
    res = reg.execute("skill_tools", {"op": "load", "path": pdf_dir})
    assert res.success
    reg.shutdown()


# ---------------------------------------------------------------------------
# 阈值与自动提示摘要注入
# ---------------------------------------------------------------------------

def test_threshold_is_055():
    assert SKILL_SUITABILITY_THRESHOLD == 0.55


class _LLM:
    def chat_stream(self, messages, tools=None, temperature=0.7):
        return iter([])


class _PermissiveCB:
    def __getattr__(self, name):
        return lambda *a, **k: None

    def is_cancelled(self):
        return False


def _hint_agent(tmp_path, monkeypatch, score: float):
    from agent.agent import Agent
    from agent import plan_review
    # 关键词能命中（.agent/skills 索引），但技能内容与部署任务无关
    _write_skill(tmp_path, "write-poem", "write-poem",
                 "按主题创作中文诗歌", ["部署", "云端"], "写诗步骤。")
    monkeypatch.setattr(plan_review, "load_plan_model",
                        lambda cfg: _Model({"write-poem": score}))
    cfg = AgentConfig()
    cfg.workspace = str(tmp_path)
    cfg.knowledge_base = ""
    agent = Agent(llm=_LLM(), config=cfg, callbacks=_PermissiveCB())
    agent.session.start_turn("把这个 Web 应用部署到云端，并给我访问链接")
    agent._last_user_text = "把这个 Web 应用部署到云端，并给我访问链接"
    return agent


def test_hint_summary_suppressed_when_irrelevant(tmp_path, monkeypatch):
    """自动提示摘要注入前判定：不匹配技能不注入 Relevant skill 提示。"""
    agent = _hint_agent(tmp_path, monkeypatch, 0.1)
    try:
        text = agent.volatile_context()
        assert "# Relevant skill" not in text
        # 该技能仍登记为候选 → 后续加载会被判定拦截（无依据可用）
        assert getattr(agent.tools, "_skill_gate_state", None) is not None
    finally:
        agent.tools.shutdown()


def test_hint_summary_shown_when_relevant(tmp_path, monkeypatch):
    agent = _hint_agent(tmp_path, monkeypatch, 0.9)
    try:
        text = agent.volatile_context()
        assert "# Relevant skill: write-poem" in text
    finally:
        agent.tools.shutdown()
