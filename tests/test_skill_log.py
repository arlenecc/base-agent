"""技能结构化日志：检索 / 加载 / 保存 / 索引事件的落盘验证。"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

from agent.config import AgentConfig
from agent.skill_log import log_skill_event, skill_log_path
from agent.tools.skill_search import (
    SkillLoadTool, SkillSaveTool, SkillSemanticSearchTool,
)


def _read_events(workspace):
    path = skill_log_path(workspace)
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_log_skill_event_writes_jsonl_and_clips(tmp_path):
    log_skill_event(str(tmp_path), "demo", query="中文查询", n=3,
                    huge="x" * 600)
    events = _read_events(str(tmp_path))
    assert len(events) == 1
    ev = events[0]
    assert ev["event"] == "demo" and ev["query"] == "中文查询" and ev["n"] == 3
    assert ev["huge"].endswith("…") and len(ev["huge"]) <= 501
    assert "ts" in ev


def test_log_redacts_secrets_and_can_be_disabled(tmp_path, monkeypatch):
    log_skill_event(str(tmp_path), "demo",
                    token="sk-abcdefgh12345678",
                    note="api_key=SECRET123", query="普通查询")
    raw = open(skill_log_path(str(tmp_path)), encoding="utf-8").read()
    assert "sk-abcdefgh" not in raw and "SECRET123" not in raw
    assert "[redacted]" in raw
    assert "普通查询" in raw

    before = len(_read_events(str(tmp_path)))
    monkeypatch.setenv("BASE_AGENT_SKILL_LOG", "0")
    log_skill_event(str(tmp_path), "hidden", query="不应记录")
    assert len(_read_events(str(tmp_path))) == before


class _StubRetriever:
    def __init__(self, skills_dir):
        self.skills_dir = skills_dir

    def retrieve(self, query, llm=None):
        return {"candidates": [
            {"dir": "demo", "name": "Demo Skill", "tags": "t",
             "description": "d", "score": 0.8, "bm25_score": 1.2},
        ]}

    def read_skill_dir(self, path):
        d = os.path.join(self.skills_dir, path)
        return d if os.path.isdir(d) else ""


class _StubRegistry:
    def __init__(self, retriever):
        self._retriever = retriever

    def get_skill_retriever(self):
        return self._retriever


def test_semantic_search_and_load_are_logged(tmp_path):
    cfg = AgentConfig(workspace=str(tmp_path))
    skills_dir = os.path.join(str(tmp_path), "skills")
    demo = os.path.join(skills_dir, "demo")
    os.makedirs(demo)
    with open(os.path.join(demo, "SKILL.md"), "w", encoding="utf-8") as f:
        f.write("---\nname: demo\ndescription: d\n---\n" + "x" * 800)

    reg = _StubRegistry(_StubRetriever(skills_dir))
    search = SkillSemanticSearchTool()
    search.bind(cfg, reg)
    assert search.run("做一个演示").success

    load = SkillLoadTool()
    load.bind(cfg, reg)
    res = load.run("demo")
    assert res.success and len(res.output) > 500

    events = _read_events(str(tmp_path))
    kinds = [e["event"] for e in events]
    assert "semantic_search" in kinds and "load" in kinds
    sem = next(e for e in events if e["event"] == "semantic_search")
    assert sem["candidates"][0]["dir"] == "demo"
    assert sem["candidates"][0]["score"] == 0.8
    loaded = next(e for e in events if e["event"] == "load")
    assert loaded["path"] == "demo" and loaded["source"] == "semantic"
    assert loaded["chars"] > 500


def test_save_event_logged(tmp_path):
    cfg = AgentConfig(workspace=str(tmp_path))
    reg = _StubRegistry(None)
    tool = SkillSaveTool()
    tool.bind(cfg, reg)
    res = tool.run("demo-save", "演示保存", "1. 做\n2. 完", tags="demo")
    assert res.success, res.error

    events = _read_events(str(tmp_path))
    saved = next(e for e in events if e["event"] == "save")
    assert saved["name"] == "demo-save"
    assert saved["path"].endswith("skills/demo-save/SKILL.md")
    assert os.path.isfile(saved["path"])

    # 同名覆盖必须显式 overwrite=true，避免自动学习覆盖手写技能
    again = tool.run("demo-save", "更新", "1. 改")
    assert not again.success and "already exists" in again.error
    assert tool.run("demo-save", "更新", "1. 改", overwrite=True).success
    events = _read_events(str(tmp_path))
    assert any(e["event"] == "save_failed" and e.get("reason") == "already_exists"
               for e in events)

    # 扫描器会跳过的目录名不能用来保存
    reserved = tool.run("tests", "保留名", "1. 做")
    assert not reserved.success and "reserved" in reserved.error


class _FakeEF:
    def __init__(self, name="fake"):
        self._name = name
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        return [[float(len(t) % 7 + 1)] * 8 for t in texts]

    def embed_query(self, text):
        return [1.0] * 8

    def name(self):
        return self._name


def test_index_build_event_logged(tmp_path):
    from agent.skill_retriever import SkillRetriever

    skills_dir = os.path.join(str(tmp_path), "skills")
    demo = os.path.join(skills_dir, "demo")
    os.makedirs(demo)
    with open(os.path.join(demo, "SKILL.md"), "w", encoding="utf-8") as f:
        f.write("---\nname: demo\ndescription: demo skill\n---\nbody")

    retriever = SkillRetriever(
        skills_dir=skills_dir,
        db_path=os.path.join(str(tmp_path), ".agent", "skill_index.lancedb"),
        embedding_function=_FakeEF(),
    )
    retriever.ensure_indexed()
    retriever.close()

    events = _read_events(str(tmp_path))
    built = next(e for e in events if e["event"] == "index_build")
    assert built["skills"] == 1 and built["rows"] == 1


def test_embedding_model_change_rebuilds_index(tmp_path):
    from agent.skill_retriever import SkillRetriever

    skills_dir = os.path.join(str(tmp_path), "skills")
    demo = os.path.join(skills_dir, "demo")
    os.makedirs(demo)
    with open(os.path.join(demo, "SKILL.md"), "w", encoding="utf-8") as f:
        f.write("---\nname: demo\ndescription: demo skill\n---\nbody")

    db = os.path.join(str(tmp_path), ".agent", "skill_index.lancedb")
    first = SkillRetriever(skills_dir=skills_dir, db_path=db,
                           embedding_function=_FakeEF("model-a"))
    first.ensure_indexed()
    first.close()

    second_ef = _FakeEF("model-b")
    second = SkillRetriever(skills_dir=skills_dir, db_path=db,
                            embedding_function=second_ef)
    second.ensure_indexed()
    second.close()

    assert second_ef.calls > 0, "切换模型后必须重新嵌入"
    sig_path = os.path.join(db, "embedding_signature.txt")
    assert open(sig_path, encoding="utf-8").read().strip() == "model-b"
    events = _read_events(str(tmp_path))
    rebuild = next(e for e in events if e["event"] == "index_rebuild")
    assert rebuild["old_model"] == "model-a"
    assert rebuild["new_model"] == "model-b"


def test_corrupt_index_degrades_without_crashing(tmp_path):
    """索引目录不可用（如被文件占位）时检索返回空候选，而不是抛异常。"""
    from agent.skill_retriever import SkillRetriever

    skills_dir = os.path.join(str(tmp_path), "skills")
    demo = os.path.join(skills_dir, "demo")
    os.makedirs(demo)
    with open(os.path.join(demo, "SKILL.md"), "w", encoding="utf-8") as f:
        f.write("---\nname: demo\ndescription: demo skill\n---\nbody")

    bad_db = os.path.join(str(tmp_path), "bad_index")
    with open(bad_db, "w", encoding="utf-8") as f:
        f.write("not a directory")

    retriever = SkillRetriever(
        skills_dir=skills_dir, db_path=bad_db, embedding_function=_FakeEF(),
    )
    result = retriever.retrieve("demo", llm=None)  # 不应抛异常
    assert result["candidates"] == []
    events = _read_events(str(tmp_path))
    assert any(e["event"] == "index_error" for e in events)
    retriever.close()
