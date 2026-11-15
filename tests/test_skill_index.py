"""Tests for SkillIndex (skill directory scanning/search) and the
skill_search / skill_load tools."""
from __future__ import annotations

import json
import os

import pytest

from agent.config import AgentConfig
from agent.skill_index import SkillIndex
from agent.skills import SkillManager
from agent.tools.base import ToolRegistry
from agent.tools.skill_search import SkillLoadTool, SkillSearchTool


def _make_skill(skills_dir, dirname, name, description="", keywords=None,
                tags=None, prompt="# instructions"):
    """Helper: create a skill directory with skill.json + prompt.md."""
    d = os.path.join(str(skills_dir), dirname)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "skill.json"), "w", encoding="utf-8") as f:
        json.dump({
            "name": name,
            "description": description,
            "keywords": keywords or [],
            "tags": tags or [],
            "entry": "prompt.md",
        }, f, ensure_ascii=False)
    with open(os.path.join(d, "prompt.md"), "w", encoding="utf-8") as f:
        f.write(prompt)
    return d


# ---------------------------------------------------------------------------
# SkillIndex
# ---------------------------------------------------------------------------

def test_scan_indexes_skill_directories(tmp_path):
    skills_dir = tmp_path / "skills"
    _make_skill(skills_dir, "pdf-extract", "PDF Extract",
                keywords=["pdf", "extract"], tags=["document"])
    _make_skill(skills_dir, "web-scrape", "Web Scrape", keywords=["web"])

    idx = SkillIndex(str(skills_dir))
    assert idx.scan(force=True) == 2
    names = {s["name"] for s in idx.list_all()}
    assert names == {"PDF Extract", "Web Scrape"}
    # Index file persisted
    assert os.path.isfile(os.path.join(str(skills_dir), "_index.json"))


def test_scan_skips_non_skill_directories(tmp_path):
    skills_dir = tmp_path / "skills"
    _make_skill(skills_dir, "real-skill", "Real")
    # Directory without skill.json must be ignored
    os.makedirs(str(skills_dir / "not-a-skill"))
    # Hidden / underscore-prefixed dirs must be ignored
    _make_skill(skills_dir, ".hidden", "Hidden")
    _make_skill(skills_dir, "_private", "Private")

    idx = SkillIndex(str(skills_dir))
    idx.scan(force=True)
    names = [s["name"] for s in idx.list_all()]
    assert names == ["Real"]


def test_scan_skips_invalid_skill_json(tmp_path):
    skills_dir = tmp_path / "skills"
    _make_skill(skills_dir, "good", "Good")
    # Malformed JSON
    bad = skills_dir / "bad-json"
    bad.mkdir()
    (bad / "skill.json").write_text("{not json", encoding="utf-8")
    (bad / "prompt.md").write_text("x", encoding="utf-8")
    # Missing name
    noname = skills_dir / "no-name"
    noname.mkdir()
    (noname / "skill.json").write_text('{"description": "x"}', encoding="utf-8")
    (noname / "prompt.md").write_text("x", encoding="utf-8")
    # Missing entry file
    noentry = skills_dir / "no-entry"
    noentry.mkdir()
    (noentry / "skill.json").write_text('{"name": "NoEntry"}', encoding="utf-8")

    idx = SkillIndex(str(skills_dir))
    idx.scan(force=True)
    assert [s["name"] for s in idx.list_all()] == ["Good"]


def test_index_persists_across_instances(tmp_path):
    skills_dir = tmp_path / "skills"
    _make_skill(skills_dir, "a-skill", "A Skill")
    idx1 = SkillIndex(str(skills_dir))
    idx1.scan(force=True)
    # New instance loads from _index.json without rescanning
    idx2 = SkillIndex(str(skills_dir))
    assert [s["name"] for s in idx2.list_all()] == ["A Skill"]


def test_scan_picks_up_new_skills(tmp_path):
    skills_dir = tmp_path / "skills"
    _make_skill(skills_dir, "first", "First")
    idx = SkillIndex(str(skills_dir))
    idx.scan(force=True)
    assert len(idx.list_all()) == 1
    # Directory mtime changes -> debounced scan should pick it up
    _make_skill(skills_dir, "second", "Second")
    names = {s["name"] for s in idx.list_all()}
    assert names == {"First", "Second"}


def test_search_ranks_by_relevance(tmp_path):
    skills_dir = tmp_path / "skills"
    _make_skill(skills_dir, "pdf", "PDF Extraction",
                description="Extract tables from PDF files",
                keywords=["pdf", "table", "extract"])
    _make_skill(skills_dir, "web", "Web Scraping",
                description="Scrape websites", keywords=["web", "scrape"])

    idx = SkillIndex(str(skills_dir))
    results = idx.search("extract table from pdf")
    assert results and results[0]["name"] == "PDF Extraction"
    # No match -> empty
    assert idx.search("unrelated xyzzy") == []
    # Empty query -> empty
    assert idx.search("") == []


def test_search_chinese_phrase_matches_keywords(tmp_path):
    """中文查询必须按 CJK bigram 命中关键词，不能把整句当一个 token。"""
    skills_dir = tmp_path / "skills"
    _make_skill(skills_dir, "kb", "知识库检索",
                description="从知识库检索书籍内容并总结",
                keywords=["知识库", "检索", "书籍"])
    _make_skill(skills_dir, "web", "网页抓取",
                description="抓取网页", keywords=["网页", "抓取"])

    idx = SkillIndex(str(skills_dir))
    for query in ("知识库检索", "检索书籍内容", "知识库"):
        results = idx.search(query)
        assert results and results[0]["path"] == "kb", query
    assert idx.search("无关的天气查询") == []


def test_read_skill_content(tmp_path):
    skills_dir = tmp_path / "skills"
    _make_skill(skills_dir, "pdf", "PDF", prompt="# PDF skill\nDo the thing.")
    idx = SkillIndex(str(skills_dir))
    content = idx.read_skill_content("pdf", "prompt.md")
    assert "Do the thing." in content
    # Missing skill / entry
    assert idx.read_skill_content("nope", "prompt.md") is None
    assert idx.read_skill_content("pdf", "missing.md") is None


def test_path_traversal_rejected(tmp_path):
    skills_dir = tmp_path / "skills"
    _make_skill(skills_dir, "pdf", "PDF")
    # A secret file outside the skills dir
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret", encoding="utf-8")

    idx = SkillIndex(str(skills_dir))
    assert idx.read_skill_content("..", "secret.txt") is None
    assert idx.read_skill_content("pdf", "../../secret.txt") is None
    assert idx.get_skill_path("..") is None


# ---------------------------------------------------------------------------
# SkillManager.create_dir_skill
# ---------------------------------------------------------------------------

def test_create_dir_skill(tmp_path):
    skills_dir = str(tmp_path / "skills")
    sm = SkillManager(path=str(tmp_path / "skills.json"), skills_dir=skills_dir)
    skill_dir = sm.create_dir_skill(
        name="PDF Extract", keywords=["pdf", "extract"],
        prompt="# Extract PDFs", description="Extract from PDFs",
        tags=["document"],
    )
    assert os.path.isdir(skill_dir)
    with open(os.path.join(skill_dir, "skill.json"), encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["name"] == "PDF Extract"
    assert meta["keywords"] == ["pdf", "extract"]
    assert meta["entry"] == "prompt.md"
    with open(os.path.join(skill_dir, "prompt.md"), encoding="utf-8") as f:
        assert "Extract PDFs" in f.read()
    # Flat record registered so match() hits it
    matched = sm.match("pdf extract")
    assert matched is not None and matched.name == "PDF Extract"
    assert "skill_tools" in matched.prompt


def test_create_dir_skill_unique_names(tmp_path):
    skills_dir = str(tmp_path / "skills")
    sm = SkillManager(path=str(tmp_path / "skills.json"), skills_dir=skills_dir)
    d1 = sm.create_dir_skill(name="Dup", keywords=["k"], prompt="p1")
    d2 = sm.create_dir_skill(name="Dup", keywords=["k"], prompt="p2")
    assert d1 != d2
    assert os.path.isdir(d1) and os.path.isdir(d2)


def test_create_dir_skill_requires_skills_dir(tmp_path):
    sm = SkillManager(path=str(tmp_path / "skills.json"))
    with pytest.raises(ValueError):
        sm.create_dir_skill(name="X", keywords=["k"], prompt="p")


def test_create_dir_skill_index_integration(tmp_path):
    """A dir skill created via SkillManager is discoverable via SkillIndex."""
    skills_dir = str(tmp_path / "skills")
    sm = SkillManager(path=str(tmp_path / "skills.json"), skills_dir=skills_dir)
    sm.create_dir_skill(name="Report Gen", keywords=["report"],
                        prompt="# make report")
    idx = SkillIndex(skills_dir)
    results = idx.search("report")
    assert results and results[0]["name"] == "Report Gen"


def test_volatile_context_hints_directory_skill(tmp_path):
    """目录技能不在 skills.json 时，Agent 也要给出 load 提示。"""
    from agent.agent import Agent

    skills_dir = os.path.join(str(tmp_path), ".agent", "skills")
    _make_skill(skills_dir, "kb", "知识库检索",
                description="从知识库检索书籍内容并总结",
                keywords=["知识库", "检索", "书籍"])
    cfg = AgentConfig(workspace=str(tmp_path))
    cfg.ensure_workspace()
    idx = SkillIndex(skills_dir)
    sm = SkillManager(path=os.path.join(str(tmp_path), ".agent", "skills.json"),
                      skills_dir=skills_dir)

    class _Tools:
        def get_rag_engine(self):
            return None

        def get_skill_index(self):
            return idx

    agent = Agent.__new__(Agent)
    agent.config = cfg
    agent.skills = sm
    agent.tools = _Tools()

    agent._last_user_text = "帮我从知识库里检索这本书的内容并总结"
    text = agent.volatile_context()
    assert "# Relevant skill: 知识库检索" in text
    assert 'skill_tools(op="load"' in text

    agent._last_user_text = "今天天气怎么样"
    assert "# Relevant skill" not in agent.volatile_context()


def test_skill_hint_index_matches_workspace_skill(tmp_path):
    """workspace/skills 的 SKILL.md 也要有轻量关键词提示（不依赖向量）。"""
    from agent.skill_hint import SkillHintIndex

    demo = tmp_path / "skills" / "invoice-organizer"
    demo.mkdir(parents=True)
    (demo / "SKILL.md").write_text(
        "---\nname: invoice-organizer\n"
        "description: 把报销发票按月份整理归档并命名\n"
        "tags: [发票, 归档]\n---\nbody",
        encoding="utf-8",
    )
    other = tmp_path / "skills" / "unity-tilemap-2d"
    other.mkdir(parents=True)
    (other / "SKILL.md").write_text(
        "---\nname: unity-tilemap-2d\ndescription: 在 Unity 里做 2D 瓦片地图\n---\n",
        encoding="utf-8",
    )

    idx = SkillHintIndex(str(tmp_path / "skills"))
    hits = idx.search("帮我按月份整理报销发票", top_k=1)
    assert hits and hits[0]["path"] == "invoice-organizer"
    assert hits[0]["score"] >= 2
    assert idx.search("今天天气怎么样") == []


def test_volatile_context_hints_workspace_skill(tmp_path):
    """没有 .agent/skills 命中时，workspace/skills 提示索引要兜底。"""
    from agent.agent import Agent
    from agent.skill_hint import SkillHintIndex

    demo = tmp_path / "skills" / "invoice-organizer"
    demo.mkdir(parents=True)
    (demo / "SKILL.md").write_text(
        "---\nname: invoice-organizer\ndescription: 整理报销发票并归档\n"
        "tags: [发票]\n---\n",
        encoding="utf-8",
    )
    cfg = AgentConfig(workspace=str(tmp_path))
    cfg.ensure_workspace()
    sm = SkillManager(path=os.path.join(str(tmp_path), ".agent", "skills.json"),
                      skills_dir=os.path.join(str(tmp_path), ".agent", "skills"))

    class _EmptyIndex:
        def search(self, query, top_k=1):
            return []

    class _Tools:
        def get_rag_engine(self):
            return None

        def get_skill_index(self):
            return _EmptyIndex()

        def get_skill_hint_index(self):
            return SkillHintIndex(str(tmp_path / "skills"))

    agent = Agent.__new__(Agent)
    agent.config = cfg
    agent.skills = sm
    agent.tools = _Tools()
    agent._last_user_text = "把这些报销发票按月份整理归档"

    text = agent.volatile_context()
    assert "# Relevant skill: invoice-organizer" in text
    assert 'path="invoice-organizer"' in text


# ---------------------------------------------------------------------------
# skill_search / skill_load tools
# ---------------------------------------------------------------------------

@pytest.fixture()
def registry(config, recording_callbacks):
    return ToolRegistry(config=config, callbacks=recording_callbacks)


def test_skill_search_tool_search_and_list(registry, config):
    skills_dir = os.path.join(config.workspace, ".agent", "skills")
    _make_skill(skills_dir, "pdf-extract", "PDF Extract",
                description="Extract tables from PDF",
                keywords=["pdf", "table"])

    tool = SkillSearchTool()
    tool.bind(config, registry)

    # search
    res = tool.run(op="search", query="extract pdf table")
    assert res.success
    hits = json.loads(res.output)
    assert hits and hits[0]["name"] == "PDF Extract"

    # list
    res = tool.run(op="list")
    assert res.success
    listed = json.loads(res.output)
    assert listed and listed[0]["path"] == "pdf-extract"

    # search without query -> error
    res = tool.run(op="search", query="")
    assert not res.success


def test_skill_load_tool(registry, config):
    skills_dir = os.path.join(config.workspace, ".agent", "skills")
    _make_skill(skills_dir, "web-scrape", "Web Scrape",
                prompt="# Scrape\nStep 1...")

    tool = SkillLoadTool()
    tool.bind(config, registry)

    res = tool.run(path="web-scrape")
    assert res.success
    assert "Step 1" in res.output

    # Missing skill -> error
    res = tool.run(path="nonexistent")
    assert not res.success

    # Traversal -> error
    res = tool.run(path="..", entry="secret.txt")
    assert not res.success


def test_skill_load_rejects_absolute_entry(registry, config):
    """`entry` is model-supplied. os.path.join drops the base when entry is
    absolute, so an unvalidated entry let the tool read any file on disk."""
    skills_dir = os.path.join(config.workspace, ".agent", "skills")
    _make_skill(skills_dir, "web-scrape", "Web Scrape", prompt="# Scrape")

    tool = SkillLoadTool()
    tool.bind(config, registry)

    # Absolute path escapes the skill directory entirely.
    res = tool.run(path="web-scrape", entry="/etc/passwd")
    assert not res.success
    assert "passwd" not in (res.output or "")

    # Parent traversal is rejected too.
    res = tool.run(path="web-scrape", entry="../../../../etc/passwd")
    assert not res.success
    assert "root:" not in (res.output or "")


def test_skill_search_tracks_workspace_change(config, recording_callbacks):
    """Switching the workspace must not keep serving the previous one's index."""
    from agent.tools.skill_search import _get_skill_index

    reg = ToolRegistry(config=config, callbacks=recording_callbacks)
    idx_a = _get_skill_index(config, reg)

    other = AgentConfig(workspace=os.path.join(str(config.workspace) + "_other"))
    other.ensure_workspace()
    idx_b = _get_skill_index(other, reg)

    assert idx_a is not idx_b
    assert idx_a._dir != idx_b._dir


def test_skill_tools_registered_by_default(registry):
    assert registry.get("skill_tools") is not None
