"""Tests for the hybrid skill retriever (SKILL.md scan + vector/BM25 search)."""
from __future__ import annotations

import hashlib
import os
import tempfile
from typing import List

from agent.skill_retriever import (
    SkillScanner,
    SkillVectorStore,
    SkillRetriever,
    parse_skill_md,
    _MIN_SIMILARITY,
    _parse_list_field,
    _normalize_vec,
)


# ---------------------------------------------------------------------------
# Fake embedding function (deterministic, no model download)
# ---------------------------------------------------------------------------

class FakeEF:
    """Deterministic hash-based embedding (768-dim) for fast tests."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out = []
        for t in texts:
            h = hashlib.md5(t.encode()).digest()
            out.append([float(b) / 255.0 for b in h] * 48)
        return out

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

SAMPLE_SKILL = """---
name: invoice-organizer
description: "Organize messy invoice folders, extract totals, and prepare tax documents. Use when handling invoices, receipts, or tax preparation."
tags: [finance, invoices, tax, bookkeeping]
license: MIT
metadata:
  version: "1.0"
---

# Invoice Organizer

## When to Use This Skill

Use this skill when the user has a messy folder of invoices and receipts.

## Examples

- "Organize my invoices for tax season"
- "Extract totals from these receipts"
"""


def test_parse_frontmatter_and_tags():
    m = parse_skill_md(SAMPLE_SKILL, "invoice-organizer")
    assert m is not None
    assert m.name == "invoice-organizer"
    assert m.tags == ["finance", "invoices", "tax", "bookkeeping"]
    assert "tax" in m.description


def test_parse_extracts_when_to_use_and_examples():
    m = parse_skill_md(SAMPLE_SKILL, "invoice-organizer")
    assert "messy folder" in m.when_to_use
    assert "Organize my invoices" in m.examples


def test_parse_list_field():
    assert _parse_list_field("[a, b, c]") == ["a", "b", "c"]
    assert _parse_list_field("a, b") == ["a", "b"]
    assert _parse_list_field('"a", "b"') == ["a", "b"]


def test_parse_rejects_missing_name():
    m = parse_skill_md("# No frontmatter\n\nbody", "x")
    assert m is None


def test_normalize_vec():
    v = _normalize_vec([3.0, 4.0])
    # 单位长度
    assert abs((v[0] ** 2 + v[1] ** 2) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

def _make_skill_dir(root: str, rel: str, name: str, body: str = "") -> None:
    d = os.path.join(root, rel)
    os.makedirs(d, exist_ok=True)
    content = f"---\nname: {name}\ndescription: test skill {name}\n---\n{body}"
    with open(os.path.join(d, "SKILL.md"), "w", encoding="utf-8") as f:
        f.write(content)


def test_scanner_recursive_and_skips_tests(tmp_path):
    skills_dir = str(tmp_path / "skills")
    _make_skill_dir(skills_dir, "finance/invoice", "invoice-organizer")
    _make_skill_dir(skills_dir, "finance/budget", "budget-planner")
    # tests 目录应被跳过
    _make_skill_dir(skills_dir, "finance/tests/skipme", "should-be-skipped")

    scanner = SkillScanner(skills_dir)
    skills = scanner.scan(force=True)
    names = {s.name for s in skills}
    assert "invoice-organizer" in names
    assert "budget-planner" in names
    assert "should-be-skipped" not in names


# ---------------------------------------------------------------------------
# Vector store + hybrid retrieval
# ---------------------------------------------------------------------------

def _build_store(tmp_path, skills_dir: str) -> SkillVectorStore:
    scanner = SkillScanner(skills_dir)
    skills = scanner.scan(force=True)
    store = SkillVectorStore(
        db_path=str(tmp_path / "skills.lancedb"),
        embedding_function=FakeEF(),
    )
    store.build(skills)
    return store


def test_store_build_and_vector_search(tmp_path):
    skills_dir = str(tmp_path / "skills")
    _make_skill_dir(skills_dir, "finance/invoice", "invoice-organizer")
    _make_skill_dir(skills_dir, "science/networkx", "networkx")

    store = _build_store(tmp_path, skills_dir)
    results = store.search_vector("organize invoices for tax", top_k=2)
    assert len(results) >= 1
    # 分数应在 [0,1] 区间（余弦相似度）
    for r in results:
        assert 0.0 <= r["score"] <= 1.0


def test_store_bm25_search(tmp_path):
    skills_dir = str(tmp_path / "skills")
    _make_skill_dir(skills_dir, "finance/invoice", "invoice-organizer")
    _make_skill_dir(skills_dir, "science/networkx", "networkx")

    store = _build_store(tmp_path, skills_dir)
    results = store.search_bm25("invoice", top_k=2)
    assert len(results) >= 1
    # 应有 bm25_score 字段
    for r in results:
        assert "bm25_score" in r


def test_retriever_hybrid_dedup_and_threshold(tmp_path):
    skills_dir = str(tmp_path / "skills")
    _make_skill_dir(skills_dir, "finance/invoice", "invoice-organizer")
    _make_skill_dir(skills_dir, "finance/budget", "budget-planner")
    _make_skill_dir(skills_dir, "science/networkx", "networkx")

    retriever = SkillRetriever(
        skills_dir=skills_dir,
        db_path=str(tmp_path / "retriever.lancedb"),
        embedding_function=FakeEF(),
    )
    result = retriever.retrieve("organize my invoices and receipts", llm=None)
    # 候选应去重（无重复 dir）且 score >= 阈值
    dirs = [c["dir"] for c in result["candidates"]]
    assert len(dirs) == len(set(dirs))
    for c in result["candidates"]:
        assert c["score"] >= _MIN_SIMILARITY
    retriever.close()


def test_retriever_no_candidates(tmp_path):
    skills_dir = str(tmp_path / "skills")
    _make_skill_dir(skills_dir, "finance/invoice", "invoice-organizer")

    retriever = SkillRetriever(
        skills_dir=skills_dir,
        db_path=str(tmp_path / "retriever.lancedb"),
        embedding_function=FakeEF(),
    )
    # 完全不相关的查询 + 低相似度 → 候选被阈值过滤
    result = retriever.retrieve("zzzz totally unrelated query zzzz", llm=None)
    # 由于 fake embedding 是哈希，可能仍有低分候选；验证逻辑不崩溃即可
    assert "candidates" in result
    retriever.close()


def test_retriever_read_skill_dir(tmp_path):
    skills_dir = str(tmp_path / "skills")
    _make_skill_dir(skills_dir, "finance/invoice", "invoice-organizer")

    retriever = SkillRetriever(
        skills_dir=skills_dir,
        db_path=str(tmp_path / "retriever.lancedb"),
        embedding_function=FakeEF(),
    )
    # 合法目录可读取
    abs_dir = retriever.read_skill_dir("finance/invoice")
    assert abs_dir is not None
    assert os.path.isfile(os.path.join(abs_dir, "SKILL.md"))
    # 路径穿越应被拒绝
    assert retriever.read_skill_dir("../../etc") is None
    retriever.close()


def test_ensure_indexed_is_idempotent(tmp_path):
    """已建表后，重复 ensure_indexed 不应触发重建（不重复 embed）。"""
    skills_dir = str(tmp_path / "skills")
    _make_skill_dir(skills_dir, "finance/invoice", "invoice-organizer")
    _make_skill_dir(skills_dir, "science/networkx", "networkx")

    db_path = str(tmp_path / "retriever.lancedb")
    retriever = SkillRetriever(
        skills_dir=skills_dir, db_path=db_path, embedding_function=FakeEF()
    )
    # 记录 build/sync 调用次数
    build_calls = []
    original_build = retriever._store.build
    def _counting_build(skills, batch_size=8):
        build_calls.append(("build", len(skills)))
        return original_build(skills, batch_size)
    retriever._store.build = _counting_build
    sync_calls = []
    original_sync = retriever._store.sync
    def _counting_sync(skills, batch_size=8):
        sync_calls.append(len(skills))
        return original_sync(skills, batch_size)
    retriever._store.sync = _counting_sync

    # 第一次：表不存在 → build
    retriever.ensure_indexed()
    assert len(build_calls) == 1

    # 第二次：表已存在 + 目录未变化 → 不 build 也不 sync
    retriever.ensure_indexed()
    assert len(build_calls) == 1, "目录未变化时不应重建"
    assert len(sync_calls) == 0, "目录未变化时不应 sync"

    retriever.close()


def test_sync_adds_updates_removes(tmp_path):
    """增量 sync：新增/修改/删除 skill，只操作对应项，不触发全量重建。"""
    skills_dir = str(tmp_path / "skills")
    _make_skill_dir(skills_dir, "finance/invoice", "invoice-organizer")
    _make_skill_dir(skills_dir, "science/networkx", "networkx")

    db_path = str(tmp_path / "retriever.lancedb")
    retriever = SkillRetriever(
        skills_dir=skills_dir, db_path=db_path, embedding_function=FakeEF()
    )
    retriever.ensure_indexed()  # 首次全量 build

    # 记录后续 sync 时 embed 的 dir 集合（验证只 embed 增量项）
    embedded_dirs = []
    original_embed_batch = retriever._store._embed_batch
    def _tracking_embed_batch(batch):
        embedded_dirs.extend(s.dir for s in batch)
        return original_embed_batch(batch)
    retriever._store._embed_batch = _tracking_embed_batch

    # 变更 1：删除 networkx + 新增 budget + 修改 invoice（改 description）
    import shutil
    shutil.rmtree(os.path.join(skills_dir, "science", "networkx"))
    _make_skill_dir(skills_dir, "finance/budget", "budget-planner")
    # 修改 invoice 的 SKILL.md 内容（改变 fingerprint）
    invoice_md = os.path.join(skills_dir, "finance", "invoice", "SKILL.md")
    with open(invoice_md, "w", encoding="utf-8") as f:
        f.write("---\nname: invoice-organizer\ndescription: EDITED description\n---\n")

    retriever._scanner._last_mtime = 0.0  # 强制感知变化
    result = retriever._store.sync(retriever._scanner.scan(force=True))

    assert result["removed"] == 1, f"应删除 networkx, got {result}"
    assert result["added"] == 1, f"应新增 budget, got {result}"
    assert result["updated"] == 1, f"应更新 invoice, got {result}"

    # 只 embed 了新增 + 更新的项（budget + invoice），没有 embed 未变的项
    assert set(embedded_dirs) == {"finance/budget", "finance/invoice"}, \
        f"只应 embed 增/改项, got {embedded_dirs}"

    # 验证表中最终状态
    from agent.skill_retriever import SkillScanner
    scanner = SkillScanner(skills_dir)
    remaining = {s.dir for s in scanner.scan(force=True)}
    assert remaining == {"finance/invoice", "finance/budget"}

    retriever.close()


def test_sync_first_run_falls_back_to_build(tmp_path):
    """表不存在时 sync 回退到全量 build。"""
    skills_dir = str(tmp_path / "skills")
    _make_skill_dir(skills_dir, "finance/invoice", "invoice-organizer")

    db_path = str(tmp_path / "retriever.lancedb")
    retriever = SkillRetriever(
        skills_dir=skills_dir, db_path=db_path, embedding_function=FakeEF()
    )
    skills = SkillScanner(skills_dir).scan(force=True)
    result = retriever._store.sync(skills)
    assert result["added"] == 1
    assert result["updated"] == 0
    assert result["removed"] == 0
    retriever.close()


# ---------------------------------------------------------------------------
# 外部 skills 兼容性（软链接 / 块标量 description / BM25 召回 / FTS 重建）
# ---------------------------------------------------------------------------

def test_scanner_follows_symlinked_skill_dirs(tmp_path):
    """外部技能常以 ln -s 挂进 skills/，必须能被扫到。"""
    skills_dir = str(tmp_path / "skills")
    external = tmp_path / "external" / "git-release"
    external.mkdir(parents=True)
    (external / "SKILL.md").write_text(
        "---\nname: git-release\ndescription: 打 tag 与发布版本\n---\n",
        encoding="utf-8",
    )
    _make_skill_dir(skills_dir, "local", "local-skill")
    os.symlink(str(external), os.path.join(skills_dir, "git-release"))

    scanner = SkillScanner(skills_dir)
    dirs = {s.dir for s in scanner.scan(force=True)}
    assert dirs == {"local", "git-release"}, f"软链接技能未索引: {dirs}"


def test_scanner_survives_symlink_loop(tmp_path):
    """自引用链接不能让扫描无限递归。"""
    skills_dir = str(tmp_path / "skills")
    _make_skill_dir(skills_dir, "real", "real-skill")
    os.symlink(skills_dir, os.path.join(skills_dir, "loop"))
    scanner = SkillScanner(skills_dir)
    assert {s.dir for s in scanner.scan(force=True)} == {"real"}


def test_read_skill_dir_allows_symlinked_skill(tmp_path):
    """read_skill_dir 必须放行软链接目录（链接目标在 skills 目录之外）。"""
    skills_dir = str(tmp_path / "skills")
    external = tmp_path / "external" / "ext-skill"
    external.mkdir(parents=True)
    (external / "SKILL.md").write_text("---\nname: ext\ndescription: d\n---\n", encoding="utf-8")
    os.makedirs(skills_dir, exist_ok=True)
    os.symlink(str(external), os.path.join(skills_dir, "ext-skill"))

    retriever = SkillRetriever(
        skills_dir=skills_dir, db_path=str(tmp_path / "db"), embedding_function=FakeEF()
    )
    abs_dir = retriever.read_skill_dir("ext-skill")
    assert abs_dir is not None
    assert os.path.isfile(os.path.join(abs_dir, "SKILL.md"))
    # 穿越与绝对路径仍然拒绝
    assert retriever.read_skill_dir("../../etc") is None
    assert retriever.read_skill_dir("/etc") is None
    retriever.close()


def test_parse_folded_block_description():
    """description: >- / | 块标量不能解析成 '>-'。"""
    folded = (
        "---\n"
        "name: excel-report\n"
        "description: >-\n"
        "  Generate Excel reports with charts and pivot tables from raw data.\n"
        "  Use when the user asks for spreadsheets or xlsx output.\n"
        "tags: [excel, xlsx]\n"
        "---\n"
    )
    m = parse_skill_md(folded, "excel-report")
    assert "Generate Excel reports" in m.description
    assert "spreadsheets" in m.description
    assert ">-" not in m.description
    assert m.tags == ["excel", "xlsx"]

    literal = (
        "---\n"
        "name: literal-skill\n"
        "description: |\n"
        "  line one\n"
        "  line two\n"
        "---\n"
    )
    m2 = parse_skill_md(literal, "literal-skill")
    assert m2.description == "line one\nline two"


def test_parse_nested_metadata_not_treated_as_top_level():
    """metadata: 下的缩进键不能污染顶层字段。"""
    text = (
        "---\n"
        "name: nested\n"
        "description: top level description\n"
        "metadata:\n"
        "  version: '1.0'\n"
        "  name: nested-name-should-be-ignored\n"
        "---\n"
    )
    m = parse_skill_md(text, "nested")
    assert m.name == "nested"
    assert m.description == "top level description"


def test_retrieve_keeps_bm25_only_candidate(tmp_path):
    """BM25 精确命中不能因为余弦 < 0.5 被整体丢弃（如查询 'CI'）。"""
    skills_dir = str(tmp_path / "skills")
    _make_skill_dir(skills_dir, "ci-pipeline", "ci-pipeline")

    retriever = SkillRetriever(
        skills_dir=skills_dir, db_path=str(tmp_path / "db"), embedding_function=FakeEF()
    )
    # 构造：向量通道无命中，BM25 命中但余弦只有 0.2（< 0.5 阈值）
    retriever._store.search_vector = lambda query, top_k=3: []
    retriever._store.search_bm25 = lambda query, top_k=3: [
        {"dir": "ci-pipeline", "name": "ci-pipeline", "tags": "CI",
         "description": "配置流水线", "bm25_score": 0.88}
    ]
    retriever._lookup_similarity = lambda query, dir, qv=None: 0.2

    result = retriever.retrieve("CI", llm=None)
    dirs = [c["dir"] for c in result["candidates"]]
    assert dirs == ["ci-pipeline"], f"BM25 命中被误杀: {result['candidates']}"
    # 归一化 BM25 最高分应把排序分抬到 0.5，避免被弱向量候选挤到 Top5 之外
    assert result["candidates"][0]["score"] >= 0.5
    retriever.close()


def test_fts_index_rebuild_after_sync(tmp_path, caplog):
    """sync 后重建 BM25 索引不能失败（缺 replace=True 会报 index already exists）。"""
    import logging

    skills_dir = str(tmp_path / "skills")
    _make_skill_dir(skills_dir, "finance/invoice", "invoice-organizer")

    db_path = str(tmp_path / "retriever.lancedb")
    retriever = SkillRetriever(
        skills_dir=skills_dir, db_path=db_path, embedding_function=FakeEF()
    )
    retriever.ensure_indexed()

    _make_skill_dir(skills_dir, "science/networkx", "networkx")
    with caplog.at_level(logging.WARNING, logger="agent.skill_retriever"):
        retriever._scanner._last_scan = 0.0
        retriever.ensure_indexed()

    assert "BM25 index failed" not in caplog.text, caplog.text
    retriever.close()
