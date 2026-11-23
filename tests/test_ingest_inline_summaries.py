"""切片入库内联生成章节/全书摘要（核心句方案）测试。

覆盖：入库后 rag_outline 立即可用、指纹去重幂等、失败不阻断入库。
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from agent.rag.chapter_summarizer import (
    build_book_summaries_inline,
    load_summaries_state,
)


def _markdown():
    return (
        "# 引言\n\n人工智能正在深刻地改变各行各业的格局。\n\n"
        "## 第一章 基础\n\n本章介绍机器学习的基本概念、算法原理与数学基础。\n\n"
        "## 第二章 应用\n\n本章讲解工程实践中的典型应用案例与部署经验。"
    )


def test_inline_summaries_generated_and_idempotent(tmp_path):
    md = _markdown()
    md_path = str(tmp_path / "book.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    assert build_book_summaries_inline(str(tmp_path), "demo", md, md_path=md_path)

    state = load_summaries_state(str(tmp_path))
    book = state["books"]["demo"]
    assert len(book["chapters"]) == 3
    assert all(c["summary"] for c in book["chapters"])
    assert book["book_summary"]
    assert book["md_fingerprint"] and book["content_hash"]

    # 相同内容重复调用应跳过（幂等，不重复抽取）
    assert not build_book_summaries_inline(
        str(tmp_path), "demo", md, md_path=md_path)


def test_content_change_regenerates(tmp_path):
    md_path = str(tmp_path / "book.md")
    md1 = _markdown()
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md1)
    build_book_summaries_inline(str(tmp_path), "demo", md1, md_path=md_path)

    md2 = md1 + "\n\n## 第三章 新章\n\n新增章节内容。"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md2)
    assert build_book_summaries_inline(str(tmp_path), "demo", md2, md_path=md_path)
    book = load_summaries_state(str(tmp_path))["books"]["demo"]
    assert len(book["chapters"]) == 4


def test_empty_markdown_skipped(tmp_path):
    assert not build_book_summaries_inline(str(tmp_path), "demo", "")
    assert not build_book_summaries_inline(str(tmp_path), "demo", "   ")


def test_scheduler_skips_empty_chapter_and_advances(tmp_path):
    """封面/献词等空章节不能被当成失败重试，必须跳过并继续后续章节。"""
    from agent.config import AgentConfig
    from agent.rag.chapter_summarizer import ChapterSummaryScheduler
    from agent.rag.document_outline import extract_chapters

    md = (
        "## 封面\n\n"
        "## 第一章 开端\n\n"
        "伦敦雾的起源可以追溯到工业革命时期，煤炭燃烧产生了大量烟尘，"
        "城市空气因此逐渐恶化，并引发社会对污染的持续关注。\n"
    )
    md_path = tmp_path / "book.md"
    md_path.write_text(md, encoding="utf-8")
    chapters = extract_chapters(md)
    info = {"md_path": str(md_path), "chapters": [
        {"level": c.level, "title": c.title} for c in chapters]}

    cfg = AgentConfig()
    cfg.workspace = str(tmp_path)
    cfg.rag_summary_use_core_sentences = True
    sched = ChapterSummaryScheduler(
        lambda: cfg, lambda: None, lambda: None, lambda: True)
    books = {}

    interval = sched._process_book(
        None, cfg, "demo", str(md_path), info, books)
    book = books["demo"]

    assert interval is not None
    assert book["chapters"][0]["idx"] == 0
    assert book["chapters"][0]["summary"] == ""
    assert book["chapters"][0]["skipped"] == "empty"
    assert any(c["idx"] == 1 and c["summary"] for c in book["chapters"])

    # 再次处理：不会重复追加第 0 章，也不会卡在空章节
    assert sched._process_book(
        None, cfg, "demo", str(md_path), info, books) is None
    assert [c["idx"] for c in book["chapters"]].count(0) == 1
