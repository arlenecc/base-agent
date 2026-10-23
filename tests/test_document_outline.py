"""Tests for document outline extraction + digest (Meta-context RAG)."""
from __future__ import annotations

from agent.rag.document_outline import (
    build_digest,
    build_toc,
    extract_chapters,
    extract_chapters_plain,
    find_chapters_by_title,
    has_markdown_headings,
    plain_text_to_markdown,
)


MARKDOWN = """# 第一章 引言

深度学习是机器学习的重要分支。本章介绍基本概念。

## 1.1 背景

神经网络的历史可以追溯到上世纪。感知机是早期的神经网络模型。

## 1.2 目标

本书目标是帮助读者系统掌握深度学习。

# 第二章 卷积神经网络

卷积神经网络在图像识别领域表现优异。LeNet 是最早的卷积网络之一。

## 2.1 卷积层

卷积层通过卷积核提取局部特征。常用参数包括卷积核大小和步长。
"""


def test_extract_chapters_heading_structure():
    chapters = extract_chapters(MARKDOWN)
    titles = [c.title for c in chapters]
    assert "第一章 引言" in titles
    assert "1.1 背景" in titles
    assert "第二章 卷积神经网络" in titles
    levels = {c.title: c.level for c in chapters}
    assert levels["第一章 引言"] == 1
    assert levels["1.1 背景"] == 2
    assert levels["第二章 卷积神经网络"] == 1


def test_extract_chapters_prefix():
    md = "这是前言内容。\n\n# 第一章\n\n正文。"
    chapters = extract_chapters(md)
    assert chapters[0].title == "前言"
    assert chapters[0].level == 0


def test_build_toc():
    chapters = extract_chapters(MARKDOWN)
    toc = build_toc(chapters)
    assert "第一章 引言" in toc
    assert "1.1 背景" in toc


def test_build_digest_is_toc_only():
    """digest 只含目录结构（标题层级），不再生成逐章摘要。"""
    digest, chapters = build_digest(MARKDOWN)
    assert "# 目录" in digest
    assert "第一章 引言" in digest
    assert "1.1 背景" in digest
    # 不再包含「章节摘要」段
    assert "# 章节摘要" not in digest
    assert "（无内容）" not in digest


def test_find_chapters_by_title():
    chapters = extract_chapters(MARKDOWN)
    hits = find_chapters_by_title(chapters, "卷积")
    # "卷积" 出现在 "第二章 卷积神经网络" 和 "2.1 卷积层" 两个标题中
    assert len(hits) == 2
    assert {c.title for c in hits} == {"第二章 卷积神经网络", "2.1 卷积层"}


def test_empty_markdown():
    assert extract_chapters("") == []
    digest, chapters = build_digest("")
    assert chapters == []


# ---------------------------------------------------------------------------
# 纯文本（无 Markdown 标题）目录提取
# ---------------------------------------------------------------------------

PLAIN_TOC_PDF = """人生的活法
（日）本多静六著

目 录
本多静六语录
作者简介
自序
1卷 我的财产告白
我的财产告白
征讨贫困和本多式储蓄法
2卷 我的人生活法
我的健康长寿法
如何健康长寿
3卷 我的人生计划
如何制定人生计划
人生为什么需要计划？

他，设计了明治神宫的森林和日比谷公园，被
称为“日本的公园之父”。
"""


def test_plain_toc_block_extraction():
    """纯文本「目录」块应提取出标题列表。"""
    chapters = extract_chapters_plain(PLAIN_TOC_PDF)
    titles = {c.title for c in chapters}
    assert "1卷 我的财产告白" in titles
    assert "2卷 我的人生活法" in titles
    assert "3卷 我的人生计划" in titles
    assert "自序" in titles
    # 目录块应在正文（以句号结尾的句子）处终止，不混入正文。
    assert not any("明治神宫" in t for t in titles)


def test_plain_cn_chapter_headings():
    """无「目录」标记、但正文含「第X章」标题的纯文本，应被识别。"""
    text = """内容提要
这是一段前言介绍。

第一章 人际关系的构成
假若黄金周这样度过，是不是很美妙。

第二章 研究方法
本章介绍研究亲密关系的方法。
"""
    chapters = extract_chapters_plain(text)
    titles = {c.title for c in chapters}
    assert "第一章 人际关系的构成" in titles
    assert "第二章 研究方法" in titles


def test_plain_epub_link_toc():
    """EPUB 的 [TITLE](#anchor) 目录格式应被清理为纯标题。"""
    text = """TABLE OF CONTENTS

[INTRODUCTION TO GENERATIVE AI](#aid_45)

[HOW GENERATIVE AI WORKS](#aid_33)

INTRODUCTION TO GENERATIVE AI

Generative AI is a revolutionary technique.
"""
    chapters = extract_chapters_plain(text)
    titles = {c.title for c in chapters}
    assert "INTRODUCTION TO GENERATIVE AI" in titles
    assert "HOW GENERATIVE AI WORKS" in titles
    # 不应残留 (#anchor) 后缀。
    assert not any("(#" in t for t in titles)


def test_build_digest_falls_back_to_plain():
    """build_digest 在无 Markdown 标题时应回退到纯文本提取。"""
    digest, chapters = build_digest(PLAIN_TOC_PDF)
    assert "1卷 我的财产告白" in digest
    assert "# 目录" in digest
    assert len(chapters) > 0


# ---------------------------------------------------------------------------
# PDF 页眉 / 页码噪声过滤
# ---------------------------------------------------------------------------

def test_plain_pdf_header_pageno_noise():
    """PDF 页眉「章节名 + 分隔符 + 页码」应被过滤，不当作标题。"""
    text = """第一章 引言
正文内容。

第二章 → 23
本章正文。

第三章 ／ 45
第三章正文。

第四章
第四章正文。
"""
    chapters = extract_chapters_plain(text)
    titles = {c.title for c in chapters}
    assert "第一章 引言" in titles
    assert "第四章" in titles
    # 页眉噪声应被丢弃
    assert "第二章 → 23" not in titles
    assert "第三章 ／ 45" not in titles


def test_plain_pdf_header_chapter_pageno_noise():
    """PDF 页眉「章节名 + 空格 + 页码」（无分隔符）应被过滤。"""
    text = """第五章
正文。

第六章 767
正文。

第七章
正文。
"""
    chapters = extract_chapters_plain(text)
    titles = {c.title for c in chapters}
    assert "第五章" in titles
    assert "第七章" in titles
    assert "第六章 767" not in titles


def test_plain_year_prefix_noise():
    """年份开头的正文行（如「1947）。…」「2007 年12月」）不应是标题。"""
    text = """第一章 引言
正文。

1947）。《薄伽梵歌》是其中的第二十三至第四十章。
2007 年12月
第二章 正文
"""
    chapters = extract_chapters_plain(text)
    titles = {c.title for c in chapters}
    assert "第一章 引言" in titles
    assert not any(t.startswith("1947") for t in titles)
    assert not any(t.startswith("2007") for t in titles)


def test_plain_toc_marker_not_title():
    """「目 录」目录标记本身不应出现在标题列表里。"""
    text = """《神之所欲》
翻 译 者 序
目 录
~1~ 不可置信之书
~2~ 史上最重要的问题
正文。
"""
    chapters = extract_chapters_plain(text)
    titles = {c.title for c in chapters}
    assert "~1~ 不可置信之书" in titles
    assert "~2~ 史上最重要的问题" in titles
    assert "目 录" not in titles


# ---------------------------------------------------------------------------
# 纯文本 → Markdown（启发式章节还原，2026-10-23）
# ---------------------------------------------------------------------------

def test_has_markdown_headings():
    assert has_markdown_headings("# 第一章\n\n正文。")
    assert has_markdown_headings("## 1.1 背景\n\n正文。")
    assert not has_markdown_headings("第一章 引言\n\n正文，没有标题行。")
    assert not has_markdown_headings("")


def test_plain_to_markdown_marks_chapter_headings():
    """强模式标题（第X章/编号）按层级标记，正文内容原样保留。"""
    text = (
        "前言说明文字，这是一段没有标题的介绍。\n"
        "\n"
        "第一章 人际关系的构成\n"
        "假若黄金周这样度过，是不是很美妙。\n"
        "\n"
        "1.1 研究方法\n"
        "本章介绍研究亲密关系的方法。\n"
        "\n"
        "第二章 研究结论\n"
        "结论是关系影响幸福。\n"
    )
    md = plain_text_to_markdown(text)
    assert "# 第一章 人际关系的构成" in md
    assert "## 1.1 研究方法" in md
    assert "# 第二章 研究结论" in md
    # 正文原样保留（只加标题标记，不删内容）
    assert "结论是关系影响幸福。" in md
    # 转换结果可被标准 Markdown 章节提取识别（digest 走标题路径而非启发式）
    titles = {c.title for c in extract_chapters(md)}
    assert "第一章 人际关系的构成" in titles
    assert "1.1 研究方法" in titles


def test_plain_to_markdown_toc_block_and_level_backfill():
    """目录条目不转标题；紧邻目录的正文标题不被吞；目录短标题正文重现时补标。"""
    text = (
        "书名：亲密关系\n"
        "\n"
        "目录\n"
        "第一章 引言\n"
        "自序\n"
        "第二章 方法\n"
        "\n"
        "第一章 引言\n"
        "这是第一章正文，探讨人际关系。\n"
        "\n"
        "自序\n"
        "写作缘起。\n"
        "\n"
        "第二章 方法\n"
        "研究方法正文。\n"
    )
    md = plain_text_to_markdown(text)
    lines = md.splitlines()
    toc_idx = lines.index("目录")
    # 目录块内不产生任何标题
    assert not any(l.startswith("#") for l in lines[toc_idx:toc_idx + 4])
    # 紧邻目录的正文「第一章 引言」必须被标记（不吞进目录区间）
    assert "# 第一章 引言" in md
    assert "# 第二章 方法" in md
    # 目录里列了、正文原样重现的短标题「自序」按目录级别补标
    assert "# 自序" in md


def test_plain_to_markdown_noop_without_structure():
    """没有可识别标题的散文原样返回（<2 个标记不强行改写）。"""
    text = "这是普通散文。\n\n全部是正文段落，含句号。\n\n另一段内容。"
    assert plain_text_to_markdown(text) == text


def test_plain_to_markdown_dedups_repeated_titles():
    """同名标题只标首现（后续重现多为页眉噪声）。"""
    text = (
        "第一章 引言\n"
        "第一章正文内容。\n"
        "\n"
        "第二章 方法\n"
        "第二章正文内容。\n"
        "\n"
        "第二章 方法\n"
        "页眉噪声行。\n"
    )
    md = plain_text_to_markdown(text)
    assert md.count("# 第二章 方法") == 1


def test_marked_headings_survive_clean_text():
    """转换必须发生在 clean_text 之前：# 行不被硬回车合并吞掉。"""
    from agent.rag.cleaner import clean_text
    text = (
        "第一章 引言\n"
        "引言正文第一行没有句末标点\n"
        "第二行接在上一行之后\n"
        "\n"
        "第二章 方法\n"
        "正文内容。"
    )
    md = plain_text_to_markdown(text)
    cleaned = clean_text(md)
    assert "# 第一章 引言" in cleaned
    assert "# 第二章 方法" in cleaned
