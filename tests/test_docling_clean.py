"""clean_docling_markdown（统一引擎 Markdown 轻清洗）的单元测试。

背景：docling 路径有意跳过 clean_text（其 HTML 标签剥离会误伤 Markdown
标题/表格语法），导致 docling 输出的 ``<!-- image -->`` 图片占位注释从未
被清洗、原样进切片。这里验证轻清洗只除注释类噪声、保留 Markdown 结构。
"""
from agent.rag.cleaner import clean_docling_markdown, clean_text


def test_removes_image_placeholders():
    md = "# 标题\n\n<!-- image -->\n\n正文段落。\n\n<!-- image -->\n\n结尾。"
    out = clean_docling_markdown(md)
    assert "<!--" not in out and "-->" not in out
    assert "# 标题" in out and "正文段落。" in out


def test_removes_multiline_comment():
    md = "前文。\n<!-- 这是一段\n跨行注释 -->\n后文。"
    out = clean_docling_markdown(md)
    assert "<!--" not in out and "跨行注释" not in out
    assert "前文。" in out and "后文。" in out


def test_removes_doctype_and_pi():
    out = clean_docling_markdown("<!DOCTYPE html>\n正文\n<?php echo 1; ?>")
    assert "<!" not in out and "<?" not in out
    assert "正文" in out


def test_preserves_markdown_structure():
    md = (
        "# 标题\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n\n"
        "```html\n<div>代码块原样保留</div>\n```\n\n- 列表项\n"
    )
    out = clean_docling_markdown(md)
    assert "| A | B |" in out
    assert "```html" in out
    assert "<div>代码块原样保留</div>" in out   # 不做标签剥离
    assert "- 列表项" in out


def test_collapses_blank_lines():
    out = clean_docling_markdown("A\n\n\n\n\nB")
    assert out == "A\n\nB"


def test_plain_text_unchanged():
    md = "# 标题\n\n一段普通的正文。"
    assert clean_docling_markdown(md) == md


def test_empty_input():
    assert clean_docling_markdown("") == ""
    assert clean_docling_markdown(None) == ""


def test_clean_text_still_strips_comments():
    # clean_text（轻量解析器路径）的注释剥离行为不回归
    assert "<!--" not in clean_text("x <!-- image --> y")
