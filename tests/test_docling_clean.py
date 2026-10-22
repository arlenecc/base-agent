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


# ---------------------------------------------------------------------------
# 扩展清理项：script/style/svg、HTML 表格、<br>、残标、实体、栅栏保护
# ---------------------------------------------------------------------------

def test_drops_script_style_svg_blocks():
    md = "正文A\n<script>alert(1)</script>\n<style>p{color:red}</style>\n" \
         "<svg><circle r=\"1\"/></svg>\n正文B"
    out = clean_docling_markdown(md)
    assert "alert" not in out and "color:red" not in out and "circle" not in out
    assert "正文A" in out and "正文B" in out


def test_converts_html_table_to_markdown():
    md = "前文。\n<table><tr><th>名称</th><th>值</th></tr>" \
         "<tr><td>甲</td><td>1</td></tr></table>\n后文。"
    out = clean_docling_markdown(md)
    assert "<table" not in out
    assert "| 名称 | 值 |" in out
    assert "| 甲 | 1 |" in out
    assert "前文。" in out and "后文。" in out


def test_br_becomes_newline():
    out = clean_docling_markdown("第一行<br>第二行<br/>第三行")
    assert "第一行\n第二行\n第三行" in out
    assert "<br" not in out


def test_strips_tags_keeps_text():
    md = "普通<u>下划线</u>与<sup>上标</sup>，<div>块内文字</div>结束。"
    out = clean_docling_markdown(md)
    assert "<" not in out
    # 行内标签剥除不插字符（HTML 渲染语义）；块级 <div> 隐含换行
    assert "普通下划线与上标，" in out
    assert "块内文字" in out and "结束。" in out


def test_unescapes_entities():
    out = clean_docling_markdown("AT&amp;T 与 A&nbsp;B")
    assert "AT&T" in out
    assert "&nbsp;" not in out and "&amp;" not in out


def test_inline_code_tags_survive():
    md = "写法：`<div class=\"x\">` 是标签。"
    out = clean_docling_markdown(md)
    assert "`<div class=\"x\">`" in out


def test_code_fence_fully_protected():
    md = (
        "说明：\n\n```html\n<!-- 注释也算内容 -->\n<table><tr><td>示例</td>"
        "</tr></table>\n<script>keep()</script>\n```\n\n<!-- image -->\n尾注。"
    )
    out = clean_docling_markdown(md)
    # 栅栏内一字不动（含注释/标签/script）
    assert "```html\n<!-- 注释也算内容 -->" in out
    assert "keep()" in out
    assert "<td>示例</td>" in out
    # 栅栏外的占位注释被清掉
    assert out.count("<!-- image -->") == 0
    assert "尾注。" in out


def test_unclosed_fence_protected_to_eof():
    md = "前文 <!-- image -->\n\n```\n<!-- 注释 --> <u>未闭合栅栏</u>"
    out = clean_docling_markdown(md)
    assert "<!-- 注释 -->" in out and "<u>未闭合栅栏</u>" in out
    assert "<!-- image -->" not in out


def test_html_table_conversion_helper():
    from agent.rag.cleaner import html_tables_to_markdown
    out = html_tables_to_markdown("<table><tr><td>A</td><td>B</td></tr></table>")
    assert out == "| A | B |\n| --- | --- |"
    # 无表格时原样返回
    assert html_tables_to_markdown("纯文本") == "纯文本"


def test_tilde_fence_protected():
    md = "~~~\n<!-- 保真 -->\n~~~\n<!-- image -->"
    out = clean_docling_markdown(md)
    assert "<!-- 保真 -->" in out
    assert "<!-- image -->" not in out


# ---------------------------------------------------------------------------
# 复杂表保留（管道表无法表达的结构不强摊平，2026-10-22 用户定版）
# ---------------------------------------------------------------------------

def test_complex_table_colspan_kept_as_html():
    md = "前文。\n<table><tr><td colspan=\"2\">合并</td></tr>" \
         "<tr><td>A</td><td>B</td></tr></table>\n后文。"
    out = clean_docling_markdown(md)
    assert "<table>" in out and 'colspan="2"' in out and "合并" in out
    assert "| A | B |" not in out          # 整块保留，不摊平
    assert "前文。" in out and "后文。" in out


def test_complex_table_rowspan_kept_as_html():
    md = "<table><tr><td rowspan=\"2\">合并</td><td>A</td></tr>" \
         "<tr><td>B</td></tr></table>"
    out = clean_docling_markdown(md)
    assert "<table>" in out and 'rowspan="2"' in out


def test_complex_table_caption_kept_as_html():
    md = "<table><caption>表标题</caption><tr><td>A</td></tr></table>"
    out = clean_docling_markdown(md)
    assert "<table>" in out and "<caption>表标题</caption>" in out


def test_nested_table_outer_block_kept():
    md = ("<table><tr><td>外\n"
          "<table><tr><td>内</td></tr></table>\n</td></tr></table>")
    out = clean_docling_markdown(md)
    # 平衡扫描：外层块完整保留（含内层表），不会被懒匹配腰斩
    assert out.count("<table>") == 2
    assert "内" in out


def test_simple_table_still_converted():
    md = "<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td>" \
         "</tr></table>"
    out = clean_docling_markdown(md)
    assert "<table" not in out
    assert "| A | B |" in out and "| 1 | 2 |" in out


def test_html_tables_to_markdown_keeps_complex():
    from agent.rag.cleaner import html_tables_to_markdown
    block = '<table><tr><td colspan="3">合并</td></tr></table>'
    assert html_tables_to_markdown(block) == block
    # 嵌套：外层块完整（懒匹配会腰斩，这里验证平衡扫描）
    nested = "<table><tr><td>x<table><tr><td>y</td></tr></table></td></tr></table>"
    assert html_tables_to_markdown(nested) == nested


def test_table_survives_inside_html_kept_block():
    # 保留的复杂表块内的标签不再被行内/块级剥标签规则二次销毁
    md = ("<table><thead><tr><th>列</th></tr></thead><tbody>"
          "<tr><td colspan=\"2\">值</td></tr></tbody></table>")
    out = clean_docling_markdown(md)
    assert "<thead>" in out and "<tbody>" in out and "<th>列</th>" in out


def test_ebook_path_table_handling():
    from agent.rag.ebook_parser import clean_ebook_markdown
    simple = "前言\n<table><tr><td>A</td><td>B</td></tr></table>\n正文"
    out = clean_ebook_markdown(simple)
    assert "<table" not in out and "A" in out and "B" in out
    complex_md = ('前言\n<table><tr><td colspan="2">合并</td></tr></table>\n正文')
    out = clean_ebook_markdown(complex_md)
    # 复杂表原样保留（不被剥标签销毁）
    assert "<table>" in out and 'colspan="2"' in out and "合并" in out
