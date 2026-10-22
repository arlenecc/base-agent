"""Text cleaner — removes garbage, tags, and normalizes whitespace."""
from __future__ import annotations

import html as _html
import logging
import re
from html.parser import HTMLParser
from typing import List, Optional

logger = logging.getLogger(__name__)

# 常见 HTML 实体表。旧实现用 ``&[a-zA-Z]+;`` 一刀切，会把正文里的
# "AT&T;"、"R&D;" 这类正常写法连同前面的词一起抹掉（PDF/OCR 抽出的文本
# 尤其常见）。改为只替换真实存在的实体名。
_HTML_ENTITIES = frozenset({
    "amp", "lt", "gt", "quot", "apos", "nbsp", "copy", "reg", "trade",
    "hellip", "mdash", "ndash", "lsquo", "rsquo", "ldquo", "rdquo",
    "laquo", "raquo", "deg", "plusmn", "times", "divide", "frac12",
    "sup2", "sup3", "micro", "para", "middot", "cent", "pound", "euro",
    "yen", "sect", "bull", "dagger", "permil", "prime", "Prime", "larr",
    "rarr", "uarr", "darr", "harr", "crarr", "spades", "clubs", "hearts",
    "diams", "alpha", "beta", "gamma", "delta", "pi", "sigma", "omega",
    "infin", "ne", "le", "ge", "sum", "prod", "radic", "int", "asymp",
    "equiv", "sub", "sup", "and", "or", "not", "cap", "cup", "isin",
})
_ENTITY_RE = re.compile(r'&([a-zA-Z][a-zA-Z0-9]{1,10});')

# 「字母/数字/空白」计数用预编译正则（C 层 findall），替代旧实现的
# 逐字符 ``c.isalnum() or c.isspace()`` Python 循环——大文档（数 MB）下
# 后者是纯解释器级 O(n) 迭代，叠加全文十余遍正则替换后清洗阶段明显偏慢。
# ``\w`` ≈ isalnum()（含 unicode 字母数字，另含下划线——纯分隔行随后
# 仍会被分隔符规则收掉，行为不变），``\s`` ≈ isspace()。
_ALNUM_OR_SPACE_RE = re.compile(r'[\w\s]', re.UNICODE)


def _replace_entity(m: "re.Match") -> str:
    return " " if m.group(1) in _HTML_ENTITIES else m.group(0)


def strip_html_entities(text: str) -> str:
    """清除冗余 HTML 实体：已知实体名与数字/十六进制实体替换为空格。

    clean_text 与电子书清洗共用（pandoc 输出的 markdown 里偶尔仍残留
    &nbsp; / &mdash; / &#8212; 之类，尤其来自源文档未解码的片段）。
    """
    text = _ENTITY_RE.sub(_replace_entity, text)
    text = re.sub(r'&#\d+;', ' ', text)
    text = re.sub(r'&#x[0-9a-fA-F]+;', ' ', text)
    return text


# 句末标点：换行符紧跟其后视为真实断行（段落/句子边界），否则视为
# 排版产生的硬回车，合并回同一段（解决 epub/PDF 提取文本的断行问题）。
_SENTENCE_ENDINGS = tuple("。！？!?.…”’」』）】>；;")
# Markdown 结构行前缀：这些行不参与硬回车合并（标题/列表/引用/表格/代码栅栏）。
_STRUCTURAL_LINE_RE = re.compile(
    r'^(#{1,6}\s|[-*+]\s|\d{1,3}[.)]\s|>|\||```|~~~|\s{4,})'
)


def join_hard_wrapped_lines(text: str, preserve_markdown: bool = True) -> str:
    """合并硬回车断行：上一行不以句末标点结尾时，把下一行接到同一行（空格连接）。

    规则（epub 电子书清洗的核心，也用于 clean_text 的通用清洗）：
    - ``<br>``/``<br/>`` 先替换为空格（由调用方或本函数前半处理）
    - 上一行以 。！?.…" 等句末标点结尾 → 保留换行（真实句子/段落边界）
    - 空行 → 保留（段落分隔）
    - Markdown 结构行（标题/列表/引用/表格/代码栅栏/缩进块）不参与合并，
      避免把 ``# 标题``、``- 列表项`` 折进正文破坏 Markdown 结构
    """
    if not text:
        return ""
    # <br> 变体 → 空格（在行合并前处理，避免残留标签）
    text = re.sub(r'<br\s*/?>', ' ', text, flags=re.IGNORECASE)

    lines = text.split('\n')
    out: List[str] = []
    buf = ""
    for line in lines:
        stripped = line.strip()
        if not stripped:
            # 空行 = 段落边界：冲刷缓冲
            if buf:
                out.append(buf)
                buf = ""
            out.append("")
            continue
        if preserve_markdown and _STRUCTURAL_LINE_RE.match(line):
            if buf:
                out.append(buf)
                buf = ""
            out.append(stripped)
            continue
        if not buf:
            buf = stripped
            continue
        if buf.rstrip().endswith(_SENTENCE_ENDINGS):
            out.append(buf)
            buf = stripped
        else:
            buf = buf + " " + stripped
    if buf:
        out.append(buf)
    return '\n'.join(out)


def clean_text(text: str) -> str:
    """Clean extracted text: remove garbage tags, normalize whitespace,
    strip non-printable characters, and collapse redundant blank lines."""
    if not text:
        return ""
    logger.debug("      清洗文本: 原始 %d 字符", len(text))

    # 1. Remove entire script/style blocks (including their content)
    text = re.sub(r'<script[^>]*>[\s\S]*?</script>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'<style[^>]*>[\s\S]*?</style>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'<noscript[^>]*>[\s\S]*?</noscript>', ' ', text, flags=re.IGNORECASE)

    # 2. Remove HTML comments
    text = re.sub(r'<!--[\s\S]*?-->', ' ', text)

    # 3. Remove remaining HTML/XML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # 4. Remove HTML entities（只替换真实存在的实体名，见 _replace_entity）
    text = strip_html_entities(text)

    # 5. Remove URLs
    text = re.sub(r'https?://\S+', ' ', text)

    # 6. Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', ' ', text)

    # 7. Remove control characters except common whitespace (\n, \t, \r)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', text)

    # 8. Replace various Unicode whitespace with normal space
    text = re.sub(r'[\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]', ' ', text)

    # 9. Remove zero-width characters
    text = re.sub(r'[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]', '', text)

    # 10. Collapse multiple spaces (but preserve newlines)
    text = re.sub(r'[ \t]+', ' ', text)

    # 11. Remove lines that are just garbage (very short, no letters/digits)
    lines = text.split('\n')
    cleaned_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append('')
            continue
        # Skip lines that are mostly non-alphanumeric garbage
        # （计数走 C 层正则，见 _ALNUM_OR_SPACE_RE 注释）
        alpha_ratio = len(_ALNUM_OR_SPACE_RE.findall(stripped)) / max(len(stripped), 1)
        if alpha_ratio < 0.3 and len(stripped) < 20:
            continue
        # Skip lines that are just page numbers or headers.
        # 只删**独立成行**的短数字（典型页码），且不再顺带删掉带字母数字混排的
        # 短行 —— 旧条件 ``alpha_ratio < 0.3 and len < 20`` 会把 "AT&T; 财报"
        # 这类行也判成垃圾。
        if re.match(r'^\d{1,4}$', stripped):
            continue
        if re.match(r'^第\s*\d{1,4}\s*页$', stripped):
            continue
        if re.match(r'^Page\s+\d{1,4}$', stripped, re.IGNORECASE):
            continue
        # Skip separator lines (long runs of the same character)
        if re.match(r'^([\-=_*#~]{3,})\s*$', stripped):
            continue
        cleaned_lines.append(stripped)

    text = '\n'.join(cleaned_lines)

    # 11.5 合并硬回车断行：上一行不以句末标点结尾时，把下一行接到同一段
    # （解决 PDF/epub 提取文本逐行断裂的问题；Markdown 结构行不合并）。
    text = join_hard_wrapped_lines(text, preserve_markdown=True)

    # 12. Collapse 3+ consecutive newlines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 13. Strip leading/trailing whitespace
    text = text.strip()

    logger.debug("      清洗文本完成: %d 字符", len(text))
    return text


# ---------------------------------------------------------------------------
# HTML 表格 → Markdown 管道表（ovis 与统一引擎轻清洗共用）
# ---------------------------------------------------------------------------

class _TableToMarkdown(HTMLParser):
    """把 <table> 块解析成行列网格（handle_data 聚合单元格文本）。"""

    def __init__(self) -> None:
        super().__init__()
        self.rows: List[List[str]] = []
        self._row: Optional[List[str]] = None
        self._cell: Optional[List[str]] = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None
        elif tag in ("td", "th") and self._cell is not None:
            if self._row is None:
                self._row = []
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def html_tables_to_markdown(text: str) -> str:
    """把文本里的 <table> 块转成 Markdown 管道表格，其余文本原样保留。

    docling/marker 的复杂表格、ovis 的 4-bit 模型输出都会给 HTML 表格。
    直接当标签剥掉会让单元格文本粘连成一行（结构信息全丢）——转成管道
    表既保结构又对 embedding 友好。
    """
    if "<table" not in text.lower():
        return text

    def _convert(m: "re.Match") -> str:
        block = m.group(0)
        p = _TableToMarkdown()
        try:
            p.feed(block)
        except Exception:
            return block
        rows = [r for r in p.rows if any(c.strip() for c in r)]
        if not rows:
            return ""
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        out = ["| " + " | ".join(rows[0]) + " |",
               "| " + " | ".join(["---"] * width) + " |"]
        for r in rows[1:]:
            out.append("| " + " | ".join(r) + " |")
        return "\n".join(out)

    return re.sub(r"<table.*?</table>", _convert, text,
                  flags=re.IGNORECASE | re.DOTALL)


# ---------------------------------------------------------------------------
# 统一引擎 Markdown 轻清洗
# ---------------------------------------------------------------------------

# 代码栅栏整块（``` 或 ~~~ 开头，允许未闭合到文末）。捕获组 1 是整块，
# re.split 的奇数下标即栅栏段——这些内容一个字符都不动。
_FENCE_BLOCK_RE = re.compile(r'((```|~~~)[^\n]*\n[\s\S]*?(?:\2|\Z))')
# 行内代码（1-3 个反引号包裹，单行）。清洗前 stash、清洗后还原，
# 避免 `` `<div>` `` 里的示例标签被剥标签规则吃掉。
_INLINE_CODE_RE = re.compile(r'(`{1,3})[^`\n]+?\1')
# 整块丢弃：<script>/<style>/<svg>（内容对检索纯噪声）
_DROP_BLOCK_RE = re.compile(
    r'<(script|style|svg)\b[^>]*>[\s\S]*?</\1\s*>', re.IGNORECASE)
_COMMENT_RE = re.compile(r'<!--[\s\S]*?-->')
_TABLE_BLOCK_RE = re.compile(r'<table[\s\S]*?</table>', re.IGNORECASE)
_BR_RE = re.compile(r'<br\s*/?>', re.IGNORECASE)
_TAG_RE = re.compile(r'</?[a-zA-Z][^>]*>')          # 属性可跨行
# 块级标签剥除后换行（HTML 渲染语义：div/p/li/h1-6 等本身隐含断行）；
# 其余（u/sup/span/a 等）剥除后不插任何字符——``普通<u>下划线</u>与``
# 渲染出来就是「普通下划线与」，插空格会把 CJK 文本撕开。
_BLOCK_TAG_RE = re.compile(
    r'</?(?:div|p|center|blockquote|section|article|header|footer|figure|'
    r'figcaption|main|aside|nav|form|fieldset|table|thead|tbody|tfoot|tr|td|th|'
    r'ul|ol|li|dl|dd|dt|h[1-6]|pre|hr)\b[^>]*>', re.IGNORECASE)
_DECL_RE = re.compile(r'<[!?][^>]*>')               # DOCTYPE / 处理指令


def _clean_md_segment(seg: str) -> str:
    """清洗单个栅栏外片段：除注释/占位块、表格转 Markdown、剥残标、还原实体。"""
    seg = _DROP_BLOCK_RE.sub(' ', seg)
    seg = _COMMENT_RE.sub(' ', seg)
    seg = _DECL_RE.sub(' ', seg)
    seg = _TABLE_BLOCK_RE.sub(
        lambda m: '\n\n' + html_tables_to_markdown(m.group(0)) + '\n\n', seg)
    seg = _BR_RE.sub('\n', seg)
    # 行内代码 stash → 剥标签 → 实体还原 → 还原行内代码
    stash: List[str] = []

    def _stash(m: "re.Match") -> str:
        stash.append(m.group(0))
        return chr(0) + str(len(stash) - 1) + chr(0)

    seg = _INLINE_CODE_RE.sub(_stash, seg)
    seg = _BLOCK_TAG_RE.sub('\n', seg)
    seg = _TAG_RE.sub('', seg)
    seg = _DECL_RE.sub(' ', seg)
    seg = _html.unescape(seg)
    seg = seg.replace('\xa0', ' ')
    for i, code in enumerate(stash):
        seg = seg.replace(chr(0) + str(i) + chr(0), code)
    return seg


def clean_docling_markdown(text: str) -> str:
    """统一引擎（docling/marker/ovis）Markdown 的**轻清洗**。

    这些解析器直接输出规范 Markdown，不能走 clean_text——它的 HTML 标签
    剥离会误伤 Markdown 标题/表格语法。这里除统一引擎的常见残留（全部
    **栅栏外**生效，代码块内容一个字符不动）：

    - HTML 注释：docling 给每张图片输出 ``<!-- image -->`` 占位等
      （此前切片混进 `<!-- image -->` 的根因——docling 路径有意跳过
      clean_text，注释从没被洗过）；
    - `<script>/<style>/<svg>` 整块丢弃；
    - HTML 表格 → Markdown 管道表（docling/marker 复杂表、ovis 输出）；
    - `<br>`/块级标签 → 换行，行内残标剥除、留内容；
    - DOCTYPE / 处理指令；`&amp;` 等实体还原为真实字符；
    - 折叠 3+ 连续空行。
    """
    if not text:
        return ""
    # 用 finditer 游标拼接（不能 re.split——split 会把**所有**捕获组插回
    # 结果，栅栏闭合符会被重复一份）。
    out: List[str] = []
    pos = 0
    for m in _FENCE_BLOCK_RE.finditer(text):
        out.append(_clean_md_segment(text[pos:m.start()]))
        out.append(m.group(0))          # 栅栏整块原样保留
        pos = m.end()
    out.append(_clean_md_segment(text[pos:]))
    cleaned = re.sub(r'\n{3,}', '\n\n', "".join(out))
    return cleaned.strip()


def normalize_markdown(text: str) -> str:
    """Convert cleaned text to clean markdown format.

    代码块内的空行保留原样，不会被当作段落分隔符。
    非代码块段落之间用一个空行分隔。
    """
    if not text:
        return ""
    logger.debug("      转换 Markdown: %d 字符", len(text))

    lines = text.split('\n')
    result: List[str] = []
    in_code_block = False
    prev_blank = False  # 上一行是否是空行（用于合并连续空行）

    for line in lines:
        stripped = line.strip()

        # Detect code blocks
        if stripped.startswith('```'):
            in_code_block = not in_code_block
            result.append(stripped)
            prev_blank = False
            continue

        if in_code_block:
            # 代码块内：保留原始行（包括空行），不做任何修改
            result.append(line)
            prev_blank = False
            continue

        if not stripped:
            # 非代码块的空行：确保段落之间只有一个空行（合并连续空行）
            if not prev_blank and result:
                result.append('')
            prev_blank = True
            continue

        # 非代码块的非空行：strip 后追加
        result.append(stripped)
        prev_blank = False

    # 去除末尾空行
    while result and not result[-1].strip():
        result.pop()

    output = '\n'.join(result)
    logger.debug("      转换 Markdown 完成: %d 行", len(result))
    return output