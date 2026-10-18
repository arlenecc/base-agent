"""Text cleaner — removes garbage, tags, and normalizes whitespace."""
from __future__ import annotations

import logging
import re
from typing import List

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