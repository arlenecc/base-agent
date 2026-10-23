"""Document outline extraction — build a TOC (目录结构) digest.

Given parsed document text (docling Markdown, EPUB, or plain-text PDF), this
module produces a condensed "缩略版本" (outline digest): the document's table
of contents — heading titles with their hierarchy — as meta-context.

Extraction strategy (best-effort, in order):
  1. Standard Markdown headings (``#`` … ``######``).
  2. Plain-text heuristics for documents without Markdown headings:
     a. A "目录/Contents" block: a marker line (``目录``/``目 录``/``Contents``/
        ``Table of Contents``) followed by a run of one-title-per-line entries.
     b. Standalone chapter-title lines in the body:
        * Chinese "第N章/节/卷/篇/部" prefixes (Arabic or CJK numerals);
        * numbered headings ("1", "1.1", "1.1.1", "第一章"…);
        * ALL-CAPS short lines (common in EPUB headings).

The digest is the *meta-context* for Meta-context + Targeted RAG: the model
first reads the whole document structure, then plans which chapters to drill
into, and RAG retrieves only those passages.

We deliberately do NOT generate per-chapter summaries — the title hierarchy
alone is sufficient and avoids the cost/slowness/emptiness of LLM summaries.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ``\s+`` 会匹配换行符：整篇文本上 ``#\n标题`` 这样的「空 ATX 标题行」
# 会把下一行正文吞并成标题（group(2) 跨行命中），污染章节提取。
# 限定为行内空白 [ \t]。
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.*)$", re.MULTILINE)

# 中文章节标题：第{数字|中文数字}章/节/卷/篇/部/部分，后跟空格或标题。
# 也兼容无「第」字的数字+卷/章格式（如「1卷 我的财产告白」）。
_CJK_NUM = "零一二三四五六七八九十百千〇"
_CN_CHAPTER_RE = re.compile(
    r"^(?:第\s*)?([0-9]+|[%s]+)\s*[章卷篇部](?:[：:\s]+.+)?$" % _CJK_NUM
)
_CN_SECTION_RE = re.compile(
    r"^(?:第\s*)?([0-9]+|[%s]+)\s*节(?:[：:\s]+.+)?$" % _CJK_NUM
)
# 编号标题：1 / 1.1 / 1.1.1 / (1) / 1、 等，后跟标题文本。
_NUMBERED_RE = re.compile(r"^(\d+(?:\.\d+){0,3})[\.、\s）)](.+)?$")
# 英文全大写标题行：≥3 个词的全大写字母，允许部分符号。
_ALLCAPS_RE = re.compile(r"^[A-Z][A-Z0-9 .,&'()/:-]{3,}$")

# EPUB 目录链接格式：[TITLE](#anchor) → TITLE。
_EPUB_LINK_RE = re.compile(r"^\[(.+)\]\(#[^)]*\)$")

# HTML 注释 / 图片占位噪声。
_HTML_NOISE_RE = re.compile(r"^\s*<!--.*?-->\s*$")

# 「目录」标记行（用于识别目录列表段）。
_TOC_MARKERS = (
    "目录", "目 录", "目錄", "contents", "table of contents",
)

# 明显非标题的噪声行（作者、版权、出版社等）。
_NOISE_TITLES = {
    "copyright", "all rights reserved", "all right reserved", "isbn",
    "references", "references and citations", "img",
}

# 「目录」标记本身（含带空格的变体），无论在哪个阶段都应跳过。
_TOC_TITLE_NOISE = {"目录", "目 录", "目錄", "目 錄"}

# PDF 页眉模式：「章节名 + 分隔符 + 纯数字页码」，如「第二章 → 23」、
# 「第十章 ／ 147」、「第六章 ～65」。这是书籍左/右页页眉（章节名+页码），
# 不是真正的标题，应整体丢弃。
_PDF_HEADER_PAGENO_RE = re.compile(
    r"^(.{2,40}?)\s*[→／/～~|—\-]\s*\d{1,4}$"
)

# 页眉变体 2：「章节名 + 空格 + 纯数字页码」（无分隔符），如「第二章 29」、
# 「第五章 753」「第九章 85」。书名/章节名后紧跟页码。
_PDF_HEADER_CHAPTER_PAGENO_RE = re.compile(
    r"^(第[%s0-9一二三四五六七八九十百]+[章卷篇部节]|[A-Za-z][A-Za-z0-9 .,&'()/:-]{2,40}?)\s+\d{1,4}$" % _CJK_NUM
)

# 页眉变体 3：「页码 + 分隔符 + 章节名」（页码在前），如「128 ／ 薄伽梵歌」。
_PDF_HEADER_PAGENO_CHAPTER_RE = re.compile(
    r"^\d{1,4}\s*[→／/～~|—\-]\s*\S.{0,38}$"
)

# 被截断的章节名（缺「第」字，如「三章」「一章」「二章」）——通常是页眉
# 「第三章」的「第」字与数字被拆到上一行所致，单独成行时是噪声。
_TRUNCATED_CN_CHAPTER_RE = re.compile(
    r"^[%s]{1,3}[章卷篇部]$" % _CJK_NUM
)

# 年份开头的正文行：如「1947）。《薄伽梵歌》是…」「2007 年12月」。
# 标题不会以 4 位年份开头（除非恰是年份书名，罕见），视为正文噪声。
_YEAR_PREFIX_RE = re.compile(r"^(?:19|20)\d{2}\s*[年）).、]")


class Chapter:
    """A single heading + its body text."""

    __slots__ = ("level", "title", "text", "summary")

    def __init__(self, level: int, title: str, text: str, summary: str = ""):
        self.level = level
        self.title = title
        self.text = text
        self.summary = summary


# ---------------------------------------------------------------------------
# Markdown headings
# ---------------------------------------------------------------------------

def _find_headings(markdown: str) -> List[Tuple[int, str, int, int]]:
    """逐行扫描找出 Markdown 标题，**跳过围栏代码块内的 ``#``**。

    返回 ``[(level, title, start, end)]``（start/end 为原文绝对偏移）。
    围栏代码块里的 shell/Python 注释（``# comment``、``#!/bin/bash``）会被
    误认为标题，把代码样本切成假章节——hybrid_chunker 已做了同样的围栏跳过，
    这里补齐（否则目录 digest 会被代码注释污染）。
    """
    results: List[Tuple[int, str, int, int]] = []
    in_fence = False
    fence_marker = ""
    offset = 0
    for line in markdown.splitlines(keepends=True):
        stripped = line.strip()
        if not in_fence:
            if stripped.startswith("```") or stripped.startswith("~~~"):
                in_fence = True
                fence_marker = stripped[:3]
                offset += len(line)
                continue
        else:
            # 只有同一种围栏才能闭合（``` 不能被 ~~~ 闭合，反之亦然）。
            if stripped.startswith(fence_marker):
                in_fence = False
                fence_marker = ""
            offset += len(line)
            continue
        m = _HEADING_RE.match(line)
        if m:
            results.append((
                len(m.group(1)), m.group(2).strip(),
                offset + m.start(), offset + m.end(),
            ))
        offset += len(line)
    return results


def extract_chapters(markdown: str) -> List[Chapter]:
    """Split Markdown into chapters at ``#`` heading boundaries."""
    if not markdown or not markdown.strip():
        return []

    headings = _find_headings(markdown)
    chapters: List[Chapter] = []

    if headings:
        prefix = markdown[: headings[0][2]].strip()
        if prefix:
            chapters.append(Chapter(0, "前言", prefix))

    for i, (level, title, _start, end) in enumerate(headings):
        next_start = headings[i + 1][2] if i + 1 < len(headings) else len(markdown)
        body = markdown[end:next_start].strip()
        if body or title:
            chapters.append(Chapter(level, title, body))

    return chapters


# ---------------------------------------------------------------------------
# Plain-text heuristics
# ---------------------------------------------------------------------------

def _split_lines(text: str) -> List[str]:
    """Split into lines with CRLF/CR normalised (blank lines preserved)."""
    return [ln.strip() for ln in re.split(r"\r\n|\r|\n", text)]


def _looks_like_title(line: str) -> bool:
    """True if a line looks like a standalone heading (short, no sentence end)."""
    if _HTML_NOISE_RE.match(line):
        return False
    n = len(line)
    if n < 2 or n > 80:
        return False
    # 不以句末标点结尾（标题一般无句号/逗号/分号）。
    if line[-1] in "。，；：！？.,;:!?":
        return False
    # 明显噪声。
    if line.lower() in _NOISE_TITLES:
        return False
    # 「目录」标记本身（含带空格变体）不是标题。
    if line in _TOC_TITLE_NOISE:
        return False
    # 年份开头的正文行（如「1947）。《薄伽梵歌》…」「2007 年12月」）。
    if _YEAR_PREFIX_RE.match(line):
        return False
    # PDF 页眉（「第二章 → 23」「第十章 ／ 147」）不是标题。
    if _PDF_HEADER_PAGENO_RE.match(line):
        return False
    # 页眉变体：「章节名 + 页码」（「第二章 29」）和「页码 + 章节名」（「128 ／ 薄伽梵歌」）。
    if _PDF_HEADER_CHAPTER_PAGENO_RE.match(line):
        return False
    if _PDF_HEADER_PAGENO_CHAPTER_RE.match(line):
        return False
    # 被截断的章节名（「三章」「一章」）不是标题。
    if _TRUNCATED_CN_CHAPTER_RE.match(line):
        return False
    return True


def _normalize_title(line: str) -> str:
    """Normalise a raw title line into a clean title.

    Strips EPUB link wrappers (``[TITLE](#anchor)`` → ``TITLE``) and other
    obvious wrappers, then collapses internal whitespace.
    """
    line = line.strip()
    m = _EPUB_LINK_RE.match(line)
    if m:
        line = m.group(1).strip()
    line = re.sub(r"\s+", " ", line)
    return line


def _cn_section_match(line: str) -> Optional[Tuple[int, str]]:
    """Match a Chinese section heading (``第X节``) → (level=2, title)."""
    m = _CN_SECTION_RE.match(line)
    if m:
        return (2, line)
    return None


def _cn_chapter_match(line: str) -> Optional[Tuple[int, str]]:
    """Match a Chinese chapter/volume/part heading → (level=1, title).

    ``第X章``/``X卷``/``第X篇``/``第X部``/``第X部分`` 等均为 level 1。
    """
    m = _CN_CHAPTER_RE.match(line)
    if m:
        return (1, line)
    return None


def _numbered_match(line: str) -> Optional[Tuple[int, str]]:
    """Match a numbered heading (``1.2.3 Title``) → (level, title)."""
    m = _NUMBERED_RE.match(line)
    if not m:
        return None
    num = m.group(1)
    level = num.count(".") + 1
    rest = (m.group(2) or "").strip()
    # 纯数字行（如页码）不算标题。
    if not rest:
        return None
    return (level, line)


def _extract_from_toc_block(lines: List[str]) -> List[Chapter]:
    """Extract titles from a "目录/Contents" block (one title per line).

    Locates the first TOC marker, then collects subsequent short lines as
    entries until a paragraph-length line (body text) is hit.  Level is inferred
    from the title's shape: chapter/volume → 1, section/numbered → 2, plain
    short titles → 1.
    """
    for i, ln in enumerate(lines):
        if ln.lower() in _TOC_MARKERS:
            break
    else:
        return []

    chapters: List[Chapter] = []
    seen = set()
    for ln in lines[i + 1:]:
        # 终止条件：正文段落通常以句末标点结尾（目录标题一般不会）。
        # 硬换行的 PDF 正文每行都短，无法靠行长度区分目录与正文，
        # 但正文句以「。，；！？」结尾，目录标题则不会。
        if len(ln) > 80 or (ln and ln[-1] in "。，；！？．,;!?"):
            break
        if not _looks_like_title(ln):
            continue
        title = _normalize_title(ln)
        hit = _cn_section_match(title) or _cn_chapter_match(title) or _numbered_match(title)
        if hit is not None:
            level, _ = hit
        elif _ALLCAPS_RE.match(title):
            level = 1
        else:
            # 目录里的普通短标题（如「自序」「我的财产告白」）→ level 1。
            level = 1
        if title in seen:
            continue
        seen.add(title)
        chapters.append(Chapter(level, title, ""))
    return chapters


def _extract_from_body_headings(lines: List[str]) -> List[Chapter]:
    """Scan the whole body for standalone chapter-title lines.

    Strong patterns (``第X章``/``第X节``/numbered) are accepted outright;
    weak patterns (ALL-CAPS) additionally require surrounding blank lines to
    avoid matching ordinary prose.
    """
    chapters: List[Chapter] = []
    seen = set()
    for idx, ln in enumerate(lines):
        if not ln or not _looks_like_title(ln):
            continue
        title = _normalize_title(ln)
        hit = _cn_section_match(title) or _cn_chapter_match(title) or _numbered_match(title)
        if hit is not None:
            level, _ = hit
        elif _ALLCAPS_RE.match(title):
            # 弱模式：要求前后有空行，避免把正文里的大写短语当标题。
            prev_blank = idx == 0 or lines[idx - 1] == ""
            nxt_blank = idx + 1 >= len(lines) or lines[idx + 1] == ""
            if not (prev_blank and nxt_blank):
                continue
            level = 1
        else:
            continue
        if title in seen:
            continue
        seen.add(title)
        chapters.append(Chapter(level, title, ""))
    return chapters


def extract_chapters_plain(text: str) -> List[Chapter]:
    """Extract a chapter list from plain text (no Markdown headings).

    Prefers a TOC block when present (more accurate and complete); otherwise
    falls back to scanning the body for standalone title lines.
    """
    lines = _split_lines(text)
    if not lines:
        return []

    chapters = _extract_from_toc_block(lines)
    if chapters:
        return chapters
    return _extract_from_body_headings(lines)


# ---------------------------------------------------------------------------
# 纯文本 → Markdown（启发式章节还原）
# ---------------------------------------------------------------------------

def has_markdown_headings(text: str) -> bool:
    """文本是否已含 Markdown 标题行。

    围栏代码块内的 ``#`` 注释会误报为 True → 转换被保守跳过（宁可不转换，
    也不给纯文本强加错误结构），方向安全。
    """
    return bool(_HEADING_RE.search(text or ""))


def _toc_block_span(lines: List[str]) -> Optional[Tuple[int, int]]:
    """定位「目录」块的行号区间 ``[start, end)``（含标记行；无则 None）。

    终止条件（除 :func:`_extract_from_toc_block` 的「变长/句末标点」外）：
    **目录里已见过的标题在区间内再次出现** —— 目录条目会在正文重现，首次
    重现即正文开始（否则紧邻目录的第一章标题会被目录区间吞掉、漏标）。
    """
    for i, ln in enumerate(lines):
        if ln.lower() in _TOC_MARKERS:
            break
    else:
        return None
    end = i + 1
    seen_titles: set = set()
    while end < len(lines):
        ln = lines[end]
        if len(ln) > 80 or (ln and ln[-1] in "。，；！？．,;!?"):
            break
        if ln and _looks_like_title(ln):
            title = _normalize_title(ln)
            if title in seen_titles:
                break           # 正文重现目录条目 → 目录块结束
            seen_titles.add(title)
        end += 1
    return (i, end)


def plain_text_to_markdown(text: str) -> str:
    """把无结构纯文本转成 Markdown：按启发式还原的章节结构给标题行加 ``#``。

    处理对象：txt 文件、OCR/解析回退产出的非结构化纯文本——它们没有标题
    层级，下游（目录 digest、分块、按章检索）全部失去章节边界；且后续
    ``clean_text`` 的硬回车合并会把「第三章 …」这类独立标题行吞进段落，
    所以转换必须发生在合并**之前**（``#`` 行是结构行，不会被合并）。

    规则：
      * 正文强模式标题（``第X章/节``、编号 ``1.2.3``）直接标记；
      * 全大写弱模式仍要求前后空行；
      * 目录块只用来提供「标题 → 级别」的权威映射与跳过区：目录里的短标题
        （自序/后记等启发式覆盖不到的）在正文原样重现时按目录级别补标；
        目录条目本身不转标题（否则封面目录页会炸成一堆假章节）；
      * 同名标题只标首现（后续重现多为页眉噪声）；
      * 标记数 < 2 时视为误判，整体不改写（保持原样最安全）。
    """
    if not text or not text.strip():
        return text
    lines = [ln.strip() for ln in re.split(r"\r\n|\r|\n", text)]

    toc_chapters = _extract_from_toc_block(lines)
    toc_level = {c.title: c.level for c in toc_chapters}
    toc_span = _toc_block_span(lines)

    marked = 0
    seen: set = set()
    out: List[str] = []
    for idx, ln in enumerate(lines):
        if toc_span and toc_span[0] <= idx < toc_span[1]:
            out.append(ln)          # 目录块原样保留，不参与标题标记
            continue
        level = None
        if ln and _looks_like_title(ln):
            title = _normalize_title(ln)
            hit = (_cn_section_match(title) or _cn_chapter_match(title)
                   or _numbered_match(title))
            if hit is not None:
                level = hit[0]
            elif _ALLCAPS_RE.match(title):
                # 弱模式：要求前后空行（与 _extract_from_body_headings 一致）。
                prev_blank = idx == 0 or lines[idx - 1] == ""
                nxt_blank = idx + 1 >= len(lines) or lines[idx + 1] == ""
                if prev_blank and nxt_blank:
                    level = 1
            elif title in toc_level:
                # 目录里列了、正文又原样出现的短标题 → 按目录级别补标。
                level = toc_level[title]
            if level is not None:
                if title in seen:
                    level = None    # 同名标题只标首现
                else:
                    seen.add(title)
        if level is not None:
            marked += 1
            out.append(f"{'#' * min(level, 6)} {ln}")
        else:
            out.append(ln)

    if marked < 2:
        return text
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Digest assembly
# ---------------------------------------------------------------------------

def build_toc(chapters: List[Chapter]) -> str:
    """Render the table of contents from a chapter list."""
    lines = []
    for c in chapters:
        indent = "  " * max(0, c.level - 1) if c.level > 0 else ""
        prefix = "#" * c.level if c.level > 0 else ""
        lines.append(f"{indent}{prefix} {c.title}".rstrip())
    return "\n".join(lines)


def build_digest(markdown: str) -> Tuple[str, List[Chapter]]:
    """Build the condensed digest (缩略版本) for a document.

    Tries Markdown headings first; if none are found, falls back to plain-text
    heuristics so pure-text PDFs/EPUBs still yield a meaningful TOC.

    Returns ``(digest_text, chapters)`` where ``digest_text`` is the TOC and
    ``chapters`` is the full chapter list (useful for targeted retrieval by
    title).
    """
    chapters = extract_chapters(markdown)
    if not chapters:
        chapters = extract_chapters_plain(markdown)

    lines = ["# 目录\n", build_toc(chapters)]
    digest = "\n".join(lines)
    return digest, chapters


def find_chapters_by_title(chapters: List[Chapter], keyword: str) -> List[Chapter]:
    """Return chapters whose title contains ``keyword`` (for targeted RAG)."""
    if not keyword:
        return []
    kw = keyword.lower()
    return [c for c in chapters if kw in c.title.lower()]
