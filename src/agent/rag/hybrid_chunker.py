"""Hybrid chunking: document structure + semantic splitting.

Combines the strengths of two approaches:

  1. **Structure-aware splitting** — Markdown headings (``#`` … ``######``)
     and blank-line paragraph boundaries delimit natural "sections" of a
     document, so a chunk never straddles a section boundary (preserving
     headings, lists, tables and code blocks as coherent units).

  2. **Semantic splitting** — within an over-long section, we fall back to
     chonkie's SemanticChunker (nomic-embed-text) to cut at topic boundaries
     rather than arbitrary character offsets.

Chunk sizing (all measured in tokens via the shared heuristic):
  * min 100 tokens — undersized sections are merged with their neighbours.
  * max 1000 tokens — over-long sections are split (semantically).
  * overlap 5% (50 tokens) — consecutive chunks borrow the tail sentence(s)
    of the previous chunk to preserve cross-boundary context.

The pipeline is:

    markdown text
      └─ _split_by_structure()          # headings + blank lines
           ├─ section ≤ 800 tokens → keep
           └─ section > 800 tokens → semantic_chunk_text() (chonkie)
      └─ _merge_to_target_size()        # merge undersized chunks to ≥100
      └─ _apply_overlap()               # 10% overlap, sentence-aligned
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

# Heading levels we treat as structural boundaries.  Lists/tables/code blocks
# stay inside their enclosing section (they follow a heading or blank line).
_HEADING_RE = re.compile(r"^(#{1,6})\s+\S", re.MULTILINE)


def hybrid_chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 50,
    min_chunk_size: int = 100,
    overlap_percent: float = 0.05,
    embedding_function=None,
) -> List[str]:
    """Split text using structure + semantic hybrid chunking.

    Args:
        text: The (Markdown) text to split.
        chunk_size: Maximum chunk size in tokens (1000).
        chunk_overlap: Overlap in tokens (default 5% of chunk_size = 50).
        min_chunk_size: Minimum chunk size in tokens (100).
        overlap_percent: Overlap as a fraction of chunk_size.
        embedding_function: Optional ``embed_documents(list[str]) -> list``
            callable for semantic splitting of over-long sections.  When
            omitted, the shared nomic-embed-text singleton is used.

    Returns:
        List of chunk strings.
    """
    if not text or not text.strip():
        return []

    from .chunker import estimate_tokens

    # 1. Structure-aware split.
    sections = _split_by_structure(text)

    # 2. Split over-long sections semantically (chonkie).
    chunks: List[str] = []
    for section in sections:
        if estimate_tokens(section) <= chunk_size:
            chunks.append(section)
        else:
            sub = _semantic_split(section, chunk_size, embedding_function)
            chunks.extend(sub if sub else _split_by_paragraph(section, chunk_size))

    # 3. Merge undersized chunks to reach min_chunk_size.
    merged = _merge_to_target_size(chunks, chunk_size, min_chunk_size, source=text)

    # 4. Apply overlap for cross-boundary continuity.
    return _apply_overlap(merged, chunk_size, chunk_overlap, overlap_percent,
                          source=text)


# ---------------------------------------------------------------------------
# structure-aware split
# ---------------------------------------------------------------------------

def _find_heading_starts(
    text: str, ignore_fences: bool = False,
) -> Optional[List[int]]:
    """返回所有标题行的**绝对偏移**；遇到未闭合围栏时返回 ``None``。

    ``ignore_fences=True`` 时不识别围栏，用于未闭合代码块的重扫回退。

    Fenced code blocks (```` ``` ```` / ``~~``) are skipped: shell and Python
    comments inside them start with ``#`` and would otherwise be mistaken for
    Markdown headings, cutting code samples in half.  Only a fence of the same
    kind can close a fence (`` ``` `` is not closed by ``~~~``).
    """
    heading_starts: List[int] = []
    in_fence = False
    fence_marker = ""
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if not in_fence:
            if not ignore_fences and (stripped.startswith("```") or stripped.startswith("~~~")):
                in_fence = True
                fence_marker = stripped[:3]
                offset += len(line)
                continue
        else:
            if stripped.startswith(fence_marker):
                in_fence = False
                fence_marker = ""
            offset += len(line)
            continue
        if _HEADING_RE.match(line):
            heading_starts.append(offset)
        offset += len(line)
    if in_fence:
        return None      # 围栏未闭合
    return heading_starts


def _split_by_structure(text: str) -> List[str]:
    """Split text at heading boundaries, preserving each heading with its body.

    Blank-line paragraph groups are kept together.  A section is: a heading
    line plus everything up to (but not including) the next heading of the
    same or higher level.

    Fenced code blocks (```` ``` ```` / ``~~``) are skipped: shell and Python
    comments inside them start with ``#`` and would otherwise be mistaken for
    Markdown headings, cutting code samples in half.
    """
    # Find heading positions, ignoring anything inside a fenced code block.
    # 记录的是「相对于整篇 text 的绝对偏移」，供下面切片使用。
    heading_starts = _find_heading_starts(text)
    if heading_starts is None:
        # 遇到**未闭合**的代码围栏：其后所有 # 标题都被当成代码内容忽略，
        # 整篇文档会退化成 1 个 section，再对全文跑语义切分（慢且失去结构
        # 保护）。此时忽略围栏重扫一次 —— 宁可切坏代码块，也别丢掉全部标题。
        heading_starts = _find_heading_starts(text, ignore_fences=True)

    if not heading_starts:
        # No headings: fall back to blank-line paragraph groups.
        return _split_by_paragraph(text, max_tokens=None)

    sections: List[str] = []
    for i, start in enumerate(heading_starts):
        end = heading_starts[i + 1] if i + 1 < len(heading_starts) else len(text)
        sections.append(text[start:end].strip())

    # Prepend any text before the first heading as its own section.
    if heading_starts and heading_starts[0] > 0:
        prefix = text[:heading_starts[0]].strip()
        if prefix:
            sections.insert(0, prefix)

    return [s for s in sections if s.strip()]


def _split_by_paragraph(text: str, max_tokens: Optional[int]) -> List[str]:
    """Split by blank-line-separated paragraphs, optionally grouping up to
    ``max_tokens`` tokens per group."""
    from .chunker import estimate_tokens

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if max_tokens is None:
        return paragraphs

    groups: List[str] = []
    current = ""
    # 增量维护当前组的 token 数：旧写法在每次追加时重新估算整个 current，
    # 组越长扫描越长 → 整段文本是 O(n²)，大章节下明显卡顿。
    current_tokens = 0
    for p in paragraphs:
        p_tokens = estimate_tokens(p)
        if not current:
            current = p
            current_tokens = p_tokens
        elif current_tokens + p_tokens <= max_tokens:
            current = current + "\n\n" + p
            current_tokens += p_tokens
        else:
            groups.append(current)
            current = p
            current_tokens = p_tokens
    if current:
        groups.append(current)
    return groups


# ---------------------------------------------------------------------------
# semantic split (chonkie) for over-long sections
# ---------------------------------------------------------------------------

def _semantic_split(section: str, chunk_size: int, embedding_function) -> Optional[List[str]]:
    try:
        from .semantic_chunker import semantic_chunk_text

        return semantic_chunk_text(
            section,
            chunk_size=chunk_size,
            chunk_overlap=0,  # overlap applied later, uniformly across all chunks
            min_chunk_size=1,  # no merging here; done globally after
            overlap_percent=0.0,
            embedding_function=embedding_function,
        )
    except Exception as e:
        logger.warning("semantic split failed, falling back to paragraphs: %s", e)
        return None


# ---------------------------------------------------------------------------
# merge + overlap (reuse the semantic_chunker helpers for consistency)
# ---------------------------------------------------------------------------

def _merge_to_target_size(chunks: List[str], chunk_size: int, min_chunk_size: int,
                          source: Optional[str] = None) -> List[str]:
    from .semantic_chunker import _merge_to_target_size as _m
    return _m(chunks, chunk_size, min_chunk_size, source=source)


def _apply_overlap(
    chunks: List[str],
    chunk_size: int,
    chunk_overlap: int,
    overlap_percent: float,
    source: Optional[str] = None,
) -> List[str]:
    from .semantic_chunker import _apply_overlap as _o
    return _o(chunks, chunk_size, chunk_overlap, overlap_percent, source=source)
