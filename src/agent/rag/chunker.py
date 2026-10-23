"""Text chunker — splits documents into overlapping chunks for RAG.

Chunk sizes are measured in *tokens* (estimated via character-based heuristic:
~2 chars/token for Chinese, ~3 chars/token for English).
"""
from __future__ import annotations

import logging
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# token estimation
# ---------------------------------------------------------------------------

# 预编译正则 + 用 finditer 计数：estimate_tokens() 在切片热路径上被调用
# 非常频繁，re.findall 会把所有匹配物化成一个列表（整篇文档几万个中文字符
# = 几万个临时单字符串对象），纯属浪费。
_CJK_RE = re.compile(r'[\u4e00-\u9fff]')


def estimate_tokens(text: str) -> int:
    """Roughly estimate token count.

    Chinese: ~2 chars per token.  English: ~3 chars per token.
    """
    chinese_chars = sum(1 for _ in _CJK_RE.finditer(text))
    other_chars = len(text) - chinese_chars
    return chinese_chars // 2 + other_chars // 3


def _token_count(text: str) -> int:
    return max(estimate_tokens(text), 1) if text else 0


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 50,
    separators: List[str] | None = None,
) -> List[str]:
    """Split text into overlapping chunks using token-based sizing.

    Args:
        text: The text to split.
        chunk_size: Target size of each chunk in **tokens**.
        chunk_overlap: Overlap between chunks in **tokens**.
        separators: Priority-ordered list of separators to split on.

    Returns:
        List of text chunks.
    """
    if not text.strip():
        return []

    if separators is None:
        separators = ["\n\n", "\n", "。", ". ", " ", ""]

    return _split_recursive(text, separators, chunk_size, chunk_overlap)


def chunk_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = 1000,
    chunk_overlap: int = 50,
    min_chunk_size: int = 100,
    overlap_percent: float = 0.05,
    embedding_function=None,
) -> List[Dict[str, Any]]:
    """Chunk multiple documents with metadata.

    ``embedding_function`` 会一路透传给 hybrid/semantic 切分器。RAGEngine 注入
    的自定义 EF（测试的 fake_ef、未来可替换的远端 embedding）必须能被语义
    切片用到——否则语义切片仍会去加载共享的本地 ONNX 模型，既浪费内存，
    在离线环境还会卡在模型下载上。

    Args:
        documents: List of dicts with 'source' (filepath) and 'text' keys.
        chunk_size: Target maximum chunk size in **tokens** (1000).
        chunk_overlap: Overlap between chunks in **tokens** (default 5% of
            ``chunk_size`` = 50).
        min_chunk_size: Target minimum chunk size in **tokens** (100) — used by
            the semantic path to merge undersized chunks.
        overlap_percent: Fraction of ``chunk_size`` used as overlap between
            consecutive chunks (0.05 => 50 tokens), preserving continuity —
            used by the semantic path.

    Returns:
        List of dicts with 'source', 'chunk_index', and 'text' keys.

    Chunking strategy: hybrid chunking (document structure + semantic split
    via chonkie SemanticChunker + shared nomic-embed-text model) when
    available, falling back to plain semantic chunking, then the recursive
    chunker.
    """
    all_chunks: List[Dict[str, Any]] = []
    for doc in documents:
        source = doc.get("source", "unknown")
        text = doc.get("text", "")
        total_tokens = estimate_tokens(text)
        logger.debug("      文档切片: %s (%d 字符, 约 %d tokens)", source, len(text), total_tokens)
        chunks = _chunk_with_hybrid_fallback(
            text, chunk_size, chunk_overlap, min_chunk_size, overlap_percent,
            embedding_function,
        )
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "source": source,
                "chunk_index": i,
                "text": chunk,
            })
    return all_chunks


def _chunk_with_hybrid_fallback(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
    overlap_percent: float,
    embedding_function=None,
) -> List[str]:
    """Hybrid-chunk first (structure + semantic), then semantic, then recursive.

    The fallback chain is deliberately ordered from highest to lowest
    fidelity:
      1. hybrid_chunk_text  — Markdown-structure-aware + semantic split
      2. semantic_chunk_text — pure semantic split (chonkie)
      3. chunk_text         — recursive character split (no model needed)
    """
    # 1. Hybrid (structure + semantic).
    try:
        from .hybrid_chunker import hybrid_chunk_text

        hybrid = hybrid_chunk_text(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            overlap_percent=overlap_percent,
            embedding_function=embedding_function,
        )
        if hybrid:
            return hybrid
    except Exception:  # pragma: no cover - defensive
        logger.warning("hybrid chunking raised, falling back to semantic", exc_info=True)

    # 2. Semantic (chonkie).
    try:
        from .semantic_chunker import semantic_chunk_text

        semantic = semantic_chunk_text(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            overlap_percent=overlap_percent,
            embedding_function=embedding_function,
        )
        if semantic is not None:
            return semantic
    except Exception:  # pragma: no cover - defensive; semantic_chunk_text already guards
        logger.warning("semantic chunking raised, falling back to recursive", exc_info=True)

    # 3. Recursive.
    return chunk_text(text, chunk_size, chunk_overlap)


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _split_recursive(
    text: str,
    separators: List[str],
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """Recursively split text at the most appropriate separator level."""
    if _token_count(text) <= chunk_size:
        return [text] if text.strip() else []

    # Try each separator in priority order
    for sep_idx, sep in enumerate(separators):
        if sep == "":
            # Character-level split
            return _split_by_tokens(text, chunk_size, chunk_overlap)

        if sep not in text:
            continue

        parts = text.split(sep)
        chunks: List[str] = []
        current = ""
        # Maintain running counts to avoid O(n²) rescans of `current`
        cur_cjk = 0
        cur_other = 0

        for part in parts:
            if current:
                sep_cjk, sep_other = _char_counts(sep)
                part_cjk, part_other = _char_counts(part)
                new_cjk = cur_cjk + sep_cjk + part_cjk
                new_other = cur_other + sep_other + part_other
                new_tokens = new_cjk // 2 + new_other // 3
                if new_tokens <= chunk_size:
                    cur_cjk = new_cjk
                    cur_other = new_other
                    current = current + sep + part
                else:
                    if current.strip():
                        chunks.append(current)
                    # If a single part is too large, split it further
                    if _token_count(part) > chunk_size:
                        sub_chunks = _split_recursive(
                            part, separators[sep_idx + 1:],
                            chunk_size, chunk_overlap,
                        )
                        chunks.extend(sub_chunks)
                        current = ""
                        cur_cjk = 0
                        cur_other = 0
                    else:
                        current = part
                        cur_cjk, cur_other = _char_counts(part)
            else:
                # First part
                part_cjk, part_other = _char_counts(part)
                if part_cjk // 2 + part_other // 3 <= chunk_size:
                    current = part
                    cur_cjk = part_cjk
                    cur_other = part_other
                else:
                    # Single part is too large — split further
                    sub_chunks = _split_recursive(
                        part, separators[sep_idx + 1:],
                        chunk_size, chunk_overlap,
                    )
                    chunks.extend(sub_chunks)
                    # Don't reset current — next part starts fresh

        if current.strip():
            chunks.append(current)

        if chunks:
            return _merge_overlap(chunks, chunk_overlap, sep, source=text)

    # Fallback: character-level split (was previously returning an oversized chunk)
    return _split_by_tokens(text, chunk_size, chunk_overlap)


def _char_counts(text: str) -> tuple:
    """Return (cjk_count, other_count) for incremental token estimation."""
    cjk = 0
    other = 0
    for ch in text:
        if 0x4e00 <= ord(ch) <= 0x9fff:
            cjk += 1
        else:
            other += 1
    return cjk, other


def _split_by_tokens(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text by character positions, respecting token budget.

    Uses incremental token counting: O(n) overall instead of O(n²).
    Maintains running CJK/other character counts and computes tokens
    as ``cjk // 2 + other // 3`` — matching ``estimate_tokens`` exactly
    but without rescanning the substring on every step.
    """
    chunks: List[str] = []
    pos = 0  # character position
    text_len = len(text)

    while pos < text_len:
        # Find a character window that fits within token budget.
        # Incremental: add one char at a time, update running counts.
        end = pos
        cjk = 0
        other = 0
        while end < text_len:
            ch = text[end]
            if 0x4e00 <= ord(ch) <= 0x9fff:
                cjk += 1
            else:
                other += 1
            if cjk // 2 + other // 3 > chunk_size:
                break
            end += 1

        # Try to break at a natural boundary within the window
        if end < text_len:
            # Search backward for a sentence or paragraph break
            for break_char in ['\n', '。', '. ', ' ']:
                last_break = text.rfind(break_char, pos, end)
                if last_break > pos + (end - pos) // 2:
                    end = last_break + len(break_char)
                    break

        chunk = text[pos:end].strip()
        if chunk:
            # 不再按内容去重：同一文档内的合法重复内容（每页页眉、重复的目录
            # 行、表格分隔行）会被静默丢弃，等于永久丢失这些片段；而切片重复
            # 对检索正确性没有影响（每条只是多占一行倒排/向量）。
            chunks.append(chunk)

        if end >= text_len:
            break

        # Advance position with overlap — incremental token count
        if chunk_overlap <= 0:
            # 零重叠：直接推进。旧实现循环条件 >= 0 恒真，仍会先吃进 1 个
            # 字符再判断，导致永远有 1 字符重叠，与「零重叠」语义不符。
            pos = end
            continue
        overlap_start = end
        ov_cjk = 0
        ov_other = 0
        while overlap_start > pos:
            ch = text[overlap_start - 1]
            if 0x4e00 <= ord(ch) <= 0x9fff:
                ov_cjk += 1
            else:
                ov_other += 1
            overlap_start -= 1
            if ov_cjk // 2 + ov_other // 3 >= chunk_overlap:
                break
        pos = max(overlap_start, pos + 1)  # ensure forward progress

    return chunks


def _gap_between(source: Optional[str], left: str, right: str, fallback: str = "\n\n") -> str:
    """Return the exact text that separates ``left`` and ``right`` in ``source``.

    拼接相邻 chunk 时硬编码分隔符会在「源文本里两者的真实间隔不是该分隔符」
    时伪造出原文没有的内容（chunk 不再是原文子串）。这里在原文中定位两个
    片段的相邻出现，返回真实间隔；定位失败/间隔异常大时退回 fallback。

    left 可能在原文出现多次（重复段落/样板文本），str.find 命中更早实例后
    j 会跨越真实 chunk 边界，算出的 gap 把不属于这两个 chunk 的文本拼进
    结果。这里限定 gap 上限，超限时尝试 left 的下一个出现位置。
    """
    if not source or not left or not right:
        return fallback
    _MAX_GAP = 256
    start = 0
    while start >= 0 and start < len(source):
        i = source.find(left, start)
        if i < 0:
            return fallback
        j = source.find(right, i + len(left))
        if j < 0:
            return fallback
        if j - (i + len(left)) <= _MAX_GAP:
            return source[i + len(left):j]
        start = i + 1
    return fallback


class _GapLocator:
    """``_gap_between`` 的游标版（对应 semantic_chunker._SourceLocator）。

    ``_merge_overlap`` 对每个 chunk 调一次 ``_gap_between``，而后者每次都从
    ``start=0`` 重新 ``str.find`` 全文档——对大书是 O(文档长度²/2)（几百页
    文档上切块耗时从秒级涨到分钟级）。这里记住「上一对片段的结束位置」，
    下一次从该位置继续找，整体退化成约 O(文档长度)。

    语义与 ``_gap_between`` 保持一致：重复片段最多依次试 8 个出现位置、
    gap 超过 256 字符即放弃。游标推进过头时（递归子片段/乱序）退回全文重找。
    """

    _MAX_GAP = 256
    _MAX_TRIES = 8

    def __init__(self, source: Optional[str]):
        self.source = source or ""
        self._pos = 0

    def gap(self, left: str, right: str, fallback: str) -> str:
        if not self.source or not left or not right:
            return fallback
        # 先从游标处找；找不到再从全文开头重试一次（游标可能被推过头）
        for base in (self._pos, 0):
            start = base
            for _ in range(self._MAX_TRIES):
                i = self.source.find(left, start)
                if i < 0:
                    break
                j = self.source.find(right, i + len(left))
                if j >= 0 and j - (i + len(left)) <= self._MAX_GAP:
                    self._pos = j          # 游标推进到 right 的起点
                    return self.source[i + len(left):j]
                start = i + 1
        return fallback


def _merge_overlap(chunks: List[str], overlap: int, sep: str,
                   source: Optional[str] = None,
                   locator: Optional["_GapLocator"] = None) -> List[str]:
    """Add token-based overlap between consecutive chunks.

    Prepends the tail of the previous chunk to the current chunk so that
    retrieval has surrounding context at chunk boundaries. The separator
    is included so the result remains a substring of the original text.

    ``locator``：传 ``_GapLocator`` 时用它定位真实间隔（带游标，O(N)）；
    不传则退化为每个 chunk 全文档 find 的 O(N²) 路径。
    """
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    if locator is None and source:
        locator = _GapLocator(source)

    merged: List[str] = [chunks[0]]
    for i in range(1, len(chunks)):
        # 取「原始前一段」的尾部，而不是 merged[-1] 的尾部。merged[-1]
        # 已经带上了一段重叠文本，若某段比 overlap 还短，向后回退会越界取
        # 到上一段的重叠部分，导致重叠逐段复合、chunk 无限膨胀。
        prev = chunks[i - 1]
        curr = chunks[i]

        # Take last N tokens from prev as overlap.
        # Incremental: walk backward, maintaining running counts.
        if _token_count(prev) > overlap:
            boundary = len(prev)
            m_cjk = 0
            m_other = 0
            while boundary > 0:
                ch = prev[boundary - 1]
                if 0x4e00 <= ord(ch) <= 0x9fff:
                    m_cjk += 1
                else:
                    m_other += 1
                boundary -= 1
                if m_cjk // 2 + m_other // 3 >= overlap:
                    break
            prev_end = prev[boundary:]

            # Prepend overlap to current chunk for context continuity.
            # prev_end + gap + curr 仍是原文子串：prev_end 是前段后缀、curr
            # 是原文里的下一段。分隔符必须是两者在原文中的**真实间隔**——
            # chunks 列表会混入深层递归的 sub_chunks（外层按 "\n\n" 切、
            # 深层按句号/空格切），相邻片段之间未必存在外层 sep；硬编码
            # sep 会伪造出原文没有的字符（与 semantic_chunker 的同类 bug）。
            # 没有原文可查时退回 sep（与旧行为一致）。
            if prev_end.strip():
                gap = (locator.gap(prev, curr, sep) if locator is not None
                       else _gap_between(source, prev, curr, fallback=sep))
                merged.append(prev_end + gap + curr)
            else:
                merged.append(curr)
        else:
            merged.append(curr)

    return merged