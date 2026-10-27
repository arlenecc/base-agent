"""Text chunker — splits documents into overlapping chunks for RAG.

Chunk sizes are measured in *tokens* estimated by ``estimate_tokens``
(character-class weighted, calibrated against bge-small-zh-v1.5's WordPiece —
see that function for the per-class rules).
"""
from __future__ import annotations

import logging
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# token estimation
# ---------------------------------------------------------------------------

# 按字符类别计权的 token 估算（bge-small-zh-v1.5 的 WordPiece 校准）。
# 旧实现把「非 CJK」统一按 3 字符/token 折算，但数字与标点（ASCII + 全角 +
# General Punctuation）在 BERT 词表里是**每字符独立 token**——标点/代码/URL
# 密集的切片真实 token 数超出估算 30%+（实测 450 → 627），嵌入时被 512 窗口
# 静默截断。分类保证「估算 ≥ 真实值」：切片只会偏小，绝不超窗。
#
#   * CJK 汉字（含扩展 A）/ 假名 / CJK 标点 / 全角形式 → 每字符 1 token
#   * 其余字符（数字、ASCII 标点、引号破折号等符号、希腊/西里尔等）→ 每字符
#     1 token（BERT 词表内单 token 或最坏 [UNK]，均不合并）
#   * ASCII 字母 → ≈3.39 字符/token（英文正文实测），按 3 取整略保守；
#     字母段数（缩写词如 API/UI 每词 1 token）参与兜底
#   * 空白 / 控制字符 → 0 token（BasicTokenizer 分词前丢弃）
#
# 实现：单次 str.translate（C 层）把字符分桶成标记字符 + str.count 统计，
# 不物化 per-match 对象——estimate_tokens() 在切片热路径上被极高频调用，
# finditer/findall 会为整篇文档产生几万个临时对象（与 kernel.budget 同款优化）。

_WIDE_MARK, _LETTER_MARK, _SPACE_MARK = "\x01", "\x02", "\x03"


def _build_bucket_table() -> dict:
    """字符码点 → 分桶标记的 translate 表（模块级构建一次）。

    \\x01-\\x03 本身是 C0 控制字符，已在表内映射为空白桶，不会与正文冲突。
    """
    table: dict = {}

    def _mark(lo: int, hi: int, mark: str) -> None:
        for cp in range(lo, hi + 1):
            table[cp] = mark

    # C0 控制字符 + ASCII 空白 + DEL → 空白桶（0 token）
    _mark(0x00, 0x20, _SPACE_MARK)
    table[0x7F] = _SPACE_MARK
    # ASCII 字母
    _mark(0x41, 0x5A, _LETTER_MARK)
    _mark(0x61, 0x7A, _LETTER_MARK)
    # CJK 标点(3000-303F) / 假名(3040-30FF) / 汉字扩展A / 汉字 / 兼容表意 / 全角
    _mark(0x3000, 0x30FF, _WIDE_MARK)
    _mark(0x3400, 0x4DBF, _WIDE_MARK)
    _mark(0x4E00, 0x9FFF, _WIDE_MARK)
    _mark(0xF900, 0xFAFF, _WIDE_MARK)
    _mark(0xFF00, 0xFFEF, _WIDE_MARK)
    return table


_BUCKET_TABLE = _build_bucket_table()
_LETTER_RUN_RE = re.compile("\x02+")  # 字母桶标记的连续段（= 英文单词/缩写）


def estimate_tokens(text: str) -> int:
    """估算文本的 embedding token 数（按 bge-small-zh-v1.5 校准，见上）。

    任何文本的真实值 ≤ 估算值 + 2（[CLS]/[SEP]），因此
    ``rag_chunk_size ≤ 510`` 时切片必不超出 bge 的 512 窗口。
    """
    if not text:
        return 0
    bucketed = text.translate(_BUCKET_TABLE)
    letter_chars = bucketed.count(_LETTER_MARK)
    # 字母段折叠成单标记（C 层 sub，不物化 run 对象）：折叠后的计数 = 段数，
    # wide/space 计数不受影响，直接在折叠串上取。
    collapsed = _LETTER_RUN_RE.sub(_LETTER_MARK, bucketed)
    wide = collapsed.count(_WIDE_MARK)         # CJK/假名/全角 → 1 token/字符
    spaces = collapsed.count(_SPACE_MARK)      # 0 token
    # 其余 = 数字/标点/符号/其它文字 → 每字符 1 token
    other = len(text) - wide - letter_chars - spaces
    # 字母段数兜底：缩写词（API/UI/SDK）在词表内是整词 1 token，但两三个
    # 字母的短词按 chars//3 会算出 0——按段数取较大者。
    letters = max(letter_chars // 3, collapsed.count(_LETTER_MARK))
    return wide + other + letters


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
    use_semantic: bool = True,
) -> List[Dict[str, Any]]:
    """Chunk multiple documents with metadata.

    ``embedding_function`` 会一路透传给 hybrid/semantic 切分器。RAGEngine 注入
    的自定义 EF（测试的 fake_ef、未来可替换的远端 embedding）必须能被语义
    切片用到——否则语义切片仍会去加载共享的本地 ONNX 模型，既浪费内存，
    在离线环境还会卡在模型下载上。

    Args:
        documents: List of dicts with 'source' (filepath) and 'text' keys.
        chunk_size: **最终切片**（含下一片头部的重叠）的 token 预算上限。
            切分器实际拿到的正文预算为 ``chunk_size - overlap``，因此相邻片
            「重叠 + 正文」不会超过该值——直接对应 embedding 模型的 token
            窗口（如 bge-small-zh 的 512）。
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
    # 重叠头部是**追加**在下一片之上的、不占该片自身的切块预算——若不补偿，
    # 最终切片会达到 chunk_size + overlap，超出 embedding 模型的 token 窗口
    # （如 500+50=550 > 512）。这里把正文预算压为 chunk_size - overlap，使
    # 「重叠头 + 正文 ≤ chunk_size」，并把显式 overlap 传下去（percent=None，
    # 避免底层再用 body 预算反推 overlap 导致数值漂移）。
    eff_overlap = int(chunk_size * overlap_percent) if overlap_percent else chunk_overlap
    body_budget = max(min_chunk_size, chunk_size - eff_overlap)
    for doc in documents:
        source = doc.get("source", "unknown")
        text = doc.get("text", "")
        total_tokens = estimate_tokens(text)
        logger.debug("      文档切片: %s (%d 字符, 约 %d tokens)", source, len(text), total_tokens)
        if use_semantic:
            chunks = _chunk_with_hybrid_fallback(
                text, body_budget, eff_overlap, min_chunk_size, None,
                embedding_function,
            )
        else:
            # 关闭语义切片：跳过混合/语义分片（不加载任何嵌入模型），直接按字数
            # （token）递归切分。递归分片确定性、可离线，且已保证切片是原文连续
            # 子串、不丢内容——适合用户明确不想做语义切片的场景。
            chunks = chunk_text(text, body_budget, eff_overlap)
        doc_chunks = [
            {"source": source, "chunk_index": i, "text": chunk}
            for i, chunk in enumerate(chunks)
        ]
        # 句末收口按文档进行：需要原文 text 定位「残句 → 下一片」的真实跨度。
        snapped = _snap_to_sentence_end(doc_chunks, source=text)
        # 覆盖性兜底（#内容保真）：若语义/混合分片在退化 EF 下漏切（出现真实内容
        # 间隔），回退到确定性递归分片——后者已验证无损。正常路径（真实 EF）覆盖
        # 完整，此检查仅 O(n) 扫描，几乎零开销。
        if not _covers_source(snapped, text):
            logger.warning(
                "chunker: 分片未覆盖原文全文（疑似丢内容），回退递归分片: %s", source
            )
            fallback = chunk_text(text, body_budget, eff_overlap)
            doc_chunks = [
                {"source": source, "chunk_index": i, "text": c}
                for i, c in enumerate(fallback)
            ]
            snapped = _snap_to_sentence_end(doc_chunks, source=text)
        all_chunks.extend(snapped)
    # 最终预算收口：兜底任何超 budget 的切片（语义分片退化 / 残句并入下一片），
    # 确保每片估算 ≤ chunk_size ⇒ 真实 ≤ chunk_size+2 ≤ embedding 模型窗口
    # （bge 512）。token 硬切不依赖 EF，确定性。
    return _clamp_chunk_budget(all_chunks, chunk_size)


def _covers_source(chunks: List[Dict[str, Any]], source: str) -> bool:
    """用布尔覆盖数组权威判定切片是否覆盖原文所有非空白字符。

    之前用 ``str.find`` 从游标贪心定位，对重叠/重复文本会误判为已覆盖（假阳性），
    导致丢内容兜底不触发。现改为：把每片文本在原文的（贪心）出现位置标记到覆盖
    数组，最后要求每个非空白字符都已被覆盖。重叠、重复、边界空白被 ``.strip()``
    掉均无影响；只有「某段非空白内容从未出现在任何切片中」才会判未覆盖——即真实
    丢内容（语义/混合分片在退化 EF 下可能发生）。
    """
    if not source.strip():
        return True
    covered = bytearray(len(source))
    cursor = 0
    for c in chunks:
        t = (c.get("text") or "").strip()
        if not t:
            continue
        # 从已覆盖末尾继续找，失败再退回全文首现——重叠片起点在前一片之内也能命中。
        p = source.find(t, cursor)
        if p < 0:
            p = source.find(t)
        if p < 0:
            return False
        covered[p:p + len(t)] = b"\x01" * len(t)
        cursor = p + len(t)
    # 非空白字符须全部被覆盖
    return all(covered[i] or ch.isspace() for i, ch in enumerate(source))


# 句末判定：句号/问号/叹号/省略号/换行，或英文句点后跟空白（". "）。
# 仅用于收口判断与边界查找，不要求覆盖全部标点风格。
_SENTENCE_END_RE = re.compile(r"[。！？!?…\n]|\.(?=\s|$)")
# 残句与下一片之间在原文里的间隔上限：超过视为误定位（跨章节重名等），放弃收口。
_SNAP_MAX_GAP = 256


def _snap_to_sentence_end(chunks: List[Dict[str, Any]],
                          source: Optional[str] = None) -> List[Dict[str, Any]]:
    """保证每个分片（除末片外）都结束在句子边界。

    语义/混合主路径的边界本就按句切（chonkie ``_DELIM`` 按句拆分、
    ``_tail_by_tokens`` 句末对齐重叠），但递归兜底 ``_split_by_tokens``
    在窗口内找不到任何句末标点时仍可能句中截断。这里做统一兜底：把片尾
    残句并入下一片。

    关键约束：**分片必须是原文的连续子串**（下游按原文命中展示）。因此并入
    残句时不做字符串拼接（那会插入原文里不存在的分隔符），而是回到原文定位
    「残句起点 → 下一片终点」的真实跨度，整段原文重建下一片——二者之间的
    空白/换行保持原样。

    定位失败（原文中找不到残句 / 间隔异常）或残句过长（多半是单个长句被硬切）
    时不强拆，保持原样——宁可句中截断也不产出原文里不存在的文本。
    """
    if not source:
        return chunks
    for i in range(len(chunks) - 1):
        cur = chunks[i].get("text") or ""
        nxt = chunks[i + 1].get("text") or ""
        if not cur or not nxt or _SENTENCE_END_RE.search(cur[-2:]):
            continue  # 已以句末标点/换行收尾
        ends = [m.end() for m in _SENTENCE_END_RE.finditer(cur)]
        if not ends:
            continue  # 整片无句末标点（代码/表格等），不强拆
        cut = ends[-1]
        tail = cur[cut:]
        if not tail.strip() or len(tail.strip()) > max(120, len(cur) // 3):
            continue  # 残句过长（多半是单个长句被硬切），保持原样最安全
        t_pos = source.find(tail)
        if t_pos < 0:
            continue  # 原文里定位不到（分片被深度改写），放弃收口
        tail_end = t_pos + len(tail)
        if tail.strip() in nxt:
            # 下一片头部已带上这段残句（重叠切片的常态）：直接收口，零丢失。
            chunks[i]["text"] = cur[:cut].rstrip()
            continue
        # 非重叠相邻：下一片应紧跟残句之后（允许 ≤ _SNAP_MAX_GAP 的间隔）。
        n_pos = source.find(nxt, t_pos)
        if n_pos < 0 or not (tail_end <= n_pos <= tail_end + _SNAP_MAX_GAP):
            continue  # 间隔异常（跨章节误匹配等），保持原样
        chunks[i]["text"] = cur[:cut].rstrip()
        chunks[i + 1]["text"] = source[t_pos:n_pos + len(nxt)]
    return chunks


def _clamp_chunk_budget(
    chunks: List[Dict[str, Any]], chunk_size: int,
) -> List[Dict[str, Any]]:
    """最终预算收口：任何超预算的切片用确定性硬切兜底。

    两道独立保险：
      1. token 预算：每片估算 ``estimate_tokens`` ≤ chunk_size ⇒ 真实 ≤
         chunk_size+2 ≤ embedding 模型窗口（bge 512）。上游语义/混合分片在退化
         情形下可能产出超大切片，或 ``_snap_to_sentence_end`` 残句并入下一片把
         切片推过预算——这里统一收口。
      2. 字数硬上限：小窗口模型（bge 等，chunk_size 已被夹到 ≤512，此时 1 字≈1
         token）额外保证每片字符数 ≤ 500——即「切片（含其头部重叠）≤ 500 字」。
         大窗口模型（nomic RoPE 8K 等）token 预算已远小于窗口，不再卡字符以免
         英文/代码被无谓碎裂成 500 字小片。

    token 硬切用 ``_split_by_tokens``（无重叠）；若仍有长串（英文 510 token ≈
    1700 字符）超字数上限，再用 ``_split_by_chars`` 按字符兜底。丢失的是语义边界
    连续性而非内容。
    """
    # 小窗口（char≈token）模型才卡字数上限；大窗口模型交给 token 预算即可。
    # 上限直接跟随（已被模型窗口夹紧后的）chunk_size，而非写死的 500：用户在
    # 大窗口模型上设长分片时，这里绝不再把切片压回 500 字（避免「设了长分片却
    # 在预算收口环节被切短」），小窗口模型则与 token 预算持平、不超出窗口。
    max_chars = chunk_size if chunk_size <= 512 else None
    out: List[Dict[str, Any]] = []
    for c in chunks:
        est = estimate_tokens(c["text"])
        over = est > chunk_size or (max_chars is not None and len(c["text"]) > max_chars)
        if not over:
            out.append(c)
            continue
        logger.info(
            "chunker: 切片超预算（估算 %d token / %d 字 > %d token / %s 字），兜底硬切",
            est, len(c["text"]), chunk_size, max_chars,
        )
        for piece in _split_by_tokens(c["text"], chunk_size, 0):
            if max_chars is None or len(piece) <= max_chars:
                out.append({"source": c["source"], "chunk_index": 0, "text": piece})
            else:
                out.extend(
                    {"source": c["source"], "chunk_index": 0, "text": p}
                    for p in _split_by_chars(piece, max_chars)
                )
    # 重排 chunk_index（硬切插入了新片；下游依赖连续索引）
    for i, c in enumerate(out):
        c["chunk_index"] = i
    return out


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
                new_tokens = new_cjk + new_other // 3
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
                if part_cjk + part_other // 3 <= chunk_size:
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
    分类规则与 ``estimate_tokens`` 完全一致（字符桶 + 字母段兜底）——旧实现
    在此维护了一份 ``cjk + other // 3`` 的独立计数器，估算器改按类别计权后
    会重新低估数字/标点，故统一走 _BUCKET_TABLE。
    """
    chunks: List[str] = []
    pos = 0  # character position
    text_len = len(text)

    while pos < text_len:
        # Find a character window that fits within token budget.
        # Incremental: add one char at a time, update running counts.
        end = pos
        wide = letter_chars = others = runs = 0
        prev_letter = False
        while end < text_len:
            mark = _BUCKET_TABLE.get(ord(text[end]))
            if mark is _WIDE_MARK:
                wide += 1
                prev_letter = False
            elif mark is _LETTER_MARK:
                letter_chars += 1
                if not prev_letter:
                    runs += 1  # 段首：字母段数兜底（API/UI 整词 1 token）
                prev_letter = True
            elif mark is _SPACE_MARK:
                prev_letter = False  # 空白/控制字符：0 token
            else:
                others += 1
                prev_letter = False
            if wide + others + max(letter_chars // 3, runs) > chunk_size:
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
            if ov_cjk + ov_other // 3 >= chunk_overlap:
                break
        pos = max(overlap_start, pos + 1)  # ensure forward progress

    return chunks


def _split_by_chars(text: str, max_chars: int) -> List[str]:
    """字符级兜底硬切：保证每片字符数 ≤ ``max_chars``。

    仅用于预算收口对「token 已达标但字符数仍超限」的最后兜底——英文/URL/代码
    等长串（510 token 英文 ≈ 1700 字符）在 token 预算下不触发拆分，但违反
    「切片 ≤ 500 字」的字数硬上限。在换行/句号/空白边界断开，找不到边界则硬切；
    不附加重叠（安全网，语义连续性已不可达）。
    """
    if max_chars <= 0 or not text:
        return []
    chunks: List[str] = []
    n = len(text)
    pos = 0
    while pos < n:
        end = min(pos + max_chars, n)
        if end < n:
            # 向后找最近的自然边界，但不得回退超一半，且不得越过 max_chars。
            for brk in ("\n", "。", " ", ". "):
                lb = text.rfind(brk, pos + max_chars // 2, end)
                if lb > pos and lb + len(brk) <= pos + max_chars:
                    end = lb + len(brk)
                    break
        piece = text[pos:end].strip()
        if piece:
            chunks.append(piece)
            pos = end
        else:
            # 整段是空白：前进 1 字符，避免死循环
            pos = pos + 1
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
                if m_cjk + m_other // 3 >= overlap:
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