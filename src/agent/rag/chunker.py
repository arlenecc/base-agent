"""Text chunker — splits documents into overlapping chunks for RAG.

Chunk sizes are measured in *tokens* as returned by ``estimate_tokens``: it uses
the embedding model's own tokenizer when one is bound via
``configure_tokenizer`` (exact, any model), and otherwise falls back to a
character-class heuristic calibrated against bge-small-zh-v1.5's WordPiece.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# token estimation
# ---------------------------------------------------------------------------

# 切片预算的单位是 embedding 模型的**真实 token 数**。首选「模型自带分词器」：
# ``configure_tokenizer`` 从模型目录的 tokenizer.json 懒加载（tokenizers 库，
# Rust 实现），任意模型（BERT WordPiece / XLM-R SentencePiece / BPE）都能得到
# 与嵌入阶段一致的计数，不再依赖针对某个模型校准的近似。
#
# 没有分词器时（fastembed hub 模型名、精简权重目录、离线场景）回退到下面这套
# 字符类别启发式。它按 bge-small-zh-v1.5 的 WordPiece 校准，中英混排足够保守，
# 但对代码/数字密集文本**会低估**（实测 bge：代码段估算 260 / 真实 280），
# 因此只作 fallback——「估算恒 ≥ 真实」不再成立，超窗由 ``_clamp_chunk_budget``
# 兜底，不再需要靠估算器本身保证安全。
#
# 启发式分类规则：
#   * CJK 汉字（含扩展 A）/ 假名 / CJK 标点 / 全角形式 → 每字符 1 token
#   * 其余字符（数字、ASCII 标点、引号破折号等符号、希腊/西里尔等）→ 每字符
#     1 token（BERT 词表内单 token 或最坏 [UNK]，均不合并）
#   * ASCII 字母 → ≈3.39 字符/token（英文正文实测），按 3 取整略保守；
#     字母段数（缩写词如 API/UI 每词 1 token）参与兜底
#   * 空白 / 控制字符 → 0 token（BasicTokenizer 分词前丢弃）
#
# 实现：单次 str.translate（C 层）把字符分桶成标记字符 + str.count 统计，
# 不物化 per-match 对象——启发式估算在切片热路径上被极高频调用，
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


_TOKENIZER: Any = None          # tokenizers.Tokenizer | None
_TOKENIZER_PATH = ""
_TOKENIZER_LOCK = threading.Lock()


def configure_tokenizer(model: Optional[str] = None) -> bool:
    """绑定 embedding 模型的分词器；``model=None`` 解绑并退回启发式估算。

    ``model`` 可以是本地模型目录、``tokenizer.json`` 路径，或仓库 ``models/``、
    ``~/models/`` 下的模型名（经 registry 解析）。找不到文件时不绑定也不报错
    ——fastembed hub 模型名、精简权重目录都属于这种正常场景。

    进程级全局状态（embedding 模型切换时重新调用）。编码本身是线程安全的
    （tokenizers 为 Rust 实现），ingest 的线程池可直接并发调用。
    """
    global _TOKENIZER, _TOKENIZER_PATH
    loaded: Any = None
    path = _find_tokenizer_path(model) if model else ""
    # 同一模型重复构造 Engine（UI 同步 + agent 工具）时不必重新解析文件
    if path and path == _TOKENIZER_PATH and _TOKENIZER is not None:
        return True
    if path:
        try:
            from tokenizers import Tokenizer  # type: ignore

            loaded = Tokenizer.from_file(path)
        except Exception:
            logger.warning("chunker: 加载分词器失败，退回启发式估算: %s", path, exc_info=True)
            loaded, path = None, ""
    with _TOKENIZER_LOCK:
        _TOKENIZER, _TOKENIZER_PATH = loaded, path
    if loaded is not None:
        logger.debug("chunker: 切片预算按真实分词器计数: %s", path)
    return loaded is not None


def tokenizer_path() -> str:
    """当前绑定的分词器文件路径（空串 = 启发式估算）。"""
    return _TOKENIZER_PATH


def _find_tokenizer_path(model: str) -> str:
    """把模型标识解析成 ``tokenizer.json`` 路径；找不到返回空串。"""
    candidates = [os.path.expanduser(str(model))]
    try:
        from ..local import registry

        candidates.append(registry.resolve_model_path("embedding", model))
    except Exception:  # pragma: no cover - registry 不可用（精简安装）
        pass
    for cand in candidates:
        if os.path.isdir(cand):
            path = os.path.join(cand, "tokenizer.json")
            if os.path.isfile(path):
                return path
        elif os.path.isfile(cand) and cand.endswith(".json"):
            return cand
    return ""


def estimate_tokens(text: str) -> int:
    """文本的 token 数：已绑定分词器时用真实计数，否则用字符类别启发式。"""
    if not text:
        return 0
    tok = _TOKENIZER
    if tok is not None:
        try:
            return len(tok.encode(text, add_special_tokens=False).ids)
        except Exception:  # pragma: no cover - 分词器异常不应中断切片
            logger.warning("chunker: 分词器编码失败，退回启发式估算", exc_info=True)
    return _estimate_tokens_heuristic(text)


def _estimate_tokens_heuristic(text: str) -> int:
    """字符类别启发式估算（bge-small-zh-v1.5 校准；无分词器时的 fallback）。"""
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
    # 重叠不能吃掉正文预算（overlap ≥ chunk_size 时 body 会退化成 min_chunk_size，
    # 叠加后必然超窗）；min_chunk_size 也不能反过来把正文顶过 chunk_size。
    eff_overlap = max(0, min(int(eff_overlap), chunk_size - min(min_chunk_size, chunk_size)))
    body_budget = max(min_chunk_size, chunk_size - eff_overlap)
    body_budget = max(1, min(body_budget, chunk_size))
    for doc in documents:
        source = doc.get("source", "unknown")
        text = doc.get("text", "")
        total_tokens = estimate_tokens(text)
        logger.debug("      文档切片: %s (%d 字符, 约 %d tokens)", source, len(text), total_tokens)
        # 排障用：把「这次到底按什么预算/哪个分词器切」写清楚。日志里看到
        # 超预算切片时，先对照这一行——body+overlap 必须等于 chunk，
        # tokenizer 为 <heuristic> 说明分词器没绑上（计数口径会偏）。
        logger.debug(
            "chunker: 预算 chunk=%d body=%d overlap=%d semantic=%s tokenizer=%s",
            chunk_size, body_budget, eff_overlap, use_semantic,
            _TOKENIZER_PATH or "<heuristic>",
        )
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
        # 收口同样受 chunk_size 约束（并入后超预算则放弃，避免下游硬切）。
        snapped = _snap_to_sentence_end(doc_chunks, source=text, max_tokens=chunk_size)
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
            snapped = _snap_to_sentence_end(doc_chunks, source=text, max_tokens=chunk_size)
            # 兜底路径本身应当是确定无损的。这里再查一次只为「发现异常」——
            # 回退后仍不覆盖说明递归分片也有洞，需要单独排查，不能再静默继续。
            if not _covers_source(snapped, text):
                logger.warning(
                    "chunker: 递归兜底后仍未覆盖全文（需排查）：%s", source
                )
        all_chunks.extend(snapped)
    # 最终预算收口：兜底任何超 budget 的切片（语义分片退化 / 残句并入下一片），
    # 确保每片估算 ≤ chunk_size ⇒ 真实 ≤ chunk_size+2 ≤ embedding 模型窗口
    # （bge 512）。token 硬切不依赖 EF，确定性。
    return _clamp_chunk_budget(all_chunks, chunk_size)


def _covers_source(chunks: List[Dict[str, Any]], source: str) -> bool:
    """用布尔覆盖数组权威判定切片是否覆盖原文所有非空白字符。

    把每片文本在原文的（贪心）出现位置标记到覆盖数组，最后要求每个非空白字符
    都已被覆盖；分片文本在原文里**完全定位不到**（拼接伪造）直接判未覆盖。

    重复文本（代码块/样板段落）会让贪心定位把某片标到另一个同名实例上，留下一段
    伪影区间，所以对未覆盖区间再做一次确认：该区间文本若完整存在于任何分片里，
    视为定位伪影而非丢失。只有「任何切片里都找不到这段文本」才是真实丢内容
    （语义/混合分片在退化 EF、错位拼接时可能发生）。
    """
    if not source.strip():
        return True
    texts = [(c.get("text") or "").strip() for c in chunks]
    texts = [t for t in texts if t]
    covered = bytearray(len(source))
    cursor = 0
    for t in texts:
        # 从已覆盖末尾继续找，失败再退回全文首现——重叠片起点在前一片之内也能命中。
        p = source.find(t, cursor)
        if p < 0:
            p = source.find(t)
        if p < 0:
            return False  # 分片文本在原文里根本不存在（拼接伪造），判未覆盖
        covered[p:p + len(t)] = b"\x01" * len(t)
        cursor = p + len(t)
    # 非空白字符须全部被覆盖。重复段落/样板文本会让贪心定位把某片标到**另一个**
    # 同名实例上，留下一段「看起来没被覆盖」的区间——这类伪影的特征是该区间文本
    # 仍完整存在于某个分片里。真正的丢失是「任何分片里都找不到这段文本」。
    i = 0
    n = len(source)
    while i < n:
        if covered[i] or source[i].isspace():
            i += 1
            continue
        j = i
        while j < n and not covered[j]:
            j += 1
        frag = source[i:j].strip()
        if not frag or not any(frag in t for t in texts):
            return False
        covered[i:j] = b"\x01" * (j - i)
        i = j
    return True


# 句末判定：句号/问号/叹号/省略号/换行，或英文句点后跟空白（". "）。
# 仅用于收口判断与边界查找，不要求覆盖全部标点风格。
_SENTENCE_END_RE = re.compile(r"[。！？!?…\n]|\.(?=\s|$)")
# 残句与下一片之间在原文里的间隔上限：超过视为误定位（跨章节重名等），放弃收口。
_SNAP_MAX_GAP = 256


def _snap_to_sentence_end(chunks: List[Dict[str, Any]],
                          source: Optional[str] = None,
                          max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
    """保证每个分片（除末片外）都结束在句子边界。

    语义/混合主路径的边界本就按句切（chonkie ``_DELIM`` 按句拆分、
    ``_tail_by_tokens`` 句末对齐重叠），但递归兜底 ``_split_by_tokens``
    在窗口内找不到任何句末标点时仍可能句中截断。这里做统一兜底：把片尾
    残句并入下一片。

    关键约束：**分片必须是原文的连续子串**（下游按原文命中展示）。因此并入
    残句时不做字符串拼接（那会插入原文里不存在的分隔符），而是回到原文定位
    「残句起点 → 下一片终点」的真实跨度，整段原文重建下一片——二者之间的
    空白/换行保持原样。

    定位失败（原文中找不到残句 / 间隔异常）、残句过长（多半是单个长句被硬切）
    或**并入后超出 ``max_tokens``** 时不强拆，保持原样——宁可句中截断也不产出
    原文里不存在的文本，更不能为了收口把切片撑过预算。
    """
    if not source:
        return chunks
    # 每片在原文中的跨度（单调定位）。残句的绝对位置直接由「前片跨度末尾」
    # 推出，不再用 source.find(tail) 全文首现——重复代码块/样板段落会让它
    # 命中更早的实例，把下一片重建成错误区间的文本（正文就此丢失）。
    spans = _locate_spans([c.get("text") or "" for c in chunks], source)
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
        span = spans[i]
        if span is None or not cur.endswith(tail):
            continue  # 原文里定位不到（分片被深度改写），放弃收口
        t_pos = span[1] - len(tail)
        tail_end = span[1]
        if tail.strip() in nxt:
            # 下一片头部已带上这段残句（重叠切片的常态）：直接收口，零丢失。
            chunks[i]["text"] = cur[:cut].rstrip()
            spans[i] = (span[0], t_pos)
            continue
        # 非重叠相邻：下一片应紧跟残句之后（允许 ≤ _SNAP_MAX_GAP 的间隔）。
        n_pos = source.find(nxt, tail_end)
        if n_pos < 0 or not (tail_end <= n_pos <= tail_end + _SNAP_MAX_GAP):
            continue  # 间隔异常（跨章节误匹配等），保持原样
        merged_text = source[t_pos:n_pos + len(nxt)]
        # 收口也吃预算：残句并入下一片会把该片推过 token 上限，此时由
        # _clamp_chunk_budget 硬切——那等于「为了句末对齐先撑爆再切」，既不
        # 保留句界又多一次无谓分片。超预算就放弃这次收口（宁可句中截断）。
        if max_tokens is not None and estimate_tokens(merged_text) > max_tokens:
            continue
        chunks[i]["text"] = cur[:cut].rstrip()
        chunks[i + 1]["text"] = merged_text
        spans[i] = (span[0], t_pos)
        spans[i + 1] = (t_pos, n_pos + len(nxt))
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
        # 累计量改用统一的 estimate_tokens（真实分词器 / 启发式同一口径），
        # 不再维护旁边那套 cjk + other // 3 独立计数器。
        cur_tokens = 0
        sep_tokens = estimate_tokens(sep)

        for part in parts:
            part_tokens = estimate_tokens(part)
            if not current:
                if part_tokens <= chunk_size:
                    current, cur_tokens = part, part_tokens
                else:
                    # Single part is too large — split further
                    chunks.extend(_split_recursive(
                        part, separators[sep_idx + 1:],
                        chunk_size, chunk_overlap,
                    ))
                continue

            if cur_tokens + sep_tokens + part_tokens <= chunk_size:
                current = current + sep + part
                cur_tokens += sep_tokens + part_tokens
                continue

            if current.strip():
                chunks.append(current)
            # If a single part is too large, split it further
            if part_tokens > chunk_size:
                chunks.extend(_split_recursive(
                    part, separators[sep_idx + 1:],
                    chunk_size, chunk_overlap,
                ))
                current, cur_tokens = "", 0
            else:
                current, cur_tokens = part, part_tokens

        if current.strip():
            chunks.append(current)

        if chunks:
            # 正文预算 + 重叠预算 = 这一层允许的总上限：重叠只在放得下时追加。
            return _merge_overlap(chunks, chunk_overlap, sep, source=text,
                                  max_tokens=chunk_size + chunk_overlap)

    # Fallback: character-level split (was previously returning an oversized chunk)
    return _split_by_tokens(text, chunk_size, chunk_overlap)


def _split_by_tokens(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text by token boundaries, respecting the token budget.

    绑定真实分词器时走 ``_split_by_tokens_exact``（按 token 下标精确切），
    否则走字符滑窗 + 分桶估算（O(n) 增量计数，不物化 per-char 对象）。
    """
    tok = _TOKENIZER
    if tok is not None:
        return _split_by_tokens_exact(tok, text, chunk_size, chunk_overlap)

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
        # 重叠同样按 token 计：二分找最长后缀使后缀 token 数 ≤ chunk_overlap
        # （旧实现自带一份 cjk/other//3 计数器，与估算器口径不一致）
        lo, hi = pos, end
        while lo < hi:
            mid = (lo + hi) // 2
            if estimate_tokens(text[mid:end]) > chunk_overlap:
                lo = mid + 1
            else:
                hi = mid
        pos = max(lo, pos + 1)  # ensure forward progress

    return chunks


def _split_by_tokens_exact(
    tokenizer: Any, text: str, chunk_size: int, chunk_overlap: int,
) -> List[str]:
    """按真实 token 边界切分：一次编码拿 offsets，再按 token 下标取子串。

    片段孤立重编码时 BPE 在片段开头的合并规则与全文中不同，可能多出 1-2 个
    token，因此每片再校验一次，超预算就按溢出量回退——保证
    ``estimate_tokens(piece) <= chunk_size`` 恒成立。
    """
    enc = tokenizer.encode(text, add_special_tokens=False)
    total = len(enc.ids)
    if total <= chunk_size:
        return [text.strip()] if text.strip() else []
    offsets = enc.offsets
    chunks: List[str] = []
    start = 0
    while start < total:
        end = min(start + chunk_size, total)
        first = offsets[start][0]
        piece = text[first:offsets[end - 1][1]].strip()
        while piece and end > start + 1:
            over = estimate_tokens(piece) - chunk_size
            if over <= 0:
                break
            end = max(start + 1, end - over)
            piece = text[first:offsets[end - 1][1]].strip()
        if piece:
            chunks.append(piece)
        if end >= total:
            break
        # 重叠也按 token 计；零重叠直接推进到下一片起点
        start = max(end - chunk_overlap, start + 1) if chunk_overlap > 0 else end
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


def _locate_spans(
    chunks: List[str], source: Optional[str],
) -> List[Optional[tuple]]:
    """按文档顺序把每个 chunk 定位成原文的 ``(start, end)``；定位失败置 None。

    游标**单调前进**：切片本就按原文顺序产出，从上一片结尾继续找即可。绝不
    退回全文首现——书里重复的代码块/样板段落会让 ``str.find`` 命中更早的实例，
    于是「A + 真实间隔 + B」被拼成「A' + 别的间隔 + B」，中间的 A..B 真实内容
    就再也没有出现在任何切片里（实测 O'Reilly 合集丢 0.5% 正文）。
    """
    if not source:
        return [None] * len(chunks)
    spans: List[Optional[tuple]] = []
    cursor = 0
    for c in chunks:
        t = c.strip()
        p = source.find(t, cursor) if t else -1
        if p < 0:
            spans.append(None)
            continue
        spans.append((p, p + len(t)))
        cursor = p + len(t)
    return spans


def _sliced_overlap(
    source: Optional[str],
    prev_span: Optional[tuple],
    curr_span: Optional[tuple],
    prev: str,
    tail: str,
) -> Optional[str]:
    """用原文跨度拼「重叠头 + 当前片」，保证结果是原文连续子串。

    两片跨度必须单调（``curr.start >= prev.end``）且 ``tail`` 能在前片内定位，
    否则返回 None 交调用方走旧的 gap 拼接逻辑。
    """
    if not source or prev_span is None or curr_span is None:
        return None
    if curr_span[0] < prev_span[1]:
        return None
    rel = prev.rfind(tail)
    if rel < 0:
        return None
    start = prev_span[0] + rel
    if not (prev_span[0] <= start < curr_span[1]):
        return None
    return source[start:curr_span[1]]


def _tail_by_tokens(text: str, budget: int) -> str:
    """返回 ``text`` 末尾尽量长、但 token 数 ≈ ``budget`` 的后缀。

    整段不足 ``budget`` 时返回空串。已绑定分词器时一次编码 + offsets 直接取
    起点（重叠在热路径上，逐段二分编码会慢一个数量级）；否则二分字符位置
    （后缀越短 token 越少，单调），估算口径与切片一致。
    """
    if budget <= 0 or not text:
        return ""
    tok = _TOKENIZER
    if tok is not None:
        try:
            enc = tok.encode(text, add_special_tokens=False)
            total = len(enc.ids)
            if total <= budget:
                return ""
            return text[enc.offsets[total - budget][0]:]
        except Exception:  # pragma: no cover - 分词器异常退回下面的二分
            logger.warning("chunker: 分词器编码失败，退回启发式估算", exc_info=True)
    if estimate_tokens(text) <= budget:
        return ""
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        if estimate_tokens(text[mid:]) > budget:
            lo = mid + 1
        else:
            hi = mid
    return text[lo:]


def _merge_overlap(chunks: List[str], overlap: int, sep: str,
                   source: Optional[str] = None,
                   locator: Optional["_GapLocator"] = None,
                   max_tokens: Optional[int] = None) -> List[str]:
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

    spans = _locate_spans(chunks, source)
    merged: List[str] = [chunks[0]]
    for i in range(1, len(chunks)):
        # 取「原始前一段」的尾部，而不是 merged[-1] 的尾部。merged[-1]
        # 已经带上了一段重叠文本，若某段比 overlap 还短，向后回退会越界取
        # 到上一段的重叠部分，导致重叠逐段复合、chunk 无限膨胀。
        prev = chunks[i - 1]
        curr = chunks[i]

        # Take last N tokens from prev as overlap（token 口径与切片一致）。
        prev_end = _tail_by_tokens(prev, overlap)
        if not prev_end.strip():
            merged.append(curr)
            continue

        # 优先按原文跨度拼接：结果恒为原文连续子串，且不会因为「重复段落命中
        # 更早实例」把不相邻的两段粘在一起——那会让中间的真实内容从所有切片里
        # 消失（语义/混合路径的历史 bug，递归路径同样受影响）。
        candidate = _sliced_overlap(source, spans[i - 1], spans[i], prev, prev_end)
        if candidate is None:
            # 兜底：无法定位跨度时按真实间隔拼接（没有原文可查则退回 sep）。
            gap = (locator.gap(prev, curr, sep) if locator is not None
                   else _gap_between(source, prev, curr, fallback=sep))
            candidate = prev_end + gap + curr
        # 重叠是**追加**在正文之上的，正文若已贴近预算，叠加后就超上限——
        # 那会直接触发下游 _clamp_chunk_budget 硬切（把刚做好的句界切碎）。
        # 超上限时放弃这一片的重叠，保留完整正文。
        if max_tokens is not None and estimate_tokens(candidate) > max_tokens:
            merged.append(curr)
            continue
        merged.append(candidate)

    return merged