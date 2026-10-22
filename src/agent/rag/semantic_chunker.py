"""Semantic chunking via chonkie.

Splits text on semantic boundaries (sentence-level embeddings + similarity
threshold) instead of naive character/recursive splits.  Semantic chunks keep
topically-related sentences together, which materially improves retrieval
quality for e-books and long documents.

Embedding backends (in priority order):
  1. Explicitly injected ``embedding_function`` → wrapped via
     ``_ChonkieEmbeddingAdapter`` (tests / external callers).
  2. model2vec static embeddings (potion-base-32M) — fast path for chunking
     decisions, ~100x faster than BERT inference (#118).
  3. Shared nomic-embed-text singleton → adapter (fallback when model2vec is
     unavailable; slow but correct).

If none is available (or ``chonkie`` is not installed), :func:`semantic_chunk_text`
returns ``None`` so callers fall back to the existing recursive chunker.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# SemanticChunker splits sentences on ". "/"! "/"? "/newline.  Chinese text
# uses "。" without a trailing space, so add CJK-friendly delimiters to avoid
# treating an entire paragraph of Chinese prose as one "sentence".
_DELIM = [". ", "! ", "? ", "\n", "。", "！", "？", "；", "，"]

# chonkie is an optional dependency.  Resolve its BaseEmbeddings base class at
# import time if available; otherwise fall back to ``object`` so the module can
# still be imported (semantic_chunk_text() returns None in that case).
try:
    from chonkie.embeddings.base import BaseEmbeddings as _BaseEmbeddings
except ImportError:  # pragma: no cover - chonkie absent
    _BaseEmbeddings = object


def _is_chonkie_available() -> bool:
    try:
        import chonkie  # noqa: F401
        return True
    except ImportError:
        return False


def semantic_chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 50,
    min_chunk_size: int = 100,
    overlap_percent: float = 0.05,
    embedding_function=None,
) -> Optional[List[str]]:
    """Split ``text`` into semantic chunks using chonkie's SemanticChunker.

    Args:
        text: The text to split.
        chunk_size: Target maximum chunk size in **tokens** (1000).
        chunk_overlap: Overlap between consecutive chunks in **tokens**.
            Defaults to 5% of ``chunk_size`` (50).  When ``overlap_percent``
            is set, ``int(chunk_size * overlap_percent)`` overrides this.
        min_chunk_size: Target minimum chunk size in **tokens** (100). Chunks
            smaller than this are greedily merged into their neighbours.
        overlap_percent: Fraction of ``chunk_size`` used as overlap (0.05 =>
            50 tokens), preserving continuity across chunk boundaries.  Takes
            precedence over ``chunk_overlap``.
        embedding_function: A callable ``embed_documents(list[str]) ->
            list[list[float]]``. When omitted, the shared nomic-embed-text
            singleton is used.

    Returns:
        List of chunk strings, or ``None`` if chonkie is unavailable or no
        embedding function could be resolved (callers should fall back).
    """
    if not text or not text.strip():
        return []

    if not _is_chonkie_available():
        logger.debug("semantic chunking skipped: chonkie not installed")
        return None

    ef = None
    try:
        from chonkie import SemanticChunker

        if embedding_function is not None:
            # 显式注入（测试/外部调用）：走适配器，行为与旧版一致。
            embeddings = _ChonkieEmbeddingAdapter(embedding_function)
        else:
            # 生产路径：优先 model2vec 静态嵌入（切片决策不需要与检索向量
            # 同空间，静态嵌入的相对相似度足够；吞吐高约两个数量级）。
            # model2vec 缺失时回退共享 nomic 适配器（慢但正确），再不行
            # 返回 None 交给上层递归切片兜底。
            embeddings = _get_static_embeddings()
            if embeddings is None:
                ef = _shared_embedding_function()
                if ef is None:
                    logger.debug("semantic chunking skipped: no embedding backend available")
                    return None
                embeddings = _ChonkieEmbeddingAdapter(ef)

        # When chonkie is present, _BaseEmbeddings is the real base class and
        # the isinstance check in SemanticChunker.__init__ will pass.  When it
        # fell back to ``object``, chonkie can't be imported anyway (handled
        # above by _is_chonkie_available), so this branch is defensive only.
        chunker = SemanticChunker(
            embedding_model=embeddings,
            threshold=0.7,          # lower => larger, more coherent groups
            chunk_size=chunk_size,  # hard ceiling in tokens
            min_sentences_per_chunk=2,
            min_characters_per_sentence=20,
            delim=_DELIM,
        )
        chunks = chunker.chunk(text)
        result = [c.text.strip() for c in chunks if c.text.strip()]
        # 传入原文 text，合并/重叠时用「源码里真实的间隔」拼接，保证产出的
        # chunk 一定是原文的连续子串（见 _gap_between）。
        merged = _merge_to_target_size(result, chunk_size, min_chunk_size, source=text)
        return _apply_overlap(
            merged, chunk_size, chunk_overlap, overlap_percent, source=text,
        )
    except Exception as e:
        # Never let a chunking failure abort ingestion — the caller falls back
        # to the recursive chunker.
        logger.warning("semantic chunking failed (%s), falling back to recursive: %s",
                       type(e).__name__, e)
        # 嵌入失败大概率是模型加载/下载失败（离线、HF 不可达）。shared 单例
        # 是惰性加载的——失败发生在 ef() 调用内部，_shared_embedding_function
        # 的负缓存永远看不到。这里把失败时间戳记回去，让 60s TTL 真正节流，
        # 否则离线时每个超长 section 都各自重试一次注定失败的模型加载。
        _mark_shared_embed_failed(ef)
        return None


class _SourceLocator:
    """在原文中定位 chunk，缓存结果避免 O(N²) 全文档扫描。

    ``_gap_between`` 旧实现每次都用 ``source.find(left)`` 从头扫，而
    ``_apply_overlap`` 对**每个 chunk** 调一次 —— 总扫描量 ≈ Σ(chunk 偏移)
    ≈ 文档长度²/2（5MB / 3000 chunk ≈ 7.5GB 字符扫描）。

    这里加一个单调游标：merge / overlap 都按文档顺序处理 chunk，位置不会
    回退，因此从游标附近起找即可，总扫描量降到 O(N)。游标 miss 时再退回
    全量 find（chonkie 偶发重排 whitespace 的情况），结果进缓存复用。

    与主线 chunker._gap_between 一样带 ``_MAX_GAP`` 防护：left 可能在原文
    出现多次（重复段落/样板文本），命中了更早的错误实例时 right 会落在
    很远的位置——算出的 gap 会把不属于这两个 chunk 的文本拼进结果。间隔
    超限就换 left 的下一个出现位置重试。
    """

    __slots__ = ("source", "_next", "_cache")

    # gap 上限：超过即认为 left 命中了错误实例（正常相邻 chunk 的间隔远
    # 小于此；只有跨章节的误匹配才会超过）。
    _MAX_GAP = 256

    def __init__(self, source: Optional[str]):
        self.source = source
        self._next = 0
        self._cache: Dict[tuple, str] = {}

    def gap(self, left: str, right: str) -> str:
        if not self.source or not left or not right:
            return "\n\n"
        key = (left, right)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        # 游标附近起找（留一点回退余量，left 可能部分落在游标之前）
        start = max(0, self._next - len(left) - 64)
        i = self.source.find(left, start)
        if i < 0:
            i = self.source.find(left)
        result = "\n\n"
        while i >= 0:
            j = self.source.find(right, i + len(left))
            if j < 0:
                # right 在 left 之后完全不存在：后续出现位置也不可能更好
                break
            if j - (i + len(left)) <= self._MAX_GAP:
                result = self.source[i + len(left):j]
                self._next = j + len(right)
                break
            # 间隔异常大：left 命中了更早的错误实例，换下一个出现位置重试
            i = self.source.find(left, i + 1)
        self._cache[key] = result
        return result


def _gap_between(source: Optional[str], left: str, right: str) -> str:
    """Return the exact text that separates ``left`` and ``right`` in ``source``.

    Merging/overlapping chunks used to hardcode ``"\\n\\n"`` as the joiner.
    When the two chunks were adjacent in the source with different whitespace
    (a single ``\\n``, a space, or nothing at all), the joined chunk contained
    characters the source never had — so chunk text was not a substring of the
    document, and the model saw text that doesn't exist in the file.

    Falls back to ``"\\n\\n"`` when the pieces can't be located in ``source``
    (no source supplied, or the chunker re-wrote whitespace).
    """
    return _SourceLocator(source).gap(left, right)


def _merge_to_target_size(
    chunks: List[str],
    chunk_size: int,
    min_chunk_size: int,
    source: Optional[str] = None,
) -> List[str]:
    """Greedily merge adjacent chunks toward the [min, max] token target.

    SemanticChunker splits on *semantic* boundaries, so chunks can be smaller
    than the target when a document has many short, topically-distinct
    sentences.  We merge adjacent chunks (which are already topically
    contiguous, since semantic splitting only cuts at low-similarity points)
    until each reaches ``min_chunk_size`` tokens, without exceeding
    ``chunk_size``.
    """
    from .chunker import estimate_tokens

    if not chunks:
        return chunks

    locator = _SourceLocator(source)
    merged: List[str] = []
    current = chunks[0]

    for nxt in chunks[1:]:
        if estimate_tokens(current) < min_chunk_size and \
                estimate_tokens(current) + estimate_tokens(nxt) <= chunk_size:
            current = current + locator.gap(current, nxt) + nxt
        else:
            merged.append(current)
            current = nxt
    merged.append(current)
    return merged


def _apply_overlap(
    chunks: List[str],
    chunk_size: int,
    chunk_overlap: int,
    overlap_percent: float,
    source: Optional[str] = None,
) -> List[str]:
    """Add token-level overlap between consecutive chunks.

    SemanticChunker produces non-overlapping chunks, so a sentence split at a
    chunk boundary loses its surrounding context.  We prepend to each chunk a
    tail slice of the *previous* chunk — chosen at a sentence boundary so we
    never split a sentence mid-way — so that retrieval across a boundary still
    sees the surrounding sentences.

    The overlap size is ``int(chunk_size * overlap_percent)`` when
    ``overlap_percent`` is set, otherwise the explicit ``chunk_overlap``.
    """
    from .chunker import estimate_tokens

    if len(chunks) <= 1:
        return chunks

    overlap = int(chunk_size * overlap_percent) if overlap_percent else chunk_overlap
    if overlap <= 0:
        return chunks

    locator = _SourceLocator(source)
    result: List[str] = [chunks[0]]
    for i in range(1, len(chunks)):
        # Overlap is always derived from the *original* previous chunk (not the
        # overlap-padded one), so overlap never compounds across boundaries.
        tail = _tail_by_tokens(chunks[i - 1], overlap)
        if tail:
            # 用源码里真实的间隔拼接，保证结果仍是原文的连续子串。
            result.append(tail + locator.gap(tail, chunks[i]) + chunks[i])
        else:
            result.append(chunks[i])
    return result


_TAIL_BOUNDARY_RE = re.compile(r"(?<=[。！？!?\.])[ \t]*\n?|\n")


def _tail_by_tokens(text: str, max_tokens: int) -> str:
    """Return the trailing portion of ``text`` up to ``max_tokens`` tokens,
    aligned to a sentence boundary (never splitting a sentence).

    切片必须取自原文 —— 旧实现先用 ``re.split`` 拆句（分隔符中的 ``\\s*``
    被丢弃）再用 ``"".join`` 拼回来，句间的换行/空格因此凭空消失，产出的
    重叠文本在原文里根本不存在（模型会看到源文件没有的内容）。这里改为直接
    在原文里找句末边界并切片，保证结果恒为原文子串。
    """
    from .chunker import estimate_tokens

    if not text or max_tokens <= 0:
        return ""

    # 所有可作为起点的句末边界（0 = 整段文本）。
    bounds = [0] + [m.end() for m in _TAIL_BOUNDARY_RE.finditer(text)]
    bounds = [b for b in bounds if b < len(text)]

    # 从最靠近末尾的边界往回找，取能满足 token 预算的最长后缀。
    start = len(text)
    for b in reversed(bounds):
        if estimate_tokens(text[b:]) > max_tokens:
            break
        start = b
    return text[start:].strip()


# ---------------------------------------------------------------------------
# embedding singleton (shared with vector store)
# ---------------------------------------------------------------------------

_shared_ef = None
_shared_ef_resolved = False
_shared_ef_failed_at = 0.0
# wrapper 已就绪但**真实 embed 调用**失败的时间戳：模型是惰性加载的，
# 加载/下载失败发生在 ef() 调用内部而不是 get_or_create_embedding_function()
# ——旧实现的负缓存 TTL 只覆盖后者，离线时每个超长 section 仍会各自重试
# 一次注定失败的加载。真实失败在这里记时间戳，由同一 TTL 节流。
_shared_ef_load_failed_at = 0.0
# 失败负缓存 TTL：模型缺失/离线时加载会抛异常（甚至尝试下载 137MB），
# 而语义切片路径对**每个超长 section** 都会调一次。无 TTL 会让每个 section
# 都重新触发一次注定失败的加载，离线环境下反复卡住。
_EF_NEGATIVE_TTL = 60.0


_shared_ef_lock = threading.Lock()


def _mark_shared_embed_failed(ef) -> None:
    """记录 shared 单例的真实 embed 失败（TTL 内后续调用直接跳过）。"""
    global _shared_ef_load_failed_at
    if ef is not None and ef is _shared_ef:
        with _shared_ef_lock:
            _shared_ef_load_failed_at = time.monotonic()


def _shared_embedding_function():
    """Return the process-wide nomic-embed-text embedding function, if available.

    Imported lazily to avoid a hard dependency on fastembed/vector_store for
    callers that never enable RAG.

    失败结果**不永久缓存**（旧实现先置位再加载，首次失败会让本进程语义切片
    永久降级，即使后来网络恢复也不重试），但也不是每次都重试 —— 用 60s TTL
    的负缓存：既避免每个 section 反复触发注定失败的模型加载，又保证网络/
    模型恢复后一分钟内自动恢复语义切片。加载过程加锁，防多 worker 并发加载。
    """
    global _shared_ef, _shared_ef_resolved, _shared_ef_failed_at
    if _shared_ef_resolved and _shared_ef is not None:
        # wrapper 已就绪，但最近一次真实 embed 失败 → TTL 内跳过语义切片
        # （模型加载/下载失败记在 _shared_ef_load_failed_at，见其定义处注释）。
        if _shared_ef_load_failed_at and (
            time.monotonic() - _shared_ef_load_failed_at < _EF_NEGATIVE_TTL
        ):
            return None
        return _shared_ef
    if _shared_ef is None and _shared_ef_failed_at and (
        time.monotonic() - _shared_ef_failed_at < _EF_NEGATIVE_TTL
    ):
        return None
    with _shared_ef_lock:
        if _shared_ef_resolved and _shared_ef is not None:
            if _shared_ef_load_failed_at and (
                time.monotonic() - _shared_ef_load_failed_at < _EF_NEGATIVE_TTL
            ):
                return None
            return _shared_ef
        if _shared_ef is None and _shared_ef_failed_at and (
            time.monotonic() - _shared_ef_failed_at < _EF_NEGATIVE_TTL
        ):
            return None
        try:
            from .vector_store import get_or_create_embedding_function
            _shared_ef = get_or_create_embedding_function()
            # 只在成功时置位，失败时留给下次重试（受 TTL 节流）。
            _shared_ef_resolved = _shared_ef is not None
            if _shared_ef_resolved:
                _shared_ef_failed_at = 0.0
            else:
                _shared_ef_failed_at = time.monotonic()
        except Exception as e:
            logger.debug("shared embedding function unavailable: %s", e)
            _shared_ef = None
            _shared_ef_failed_at = time.monotonic()
    return _shared_ef


# ---------------------------------------------------------------------------
# static embeddings for chunking decisions (potion via model2vec)
# ---------------------------------------------------------------------------

# 语义切片的「快路径」嵌入后端：model2vec 静态嵌入（potion-base-32M，
# ~30MB，纯 numpy 查表 + 平均池化，无 BERT 前向）。它只服务「相邻窗口是否
# 语义断裂」的切片决策——决策只需要相邻相似度的相对大小，不需要与检索
# 向量同空间，静态嵌入完全够用。实测吞吐比共享 nomic 适配器（BERT 推理
# + 2N 次 embed）高约两个数量级（88k 字符章节 90-150s → 秒级，#118）。
#
# chonkie 自带的 AutoEmbeddings 会把 minishlab/* 路由到 Model2VecEmbeddings
# （chonkie.embeddings.registry），model2vec 未安装时其构造抛 ImportError。
_STATIC_EMBS = None
_STATIC_EMBS_RESOLVED = False
_STATIC_EMBS_FAILED_AT: Optional[float] = None
_STATIC_EMBS_RETRY_INTERVAL = 60.0  # 失败负缓存：60s 内不重复尝试
_STATIC_EMBS_LOCK = threading.Lock()

_POTION_MODEL = "minishlab/potion-base-32M"


def _get_static_embeddings():
    """Return the model2vec static embeddings for chunking, or None.

    失败负缓存与 ``_shared_embedding_function`` 同款：deps 在同步前自动
    安装 model2vec，安装成功后的下一次尝试（60s 负缓存过期后）自动升级
    到快路径，无需重启。首次使用会下载 potion 权重（~30MB，HF 缓存）。
    """
    global _STATIC_EMBS, _STATIC_EMBS_RESOLVED, _STATIC_EMBS_FAILED_AT
    if _STATIC_EMBS_RESOLVED:
        return _STATIC_EMBS
    if (
        _STATIC_EMBS_FAILED_AT is not None
        and time.monotonic() - _STATIC_EMBS_FAILED_AT < _STATIC_EMBS_RETRY_INTERVAL
    ):
        return None
    with _STATIC_EMBS_LOCK:
        if _STATIC_EMBS_RESOLVED:
            return _STATIC_EMBS
        try:
            from chonkie.embeddings import AutoEmbeddings

            _STATIC_EMBS = AutoEmbeddings.get_embeddings(_POTION_MODEL)
            _STATIC_EMBS_RESOLVED = True
            _STATIC_EMBS_FAILED_AT = None
            logger.info("语义切片使用静态嵌入 %s（model2vec 快路径）", _POTION_MODEL)
        except Exception as e:
            logger.debug("model2vec 静态嵌入不可用（%s），回退 nomic 适配器", e)
            _STATIC_EMBS = None
            _STATIC_EMBS_FAILED_AT = time.monotonic()
    return _STATIC_EMBS


# ---------------------------------------------------------------------------
# chonkie embedding adapter
# ---------------------------------------------------------------------------

# 已知嵌入维度（nomic-embed-text-v1.5 = 768）。chonkie 可能在构造期读
# ``dimension``，用真实 embed 探测会把 ONNX 模型加载提前到 ingest worker
# 线程里；先用已知值占位，首批 embed 后再用实测长度刷新。
_KNOWN_DIM = 768


class _ChonkieEmbeddingAdapter(_BaseEmbeddings):
    """Adapt a FastEmbed embedding function to chonkie's BaseEmbeddings interface.

    chonkie's SemanticChunker accepts either a model-name string or a
    ``BaseEmbeddings`` instance.  To keep the exact nomic-embed-text ONNX model
    shared with the vector store (instead of loading potion-base-32M or a
    second nomic copy), we subclass BaseEmbeddings and wrap the existing
    embedding function.  Required members:

      * ``embed(text)`` / ``embed_batch(texts)`` — sentence embeddings
      * ``similarity(u, v)`` — cosine similarity
      * ``get_tokenizer()`` — returns our ``estimate_tokens`` callable (chonkie
        wraps it via CallableAutoTokenizer for ``count_tokens``)
      * ``dimension`` — embedding vector dimension
    """

    def __init__(self, ef):
        super().__init__()
        self._ef = ef
        # Resolve dimension lazily (requires one embedding call) to avoid
        # forcing model load at construction time.
        self._dim = None

    # -- embedding ---------------------------------------------------------
    def _embed_many(self, texts: List[str]):
        """Embed a list of texts, tolerating either ``embed_documents`` or the
        callable ``__call__`` protocol (the test fake exposes ``__call__``)."""
        if hasattr(self._ef, "embed_documents"):
            return self._ef.embed_documents(texts)
        return self._ef(texts)

    def embed(self, text: str):
        return self._to_array(self._embed_many([text])[0])

    def embed_batch(self, texts: List[str]):
        vecs = [self._to_array(v) for v in self._embed_many(list(texts))]
        # 顺手把真实维度记下来：之后的 dimension 就用实测值，不再依赖占位。
        if vecs and self._dim is None:
            try:
                self._dim = int(vecs[0].shape[-1])
            except Exception:
                pass
        return vecs

    @staticmethod
    def _to_array(vec):
        import numpy as np
        return np.asarray(vec, dtype="float32")

    # -- similarity --------------------------------------------------------
    def similarity(self, u, v) -> float:
        import numpy as np
        u = np.asarray(u, dtype="float32")
        v = np.asarray(v, dtype="float32")
        denom = float(np.linalg.norm(u) * np.linalg.norm(v))
        if denom == 0.0:
            return 0.0
        return float(np.dot(u, v) / denom)

    # -- dimension ---------------------------------------------------------
    @property
    def dimension(self) -> int:
        """向量维度。

        旧实现用一次真实 embed 探测（``len(self.embed("probe"))``）。chonkie
        可能在**构造期**就访问这个属性，于是语义分块的初始化会在 ingest worker
        线程里强制加载 ONNX 模型 —— 而模型加载受全局锁串行化，第一个 worker
        会把整体启动时间拖长，4 个 worker 之间还会互相干等。

        改为惰性：已经算过向量就用它的长度；否则退回已知维度（本项目默认的
        nomic-embed-text 就是 768）。真正开始分批 embed 后 ``_dim`` 会被真实
        长度刷新（见 ``embed_batch``）。
        """
        if self._dim is not None:
            return self._dim
        return _KNOWN_DIM

    # -- tokenizer ---------------------------------------------------------
    def get_tokenizer(self) -> Any:
        """Return a token counter callable.

        chonkie wraps callables via ``CallableAutoTokenizer`` whose
        ``count_tokens(text)`` calls it directly.  We use the same heuristic
        as the recursive chunker (``estimate_tokens``) so chunk sizes are
        measured consistently across both paths.
        """
        from .chunker import estimate_tokens
        return estimate_tokens
