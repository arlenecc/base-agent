"""长文档 ColBERT 索引与检索（LanceDB 三表 + 残差量化 + MaxSim）。

表结构：
* ``document``         doc_id / title / content（另加内部 content_fts 供 BM25）
* ``doc_summary_vec``  doc_id / vector(float16[dim])：文档 token 中心向量
* ``doc_residuals``    doc_id / token_count / dim / quant_bits / scale /
                       offset / residual_blob（4bit：2 值/字节）

检索：
  BM25(content_fts) + 中心向量 LanceDB 向量检索 → RRF 融合取 top20 →
  流式解压残差、反量化还原 token 向量 → 与 query token 向量做 MaxSim → topN。
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa

from .colbert_runtime import ColBertRuntime, get_runtime

logger = logging.getLogger(__name__)
from .doc_chunker import (
    DEFAULT_MAX_CHARS, DEFAULT_MIN_CHARS, DEFAULT_OVERLAP_RATIO,
    DEFAULT_SEMANTIC_THRESHOLD, prune_tokens, chunk_document,
)

_INDEX_SUBDIR = ".agent/colbert_docs.lancedb"
_RRF_K = 60.0
_BATCH_TOKENS = 2048
_MAX_COARSE = 100
_MAX_TOP_K = 30


class ColBertStoreError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# 残差量化 / 反量化
# ---------------------------------------------------------------------------


def quantize_vectors(vectors: np.ndarray, bits: int = 4
                     ) -> Tuple[np.ndarray, bytes, float, float]:
    """fp32 token 向量 → fp16 中心 + 量化残差 blob。

    返回 ``(center_fp16, blob, scale, offset)``。量化公式：
    ``q = round((residual - offset) / scale)``，``scale = 2*maxAbs/(2^bits-1)``，
    ``offset = -maxAbs``；q 为无符号 0..2^bits-1。
    """
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ColBertStoreError("empty token vectors")
    if bits not in (2, 4):
        raise ColBertStoreError("quant_bits must be 2 or 4")
    vec16 = arr.astype(np.float16)
    center = vec16.astype(np.float32).mean(axis=0).astype(np.float16)
    residual = (vec16.astype(np.float32)
                - center.astype(np.float32)).astype(np.float16)
    residual = residual.astype(np.float32)
    max_abs = float(np.max(np.abs(residual)))
    levels = (1 << bits) - 1
    if max_abs <= 1e-12:
        scale, offset = 1.0, 0.0
        q = np.zeros(residual.shape, dtype=np.uint8)
    else:
        scale = (2.0 * max_abs) / levels
        offset = -max_abs
        q = np.clip(np.rint((residual - offset) / scale),
                    0, levels).astype(np.uint8)
    blob = _pack_quantized(q, bits)
    return center, blob, float(scale), float(offset)


def _pack_quantized(q: np.ndarray, bits: int) -> bytes:
    flat = np.asarray(q, dtype=np.uint8).ravel()
    if bits == 4:
        if flat.size % 2:
            flat = np.append(flat, 0)
        packed = flat[0::2] | (flat[1::2] << 4)
    else:
        pad = (-flat.size) % 4
        if pad:
            flat = np.pad(flat, (0, pad))
        packed = (flat[0::4] | (flat[1::4] << 2)
                  | (flat[2::4] << 4) | (flat[3::4] << 6))
    return packed.astype(np.uint8).tobytes()


def _unpack_quantized(blob: bytes, count: int, bits: int) -> np.ndarray:
    raw = np.frombuffer(blob, dtype=np.uint8)
    if bits == 4:
        out = np.empty(raw.size * 2, dtype=np.uint8)
        out[0::2] = raw & 0x0F
        out[1::2] = (raw >> 4) & 0x0F
    else:
        out = np.empty(raw.size * 4, dtype=np.uint8)
        out[0::4] = raw & 0x03
        out[1::4] = (raw >> 2) & 0x03
        out[2::4] = (raw >> 4) & 0x03
        out[3::4] = (raw >> 6) & 0x03
    return out[:count]


def iter_residual_vectors(blob: bytes, token_count: int, dim: int, bits: int,
                          scale: float, offset: float,
                          center: np.ndarray,
                          batch_tokens: int = _BATCH_TOKENS
                          ) -> Iterator[np.ndarray]:
    """流式反量化：按 token 批次产出 ``(batch, dim)`` 的 fp32 向量。"""
    if token_count <= 0 or dim <= 0:
        return
    bytes_per_token = (dim * bits + 7) // 8
    aligned = (dim * bits) % 8 == 0
    if not aligned:
        # 极端维度不按字节对齐：整块解一次，再按 token 分批返回。
        q_all = _unpack_quantized(blob, token_count * dim, bits)
        res_all = q_all.astype(np.float32) * scale + offset
        res_all = (res_all.reshape(token_count, dim)
                   + center.astype(np.float32))
        for start in range(0, token_count, batch_tokens):
            yield res_all[start:start + batch_tokens]
        return
    center32 = center.astype(np.float32)
    for start in range(0, token_count, batch_tokens):
        count = min(batch_tokens, token_count - start)
        raw = blob[start * bytes_per_token:(start + count) * bytes_per_token]
        q = _unpack_quantized(raw, count * dim, bits)
        vecs = q.astype(np.float32) * scale + offset
        yield vecs.reshape(count, dim) + center32


def maxsim_stream(query_vectors: np.ndarray, blob: bytes, token_count: int,
                  dim: int, bits: int, scale: float, offset: float,
                  center: np.ndarray,
                  batch_tokens: int = _BATCH_TOKENS) -> float:
    """边解压边算 MaxSim，避免一次性还原整篇 token 向量。"""
    q = np.asarray(query_vectors, dtype=np.float32)
    if q.ndim != 2 or q.size == 0 or token_count <= 0:
        return float("-inf")
    best = np.full(q.shape[0], -np.inf, dtype=np.float32)
    for vecs in iter_residual_vectors(
            blob, token_count, dim, bits, scale, offset, center,
            batch_tokens=batch_tokens):
        if vecs.size == 0:
            continue
        sims = vecs @ q.T  # BLAS matmul
        best = np.maximum(best, sims.max(axis=0))
    if not np.isfinite(best).any():
        return float("-inf")
    return float(best.sum())


# ---------------------------------------------------------------------------
# RRF
# ---------------------------------------------------------------------------


def rrf_fuse(rankings: Sequence[Sequence[str]], k: float = _RRF_K,
             limit: int = 20) -> List[Tuple[str, float]]:
    """Reciprocal Rank Fusion：score = Σ 1/(k + rank)。"""
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            if not doc_id:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1.0)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]


# ---------------------------------------------------------------------------
# FTS 预分词
# ---------------------------------------------------------------------------

_jieba = None
_jieba_lock = threading.Lock()
_jieba_tried = False


def _get_jieba():
    global _jieba, _jieba_tried
    if _jieba_tried:
        return _jieba
    with _jieba_lock:
        if _jieba_tried:
            return _jieba
        _jieba_tried = True
        try:
            import jieba
            _jieba = jieba
        except ImportError:
            _jieba = None
    return _jieba


def fts_text(text: str) -> str:
    jieba = _get_jieba()
    if jieba is not None and text:
        return " ".join(t.strip() for t in jieba.cut(text) if t.strip())
    # 无 jieba：CJK 双字 + 英文/数字，保证 simple tokenizer 能命中。
    tokens: List[str] = []
    for m in re.finditer(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", text or ""):
        s = m.group(0)
        if re.match(r"[A-Za-z0-9_]", s):
            tokens.append(s.lower())
        else:
            tokens.extend(s[i:i + 2] for i in range(max(1, len(s) - 1)))
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

_DOC_SCHEMA = pa.schema([
    ("doc_id", pa.string()),
    # title 存文档名（同一文档的所有切片相同）；切片内的章节标题仍在 content
    # 首行，检索结果原样返回，模型/气泡都能看到。
    ("title", pa.string()),
    ("content", pa.string()),
    # 内部列：jieba 预分词文本，供 LanceDB simple FTS 做中文 BM25。
    ("content_fts", pa.string()),
    # 内部列：原始文件路径与切片序号，供检索即展示的前/后片翻页。
    ("source", pa.string()),
    ("chunk_index", pa.int32()),
])


def _vec_schema(dim: int) -> pa.schema:
    return pa.schema([
        ("doc_id", pa.string()),
        ("vector", pa.list_(pa.float16(), dim)),
    ])


_RES_SCHEMA = pa.schema([
    ("doc_id", pa.string()),
    ("token_count", pa.int32()),
    ("dim", pa.int32()),
    ("quant_bits", pa.int32()),
    ("scale", pa.float32()),
    ("offset", pa.float32()),
    ("residual_blob", pa.binary()),
])


class ColBertDocStore:
    def __init__(self, db_dir: Optional[str] = None,
                 workspace: Optional[str] = None,
                 runtime: Optional[ColBertRuntime] = None,
                 dim: Optional[int] = None,
                 model_dir: str = ""):
        if db_dir is None:
            base = workspace or os.path.expanduser("~/base-agent-workspace")
            db_dir = os.path.join(base, _INDEX_SUBDIR)
        self.db_dir = os.path.abspath(db_dir)
        os.makedirs(self.db_dir, exist_ok=True)
        self._runtime = runtime
        self._model_dir = (os.path.abspath(os.path.expanduser(model_dir))
                           if model_dir else "")
        self.dim = int(dim or (runtime.dim if runtime is not None else 64))
        self._write_lock = threading.RLock()
        self._fts_ready = False
        self._ann_ready = False
        # 语义切片编码器：默认 model2vec 静态句向量（测试可注入 Fake）。
        self._chunk_encoder = None
        try:
            import lancedb
        except ImportError as e:
            raise ColBertStoreError(f"lancedb 未安装：{e}") from e
        self._db = lancedb.connect(self.db_dir)
        self._doc = self._open("document", _DOC_SCHEMA)
        self._vec = self._open("doc_summary_vec", _vec_schema(self.dim))
        self._res = self._open("doc_residuals", _RES_SCHEMA)

    def _open(self, name: str, schema: pa.Schema):
        try:
            table = self._db.open_table(name)
        except Exception:
            return self._db.create_table(name, schema=schema)
        # 旧版本表可能缺少 source/chunk_index/content_fts；补空列比整表重建
        # 安全（已入库的向量/残差全部保留），旧行由查询侧回退解析。
        try:
            existing = {f.name for f in table.schema}
            missing = [f for f in schema if f.name not in existing]
            if missing:
                table.add_columns(missing)
                logger.info("ColBERT 表 %s 已补充列: %s", name,
                            ", ".join(f.name for f in missing))
        except Exception as e:
            logger.warning("ColBERT 表 %s 列迁移失败（继续使用）: %s", name, e)
        return table

    def set_model_dir(self, model_dir: str) -> None:
        """更新模型目录；模型尚未加载时生效（应用配置后无需重建 store）。"""
        if self._runtime is None:
            self._model_dir = (os.path.abspath(os.path.expanduser(model_dir))
                               if model_dir else "")

    @property
    def runtime(self) -> ColBertRuntime:
        # list/stats 不需要加载 587MB 模型；首次真正编码时才惰性加载。
        if self._runtime is None:
            self._runtime = get_runtime(self._model_dir or None)
            self.dim = int(self._runtime.dim)
        return self._runtime

    # ------------------------------------------------------------------ index
    def _delete_prefix(self, prefix: str) -> None:
        expr = f"doc_id LIKE '{prefix}%'"
        for table in (self._doc, self._vec, self._res):
            try:
                table.delete(expr)
            except Exception:
                pass

    def index_text(self, title: str, text: str, source: str = "",
                   overwrite: bool = True,
                   max_chars: int = DEFAULT_MAX_CHARS,
                   min_chars: int = DEFAULT_MIN_CHARS,
                   overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
                   semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
                   quant_bits: int = 4) -> int:
        """切分 + 编码 + 量化 + 写入三表；返回切片数。"""
        if not (text or "").strip():
            raise ColBertStoreError("document text is empty")
        title = (title or "Untitled").strip()
        # 语义低谷用 model2vec 静态句向量，不用 ColBERT；ColBERT 只在
        # add_chunks() 里计算入库词向量。model2vec 不可用时 chunk_document
        # 自动退回词面相似度。
        encoder = self._chunk_encoder
        if encoder is None:
            from .semantic_chunker import get_chunking_encoder
            encoder = get_chunking_encoder()
        chunks = chunk_document(
            text, title=title, encoder=encoder,
            max_chars=max_chars, min_chars=min_chars,
            overlap_ratio=overlap_ratio,
            semantic_threshold=semantic_threshold)
        if not chunks:
            raise ColBertStoreError("no chunks produced")
        return self.add_chunks(
            title, source or title, [c.content for c in chunks],
            quant_bits=quant_bits, overwrite=overwrite)

    def add_chunks(self, title: str, source: str, texts: Sequence[str],
                   quant_bits: int = 4, overwrite: bool = True) -> int:
        """对**已切好的**切片做剪枝/编码/量化并写入三表。

        供 RAG 检索阅读模式复用：引擎已完成 Markdown 切片，这里不再二次
        切分，直接按传入顺序生成 ``chunk_index``。返回实际写入切片数。
        """
        clean_texts = [str(t or "").strip() for t in texts]
        clean_texts = [t for t in clean_texts if t]
        if not clean_texts:
            raise ColBertStoreError("no non-empty chunks")
        title = (title or "Untitled").strip()
        source = source or title
        prefix = hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]
        doc_rows: List[Dict[str, Any]] = []
        vec_rows: List[Dict[str, Any]] = []
        res_rows: List[Dict[str, Any]] = []
        for idx, body in enumerate(clean_texts):
            is_code = "```" in body
            pruned = prune_tokens(body, is_code=is_code) or body
            try:
                vectors = self.runtime.encode_document(pruned)
            except Exception as e:
                raise ColBertStoreError(
                    f"ColBERT 编码失败（chunk {idx}）：{e}") from e
            if vectors.ndim != 2 or vectors.shape[1] != self.dim:
                raise ColBertStoreError(
                    f"ColBERT 输出维度异常：{getattr(vectors, 'shape', None)}，"
                    f"期望 dim={self.dim}")
            center, blob, scale, offset = quantize_vectors(vectors, quant_bits)
            doc_id = f"{prefix}:{idx:05d}"
            doc_rows.append({
                "doc_id": doc_id,
                "title": title,
                "content": body,
                "content_fts": fts_text(body),
                "source": source,
                "chunk_index": idx,
            })
            vec_rows.append({
                "doc_id": doc_id,
                "vector": center.astype(np.float16).tolist(),
            })
            res_rows.append({
                "doc_id": doc_id,
                "token_count": int(vectors.shape[0]),
                "dim": int(self.dim),
                "quant_bits": int(quant_bits),
                "scale": float(scale),
                "offset": float(offset),
                "residual_blob": blob,
            })
        with self._write_lock:
            if overwrite:
                self._delete_prefix(prefix)
            self._doc.add(doc_rows)
            self._vec.add(vec_rows)
            self._res.add(res_rows)
            self._fts_ready = False
            self._ensure_fts()
        return len(doc_rows)

    def index_path(self, path: str, overwrite: bool = True,
                   quant_bits: int = 4) -> Dict[str, Any]:
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.exists(path):
            raise ColBertStoreError(f"path not found: {path}")
        exts = (".md", ".markdown", ".txt", ".rst", ".text")
        files: List[str] = []
        if os.path.isdir(path):
            for root, _dirs, names in os.walk(path):
                for name in sorted(names):
                    if name.lower().endswith(exts):
                        files.append(os.path.join(root, name))
        elif path.lower().endswith(exts):
            files = [path]
        else:
            raise ColBertStoreError(
                "only .md/.markdown/.txt/.rst files are supported")
        if not files:
            raise ColBertStoreError(f"no supported documents under {path}")
        indexed = 0
        chunks = 0
        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
            except OSError as e:
                continue
            title = os.path.splitext(os.path.basename(file_path))[0]
            n = self.index_text(title, text, source=file_path,
                                overwrite=overwrite, quant_bits=quant_bits)
            indexed += 1
            chunks += n
        return {"files": indexed, "chunks": chunks}

    # ----------------------------------------------------------------- search
    def _ensure_fts(self) -> bool:
        if self._fts_ready:
            return True
        try:
            self._doc.create_fts_index(
                "content_fts", use_tantivy=False, replace=True)
            self._fts_ready = True
            return True
        except Exception:
            self._fts_ready = False
            return False

    def _fetch(self, table, ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        if not ids:
            return {}
        quoted = ", ".join("'" + str(i).replace("'", "''") + "'" for i in ids)
        try:
            rows = table.search().where(f"doc_id IN ({quoted})").limit(
                len(ids) + 1).to_list()
        except Exception:
            rows = []
        return {row["doc_id"]: row for row in rows}

    def _source_ids(self, source: str) -> Optional[set]:
        """按来源路径/标题子串预取允许的 doc_id 集合（None = 不限制）。"""
        source = (source or "").strip()
        if not source:
            return None
        escaped = source.replace("'", "''")
        try:
            rows = self._doc.search().where(
                f"source LIKE '%{escaped}%'").limit(5000).to_list()
        except Exception:
            rows = []
        return {r.get("doc_id") for r in rows if r.get("doc_id")}

    def _vector_search(self, q_center: np.ndarray, limit: int,
                       use_ann: bool) -> List[Dict[str, Any]]:
        q = self._vec.search(q_center.astype(np.float32)).metric("cosine")
        if use_ann:
            try:
                return q.nprobes(20).limit(limit).to_list()
            except Exception as e:
                logger.info("ColBERT ANN 查询失败（回退精确扫描）: %s", e)
        try:
            q = q.bypass_vector_index()
        except Exception:
            pass
        return q.limit(limit).to_list()

    def search(self, query: str, top_k: int = 5, coarse_k: int = 20,
               include_content: bool = True,
               max_content_chars: int = 1500,
               source: str = "", use_ann: bool = False,
               min_score: float = 0.0
               ) -> List[Dict[str, Any]]:
        """粗筛 + MaxSim 精排（BM25 + 中心向量并集 → MaxSim）。

        ``min_score`` > 0 时按「MaxSim ÷ 查询 token 数」（每个查询词的平均
        最优相似度）过滤：低于该值的切片视为不相关，不返回。该阈值与查询
        长度无关，可直接跨查询比较；0 表示不过滤。
        """
        query = (query or "").strip()
        if not query:
            raise ColBertStoreError("query is empty")
        top_k = max(1, min(int(top_k), _MAX_TOP_K))
        coarse_k = max(top_k, min(int(coarse_k), _MAX_COARSE))
        q_vectors = self.runtime.encode_query(query)
        if q_vectors.ndim != 2 or q_vectors.size == 0:
            return []
        q_token_count = max(1, int(q_vectors.shape[0]))
        q_center = q_vectors.astype(np.float32).mean(axis=0)
        norm = float(np.linalg.norm(q_center))
        q_center = q_center / max(norm, 1e-12)

        allowed = self._source_ids(source)
        if allowed is not None and not allowed:
            return []
        # source 过滤在查询层完成：BM25 直接 where source；向量表没有 source
        # 列，就对允许的 doc_id 精确点积（绕过 ANN，避免全局 top-k 把目标
        # 文档挤掉）。无 source 时才走 LanceDB 的 ANN/精确扫描。
        # 两条通道各取 top coarse_k 即可：后续精排候选是两通道的并集
        # （≤2×coarse_k），再按 len(allowed) 放大只会让并集无上界。
        fetch_k = coarse_k
        bm25_ids: List[str] = []
        if self._ensure_fts():
            try:
                q_fts = self._doc.search(
                    fts_text(query), query_type="fts")
                if source:
                    escaped = source.replace("'", "''")
                    q_fts = q_fts.where(f"source LIKE '%{escaped}%'")
                rows = q_fts.limit(fetch_k).to_list()
                bm25_ids = [r["doc_id"] for r in rows]
            except Exception:
                bm25_ids = []
        vec_ids: List[str] = []
        try:
            if allowed is not None:
                scored: List[Tuple[float, str]] = []
                for doc_id, row in self._fetch(
                        self._vec, sorted(allowed)).items():
                    vec = np.asarray(row.get("vector"), dtype=np.float32)
                    if vec.shape == q_center.shape:
                        # q_center 已 L2 归一化，中心向量必须同样归一化：
                        # 存的是「token 向量均值」而非单位向量，直接点积
                        # 等于把模长更大的切片整体抬高（定向检索时实测会
                        # 把正确切片从第 1 挤到第 5），必须按余弦比较。
                        vnorm = float(np.linalg.norm(vec))
                        scored.append((float(np.dot(
                            q_center, vec / max(vnorm, 1e-12))), doc_id))
                scored.sort(reverse=True)
                vec_ids = [doc_id for _, doc_id in scored[:fetch_k]]
            else:
                rows = self._vector_search(q_center, fetch_k, use_ann)
                vec_ids = [r["doc_id"] for r in rows]
        except Exception as e:
            logger.warning("ColBERT 中心向量检索失败: %s", e)
            vec_ids = []

        # 精排候选 = 两条通道各自 top coarse_k 的**并集**，而不是 RRF 融合后
        # 截断到 coarse_k：RRF 截断会把只被单通道命中的头部结果挤掉（被
        # 「双通道都命中但排名靠后」的候选取代）。实测 126 切片、16 条查询，
        # 精排最优切片能被粗筛召回的比例：RRF 截断 12/16，并集 14/16；而
        # MaxSim 每候选仅约 0.2ms，扩大候选几乎不增加检索延迟。
        rrf_scores = dict(rrf_fuse([bm25_ids, vec_ids], limit=len(bm25_ids)
                                   + len(vec_ids)))
        ids: List[str] = []
        seen_ids: set = set()
        for doc_id in bm25_ids + vec_ids:
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                ids.append(doc_id)
        if not ids:
            return []
        docs = self._fetch(self._doc, ids)
        vecs = self._fetch(self._vec, ids)
        residuals = self._fetch(self._res, ids)

        hits: List[Dict[str, Any]] = []
        for doc_id in ids:
            rrf_score = rrf_scores.get(doc_id, 0.0)
            doc = docs.get(doc_id)
            center_row = vecs.get(doc_id)
            res = residuals.get(doc_id)
            if not doc or not center_row or not res:
                continue
            center = np.asarray(center_row["vector"], dtype=np.float32)
            try:
                score = maxsim_stream(
                    q_vectors,
                    res["residual_blob"], int(res["token_count"]),
                    int(res["dim"]), int(res["quant_bits"]),
                    float(res["scale"]), float(res["offset"]), center)
            except Exception:
                continue
            per_token = float(score) / q_token_count
            # 相关度门限：MaxSim 是查询词数决定的绝对分，必须按 token 数
            # 归一化后才能跨查询比较（长问句的绝对分天然更高）。
            if min_score > 0.0 and per_token < float(min_score):
                continue
            content = doc.get("content", "")
            if not include_content:
                content = ""
            elif len(content) > max_content_chars:
                content = content[:max_content_chars].rstrip() + "…"
            idx = doc.get("chunk_index")
            if idx is None:
                try:
                    idx = int(str(doc_id).rsplit(":", 1)[-1])
                except (TypeError, ValueError):
                    idx = 0
            hits.append({
                "doc_id": doc_id,
                "title": doc.get("title", ""),
                "source": doc.get("source") or doc.get("title", ""),
                "chunk_index": int(idx),
                "content": content,
                "score": float(score),
                "score_per_token": per_token,
                "rrf_score": float(rrf_score),
                "token_count": int(res["token_count"]),
            })
        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[:top_k]

    # ------------------------------------------------------------------ admin
    def list_documents(self, limit: int = 50) -> List[Dict[str, Any]]:
        """按来源去重列出文档（同一文档的多个切片只出现一次）。"""
        try:
            rows = self._doc.search().limit(max(1, min(limit, 500)) * 20).to_list()
        except Exception:
            return []
        out: List[Dict[str, Any]] = []
        seen = set()
        for r in rows:
            key = r.get("source") or r.get("title") or r.get("doc_id")
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "doc_id": r.get("doc_id", ""),
                "title": r.get("title", "") or r.get("source", ""),
                "source": r.get("source", "") or r.get("title", ""),
            })
            if len(out) >= max(1, min(limit, 500)):
                break
        return out

    def get_chunk_window(self, source: str,
                         chunk_index: Any) -> Dict[str, Any]:
        """返回当前切片 + 同文档前/后片（供切片阅读器翻页）。"""
        try:
            idx = int(chunk_index)
        except (TypeError, ValueError):
            return {}
        source = (source or "").strip()
        if not source:
            return {}
        escaped = source.replace("'", "''")
        try:
            rows = self._doc.search().where(
                f"source = '{escaped}' AND chunk_index IN "
                f"({idx - 1}, {idx}, {idx + 1})").limit(5).to_list()
        except Exception as e:
            logger.debug("ColBERT 读取切片窗口失败: %s", e)
            return {}

        def _to_chunk(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if not row:
                return None
            return {
                "doc_id": row.get("doc_id", ""),
                "title": row.get("title", ""),
                "source": row.get("source", "") or source,
                "chunk_index": int(row.get("chunk_index", idx)),
                "text": row.get("content", ""),
                "score": None,
            }

        by_idx = {int(r.get("chunk_index", -1)): r for r in rows}
        cur = _to_chunk(by_idx.get(idx))
        if cur is None:
            return {}
        return {"current": cur,
                "prev": _to_chunk(by_idx.get(idx - 1)),
                "next": _to_chunk(by_idx.get(idx + 1))}

    def build_ann_index(self, rebuild: bool = True) -> Dict[str, Any]:
        """构建/续建 LanceDB 中心向量 ANN 索引。

        ``rebuild=False`` 先 optimize（合并新 fragment、增量更新索引），
        再按需重建；任何失败都返回可读信息，不抛异常打断 UI。
        """
        with self._write_lock:
            rows = self._vec.count_rows()
            if rows == 0:
                return {"built": False, "vectors": 0,
                        "message": "向量库为空，请先同步知识库"}
            try:
                if rebuild:
                    self._vec.optimize()
                else:
                    self._vec.optimize(retrain=False)
                self._vec.create_index(
                    metric="cosine", vector_column_name="vector",
                    replace=True)
                self._ann_ready = True
                return {"built": True, "vectors": rows,
                        "message": f"ANN 索引已构建/续建（{rows} 个中心向量）"}
            except Exception as e:
                self._ann_ready = False
                logger.warning("ColBERT ANN 索引构建失败: %s", e)
                return {"built": False, "vectors": rows,
                        "message": f"ANN 索引构建失败（可继续使用精确扫描）：{e}"}

    def delete_matching(self, keyword: str) -> int:
        """按 source/title 子串删除所有匹配文档（知识遗忘/按名删除）。"""
        keyword = (keyword or "").strip()
        if not keyword:
            return 0
        escaped = keyword.replace("'", "''")
        try:
            rows = self._doc.search().where(
                f"source LIKE '%{escaped}%' OR title LIKE '%{escaped}%'"
            ).limit(5000).to_list()
        except Exception as e:
            logger.debug("ColBERT 按名查找待删文档失败: %s", e)
            return 0
        sources = {r.get("source") or r.get("title") for r in rows}
        removed = 0
        for source in sources:
            if source:
                removed += self.delete(source=source)
        return removed

    def delete(self, doc_id: str = "", source: str = "",
               title: str = "") -> int:
        prefix = ""
        if doc_id:
            prefix = doc_id
        elif source or title:
            prefix = hashlib.sha1(
                (source or title).encode("utf-8")).hexdigest()[:16] + ":"
        if not prefix:
            raise ColBertStoreError("doc_id or source/title is required")
        with self._write_lock:
            before = self._doc.count_rows()
            expr = (f"doc_id = '{prefix}'" if ":" in prefix
                    and not prefix.endswith(":") else f"doc_id LIKE '{prefix}%'")
            for table in (self._doc, self._vec, self._res):
                try:
                    table.delete(expr)
                except Exception:
                    pass
            self._fts_ready = False
            return max(0, before - self._doc.count_rows())

    def clear(self) -> None:
        """清空三表（知识库清空/重建时调用）。"""
        with self._write_lock:
            for name in ("document", "doc_summary_vec", "doc_residuals"):
                try:
                    self._db.drop_table(name, ignore_missing=True)
                except Exception as e:
                    logger.warning("ColBERT 表 %s 删除失败: %s", name, e)
            self._doc = self._open("document", _DOC_SCHEMA)
            self._vec = self._open("doc_summary_vec", _vec_schema(self.dim))
            self._res = self._open("doc_residuals", _RES_SCHEMA)
            self._fts_ready = False
            self._ann_ready = False

    def stats(self) -> Dict[str, int]:
        return {
            "documents": self._doc.count_rows(),
            "vectors": self._vec.count_rows(),
            "residuals": self._res.count_rows(),
        }


# ---------------------------------------------------------------------------
# 进程级 store 缓存：DocSearchTool 与 RAG 检索阅读模式共用同一份 LanceDB
# ---------------------------------------------------------------------------

_STORES: Dict[str, ColBertDocStore] = {}
_STORES_LOCK = threading.RLock()


def get_store_for_workspace(workspace: str, model_dir: str = ""
                            ) -> ColBertDocStore:
    """按 workspace 取（或创建）共享的 ColBertDocStore。"""
    base = os.path.abspath(os.path.expanduser(workspace or "."))
    db_dir = os.path.join(base, _INDEX_SUBDIR)
    with _STORES_LOCK:
        store = _STORES.get(db_dir)
        if store is None:
            store = ColBertDocStore(db_dir=db_dir, workspace=base,
                                    model_dir=model_dir)
            _STORES[db_dir] = store
        elif model_dir:
            store.set_model_dir(model_dir)
        return store
