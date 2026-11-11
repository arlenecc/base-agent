"""KG 专属 embedding 适配（spec 第五步）。

要求：
* 模型 jina-embeddings-v5-text-nano（项目默认本地 embedding，MLX int4）；
* 任务类型：入库用 ``doc``，检索查询用 ``query``，文本前缀
  ``Document: `` / ``Query: `` 由 local/embedding.py 的 jina-v5 前缀族统一处理；
* 向量截取前 128 维 → **重新 L2 归一化** → int8 量化后入库。

复用 vector_store 既有管线：``_prepare_vector_for_index``（截断 + 重归一化 +
int8）与 Matryoshka 开关/维度（config.rag_matryoshka_enabled / rag_embedding_dims，
默认即 128）。查询向量与入库向量共用同一变换，保证同一表示空间。
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..vector_store import _prepare_vector_for_index

logger = logging.getLogger(__name__)

_INT8_SCALE = 127.0


def _matryoshka_state() -> tuple:
    """读取 vector_store 的模块级 Matryoshka 开关与维度（由 config 同步）。"""
    from .. import vector_store as vs
    enabled = bool(getattr(vs, "_MATRYOSHKA_ENABLED", True))
    dims = int(getattr(vs, "_MATRYOSHKA_DIMS", 128) or 0)
    return enabled, dims


def _is_matryoshka_model(name: str) -> bool:
    n = (name or "").lower()
    return "nomic-embed-text" in n or "jina-embeddings" in n


class KGEmbedder:
    """把文本 → (截断128维 → L2重归一化 → int8) 的向量，并提供 int8 余弦检索。"""

    def __init__(self, embedding_function: Any):
        # 兼容 LocalEmbeddingFunction（embed_task）与 FastEmbedEmbeddingFunction
        # （embed_query/embed_documents）两种实现。
        self._ef = embedding_function

    @property
    def model_name(self) -> str:
        try:
            return str(self._ef.name())
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # 文本构造（spec 第五步）
    # ------------------------------------------------------------------

    @staticmethod
    def entity_text(name: str, description: str) -> str:
        """实体向量文本：name + "\n" + description。"""
        return f"{name or ''}\n{description or ''}"

    @staticmethod
    def relationship_text(src: str, dst: str, keywords: str,
                          description: str, reverse: bool = False) -> str:
        """关系向量文本：keywords + "\t" + src + "\n" + dst + "\n" + description。

        无向边 → 正反序各一个 id（reverse=True 时交换 src/dst）。
        """
        a, b = (dst, src) if reverse else (src, dst)
        return f"{keywords or ''}\t{a or ''}\n{b or ''}\n{description or ''}"

    @staticmethod
    def content_hash(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # 编码
    # ------------------------------------------------------------------

    def _encode_task(self, texts: List[str], task: str) -> List[List[float]]:
        ef = self._ef
        if hasattr(ef, "embed_task"):
            return list(ef.embed_task(texts, task))
        # fastembed 路径：无任务概念，退化为 query/doc 两个既有方法。
        if task == "query" and hasattr(ef, "embed_query"):
            return [list(ef.embed_query(t)) for t in texts]
        if hasattr(ef, "embed_documents"):
            return list(ef.embed_documents(texts))
        return list(ef(texts))

    def embed_for_index(self, texts: List[str]) -> List[List[int]]:
        """入库向量：doc 任务 + 截断/重归一化/int8（vector_store 同管线）。"""
        raw = self._encode_task(texts, "doc")
        enabled, dims = _matryoshka_state()
        matry = enabled and _is_matryoshka_model(self.model_name)
        return [_prepare_vector_for_index(v, matry, dims) for v in raw]

    def embed_query(self, texts: List[str]) -> List[List[int]]:
        """查询向量：query 任务 + 同一变换（与入库同空间）。"""
        raw = self._encode_task(texts, "query")
        enabled, dims = _matryoshka_state()
        matry = enabled and _is_matryoshka_model(self.model_name)
        return [_prepare_vector_for_index(v, matry, dims) for v in raw]

    # ------------------------------------------------------------------
    # int8 余弦检索
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_int8(a: bytes, b: bytes, dim_hint: Optional[int] = None) -> float:
        """两条 int8 向量的余弦相似度（对称量化 ×127，模长在量化时已归一）。"""
        n = min(len(a), len(b))
        if n == 0:
            return 0.0
        va = np.frombuffer(a[:n], dtype=np.int8).astype(np.float32)
        vb = np.frombuffer(b[:n], dtype=np.int8).astype(np.float32)
        na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))

    def search_vectors(
        self,
        query_vec: Sequence[int],
        vectors: List[Dict[str, Any]],
        top_k: int,
        min_similarity: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """在候选向量集内做 int8 余弦检索，余弦 < min_similarity 直接丢弃。"""
        if not vectors or not query_vec:
            return []
        q = np.asarray(list(query_vec), dtype=np.float32)
        qn = float(np.linalg.norm(q))
        if qn == 0.0:
            return []
        scored: List[Dict[str, Any]] = []
        for item in vectors:
            emb = item.get("embedding") or b""
            if not emb:
                continue
            v = np.frombuffer(emb, dtype=np.int8).astype(np.float32)
            n = min(len(v), len(q))
            if n == 0:
                continue
            v, qs = v[:n], q[:n]
            vn = float(np.linalg.norm(v))
            if vn == 0.0:
                continue
            sim = float(np.dot(v, qs) / (vn * qn))
            if sim < min_similarity:
                continue
            scored.append({**item, "similarity": sim})
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]
