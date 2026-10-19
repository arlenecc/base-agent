"""本地 embedding 后端：MLX / ONNX / GGUF。

对外暴露的 ``LocalEmbeddingFunction`` 与 ``FastEmbedEmbeddingFunction``
（vector_store.py）**同接口**：``__call__ / embed_query / embed_documents /
name``。因此调用方不需要任何改动就能在 fastembed 与本地模型之间切换。

三种后端的加载方式：
    mlx   — nomic_bert 走本包的 MLX 实现（``mlx_nomic``，因为
             ``mlx-embeddings`` 不认这个 model_type）；其它架构走
             ``mlx_embeddings.load``。
    onnx  — onnxruntime 直接跑 ``model.onnx`` / ``model_quantized.onnx``，
             tokenizer 用 transformers，池化与归一化用 numpy。
    gguf  — llama-cpp-python 的 ``embedding=True`` 模式。
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# nomic-embed-text-v1.5 的前缀约定（与 fastembed / sentence-transformers 一致）
_NOMIC_DOC_PREFIX = "search_document: "
_NOMIC_QUERY_PREFIX = "search_query: "

# 各模型族的 (doc_prefix, query_prefix)。未知族 → 空前缀（bge-m3 / e5 等
# 对称模型不需要前缀；需要 query instruction 的模型可后续在此扩展）。
_FAMILY_PREFIXES = {
    "nomic_bert": (_NOMIC_DOC_PREFIX, _NOMIC_QUERY_PREFIX),
}


def _model_family(path: str) -> str:
    """从 config.json 推断模型族（model_type 优先，architectures 兜底）。"""
    cfg_path = os.path.join(path, "config.json")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return ""
    mt = str(cfg.get("model_type") or "").strip().lower()
    if mt:
        return mt
    archs = cfg.get("architectures") or []
    if isinstance(archs, list) and archs:
        return str(archs[0]).strip().lower()
    return ""


def _onnx_file(path: str) -> Optional[str]:
    """挑选目录里的 ONNX 权重：优先非量化（精度高），否则量化版。"""
    for name in ("model.onnx", "model_quantized.onnx"):
        p = os.path.join(path, name)
        if os.path.exists(p):
            return p
    return None


class LocalEmbeddingFunction:
    """本地 embedding 函数（fastembed 的并行可选项）。

    线程安全：模型懒加载受 ``_load_lock`` 保护；推理本身由后端保证
    （ONNX Runtime / MLX 的 session 级线程安全）。
    """

    def __init__(self, path: str, backend: str = "mlx",
                 doc_prefix: str = "", query_prefix: str = "",
                 dtype: str = "float32", max_length: Optional[int] = None):
        self._path = path
        self._backend = (backend or "mlx").strip().lower()
        self._model = None
        self._tokenizer = None
        self._session = None
        self._load_lock = threading.Lock()
        self._loaded = False
        # 前缀：调用方未显式指定时按模型族给默认值
        family = _model_family(path)
        defaults = _FAMILY_PREFIXES.get(family, ("", ""))
        self._doc_prefix = doc_prefix or defaults[0]
        self._query_prefix = query_prefix or defaults[1]
        self._dtype = dtype
        self._family = family
        # 最大序列长度：未显式指定时按模型位置编码类型夹紧（见
        # registry.resolve_model_max_length：RoPE→8192，绝对位置→max_position_embeddings）。
        if max_length and max_length > 0:
            self._max_length = int(max_length)
        else:
            from .registry import resolve_model_max_length
            self._max_length = resolve_model_max_length(path)

    # ------------------------------------------------------------------
    # lazy loading
    # ------------------------------------------------------------------
    def _ensure_model(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:      # double-check
                return
            if not os.path.isdir(self._path):
                raise FileNotFoundError(f"本地 embedding 模型目录不存在: {self._path}")
            if self._backend == "mlx":
                self._load_mlx()
            elif self._backend == "onnx":
                self._load_onnx()
            elif self._backend == "gguf":
                self._load_gguf()
            else:
                raise ValueError(f"未知 embedding 后端: {self._backend!r}")
            self._loaded = True

    def _load_mlx(self) -> None:
        if self._family == "nomic_bert":
            # mlx-embeddings 不认 nomic_bert → 用本包的 MLX 实现
            from .mlx_nomic import load_nomic_bert
            self._model, _cfg = load_nomic_bert(self._path, dtype=self._dtype)
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._path)
            return
        try:
            from mlx_embeddings import load as _load
        except Exception as e:
            raise RuntimeError(
                "MLX embedding 后端不可用：未安装 mlx-embeddings。\n"
                "安装：pip install mlx-embeddings\n"
                f"原始错误: {type(e).__name__}: {e}"
            ) from e
        self._model, self._tokenizer = _load(self._path)

    def _load_onnx(self) -> None:
        onnx_path = _onnx_file(self._path)
        if onnx_path is None:
            raise FileNotFoundError(
                f"目录中没有 ONNX 权重（model.onnx / model_quantized.onnx）: {self._path}"
            )
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except Exception as e:
            raise RuntimeError(
                "ONNX embedding 后端不可用：需要 onnxruntime 与 transformers。\n"
                "安装：pip install onnxruntime transformers\n"
                f"原始错误: {type(e).__name__}: {e}"
            ) from e
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            onnx_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self._path)
        logger.info("local-embedding[onnx]: %s", os.path.basename(onnx_path))

    def _load_gguf(self) -> None:
        gguf = self._find_gguf()
        try:
            from llama_cpp import Llama
        except Exception as e:
            raise RuntimeError(
                "GGUF embedding 后端不可用：未安装 llama-cpp-python。\n"
                "安装：pip install llama-cpp-python\n"
                f"原始错误: {type(e).__name__}: {e}"
            ) from e
        # embedding=True：拿句向量；n_ctx 只需覆盖单条文本
        self._model = Llama(model_path=gguf, embedding=True, n_ctx=2048, verbose=False)

    def _find_gguf(self) -> str:
        if os.path.isfile(self._path) and self._path.endswith(".gguf"):
            return self._path
        if os.path.isdir(self._path):
            for f in sorted(os.listdir(self._path)):
                if f.endswith(".gguf"):
                    return os.path.join(self._path, f)
        raise FileNotFoundError(f"未找到 .gguf 文件: {self._path}")

    # ------------------------------------------------------------------
    # encoding
    # ------------------------------------------------------------------
    def _encode(self, texts: List[str], is_query: bool) -> List[List[float]]:
        self._ensure_model()
        prefix = self._query_prefix if is_query else self._doc_prefix
        prefixed = [prefix + t if prefix and not t.startswith(prefix) else t
                    for t in texts]
        if self._backend == "mlx":
            return self._encode_mlx(prefixed)
        if self._backend == "onnx":
            return self._encode_onnx(prefixed)
        return self._encode_gguf(prefixed)

    def _encode_mlx(self, texts: List[str]) -> List[List[float]]:
        import mlx.core as mx
        if self._family == "nomic_bert":
            from .mlx_nomic import l2_normalize, mean_pool
            enc = self._tokenizer(texts, padding=True, truncation=True,
                                  max_length=self._max_length, return_tensors="np")
            hidden = self._model(mx.array(enc["input_ids"]),
                                 mx.array(enc["attention_mask"]))
            pooled = l2_normalize(mean_pool(hidden, mx.array(enc["attention_mask"])))
            return [list(map(float, row)) for row in pooled.tolist()]
        # 其它架构：mlx_embeddings 的 generate()
        from mlx_embeddings import generate as _generate
        out = _generate(self._model, self._tokenizer, texts)
        return [list(map(float, row)) for row in out.tolist()]

    def _encode_onnx(self, texts: List[str]) -> List[List[float]]:
        import numpy as np
        enc = self._tokenizer(texts, padding=True, truncation=True,
                              max_length=self._max_length, return_tensors="np")
        feeds = {}
        for inp in self._session.get_inputs():
            if inp.name in ("input_ids", "attention_mask", "token_type_ids"):
                feeds[inp.name] = enc[inp.name]
        outs = self._session.run(None, feeds)
        last = np.asarray(outs[0], dtype=np.float32)      # [B, T, H]
        mask = enc["attention_mask"][:, :, None].astype(np.float32)
        pooled = (last * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        pooled = pooled / np.clip(norms, 1e-12, None)
        return [list(map(float, row)) for row in pooled]

    def _encode_gguf(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for t in texts:
            vec = self._model.embed(t)          # List[float]
            out.append([float(x) for x in vec])
        return out

    # ------------------------------------------------------------------
    # public API（与 FastEmbedEmbeddingFunction 一致）
    # ------------------------------------------------------------------
    def __call__(self, input: List[str]) -> List[List[float]]:
        if not input:
            return []
        return self._encode(list(input), is_query=False)

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text], is_query=True)[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self(texts)

    def name(self) -> str:
        base = os.path.basename(self._path.rstrip("/")) or self._path
        return f"local:{self._backend}:{base}"
