"""本地 reranker（cross-encoder）后端：MLX / ONNX / GGUF。

对外暴露的 ``LocalReranker`` 与 fastembed ``TextCrossEncoder`` 同接口：
``rerank(query, documents) -> List[float]``（原始分数，由调用方决定是否
sigmoid）。``VectorStore`` 因此可以在不改检索逻辑的前提下切换精排模型。
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def _stable_sigmoid(x: float) -> float:
    """数值稳定的 sigmoid（与 vector_store 的同名实现一致）。"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _onnx_file(path: str) -> Optional[str]:
    for name in ("model.onnx", "model_quantized.onnx"):
        p = os.path.join(path, name)
        if os.path.exists(p):
            return p
    return None


class LocalReranker:
    """本地 cross-encoder 精排（fastembed TextCrossEncoder 的并行可选项）。"""

    def __init__(self, path: str, backend: str = "mlx",
                 max_length: Optional[int] = None, batch_size: int = 8):
        self._path = path
        self._backend = (backend or "mlx").strip().lower()
        # 最大序列长度：未显式指定时按模型位置编码类型夹紧（见
        # registry.resolve_model_max_length：RoPE→8192，绝对位置→max_position_embeddings）。
        if max_length and max_length > 0:
            self._max_length = int(max_length)
        else:
            from .registry import resolve_model_max_length
            self._max_length = resolve_model_max_length(path)
        self._batch_size = max(1, int(batch_size))
        self._model = None
        self._tokenizer = None
        self._session = None
        self._load_lock = threading.Lock()
        self._loaded = False
        self._is_logits: Optional[bool] = None   # 首次打分时探测

    def close(self) -> None:
        """释放后端权重 / ONNX InferenceSession（单例淘汰与测试清理用）。"""
        with self._load_lock:
            self._model = None
            self._tokenizer = None
            self._session = None
            self._loaded = False
            self._is_logits = None

    # ------------------------------------------------------------------
    def _ensure_model(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            if self._backend == "mlx":
                self._load_mlx()
            elif self._backend == "onnx":
                self._load_onnx()
            elif self._backend == "gguf":
                self._load_gguf()
            else:
                raise ValueError(f"未知 reranker 后端: {self._backend!r}")
            self._loaded = True

    def _load_mlx(self) -> None:
        if not os.path.isdir(self._path):
            raise FileNotFoundError(f"本地 reranker 模型目录不存在: {self._path}")
        try:
            from mlx_lm import load
        except Exception as e:
            raise RuntimeError(
                "MLX reranker 后端不可用：未安装 mlx-lm。\n"
                "安装：pip install mlx-lm\n"
                f"原始错误: {type(e).__name__}: {e}"
            ) from e
        # 模型目录带自定义 modeling 代码（如 reranker_xlm_roberta.py，
        # 经 config.json 的 model_file 引用）→ 必须 trust_remote_code
        self._model, self._tokenizer = load(self._path, trust_remote_code=True)
        logger.info("local-reranker[mlx]: %s", os.path.basename(self._path))

    def _load_onnx(self) -> None:
        onnx_path = _onnx_file(self._path) if os.path.isdir(self._path) else self._path
        if not onnx_path or not os.path.exists(onnx_path):
            raise FileNotFoundError(f"未找到 reranker ONNX 权重: {self._path}")
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except Exception as e:
            raise RuntimeError(
                "ONNX reranker 后端不可用：需要 onnxruntime 与 transformers。\n"
                f"原始错误: {type(e).__name__}: {e}"
            ) from e
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            onnx_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        tok_dir = self._path if os.path.isdir(self._path) else os.path.dirname(self._path)
        self._tokenizer = AutoTokenizer.from_pretrained(tok_dir)
        logger.info("local-reranker[onnx]: %s", os.path.basename(onnx_path))

    def _load_gguf(self) -> None:
        if os.path.isfile(self._path) and self._path.endswith(".gguf"):
            gguf = self._path
        else:
            gguf = None
            for f in sorted(os.listdir(self._path)):
                if f.endswith(".gguf"):
                    gguf = os.path.join(self._path, f)
                    break
            if gguf is None:
                raise FileNotFoundError(f"未找到 .gguf 文件: {self._path}")
        try:
            from llama_cpp import Llama
        except Exception as e:
            raise RuntimeError(
                "GGUF reranker 后端不可用：未安装 llama-cpp-python。\n"
                f"原始错误: {type(e).__name__}: {e}"
            ) from e
        # 新版本 llama-cpp-python 提供 rerank 能力（create_rerank）
        self._model = Llama(model_path=gguf, n_ctx=2048, verbose=False)

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------
    def rerank(self, query: str, documents: List[str]) -> List[float]:
        """对 (query, doc) 对打分，返回与 documents 等长同序的分数列表。"""
        if not documents:
            return []
        self._ensure_model()
        if self._backend == "mlx":
            scores = self._score_mlx(query, documents)
        elif self._backend == "onnx":
            scores = self._score_onnx(query, documents)
        else:
            scores = self._score_gguf(query, documents)
        # 输出形态探测（原始 logits 还是已归一化概率）—— 只探一次，
        # 之后按模型固定处理，避免分数随查询漂移。
        if self._is_logits is None:
            self._is_logits = any(abs(s) > 1.0 for s in scores)
            logger.info("local-reranker: 输出探测: %s",
                        "原始 logits（调用方将做 sigmoid）" if self._is_logits
                        else "已归一化概率")
        return scores

    # ---- MLX ----
    def _score_mlx(self, query: str, documents: List[str]) -> List[float]:
        import mlx.core as mx
        tok = self._tokenizer
        # mlx_lm 的 TokenizerWrapper 把真实 HF tokenizer 存在 _tokenizer 属性里
        # （直接 callable 的是 wrapper，不可用）；少数情况下是直接的 AutoTokenizer。
        hf = getattr(tok, "_tokenizer", getattr(tok, "tokenizer", tok))
        scores: List[float] = []
        for i in range(0, len(documents), self._batch_size):
            batch = documents[i:i + self._batch_size]
            pairs = [(query, d) for d in batch]
            enc = hf(pairs, padding=True, truncation=True,
                     max_length=self._max_length, return_tensors="np")
            out = self._model(mx.array(enc["input_ids"]),
                              mx.array(enc["attention_mask"]))
            arr = out.reshape(-1)
            scores.extend(float(x) for x in arr.tolist())
            mx.eval(arr)
        return scores

    # ---- ONNX ----
    def _score_onnx(self, query: str, documents: List[str]) -> List[float]:
        import numpy as np
        hf = getattr(self._tokenizer, "_tokenizer",
                     getattr(self._tokenizer, "tokenizer", self._tokenizer))
        scores: List[float] = []
        for i in range(0, len(documents), self._batch_size):
            batch = documents[i:i + self._batch_size]
            enc = hf([(query, d) for d in batch], padding=True, truncation=True,
                     max_length=self._max_length, return_tensors="np")
            feeds = {}
            for inp in self._session.get_inputs():
                if inp.name in enc:
                    feeds[inp.name] = enc[inp.name]
            outs = self._session.run(None, feeds)
            logits = np.asarray(outs[0], dtype=np.float32)
            scores.extend(float(x) for x in logits.reshape(-1))
        return scores

    # ---- GGUF ----
    def _score_gguf(self, query: str, documents: List[str]) -> List[float]:
        # llama-cpp-python 0.3.x 提供 create_rerank（兼容 OpenAI 的 rerank 接口）
        create = getattr(self._model, "create_rerank", None)
        if create is None:
            raise RuntimeError(
                "当前 llama-cpp-python 版本不支持 rerank（需要 create_rerank）。\n"
                "请升级：pip install -U llama-cpp-python"
            )
        res = create(query=query, documents=documents)
        # 返回 [{index, relevance_score}, ...] → 按原顺序还原
        by_idx = {int(r["index"]): float(r["relevance_score"]) for r in res}
        return [by_idx.get(i, 0.0) for i in range(len(documents))]

    # ------------------------------------------------------------------
    def name(self) -> str:
        base = os.path.basename(self._path.rstrip("/")) or self._path
        return f"local-rerank:{self._backend}:{base}"


def normalize_scores(scores: List[float]) -> List[float]:
    """把 reranker 原始分归一到 [0,1]（logits 走 sigmoid）。"""
    return [_stable_sigmoid(s) if abs(s) > 1.0 else s for s in scores]
