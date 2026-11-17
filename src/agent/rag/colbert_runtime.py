"""Jina ColBERT v2（64 维 INT8 ONNX）的本地推理封装。

模型目录里的 ``infer.py`` 是参考实现；这里保留同样的核心语义：

* query 文本前加 ``[QueryMarker]``（token 250002），文档前加
  ``[DocumentMarker]``（token 250003）；
* 每个 token 输出 64 维向量并做 L2 归一化；
* 检索打分用 ColBERT 的 MaxSim：``sum_i max_j (q_i · d_j)``。

差异：
* 默认不做 query augmentation（不补 ``<mask>``），因为下游要用「问句里
  所有真实词」的中心向量做粗筛；
* **保留** BOS/EOS（``<s>`` / ``</s>``）与 ``[marker]``，只丢弃 pad；
  实测去掉 BOS/EOS 会让 JinaBERT 输出退化成近各向同性向量（query token
  两两余弦 ≈0.99，MaxSim 无法区分文档），因此必须与参考实现一致；
* 默认不做 query augmentation（不补 MASK），因为下游要用「问句真实词」
  的中心向量做粗筛；需要时可通过 ``query_augmentation=True`` 打开；
* 长文本按 ``batch_size`` 分批跑 ONNX，避免一次 padding 到 8k 造成内存峰值。
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MODEL_REPO_ID = "arlenecc/jina-colbert-v2-64-onnx-int8"
MODEL_DIRNAME = "jina-colbert-v2-64-onnx-int8"
HF_MIRROR = "https://hf-mirror.com"
_MODEL_FILES = ("model_int8.onnx", "onnx/model_int8.onnx",
                "model_quantized.onnx", "onnx/model_quantized.onnx",
                "model.onnx", "onnx/model.onnx")
_MODEL_COMPANIONS = ("tokenizer.json", "config.json")

QUERY_PREFIX = "[QueryMarker]"
DOC_PREFIX = "[DocumentMarker]"
QUERY_MARKER_ID = 250002
DOC_MARKER_ID = 250003
MASK_ID = 250001
PAD_ID = 1
BOS_ID = 0
EOS_ID = 2

# 模型 max_position_embeddings=8194；留出 prefix/special 的余量。
DEFAULT_MAX_LENGTH = 8190


class ColBertUnavailable(RuntimeError):
    """模型/依赖不可用（调用方应转成可读的 ToolResult 错误）。"""


def _first_existing(directory: str, relative_paths: Sequence[str]) -> str:
    for rel in relative_paths:
        path = os.path.join(directory, *rel.split("/"))
        if os.path.isfile(path):
            return path
    return ""


def model_is_complete(directory: str) -> bool:
    """目录是否具备运行所需的 ONNX 权重 + tokenizer + config。"""
    if not directory or not os.path.isdir(directory):
        return False
    if not _first_existing(directory, _MODEL_FILES):
        return False
    return all(_first_existing(directory, (name,))
               for name in _MODEL_COMPANIONS)


def default_model_dir() -> str:
    """模型目录解析：环境变量 > 已下载的配置目录 > 仓库自带 models/。

    都不存在时返回配置目录下的下载落点（``~/.base-agent/models/...``），
    由 ``ColBertRuntime`` 首次加载时自动从 hf-mirror 下载。
    """
    env = os.environ.get("BASE_AGENT_COLBERT_MODEL", "").strip()
    if env:
        return env
    try:
        from ..local.registry import config_models_dir
        config_dir = os.path.join(config_models_dir(), MODEL_DIRNAME)
    except Exception:
        config_dir = os.path.expanduser(
            f"~/.base-agent/models/{MODEL_DIRNAME}")
    # src/agent/rag/colbert_runtime.py -> 仓库根目录
    root = Path(__file__).resolve().parents[3]
    repo_dir = str(root / "models" / MODEL_DIRNAME)
    for candidate in (config_dir, repo_dir):
        if model_is_complete(candidate):
            return candidate
    return config_dir


def ensure_model_dir(directory: str, progress_cb=None,
                     cancel_event=None) -> str:
    """确保目录内有 ColBERT 模型；缺失时从 hf-mirror.com 自动下载。

    复用 ``local.model_downloader`` 的 HF REST 清单 + 白名单 + 断点续传实现，
    只拉运行所需文件（INT8 ONNX + tokenizer/config），不下载 README 等。
    """
    directory = os.path.abspath(os.path.expanduser(directory))
    if model_is_complete(directory):
        return directory
    os.makedirs(directory, exist_ok=True)
    try:
        from ..local.model_downloader import (
            DEFAULT_BASE_URL, DEFAULT_NAMESPACE, ModelTask, download_model,
        )
        repo_name = MODEL_REPO_ID.split("/", 1)[-1]
        task = ModelTask(
            key="colbert", display_name="Jina ColBERT v2（检索阅读模式）",
            repo_name=repo_name, target_dirs=[directory], checker="onnx")
        state = {"pct": -5.0}

        def _progress(done: int, total: int, label: str) -> None:
            pct = (done / total * 100.0) if total else 0.0
            if pct >= state["pct"] + 5.0 or (total and done >= total):
                state["pct"] = pct
                logger.info("ColBERT 模型下载 %.0f%%（%s）", pct, label)
            if progress_cb is not None:
                progress_cb(done, total, label)

        logger.info("未找到 ColBERT 模型，开始从 %s 下载 %s → %s",
                    DEFAULT_BASE_URL, MODEL_REPO_ID, directory)
        download_model(task, DEFAULT_BASE_URL, DEFAULT_NAMESPACE,
                       _progress, cancel_event)
    except Exception as e:
        raise ColBertUnavailable(
            f"ColBERT 模型缺失且自动下载失败：{directory}。"
            f"请检查网络，或手动从 {HF_MIRROR}/{MODEL_REPO_ID} 下载 "
            f"model_int8.onnx / tokenizer.json / config.json 到该目录。"
            f"（{type(e).__name__}: {e}）") from e
    if not model_is_complete(directory):
        raise ColBertUnavailable(
            f"ColBERT 模型下载后仍不完整：{directory}；需要 model_int8.onnx、"
            f"tokenizer.json、config.json")
    return directory


class ColBertRuntime:
    """ONNX Runtime 版 ColBERT 编码器（线程安全、惰性加载）。"""

    def __init__(self, model_dir: Optional[str] = None,
                 providers: Optional[Sequence[str]] = None,
                 max_length: int = DEFAULT_MAX_LENGTH,
                 auto_download: bool = True,
                 query_augmentation: bool = False,
                 query_maxlen: int = 32,
                 doc_window_tokens: int = 512):
        self.dir = os.path.abspath(os.path.expanduser(
            model_dir or default_model_dir()))
        if not model_is_complete(self.dir) and auto_download:
            ensure_model_dir(self.dir)
        model_path = _first_existing(self.dir, _MODEL_FILES)
        tokenizer_path = _first_existing(self.dir, ("tokenizer.json",))
        config_path = _first_existing(self.dir, ("config.json",))
        if not model_path or not tokenizer_path or not config_path:
            raise ColBertUnavailable(
                f"ColBERT 模型文件缺失：{self.dir}（需要 model_int8.onnx、"
                f"tokenizer.json、config.json）")
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:
            raise ColBertUnavailable(
                f"缺少 onnxruntime/tokenizers 依赖：{e}") from e

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.dim = int(self.config.get("projection_dim") or 64)
        self.max_length = int(max_length)
        self.query_augmentation = bool(query_augmentation)
        self.query_maxlen = max(1, int(query_maxlen))
        self.doc_window_tokens = max(64, int(doc_window_tokens))
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        # longest_first 会优先保留句首（含 marker），截断尾部。
        self.tokenizer.enable_truncation(
            max_length=self.max_length, strategy="longest_first")
        self.session = ort.InferenceSession(
            model_path,
            providers=list(providers or ["CPUExecutionProvider"]),
        )
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ core
    def _tokenize(self, text: str, kind: str) -> List[int]:
        prefix = QUERY_PREFIX if kind == "query" else DOC_PREFIX
        body = (text or "").strip()
        encoded = self.tokenizer.encode(
            f"{prefix} {body}", add_special_tokens=True)
        # 与参考实现一致：保留 BOS/EOS 与 marker，只过滤 pad。
        # BOS/EOS 参与 JinaBERT 的上下文编码；去掉后模型输出会退化成
        # 近各向同性向量，MaxSim 失去区分度（实测两两余弦 ≈0.99）。
        ids = [tid for tid in encoded.ids if tid != PAD_ID]
        if kind == "query" and self.query_augmentation and ids:
            target = max(self.query_maxlen, len(ids))
            ids = ids + [MASK_ID] * (target - len(ids))
        return ids

    def _run_ids(self, ids_list: Sequence[Sequence[int]],
                 batch_size: int) -> List[np.ndarray]:
        """按 ids 批量跑 ONNX，返回与每条 ids 对齐的 L2 归一化向量。"""
        out: List[np.ndarray] = []
        if not ids_list:
            return out
        with self._lock:
            for start in range(0, len(ids_list), max(1, batch_size)):
                chunk = ids_list[start:start + max(1, batch_size)]
                maxlen = max(1, max(len(ids) for ids in chunk))
                input_ids = np.full(
                    (len(chunk), maxlen), PAD_ID, dtype=np.int64)
                attention = np.zeros((len(chunk), maxlen), dtype=np.int64)
                for i, ids in enumerate(chunk):
                    input_ids[i, :len(ids)] = ids
                    attention[i, :len(ids)] = 1
                raw = self.session.run(
                    None,
                    {"input_ids": input_ids, "attention_mask": attention},
                )[0]
                raw = np.asarray(raw, dtype=np.float32)
                for i, ids in enumerate(chunk):
                    vecs = raw[i, :len(ids)].copy()
                    if vecs.size == 0:
                        vecs = np.zeros((1, self.dim), dtype=np.float32)
                    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                    vecs = vecs / np.maximum(norms, 1e-12)
                    out.append(vecs)
        return out

    def _encode_batch(self, texts: Sequence[str], kind: str,
                      batch_size: int) -> List[np.ndarray]:
        if isinstance(texts, str):
            texts = [texts]
        return self._run_ids([self._tokenize(t, kind) for t in texts],
                             batch_size)

    def _document_sequences(self, text: str
                            ) -> List[Tuple[List[int], int]]:
        """把文档 ids 切成局部窗口（含 BOS/DOC/EOS），返回 (seq, 内容长度)。

        长切片一次前向会让 JinaBERT 的 token 表示随长度退化（实测 4k token
        的 MaxSim 从 11.7 降到 7.1，正确切片会被短切片挤出 topN）。按 512
        token 窗口局部编码后，所有词向量仍在（上下文局部化），长切片的
        MaxSim 恢复到 9~10，与短切片可比。
        """
        ids = self._tokenize(text, "document")
        content = list(ids[2:]) if len(ids) > 2 else []
        if content and content[-1] == EOS_ID:
            content.pop()
        if not content:
            return []
        window = max(64, int(self.doc_window_tokens))
        seqs: List[Tuple[List[int], int]] = []
        for start in range(0, len(content), window):
            piece = content[start:start + window]
            seqs.append(([BOS_ID, DOC_MARKER_ID] + piece + [EOS_ID],
                         len(piece)))
        return seqs

    def encode_query(self, text: str) -> np.ndarray:
        """返回问句 token 向量 (Q, dim)，含 BOS/EOS 与 [QueryMarker]。"""
        return self._encode_batch([text], "query", batch_size=1)[0]

    def encode_queries(self, texts: Sequence[str],
                       batch_size: int = 8) -> List[np.ndarray]:
        return self._encode_batch(texts, "query", batch_size=batch_size)

    def encode_document(self, text: str) -> np.ndarray:
        """返回文档所有词的 token 向量（长文本按局部窗口编码）。"""
        return self.encode_documents([text], batch_size=1)[0]

    def encode_documents(self, texts: Sequence[str],
                         batch_size: int = 1) -> List[np.ndarray]:
        if isinstance(texts, str):
            texts = [texts]
        prepared: List[Tuple[int, List[int], int]] = []
        for idx, text in enumerate(texts):
            for seq, length in self._document_sequences(text):
                prepared.append((idx, seq, length))
        raw = self._run_ids([p[1] for p in prepared], batch_size)
        grouped: List[List[np.ndarray]] = [[] for _ in texts]
        for (idx, _seq, length), vecs in zip(prepared, raw):
            # 丢弃窗口自带的 BOS / [DocumentMarker] / EOS，仅保留内容词向量。
            grouped[idx].append(vecs[2:2 + length])
        out: List[np.ndarray] = []
        for vecs_list in grouped:
            if not vecs_list:
                out.append(np.zeros((1, self.dim), dtype=np.float32))
            else:
                out.append(np.vstack(vecs_list))
        return out

    @staticmethod
    def maxsim(query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
        q = np.asarray(query_emb, dtype=np.float32)
        d = np.asarray(doc_emb, dtype=np.float32)
        if q.size == 0 or d.size == 0:
            return float("-inf")
        return float((q @ d.T).max(axis=1).sum())

    def self_test(self) -> bool:
        q = self.encode_query("What does ColBERT do?")
        d = self.encode_document(
            "ColBERT is a late interaction retrieval model.")
        ok = (q.shape[1] == self.dim and d.shape[1] == self.dim
              and np.isfinite(q).all() and np.isfinite(d).all())
        return bool(ok)


_RUNTIME_LOCK = threading.RLock()
_RUNTIME: Optional[ColBertRuntime] = None
_RUNTIME_KEY = ""


def get_runtime(model_dir: Optional[str] = None) -> ColBertRuntime:
    """进程级单例：模型 587MB，避免每次工具调用重复加载。"""
    global _RUNTIME, _RUNTIME_KEY
    key = os.path.abspath(model_dir or default_model_dir())
    with _RUNTIME_LOCK:
        if _RUNTIME is None or _RUNTIME_KEY != key:
            _RUNTIME = ColBertRuntime(key)
            _RUNTIME_KEY = key
        return _RUNTIME
