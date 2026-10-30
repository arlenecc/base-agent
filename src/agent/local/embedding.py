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

# ---------------------------------------------------------------------------
# 任务前缀策略（按模型族统一收敛，唯一权威表）
#
# 各 embedding 模型族的「任务前缀」约定不同。这里以「前缀族」为单位统一
# 维护一张表：本地路径（LocalEmbeddingFunction，mlx/onnx/gguf）与 fastembed
# 路径（vector_store.py 的 FastEmbedEmbeddingFunction）都通过
# resolve_task_prefixes() 取同一份数据，避免两处各写一份导致漂移。
#
# 键为任务名：doc（入库文档）/ query（检索查询）/ clustering（聚类）/
# classification（分类）。某族未定义的任务回退到该族 doc 前缀（对称语义，
# bge 的 doc 前缀为空即不加）；完全未知的族 → 全空前缀。
_FAMILY_TASK_PREFIXES: Dict[str, Dict[str, str]] = {
    # nomic-embed-text：检索前缀（官方约定，与 fastembed 一致）
    "nomic": {"doc": "search_document: ", "query": "search_query: "},
    # bge 系列：非对称检索，query 带指令，document 不加前缀
    "bge": {"doc": "", "query": "为这个句子生成表示以检索相关文档："},
    # jina 系列：检索 / 聚类 / 分类各有一套前缀
    "jina": {"doc": "passage: ", "query": "query: ",
             "clustering": "clustering: ", "classification": "classification: "},
    # jina-embeddings-v5（EuroBERT，last-token 池化）：前缀与 v3 不同
    "jina-v5": {"doc": "Document: ", "query": "Query: ",
                "clustering": "Document: ", "classification": "Document: "},
    # e5 系列：检索前缀
    "e5": {"doc": "passage: ", "query": "query: "},
}

# 前缀族检测：bge / e5 的 config.json model_type 都是 bert（jina v2 也是
# bert、v3 是 qwen2），架构字段分不出族，只能先按模型目录名关键词判断；
# 名称不含关键词时再回退到 model_type（nomic_bert → nomic）。
_PREFIX_FAMILY_NAME_KEYWORDS = ("nomic", "bge", "jina", "e5")

# jina-embeddings-v3 是「统一模型」：单文件通过 task 选择 LoRA 适配器……（见下）
# 高维前缀（passage:/query: 等）已在 _FAMILY_TASK_PREFIXES 的 jina 项中处理；
# 这里与模型 task 是两件不同的事，要同时叠加：模型 task 选 LoRA，文本前缀做标签。

# embedding 输入 token 硬上限：切片标准是 800 token，留余量截到 1024。模型原生
# 虽支持 8192（RoPE 外推），但这里强制不超 1024——避免对超长 / 异常输入做无意义的
# 8192 token 前向计算（RoPE 下长序列算力近似随长度线性增长）。正常 800 token 切片
# 永不触发截断，仅保护异常长输入。所有编码路径的 tokenizer 都按此上限截断。
_MAX_EMBED_INPUT_TOKENS = 1024

# jina-embeddings-v3 是「统一模型」：单文件服务全部任务，推理时通过 task 选择
# LoRA 适配器。这里把高层意图（doc/query/clustering/classification）映射到模型
# 的 task 名。切片（入库）用 retrieval.passage，检索查询用 retrieval.query。
_JINA_MODEL_TASK = {
    "doc": "retrieval.passage",
    "query": "retrieval.query",
    "clustering": "clustering",
    "classification": "classification",
}
# 高维前缀（passage:/query: 等）已在 _FAMILY_TASK_PREFIXES 的 jina 项中处理；
# 这里与模型 task 是两件不同的事，要同时叠加：模型 task 选 LoRA，文本前缀做标签。


def _model_family(path: str) -> str:
    """从 config.json 推断模型架构族（model_type 优先，architectures 兜底）。

    仅用于加载 / 池化调度（nomic_bert / bert 等），与「前缀族」是两回事。
    """
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


def _prefix_family(model_ref: str) -> str:
    """从模型目录名 / model_type 推断前缀族（未知 → ""，即全空前缀）。"""
    name = os.path.basename(str(model_ref).rstrip("/\\")).lower()
    if "jina" in name and "v5" in name:
        return "jina-v5"      # v5 前缀（Query:/Document:）与 v3 不同，先于通用 jina 判定
    for key in _PREFIX_FAMILY_NAME_KEYWORDS:
        if key in name:
            return key
    family = _model_family(model_ref)
    if family == "eurobert":
        return "jina-v5"
    return "nomic" if family == "nomic_bert" else ""


def resolve_task_prefixes(model_ref: str) -> Dict[str, str]:
    """按模型名 / 目录返回该族的任务前缀表（各 embedding 后端统一取用）。"""
    return dict(_FAMILY_TASK_PREFIXES.get(_prefix_family(model_ref), {}))


def _onnx_file(path: str) -> Optional[str]:
    """挑选目录里的 ONNX 权重：优先非量化（精度高），否则量化版。"""
    for name in ("model.onnx", "model_quantized.onnx"):
        p = os.path.join(path, name)
        if os.path.exists(p):
            return p
    return None


_MLX_WEIGHT_SUFFIXES = (".safetensors", ".npz", ".mlx")


def _has_mlx_weights(path: str) -> bool:
    """目录里是否存在 MLX 权重（safetensors / npz / .mlx）。"""
    if not os.path.isdir(path):
        return False
    return any(f.endswith(_MLX_WEIGHT_SUFFIXES) for f in os.listdir(path))


class LocalEmbeddingFunction:
    """本地 embedding 函数（fastembed 的并行可选项）。

    线程安全：模型懒加载受 ``_load_lock`` 保护；推理本身由后端保证
    （ONNX Runtime / MLX 的 session 级线程安全）。
    """

    def __init__(self, path: str, backend: str = "mlx",
                 doc_prefix: str = "", query_prefix: str = "",
                 dtype: str = "float32", max_length: Optional[int] = None,
                 truncate_dim: Optional[int] = None):
        self._path = path
        self._backend = (backend or "mlx").strip().lower()
        # 按目录内容自动回退 backend：用户选错（如指向 ONNX 目录却留默认 mlx）
        # 时也能开箱即用，且不覆盖任何「权重确实存在」的有效选择。
        if (self._backend == "mlx"
                and not _has_mlx_weights(self._path)
                and _onnx_file(self._path)):
            self._backend = "onnx"
        elif (self._backend == "onnx"
              and not _onnx_file(self._path)
              and _has_mlx_weights(self._path)):
            self._backend = "mlx"
        self._model = None
        self._tokenizer = None
        self._session = None
        self._load_lock = threading.Lock()
        self._loaded = False
        # 前缀：统一按前缀族表取默认值（见 _FAMILY_TASK_PREFIXES）；
        # 显式传 doc_prefix/query_prefix 仅用于覆盖，未传的任务项保持表默认。
        self._family = _model_family(path)            # 架构族：MLX 加载/池化调度用
        self._prefix_family = _prefix_family(path)    # 前缀族：doc/query 等文本前缀
        self._prefixes = resolve_task_prefixes(path)  # 前缀族 → 任务前缀表
        if doc_prefix:
            self._prefixes["doc"] = doc_prefix
        if query_prefix:
            self._prefixes["query"] = query_prefix
        self._doc_prefix = self._prefixes.get("doc", "")
        self._query_prefix = self._prefixes.get("query", "")
        self._dtype = dtype
        # Matryoshka 截维（可选）：非 None 时 MLX jina 系（v3/v5）在模型内
        # 截取前 N 维并重新 L2 归一化后再返回（v3 的 ONNX 路径不支持，
        # 仍由 vector_store 的 Matryoshka 流程统一处理）。
        self._truncate_dim = int(truncate_dim) if truncate_dim else None
        # 最大序列长度：未显式指定时按模型位置编码类型夹紧（见
        # registry.resolve_model_max_length：RoPE→8192，绝对位置→max_position_embeddings），
        # 但无论模型原生支持多长，统一强制截到 _MAX_EMBED_INPUT_TOKENS（1024）：
        # 切片标准是 800 token，留余量到 1024 即可覆盖全部切片，避免对超长 / 异常输入
        # 做浪费的 8192 token 前向计算。
        if max_length and max_length > 0:
            eff = int(max_length)
        else:
            from .registry import resolve_model_max_length
            eff = resolve_model_max_length(path)
        self._max_length = min(eff, _MAX_EMBED_INPUT_TOKENS)

    def close(self) -> None:
        """释放后端权重 / ONNX InferenceSession。

        单例淘汰（provider 配置变更）与测试清理走这里显式释放；只靠
        字典摘引用的话，ORT session 的 C++ 析构要等 GC，切换配置的瞬间
        内存仍是两份权重并存。
        """
        with self._load_lock:
            self._model = None
            self._tokenizer = None
            self._session = None
            self._loaded = False

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
        if self._family == "jina-v3-mlx":
            # 本包 vendoring 的 MLX 实现（models/jina-embeddings-v3-mlx-*）：
            # XLM-RoBERTa + RoPE + 任务 LoRA，mlx-embeddings / mlx-lm 都不认这类权重。
            from .mlx_jina import load_model
            self._model, _cfg = load_model(self._path)
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._path)
            return
        if self._family == "eurobert":
            # jina-embeddings-v5-text-nano（EuroBERT-210M，last-token 池化）。
            # vendored 实现（mlx_jina_v5）支持 config.json quantization 的
            # int4/int8 量化权重；fp16 版自动预合并 retrieval LoRA。
            from .mlx_jina_v5 import load_model
            self._model, _cfg = load_model(self._path)
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._path)
            return
        if self._family == "nomic_bert":
            # mlx-embeddings 不认 nomic_bert → 用本包的 MLX 实现
            from .mlx_nomic import load_nomic_bert
            self._model, _cfg = load_nomic_bert(self._path, dtype=self._dtype)
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._path)
            return
        if self._family == "bert":
            # 仓库里的 bge MLX 权重是 mlx-lm 格式键名（query_proj/linear1/ln1/
            # pooler），mlx-embeddings 不认 → 用本包的 MLX BERT 实现。
            from .mlx_bert import load_bert
            self._model, _cfg = load_bert(self._path, dtype=self._dtype)
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
        # jina-embeddings-v3 的 ONNX 导出需要在输入里额外传 task_id（runtime 选择
        # LoRA），映射表写在 config.json 的 task_id_map 里，提前读出来备用。
        self._jina_task_id_map = (
            self._read_jina_task_id_map() if self._family == "jina-v3-mlx" else {}
        )
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
    def _task_prefix(self, task: str) -> str:
        """任务 → 前缀。族表未定义的任务（如 nomic 的 clustering）回退 doc 前缀。"""
        return self._prefixes.get(task) or self._prefixes.get("doc", "") or ""

    def _jina_task(self, task: str) -> Optional[str]:
        """高层意图 → jina 模型 task 名（retrieval.passage / retrieval.query …）。

        非 jina 模型返回 None；jina 但不识别的任务也返回 None（模型退化到无 LoRA）。
        """
        return _JINA_MODEL_TASK.get(task)

    def _encode(self, texts: List[str], task: str = "doc") -> List[List[float]]:
        self._ensure_model()
        prefix = self._task_prefix(task)
        prefixed = [prefix + t if prefix and not t.startswith(prefix) else t
                    for t in texts]
        jina_task = self._jina_task(task)
        if self._backend == "mlx":
            if self._family == "jina-v3-mlx":
                return self._encode_mlx_jina(prefixed, jina_task)
            if self._family == "eurobert":
                return self._encode_mlx_eurobert(prefixed)
            return self._encode_mlx(prefixed)
        if self._backend == "onnx":
            if self._family == "jina-v3-mlx":
                return self._encode_onnx_jina(prefixed, jina_task)
            if self._prefix_family == "jina-v5":
                return self._encode_onnx_eurobert(prefixed)
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
        if self._family == "bert":
            from .mlx_bert import l2_normalize, mean_pool
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

    def _read_jina_task_id_map(self) -> Dict[str, int]:
        """从 config.json 读 jina ONNX 的 task→task_id 映射（缺省回退空表）。"""
        cfg_path = os.path.join(self._path, "config.json")
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                return json.load(f).get("task_id_map", {})
        except (OSError, json.JSONDecodeError):
            return {}

    def _encode_mlx_jina(self, texts: List[str], jina_task: Optional[str]) -> List[List[float]]:
        """jina-embeddings-v3（MLX）编码：tokenize → 模型按 task 选 LoRA → 输出。

        模型 encode() 内部已 mean-pool + L2 归一化（1024 维），这里原样返回；
        256 维截断 + 重归一化 + int8 量化由 vector_store 的 Matryoshka 流程统一做。
        """
        import numpy as np
        import mlx.core as mx
        enc = self._tokenizer(texts, padding=True, truncation=True,
                              max_length=self._max_length, return_tensors="np")
        input_ids = mx.array(enc["input_ids"])
        attention_mask = mx.array(enc["attention_mask"])
        out = self._model.encode(input_ids, attention_mask,
                                 truncate_dim=self._truncate_dim, task=jina_task)
        return [list(map(float, row)) for row in np.array(out, dtype=np.float32)]

    def _encode_mlx_eurobert(self, texts: List[str]) -> List[List[float]]:
        """jina-embeddings-v5-text-nano（MLX/eurobert）编码。

        模型内部 last-token 池化 + L2 归一化（768 维）；``truncate_dim`` 生效时
        截取前 N 维并重新归一化（MRL）。量化版已预合并 retrieval LoRA，
        query/passage 仅靠文本前缀（Query:/Document:）区分，无需运行时选任务。
        """
        import numpy as np
        import mlx.core as mx
        enc = self._tokenizer(texts, padding=True, truncation=True,
                              max_length=self._max_length, return_tensors="np")
        out = self._model.encode(mx.array(enc["input_ids"]),
                                 mx.array(enc["attention_mask"]),
                                 truncate_dim=self._truncate_dim)
        return [list(map(float, row)) for row in np.asarray(out, dtype=np.float32)]

    def _encode_onnx_jina(self, texts: List[str], jina_task: Optional[str]) -> List[List[float]]:
        """jina-embeddings-v3（ONNX）编码：额外传 task_id 选 LoRA。

        导出图已包含 mean-pool + L2 归一化，输出直接是 (B, 1024) 归一化向量，
        不要再二次池化；256 维截断 + 重归一化 + int8 交给 vector_store。
        """
        import numpy as np
        enc = self._tokenizer(texts, padding=True, truncation=True,
                              max_length=self._max_length, return_tensors="np")
        task_id = int(self._jina_task_id_map.get(jina_task, 0))
        feeds = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
            "task_id": np.array(task_id, dtype=np.int64),
        }
        out = self._session.run(None, feeds)[0]  # (B, 1024) 已归一化
        return [list(map(float, row)) for row in out]

    def _encode_onnx_eurobert(self, texts: List[str]) -> List[List[float]]:
        """jina-embeddings-v5-text-nano（ONNX）编码。

        导出图已包含 last-token 池化 + L2 归一化，输出直接是 (B, 768) 归一化向量，
        不要再二次池化；``truncate_dim`` 生效时截取前 N 维并重新归一化（MRL）。
        量化版（int4/int8）已预合并 retrieval LoRA，query/doc 仅靠文本前缀
        （Query:/Document:）区分，无需运行时选任务。
        """
        import numpy as np
        enc = self._tokenizer(texts, padding=True, truncation=True,
                              max_length=self._max_length, return_tensors="np")
        feeds = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        out = np.asarray(self._session.run(None, feeds)[0],
                         dtype=np.float32)  # (B, 768) 已 L2 归一化
        if self._truncate_dim:
            out = out[:, :self._truncate_dim]
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / np.clip(norms, 1e-12, None)
        return [list(map(float, row)) for row in out]

    # ------------------------------------------------------------------
    # public API（与 FastEmbedEmbeddingFunction 一致）
    # ------------------------------------------------------------------
    def __call__(self, input: List[str]) -> List[List[float]]:
        if not input:
            return []
        return self._encode(list(input), task="doc")

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text], task="query")[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self(texts)

    def embed_task(self, texts: List[str], task: str = "doc") -> List[List[float]]:
        """按任务编码（doc / query / clustering / classification）。

        族表未定义该任务时回退到 doc 前缀；未知名族（空前缀表）恒不加前缀。
        """
        if not texts:
            return []
        return self._encode(list(texts), task=task)

    def name(self) -> str:
        base = os.path.basename(self._path.rstrip("/")) or self._path
        return f"local:{self._backend}:{base}"
