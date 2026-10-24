"""Vector store — embedding + similarity search using LanceDB, with FastEmbed reranker.

Embeddings are generated locally via FastEmbed (ONNX Runtime) using
nomic-ai/nomic-embed-text-v1.5 — no API server or HuggingFace dependency.
Reranking uses FastEmbed's TextCrossEncoder (bge-reranker-v2-m3, also ONNX
Runtime) — same stack as embedding, no torch/FlagEmbedding dependency.
Vectors are stored in LanceDB (Lance columnar format on local disk).
"""
from __future__ import annotations

import functools
import hashlib
import inspect
import logging
import math
import os
import re
import threading
import time
import warnings
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default embedding model — quantized nomic-embed-text-v1.5 via FastEmbed (ONNX Runtime).
# Using the -Q variant: model_quantized.onnx (~137MB) instead of model.onnx (~548MB).
# Same 768-dim embeddings, same quality, 4x smaller download.
DEFAULT_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5-Q"
# 重排序走 fastembed TextCrossEncoder（ONNX，无 PyTorch 依赖）。选官方 INT8
# 量化版：~570MB（fp32 全量 2.3GB 的一半不到），且 v2-m3 的多语种/中文精排
# 能力优于 bge-reranker-base。旧配置里的 bge-reranker-base 仍被 fastembed
# 支持（fp32 onnx ~1.1GB），无需迁移。
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3-int8"
_TABLE_NAME = "knowledge_base"

# 文档元数据表：存储每个文档的缩略版本（目录结构 + 章节标题）与完整 Markdown，
# 用于 Meta-context + Targeted RAG（先返回全文档结构概览，再按需检索细节）。
_DOCUMENTS_TABLE_NAME = "documents"

# nomic-embed-text-v1.5 requires query/document prefixes for best results
_DOC_PREFIX = "search_document: "
_QUERY_PREFIX = "search_query: "

# BM25 全文检索的预分词列名（jieba 分词后空格连接，供原生 FTS 的 simple
# tokenizer 按空格切分索引，解决中文无空格无法分词的问题）。
_FTS_COLUMN = "text_fts"

# jieba 分词器缓存（进程级单例，避免每次调用重复初始化词典）。
_jieba = None
_jieba_lock = threading.Lock()
_jieba_tried = False


def _stable_sigmoid(x: float) -> float:
    """数值稳定的 sigmoid。

    ``1/(1+exp(-x))`` 在 x 为很大的负数时 ``exp(-x)`` 会 OverflowError
    （reranker 的原始 logits 完全可能到 -30 以下），异常被上层 except 吞掉后
    会静默回退到向量排序 —— 精排看起来"生效"了其实一次都没用。分支写法在两个
    方向上都只算一次 exp，不会溢出。
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _get_jieba():
    """Lazily import and return jieba, or None if unavailable."""
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
            logger.debug("jieba 分词器已加载（用于 BM25 中文全文检索）")
        except ImportError:
            _jieba = None
            logger.info("jieba 未安装，BM25 中文分词退化为空白切分")
    return _jieba


def _tokenize_for_fts(text: str) -> str:
    """Tokenize text for the BM25 full-text index.

    Uses jieba for CJK word segmentation, then joins tokens with spaces so the
    native FTS ``simple`` tokenizer can index each word.  English words are
    kept intact (jieba segments them on whitespace/punctuation).  When jieba
    is unavailable, returns the original text (English still works via
    whitespace; CJK degrades to whole-sentence tokens).
    """
    jieba = _get_jieba()
    if jieba is None or not text:
        return text or ""
    tokens = [t.strip() for t in jieba.cut(text) if t.strip()]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# FastEmbed-based embedding function (local ONNX Runtime, no API needed)
# ---------------------------------------------------------------------------


class FastEmbedEmbeddingFunction:
    """Embedding function using FastEmbed (ONNX Runtime) for local inference.

    Loads nomic-ai/nomic-embed-text-v1.5 via ONNX Runtime — no GPU, no
    HuggingFace network calls, no PyTorch dependency. The model is cached
    after first download (~130MB quantized).

    Thread-safety: ONNX Runtime InferenceSession.run() is thread-safe,
    so a single instance can be shared across RAG, graph_memory, etc.
    Use get_or_create_embedding_function() to get a shared singleton.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self._model_name = model_name
        self._model = None  # lazy-init

    def _ensure_model(self):
        if self._model is not None:
            return
        with _MODEL_LOAD_LOCK:
            if self._model is not None:  # double-check：并发方可能刚加载完
                return
            from fastembed import TextEmbedding
            logger.info("  🔧 正在加载嵌入模型: %s (首次加载, 约 130MB)...", self._model_name)
            import time as _time
            _time.sleep(0.01)  # 释放 GIL 让 UI 更新
            try:
                # Use a persistent cache dir instead of the default system
                # temp dir (/var/folders/.../T/) which macOS cleans regularly.
                # This prevents the model from being re-downloaded after
                # every system cleanup.
                cache_dir = os.path.join(
                    os.path.expanduser("~"),
                    ".cache",
                    "fastembed",
                )
                os.makedirs(cache_dir, exist_ok=True)
                # Suppress FastEmbed's "model has been updated on HuggingFace"
                # UserWarning — it fires on every instantiation and is not
                # actionable for the user.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    model = TextEmbedding(
                        model_name=self._model_name,
                        cache_dir=cache_dir,
                        # ORT intra-op 线程数：默认值在本机实测与 4 相当
                        # （57 vs 63 tok/s），留一半核心给 OCR/解析 worker。
                        # 嵌入吞吐的主要杠杆是长度分桶（见 _embed_chunks），
                        # 线程数影响很小（#122）。
                        threads=4,
                        # **显式强制 CPU-only**（见 _cpu_only_kwargs 注释）：
                        # 不传时 FastEmbed 的 ``cuda=Device.AUTO`` 只在该
                        # 环境没有 CUDAExecutionProvider 时才回退 CPU，装了
                        # onnxruntime-gpu 的机器上会静默切到 CUDA。
                        **_cpu_only_kwargs(TextEmbedding),
                    )
                self._model = model
            except Exception as e:
                # 模型加载失败最常见的原因是首次使用时需要从 HuggingFace
                # 下载 ONNX 模型文件（约 130MB），但网络无法访问 huggingface.co。
                # 给用户提供明确的错误信息和解决方案。
                msg = (
                    f"无法加载嵌入模型 '{self._model_name}'：{e}\n\n"
                    f"可能原因及解决方案：\n"
                    f"1. 首次使用需从 HuggingFace 下载模型（~130MB），网络无法访问\n"
                    f"   → 设置镜像站: export HF_ENDPOINT=https://hf-mirror.com\n"
                    f"2. 新版 huggingface_hub 默认使用 Xet 传输，镜像站可能不支持\n"
                    f"   → 禁用 Xet: export HF_HUB_DISABLE_XET=1\n"
                    f"3. 模型缓存目录权限不足\n"
                    f"   → 检查 ~/.cache/huggingface/ 目录权限\n"
                    f"4. 也可手动下载模型到本地缓存：\n"
                    f"   HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \\\n"
                    f"     python3 -c \"from huggingface_hub import snapshot_download; \\\n"
                    f"     snapshot_download('nomic-ai/nomic-embed-text-v1.5')\"\n\n"
                    f"原始错误: {type(e).__name__}: {e}"
                )
                logger.error(msg)
                raise RuntimeError(msg) from e
            logger.info("  ✅ 嵌入模型加载完成: %s", self._model_name)
            _time.sleep(0.01)  # 释放 GIL 让 UI 更新

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Embed a list of document texts. Returns a list of embedding vectors."""
        if not input:
            return []
        self._ensure_model()
        # Add document prefix for nomic-embed-text-v1.5
        prefixed = [_DOC_PREFIX + t if not t.startswith(_DOC_PREFIX) else t for t in input]
        embeddings = list(self._model.embed(prefixed))
        return [e.tolist() for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text with the query prefix."""
        self._ensure_model()
        prefixed = _QUERY_PREFIX + text if not text.startswith(_QUERY_PREFIX) else text
        embeddings = list(self._model.embed([prefixed]))
        return embeddings[0].tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document texts (with document prefix).

        Batched: a single ONNX forward pass for the whole list, much faster
        than calling embed_query() per text.
        """
        return self(texts)

    def name(self) -> str:
        return f"fastembed:{self._model_name}"

    def close(self) -> None:
        """释放 FastEmbed TextEmbedding（ONNX session ~137MB）。

        单例淘汰（embedding 模型配置变更）走这里显式释放；只摘字典引用
        的话要等 GC 才析构 C++ session，切换配置瞬间内存双份并存。
        """
        self._model = None


# ---------------------------------------------------------------------------
# Process-level embedding function singleton
# ---------------------------------------------------------------------------

_EF_SINGLETONS: Dict[str, "FastEmbedEmbeddingFunction"] = {}
_EF_SINGLETONS_LOCK = threading.Lock()

# 保护「检查表不存在 → create_table」的 check-then-act：两个 VectorStore 实例
# 共享同一 persist_dir（如 SyncKnowledgeWorker 与 agent 引擎交叠）时，无锁会让
# 第二个线程的 create_table 抛 "Table already exists"，该批次数据丢失。
_TABLE_CREATE_LOCK = threading.Lock()
# FTS 建索引的跨实例互斥（见 _ensure_fts_index 内注释）。
_FTS_CREATE_LOCK = threading.Lock()


def _store_op(func):
    """装饰 VectorStore 的公开方法：登记在途操作，与 clear()/close() 互斥。

    不装饰内部方法（``_*``）：它们由已登记的操作调用，重复计数会让
    clear() 永远等不到排空（自己等自己）。
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self._begin_op()
        try:
            return func(self, *args, **kwargs)
        finally:
            self._end_op()

    return wrapper


# ---------------------------------------------------------------------------
# 自适应嵌入子批大小控制器
# ---------------------------------------------------------------------------

# 子批大小探索范围与步长（#125，#126/#127/#128 生产验证后修订）：真实
# 切片上的实测是「吞吐与内存都随子批增大单调变差」——激活工作集 ∝ 子批
# ×批内最长序列，超出 CPU 缓存后被内存带宽卡死，「大批次摊薄开销」在此
# 不成立。真实 200 切片实测：size2 = 2071 tok/s @ 1.28GB，size4 = 1864 @
# 2.14GB，size12 = 1163 @ 4.99GB。调优范围按用户规格改为 [2, 64]，步进 2，
# 初始 16：下限 2 仍避开「单条最省内存」的过碎批，上限放开给长切片/高带宽
# 机器爬升空间，爬山步长 2 在更宽区间内更快收敛。
_SUBBATCH_MIN = 2
_SUBBATCH_MAX = 64
_SUBBATCH_START = 16
_SUBBATCH_STEP = 2
# RSS 软墙（#129，用户定参 8GB）：内存偶尔冲高是正常的（OCR 缓存、压缩
# 计费），只要求「超墙时降档避让、回墙下即停」——无闩锁。旧 5GB 闩锁版
# 会在真实同步基线（6-10GB）下首个子批就触发且永不解除，是「降到 1 再
# 也不回去」的根因之一。注意：wall 只影响运行中的进程，改这里不会让
# 已启动的应用生效，需要完全重启。
_SUBBATCH_RSS_LIMIT = 8.0  # GB
# 恢复阈值（#129）：monitor 下速率回到历史最优 95% 且当前尺寸低于最优
# 尺寸时，回到最优尺寸重新爬升——消除「降档单向棘轮」（降需连续 2 窗口
# <75%，回却需 >110%，一次短暂内存尖峰会永久降档）。低于 100% 是安全
# 的：回到最优尺寸后 probe 会重新验证，验证不过自然回退。
_SUBBATCH_RECOVER = 0.95
# 形状预算（#126 修订）：嵌入激活 ∝ 子批条数 × 批内最长序列²（attention 项
# 主导）。基准标定：size8 × 1311² ≈ 13.75M 单位 ≈ 4.7GB 激活（≈0.34GB/M
# 单位）——预算 6M 单位把瞬时激活压在 ~2GB。不用绝对 RSS 阈值：真实同步
# 进程基线本就有 6-10GB（OCR 原生缓存、LanceDB 等），绝对阈值会在首个子批
# 就触发降档并永久卡死在下限（实测「子批很快降到 1」的根因）。形状预算
# 只对超长文本钳制条数，与进程里其它组件的内存正交。
_SUBBATCH_SHAPE_BUDGET = 6_000_000  # 条 × 字符²


def _shape_cap_size(size: int, max_len: int) -> int:
    """按形状预算收缩子批条数：超长文本下自动变小，短文本不受影响。"""
    if max_len <= 0:
        return size
    budget = _SUBBATCH_SHAPE_BUDGET // (max_len * max_len)
    return max(1, min(size, budget))


# 每个窗口聚合的子批数：太小易被单批噪声带偏，太大响应迟钝。
_SUBBATCH_WINDOW = 2
# 爬升判定：候选子批的算力速率需比基线高 3% 才采纳——低于该阈值的波动
# 视为噪声，避免在吞吐平坦段来回抖动。
_SUBBATCH_IMPROVE = 1.03
# 降档判定：连续 2 个窗口的速率低于历史最优的 75% 视为环境劣化（OCR
# worker 抢占 CPU、热降频等），降 1 档避让，直到下限 1。
_SUBBATCH_DEGRADE = 0.75
_SUBBATCH_DEGRADE_STREAK = 2
# 历史最优速率的逐窗口衰减：环境永久变慢后旧纪录缓慢失效，不至于永远
# 拿旧峰值当标尺。
_SUBBATCH_BEST_DECAY = 0.999
# 重新爬升判定：monitor 阶段实测速率比历史最优高出 10%（如 OCR worker
# 结束、CPU 空出来）→ 环境已变，重新从小尺寸向上爬山，否则会永远停在
# 忙时降下来的小尺寸上。
_SUBBATCH_RE_CLIMB = 1.10


class _SubBatchTuner:
    """在 [2, 64] 内自动寻找吞吐最优的嵌入子批大小（爬山 + 降档）。

    度量：不用原始 tok/s——长度分桶后各批序列长度天然不同，直接比
    tok/s 有系统性偏差。改用算力速率 ``ops/s = n × max_len² / dt``（以
    字符数近似 token 数，单调代理即可），对批大小与批内最长长度同时
    归一，跨窗口、跨尺寸可比。

    状态机（#125 用户规格，#126/#127/#128 生产验证修订）：
      climb   初始 2：基线窗口（2 子批）→ 试探 +1 → 更快（≥3%）则采纳并
              继续上调，变慢则回退到上一个数字，进入 monitor。起步即落
              在实测最优区，步长 1 在吞吐平坦段也能精细寻优；
      monitor 持续盯速率：连续 2 窗口低于历史最优 75% → 降 1 避让（下限
              1），速率恢复即停；速率回到历史最优 95% 且尺寸低于最优尺寸 →
              尖峰已过去，回到最优尺寸重新爬升（#129，消除降档单向棘轮）；
              实测速率比历史最优高 10% → 环境变快，重新爬升。

    内存安全双保险（#126/#129）：
      ① 形状预算：_embed_chunks 组装每个子批时按 ``_shape_cap_size`` 钳制
         条数（激活 ∝ 条数 × 批内最长序列²），超长文本自动变小。绝对 RSS
         阈值在真实同步进程（基线 6-10GB）必然误触发，曾导致永久卡死。
      ② RSS 软墙 8GB（无闩锁）：超墙每子批 force_down 降 1 档，回墙下即
         停，恢复交给 monitor 恢复分支——旧 5GB 闩锁版是「降到 1 再也不
         回去」的根因。调优器 report() 收到的是钳制后的实际条数，状态机
         在可行域内正常爬山。

    预热丢弃（#126 修订）：首个完整窗口包含模型加载 / ONNX 会话初始化
    （可达 10s+），会把爬升基线拉到虚低，让热身的试探窗口显得「暴涨」
    被误采纳——第一个窗口只用于预热，丢弃不计入判定。

    线程安全：内部状态由锁保护（add() 与 add_streaming() 可能来自不同
    线程，窗口聚合不会交错损坏）。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.size = max(_SUBBATCH_MIN, min(_SUBBATCH_MAX, _SUBBATCH_START))
        self._phase = "climb"  # climb | monitor
        self._climb_state = "baseline"  # baseline | probe
        self._prev_size = self.size  # 爬升失败时回退到此处
        self._baseline_score: Optional[float] = None
        self._win_ops = 0.0
        self._win_dt = 0.0
        self._win_n = 0
        self._ema: Optional[float] = None
        self._best: Optional[float] = None
        # 历史最优速率对应的尺寸（#129）：恢复判定的参照——降档棘轮的解药
        self._best_size = self.size
        self._degrade_streak = 0
        self._warmed = False  # 预热窗口是否已丢弃

    def propose(self) -> int:
        """返回下一个子批应使用的大小。"""
        with self._lock:
            return self.size

    def report(self, n: int, max_len: int, dt: float) -> None:
        """上报一个子批的实测结果（条数、批内最长字符数、耗时）。"""
        ops = float(n) * float(max(max_len, 1)) ** 2
        with self._lock:
            self._win_ops += ops
            self._win_dt += max(dt, 1e-6)
            self._win_n += 1
            if self._win_n < _SUBBATCH_WINDOW:
                return
            score = self._win_ops / self._win_dt
            self._win_ops = 0.0
            self._win_dt = 0.0
            self._win_n = 0
            if not self._warmed:
                # 预热窗口丢弃：含模型加载/会话初始化，基线会虚低
                self._warmed = True
                return
            if self._phase == "climb":
                self._on_climb_window(score)
            else:
                self._on_monitor_window(score)

    def _on_climb_window(self, score: float) -> None:
        """爬升阶段窗口结果。调用时已持锁。"""
        if self._climb_state == "baseline":
            self._baseline_score = score
            nxt = self.size + _SUBBATCH_STEP
            if nxt > _SUBBATCH_MAX:
                self._phase = "monitor"
                self._best_size = self.size  # 操作点已重新验证（#129）
                return
            self._prev_size = self.size
            self.size = nxt
            self._climb_state = "probe"
            logger.info(
                "  ├─ 子批自适应: %d → %d (试探吞吐)", self._prev_size, nxt
            )
            return
        # probe 窗口结果：与基线比较
        assert self._baseline_score is not None
        ratio = score / self._baseline_score
        if ratio >= _SUBBATCH_IMPROVE:
            logger.info(
                "  ├─ 子批自适应: %d 吞吐 %+.0f%%, 采纳并继续上调",
                self.size, (ratio - 1) * 100,
            )
            self._baseline_score = score
            self._best_size = self.size  # 采纳 = 已验证（#129）
            nxt = self.size + _SUBBATCH_STEP
            if nxt > _SUBBATCH_MAX:
                self._phase = "monitor"
            else:
                self._prev_size = self.size
                self.size = nxt
        else:
            logger.info(
                "  ├─ 子批自适应: %d 吞吐 %+.0f%%, 回退到 %d",
                self.size, (ratio - 1) * 100, self._prev_size,
            )
            self.size = self._prev_size
            self._phase = "monitor"
            # 回退 = 大尺寸被否决，操作点锚在此处，防止恢复循环（#129）
            self._best_size = self.size

    def _on_monitor_window(self, score: float) -> None:
        """监控阶段窗口结果：盯劣化降档，盯变快重新爬升。调用时已持锁。"""
        self._ema = score if self._ema is None else self._ema * 0.5 + score * 0.5
        prev_best = self._best
        # 恢复（#129）：当前尺寸低于最优尺寸、且速率回到历史最优 95% →
        # 尖峰（内存压力 / 抢 CPU）已过去，直接回到最优尺寸（不走路爬山：
        # plateau 上试探必然持平 <3% 被回退，回退锚会把最优尺寸永久改小，
        # 恰好复现「降到 1 再也不回去」）。留在 monitor：若最优尺寸在新
        # 环境下确实过慢，降档路径会再次接管，自校正无循环。放在 best
        # 更新之前，避免恢复窗口把 best/best_size 就地改写成小尺寸。
        if (
            prev_best is not None
            and prev_best > 0
            and self._ema is not None
            and self.size < self._best_size
            and self._ema >= prev_best * _SUBBATCH_RECOVER
        ):
            logger.info(
                "  ├─ 子批自适应: 速率恢复到最优的 %.0f%%, %d → %d 回到最优尺寸",
                self._ema / prev_best * 100, self.size, self._best_size,
            )
            self.size = self._best_size
            self._ema = None
            self._degrade_streak = 0
            return
        if self._best is None or self._ema > self._best:
            self._best = self._ema
        else:
            self._best *= _SUBBATCH_BEST_DECAY
        # 环境明显变快（新 EMA 比旧纪录高 10%）→ 重新爬升
        if (
            prev_best is not None
            and self._ema is not None
            and prev_best > 0
            and self._ema >= prev_best * _SUBBATCH_RE_CLIMB
        ):
            logger.info(
                "  ├─ 子批自适应: 速率比历史最优高 %.0f%%, 环境变快, 重新爬升",
                (self._ema / prev_best - 1) * 100,
            )
            self._phase = "climb"
            self._climb_state = "baseline"
            self._baseline_score = None
            self._ema = None
            self._degrade_streak = 0
            return
        if self._ema is not None and self._best and self._ema < self._best * _SUBBATCH_DEGRADE:
            self._degrade_streak += 1
        else:
            self._degrade_streak = 0
        if self._degrade_streak >= _SUBBATCH_DEGRADE_STREAK and self.size > _SUBBATCH_MIN:
            new_size = max(_SUBBATCH_MIN, self.size - _SUBBATCH_STEP)
            logger.info(
                "  ├─ 子批自适应: 吞吐降至最优的 %.0f%%, %d → %d 避让",
                self._ema / self._best * 100, self.size, new_size,
            )
            self.size = new_size
            self._ema = None
            self._degrade_streak = 0

    def force_down(self) -> None:
        """RSS 软墙（#129）：降 1 档并丢弃半满窗口。

        与速率状态机独立，**无闩锁**：超墙时每个子批调一次（激活随之变小、
        RSS 自然回落），回墙下即停，之后由恢复分支 / 重爬把尺寸带回来。
        旧 5GB 闩锁版（mem_pressed 永不解除）会在真实同步基线 6-10GB 下
        首个子批就触发并永久卡死在下限，是「降到 1 再也不回去」的根因。
        """
        with self._lock:
            self._win_ops = 0.0
            self._win_dt = 0.0
            self._win_n = 0
            if self.size <= _SUBBATCH_MIN:
                return
            new_size = max(_SUBBATCH_MIN, self.size - _SUBBATCH_STEP)
            logger.info(
                "  ├─ 子批自适应: RSS 超 %.0fGB, %d → %d 降档避让",
                _SUBBATCH_RSS_LIMIT, self.size, new_size,
            )
            self.size = new_size
            self._phase = "monitor"
            self._ema = None
            self._degrade_streak = 0


# ---------------------------------------------------------------------------
# 本地模型 provider 开关：embedding / reranker 可切换为「应用内加载本地权重」。
# 由 local.factory.configure_local_models(config) 在初始化时写入一次。
# ---------------------------------------------------------------------------
_EMBED_PROVIDER: Dict[str, Any] = {"provider": "api", "path": "", "backend": "mlx"}
_RERANK_PROVIDER: Dict[str, Any] = {"provider": "api", "path": "", "backend": "mlx"}


def _evict_local_singletons(key_prefix: str) -> int:
    """淘汰并释放某一类本地模型单例（provider 配置变更时调用）。

    ``set_*_provider`` 改变进程级开关后，旧 path/backend 对应的本地实例
    （ONNX InferenceSession ~137MB / MLX 权重 GB 级）会永久滞留
    ``_EF_SINGLETONS`` 直到进程退出——旧实现如此，用户每次在「应用配置」
    里换 embedding/reranker 模型就泄漏一份权重。本地单例语义上是「当前
    配置」的进程级共享实例，配置变更即应全部淘汰。
    """
    with _EF_SINGLETONS_LOCK:
        keys = [k for k in list(_EF_SINGLETONS)
                if isinstance(k, tuple) and k and k[0] == key_prefix]
        for k in keys:
            obj = _EF_SINGLETONS.pop(k, None)
            closer = getattr(obj, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass
    return len(keys)


def set_embedding_provider(provider: str, path: str = "", backend: str = "mlx") -> None:
    """配置 embedding provider：``api``（fastembed 网络）或 ``local``（本地权重）。"""
    new = {"provider": provider or "api", "path": path or "", "backend": backend or "mlx"}
    if new != _EMBED_PROVIDER:
        # 配置真的变了 → 旧本地单例（GB 级权重）立即淘汰，不留给进程退出
        _evict_local_singletons("__local_embed__")
    _EMBED_PROVIDER.update(new)


def set_reranker_provider(provider: str, path: str = "", backend: str = "mlx") -> None:
    """配置 reranker provider：``api``（fastembed cross-encoder）或 ``local``。"""
    new = {"provider": provider or "api", "path": path or "", "backend": backend or "mlx"}
    if new != _RERANK_PROVIDER:
        _evict_local_singletons("__local_rerank__")
    _RERANK_PROVIDER.update(new)


def reset_local_providers() -> None:
    """把 embedding / reranker provider 复位到默认（``api``）。

    ``set_*_provider`` 写的是**进程级模块状态**：GUI 初始化（或任何调用
    ``configure_local_models`` 的代码路径）一旦切到 ``local``，后续所有
    ``VectorStore`` 实例——包括测试里显式用 fastembed 的实例——都会拿到
    ``LocalEmbeddingFunction`` / ``LocalReranker``。本函数供测试夹具与
    「恢复默认配置」路径调用，避免全局开关跨上下文泄漏。
    """
    _EMBED_PROVIDER.update(provider="api", path="", backend="mlx")
    _RERANK_PROVIDER.update(provider="api", path="", backend="mlx")


def get_or_create_embedding_function(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> Any:
    """Return a process-wide shared embedding function.

    当 ``_EMBED_PROVIDER["provider"] == "local"`` 时返回本地加载的
    ``LocalEmbeddingFunction``（与 ``FastEmbedEmbeddingFunction`` 同接口）；否则
    走原有 fastembed 路径。两者都以 ``model_name`` 为键缓存单例，避免重复加载
    ~GB 级权重。

    The ONNX model (~137MB) is loaded once and shared across all callers
    (RAG VectorStore, graph_memory GraphMemoryStore, etc.).  ONNX Runtime
    InferenceSession.run() is thread-safe, so concurrent callers can share
    a single model instance without locking on inference.

    This avoids loading multiple copies of the same ~137MB model when both
    RAG and graph_memory are active (which previously cost ~274MB).
    """
    if _EMBED_PROVIDER["provider"] == "local":
        key = ("__local_embed__", _EMBED_PROVIDER["path"] or "default",
               _EMBED_PROVIDER["backend"])
        with _EF_SINGLETONS_LOCK:
            ef = _EF_SINGLETONS.get(key)
            if ef is None:
                from ..local.embedding import LocalEmbeddingFunction
                from ..local import registry
                path = registry.resolve_model_path("embedding", _EMBED_PROVIDER["path"])
                ef = LocalEmbeddingFunction(path, backend=_EMBED_PROVIDER["backend"])
                _EF_SINGLETONS[key] = ef
            return ef
    with _EF_SINGLETONS_LOCK:
        ef = _EF_SINGLETONS.get(model_name)
        if ef is None:
            ef = FastEmbedEmbeddingFunction(model_name=model_name)
            _EF_SINGLETONS[model_name] = ef
        return ef


def get_or_create_reranker() -> Any:
    """返回与 fastembed ``TextCrossEncoder`` 同接口（``.rerank(query, docs)``）的精排器。

    当 provider 为 ``local`` 时返回进程级单例 ``LocalReranker``；否则返回 None，
    交由 ``VectorStore._get_reranker`` 走原有 fastembed 后台加载流程。
    """
    if _RERANK_PROVIDER["provider"] != "local":
        return None
    key = ("__local_rerank__", _RERANK_PROVIDER["path"] or "default",
           _RERANK_PROVIDER["backend"])
    with _EF_SINGLETONS_LOCK:
        rk = _EF_SINGLETONS.get(key)
        if rk is None:
            from ..local.reranker import LocalReranker
            from ..local import registry
            path = registry.resolve_model_path("reranker", _RERANK_PROVIDER["path"])
            rk = LocalReranker(path, backend=_RERANK_PROVIDER["backend"])
            _EF_SINGLETONS[key] = rk
        return rk


# 懒加载互斥：_ensure_model 的「if self._model is None → TextEmbedding(...)」
# 之间无锁，ingest worker / 主线程 add_streaming / agent 检索 / graph_memory /
# skill_retriever 首次并发调用会各自加载一份 ~137MB ONNX 模型（后写者覆盖
# 实例属性，先加载那份纯泄漏）。加载完成前的 GIL 释放点（_time.sleep）正好
# 放大这个窗口。进程级锁 + 双重检查。
_MODEL_LOAD_LOCK = threading.Lock()


def _cpu_only_kwargs(cls) -> Dict[str, Any]:
    """返回强制 CPU-only 所需的构造参数（当前 fastembed 不支持时返回 {}）。

    显式传 ``providers=["CPUExecutionProvider"]`` 让 ONNX Runtime 的 EP 选择
    不依赖 FastEmbed 的默认回退——否则一旦环境里装了 onnxruntime-gpu，
    ``cuda=Device.AUTO`` 会静默切到 CUDA，与 OCR/解析全程 CPU 的设计不符，
    同一向量库在不同机器上的数值行为也会漂移。

    ``providers`` 参数在较老的 fastembed 上不存在（poetry.lock 仍锁 0.8.0，
    而 pyproject 已要求 >=0.9.0）。用签名探测而不是 try/except TypeError：
    加载失败的原因很多，把「参数不支持」和「模型下载失败」混在同一个异常
    分支里会让错误信息误导用户。探测不到就退化为不传（旧版本行为不变）。
    """
    try:
        import inspect
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover - 内置/C 实现类
        return {}
    if "providers" not in params:
        return {}
    return {"providers": ["CPUExecutionProvider"]}


# ---------------------------------------------------------------------------
# Vector store backed by LanceDB
# ---------------------------------------------------------------------------


def _call_on_batch(on_batch, n_added: int, total: int,
                   dt_embed: float, dt_store: float) -> None:
    """回调批次进度：**兼容两种签名**。

    新签名 ``on_batch(n_added, total, timing)`` 能拿到本批的向量化（embed）
    与入库（table.add）耗时——同步日志靠它按文件分摊出「哪一段最慢」；旧的
    两参数回调继续按原样调用（第三方/旧调用方零影响）。
    """
    if on_batch is None:
        return
    try:
        if len(inspect.signature(on_batch).parameters) >= 3:
            on_batch(n_added, total, {"embed": dt_embed, "store": dt_store})
            return
    except (TypeError, ValueError):
        pass
    on_batch(n_added, total)


class VectorStore:
    """Local vector store backed by LanceDB with FastEmbed embeddings.

    Uses nomic-ai/nomic-embed-text-v1.5 via ONNX Runtime for local inference.
    Supports optional FastEmbed reranker (bge-reranker-v2-m3) for precise
    post-retrieval scoring.
    """

    def __init__(
        self,
        persist_dir: str,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        rerank_model: str = DEFAULT_RERANK_MODEL,
        embedding_function: Optional[Any] = None,
    ):
        self._persist_dir = persist_dir
        self._embedding_model = embedding_model
        self._rerank_model = rerank_model
        self._custom_ef = embedding_function  # for testing injection
        self._db = None
        self._table = None
        self._documents_table = None  # 文档元数据表（缩略版本 + markdown）
        # 保护 documents 表懒创建 + 写操作。用 RLock：upsert_document /
        # delete_document 会「持锁后再调用 _ensure_documents_table()」，普通
        # Lock 会在此自锁死（见 _ensure_documents_table 的注释）。
        self._documents_lock = threading.RLock()
        # 保护整个 store 的懒初始化（连接/建表/schema 迁移只允许一次）。
        self._init_lock = threading.RLock()
        # 保护 reranker 状态机（避免并发加载两份 ~500MB 模型）。
        self._reranker_lock = threading.RLock()
        # reranker 输出是否为原始 logits（需 sigmoid 归一化）。首次打分时探测
        # 一次并缓存，之后按模型固定处理 —— 见 rerank() 内注释。
        self._reranker_is_logits: Optional[bool] = None
        # close() 后置位：后台加载线程完成时不得把模型写回本实例。
        self._closed = False
        # ---- 在途操作计数：与 clear()/close() 互斥 ----
        # clear() 会 drop_table、close() 会把 _db 置 None，二者若与在途的
        # add/search 并发，在途操作会打到已删除的表或 None 连接上（表现是
        # 随机 AttributeError / 检索到已删数据）。这里用在途计数 + 条件变量：
        # 业务操作进出各记一次，clear/close 先排空在途操作再动手。
        self._op_cv = threading.Condition(threading.Lock())
        self._inflight = 0
        self._closing = False
        # documents 表内容缓存：避免 get_document_digest / list_documents 每次
        # 都全表 to_pylist()（表内含完整 markdown，反复全量扫描既慢又占内存）。
        # 写入（upsert/delete/clear）时置 None 失效，下次读取时重新加载。
        self._documents_cache: Optional[List[Dict[str, Any]]] = None
        self._ef = None  # embedding function
        self._subbatch_tuner = _SubBatchTuner()  # 自适应子批大小（#125）
        self._initialized = False
        self._table_checked = False  # 是否已检查过 table_names（避免重复磁盘扫描）
        self._fts_ready = False  # BM25 全文索引是否已就绪
        self._fts_checked = False  # 是否已尝试创建过 FTS 索引
        self._fts_failed = False  # 建索引曾失败 → 不再重试（见 _ensure_fts_index）

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 在途操作计数（与 clear/close 互斥）
    # ------------------------------------------------------------------

    def _begin_op(self) -> None:
        """登记一次在途操作。store 已关闭/正在关闭时抛明确的错误。

        ``_closing`` 必须一起检查：``clear()`` 排空完在途操作后会执行
        drop_table，若此时新操作仍能进来，就正好撞上这次改动想消除的竞态
        （读到/写到一张已被删除的表）。旧实现只检查 ``_closed``，漏了它。
        """
        with self._op_cv:
            if self._closed:
                raise RuntimeError("VectorStore is closed")
            if self._closing:
                raise RuntimeError("VectorStore is shutting down (clear in progress)")
            self._inflight += 1

    def _end_op(self) -> None:
        with self._op_cv:
            self._inflight = max(0, self._inflight - 1)
            if self._inflight == 0:
                self._op_cv.notify_all()

    def _drain_ops(self, timeout: float = 10.0) -> bool:
        """排空在途操作，期间拒绝新的操作。返回 True 表示已排空。

        超时后仍会继续：关闭/清空不能因为某个卡住的操作（例如慢 embedding）
        而永久挂起，宁可冒一次竞态也不能把 UI 卡死。
        """
        deadline = time.monotonic() + timeout
        with self._op_cv:
            self._closing = True
            while self._inflight > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning(
                        "VectorStore: %d 个在途操作未在 %.0fs 内结束，将继续关闭",
                        self._inflight, timeout,
                    )
                    return False
                self._op_cv.wait(remaining)
            return True

    def _resume_ops(self) -> None:
        """clear() 结束后解除「拒绝新操作」状态。"""
        with self._op_cv:
            self._closing = False
            self._op_cv.notify_all()

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        # ingest 的 4 个 worker 会并发走到这里（upsert_document / add_streaming
        # 都会触发），若不加锁，多个线程会同时执行 lancedb.connect /
        # create_table / schema 迁移，轻则重复建表报错，重则 schema 迁移互相
        # 覆盖。与 _ensure_documents_table 相同的双重检查模式。
        with self._init_lock:
            if self._initialized:
                return
            self._initialize_locked()

    def _initialize_locked(self) -> None:
        """``_ensure_initialized`` 的实现，调用方必须已持有 _init_lock。"""
        logger.info("  正在初始化向量库...")
        try:
            import lancedb
        except ImportError:
            raise ImportError(
                "lancedb is required for RAG vector storage. pip install lancedb"
            )

        os.makedirs(self._persist_dir, exist_ok=True)

        logger.info("  连接 LanceDB: %s", self._persist_dir)
        self._db = lancedb.connect(self._persist_dir)
        # 实例可以被 close() 之后重新初始化（_initialized 会被置回 False）。
        # 不复位 _closed 的话，后台 reranker 加载线程一看到它就丢弃模型并
        # return —— 重排对该实例永久失效，且没有任何日志。
        self._closed = False
        # _closing 同理：close() 的 _drain_ops() 会把它置 True 且不再复位
        # （close 后本应永久拒绝新操作）。但本引擎的实例支持复活重连，
        # 不复位会让复活后的实例被 _begin_op 永久拒绝（「clear in progress」）。
        self._closing = False

        # Use injected embedding function (for testing) or create FastEmbed one
        if self._custom_ef is not None:
            self._ef = self._custom_ef
        else:
            # Use the process-wide shared singleton to avoid loading
            # multiple copies of the ~137MB ONNX model (RAG + graph_memory).
            self._ef = get_or_create_embedding_function(self._embedding_model)

        # Open existing table or defer creation until first add
        if _TABLE_NAME in self._db.table_names():
            self._table = self._db.open_table(_TABLE_NAME)
            self._check_schema_migration()

        # Open the documents metadata table (if it already exists).
        if _DOCUMENTS_TABLE_NAME in self._db.table_names():
            self._documents_table = self._db.open_table(_DOCUMENTS_TABLE_NAME)
            self._migrate_documents_schema()

        self._initialized = True
        count = self._table.count_rows() if self._table is not None else 0
        logger.info("  向量库初始化完成: 当前存储 %d 条向量记录", count)

    def _check_schema_migration(self) -> None:
        """Migrate an existing table to include the ``text_fts`` column.

        The BM25 full-text index lives on a pre-tokenized ``text_fts`` column.
        Tables created before this feature lack the column; since LanceDB
        schemas are immutable, we rebuild the table (data survives — it is
        re-read from the old table and re-written with the new column).
        """
        if self._table is None:
            return
        try:
            schema = self._table.schema
            if _FTS_COLUMN in schema.names:
                return
        except Exception:
            return

        logger.info("  ⚙ 检测到旧版向量表（缺少 %s 列），自动迁移...", _FTS_COLUMN)
        try:
            # Read all existing rows via PyArrow (无 pandas 依赖)。
            arrow_table = self._table.to_arrow()
            records = []
            for row in arrow_table.to_pylist():
                rec = {
                    "id": row["id"],
                    "text": row["text"],
                    "vector": row["vector"],
                    "source": row["source"],
                    "chunk_index": row["chunk_index"],
                }
                rec[_FTS_COLUMN] = _tokenize_for_fts(rec["text"])
                records.append(rec)

            # LanceDB OSS 无 rename_table，只能 drop 后重建。数据已在内存
            # 的 records 中，drop 旧表后 create 失败可再次尝试；即使最终
            # 失败，RAG 数据也可从源文件重新同步恢复（幂等）。
            self._db.drop_table(_TABLE_NAME)
            self._table = None
            if records:
                self._table = self._db.create_table(_TABLE_NAME, records)
            self._table_checked = True
            logger.info("  ⚙ 向量表迁移完成（新增 %s 列，%d 条记录）", _FTS_COLUMN, len(records))
        except Exception as e:
            logger.warning("  ⚠ 向量表迁移失败（BM25 索引将不可用）: %s", e)
            self._table = None
            self._table_checked = False

    def _ensure_table(self) -> Optional[Any]:
        """Open existing table if it exists, return None if not yet created.

        Re-checks ``table_names()`` on every call when ``_table`` is None,
        because the table may have been created by another RAGEngine instance
        (e.g. SyncKnowledgeWorker) after this VectorStore was first initialized.
        Caching "table doesn't exist" was a premature optimization that caused
        a real bug: after sync wrote data via a separate engine, this engine's
        search() still returned empty because it never re-checked.

        LanceDB's ``table_names()`` is cheap (directory listing), so the
        per-call cost is acceptable for correctness.
        """
        if self._table is not None:
            return self._table
        if _TABLE_NAME in self._db.table_names():
            self._table = self._db.open_table(_TABLE_NAME)
            self._table_checked = True
        return self._table

    def _create_or_open_table(self, first_records: List[Dict[str, Any]]):
        """竞态安全的建表：返回 ``(table, created)``。

        锁内重查 —— 另一实例/线程可能已建好；竞争失败时打开已存在的表并返回
        ``created=False``，让调用方改走 upsert 路径，而不是抛异常丢数据。
        """
        with _TABLE_CREATE_LOCK:
            table = self._ensure_table()
            if table is not None:
                return table, False
            try:
                self._table = self._db.create_table(_TABLE_NAME, first_records)
                self._table_checked = True
                return self._table, True
            except Exception as e:
                if "exist" in str(e).lower():
                    try:
                        self._table = self._db.open_table(_TABLE_NAME)
                        self._table_checked = True
                        return self._table, False
                    except Exception:
                        pass
                raise

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    @_store_op
    def add(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """Add document chunks to the vector store. Returns number of chunks added.

        Upsert 语义：先按 source 删除该来源的全部旧记录（处理文档缩减后
        残留旧 chunks 的情况），再插入新记录。embedding 分批进行以避免 OOM。
        """
        self._ensure_initialized()
        if not chunks:
            return 0

        # Build records (embed in batches to avoid OOM on large knowledge bases)
        records: List[Dict[str, Any]] = []
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            texts = [c["text"] for c in batch_chunks]
            vectors = self._ef(texts)
            for j, chunk in enumerate(batch_chunks):
                idx = i + j
                source = chunk.get("source", "unknown")
                chunk_idx = chunk.get("chunk_index", idx)
                chunk_id = f"{_hash_source(source)}_{chunk_idx}"
                records.append({
                    "id": chunk_id,
                    "text": chunk["text"],
                    "vector": _normalize(vectors[j]),
                    "source": source,
                    "chunk_index": chunk_idx,
                    _FTS_COLUMN: _tokenize_for_fts(chunk["text"]),
                })

        table = self._ensure_table()

        created = False
        if table is None:
            # First add: create table (race-safe — another instance may have won)
            table, created = self._create_or_open_table(records)
        if created:
            logger.info("LanceDB table '%s' created with %d records", _TABLE_NAME, len(records))
            return len(records)

        # Upsert: 按 source 删除该来源的全部旧记录。
        # 用 source 的 md5 hash 作为 LIKE 模式（hash 是 hex 字符串，无 SQL 注入风险），
        # 既能覆盖 chunk_index 0..N 的所有旧记录，也能清理文档缩减后多出的旧 chunks。
        sources_to_replace = sorted({c.get("source", "unknown") for c in chunks})
        for source in sources_to_replace:
            h = _hash_source(source)
            try:
                # id 格式为 "{hash}_{chunk_index}"，LIKE '{hash}_%' 匹配该 source 的所有 chunk
                table.delete(f"id LIKE '{h}_%'")
            except Exception as e:
                logger.debug("VectorStore: delete by source '%s' failed: %s", source, e)

        # Add records in batches（单批失败不中断，最大限度保留数据）
        total = 0
        failed_batches = 0
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            try:
                table.add(batch)
                total += len(batch)
            except Exception as e:
                failed_batches += 1
                logger.error(
                    "VectorStore: add batch %d/%d failed: %s",
                    i // batch_size + 1, (len(records) + batch_size - 1) // batch_size, e,
                )
                if failed_batches >= 3:
                    # 连续失败太多，可能系统性问题，中断避免持续报错
                    raise
        if failed_batches > 0:
            logger.warning("VectorStore: %d batch(es) failed during add", failed_batches)

        # 数据写入后重建 BM25 全文索引，覆盖新增的行。
        self._fts_ready = False
        self._ensure_fts_index(self._table)

        logger.info("VectorStore: added %d chunks", total)
        return total

    @_store_op
    def delete_by_source(self, source: str) -> int:
        """删除指定 source 的全部记录。返回删除的记录数（无法统计时为 1/0）。

        用于 force 重新 ingest 时清理已从知识库中删除的文件对应的旧向量。
        旧实现 docstring 声称返回删除前记录数，实际恒返回 1 —— 与语义不符。
        LanceDB 的 delete 不返回行数，无法精确统计；用前后 count 差值尽力
        估计（并发写入时可能不准），失败为 0。
        """
        self._ensure_initialized()
        table = self._ensure_table()
        if table is None:
            return 0
        h = _hash_source(source)
        try:
            try:
                before = table.count_rows()
            except Exception:
                before = None
            table.delete(f"id LIKE '{h}_%'")
            # 向量删了，但 BM25 倒排索引里的 term 还在——必须失效索引，
            # 否则已删除文档的内容仍会被关键词检索召回并返回给 LLM。
            self._fts_ready = False
            logger.info("VectorStore: deleted records for source '%s'", source)
            if before is None:
                return 1
            try:
                after = table.count_rows()
            except Exception:
                return 1
            return max(0, before - after)
        except Exception as e:
            # 删除失败必须抛出而不是吞掉：engine._cleanup_deleted_files 依赖
            # 「无异常 = 删除成功」决定是否从 manifest 移除条目——吞掉异常会让
            # 失败文件被无条件 pop，向量永久残留且下次同步没有重试机会。
            logger.warning("VectorStore: delete_by_source '%s' failed: %s", source, e)
            raise

    @_store_op
    def deduplicate_vectors(
        self,
        order_key: Optional[Any] = None,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, int]:
        """清理内容完全相同的重复向量：每组只保留一份，其余删除。

        分组键 = 归一化文本（strip + 内部空白折叠）的 sha256——**内容一字
        不差**才算重复（不同文档里的相同样板文字，如纯「目录」页，同样只留
        一份，降低检索噪声）；近似重复不处理，避免误删。

        组内胜者由 ``order_key(record)`` 取最小决定：引擎侧传入
        「markdown 缓存 mtime → source → chunk_index」，即保留**入库时间早**
        的一份；order_key 缺省时按 (source, chunk_index) 字典序兜底。

        Args:
            order_key: 可选 ``record -> tuple``，值越小越优先保留。
            progress_callback: 可选 ``(cur, total, msg) -> None`` 扫描进度。

        Returns:
            ``{"scanned": 扫描条数, "groups": 有重复的组数, "removed": 删除条数}``
        """
        self._ensure_initialized()
        table = self._ensure_table()
        if table is None:
            return {"scanned": 0, "groups": 0, "removed": 0}
        try:
            total = table.count_rows()
        except Exception:
            total = 0
        if total == 0:
            return {"scanned": 0, "groups": 0, "removed": 0}
        # key -> [best_order, best_id, 所有见过的 id]。胜者可能后出现（入库
        # 更早的文档扫描序不保证在前），所以每组记住全部 id，最后统一删。
        groups: Dict[str, list] = {}
        _BATCH = 2000
        _COLS = ["id", "text", "source", "chunk_index"]
        scanned = 0
        while scanned < total:
            # 只投影四列（不加载 768 维向量）：search(None) 是全表扫描，
            # select 做列裁剪，limit/offset 分页（lancedb 0.40 的
            # to_arrow() 不接受任何参数，只能走 query builder）。
            rows = table.search(None).select(_COLS).limit(_BATCH).offset(
                scanned).to_arrow().to_pylist()
            if not rows:
                break
            for r in rows:
                text = re.sub(r"\s+", " ", (r.get("text") or "")).strip()
                if not text:
                    continue
                key = hashlib.sha256(text.encode("utf-8")).hexdigest()
                ok = (order_key(r) if order_key is not None
                      else (r.get("source") or "", r.get("chunk_index") or 0))
                g = groups.get(key)
                if g is None:
                    groups[key] = [ok, r["id"], [r["id"]]]
                else:
                    g[2].append(r["id"])
                    if ok < g[0]:
                        g[0], g[1] = ok, r["id"]
            scanned += len(rows)
            if progress_callback is not None:
                try:
                    progress_callback(scanned, total,
                                      f"已扫描 {scanned}/{total} 条向量")
                except Exception as e:
                    logger.warning("VectorStore: dedup progress error: %s", e)
        to_delete: List[str] = []
        dup_groups = 0
        for g in groups.values():
            if len(g[2]) > 1:
                dup_groups += 1
                to_delete.extend(i for i in g[2] if i != g[1])
        removed = 0
        if to_delete:
            # 分批 IN 删除。id 形如 "{md5}_{idx}"（十六进制+下划线+数字），
            # 无引号注入面；只做单引号翻倍，**不能**用 _escape_like（它会转义
            # 下划线，在 IN 列表里反而匹配不上）。
            def _lit(x: Any) -> str:
                return "'" + str(x).replace("'", "''") + "'"

            for i in range(0, len(to_delete), 500):
                batch = to_delete[i:i + 500]
                table.delete("id IN (" + ", ".join(_lit(x) for x in batch) + ")")
                removed += len(batch)
            # BM25 倒排索引里的 term 还在——失效重建，否则已删内容仍会被
            # 关键词检索召回（与 delete_by_source 同一理由）。
            self._fts_ready = False
            self._ensure_fts_index(self._table)
            logger.info("VectorStore: 向量去重完成: %d 组重复, 删除 %d 条",
                        dup_groups, removed)
        return {"scanned": scanned, "groups": dup_groups, "removed": removed}

    @_store_op
    def add_streaming(  # noqa: C901 - 分批写入的控制流较长，保持线性可读
        self,
        chunk_iter,
        batch_size: int = 100,
        on_batch: Optional[Any] = None,
        is_cancelled: Optional[Any] = None,
    ) -> int:
        """流式增量写入：接受 chunk 迭代器，每批 embed 完立即写入并释放。

        用于大知识库 ingest，避免 H3 隐患（全部 records 含 vector 同时在内存）。
        内存占用 = O(batch_size) 而非 O(总 chunks 数)。

        Args:
            chunk_iter: 产出 chunk dict 的迭代器（必须有 source / chunk_index / text）
            batch_size: 每批 embed + 写入的 chunk 数
            on_batch: 可选回调 (batch_added, total_added) -> None
            is_cancelled: 可选回调 () -> bool，返回 True 时中断写入并返回当前计数

        Returns:
            成功写入的 chunk 总数（取消时返回已写入数）
        """
        self._ensure_initialized()

        # 第一批特殊处理：用于判断是否需要 create_table
        first_batch: List[Dict[str, Any]] = []
        exhausted = False
        for _ in range(batch_size):
            try:
                first_batch.append(next(chunk_iter))
            except StopIteration:
                exhausted = True
                break

        if not first_batch:
            logger.info("  向量化写入: 无切片需要写入（空迭代器）")
            return 0

        # 第一批处理前检查取消标志
        if is_cancelled is not None and is_cancelled():
            return 0

        # Embed 第一批
        logger.info("  ├─ 向量化: 首批 %d 个切片正在嵌入...", len(first_batch))
        _t_emb0 = time.monotonic()
        first_records = self._embed_chunks(first_batch, 0)
        _dt_emb0 = time.monotonic() - _t_emb0
        logger.info("  ├─ 向量化完成: 首批 %d 个切片", len(first_records))

        table = self._ensure_table()
        # 跟踪已删除过旧数据的 source，避免后续批次重复写入时新旧数据共存。
        sources_seen: set = set()
        created = False
        if table is None:
            # 首次：创建表（用第一批 records），无需删除旧数据。竞态安全：
            # 若另一实例已建好，created=False，改走下面的「已有表」删除+写入。
            table, created = self._create_or_open_table(first_records)
        if created:
            logger.info("  ├─ 向量库: 新建 LanceDB 表 '%s', 写入 %d 条记录", _TABLE_NAME, len(first_records))
            total = len(first_records)
            sources_seen = {r["source"] for r in first_records}
            if on_batch:
                # 建表即写入，embed 单独计时、store 计入建表耗时
                _call_on_batch(on_batch, len(first_records), total,
                               _dt_emb0, time.monotonic() - _t_emb0 - _dt_emb0)
            if exhausted:
                logger.info("  └─ 向量化写入完成: 共 %d 个切片 (仅一批)", total)
                self._fts_ready = False
                self._ensure_fts_index(self._table)
                return total
        else:
            # 已有表（或并发竞争后打开）：先按 source 删除旧记录
            existing_count = table.count_rows()
            sources_seen = {r["source"] for r in first_records}
            logger.info(
                "  ├─ 向量库: 已有表 (%d 行), 将替换 %d 个来源的旧数据",
                existing_count, len(sources_seen),
            )
            for source in sources_seen:
                h = _hash_source(source)
                try:
                    table.delete(f"id LIKE '{h}_%'")
                except Exception as e:
                    logger.debug("VectorStore: delete by source '%s' failed: %s", source, e)
            # 写入第一批
            try:
                _t_add = time.monotonic()
                table.add(first_records)
                _dt_add = time.monotonic() - _t_add
                total = len(first_records)
                logger.info("  ├─ 向量库写入: 首批 %d 条记录", total)
                if on_batch:
                    _call_on_batch(on_batch, len(first_records), total,
                                   _dt_emb0, _dt_add)
            except Exception as e:
                logger.error("  向量库写入失败(首批): %s", e)
                self._fts_ready = False
                raise
            if exhausted:
                logger.info("  └─ 向量化写入完成: 共 %d 个切片 (仅一批)", total)
                # 单批写入也要重建 BM25 索引——否则这批新数据永远查不到，
                # 混合检索会静默退化成纯向量检索（无报错）。
                self._fts_ready = False
                self._ensure_fts_index(self._table)
                return total

        # 后续批次
        chunk_idx = len(first_batch)
        batch_buf: List[Dict[str, Any]] = []
        batch_num = 1
        deleted_sources: set = sources_seen.copy() if table is not None else set()
        for chunk in chunk_iter:
            batch_buf.append(chunk)
            if len(batch_buf) >= batch_size:
                if is_cancelled is not None and is_cancelled():
                    # 已 delete 旧数据/写入部分新数据，BM25 索引却是旧快照：
                    # 置失效让下次 search_fts 自愈重建，否则新数据在 BM25 通道静默不可见。
                    self._fts_ready = False
                    return total
                batch_num += 1
                logger.info("  ├─ 向量化: 第 %d 批 (%d 切片) 正在嵌入...", batch_num, len(batch_buf))
                # 先 embed 再 delete+add（与首批同序）：旧实现先 delete 旧数据
                # 再 embed 再 add——embed 耗时数秒到数分钟，这段时间里该
                # source 在库里被删空，并发检索会返回空（「文档凭空消失」）。
                # 调整后，删旧与写新背靠背执行，空窗缩到一次 table.add 的耗时。
                _t_e = time.monotonic()
                records = self._embed_chunks(batch_buf, chunk_idx)
                _dt_e = time.monotonic() - _t_e
                if table is not None:
                    new_sources = {c["source"] for c in batch_buf} - deleted_sources
                    if new_sources:
                        for source in new_sources:
                            h = _hash_source(source)
                            try:
                                table.delete(f"id LIKE '{h}_%'")
                            except Exception:
                                pass
                        deleted_sources.update(new_sources)
                try:
                    _t_a = time.monotonic()
                    table.add(records)
                    _dt_a = time.monotonic() - _t_a
                    total += len(records)
                    logger.info("  ├─ 向量库写入: 第 %d 批 %d 条 (累计 %d)", batch_num, len(records), total)
                    if on_batch:
                        _call_on_batch(on_batch, len(records), total, _dt_e, _dt_a)
                except Exception as e:
                    logger.error("  向量库写入失败(第 %d 批): %s", batch_num, e)
                    self._fts_ready = False
                    raise
                chunk_idx += len(batch_buf)
                batch_buf.clear()
        # 处理剩余
        if batch_buf:
            if is_cancelled is not None and is_cancelled():
                self._fts_ready = False
                return total
            batch_num += 1
            logger.info("  ├─ 向量化: 最后批次 (%d 切片) 正在嵌入...", len(batch_buf))
            if table is not None:
                new_sources = {c["source"] for c in batch_buf} - deleted_sources
                if new_sources:
                    for source in new_sources:
                        h = _hash_source(source)
                        try:
                            table.delete(f"id LIKE '{h}_%'")
                        except Exception:
                            pass
            records = self._embed_chunks(batch_buf, chunk_idx)
            try:
                table.add(records)
                total += len(records)
                logger.info("  ├─ 向量库写入: 最后批次 %d 条 (累计 %d)", len(records), total)
                if on_batch:
                    on_batch(len(records), total)
            except Exception as e:
                logger.error("  向量库写入失败(最后批次): %s", e)
                self._fts_ready = False
                raise

        logger.info("  └─ 向量化写入完成: 共 %d 个切片 (%d 批次)", total, batch_num)

        # 数据写入后重建 BM25 全文索引，覆盖新增的行。
        # 注：LanceDB 的 FTS 索引不会在 add() 后自动增量更新，
        # 必须用 replace=True 重建才能检索到新写入的切片。
        self._fts_ready = False
        self._ensure_fts_index(self._table)

        return total

    def _embed_chunks(
        self, chunks: List[Dict[str, Any]], start_idx: int,
    ) -> List[Dict[str, Any]]:
        """Embed 一批 chunks，返回 records（含 vector）。内部辅助方法。

        将每批拆分为更小的子批次进行 embedding，每个子批次完成后主动
        time.sleep() 释放 GIL，让主线程有机会处理 Qt 事件（日志渲染、进度
        更新）。ONNX Runtime 的 embed() 在 C++ 计算期间持有 GIL，整批一次
        性嵌入会导致 UI 长时间冻结。

        子批大小不固定（历史值 5/12）：由 :class:`_SubBatchTuner` 在 [2, 64]
        内自适应（用户规格：最小 2 / 最大 64 / 初始 16 / 步进 2）。真实切片上
        吞吐与内存随子批增大单调变差，调优器会自动在 2~最优区间收敛；下限 2
        避开单条过碎批，上限放开给长切片/高带宽机器爬升。长度分桶后 padding 已归零。

        形状预算（#126 修订）：每个子批组装时按 ``_shape_cap_size`` 钳制
        条数（激活 ∝ 条数 × 批内最长序列²），超长文本自动变小，把瞬时激活
        压在 ~2GB。不再使用绝对 RSS 阈值——真实同步进程基线 6-10GB，绝对
        阈值会误触发并永久卡死在下限。

        关键优化：**按文本长度排序分桶**（#122）。fastembed 的批内所有序列
        padding 到批内最长——长度混杂的批 padding 浪费 2-4 倍算力（实测
        32 条混杂批 309s ≈ 40 tok/s，而同模型「长度排序 + 8-12 条/批」
        ~2000 tok/s，差 30 倍）。排序后每批长度相近、padding 归零，向量按
        原顺序回填，结果与混杂批完全一致。
        """
        # 让出 GIL 的累计耗时阈值：不是「每子批都 sleep」，而是累计占用 GIL
        # 超过 50ms 才让出一次，保证 UI 响应又不为 sleep 付出固定开销。
        _YIELD_INTERVAL = 0.05
        texts = [c["text"] for c in chunks]
        # 长度降序分桶：批内长度相近，padding 浪费最小（降序让最长批最先
        # 出现，提早把激活水位顶到峰值，后续批更轻）。
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        vectors_by_idx: List[Optional[List[float]]] = [None] * len(texts)
        tuner = self._subbatch_tuner
        _last_yield = time.monotonic()
        _t_embed = time.monotonic()
        sub_idx = 0
        sub_start = 0
        while sub_start < len(order):
            size = tuner.propose()
            # 形状预算钳制（#126 修订）：激活 ∝ 条数 × 批内最长序列²，对
            # 超长文本收缩本子批条数，把瞬时激活压在 ~2GB（常量注释含标定）。
            # 降序分桶下切片首条即批内最长，钳制后 report() 收到实际条数。
            size = _shape_cap_size(size, len(texts[order[sub_start]]))
            idxs = order[sub_start:sub_start + size]
            _t_sub = time.monotonic()
            sub_vectors = self._ef([texts[i] for i in idxs])
            _dt = time.monotonic() - _t_sub
            for i, v in zip(idxs, sub_vectors):
                vectors_by_idx[i] = v
            sub_idx += 1
            sub_start += len(idxs)
            # 子批级进度日志：嵌入一个子批要数十秒且中间完全静默，用户会
            # 误判为「卡死」（#122）。每个子批一条，给出速率与当前子批大小。
            n_chars = sum(len(texts[i]) for i in idxs)
            logger.info(
                "  ├─ 嵌入子批 #%d [size=%d]: %d 条 %.1fk字符, %.1fs (%.0f k字符/s, 累计 %.0fs)",
                sub_idx, size, len(idxs), n_chars / 1000, _dt,
                n_chars / max(_dt, 1e-6) / 1000,
                time.monotonic() - _t_embed,
            )
            tuner.report(len(idxs), max(len(texts[i]) for i in idxs), _dt)
            # RSS 软墙（#129，8GB）：内存偶尔冲高是正常波动，超墙只要求降档
            # 避让——每子批检查一次，降 1 档、激活变小、RSS 自然回落，回墙
            # 下即停。无闩锁（旧 5GB 闩锁版是「卡死在 1」的根因），恢复交给
            # 调优器自身的恢复分支 / 重爬。psutil 缺失时静默跳过。
            try:
                import psutil as _psutil

                if (
                    _psutil.Process().memory_info().rss / 1024**3
                    > _SUBBATCH_RSS_LIMIT
                ):
                    tuner.force_down()
            except Exception:
                pass
            # 主动释放 GIL，让 Qt 事件循环有机会处理排队的日志/进度信号。
            # 这是关键：ONNX Runtime 在 embed() 期间持有 GIL，只有在 Python
            # 层主动 sleep 时才释放。
            if time.monotonic() - _last_yield >= _YIELD_INTERVAL:
                time.sleep(0.01)
                _last_yield = time.monotonic()
        vectors = vectors_by_idx
        records: List[Dict[str, Any]] = []
        # 一批 chunk 通常来源相同 → 缓存 source→hash 映射，避免每个 chunk
        # 都重算一次 md5（写入侧纯浪费）。
        _hash_cache: Dict[str, str] = {}
        # FTS 分词循环是纯 Python（jieba 全程持 GIL）：书籍切片动辄几千字，
        # 200 个切片连跑数秒 UI 线程完全拿不到 GIL——表现为「嵌入子批之间
        # 阶段性卡住不动」（#131）。每 32 条主动让出一次 GIL。
        for j, chunk in enumerate(chunks):
            if j and j % 32 == 0:
                time.sleep(0.01)
            idx = start_idx + j
            source = chunk.get("source", "unknown")
            chunk_idx = chunk.get("chunk_index", idx)
            h = _hash_cache.get(source)
            if h is None:
                h = _hash_source(source)
                _hash_cache[source] = h
            chunk_id = f"{h}_{chunk_idx}"
            records.append({
                "id": chunk_id,
                "text": chunk["text"],
                "vector": _normalize(vectors[j]),
                "source": source,
                "chunk_index": chunk_idx,
                _FTS_COLUMN: _tokenize_for_fts(chunk["text"]),
            })
        return records

    # ------------------------------------------------------------------
    # BM25 full-text search (hybrid retrieval)
    # ------------------------------------------------------------------

    def _ensure_fts_index(self, table) -> bool:
        """Ensure a BM25 full-text index exists on the ``text_fts`` column.

        Creates the index on first use and after each ingest (the index must
        be refreshed to cover newly written rows).  The ``text_fts`` column is
        pre-tokenized with jieba (space-joined).  We use LanceDB's *native*
        FTS (Tantivy support was removed in lancedb 0.37); the native ``simple``
        tokenizer splits on whitespace, so each space-separated jieba token is
        indexed as its own term — sidestepping the lack of a compiled CJK
        tokenizer in the native FTS backend.

        Returns True if the index is available for querying.
        """
        if self._fts_ready:
            return True
        # 建索引失败过就不再重试：旧实现只把 _fts_checked 置 True，而
        # _fts_checked 全项目没有任何读取点 —— 于是每次检索都会重跑一遍
        # 昂贵的全库建索引，然后再次失败。
        if getattr(self, "_fts_failed", False):
            return False

        # 原生 FTS 配置（两种 lancedb 版本共用同一语义）：
        #  - simple base tokenizer 按空格/标点切分，配合 jieba 预分词列，
        #    每个空格分隔的 jieba token 即为一个独立 term；
        #  - ascii_folding=False 保留 CJK 码点；
        #  - stem=False / remove_stop_words=False 避免破坏中文词形。
        fts_kwargs = dict(
            base_tokenizer="simple",
            lower_case=True,
            stem=False,
            remove_stop_words=False,
            ascii_folding=False,
            with_position=True,
        )

        # lancedb >= 0.37：create_index(config=FTS(...))（Tantivy 已移除）。
        # lancedb 0.25.x：create_fts_index(use_tantivy=False, ...)。
        #
        # 必须持**模块级**锁：写入侧（SyncKnowledgeWorker 的引擎实例）与检索
        # 侧（agent 的引擎实例）是不同的 VectorStore 实例、同一张磁盘表，双方
        # _fts_ready 都为 False 时会对同一张表并发 replace 建索引（同步中用户
        # 同时检索即可触发），可能报错或索引不一致。建索引是低频操作，全局
        # 锁没有吞吐影响。
        #
        # 注意：ImportError 必须单独捕获——0.25.x 上根本没有 lancedb.index.FTS
        # 这个模块，裸 import 会抛 ImportError；若掉进通用 except Exception 分支，
        # _fts_failed 被置 True 后 0.25.x 的回退路径就永远走不到了（BM25 永久禁用）。
        try:
            from lancedb.index import FTS
        except ImportError:
            FTS = None
        with _FTS_CREATE_LOCK:
            try:
                if FTS is not None:
                    table.create_index(
                        _FTS_COLUMN,
                        config=FTS(**fts_kwargs),
                        replace=True,
                    )
                else:
                    raise TypeError  # 走 0.25.x 回退分支
            except TypeError:
                # 0.25.x 的 create_index 无 config 参数（或 FTS 不可用），回退到
                # create_fts_index。
                try:
                    table.create_fts_index(
                        _FTS_COLUMN,
                        use_tantivy=False,
                        replace=True,
                        **fts_kwargs,
                    )
                except Exception as e:
                    logger.warning("  ⚠ BM25 全文索引创建失败 (回退纯向量检索): %s", e)
                    self._fts_checked = True
                    self._fts_ready = False
                    self._fts_failed = True
                    return False
            except Exception as e:
                # 并发对方刚建好 → 本方报「已存在」类错误时按成功处理，
                # 索引在磁盘上已可用，没必要回退纯向量检索。
                msg = str(e).lower()
                if "exist" in msg or "already" in msg:
                    logger.debug("  BM25 索引已由并发方创建: %s", e)
                else:
                    logger.warning("  ⚠ BM25 全文索引创建失败 (回退纯向量检索): %s", e)
                    self._fts_checked = True
                    self._fts_ready = False
                    self._fts_failed = True
                    return False

        self._fts_ready = True
        self._fts_checked = True
        logger.info("  ├─ BM25 全文索引已创建 (原生 FTS simple tokenizer + jieba 预分词)")
        return True

    @_store_op
    def search_fts(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Keyword (BM25) full-text search over the ``text_fts`` column.

        Returns list of {text, source, chunk_index, score, fts_score} dicts,
        where ``score`` is the BM25 score (higher is better).  Used together
        with vector search for hybrid retrieval.

        ``where`` is an optional SQL filter (e.g. ``source LIKE '%book.md'``)
        applied to the FTS search as well, so targeted retrieval restricts both
        channels consistently.
        """
        self._ensure_initialized()
        # 空查询直接短路：否则会白白触发一次全库 FTS 建索引 + 一次空查询。
        if not (query or "").strip():
            return []
        table = self._ensure_table()
        if table is None:
            return []

        if not self._ensure_fts_index(table):
            return []

        try:
            # Tokenize the query the same way as the indexed text.
            tokenized_query = _tokenize_for_fts(query)
            q = (
                table.search(tokenized_query, query_type="fts", fts_columns=[_FTS_COLUMN])
            )
            if where:
                q = q.where(where)
            results = q.limit(top_k).to_list()
        except Exception as e:
            logger.warning("  ⚠ BM25 全文检索失败: %s", e)
            return []

        output: List[Dict[str, Any]] = []
        for r in results:
            # Tantivy/原生 FTS 返回的 BM25 分在 `_score`（越大越好），是无上界
            # 的原始分（常见 5~30+）。绝不能把它直接塞进 `score`：
            # `score` 在有向量候选的场合被当作 [0,1] 余弦相似度使用，
            # _rerank_fallback 会对其做 `1 - s²/2` 换算 —— BM25=15 会被算成
            # 1-112.5 → clamp 到 0.0，纯关键词候选集体排到末尾，混合检索
            # 在 reranker 不可用时静默退化成纯向量检索。
            # 这里用 s/(1+s) 归一化到 [0,1)，原始分另存 fts_score。
            raw = float(r.get("_score", 0.0))
            norm = raw / (1.0 + raw) if raw > 0 else 0.0
            output.append({
                "text": r.get("text", ""),
                "source": r.get("source", ""),
                "chunk_index": r.get("chunk_index", 0),
                "score": norm,
                "fts_score": raw,
            })
        return output

    @_store_op
    def search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search the vector store for relevant chunks.

        Returns list of {text, source, chunk_index, score} dicts.
        """
        self._ensure_initialized()
        table = self._ensure_table()
        if table is None:
            logger.info("  向量检索: 向量表不存在 (查询=%r)", query[:60])
            return []
        row_count = table.count_rows()
        if row_count == 0:
            logger.info("  向量检索: 向量表为空 (查询=%r)", query[:60])
            return []

        # Embed query (use embed_query for proper prefix handling)
        logger.info("  ├─ 查询向量化: 将查询文本转为向量...")
        if hasattr(self._ef, 'embed_query'):
            query_vec = _normalize(self._ef.embed_query(query))
        else:
            query_vec = _normalize(self._ef([query])[0])

        # Vector search
        logger.info("  ├─ 向量相似度搜索: 在 %d 条向量中检索 top %d...", row_count, top_k)
        search = table.search(query_vec).limit(top_k)
        if where:
            # prefilter=True：先按条件过滤再做 ANN。默认的后置过滤是「先取
            # ANN top-k 再过滤」，定向检索（rag_search(source=...)）下候选
            # 会被先截断再筛，可能漏召回甚至返回空。
            try:
                search = search.where(where, prefilter=True)
            except TypeError:
                # 旧版 lancedb 的 where() 不接受 prefilter 关键字。
                search = search.where(where)
        results = search.to_list()
        logger.info("  ├─ 向量检索完成: 获得 %d 个候选切片", len(results))

        output: List[Dict[str, Any]] = []
        for r in results:
            # LanceDB 返回 _distance（L2 距离，越小越相似）。
            # 转换为相似度分数 [0,1]（越高越相似），统一下游接口语义。
            if "_distance" in r:
                distance = float(r["_distance"])
                # 对归一化向量，L2 距离 d ∈ [0,2]，余弦相似度 cos = 1 - d²/2
                similarity = max(0.0, 1.0 - (distance * distance) / 2.0)
            else:
                # _distance 字段缺失（罕见：表 schema 不兼容或旧表迁移）。
                # 使用中等分数避免将无效结果排到最前或最后。
                logger.debug("VectorStore.search: _distance field missing for result, using default score 0.5")
                similarity = 0.5
            output.append({
                "text": r.get("text", ""),
                "source": r.get("source", ""),
                "chunk_index": r.get("chunk_index", 0),
                "score": similarity,
            })

        return output

    # ------------------------------------------------------------------
    # reranking with FastEmbed
    # ------------------------------------------------------------------

    def _get_reranker(self):
        """Lazy-load the reranker model (FastEmbed TextCrossEncoder) with timeout protection.

        Returns the reranker instance, or None if loading fails (network
        timeout, missing dependency, etc.). Callers must handle None by
        falling back to distance-based ranking.

        Uses ``_reranker_state`` to track loading state so concurrent callers
        don't each spawn their own load thread (which would leak the model
        weights per extra thread).  On timeout, the background thread
        is NOT abandoned — when it eventually finishes it stores the result
        (or error) on ``self`` so a later call can pick it up without reloading.
        """
        # close() 之后的实例不允许再拉起 reranker：后台加载线程看到 _closed
        # 会直接 return（状态置 failed），避免状态机卡死或「复活」持有权重。
        if getattr(self, "_closed", False):
            return None
        # 本地 reranker 优先：单例、懒加载，直接复用，不走 fastembed 后台状态机。
        if _RERANK_PROVIDER["provider"] == "local":
            return get_or_create_reranker()
        state = getattr(self, "_reranker_state", "idle")
        if state == "done":
            return self._reranker
        if state in ("failed", "loading"):
            # failed → 不可用；loading → 另一个线程已在加载，本次调用回退到
            # 距离排序，加载完成后的调用自然能拿到模型。
            return None

        # 状态转换必须串行：import fastembed + 构造 TextCrossEncoder 可触发模型
        # 下载/加载（v2-m3 约 2.3GB），耗时数秒到数分钟。旧实现从「读到 idle」
        # 到「写入 loading」之间没有任何锁，两个并发调用方都会各自起一个加载
        # 线程——各加载一份模型，后写者覆盖前者，先建的那份成为纯泄漏。
        with self._reranker_lock:
            state = getattr(self, "_reranker_state", "idle")
            if state == "done":
                return self._reranker
            if state in ("failed", "loading"):
                return None

            # FastEmbed 的 TextCrossEncoder 基于 ONNX Runtime（与 embedding 同源），
            # 不依赖 torch / FlagEmbedding，因此在 numpy 2.x 下也能正常工作——
            # 旧实现需要 torch 探测 + FlagEmbedding，正是 numpy 2.x 不兼容的根源。
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
            except ImportError:
                logger.warning(
                    "  ⚠ fastembed.rerank 不可用, 重排序功能不可用. "
                    "请升级 fastembed: pip install 'fastembed>=0.9.0'"
                )
                self._reranker = None
                self._reranker_state = "failed"
                return None

            logger.info("  🔧 正在加载 Reranker 模型: %s (首次加载)...", self._rerank_model)
            import time as _time
            _time.sleep(0.01)  # 释放 GIL 让 UI 更新

            def _load():
                try:
                    cache_dir = os.path.join(
                        os.path.expanduser("~"), ".cache", "fastembed",
                    )
                    os.makedirs(cache_dir, exist_ok=True)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        reranker = TextCrossEncoder(
                            model_name=self._rerank_model,
                            cache_dir=cache_dir,
                            # 与嵌入模型同款：显式强制 CPU-only，不依赖 FastEmbed
                            # 的默认回退（见 _cpu_only_kwargs 注释）。
                            **_cpu_only_kwargs(TextCrossEncoder),
                        )
                    # 写回实例必须持 _reranker_lock 并在锁内复查 _closed：
                    # 旧实现裸写 _reranker/_reranker_state，与 close() 的裸清
                    # 交错时，已 close 的实例会「复活」持有模型权重，且
                    # state 变成 done 后下次还会再起加载线程。
                    with self._reranker_lock:
                        if getattr(self, "_closed", False):
                            # close() 已经执行：不要再把模型写回实例
                            # （旧实现会让 close 之后的实例「复活」持有权重）。
                            del reranker
                            # 必须把状态机从 "loading" 里救出来：旧实现直接
                            # return，状态永远停在 loading → 之后所有 rerank()
                            # 都命中 loading 分支永久返回 None，重排序静默失效
                            # 且无日志。
                            self._reranker_state = "failed"
                            return
                        self._reranker = reranker
                        self._reranker_state = "done"
                    logger.info("  ✅ Reranker 加载完成: %s", self._rerank_model)
                except Exception as e:
                    logger.warning("  ⚠ Reranker 加载失败: %s", e)
                    with self._reranker_lock:
                        self._reranker = None
                        self._reranker_state = "failed"

            # 先置 loading 再启动线程，并发的其他调用方看到 loading 即返回。
            self._reranker_state = "loading"
            t = threading.Thread(target=_load, daemon=True)
            t.start()

        # join 放在锁外：避免其他调用方为等这个 join 阻塞。
        try:
            t.join(timeout=180)  # 3 分钟超时（v2-m3 约 2.3GB，网络慢时更久）
            if t.is_alive():
                # 加载仍在后台进行（网络慢）。不丢弃结果——后台线程完成时会
                # 写入 self._reranker/_reranker_state，后续调用直接复用。
                logger.warning(
                    "  ⚠ Reranker 加载超时 (3 分钟) — 仍在后台下载中。"
                    " 本次重排序回退到向量距离排序；加载完成后自动启用。"
                )
        except Exception as e:
            logger.warning("  ⚠ Reranker 加载失败: %s", e)
            with self._reranker_lock:
                self._reranker = None
                self._reranker_state = "failed"

        with self._reranker_lock:
            return self._reranker if getattr(self, "_reranker_state", "idle") == "done" else None

    def _rerank_fallback(
        self,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Fall back to vector-distance ranking when the reranker is unavailable.

        candidates 的 score 已由 search() 转换为相似度 [0,1]（越高越好）。
        回退模式下分数是余弦距离的近似（非 reranker 语义分），因此不做
        0.7 阈值过滤——阈值仅对 reranker 的语义分数有意义。
        """
        logger.info("  ├─ 重排序: Reranker 不可用, 使用向量距离排序 (回退模式)")
        for c in candidates:
            try:
                s = float(c.get("score", 0.0))
            except (TypeError, ValueError):
                c["score"] = 0.0
                continue
            if s > 1.0:
                c["score"] = max(0.0, 1.0 - (s * s) / 2.0)
            else:
                c["score"] = s
        candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        result = candidates[:top_k]
        logger.info("  ├─ 重排序完成(回退): %d → %d 条结果", len(candidates), len(result))
        return result

    @_store_op
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 3,
        min_similarity: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Re-rank candidates using FastEmbed reranker for precise scoring.

        Falls back to distance-based similarity scoring when the reranker
        is unavailable. In both cases scores are normalized to [0, 1]
        (higher is better) and results are returned sorted descending.

        Results whose final score is below ``min_similarity`` are dropped
        (they are not semantically relevant enough to send to the LLM), so
        the returned list may be shorter than ``top_k``.
        """
        if not candidates:
            return []

        # 使用注入的自定义 embedding（如测试的 FakeEmbeddingFunction）时，
        # 向量空间与 reranker 的语义空间不一致，rerank 打分无意义，
        # 直接回退到向量距离排序。
        if self._custom_ef is not None:
            return self._rerank_fallback(candidates, top_k)

        reranker = self._get_reranker()
        if reranker is None:
            return self._rerank_fallback(candidates, top_k)

        docs = [c["text"] for c in candidates]
        logger.info("  ├─ FastEmbed Reranker 精排: 对 %d 个候选进行交叉编码打分...", len(candidates))
        try:
            # fastembed 直接返回模型输出（与输入顺序一一对应）。不同 ONNX 导出
            # 可能是原始 logits 或已 sigmoid 归一化的概率。
            #
            # 归一化按**模型固定**（首次探测一次并缓存结论），不再按「本批候选」
            # 自适应：旧实现用 any(abs(s) > 1.0) 决定是否对整个 batch 做 sigmoid，
            # 同一个 chunk 在不同查询下可能走/不走归一化，绝对值之间不可比，而
            # min_similarity=0.7 是绝对语义的阈值 —— 语义会随查询漂移。
            #
            # 数值稳定 sigmoid：s 为很大负数时 1/(1+e^-s) 会 OverflowError
            # （被外层 except 吞掉后静默回退到向量排序，精排形同失效）。
            raw_scores = [float(s) for s in reranker.rerank(query, docs)]
            if self._reranker_is_logits is None:
                self._reranker_is_logits = any(abs(s) > 1.0 for s in raw_scores)
                logger.info(
                    "  ├─ Reranker 输出探测: %s",
                    "原始 logits（将做 sigmoid 归一化）" if self._reranker_is_logits
                    else "已归一化概率（直接使用）",
                )
            if self._reranker_is_logits:
                scores = [_stable_sigmoid(s) for s in raw_scores]
            else:
                scores = raw_scores
        except Exception as e:
            logger.warning(
                "  ⚠ Reranker 打分失败, 回退到向量距离排序: %s", e,
            )
            # 标记不可用，后续直接走回退分支。必须同步置 failed，否则
            # _reranker_state 长期停在 "done" 而模型为 None（状态机失真）。
            # 持锁写回：与 close() / 后台加载线程的持锁写回互斥。
            with self._reranker_lock:
                self._reranker = None
                self._reranker_state = "failed"
            return self._rerank_fallback(candidates, top_k)

        # 防御：分数数量与候选数不一致（异常），回退避免错位。
        if len(scores) != len(candidates):
            logger.warning(
                "  ⚠ Reranker 返回分数数量(%d)与候选数(%d)不一致, 回退到向量距离排序",
                len(scores), len(candidates),
            )
            with self._reranker_lock:
                self._reranker = None
                self._reranker_state = "failed"
            return self._rerank_fallback(candidates, top_k)

        # Attach rerank scores and sort (higher is better)
        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = float(score)

        candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        # Return top_k, mapping rerank_score to score for downstream
        results = candidates[:top_k]
        for r in results:
            r["score"] = r.get("rerank_score", r.get("score", 0))

        # 输出重排序前后的分数变化
        if results:
            logger.info("  ├─ Reranker 重排序完成: %d → %d 条结果", len(candidates), len(results))
            for i, r in enumerate(results[:3]):
                source = os.path.basename(r.get("source", ""))
                logger.info("  │   [%d] %s (重排序分: %.4f)", i + 1, source, r["score"])

        return self._filter_by_similarity(results, min_similarity)

    def _filter_by_similarity(
        self,
        results: List[Dict[str, Any]],
        min_similarity: float,
    ) -> List[Dict[str, Any]]:
        """Drop results whose final score is below ``min_similarity``.

        A rerank score (BGE sigmoid, or cosine-similarity fallback) below the
        threshold means the chunk is not semantically close enough to the
        query to be worth sending to the LLM.
        """
        if not results or min_similarity <= 0:
            return results
        kept = [r for r in results if r.get("score", 0.0) >= min_similarity]
        if len(kept) < len(results):
            dropped = len(results) - len(kept)
            logger.info(
                "  ├─ 相似度过滤: 丢弃 %d 条 < %.2f 的结果, 保留 %d 条",
                dropped, min_similarity, len(kept),
            )
        return kept

    @_store_op
    def search_with_rerank(
        self,
        query: str,
        top_k: int = 3,
        candidate_multiplier: int = 4,
        where: Optional[str] = None,
        min_similarity: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Hybrid retrieval: vector similarity + BM25 keyword, then BGE rerank.

        Pipeline:
          1. Vector search: top_k * candidate_multiplier candidates.
          2. BM25 keyword search: top_k * candidate_multiplier candidates.
          3. Merge & dedupe by (source, chunk_index).
          4. BGE rerank the merged candidate set.
          5. Drop any result whose rerank score < min_similarity (0.7).
          6. Return up to top_k results.

        Args:
            query: Search query.
            top_k: Final number of results to return.
            candidate_multiplier: Candidates per retrieval channel = top_k × N.
            where: Optional SQL filter (applied to vector search only).
            min_similarity: Reject rerank results below this score.
        """
        n_candidates = top_k * candidate_multiplier

        # 1. Vector similarity search.
        vec_candidates = self.search(query, top_k=n_candidates, where=where)
        logger.info(
            "VectorStore.search_with_rerank: 向量检索 %d 条候选", len(vec_candidates),
        )

        # 2. BM25 keyword search (同样的 where 过滤，保证定向检索一致)。
        fts_candidates = self.search_fts(query, top_k=n_candidates, where=where)
        logger.info(
            "VectorStore.search_with_rerank: BM25 检索 %d 条候选", len(fts_candidates),
        )

        # 3. Merge & dedupe by (source, chunk_index), keeping vector candidate
        #    first (its score is a cosine similarity in [0,1]).
        merged: List[Dict[str, Any]] = []
        seen: set = set()
        for c in vec_candidates + fts_candidates:
            key = (c.get("source", ""), c.get("chunk_index", 0))
            if key in seen:
                continue
            seen.add(key)
            merged.append(c)

        if not merged:
            logger.info("VectorStore.search_with_rerank: 混合检索无候选")
            return []

        logger.info(
            "VectorStore.search_with_rerank: 混合候选 %d 条 (向量 %d + BM25 %d, 去重后 %d)",
            len(merged), len(vec_candidates), len(fts_candidates), len(merged),
        )

        # 4. BGE rerank + 5. min_similarity filter.
        results = self.rerank(query, merged, top_k=top_k, min_similarity=min_similarity)
        logger.info(
            "VectorStore.search_with_rerank: %d 候选 → %d 条 (rerank + 相似度≥%.2f 过滤)",
            len(merged), len(results), min_similarity,
        )

        # 6. 当 rerank 严格过滤后无结果（例如查询词质量差——拼音、口语化、
        #    噪声词等导致 rerank 语义分普遍低于阈值）时，回退到向量距离排序，
        #    至少返回 top_k 条候选，避免 LLM 完全拿不到上下文。
        if not results and merged:
            logger.info(
                "VectorStore.search_with_rerank: rerank 过滤后无结果, 回退到向量距离排序",
            )
            results = self._rerank_fallback(merged, top_k)
        return results

    def _clear_locked(self) -> None:
        """Delete all documents from the store.

        调用方必须已排空在途操作（见 ``clear``）：drop_table 与在途的
        add/search 并发，会让后者打到一张已被删除的表上。
        """
        self._ensure_initialized()
        table = self._ensure_table()
        if table is None:
            return
        count = table.count_rows()
        if count > 0:
            # Drop and recreate empty table
            self._db.drop_table(_TABLE_NAME)
            self._table = None
            self._table_checked = False  # 重置标记，允许下次 add 重新创建表
            # 必须同时重置 FTS 三标记（与 close() 同款）：漏掉的话 _fts_ready
            # 仍为 True，add() 重建新表时会跳过建索引，search_fts 也认为索引
            # 就绪 → 在未索引列上跑 FTS 抛异常被吞掉 → BM25 通道从此永久
            # 静默返回空，无任何告警。
            self._fts_ready = False
            self._fts_checked = False
            self._fts_failed = False
            logger.info("VectorStore: cleared %d documents", count)
        # 同步清空文档元数据表。
        if _DOCUMENTS_TABLE_NAME in self._db.table_names():
            self._db.drop_table(_DOCUMENTS_TABLE_NAME)
        # documents 表**不存在**时也要清缓存/句柄：否则残留的旧缓存会让
        # get_document_digest 继续返回已删除文档的 digest。清缓存必须与
        # _load_documents 一样持 _documents_lock。
        with self._documents_lock:
            self._documents_table = None
            self._documents_cache = None
            logger.info("VectorStore: cleared documents metadata table")

    def clear(self) -> None:
        """清空向量库与文档元数据。

        真正实现在 ``_clear_locked``：``clear`` 需要在拿到在途操作排空权之后
        再动手——drop_table 与在途的 add/search 并发会让后者打到已删除的表上。
        注意这里**不能**加 ``@_store_op``：它会先把自己登记为在途操作，随后
        ``_drain_ops()`` 就会等自己结束（自锁）。
        """
        self._drain_ops()
        try:
            self._clear_locked()
        finally:
            self._resume_ops()

    @_store_op
    def count(self) -> int:
        self._ensure_initialized()
        table = self._ensure_table()
        if table is None:
            return 0
        return table.count_rows()

    def close(self) -> None:
        """Release heavy resources (LanceDB connection, embedding model, reranker).

        After close(), the VectorStore cannot be used; create a new instance to
        access the same data (LanceDB persists to disk, so data survives).

        Important: FastEmbed's embedding and reranker ONNX Runtime sessions
        hold significant C++ heap memory (~130MB + ~570MB INT8). Python's GC may
        not reclaim these promptly when the Python object is dropped — explicitly
        nulling the references here ensures the C++ destructors run as soon
        as possible, which is critical when SyncKnowledgeWorker creates a fresh
        RAGEngine per sync run.

        Waits for in-flight ``add``/``search`` operations to finish first:
        tearing down the LanceDB connection underneath them produces random
        AttributeErrors (``self._db`` is None) or silently wrong results.
        """
        # 先排空并**保持**排空态：若此时正有 add/search 在跑，它们会撞上被置
        # None 的 _db 或已被 drop 的表。这里不能像 clear() 那样 _resume_ops()——
        # close() 之后本实例永久不可用，_closing 保持 True 正好与
        # _begin_op() 的检查一致（close 后新操作一律被拒）。
        self._drain_ops()
        # Try to explicitly release the ONNX InferenceSession held by
        # FastEmbed's OnnxEmbeddings. FastEmbed wraps the model in a
        # `model` attribute (ort.InferenceSession). Calling .release() on
        # the session frees the C++ memory immediately instead of waiting
        # for Python's cyclic GC to eventually notice the dead reference.
        # This is best-effort: if the attribute layout changes in a future
        # FastEmbed version, the AttributeError is swallowed and we fall
        # back to simply dropping the Python reference.
        # 先标记关闭：后台 reranker 加载线程完成时看到 _closed 就不再把模型
        # 写回本实例（否则 close 之后的实例会「复活」持有 ~500MB 权重）。
        # 注意 _closed 必须在重新初始化时复位（见 _initialize_locked）：
        # 该实例可以被复活重连，若标志一直为真，reranker 会对它永久失效。
        self._closed = True
        self._table = None
        self._db = None
        # documents 表也必须一起置空：它指向旧 lancedb.connect() 打开的表，
        # 而 LanceDB table 持有打开时的版本快照。只置空 _table/_db 会让
        # 复活的实例（_initialize_locked 支持 close 后重连）继续往旧连接上写，
        # upsert_document 对新连接不可见甚至直接失败。
        self._documents_table = None
        self._documents_cache = None
        if self._ef is not None:
            # Only release the ONNX session if this VectorStore owns a
            # private (non-shared) embedding function.  When using the
            # process-wide singleton (get_or_create_embedding_function),
            # the session may still be in use by graph_memory or another
            # VectorStore — releasing it would crash those callers.
            is_shared = self._ef in _EF_SINGLETONS.values()
            if not is_shared:
                try:
                    inner = getattr(self._ef, "model", None) or getattr(self._ef, "_model", None)
                    if inner is not None:
                        session = getattr(inner, "session", None) or getattr(inner, "_session", None)
                        if session is not None and hasattr(session, "release"):
                            try:
                                session.release()
                            except Exception:
                                pass
                except Exception:
                    pass
            self._ef = None
        if hasattr(self, "_reranker"):
            # FastEmbed reranker 是 ONNX 模型（无 torch/PyTorch 张量），del 后
            # 由 Python GC 回收 C++ 堆内存。必须持 _reranker_lock：与后台加载
            # 线程的写回（同样持锁）互斥，否则 close 交错会把已清空的实例
            # 「复活」持有模型权重。
            # 旧实现这里还对 reranker.model 调 .cpu() —— TextCrossEncoder 的
            # model 是 ONNX 封装，没有 cpu()，属死代码（异常还被吞掉）。
            with self._reranker_lock:
                self._reranker = None
                self._reranker_state = "idle"
        self._initialized = False
        self._table_checked = False
        # 表句柄已丢弃，重连后打开的是「另一张」表（可能被 clear()/迁移重建
        # 过）。不重置 FTS 标记的话，_ensure_fts_index() 会直接返回 True，
        # 于是 BM25 检索静默失效（新表上没有索引）。
        self._fts_ready = False
        self._fts_checked = False
        # close() 后可以重新 open：失败标记必须一并复位，否则新建的实例会
        # 永久跳过全文索引。
        self._fts_failed = False
        logger.info("VectorStore: resources released")

    def reload(self) -> None:
        """Drop the cached table handle so the next operation re-opens it.

        LanceDB table objects hold a version snapshot at open time. When another
        VectorStore instance (e.g. SyncKnowledgeWorker's) writes new rows to the
        same table, this instance's cached table object won't see them —
        count_rows() and search() still return the old snapshot's data.

        Call this after a sync completes to ensure the agent's VectorStore sees
        the freshly written data on its next search.

        与 clear()/close() 一样先排空在途操作：``_table = None`` 与在途的
        search() 并发时，后者可能正持有旧句柄做查询，行为未定义。
        """
        self._drain_ops()
        try:
            self._reload_locked()
        finally:
            self._resume_ops()

    def _reload_locked(self) -> None:
        """``reload()`` 的实现，调用方必须已排空在途操作。"""
        self._table = None
        self._table_checked = False
        # documents 元数据表与其缓存同样持有打开时的快照：同步新增/更新了
        # 文档后，agent 侧的 rag_outline 必须丢弃旧句柄与旧缓存才能看到
        # 新 digest。
        with self._documents_lock:
            self._documents_table = None
            self._documents_cache = None
        # FTS 标志：reload 只丢弃**内存句柄快照**，磁盘上的索引与数据都没动。
        # 旧实现把 _fts_ready/_fts_checked 一并清 False → 下次检索走
        # _ensure_fts_index 的 replace=True 全库重建（秒级到分钟级），而索引
        # 其实早在磁盘上、本就可用。这里探测一次磁盘索引是否仍在：在就在
        # _fts_ready=True（检索直接用），不在才回落到重建路径。
        # _fts_failed 保留旧值：曾经建失败的实例，reload 也不该自愈（避免
        # 每次检索都重试一次必败的建索引）。
        self._fts_checked = False
        self._fts_ready = False
        try:
            tbl = self._ensure_table()
            if tbl is not None:
                indices = tbl.list_indices()
                names = set()
                for it in indices:
                    n = getattr(it, "name", None) or (it.get("name") if isinstance(it, dict) else None)
                    if n:
                        names.add(str(n))
                # 命中已知列名或任意 FTS 索引名（不同 lancedb 版本命名不同）。
                if _FTS_COLUMN in names or any("fts" in n.lower() for n in names):
                    self._fts_ready = True
                    self._fts_checked = True
                    logger.debug("VectorStore: reload 探测到磁盘 FTS 索引，跳过重建")
        except Exception as e:
            logger.debug("VectorStore: reload FTS 探测失败，回落到重建: %s", e)
        logger.debug("VectorStore: table handles dropped for reload")

    @_store_op
    def list_sources(self) -> List[str]:
        """Return unique source filenames in the store."""
        self._ensure_initialized()
        table = self._ensure_table()
        if table is None or table.count_rows() == 0:
            return []
        # 只投影 source 列：lancedb 0.25（pyproject 锁 <0.26）的
        # ``Table.to_arrow()`` 签名就是 ``(self)``，**不接受 columns 参数**，
        # 旧实现两次调用都抛 TypeError 后回退到 ``to_arrow().to_pylist()``
        # —— 把全部行 × 全部列（含 768 维向量）转成 Python 对象。实测 2 万
        # 切片 7.5s、内存峰值 GB 级，而它每轮对话都被 _system_prompt() 调一次。
        try:
            n = table.count_rows()
            rows = (
                table.search()
                .limit(max(n, 1))
                .select(["source"])
                .to_arrow()
                .to_pylist()
            )
            return sorted({r.get("source", "") for r in rows if r and r.get("source")})
        except Exception as e:
            logger.debug("VectorStore: list_sources 列投影不可用，回退 Arrow 去重: %s", e)
        # Fallback: Arrow 层去重（不转 Python 对象）；再不行才整表物化。
        try:
            import pyarrow.compute as pc
            unique_sources = pc.unique(table.to_arrow()["source"]).to_pylist()
            return sorted([s for s in unique_sources if s])
        except Exception as e:
            logger.debug("VectorStore: list_sources Arrow 去重失败，回退全表: %s", e)
        try:
            all_data = table.to_arrow().to_pylist()
            sources = sorted({r.get("source", "") for r in all_data if r and r.get("source")})
            return sources
        except Exception as e:
            logger.warning("VectorStore: list_sources failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # documents metadata table (Meta-context + Targeted RAG)
    # ------------------------------------------------------------------

    def _ensure_documents_table(self):
        """Open or create the documents metadata table (thread-safe).

        用双重检查锁保护：ingest 的多 worker 并行调用 upsert_document 时，
        首次创建表会并发竞争——若不加锁，第二个 worker 会在 create_table 时报
        "Table already exists"。锁内再次检查后，创建或打开由唯一一个线程完成。

        注意：调用方（upsert_document / delete_document）通常已经持有
        ``_documents_lock``。锁现在是 RLock，所以这里再次 acquire 是安全的；
        但真正的首次建表路径过去是 **死锁**：upsert_document 持锁 →
        调用本函数 → 再次 acquire 同一个非可重入 Lock → 永久阻塞。第一次同步
        （documents 表尚不存在）会卡死整个 ingest，UI 一直停在「同步中」。
        """
        if self._documents_table is not None:
            return self._documents_table
        with self._documents_lock:
            return self._ensure_documents_table_locked()

    def _migrate_documents_schema(self) -> None:
        """旧 schema 迁移（#131）：补 md_path 列。

        早期版本的 documents 表没有 md_path 列，而新代码的 delete 过滤
        （``md_path != ...``）和 upsert 写入都引用它——不补列会报
        ``field 'md_path' does not exist in table schema``，每个文件的
        缩略版本（digest）写入都失败。add_columns 补空串默认值即可，
        老行的 md_path 为空串、下次重解析时自然被 upsert 覆盖。
        """
        try:
            col_names = [f.name for f in self._documents_table.schema]
            if "md_path" not in col_names:
                logger.info("documents 表缺 md_path 列，补列（schema 迁移）")
                self._documents_table.add_columns({"md_path": "''"})
        except Exception as e:
            logger.warning("documents 表 md_path 迁移失败: %s", e)

    def _ensure_documents_table_locked(self):
        """``_ensure_documents_table`` 的实现，调用方必须已持有 _documents_lock。"""
        if self._documents_table is not None:
            return self._documents_table
        self._ensure_initialized()
        if _DOCUMENTS_TABLE_NAME in self._db.table_names():
            self._documents_table = self._db.open_table(_DOCUMENTS_TABLE_NAME)
            self._migrate_documents_schema()
        else:
            # 首次创建：空表（含占位空记录，后续 upsert 时清理）。
            # 注意：不建 markdown 列——全文已落在 rag/documents/*.md 缓存里，
            # 表内再存一份是纯浪费（且会被 _documents_cache 全量载入内存）。
            self._documents_table = self._db.create_table(
                _DOCUMENTS_TABLE_NAME,
                data=[{
                    "doc_id": "",
                    "doc_name": "",
                    "digest": "",
                    "chapters": "",
                    "md_path": "",
                }],
            )
        return self._documents_table

    @_store_op
    def upsert_document(
        self,
        doc_name: str,
        digest: str,
        markdown: str = "",
        chapters: str = "",
        md_path: str = "",
    ) -> None:
        """Store or update a document's digest (缩略版本) + chapter list.

        ``doc_id`` is a stable md5 of ``md_path`` (falling back to ``doc_name``)
        so re-ingesting the same document overwrites the previous entry (upsert
        semantics).  Keying on the basename would collide for same-named files
        in different subdirectories.

        The full markdown is **not** stored in this table: it already lives on
        disk in ``rag/documents/*.md`` (the ingest cache), and no consumer
        reads it from here — only ``digest`` is used (by ``rag_outline``).
        Keeping it would duplicate the whole corpus a third time on disk and
        load it entirely into memory via ``_documents_cache``.  We persist
        ``md_path`` instead so a consumer can read it on demand.

        The delete+add pair is serialized under ``_documents_lock`` because
        ingest workers call this concurrently (one per file). LanceDB table
        writes are not thread-safe, and interleaving delete/add from multiple
        threads could corrupt rows or raise "Table already exists"-style races.
        """
        self._ensure_initialized()
        with self._documents_lock:
            # 已在锁内 → 用不重复加锁的版本，避免嵌套 acquire。
            table = self._ensure_documents_table_locked()
            # doc_id 用 md_path（每篇文档的缓存路径唯一）而不是 basename：
            # 知识库里不同子目录的同名文件（a/report.pdf 与 b/report.pdf）
            # basename 相同，旧实现它们的 doc_id 相同 → upsert 互相覆盖、
            # delete 误删另一文件的元数据，rag_outline 可能返回错误文档的目录。
            doc_id = _hash_source(md_path or doc_name)

            # Delete any previous entry (and the placeholder row), then add the new one.
            try:
                table.delete(f"doc_id = '{doc_id}'")
                table.delete("doc_id = ''")
                # 兼容旧数据：历史版本 doc_id = hash(basename)，同名的旧行
                # doc_id 不同、不会被上面删掉，会成为影子行（同名匹配时可能
                # 返回过期 digest）。按 doc_name + md_path 不匹配清掉它们。
                if md_path:
                    safe_name = doc_name.replace("'", "''")
                    safe_md = md_path.replace("'", "''")
                    table.delete(
                        f"doc_name = '{safe_name}' AND md_path != '{safe_md}'"
                    )
            except Exception:
                pass
            table.add([{
                "doc_id": doc_id,
                "doc_name": doc_name,
                "digest": digest,
                "chapters": chapters,
                "md_path": md_path,
            }])
            logger.info("  ├─ 文档元数据已写入: %s (digest %d 字符)", doc_name, len(digest))
            # 写入后失效缓存，下次读取时重新加载。
            self._documents_cache = None

    def _load_documents(self) -> List[Dict[str, Any]]:
        """Load (and cache) all non-placeholder document rows.

        Returns a list of dicts, filtered to drop the placeholder row
        (``doc_name`` empty).  Cached in ``_documents_cache``.

        Only the lightweight columns are read — the (now unused) ``markdown``
        column must never be pulled into the cache, otherwise the whole corpus
        is duplicated in memory for the process lifetime.
        """
        with self._documents_lock:
            if self._documents_cache is not None:
                return self._documents_cache
            table = self._ensure_documents_table_locked()
            if table is None:
                return []
            try:
                cols = ["doc_id", "doc_name", "digest", "chapters", "md_path"]
                try:
                    rows = table.to_arrow(columns=cols).to_pylist()
                except Exception:
                    # 旧表没有 md_path 列（lancedb 0.25 的 to_arrow 也不支持
                    # columns 参数）→ 用存储层列裁剪，绝不能全表物化：
                    # 旧 schema 里还有完整 markdown 列，全读 = 整份语料进内存。
                    # 按 schema 实际存在的列投影（markdown 永不读取）。
                    avail = [c for c in getattr(table.schema, "names", []) if c != "markdown"]
                    pick = [c for c in cols if c in avail] or avail
                    n = table.count_rows()
                    rows = (
                        table.search()
                        .limit(max(n, 1))
                        .select(pick)
                        .to_arrow()
                        .to_pylist()
                    )
            except Exception as e:
                logger.warning("读取文档元数据失败: %s", e)
                return []
            rows = [r for r in rows if r.get("doc_name")]
            for r in rows:
                r.pop("markdown", None)  # 兼容旧数据：绝不缓存 markdown
            self._documents_cache = rows
            return rows

    @_store_op
    def get_document_digest(self, doc_name: str) -> Optional[Dict[str, Any]]:
        """Return {doc_name, digest, markdown, chapters} for a document by name.

        Matches by exact basename or by substring (so the model can reference
        a book by a partial title).  Returns None if not found.
        """
        self._ensure_initialized()
        rows = self._load_documents()
        if not rows:
            return None

        # 1. Exact match (case-insensitive) on doc_name or basename.
        target = doc_name.strip().lower()
        for r in rows:
            name = (r.get("doc_name") or "").lower()
            if name == target or os.path.basename(name) == target:
                return {
                    "doc_name": r.get("doc_name", ""),
                    "digest": r.get("digest", ""),
                    "chapters": r.get("chapters", ""),
                    "md_path": r.get("md_path", ""),
                }
        # 2. Substring match (partial title).
        for r in rows:
            name = (r.get("doc_name") or "").lower()
            if target and (target in name or name in target):
                return {
                    "doc_name": r.get("doc_name", ""),
                    "digest": r.get("digest", ""),
                    "chapters": r.get("chapters", ""),
                    "md_path": r.get("md_path", ""),
                }
        return None

    @_store_op
    def list_documents(self) -> List[str]:
        """Return the list of stored document names (for the outline tool)."""
        self._ensure_initialized()
        rows = self._load_documents()
        return sorted({r.get("doc_name", "") for r in rows if r.get("doc_name")})

    @_store_op
    def all_document_rows(self) -> List[Dict[str, Any]]:
        """返回 documents 表全部行的**浅拷贝**（按列读取，不含 markdown）。

        供「按 md_path/doc_name 批量对账」的调用方一次性快照 —— 逐个调
        ``get_document_by_md_path`` 会让每次写入失效缓存后重新全表加载，
        大知识库下退化成 O(N) 次全表物化。浅拷贝避免调用方改到缓存。
        """
        self._ensure_initialized()
        return [dict(r) for r in self._load_documents()]

    @_store_op
    def delete_document(self, doc_name: str) -> None:
        """Remove a document's metadata entry.

        按**两种键**删：``doc_id``（= md5(doc_name)）与 ``doc_name`` 字面量。
        ``upsert_document`` 的 doc_id 在传入 ``md_path`` 时取 md5(md_path)、
        没传时取 md5(doc_name)——只按 doc_id 删会漏掉后一类（以及历史上按
        basename 建键的旧行），留下永远不会被清理的孤儿元数据。
        """
        self._ensure_initialized()
        with self._documents_lock:
            table = self._ensure_documents_table_locked()
            if table is None:
                return
            doc_id = _hash_source(doc_name or "")
            safe_name = (doc_name or "").replace("'", "''")
            try:
                table.delete(f"doc_id = '{doc_id}' OR doc_name = '{safe_name}'")
            except Exception as e:
                logger.warning("删除文档元数据失败: %s", e)
            self._documents_cache = None

    @_store_op
    def get_document_by_md_path(self, md_path: str) -> Optional[Dict[str, Any]]:
        """按 markdown 缓存路径精确查找文档元数据。

        缓存文件名经过 ``_safe_filename`` 清洗（空格、括号、中文标点都被丢
        掉），从文件名反推 doc_name 必然失真 —— 于是「Chapter 1.pdf」的缓存
        被认成「Chapter1.pdf」，每轮同步都被判为缺记录并写入一条重复行。
        ``md_path`` 是入库时原样记录的，用它匹配才可靠。
        """
        if not md_path:
            return None
        self._ensure_initialized()
        target = os.path.abspath(md_path)
        for r in self._load_documents():
            stored = str(r.get("md_path") or "")
            if stored and os.path.abspath(stored) == target:
                return {
                    "doc_name": r.get("doc_name", ""),
                    "digest": r.get("digest", ""),
                    "chapters": r.get("chapters", ""),
                    "md_path": stored,
                }
        return None

    @_store_op
    def delete_document_by_md_path(self, md_path: str) -> bool:
        """删除指定 markdown 缓存对应的元数据行。"""
        if not md_path:
            return False
        self._ensure_initialized()
        with self._documents_lock:
            table = self._ensure_documents_table_locked()
            if table is None:
                return False
            target = os.path.abspath(md_path)
            safe = target.replace("'", "''")
            try:
                table.delete(f"md_path = '{safe}'")
            except Exception as e:
                logger.warning("按 md_path 删除文档元数据失败: %s", e)
                return False
            self._documents_cache = None
            return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash_source(source: str) -> str:
    """Create a short stable hash for a source path."""
    import hashlib
    return hashlib.md5(source.encode()).hexdigest()[:12]


def _normalize(vec: List[float]) -> List[float]:
    """Normalize a vector to unit length for cosine similarity via L2 distance."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]
