"""RAG engine — orchestrates document ingestion, chunking, embedding, and retrieval."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from .parsers import extract_text, SUPPORTED_EXTENSIONS
from .cleaner import clean_text, clean_docling_markdown, normalize_markdown
from .chunker import chunk_documents
from .document_outline import (
    build_digest,
    has_markdown_headings,
    plain_text_to_markdown,
)
from .vector_store import DEFAULT_RERANK_MODEL, VectorStore
from ..atomic_io import atomic_write_json

logger = logging.getLogger(__name__)


def _is_stale_digest(old_digest: str) -> bool:
    """判断 documents 表中已存的 digest 是否为旧格式、需要重刷。

    旧格式特征（任一即判定为 stale）：
      * 含「# 章节摘要」段 —— 历史版本会生成逐章摘要，已废弃；
      * 含「# 目录」但后面只有空白（目录为空）—— 说明标题提取失败。
    """
    if "# 章节摘要" in old_digest:
        return True
    # 目录为空的旧记录：`# 目录\n` 后紧跟空白，无任何标题行。
    if "# 目录" in old_digest:
        body = old_digest.split("# 目录", 1)[1].strip()
        if not body:
            return True
    return False

# Default number of worker threads for parallel ingestion. Tuned for typical
# local-disk + CPU-bound parsing workloads (PDF OCR, docx parsing, etc.).
# Embedding/vector-store writes stay serial because LanceDB's table lock is
# not thread-safe and the embedding model instance can't be shared across threads.
# 4 → 2（本地模式内存实测修订）：每个并行 worker 都持有完整解析结果（大
# PDF/电子书可达数百 MB），叠加进程内常驻的本地模型（主 LLM 2.2GB + 辅助
# 1.6GB + OCR 0.6GB + 嵌入 0.3GB），4 路并发在 16GB 机器上峰值逼近系统上限，
# 长同步触发 macOS 内存压力（系统级卡死/崩溃）。解析是子进程/CPU 密集型，
# 2 路已足够喂饱串行嵌入。
DEFAULT_PARALLEL_WORKERS = 2


def configure_mlx_memory_limits(cache_bytes: int = 1_000_000_000) -> None:
    """给 MLX 显式设 Metal 缓冲缓存上限（#131：同步 8→18GB 的根因）。

    MLX 0.32 在 macOS 上默认 ``cache_limit = recommendedMaxWorkingSetSize``
    （统一内存的 ~75%）——空闲后的 Metal buffer 不归还系统，而是留在池里
    一直涨到这个上限。16GB 机器上意味着缓存 alone 可到 ~12GB；再叠加
    docling converter（torch + 布局模型，~4.6GB）驻留，进程 phys_footprint
    冲到 19GB（实测 2026-10-20），系统开始疯狂 swap。

    只限缓存：``clear_cache`` 后的空闲缓冲超过 cache_bytes 就立即归还系统。
    活跃内存（模型权重 + 推理激活）不受影响，不设 memory_limit 以免干扰
    同进程的 aux LLM 长上下文推理。同步开始时调用一次即可。
    """
    try:
        import mlx.core as mx
        mx.metal.set_cache_limit(cache_bytes)
        logger.info("MLX cache limit set to %.1f GB", cache_bytes / 1e9)
    except Exception as e:
        logger.debug("MLX cache limit not set: %s", e)


def release_mlx_pool() -> None:
    """把 MLX 统一内存池的空闲缓冲与解析器产生的引用循环还给系统。

    MLX 的 buffer pool 只借不还：嵌入/OCR 的瞬时激活在池里留下碎片后 RSS
    只升不降；marker/docling 等解析器会制造引用循环，仅靠引用计数不释放。
    长知识库同步里两者叠加，RSS 单调爬升直到触发 macOS 内存压力（系统卡死
    /崩溃）。在消费线程每处理完一个文件调用一次（串行、有界队列节拍）。
    """
    try:
        import gc
        gc.collect()
    except Exception:
        pass
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass

# worker→主线程队列容量：限制 worker 最多超前产出多少文件的 chunks，
# 超过后 worker 阻塞（背压），避免大知识库时 chunks 在内存堆积。
_QUEUE_CAPACITY_FACTOR = 2  # = parallel_workers * 2

# 哨兵：worker 全部完成后投递到队列，主线程据此退出消费循环
_SENTINEL = object()

# 向量库写入批大小。LanceDB 每次 add 都会产生新 fragment，批太小会在大库上
# 留下海量碎片，检索退化为逐段扫描；批太大则单次失败要重做的量变多。
_WRITE_BATCH_SIZE = 200


def _md_meta_path(md_path: str) -> str:
    """Markdown 缓存的 sidecar 路径：保存「缓存文件名 → 原始文档名」映射。"""
    return md_path + ".meta.json"


def _write_md_meta(md_path: str, payload: Dict[str, Any]) -> None:
    """写 sidecar。失败只告警——它只是加速/纠错的辅助数据，不影响主流程。"""
    try:
        with open(_md_meta_path(md_path), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except OSError as e:
        logger.warning("Markdown meta 写入失败: %s: %s", md_path, e)


def _read_md_meta(md_path: str) -> Optional[Dict[str, Any]]:
    """读 sidecar；不存在或损坏返回 None。"""
    try:
        with open(_md_meta_path(md_path), "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------------------
# ingest 互斥锁（跨实例，按 rag_dir 键控）
#
# UI 的 SyncKnowledgeWorker 与 agent 的 rag_ingest 工具各自持有**不同的**
# RAGEngine 实例，但 workspace 相同 → rag_dir 相同 → 底层是同一份 LanceDB
# 表与 manifest.json。实例级锁对这两条真实并发路径无效，因此锁提升到模块
# 级并按 rag_dir 索引：同一工作区的任何两条 ingest 都真正互斥。
# ---------------------------------------------------------------------------

_INGEST_LOCKS: Dict[str, threading.Lock] = {}
_INGEST_LOCKS_GUARD = threading.Lock()


def _ingest_lock_for(rag_dir: str) -> threading.Lock:
    with _INGEST_LOCKS_GUARD:
        lock = _INGEST_LOCKS.get(rag_dir)
        if lock is None:
            lock = threading.Lock()
            _INGEST_LOCKS[rag_dir] = lock
        return lock


# ---------------------------------------------------------------------------
# Manifest — tracks which documents have been processed so incremental syncs
# can skip unchanged files entirely (no chunk, no embed, no upsert).
# ---------------------------------------------------------------------------

def _manifest_path(rag_dir: str) -> str:
    """Path to the incremental-sync manifest inside the RAG directory."""
    return os.path.join(rag_dir, "manifest.json")


# 进程内的 manifest 串行化（跨进程由 _ManifestLock 的 flock 保证）。
#
# 必须是**不可重入**的 Lock，并且与 flock 的重入计数解耦：flock 绑定在 open
# file description 上，同一线程对同一文件再 open 一次拿到的 fd 与已有的
# LOCK_EX **互斥**（不区分同进程）。若这里用 RLock，嵌套进入就会让线程第二次
# flock 自己等自己 → 永久挂起。用普通 Lock 时嵌套会立刻在线程锁上暴露。
_MANIFEST_THREAD_LOCK = threading.Lock()
# 线程本地的重入深度：同一线程嵌套进入只加计数，不再重复 flock。
_MANIFEST_TLS = threading.local()


def _manifest_lock_path(rag_dir: str) -> str:
    """Path to the manifest lock file (cross-process advisory lock)."""
    return os.path.join(rag_dir, "manifest.lock")


def _load_manifest_unlocked(rag_dir: str) -> Dict[str, Dict[str, Any]]:
    """``_load_manifest`` 的实现（不加锁，供持锁路径内部调用）。"""
    path = _manifest_path(rag_dir)
    if not os.path.exists(path):
        logger.debug("RAG: manifest not found at %s (first sync)", path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("files"), dict):
            logger.debug("RAG: loaded manifest with %d entries from %s", len(data["files"]), path)
            return data["files"]
        logger.warning("RAG: manifest has unexpected format, ignoring")
        return {}
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("RAG: failed to load manifest: %s", e)
        return {}


def _load_manifest(rag_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load the document manifest (加锁读，避免读到写入一半的状态）。"""
    with _ManifestLock(rag_dir):
        return _load_manifest_unlocked(rag_dir)


class _ManifestLock:
    """manifest 的跨进程/跨实例互斥锁（fcntl.flock advisory lock）。

    同步可能由多个入口并发触发：UI 的「同步知识库」用一个 RAGEngine，agent
    侧的自动 ingest 用另一个，二者各持一份内存中的 manifest 快照。旧实现
    ``_save_manifest`` 是**全量覆盖写**——后写者会用自己那份旧快照抹掉前者
    刚写入的条目，于是「已同步」的文件被判定为「未同步」，下次全量重跑。

    Windows 上没有 fcntl：退化为「只加线程锁 + 直接写」（该平台不支持多进程
    同步场景，且不会比现状更差）。
    """

    _HAVE_FCNTL = True
    try:
        import fcntl  # noqa: F401
    except ImportError:  # pragma: no cover - Windows
        _HAVE_FCNTL = False

    def __init__(self, rag_dir: str):
        self._rag_dir = rag_dir
        self._fh = None
        self._thread_lock = _MANIFEST_THREAD_LOCK
        # 本次 __enter__ 是否真的拿到了 flock（嵌套时不重复加锁/解锁）。
        self._acquired = False

    def __enter__(self):
        depth = getattr(_MANIFEST_TLS, "depth", 0)
        if depth > 0:
            # 同一线程重入：只加深计数，不再 flock（否则会等自己）。
            _MANIFEST_TLS.depth = depth + 1
            return self
        self._thread_lock.acquire()
        _MANIFEST_TLS.depth = 1
        self._acquired = True
        if self._HAVE_FCNTL:
            try:
                import fcntl
                os.makedirs(self._rag_dir, exist_ok=True)
                self._fh = open(_manifest_lock_path(self._rag_dir), "a+")
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
            except Exception as e:
                logger.debug("RAG: manifest 文件锁不可用 (%s)，退化为仅线程锁", e)
                if self._fh is not None:
                    try:
                        self._fh.close()
                    except Exception:
                        pass
                    self._fh = None
        return self

    def __exit__(self, *exc):
        depth = getattr(_MANIFEST_TLS, "depth", 0)
        if depth > 1:
            _MANIFEST_TLS.depth = depth - 1
            return False
        _MANIFEST_TLS.depth = 0
        if self._fh is not None:
            try:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None
        if self._acquired:
            self._acquired = False
            self._thread_lock.release()
        return False


def _save_manifest(
    rag_dir: str,
    files: Dict[str, Dict[str, Any]],
    removed: Optional[set] = None,
) -> None:
    """Persist the manifest atomically (unique tmp + os.replace), **读改写合并**。

    不再全量覆盖：写入前重新读一次磁盘上的最新内容，把本次调用没碰过的条目
    保留下来。否则两个并发的同步（UI 一个、agent 一个）会互相抹掉对方的
    记录。

    ``removed`` 是**本次有意删除**的 source 集合：这些键即使在磁盘版本里
    存在也必须删掉（否则「别的实例刚写入的条目」和「我刚删掉的条目」无法
    区分，删除会被合并逻辑复活）。
    """
    os.makedirs(rag_dir, exist_ok=True)
    with _ManifestLock(rag_dir):
        merged = dict(_load_manifest_unlocked(rag_dir))
        for key in removed or ():
            merged.pop(key, None)
        merged.update(files)
        atomic_write_json(_manifest_path(rag_dir), {"files": merged})


def _file_signature(filepath: str) -> Optional[Dict[str, Any]]:
    """Compute the lightweight signature of a file (no content read).

    Used to decide whether a file needs reprocessing without parsing it.
    返回 None 表示 stat 失败（文件在扫描后被删 / 权限变化 / 网络挂载抖动）——
    调用方应跳过该文件并计入 errors，而不是让整轮同步中断。
    """
    try:
        st = os.stat(filepath)
    except OSError:
        return None
    return {"mtime": st.st_mtime, "size": st.st_size}


def _content_hash(text: str) -> str:
    """Stable hash of cleaned text content.

    Stored in the manifest so manual inspection is traceable.  Note this is
    the hash of the *parsed* text — too expensive to compute at scan time,
    so change detection uses :func:`_source_fingerprint` (raw bytes) instead.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# 超过这个体积就改用「头+中+尾」取样指纹：全量哈希要读整个文件，
# 而指纹在「本来要跳过」的路径上逐文件调用 —— 10GB 知识库每次同步固定
# 10GB 读盘，机械盘/网络盘上直接把同步拖到几分钟。
_FULL_HASH_MAX_BYTES = 4 * 1024 * 1024
_SAMPLE_BYTES = 64 * 1024


def _source_fingerprint(path: str, size: Optional[int] = None) -> str:
    """Cheap fingerprint of the **raw file bytes** (not the parsed text).

    Only called when mtime+size already match, i.e. on the would-skip path.
    Reading the bytes is orders of magnitude cheaper than re-parsing and
    re-embedding, and it catches the `cp -p` / zip-extract / cloud-sync
    rollback cases where mtime and size are preserved but content differs.

    小文件（≤4MB）全量哈希；大文件取「头 + 中 + 尾」三段各 64KB。三段采样
    已覆盖绝大多数「内容被换掉」的情形（改写头部/中部/尾部/追加），同时把
    单文件读盘量从 size 降到固定 192KB。
    """
    try:
        if size is None:
            size = os.path.getsize(path)
        h = hashlib.sha256()
        h.update(str(size).encode())
        with open(path, "rb") as f:
            if size <= _FULL_HASH_MAX_BYTES:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            else:
                h.update(f.read(_SAMPLE_BYTES))
                mid = max(0, size // 2 - _SAMPLE_BYTES // 2)
                f.seek(mid)
                h.update(f.read(_SAMPLE_BYTES))
                f.seek(max(0, size - _SAMPLE_BYTES))
                h.update(f.read(_SAMPLE_BYTES))
        return h.hexdigest()[:16]
    except OSError:
        return ""


def _escape_like(s: str) -> str:
    """Escape a value used inside a SQL ``LIKE`` literal (with ``ESCAPE '\\'``).

    1. SQL 字符串字面量：单引号翻倍，否则文件名含 ``'``（如 ``it's.pdf``）
       会拼出畸形 SQL / 注入。
    2. LIKE 通配符：``%`` ``_`` ``\\`` 前面加反斜杠。LanceDB 的 LIKE 默认
       不带 ESCAPE 子句，必须在使用处显式写 ``ESCAPE '\\'``，否则 ``\\%``
       不会被当作字面量，含 ``%``/``_`` 的文件名会匹配到所有文档。
    """
    s = s.replace("\\", "\\\\")
    for ch in ("%", "_"):
        s = s.replace(ch, "\\" + ch)
    return s.replace("'", "''")


# 统一 PDF/office 解析引擎（"docling" | "marker" | "ovis"）。经 set_pdf_engine
# 注入，默认 docling——与 rag_pdf_engine 配置项的默认值保持一致。
_pdf_engine = "docling"

# pandoc 优先的格式集合（与 ebook_parser.PANDOC_EXTENSIONS 一致，模块级导入
# 失败时的兜底副本——ebook_parser 导入本身无重依赖，直接复用）。
from .ebook_parser import PANDOC_EXTENSIONS as _PANDOC_FORMAT_EXTENSIONS  # noqa: E402
from .ebook_parser import EBOOK_EXTENSIONS as _EBOOK_FORMAT_EXTENSIONS  # noqa: E402


def set_pdf_engine(engine: str) -> None:
    """注入 rag_pdf_engine 配置（同 parsers.set_ocr_config 的模块级注入模式）。

    SyncKnowledgeWorker / agent 工具同进程运行，MainWindow 初始化时注入一次
    即对两条路径生效。非法值回退 "docling"（保守默认）。
    """
    global _pdf_engine
    _pdf_engine = engine if engine in ("docling", "marker", "ovis") else "docling"
    # 一次性提示（#133）：marker / ovis 是重依赖，不进 deps.py 的自动安装
    # 列表（auto_install 会装全部 optional，marker-pdf 拖 torch+1.4GB VLM）。
    # 用户手选却没装时回退链会静默落到 docling——这里显式告警。
    if _pdf_engine in ("marker", "ovis"):
        try:
            import importlib.util
            pkg = "marker" if _pdf_engine == "marker" else "mlx_vlm"
            pip_name = "marker-pdf" if _pdf_engine == "marker" else "mlx-vlm"
            if importlib.util.find_spec(pkg) is None:
                logger.warning(
                    "rag_pdf_engine=%s 但 %s 未安装，将回退 docling"
                    "（如需启用：pip install %s）",
                    _pdf_engine, pkg, pip_name,
                )
        except Exception:
            pass


def set_ovis_model_config(engine: str, name: str, path: str) -> None:
    """注入 OvisOCR2 模型配置（rag_ocr_engine/name/path）到 ovis 解析器。"""
    try:
        from .ovis_ocr_parser import set_ovis_model_config as _set
        _set(engine, name, path)
    except Exception:
        pass


# 回退链引擎的连续失败熔断（2026-10-23）：主引擎**系统性**失败（docling 版本
# 不兼容、权重损坏等）时，回退链会把下一棒引擎逐个拉起来试——marker ~3GB /
# ovis ~0.9GB 的权重常驻到同步结束，与 docling 的 4.6GB 叠加；且每次失败要
# 实跑数分钟。同一引擎连续失败 _ENGINE_FAIL_LIMIT 次即本轮同步内停用，并
# **显式释放其已加载的单例**（峰值从「全部叠加」降为「一次一个」）。
# 只对**异常**计数：返回 None 多为「格式不适用」（如 txt 对 docling），熔断
# 它会把本该由它解析的 PDF 悄悄降级到纯文本。
_ENGINE_FAIL_LIMIT = 3
_engine_fail_streak: Dict[str, int] = {}
_engine_fail_lock = threading.Lock()


def reset_engine_fail_streak() -> None:
    """复位回退链熔断计数（每轮 ingest 开始时调用）。"""
    global _engine_fail_streak
    with _engine_fail_lock:
        _engine_fail_streak = {}


def _note_engine_failure(engine: str) -> bool:
    """记录一次引擎失败；返回 True 表示该引擎已触发熔断。"""
    with _engine_fail_lock:
        n = _engine_fail_streak.get(engine, 0) + 1
        _engine_fail_streak[engine] = n
    return n >= _ENGINE_FAIL_LIMIT


def _release_engine_singleton(engine: str) -> None:
    """释放指定引擎的进程级单例（熔断时调用，回收其权重内存）。"""
    try:
        if engine == "marker":
            from .marker_parser import release_converter as _release
        elif engine == "ovis":
            from .ovis_ocr_parser import release_model as _release
        else:
            from .docling_parser import release_converter as _release
        _release()
    except Exception as e:
        logger.debug("RAG: release engine %s failed: %s", engine, e)


def _extract_with_docling(filepath: str) -> tuple:
    """Extract text, preferring the configured unified parser (Markdown output).

    Returns ``(text, is_markdown)`` where ``is_markdown`` is True when the
    text came from docling/marker and is already well-formed Markdown (so the
    caller can skip clean/normalize).  When the preferred engine is unavailable
    or unsuitable for the format, falls back to the other unified parser and
    then to the per-format parsers.

    引擎选择：``rag_pdf_engine`` 配置（经 :func:`set_pdf_engine` 注入）。
    默认 "docling"；"marker" 时 marker-pdf 优先；"ovis" 时 OvisOCR2 VLM
    优先（#133，PyMuPDF 取页/裁表 → VLM）。三者构成回退链，最后落到
    PyMuPDF 纯文本兜底。

    例外：html/docx/tex 等 pandoc 支持格式优先走 pandoc（输出更干净的
    Markdown 且速度快一个量级），pandoc 不可用才回落 docling/按格式解析器。
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext in _PANDOC_FORMAT_EXTENSIONS:
        from .ebook_parser import pandoc_to_markdown, clean_ebook_markdown
        md = pandoc_to_markdown(filepath)
        if md is not None:
            # pandoc gfm 会保留 raw HTML（pandoc 输出未清洗直接 return 是
            # 入库文本带 HTML 标签的另一条泄漏路径）；统一过 ebook 清洗。
            cleaned = clean_ebook_markdown(md)
            if cleaned:
                return cleaned, True
    # 回退链（#131/#133 用户定版）：主引擎（rag_pdf_engine 配置，默认 docling）
    # → 其余统一引擎（marker/docling）→ ovis（OvisOCR2 VLM，#133 起取代
    # PyMuPDF+RapidOCR 成为第三选项/兜底——输出带版面的 Markdown 而非纯文本）
    # → PyMuPDF 纯文本（最后手段）。ovis 不可用/失败则静默落到纯文本。
    order: List[str] = []
    for e in (_pdf_engine, "docling", "marker", "ovis"):
        if e and e not in order:
            order.append(e)
    for engine in order:
        if _engine_fail_streak.get(engine, 0) >= _ENGINE_FAIL_LIMIT:
            logger.debug("RAG: 引擎 %s 已熔断（连续失败 %d 次），本轮同步跳过",
                         engine, _ENGINE_FAIL_LIMIT)
            continue
        try:
            if engine == "marker":
                from .marker_parser import extract_markdown
            elif engine == "ovis":
                from .ovis_ocr_parser import extract_markdown
            else:
                from .docling_parser import extract_markdown
            markdown = extract_markdown(filepath)
            if markdown is not None:
                with _engine_fail_lock:
                    _engine_fail_streak[engine] = 0
                return markdown, True
        except Exception as e:
            # 不再静默吞掉：2026-10-22 docling 的 NameError（池化改造误删
            # 模块级 _converter_unavailable）就是被这里吞掉的，症状是「PDF
            # 全部静默降级 marker、docling 毫无报错」，极难定位。debug 级
            # 记录引擎名 + 异常类型 + 信息，日志窗口开 debug 即可见。
            logger.debug("RAG: 解析引擎 %s 失败（走下一棒）: %s: %s",
                         engine, type(e).__name__, e)
            if _note_engine_failure(engine):
                logger.warning(
                    "RAG: 解析引擎 %s 连续失败 %d 次，本轮同步内停用并释放其"
                    "模型（后续文件走下一棒；若这是误判请检查该引擎依赖）",
                    engine, _ENGINE_FAIL_LIMIT,
                )
                _release_engine_singleton(engine)
    return extract_text(filepath), False


def _shard_files_balanced(files: List[str], num_workers: int) -> List[List[str]]:
    """把文件列表分片给 worker：按大小降序 + 贪心最轻负载（LPT 调度）。

    旧实现按 ``ceil(n/workers)`` 连续切片：n=9、workers=4 时第 4 片为空
    （3×3=9 已分完），第 4 个 worker 领空启动即退；且慢文件（扫描书要
    逐页 OCR）按文件名顺序扎在同一片时，其余 worker 早早干完长时空转，
    同步尾延迟被拉长。

    现按文件大小降序，每次把文件分给当前累计字节数最小、并列时当前
    已分配文件数最少的 worker：最大的 ``num_workers`` 个文件必然分属
    不同的 worker，n ≥ workers 时空片不可能出现，慢书自动摊开。大小
    只是处理时长的近似（扫描书按页数 OCR、文本书直读），但显著优于
    随机顺序的连续切片。
    """
    if num_workers <= 1:
        return [list(files)]
    sized: List[List[int]] = []
    for f in files:
        try:
            sized.append([os.path.getsize(f), 0, 0])
        except OSError:
            # 扫描后文件消失（TOCTOU）：按 0 处理，worker 内再行隔离
            sized.append([0, 0, 0])
    # 大小降序（并列时按原序号稳定），随后贪心分配
    order = sorted(range(len(files)),
                   key=lambda i: (-sized[i][0], i))
    shards: List[List[str]] = [[] for _ in range(num_workers)]
    totals = [0] * num_workers
    counts = [0] * num_workers
    for i in order:
        # 负载键：累计字节 → 已分配文件数 → worker 序号（保证确定性）
        k = min(range(num_workers), key=lambda w: (totals[w], counts[w], w))
        shards[k].append(files[i])
        totals[k] += sized[i][0]
        counts[k] += 1
    return shards


class RAGEngine:
    """Local knowledge base engine.

    Ingests documents from a knowledge base directory, extracts text,
    cleans it, chunks it, embeds it, and stores vectors for retrieval.

    The per-document pipeline (parse → clean → write markdown → chunk) runs
    in a thread pool to parallelize CPU-bound parsing (esp. PDF OCR). Vector
    store writes remain serial for thread safety.
    """

    def __init__(
        self,
        workspace: str,
        knowledge_base: str = "",
        embedding_model: str = "",
        embedding_function: Any = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
        overlap_percent: float = 0.05,
        rerank_model: str = DEFAULT_RERANK_MODEL,
        rerank_enabled: bool = True,
        parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
    ):
        self._workspace = workspace
        self._knowledge_base = knowledge_base
        self._rag_dir = os.path.join(workspace, "rag")
        self._markdown_dir = os.path.join(self._rag_dir, "documents")
        self._vector_dir = os.path.join(self._rag_dir, "vectors")
        self._embedding_model = embedding_model or "nomic-ai/nomic-embed-text-v1.5-Q"
        self._custom_ef = embedding_function  # for testing injection
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._min_chunk_size = min_chunk_size
        self._overlap_percent = overlap_percent
        self._rerank_model = rerank_model
        # 重排序总开关。config.rag_rerank_enabled 此前从未被读取——用户在配置里
        # 关掉重排序或换模型都完全无效（README / 设计文档却宣称可用）。这里接线：
        # 关闭时 search 直接走向量+BM25 混合排序，不加载 ~2.3GB 的 reranker。
        self._rerank_enabled = bool(rerank_enabled)
        self._parallel_workers = max(1, int(parallel_workers))
        self._store: Optional[VectorStore] = None
        self._store_lock = threading.Lock()  # 保护 _store 懒初始化的线程安全
        self._cancelled = threading.Event()  # 取消标志，cancel() 设置后各阶段检查退出
        # ingest 互斥：**跨实例**。UI 的 SyncKnowledgeWorker 与 agent 的
        # rag_ingest 工具各自 new 一个 RAGEngine（workspace 相同 → rag_dir
        # 相同 → 底层是同一份 LanceDB 表与 manifest 文件）。实例级锁对两条
        # 真实并发路径完全无效：两路 ingest 会同时写同一张表、同时
        # _save_manifest（temp+replace 后者覆盖），manifest 合并状态互相踩。
        # 改用模块级、按 rag_dir 键控的锁才能真正互斥。
        self._ingest_lock = _ingest_lock_for(self._rag_dir)

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def rag_dir(self) -> str:
        return self._rag_dir

    @property
    def markdown_dir(self) -> str:
        return self._markdown_dir

    @property
    def knowledge_base(self) -> str:
        return self._knowledge_base

    @knowledge_base.setter
    def knowledge_base(self, value: str) -> None:
        self._knowledge_base = value

    # ------------------------------------------------------------------
    # ingestion pipeline (流式：worker 并行解析+chunk，主线程增量 embed+store)
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        """请求取消正在进行的 ingest。各阶段检查 _cancelled 后提前退出。"""
        logger.info("RAG: cancel requested")
        self._cancelled.set()

    def _reset_cancel(self) -> None:
        self._cancelled.clear()

    def _is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def _dedup_manifest_by_content(
        self, manifest: Dict[str, Dict[str, Any]]
    ) -> int:
        """按 content_hash 清理重复副本的向量（B 方案后半程，2026-10-22 定版）。

        覆盖两类存量问题：
        1. 去重功能上线**前**，不同目录的同一本书已各自入库（两套向量、
           documents 表两行）——检索时同段内容双倍召回；
        2. 预扫描阶段漏掉的（如同一同步里两个副本都未变、走了跳过路径，
           或历史 manifest 条目没有 source_hash 没进 seen 表）。

        每组同 content_hash 保留一个 owner（无 duplicate_of 且路径字典序
        最小，确定性选择），其余副本：删除向量 + documents 表记录（rel_path /
        basename / 完整路径三种 id 都删，兼容旧版），manifest 条目标记
        ``duplicate_of=owner``——下次同步直接走跳过路径，不重新解析。
        已标记过的副本幂等跳过。另外把 owner 缺失/解析失败（无 content_hash）
        的僵尸副本提升为正常条目，下轮同步重新入库。返回删除向量的副本数。
        """
        # 1) 僵尸副本提升：duplicate_of 指向的条目没了/没入库成功。
        # 判据是**向量是否还在**（chunk_count），而不是有没有 content_hash：
        # 第 2 步清理副本时会删掉向量但保留 content_hash，此时若只摘标记，
        # 下轮会因 mtime/size/指纹未变被判「未修改」跳过 —— 向量已删且不会
        # 重新入库，内容永久检索不到。
        promoted = []
        for _fp, entry in manifest.items():
            owner = entry.get("duplicate_of")
            if not owner:
                continue
            owner_entry = manifest.get(owner)
            if owner_entry is None or not owner_entry.get("content_hash"):
                if entry.get("chunk_count"):
                    entry.pop("duplicate_of", None)   # 向量还在 → 摘标记转正
                else:
                    promoted.append(_fp)      # 迭代后删（避免边遍历边删）
        for _fp in promoted:
            manifest.pop(_fp, None)
        # 2) 按 content_hash 分组清理。**包含已标记条目**——它可能还带着
        # 旧向量（force 模式下字典序在前的副本被重解析、另一个被标记时，
        # 或历史 manifest 里两个副本都入过库），不能只按「无标记」分组，
        # 否则这批旧向量永远清理不掉、检索继续双倍召回。
        groups: Dict[str, List[str]] = {}
        for fp, entry in manifest.items():
            ch = entry.get("content_hash")
            if ch:
                groups.setdefault(ch, []).append(fp)
        removed = 0
        store = None
        for _ch, paths in groups.items():
            if len(paths) < 2:
                continue
            # owner：优先无 duplicate_of 标记的，其次字典序最小（确定性）。
            unmarked = [p for p in paths if not manifest[p].get("duplicate_of")]
            owner = sorted(unmarked)[0] if unmarked else sorted(paths)[0]
            for dup_path in paths:
                if dup_path == owner:
                    continue
                entry = manifest[dup_path]
                if (entry.get("duplicate_of") == owner
                        and not entry.get("chunk_count")):
                    continue                  # 上轮已清理（幂等）
                if entry.get("chunk_count"):
                    # 有旧向量才动 store（从未入库的副本无向量可删）
                    if store is None:
                        store = self._get_store()
                    try:
                        store.delete_by_source(dup_path)
                        # documents 表同样清（三种 id 兼容，见 _cleanup_deleted_files）
                        store.delete_document(
                            os.path.relpath(dup_path, self._knowledge_base))
                        store.delete_document(os.path.basename(dup_path))
                    except Exception as e:
                        logger.warning("RAG: 清理副本向量失败 %s: %s", dup_path, e)
                    # Markdown 缓存 + digest 行必须一起清：否则紧接着的 Step 7
                    # （ensure_digests_for_cached_documents）看到「有缓存、无
                    # 记录」会用旧缓存重新生成一条 digest → documents 表出现
                    # 「有目录、无向量」的孤儿行，rag_outline 持续列出一个
                    # 检索不到的文档（与 _cleanup_deleted_files 保持自洽）。
                    md_path = self._cache_md_path(dup_path)
                    try:
                        store.delete_document_by_md_path(md_path)
                    except Exception as e:
                        logger.debug("RAG: 清理副本 digest 失败 %s: %s", dup_path, e)
                    try:
                        if os.path.isfile(md_path):
                            os.remove(md_path)
                        sidecar = _md_meta_path(md_path)
                        if os.path.isfile(sidecar):
                            os.remove(sidecar)
                    except OSError as e:
                        logger.warning("RAG: 删除副本 Markdown 缓存失败 %s: %s",
                                       md_path, e)
                entry["duplicate_of"] = owner
                entry["chunk_count"] = 0       # 向量已删，计数归零（幂等依据）
                removed += 1
        if promoted or removed:
            logger.info(
                "RAG: 内容去重清理: %d 个副本向量已删, %d 个僵尸副本已处理",
                removed, len(promoted),
            )
        return removed

    def dedup_vectors(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, int]:
        """排查并清理向量库内内容完全相同的重复向量（保留每组最早入库的一份）。

        「入库时间」以 markdown 缓存文件的 mtime 为代理——缓存写入时刻即
        解析入库时刻（ingest 处理后立即写盘）；无缓存的源（缓存被清/文档
        已删）记 0，按 (source, chunk_index) 字典序兜底，保证结果确定性。

        内容一字不差才算重复（归一化空白后哈希）；不同文档的相同样板文字
        （如纯「目录」页）同样只留一份——降低检索噪声。近似重复不处理。
        """
        # 缓存：同一 source 的数千个切片共用一个 stat（旧实现**每行一次**
        # os.path.getmtime，全表 N 行 N 次系统调用且完全重复）。
        _ts_cache: Dict[str, float] = {}

        def _order_key(rec: Dict[str, Any]) -> tuple:
            src = rec.get("source") or ""
            ts = _ts_cache.get(src)
            if ts is None:
                try:
                    rel = os.path.relpath(src, self._knowledge_base)
                    md_path = os.path.join(
                        self._markdown_dir, _safe_filename(rel) + ".md")
                    ts = os.path.getmtime(md_path)
                except OSError:
                    ts = 0.0
                _ts_cache[src] = ts
            return (ts, src, rec.get("chunk_index") or 0)

        store = self._get_store()
        return store.deduplicate_vectors(order_key=_order_key,
                                         progress_callback=progress_callback)

    def ingest(
        self,
        force: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Ingest entry point — mutually exclusive (see ``_ingest_lock``).

        Returns:
            统计 dict，或 ``{"error": ...}`` 表示本次请求被拒绝（已有同步在跑）。
        """
        if not self._ingest_lock.acquire(blocking=False):
            # 不阻塞等待：UI 同步与 agent 的 rag_ingest 是两条独立线程，排队
            # 等一次完整同步（可能几分钟）只会让调用方看起来卡死。
            logger.info("RAG ingest: another sync is already running; ignoring request")
            return {"error": "另一次同步正在进行，已忽略本次请求"}
        try:
            return self._ingest_locked(force, progress_callback, log_callback)
        finally:
            # 无论正常结束/取消/异常/提前 return 都要释放解析单例：agent 的
            # rag_ingest 工具路径不会调 close()（引擎长期持有），不释放会让
            # docling 池 + OCR + ovis 常驻到会话结束（数 GB）。
            try:
                self.release_parsers()
            except Exception as e:
                logger.debug("RAG: release_parsers error: %s", e)
            self._ingest_lock.release()

    def _ingest_locked(
        self,
        force: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Run the ingestion pipeline on the knowledge base directory.

        增量同步：通过 manifest.json 记录每个已处理文档的签名（mtime+size）。
        - 未修改文件：完全跳过（不 parse、不 chunk、不 embed、不 upsert）
        - 新增/修改文件：完整处理（parse → clean → chunk → embed → upsert）
        - 已删除文件：清理对应向量和 manifest 条目（无论 force 与否）
        - 中途取消：已完整入库的文件记入 manifest（下次跳过，不重做解析/
          清洗/切片/嵌入）；进行中或只入库一半的文件不记录，下次同步从断点
          续传（幂等 upsert，不留「记录完成却只入库一半」的永久缺失）

        流式架构：worker 池并行解析+chunk，主线程通过有界队列消费并增量写入
        vector store。内存占用 = O(parallel_workers * 单文件 chunks) + O(batch_size)，
        不再随知识库总规模线性增长。

        Args:
            force: 强制重新处理所有文件（忽略 manifest，仍走 upsert，不清库）
            progress_callback: 回调 (done, total, current_file) -> None

        Returns:
            统计 dict
        """
        self._reset_cancel()
        # 辅助函数：写入日志。当提供 log_callback（UI 同步场景）时，只走
        # callback——否则会与临时 logging handler 捕获的 logger.info 重复，
        # 导致同一条消息在 UI 日志面板出现两遍。无 callback 时（命令行/测试
        # 直接调用 ingest）走 logger.info 输出到终端。
        def _log(msg: str, *args) -> None:
            if args:
                msg = msg % args
            if log_callback is not None:
                # 回调是调用方（UI / 测试）的代码，可能抛异常。_progress 系列
                # 调用都包了 try/except，唯独这里没有 —— 一旦抛出，异常会穿出
                # ingest()，finally 里的 _abort.set() 就不执行，4 个 worker
                # 会永久阻塞在 q.put(timeout=2.0) 上，连同各自的切片内存一起
                # 泄漏。日志失败绝不该拖垮整条同步。
                try:
                    log_callback(msg)
                except Exception as e:  # pragma: no cover - 防御
                    logger.warning("RAG log_callback error: %s", e)
                return
            logger.info(msg)

        if not self._knowledge_base or not os.path.isdir(self._knowledge_base):
            logger.warning("RAG ingest: knowledge base dir not found: %s", self._knowledge_base)
            return {"error": f"Knowledge base directory not found: {self._knowledge_base}"}

        # 内存护栏前置到引擎层（覆盖**所有**调用方）：UI 同步路径在 worker 里
        # 调过，但 agent 的 rag_ingest 工具路径不会——那时主/辅 LLM（~4GB）+
        # KV 缓存常驻，MLX 缓存默认上限是统一内存的 ~75%（16GB 机器 ~12GB），
        # 再叠加 docling 4.6GB 直接爆。只设上限**不清缓存**：清缓存会把 agent
        # 预热好的系统提示 KV 一并 evict（UI 路径已卸载模型，清而无害）。
        configure_mlx_memory_limits()
        # 回退链熔断计数随每轮同步复位（见 _extract_with_docling）。
        reset_engine_fail_streak()

        t_start = time.monotonic()
        os.makedirs(self._rag_dir, exist_ok=True)
        os.makedirs(self._markdown_dir, exist_ok=True)

        stats: Dict[str, Any] = {
            "knowledge_base": self._knowledge_base,
            "files_found": 0,
            "files_extracted": 0,
            "files_skipped": 0,
            "files_deleted": 0,
            "files_duplicated": 0,
            "total_chars": 0,
            "chunks": 0,
            "errors": [],
            "cancelled": False,
        }
        stats_lock = threading.Lock()

        # Step 1: 扫描知识库目录
        _log("━━━ Step 1/6: 扫描知识库目录 ━━━")
        _log("  正在扫描: %s", self._knowledge_base)
        root = Path(self._knowledge_base)
        all_files: List[str] = []
        ext_counts: Dict[str, int] = {}
        for fp in root.rglob("*"):
            if (fp.is_file()
                    and fp.suffix.lower() in SUPPORTED_EXTENSIONS
                    and not fp.name.startswith(".")):
                all_files.append(str(fp))
                ext = fp.suffix.lower()
                ext_counts[ext] = ext_counts.get(ext, 0) + 1
        total_files = len(all_files)
        stats["files_found"] = total_files
        # 排序让全管线确定性：内容去重的 owner（首个副本）、worker 分片、
        # 日志顺序都不再随文件系统 inode 顺序波动。
        all_files.sort()
        ext_summary = ", ".join(f"{ext}({cnt})" for ext, cnt in sorted(ext_counts.items()))
        _log("  扫描完成: 共发现 %d 个文件 [%s]", total_files, ext_summary if ext_summary else "无")
        if progress_callback is not None:
            try:
                progress_callback(-1, 0, f"扫描完成: {total_files} 个文件")
            except Exception:
                pass
        if total_files == 0:
            # Even with no files, clean up manifest entries for deleted sources
            # and save the (now empty) manifest so the deletion persists.
            new_manifest = _load_manifest(self._rag_dir)
            removed = self._cleanup_deleted_files(all_files, stats, stats_lock,
                                                  old_manifest=dict(new_manifest),
                                                  new_manifest=new_manifest)
            try:
                self._dedup_manifest_by_content(new_manifest)
            except Exception as e:
                logger.warning("RAG: content dedup failed: %s", e)
            try:
                _save_manifest(self._rag_dir, new_manifest, removed=removed)
            except OSError as e:
                logger.warning("RAG: failed to save manifest: %s", e)
            return stats
        if self._is_cancelled():
            stats["cancelled"] = True
            return stats

        # Step 2: load manifest and partition
        _log("━━━ Step 2/6: 增量对比（Manifest） ━━━")
        manifest = _load_manifest(self._rag_dir)
        if manifest:
            _log("  已加载同步记录: %d 个文件有历史记录", len(manifest))
        else:
            _log("  无历史同步记录，将全量处理")
        new_manifest: Dict[str, Dict[str, Any]] = {}
        files_to_process: List[str] = []  # new/modified files needing full pipeline
        skipped_detail: List[str] = []
        new_detail: List[str] = []
        dup_detail: List[str] = []
        # 内容去重（B 方案，2026-10-22 用户定版）：采样指纹 → owner 路径。
        # 同一本书放在不同子目录/改名，只解析入库 owner，其余记 duplicate_of
        # 跳过——省一遍解析+切片+嵌入，也避免同段内容在检索里双倍召回。
        seen_fp: Dict[str, str] = {}
        # 预扫描已算出的采样指纹，消费端直接复用（大文件每次 192KB 读盘，
        # 同一文件算两遍纯属浪费）。
        fp_by_path: Dict[str, str] = {}
        # 预播种：**已入库**文件的指纹先进 seen 表。新副本（新建文件夹拷贝
        # 进来的）遇到它们时把 owner 让给「已在库」的文件——复用现成向量，
        # 而不是按路径排序把新副本当成 owner 白解析一遍（2026-10-22 用户
        # 实测：拷贝进新文件夹的文件仍被重新解析，根因即遍历顺序不确定）。
        # 只播种磁盘仍存在、且**自己有向量**的条目：
        # - 已删除文件的指纹不能让新副本让位；
        # - 被标记为副本（duplicate_of）的条目向量已被清理，让它当 owner 会
        #   使同内容的新文件被判成它的副本 → 永不解析、永不入库（内容丢失）。
        for _p, _e in manifest.items():
            _h = _e.get("source_hash")
            if (_h and not _e.get("duplicate_of")
                    and _e.get("chunk_count") and os.path.exists(_p)):
                seen_fp.setdefault(_h, _p)
        for filepath in all_files:
            sig = _file_signature(filepath)
            if sig is None:
                # 扫描后文件消失/不可读：跳过并记账，别让整轮同步中断
                # （旧实现裸 os.stat，一个文件就让 manifest 不写、清理不跑）。
                msg = f"  ⚠ 跳过(无法读取): {os.path.basename(filepath)}"
                _log(msg)
                with stats_lock:
                    stats["errors"].append(msg)
                continue
            prev = manifest.get(filepath)
            # 僵尸副本提升：记录里的主文件已不在（删除/移走）。分两种：
            # - 曾真实入库（有 content_hash，向量还在）：只摘 duplicate_of，
            #   条目保留走「未修改」跳过，内容继续可检索；
            # - 从没入过库（预扫描标记的副本，content_hash 为空）：必须整条
            #   丢掉历史记录让它重新走完整管线——只摘标记的话，mtime/size/
            #   指纹未变，下面就被当「未修改」跳过，内容永远回不到知识库。
            dup_owner = (prev or {}).get("duplicate_of")
            if dup_owner and not os.path.exists(dup_owner):
                # 判据必须是 **向量是否还在**（chunk_count），而不是有没有
                # content_hash：去重清理会把副本的向量删掉但保留 content_hash，
                # 这时只摘标记 → 下面按 mtime/size/指纹判「未修改」跳过，
                # 而向量早已被删且没有路径会重新入库 → 内容永久检索不到。
                if (prev or {}).get("chunk_count"):
                    prev = dict(prev)
                    prev.pop("duplicate_of", None)
                    manifest[filepath] = prev
                else:
                    manifest.pop(filepath, None)
                    prev = None
            prev_hash = (prev or {}).get("source_hash")
            # mtime+size 未变时还要再看一眼内容指纹：只看 mtime+size 会
            # **永久**漏同步 —— `cp -p` / `rsync -t` / 解压保留 mtime /
            # git checkout / 云盘回滚都会造成「内容换了但 mtime+size 不变」；
            # HFS+/SMB 挂载的 mtime 粒度只有 1~2 秒，同一秒内改一个字也识别
            # 不出来。指纹只在「本来要跳过」时才读盘算，正常路径零额外开销。
            # 旧记录没有 source_hash 时按旧逻辑判定（不影响已同步的库）。
            if (not force
                    and prev is not None
                    and prev.get("mtime") == sig["mtime"]
                    and prev.get("size") == sig["size"]
                    and (not prev_hash
                         or prev_hash == _source_fingerprint(filepath, sig["size"]))):
                # Unchanged: carry over manifest entry, skip entirely.
                new_manifest[filepath] = prev
                with stats_lock:
                    stats["files_skipped"] += 1
                skipped_detail.append(f"  ✓ 跳过(未修改): {os.path.basename(filepath)}")
                # 去重种子：已入库文件的指纹也要进 seen 表——否则本同步里
                # 新出现的同内容副本检测不到「库里已经有一个」。
                fp = prev_hash or _source_fingerprint(filepath, sig["size"])
                if fp:
                    seen_fp.setdefault(fp, filepath)
                if progress_callback is not None:
                    try:
                        progress_callback(stats["files_extracted"] + stats["files_skipped"],
                                          total_files, os.path.basename(filepath))
                    except Exception as e:
                        logger.warning("RAG: progress_callback error: %s", e)
            else:
                # 待处理前先做内容去重：指纹 ≤4MB 全量 / 大文件头中尾三段
                # 64KB（192KB 读盘）——远小于解析成本，与跳过路径同源。
                fp = _source_fingerprint(filepath, sig["size"])
                owner = seen_fp.get(fp) if fp else None
                if owner is not None and owner != filepath:
                    # 同一内容的另一个副本（不同目录/改名）：不解析、不切片、
                    # 不入库，只记一条 duplicate_of；下次同步继续走跳过路径。
                    new_manifest[filepath] = {
                        **sig,
                        "source_hash": fp,
                        "duplicate_of": owner,
                        # 带上旧条目的 content_hash/chunk_count：这个副本
                        # 此前若真入过库（force 场景），收尾清理要靠它们
                        # 认出「还有旧向量要删」；从没入库过的副本两项为空。
                        "content_hash": (prev or {}).get("content_hash"),
                        "chunk_count": (prev or {}).get("chunk_count", 0),
                    }
                    with stats_lock:
                        stats["files_duplicated"] += 1
                    dup_detail.append(
                        f"  ⧗ 跳过(内容重复): {os.path.basename(filepath)}"
                        f" ≡ {os.path.basename(owner)}")
                    continue
                files_to_process.append(filepath)
                if fp:
                    seen_fp.setdefault(fp, filepath)
                    fp_by_path[filepath] = fp
                reason = "新增" if prev is None else "已修改"
                new_detail.append(f"  → 待处理({reason}): {os.path.basename(filepath)}")
        # 汇总输出
        if skipped_detail:
            for line in skipped_detail:
                _log(line)
        if new_detail:
            for line in new_detail:
                _log(line)
        if dup_detail:
            for line in dup_detail:
                _log(line)
        _log(
            "  对比结果: 总计%d 文件, %d 个待处理, %d 个跳过(未修改), %d 个内容重复",
            total_files, len(files_to_process), stats["files_skipped"],
            stats["files_duplicated"],
        )
        if progress_callback is not None:
            try:
                progress_callback(-1, 0,
                    f"增量对比: {len(files_to_process)}待处理, {stats['files_skipped']}跳过")
            except Exception:
                pass

        # Step 2b: 初始化向量库
        if progress_callback is not None:
            try:
                progress_callback(-1, 0, "初始化向量库...")
            except Exception:
                pass
        store = self._get_store()
        if progress_callback is not None:
            try:
                progress_callback(-1, 0, "向量库就绪")
            except Exception:
                pass

        # Step 3: worker 并行解析+chunk，主线程消费并增量写入
        # 只有 files_to_process 才进入 worker 池——跳过的文件不解析不 chunk
        # 本次实际写入的切片数（取消/失败分支要用它，而不是全库总数）。
        # 必须在 if 之前初始化：全跳过时下面的取消分支也要读它。
        added = 0
        if files_to_process:
            # 实际 worker 数 = min(配置并发, 文件数)——文件少时只启必要的
            # 线程（旧实现头部照写 self._parallel_workers，1 个文件也显示
            # 「2 个并行 Worker」，与实际启动的线程数不符）。
            _num_workers = min(self._parallel_workers, len(files_to_process))
            _log("━━━ Step 3/6: 解析文档 + 清洗 + 切片（%d 个文件，%d 个并行 Worker） ━━━",
                        len(files_to_process), _num_workers)
            q: "queue.Queue[Any]" = queue.Queue(maxsize=self._parallel_workers * _QUEUE_CAPACITY_FACTOR)

            # 全局文件序号计数器（用于日志显示 "文件 3/10"）
            _file_counter = [0]  # 用 list 包装以在闭包中修改
            _file_counter_lock = threading.Lock()

            # 辅助函数：触发 UI 日志刷新。
            # progress_callback(done=-1, ...) 是"纯日志刷新"信号，
            # _on_sync_progress 收到 done<0 时只消费 _log_buffer 不更新进度条。
            def _flush_logs(hint: str = "") -> None:
                if progress_callback is not None:
                    try:
                        progress_callback(-1, 0, hint)
                    except Exception:
                        pass

            def _worker(filepaths: List[str]) -> None:
                """单个 worker：处理分配给它的文件列表，把 chunks 推入队列。"""
                worker_id = threading.get_ident()
                _log("  [Worker-%d] 启动，负责 %d 个文件", worker_id % 1000, len(filepaths))
                try:
                    for filepath in filepaths:
                        if self._is_cancelled():
                            _log("  [Worker-%d] 收到取消信号，退出", worker_id % 1000)
                            return
                        # 获取全局文件序号
                        with _file_counter_lock:
                            _file_counter[0] += 1
                            file_no = _file_counter[0]
                        fname = os.path.basename(filepath)
                        t0 = time.monotonic()
                        _log("  [%d/%d] 开始处理: %s", file_no, len(files_to_process), fname)
                        _flush_logs(fname)

                        # ---- 电子书快速路径（epub/mobi/azw3/azw）----
                        # pandoc 全书转 Markdown + ebooklib 目录缩略版 + 按章节
                        # 直接切块（跳过通用切块器），切片直接进 embed/入库流程。
                        # 解析失败时落回下方通用解析路径。
                        if os.path.splitext(filepath)[1].lower() in _EBOOK_FORMAT_EXTENSIONS:
                            book = None
                            try:
                                from .ebook_parser import extract_ebook
                                book = extract_ebook(filepath)
                            except Exception as e:
                                logger.warning("    └─ ⚠ 电子书解析异常，回退通用路径: %s: %s", fname, e)
                            if book is not None:
                                try:
                                    rel_path = os.path.relpath(filepath, self._knowledge_base)
                                    md_name = _safe_filename(rel_path) + ".md"
                                    md_path = os.path.join(self._markdown_dir, md_name)
                                    cleaned = book["full_markdown"]
                                    try:
                                        with open(md_path, "w", encoding="utf-8") as f:
                                            f.write(cleaned)
                                    except OSError as e:
                                        logger.warning("    ├─ Markdown 缓存写入失败: %s: %s", fname, e)
                                    _write_md_meta(md_path, {
                                        "doc_name": os.path.basename(filepath),
                                        "source": rel_path,
                                    })
                                    # 目录缩略版：直接采用 ebooklib 目录树
                                    try:
                                        self.build_and_store_digest(
                                            rel_path, cleaned, md_path=md_path,
                                            digest=book["toc_digest"],
                                            chapters=book["toc_chapters"],
                                        )
                                        _log("    ├─ 目录缩略版已生成 (ebooklib 目录树, %d 章)", len(book["toc_chapters"]))
                                    except Exception as e:
                                        logger.warning("    ├─ ⚠ 目录缩略版写入失败: %s: %s", fname, e)
                                    # 章节即切片；超长章节再用通用切块器二次切分
                                    # （避免单块超出 embedding 模型 token 上限）
                                    doc_chunks: List[Dict[str, str]] = []
                                    max_chars = self._chunk_size * 6
                                    for ch in book["chapters"]:
                                        if len(ch) <= max_chars:
                                            doc_chunks.append({"source": filepath, "text": ch})
                                        else:
                                            doc_chunks.extend(chunk_documents(
                                                [{"source": filepath, "text": ch}],
                                                chunk_size=self._chunk_size,
                                                chunk_overlap=self._chunk_overlap,
                                                min_chunk_size=self._min_chunk_size,
                                                overlap_percent=self._overlap_percent,
                                            ))
                                    if not doc_chunks:
                                        raise ValueError("no chapters produced")
                                    c_hash = _content_hash(cleaned)
                                    with stats_lock:
                                        stats["files_extracted"] += 1
                                        stats["total_chars"] += len(cleaned)
                                    _log(
                                        "    └─ ✅ 电子书处理完成: %s (%d 章 → %d 切片, 总耗时 %.2fs)",
                                        fname, len(book["chapters"]), len(doc_chunks), time.monotonic() - t0,
                                    )
                                    _flush_logs(f"✅ {fname} ({len(doc_chunks)}切片)")
                                    del book, cleaned
                                    if self._is_cancelled():
                                        return
                                    item = (filepath, doc_chunks, c_hash, len(doc_chunks))
                                    while not self._is_cancelled() and not _abort.is_set():
                                        try:
                                            q.put(item, timeout=2.0)
                                            break
                                        except queue.Full:
                                            continue
                                    else:
                                        return
                                    del doc_chunks, item
                                    continue
                                except Exception as e:
                                    msg = f"{os.path.basename(filepath)}: ebook clean/chunk failed: {e}"
                                    logger.error("    └─ ❌ 电子书处理失败: %s: %s", fname, e, exc_info=True)
                                    with stats_lock:
                                        stats["errors"].append(msg)
                                    continue
                            # book is None → 落回通用解析路径

                        try:
                            # ---- 1. 解析文件 ----
                            # 优先用 docling 统一解析（直接输出 Markdown），
                            # 不可用时回退到按格式的轻量解析器。
                            _log("    ├─ 解析文件: %s", fname)
                            _flush_logs(f"解析: {fname}")
                            text, is_markdown = _extract_with_docling(filepath)
                            t_extract = time.monotonic() - t0
                            _log("    ├─ 解析完成: %s (提取 %d 字符, 耗时 %.2fs%s)",
                                 fname, len(text), t_extract, " [docling]" if is_markdown else "")
                            _flush_logs(f"解析完成: {fname}")
                        except Exception as e:
                            msg = f"{os.path.basename(filepath)}: {e}"
                            logger.warning("    └─ ❌ 解析失败: %s (耗时 %.2fs): %s", fname, time.monotonic() - t0, e)
                            with stats_lock:
                                stats["errors"].append(msg)
                            continue
                        if not text.strip():
                            msg = f"{os.path.basename(filepath)}: extracted text is empty"
                            logger.warning("    └─ ⚠ 提取文本为空: %s", fname)
                            with stats_lock:
                                stats["errors"].append(msg)
                            continue
                        # clean → chunk → put 包在 try 中，避免 clean_text/
                        # normalize_markdown/chunk_documents 异常导致 worker
                        # 静默退出（异常会丢失，不记录到 stats["errors"]）。
                        try:
                            rel_path = os.path.relpath(filepath, self._knowledge_base)
                            md_name = _safe_filename(rel_path) + ".md"
                            md_path = os.path.join(self._markdown_dir, md_name)
                            # 缓存是否可用。旧实现把「缓存命中」的判断写了两次：
                            # 读缓存失败的 except 分支只把 cleaned 置空，但第二处
                            # 判断仍然命中 → cleaned 保持 ""，后面既生成一份空
                            # digest 又跳过切片。这里用同一个变量，读失败即视为
                            # 缓存未命中，强制走重新清洗。
                            cache_is_fresh = (not force
                                    and os.path.exists(md_path)
                                    and os.path.getmtime(filepath) <= os.path.getmtime(md_path))
                            if cache_is_fresh:
                                # ---- 2. 使用缓存的 Markdown ----
                                try:
                                    with open(md_path, "r", encoding="utf-8") as f:
                                        cleaned = f.read()
                                    if not cleaned.strip():
                                        raise OSError("cached markdown is empty")
                                    _log("    ├─ 清洗文本: %s (使用缓存 Markdown, %d 字符)", fname, len(cleaned))
                                except OSError:
                                    cleaned = ""
                                    cache_is_fresh = False
                            if not cache_is_fresh:
                                if is_markdown:
                                    # docling/统一引擎已输出规范 Markdown，不做
                                    # clean_text 的 HTML 标签剥离（会误删 Markdown
                                    # 标题/表格语法），但注释类占位噪声必须洗掉——
                                    # docling 给每张图片输出 `<!-- image -->`，
                                    # 不洗会原样进切片、污染检索命中。
                                    cleaned = clean_docling_markdown(text)
                                    _log("    ├─ docling Markdown 就绪: %s (%d 字符)", fname, len(cleaned))
                                else:
                                    # ---- 1.5 纯文本 → Markdown（启发式章节还原）----
                                    # txt / 解析回退产出的非结构化纯文本没有标题结构；
                                    # 若直接 clean_text，硬回车合并会把「第三章 …」这类
                                    # 独立标题行吞进段落——目录 digest 与分块全部失去
                                    # 章节边界。必须在合并**之前**用启发式把标题标成
                                    # # 行（# 是结构行，join_hard_wrapped_lines 不合并）。
                                    if not has_markdown_headings(text):
                                        marked = plain_text_to_markdown(text)
                                        if marked != text:
                                            n_head = sum(
                                                1 for l in marked.splitlines()
                                                if l.startswith("#"))
                                            _log("    ├─ 纯文本章节还原: 标记 %d 个标题 → Markdown",
                                                 n_head)
                                            text = marked
                                    # ---- 2. 清洗文本 ----
                                    t_clean = time.monotonic()
                                    _log("    ├─ 清洗文本: %s (原始 %d 字符)", fname, len(text))
                                    _flush_logs(f"清洗: {fname}")
                                    cleaned = clean_text(text)
                                    t_clean_end = time.monotonic()
                                    _log("    ├─ 清洗完成: %s → %d 字符 (耗时 %.2fs)", fname, len(cleaned), t_clean_end - t_clean)
                                    # ---- 3. 转换为 Markdown ----
                                    _log("    ├─ 转换 Markdown: %s", fname)
                                    _flush_logs(f"Markdown: {fname}")
                                    cleaned = normalize_markdown(cleaned)
                                    _log("    ├─ Markdown 完成: %s (%d 字符)", fname, len(cleaned))
                                    _flush_logs(f"Markdown完成: {fname}")
                                try:
                                    with open(md_path, "w", encoding="utf-8") as f:
                                        f.write(cleaned)
                                    logger.debug("    ├─ Markdown 已缓存: %s", md_name)
                                except OSError as e:
                                    logger.warning("    ├─ Markdown 缓存写入失败: %s: %s", filepath, e)
                            # 无论走缓存还是重新清洗，都补写一次 sidecar：
                            # 「缓存文件名 → 原始文档名」的映射无法从文件名反推
                            # （_safe_filename 会删空格、截断），历史缓存文件也
                            # 借此在下次同步时自动获得正确映射。
                            _write_md_meta(md_path, {
                                "doc_name": os.path.basename(filepath),
                                "source": rel_path,
                            })
                            del text

                            if not cleaned.strip():
                                logger.warning("    └─ ⚠ 清洗后文本为空: %s", fname)
                                continue

                            # ---- 构建文档缩略版本（目录结构 + 标题）----
                            # 只存目录结构（标题层级），不做逐章摘要。
                            # 放在空文本检查之后：空内容不该生成一份
                            # "# 目录\n\n文件名" 的空 digest 入库。
                            try:
                                self.build_and_store_digest(
                                    # 传相对路径而非 basename：documents 表的
                                    # doc_id 以它为输入，basename 会让不同目录
                                    # 下的同名文件互相覆盖/误删。
                                    rel_path,
                                    cleaned,
                                    md_path=md_path,
                                )
                                _log("    ├─ 文档缩略版本已生成: %s", fname)
                            except Exception as e:
                                logger.warning("    ├─ ⚠ 缩略版本生成失败: %s: %s", fname, e)

                            # ---- 4. 文档切片 ----
                            t_chunk = time.monotonic()
                            _flush_logs(f"切片: {fname}")
                            c_hash = _content_hash(cleaned)
                            cleaned_len = len(cleaned)
                            doc_chunks = chunk_documents(
                                [{"source": filepath, "text": cleaned}],
                                chunk_size=self._chunk_size,
                                chunk_overlap=self._chunk_overlap,
                                min_chunk_size=self._min_chunk_size,
                                overlap_percent=self._overlap_percent,
                                # 把引擎的 EF（可能是注入的自定义 EF）透传给
                                # 语义切片，避免它另起一份本地 ONNX 模型。
                                embedding_function=self._custom_ef,
                            )
                            t_chunk_end = time.monotonic()
                            _log(
                                "    ├─ 文档切片: %s → %d 个切片 (每片约 %d tokens, 重叠 %d tokens, 耗时 %.2fs)",
                                fname, len(doc_chunks), self._chunk_size, self._chunk_overlap,
                                t_chunk_end - t_chunk,
                            )
                            _flush_logs(f"切片完成: {fname} ({len(doc_chunks)}片)")
                            del cleaned
                        except Exception as e:
                            # clean/chunk 阶段异常：记录并跳过该文件，不中断整个同步。
                            msg = f"{os.path.basename(filepath)}: clean/chunk failed: {e}"
                            logger.error("    └─ ❌ 清洗/切片失败: %s: %s", fname, e, exc_info=True)
                            with stats_lock:
                                stats["errors"].append(msg)
                            continue

                        with stats_lock:
                            stats["files_extracted"] += 1
                            stats["total_chars"] += cleaned_len

                        # 文件处理完成总耗时
                        _log(
                            "    └─ ✅ 文件处理完成: %s (%d 切片, 总耗时 %.2fs)",
                            fname, len(doc_chunks), time.monotonic() - t0,
                        )
                        _flush_logs(f"✅ {fname} ({len(doc_chunks)}切片)")

                        if self._is_cancelled():
                            return
                        # 背压：队列满时阻塞，避免内存堆积。
                        # 使用 timeout + 循环检查取消标志，避免主线程因
                        # 取消/异常停止消费后 worker 永久阻塞在 put 上。
                        item = (filepath, doc_chunks, c_hash, len(doc_chunks))
                        while not self._is_cancelled() and not _abort.is_set():
                            try:
                                q.put(item, timeout=2.0)
                                break
                            except queue.Full:
                                continue
                        else:
                            return
                        # Release local references so memory is freed before
                        # the next file is loaded (important for large PDFs:
                        # without this, doc_chunks stays alive until next iter).
                        del doc_chunks, item
                except Exception as e:
                    # 兜底：任何未预期的异常都记录，避免 worker 静默死亡。
                    logger.error("  [Worker-%d] 异常退出: %s", worker_id % 1000, e, exc_info=True)
                    with stats_lock:
                        stats["errors"].append(f"worker: {e}")
                finally:
                    # 投递哨兵也要带超时，防止队列满且主线程已退出消费时
                    # worker 永久卡在 put。
                    _log("  [Worker-%d] 完成所有文件，发送完成信号", worker_id % 1000)
                    try:
                        q.put(_SENTINEL, timeout=5.0)
                    except queue.Full:
                        pass

            # 消费端（add_streaming）失败/中途放弃时置位，让所有 worker 立即
            # 退出 put 循环。旧实现只靠 _is_cancelled()——向量化写入异常并非
            # 用户取消，不会置位该标志，worker 会永远阻塞在 q.put 上：线程与
            # 其持有的整文档切片内存一起泄漏到进程退出。
            _abort = threading.Event()

            # 把文件列表分片给各 worker（LPT 负载均衡，见 _shard_files_balanced）
            # 动态调整并发数：文件数少于默认 worker 数时，只启动必要数量的线程
            num_workers = min(self._parallel_workers, len(files_to_process))
            file_shards = _shard_files_balanced(files_to_process, num_workers)

            threads = []
            for shard in file_shards:
                t = threading.Thread(target=_worker, args=(shard,), daemon=True)
                t.start()
                threads.append(t)

            _flush_logs(f"启动 {len(threads)} 个 Worker ({len(files_to_process)} 个文件)")

            # Step 4: 主线程消费队列，逐文件增量写入 vector store
            _log("━━━ Step 4/6: 向量化 + 存入向量库 ━━━")
            def _chunk_iter() -> Iterator[Dict[str, Any]]:
                sentinels_seen = 0
                # 必须等于实际启动的 worker 数（num_workers），而不是
                # self._parallel_workers：文件数少于默认并发数时，实际启动的
                # 线程更少，若仍按 _parallel_workers 期待哨兵，会永远等不到
                # 足够数量，只能靠 15s 空超时兜底退出（白白空等）。
                expected_sentinels = num_workers
                empty_streak = 0
                # 使用较短的 q.get 超时（0.5s），确保取消后最多 0.5s 内能响应。
                _QGET_TIMEOUT = 0.5
                # 总超时 = empty_streak * _QGET_TIMEOUT，设为 15s（30 次 × 0.5s）
                _MAX_EMPTY_STREAK = 30
                consumer_file_no = 0
                while sentinels_seen < expected_sentinels:
                    if self._is_cancelled():
                        return
                    try:
                        item = q.get(timeout=_QGET_TIMEOUT)
                        empty_streak = 0
                    except queue.Empty:
                        empty_streak += 1
                        if empty_streak >= _MAX_EMPTY_STREAK and all(not t.is_alive() for t in threads):
                            logger.warning(
                                "  向量化消费者: %d/%d 个 Worker 完成信号已收到, 所有 Worker 已退出, 退出消费",
                                sentinels_seen, expected_sentinels,
                            )
                            return
                        continue
                    if item is _SENTINEL:
                        sentinels_seen += 1
                        _log("  收到 Worker 完成信号 (%d/%d)", sentinels_seen, expected_sentinels)
                        continue
                    filepath, doc_chunks, c_hash, n_chunks = item
                    consumer_file_no += 1
                    fname = os.path.basename(filepath)
                    _log(
                        "  [消费 %d] 向量化排队: %s (%d 个切片等待嵌入)",
                        consumer_file_no, fname, n_chunks,
                    )
                    # Update manifest for this processed file.
                    # 先记到 pending，等 add_streaming 整体成功后才并入
                    # new_manifest（见下方 except/cancel 分支）：旧实现在产出
                    # chunks 的同时就写 manifest，一旦后续 embedding/写库失败
                    # 或用户中途取消，文件只入库了一部分，manifest 却已记录
                    # 完成——下次同步因 mtime/size 未变被跳过，内容永久缺失。
                    try:
                        sig_now = _file_signature(filepath)
                        if sig_now is None:
                            # 文件在扫描后被删 / 权限变化：跳过它并计入 errors。
                            # 旧实现直接 ``{**sig_now}`` 解包 None 抛 TypeError，
                            # 异常穿过 _chunk_iter → add_streaming 的 except，
                            # write_failed=True —— 剩余文件的向量全部丢弃、整个
                            # manifest 作废，一次抖动毁掉整轮同步。
                            # errors 是**列表**（其他错误路径都 append dict），这里
                            # 旧实现写成 ``stats["errors"] = stats.get("errors", 0) + 1``
                            # 会把列表覆盖成 int 且不持 stats_lock —— 同步收尾的
                            # ``len(stats["errors"])`` 随即 TypeError，整轮已完成的
                            # 同步以异常告终（UI 显示「显示结果时出错」）。
                            with stats_lock:
                                stats["errors"].append(
                                    f"{os.path.basename(filepath)}: 文件在扫描后不可读（被删除/权限变化），已跳过"
                                )
                            _log("  ⚠ 跳过（文件已不可读）: %s", fname)
                            continue
                        pending_manifest[filepath] = {
                            **sig_now,
                            # 复用预扫描算好的指纹（该文件本次未被改动的
                            # 情况下一致；算不出来才回退读盘）。
                            "source_hash": (fp_by_path.get(filepath)
                                            or _source_fingerprint(
                                                filepath, sig_now["size"])),
                            "content_hash": c_hash,
                            "chunk_count": n_chunks,
                        }
                    except OSError as e:
                        logger.warning("RAG: failed to update manifest for %s: %s", filepath, e)
                    for c in doc_chunks:
                        yield c
                    # 进度反馈：每消费完一个文件回调一次。
                    if progress_callback is not None:
                        try:
                            with stats_lock:
                                done = stats["files_extracted"] + stats["files_skipped"]
                            progress_callback(done, total_files, os.path.basename(filepath))
                        except Exception as e:
                            logger.warning("RAG: progress_callback error: %s", e)
                    # 单文件消费完毕：释放解析器引用循环与 MLX 内存池空闲
                    # 缓冲，防止长同步 RSS 单调爬升（见 release_mlx_pool 注释）。
                    release_mlx_pool()

            def _on_batch(batch_added: int, total_added: int) -> None:
                _log("  ├─ 向量化批次写入: +%d 切片 (累计 %d)", batch_added, total_added)
                # 记录「已完整写入向量库」的文件：按 pending 顺序累加各文件
                # chunk_count，一旦某文件累计 chunk 数 <= total_added，说明它的
                # 全部切片都已落库，可安全记入同步记录（取消时仅提交这些）。
                # 一旦某文件累计 > total_added（尚未完整写入），其后所有文件也
                # 必然未完整写入，直接 break（保持有序、提前结束）。
                cum = 0
                for fp, entry in pending_manifest.items():
                    cum += entry.get("chunk_count", 0)
                    if cum <= total_added:
                        if fp not in committed_manifest:
                            committed_manifest[fp] = entry
                    else:
                        break
                if progress_callback is not None:
                    try:
                        progress_callback(stats["files_extracted"] + stats["files_skipped"],
                                          total_files, f"已写入 {total_added} 切片")
                    except Exception as e:
                        logger.warning("RAG: on_batch callback error: %s", e)

            write_failed = False
            pending_manifest: Dict[str, Dict[str, Any]] = {}
            # 已「完整写入向量库」的文件（取消时只提交这些，避免半截记录）。
            # 与 pending_manifest 不同：pending 记录「已产出 chunks 的文件」，
            # committed 记录「chunks 已实际落库的文件」——取消时二者之差就是
            # 正在处理 / 只写入一半的文件，不记入同步记录，下次同步重新处理即可。
            committed_manifest: Dict[str, Dict[str, Any]] = {}
            try:
                # 批量 200（旧值 20）：LanceDB 每次 add 都会产生一个新
                # fragment，小批量在大库上留下海量碎片，检索要逐段扫描。
                _log("  开始流式向量化写入 (每批 %d 个切片)", _WRITE_BATCH_SIZE)
                added = store.add_streaming(
                    _chunk_iter(),
                    batch_size=_WRITE_BATCH_SIZE,
                    on_batch=_on_batch,
                    is_cancelled=self._is_cancelled,
                )
                stats["chunks"] = added
                stats["chunks_total"] = store.count()
                _log("  向量化写入完成: 共 %d 个切片已存入向量库", added)
            except Exception as e:
                logger.error("  向量化写入失败: %s", e, exc_info=True)
                # 注意：这里不能调用 self.cancel()——向量化写入异常（如模型下载
                # 失败、磁盘满、LanceDB 写入异常）不是用户主动取消，不应该设置
                # cancelled 标志。否则 UI 会错误地显示"同步已取消"并丢弃已
                # 解析的数据。改为将异常记录到 errors，让 ingest 正常结束。
                #
                # stats["chunks"] 必须记「本次写入量」而不是全库总数：UI 用它
                # 展示「本次新增多少切片」，写全库数会让用户以为全部重算了。
                stats["chunks"] = added
                stats["errors"].append(f"Vector store add failed: {e}")
                write_failed = True
            finally:
                # 让阻塞在 q.put 上的 worker 立即退出（成功路径是空操作）。
                _abort.set()

            # 只有整条写入链路成功完成才提交 manifest；失败/取消路径下这些
            # 文件下次同步会被重新处理（幂等），保证不会出现「manifest 记录
            # 了但向量库里只有半个文件」的永久性内容缺失。
            if write_failed:
                pending_manifest.clear()
                _log("  ⚠ 写入失败，本次处理的 %d 个文件将不在同步记录中（下次同步会重新处理）",
                     stats["files_extracted"])
            elif self._is_cancelled():
                # 取消时不丢弃已完成的工作：只提交「已完整写入向量库」的文件
                # （committed_manifest），其余（正在处理 / 仅写入一半）不记录，
                # 下次同步会被重新处理（幂等 upsert，不重复留垃圾）。这样已完成
                # 的解析/清洗/切片/入库不会被浪费，也未完成的不留「记录完成却只
                # 入库一半」的永久性内容缺失。
                new_manifest.update(committed_manifest)
                pending_manifest.clear()
                _log(
                    "  ⚠ 同步被取消: %d 个文件已完整入库并记入同步记录（下次从中断处续传）, "
                    "%d 个文件待下次续传",
                    len(committed_manifest), len(pending_manifest),
                )
            else:
                new_manifest.update(pending_manifest)

            # 等待所有 worker 线程结束
            _log("  等待所有 Worker 线程结束...")
            for i, t in enumerate(threads):
                t.join(timeout=5.0)
                if t.is_alive():
                    logger.warning("  Worker 线程 %d 未在 5s 内结束", i + 1)
            _log("  所有 Worker 线程已结束")
        else:
            # All files skipped — no new chunks written. Report current store count.
            stats["chunks"] = store.count()
            _log("  所有 %d 个文件均未修改，无需重新处理", total_files)

        cancelled = self._is_cancelled()
        if cancelled:
            stats["cancelled"] = True
            # 记本次写入量而非全库总数：UI 展示的是「这次同步了多少切片」。
            stats["chunks"] = added
            _log("⚠ 同步已被用户取消: %d/%d 文件已提取, %d 切片已存储",
                        stats["files_extracted"], stats["files_found"], stats["chunks"])

        # Step 5: 清理已删除文件
        _log("━━━ Step 5/6: 清理已删除文件 ━━━")
        removed_sources = self._cleanup_deleted_files(
            all_files, stats, stats_lock, manifest, new_manifest)
        if stats["files_deleted"] > 0:
            _log("  已清理 %d 个已删除文件的旧向量", stats["files_deleted"])
        else:
            _log("  无已删除文件需要清理")

        # Step 6: 保存 manifest
        _log("━━━ Step 6/6: 保存同步记录 ━━━")
        # 内容去重收尾（B 方案后半程）：清理存量重复副本的向量 + 标记
        # duplicate_of，避免检索时同段内容双倍召回。
        try:
            deduped = self._dedup_manifest_by_content(new_manifest)
            if deduped:
                _log("  ⧗ 内容去重: 清理 %d 个重复副本的向量（content_hash 相同）",
                     deduped)
        except Exception as e:
            logger.warning("RAG: content dedup failed: %s", e)
        try:
            # 读改写合并 + 跨进程锁：并存的另一个 RAGEngine（agent 侧自动
            # ingest）刚写入的条目不能被这份旧快照抹掉。
            _save_manifest(self._rag_dir, new_manifest, removed=removed_sources)
            _log("  同步记录已保存: %d 个文件条目 → %s",
                        len(new_manifest), _manifest_path(self._rag_dir))
        except OSError as e:
            logger.warning("  同步记录保存失败: %s", e)

        # Step 7: 补做/重刷文档目录结构（缩略版本）。
        # 这一步不依赖「文件是否修改」——即使所有文件都被增量跳过（未修改），
        # 也要扫描 markdown 缓存，为缺记录或旧格式（含废弃的「章节摘要」段、
        # 空目录）的文档重刷为纯目录结构，确保 rag_outline 始终返回正确结果。
        # 用户已取消时跳过：该步会全量重读/重建所有缓存 digest，同步都取消了
        # 还多跑几秒（Step5 清理保留——它便宜且保证不残留已删文件的向量）。
        if not cancelled:
            try:
                refreshed = self.ensure_digests_for_cached_documents(refresh_stale=True)
                if refreshed:
                    _log("  目录结构（缩略版本）补做/重刷: %d 篇", refreshed)
            except Exception as e:
                logger.warning("  目录结构补做/重刷失败: %s", e)

        if cancelled:
            t_total = time.monotonic() - t_start
            _log("══════════════════════════════════════════")
            _log("⏹ 知识库同步已取消 (耗时 %.1fs): 扫描 %d 文件, 提取 %d, 跳过 %d, 切片 %d",
                t_total, stats["files_found"], stats["files_extracted"],
                stats["files_skipped"], stats["chunks"],
            )
            _log("══════════════════════════════════════════")
            return stats

        t_total = time.monotonic() - t_start
        _log("══════════════════════════════════════════")
        _log("✅ 知识库同步完成 (耗时 %.1fs)", t_total)
        _log("   扫描文件: %d | 新增/更新: %d | 跳过(未修改): %d | 清理删除: %d",
            stats["files_found"], stats["files_extracted"],
            stats["files_skipped"], stats["files_deleted"],
        )
        _log("   总字符数: %d | 向量切片: %d | 错误: %d",
            stats["total_chars"], stats["chunks"], len(stats.get("errors", [])),
        )
        if stats.get("errors"):
            _log("   错误详情:")
            for err in stats["errors"][:10]:
                _log("     - %s", err)
            if len(stats["errors"]) > 10:
                _log("     ... (还有 %d 个错误)", len(stats["errors"]) - 10)
        _log("══════════════════════════════════════════")
        return stats

    def _cleanup_deleted_files(
        self,
        current_files: List[str],
        stats: Dict[str, Any],
        stats_lock: threading.Lock,
        old_manifest: Optional[Dict[str, Dict[str, Any]]] = None,
        new_manifest: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> set:
        """Delete vectors and manifest entries for files removed from the KB.

        Runs on every ingest (force or not): if a file was in the manifest
        but is no longer on disk, its vectors are stale and must be removed.

        Returns 本次有意删除的 source 集合——``_save_manifest`` 需要它来区分
        「我删掉的条目」（必须删）与「别的实例并发写入的条目」（必须保留）。
        """
        if old_manifest is None:
            old_manifest = _load_manifest(self._rag_dir)
        if new_manifest is None:
            new_manifest = dict(old_manifest)  # caller will update
        current_set = set(current_files)
        removed = [fp for fp in old_manifest if fp not in current_set]
        if not removed:
            return set()
        logger.info("  发现 %d 个已删除文件，清理对应向量与目录概要...", len(removed))
        store = self._get_store()
        # 只把**真正删成功**的文件交给调用方从 manifest 移除：返回值会喂给
        # _save_manifest(removed=…)，旧实现直接返回全部 removed —— 失败文件
        # 的条目连同「下次同步重试」的机会一起被 pop 掉，而 _cleanup_deleted_files
        # 只遍历 manifest 键 → 孤儿向量永久残留且仍可被检索。
        deleted_ok: set = set()
        for source in removed:
            # 单个文件的删除失败不能中断整个清理：此时向量写入已完成、manifest
            # 尚未保存（Step 6 在其后），一次 LanceDB 异常带着未保存的 manifest
            # 崩溃退出会让增量同步状态不一致。逐文件隔离；失败的文件**保留在
            # manifest 里**，下次同步重试——旧实现无条件 pop，「下次重试」的
            # 承诺永远不成立（条目已不在 manifest，_cleanup_deleted_files 只
            # 遍历 manifest 键），失败文件的向量会永久残留且仍可被检索。
            try:
                store.delete_by_source(source)
                # 同步清理 documents 元数据表里的目录概要（digest）/markdown 记录。
                # 三个名字都删：doc_name 存的是相对路径（子目录文件），其
                # doc_id 是 md 缓存路径的哈希——只按 basename/source 找永远
                # 找不到子目录文件的行（digest 孤儿行永久累积，rag_outline
                # 继续列出已删文档）。rel_path 按 doc_name 精确命中。
                store.delete_document(os.path.relpath(source, self._knowledge_base))
                # 两个旧版兼容：basename 哈希（最早版本）与完整路径哈希。
                store.delete_document(os.path.basename(source))
                store.delete_document(source)
                # 缓存的 Markdown 也必须一起删。旧实现只删了向量和元数据，
                # 而紧接着的 Step 7（ensure_digests_for_cached_documents）会发现
                # 「有缓存文件、无 digest 记录」，于是用旧缓存重新生成一条 ——
                # 表现是日志打了「🗑 已清理」，但 rag_outline 仍列出该文档，
                # 且 rag/documents/ 下的 .md 永久堆积。
                md_path = self._cache_md_path(source)
                store.delete_document_by_md_path(md_path)
                try:
                    if os.path.isfile(md_path):
                        os.remove(md_path)
                    # sidecar 一并清掉：留着会在同名新文件复用时读到旧映射
                    sidecar = _md_meta_path(md_path)
                    if os.path.isfile(sidecar):
                        os.remove(sidecar)
                except OSError as e:
                    logger.warning("  ⚠ 删除 Markdown 缓存失败 %s: %s", md_path, e)
                new_manifest.pop(source, None)
                deleted_ok.add(source)
                with stats_lock:
                    stats["files_deleted"] += 1
                logger.info("  🗑 已清理: %s", os.path.basename(source))
            except Exception as e:
                logger.warning(
                    "  ⚠ 清理已删除文件失败 %s: %s（保留 manifest 条目，下次同步重试）",
                    source, e,
                )
        return deleted_ok

    # ------------------------------------------------------------------
    # retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 3,
        source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search the knowledge base with hybrid retrieval + reranking.

        Retrieves top_k * 4 candidates each from vector-similarity search and
        BM25 keyword search, merges/dedupes, then uses FastEmbed reranker to pick
        the most relevant results.  Results below the 0.7 similarity threshold
        are dropped before returning.

        ``source`` 为可选文档名/关键词过滤：定向检索某本书/文档内的切片
        （配合 rag_outline 实现 Targeted RAG——先掌握全局结构，再按需检索
        特定文档的细节）。
        """
        t0 = time.monotonic()
        store = self._get_store()
        store_count = store.count()
        if store_count == 0:
            logger.info("🔍 知识库检索: 向量库为空，无结果返回 (查询=%r)", query[:60])
            return []
        # 构造 SQL 过滤条件：source 字段存完整路径，用 LIKE 匹配文档名。
        where = None
        if source:
            where = f"source LIKE '%{_escape_like(source)}%' ESCAPE '\\'"
            logger.info("  ├─ 定向检索: 限定来源包含 %r", source)
        logger.info(
            "🔍 知识库检索开始: 查询=%r 目标结果数=%d 向量库总量=%d",
            query[:80], top_k, store_count,
        )
        logger.info("  ├─ 混合初检: 向量相似度 + BM25 关键词, 各取 top %d 候选", top_k * 4)
        if self._rerank_enabled:
            results = store.search_with_rerank(query, top_k=top_k, where=where)
        else:
            logger.info("  ├─ 重排序已在配置中关闭 (rag_rerank_enabled=false)，"
                        "直接使用混合检索排序")
            results = store.search(query, top_k=top_k, where=where)
        t_total = time.monotonic() - t0
        if results:
            logger.info("  └─ ✅ 检索完成: 返回 %d 条结果 (耗时 %.2fs)", len(results), t_total)
            for i, r in enumerate(results, 1):
                # 循环变量不能叫 source：会覆盖函数形参 source（定向检索
                # 过滤条件），后续任何对它的引用都会拿到最后一条结果的文件名。
                src_name = os.path.basename(r.get("source") or "")
                score = r.get("score") or 0
                text_preview = (r.get("text") or "")[:80].replace("\n", " ")
                logger.info("      [%d] %s (相关度: %.4f) | %s...", i, src_name, score, text_preview)
        else:
            logger.info("  └─ ⚠ 检索完成: 无匹配结果 (耗时 %.2fs)", t_total)
        return results

    def search_formatted(
        self,
        query: str,
        top_k: int = 3,
        source: Optional[str] = None,
        max_chars_per_result: int = 1500,
        max_total_chars: int = 8000,
    ) -> str:
        """Search and return formatted results as a string.

        Each result's text is truncated to ``max_chars_per_result`` and the
        total output is capped at ``max_total_chars`` to avoid blowing up the
        LLM context window with oversized chunks (some documents produce very
        long chunks, which previously caused ``exceed_context_size_error``).

        ``source`` 为可选文档名过滤（定向检索某本书）。
        """
        results = self.search(query, top_k, source=source)
        if not results:
            return (
                "（知识库中未找到与该查询相关的内容。"
                "若是想了解知识库整体有哪些资料，请改用 rag_outline（不带参数）"
                "列出全部文档；若是检索词太偏，可换用资料中可能出现的措辞重试。）"
            )

        lines = [f"从知识库检索到 {len(results)} 条相关内容（已重排序）：\n"]
        total = 0
        for i, r in enumerate(results, 1):
            # 回退结果（向量排序）可能缺键，直接下标会 KeyError 打断整个工具
            # 调用；用 .get() 兜底。
            src_name = os.path.basename(r.get("source") or "")
            header = f"### [{i}] 来源: {src_name} (相关度: {(r.get('score') or 0):.4f})"
            text = r.get("text") or ""
            if len(text) > max_chars_per_result:
                text = text[:max_chars_per_result] + "…"
            lines.append(header)
            total += len(header) + 1
            lines.append(text)
            total += len(text)
            lines.append("")
            total += 1
            if total >= max_total_chars:
                lines.append("…（结果过长，已截断）")
                break
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Meta-context + Targeted RAG (文档缩略版本)
    # ------------------------------------------------------------------

    def _cache_md_path(self, source: str) -> str:
        """源文件 → 它对应的 Markdown 缓存文件路径。

        与 ingest 里的命名规则保持一致（``_safe_filename(rel) + ".md"``），
        删除清理时才能定位到该删哪个缓存（此时源文件已不存在，只能算出来）。
        """
        try:
            rel = os.path.relpath(source, self._knowledge_base)
        except ValueError:      # 不同盘符（Windows）
            rel = os.path.basename(source)
        return os.path.join(self._markdown_dir, _safe_filename(rel) + ".md")

    def build_and_store_digest(
        self,
        doc_name: str,
        markdown: str,
        md_path: str = "",
        digest: Optional[str] = None,
        chapters: Optional[list] = None,
    ) -> str:
        """Build a document's digest (缩略版本) and store it in the metadata table.

        The digest is the document's 目录结构（标题层级），不含逐章摘要——避免
        对每章调用 LLM 带来的慢速与空结果问题。标题层级已足以作为 Targeted RAG
        的全局结构元信息。

        ``markdown`` 只用于**构建**目录结构，不会写入元数据表（全文已缓存在
        ``rag/documents/*.md``，表内再存一份是第三份冗余且会被全量载入内存）。

        ``digest`` / ``chapters`` 允许调用方传入已计算好的结果（批量重刷场景
        里 build_digest 的标题提取很昂贵，避免算两遍）。
        """
        if chapters is None:
            digest, chapters = build_digest(markdown)
        # 兜底：纯文本解析结果（如 PDF OCR）通常没有 Markdown 标题，提取不到
        # 任何章节。此时用文档名作为最小缩略版本，避免 rag_outline 返回空目录。
        if not chapters:
            digest = f"# 目录\n\n{os.path.basename(doc_name)}"
        # Serialize chapters (title + level) for targeted retrieval.
        chapters_json = json.dumps(
            [{"level": c.level, "title": c.title}
             for c in chapters],
            ensure_ascii=False,
        )
        store = self._get_store()
        # 存完整 doc_name（通常是相对路径）：basename 会让不同目录下的同名
        # 文件在 documents 表里共用 doc_id 互相覆盖。doc_name 列同时承担
        # rag_outline 的展示与按名检索（含 basename/子串匹配），存相对路径
        # 不影响这两种用法。
        store.upsert_document(
            doc_name=doc_name,
            digest=digest,
            chapters=chapters_json,
            md_path=md_path,
        )
        return digest

    def get_document_outline(self, doc_name: str) -> Optional[Dict[str, Any]]:
        """Return a document's stored digest + chapter list by (partial) name.

        The full markdown is no longer persisted in the metadata table; use
        the returned ``md_path`` (or the on-disk ingest cache) to read it.
        """
        store = self._get_store()
        return store.get_document_digest(doc_name)

    def list_documents(self) -> List[str]:
        """Return stored document names that have a digest."""
        store = self._get_store()
        return store.list_documents()

    def ensure_digests_for_cached_documents(self, refresh_stale: bool = False) -> int:
        """为已缓存 Markdown 的文档补做/重刷 digest（目录结构）。

        扫描 ``markdown_dir`` 缓存目录，对每个 .md 缓存文件：
          * documents 表中缺记录的 → 补做 digest（关键场景：功能上线前已同步
            的文档，有 markdown 缓存 + 向量，但无 metadata 记录）；
          * ``refresh_stale=True`` 时，对已存在但 digest 为旧格式（含「章节摘要」
            段）或目录为空的记录 → 用当前目录结构重刷。

        旧格式判定：历史版本曾生成「# 章节摘要」段（逐章 LLM/提取式摘要），
        该功能已废弃，残留数据需重刷为纯目录结构。

        Returns 本次补做/重刷的文档数量。
        """
        if not os.path.isdir(self._markdown_dir):
            return 0

        store = self._get_store()
        # 一次性快照 documents 表，在内存里建索引。
        # 旧实现逐个文件调 get_document_by_md_path/get_document_digest ——
        # 每次 build_and_store_digest 都会让 _documents_cache 失效，于是下一轮
        # 迭代重新从磁盘全表加载：N 篇文档 = N 次全表物化 + O(N²) 次线性扫描
        # （每行还要 abspath）。这里读一次，之后全部走内存 dict。
        rows = store.all_document_rows()
        by_md_path: Dict[str, Dict[str, Any]] = {}
        by_doc_name: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            mp = r.get("md_path") or ""
            if mp:
                by_md_path[os.path.abspath(mp)] = r
            dn = r.get("doc_name") or ""
            if dn:
                by_doc_name.setdefault(dn, r)

        created = 0
        for fn in sorted(os.listdir(self._markdown_dir)):
            if not fn.endswith(".md"):
                continue
            md_path = os.path.join(self._markdown_dir, fn)
            try:
                # doc_name 的解析优先级：
                #   1) sidecar（写缓存时落下的原始 source 相对路径，最权威）
                #   2) documents 表按 md_path 反查（表里的 md_path 也是原值）
                #   3) 从缓存文件名反推（**不可靠**：_safe_filename 会删掉空格、
                #      并把名字截断到 32 字符，"Report 2024.pdf" → "Report2024.pdf"）
                # 只用第 3 种会导致每次同步都查不到已有记录 → 全文重读 + 重建
                # digest，并用错名字再写一条 → documents 表出现永久重复条目。
                meta = _read_md_meta(md_path)
                doc_name = ""
                if meta:
                    doc_name = (meta.get("source") or meta.get("doc_name") or "").strip()
                if not doc_name:
                    # 表记录按 md_path 精确匹配：缓存文件名经过清洗（空格/括号/
                    # 中文标点都被丢掉），从文件名反推 doc_name 必然失真，会把
                    # "Chapter 1.pdf" 认成 "Chapter1.pdf"，于是每轮同步都判定
                    # 缺记录并写入一条重复行。md_path 是入库时原样记录的。
                    existing = by_md_path.get(os.path.abspath(md_path))
                    doc_name = existing["doc_name"] if existing else ""
                if not doc_name:
                    # 兜底：老缓存没有 md_path 记录时才退回按文件名猜。
                    # 缓存文件名格式：{safe_base}_{md5[:12]}.md → 还原 doc_name。
                    stem = fn[:-3]  # 去掉 .md
                    doc_name = stem
                    if len(stem) > 13 and stem[-13] == "_":
                        suffix = stem[-12:]
                        if all(c in "0123456789abcdef" for c in suffix):
                            doc_name = stem[:-13]
                # 统一用内存快照判定「记录是否存在」，不再逐文件查表。
                existing = by_md_path.get(os.path.abspath(md_path))
                if existing is None:
                    # 廉价预检：先查记录再决定要不要读全文。绝大多数缓存文件的
                    # digest 记录已是最新——旧实现无条件全文读入 + build_digest，
                    # 大知识库下每次同步都白跑一遍全文 IO 和解析。
                    existing = by_doc_name.get(doc_name)
                if existing is not None:
                    if not refresh_stale:
                        continue
                    if not _is_stale_digest(existing.get("digest", "")):
                        continue
                with open(md_path, "r", encoding="utf-8") as f:
                    markdown = f.read()
                if not markdown.strip():
                    continue
                digest, chapters = build_digest(markdown)
                # 记录存在但格式合法时只有内容真的变了才需要重写。
                if existing is not None and digest == existing.get("digest", ""):
                    continue
                # 传入已计算的 digest/chapters，避免 build_and_store_digest 内部
                # 再跑一遍昂贵的标题提取。md_path 必须传：upsert 的 doc_id 以它
                # 为输入——漏传会落库 hash(doc_name) 的重复行，与已有记录
                # （hash(md_path)）并存，get_document_digest 精确匹配永远命中
                # 旧行，重刷结果对用户不可见且 by_md_path 反查永久失效。
                self.build_and_store_digest(
                    doc_name, markdown,
                    digest=digest, chapters=chapters,
                    md_path=md_path,
                )
                created += 1
                logger.info("  ├─ 补做/重刷缩略版本: %s (%d 字符)", doc_name, len(markdown))
            except Exception as e:
                logger.warning("  ⚠ 补做缩略版本失败 %s: %s", fn, e)
        return created

    # ------------------------------------------------------------------
    # management
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Return current status of the knowledge base."""
        store = self._get_store()
        chunk_count = store.count()
        sources = store.list_sources()
        logger.info("📊 知识库状态查询: 切片数=%d 来源文件数=%d", chunk_count, len(sources))
        return {
            "knowledge_base": self._knowledge_base,
            "rag_dir": self._rag_dir,
            "chunks_stored": chunk_count,
            "sources": sources,
            "has_knowledge_base": bool(self._knowledge_base and os.path.isdir(self._knowledge_base)),
        }

    def clear(self) -> None:
        """Clear all stored vectors, cached markdown files, and manifest."""
        logger.info("🗑 正在清空知识库...")
        store = self._get_store()
        store.clear()
        # Also clear cached markdown
        if os.path.isdir(self._markdown_dir):
            import shutil
            shutil.rmtree(self._markdown_dir, ignore_errors=True)
            os.makedirs(self._markdown_dir, exist_ok=True)
            logger.info("  已清除缓存 Markdown 文件")
        # Clear the incremental-sync manifest so the next ingest processes
        # everything from scratch (no stale "already processed" entries).
        # 必须持 manifest 锁：并发的另一个实例可能已进入 _save_manifest 的
        # 读改写（它 merged 里是旧条目），我们删完后它 os.replace 一写，
        # manifest **立刻被复活** → 向量库空、记录满，此后所有文件都被判为
        # 「已同步」而跳过，检索永远返回空且不会自愈。
        manifest = _manifest_path(self._rag_dir)
        with _ManifestLock(self._rag_dir):
            if os.path.exists(manifest):
                try:
                    os.remove(manifest)
                    logger.info("  已清除同步记录文件")
                except OSError as e:
                    logger.warning("  同步记录文件删除失败: %s", e)
        logger.info("  知识库已清空")

    def close(self) -> None:
        """Release the underlying VectorStore's resources.

        Drops the cached VectorStore (LanceDB connection, FastEmbed ONNX model,
        FastEmbed reranker). After close(), a new VectorStore is lazily created on
        the next operation. Call this when the RAGEngine is no longer needed
        (e.g. after SyncKnowledgeWorker finishes) to avoid leaking ~600MB of
        C++ heap memory per instance.
        """
        # 先请求取消并等待 ingest 的 worker 退出：worker 的 join(timeout=5) 超时
        # 后 ingest 仍会返回，调用方（SyncKnowledgeWorker）紧接着就 close()。
        # 不先撤 worker 的话，它们随后会对着已关闭的 LanceDB/ONNX 会话报错，
        # 并把大文档的切片内存继续占着（线程是 daemon，不会阻止进程退出）。
        if self._ingest_lock.acquire(blocking=False):
            try:
                self._cancelled.set()
            finally:
                self._ingest_lock.release()
        self._cancelled.set()
        if self._store is not None:
            with self._store_lock:
                if self._store is not None:
                    try:
                        self._store.close()
                    except Exception as e:
                        logger.debug("RAG: VectorStore close error: %s", e)
                    self._store = None
        self.release_parsers()

    def release_parsers(self) -> None:
        """释放同步期占用的重量级解析单例（OCR / docling / marker / ovis）。

        UI 同步路径会调 ``close()``，但 **agent 的 rag_ingest 工具路径不会**
        （引擎被 ToolRegistry 长期持有，只在 shutdown 时 close）——一次
        rag_ingest 后 docling 池（~4.6GB）+ OCR（~0.5GB）+ ovis（~0.9GB）
        会常驻到会话结束。所以 ingest 结束时**无条件**调用本方法（见
        ``ingest()`` 的 finally），下次需要时惰性重建。
        """
        # Release the global OCR singleton (~500MB) as well.  It is only needed
        # during ingestion of image-based PDFs; after a sync finishes the
        # engine can reclaim it.  Lazy re-created on the next OCR-requiring run.
        try:
            from .parsers import release_ocr_engine
            release_ocr_engine()
        except Exception as e:
            logger.debug("RAG: release_ocr_engine error: %s", e)
        # 同样释放 docling 的共享转换器池（重量级模型常驻）。
        try:
            from .docling_parser import release_converter
            release_converter()
        except Exception as e:
            logger.debug("RAG: release_converter error: %s", e)
        # marker（rag_pdf_engine=marker 时启用）共享 PdfConverter 同步释放；
        # 未装 marker 时导入失败静默跳过。
        try:
            from .marker_parser import release_converter as _release_marker
            _release_marker()
        except Exception as e:
            logger.debug("RAG: release marker converter error: %s", e)
        # ovis（OvisOCR2 VLM，#133）模型单例同步释放（~0.9GB MLX）。
        try:
            from .ovis_ocr_parser import release_model
            release_model()
        except Exception as e:
            logger.debug("RAG: release ovis model error: %s", e)

    def reload(self) -> None:
        """Drop the cached table handle so the next search sees fresh data.

        Call this after SyncKnowledgeWorker finishes writing — the agent's
        RAGEngine holds a stale table snapshot that won't reflect newly
        written rows until the table is re-opened.
        """
        with self._store_lock:
            if self._store is not None:
                try:
                    self._store.reload()
                    logger.info("RAG: VectorStore reloaded (table handle refreshed)")
                except Exception as e:
                    logger.warning("RAG: VectorStore reload error: %s", e)

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _get_store(self) -> VectorStore:
        # Double-check locking：避免并发 search() 时创建多个 VectorStore 实例
        if self._store is None:
            with self._store_lock:
                if self._store is None:
                    self._store = VectorStore(
                        persist_dir=self._vector_dir,
                        embedding_model=self._embedding_model,
                        rerank_model=self._rerank_model,
                        embedding_function=self._custom_ef,
                    )
        return self._store


def _safe_filename(path: str) -> str:
    """Convert a relative path to a safe, collision-free filename.

    Uses a short hash of the full relative path to avoid collisions between
    e.g. ``dir1/file.txt`` and ``dir1_file.txt`` which would otherwise produce
    the same safe name.
    """
    import hashlib
    h = hashlib.md5(path.encode("utf-8")).hexdigest()[:12]
    # Build a readable prefix from the basename for easier debugging
    base = os.path.basename(path)
    safe_base = "".join(c for c in base if c.isalnum() or c in "._-")[:32]
    prefix = safe_base or "document"
    return f"{prefix}_{h}"