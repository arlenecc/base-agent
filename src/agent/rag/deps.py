"""Dependency check, auto-install, and model pre-download for RAG.

Covers three layers:
1. Python packages (lancedb, fastembed, PyMuPDF, python-docx, etc.)
2. OCR engine (rapidocr 3.x，兼容旧版 rapidocr-onnxruntime) — auto-detected if PDF has image pages
3. Embedding model (nomic-ai/nomic-embed-text-v1.5) — pre-downloaded from HF

Runs before every sync so users never hit missing-dependency errors mid-ingest.
"""
from __future__ import annotations

import importlib
import logging
import os
import subprocess
import warnings
import glob
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# 重排序模型下载失败的短时负缓存（monotonic 时间戳）：见 ensure_rag_deps Step 3。
_RERANKER_DOWNLOAD_FAILED_AT: Optional[float] = None
_RERANKER_RETRY_INTERVAL = 600.0  # 10 分钟内不重复尝试下载

# ---------------------------------------------------------------------------
# Dependency registry
# ---------------------------------------------------------------------------


@dataclass
class DepSpec:
    """A single dependency requirement."""
    import_name: str       # module to check with importlib
    pip_spec: str          # pip install spec (e.g., "python-docx", "numpy<2.0")
    label: str             # human-readable label
    required: bool = True  # if False, only warn (e.g., OCR is optional unless needed)


# Map each file extension to its required dependencies.
# 旧版二进制格式（.doc/.xls/.ppt）不在表内：python-docx/openpyxl/python-pptx
# 均只支持 OOXML，装了也解析不了（parsers 对它们给出明确的「另存为」提示），
# 列在这里只会制造「已支持」的假象并白装依赖。
EXTENSION_DEPS: Dict[str, List[DepSpec]] = {
    ".docx": [DepSpec("docx", "python-docx", "Word 文档解析")],
    ".xlsx": [DepSpec("openpyxl", "openpyxl", "Excel 表格解析")],
    ".pptx": [DepSpec("pptx", "python-pptx", "PPT 幻灯片解析")],
    ".pdf":  [
        DepSpec("fitz", "PyMuPDF", "PDF 文字解析"),
        # OCR image pre-processing (adaptive DPI + scale normalisation + CLAHE +
        # sharpen) needs cv2. Missing → ocr_engine falls back to the raw bitmap,
        # so this stays optional rather than blocking every PDF sync.
        # headless build: no GUI/system-lib dependency (runs on servers/CI too).
        DepSpec("cv2", "opencv-python-headless", "OCR 图像预处理（归一化/增强，提质提速）", required=False),
    ],
    ".epub": [
        DepSpec("ebooklib", "ebooklib", "EPUB 电子书解析"),
        DepSpec("bs4", "beautifulsoup4", "HTML 内容清洗"),
    ],
    ".mobi": [],
    ".azw3": [],
    ".azw":  [],
    ".txt": [],
    ".md": [],
    ".markdown": [],
    ".rst": [],
    ".csv": [],
    ".tsv": [],
    ".html": [DepSpec("bs4", "beautifulsoup4", "HTML 解析")],
    ".htm":  [DepSpec("bs4", "beautifulsoup4", "HTML 解析")],
}

# Core RAG infrastructure (always checked, always required)
RAG_CORE_DEPS: List[DepSpec] = [
    DepSpec("lancedb", "lancedb", "向量数据库"),
    DepSpec("fastembed", "fastembed", "嵌入模型 (ONNX Runtime)"),
]

# Optional but recommended enhancements — missing → graceful fallback (no sync block).
# Always checked so they get auto-installed on first sync and silently skipped if
# install fails.  (chonkie: semantic chunking. Reranker now uses FastEmbed's
# cross-encoder — same ONNX stack as the embedding model, so it needs NO separate
# pip package; fastembed in RAG_CORE_DEPS already covers it.)
SEMANTIC_CHUNK_DEPS: List[DepSpec] = [
    DepSpec("chonkie", "chonkie", "语义分块 (chonkie SemanticChunker)", required=False),
]
# Reranker 不再需要独立 pip 依赖（FlagEmbedding/torch），保留空列表占位。
RERANKER_DEPS: List[DepSpec] = []

# Embedding model info for pre-download
# Embedding model: quantized ONNX variant — only 137MB (vs 548MB for FP32).
# FastEmbed uses model_quantized.onnx; we download only the required files
# instead of the full snapshot (which is ~1.6GB with all quantization formats).
EMBEDDING_MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_MODEL_LABEL = "嵌入模型 nomic-embed-text-v1.5 (~137MB, 仅下载 ONNX 量化版)"

# FastEmbed model name — uses the -Q variant (quantized ONNX, model_quantized.onnx).
# Same 768-dim embeddings, 4x smaller than FP32, 12x smaller than full snapshot.
FASTEMBED_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5-Q"

# Files actually needed by FastEmbed for the quantized model.
# Only these files are downloaded — not the entire ~1.6GB repo snapshot.
_EMBEDDING_REQUIRED_FILES = [
    "onnx/model_quantized.onnx",   # quantized ONNX model (~137MB)
    "tokenizer.json",              # tokenizer (~0.7MB)
    "config.json",                 # model config (~3KB)
    "tokenizer_config.json",       # tokenizer config (~4KB)
    "vocab.txt",                   # vocabulary (~0.2MB)
]

# HuggingFace 仓库与「必须完整落盘」的文件清单。
#
# 注意 fastembed 的**模型名**与它实际拉取的 **HF 仓库**不是一回事：
#   nomic-ai/nomic-embed-text-v1.5-Q  →  nomic-ai/nomic-embed-text-v1.5
#   BAAI/bge-reranker-v2-m3-int8      →  onnx-community/bge-reranker-v2-m3-ONNX
# 缓存目录名由 HF 仓库推导（``models--{org}--{repo}``），所以探测必须用仓库名，
# 用 fastembed 模型名会永远找不到已下载的模型。
#
# 重排序选官方 INT8 量化版（~570MB，fp32 全量 2.3GB 的一半不到；CPU EP 上有
# int8 加速，且 bge-reranker-v2-m3 的多语种/中文精排能力优于 bge-reranker-base）。
# INT8 版没有外挂 ``model.onnx_data``（fp32 才有），权重是单文件
# ``onnx/model_int8.onnx``——它是下载的最后一个文件，它完整落盘即意味着
# tokenizer 等前置文件也已就绪（中断的下载只会留下前置文件，见
# _model_file_cached 的 .incomplete 处理）。
_EMBEDDING_HF_REPO = "nomic-ai/nomic-embed-text-v1.5"
_RERANK_HF_REPO = "onnx-community/bge-reranker-v2-m3-ONNX"
_RERANK_REQUIRED_FILES = [
    "onnx/model_int8.onnx",
]

# Reranker model (FastEmbed cross-encoder, used at search time, not ingest —
# pre-download to avoid blocking the first search). ~570MB ONNX download,
# optional: search falls back to distance-based ranking if unavailable.
RERANK_MODEL_ID = "BAAI/bge-reranker-v2-m3-int8"
RERANK_MODEL_LABEL = "重排序模型 bge-reranker-v2-m3 (INT8, ~570MB, 可选)"

# Version pins: packages that need specific version constraints
VERSION_PINS: Dict[str, str] = {
    "numpy": "numpy<2.0",  # FastEmbed ONNX Runtime compat
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class DependencyReport:
    """Result of a full dependency check (packages + models)."""
    # Python packages
    missing_required: List[DepSpec] = field(default_factory=list)
    missing_optional: List[DepSpec] = field(default_factory=list)
    scanned_extensions: Set[str] = field(default_factory=set)

    # Models
    embedding_model_missing: bool = False
    reranker_model_missing: bool = False

    # OCR detection
    has_image_pdf: bool = False   # knowledge base contains image-based PDF pages
    ocr_installed: bool = False

    # Install results
    installed_packages: List[str] = field(default_factory=list)
    install_errors: List[str] = field(default_factory=list)

    # Non-blocking suggestions (e.g. upgrade onnxruntime on Apple Silicon for
    # CoreML EP). Not installed automatically, not blocking.
    suggestions: List[str] = field(default_factory=list)

    @property
    def all_missing(self) -> List[DepSpec]:
        return self.missing_required + self.missing_optional

    @property
    def has_missing(self) -> bool:
        return bool(self.missing_required or self.missing_optional)

    @property
    def has_blocking(self) -> bool:
        """True if any *required* deps or the embedding model is missing."""
        return bool(self.missing_required) or self.embedding_model_missing

    def summary(self) -> str:
        """Human-readable summary of what's missing."""
        lines: List[str] = []
        if self.missing_required:
            lines.append("缺少必要 Python 包：")
            for d in self.missing_required:
                lines.append(f"  - {d.label} (pip install {d.pip_spec})")
        if self.embedding_model_missing:
            lines.append(f"缺少嵌入模型：{EMBEDDING_MODEL_LABEL}")
            lines.append(f"  → fastembed 模型名: {FASTEMBED_MODEL_NAME}")
            lines.append(f"  → HuggingFace 仓库: {_EMBEDDING_HF_REPO} (仅需 {len(_EMBEDDING_REQUIRED_FILES)} 个文件)")
            lines.append("  → 手动下载命令（必须落到 fastembed 自己的缓存目录，")
            lines.append("     下到默认 ~/.cache/huggingface 的话 fastembed 看不到，报错依旧）：")
            lines.append("     export HF_ENDPOINT=https://hf-mirror.com")
            lines.append("     export HF_HUB_DISABLE_XET=1")
            lines.append(f'     python3 -c "from fastembed import TextEmbedding; '
                         f"TextEmbedding(model_name='{FASTEMBED_MODEL_NAME}', "
                         f"cache_dir='{_get_fastembed_cache_dir()}')\"")

        if self.has_image_pdf and not self.ocr_installed:
            lines.append("知识库包含图片版 PDF，但 OCR 引擎未安装：")
            lines.append("  - rapidocr-onnxruntime (pip install rapidocr-onnxruntime)")
        if self.missing_optional:
            lines.append("缺少可选依赖（部分功能受限）：")
            for d in self.missing_optional:
                lines.append(f"  - {d.label} (pip install {d.pip_spec})")
        if self.installed_packages:
            lines.append(f"已自动安装 {len(self.installed_packages)} 个包：")
            for pkg in self.installed_packages:
                lines.append(f"  ✅ {pkg}")
        if self.install_errors:
            lines.append("安装失败：")
            for err in self.install_errors:
                lines.append(f"  ❌ {err}")
        if self.suggestions:
            lines.append("优化建议（非必须，不影响使用）：")
            for s in self.suggestions:
                lines.append(f"  💡 {s}")
        if not lines:
            lines.append("所有依赖已就绪（Python 包 + 嵌入模型 + OCR 引擎）。")
        return "\n".join(lines)


def scan_extensions(kb_path: str) -> Set[str]:
    """Scan a knowledge base directory and return the set of file extensions present."""
    root = Path(kb_path)
    if not root.is_dir():
        logger.warning("RAG deps: knowledge base dir not found: %s", kb_path)
        return set()
    exts: Set[str] = set()
    for fp in root.rglob("*"):
        if fp.is_file() and not fp.name.startswith("."):
            exts.add(fp.suffix.lower())
    logger.info("RAG deps: scanned %s — found extensions: %s", kb_path, sorted(exts) if exts else "(none)")
    return exts


# ---------------------------------------------------------------------------
# PDF image-page detection
# ---------------------------------------------------------------------------


def _pdf_has_image_pages(kb_path: str, max_pages_per_pdf: int = 10) -> bool:
    """Quickly sample PDF files to check if any have image-based pages.

    Only checks the first ``max_pages_per_pdf`` pages of each PDF to keep
    the pre-sync check fast. Returns True as soon as one image page is found.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        # Can't check without PyMuPDF — assume no image PDFs
        return False

    root = Path(kb_path)
    # 单遍遍历 + 小写后缀判断：旧实现 rglob 两遍（"*.pdf" 和 "*.PDF"），
    # 在大知识库上等于把整棵目录树扫两次。
    pdf_files = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() == ".pdf" and not p.name.startswith(".")
    ]
    if not pdf_files:
        return False

    logger.info("RAG deps: checking %d PDF(s) for image-based pages...", len(pdf_files))
    for pdf_path in pdf_files:
        doc = None
        try:
            doc = fitz.open(str(pdf_path))
            pages_to_check = min(len(doc), max_pages_per_pdf)
            for i in range(pages_to_check):
                page = doc[i]
                text = page.get_text("text").strip()
                if not text and page.get_images(full=True):
                    logger.info(
                        "RAG deps: found image-based page in '%s' (page %d)",
                        pdf_path.name, i + 1,
                    )
                    return True
        except Exception as e:
            logger.debug("RAG deps: failed to check PDF '%s': %s", pdf_path.name, e)
        finally:
            # 旧实现在异常分支直接漏掉 doc.close()，损坏/加密 PDF 会持续
            # 泄漏文件句柄与解码缓存。
            if doc is not None:
                try:
                    doc.close()
                except Exception:
                    pass

    return False


# ---------------------------------------------------------------------------
# Embedding model pre-download
# ---------------------------------------------------------------------------


def _get_fastembed_cache_dir() -> str:
    """Return the persistent cache directory for FastEmbed models.

    FastEmbed defaults to a system temp dir (/var/folders/.../T/) which
    macOS cleans regularly, causing the model to be re-downloaded.  We
    use ~/.cache/fastembed instead — same as vector_store.py.
    """
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "fastembed")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _model_cache_dirs(cache_dir: str, hf_repo: str) -> List[str]:
    """某个 HF 仓库在 fastembed 缓存里的候选目录（覆盖两种布局）。

    * fastembed >= 0.9 用 HuggingFace hub 风格缓存：
      ``models--{org}--{repo}/``，内容在 ``blobs/<sha>``，清单在
      ``trees/*.json``；
    * 旧版 fastembed 直接把文件放在 ``{org}/{repo}/`` 下。

    两种都探，避免升级 fastembed 后探测逻辑静默失效（表现为「明明下载了却
    每次都重新下」）。
    """
    org, _, name = hf_repo.partition("/")
    return [
        os.path.join(cache_dir, "models--" + hf_repo.replace("/", "--")),
        os.path.join(cache_dir, org, name),
    ]


def _model_file_cached(model_dir: str, filename: str) -> bool:
    """判断某个模型文件是否已**完整**落盘。

    中断的下载会在 hub 缓存里留下 ``blobs/<sha>.incomplete``，而完整的
    ``blobs/<sha>`` 并不存在 —— 只按文件名判断会把半成品当成已缓存，从而跳过
    预下载，把 ~2.3GB 的下载推迟到首次检索时现场发生。
    """
    # 1) 旧版布局：文件直接就在目录里
    if os.path.isfile(os.path.join(model_dir, filename)):
        return True
    # 2) hub 布局：从 trees/*.json 清单找到 blob_id，确认 blobs/<blob_id> 存在
    for tree_path in sorted(glob.glob(os.path.join(model_dir, "trees", "*.json"))):
        try:
            with open(tree_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        entry = (manifest.get("files") or {}).get(filename)
        if not isinstance(entry, dict):
            continue
        blob_id = entry.get("blob_id")
        if blob_id and os.path.isfile(os.path.join(model_dir, "blobs", blob_id)):
            return True
    return False


def _is_embedding_model_cached() -> bool:
    """Check whether the embedding model is already cached locally.

    FastEmbed uses its OWN cache directory (NOT the standard HuggingFace
    cache).  We check the persistent cache dir (~/.cache/fastembed) that
    we configure in vector_store.py and _download_embedding_model().

    **不实例化模型**：旧实现用 ``TextEmbedding(..., lazy_load=True)`` 探路径，
    但 fastembed 部分版本的下载逻辑位于基类 ``__init__`` 而非
    ``OnnxTextEmbedding.__init__``，``lazy_load`` 只跳过 ONNX Session 创建
    —— 最坏情况下「依赖检查」这一步就会在 UI 线程上开始下载 137MB / 2.3GB。
    纯文件系统探测没有任何下载副作用。
    """
    cache_dir = _get_fastembed_cache_dir()
    for model_dir in _model_cache_dirs(cache_dir, _EMBEDDING_HF_REPO):
        if not os.path.isdir(model_dir):
            continue
        if all(_model_file_cached(model_dir, f) for f in _EMBEDDING_REQUIRED_FILES):
            logger.debug("RAG deps: embedding model cached at %s", model_dir)
            return True
    return False


def _download_embedding_model(progress_callback=None) -> Tuple[bool, str]:
    """Pre-download the embedding model by instantiating FastEmbed.

    Previously this used a subprocess to call huggingface_hub.hf_hub_download,
    but that had two problems:
    1. It downloaded to the HF cache, but FastEmbed uses its OWN cache —
       so the model was downloaded but FastEmbed didn't see it.
    2. In a PyInstaller-frozen .app, sys.executable is the frozen binary
       and the subprocess can't import huggingface_hub.

    The fix: just instantiate TextEmbedding directly.  FastEmbed handles
    the download to its own cache directory, which is where it will look
    for the model at runtime.  This works in both dev and frozen envs.
    """
    logger.info("RAG deps: downloading embedding model via FastEmbed...")
    if progress_callback:
        progress_callback(f"正在下载{EMBEDDING_MODEL_LABEL}（FastEmbed 自动缓存）...")

    try:
        from fastembed import TextEmbedding
        # Instantiating TextEmbedding triggers the download to FastEmbed's
        # own cache directory.  Use the same persistent cache_dir as
        # vector_store.py to avoid re-downloads after macOS temp cleanup.
        cache_dir = _get_fastembed_cache_dir()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            TextEmbedding(model_name=FASTEMBED_MODEL_NAME, cache_dir=cache_dir)
        logger.info("RAG deps: embedding model downloaded and cached by FastEmbed")
        if progress_callback:
            progress_callback(f"✅ {EMBEDDING_MODEL_LABEL} 下载完成")
        return True, "下载成功"
    except Exception as e:
        msg = str(e)
        logger.error("RAG deps: embedding model download failed: %s", e)
        if progress_callback:
            progress_callback(f"❌ 模型下载失败: {msg}")
        return False, msg


# ---------------------------------------------------------------------------
# Package checks
# ---------------------------------------------------------------------------


def check_dependencies(kb_path: str, include_core: bool = True) -> DependencyReport:
    """Check Python packages, embedding model, and OCR engine availability.

    Args:
        kb_path: Path to the knowledge base directory.
        include_core: Also check RAG core dependencies (lancedb, fastembed).

    Returns:
        DependencyReport with full status of all dependencies.
    """
    report = DependencyReport()
    exts = scan_extensions(kb_path)
    report.scanned_extensions = exts

    # ---- Python packages ----
    deps_to_check: List[DepSpec] = []
    if include_core:
        deps_to_check.extend(RAG_CORE_DEPS)
        # 可选增强：纳入检查 + 自动安装，但缺失不阻塞同步（都有回退路径）。
        deps_to_check.extend(SEMANTIC_CHUNK_DEPS)
        deps_to_check.extend(RERANKER_DEPS)
    for ext in exts:
        if ext in EXTENSION_DEPS:
            deps_to_check.extend(EXTENSION_DEPS[ext])

    # Deduplicate by pip_spec
    seen: Set[str] = set()
    unique_deps: List[DepSpec] = []
    for d in deps_to_check:
        if d.pip_spec not in seen:
            seen.add(d.pip_spec)
            unique_deps.append(d)

    for dep in unique_deps:
        if not _is_importable(dep.import_name):
            if dep.required:
                report.missing_required.append(dep)
            else:
                report.missing_optional.append(dep)

    # ---- OCR engine: upgrade from optional to required if image PDFs exist ----
    if ".pdf" in exts:
        has_images = _pdf_has_image_pages(kb_path)
        report.has_image_pdf = has_images
        # Check if OCR is installed.
        # 新版 rapidocr (3.x) 与旧版 rapidocr-onnxruntime (1.x) 任一可用即可，
        # 引擎选择见 parsers._get_ocr_engine。
        report.ocr_installed = (_is_importable("rapidocr")
                                or _is_importable("rapidocr_onnxruntime"))
        if has_images and not report.ocr_installed:
            # Move OCR from optional to required.
            # 必须同时从 missing_optional 里移除：否则同一个 pip_spec 会同时
            # 出现在「缺失必需」和「缺失可选」里 → pip 装两遍、summary 里
            # 重复列出同一项。按 pip_spec 去重（DepSpec 是 dataclass，值相等
            # 即可比较，但这里显式按 pip_spec 匹配更稳）。
            ocr_spec = DepSpec("rapidocr", "rapidocr", "图片型 PDF OCR 识别")
            report.missing_optional = [
                d for d in report.missing_optional if d.pip_spec != ocr_spec.pip_spec
            ]
            if all(d.pip_spec != ocr_spec.pip_spec for d in report.missing_required):
                report.missing_required.append(ocr_spec)

    # ---- docling: optional unified parser (falls back to per-format parsers) ----
    if exts & {".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt", ".pdf", ".epub"}:
        if not _is_importable("docling"):
            report.missing_optional.append(
                DepSpec("docling", "docling",
                        "统一文档解析引擎 docling（可选，未安装时回退按格式解析）",
                        required=False)
            )

    # ---- jieba: BM25 中文预分词（RAG 全文检索 + 技能语义检索共用）----
    # 缺失时 _tokenize_for_fts 静默退化为空白切分：中文整句作为一个 token，
    # BM25 召回大幅下降且用户无感知。纯 Python + 内置词典（~19MB）无重依赖，
    # 与 chonkie 同级处理——缺失时随同步自动安装。
    # 注意：写入侧退化是持久的（text_fts 列在 add 时已按坏分词落库），安装
    # 后需 force 重新同步才能修复历史向量；查询侧装完即生效。
    if not _is_importable("jieba"):
        report.missing_optional.append(
            DepSpec("jieba", "jieba",
                    "中文分词 jieba（可选，未安装时 BM25 中文检索退化）",
                    required=False)
        )

    # ---- model2vec: chonkie 语义切片的静态嵌入快路径（potion-base-32M）----
    # 没有它，chonkie 的 AutoEmbeddings 构造 potion 会失败，semantic_chunker
    # 回退共享 nomic 适配器——BERT 推理 + 2N 次 embed，88k 字符章节要
    # 90-150s（#118）。model2vec 纯 numpy 查表，秒级。仅在 chonkie 已
    # 安装时才有意义（chonkie 都没有话语义切片整体回退递归）。
    if _is_importable("chonkie") and not _is_importable("model2vec"):
        report.missing_optional.append(
            DepSpec("model2vec", "model2vec",
                    "语义切片静态嵌入 model2vec（可选，缺失时切片走 nomic 慢路径）",
                    required=False)
        )

    # ---- Embedding model ----
    if include_core:
        report.embedding_model_missing = not _is_embedding_model_cached()

    # ---- Reranker model (optional, only warn) ----
    report.reranker_model_missing = not _is_reranker_cached()

    # ---- Apple Silicon: CoreML EP needs onnxruntime >= 1.31.0 ----
    # Non-blocking suggestion only — we do NOT force-upgrade an existing
    # onnxruntime (that could break fastembed/rapidocr in the user's env).
    # ocr_engine already refuses CoreML when the version is too old; this
    # surfaces the upgrade command so the user can opt in.
    try:
        from .ocr_engine import is_apple_silicon, ort_version
        if is_apple_silicon():
            ov = ort_version()
            if ov is not None and ov < (1, 31, 0):
                report.suggestions.append(
                    "Apple Silicon 上启用 OCR 的 CoreML EP 加速（CPU+GPU+ANE）需要 "
                    f"onnxruntime>={'.'.join(str(x) for x in (1, 31, 0))}"
                    f"（当前 {'.'.join(str(x) for x in ov)}）。可运行："
                    "pip install -U 'onnxruntime>=1.31.0'"
                )
    except Exception:
        pass

    # Log summary
    if report.has_missing or report.embedding_model_missing:
        logger.info(
            "RAG deps: check complete — %d required pkgs missing, %d optional, "
            "embedding_model=%s, ocr=%s, image_pdf=%s",
            len(report.missing_required), len(report.missing_optional),
            "MISSING" if report.embedding_model_missing else "OK",
            "OK" if report.ocr_installed else "MISSING",
            report.has_image_pdf,
        )
    else:
        logger.info("RAG deps: all %d packages + embedding model + OCR are ready", len(unique_deps))

    return report


def install_packages(specs: List[str], progress_callback=None) -> Tuple[bool, str]:
    """Install pip packages in the current Python environment.

    Args:
        specs: List of pip install specs (e.g., ["python-docx", "numpy<2.0"]).
        progress_callback: Optional callable(str) for progress updates.

    Returns:
        (success, output_message)
    """
    if not specs:
        return True, "无需安装。"

    logger.info("RAG deps: installing %s", ", ".join(specs))

    # Apply version pins
    final_specs: List[str] = []
    for spec in specs:
        pkg_name = spec.split("<")[0].split(">")[0].split("=")[0].split("[")[0].strip().lower()
        if pkg_name in VERSION_PINS and "<" not in spec and ">" not in spec:
            final_specs.append(VERSION_PINS[pkg_name])
        else:
            final_specs.append(spec)

    # PyInstaller 冻结环境下 sys.executable 是 .app 二进制，`-m pip` 必然失败
    # （同文件 _download_embedding_model 已经为冻结环境改过子进程方案，这里却
    # 漏了）。优先用启动时的真实解释器；拿不到再退回 sys.executable 并明确报错。
    python_exe = getattr(sys, "_base_executable", None) or sys.executable
    frozen = bool(getattr(sys, "frozen", False))

    cmd = [python_exe, "-m", "pip", "install"] + final_specs
    if progress_callback:
        progress_callback(f"正在安装: {' '.join(final_specs)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5-minute timeout
        )
        if result.returncode == 0:
            msg = f"安装成功: {' '.join(final_specs)}"
            logger.info("RAG deps: install succeeded: %s", " ".join(final_specs))
            if progress_callback:
                progress_callback(msg)
            return True, msg
        else:
            error_tail = result.stderr.strip().split("\n")[-5:] if result.stderr else []
            msg = f"安装失败 (exit {result.returncode}):\n" + "\n".join(error_tail)
            logger.error("RAG deps: install failed (exit %d): %s", result.returncode, "\n".join(error_tail))
            if progress_callback:
                progress_callback(msg)
            return False, msg
    except subprocess.TimeoutExpired:
        msg = "安装超时（5 分钟），请手动执行: pip install " + " ".join(final_specs)
        logger.error("RAG deps: install timed out after 5 min: %s", " ".join(final_specs))
        if progress_callback:
            progress_callback(msg)
        return False, msg
    except FileNotFoundError as e:
        # 冻结环境里 _base_executable 也可能不可执行 —— 明确告知而不是一句
        # 「安装异常」让用户无从下手。
        msg = ("找不到可用的 Python 解释器执行 pip"
               + ("（当前为打包环境，请在宿主环境手动安装）" if frozen else "")
               + f": {e}")
        logger.error("RAG deps: install interpreter unavailable: %s", e)
        if progress_callback:
            progress_callback(msg)
        return False, msg
    except Exception as e:
        msg = f"安装异常: {e}"
        logger.error("RAG deps: install exception: %s", e)
        if progress_callback:
            progress_callback(msg)
        return False, msg


def ensure_dependencies(
    kb_path: str,
    auto_install: bool = True,
    include_core: bool = True,
    progress_callback=None,
) -> DependencyReport:
    """Check and optionally auto-install ALL missing dependencies.

    This is the main entry point called by InstallDepsWorker before every sync.
    It handles:
    1. Python packages (auto-install via pip)
    2. Embedding model (pre-download from HuggingFace)
    3. OCR engine (auto-install if image-based PDFs are detected)

    Args:
        kb_path: Path to knowledge base directory.
        auto_install: If True, install missing deps automatically.
        include_core: Also check RAG core dependencies.
        progress_callback: Optional callable(str) for progress updates.

    Returns:
        DependencyReport describing what was missing and what was installed.
    """
    report = check_dependencies(kb_path, include_core=include_core)

    if not auto_install:
        return report

    installed_anything = False

    # ---- Step 1: Install missing Python packages ----
    if report.has_missing:
        specs_to_install: List[str] = []
        for dep in report.missing_required:
            specs_to_install.append(dep.pip_spec)
        for dep in report.missing_optional:
            specs_to_install.append(dep.pip_spec)

        if specs_to_install:
            logger.info("RAG deps: auto-installing %d packages: %s",
                        len(specs_to_install), ", ".join(specs_to_install))
            if progress_callback:
                progress_callback(f"检测到缺少 {len(specs_to_install)} 个依赖，开始自动安装…")

            success, msg = install_packages(specs_to_install, progress_callback)

            if success:
                report.installed_packages.extend(specs_to_install)
                installed_anything = True
                # docling 一旦初始化失败过（未安装 / 瞬时错误）就会被永久禁用，
                # 装好之后必须清掉这个标志，否则本次进程再也不会用 docling。
                try:
                    from . import docling_parser
                    docling_parser.reset_unavailable()
                except Exception:
                    pass
                # 只复查刚装的那些包的导入情况，**不**重跑 check_dependencies：
                # 它内含 scan_extensions()（rglob 整棵知识库树）与
                # _pdf_has_image_pages()（逐本打开 PDF 抽前 10 页文本），
                # 装完包后再跑一遍等于把最贵的一步做两次，同步前先白等一轮。
                still_missing = [
                    d for d in report.all_missing if not _is_importable(d.import_name)
                ]
                report.missing_required = [d for d in still_missing if d.required]
                report.missing_optional = [d for d in still_missing if not d.required]
                # Preserve install info
                report.installed_packages = specs_to_install
                if report.has_blocking:
                    logger.warning("RAG deps: post-install — %d deps still missing",
                                   len(report.missing_required))
                else:
                    logger.info("RAG deps: all packages installed successfully")
                if progress_callback:
                    if report.has_blocking:
                        progress_callback("部分依赖安装后仍无法导入，请手动检查。")
                    else:
                        progress_callback("所有 Python 依赖已安装成功。")
            else:
                report.install_errors.append(msg)
                if progress_callback:
                    progress_callback(f"自动安装失败: {msg}")

    # ---- Step 2: Pre-download embedding model ----
    if report.embedding_model_missing:
        if progress_callback:
            progress_callback(f"嵌入模型未缓存，开始下载 {EMBEDDING_MODEL_LABEL}…")
        success, msg = _download_embedding_model(progress_callback)
        if success:
            report.embedding_model_missing = False
            installed_anything = True
        else:
            report.install_errors.append(f"嵌入模型下载失败: {msg}")

    # ---- Step 3: Pre-download reranker model (optional, non-blocking) ----
    # 不依赖 report.reranker_model_missing：它是在 pip 安装**之前**判定的，
    # fastembed 刚被 Step 1 从旧版升级上来的场景里该标志会被跳过，模型下载
    # 就要延迟到下一轮同步。这里用安装完成后的实时状态重新判定。
    if _is_importable("fastembed") and not _is_reranker_cached():
        global _RERANKER_DOWNLOAD_FAILED_AT
        recent_failure = (
            _RERANKER_DOWNLOAD_FAILED_AT is not None
            and time.monotonic() - _RERANKER_DOWNLOAD_FAILED_AT < _RERANKER_RETRY_INTERVAL
        )
        if recent_failure:
            # 负缓存：上次下载失败（网络不通等）后不再每次同步都同步阻塞一整轮
            # 网络超时。重排序是可选能力，跳过不影响同步。
            report.install_errors.append(
                "重排序模型此前下载失败，短时间内不再重试（可选，不影响同步）"
            )
        else:
            if progress_callback:
                progress_callback(f"重排序模型未缓存，开始后台下载 {RERANK_MODEL_LABEL}…")
            success, msg = _download_reranker_model(progress_callback)
            if success:
                report.reranker_model_missing = False
                installed_anything = True
                _RERANKER_DOWNLOAD_FAILED_AT = None
            else:
                # Reranker is optional — don't block sync if download fails
                report.install_errors.append(f"重排序模型下载失败（可选，不影响同步）: {msg}")
                _RERANKER_DOWNLOAD_FAILED_AT = time.monotonic()

    if installed_anything and progress_callback:
        progress_callback("依赖检查完成。")

    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_importable(module_name: str) -> bool:
    """Check if a module can be imported without importing it globally."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False
    except Exception:
        # Some packages raise non-ImportError on import but are still usable.
        return True


def _is_reranker_cached() -> bool:
    """Check whether the reranker model is already cached locally.

    与 ``_is_embedding_model_cached`` 同样做纯文件系统探测（不实例化模型，
    避免依赖检查阶段触发 2.3GB 下载）。bge-reranker-v2-m3 的权重是
    ``onnx/model.onnx`` + 外挂 ``onnx/model.onnx_data``，两个都要完整才算缓存。
    """
    cache_dir = _get_fastembed_cache_dir()
    for model_dir in _model_cache_dirs(cache_dir, _RERANK_HF_REPO):
        if not os.path.isdir(model_dir):
            continue
        if all(_model_file_cached(model_dir, f) for f in _RERANK_REQUIRED_FILES):
            logger.debug("RAG deps: reranker model cached at %s", model_dir)
            return True
    return False


def _download_reranker_model(progress_callback=None) -> Tuple[bool, str]:
    """Pre-download the reranker model via FastEmbed.

    用 ``TextCrossEncoder(..., lazy_load=True)``：fastembed 会把 catalog 里的
    权重 + tokenizer 下到 HF-hub 风格缓存，但**不创建** onnxruntime 会话——
    依赖检查跑在同步前的 UI 线程上，少了 ~0.5GB 常驻内存与数秒模型加载。
    调用是幂等的：已缓存时瞬间返回。
    cache_dir 显式指定为持久目录（与嵌入模型同一目录），不用 fastembed 默认的
    临时目录（macOS 会定期清理，曾导致每次同步重新下载，#121）。

    FastEmbed handles the download to its own cache directory (same as the
    embedding model), which is where it will look for the model at runtime.
    Works in both dev and frozen (PyInstaller) environments.
    """
    logger.info("RAG deps: downloading reranker model via FastEmbed...")
    if progress_callback:
        progress_callback(f"正在下载{RERANK_MODEL_LABEL}（FastEmbed 自动缓存）...")

    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
    except ImportError:
        try:
            # fastembed 0.6–0.8 在顶层导出 TextCrossEncoder，0.9 起移入子模块。
            from fastembed import TextCrossEncoder  # type: ignore[no-redef]
        except ImportError:
            msg = "fastembed 未安装或版本过旧（需要 >= 0.6）"
            logger.warning("RAG deps: reranker download failed: %s", msg)
            if progress_callback:
                progress_callback(f"⚠ 重排序模型下载失败（可选）: {msg}")
            return False, msg

    try:
        cache_dir = _get_fastembed_cache_dir()
        # providers 显式强制 CPU-only（lazy_load=True 本就不建 onnx 会话，
        # 这里是防御性的：万一将来懒加载语义变化，EP 选择也不会漂到 CUDA）。
        # 与 vector_store._cpu_only_kwargs 同一套签名探测逻辑——本地重复一份，
        # 避免 deps 反向依赖 vector_store（会把 lancedb/fastembed 的导入拉进
        # 依赖检查路径）。
        _cpu_only: Dict[str, Any] = {}
        try:
            import inspect as _inspect
            if "providers" in _inspect.signature(TextCrossEncoder.__init__).parameters:
                _cpu_only = {"providers": ["CPUExecutionProvider"]}
        except (TypeError, ValueError):  # pragma: no cover
            pass
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            TextCrossEncoder(
                RERANK_MODEL_ID,
                lazy_load=True,
                cache_dir=cache_dir,
                **_cpu_only,
            )
        logger.info("RAG deps: reranker model downloaded and cached by FastEmbed")
        if progress_callback:
            progress_callback(f"✅ {RERANK_MODEL_LABEL} 下载完成")
        return True, "下载成功"
    except Exception as e:
        # Keep only the last line — HF/ONNX errors can be multi-page tracebacks.
        msg = str(e).strip().split("\n")[-1] if str(e).strip() else "unknown error"
        logger.warning("RAG deps: reranker download exception: %s", e)
        if progress_callback:
            progress_callback(f"⚠ 重排序模型下载失败（可选）: {msg}")
        return False, msg
