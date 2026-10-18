"""Unified document parsing via docling.

docling is a single, high-quality document-understanding engine that converts
Word, Excel, PowerPoint, PDF and EPUB into structured Markdown (with layout
awareness, tables, headings, and image OCR).  This module wraps it behind the
same ``extract_text``-style interface so the RAG engine can use one parser for
all office/document formats.

Design:
  * ``extract_markdown(filepath)`` — convert any supported document to Markdown
    via docling.  Returns ``None`` when docling is unavailable or conversion
    fails (callers fall back to the per-format parsers in
    :mod:`agent.rag.parsers`).
  * **docling 全权接管所有 PDF**（文本层 / 扫描件 / 图片重混排），不再走
    PyMuPDF + RapidOCR 的自定义路径。PDF 的 OCR 后端由 :func:`_choose_ocr_options`
    选定：优先 ocrmac（Apple Vision 原生 OCR，离线、零额外模型、比 RapidOCR
    快），不可用时（非 macOS / 未装）回退 docling 自带的 RapidOCR。
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Formats docling can handle natively.  Text/markdown/csv/html are handled by
# the lightweight parsers (docling is overkill and slower for them).
_DOCLING_EXTENSIONS = {".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt", ".pdf", ".epub"}

# Converter singleton: docling models are heavy, so share one instance across
# all workers (access is guarded by a lock — DocumentConverter is not
# documented as thread-safe).
#
# 锁必须在模块导入时就创建好。旧实现在 _get_converter() 里「if _converter_lock
# is None: _converter_lock = threading.Lock()」，这在并发首次调用时是典型的
# check-then-act 竞态：两个线程各自新建一把锁，于是各自持自己的锁去创建
# DocumentConverter —— 同时加载两份数百 MB 的模型。
_converter = None
_converter_lock = threading.Lock()
_converter_unavailable = False


def release_converter() -> None:
    """Drop the shared docling converter so its models can be freed.

    Mirrors ``parsers.release_ocr_engine()``: ingestion runs in short bursts
    (a sync job), after which the converter's models would otherwise stay
    resident for the whole process lifetime.
    """
    global _converter
    with _converter_lock:
        _converter = None


def reset_unavailable() -> None:
    """清掉「docling 不可用」的永久禁用标志。

    ``_converter_unavailable`` 在任何异常（不只是 ImportError）下都会被置位
    且没有复位路径。典型后果：进程早期跑过一次 ingest（docling 未装 → 禁用），
    随后依赖检查自动装上了 docling，之后的同步仍然全程回退到按格式解析
    （PDF 表格/版面质量下降），且毫无提示。依赖装好后调用本函数即可恢复。
    """
    global _converter_unavailable
    with _converter_lock:
        _converter_unavailable = False


def is_docling_available() -> bool:
    """Return True if docling can be imported."""
    global _converter_unavailable
    if _converter_unavailable:
        return False
    try:
        import docling  # noqa: F401
        return True
    except ImportError:
        _converter_unavailable = True
        return False


def is_supported(ext: str) -> bool:
    return ext.lower() in _DOCLING_EXTENSIONS


# ---------------------------------------------------------------------------
# PDF OCR 后端选择：ocrmac（Apple Vision）默认，RapidOCR 兜底
# ---------------------------------------------------------------------------
# ocrmac 是 macOS 原生 OCR（Vision 框架），离线、零额外模型，识别质量与速度
# 都优于 RapidOCR（后者要加载 ~500MB ONNX 模型并跑 CoreML/CPU）。仅当运行环境
# 不是 macOS 或未安装 ocrmac 时，才回退 docling 自带的 RapidOCR。
#
# 语种以中文为主（简体 zh-Hans + 繁体 zh-Hant），兼容英文 en-US。
# recognition 固定 "accurate"——macOS 的 fast 档在本机（15.8.1）只支持拉丁语种，
# 用 fast 处理中文 PDF 会直接抛 ValueError（ocrmac 校验 language_preference
# 必须是该档可用语种的子集），已实测。
_OCR_LANG = ["zh-Hans", "zh-Hant", "en-US"]


def _choose_ocr_options():
    """docling 的 PDF OCR 后端选项：优先 ocrmac，回退 RapidOCR。

    返回 docling 的 ``OcrOptions`` 实例，挂到 ``PdfPipelineOptions.ocr_options``。
    ocrmac 不可用时（非 macOS / 未安装）回退 ``RapidOcrOptions``（docling 内部
    走默认 RapidOCR，不再是我们自己注入 CoreML EP 的版本——那是 parsers 兜底
    路径用的）。
    """
    try:
        import ocrmac  # noqa: F401  —— 仅 macOS 可用
        from docling.datamodel.pipeline_options import OcrMacOptions
        opts = OcrMacOptions(
            lang=list(_OCR_LANG),
            framework="vision",
            recognition="accurate",
        )
        logger.info("docling OCR 后端：ocrmac（Apple Vision，%s）", ",".join(_OCR_LANG))
        return opts
    except ImportError:
        from docling.datamodel.pipeline_options import RapidOcrOptions
        logger.info("docling OCR 后端：ocrmac 不可用（非 macOS / 未安装），回退 RapidOCR")
        return RapidOcrOptions()


def _ocr_backend_name() -> str:
    """供日志使用的 OCR 后端名（不实际构造引擎，零副作用）。"""
    try:
        import ocrmac  # noqa: F401
        return "ocrmac(Apple Vision)"
    except ImportError:
        return "RapidOCR(回退)"


def _get_converter():
    """Lazily build and cache the docling DocumentConverter singleton.

    管线降配（版本兼容，全部包在 try/except 中，任何失败都回退默认构造）：
    accelerator 强制 CPU + 限 4 线程。docling 2.x 在 macOS 上默认
    device='auto' → MPS（torch Metal 后端），其缓存分配器只增不放——
    实测同步图多的书时 RSS 可涨到 30GB+ 且不回落（#116 根因之一）。
    CPU 模式下布局/表格模型对文本为主的文档足够快，且内存有界。
    """
    global _converter, _converter_unavailable
    if _converter is not None:
        return _converter
    if _converter_unavailable:
        return None
    with _converter_lock:
        if _converter is not None:
            return _converter
        try:
            from docling.document_converter import DocumentConverter

            converter = None
            try:
                from docling.datamodel.accelerator_options import (
                    AcceleratorDevice,
                    AcceleratorOptions,
                )
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.document_converter import PdfFormatOption

                pdf_opts = PdfPipelineOptions()
                pdf_opts.accelerator_options = AcceleratorOptions(
                    num_threads=4, device=AcceleratorDevice.CPU
                )
                # PDF OCR 后端：ocrmac（Apple Vision）优先，RapidOCR 兜底。
                pdf_opts.ocr_options = _choose_ocr_options()
                converter = DocumentConverter(
                    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
                )
                logger.info(
                    "docling DocumentConverter initialized (PDF 管线强制 CPU + %s)",
                    _ocr_backend_name(),
                )
            except Exception as e:
                # docling 版本差异（2.13x 前后多次移动模块路径）：拿不到
                # 降配入口就用默认构造——正确性不受影响，只是 macOS 上
                # MPS 缓存可能再次推高内存。
                logger.info("docling 管线降配不可用（%s），使用默认配置", e)
                converter = DocumentConverter()
            _converter = converter
            logger.info("docling DocumentConverter initialized (global singleton)")
            return converter
        except ImportError:
            logger.info("docling not installed; falling back to per-format parsers")
            _converter_unavailable = True
            return None
        except Exception as e:
            # 只在**缺失**时永久禁用。瞬时错误（模型下载中断、磁盘抖动、
            # OOM）绝不能置位 —— 那会让本进程后续所有同步都静默降级到轻量
            # 解析器，即使原因早已消失了也不自愈。
            logger.warning("failed to init docling converter (will retry): %s", e)
            return None


def extract_markdown(filepath: str) -> Optional[str]:
    """Convert a document to Markdown using docling.

    Returns the Markdown text, or ``None`` if docling is unavailable or the
    format is not handled by docling (callers should fall back to the
    per-format parsers).
    """
    if not is_docling_available():
        return None

    path = Path(filepath)
    suffix = path.suffix.lower()
    if suffix not in _DOCLING_EXTENSIONS:
        return None

    # 所有 PDF（文本 / 扫描 / 图片重）一律交给 docling：其 PDF 管线已配置
    # ocrmac（Apple Vision）OCR 后端，扫描件与图片页由 docling 内部完成 OCR，
    # 不再分流转 PyMuPDF + RapidOCR 的自定义路径（#116 的内存/耗时问题在
    # ocrmac 原生 OCR 下不复存在——无需加载 500MB ONNX、无 MPS 缓存膨胀）。
    # 仅当 docling 自身不可用或转换失败时才由调用方回退到 parsers 的 PyMuPDF
    # 兜底路径。

    converter = _get_converter()
    if converter is None:
        return None

    try:
        # DocumentConverter 没有声明线程安全，而 ingest 用 4 个 worker 线程
        # 并行解析。旧实现只对「创建单例」加锁，convert() 本身完全不设防，
        # 并发转换会互相踩踏（状态污染/崩溃）。这里用同一把锁串行化转换。
        with _converter_lock:
            result = converter.convert(str(path))
            doc = result.document
            markdown = doc.export_to_markdown()
        # 及时释放大对象引用，避免整棵文档树在切片前一直常驻。
        del result, doc
        if not markdown or not markdown.strip():
            logger.warning("docling extracted empty markdown from %s", path.name)
            return None
        return markdown
    except Exception as e:
        # Never let a docling failure abort ingestion — caller falls back.
        logger.warning("docling conversion failed for %s: %s", path.name, e)
        return None



