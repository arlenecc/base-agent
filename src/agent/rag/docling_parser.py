"""Unified document parsing via docling.

docling is a single, high-quality document-understanding engine that converts
Word, Excel, PowerPoint, PDF and EPUB into structured Markdown (with layout
awareness, tables, headings, and image OCR).  This module wraps it behind the
same ``extract_text``-style interface so the RAG engine can use one parser for
all office/document formats.

Design:
  * ``extract_markdown(filepath)`` — convert any supported document to Markdown
    via docling.  Returns ``None`` when docling is unavailable (callers fall
    back to the per-format parsers in :mod:`agent.rag.parsers`).
  * Image-based PDF pages are OCR'd by ocrmac (Apple Vision, zh-Hans first)
    when available, RapidOCR otherwise; for pure-scan PDFs without a text
    layer we route through the shared OCR entry in :mod:`agent.rag.parsers`
    (same ocrmac-first policy).
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
                # OCR 后端：Apple Vision（ocrmac）优先——中文印刷体识别质量
                # 高于 RapidOCR，ANE 推理不占 CPU/内存；不可用（非 macOS /
                # 未安装 ocrmac）回退 RapidOCR。lang 必须显式给中文：docling
                # 的 OcrMacOptions 默认 ['en-US', 'es-ES', ...] 不含中文，
                # 对中文扫描页会大面积漏识别。
                from .ocr_engine import ocrmac_available
                if ocrmac_available():
                    try:
                        from docling.datamodel.pipeline_options import OcrMacOptions
                        pdf_opts.ocr_options = OcrMacOptions(
                            lang=["zh-Hans", "en-US"], recognition="accurate")
                        pdf_opts.do_ocr = True
                        logger.info("docling OCR 后端: Apple Vision (ocrmac, zh-Hans 优先)")
                    except Exception as e:
                        # docling 版本差异导致 OcrMacOptions 不可用 → 回退。
                        logger.info("OcrMacOptions 不可用（%s），回退 RapidOCR", e)
                if getattr(pdf_opts, "ocr_options", None) is None:
                    try:
                        from docling.datamodel.pipeline_options import RapidOcrOptions
                        pdf_opts.ocr_options = RapidOcrOptions()
                        pdf_opts.do_ocr = True
                        logger.info("docling OCR 后端: RapidOCR")
                    except Exception:
                        pass
                converter = DocumentConverter(
                    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
                )
                logger.info("docling DocumentConverter initialized (PDF 管线强制 CPU 加速)")
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

    # Image-based PDF handling: docling OCRs image pages via RapidOCR, but a
    # pure-scan PDF (no text layer at all) is better served by the dedicated
    # RapidOCR path in parsers.py which reuses the shared engine.  Detect that
    # case cheaply and delegate.
    #
    # 混合型 PDF（文本层 + 大量图片页）同样要拦下：docling 对它们跑全量
    # 管线（布局模型 + TableFormer + 逐页 OCR），单本耗时 40-70 分钟；而
    # parsers.py 的 PyMuPDF + RapidOCR 路径逐页流式处理（共享单例引擎 +
    # CoreML 加速 + 归一化），快数倍且内存有界（#116：实测 2 本图多的书
    # 触发 30GB+ 内存与 2 小时同步）。
    if suffix == ".pdf" and (_is_scan_pdf(path) or _pdf_is_image_heavy(path)):
        return None

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


def _is_scan_pdf(path: Path) -> bool:
    """Detect whether a PDF looks like a scan (no text layer at the front).

    只探测前几页：旧实现 ``for page in doc`` 把整本 PDF 的文本全提取一遍——
    而「判定为扫描件」后走的解析器又要逐页提取/OCR 一遍，大 PDF 的解析时间
    近乎翻倍。扫描件几乎不可能前 5 页都没有文本层，这里采样首页+中间页即可。

    Uses PyMuPDF; returns False if PyMuPDF is unavailable (let docling try).
    """
    _PROBE_PAGES = 5
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return False
    try:
        with fitz.open(str(path)) as doc:
            n = min(_PROBE_PAGES, doc.page_count)
            for i in range(n):
                if doc[i].get_text("text").strip():
                    return False
        return True
    except Exception:
        return False


# 图片页占比超过该阈值的 PDF 不进 docling：采样窗口内「docling 会为之
# 付出逐页位图处理代价」的页越多，其全量管线（逐页布局/表格/OCR，torch
# MPS 内存只增不放）越失控，交给 parsers.py 的 PyMuPDF+RapidOCR 路径
# （CoreML、共享单例、有界内存）更合适。
#
# 「会付出位图代价」的页 = 纯图页，或嵌入图片占页面面积 ≥15% 的文本页。
# 实测（#116）：中文电子书常见「每页文本层 + 满页背景图」的排版，纯图页
# 占比判定为 0%，但 docling 对每个位图区域都跑 OCR——单本 225-280 页
# 耗时 37-74 分钟、进程 RSS 涨到 36GB；而文本层直读只需数秒。
_IMAGE_HEAVY_PROBE_PAGES = 20
_IMAGE_HEAVY_RATIO = 0.3
_IMAGE_SIGNIFICANT_AREA_RATIO = 0.15


def _pdf_is_image_heavy(path: Path) -> bool:
    """Detect whether a PDF would make docling pay for per-page bitmap work.

    采样前 ``_IMAGE_HEAVY_PROBE_PAGES`` 页，统计以下两类页面占比：
      1. 纯图页（无文本层且嵌有图片）——docling 整页 OCR；
      2. 嵌有显著图片（面积 ≥ ``_IMAGE_SIGNIFICANT_AREA_RATIO`` × 页面）的
         文本页——典型如满页背景图/花边的中文电子书，docling 的 auto OCR
         仍会对位图区域逐页推理。

    占比达到 ``_IMAGE_HEAVY_RATIO`` 即判定为图片重。只开一次文档、只读前
    几页，开销毫秒级。与 ``_is_scan_pdf``（全无文本层）互补。
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return False
    try:
        with fitz.open(str(path)) as doc:
            n = min(_IMAGE_HEAVY_PROBE_PAGES, doc.page_count)
            if n == 0:
                return False
            costly = 0
            for i in range(n):
                page = doc[i]
                text = page.get_text("text").strip()
                images = page.get_images(full=True)
                if not text:
                    if images:
                        costly += 1
                    continue
                # 文本页：只有显著大图才计入（页角小 logo 不会让 docling
                # 变慢到失控，不值得为它放弃 docling 的版面分析质量）。
                page_area = abs(page.rect)
                if page_area <= 0:
                    continue
                for img in images:
                    try:
                        rects = page.get_image_rects(img[0])
                    except Exception:
                        continue
                    if any(abs(r) >= _IMAGE_SIGNIFICANT_AREA_RATIO * page_area
                           for r in rects):
                        costly += 1
                        break
            return costly / n >= _IMAGE_HEAVY_RATIO
    except Exception:
        return False
