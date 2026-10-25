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
import os
import os
import threading
import time
from pathlib import Path
from typing import List, Optional

# 强制 HuggingFace 离线（#131）：docling convert 首次运行会向 huggingface.co
# 检查模型更新，网络不通时 connect() 要等完整 TCP 超时 + hub 客户端重试，
# 实测把整个同步卡死 20+ 分钟（2026-10-20 抓到 worker 阻塞在 __connect，
# 上层是 pydantic 模型构造，OCR 的 VN 线程同时挂起）。docling 所需模型
# （layout-heron / TableFormer）早已缓存在 ~/.cache/huggingface，离线模式
# 直接走缓存；未缓存的全新环境会立刻报错 → 走轻量解析器回退，不会挂死。
# setdefault：用户显式设置的值优先。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# 保险：若 huggingface_hub 在本模块之前已被导入（其 constants 已固化
# HF_HUB_OFFLINE=False），直接改写常量兜底。部分 hf 模块在 import 时绑定了
# 旧值，改不干净——根治靠 agent/__init__.py 的最早设置，这里只是兜底。
try:
    import huggingface_hub.constants as _hf_constants
    if not _hf_constants.HF_HUB_OFFLINE:
        _hf_constants.HF_HUB_OFFLINE = True
except Exception:
    pass

from .ocr_scale_policy import (
    analyze_pdf as _analyze_ocr_scale,
    make_scale_policy_pipeline_cls,
    set_current_policy as _set_scale_policy,
    set_scale_config as _set_scale_cfg,
)

logger = logging.getLogger(__name__)

# OCR 渲染 scale 配置（#132，经 set_ocr_scale_config 注入）。默认固定
# 2.0、动态策略关闭——位图内存 ∝ scale²，2.0 相比旧默认 3.0 省 56%。
#
# 动态策略关闭时（默认）：**完全不挂载**策略管线、不做预分析、不包装
# OCR 模型——scale 只在 converter 构建时设进 OcrMacOptions 一次，默认
# 路径零额外开销。动态开启时才构建带 ScalePolicyOcrModel 的管线。
_ocr_scale_base = 2.0
_ocr_scale_dynamic = False

# 大 PDF 分批：仅当**文件大小 > _PDF_BATCH_BYTES (50MB)** 才拆批，否则整文件
# 直接 convert（小文件整载入更快、且内存可控）。超过阈值后用 PyMuPDF 按
# _PDF_BATCH_PAGES (50 页) 切成临时小文件、逐批 convert 再拼接，单批内存上限
# 锁死在 50 页，避免整文件转换时整本文档树常驻内存把大 PDF 一次性撑爆。
_PDF_BATCH_BYTES = 50 * 1024 * 1024
_PDF_BATCH_PAGES = 50
# docling 并行实例池大小（#本任务）：同时转换不同文件的 converter 实例数。
# ⚠️ 内存警示：每个实例常驻模型权重约 4.6GB，N 实例 ≈ 4.6×N GB 常驻——
# 2 实例 ≈ 9GB 常驻，建议 RAM ≥16GB；若仍紧张可继续调小到 1（退化为串行）。
# 分批只约束单批文档树这种瞬时内存，消除不了常驻权重，故并行度受 RAM 硬约束。
_DOCLING_POOL_SIZE = 2
_ocr_scale_table = 2.5


def set_ocr_scale_config(base: float = 2.0, dynamic: bool = False,
                         table: float = 2.5) -> None:
    """注入 OCR scale 配置（UI 保存设置 / 应用配置时调用）。

    base   固定 scale（rag_ocr_scale_base）
    dynamic 动态策略开关（rag_ocr_scale_dynamic，默认 False）
    table  动态策略下表格页的 scale（rag_ocr_scale_table）

    任一值变化时释放已构建的 converter 单例——scale / 动态开关在 converter
    构建时固化（OcrMacOptions 构造 + pipeline_cls 选择），不重建的话改配置
    要重启应用才生效。
    """
    global _ocr_scale_base, _ocr_scale_dynamic, _ocr_scale_table
    new_base = float(base) if base else 2.0
    new_dynamic = bool(dynamic)
    new_table = float(table) if table else 2.5
    changed = (new_base != _ocr_scale_base
               or new_dynamic != _ocr_scale_dynamic
               or new_table != _ocr_scale_table)
    _ocr_scale_base = new_base
    _ocr_scale_dynamic = new_dynamic
    _ocr_scale_table = new_table
    _set_scale_cfg(new_dynamic, new_base, new_table)
    if changed and _converters:
        logger.info(
            "OCR scale 配置变化（base=%.1f dynamic=%s table=%.1f），"
            "释放 converter 以重建",
            new_base, new_dynamic, new_table,
        )
        release_converter()

# Formats docling can handle natively.  Text/markdown/csv/html are handled by
# the lightweight parsers (docling is overkill and slower for them).
# 不含 .epub：EPUB 有专门的 ebook_parser（pandoc → Markdown + 章节切块，
# 秒级完成）；docling 处理 epub 要先构建 ~3-4GB 的 DocumentConverter
# （布局模型 + TableFormer，torch），与 ebook_parser 完全重复且慢得多。
# 2026-10 实测：同步中 docling converter 构建是 2 秒内 +3.7GB 内存突增的
# 元凶。.doc/.xls 等 legacy 格式由调用方提前拦截（docling 同样不支持）。
_DOCLING_EXTENSIONS = {".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt", ".pdf"}

# docling 并行实例池（#本任务）：docling 转换是 PDF 解析的吞吐瓶颈，单实例串行
# 无法利用多核。现改为固定大小的 converter 实例池（_DOCLING_POOL_SIZE），不同
# 实例可同时 convert 不同文件——大 PDF 已按 50MB/50 页分批（见下方常量），单批
# 文档树内存可控，因此并行不会瞬间撑爆内存。
#
# ⚠️ 内存警示：每个实例常驻模型权重约 4.6GB，N 实例 ≈ 4.6×N GB 常驻（4 实例
# ≈ 18GB+）。这是分批消除不了的——分批只约束「单批文档树」这种瞬时内存，消除
# 不了常驻权重。故 4 实例需机器有充足 RAM（建议 ≥32GB）；16GB 机器请调小
# _DOCLING_POOL_SIZE 到 2。2026-10-23 曾因两实例 + 工作集把 16GB 机器打崩，当时
# 的教训是「常驻权重线性叠加」，本改动用「分批 + 可调池大小」重新打开并行。
#
# 池槽在模块导入时不创建；首次借出时按需在锁内 lazy 构建（最多 _DOCLING_POOL_SIZE
# 个），避免启动即一次性加载数份权重。锁/条件变量必须在模块导入时就建好，防止
# 并发首次借出时的 check-then-act 竞态（两个线程各自新建锁 → 各自持锁创建实例 →
# 同时加载多份模型）。
_converters: List = []            # 池槽：DocumentConverter 实例或 None
_converter_busy: List[bool] = []  # 与 _converters 同长的占用标记
_converter_pool_lock = threading.Lock()
_converter_pool_cond = threading.Condition(_converter_pool_lock)
_converter_unavailable = False

# ---- 空闲自动释放（#131）----
# 2026-10-20 实测：同步第 3 分钟排到第一个 PDF/office 文件时构建 converter
# （import torch + docling-layout-heron 权重 + TableFormer + 首次推理激活）
# 一次性 +4.6GB，此后常驻到同步结束才被 engine 的 release_converter 释放。
# 对「大部分是 txt/epub、只夹了一两个 PDF」的语料，这 4GB+ 要白占几十分钟。
# 这里加 in-use 计数 + 空闲计时：converter 空闲超过 _IDLE_RELEASE_SECS 且无
# 活跃转换时由后台线程释放；下一个 docling 文件到来时 _acquire_converter 会重建
# 池实例（构建 ~2s，代价远小于长时间白占 4GB+）。
_IDLE_RELEASE_SECS = 90.0
_IDLE_CHECK_INTERVAL = 15.0
_converter_in_use = 0        # 正在 convert() 的调用数（含构建期间）
_converter_last_used = 0.0   # monotonic 时间戳
_reaper_started = False
# reaper 启动标志的专用锁：绝不能用 _converter_pool_lock——它在 converter.convert()
# 期间被长时间持有（整份文档转换数分钟），_start_idle_reaper 在 _build_converter
# 成功路径上被调用，抢这把锁会让刚构建完 converter 的 worker 卡在另一个
# worker 的整个转换期间（2026-10-20 实测：正是这个等待让 py-spy 抓到
# Thread-4 卡在 _start_idle_reaper，掩盖了真正的 convert 挂死）。
_reaper_start_lock = threading.Lock()


def _touch_converter() -> None:
    """记录「converter 刚被使用」，重置空闲计时。"""
    global _converter_last_used
    _converter_last_used = time.monotonic()


def _start_idle_reaper() -> None:
    """启动后台空闲释放线程（进程内仅启动一次）。"""
    global _reaper_started
    if _reaper_started:
        return
    with _reaper_start_lock:
        if _reaper_started:
            return
        _reaper_started = True

    def _reap() -> None:
        while True:
            time.sleep(_IDLE_CHECK_INTERVAL)
            with _converter_pool_lock:
                # 仅当无活跃转换且空闲超时才释放全部实例；正在进行的转换持有
                # 局部引用，置 None 不影响其执行。
                if (_converter_in_use or _converter_last_used <= 0.0
                        or not _converters):
                    continue
                idle = time.monotonic() - _converter_last_used
                if idle < _IDLE_RELEASE_SECS:
                    continue
                n = len([c for c in _converters if c is not None])
                if n == 0:
                    continue
                logger.info(
                    "docling converter 空闲 %.0fs，释放 %d 个布局/表格模型（数 GB）",
                    idle, n,
                )
                for i in range(len(_converters)):
                    _converters[i] = None
                    _converter_busy[i] = False

    threading.Thread(
        target=_reap, name="docling-idle-reaper", daemon=True
    ).start()


def release_converter() -> None:
    """释放 docling converter 池的全部实例，归还模型权重占用的内存。

    同步结束 / 配置变更时由 engine 调用。仅清空池槽引用；正在进行的转换持有
    局部引用，不受影响。
    """
    with _converter_pool_lock:
        for i in range(len(_converters)):
            _converters[i] = None
            _converter_busy[i] = False


def reset_unavailable() -> None:
    """清掉「docling 不可用」的永久禁用标志。

    ``_converter_unavailable`` 在任何异常（不只是 ImportError）下都会被置位
    且没有复位路径。典型后果：进程早期跑过一次 ingest（docling 未装 → 禁用），
    随后依赖检查自动装上了 docling，之后的同步仍然全程回退到按格式解析
    （PDF 表格/版面质量下降），且毫无提示。依赖装好后调用本函数即可恢复。
    """
    global _converter_unavailable
    _converter_unavailable = False


def is_docling_available() -> bool:
    """Return True if docling can be imported.

    用 ``find_spec`` 探测而**不是真实 import**：import docling 会连带加载
    torch 全家桶（~1-2GB）。这条探测在每个 PDF/EPUB 的解析入口都会走到，
    为探测而加载 torch 纯属浪费（2026-10 实测：同步中 2 秒内 +3.7GB 的
    元凶之一）。
    """
    global _converter_unavailable
    if _converter_unavailable:
        return False
    try:
        import importlib.util
        return importlib.util.find_spec("docling") is not None
    except Exception:
        return False


def is_supported(ext: str) -> bool:
    return ext.lower() in _DOCLING_EXTENSIONS


def _build_converter():
    """构建**一个** docling DocumentConverter 实例（不缓存到模块级单例）。

    global 声明必须加：函数内 except ImportError 分支会赋值
    ``_converter_unavailable = True``，Python 会据此把该名视为局部变量，导致
    函数开头的 ``if _converter_unavailable`` 读即 UnboundLocalError（实测 docling
    因此对全部 PDF 抛错、整体回退 marker——#本任务回归）。
    """
    global _converter_unavailable

    if _converter_unavailable:
        return None
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
            # 加速器：MPS（torch Metal）。热转换快 ~35%、峰值平坦；布局模型移到
            # GPU 后 CPU 核让给 RapidOCR/嵌入。注意：不要设
            # PYTORCH_MPS_HIGH_WATERMARK_RATIO——docling 的 transformers 引擎
            # 内部设 low=1.4，高水位低于它会直接报错。
            pdf_opts.accelerator_options = AcceleratorOptions(
                num_threads=4, device=AcceleratorDevice.MPS
            )
            # 单文档硬超时（#131）：convert 内部可能永久挂死。docling 原生支持
            # document_timeout，超时抛错 → extract_markdown 捕获 → 回退轻量解析器。
            # 3600s：图片书逐页 OCR + 布局 + TableFormer 较慢，超时只是防永久挂死。
            try:
                pdf_opts.document_timeout = 3600.0
            except Exception:
                pass  # 老版本无此字段
            # OCR 后端：Apple Vision（ocrmac）优先，中文印刷体识别质量高于
            # RapidOCR；不可用回退 RapidOCR。lang 必须显式给中文。
            from .ocr_engine import ocrmac_available
            if ocrmac_available():
                try:
                    from docling.datamodel.pipeline_options import OcrMacOptions
                    # OCR 渲染 scale（#132）：默认 2.0（3.0→2.0 省 56% 位图内存）。
                    pdf_opts.ocr_options = OcrMacOptions(
                        lang=["zh-Hans", "en-US"], recognition="accurate",
                        scale=_ocr_scale_base,
                    )
                    pdf_opts.do_ocr = True
                    logger.info("docling OCR 后端: Apple Vision (ocrmac, zh-Hans 优先)")
                except Exception as e:
                    logger.info("OcrMacOptions 不可用（%s），回退 RapidOCR", e)
            if getattr(pdf_opts, "ocr_options", None) is None:
                try:
                    from docling.datamodel.pipeline_options import RapidOcrOptions
                    try:
                        pdf_opts.ocr_options = RapidOcrOptions(scale=_ocr_scale_base)
                    except Exception:
                        pdf_opts.ocr_options = RapidOcrOptions()
                    pdf_opts.do_ocr = True
                    logger.info("docling OCR 后端: RapidOCR")
                except Exception:
                    pass
            # OCR scale 动态策略（#132）：仅动态开关开启时挂载策略管线。
            fmt_kwargs: dict = {"pipeline_options": pdf_opts}
            if _ocr_scale_dynamic:
                policy_pipeline_cls = make_scale_policy_pipeline_cls()
                if policy_pipeline_cls is not None:
                    fmt_kwargs["pipeline_cls"] = policy_pipeline_cls
                    logger.info(
                        "OCR scale 动态策略启用（表格页 %.1f / 其余 %.1f）",
                        _ocr_scale_table, _ocr_scale_base,
                    )
            converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(**fmt_kwargs)}
            )
        except Exception as e:
            # docling 版本差异（2.13x 前后多次移动模块路径）：拿不到降配入口就
            # 用默认构造——正确性不受影响，只是 macOS 上 MPS 缓存可能推高内存。
            logger.info("docling 管线降配不可用（%s），使用默认配置", e)
            converter = DocumentConverter()
        logger.info("docling DocumentConverter 实例已构建（并行池）")
        _start_idle_reaper()
        return converter
    except ImportError:
        logger.info("docling not installed; falling back to per-format parsers")
        _converter_unavailable = True
        return None


def _acquire_converter():
    """从并行池中借出一个 converter 实例（阻塞直到有空闲槽）。

    池大小上限 _DOCLING_POOL_SIZE；槽位首次借出时 lazy 构建（最多构建池大小个）。
    每个实例被借出期间标记 busy，保证同一实例不被两个线程同时使用
    （DocumentConverter 非线程安全）。
    """
    global _converter_in_use
    if _converter_unavailable:
        return None
    with _converter_pool_cond:
        while True:
            for i in range(len(_converters)):
                if not _converter_busy[i]:
                    c = _converters[i]
                    if c is None:
                        c = _build_converter()
                        if c is None:
                            return None
                        _converters[i] = c
                    _converter_busy[i] = True
                    _converter_in_use += 1
                    _touch_converter()
                    return c
            if len(_converters) < _DOCLING_POOL_SIZE:
                _converters.append(None)
                _converter_busy.append(False)
                continue
            _converter_pool_cond.wait()


def _release_converter_instance(c) -> None:
    """归还一个 converter 实例到池（标记空闲并唤醒等待者）。"""
    global _converter_in_use
    with _converter_pool_cond:
        for i in range(len(_converters)):
            if _converters[i] is c:
                _converter_busy[i] = False
                break
        _converter_in_use = max(0, _converter_in_use - 1)
        _touch_converter()
        _converter_pool_cond.notify_all()


def _split_pdf_into_batches(src: str, batch_size: int) -> List[str]:
    """把 PDF 按 batch_size 页切成临时小文件，返回临时路径列表。

    返回空列表表示「无需拆分」：页数 <= batch_size，或无法用 PyMuPDF 读取
    （此时调用方回退整文件转换）。每个临时文件可单独交给 docling convert，
    调用方负责用完删除。目的：整文件 convert 会把整本文档树常驻内存（图片书
    尤甚），拆批后单批内存上限被锁死在 batch_size 页。
    """
    try:
        import fitz, tempfile
    except Exception:
        return []
    doc = None
    tmp_paths: List[str] = []
    try:
        doc = fitz.open(src)
        n = doc.page_count
        if n <= batch_size:
            return []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            sub = fitz.open()
            sub.insert_pdf(doc, from_page=start, to_page=end - 1)
            fd, p = tempfile.mkstemp(suffix=".pdf", prefix="docling_batch_")
            os.close(fd)
            sub.save(p)
            sub.close()
            tmp_paths.append(p)
        return tmp_paths
    except Exception as e:
        logger.warning("PDF 分页拆分失败（回退整文件转换）: %s", e)
        for p in tmp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass
        return []
    finally:
        if doc is not None:
            doc.close()


def _convert_locked(converter, path: str, scale_policy) -> Optional[str]:
    """convert 单个文件（或单批），返回 Markdown 或 None。

    封装心跳看门狗、convert + export_to_markdown、大对象释放、MPS 缓存清理。
    整文件与「按页分批」两条路径共用——分批时每批独立走一遍，使单批内存上限
    锁死在 batch_size 页。并发安全由调用方保证：converter 实例从并行池借出、
    期间独占（busy 标记），不同线程使用不同实例，无需再持全局转换锁。in-use
    计数与实例归还也由池的借出/归还负责，这里不再管理。
    """
    _convert_done = threading.Event()

    def _heartbeat() -> None:
        t0 = time.monotonic()
        while not _convert_done.wait(60.0):
            logger.info(
                "docling 仍在转换 %s（已 %.0f 分钟，图片书 10-20 分钟属正常）",
                os.path.basename(path), (time.monotonic() - t0) / 60.0,
            )

    threading.Thread(target=_heartbeat, name="docling-heartbeat", daemon=True).start()

    try:
        # OCR scale 动态策略仅在开启时注入/复位（默认关闭时 scale 已在 converter
        # 构建时固化，无需、也不应每批 set/reset——避免多实例并发时全局变量互相
        # 踩踏）。动态 + 并行池不兼容，见 extract_markdown 注释。
        if _ocr_scale_dynamic:
            _set_scale_policy(scale_policy)
        try:
            result = converter.convert(path)
            doc = result.document
            markdown = doc.export_to_markdown()
        finally:
            if _ocr_scale_dynamic:
                _set_scale_policy(None)
        # 及时释放大对象引用，避免整棵文档树在切片前一直常驻。
        del result, doc
        if not markdown or not markdown.strip():
            logger.warning("docling extracted empty markdown from %s", os.path.basename(path))
            return None
        return markdown
    except Exception as e:
        # Never let a docling failure abort ingestion — caller falls back.
        logger.warning("docling conversion failed for %s: %s", os.path.basename(path), e)
        return None
    finally:
        _convert_done.set()
        # MPS 缓存护栏（#131）：torch 的 MPS caching allocator 只借不还，
        # 逐页激活碎片会在长书转换间累积（#116 的 36GB 根因）。每次转换
        # 结束立刻把空闲缓冲归还系统——分批后每批都清，峰值更平坦。
        try:
            import torch
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass


def extract_markdown(filepath: str) -> Optional[str]:
    """Convert a document to Markdown using docling.

    Returns the Markdown text, or ``None`` if docling is unavailable or the
    format is not handled by docling (callers should fall back to the
    per-format parsers).

    大 PDF 分批：仅当文件大小超过 ``_PDF_BATCH_BYTES`` (100MB) 时，先用 PyMuPDF
    按 ``_PDF_BATCH_PAGES`` (100 页) 切临时小文件、逐批 convert 再拼接——单批
    内存上限锁死在 100 页，避免整文件转换时整本文档树常驻内存把大 PDF 一次性
    撑爆；小文件（<= 100MB）整文件直接载入转换，更快且内存可控。
    """
    if not is_docling_available():
        return None

    path = Path(filepath)
    suffix = path.suffix.lower()
    if suffix not in _DOCLING_EXTENSIONS:
        return None

    # 图片型 PDF（扫描件/满页背景图的电子书）**不再拦截**：设计上所有 PDF
    # 统一走 docling（内部 OCR 优先 Apple Vision/ocrmac，输出版面良好的
    # Markdown）；PyMuPDF+RapidOCR 只是 docling 不可用/转换失败时的兜底。
    # 旧的「扫描件/图片重 → 转投 PyMuPDF」拦截（#116）已撤销——MPS 加速器、
    # 无超时、无内存约束等根因都已修复：accelerator 强制 CPU、document_timeout
    # 硬超时、MLX/缓存上限，且现在大 PDF 还按页分批。

    converter = _acquire_converter()
    if converter is None:
        return None

    # OCR scale 动态策略（#132，可配置）：仅 rag_ocr_scale_dynamic 开启时
    # convert 前预分析 PDF（文本层/原生 DPI/表格启发式），表格页逐页渲染
    # 提到 rag_ocr_scale_table，其余用 rag_ocr_scale_base。默认关闭——此处
    # 直接跳过，convert 前零开销（scale 已在 converter 构建时固化）。
    # ⚠️ 并行池下动态策略是**全局**的（模块级变量）：多实例并发 convert 会
    # 共享同一份策略，不同 PDF 的差异化 scale 会相互覆盖——动态 + 并行池不
    # 兼容，需要差异化表格 scale 时请把 _DOCLING_POOL_SIZE 调成 1。
    _scale_policy = (_analyze_ocr_scale(path)
                     if suffix == ".pdf" and _ocr_scale_dynamic else None)

    # 大 PDF 分批：仅当文件 > 50MB 才拆批（按 50 页/批），否则整文件直接
    # 转换——小文件整载入更快且内存可控。_split_pdf_into_batches 内部对页数
    # <= 批次的也会返回空（等于整文件单次转换），不影响结论。
    need_batch = suffix == ".pdf"
    if need_batch:
        try:
            need_batch = path.stat().st_size > _PDF_BATCH_BYTES
        except OSError:
            need_batch = False
    batches = (_split_pdf_into_batches(str(path), _PDF_BATCH_PAGES)
               if need_batch else [])
    try:
        if batches:
            parts: List[str] = []
            try:
                for bp in batches:
                    md = _convert_locked(converter, bp, _scale_policy)
                    if md:
                        parts.append(md)
            finally:
                for bp in batches:
                    try:
                        os.unlink(bp)
                    except OSError:
                        pass
            markdown = "\n\n".join(p for p in parts if p.strip())
            if not markdown:
                logger.warning("docling 分批转换产出空 Markdown: %s", path.name)
                return None
            return markdown

        return _convert_locked(converter, str(path), _scale_policy)
    finally:
        _release_converter_instance(converter)


