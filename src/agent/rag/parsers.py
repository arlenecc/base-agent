"""Document parsers for Word, Excel, PPT, PDF, EPUB, MOBI, AZW3 and more.

Each parser accepts a file path and returns extracted text. Lazy imports keep
optional dependencies optional — only the formats you actually use need to be
installed.
"""
from __future__ import annotations

import gc
import logging
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 全局 OCR 引擎（单例）+ 并发信号量
# ---------------------------------------------------------------------------
# RapidOCR ONNX Runtime 单实例约 500MB 内存。如果每个 worker 各加载一份，
# 4 worker 同时遇到图片 PDF 时会占用 ~2GB 仅仅用于 OCR 模型。
# 改为全局单例：所有 worker 共享同一个引擎实例，通过信号量串行执行
# （RapidOCR 可能非线程安全，串行调用 engine(img) 避免竞态）。
_OCR_SEMAPHORE = threading.Semaphore(1)
# 等待 OCR 信号量的上限。单页 OCR 正常远小于此；只有卡死才会走到超时，
# 此时跳过该页而不是让所有 worker 永久挂起。
_OCR_ACQUIRE_TIMEOUT = 300.0
_OCR_ENGINE = None  # 全局单例；None 表示尚未尝试加载
_OCR_ENGINE_LOCK = threading.Lock()
# 初始化失败后的冷却截止时间（monotonic）。已安装但加载失败（模型下载
# 不了 / ONNX 会话建不起来）时，负缓存避免每个图片型 PDF 都重试一次
# 完整加载（每次都要跑一遍可能耗时数分钟的下载尝试）。
_OCR_RETRY_AFTER = 0.0
_OCR_RETRY_COOLDOWN = 60.0

# OCR 引擎周期性重建页数。CoreML/ORT 的原生缓存按输入形状保留（实测真实
# 扫描页 ~35MB/页，#119：400 页扫描书滞留 ~14GB，进程退出前不释放，最终
# 被系统压缩到 17GB 直至 50GB 崩溃）。每 N 页强制重建引擎冲刷全部原生
# 缓存：#130 系统扫描（N=10..100，单线程真实 200dpi 页）证实热重建仅
# 1-2s/次、峰值足迹在各 N 下持平 ~5.2GB（压缩器吸收缓存增长），N 对速度
# 影响仅 1-3%/400 页书。取 50：单代水位 ~1.8GB（#123 后单代存活，4 worker
# 聚合约同量级），重建开销摊薄到 ~0.5%。合成小图触发不了该积累，只有真实
# 200dpi 大页 + 密集文字框才会——不要因为「合成实验没复现」而移除。
_OCR_RECYCLE_PAGES = 50

# OCR 内存护栏：CoreML EP 的原生缓存按**输入形状**保留，实测真实扫描书
# ~67MB/页（182 页扫描件单文件即推到 4.99GB 峰值），叠加常驻的本地 LLM
# （MLX 统一内存不可换出）会在 16GB 机器上触发系统级内存压力（整机卡死/
# 崩溃）。周期重建只能冲刷整代，挡不住单代内的持续爬升——400 页的书在一
# 代内就能吃掉十几 GB。护栏按页监控 RSS 增长：超预算立即冲刷引擎；冲刷
# 后仍超则判定该 EP 的缓存不可控，永久降级到 CPU EP。实测同一本书 CPU EP
# 反而更快（1.06 vs 1.73 s/页）且输出一字不差，降级不损失质量。
# 预算 1.2GB + 每 5 页一查：CoreML 实测 ~67MB/页，5 页的检测延迟约 0.33GB，
# 即触发时实际增长 ≈1.5GB。注意 CoreML 引擎自身的常驻基线就有 ~1.7GB
# （CPU EP 仅 ~0.3GB），压低预算才能把 CoreML 场景的峰值从 5.0GB 拉下来；
# 代价只是更频繁的热重建（1-2s/次，182 页书约 10 次 ≈ 15s，占 8%）。
_OCR_RSS_BUDGET_GB = 1.2    # 单个文件 OCR 期间允许的新增 RSS
_OCR_RSS_CHECK_PAGES = 5    # 每 N 个 OCR 页检查一次
_OCR_FORCE_CPU = False      # 降级标志：置位后新建的引擎一律走 CPU EP

# 进 OCR 的位图长边硬上限（与 ocr_engine._MAX_LONG_SIDE 同源）。渲染侧必须
# 自己兜一遍：长边归一化那一步在 preprocess_for_ocr 里，依赖 cv2/numpy——
# 预处理被关闭或导入失败时它会被整个跳过，位图原样进引擎（大开本扫描件
# 实测可到 2399×3725，8.9MP）。这里用 fitz 自身的缩放重渲染，使「所有进
# OCR 的图长边 ≤ 上限」与预处理是否可用无关。
try:
    from .ocr_engine import _MAX_LONG_SIDE as _PIXMAP_MAX_LONG_SIDE
except Exception:  # pragma: no cover - ocr_engine 不可用（缺 cv2/numpy 等）
    _PIXMAP_MAX_LONG_SIDE = 1500

# AgentConfig（可选注入）。未提供时按「全部开启」处理 —— config 的默认值
# 本身就是 True，不注入不会改变行为；注入后 UI 的开关才生效。
_OCR_CONFIG = None


def _current_rss_gb() -> Optional[float]:
    """当前进程 RSS（GB）。psutil/resource 都不可用时返回 None（护栏静默跳过）。"""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        pass
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS 单位是 bytes，Linux 是 KB：按量级区分后再统一成 GB。
        return (ru / 1e9) if ru > 1e7 else (ru * 1024.0 / 1e9)
    except Exception:
        return None


def _force_cpu_ep() -> None:
    """永久降级为 CPU EP（CoreML 原生缓存不可控时调用一次）。"""
    global _OCR_FORCE_CPU
    if not _OCR_FORCE_CPU:
        _OCR_FORCE_CPU = True
        logger.warning(
            "OCR: CoreML EP 原生缓存增长超预算，永久降级为 CPU EP"
            "（实测真实扫描件上 CPU EP 更快且输出一致）"
        )


def _ocr_coreml_requested() -> bool:
    """当前配置是否请求了 CoreML EP（用于判断冲刷后该不该降级）。

    兜底与 AgentConfig.rag_ocr_coreml 默认值（False）一致——兜底 True 会让
    未注入配置的调用路径意外启用 CoreML 相关行为（实测比 CPU 慢 2 倍）。
    """
    return bool(getattr(_OCR_CONFIG, "rag_ocr_coreml", False)) and _is_apple_silicon()


def set_ocr_config(config) -> None:
    """让解析层感知 AgentConfig（决定是否启用 CoreML EP / 预处理）。

    引擎是进程级单例且可能已被创建，所以这里只影响**后续**创建的引擎；
    调用方若需要立刻生效，应先 :func:`release_ocr_engine`。
    """
    global _OCR_CONFIG
    _OCR_CONFIG = config

# ---------------------------------------------------------------------------
# Format → extension mapping
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".docx",
    ".xlsx",
    ".pptx",
    ".pdf",
    ".epub",
    ".mobi", ".azw3", ".azw",
    ".txt", ".md", ".markdown", ".rst",
    ".csv", ".tsv",
    ".html", ".htm",
    ".tex", ".latex",
}
# 旧版二进制格式（.doc/.xls/.ppt）不在支持列表：python-docx/openpyxl/
# python-pptx 与 docling 均只支持 OOXML，解析必然失败（extract_text 对它们
# 给出明确的「另存为」提示）。保留在支持列表里只会让用户误以为可用。

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_text(filepath: str) -> str:
    """Extract text from a document. Returns the full text as a string."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    suffix = path.suffix.lower()

    # 文件类型→解析器映射（用于日志）
    _PARSER_LABELS = {
        ".docx": "Word 文档", ".doc": "Word 文档",
        ".xlsx": "Excel 表格", ".xls": "Excel 表格",
        ".pptx": "PPT 幻灯片", ".ppt": "PPT 幻灯片",
        ".pdf": "PDF 文档",
        ".epub": "EPUB 电子书",
        ".mobi": "MOBI 电子书", ".azw3": "AZW3 电子书", ".azw": "AZW 电子书",
        ".txt": "纯文本", ".md": "Markdown", ".markdown": "Markdown", ".rst": "reStructuredText",
        ".csv": "CSV 表格", ".tsv": "TSV 表格",
        ".html": "HTML 网页", ".htm": "HTML 网页",
    }
    label = _PARSER_LABELS.get(suffix, suffix)
    logger.debug("    解析文件类型: %s (%s)", path.name, label)

    if suffix in (".docx", ".doc"):
        if suffix == ".doc":
            # python-docx 只支持 OOXML（.docx）；旧版 OLE2 二进制 .doc 传进来
            # 100% 抛异常（docling 同样不支持 legacy 格式），之前表现为一条
            # 难懂的库报错。直接给出可行动的提示。
            raise ValueError(
                "旧版 .doc 二进制格式不受支持（python-docx/docling 均只支持 .docx），"
                "请先用 Word/WPS 另存为 .docx 后重新同步"
            )
        return _extract_docx(path)
    elif suffix in (".xlsx", ".xls"):
        if suffix == ".xls":
            # openpyxl 明确不支持旧版 .xls（BIFF），必然抛
            # "does not support the old .xls file format"。
            raise ValueError(
                "旧版 .xls 二进制格式不受支持（openpyxl 只支持 .xlsx），"
                "请先用 Excel/WPS 另存为 .xlsx 后重新同步"
            )
        return _extract_xlsx(path)
    elif suffix in (".pptx", ".ppt"):
        if suffix == ".ppt":
            # python-pptx 只支持 OOXML（.pptx），同上。
            raise ValueError(
                "旧版 .ppt 二进制格式不受支持（python-pptx 只支持 .pptx），"
                "请先用 PowerPoint/WPS 另存为 .pptx 后重新同步"
            )
        return _extract_pptx(path)
    elif suffix == ".pdf":
        return _extract_pdf(path)
    elif suffix == ".epub":
        return _extract_epub(path)
    elif suffix in (".mobi", ".azw3", ".azw"):
        return _extract_mobi(path)
    elif suffix in (".txt", ".md", ".markdown", ".rst"):
        return _extract_text(path)
    elif suffix in (".csv", ".tsv"):
        return _extract_csv(path)
    elif suffix in (".html", ".htm"):
        return _extract_html(path)
    else:
        raise ValueError(f"Unsupported format: {suffix}")


# ---------------------------------------------------------------------------
# Individual parsers
# ---------------------------------------------------------------------------


def _extract_docx(path: Path) -> str:
    try:
        from docx import Document
    except ImportError:
        raise ImportError("python-docx is required for Word files. pip install python-docx")

    # python-docx/openpyxl/python-pptx 都会持有打开的文件句柄（openpyxl
    # 非 read_only 时还留着 zip 句柄），只靠 GC 回收在 4 worker × 数千文件
    # 的同步里会累积句柄（Windows 上还会锁住文件，导致后续无法删除/覆盖）。
    doc = Document(str(path))
    try:
        parts: List[str] = []
        for para in doc.paragraphs:
            if para.text.strip():
                parts.append(para.text)
        # Also extract tables
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                parts.append(" | ".join(cells))
        return "\n\n".join(parts)
    finally:
        try:
            doc.part.package.stream.close()
        except Exception:
            pass


def _extract_xlsx(path: Path) -> str:
    try:
        import openpyxl
    except ImportError:
        raise ImportError("openpyxl is required for Excel files. pip install openpyxl")

    wb = openpyxl.load_workbook(str(path), data_only=True)
    try:
        parts: List[str] = []
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            parts.append(f"# Sheet: {sheet_name}")
            for row in ws.iter_rows(values_only=True):
                cells = [str(c) if c is not None else "" for c in row]
                if any(c.strip() for c in cells):
                    parts.append(" | ".join(cells))
        return "\n".join(parts)
    finally:
        # 关闭底层 zip 句柄（非 read_only 模式会一直持有）
        try:
            wb.close()
        except Exception:
            pass


def _extract_pptx(path: Path) -> str:
    try:
        from pptx import Presentation
    except ImportError:
        raise ImportError("python-pptx is required for PPT files. pip install python-pptx")

    prs = Presentation(str(path))
    try:
        parts: List[str] = []
        for i, slide in enumerate(prs.slides, 1):
            slide_parts = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        t = para.text.strip()
                        if t:
                            slide_parts.append(t)
                if shape.has_table:
                    table = shape.table
                    for row in table.rows:
                        cells = [cell.text.strip() for cell in row.cells]
                        slide_parts.append(" | ".join(cells))
            if slide_parts:
                parts.append(f"## Slide {i}\n\n" + "\n".join(slide_parts))
        return "\n\n".join(parts)
    finally:
        try:
            prs.part.package.stream.close()
        except Exception:
            pass


def _pixmap_is_blank(pix) -> bool:
    """检测渲染后的整页位图是否为纯色/近纯色（无有效内容）。

    均匀采样至多 8192 字节取 min/max：极差 ≤ 12 视为近纯色。覆盖两类
    场景：页面上只有纯色占位图/装饰条；扫描件里的空白纸页（整页白纸
    带轻微压缩噪点）。这些页送 OCR 必然返回 "text detection result is
    empty"，每页白白消耗 2-5 秒。极差阈值 12 足以容纳 JPEG/扫描噪点，
    而任何含文字的页都有大量暗像素，极差远超此值。
    """
    samples = pix.samples
    if not samples:
        return True
    step = max(1, len(samples) // 8192)
    data = samples[::step]
    return (max(data) - min(data)) <= 12


def _extract_pdf(path: Path) -> str:
    """Extract text from PDF. Auto-detects text vs image-based pages.

    Strategy:
    1. Text-based pages: extract text directly via PyMuPDF get_text("text").
    2. Image-based pages (no text layer, has embedded images): OCR via RapidOCR.
    3. Mixed PDFs: text pages extracted directly, image pages OCR'd.

    If RapidOCR is not installed and the PDF contains image-based pages,
    a warning is logged and the image pages are skipped — but the text
    pages are still extracted.
    """
    path = Path(path)  # 兼容 str 和 Path 两种入参
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF is required for PDF files. pip install PyMuPDF")

    doc = fitz.open(str(path))
    try:
        # 加密 PDF：没有密码就取不到任何文本，旧行为是整份文件静默产出空
        # 内容并计入「提取失败」。用空密码尝试解锁，能解就继续正常解析。
        if doc.needs_pass:
            if not doc.authenticate(""):
                raise ValueError("PDF 已加密且无法用空密码解锁")

        total_pages = len(doc)
        parts: List[str] = []
        ocr_unavailable = False  # OCR 引擎加载失败标记，避免每页重复尝试
        # 汇总日志用：记录「本本书 OCR 是否可用」。循环内每页重新绑定
        # （页末即释放，不跨页持有旧引擎——见 _OCR_RECYCLE_PAGES 注释），
        # 纯文本书不进 OCR 分支时保持 None。
        ocr_engine = None
        # 内存护栏基线：到第一个真正要 OCR 的页才取值，避免把本文件之外的
        # 常驻开销（其它 worker 的文档、解析缓存）算进 OCR 的预算里。
        ocr_rss_base: Optional[float] = None

        # OCR 图像预处理（自适应 DPI + 尺寸归一化 + CLAHE + 锐化）：默认开启，
        # 可由 AgentConfig.rag_ocr_preprocess 关闭。放在循环外只解析一次，
        # 且导入失败（缺 cv2/numpy）时降级为原始的固定 200dpi + 裸位图。
        def _fallback_dpi(_page) -> int:
            # 与 ocr_engine._MAX_RENDER_DPI 一致：预处理缺席时没有长边反推，
            # 但仍不该渲染出超过上限 dpi 的位图（渲染侧长边兜底会再缩一次）。
            return 150

        render_dpi_for_page = _fallback_dpi
        decode_image = None
        preprocess_for_ocr = None
        preprocess_enabled = bool(getattr(_OCR_CONFIG, "rag_ocr_preprocess", True))
        # 二值化：扫描件上是纯收益（更快、字符数一致），但有阴影/渐变/彩色
        # 插图的页面会丢信息 → 交给配置决定。
        binarize = bool(getattr(_OCR_CONFIG, "rag_ocr_binarize", True))
        if preprocess_enabled:
            try:
                from .ocr_engine import (
                    decode_image as _decode,
                    preprocess_for_ocr as _preprocess,
                    render_dpi_for_page as _dpi,
                )
                decode_image, preprocess_for_ocr, render_dpi_for_page = (
                    _decode, _preprocess, _dpi,
                )
            except Exception as e:  # pragma: no cover - 缺少 cv2/numpy
                logger.debug("OCR 预处理不可用，回退原始位图: %s", e)
                preprocess_enabled = False

        # 统计各类型页面数量，用于最终日志
        text_page_count = 0
        image_page_count = 0
        ocr_page_count = 0
        ocr_skipped_count = 0
        blank_page_count = 0

        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text("text").strip()

            if text:
                # 文字版页面：直接提取文本，不需要 OCR
                parts.append(text)
                text_page_count += 1
                continue

            # 无文字层 → 可能是图片版或空白页。检查是否含嵌入图片。
            images = page.get_images(full=True)
            if not images:
                # 既无文字也无图片 → 空白页，跳过
                blank_page_count += 1
                continue

            # 图片版页面 → 需要 OCR
            image_page_count += 1

            if ocr_unavailable:
                ocr_skipped_count += 1
                continue

            # 无超时的 acquire 在这里是个隐患：单页 OCR 卡死（ONNX 死锁 /
            # 病态大图）时，所有 ingest worker 连同它们的 daemon 线程一起永久
            # 挂起，用户侧表现为「同步进度再也不动且没有任何报错」。
            # 加一个宽松的超时，拿不到就跳过该页并计数。
            if not _OCR_SEMAPHORE.acquire(timeout=_OCR_ACQUIRE_TIMEOUT):
                ocr_skipped_count += 1
                logger.warning(
                    "RAG: OCR 信号量等待超过 %ss，跳过第 %d 页 (%s)",
                    _OCR_ACQUIRE_TIMEOUT, page_num + 1, path.name,
                )
                continue
            try:
                # 每页从进程级单例取引擎，页末即释放局部引用（#123）。
                # 旧实现把引擎缓存在 worker 的循环级局部变量里：周期回收
                # （release + 全局置 None）只影响触发回收的 worker，其它
                # worker 仍抱着旧引擎继续用——最多 4 代引擎的原生缓存并存
                # （每代 ~1-2GB），回收形同虚设，实测内存仍涨到 16GB。
                # 每页取用后，任意时刻至多一代引擎存活，回收立即全局生效。
                ocr_engine = _get_ocr_engine()
                if ocr_engine is None:
                    ocr_unavailable = True
                    ocr_skipped_count += 1
                    continue
                if ocr_rss_base is None:
                    # 引擎已加载后再取基线：OCR 模型自身的占用不算进预算，
                    # 之后每页新增的原生缓存才算。
                    ocr_rss_base = _current_rss_gb()
                # 自适应 DPI：页面物理尺寸差异很大（A4 / letter / 裁剪的小图），
                # 固定 200dpi 会让检测器看到完全不同的字形大小。先按页面尺寸
                # 解出「短边落在目标像素」的 DPI，再做归一化。
                dpi = render_dpi_for_page(page)
                pix = page.get_pixmap(dpi=dpi)
                # 渲染侧长边兜底（见 _PIXMAP_MAX_LONG_SIDE 注释）：预处理可用
                # 时下面的 preprocess_for_ocr 也会缩到同一上限，这里重渲染只
                # 在它缺席时真正生效，代价是一次额外的缩放渲染。
                _long_side = max(pix.width, pix.height)
                if _long_side > _PIXMAP_MAX_LONG_SIDE:
                    _zoom = (dpi / 72.0) * (_PIXMAP_MAX_LONG_SIDE / float(_long_side))
                    pix = None
                    pix = page.get_pixmap(matrix=fitz.Matrix(_zoom, _zoom))
                try:
                    if _pixmap_is_blank(pix):
                        # 渲染后确认的纯色/近纯色页（纯色占位图、扫描件的
                        # 空白纸页）：OCR 检测必然为空，直接跳过，省下每页
                        # 2-5 秒的 OCR（真实日志中单文档可浪费 1 分钟以上）。
                        blank_page_count += 1
                        continue
                    img_bytes = pix.tobytes("png")
                finally:
                    # Pixmap 持有整页位图（数 MB），显式释放避免多页循环时
                    # 峰值内存持续累积。
                    pix = None

                image_for_ocr = img_bytes
                if preprocess_enabled:
                    decoded = decode_image(img_bytes)
                    if decoded is not None:
                        image_for_ocr = preprocess_for_ocr(decoded, binarize=binarize)
                    # 解码/预处理失败 → 把原始 PNG bytes 交给引擎（旧路径）

                ocr_text = _ocr_image_cached(ocr_engine, image_for_ocr,
                                             page_num + 1, path.name)
            finally:
                _OCR_SEMAPHORE.release()

            if ocr_text:
                parts.append(ocr_text)
                ocr_page_count += 1
            else:
                # OCR 返回空结果（可能是图片质量太差或引擎异常）
                ocr_skipped_count += 1

            # ---- 内存护栏（依据见 _OCR_RSS_BUDGET_GB 注释）----
            # 只在真的超预算时才冲刷：小文档/无扫描页的正常场景 RSS 几乎不
            # 动，不会白白付出引擎重建（热重建 1-2s）的代价。
            if (ocr_rss_base is not None
                    and (ocr_page_count + ocr_skipped_count) % _OCR_RSS_CHECK_PAGES == 0):
                _rss_now = _current_rss_gb()
                if _rss_now is not None and _rss_now - ocr_rss_base > _OCR_RSS_BUDGET_GB:
                    release_ocr_engine()
                    logger.warning(
                        "PDF '%s': OCR 原生缓存 +%.2fGB 超预算 %.1fGB，已冲刷引擎",
                        path.name, _rss_now - ocr_rss_base, _OCR_RSS_BUDGET_GB,
                    )
                    if not _OCR_FORCE_CPU and _ocr_coreml_requested():
                        _force_cpu_ep()
                    # 冲刷后重取基线：新一代引擎（可能已降级到 CPU EP）的常驻
                    # 水位与上一代不同，沿用旧基线会让 growth 虚高，导致后面
                    # 每 5 页就白冲刷一次。
                    _rss_after = _current_rss_gb()
                    if _rss_after is not None:
                        ocr_rss_base = _rss_after

            # 周期性重建引擎：冲刷 CoreML/ORT 按形状保留的原生缓存（见
            # _OCR_RECYCLE_PAGES 注释）。全局置 None 后，所有 worker 的
            # 下一页都会重新取用/重建——旧引擎随各 worker 页末的引用释放
            # 而销毁，原生缓存真正被冲刷。
            # **只在 CoreML 模式下执行**：CPU EP 的 ORT arena 增量实测为
            # 0（2026-10 对照：3 页推理后内存增量 0.00GB），重建纯属代价——
            # 每次新建 ~0.5-1GB 引擎，4 worker 并发交叉时新旧两代共存，
            # 实测把 RSS 顶到 6.36GB + swap 26.6GB（整机卡死）。CoreML 的
            # 按形状缓存问题不存在于 CPU EP。
            if ((ocr_page_count + ocr_skipped_count) % _OCR_RECYCLE_PAGES == 0
                    and _ocr_coreml_requested() and not _OCR_FORCE_CPU):
                release_ocr_engine()
                logger.info("PDF '%s': OCR 引擎已周期重建（每 %d 页，冲刷原生缓存）",
                            path.name, _OCR_RECYCLE_PAGES)
    finally:
        # 必须放在 finally：中途抛异常（加密 PDF、损坏文件、OCR 崩溃）时
        # 旧实现直接跳过 doc.close()，泄漏文件句柄与整份文档的解码缓存。
        doc.close()
        # MuPDF 的 store 会缓存解码后的嵌入图（每张满页扫描图解码后
        # ~10MB+，数百页累计可观且进程内常驻）；全书解析完立即收缩并
        # 回收 Python 侧临时对象。store 是进程级全局的，收缩会影响其它
        # worker 正在打开的文档——只是缓存失效（重解码），不影响正确性。
        try:
            fitz.TOOLS.store_shrink(100)
        except Exception:
            pass
        gc.collect()

    # 汇总日志：清晰报告各类型页面的处理情况
    detail_parts = [f"text={text_page_count}"]
    if image_page_count > 0:
        detail_parts.append(f"image={image_page_count}(OCR={ocr_page_count})")
    if ocr_skipped_count > 0:
        detail_parts.append(f"OCR-skipped={ocr_skipped_count}")
    if blank_page_count > 0:
        detail_parts.append(f"blank={blank_page_count}")

    # ---- 全图片版 PDF 且 OCR 不可用 → 抛出明确错误 ----
    if not parts and image_page_count > 0 and ocr_engine is None:
        msg = (
            f"PDF '{path.name}' 是图片版（{total_pages} 页全部为扫描图片），"
            f"但 RapidOCR 未安装，无法进行 OCR 识别。"
            f"请安装: pip install rapidocr-onnxruntime"
        )
        logger.error(msg)
        raise RuntimeError(msg)

    if not parts:
        logger.warning(
            "PDF '%s': no text extracted from %d pages "
            "(text=%d, image=%d, blank=%d)",
            path.name, total_pages, text_page_count, image_page_count, blank_page_count,
        )
        return ""

    if ocr_skipped_count > 0 and ocr_engine is None:
        # 混合 PDF：部分图片页因 OCR 不可用被跳过
        logger.warning(
            "PDF '%s': %d/%d pages are image-based but RapidOCR is not installed. "
            "Image pages were SKIPPED — only text pages were extracted (%d chars). "
            "Install with: pip install rapidocr-onnxruntime",
            path.name, ocr_skipped_count, total_pages, sum(len(p) for p in parts),
        )
    elif image_page_count > 0:
        logger.info(
            "PDF '%s': %d pages processed (%s)",
            path.name, total_pages, ", ".join(detail_parts),
        )
    else:
        logger.info(
            "PDF '%s': %d pages, all text-based (%d chars extracted)",
            path.name, total_pages, sum(len(p) for p in parts),
        )

    return "\n\n".join(parts)


def _get_ocr_engine():
    """Lazy-load RapidOCR engine as a global singleton.

    Returns None if not installed. All workers share the same engine instance
    (~500MB) — without this, 4 workers loading 4 instances would OOM. The
    _OCR_SEMAPHORE serializes actual OCR calls so the shared engine is only
    invoked by one thread at a time.

    引擎选择：优先 rapidocr 3.x（支持 CoreML EP、模型目录与 docling 共享、
    懒加载省内存），未安装再回退旧版 rapidocr-onnxruntime 1.x（经
    :mod:`agent.rag.ocr_engine` 注入 CoreML EP）。Apple Silicon（M 系列）上
    3.x 启用 onnxruntime 的 CoreML EP（ANE+GPU 推理）；CoreML 初始化失败
    自动回退 CPU EP。两种版本都会预热一次（空图）——模型本来马上就会被用到，
    预热同时消除首页 OCR 的加载延迟并探测实际生效的 EP。
    """
    global _OCR_ENGINE, _OCR_RETRY_AFTER
    if _OCR_ENGINE is not None:
        return _OCR_ENGINE
    with _OCR_ENGINE_LOCK:
        if _OCR_ENGINE is not None:
            return _OCR_ENGINE
        # 冷却期内不重试：持续失败的加载（模型下载不了）不该每个文件
        # 都重新触发一遍。到期后放行一次，网络恢复后能自愈。
        if time.monotonic() < _OCR_RETRY_AFTER:
            return None
        try:
            _OCR_ENGINE = _build_ocr_engine()
            _warmup_ocr_engine(_OCR_ENGINE)
            logger.info("RapidOCR engine loaded (global singleton)")
            return _OCR_ENGINE
        except ImportError:
            logger.warning(
                "rapidocr is required for OCR on image-based PDFs. "
                "pip install rapidocr"
            )
            return None
        except Exception as e:
            logger.warning("Failed to init RapidOCR: %s", e)
            _OCR_RETRY_AFTER = time.monotonic() + _OCR_RETRY_COOLDOWN
            return None


def _build_ocr_engine():
    """按可用性构造 RapidOCR 引擎：3.x 优先，1.x 回退（均带 CoreML 加速）。

    ``_OCR_FORCE_CPU`` 置位后（内存护栏判定 CoreML 缓存不可控）一律走 CPU EP，
    无论配置里 ``rag_ocr_coreml`` 是什么——配置是「请求」，护栏是「否决」。
    """
    # getattr 的兜底必须与 AgentConfig.rag_ocr_coreml 的默认值（False）一致：
    # _OCR_CONFIG 未注入时（如直接调用 _extract_with_docling 的脚本路径）
    # 兜底 True 会把 CoreML 意外打开——实测 CoreML EP 比 CPU 慢约 2 倍
    # （3 页 4.95s vs 2.43s），见 AgentConfig.rag_ocr_coreml 注释。
    use_coreml = (bool(getattr(_OCR_CONFIG, "rag_ocr_coreml", False))
                  and not _OCR_FORCE_CPU)
    params = None
    try:
        from rapidocr import RapidOCR  # 3.x
        if use_coreml and _is_apple_silicon():
            params = {"EngineConfig.onnxruntime.use_coreml": True}
    except ImportError:
        from .ocr_engine import create_ocr_engine, coreml_reason
        if not use_coreml:
            logger.info("OCR: CoreML EP 按配置关闭")
        else:
            _, reason = coreml_reason()
            logger.info("OCR: CoreML EP 检测 — %s", reason)
        return create_ocr_engine(use_coreml=use_coreml)
    if params:
        try:
            return RapidOCR(params=params)
        except Exception as e:
            # CoreML 会话建不起来（不支持的算子 / MLProgram 缺失）→ 回退 CPU EP。
            logger.info(
                "RapidOCR CoreML EP init failed (%s); using CPU EP", type(e).__name__
            )
            return RapidOCR()
    return RapidOCR()


def _is_apple_silicon() -> bool:
    """当前是否为 Apple Silicon（M 系列）Mac。

    优先用 :mod:`agent.rag.ocr_engine` 的探测（``sysctl`` 读真实 CPU 型号，
    Rosetta 下 ``platform.machine()`` 会谎报 x86_64）；拿不到时退回简单的
    arm64 判断。Intel Mac 返回 False，继续用 CPU EP。
    """
    try:
        from .ocr_engine import is_apple_silicon as _detect
        return bool(_detect())
    except Exception:
        import platform
        return platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")


def _warmup_ocr_engine(engine) -> None:
    """用一张空白图预热：提前加载模型并探测实际生效的 EP（失败无感跳过）。"""
    try:
        import numpy as np
        engine(np.full((64, 64), 255, dtype=np.uint8))
        inner = getattr(getattr(engine, "text_det", None), "session", None)
        inner = getattr(inner, "session", inner)
        if inner is not None and hasattr(inner, "get_providers"):
            logger.info("RapidOCR engine loaded, EP: %s", inner.get_providers())
    except Exception:
        logger.debug("OCR 预热跳过（不影响使用）")


def release_ocr_engine() -> None:
    """Release the global OCR singleton to reclaim its ~500MB memory.

    The RapidOCR engine holds an ONNX Runtime inference session (~500MB of
    native heap) that Python GC cannot reclaim promptly.  GUI apps that OCR
    image-based PDFs would otherwise hold this memory until process exit.
    Call this from RAGEngine.close() after a sync finishes; the engine is
    lazily re-created on the next OCR-requiring ingestion.
    """
    global _OCR_ENGINE, _OCR_RETRY_AFTER
    with _OCR_ENGINE_LOCK:
        if _OCR_ENGINE is not None:
            # RapidOCR wraps an onnxruntime InferenceSession.  Drop the
            # reference so the session's C++ heap is freed on GC.
            _OCR_ENGINE = None
            # 同时清掉负缓存冷却：显式释放意味着调用方要求「可重新加载」，
            # 残留冷却会让下一次 ingest 在冷却期内直接跳过 OCR。
            _OCR_RETRY_AFTER = 0.0
            logger.info("RapidOCR engine released (memory reclaimed)")


# ocrmac 输出低于该字符数视为「疑似识别困难」——繁体竖排、艺术字体等
# Apple Vision 处理不好的版式往往只能零星识别几个字。正常正文页 ocrmac
# 输出一般在 200 字以上；空白/封面/扉页低于阈值时会多跑一次 RapidOCR
# （CPU 0.6-1.2s/页），可接受。
_OCRMAC_LOW_OUTPUT = 80
# RapidOCR 补跑结果需明显多于 ocrmac 才采用（避免把噪声当收益）。
_OCRMAC_FALLBACK_GAIN = 1.2


def _rapidocr_text(engine, image, page_num: int, filename: str) -> str:
    """纯 RapidOCR 识别（ocrmac 补跑 / 整体回退共用）。

    兼容两代 API 的返回：3.x 返回 RapidOCROutput（行文本在 .txts）；
    1.x 返回 (result, elapse)，result 是 [box, text, score] 列表。
    """
    try:
        out = engine(image)
        if isinstance(out, tuple):
            out = out[0]
        txts = getattr(out, "txts", None)
        if txts is not None:
            return "\n".join(t for t in txts if t and t.strip())
        if not out:
            return ""
        lines = []
        for item in out:
            text = item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else str(item)
            if text.strip():
                lines.append(text)
        return "\n".join(lines)
    except Exception as e:
        logger.warning("OCR failed on page %d of '%s': %s", page_num, filename, e)
        return ""


def _ocr_image_cached(engine, image, page_num: int, filename: str) -> str:
    """统一 OCR 入口：Apple Vision（ocrmac）优先，困难页用 RapidOCR 补充。

    ``image`` 可以是 PNG bytes（旧路径）或已预处理的 BGR ndarray。三段策略：

    1. ocrmac（Apple Vision）识别——中文横排印刷体质量最高、ANE 推理；
       引擎不可用或调用失败（返回 None）→ 该页整体走 RapidOCR。
    2. 输出过少（< ``_OCRMAC_LOW_OUTPUT``）→ 疑似繁体竖排等 Vision 不擅长
       的版式，**仅对该页**用 RapidOCR 重跑对比：结果明显更多
       （> ``_OCRMAC_FALLBACK_GAIN`` 倍）才采用。是单页级补充——后续页面
       仍然优先 ocrmac，不会因为某一页而整体切换引擎。
    3. RapidOCR 识别本体见 :func:`_rapidocr_text`。

    不做双引擎结果拼接：同一页两种引擎的行会大量重复/交错，模糊去重的
    复杂度和出错面远大于收益，二选一（取文本明显更多者）更稳。
    """
    from .ocr_engine import ocr_with_ocrmac
    ocr_text = ocr_with_ocrmac(image)
    if ocr_text is None:
        return _rapidocr_text(engine, image, page_num, filename)
    if len(ocr_text.strip()) >= _OCRMAC_LOW_OUTPUT:
        return ocr_text
    rapid_text = _rapidocr_text(engine, image, page_num, filename)
    if len(rapid_text) > len(ocr_text) * _OCRMAC_FALLBACK_GAIN:
        logger.info(
            "页 %d '%s': ocrmac 仅 %d 字（疑似竖排/特殊版式），"
            "RapidOCR 得 %d 字 → 采用 RapidOCR 结果",
            page_num, filename, len(ocr_text.strip()), len(rapid_text))
        return rapid_text
    return ocr_text


def _extract_epub(path: Path) -> str:
    try:
        from ebooklib import epub
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("ebooklib and beautifulsoup4 are required for EPUB. pip install ebooklib beautifulsoup4")

    book = epub.read_epub(str(path))
    parts: List[str] = []
    try:
        for item in book.get_items_of_type(9):  # ITEM_DOCUMENT = 9
            content = item.get_content().decode("utf-8", errors="replace")
            soup = BeautifulSoup(content, "html.parser")
            text = soup.get_text(separator="\n")
            if text.strip():
                parts.append(text)
    finally:
        # ebooklib 内部持有 zip 文件句柄。不显式关闭时，4 个 ingest worker ×
        # 数千本 EPUB 会让 fd 持续累积（Windows 上还会锁住文件导致无法删除/覆盖）。
        for closer in ("close", "_cleanup"):
            fn = getattr(book, closer, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
                break
    return "\n\n".join(parts)


def _extract_mobi(path: Path) -> str:
    """Extract text from MOBI/AZW3 files.

    Strategy: try to convert via calibre ebook-convert CLI first (most reliable),
    then fall back to a basic raw-text extraction.
    """
    # Try calibre's ebook-convert first
    import subprocess
    import shutil

    if shutil.which("ebook-convert"):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            txt_path = tmp.name
        try:
            result = subprocess.run(
                ["ebook-convert", str(path), txt_path],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0 and os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                    converted = f.read()
                # 只有非空才算转换成功：ebook-convert 对个别文件会
                # 返回 0 但产出空文件，旧实现直接 return 空串，静默
                # 跳过下面承诺的 raw 回退路径（文档变成零内容）。
                if converted.strip():
                    return converted
                logger.warning(
                    "ebook-convert produced empty output for '%s', "
                    "falling back to raw extraction", path.name,
                )
        except subprocess.TimeoutExpired:
            logger.warning("ebook-convert timed out for '%s'", path.name)
        except Exception as e:
            logger.warning("ebook-convert failed for '%s': %s", path.name, e)
        finally:
            try:
                os.unlink(txt_path)
            except Exception:
                pass

    # Fallback: basic raw extraction
    try:
        return _extract_mobi_raw(path)
    except Exception as e:
        logger.warning("Raw MOBI extraction failed for '%s': %s", path.name, e)
        return ""


def _extract_mobi_raw(path: Path) -> str:
    """Basic MOBI text extraction without external tools."""
    # 只处理前 N MB：MOBI 常有几十 MB，整份读入 + 整份 decode + findall
    # 物化会同时存在 3-4 份副本（latin-1 回退时再翻一倍）。
    _MOBI_MAX_BYTES = 8 * 1024 * 1024
    with open(path, "rb") as f:
        data = f.read(_MOBI_MAX_BYTES)

    # Try to find text content in the raw bytes
    text_parts: List[str] = []
    # Look for UTF-8 text segments
    try:
        decoded = data.decode("utf-8", errors="ignore")
        # finditer 边匹配边消费，避免把所有匹配块一次物化成列表
        text_parts.extend(
            m.group(0) for m in re.finditer(
                r'[\u4e00-\u9fff\u3000-\u303f\w\s.,!?;:()\[\]{}\'\"\-–—]{30,}', decoded)
        )
        del decoded
    except Exception:
        pass

    # Also try Latin-1
    # 注意 data 不能在 utf-8 分支里 del：utf-8 的 errors="ignore" 几乎不会抛
    # 异常，但 text_parts 为空时 latin-1 回退需要原始字节。旧实现在 utf-8
    # 分支 del data，回退触发 NameError 被 except 吞掉 → 非 utf-8 的 MOBI
    # 静默返回空。这里用显式置 None 释放引用，回退分支判空跳过。
    if not text_parts:
        if data is None:
            return "\n\n".join(text_parts)
        try:
            decoded = data.decode("latin-1", errors="ignore")
            text_parts.extend(
                m.group(0) for m in re.finditer(
                    r'[\w\s.,!?;:()\[\]{}\'\"\-–—]{30,}', decoded)
            )
        except Exception:
            pass

    return "\n\n".join(text_parts)


def _extract_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _extract_csv(path: Path) -> str:
    import csv
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel  # default: comma-delimited
        reader = csv.reader(f, dialect)
        # 单次流式消费：旧实现 rows=list(reader) → lines=[...] → join，
        # 三个全量副本同时存在（rows 与 lines 长期共存），200MB CSV 峰值 1GB+。
        parts: List[str] = []
        append = parts.append
        for row in reader:
            append(" | ".join(row))
    if not parts:
        return ""
    return "\n".join(parts)


def _extract_html(path: Path) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        # Fallback: basic regex stripping
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        content = re.sub(r'<[^>]+>', ' ', content)
        content = re.sub(r'\s+', ' ', content)
        return content.strip()

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    # Remove script/style tags
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n")


# ---------------------------------------------------------------------------
# Directory-level extraction
# ---------------------------------------------------------------------------


def extract_directory(
    dirpath: str,
    recursive: bool = True,
    extensions: Optional[set] = None,
    errors: Optional[List[str]] = None,
) -> List[tuple]:
    """Walk a directory and extract text from all supported files.

    Returns a list of (filepath, extracted_text) tuples.
    If *errors* is provided, extraction failure messages are appended to it.
    """
    return list(iter_directory(dirpath, recursive=recursive,
                               extensions=extensions, errors=errors))


def iter_directory(
    dirpath: str,
    recursive: bool = True,
    extensions: Optional[set] = None,
    errors: Optional[List[str]] = None,
) -> Iterator[Tuple[str, str]]:
    """流式版本的 extract_directory：逐文件 yield (filepath, text)。

    避免大知识库一次性把全部文件全文累积在内存（H1 隐患）。
    调用方消费完一个文件后即可释放该文件的全文字符串。
    """
    if extensions is None:
        extensions = SUPPORTED_EXTENSIONS

    root = Path(dirpath)
    iterator = root.rglob("*") if recursive else root.glob("*")
    for filepath in iterator:
        if not filepath.is_file():
            continue
        if filepath.suffix.lower() not in extensions:
            continue
        # Skip hidden files
        if filepath.name.startswith("."):
            continue
        try:
            text = extract_text(str(filepath))
            if text.strip():
                yield (str(filepath), text)
                logger.info("Extracted: %s (%d chars)", filepath.name, len(text))
            else:
                msg = f"{filepath.name}: extracted text is empty"
                logger.warning(msg)
                if errors is not None:
                    errors.append(msg)
        except Exception as e:
            msg = f"{filepath.name}: {e}"
            logger.warning("Failed to extract '%s': %s", filepath.name, e)
            if errors is not None:
                errors.append(msg)