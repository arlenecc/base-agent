"""OCR 识别测试：按页返回各引擎的识别结果，供 UI 左右对照。

与 RAG 入库链路（``parsers`` / ``ovis_ocr_parser``）共用同一批引擎、同一套
归一化与同一份配置，差别有三点：

1. 返回**逐页**结果（``List[PageResult]``），而不是拼好的整串——测试窗口要
   按页与原件对照，拼成一整串就没法比对了。
2. 引擎由调用方显式指定，不走 RAG 的「统一引擎回退链」——测试的目的正是
   比较单个引擎的效果，自动回退会把结果混成别人家的输出。
3. PDF 按页自动判别：文本型页用 **PyMuPDF4LLM** 直接转 Markdown（保留标题/
   列表/表格结构，优于纯文本层，且不必跑 OCR）；只有扫描型页才按选定引擎做
   OCR。PyMuPDF4LLM 不可用时自动回退到 PyMuPDF 纯文本层提取。

引擎标识：

- ``apple``  Apple Vision（ocrmac），仅 macOS
- ``ovis``   OvisOCR2 VLM（配置里的后端 / 模型路径）
- ``hybrid`` Vision + Ovis 表格（仅 PDF）：正文走 Vision，表格区域走 VLM
- ``rapid``  RapidOCR（本地轻量 OCR 引擎，受 CoreML / 预处理 / 二值化开关影响）
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

APPLE = "apple"
OVIS = "ovis"
HYBRID = "hybrid"
RAPID = "rapid"
ENGINE_IDS = (APPLE, OVIS, HYBRID, RAPID)

PDF_EXTS = {".pdf"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# 文本层判定阈值（与 ovis_ocr_parser._TEXT_LAYER_MIN_CHARS 保持一致）：
# 少于这么多字符的页面视为「没有可用文本层」，按扫描页处理。
_TEXT_LAYER_MIN_CHARS = 50


@dataclass
class PageResult:
    """单页识别结果。

    ``markdown`` 与 ``error`` 可以同时有值（如混合模式正文失败但表格成功），
    也可以同时为空（空白页）。
    """

    index: int = 0          # 0-based 页序
    kind: str = ""          # text | image | blank
    engine: str = ""        # 实际生效的引擎标识
    markdown: str = ""      # 识别结果（Markdown / HTML / 纯文本）
    error: str = ""         # 该页的失败/提示信息


# ---------------------------------------------------------------------------
# 可用性 / 文件探测
# ---------------------------------------------------------------------------

def is_supported(path: str) -> bool:
    """图片与 PDF 才支持识别测试。"""
    return os.path.splitext(path or "")[1].lower() in (PDF_EXTS | IMAGE_EXTS)


def is_pdf(path: str) -> bool:
    return os.path.splitext(path or "")[1].lower() in PDF_EXTS


def probe_pages(path: str) -> int:
    """文件页数（图片恒为 1，PDF 读取真实页数）。打开失败返回 0。"""
    if not is_pdf(path):
        return 1
    try:
        import fitz
    except Exception:
        return 0
    try:
        doc = fitz.open(path)
        try:
            return int(doc.page_count)
        finally:
            doc.close()
    except Exception as e:
        logger.debug("probe_pages 失败 %s: %s", path, e)
        return 0


def _rapidocr_installed() -> bool:
    try:
        import rapidocr  # noqa: F401  (3.x 统一包名)
        return True
    except ImportError:
        try:
            import rapidocr_onnxruntime  # noqa: F401  (1.x 旧包名)
            return True
        except ImportError:
            return False


def engine_status() -> Dict[str, str]:
    """各引擎的可用性：``engine_id -> 不可用原因``（空串 = 可用）。"""
    out: Dict[str, str] = {}
    try:
        from .ocr_engine import ocrmac_available
        out[APPLE] = "" if ocrmac_available() else \
            "Apple Vision 不可用（仅 macOS；需 pip install ocrmac）"
    except Exception as e:  # pragma: no cover - 导入失败即不可用
        out[APPLE] = f"Apple Vision 不可用：{type(e).__name__}: {e}"
    try:
        from .ovis_ocr_parser import is_ovis_available
        out[OVIS] = "" if is_ovis_available() else \
            "OvisOCR2 不可用（模型目录缺失或后端依赖未安装）"
    except Exception as e:  # pragma: no cover
        out[OVIS] = f"OvisOCR2 不可用：{type(e).__name__}: {e}"
    out[RAPID] = "" if _rapidocr_installed() else \
        "RapidOCR 未安装（pip install rapidocr）"
    # 混合模式：正文依赖 Vision、表格依赖 VLM，两者缺一就只能跑半边。
    if out[APPLE] or out[OVIS]:
        out[HYBRID] = "；".join(p for p in (out[APPLE], out[OVIS]) if p)
    else:
        out[HYBRID] = ""
    return out


# ---------------------------------------------------------------------------
# 图像归一化 / 引擎调用
# ---------------------------------------------------------------------------

def _pil_to_bgr(pil_img) -> Optional["object"]:
    """PIL → BGR ndarray（RapidOCR/ocrmac 都在 BGR 上工作）。"""
    try:
        import numpy as np
    except Exception:
        return None
    try:
        arr = np.asarray(pil_img.convert("RGB"))
        return arr[:, :, ::-1].copy()          # RGB → BGR
    except Exception as e:
        logger.debug("PIL→BGR 转换失败: %s", e)
        return None


def _normalize(img, preprocess: bool, binarize: bool):
    """统一的送入引擎前归一化：尺寸归一 + CLAHE + 锐化（+ 可选二值化）。

    与 ``parsers._extract_pdf`` 的扫描页路径完全同一策略（同一个
    ``preprocess_for_ocr``），测试结果因此与真实入库结果可比。
    """
    if img is None or not preprocess:
        return img
    try:
        from .ocr_engine import preprocess_for_ocr
        return preprocess_for_ocr(img, binarize=binarize)
    except Exception as e:  # pragma: no cover - 缺 cv2/numpy
        logger.debug("OCR 预处理不可用，使用原始位图: %s", e)
        return img


def _plain_engine_text(img, engine: str, page_no: int,
                       filename: str) -> Tuple[str, str]:
    """Apple Vision / RapidOCR 识别一张已归一化的图，返回 ``(text, error)``。"""
    if img is None:
        return "", "图像解码失败"
    if engine == APPLE:
        try:
            from .ocr_engine import ocr_with_ocrmac
        except Exception as e:  # pragma: no cover
            return "", f"ocrmac 导入失败：{type(e).__name__}: {e}"
        text = ocr_with_ocrmac(img)
        if text is None:
            return "", "Apple Vision 不可用或识别失败"
        return (text or "").strip(), ""

    if engine != RAPID:
        return "", f"未知引擎: {engine}"

    try:
        from .parsers import (
            _acquire_ocr_engine, _rapidocr_text, _return_ocr_engine,
        )
    except Exception as e:  # pragma: no cover
        return "", f"RapidOCR 不可用：{type(e).__name__}: {e}"
    engine_obj, gen = _acquire_ocr_engine()
    if engine_obj is None:
        return "", "RapidOCR 引擎不可用（未安装 rapidocr，或加载处在冷却期）"
    try:
        text = _rapidocr_text(engine_obj, img, page_no, filename)
    finally:
        # 必须归还：池只有 rag_ocr_pool_size 个实例，漏还会让后续页全部拿不到
        # 引擎（parsers 里同样是「每页借、页末还」）。
        _return_ocr_engine(engine_obj, gen)
    return (text or "").strip(), ""


# ---------------------------------------------------------------------------
# 图片文件
# ---------------------------------------------------------------------------

def _recognize_image_file(path: str, engine: str, preprocess: bool,
                          binarize: bool) -> PageResult:
    """单张图片：统一归一化后用指定引擎识别。"""
    res = PageResult(index=0, kind="image", engine=engine)
    if engine == HYBRID:
        res.error = "混合模式仅支持 PDF（依赖页面的表格区域检测）"
        return res
    try:
        from PIL import Image
        with Image.open(path) as pil:
            pil.load()
            pil_rgb = pil.convert("RGB")
    except Exception as e:
        res.error = f"图片打开失败：{type(e).__name__}: {e}"
        return res

    if engine == OVIS:
        # VLM 走自己的归一化（_save_normalized：最长边 1408），不做二值化——
        # 二值化会抹掉字形灰度层次，对 VLM 是净损失。
        try:
            from .ovis_ocr_parser import _vlm_ocr_pil
            md = _vlm_ocr_pil(pil_rgb)
        except Exception as e:
            res.error = f"OvisOCR2 识别失败：{type(e).__name__}: {e}"
            return res
        if not md:
            res.error = "OvisOCR2 未返回结果（模型不可用或输出为空）"
        else:
            res.markdown = md
        return res

    img = _pil_to_bgr(pil_rgb)
    img = _normalize(img, preprocess, binarize)
    text, err = _plain_engine_text(img, engine, 1, os.path.basename(path))
    res.markdown, res.error = text, err
    return res


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def _render_page_bgr(page):
    """PDF 页 → BGR ndarray（自适应 DPI，与入库链路同一算法）。"""
    try:
        from .ocr_engine import decode_image, render_dpi_for_page
        dpi = render_dpi_for_page(page)
    except Exception:
        dpi = 150
        decode_image = None
    pix = None
    try:
        pix = page.get_pixmap(dpi=dpi)
        png = pix.tobytes("png")
    finally:
        pix = None                      # 整页位图数 MB，尽早释放
    if decode_image is None:
        try:
            from .ocr_engine import decode_image as _dec
            decode_image = _dec
        except Exception:
            return None
    return decode_image(png)


def _table_markdowns(page) -> List[str]:
    """页面内表格区域 → VLM Markdown 列表（复用 ovis 的表格检测与渲染倍率）。"""
    try:
        import fitz
        from .ovis_ocr_parser import (
            _detect_table_rects, _pixmap_to_png, _vlm_ocr, _TABLE_SCALE,
        )
    except Exception as e:  # pragma: no cover
        logger.debug("表格 OCR 不可用: %s", e)
        return []
    rects = _detect_table_rects(page)
    if not rects:
        return []
    mds: List[str] = []
    for rect in rects:
        try:
            pix = page.get_pixmap(
                matrix=fitz.Matrix(_TABLE_SCALE, _TABLE_SCALE), clip=rect)
        except Exception:
            continue
        png = None
        try:
            png = _pixmap_to_png(pix)
        finally:
            pix = None
        if png is None:
            continue
        try:
            md = _vlm_ocr(png)
        finally:
            try:
                os.unlink(png)
            except OSError:
                pass
        if md:
            mds.append(md)
    return mds


def _text_page_markdown(doc, index: int) -> str:
    """文本型页用 PyMuPDF4LLM 直接转 Markdown（保留标题/列表/表格结构）。

    失败或该库不可用时回退到 PyMuPDF 的纯文本层提取——文本型 PDF 即使回退
    也拿得到全部文字，只是少了排版结构。
    """
    try:
        import pymupdf4llm
        md = pymupdf4llm.to_markdown(doc, pages=[index])
        if isinstance(md, str) and md.strip():
            return md.strip()
    except Exception as e:  # pragma: no cover - 库未装或单页转换异常
        logger.debug("PyMuPDF4LLM 文本页转换失败[%d]: %s", index, e)
    try:
        return (doc[index].get_text("text") or "").strip()
    except Exception:
        return ""


def _pdf_page_ovis(page) -> Tuple[str, str]:
    """OvisOCR2 扫描页：整页 VLM 转 Markdown。

    文本型页不进这里——主循环已用 PyMuPDF4LLM 转好结构化 Markdown，VLM 看图
    反而丢结构、且更慢。
    """
    from .ovis_ocr_parser import _ocr_image_page
    md = _ocr_image_page(page)
    return (md or ""), ("" if md else "OvisOCR2 未返回结果")


def _pdf_page_hybrid(page, preprocess: bool, binarize: bool, page_no: int,
                     filename: str) -> Tuple[str, str]:
    """混合模式（扫描页）：正文 Vision + 表格区域 OvisOCR2。

    文本型页不进这里（主循环已用 PyMuPDF4LLM 转好）；只有扫描页才需要 Vision
    看图 OCR。表格无论哪种页都交给 VLM——表格结构正是 VLM 相对 Vision 的优势。
    """
    img = _normalize(_render_page_bgr(page), preprocess, binarize)
    body, err = _plain_engine_text(img, APPLE, page_no, filename)
    table_mds = _table_markdowns(page)
    parts = [p for p in (body, "\n\n".join(table_mds)) if p]
    return "\n\n".join(parts), err


def _recognize_pdf(path: str, engine: str, preprocess: bool, binarize: bool,
                   on_page: Optional[Callable[[PageResult], None]],
                   progress: Optional[Callable[[int, int, str], None]],
                   cancelled: Optional[Callable[[], bool]]) -> List[PageResult]:
    import fitz

    name = os.path.basename(path)
    results: List[PageResult] = []
    doc = fitz.open(path)
    try:
        if doc.needs_pass and not doc.authenticate(""):
            raise ValueError("PDF 已加密且无法用空密码解锁")
        total = int(doc.page_count)
        for i in range(total):
            if cancelled is not None and cancelled():
                break
            page = doc[i]
            text_layer = (page.get_text("text") or "").strip()
            # 自动判别：有文本层 → 文本型；无文本层但有图 → 扫描型；都没有 → 空白
            has_text = len(text_layer) >= _TEXT_LAYER_MIN_CHARS
            kind = "text" if has_text else (
                "image" if page.get_images(full=True) else "blank")
            res = PageResult(index=i, kind=kind, engine=engine)
            if kind == "blank":
                res.error = "空白页（无文本层且无图像）"
            elif kind == "text":
                # 文本型 PDF：PyMuPDF4LLM 转 Markdown（结构优于纯文本层），不必
                # 跑任何 OCR 引擎；库不可用时自动回退 get_text。
                res.markdown = _text_page_markdown(doc, i)
            elif engine == OVIS:
                res.markdown, res.error = _pdf_page_ovis(page)
            elif engine == HYBRID:
                res.markdown, res.error = _pdf_page_hybrid(
                    page, preprocess, binarize, i + 1, name)
            else:  # apple / rapid 扫描页
                img = _normalize(_render_page_bgr(page), preprocess, binarize)
                res.markdown, res.error = _plain_engine_text(
                    img, engine, i + 1, name)
            results.append(res)
            if on_page is not None:
                try:
                    on_page(res)
                except Exception:  # pragma: no cover - 回调失败不影响识别
                    pass
            if progress is not None:
                try:
                    progress(len(results), total, f"第 {i + 1}/{total} 页")
                except Exception:  # pragma: no cover
                    pass
    finally:
        doc.close()
    return results


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------

def recognize_file(
    path: str,
    engine: str,
    *,
    preprocess: bool = True,
    binarize: bool = True,
    on_page: Optional[Callable[[PageResult], None]] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
) -> List[PageResult]:
    """用指定引擎识别文件，返回逐页结果。

    ``on_page`` 每识别完一页就回调一次（UI 用它增量显示，长 PDF 不必等到
    最后）；``cancelled`` 返回 True 时在下一页前停下（已完成的页照常保留）。
    """
    engine = (engine or APPLE).strip()
    if engine not in ENGINE_IDS:
        raise ValueError(f"未知 OCR 引擎: {engine}")
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(path)
    if not is_supported(path):
        raise ValueError("仅支持图片（png/jpg/jpeg/bmp/tif/tiff/webp）与 PDF")

    if is_pdf(path):
        return _recognize_pdf(path, engine, preprocess, binarize,
                              on_page, progress, cancelled)
    res = _recognize_image_file(path, engine, preprocess, binarize)
    if on_page is not None:
        on_page(res)
    if progress is not None:
        progress(1, 1, "完成")
    return [res]
