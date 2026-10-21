"""OvisOCR2 VLM 解析器（#133）——引擎链第三棒，取代 PyMuPDF+RapidOCR 兜底。

定位
----
统一引擎回退链：docling → marker → **ovis** → 纯文本兜底。ovis 负责
「扫描版 PDF / 图像类文件 → Markdown」，替换原先的 PyMuPDF+RapidOCR
方案（后者只能出无版面的纯文本）。

处理流程（2026-10-21 用户定版）
------------------------------
- **文本 PDF**：PyMuPDF 直接提取每页文本；页面检测到表格时，把表格区域
  裁剪成图 → 归一化 → OvisOCR2 转 Markdown → 填回该页（表格区域的文本层
  内容先用 redaction 剔除，避免与 VLM 输出重复）
- **图片 PDF**：PyMuPDF 取出每页图片（优先抽原始内嵌图，失败退渲染）
  → 归一化 → OvisOCR2 整页转 Markdown
- **图像文件**（png/jpg/bmp/tiff/webp）：归一化 → OvisOCR2

模型配置：``rag_ocr_engine``（mlx-vlm / onnx / gguf）+
``rag_ocr_model_name`` / ``rag_ocr_model_path``，经
:func:`set_ovis_model_config` 注入。默认自动加载 models/ 下
``OvisOCR2-0.8B-MLX-4bit``。onnx/gguf 后端预留了配置位，当前仅实现
MLX（mlx_vlm）。
"""
from __future__ import annotations

import logging
import os
import re
import threading
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 模型配置（模块级注入，同 parsers.set_ocr_config 模式）
# ---------------------------------------------------------------------------
_engine = "mlx-vlm"          # mlx-vlm | onnx | gguf
_model_name = "OvisOCR2-0.8B-MLX-4bit"
_model_path = ""             # 显式本地路径优先；空则 models/<name>
_TABLE_SCALE = 3.0           # 文本页表格裁剪的渲染倍率（细线表格要清晰）
_IMAGE_PAGE_SCALE = 2.0      # 图片 PDF 整页渲染倍率（无内嵌图时）
_MAX_SIDE = 1408             # 送 VLM 前的最长边。#133 实测：扫描书内嵌图普遍
                             # 1504×2336，全分辨率送 VLM 单页 prefill 6.8s（视觉
                             # 塔 bf16，patch 数 ∝ 面积）；压到 1408 后单页
                             # 5.0s（-46%）且转录输出逐字一致。再降（896）
                             # 开始出现字形级差异（导言→导音），1408 是甜点。
                             # 32 的倍数：Qwen3VL patch16 × merge2。
_MIN_SIDE = 256
_TEXT_LAYER_MIN_CHARS = 50   # 与 ocr_scale_policy 一致的文本层判定
_OVIS_PROMPT = (
    "<|vision_start|><|image_pad|><|vision_end|>\n"
    "Transcribe this document to markdown."
)

_OVIS_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def set_ovis_model_config(engine: str, name: str, path: str) -> None:
    """注入 OCR 模型配置（UI 保存设置时调用）。"""
    global _engine, _model_name, _model_path
    _engine = (engine or "mlx-vlm").strip() or "mlx-vlm"
    _model_name = (name or "OvisOCR2-0.8B-MLX-4bit").strip() or _model_name
    _model_path = (path or "").strip()


def _candidate_model_dirs() -> List[Path]:
    """按优先级给出候选模型目录。"""
    dirs: List[Path] = []
    if _model_path:
        dirs.append(Path(_model_path).expanduser())
    # models/<name>（应用根目录下的 models/）
    for root in (Path.cwd(), Path(__file__).resolve().parents[3]):
        dirs.append(root / "models" / _model_name)
        dirs.append(root / "models" / "OvisOCR2-0.8B-MLX-4bit")
    return dirs


def resolve_model_dir() -> Optional[Path]:
    """找到可用的模型目录（含 model.safetensors）。"""
    seen = set()
    for d in _candidate_model_dirs():
        key = str(d)
        if key in seen:
            continue
        seen.add(key)
        try:
            if (d / "model.safetensors").is_file() or (
                d / "model.safetensors.index.json"
            ).is_file():
                return d
        except OSError:
            continue
    return None


def is_ovis_available() -> bool:
    """ovis 引擎可用性：mlx_vlm 已装 + 模型目录存在。

    探测只用 find_spec——真实 import mlx_vlm 会连带加载 mlx/torch 全家桶。
    """
    if _engine != "mlx-vlm":
        # onnx/gguf 后端尚未实现：不可用（回退链继续往下走）
        return False
    try:
        import importlib.util
        if importlib.util.find_spec("mlx_vlm") is None:
            return False
    except Exception:
        return False
    return resolve_model_dir() is not None


# ---------------------------------------------------------------------------
# 模型单例
# ---------------------------------------------------------------------------
_model = None
_processor = None
_model_lock = threading.Lock()
_model_load_error: Optional[str] = None


def _get_model():
    """加载（并缓存）OvisOCR2 模型。线程安全；失败返回 None。"""
    global _model, _processor, _model_load_error
    if _model is not None:
        return _model, _processor
    if _model_load_error is not None:
        return None
    with _model_lock:
        if _model is not None:
            return _model, _processor
        model_dir = resolve_model_dir()
        if model_dir is None:
            _model_load_error = "模型目录不存在"
            return None
        try:
            from mlx_vlm import load

            t0 = __import__("time").monotonic()
            model, processor = load(str(model_dir))
            _model, _processor = model, processor
            logger.info(
                "OvisOCR2 模型加载完成: %s (%.1fs)", model_dir.name,
                __import__("time").monotonic() - t0,
            )
            return _model, _processor
        except Exception as e:
            _model_load_error = str(e)
            logger.warning("OvisOCR2 模型加载失败: %s", e)
            return None


def release_model() -> None:
    """释放模型引用（同步结束后调用，让 MLX 归还显存）。"""
    global _model, _processor, _model_load_error
    with _model_lock:
        _model = None
        _processor = None
        _model_load_error = None  # 允许下次重新尝试加载
    try:
        import gc
        gc.collect()
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# HTML → Markdown（4-bit 量化版模型倾向输出 HTML 表格，README 已说明）
# ---------------------------------------------------------------------------

class _TableToMarkdown(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: List[List[str]] = []
        self._row: Optional[List[str]] = None
        self._cell: Optional[List[str]] = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None
        elif tag in ("td", "th") and self._cell is not None:
            if self._row is None:
                self._row = []
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def _html_tables_to_markdown(text: str) -> str:
    """把输出里的 <table> 块转成 Markdown 管道表格，其余文本原样保留。"""
    if "<table" not in text.lower():
        return text

    def _convert(m: "re.Match") -> str:
        block = m.group(0)
        p = _TableToMarkdown()
        try:
            p.feed(block)
        except Exception:
            return block
        rows = [r for r in p.rows if any(c.strip() for c in r)]
        if not rows:
            return ""
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        out = ["| " + " | ".join(rows[0]) + " |",
               "| " + " | ".join(["---"] * width) + " |"]
        for r in rows[1:]:
            out.append("| " + " | ".join(r) + " |")
        return "\n".join(out)

    return re.sub(r"<table.*?</table>", _convert, text,
                  flags=re.IGNORECASE | re.DOTALL)


def _clean_vlm_output(text: str) -> str:
    text = _html_tables_to_markdown(text or "")
    text = re.sub(r"</?(?:html|body|div|span|p)\b[^>]*>", "", text,
                  flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# 图像归一化
# ---------------------------------------------------------------------------

def _save_normalized(pil_image, out_dir: str = "") -> Optional[str]:
    """PIL 图像归一化为 PNG：RGB + 最长边封顶 _MAX_SIDE / 保底 _MIN_SIDE。"""
    try:
        from PIL import Image

        if not isinstance(pil_image, Image.Image):
            img = Image.fromarray(pil_image)
        else:
            img = pil_image
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        w, h = img.size
        longest = max(w, h)
        if longest > _MAX_SIDE:
            ratio = _MAX_SIDE / longest
            img = img.resize((max(1, int(w * ratio)), max(1, int(h * ratio))),
                             Image.LANCZOS)
        elif longest < _MIN_SIDE:
            ratio = _MIN_SIDE / max(1, longest)
            img = img.resize((max(1, int(w * ratio)), max(1, int(h * ratio))),
                             Image.LANCZOS)
        if img.mode == "L":
            img = img.convert("RGB")
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".png", prefix="ovis_",
                                    dir=out_dir or None)
        os.close(fd)
        img.save(path, "PNG")
        return path
    except Exception as e:
        logger.debug("图像归一化失败: %s", e)
        return None


def _pixmap_to_png(pix, out_dir: str = "") -> Optional[str]:
    """fitz Pixmap → 归一化 PNG。"""
    try:
        if pix.colorspace and pix.colorspace.n > 3:
            pix = fitz_pixmap_rgb(pix)
        from PIL import Image
        img = Image.frombytes("RGB" if pix.n >= 3 else "L",
                              (pix.width, pix.height), pix.samples)
        return _save_normalized(img, out_dir)
    except Exception as e:
        logger.debug("Pixmap 转 PNG 失败: %s", e)
        return None


def fitz_pixmap_rgb(pix):
    import fitz
    return fitz.Pixmap(fitz.csRGB, pix)


# ---------------------------------------------------------------------------
# VLM 推理
# ---------------------------------------------------------------------------

def _vlm_ocr(image_path: str) -> Optional[str]:
    got = _get_model()
    if got is None:
        return None
    model, processor = got
    try:
        from mlx_vlm import generate

        result = generate(
            model, processor, _OVIS_PROMPT, image=image_path,
            max_tokens=1024, verbose=False,
        )
        return _clean_vlm_output(getattr(result, "text", "") or "")
    except Exception as e:
        logger.debug("OvisOCR2 推理失败: %s", e)
        return None


def _vlm_ocr_pil(pil_image, out_dir: str = "") -> Optional[str]:
    path = _save_normalized(pil_image, out_dir)
    if path is None:
        return None
    try:
        return _vlm_ocr(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# 表格区域检测（文本页用：矢量线 + 词网格）
# ---------------------------------------------------------------------------

def _cluster_rects(boxes: List, gap: float = 24.0) -> List[List]:
    """把线段/词组的 bbox 按 proximity 聚成连通组（并查集）。

    旧实现把所有表格线求并集成一个整页大 bbox——一页有多张表（或表+
    插图）时会并成一张送给 VLM，输出必坏。这里按「bbox 外扩 gap 后
    相交即同组」聚类，每组独立成一个表格区域。
    """
    n = len(boxes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            a, b = boxes[i], boxes[j]
            if (a[0] - gap <= b[2] and b[0] - gap <= a[2]
                    and a[1] - gap <= b[3] and b[1] - gap <= a[3]):
                union(i, j)
    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(boxes[i])
    return list(groups.values())


def _detect_table_rects(page) -> List:
    """返回页面上各表格区域的 fitz.Rect 列表（按连通性聚类，多表不合并）。

    文本页（born-digital）有矢量线和词坐标可用；扫描页没有文本层，走
    整页 VLM，不需要这个。
    """
    rects: List = []
    try:
        import fitz

        page_rect = page.rect
        # 1) 矢量线：每条线一个 bbox，连通聚类后每组 ≥2 横 + ≥2 竖才算表格
        h_boxes, v_boxes = [], []
        try:
            drawings = page.get_drawings()
        except Exception:
            drawings = []
        for d in drawings:
            for item in d.get("items", ()):
                if item[0] == "l":
                    p1, p2 = item[1], item[2]
                    if abs(p1.y - p2.y) < 1.0 and abs(p1.x - p2.x) > 30:
                        h_boxes.append((min(p1.x, p2.x), p1.y - 0.5,
                                        max(p1.x, p2.x), p1.y + 0.5))
                    elif abs(p1.x - p2.x) < 1.0 and abs(p1.y - p2.y) > 30:
                        v_boxes.append((p1.x - 0.5, min(p1.y, p2.y),
                                        p1.x + 0.5, max(p1.y, p2.y)))
                elif item[0] == "re":
                    r = item[1]
                    if r.width > 30 and r.height < 2:
                        h_boxes.append((r.x0, r.y0, r.x1, r.y1))
                    elif r.height > 30 and r.width < 2:
                        v_boxes.append((r.x0, r.y0, r.x1, r.y1))
        if h_boxes and v_boxes:
            for group in _cluster_rects(h_boxes + v_boxes):
                n_h = sum(1 for b in group
                          if any(b == hb for hb in h_boxes))
                n_v = len(group) - n_h
                if n_h >= 2 and n_v >= 2:
                    r = fitz.Rect(
                        min(b[0] for b in group), min(b[1] for b in group),
                        max(b[2] for b in group), max(b[3] for b in group),
                    ) & page_rect
                    if not r.is_empty and r.width > 60 and r.height > 40:
                        rects.append(r)

        # 2) 词网格（无框线表格）：多词行按 y 相邻聚类，每组 ≥3 行 × ≥3 列
        try:
            words = page.get_text("words")
        except Exception:
            words = []
        if len(words) >= 9:
            rows: dict = {}
            for w in words:
                rows.setdefault(int(w[1] // 4), []).append(w)
            table_rows = [ws for ws in rows.values() if len(ws) >= 3]
            if len(table_rows) >= 3:
                row_boxes = [(min(w[0] for w in ws), min(w[1] for w in ws),
                              max(w[2] for w in ws), max(w[3] for w in ws))
                             for ws in table_rows]
                for group in _cluster_rects(row_boxes, gap=40.0):
                    if len(group) < 3:
                        continue
                    gw = [w for ws in table_rows
                          if (min(w[0] for w in ws), min(w[1] for w in ws),
                              max(w[2] for w in ws), max(w[3] for w in ws)) in group]
                    xs = sorted({int(w[0] // 10) for w in gw})
                    cols = 0
                    last = None
                    for x in xs:
                        if last is None or x - last > 1:
                            cols += 1
                        last = x
                    if cols < 3:
                        continue
                    r = fitz.Rect(
                        min(w[0] for w in gw), min(w[1] for w in gw),
                        max(w[2] for w in gw), max(w[3] for w in gw),
                    ) & page_rect
                    if r.is_empty or r.width <= 60 or r.height <= 40:
                        continue
                    # 与矢量线结果重叠则跳过（同一张表）
                    if not any((r & e).get_area() > 0.5 * r.get_area()
                               for e in rects):
                        rects.append(r)
    except Exception as e:
        logger.debug("表格区域检测失败: %s", e)
    return rects


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def extract_markdown(filepath: str) -> Optional[str]:
    """OvisOCR2 解析：PDF（文本/图片）与图像文件 → Markdown。

    不可用 / 失败返回 None（调用方继续回退链）。
    """
    if not is_ovis_available():
        return None
    path = Path(filepath)
    suffix = path.suffix.lower()

    try:
        if suffix == ".pdf":
            return _extract_pdf(path)
        if suffix in _OVIS_IMAGE_EXTS:
            from PIL import Image
            with Image.open(path) as img:
                img = img.convert("RGB") if img.mode not in ("RGB", "L") else img
                md = _vlm_ocr_pil(img)
            if md:
                logger.info("OvisOCR2 图像解析: %s → %d 字符",
                            path.name, len(md))
            return md or None
    except Exception as e:
        logger.warning("OvisOCR2 解析失败 %s: %s", path.name, e)
        return None
    return None


def _extract_pdf(path: Path) -> Optional[str]:
    import fitz

    doc = fitz.open(str(path))
    try:
        pages_md: List[str] = []
        n = doc.page_count
        for page in doc:
            page_no = page.number + 1
            text = page.get_text("text") or ""
            if len(text.strip()) >= _TEXT_LAYER_MIN_CHARS:
                # ---- 文本页：直提文本 + 表格区域 VLM 回填 ----
                table_rects = _detect_table_rects(page)
                if not table_rects:
                    pages_md.append(text.strip())
                    continue
                table_mds: List[str] = []
                for rect in table_rects:
                    clip = page.get_pixmap(matrix=fitz.Matrix(
                        _TABLE_SCALE, _TABLE_SCALE), clip=rect)
                    png = _pixmap_to_png(clip)
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
                        table_mds.append(md)
                # redaction 剔除表格区域的文本层内容，避免与 VLM 输出重复
                try:
                    for rect in table_rects:
                        page.add_redact_annot(rect)
                    page.apply_redactions()
                    rest = page.get_text("text") or ""
                except Exception:
                    rest = text
                parts = [p for p in (rest.strip(),
                                     "\n\n".join(table_mds)) if p]
                pages_md.append("\n\n".join(parts))
                logger.info(
                    "OvisOCR2 文本页 %d/%d: 表格 %d 个 → VLM 回填",
                    page_no, n, len(table_rects),
                )
            else:
                # ---- 图片页：抽内嵌原图优先，失败退渲染 ----
                img_md = _ocr_image_page(page)
                if img_md:
                    pages_md.append(img_md)
                logger.info("OvisOCR2 图片页 %d/%d → %d 字符",
                            page_no, n, len(img_md or ""))
        md = "\n\n".join(p for p in pages_md if p.strip())
        if not md.strip():
            return None
        logger.info("OvisOCR2 PDF 解析完成: %s (%d 页, %d 字符)",
                    path.name, n, len(md))
        return md
    finally:
        doc.close()


def _ocr_image_page(page) -> Optional[str]:
    """图片 PDF 的单页 OCR：优先抽取内嵌原图（保真），失败退整页渲染。"""
    import fitz

    # 1) 内嵌原图
    try:
        images = page.get_images(full=True)
        if images:
            from PIL import Image
            import io
            best = None
            for img in images:
                xref = img[0]
                try:
                    info = page.parent.extract_image(xref)
                except Exception:
                    continue
                if not info or not info.get("image"):
                    continue
                w, h = info.get("width", 0), info.get("height", 0)
                if best is None or w * h > best[0] * best[1]:
                    best = (w, h, info["image"])
            if best is not None:
                pil = Image.open(io.BytesIO(best[2]))
                md = _vlm_ocr_pil(pil)
                if md:
                    return md
    except Exception as e:
        logger.debug("内嵌图抽取失败，退渲染: %s", e)
    # 2) 整页渲染
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(_IMAGE_PAGE_SCALE,
                                                 _IMAGE_PAGE_SCALE))
        png = _pixmap_to_png(pix)
        if png is None:
            return None
        try:
            return _vlm_ocr(png)
        finally:
            try:
                os.unlink(png)
            except OSError:
                pass
    except Exception as e:
        logger.debug("整页渲染失败: %s", e)
        return None
