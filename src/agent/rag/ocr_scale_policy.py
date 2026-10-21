"""docling OCR 渲染 scale 的动态策略（#132）。

背景
----
docling 标准管线按 预处理 → OCR → 布局 → 表格 的顺序执行：表格检测发生在
OCR **之后**，所以「表格页用更高 OCR scale」不能依赖 docling 自己的检测，
必须在 convert 之前用 PyMuPDF 做一次轻量预分析。

好在 docling 的 ocrmac 引擎每渲染一页都重新读 ``self.scale``
（``ocr_mac_model.py:172: page.render(scale=self.scale, ...)``），实例属性
可动态修改——我们用一个自定义 pipeline 类包装 OCR 模型，在每批页面前按
策略改写它，实现逐页 scale。

策略（2026-10-21 用户定版）
--------------------------
- 纯文本页（有文本层）        → 2.0（docling 对这些页根本不触发 OCR，占位统一）
- 扫描页且原生分辨率 < 150dpi → 2.0（放大不产生新信息，纯浪费位图内存）
- 其余扫描页                  → 含表格的页 2.5，其他页 2.0

代价收益：整页位图内存 ∝ scale²，3.0→2.0 省 56%（实测 25GB 尖峰的主要来源）。
表格页保留较高 scale 保住细线表格的识别质量。
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# 策略常量（可经 set_scale_config 配置注入；默认动态关闭、固定 2.0）
BASE_SCALE = 2.0      # 普通页 / 文本页 / 低 DPI 扫描页
TABLE_SCALE = 2.5     # 检测到表格的扫描页
_DYNAMIC_ENABLED = False   # 动态策略开关（默认关：固定用 BASE_SCALE）
LOW_DPI_THRESHOLD = 150.0
TEXT_LAYER_MIN_CHARS = 50     # 页面文本层超过该字数视为「有文本层」
_TABLE_MIN_LINES = 2          # 横/竖规则线各至少 N 条才算有线表格
_TABLE_MIN_ROWS = 3           # 词网格至少 N 行
_TABLE_MIN_COLS = 3           # 词网格至少 N 列

# ---------------------------------------------------------------------------
# 逐页策略的当前值（convert 期间独占）
# ---------------------------------------------------------------------------
# docling_parser.extract_markdown 在 _converter_lock 内串行调用 convert()，
# 只有一次转换在跑，所以这个模块级变量是安全的。set/clear 成对出现在
# extract_markdown 的 try/finally 里。
_current_policy: Optional[Dict[int, float]] = None
_policy_lock = threading.Lock()


def set_scale_config(enabled: bool, base: float = 2.0,
                     table: float = 2.5) -> None:
    """配置注入（UI 保存设置时调用）。

    enabled=False 时 analyze_pdf 直接返回 None——docling 固定用
    base 值，不做任何预分析。
    """
    global BASE_SCALE, TABLE_SCALE, _DYNAMIC_ENABLED
    BASE_SCALE = float(base) if base else 2.0
    TABLE_SCALE = float(table) if table else 2.5
    _DYNAMIC_ENABLED = bool(enabled)


def set_current_policy(policy: Optional[Dict[int, float]]) -> None:
    global _current_policy
    with _policy_lock:
        _current_policy = policy


def get_scale_for_page(page_no: int) -> float:
    with _policy_lock:
        pol = _current_policy
    if pol:
        return pol.get(page_no, BASE_SCALE)
    return BASE_SCALE


# ---------------------------------------------------------------------------
# PDF 预分析
# ---------------------------------------------------------------------------

_analysis_cache: Dict[Any, Optional[Dict[int, float]]] = {}
_ANALYSIS_CACHE_MAX = 64


def _cache_key(path) -> Any:
    import os
    st = os.stat(path)
    return (str(path), st.st_mtime_ns, st.st_size)


def _page_native_dpi(page) -> Optional[float]:
    """页面内嵌图片的有效 DPI（图片像素数 / 版面尺寸），取全页最大值。"""
    try:
        images = page.get_images(full=True)
    except Exception:
        return None
    best: Optional[float] = None
    for img in images:
        try:
            xref = img[0]
            pix_w, pix_h = img[2], img[3]
            if pix_w <= 0 or pix_h <= 0:
                continue
            for rect in page.get_image_rects(xref):
                if rect.width <= 0 or rect.height <= 0:
                    continue
                dpi_x = pix_w / (rect.width / 72.0)
                dpi_y = pix_h / (rect.height / 72.0)
                dpi = max(dpi_x, dpi_y)
                if best is None or dpi > best:
                    best = dpi
        except Exception:
            continue
    return best


def _page_has_table_heuristic(page) -> bool:
    """表格启发式：优先规则线（横竖线网格），退化为词网格对齐检测。"""
    # 1) 矢量规则线：扫描页通常没有；born-digital 表格页有
    try:
        drawings = page.get_drawings()
    except Exception:
        drawings = []
    h_lines = v_lines = 0
    for d in drawings:
        for item in d.get("items", ()):
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                if abs(p1.y - p2.y) < 1.0 and abs(p1.x - p2.x) > 30:
                    h_lines += 1
                elif abs(p1.x - p2.x) < 1.0 and abs(p1.y - p2.y) > 30:
                    v_lines += 1
            elif item[0] == "re":
                r = item[1]
                # 细长矩形 ≈ 线
                if r.width > 30 and r.height < 2:
                    h_lines += 1
                elif r.height > 30 and r.width < 2:
                    v_lines += 1
    if h_lines >= _TABLE_MIN_LINES and v_lines >= _TABLE_MIN_LINES:
        return True

    # 2) 词网格：≥3 行、每行 ≥3 个词，且列位置跨行重合（表格化排布）
    try:
        words = page.get_text("words")
    except Exception:
        words = []
    if len(words) >= _TABLE_MIN_COLS * _TABLE_MIN_ROWS:
        rows: Dict[int, list] = {}
        for w in words:
            y_key = int(w[1] // 4)  # 4pt 容差聚成一行
            rows.setdefault(y_key, []).append(w)
        table_rows = [ws for ws in rows.values() if len(ws) >= _TABLE_MIN_COLS]
        if len(table_rows) >= _TABLE_MIN_ROWS:
            # 收集所有行的词左边界 x，聚成列；表格要求 ≥3 个跨行稳定的列
            xs = sorted({int(w[0] // 10) for ws in table_rows for w in ws})
            col_count = 0
            last = None
            for x in xs:
                if last is None or x - last > 1:
                    col_count += 1
                last = x
            if col_count >= _TABLE_MIN_COLS:
                return True

    # 3) 扫描页（纯位图）：矢量线和文本层都为空，用低分辨率渲染 + numpy
    #    暗线检测。有线表格的横/竖贯穿线在灰度图上表现为细长暗段。
    return _page_has_scan_table(page)


def _longest_run(arr) -> int:
    """一维布尔数组的最长连续 True 段长度。"""
    import numpy as np

    if arr.size == 0:
        return 0
    idx = np.flatnonzero(np.diff(np.concatenate(([0], arr.view(np.int8), [0]))))
    if idx.size == 0:
        return int(arr.sum()) if arr.all() else 0
    # idx 成对出现（段的起止），最长段 = max(diff)//2... 直接用 runs 长度
    starts, ends = idx[0::2], idx[1::2]
    return int((ends - starts).max())


def _page_has_scan_table(page) -> bool:
    """扫描页表格检测：低分辨率灰度渲染 + 长暗线网格。"""
    try:
        import numpy as np
        import fitz

        pix = page.get_pixmap(
            matrix=fitz.Matrix(1.0, 1.0), colorspace=fitz.csGRAY
        )
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width
        )
    except Exception:
        return False
    h, w = arr.shape
    if h < 50 or w < 50:
        return False
    dark = arr < 190  # 190：抗锯齿后细线边缘被抬到 128 以上，190 能兜住；
                     # 纸面背景通常 >220，文字/线条本体 <100，居中取值
    # 整页深色（照片页/深底纹）不可能是线表格——暗像素占比过高的直接否掉，
    # 否则每行都是"长暗段"假阳性。
    if dark.mean() > 0.3:
        return False
    min_len = max(60, int(w * 0.2))
    h_hits = 0
    for row in dark:
        if _longest_run(row) >= min_len:
            h_hits += 1
    if h_hits < _TABLE_MIN_LINES:
        return False
    min_len_v = max(60, int(h * 0.2))
    v_hits = 0
    for col in dark.T:
        if _longest_run(col) >= min_len_v:
            v_hits += 1
    return v_hits >= _TABLE_MIN_LINES


def analyze_pdf(path) -> Optional[Dict[int, float]]:
    """预分析 PDF，返回 {page_no: TABLE_SCALE}（仅表格页）。

    返回 None = 无需动态策略（全部按默认 BASE_SCALE）。结果按
    (path, mtime, size) 缓存，重复同步不重分析。动态开关关闭时直接
    返回 None（默认行为：固定 scale，零分析开销）。
    """
    if not _DYNAMIC_ENABLED:
        return None
    key = None
    try:
        key = _cache_key(path)
        if key in _analysis_cache:
            return _analysis_cache[key]
    except Exception:
        key = None

    policy: Dict[int, float] = {}
    try:
        import fitz

        doc = fitz.open(path)
        try:
            for page in doc:
                page_no = page.number + 1
                text = page.get_text("text") or ""
                if len(text.strip()) >= TEXT_LAYER_MIN_CHARS:
                    continue  # 文本页：不触发 OCR，无需特殊 scale
                dpi = _page_native_dpi(page)
                if dpi is not None and dpi < LOW_DPI_THRESHOLD:
                    continue  # 低分辨率扫描页：放大无新信息
                if _page_has_table_heuristic(page):
                    policy[page_no] = TABLE_SCALE
        finally:
            doc.close()
    except Exception as e:
        logger.debug("OCR scale 策略预分析失败 %s: %s", path, e)
        policy = {}

    result = policy or None
    if key is not None:
        if len(_analysis_cache) >= _ANALYSIS_CACHE_MAX:
            _analysis_cache.clear()
        _analysis_cache[key] = result
    if result:
        logger.info(
            "OCR scale 策略: %s 表格页 %d 个 → %.1f，其余 %.1f",
            getattr(path, "name", path), len(result), TABLE_SCALE, BASE_SCALE,
        )
    return result


# ---------------------------------------------------------------------------
# docling pipeline 包装
# ---------------------------------------------------------------------------

class ScalePolicyOcrModel:
    """OCR 模型包装：每批页面前按当前策略改写 inner.scale。

    docling 的 ocrmac 引擎每页渲染时读取 ``self.scale``
    （ocr_mac_model.py:172），所以在页面之间改这个属性即可生效。
    无策略时原样批量委派（不逐页拆分，避免无谓开销）。
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __call__(self, conv_res: Any, pages: Any) -> Any:
        if not _current_policy:
            return self._inner(conv_res, pages)
        out = []
        for page in pages:
            page_no = getattr(page, "page_no", 0)
            self._inner.scale = get_scale_for_page(page_no)
            out.extend(self._inner(conv_res, [page]))
        return out

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def make_scale_policy_pipeline_cls() -> Optional[Callable]:
    """构造带 OCR scale 策略的 StandardPdfPipeline 子类。

    失败（docling 内部结构变化等）返回 None，调用方退回默认管线。
    """
    try:
        from docling.pipeline.standard_pdf_pipeline import (
            StandardPdfPipeline,
        )

        class _ScalePolicyPdfPipeline(StandardPdfPipeline):
            def _make_ocr_model(self, art_path: Any) -> Any:
                inner = super()._make_ocr_model(art_path)
                return ScalePolicyOcrModel(inner)

        return _ScalePolicyPdfPipeline
    except Exception as e:
        logger.debug("无法构造 OCR scale 策略管线，退回默认: %s", e)
        return None
