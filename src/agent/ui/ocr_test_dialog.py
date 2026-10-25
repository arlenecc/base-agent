"""OCR 识别测试窗口：左侧原件、右侧识别结果，多页 PDF 左右联动对照。

用途：在真正入库之前先看看「这个引擎对这个文件的效果到底如何」，觉得可用
再点「保存结果」。识别走的是与入库链路完全相同的引擎与归一化
（``rag/ocr_tester.py``），所以这里的结论对真实解析有效。

线程模型：识别在 ``OcrTestWorker`` 里跑（PyMuPDF / Vision / VLM 都是阻塞
且耗时的），逐页 emit 回 UI 线程；关闭窗口时先停 worker 再销毁控件，否则
信号会打进已释放的 C++ 对象。
"""
from __future__ import annotations

import collections
import logging
import os
import re
from typing import Dict, List, Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QProgressBar, QPushButton, QScrollArea,
    QSizePolicy, QSpinBox, QSplitter, QTextEdit, QVBoxLayout, QWidget,
)

from ..atomic_io import atomic_write_text
from ..rag import ocr_tester
from ..rag.ocr_tester import PageResult

logger = logging.getLogger(__name__)

_FILE_FILTER = (
    "图片或 PDF (*.pdf *.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;"
    "PDF (*.pdf);;图片 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;"
    "所有文件 (*)"
)

# 下拉前三项固定；第四项「自定义OCR」动态命名（模型名从配置路径/名称提取）。
_ENGINE_ITEMS = (
    ("Apple OCR", ocr_tester.APPLE),
    ("RapidOCR", ocr_tester.RAPID),
    ("混合（Vision+Ovis表格，仅PDF）", ocr_tester.HYBRID),
)
# 「自定义OCR」在 engine_combo 中的固定位置（数据仍是 ovis 引擎，区别只在于
# 用配置面板填的模型/后端，而非内置默认 OvisOCR3）。
_CUSTOM_ENGINE_INDEX = 3

_KIND_LABEL = {"text": "文本层", "image": "扫描页", "blank": "空白页"}

# 结果是 HTML（VLM 常直接吐 <table>）时走 setHtml，否则按 Markdown 渲染。
_HTML_RE = re.compile(
    r"<(html|body|table|thead|tbody|tr|td|th|div|p|h[1-6])\b", re.IGNORECASE)

# 预览位图缓存页数：1500px 级别的 QPixmap 每页十几 MB，长 PDF 不能全缓存。
_PREVIEW_CACHE_MAX = 8
# 预览渲染 DPI：只要能看清版式即可，不必到 OCR 的分辨率。
_PREVIEW_DPI = 110


class OcrTestWorker(QThread):
    """后台逐页识别。

    信号用 ``object`` 携带 ``PageResult``：跨线程走 queued 投递，UI 线程里
    拿到的是同一 Python 对象，不需要再序列化。
    """

    page_done = pyqtSignal(object)        # PageResult
    progress = pyqtSignal(int, int, str)  # done, total, msg
    failed = pyqtSignal(str)

    def __init__(self, path: str, engine: str, preprocess: bool,
                 binarize: bool, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._path = path
        self._engine = engine
        self._preprocess = preprocess
        self._binarize = binarize
        # 协作式取消：只保证「下一页不再开始」，不打断正在跑的那一页
        # （强杀线程会让 VLM / Vision 的原生调用停在半路）。
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:  # type: ignore[override]
        try:
            ocr_tester.recognize_file(
                self._path,
                self._engine,
                preprocess=self._preprocess,
                binarize=self._binarize,
                on_page=lambda r: self.page_done.emit(r),
                progress=lambda d, t, m: self.progress.emit(d, t, m),
                cancelled=lambda: self._cancel,
            )
        except Exception as e:
            logger.warning("OCR 测试失败: %s", e)
            self.failed.emit(f"{type(e).__name__}: {e}")


class OcrTestDialog(QDialog):
    """识别测试窗口（配置面板「识别测试」按钮打开）。"""

    def __init__(self, config, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._config = config
        self._path = ""
        self._page_count = 0
        self._results: Dict[int, PageResult] = {}
        self._current = 0
        self._worker: Optional[OcrTestWorker] = None
        # 预览用的独立文档句柄：识别进程会对页面做 apply_redactions（剔除表格
        # 区文本层），共用同一份文档会把预览里的表格涂黑。
        self._preview_doc = None
        self._preview_cache: "collections.OrderedDict[int, QPixmap]" = \
            collections.OrderedDict()

        self.setWindowTitle("OCR 识别测试")
        self.resize(1200, 840)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _custom_model_label(self) -> str:
        """第四项「自定义OCR」的显示名：优先用模型路径的目录名，其次模型名。

        这样用户在配置面板设好自定义模型并「保存设置」后，下拉里立刻能看到
        这个模型是谁，不必去翻路径。
        """
        cfg = self._config
        path = (getattr(cfg, "rag_ocr_model_path", "") or "").strip().rstrip("/")
        name = (getattr(cfg, "rag_ocr_model_name", "") or "").strip()
        engine = (getattr(cfg, "rag_ocr_engine", "") or "").strip()
        if path:
            return f"自定义OCR（{os.path.basename(path)}）"
        if name and name != "OvisOCR3-MLX-4bit-mixed46":
            return f"自定义OCR（{name}）"
        if engine and engine != "mlx-vlm":
            return f"自定义OCR（{engine}）"
        return "自定义OCR（OvisOCR3-MLX-4bit-mixed46）"

    def refresh_engine_items(self) -> None:
        """配置面板「保存设置」后刷新第四项文字（模型名来自路径）。"""
        if self.engine_combo.count() > _CUSTOM_ENGINE_INDEX:
            self.engine_combo.setItemText(
                _CUSTOM_ENGINE_INDEX, self._custom_model_label())

    def showEvent(self, event) -> None:  # type: ignore[override]
        # 每次打开都按当前 config 刷新自定义项名字（用户可能已改过设置）。
        self.refresh_engine_items()
        super().showEvent(event)

    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        # ---- 顶部：文件 + 引擎 + 开始 / 保存 ----
        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(QLabel("待识别文件"))
        self.path_edit = QLineEdit(placeholderText="选择图片（png/jpg/…）或 PDF 文件")
        self.path_edit.setReadOnly(True)
        self.pick_btn = QPushButton("选择")
        self.pick_btn.setFixedWidth(64)
        self.pick_btn.clicked.connect(self._on_pick_file)
        top.addWidget(self.path_edit, 1)
        top.addWidget(self.pick_btn)

        top.addSpacing(12)
        top.addWidget(QLabel("识别模型"))
        self.engine_combo = QComboBox()
        self.engine_combo.setMinimumWidth(220)
        for label, eid in _ENGINE_ITEMS:
            self.engine_combo.addItem(label, eid)
        # 第四项：自定义OCR（数据走 ovis 引擎，名字来自配置里用户设置的模型）。
        self.engine_combo.addItem(self._custom_model_label(), ocr_tester.OVIS)
        self.engine_combo.setToolTip(
            "· Apple OCR：系统 Vision 框架（ocrmac），中文印刷体质量好、ANE 推理\n"
            "· RapidOCR：本地轻量 OCR 引擎（可换 det/rec 模型），输出无版面纯文本\n"
            "· 混合：正文走 Vision、表格区域走 OvisOCR3（仅 PDF）\n"
            "· 自定义OCR：使用配置面板「自定义OCR模型」行所填的模型/后端，模型名取自设置路径\n\n"
            "PDF 会自动判别文本型 / 扫描型：文本页直提文本层，扫描页才跑 OCR。"
        )
        top.addWidget(self.engine_combo)

        self.start_btn = QPushButton("开始识别")
        self.start_btn.setFixedWidth(92)
        self.start_btn.setStyleSheet("font-weight:600;")
        self.start_btn.clicked.connect(self._on_start_or_stop)
        self.save_btn = QPushButton("保存结果")
        self.save_btn.setFixedWidth(92)
        self.save_btn.clicked.connect(self._on_save_result)
        top.addWidget(self.start_btn)
        top.addWidget(self.save_btn)
        lay.addLayout(top)

        # ---- 中部：左原件 / 右结果 ----
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)

        left = QWidget()
        llay = QVBoxLayout(left)
        llay.setContentsMargins(0, 0, 0, 0)
        llay.setSpacing(6)
        nav = QHBoxLayout()
        nav.setSpacing(6)
        self.prev_btn = QPushButton("上一页")
        self.prev_btn.setFixedWidth(72)
        self.prev_btn.clicked.connect(lambda: self._goto_page(self._current - 1))
        self.next_btn = QPushButton("下一页")
        self.next_btn.setFixedWidth(72)
        self.next_btn.clicked.connect(lambda: self._goto_page(self._current + 1))
        self.spin = QSpinBox()
        self.spin.setRange(1, 1)
        self.spin.setFixedWidth(64)
        self.spin.valueChanged.connect(self._on_spin_changed)
        self.page_total_label = QLabel("/ -")
        nav.addWidget(QLabel("第"))
        nav.addWidget(self.spin)
        nav.addWidget(self.page_total_label)
        nav.addStretch(1)
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)
        llay.addLayout(nav)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.image_label = QLabel("尚未选择文件")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignHCenter
                                      | Qt.AlignmentFlag.AlignTop)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored,
                                       QSizePolicy.Policy.Ignored)
        self.scroll.setWidget(self.image_label)
        llay.addWidget(self.scroll, 1)
        split.addWidget(left)

        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(0, 0, 0, 0)
        rlay.setSpacing(6)
        rtool = QHBoxLayout()
        rtool.setSpacing(8)
        rtool.addWidget(QLabel("识别结果"))
        self.scope_combo = QComboBox()
        self.scope_combo.addItem("当前页")
        self.scope_combo.addItem("全部页")
        self.scope_combo.setToolTip(
            "「当前页」与左侧联动对照；「全部页」拼接所有页结果（保存时用这个）。")
        self.scope_combo.currentIndexChanged.connect(self._refresh_result)
        self.raw_check = QCheckBox("显示源码")
        self.raw_check.setToolTip("按纯文本显示（不渲染 Markdown / HTML），便于核对原始输出。")
        self.raw_check.stateChanged.connect(self._refresh_result)
        self.page_info_label = QLabel("")
        rtool.addWidget(self.scope_combo)
        rtool.addWidget(self.raw_check)
        rtool.addWidget(self.page_info_label, 1)
        rlay.addLayout(rtool)

        self.result_view = QTextEdit()
        self.result_view.setReadOnly(True)
        self.result_view.setPlaceholderText("点击「开始识别」后，这里显示识别结果")
        rlay.addWidget(self.result_view, 1)
        split.addWidget(right)

        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        split.setSizes([560, 560])
        lay.addWidget(split, 1)

        # ---- 底部：状态 + 进度 + 关闭 ----
        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        self.status_label = QLabel("就绪")
        self.progress = QProgressBar()
        self.progress.setFixedWidth(200)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        bottom.addWidget(self.status_label, 1)
        bottom.addWidget(self.progress)
        close_btn = QPushButton("关闭")
        close_btn.setFixedWidth(84)
        close_btn.clicked.connect(self.close)
        bottom.addWidget(close_btn)
        lay.addLayout(bottom)

    # ------------------------------------------------------------------
    # 文件
    # ------------------------------------------------------------------

    def _on_pick_file(self) -> None:
        start = os.path.dirname(self._path) if self._path else os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(self, "选择待识别文件", start,
                                              _FILE_FILTER)
        if not path:
            return
        self._load_file(path)

    def load_file(self, path: str) -> None:
        """外部（配置面板）直接指定文件。"""
        self._load_file(path)

    def _load_file(self, path: str) -> None:
        path = os.path.abspath(os.path.expanduser(path or ""))
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "文件无效", f"找不到文件：\n{path}")
            return
        if not ocr_tester.is_supported(path):
            QMessageBox.warning(
                self, "格式不支持",
                "仅支持图片（png/jpg/jpeg/bmp/tif/tiff/webp）与 PDF。")
            return

        self._stop_worker()
        self._close_preview()
        self._path = path
        self._results = {}
        self._current = 0
        self.path_edit.setText(path)

        pages = ocr_tester.probe_pages(path)
        if pages <= 0:
            QMessageBox.warning(self, "打开失败", f"无法读取该文件：\n{path}")
            self._page_count = 0
            self.page_total_label.setText("/ -")
            return
        self._page_count = pages
        self.spin.setRange(1, max(1, pages))
        self.spin.blockSignals(True)
        self.spin.setValue(1)
        self.spin.blockSignals(False)
        self.page_total_label.setText(f"/ {pages}")
        self.progress.setRange(0, max(1, pages))
        self.progress.setValue(0)
        self.status_label.setText(f"已载入 {os.path.basename(path)}（{pages} 页）")
        self._render_preview()
        self._refresh_result()

    # ------------------------------------------------------------------
    # 预览（左侧）
    # ------------------------------------------------------------------

    def _close_preview(self) -> None:
        self._preview_cache.clear()
        doc = self._preview_doc
        self._preview_doc = None
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass

    def _pixmap_for(self, index: int) -> Optional[QPixmap]:
        """取得某页预览位图（LRU 缓存，避免长 PDF 撑爆内存）。"""
        cached = self._preview_cache.get(index)
        if cached is not None:
            self._preview_cache.move_to_end(index)
            return cached
        if not self._path:
            return None
        pm = QPixmap()
        try:
            if ocr_tester.is_pdf(self._path):
                import fitz
                if self._preview_doc is None:
                    self._preview_doc = fitz.open(self._path)
                page = self._preview_doc[index]
                pix = page.get_pixmap(dpi=_PREVIEW_DPI)
                try:
                    pm.loadFromData(pix.tobytes("png"), "PNG")
                finally:
                    pix = None
            else:
                pm.load(self._path)
        except Exception as e:
            logger.debug("预览渲染失败[%d]: %s", index, e)
            return None
        if pm.isNull():
            return None
        self._preview_cache[index] = pm
        while len(self._preview_cache) > _PREVIEW_CACHE_MAX:
            self._preview_cache.popitem(last=False)
        return pm

    def _render_preview(self) -> None:
        if not self._path or self._page_count <= 0:
            self.image_label.setText("尚未选择文件")
            self.image_label.setPixmap(QPixmap())
            return
        pm = self._pixmap_for(self._current)
        if pm is None:
            self.image_label.setText(f"第 {self._current + 1} 页预览失败")
            self.image_label.setPixmap(QPixmap())
            return
        # 按视口宽度缩放：整页位图（1500px 级）直接铺会大到没法对照，
        # 也不该缩得比视口小（小字看不清）。
        width = max(240, self.scroll.viewport().width() - 16)
        self.image_label.setPixmap(
            pm.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        # 视口宽度变了就按新宽度重缩放（不重新渲染页面，只是缩放显示）。
        try:
            self._render_preview()
        except Exception:  # pragma: no cover - 关闭途中的竞态
            pass

    def _goto_page(self, index: int) -> None:
        if self._page_count <= 0:
            return
        index = max(0, min(int(index), self._page_count - 1))
        if index == self._current:
            return
        self._current = index
        self.spin.blockSignals(True)
        self.spin.setValue(index + 1)
        self.spin.blockSignals(False)
        self._render_preview()
        self._refresh_result()

    def _on_spin_changed(self, value: int) -> None:
        self._goto_page(int(value) - 1)

    # ------------------------------------------------------------------
    # 结果（右侧）
    # ------------------------------------------------------------------

    def _refresh_result(self) -> None:
        if self.scope_combo.currentIndex() == 1:
            self._set_result_text(self.all_text())
            res = self._results.get(self._current)
            self.page_info_label.setText(
                f"共 {len([r for r in self._results.values() if r.markdown])} 页有内容")
            return
        res = self._results.get(self._current)
        if res is None:
            self._set_result_text("")
            self.page_info_label.setText(
                "该页尚未识别" if self._path else "")
            return
        marks = [_KIND_LABEL.get(res.kind, res.kind)]
        if res.error:
            marks.append(res.error)
        self.page_info_label.setText(" · ".join(m for m in marks if m))
        self._set_result_text((res.markdown or "").strip())

    def _set_result_text(self, text: str) -> None:
        view = self.result_view
        if self.raw_check.isChecked():
            view.setPlainText(text or "")
            return
        if not text:
            view.clear()
            return
        if _HTML_RE.search(text):
            view.setHtml(text)
        else:
            # QTextEdit 的 Markdown 子集含表格，VLM/文本层输出都能正常显示。
            view.setMarkdown(text)

    def all_text(self) -> str:
        """全部页结果（按页序拼接，带页标注，供保存）。"""
        parts: List[str] = []
        for i in range(self._page_count):
            res = self._results.get(i)
            if res is None:
                continue
            body = (res.markdown or "").strip()
            if not body:
                continue
            parts.append(f"<!-- 第 {i + 1} 页 -->\n{body}"
                         if self._page_count > 1 else body)
        return "\n\n".join(parts)

    def _current_text(self) -> str:
        if self.scope_combo.currentIndex() == 1:
            return self.all_text()
        res = self._results.get(self._current)
        return (res.markdown or "").strip() if res else ""

    # ------------------------------------------------------------------
    # 识别
    # ------------------------------------------------------------------

    def _on_start_or_stop(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._stop_worker()
            self.status_label.setText("已停止（保留已识别的页）")
            self.start_btn.setText("开始识别")
            return
        self._start()

    def _start(self) -> None:
        if not self._path:
            QMessageBox.information(self, "提示", "请先选择待识别的图片或 PDF 文件。")
            return
        engine = self.engine_combo.currentData() or ocr_tester.APPLE
        if engine == ocr_tester.HYBRID and not ocr_tester.is_pdf(self._path):
            QMessageBox.warning(self, "引擎不支持",
                                "混合模式（Vision+Ovis表格）仅支持 PDF。")
            return
        reason = ocr_tester.engine_status().get(engine, "")
        if reason:
            QMessageBox.warning(self, "引擎不可用", reason)
            return

        self._results = {}
        self._current = 0
        self.spin.blockSignals(True)
        self.spin.setValue(1)
        self.spin.blockSignals(False)
        self.progress.setRange(0, max(1, self._page_count))
        self.progress.setValue(0)
        self._refresh_result()

        preprocess = bool(getattr(self._config, "rag_ocr_preprocess", True))
        binarize = bool(getattr(self._config, "rag_ocr_binarize", True))
        self._worker = OcrTestWorker(self._path, engine, preprocess, binarize, self)
        self._worker.page_done.connect(self._on_page_done)
        self._worker.progress.connect(self._on_progress)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()
        self.start_btn.setText("停止")
        self.status_label.setText(
            f"识别中：{self.engine_combo.currentText()} · {os.path.basename(self._path)}")

    def _stop_worker(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is None:
            return
        # 先断信号：即使线程跑完最后一跳 emit，也不会打进正在销毁的控件。
        for sig in (worker.page_done, worker.progress, worker.failed,
                    worker.finished):
            try:
                sig.disconnect()
            except (TypeError, RuntimeError):
                pass
        # 协作式取消：只保证下一页不再开始，正在跑的那一页要等它自己结束。
        worker.cancel()
        if worker.isRunning():
            worker.wait(3000)
        if worker.isRunning():
            # 兜底：VLM/Vision 的原生调用可能卡在长推理里。强杀不安全，但比
            # 「对话框已销毁、线程还在往已释放的控件 emit」更可控（与主窗口
            # AgentWorker 的取舍一致）。
            worker.terminate()
            worker.wait(1000)

    def _on_page_done(self, res: PageResult) -> None:
        self._results[res.index] = res
        self.progress.setValue(min(len(self._results), self.progress.maximum()))
        self.status_label.setText(
            f"已识别第 {res.index + 1} 页 · {_KIND_LABEL.get(res.kind, res.kind)}")
        # 跟随最新页：识别过程中能一路看到结果，不必等全部完成。
        if res.index != self._current:
            self._current = res.index
            self.spin.blockSignals(True)
            self.spin.setValue(res.index + 1)
            self.spin.blockSignals(False)
            self._render_preview()
        self._refresh_result()

    def _on_progress(self, done: int, total: int, msg: str) -> None:
        if total > 0:
            self.progress.setRange(0, total)
        self.progress.setValue(min(done, self.progress.maximum()))

    def _on_failed(self, msg: str) -> None:
        QMessageBox.warning(self, "识别失败", msg)
        self.status_label.setText(f"识别失败：{msg}")

    def _on_finished(self) -> None:
        self.start_btn.setText("开始识别")
        chars = sum(len(r.markdown or "") for r in self._results.values())
        done = len(self._results)
        self.status_label.setText(
            f"完成：{done} 页 · {chars} 字符 · {self.engine_combo.currentText()}")

    # ------------------------------------------------------------------
    # 保存
    # ------------------------------------------------------------------

    def _on_save_result(self) -> None:
        text = self._current_text()
        if not text.strip():
            QMessageBox.information(self, "无可保存内容",
                                    "当前视图没有识别结果（先识别，或切到「全部页」）。")
            return
        engine = self.engine_combo.currentData() or ocr_tester.APPLE
        stem = os.path.splitext(os.path.basename(self._path or "ocr"))[0]
        default = os.path.join(os.path.dirname(self._path) or ".",
                               f"{stem}.{engine}.md")
        path, _ = QFileDialog.getSaveFileName(
            self, "保存识别结果", default,
            "Markdown (*.md);;HTML (*.html);;文本 (*.txt);;所有文件 (*)")
        if not path:
            return
        try:
            atomic_write_text(path, text)
        except OSError as e:
            QMessageBox.warning(self, "保存失败", f"{type(e).__name__}: {e}")
            return
        self.status_label.setText(f"已保存：{path}")

    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # type: ignore[override]
        """先停 worker 再销毁：识别线程晚一步 emit 会打进已释放的控件。"""
        self._stop_worker()
        self._close_preview()
        super().closeEvent(event)
