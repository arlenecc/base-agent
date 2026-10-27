"""文件转 Markdown 转换工具窗口。

从配置面板「PDF OCR 设置」一行末尾的「转换工具」按钮打开。用途是让用户在
入库之外，单独把各类文件（PDF / EPUB / Word / Excel / PPT / HTML / 图片 / TXT
等）批量提取转换成 Markdown，并保存到指定目录。

转换逻辑**完全复用** RAG 入库链路的同一套处理（解析、提取、图片归一化、
图片识别、结果清洗、Markdown 转换），所以这里得到的 Markdown 与真正入库时
是一致的：

- PDF / Office / HTML / DOCX / TEX：走 ``rag.engine._extract_with_docling``
  的「统一引擎回退链」（docling / marker / ovis），其扫描页 OCR 与图片归一化
  已在引擎内部完成；
- 其余格式（EPUB / TXT / CSV 等）回退到 ``rag.parsers.extract_text`` 纯文本提取；
- 图片（jpg/png/...）走 ``rag.ocr_tester.recognize_file`` 做图片归一化 +
  OCR 识别；
- 清洗 / Markdown 转换复用 ``rag.cleaner`` 与 ``rag.document_outline`` 的
  ``clean_docling_markdown`` / ``clean_text`` / ``normalize_markdown`` /
  ``plain_text_to_markdown``。

线程模型：转换在 ``ConvertWorker`` 里跑（文档解析 / OCR 都是阻塞且耗时的），
逐文件 emit 回 UI 线程；关闭窗口时先停 worker 再销毁控件，否则信号会打进已
释放的 C++ 对象（与 ``ocr_test_dialog`` 同理）。
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QProgressBar,
    QPushButton, QTextEdit, QVBoxLayout, QWidget,
)

from ..atomic_io import atomic_write_text
from ..rag import parsers
from ..rag.parsers import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

# 转换工具额外支持的图片格式（RAG 入库不收图片，但转换工具要支持）。
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
_CONVERT_EXTS = set(SUPPORTED_EXTENSIONS) | _IMAGE_EXTS

# 文件选择过滤器（选择文件对话框）。
_FILE_FILTER = (
    "所有支持的文件 (*.pdf *.epub *.docx *.xlsx *.pptx *.html *.htm *.txt "
    "*.md *.markdown *.csv *.tsv *.tex *.rst *.jpg *.jpeg *.png *.bmp *.tif "
    "*.tiff *.webp);;"
    "PDF (*.pdf);;EPUB (*.epub);;Word (*.docx);;Excel (*.xlsx);;"
    "PPT (*.pptx);;网页 (*.html *.htm);;文本 (*.txt *.md *.markdown *.csv "
    "*.tsv *.rst);;图片 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp);;"
    "所有文件 (*)"
)


def _supported(filepath: str) -> bool:
    return os.path.splitext(filepath)[1].lower() in _CONVERT_EXTS


def _pick_image_engine(config) -> str:
    """为图片 OCR 选一个可用引擎：优先配置里的 ovis，否则 Apple Vision，再次 RapidOCR。

    与配置面板「PDF 解析引擎 = ovis」时图片识别走 OvisOCR2 的取舍一致。
    """
    from ..rag import ocr_tester
    pdf_engine = (getattr(config, "rag_pdf_engine", "docling") or "").strip()
    status = ocr_tester.engine_status()
    if pdf_engine == "ovis" and status.get(ocr_tester.OVIS, "x") == "":
        return ocr_tester.OVIS
    if status.get(ocr_tester.APPLE, "x") == "":
        return ocr_tester.APPLE
    if status.get(ocr_tester.RAPID, "x") == "":
        return ocr_tester.RAPID
    return ocr_tester.APPLE


def _convert_image(path: str, config) -> str:
    """单张图片：统一归一化 + OCR 识别 → Markdown（清洗 / 转 MD 复用现有逻辑）。"""
    from ..rag import ocr_tester
    engine = _pick_image_engine(config)
    preprocess = bool(getattr(config, "rag_ocr_preprocess", True))
    binarize = bool(getattr(config, "rag_ocr_binarize", True))
    results = ocr_tester.recognize_file(
        path, engine, preprocess=preprocess, binarize=binarize)
    md = "\n\n".join(r.markdown for r in results
                     if getattr(r, "markdown", None) and r.markdown.strip())
    if not md.strip():
        return ""
    from ..rag.cleaner import clean_text, normalize_markdown
    return normalize_markdown(clean_text(md))


def convert_file(path: str, config) -> str:
    """把单个文件转成清洗后的 Markdown 文本（沿用 RAG 入库处理逻辑）。

    返回 Markdown 字符串；提取不到内容时返回空串。
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in _IMAGE_EXTS:
        return _convert_image(path, config)

    # 主路径：统一引擎回退链（PDF / Office / HTML / DOCX / TEX），已含扫描页
    # OCR 与图片归一化；其余格式回退到按格式解析器纯文本提取。
    try:
        from ..rag.engine import _extract_with_docling
        text, is_markdown = _extract_with_docling(path)
    except Exception as e:  # 兜底：解析器级异常，再退一步用纯文本提取
        logger.debug("转换工具：统一引擎失败，回退纯文本提取 %s: %s",
                     os.path.basename(path), e)
        try:
            text, is_markdown = parsers.extract_text(path), False
        except Exception as e2:
            raise RuntimeError(f"无法解析 {os.path.basename(path)}：{e2}") from e2

    if not text or not text.strip():
        return ""

    if is_markdown:
        # docling/marker/ovis/pandoc 已输出规范 Markdown：只做轻清洗（去注释/
        # 去 <script> 等），不剥 Markdown 结构标签。
        from ..rag.cleaner import clean_docling_markdown
        return clean_docling_markdown(text)

    # 纯文本 / 解析回退路径：先尝试启发式还原章节标题（# 行），再清洗、转 MD。
    from ..rag.cleaner import clean_text, normalize_markdown
    from ..rag.document_outline import (
        has_markdown_headings, plain_text_to_markdown,
    )
    if not has_markdown_headings(text):
        marked = plain_text_to_markdown(text)
        if marked != text:
            text = marked
    cleaned = clean_text(text)
    return normalize_markdown(cleaned)


class ConvertWorker(QThread):
    """后台逐文件转换。

    信号：
    - progress(done, total, filename, stage)：整体进度 + 当前文件 + 当前环节；
    - file_done(filename, ok, info)：单文件完成（成功/失败 + 信息）；
    - log(line)：追加日志文本；
    - finished(summary)：全部完成（{total, ok, failed}）；
    - failed(msg)：整体失败。
    """

    progress = pyqtSignal(int, int, str, str)
    file_done = pyqtSignal(str, bool, str)
    log = pyqtSignal(str)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, files: List[str], save_dir: str, base_dir: str,
                 config, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._files = files
        self._save_dir = save_dir
        self._base_dir = base_dir
        self._config = config
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def _out_path(self, filepath: str) -> str:
        rel = os.path.relpath(filepath, self._base_dir)
        stem, _ = os.path.splitext(rel)
        out = os.path.join(self._save_dir, stem + ".md")
        return out

    def run(self) -> None:  # type: ignore[override]
        total = len(self._files)
        ok = 0
        failed = 0
        try:
            for i, f in enumerate(self._files):
                if self._cancel:
                    break
                name = os.path.basename(f)
                self.progress.emit(i, total, name, "解析 / 提取（含图片识别）")
                self.log.emit(f"[{i + 1}/{total}] 解析：{name}")
                try:
                    md = convert_file(f, self._config)
                except Exception as e:
                    failed += 1
                    self.file_done.emit(name, False, str(e))
                    self.log.emit(f"  ✗ 失败：{e}")
                    continue

                if self._cancel:
                    break

                self.progress.emit(i, total, name, "结果清洗 / Markdown 转换")
                if not md.strip():
                    failed += 1
                    self.file_done.emit(name, False, "提取内容为空")
                    self.log.emit("  ✗ 提取内容为空，跳过写入")
                    continue

                self.progress.emit(i, total, name, "写入文件")
                out = self._out_path(f)
                try:
                    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
                    atomic_write_text(out, md)
                except OSError as e:
                    failed += 1
                    self.file_done.emit(name, False, f"写入失败：{e}")
                    self.log.emit(f"  ✗ 写入失败：{e}")
                    continue

                ok += 1
                chars = len(md)
                self.file_done.emit(name, True, f"{chars} 字符 → {out}")
                self.log.emit(f"  ✓ 已保存：{out}（{chars} 字符）")

            done = ok + failed
            # 整体进度条推进到 100%（或全部已完成的数量）。
            self.progress.emit(min(done, total), total, "", "完成")
            self.finished.emit({"total": total, "ok": ok, "failed": failed})
        except Exception as e:
            logger.warning("转换工具执行失败: %s", e)
            self.failed.emit(f"{type(e).__name__}: {e}")


class ConvertToolDialog(QDialog):
    """文件转 Markdown 转换工具窗口（配置面板「转换工具」按钮打开）。"""

    def __init__(self, main_window, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._main_window = main_window
        self._config = main_window.config
        self._worker: Optional[ConvertWorker] = None

        self.setWindowTitle("文件转 Markdown 转换工具")
        self.resize(900, 620)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        # ---- 顶部第一行：待转换文件 + 路径选取 ----
        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(QLabel("待转换文件"))
        self.src_edit = QLineEdit(placeholderText="选择单个文件，或指定一个目录")
        self.src_edit.setReadOnly(True)
        top.addWidget(self.src_edit, 1)
        self.pick_file_btn = QPushButton("选择文件")
        self.pick_file_btn.setFixedWidth(84)
        self.pick_file_btn.clicked.connect(self._on_pick_file)
        self.pick_dir_btn = QPushButton("选择目录")
        self.pick_dir_btn.setFixedWidth(84)
        self.pick_dir_btn.clicked.connect(self._on_pick_dir)
        top.addWidget(self.pick_file_btn)
        top.addWidget(self.pick_dir_btn)
        lay.addLayout(top)

        # ---- 顶部第二行：保存目录 + 路径选择 ----
        mid = QHBoxLayout()
        mid.setSpacing(8)
        mid.addWidget(QLabel("保存目录"))
        self.out_edit = QLineEdit(placeholderText="转换结果（.md）保存目录")
        self.out_edit.setReadOnly(True)
        mid.addWidget(self.out_edit, 1)
        self.pick_out_btn = QPushButton("选择目录")
        self.pick_out_btn.setFixedWidth(84)
        self.pick_out_btn.clicked.connect(self._on_pick_out)
        mid.addWidget(self.pick_out_btn)
        lay.addLayout(mid)

        # ---- 顶部第三行：开始转换 / 停止 ----
        bar = QHBoxLayout()
        bar.setSpacing(8)
        self.start_btn = QPushButton("开始转换")
        self.start_btn.setFixedWidth(110)
        self.start_btn.setStyleSheet("font-weight:600;")
        self.start_btn.clicked.connect(self._on_start_or_stop)
        bar.addWidget(self.start_btn)
        self.file_count_label = QLabel("未选择文件")
        bar.addWidget(self.file_count_label, 1)
        lay.addLayout(bar)

        # ---- 进度条 + 状态 ----
        prog = QHBoxLayout()
        prog.setSpacing(8)
        self.status_label = QLabel("就绪")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        prog.addWidget(self.status_label, 1)
        prog.addWidget(self.progress)
        lay.addLayout(prog)

        # ---- 日志 ----
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("转换过程日志（每个文件的处理结果）")
        lay.addWidget(self.log_view, 1)

        # ---- 底部：关闭 ----
        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        bottom.addStretch(1)
        close_btn = QPushButton("关闭")
        close_btn.setFixedWidth(84)
        close_btn.clicked.connect(self.close)
        bottom.addWidget(close_btn)
        lay.addLayout(bottom)

    # ------------------------------------------------------------------
    # 选择
    # ------------------------------------------------------------------

    def _on_pick_file(self) -> None:
        start = (os.path.dirname(self.src_edit.text())
                 if self.src_edit.text() else os.path.expanduser("~"))
        path, _ = QFileDialog.getOpenFileName(
            self, "选择待转换文件", start, _FILE_FILTER)
        if path:
            self.src_edit.setText(os.path.abspath(path))
            self._refresh_file_count()

    def _on_pick_dir(self) -> None:
        start = (self.src_edit.text()
                 if self.src_edit.text() else os.path.expanduser("~"))
        d = QFileDialog.getExistingDirectory(self, "选择待转换目录", start)
        if d:
            self.src_edit.setText(os.path.abspath(d))
            self._refresh_file_count()

    def _on_pick_out(self) -> None:
        start = (self.out_edit.text()
                 if self.out_edit.text() else os.path.expanduser("~"))
        d = QFileDialog.getExistingDirectory(self, "选择保存目录", start)
        if d:
            self.out_edit.setText(os.path.abspath(d))

    def _collect_files(self) -> List[str]:
        src = self.src_edit.text().strip()
        if not src:
            return []
        if os.path.isfile(src):
            return [src] if _supported(src) else []
        if os.path.isdir(src):
            files: List[str] = []
            for root, _dirs, names in os.walk(src):
                for n in names:
                    if n.startswith("."):
                        continue
                    p = os.path.join(root, n)
                    if _supported(p):
                        files.append(p)
            return sorted(files)
        return []

    def _refresh_file_count(self) -> None:
        files = self._collect_files()
        if not files:
            self.file_count_label.setText("未选择支持的文件")
        else:
            self.file_count_label.setText(f"待转换：{len(files)} 个文件")

    # ------------------------------------------------------------------
    # 开始 / 停止
    # ------------------------------------------------------------------

    def _on_start_or_stop(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._stop_worker()
            self.status_label.setText("已停止（已完成的文件已保存）")
            self.start_btn.setText("开始转换")
            return
        self._start()

    def _start(self) -> None:
        src = self.src_edit.text().strip()
        out = self.out_edit.text().strip()
        if not src:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, "提示", "请先选择待转换文件或目录。")
            return
        if not out:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, "提示", "请先选择转换结果保存目录。")
            return
        files = self._collect_files()
        if not files:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "提示",
                "所选路径下没有支持的文件（支持 PDF/EPUB/Word/Excel/PPT/"
                "HTML/图片/TXT 等）。")
            return

        # 自动保存并采用配置主窗口内的相关设置（PDF 引擎 / OCR 模型 / 归一化 /
        # 预处理等），让转换过程与「应用配置」后完全一致。
        try:
            self._main_window.sync_rag_config_to_rag_layer()
        except Exception as e:  # 注入失败不应阻断转换
            logger.debug("转换工具：RAG 配置注入失败（忽略）: %s", e)

        base_dir = src if os.path.isdir(src) else os.path.dirname(src)
        try:
            os.makedirs(out, exist_ok=True)
        except OSError as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "目录无效", f"无法创建保存目录：\n{e}")
            return

        self.log_view.clear()
        self.log_view.append(f"共 {len(files)} 个文件，保存目录：{out}")
        self.progress.setRange(0, len(files))
        self.progress.setValue(0)
        self.status_label.setText("准备中…")

        self._worker = ConvertWorker(files, out, base_dir, self._config, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.file_done.connect(self._on_file_done)
        self._worker.log.connect(self._on_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()
        self.start_btn.setText("停止")
        self.pick_file_btn.setEnabled(False)
        self.pick_dir_btn.setEnabled(False)
        self.pick_out_btn.setEnabled(False)

    def _stop_worker(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is None:
            return
        for sig in (worker.progress, worker.file_done, worker.log,
                    worker.finished, worker.failed):
            try:
                sig.disconnect()
            except (TypeError, RuntimeError):
                pass
        worker.cancel()
        if worker.isRunning():
            worker.wait(5000)
        if worker.isRunning():
            worker.terminate()
            worker.wait(1000)

    # ------------------------------------------------------------------
    # 信号
    # ------------------------------------------------------------------

    def _on_progress(self, done: int, total: int, filename: str,
                     stage: str) -> None:
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(min(done, total))
        if filename:
            self.status_label.setText(
                f"处理：{filename} · 环节：{stage} · 总进度 {done}/{total}")
        else:
            self.status_label.setText(
                f"完成 · 总进度 {done}/{total}")

    def _on_file_done(self, filename: str, ok: bool, info: str) -> None:
        pass  # 日志已在 _on_log 中追加

    def _on_log(self, line: str) -> None:
        self.log_view.append(line)
        # 跟随最新日志
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_finished(self, summary: dict) -> None:
        self.start_btn.setText("开始转换")
        self.pick_file_btn.setEnabled(True)
        self.pick_dir_btn.setEnabled(True)
        self.pick_out_btn.setEnabled(True)
        self.status_label.setText(
            f"完成：共 {summary['total']} 个，成功 {summary['ok']} 个，"
            f"失败 {summary['failed']} 个")
        self.log_view.append(
            f"==== 全部完成：成功 {summary['ok']} / 失败 {summary['failed']} ====")

    def _on_failed(self, msg: str) -> None:
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.warning(self, "转换失败", msg)
        self.status_label.setText(f"转换失败：{msg}")
        self.start_btn.setText("开始转换")
        self.pick_file_btn.setEnabled(True)
        self.pick_dir_btn.setEnabled(True)
        self.pick_out_btn.setEnabled(True)

    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # type: ignore[override]
        """先停 worker 再销毁：转换线程晚一步 emit 会打进已释放的控件。"""
        self._stop_worker()
        super().closeEvent(event)
