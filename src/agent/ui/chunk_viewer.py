"""切片阅读器窗口（检索即展示 / Retrieval-as-Presentation）。

用户点击回复末尾的「参考切片」气泡后打开：显示该切片完整原文，并可切换到
同文档的**前一片/后一片**——切片在边界处被截断时，用户可以直接续读，不必
回到对话里再检索一次。

数据来源：
* 首屏用气泡附件里携带的文本（离线可用、零等待）；
* 若能拿到 RAG 引擎，再按 (source, chunk_index) 取相邻切片，用于翻页；
  引擎不可用（同步中/已释放）时翻页按钮禁用，仅保留当前片阅读。
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout,
)


class ChunkViewerDialog(QDialog):
    """单个召回切片的原文阅读器（支持同文档前后翻页）。"""

    def __init__(self, parent, attachment: Dict[str, Any],
                 fetch: Optional[Callable[[str, int], Dict[str, Any]]] = None):
        super().__init__(parent)
        self._fetch = fetch
        self._source = attachment.get("source") or ""
        self._score = attachment.get("score")
        self._current: Dict[str, Any] = dict(attachment)
        self._prev: Optional[Dict[str, Any]] = None
        self._next: Optional[Dict[str, Any]] = None

        self.setWindowTitle("参考切片")
        self.resize(780, 640)
        lay = QVBoxLayout(self)
        lay.setSpacing(6)

        self._title = QLabel("")
        self._title.setStyleSheet("font-weight:600;")
        self._title.setWordWrap(True)
        lay.addWidget(self._title)

        self._meta = QLabel("")
        self._meta.setStyleSheet("color:#6b7280;")
        self._meta.setWordWrap(True)
        lay.addWidget(self._meta)

        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setPlaceholderText("（该切片无文本内容）")
        lay.addWidget(self._text, 1)

        row = QHBoxLayout()
        self._prev_btn = QPushButton("← 上一片")
        self._next_btn = QPushButton("下一片 →")
        close_btn = QPushButton("关闭")
        row.addWidget(self._prev_btn)
        row.addWidget(self._next_btn)
        row.addStretch(1)
        row.addWidget(close_btn)
        lay.addLayout(row)

        self._prev_btn.clicked.connect(lambda: self._go(-1))
        self._next_btn.clicked.connect(lambda: self._go(1))
        close_btn.clicked.connect(self.close)

        self._show(self._current)
        self._refresh_window()

    # ------------------------------------------------------------------

    @property
    def _index(self) -> Optional[int]:
        try:
            return int(self._current.get("chunk_index"))
        except (TypeError, ValueError):
            return None

    def _doc_name(self) -> str:
        return os.path.basename(self._source) or "（未知文档）"

    def _show(self, chunk: Dict[str, Any]) -> None:
        """把一片切片渲染到窗口（只换内容，不改窗口尺寸/位置）。"""
        self._current = dict(chunk)
        idx = self._index
        score = chunk.get("score", self._score)
        score_s = ""
        try:
            if score is not None:
                score_s = f" · 相关度 {float(score):.4f}"
        except (TypeError, ValueError):
            score_s = ""
        self._title.setText(f"📄 {self._doc_name()}")
        pos_s = f"第 {idx + 1} 片" if idx is not None else "片段"
        self._meta.setText(f"{pos_s}{score_s}")
        self._text.setPlainText(chunk.get("text") or "")
        self._text.moveCursor(QTextCursor.MoveOperation.Start)

    def _refresh_window(self) -> None:
        """拉取当前片的前/后各一片（失败则禁用翻页，不影响阅读）。"""
        self._prev = self._next = None
        if (self._fetch is not None and self._source
                and self._index is not None):
            try:
                win = self._fetch(self._source, self._index) or {}
            except Exception:
                win = {}
            cur = win.get("current")
            # 库内文本是权威版本；气泡里的文本只是离线兜底。
            if cur and (cur.get("text") or "").strip():
                self._show(cur)
            self._prev = win.get("prev")
            self._next = win.get("next")
        self._update_buttons()

    def _update_buttons(self) -> None:
        self._prev_btn.setEnabled(self._prev is not None)
        self._next_btn.setEnabled(self._next is not None)
        if self._fetch is None:
            self._meta.setToolTip("知识库引擎不可用，无法翻页")
        else:
            self._meta.setToolTip("")

    def _go(self, delta: int) -> None:
        if self._index is None:
            return
        target = self._index + delta
        cached = self._next if delta > 0 else self._prev
        try:
            cached_idx = int(cached.get("chunk_index")) if cached else None
        except (TypeError, ValueError):
            cached_idx = None
        if cached is not None and cached_idx == target:
            self._show(cached)
            self._refresh_window()
            return
        # 预取的相邻片不可用 → 直接按目标序号取；取不到就保持当前片。
        if self._fetch is None or not self._source:
            return
        try:
            win = self._fetch(self._source, target) or {}
        except Exception:
            win = {}
        if win.get("current"):
            self._show(win["current"])
            self._prev = win.get("prev")
            self._next = win.get("next")
            self._update_buttons()

    # ------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        # 方向键翻页；Esc 关闭（QDialog 默认行为）。
        if event.key() == Qt.Key.Key_Left:
            self._go(-1)
            event.accept()
            return
        if event.key() == Qt.Key.Key_Right:
            self._go(1)
            event.accept()
            return
        super().keyPressEvent(event)
