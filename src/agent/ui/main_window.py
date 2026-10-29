"""PyQt6 main window: top bar with config/log entries, chat column, and
plan/thinking column.

The agent runs in a background QThread; a signal Bridge carries streaming
events to the UI thread. Blocking confirm/ask_user dialogs are synchronised
with threading.Event so the worker blocks until the user answers.

布局（本次调整）：
    对话标题行      「对话」+ 清空对话 / 终止对话 + ⚙ 配置 / 日志（全部并入一行）
    左栏            对话区（占满）→ token 速度（居中）→ 发送框 + 发送按钮（贴底）
    右栏            执行计划面板（上）→ 思考过程（下）

原顶部配置区已整体移入 ``ConfigDialog``；原右栏日志面板移入 ``LogWindow``；
思考过程调到右栏原日志的位置，左栏整块都给对话区；token 速度/总量居中放在
发送框与发送按钮上方。配置 / 日志入口并入对话标题行，不再单独占一条顶部工具条。
"""
from __future__ import annotations

import io
import logging
import os
import queue
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer
from PyQt6.QtGui import QFont, QTextCursor, QKeyEvent
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QLabel, QLineEdit, QPushButton, QComboBox, QTextEdit,
    QMessageBox, QInputDialog, QPlainTextEdit, QProgressBar,
    QStatusBar, QFileDialog, QDialog, QGroupBox, QCheckBox,
)

from ..agent import Agent, AgentCallbacks
from ..config import AgentConfig
from ..llm_client import LLMClient, LLMError
from ..skills import Skill
from ..tools import ToolRegistry
from ..rag.engine import RAGEngine, effective_embedding_model
from ..rag.vector_store import DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANK_MODEL
from .ocr_test_dialog import OcrTestDialog
from .convert_tool_dialog import ConvertToolDialog


CONFIG_PATH = os.path.expanduser("~/.base-agent/config.json")

logger = logging.getLogger(__name__)

# 配置窗口「模型类型」下拉标签 ↔ 本地后端标识的映射。
# 用户可见标签用 MLX-LM / MLX-VLM / ONNX / GGUF（旧的 llama_cpp 已改名 GGUF）。
_LLM_TYPE_TO_BACKEND = {"MLX-LM": "mlx", "MLX-VLM": "mlx", "ONNX": "onnx", "GGUF": "gguf"}
_BACKEND_TO_LLM_TYPE = {"mlx": "MLX-LM", "onnx": "ONNX", "gguf": "GGUF"}

# PDF 解析引擎下拉标签 ↔ 内部标识（#133：rag_pdf_engine 取值 docling/marker/ovis）。
_PDF_DISPLAY_TO_VALUE = {"docling": "docling", "marker": "marker", "OvisOCR": "ovis"}
_PDF_VALUE_TO_DISPLAY = {"docling": "docling", "marker": "marker", "ovis": "OvisOCR"}

# 本地模型加载成功后的按钮样式：文字改为「重新加载」+ 绿色背景，代表模型可用。
_MODEL_READY_STYLE = "font-weight:600; background-color:#16a34a; color:#ffffff;"
_MODEL_IDLE_STYLE = "font-weight:600;"


def _impl_to_provider_backend(impl: str):
    """Embedding/Reranker 实现下拉标签 → (provider, backend)。

    fastembed → 走网络基线（provider=api）；mlx/onnx/gguf → 本地加载。
    """
    if impl == "fastembed":
        return "api", "mlx"
    return "local", impl


def _provider_backend_to_impl(provider: str, backend: str) -> str:
    """(provider, backend) → Embedding/Reranker 实现下拉标签。"""
    return backend if provider == "local" else "fastembed"


# RAG / sync 相关模块的 logger 名称前缀，用于 UI 日志面板过滤
# 覆盖 RAG 全链路：同步引擎、向量库、解析器、清洗器、切片器、依赖检查、工具调用
#
# logger 名取自各模块的 __name__，而包名随运行方式变化：
#   - python main.py / pip 安装 / PyInstaller 冻结 → "agent.rag.engine"
#   - 从仓库根目录以 src.agent 导入（如部分测试）   → "src.agent.rag.engine"
# 这里从本模块的 __name__ 动态推导包根，写死 "src.agent.*" 会导致
# 生产环境（包名为 agent.*）下过滤器和临时 handler 永远匹配不到任何
# 日志记录，UI 日志面板静默丢失全部 RAG 日志。
_PKG_ROOT = __name__.rsplit(".ui", 1)[0]
_RAG_LOG_PREFIXES = (
    f"{_PKG_ROOT}.rag",
    f"{_PKG_ROOT}.tools.rag_tool",
    f"{_PKG_ROOT}.tools.base",
    f"{_PKG_ROOT}.ui.main_window",
)
# RAG 子模块 logger 全名（同步 worker 临时 handler / 日志桥接 level 设置共用）
_RAG_SUB_LOGGERS = tuple(
    f"{_PKG_ROOT}.rag.{m}"
    for m in ("engine", "vector_store", "chunker", "cleaner", "parsers", "deps")
)

# QSignalLogHandler 实例引用，保存在模块级别以便 closeEvent 清理。
# _setup_log_bridge() 创建并赋值，closeEvent() 从 root logger 移除。
_LOG_HANDLER: Optional[Any] = None

# 「终止对话」看门狗：协作式取消在这个时长内没生效就强制结束线程。
# 太长会让用户以为卡死，太短会在正常的长工具调用（如 30s 的 code_run）里误杀。
_STOP_FORCE_AFTER_MS = 30_000

# 流式文本的合并刷新间隔（ms）。~25fps —— 再低肉眼可见卡顿，再高则失去
# 流式观感。目的是把每秒数百次 QTextDocument 操作压到每秒 25 次。
_STREAM_FLUSH_MS = 40


class QSignalLogHandler(logging.Handler, QObject):
    """将 Python logging 日志桥接到 Qt UI 的 log_view。

    通过 pyqtSignal 将日志消息安全地传递到 UI 线程，避免跨线程
    直接操作 QPlainTextEdit 导致的崩溃。

    必须同时继承 QObject 和 logging.Handler，因为 pyqtSignal
    需要 QObject 作为基类才能正常工作。
    """

    message_received = pyqtSignal(str)

    def __init__(self) -> None:
        # 同时继承 QObject 和 logging.Handler 时，super().__init__() 只调用
        # MRO 中第一个父类的 __init__。logging.Handler.__init__() 不会调用
        # super().__init__()，因此 QObject.__init__() 被跳过，导致 pyqtSignal
        # 初始化失败。必须显式调用两个父类的 __init__。
        QObject.__init__(self)
        logging.Handler.__init__(self)
        self.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
        self._emit_count = 0
        self._emit_errors = 0
        self._sync_active = False  # 同步期间为 True，跳过 RAG 日志避免并发崩溃

    def close(self) -> None:
        """Override close() to also remove this handler from
        logging._handlerList IN-PLACE.

        The base Handler.close() only removes the handler from the _handlers
        dict (by name) and sets _closed = True.  But logging.shutdown() —
        registered via atexit — iterates _handlerList, a global list of weak
        references.  If the underlying Qt C++ object has already been
        deleted by the time atexit fires, getattr(h, 'flushOnClose') raises
        RuntimeError: wrapped C/C++ object has been deleted.

        Note: we must use slice assignment (_handlerList[:] = [...]) to
        mutate the list in-place, because shutdown()'s default argument
        binds to the list object at function definition time.  Replacing
        _handlerList with a new list would leave the default argument
        pointing at the stale list.
        """
        try:
            import logging as _logging
            _logging._handlerList[:] = [
                wr for wr in _logging._handlerList
                if wr() is not self
            ]
        except Exception:
            pass
        try:
            super().close()
        except RuntimeError:
            pass

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # 同步期间跳过 RAG 日志——这些日志由 _log_buffer + QTimer 处理，
            # 避免 QSignalLogHandler 的 queued signal 和 _poll_sync_logs 的
            # appendPlainText 同时操作 log_view 导致的并发崩溃。
            if self._sync_active and any(
                record.name.startswith(p) for p in _RAG_LOG_PREFIXES
            ):
                return
            msg = self.format(record)
            self._emit_count += 1
            self.message_received.emit(msg)
        except RuntimeError:
            # Qt 对象在 atexit/logging.shutdown 时可能已被销毁，
            # 忽略以避免 "wrapped C/C++ object has been deleted" 异常。
            pass
        except Exception:
            self._emit_errors += 1
            self.handleError(record)

    def flush(self) -> None:
        """Override flush to prevent RuntimeError during logging.shutdown."""
        try:
            super().flush()
        except RuntimeError:
            pass


def _sanitize_stream_text(text: str) -> str:
    """Normalize line endings and drop overstrike carriage returns.

    Some reasoning models emit bare ``\\r`` characters (spinner / line-overwrite
    animation). Inserting those into a text widget returns the cursor to the
    start of the current line, so subsequent text overprints what is already
    there -> visual overlap. We turn CRLF into ``\\n`` and any lone ``\\r`` into
    ``\\n`` so each chunk lands on its own line and nothing overprints.
    """
    if not text:
        return ""
    return text.replace("\r\n", "\n").replace("\r", "\n")


# ---------------------------------------------------------------------------
# 执行计划面板：plan 快照 → 带状态标记的 HTML / 进度摘要
#
# 数据来自 Bridge.plan 信号携带的**快照**（不能去读 agent.session：工作线程
# 的 update_plan 工具会原地改 step dict，跨线程读共享可变状态会渲染出半新半旧
# 的中间态）。渲染与 kernel.session.render_plan_dict 的纯文本格式逐行对应，
# 额外按状态着色并把「正在进行」的步骤高亮。
# ---------------------------------------------------------------------------

# 终态（计入完成数）与非终态
_PLAN_DONE_STATES = ("done", "blocked", "skipped")
_PLAN_OPEN_STATES = ("pending", "doing")

# 状态 → (标记, 颜色)
_PLAN_STATUS_STYLE = {
    "done": ("✓", "#059669"),      # 绿：已完成
    "doing": ("▶", "#b45309"),     # 琥珀：正在进行（高亮）
    "pending": (" ", "#6b7280"),
    "blocked": ("✗", "#dc2626"),
    "skipped": ("⊘", "#6b7280"),
}


def _plan_steps(plan: Any) -> List[Dict[str, Any]]:
    if not isinstance(plan, dict):
        return []
    return [s for s in (plan.get("steps") or []) if isinstance(s, dict)]


def _plan_status(step: Dict[str, Any]) -> str:
    return str(step.get("status", "pending") or "pending").strip().lower()


def _plan_html(plan: Any) -> str:
    """把 plan 快照渲染成带状态标记的 HTML（右栏「执行计划」面板）。"""
    steps = _plan_steps(plan)
    goal = str(plan.get("goal") or "").strip() if isinstance(plan, dict) else ""
    if not goal and not steps:
        return ""
    import html as _html

    def esc(value: Any) -> str:
        return _html.escape(str(value))

    lines: List[str] = []
    if goal:
        lines.append(f'<p style="margin:2px 0;"><b>Goal:</b> {esc(goal)}</p>')
    doing_step: Optional[Dict[str, Any]] = None
    for s in steps:
        status = _plan_status(s)
        mark, color = _PLAN_STATUS_STYLE.get(status, _PLAN_STATUS_STYLE["pending"])
        title = str(s.get("title", "") or "")
        note = str(s.get("note", "") or "")
        line = f"- [{mark}] {esc(s.get('id', '?'))}. {esc(title)}"
        if note:
            # 两个前导空格必须原样保留（与纯文本版 render_plan_dict 一致），
            # HTML 默认折叠连续空白 → 用 white-space:pre-wrap 保住。
            line += (
                '<span style="white-space:pre-wrap;color:#6b7280;">'
                f'  # {esc(note)}</span>'
            )
        if status == "doing" and doing_step is None:
            doing_step = s
            # 当前正在进行的步骤：琥珀底纹 + 明确标记
            lines.append(
                '<p style="margin:2px 0; padding:1px 4px; background:#fef3c7; '
                f'color:#b45309; font-weight:600;">{line}  ◀ 正在进行</p>'
            )
        else:
            lines.append(f'<p style="margin:2px 0; color:{color};">{line}</p>')
    if doing_step is not None:
        now = f"{doing_step.get('id', '?')}. {doing_step.get('title', '')}"
        lines.append(f'<p style="margin:2px 0; color:#b45309;"><b>Now:</b> {esc(now)}</p>')
    open_steps = [s for s in steps if _plan_status(s) in _PLAN_OPEN_STATES]
    if open_steps:
        lines.append(
            '<p style="margin:2px 0; color:#6b7280;">'
            f"Remaining: {len(open_steps)} step(s) not finished</p>"
        )
    if steps:
        done = sum(1 for s in steps if _plan_status(s) in _PLAN_DONE_STATES)
        lines.append(
            '<p style="margin:2px 0; color:#374151;"><b>进度:</b> '
            f"{done}/{len(steps)} 步已完成</p>"
        )
    return "\n".join(lines)


def _plan_progress_text(plan: Any) -> str:
    """计划面板标题栏的进度摘要：``2/5 步完成 · 进行中：部署``。"""
    steps = _plan_steps(plan)
    if not steps:
        return ""
    done = sum(1 for s in steps if _plan_status(s) in _PLAN_DONE_STATES)
    text = f"{done}/{len(steps)} 步完成"
    doing = next((s for s in steps if _plan_status(s) == "doing"), None)
    if doing is not None:
        title = str(doing.get("title", "") or "")
        if len(title) > 16:
            title = title[:16] + "…"
        text += f" · 进行中：{title}"
    return text


# ===========================================================================
# Bridge: thread-safe signal channel between worker and UI
# ===========================================================================


class _SummaryStripper:
    """把协议标签 ``<summary>…</summary>`` 从流式文本里剔掉。

    协议要求模型在回复**开头**输出这个标签（L2 摘要的原料），但它是给内核
    用的机器标记，不该出现在对话气泡里。流式场景下标签可能跨 chunk 到达，
    因此用一个带 carry 的小状态机处理，并对末尾「可能是标签前缀」的片段
    做保留，避免把 ``<summ`` 当成正文输出、下一 chunk 又来一个 ``ary>``。

    标签**内部**的内容先缓存而不是立刻丢弃：若到回合结束标签始终没闭合
    （模型没遵守协议），说明它根本不是标签，要把吞掉的内容还给用户——
    直接丢弃会让整条回复凭空消失。
    """

    _OPEN = "<summary>"
    _CLOSE = "</summary>"

    def __init__(self) -> None:
        self._inside = False
        self._carry = ""
        self._inside_buf: List[str] = []

    def feed(self, text: str) -> str:
        buf = self._carry + (text or "")
        self._carry = ""
        if not buf:
            return ""
        out: List[str] = []
        i, n = 0, len(buf)
        while i < n:
            if self._inside:
                # 等闭合标签
                j = buf.find(self._CLOSE, i)
                if j >= 0:
                    self._inside = False
                    self._inside_buf = []           # 正常闭合 → 摘要正文丢弃
                    i = j + len(self._CLOSE)
                    continue
                # 未闭合：除尾部可能的部分闭合标签外，全部吞进摘要缓冲
                rest, carry = self._split_partial(buf, i, self._CLOSE)
                self._carry = carry
                self._inside_buf.append(rest)
                break
            # 未在摘要内：吞掉孤立的闭合标签（开标签落在推理流里时会出现）
            if buf.startswith(self._CLOSE, i):
                i += len(self._CLOSE)
                continue
            j = buf.find(self._OPEN, i)
            jc = buf.find(self._CLOSE, i)
            if jc != -1 and (j == -1 or jc < j):
                out.append(buf[i:jc])
                i = jc + len(self._CLOSE)
                continue
            if j >= 0:
                out.append(buf[i:j])
                self._inside = True
                i = j + len(self._OPEN)
                continue
            # 无完整标签：尾部可能是任一标签的**任意长度**真前缀
            rest, carry = self._split_partial(buf, i, self._OPEN, self._CLOSE)
            self._carry = carry
            out.append(rest)
            break
        return "".join(out)

    @staticmethod
    def _split_partial(buf: str, i: int, *tags: str) -> Tuple[str, str]:
        """返回 (可安全输出的部分, 尾部 carry)。

        流式 delta 可能把标签在**任意位置**切碎（如 ``…<sum`` + ``mary>…``）。
        carry 取 buf 尾部最长的、且是任一 tag 真前缀的后缀——旧实现只检查
        「恰好占满 tag_len-1 窗口」的情况，短前缀会把半个标签当正文发出，
        下一 chunk 的 ``mary>`` 就再也拼不回去。
        """
        n = len(buf)
        best = 0
        for tag in tags:
            for plen in range(min(len(tag) - 1, n - i), 0, -1):
                if tag.startswith(buf[n - plen:]):
                    if plen > best:
                        best = plen
                    break
        return buf[i:n - best], (buf[n - best:] if best else "")

    def flush(self) -> str:
        """收尾：标签始终没闭合时，把吞进去的内容还回去。"""
        c, self._carry = self._carry, ""
        if self._inside:
            self._inside = False
            rescued = "".join(self._inside_buf) + c
            self._inside_buf = []
            return rescued
        self._inside_buf = []
        return c


class Bridge(QObject):
    content = pyqtSignal(str)
    reasoning = pyqtSignal(str)
    tool_start = pyqtSignal(str, object)
    tool_end = pyqtSignal(str, object)
    log = pyqtSignal(str)
    usage = pyqtSignal(dict)
    speed = pyqtSignal(int, float)
    skill = pyqtSignal(object)
    plan = pyqtSignal(object)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    # 通用状态栏提示（worker / 预加载线程 → 主线程底部状态栏，跨线程安全入队）。
    status = pyqtSignal(str)
    # blocking requests (worker -> UI)
    confirm_request = pyqtSignal(str)
    ask_request = pyqtSignal(str)


class BridgeCallbacks(AgentCallbacks):
    """AgentCallbacks that forwards everything through the Bridge."""

    def __init__(self, bridge: Bridge):
        self.bridge = bridge
        self._confirm_event = threading.Event()
        self._ask_event = threading.Event()
        self._lock = threading.Lock()
        # 默认「拒绝」而不是「同意」：确认弹窗用于 shell_run / code_run 等
        # 高风险操作，若对话框因任何原因没有回话（超时、事件循环被卡住、
        # 信号丢失），安全的一侧是拒绝执行，而不是默默放行。
        self._confirm_result: bool = False
        self._ask_result: Optional[str] = None
        self._cancelled = False

    # streaming
    def on_content(self, text: str) -> None:
        self.bridge.content.emit(text)
    def on_reasoning(self, text: str) -> None:
        self.bridge.reasoning.emit(text)
    def on_tool_start(self, name: str, args: Dict[str, Any]) -> None:
        self.bridge.tool_start.emit(name, args)
    def on_tool_end(self, name: str, result) -> None:
        self.bridge.tool_end.emit(name, result)
    def on_log(self, line: str) -> None:
        self.bridge.log.emit(line)
        # 诊断开关：BASE_AGENT_KERNEL_LOG=1 时把内核每轮决策日志落盘，
        # 便于复现「问候语卡在反思门循环」类问题（不会进 stdout，不影响正常 UI）。
        if os.environ.get("BASE_AGENT_KERNEL_LOG"):
            try:
                with open("/tmp/base-agent-kernel.log", "a", encoding="utf-8") as _fk:
                    _fk.write(line + "\n")
            except Exception:
                pass
    def on_usage(self, usage: Dict[str, int]) -> None:
        self.bridge.usage.emit(usage)
    def on_token_speed(self, total_tokens: int, speed: float) -> None:
        self.bridge.speed.emit(total_tokens, speed)
    def on_skill_suggested(self, skill) -> None:
        self.bridge.skill.emit(skill)
    def on_plan_updated(self, plan) -> None:
        self.bridge.plan.emit(plan)
    def on_error(self, message: str) -> None:
        self.bridge.error.emit(message)
    def on_finished(self) -> None:
        self.bridge.finished.emit()

    # blocking dialogs (called from worker thread)
    # 等待用户回话的最长时间（秒）。超过则视为未获授权 → 拒绝 / 无回答。
    _DIALOG_TIMEOUT = 600.0

    def confirm(self, message: str) -> bool:
        if self._cancelled:
            return False
        self._confirm_event.clear()
        with self._lock:
            self._confirm_result = False
        self.bridge.confirm_request.emit(message)
        # Event.wait() 返回 False 表示超时——此时绝不能沿用默认/上一次结果，
        # 否则高风险操作会在用户没点确认的情况下被执行。
        if not self._confirm_event.wait(timeout=self._DIALOG_TIMEOUT):
            return False
        with self._lock:
            return self._confirm_result

    def ask_user(self, prompt: str) -> Optional[str]:
        if self._cancelled:
            return None
        self._ask_event.clear()
        with self._lock:
            self._ask_result = None
        self.bridge.ask_request.emit(prompt)
        if not self._ask_event.wait(timeout=self._DIALOG_TIMEOUT):
            return None
        with self._lock:
            return self._ask_result

    # setters called from UI thread
    def set_confirm_result(self, val: bool) -> None:
        with self._lock:
            self._confirm_result = val
        self._confirm_event.set()

    def set_ask_result(self, val: Optional[str]) -> None:
        with self._lock:
            self._ask_result = val
        self._ask_event.set()

    def reset(self) -> None:
        """Clear per-run state so the next turn starts from a clean slate.

        Without this, ``_cancelled`` stays True after 「终止对话」 and every
        later confirm/ask would be auto-declined; the stale ``_ask_result``
        would also be returned again for the next question.
        """
        self._cancelled = False
        with self._lock:
            self._confirm_result = False
            self._ask_result = None
        self._confirm_event.clear()
        self._ask_event.clear()

    def is_cancelled(self) -> bool:
        """供 Agent.run() 在流式/迭代/工具边界轮询（协作式取消）。"""
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True
        self._confirm_event.set()
        self._ask_event.set()


# ===========================================================================
# Worker: runs the agent in a background thread
# ===========================================================================


class AgentWorker(QThread):
    """常驻工作线程：所有用户消息都在**同一条线程**上执行 agent.run()。

    旧实现每回合新建 QThread。这不只是开销问题：MLX 的 KV 缓存绑定创建线程
    的 GPU stream（跨线程复用会抛 "There is no Stream(gpu, N) in current
    thread"，llm.py 据此按线程隔离缓存），每回合换线程 → 缓存每回合都被判定
    跨线程而丢弃 → 多轮对话退回全量预填充。常驻线程让 KV 复用真正命中
    （多轮场景 6.5s → 0.2s 量级）。

    生命周期：
    · ``set_input()`` 入队一条用户消息；``run()`` 循环逐条执行，每条结束
      emit ``bridge.finished``（UI 的 ``_on_finished`` 靠它复位 ``_busy``）。
    · ``shutdown()`` 入队哨兵让循环退出（closeEvent / 重建 agent 时调用）。
      正在执行的回合不受影响——协作取消交给 ``callbacks.cancel()``。
    · 强杀（``_force_stop_agent`` 的 terminate）会让线程死掉但 QThread 对象
      还在：置 ``_dead`` 后 ``set_input`` 拒绝新任务，``_on_send`` 换新线程。
    """

    def __init__(self, agent: Agent, callbacks: BridgeCallbacks):
        super().__init__()
        self._agent = agent
        self._callbacks = callbacks
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._dead = False

    def set_input(self, text: str) -> bool:
        """入队一条用户消息。线程已被强杀时返回 False（调用方需换新 worker）。"""
        if self._dead:
            return False
        self._queue.put(text)
        return True

    def shutdown(self) -> None:
        """请求常驻循环退出；正在执行的回合结束后生效。"""
        self._dead = True
        self._queue.put(None)

    def disconnect_bridge(self) -> None:
        """断开 ``_callbacks.bridge`` 的所有信号连接。

        AgentWorker 自身**不定义任何 pyqtSignal**——``run()`` 里 emit 的是
        ``self._callbacks.bridge.*``（Bridge 的信号）。``QObject.disconnect()``
        只断本对象自身的连接，对 worker 调用恒为 no-op。强杀 / 关窗口时若
        worker 线程还在跑并 emit bridge 信号，会打到已删除的 C++ 对象
        （``RuntimeError: wrapped C/C++ object has been deleted``，PyQt6 下
        可能直接 qFatal）。要断的必须是 bridge。
        """
        bridge = getattr(getattr(self, "_callbacks", None), "bridge", None)
        if bridge is not None:
            bridge.disconnect()

    def run(self) -> None:  # type: ignore[override]
        # 模型加载完成后本线程即启动，先做两件「空闲期预热」，都跑在 worker 线程、
        # 与真实生成同线程（缓存不被线程隔离丢弃），且二者串行避免并发抢同一份权重
        # （MLX 前向非线程安全）：
        #  (1) 用 system prompt 预填 KV（数秒）——首句直接命中缓存、跳过数千 token
        #      全量预填充；
        #  (2) warm_up 跑一次真实形状推理，触发 MLX 首推图编译（20-30s），首句不再
        #      等冷启动。
        # 两步都失败不致命；预填/编译完才进入消息循环，用户打字前本线程已空闲。
        # 但若用户手快、消息已经在队列里等着：跳过空闲期预热，先处理真实消息
        # （首句自己会把编译/预填一并付掉——反正躲不掉；让它干等 warm_up 完成是
        # 纯浪费，实测表现为「发了消息 18s 没动静」）。空队列时才值得预热。
        if not self._queue.empty():
            self._callbacks.bridge.status.emit(
                "已有消息在排队，跳过空闲期预热（首句将自行完成编译）")
            logger.info("AgentWorker: 队列已有待处理消息，跳过空闲期预热")
        else:
            try:
                self._callbacks.bridge.status.emit(
                    "系统提示预热中（预填上下文，请稍候）")
                self._agent.preload_kv()
                self._callbacks.bridge.status.emit("系统提示已预热，可开始对话")
            except Exception:
                self._callbacks.bridge.status.emit("系统提示预热失败（首句会偏慢）")
            # warm_up（首次推理编译）：与预填串行、同在 worker 线程，不另起并发抢权重。
            try:
                self._callbacks.bridge.status.emit(
                    "主模型预热中（首次推理编译，约 20-30s，请稍候）")
                self._agent.warm_up()
                self._callbacks.bridge.status.emit("主模型预热完成，可开始对话")
            except Exception as e:
                logger.warning("主模型预热失败(忽略): %s", e)
                self._callbacks.bridge.status.emit(
                    "预热失败，但仍可对话（首句会偏慢）")
        while True:
            text = self._queue.get()
            if text is None:
                break
            try:
                self._agent.run(text)
            except Exception as e:
                self._callbacks.bridge.error.emit(f"{type(e).__name__}: {e}")
            finally:
                self._callbacks.bridge.finished.emit()


class FetchModelsWorker(QThread):
    fetched = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, base_url: str, api_key: str):
        super().__init__()
        self._base_url = base_url
        self._api_key = api_key

    def run(self) -> None:  # type: ignore[override]
        client = None
        try:
            client = LLMClient(base_url=self._base_url, api_key=self._api_key, model="",
                               timeout=30.0, max_tokens=1)
            models = client.list_models()
            # 先取结果再 emit：emit 之后 close 若抛异常，错误会出现在
            # finished 之后无法显示。
            client.close()
            client = None
            self.fetched.emit(models)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
        finally:
            # close 不放进 try 的正常路径：list_models 对 4xx/5xx、网络错误
            # 都会抛异常，旧实现此时跳过了 close，httpx 连接池（socket +
            # 线程池）依赖 GC 兜底——每次「获取模型失败」泄漏一个。
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass


class VectorDedupWorker(QThread):
    """后台执行向量库去重：清理内容完全相同的重复向量。

    每组同文本只保留「入库时间早」的一份（markdown 缓存 mtime 为代理），
    其余删除，降低同段内容在检索里的双倍召回。全程只读 id/text/source/
    chunk_index 四列（不加载向量），分批删除。

    自定义信号统一 ``dedup_`` 前缀——绝不声明 ``finished``：那会 shadow
    QThread 内建 finished，挂在上面的 deleteLater 会在 run() 执行途中
    销毁线程对象（见 SyncKnowledgeWorker 的同款警告）。
    """

    dedup_log = pyqtSignal(str)
    dedup_progress = pyqtSignal(int, int, str)   # cur, total, msg
    dedup_finished = pyqtSignal(dict)            # {"scanned","groups","removed"}
    dedup_failed = pyqtSignal(str)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self._config = config

    def run(self):
        try:
            from ..rag.engine import RAGEngine
            eng = RAGEngine(
                workspace=self._config.workspace,
                knowledge_base=self._config.knowledge_base,
                embedding_model=getattr(self._config, "rag_embedding_model",
                                        DEFAULT_EMBEDDING_MODEL),
            )
            self.dedup_log.emit("[dedup] 开始扫描向量库（只读文本列，不加载向量）…")
            stats = eng.dedup_vectors(progress_callback=self._on_progress)
            self.dedup_finished.emit(stats)
        except Exception as e:
            logger.warning("VectorDedupWorker failed: %s", e)
            self.dedup_failed.emit(str(e))

    def _on_progress(self, cur: int, total: int, msg: str) -> None:
        self.dedup_progress.emit(cur, total, msg)


class SyncKnowledgeWorker(QThread):
    """Background worker for RAG knowledge base ingestion.

    IMPORTANT: Custom signals are prefixed with `sync_` to avoid shadowing
    QThread's built-in `finished` signal. Shadowing `finished` breaks Qt's
    thread lifecycle management — `deleteLater` connected to the custom
    signal fires *inside* run() (before cleanup), while the built-in
    `finished` (emitted after run() returns) is left with no listeners.
    """
    sync_finished = pyqtSignal(dict)   # stats dict
    sync_failed = pyqtSignal(str)       # error message
    sync_progress = pyqtSignal(int, int, str)  # done, total, current_file

    def __init__(self, workspace: str, knowledge_base: str, embedding_model: str = "",
                 force: bool = False, config=None):
        super().__init__()
        self._workspace = workspace
        self._knowledge_base = knowledge_base
        self._embedding_model = embedding_model
        self._force = force
        self._config = config
        self._engine: Optional[RAGEngine] = None
        # 日志缓冲区：子线程写入，主线程在 _poll_sync_logs 中消费。
        # 用 Lock 保护，避免多 worker 线程 append 与主线程清空之间的竞态。
        self._log_buffer: list = []
        self._log_buffer_lock = threading.Lock()

    def _emit_log(self, msg: str) -> None:
        """将日志追加到缓冲区，由 _poll_sync_logs 消费写入日志面板。"""
        with self._log_buffer_lock:
            self._log_buffer.append(msg)

    def run(self) -> None:  # type: ignore[override]
        self._emit_log("[rag] ────────── 同步引擎启动 ──────────")
        logger.info(
            "SyncKnowledgeWorker start: workspace=%s kb=%s force=%s embedding=%s",
            self._workspace, self._knowledge_base, self._force,
            self._embedding_model or "(default)",
        )
        # 安装临时 handler：捕获 RAG 各模块通过 logging 输出的日志
        # （如 vector_store 的向量化过程日志），写入 _log_buffer。
        # engine.py 的 _log() 则直接通过 log_callback 写 buffer，
        # 不依赖 logging handler。
        #
        # IMPORTANT: 临时 handler 只挂在父 logger `agent.rag` 上，利用 logging
        # 的传播机制捕获所有子模块（engine / vector_store / ...）的日志。
        # 若同时给父 logger 和每个子 logger 都挂 handler，子 logger 的消息
        # 会被自己的 handler 捕获一次，再传播到父 logger 的 handler 捕获一次，
        # 导致每条日志重复输出两遍（叠加 log_callback 即三遍）。
        _tmp_handler = logging.StreamHandler(io.StringIO())
        _tmp_handler.setLevel(logging.INFO)
        _tmp_handler.setFormatter(
            logging.Formatter("%(message)s")
        )
        def _on_rag_log(record):
            msg = _tmp_handler.format(record)
            with self._log_buffer_lock:
                self._log_buffer.append(msg)
        _tmp_handler.emit = _on_rag_log
        # 只挂父 logger；子 logger 保持 NOTSET（默认传播到父 logger）。
        _rag_loggers = [f"{_PKG_ROOT}.rag"]
        for _name in _rag_loggers:
            _l = logging.getLogger(_name)
            _l.setLevel(logging.INFO)
            _l.addHandler(_tmp_handler)
        emit_stats = None
        emit_error = None
        try:
            # 同步前先归还 MLX 的空闲缓冲：GUI 里主/辅模型常驻（MLX 统一内存
            # 不可换出），其推理留下的缓存会占用几百 MB~1GB 水位。同步本身
            # 还要叠加嵌入（~2GB）与扫描件 OCR（数 GB），起步水位越低越不容
            # 易触发系统内存压力。只清空闲缓冲，不影响已加载的模型权重。
            try:
                from ..rag.engine import configure_mlx_memory_limits, release_mlx_pool
                # MLX 默认 cache_limit = 统一内存的 ~75%，空闲 Metal buffer 会
                # 一路涨到上限才归还——16GB 机器上光缓存就能到 ~12GB（实测
                # 同步峰值 footprint 19GB 的主因）。按配置压到指定上限（默认
                # 2GB），空闲即还；2GB 比 1GB 给编译核更宽松缓存、略提速度。
                _cache_mb = int(getattr(self.config, "mlx_cache_limit_mb", 2000))
                configure_mlx_memory_limits(_cache_mb * 1_000_000)
                release_mlx_pool()
            except Exception as e:  # 防御：清理失败不该中断同步
                logger.debug("SyncKnowledgeWorker: pre-sync mlx cleanup failed: %s", e)
            # 解析并发说明：docling 转换器现为**并行实例池**（`_DOCLING_POOL_SIZE`，
            # 本机 16GB 默认 1）。2 个 MPS 实例与 MLX embedding 并发抢 GPU 会触发 GPU 重置
            # → torch-MPS 状态损坏 → 原生 SIGSEGV 硬崩（2026-10-25 实测多次，崩溃报告
            # Malloc 仅 3.8G、非 OOM），故 16GB 默认回到 1（1 个 MPS 实例稳定）。
            # ⚠️ 约束是 GPU 稳定性不是内存（每实例常驻权重约 3-3.5GB）；≥32GB 可试 2、
            # ≥64GB 可 4。embedding 已改为单实例（self._ef 共享单例，多实例无收益且有
            # 死锁/线程安全风险），store 仍串行。
            self._engine = RAGEngine(
                workspace=self._workspace,
                knowledge_base=self._knowledge_base,
                embedding_model=self._embedding_model,
                rerank_model=getattr(self._config, "rag_rerank_model",
                                     DEFAULT_RERANK_MODEL),
                rerank_enabled=bool(getattr(self._config, "rag_rerank_enabled", True)),
                # 切片大小/重叠从 config 接线（此前 GUI 同步路径没传，导致引擎
                # 用默认 1000，远超 bge 的 512 窗口、切片被静默截断——见
                # _clamp_chunk_size_to_model_window 兜底，但这里仍尊重用户配置）。
                chunk_size=getattr(self._config, "rag_chunk_size", 800),
                chunk_overlap=getattr(self._config, "rag_chunk_overlap", 50),
                overlap_percent=getattr(self._config, "rag_chunk_overlap_percent", 0.10),
                use_semantic=getattr(self._config, "rag_chunk_semantic", True),
                # 窗口夹紧用实际生效的本地模型（local 模式取 local_embedding_model），
                # 避免 rag_embedding_model 的遗留值（如 bge 目录名）把 800 夹成 510。
                clamp_model=effective_embedding_model(self._config),
            )
            self._emit_log("[rag] RAG 引擎初始化完成，开始同步...")
            stats = self._engine.ingest(
                force=self._force,
                progress_callback=self._on_progress,
                log_callback=self._emit_log,
            )
            logger.info(
                "SyncKnowledgeWorker done: found=%d extracted=%d skipped=%d deleted=%d duplicated=%d chunks=%d errors=%d cancelled=%s",
                stats.get("files_found", 0), stats.get("files_extracted", 0),
                stats.get("files_skipped", 0), stats.get("files_deleted", 0),
                stats.get("files_duplicated", 0),
                stats.get("chunks", 0), len(stats.get("errors", [])),
                stats.get("cancelled", False),
            )
            self._emit_log(
                f"[rag] 同步完成: 扫描{stats.get('files_found',0)}文件, "
                f"提取{stats.get('files_extracted',0)}, 跳过{stats.get('files_skipped',0)}, "
                f"切片{stats.get('chunks',0)}"
            )
            emit_stats = stats
        except Exception as e:
            self._emit_log(f"[rag] ❌ 同步失败: {e}")
            logger.error("SyncKnowledgeWorker failed: %s", e, exc_info=True)
            emit_error = f"{type(e).__name__}: {e}"
        finally:
            # 移除临时 handler
            for _name in _rag_loggers:
                try:
                    logging.getLogger(_name).removeHandler(_tmp_handler)
                except Exception:
                    pass
            # Release the VectorStore's resources (LanceDB connection, FastEmbed
            # ONNX model ~130MB, BGE reranker ~500MB). Without this, every sync
            # run leaks ~600MB of C++ heap memory because the ONNX Runtime
            # InferenceSession is not promptly GC'd by Python.
            if self._engine is not None:
                try:
                    self._engine.close()
                    logger.info("SyncKnowledgeWorker: RAG engine closed")
                except Exception as e:
                    logger.warning("SyncKnowledgeWorker: engine close failed: %s", e)
            # Emit the signal only after cleanup is complete so the main
            # thread's UI reset / reload_rag() runs against a fully released
            # worker-side engine.
            if emit_error is not None:
                logger.info("SyncKnowledgeWorker: emitting sync_failed")
                self.sync_failed.emit(emit_error)
            elif emit_stats is not None:
                logger.info("SyncKnowledgeWorker: emitting sync_finished with stats keys=%s",
                            list(emit_stats.keys()) if isinstance(emit_stats, dict) else type(emit_stats))
                self.sync_finished.emit(emit_stats)
            else:
                # 防御：如果 emit_error 和 emit_stats 都是 None（极端情况：
                # RAGEngine 构造或 ingest 抛出非 Exception 的 BaseException，
                # 如 KeyboardInterrupt），确保 UI 不会永远卡在"同步中"。
                logger.error(
                    "SyncKnowledgeWorker: BOTH emit_error and emit_stats are None! "
                    "Emitting sync_failed as fallback to unblock UI."
                )
                self.sync_failed.emit("Unknown sync error (no stats or error captured)")

    def cancel(self) -> None:
        """请求取消 ingest（在当前文件/批处理完成后生效）。"""
        if self._engine is not None:
            self._engine.cancel()

    def _on_progress(self, done: int, total: int, current: str) -> None:
        self.sync_progress.emit(done, total, current)


class InstallDepsWorker(QThread):
    """Background worker for checking and installing missing RAG dependencies.

    IMPORTANT: the completion signal is named ``deps_finished``, NOT
    ``finished``.  ``QThread`` already defines a built-in ``finished`` signal
    that is emitted *after* ``run()`` returns; declaring a custom signal with
    the same name shadows it, so anything connected to it (e.g.
    ``deleteLater``) fires from inside ``run()`` — deleting the QThread while
    it is still running.  See the same note on SyncKnowledgeWorker.
    """
    progress = pyqtSignal(str)
    deps_finished = pyqtSignal(object)  # DependencyReport

    def __init__(self, kb_path: str):
        super().__init__()
        self._kb_path = kb_path
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        """Signal the worker to stop (best-effort, pip install can't be interrupted)."""
        self._cancelled.set()

    def run(self) -> None:  # type: ignore[override]
        try:
            if self._cancelled.is_set():
                self.progress.emit("依赖检查已取消")
                self.deps_finished.emit(None)
                return
            from ..rag.deps import ensure_dependencies
            report = ensure_dependencies(
                self._kb_path,
                auto_install=True,
                include_core=True,
                progress_callback=self.progress.emit,
            )
            if self._cancelled.is_set():
                self.progress.emit("依赖检查已取消")
                self.deps_finished.emit(None)
                return
            self.deps_finished.emit(report)
        except Exception as e:
            self.progress.emit(f"依赖检查异常: {e}")
            self.deps_finished.emit(None)


# ===========================================================================
# Hosted-panel dialogs: config window / log window
# ===========================================================================


class _HostedPanelDialog(QDialog):
    """承载「主窗口长期持有的面板控件树」的对话框基类。

    面板在对话框与主窗口的隐藏宿主之间搬移：打开窗口时挂进对话框，关闭时移回
    宿主。控件对象因此全程常驻 —— 主窗口里各处引用（``self.base_url_edit`` /
    ``self.log_view`` …）不必关心父窗口是谁，也不必在每次开关窗口时重建：
    重建会丢掉用户已填未应用的输入，也会丢掉还没被查看的日志。
    """

    def __init__(self, title: str, min_w: int, min_h: int,
                 panel: Optional[QWidget] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(10, 10, 10, 10)
        self._lay.setSpacing(8)
        self._panel: Optional[QWidget] = None
        self.setWindowTitle(title)
        self.setMinimumSize(min_w, min_h)
        if panel is not None:
            self.attach(panel)
        self._build_footer(self._lay)

    def _build_footer(self, lay: QVBoxLayout) -> None:
        """对话框底部按钮行（子类可覆盖）。"""
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        close_btn = QPushButton("关闭")
        close_btn.setFixedWidth(96)
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        lay.addLayout(btn_row)

    def attach(self, panel: QWidget) -> None:
        """把面板挂进对话框（重复调用只是移动父窗口，不重建控件）。"""
        self._panel = panel
        self._lay.addWidget(panel)
        panel.show()

    def detach(self, host: QWidget) -> None:
        """把面板移回隐藏宿主，避免它随对话框一起被销毁。"""
        panel = self._panel
        self._panel = None
        if panel is None:
            return
        host_lay = host.layout()
        if host_lay is not None:
            host_lay.addWidget(panel)
        else:
            panel.setParent(host)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        parent = self.parent()
        host = getattr(parent, self._host_attr, None)
        if host is not None:
            self.detach(host)
        super().closeEvent(event)

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._center_on_parent()

    def _center_on_parent(self) -> None:
        parent = self.parent()
        if parent is None:
            return
        try:
            geo = parent.geometry()
            self.move(geo.center().x() - self.width() // 2,
                      geo.center().y() - self.height() // 2)
        except Exception:
            pass


class ConfigDialog(_HostedPanelDialog):
    """配置窗口：承载原主窗口顶部配置区的全部内容。"""

    _host_attr = "_config_host"

    def __init__(self, panel: QWidget, parent: Optional[QWidget] = None):
        super().__init__("配置", 780, 420, panel, parent)


class LogWindow(_HostedPanelDialog):
    """日志窗口：承载原右栏的日志面板（右上角「日志」按钮打开）。

    窗口关闭期间日志仍会持续写入面板（``_append_log`` / 知识库同步轮询都直接
    写 ``self.log_view``），重新打开即可看到完整历史，不会丢行。
    """

    _host_attr = "_log_host"

    def __init__(self, panel: QWidget, parent: Optional[QWidget] = None):
        super().__init__("日志", 900, 620, panel, parent)


# ===========================================================================
# Keyed plain text editor: Enter sends, Shift+Enter newline
# ===========================================================================


class SendEditor(QPlainTextEdit):
    send_requested = pyqtSignal()

    def keyPressEvent(self, e: QKeyEvent) -> None:  # type: ignore[override]
        # 裸 Enter/Return 发送；Shift/Ctrl/Meta+Enter 走 super()（换行）。
        #
        # IME 预编辑态按回车是「确认候选词」而非「发送」：此时 Return 事件的
        # text() 为空；普通回车的 text() 为 "\r"（非空）。用「text 非空」区分
        # 二者——旧实现多写了 .strip("\r\n")，把普通回车的 "\r" 也剥成空，
        # 导致普通回车永远不进入发送分支、回车发送彻底失效（只能点按钮）。
        if (e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
                and not (e.modifiers() & ~Qt.KeyboardModifier.KeypadModifier)
                and (e.text() or "") != ""):
            self.send_requested.emit()
            e.accept()
            return
        super().keyPressEvent(e)


# ===========================================================================
# Main window
# ===========================================================================


class MainWindow(QMainWindow):
    # 本地模型后台预加载完成（跨线程）→ UI 线程更新按钮态：key=「main」/「aux」，success=是否成功。
    model_loaded = pyqtSignal(str, bool)
    # 预热阶段的状态提示：后台预加载线程 → 主线程底部状态栏。
    # 预热在子线程阻塞跑，不能直接碰 QWidgets（跨线程崩溃），统一走信号
    # queued 投到 UI 线程更新 status_label。
    warmup_status = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.model_loaded.connect(self._on_model_loaded)
        self.warmup_status.connect(self._on_warmup_status)
        self.setWindowTitle("Base Augment Agent")
        self.resize(1280, 820)

        self.config = AgentConfig.load(CONFIG_PATH)
        self.config.ensure_workspace()
        # 让 RAG 解析层感知 OCR 配置开关（CoreML EP / 预处理）。
        # _OCR_CONFIG 是模块级全局，SyncKnowledgeWorker 同进程运行，全局注入一次即可生效。
        from agent.rag import parsers as _rag_parsers
        _rag_parsers.set_ocr_config(self.config)
        # docling 转换器池大小（UI「并发实例」），默认 1。
        from agent.rag.docling_parser import set_docling_pool_size
        set_docling_pool_size(getattr(self.config, "rag_docling_pool_size", 1))
        # PDF 统一解析引擎选择（docling | marker | ovis），同上全局注入一次。
        from agent.rag import engine as _rag_engine_mod
        _rag_engine_mod.set_pdf_engine(getattr(self.config, "rag_pdf_engine", "docling"))
        # OvisOCR2 模型配置（#133，ovis 引擎用；docling/marker 模式下不加载）。
        _rag_engine_mod.set_ovis_model_config(
            getattr(self.config, "rag_ocr_engine", "mlx-vlm"),
            getattr(self.config, "rag_ocr_model_name", "OvisOCR2-0.8B-MLX-4bit"),
            getattr(self.config, "rag_ocr_model_path", ""),
            getattr(self.config, "rag_ocr_mmproj_path", ""),
            getattr(self.config, "rag_ocr_tokenizer_path", ""),
        )
        # docling OCR scale 配置（#132：固定值默认 2.0，动态策略默认关）。
        from agent.rag.docling_parser import set_ocr_scale_config
        set_ocr_scale_config(
            base=getattr(self.config, "rag_ocr_scale_base", 2.0),
            dynamic=getattr(self.config, "rag_ocr_scale_dynamic", False),
            table=getattr(self.config, "rag_ocr_scale_table", 2.5),
        )
        self._bridge = Bridge()
        self._callbacks = BridgeCallbacks(self._bridge)
        # bridge.status 复用 warmup 状态入口：worker 线程预填系统提示 KV 时，
        # 跨线程把「系统提示预热中…/已预热」投到状态栏 + 进度条不确定态。
        self._bridge.status.connect(self._on_warmup_status)
        self._llm: Optional[LLMClient] = None
        self._agent: Optional[Agent] = None
        self._worker: Optional[AgentWorker] = None
        self._fetch_worker: Optional[FetchModelsWorker] = None
        self._sync_worker: Optional[SyncKnowledgeWorker] = None
        self._dedup_worker: Optional[VectorDedupWorker] = None
        self._deps_worker: Optional[InstallDepsWorker] = None
        self._deps_cancelled: bool = False  # True if user cancelled during deps check
        self._streaming_assistant = False  # True while appending content to current assistant bubble
        self._busy = False
        self._total_tokens = 0
        # 工作线程被 terminate() 强杀后置位：此后不能再走任何需要取锁的
        # 资源释放（tools.shutdown 等）—— 被杀线程持有的锁永远不会被释放。
        self._worker_terminated = False
        # wait 超时的 QThread 孤儿：最后一个 Python 引用归零会触发 sip 析构
        # → qFatal。保留引用到进程退出，closeEvent 时再统一 wait 一次。
        self._orphan_workers: list = []
        self._preload_threads: dict = {}  # key(main/aux) → 预加载线程，见 _preload_client
        # 同步进行中标志：供 _set_controls_busy 判定是否跳过对同步控件的重置。
        self._sync_active = False
        self._stop_watchdog: Optional[QTimer] = None
        # 流式输出缓冲（配合 _stream_timer 合并刷新）
        self._pending_content: List[str] = []
        self._pending_reasoning: List[str] = []
        self._stream_timer: Optional[QTimer] = None
        # 协议标签剥离器（每个 assistant 回合重建）
        self._summary_stripper = _SummaryStripper()
        # 日志轮询定时器：每 100ms 检查 sync_worker._log_buffer，
        # 把新日志追加到 log_view。绕过 Qt queued signal 机制，
        # 避免在 GIL 竞争下信号无法被主线程处理的问题。
        self._log_poll_timer = QTimer()
        self._log_poll_timer.setInterval(100)  # 100ms
        self._log_poll_timer.timeout.connect(self._poll_sync_logs)
        # 闲时后台逐章摘要调度器（rag/chapter_summarizer.py）：agent 空闲且
        # 无同步类 worker 运行时，每 15 分钟（复用已加载辅助引擎）/ 5 分钟
        # （临时加载）为知识库书籍补一章摘要。无论辅助引擎是否「启用」都会
        # 执行——未加载时临时加载、用完即卸载。全部依赖经 callable 注入，
        # 调度器不持有 UI/引擎/辅助引擎强引用。
        from ..rag.chapter_summarizer import ChapterSummaryScheduler
        self._summary_scheduler = ChapterSummaryScheduler(
            get_config=lambda: self.config,
            get_engine=lambda: (
                self._agent.tools.get_rag_engine()
                if (getattr(self, "_agent", None) is not None
                    and getattr(self._agent, "tools", None) is not None)
                else None),
            get_shared_aux=lambda: getattr(self, "_aux_llm", None),
            is_idle=lambda: (
                not self._busy
                and not self._sync_active
                and not self._is_worker_running(getattr(self, "_sync_worker", None))
                and not self._is_worker_running(getattr(self, "_deps_worker", None))
                and not self._is_worker_running(getattr(self, "_dedup_worker", None))
                and not self._is_worker_running(getattr(self, "_fetch_worker", None))
            ),
        )
        self._summary_scheduler.start()
        # Pending skill suggestion — cached when the agent emits
        # on_skill_suggested during a run, and shown as a dialog AFTER the
        # run finishes. Showing a modal dialog during streaming would block
        # the UI event loop and prevent content/reasoning chunks from
        # rendering, making the app look frozen.
        self._pending_skill: Optional[Skill] = None

        self._build_ui()
        self._wire_signals()
        self._setup_log_bridge()
        self._apply_config_to_widgets()
        self._rebuild_agent()
        # 启动主动加载：本地模式自动加载主模型；辅助模型仅在勾选「启用」时加载。
        self._auto_load_startup()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        # ---- Body: chat column (left) + plan/thinking column (right) ----
        # 配置 / 日志入口并入对话区上方的标题行（见 _build_chat_panel）：原先单独
        # 一条顶部工具条在去掉标题文本后左侧只剩空条，是多余的视觉噪音。
        body_split = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(body_split, 1)
        body_split.addWidget(self._build_chat_panel())
        body_split.addWidget(self._build_side_panel())
        body_split.setStretchFactor(0, 4)
        body_split.setStretchFactor(1, 1)
        body_split.setSizes([1000, 300])

        # 配置面板与日志面板都不进入主布局：各自托管在一个隐藏宿主里，打开对应
        # 窗口时挂进对话框，关闭时移回宿主 —— 控件对象常驻，主窗口各处引用无需
        # 感知父窗口变化。
        self._config_host = QWidget(self)
        host_lay = QVBoxLayout(self._config_host)
        host_lay.setContentsMargins(0, 0, 0, 0)
        self._config_panel = self._build_config_panel()
        host_lay.addWidget(self._config_panel)
        # 宿主仅是面板在「对话框 ↔ 主窗口」间 reparent 的中间容器，本身不进任何布局，
        # 必须隐藏——否则会作为浮动控件出现在主窗口 (0,0) 左上角（盖住对话区）。
        self._config_host.hide()
        self._config_dialog: Optional[ConfigDialog] = None
        # OCR 识别测试窗口（配置面板「识别测试」打开）：控件由对话框自己持有，
        # 关闭即销毁，这里只保存引用以便重复打开时复用同一个实例。
        self._ocr_test_dialog: Optional[OcrTestDialog] = None
        # 文件转 Markdown 转换工具窗口（配置面板「转换工具」打开）。
        self._convert_tool_dialog = None

        self._log_host = QWidget(self)
        log_host_lay = QVBoxLayout(self._log_host)
        log_host_lay.setContentsMargins(0, 0, 0, 0)
        self._log_panel = self._build_log_panel()
        log_host_lay.addWidget(self._log_panel)
        self._log_host.hide()
        self._log_window: Optional[LogWindow] = None

        # Status bar with progress
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(220)
        self.progress.setRange(0, 0)  # indeterminate off by default
        self.progress.setRange(0, 1)
        self.status.addPermanentWidget(QLabel(""))
        self.status.addPermanentWidget(self.progress)
        self.status_label = QLabel("就绪")
        self.status.addWidget(self.status_label)

    def _on_open_config(self) -> None:
        """右上角「⚙ 配置」：打开独立配置窗口。

        不做「打开时刷新」：控件里的值本来就是上次「应用配置」后的生效值
        （``__init__`` / ``_gather_config_from_widgets`` 都会同步），再刷一次
        只会把用户改到一半、还没点应用的编辑冲掉。
        """
        if self._config_dialog is None:
            self._config_dialog = ConfigDialog(self._config_panel, self)
        else:
            # 上一次关闭时面板已移回宿主，这里重新挂进对话框。
            self._config_dialog.attach(self._config_panel)
        self._config_dialog.show()
        self._config_dialog.raise_()
        self._config_dialog.activateWindow()

    def _on_open_log(self) -> None:
        """右上角「日志」：打开独立日志窗口（原右栏日志面板迁入）。

        与配置窗口同理不做「打开时刷新」：日志在窗口关闭期间持续写入面板，
        重新打开看到的就是完整历史。
        """
        if self._log_window is None:
            self._log_window = LogWindow(self._log_panel, self)
        else:
            # 上一次关闭时面板已移回宿主，这里重新挂进窗口。
            self._log_window.attach(self._log_panel)
        self._log_window.show()
        self._log_window.raise_()
        self._log_window.activateWindow()

    def _on_save_ocr_settings(self) -> None:
        """保存自定义 OCR 设置：落盘 + 注入 RAG 层 + 刷新测试窗口下拉。

        只动 OCR 这一段（不触发整个 RAG 重建）。下拉第四项「自定义OCR」的名字
        取自保存路径，所以保存后立刻在识别测试窗口看到该模型。
        """
        engine = self.ocr_impl_combo.currentText().strip() or "mlx-vlm"
        path = self.ocr_path_edit.text().strip()
        mmproj = self.ocr_mmproj_edit.text().strip()
        tokenizer = self.ocr_tokenizer_edit.text().strip()
        self.config.rag_ocr_engine = engine
        self.config.rag_ocr_model_path = path
        self.config.rag_ocr_mmproj_path = mmproj
        self.config.rag_ocr_tokenizer_path = tokenizer
        name = getattr(self.config, "rag_ocr_model_name",
                      "OvisOCR2-0.8B-MLX-4bit")
        try:
            self.config.save(CONFIG_PATH)
        except Exception as e:
            logger.warning("保存 OCR 配置失败: %s", e)
            QMessageBox.warning(self, "保存失败", f"{type(e).__name__}: {e}")
            return
        try:
            from agent.rag import engine as _rag_engine_mod
            _rag_engine_mod.set_ovis_model_config(
                engine, name, path, mmproj, tokenizer)
        except Exception as e:  # pragma: no cover - 注入失败不阻塞
            logger.debug("注入 OCR 模型配置失败（忽略）: %s", e)
        dlg = getattr(self, "_ocr_test_dialog", None)
        if dlg is not None:
            dlg.refresh_engine_items()
        QMessageBox.information(
            self, "已保存",
            "自定义 OCR 设置已保存。识别测试下拉框已更新，可立即用它识别。")

    def _on_open_ocr_test(self) -> None:
        """配置面板「识别测试」：打开 OCR 识别测试窗口。

        打开前先把面板里**当前填写**的 OCR 模型设置（后端 / 路径 / mmproj /
        tokenizer）注入 RAG 层——用户填完路径就想立刻试效果，不该被要求先点
        「应用配置」。配置确实变了会触发 release_model()，旧权重先卸掉，测试
        用的必定是新配置。
        """
        try:
            from agent.rag import engine as _rag_engine_mod
            _rag_engine_mod.set_ovis_model_config(
                self.ocr_impl_combo.currentText().strip() or "mlx-vlm",
                getattr(self.config, "rag_ocr_model_name",
                        "OvisOCR2-0.8B-MLX-4bit"),
                self.ocr_path_edit.text().strip(),
                self.ocr_mmproj_edit.text().strip(),
                self.ocr_tokenizer_edit.text().strip(),
            )
        except Exception as e:  # pragma: no cover - 注入失败不阻塞开窗
            logger.debug("OCR 测试：注入模型配置失败（忽略）: %s", e)
        if self._ocr_test_dialog is None:
            self._ocr_test_dialog = OcrTestDialog(self.config, self)
        self._ocr_test_dialog.show()
        self._ocr_test_dialog.raise_()
        self._ocr_test_dialog.activateWindow()

    def _on_open_convert_tool(self) -> None:
        """配置面板「转换工具」：打开「文件转 Markdown」独立窗口。

        转换窗口在「开始转换」时会调用 :meth:`sync_rag_config_to_rag_layer`
        自动采用当前面板里的 PDF 引擎 / OCR 模型等设置，所以这里无需提前注入，
        只负责创建 / 置顶窗口。
        """
        if getattr(self, "_convert_tool_dialog", None) is None:
            self._convert_tool_dialog = ConvertToolDialog(self, self)
        self._convert_tool_dialog.show()
        self._convert_tool_dialog.raise_()
        self._convert_tool_dialog.activateWindow()

    def sync_rag_config_to_rag_layer(self) -> None:
        """把面板当前控件值写入 config 并重新注入 RAG 解析层全局（不重建 Agent）。

        转换工具「开始转换」时调用：用户可能改了 PDF 引擎 / OCR 模型等却还没点
        「应用配置」，这里把最新面板值落进 config 对象并重新注入 ``rag`` 层的
        模块级全局（set_pdf_engine / set_ovis_model_config / set_ocr_config /
        set_ocr_scale_config），确保转换过程与「应用配置」后完全一致。与
        ``_rebuild_agent`` 内的同款注入一致，但不触发 Agent 重建（解析链路与
        LLM Agent 互不依赖）。
        """
        self._gather_config_from_widgets()
        try:
            from agent.rag import engine as _rag_engine_mod
            _rag_engine_mod.set_pdf_engine(
                getattr(self.config, "rag_pdf_engine", "docling"))
            _rag_engine_mod.set_ovis_model_config(
                getattr(self.config, "rag_ocr_engine", "mlx-vlm"),
                getattr(self.config, "rag_ocr_model_name",
                        "OvisOCR2-0.8B-MLX-4bit"),
                getattr(self.config, "rag_ocr_model_path", ""),
                getattr(self.config, "rag_ocr_mmproj_path", ""),
                getattr(self.config, "rag_ocr_tokenizer_path", ""),
            )
            from agent.rag import parsers as _rag_parsers
            _rag_parsers.set_ocr_config(self.config)
            from agent.rag.docling_parser import set_docling_pool_size
            set_docling_pool_size(getattr(self.config, "rag_docling_pool_size", 1))
            from agent.rag.docling_parser import set_ocr_scale_config
            set_ocr_scale_config(
                base=getattr(self.config, "rag_ocr_scale_base", 2.0),
                dynamic=getattr(self.config, "rag_ocr_scale_dynamic", False),
                table=getattr(self.config, "rag_ocr_scale_table", 2.5),
            )
        except Exception as e:  # pragma: no cover - 注入失败不阻塞转换
            logger.debug("转换工具：RAG 配置注入失败（忽略）: %s", e)

    def _build_config_panel(self) -> QWidget:
        """配置窗口内容：两大区 —— Agent 模型配置 / 知识同步配置。"""
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(10)

        def _combo(items, min_w=None):
            c = QComboBox()
            c.addItems(items)
            if min_w:
                c.setMinimumWidth(min_w)
            return c

        def _path(placeholder="", min_w=200):
            e = QLineEdit(placeholderText=placeholder)
            if min_w:
                e.setMinimumWidth(min_w)
            return e

        def _num(text, width=56):
            e = QLineEdit(text)
            e.setFixedWidth(width)
            return e

        def _pick_btn(edit, combo=None):
            b = QPushButton("选择")
            b.setFixedWidth(56)
            b.setToolTip("选择本地模型：GGUF/ONNX 选权重文件，MLX/fastembed 选目录")
            b.clicked.connect(lambda: self._pick_model_path(edit, combo))
            return b

        def _pick_dir_btn(edit):
            b = QPushButton("选择")
            b.setFixedWidth(56)
            b.setToolTip("选择目录（tokenizer 等所在目录）")
            b.clicked.connect(lambda: self._pick_dir_path(edit))
            return b

        def _params_widget(mt, temp, tp, mp, tk, rp, thinking, *extras):
            w = QWidget()
            row = QHBoxLayout(w)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            row.addWidget(QLabel("context")); row.addWidget(mt)
            row.addWidget(QLabel("Temp"));     row.addWidget(temp)
            row.addWidget(QLabel("top_p"));    row.addWidget(tp)
            row.addWidget(QLabel("min_p"));    row.addWidget(mp)
            row.addWidget(QLabel("top_k"));    row.addWidget(tk)
            row.addWidget(QLabel("repeat-penalty")); row.addWidget(rp)
            row.addWidget(thinking)
            for ex in extras:
                if ex is not None:
                    row.addWidget(ex)
            return w

        # ==================================================================
        # 区一：Agent 模型配置（主模型 + 辅助模型）
        # ==================================================================
        agent_box = QGroupBox("Agent 模型配置")
        alay = QVBoxLayout(agent_box)
        alay.setContentsMargins(10, 8, 10, 8)
        alay.setSpacing(8)

        # ---- 主模型：模式切换（本地模式 / OpenAI 兼容接口）----
        mode_row = QHBoxLayout(); mode_row.setSpacing(8)
        mode_row.addWidget(QLabel("主模型"))
        mode_row.addSpacing(8)
        mode_row.addWidget(QLabel("模式:"))
        self.main_mode_combo = _combo(["本地模式", "OpenAI兼容接口"])
        mode_row.addWidget(self.main_mode_combo)
        mode_row.addStretch(1)
        alay.addLayout(mode_row)

        # 主模型 - 本地模式 row1：模型类型 + 本地路径 + 选择
        self.main_type_combo = _combo(["MLX-LM", "MLX-VLM", "ONNX", "GGUF"], 110)
        self.local_llm_model_edit = _path("留空=默认 qwen3.5-4b-agentic-coder-v4-4bit-mlx")
        self.main_local_row1_w = QWidget()
        r = QHBoxLayout(self.main_local_row1_w); r.setContentsMargins(0, 0, 0, 0); r.setSpacing(8)
        r.addWidget(QLabel("模型类型")); r.addWidget(self.main_type_combo)
        r.addWidget(QLabel("本地路径")); r.addWidget(self.local_llm_model_edit, 1)
        r.addWidget(_pick_btn(self.local_llm_model_edit, self.main_type_combo))
        alay.addWidget(self.main_local_row1_w)

        # 主模型 - 第二路径（gguf→mmproj 视觉塔 / onnx→tokenizer 目录），按类型显隐
        self.main_second_label = QLabel("mmproj 路径")
        self.main_mmproj_edit = _path("mmproj 视觉塔路径（可选，多模态才需要）")
        self.main_mmproj_pick = _pick_btn(self.main_mmproj_edit, self.main_type_combo)
        self.main_tokenizer_edit = _path("tokenizer 路径（可选）")
        self.main_tokenizer_pick = _pick_dir_btn(self.main_tokenizer_edit)
        self.main_second_w = QWidget()
        r = QHBoxLayout(self.main_second_w); r.setContentsMargins(0, 0, 0, 0); r.setSpacing(8)
        r.addWidget(self.main_second_label)
        r.addWidget(self.main_mmproj_edit, 1); r.addWidget(self.main_mmproj_pick)
        r.addWidget(self.main_tokenizer_edit, 1); r.addWidget(self.main_tokenizer_pick)
        alay.addWidget(self.main_second_w)
        self.main_type_combo.currentTextChanged.connect(
            lambda _: self._sync_second_path(
                self.main_second_w, self.main_type_combo, self.main_second_label,
                self.main_mmproj_edit, self.main_mmproj_pick,
                self.main_tokenizer_edit, self.main_tokenizer_pick))

        # 主模型 - OpenAI 模式 row1：Base URL + API Key + 模型 + 获取模型
        self.base_url_edit = QLineEdit(placeholderText="https://api.openai.com/v1")
        self.api_key_edit = QLineEdit(placeholderText="sk-...", echoMode=QLineEdit.EchoMode.Password)
        self.model_combo = QComboBox(editable=True); self.model_combo.setMinimumWidth(180)
        self.fetch_btn = QPushButton("获取模型")
        self.main_openai_row1_w = QWidget()
        r = QHBoxLayout(self.main_openai_row1_w); r.setContentsMargins(0, 0, 0, 0); r.setSpacing(8)
        r.addWidget(QLabel("Base URL")); r.addWidget(self.base_url_edit, 3)
        r.addWidget(QLabel("API Key")); r.addWidget(self.api_key_edit, 2)
        r.addWidget(QLabel("模型")); r.addWidget(self.model_combo, 2)
        r.addWidget(self.fetch_btn)
        alay.addWidget(self.main_openai_row1_w)

        # 主模型 - 参数行（两模式共用）+ 动作按钮（本地: 加载/卸载；API: 应用配置）
        self.max_tokens_edit = _num("32768", 80)
        self.temp_edit = _num("0.7")
        self.top_p_edit = _num("0.95")
        self.min_p_edit = _num("0.05")
        self.top_k_edit = _num("20")
        self.repeat_penalty_edit = _num("1.0")
        self.thinking_check = QCheckBox("思考模式")
        self.thinking_check.setToolTip(
            "启用 Qwen3 系思考模式（enable_thinking）。关闭后模型跳过思考直接作答，"
            "首字延迟显著降低，但复杂任务的推理质量可能下降。")
        self.kv_reuse_check = QCheckBox("跨轮复用")
        self.kv_reuse_check.setToolTip(
            "跨轮 KV / 前缀复用：系统提示与历史不变的部分直接复用上一轮算好的 KV，"
            "多轮对话首字延迟显著下降（MLX 与 GGUF 后端都生效）。\n"
            "关闭后每轮整体重建（全量预填充，最慢但最保守）——用于排查前缀错位导致"
            "的罕见乱码，或在上下文/显存极紧张时以时间换空间。")
        self.kv_quant_check = QCheckBox("KV量化")
        self.kv_quant_check.setToolTip(
            "KV cache INT8 量化（省显存，质量近乎无损）：同等上下文长度下 KV 显存约"
            "减半。MLX 在模型默认缓存为标准 KV 缓存时生效（Qwen3.5 等线性注意力模型"
            "无标准 KV，自动跳过）；GGUF 通过 llama.cpp 的 type_k/type_v=q8_0 生效。"
            "主模型与辅助模型都启用。")
        self.main_params_w = QWidget()
        r = QHBoxLayout(self.main_params_w); r.setContentsMargins(0, 0, 0, 0); r.setSpacing(6)
        r.addWidget(_params_widget(self.max_tokens_edit, self.temp_edit, self.top_p_edit,
                                   self.min_p_edit, self.top_k_edit, self.repeat_penalty_edit,
                                   self.thinking_check,
                                   self.kv_reuse_check, self.kv_quant_check))
        r.addStretch(1)
        self.load_main_btn = QPushButton("加载模型")
        self.load_main_btn.setStyleSheet("font-weight:600;")
        self.unload_main_btn = QPushButton("卸载")
        self.connect_btn = QPushButton("应用配置")
        self.connect_btn.setStyleSheet("font-weight:600;")
        r.addWidget(self.load_main_btn); r.addWidget(self.unload_main_btn); r.addWidget(self.connect_btn)
        alay.addWidget(self.main_params_w)

        # ---- 辅助模型（本地专用）----
        aux_head = QHBoxLayout(); aux_head.setSpacing(8)
        aux_head.addWidget(QLabel("辅助模型（本地专用，用于事实抽取等轻量子任务）"))
        aux_head.addStretch(1)
        # 启用开关：默认不启用。关闭时应用启动不自动加载、知识库同步结束后
        # 不自动重载（开始/停止同步链路都会检查此选择）；「加载模型」按钮
        # 仍可手动临时加载。
        self.aux_enabled_check = QCheckBox("启用")
        self.aux_enabled_check.setChecked(False)
        self.aux_enabled_check.setToolTip(
            "启用后才会自动加载辅助模型（事实抽取 / RAG 检索结果归纳 / 投机解码草稿）。\n"
            "关闭时：应用启动不自动加载，知识库同步结束后也不会自动重载；\n"
            "「加载模型」按钮仍可手动临时加载。")
        aux_head.addWidget(self.aux_enabled_check)
        alay.addLayout(aux_head)
        self.aux_type_combo = _combo(["MLX-LM", "MLX-VLM", "ONNX", "GGUF"], 110)
        self.aux_local_llm_model_edit = _path("留空=默认 Qwen3.5-2B-MLX-4bit")
        r = QHBoxLayout(); r.setSpacing(8)
        r.addWidget(QLabel("模型类型")); r.addWidget(self.aux_type_combo)
        r.addWidget(QLabel("本地路径")); r.addWidget(self.aux_local_llm_model_edit, 1)
        r.addWidget(_pick_btn(self.aux_local_llm_model_edit, self.aux_type_combo))
        alay.addLayout(r)
        # 辅助模型 - 第二路径（gguf→mmproj / onnx→tokenizer），按类型显隐
        self.aux_second_label = QLabel("mmproj 路径")
        self.aux_mmproj_edit = _path("mmproj 视觉塔路径（可选，多模态才需要）")
        self.aux_mmproj_pick = _pick_btn(self.aux_mmproj_edit, self.aux_type_combo)
        self.aux_tokenizer_edit = _path("tokenizer 路径（可选）")
        self.aux_tokenizer_pick = _pick_dir_btn(self.aux_tokenizer_edit)
        self.aux_second_w = QWidget()
        r = QHBoxLayout(self.aux_second_w); r.setContentsMargins(0, 0, 0, 0); r.setSpacing(8)
        r.addWidget(self.aux_second_label)
        r.addWidget(self.aux_mmproj_edit, 1); r.addWidget(self.aux_mmproj_pick)
        r.addWidget(self.aux_tokenizer_edit, 1); r.addWidget(self.aux_tokenizer_pick)
        alay.addWidget(self.aux_second_w)
        self.aux_type_combo.currentTextChanged.connect(
            lambda _: self._sync_second_path(
                self.aux_second_w, self.aux_type_combo, self.aux_second_label,
                self.aux_mmproj_edit, self.aux_mmproj_pick,
                self.aux_tokenizer_edit, self.aux_tokenizer_pick))
        self.aux_max_tokens_edit = _num("32768", 80)
        self.aux_temp_edit = _num("0.7")
        self.aux_top_p_edit = _num("0.95")
        self.aux_min_p_edit = _num("0.05")
        self.aux_top_k_edit = _num("20")
        self.aux_repeat_penalty_edit = _num("1.0")
        self.aux_thinking_check = QCheckBox("思考模式")
        self.aux_thinking_check.setToolTip(self.thinking_check.toolTip())
        # 辅助模型同样暴露 KV 复用 / KV INT8 量化开关（与主模型同款）。
        self.aux_kv_reuse_check = QCheckBox("跨轮复用")
        self.aux_kv_reuse_check.setToolTip(self.kv_reuse_check.toolTip())
        self.aux_kv_quant_check = QCheckBox("KV量化")
        self.aux_kv_quant_check.setToolTip(self.kv_quant_check.toolTip())
        r = QHBoxLayout(); r.setSpacing(6)
        r.addWidget(_params_widget(self.aux_max_tokens_edit, self.aux_temp_edit, self.aux_top_p_edit,
                                   self.aux_min_p_edit, self.aux_top_k_edit, self.aux_repeat_penalty_edit,
                                   self.aux_thinking_check,
                                   self.aux_kv_reuse_check, self.aux_kv_quant_check))
        r.addStretch(1)
        self.load_aux_btn = QPushButton("加载模型")
        self.load_aux_btn.setStyleSheet("font-weight:600;")
        self.unload_aux_btn = QPushButton("卸载")
        r.addWidget(self.load_aux_btn); r.addWidget(self.unload_aux_btn)
        alay.addLayout(r)

        lay.addWidget(agent_box)

        # ==================================================================
        # 区二：知识同步配置（工作/知识目录 + Embedding + Reranker）
        # ==================================================================
        kb_box = QGroupBox("知识同步配置")
        klay = QVBoxLayout(kb_box)
        klay.setContentsMargins(10, 8, 10, 8)
        klay.setSpacing(8)

        # row1: 工作目录 + 知识库目录
        self.workspace_edit = QLineEdit()
        self.workspace_btn = QPushButton("…"); self.workspace_btn.setFixedWidth(28)
        self.knowledge_base_edit = QLineEdit(placeholderText="（可选）知识库目录")
        self.knowledge_base_btn = QPushButton("…"); self.knowledge_base_btn.setFixedWidth(28)
        r = QHBoxLayout(); r.setSpacing(8)
        r.addWidget(QLabel("工作目录")); r.addWidget(self.workspace_edit, 1); r.addWidget(self.workspace_btn)
        r.addWidget(QLabel("源知识库")); r.addWidget(self.knowledge_base_edit, 1); r.addWidget(self.knowledge_base_btn)
        klay.addLayout(r)

        # row1.2: 知识同步动作行（紧跟「工作目录配置行」）。按需求，在「应用配置」
        # 之后依次排列：开始同步 / 停止同步 / 向量去重。
        self.kb_apply_btn = QPushButton("应用配置")
        self.kb_apply_btn.setStyleSheet("font-weight:600;")
        self.kb_apply_btn.setToolTip(
            "保存当前所有设置并重新注入 RAG 解析层 / 重建 Agent（含知识同步区的\n"
            "Embedding / Reranker 选择），使新建向量库立即采用所选实现。")
        self.kb_apply_btn.clicked.connect(self._on_apply_config)

        self.sync_knowledge_btn = QPushButton("开始同步")
        self.sync_knowledge_btn.setToolTip(
            "解析知识库文档、建立向量索引，支持 RAG 检索\n"
            "左键：增量同步（只处理新增/修改的文件）\n"
            "右键：强制全量重新处理"
        )
        self.sync_knowledge_btn.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
        self._force_sync_action = self.sync_knowledge_btn.addAction("强制全量重新处理")
        self._force_sync_action.triggered.connect(lambda: self._on_sync_knowledge(force=True))

        self.stop_sync_btn = QPushButton("停止同步")
        self.stop_sync_btn.setEnabled(False)
        self.stop_sync_btn.setToolTip("停止知识库同步（在当前文件处理完成后生效）")
        self.stop_sync_btn.clicked.connect(self._on_stop_sync)

        self.dedup_vectors_btn = QPushButton("向量去重")
        self.dedup_vectors_btn.setToolTip(
            "扫描向量库，清理内容完全相同的重复向量\n"
            "（每组只保留最早入库的一份，降低检索噪声）"
        )
        self.dedup_vectors_btn.clicked.connect(self._on_dedup_vectors)

        r = QHBoxLayout(); r.setSpacing(8); r.setContentsMargins(0, 0, 0, 0)
        r.addStretch(1)
        r.addWidget(self.kb_apply_btn)
        r.addWidget(self.sync_knowledge_btn)
        r.addWidget(self.stop_sync_btn)
        r.addWidget(self.dedup_vectors_btn)
        klay.addLayout(r)

        # row1.5: PDF 解析引擎 + 自定义 OCR 模型（#133 统一引擎回退链）
        self.pdf_engine_combo = _combo(["docling", "marker", "OvisOCR"], 110)
        self.pdf_engine_combo.setToolTip(
            "PDF/Office 统一解析引擎（rag_pdf_engine）：\n"
            "· docling（默认）：docling 统一管线，版面/表格良好\n"
            "· marker：marker-pdf 优先（需 pip install marker-pdf；未装静默回退 docling）\n"
            "· OvisOCR：OvisOCR2 VLM 转 Markdown（docling/marker 之外的兜底）"
        )
        # 并发实例：docling 转换器池大小（rag_docling_pool_size）。可选 1-6，默认 1。
        # 控制 docling 解析器同时 convert 几个文件；⚠️ docling 的 MPS 推理非线程安全，
        # 进程内多实例在 16GB 机器上会触发 GPU 重置硬崩——默认 1 最稳，≥32GB 可试 2-6。
        self.parse_concurrency_combo = _combo(["1", "2", "3", "4", "5", "6"], 56)
        self.parse_concurrency_combo.setToolTip(
            "docling 解析器的并发实例数（rag_docling_pool_size）：\n"
            "1 = 单实例串行（最省内存，默认，16GB 机器最稳）；\n"
            "2-6 = 多实例并发解析，吞吐更高但常驻显存/内存同比上升。\n"
            "⚠️ docling 的 MPS 推理非线程安全，进程内多实例在 16GB 机器上\n"
            "可能触发 GPU 重置硬崩；仅 ≥32GB 内存时建议尝试 2 及以上。")
        # OCR并发：RapidOCR 引擎池大小（rag_ocr_pool_size）。可选 1-6，默认 1。
        # 控制 ocrmac 透传页的 RapidOCR 并发识别数；纯 CPU EP，多实例安全。
        self.ocr_concurrency_combo = _combo(["1", "2", "3", "4", "5", "6"], 56)
        self.ocr_concurrency_combo.setToolTip(
            "OCR 并发数（rag_ocr_pool_size，RapidOCR 引擎池）：\n"
            "1 = 单实例串行（最省内存，默认）；\n"
            "2-6 = 多实例并发识别，吞吐更高但常驻内存同比上升。\n"
            "对扫描件/图片型 PDF 的 OCR 阶段收益最明显，纯文本 PDF 基本无感。")
        self.ocr_impl_combo = _combo(["mlx-vlm", "onnx", "gguf"], 110)
        # 仅「类型 + 本地路径」：OCR 始终本地 VLM，模型名由 config 默认值
        # （rag_ocr_model_name=OvisOCR2-0.8B-MLX-4bit）决定，留空自动从
        # models/ 下解析；暴露可编辑的「模型名」只会制造拼写错误（同 reranker 旧
        # 的 -mx-fp8 笔误事故），故不显示在设置区。
        self.ocr_path_edit = _path("本地路径（留空自动用 models/ 下的模型名）", 160)
        # 「识别测试」：就地试跑各 OCR 引擎并左右对照原件与结果，避免每次验证
        # 效果都要走一遍完整知识库同步。打开的窗口直接用本行当前填写的配置。
        self.ocr_test_btn = QPushButton("识别测试")
        self.ocr_test_btn.setFixedWidth(84)
        self.ocr_test_btn.setToolTip(
            "打开 OCR 识别测试窗口：\n"
            "选一个图片/PDF，用 Apple OCR / OvisOCR2 / 混合（Vision+Ovis 表格）/"
            "自定义 OCR 试识别；\n左侧原件、右侧结果，多页 PDF 可逐页联动对照；"
            "确认可用后保存为 Markdown。")
        self.ocr_test_btn.clicked.connect(self._on_open_ocr_test)
        r = QHBoxLayout(); r.setSpacing(8)
        r.addWidget(QLabel("PDF解析引擎")); r.addWidget(self.pdf_engine_combo)
        r.addWidget(QLabel("并发实例")); r.addWidget(self.parse_concurrency_combo)
        r.addWidget(QLabel("自定义OCR")); r.addWidget(self.ocr_impl_combo)
        r.addWidget(QLabel("OCR并发")); r.addWidget(self.ocr_concurrency_combo)
        r.addWidget(self.ocr_path_edit, 2); r.addWidget(_pick_btn(self.ocr_path_edit, self.ocr_impl_combo))
        # 「保存设置」：把当前「自定义OCR」行的设置落盘，并让识别测试下拉框
        # 出现这个自定义模型（下拉第四项名字取自路径）。不重建整个 RAG。
        self.save_ocr_settings_btn = QPushButton("保存设置")
        self.save_ocr_settings_btn.setFixedWidth(84)
        self.save_ocr_settings_btn.setToolTip(
            "保存当前「自定义OCR」行的设置（后端/路径/mmproj/tokenizer）。\n"
            "保存后，识别测试窗口下拉框的「自定义OCR」项即显示为该模型，可立即用它识别。")
        self.save_ocr_settings_btn.clicked.connect(self._on_save_ocr_settings)
        r.addWidget(self.save_ocr_settings_btn)
        r.addWidget(self.ocr_test_btn)
        # 「转换工具」：打开独立的「文件转 Markdown」窗口，单独把各类文件批量
        # 提取转换成 Markdown 并保存到指定目录（解析/提取/图片识别/清洗/MD
        # 转换均复用 RAG 入库链路同一套处理逻辑）。
        self.convert_tool_btn = QPushButton("转换工具")
        self.convert_tool_btn.setFixedWidth(84)
        self.convert_tool_btn.setToolTip(
            "打开「文件转 Markdown 转换工具」：\n"
            "选单个文件或指定目录（支持 PDF/EPUB/Word/Excel/PPT/HTML/图片/TXT 等），"
            "设置保存目录后一键批量转换成 Markdown；\n"
            "自动采用当前配置面板里的 PDF 引擎 / OCR 模型等设置，进度条实时显示当前"
            "文件、处理环节与总进度。")
        self.convert_tool_btn.clicked.connect(self._on_open_convert_tool)
        r.addWidget(self.convert_tool_btn)
        klay.addLayout(r)

        # OCR 第二路径（gguf→mmproj 视觉塔 / onnx→tokenizer），按类型显隐
        self.ocr_second_label = QLabel("mmproj 路径")
        self.ocr_mmproj_edit = _path("mmproj 视觉塔路径（可选，多模态才需要）")
        self.ocr_mmproj_pick = _pick_btn(self.ocr_mmproj_edit, self.ocr_impl_combo)
        self.ocr_tokenizer_edit = _path("tokenizer 路径（可选）")
        self.ocr_tokenizer_pick = _pick_dir_btn(self.ocr_tokenizer_edit)
        self.ocr_second_w = QWidget()
        r2 = QHBoxLayout(self.ocr_second_w); r2.setContentsMargins(0, 0, 0, 0); r2.setSpacing(8)
        r2.addWidget(self.ocr_second_label)
        r2.addWidget(self.ocr_mmproj_edit, 1); r2.addWidget(self.ocr_mmproj_pick)
        r2.addWidget(self.ocr_tokenizer_edit, 1); r2.addWidget(self.ocr_tokenizer_pick)
        # 注意：r2 已在上面通过 QHBoxLayout(self.ocr_second_w) 设为该容器的布局，
        # 这里必须 addWidget（添加容器控件），不能 addLayout(r2)——后者会把已有
        # 父对象的布局再挂到 klay，触发 "QHBoxLayout already has a parent" 警告，
        # 且会让 ocr_second_w 变成没有布局的孤儿控件（显隐切换失效）。
        klay.addWidget(self.ocr_second_w)
        self.ocr_impl_combo.currentTextChanged.connect(
            lambda _: self._sync_second_path(
                self.ocr_second_w, self.ocr_impl_combo, self.ocr_second_label,
                self.ocr_mmproj_edit, self.ocr_mmproj_pick,
                self.ocr_tokenizer_edit, self.ocr_tokenizer_pick))
        self.ocr_impl_combo.setToolTip(
            "OvisOCR2 VLM 后端：\n"
            "· mlx-vlm：目录含 model.safetensors（Apple Silicon 推荐）\n"
            "· gguf：目录含 主模型.gguf + mmproj*.gguf（llama-cpp 多模态）\n"
            "· onnx：目录含 genai_config.json（onnxruntime-genai 导出，"
            "需 pip install onnxruntime-genai）"
        )

        # row2: Embedding（实现下拉 + 模型名 + 本地路径选择 + 同步知识）
        self.embedding_impl_combo = _combo(["fastembed", "mlx", "onnx", "gguf"], 110)
        self.embedding_model_name_edit = _path("模型名（fastembed 模型 id）", 200)
        self.embedding_path_edit = _path("本地路径（mlx/onnx/gguf 时选用）", 160)
        r = QHBoxLayout(); r.setSpacing(8)
        r.addWidget(QLabel("Embedding")); r.addWidget(self.embedding_impl_combo)
        r.addWidget(self.embedding_model_name_edit, 2)
        r.addWidget(self.embedding_path_edit, 2); r.addWidget(_pick_btn(self.embedding_path_edit, self.embedding_impl_combo))
        # 模型名只在 fastembed（网络模型 id）模式有意义；本地模式由「路径」决定
        # 模型，显示一个用不到又允许编辑的字段只会制造拼写错误（实测用户把
        # reranker 名写成 -mx-fp8，加载直接失败）。按实现下拉显隐。
        self.embedding_impl_combo.currentTextChanged.connect(
            lambda _t: self._sync_impl_fields(self.embedding_impl_combo,
                                              self.embedding_model_name_edit,
                                              self.embedding_path_edit))
        # 切片参数（语义切片开关 + 大小 / 重叠，单位 token）：与 Embedding 同处切片
        # 链路，故置于 Embedding 行内——模型目录选择框后、向量去重前。切片大小 /
        # 重叠同步到所有分片环节（递归 / 语义 / 混合分片 + 预算收口）；关闭语义
        # 切片时直接按字数递归切分，不加载任何嵌入模型。
        self.semantic_chunk_chk = QCheckBox("语义切片")
        self.semantic_chunk_chk.setChecked(
            bool(getattr(self.config, "rag_chunk_semantic", True)))
        self.semantic_chunk_chk.setToolTip(
            "勾选（默认）：按语义边界切分（混合分片 → 语义分片 → 递归分片三级回退，"
            "会使用嵌入模型）。\n取消勾选：不做语义切片，直接按字数（token）递归切分，"
            "确定性、可离线、不加载任何嵌入模型。")
        r.addWidget(self.semantic_chunk_chk)
        r.addSpacing(8)
        r.addWidget(QLabel("切片大小"))
        self.chunk_size_edit = _num(str(getattr(self.config, "rag_chunk_size", 800)), 64)
        self.chunk_size_edit.setToolTip(
            "切片大小（单位：token）。送入 embedding 前的单块预算上限；\n"
            "实际切片 = 正文预算 + 重叠头，均不超过该值。\n"
            "若 embedding 模型上下文窗口较小（如 bge 512），引擎会自动夹紧到窗口上限，"
            "避免嵌入阶段静默截断。")
        r.addWidget(self.chunk_size_edit)
        r.addSpacing(6)
        r.addWidget(QLabel("重叠"))
        self.chunk_overlap_edit = _num(str(getattr(self.config, "rag_chunk_overlap", 50)), 64)
        self.chunk_overlap_edit.setToolTip(
            "相邻分片重叠（单位：token）。用于保持分片边界处的上下文连续性；\n"
            "值过大会浪费嵌入算力，过小会丢失跨边界语义。建议为切片大小的 5%~15%。")
        r.addWidget(self.chunk_overlap_edit)
        r.addSpacing(6)
        self.matryoshka_chk = QCheckBox("Matryoshka")
        self.matryoshka_chk.setChecked(
            bool(getattr(self.config, "rag_matryoshka_enabled", True)))
        self.matryoshka_chk.setToolTip(
            "对支持 Matryoshka（俄罗斯套娃）表示的 embedding 模型\n"
            "（nomic-embed-text-v1.5、jina-embeddings-v3 等）启用维度裁剪：\n"
            "截取前 N 维、**重新 L2 归一化**后量化到 int8 入库/查询。\n"
            "关闭则保留模型原始维度（仍做 int8 量化）。")
        r.addWidget(self.matryoshka_chk)
        r.addSpacing(6)
        r.addWidget(QLabel("裁剪维度"))
        self.embedding_dims_combo = _combo(["128", "256", "512"])
        _dims_init = str(getattr(self.config, "rag_embedding_dims", 256))
        if _dims_init not in ("128", "256", "512"):
            _dims_init = "256"
        self.embedding_dims_combo.setCurrentText(_dims_init)
        self.embedding_dims_combo.setToolTip(
            "Matryoshka 裁剪维度（仅对支持 Matryoshka 的模型生效）：\n"
            "128：最小体积/最快，检索质量最低；\n"
            "256：体积与质量均衡（默认）；\n"
            "512：质量最高，体积最大。\n"
            "设置后自动完成 L2 归一化 + int8 量化，入库与查询同步。\n"
            "改动后旧向量库需清空重建。")
        r.addWidget(self.embedding_dims_combo)
        klay.addLayout(r)

        # row3: Reranker（实现下拉 + 模型名 + 本地路径选择 + 停止同步）
        self.reranker_impl_combo = _combo(["fastembed", "mlx", "onnx", "gguf"], 110)
        self.reranker_model_name_edit = _path("模型名（fastembed 模型 id）", 200)
        self.reranker_path_edit = _path("本地路径（mlx/onnx/gguf 时选用）", 160)
        r = QHBoxLayout(); r.setSpacing(8)
        r.addWidget(QLabel("Reranker")); r.addWidget(self.reranker_impl_combo)
        r.addWidget(self.reranker_model_name_edit, 2)
        r.addWidget(self.reranker_path_edit, 2); r.addWidget(_pick_btn(self.reranker_path_edit, self.reranker_impl_combo))
        self.reranker_impl_combo.currentTextChanged.connect(
            lambda _t: self._sync_impl_fields(self.reranker_impl_combo,
                                              self.reranker_model_name_edit,
                                              self.reranker_path_edit))
        # TopN：reranker 重排后最终保留的结果条数（对应 rag_search_top_k，
        # engine.search(top_k=...) → store.search_with_rerank(top_k=...)）。
        # 模型未显式传 top_k 时，rag_search 默认就用这一配置值。
        r.addWidget(QLabel("TopN"))
        self.reranker_topn_edit = _num(
            str(getattr(self.config, "rag_search_top_k", 5)), 56)
        self.reranker_topn_edit.setToolTip(
            "重排序（及最终检索）后保留的结果条数。\n"
            "对应 rag_search_top_k（默认 5）；检索时若未显式指定 top_k 即采用此值。\n"
            "取值 1~10（引擎在检索时会夹紧到该范围）。")
        r.addWidget(self.reranker_topn_edit)
        klay.addLayout(r)

        lay.addWidget(kb_box)
        lay.addStretch(1)

        # 主模型模式切换 & 加载/卸载按钮
        self.main_mode_combo.currentTextChanged.connect(self._on_main_mode_changed)
        self.load_main_btn.clicked.connect(self._on_load_main_model)
        self.unload_main_btn.clicked.connect(self._on_unload_main_model)
        self.load_aux_btn.clicked.connect(self._on_load_aux_model)
        self.unload_aux_btn.clicked.connect(self._on_unload_aux_model)

        return box

    # ------------------------------------------------------------------
    # 配置窗口：模型路径选择 / 主模型模式切换 / 主·辅助模型加载·卸载
    # ------------------------------------------------------------------
    def _pick_model_path(self, edit: "QLineEdit",
                         combo: "QComboBox" = None) -> None:
        """按「模型类型/实现」下拉选择本地模型：GGUF/ONNX 选权重文件，其余选目录。

        旧实现只弹目录选择框——GGUF/ONNX 后端的权重是**单文件**，用户根本选不到
        文件，只能手工输入路径（还踩过 ``~`` 不展开、无效路径报错不可读的坑）。
        这里按下拉文本分流（``gguf``/``onnx`` 子串匹配，覆盖主/辅模型、OCR、
        Embedding、Reranker 五处）；GGUF 也接受目录形式（目录内有 .gguf 时加载
        层按字典序取第一个），需要时手工输入即可。
        """
        text = (combo.currentText() if combo is not None else "").strip().lower()
        if "gguf" in text:
            path, _ = QFileDialog.getOpenFileName(
                self, "选择 GGUF 权重文件", edit.text() or "models",
                "GGUF 权重 (*.gguf);;所有文件 (*)")
        elif "onnx" in text:
            path, _ = QFileDialog.getOpenFileName(
                self, "选择 ONNX 权重文件", edit.text() or "models",
                "ONNX 权重 (*.onnx);;所有文件 (*)")
        else:
            path = QFileDialog.getExistingDirectory(
                self, "选择本地模型目录", edit.text() or "models")
        if path:
            edit.setText(path)

    def _pick_dir_path(self, edit: "QLineEdit") -> None:
        """与 _pick_model_path 互补：选目录（tokenizer 等路径）。"""
        path = QFileDialog.getExistingDirectory(
            self, "选择目录", edit.text() or "models")
        if path:
            edit.setText(path)

    def _sync_second_path(self, second_w, combo, label, mmproj_edit, mmproj_pick,
                          tokenizer_edit, tokenizer_pick) -> None:
        """按模型类型下拉显隐第二路径（gguf→mmproj / onnx→tokenizer）。

        仅当类型含 'gguf'/'onnx' 时显示对应一组；其余（mlx / mlx-vlm 等）整体隐藏。
        """
        t = combo.currentText().strip().lower()
        is_gguf = "gguf" in t
        is_onnx = "onnx" in t
        label.setText("mmproj 路径" if is_gguf else "tokenizer 路径")
        label.setVisible(is_gguf or is_onnx)
        mmproj_edit.setVisible(is_gguf); mmproj_pick.setVisible(is_gguf)
        tokenizer_edit.setVisible(is_onnx); tokenizer_pick.setVisible(is_onnx)
        second_w.setVisible(is_gguf or is_onnx)

    def _on_main_mode_changed(self, text: str) -> None:
        """主模型在「本地模式」与「OpenAI兼容接口」间切换：显示对应行与动作按钮。"""
        local = (text == "本地模式")
        self.main_local_row1_w.setVisible(local)
        self.load_main_btn.setVisible(local)
        self.unload_main_btn.setVisible(local)
        self.main_openai_row1_w.setVisible(not local)
        self.connect_btn.setVisible(not local)
        # 第二路径（gguf→mmproj / onnx→tokenizer）仅本地模式可见，且按类型显隐。
        if local:
            self._sync_second_path(
                self.main_second_w, self.main_type_combo, self.main_second_label,
                self.main_mmproj_edit, self.main_mmproj_pick,
                self.main_tokenizer_edit, self.main_tokenizer_pick)
        else:
            self.main_second_w.setVisible(False)

    def _mark_model_loaded(self, key: str) -> None:
        btn = self.load_main_btn if key == "main" else self.load_aux_btn
        btn.setText("重新加载")
        btn.setStyleSheet(_MODEL_READY_STYLE)

    def _mark_model_unloaded(self, key: str) -> None:
        btn = self.load_main_btn if key == "main" else self.load_aux_btn
        btn.setText("加载模型")
        btn.setStyleSheet(_MODEL_IDLE_STYLE)

    def _on_model_loaded(self, key: str, success: bool) -> None:
        """后台预加载完成（跨线程信号入队到主线程）后更新按钮态。"""
        if success:
            self._mark_model_loaded(key)
            self.status_label.setText(
                "主模型已就绪，可对话" if key == "main" else "辅助模型已就绪")
        else:
            self._mark_model_unloaded(key)

    def _on_warmup_status(self, msg: str) -> None:
        """预热/加载阶段的状态提示（warmup_status / bridge.status 信号，跨线程安全入队）。

        模型加载完成后会在后台跑两次预热：aux 预热（触发 MLX 首次推理编译）与
        系统提示 KV 预填（预填数千 token 上下文）。这段时间把提示投到状态栏，
        并让进度条进入不确定态，用户能明确看到「在预热」而不是以为卡死。
        """
        self.status_label.setText(msg)
        # 含「请稍候」= 进行中 → 进度条不确定态；否则（完成/失败）复位。
        if "请稍候" in msg:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 1)
            self.progress.setValue(0)

    def _start_worker(self) -> None:
        """启动常驻 worker 线程（空闲期预填系统提示 KV，首句直接命中缓存）。

        必须在模型已加载、agent 就绪后调用。旧实现在用户首次点发送时才创建
        worker，导致 preload_kv() 作为首句处理的一部分执行、首句仍需全量预填充。
        改为加载完成后立即启动，预填发生在用户打字前的空闲期——worker 线程与真实
        生成同线程，缓存不被线程隔离丢弃。重复调用幂等（已在跑则跳过）。
        """
        if self._worker is not None and self._worker.isRunning():
            return
        if self._agent is None:
            return
        self._worker = AgentWorker(self._agent, self._callbacks)
        self._worker.start()

    def _preload_client(self, client, label: str, key: str) -> None:
        """后台预加载本地模型权重（LocalLLMClient._ensure_model），不阻塞 UI。

        加载完成/失败都通过 model_loaded 信号（跨线程入队）回到主线程更新
        按钮态：成功→绿色「重新加载」，失败→恢复「加载模型」。
        """
        ensure = getattr(client, "_ensure_model", None)
        if ensure is None:
            return

        def _run():
            try:
                ensure()
                logger.info("%s 权重已预加载", label)
                # 先置绿：加载成功即标记为「已就绪」（变绿），用户可立即开始对话。
                self.model_loaded.emit(key, True)
                # 主模型加载完成后**立即**启动常驻 worker：run() 内会先用 system
                # prompt 预填 KV（数秒）、再串行跑 warm_up 编译（20-30s），都在
                # worker 线程内完成、与真实生成同线程。用户打字前 worker 已空闲，
                # 首句直接命中缓存。绝不在本（预加载）线程里再跑 warm_up——否则会和
                # worker 内的预填/编译并发抢同一份权重（MLX 前向非线程安全）。
                if key == "main" and not self._busy:
                    self._start_worker()
            except Exception as e:  # 预加载失败不影响主流程，首次用时再报错
                logger.warning("%s 预加载失败: %s", label, e)
                self.model_loaded.emit(key, False)

        # 旧实现 daemon 线程不跟踪：连续点「加载模型」会同时起 N 个加载线程
        # 抢 GPU 内存，各自独立 _ensure_model → 同一进程里 MLX/Llama 加载
        # N 次，内存翻倍。同 key 的前一个没结束就不再起新的；线程引用归
        # _preload_threads 持有，closeEvent 时 join 一遍避免 daemon 在退出
        # 时被强杀、丢落盘写入。
        prev = self._preload_threads.get(key)
        if prev is not None and prev.is_alive():
            logger.info("%s 预加载已在进行，跳过本次", label)
            return
        t = threading.Thread(target=_run, daemon=True, name=f"preload-{label}")
        self._preload_threads[key] = t
        t.start()

    def _auto_load_startup(self) -> None:
        """应用启动后主动预加载模型（后台线程，不阻塞 UI）。

        · 主模型：仅本地模式（``llm_provider == "local"``）下自动加载——API 模式
          无本地权重可加载，跳过（按钮保持未加载态，由网络调用兜底）。
        · 辅助模型：恒为本地模型（无 OpenAI 模式）。**默认不启用**——仅在
          「启用」勾选框勾选（``aux_llm_enabled``）时启动即加载，用于事实
          抽取 / RAG 归纳等子任务，避免首次用到才冷启动；未勾选绝不加载。
        """
        if getattr(self, "_busy", False):
            return
        started = False
        if getattr(self.config, "llm_provider", "local") == "local":
            main = getattr(self, "_llm", None)
            if main is not None and getattr(main, "_ensure_model", None) is not None:
                self._preload_client(main, "主模型", "main")
                started = True
        # 辅助模型自动加载必须以「启用」勾选为前提——用户未启用时启动绝不
        # 偷偷把权重拉进内存。优先读勾选框当前状态（可能已改未应用）。
        chk = getattr(self, "aux_enabled_check", None)
        aux_enabled = bool(chk.isChecked()) if chk is not None else bool(
            getattr(self.config, "aux_llm_enabled", False))
        if aux_enabled:
            aux = getattr(self, "_aux_llm", None)
            if aux is not None and getattr(aux, "_ensure_model", None) is not None:
                self._preload_client(aux, "辅助模型", "aux")
                started = True
        if started:
            self.status_label.setText("正在后台加载模型…")

    def _on_load_main_model(self) -> None:
        if self._busy:
            QMessageBox.warning(
                self, "正在运行", "Agent 正在运行，请先「终止对话」再加载模型。")
            return
        # 先收集控件再保存：旧实现先 save 再 rebuild（rebuild 内部才 gather），
        # 新填的路径/后端只进了内存、没写进 config.json——重启应用后丢失，
        # 且磁盘上存的是旧值，与内存不一致。
        self._gather_config_from_widgets()
        self.config.save(CONFIG_PATH)
        # 仅重载主模型：保留已加载的辅助模型（不销毁权重、不动其按钮态）。
        # 若要同时换辅助模型路径，点「重新加载辅助模型」或「应用配置」。
        self._rebuild_agent(rebuild_aux=False)
        self._preload_client(getattr(self, "_llm", None), "主模型", "main")
        self.status_label.setText("主模型加载中…")

    def _on_unload_main_model(self) -> None:
        if self._busy:
            QMessageBox.warning(
                self, "正在运行", "Agent 正在运行，请先「终止对话」再卸载模型。")
            return
        old = getattr(self, "_llm", None)
        if old is not None:
            try:
                old.close()
            except Exception:
                pass
        # 必须收尾 agent：旧实现只把 self._agent 置 None，旧 agent 持有的
        # MCP 子进程 / httpx / 本地模型句柄 / ONNX session / LanceDB 连接
        # 全部泄漏，且常驻 worker 仍引用它、下次还会把消息打进已 close 的
        # agent（各种「AttributeError on closed client」）。
        old_agent = getattr(self, "_agent", None)
        if old_agent is not None:
            try:
                old_agent.close()
            except Exception:
                pass
            try:
                old_agent.tools.shutdown()
            except Exception:
                pass
        self._llm = None
        self._agent = None
        self._mark_model_unloaded("main")
        self.status_label.setText("主模型已卸载，请选择新模型并加载")

    def _on_load_aux_model(self) -> None:
        if self._busy:
            QMessageBox.warning(
                self, "正在运行", "Agent 正在运行，请先「终止对话」再加载模型。")
            return
        # 先收集控件再保存（与 _on_load_main_model 同款修正：旧实现 save 在
        # gather 之前，新填的辅助模型路径不落盘，重启即丢）。
        self._gather_config_from_widgets()
        self.config.save(CONFIG_PATH)
        old = getattr(self, "_aux_llm", None)
        if old is not None:
            try:
                old.close()
            except Exception:
                pass
            self._aux_llm = None
        try:
            from ..local.factory import build_aux_llm_client
            aux = build_aux_llm_client(self.config)
        except Exception as e:
            self._on_error(f"无法加载辅助模型: {e}")
            return
        self._aux_llm = aux
        if self._agent is not None:
            self._agent.aux_llm = aux
            # 主模型复用刚加载的辅助模型当投机解码草稿（不另起权重）。
            if self._llm is not None:
                self._llm.set_draft_source(aux)
        self._preload_client(aux, "辅助模型", "aux")
        self.status_label.setText("辅助模型加载中…")

    def _on_unload_aux_model(self) -> None:
        if self._busy:
            QMessageBox.warning(
                self, "正在运行", "Agent 正在运行，请先「终止对话」再卸载模型。")
            return
        old = getattr(self, "_aux_llm", None)
        if old is not None:
            try:
                old.close()
            except Exception:
                pass
        self._aux_llm = None
        if self._agent is not None:
            self._agent.aux_llm = None
            # 辅助模型卸载 → 主模型关闭投机解码（退化为普通生成）。
            if self._llm is not None:
                self._llm.set_draft_source(None)
        self._mark_model_unloaded("aux")
        self.status_label.setText("辅助模型已卸载")

    def _build_chat_panel(self) -> QWidget:
        """左栏：对话区（占满）→ token 速度 → 发送框 + 发送按钮（贴底）。

        思考过程已移到右栏（见 ``_build_side_panel``），日志移入独立日志窗口，
        左栏因此整块都给对话区；token 速度/总量居中放在发送框上方。
        """
        left = QWidget()
        llay = QVBoxLayout(left)
        llay.setContentsMargins(0, 0, 0, 0)
        llay.setSpacing(4)

        # 对话控制按钮（原配置区里的两个对话动作，随对话栏走）
        self.clear_btn = QPushButton("清空对话")
        self.stop_btn = QPushButton("终止对话")
        self.stop_btn.setEnabled(False)
        # 配置 / 日志入口并入本行右侧（不再单独占一条顶部工具条）
        self.config_btn = QPushButton("⚙ 配置")
        self.config_btn.setFixedWidth(96)
        self.config_btn.setStyleSheet("font-weight:600;")
        self.config_btn.setToolTip(
            "打开配置窗口：Base URL / API Key / 模型 / 采样参数 / 工作目录 / 源知识库"
        )
        self.log_btn = QPushButton("日志")
        self.log_btn.setFixedWidth(96)
        self.log_btn.setStyleSheet("font-weight:600;")
        self.log_btn.setToolTip("打开日志窗口：运行日志 / RAG 同步 / 工具调用记录")

        # 知识同步快捷按钮：不必进设置窗口即可直接同步（与设置区「同步知识」同流程）。
        self.kb_sync_btn = QPushButton("知识同步")
        self.kb_sync_btn.setFixedWidth(96)
        self.kb_sync_btn.setStyleSheet("font-weight:600;")
        self.kb_sync_btn.setToolTip(
            "直接同步知识库（无需打开设置窗口）：\n"
            "左键：增量同步（只处理新增/修改的文件）\n"
            "右键：强制全量重新处理"
        )
        self.kb_sync_btn.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
        self._kb_force_sync_action = self.kb_sync_btn.addAction("强制全量重新处理")
        self._kb_force_sync_action.triggered.connect(
            lambda: self._on_sync_knowledge(force=True))

        # chat header with controls
        chat_header = QHBoxLayout()
        chat_header.setSpacing(8)
        chat_header.addWidget(QLabel("对话"))
        chat_header.addStretch(1)
        chat_header.addWidget(self.clear_btn)
        chat_header.addWidget(self.stop_btn)
        chat_header.addSpacing(12)
        chat_header.addWidget(self.config_btn)
        chat_header.addWidget(self.kb_sync_btn)
        chat_header.addWidget(self.log_btn)
        llay.addLayout(chat_header)

        # chat view —— 思考/日志移走后，左栏整块都是对话区
        self.chat_view = QTextEdit()
        self.chat_view.setReadOnly(True)
        # 限制对话区保留的段落数。长会话（多轮工具调用 + 大段输出）会让
        # QTextDocument 无限增长，最终吃光内存并让滚动/重排越来越卡。
        self.chat_view.document().setMaximumBlockCount(5000)
        self.chat_view.document().setDefaultStyleSheet(
            ".u{color:#2563eb}.a{color:#059669}.t{color:#7c3aed}.sys{color:#9ca3af}"
        )
        llay.addWidget(self.chat_view, 1)

        # token 速度 / 总量：居中放在发送框与发送按钮上方
        stats_row = QHBoxLayout()
        stats_row.setSpacing(8)
        stats_row.addStretch(1)
        self.speed_label = QLabel("0.0 tok/s")
        self.speed_label.setStyleSheet("color:#6b7280;")
        self.token_label = QLabel("总计 0 tokens")
        self.token_label.setStyleSheet("color:#6b7280;")
        stats_row.addWidget(self.speed_label)
        stats_row.addWidget(QLabel("|"))
        stats_row.addWidget(self.token_label)
        stats_row.addStretch(1)
        llay.addLayout(stats_row)

        # send（贴在最底部）
        send_row = QHBoxLayout()
        self.send_editor = SendEditor()
        self.send_editor.setFixedHeight(64)
        self.send_editor.setPlaceholderText("输入消息…  (Enter 发送 / Shift+Enter 换行)")
        send_row.addWidget(self.send_editor)
        send_col = QVBoxLayout()
        self.send_btn = QPushButton("发送")
        self.send_btn.setFixedHeight(64)
        self.send_btn.setStyleSheet("font-weight:600;")
        send_col.addWidget(self.send_btn)
        send_row.addLayout(send_col)
        llay.addLayout(send_row)

        return left

    def _build_side_panel(self) -> QWidget:
        """右栏：执行计划（上）+ 思考过程（下）—— 日志已移入独立日志窗口。"""
        vsplit = QSplitter(Qt.Orientation.Vertical)
        vsplit.addWidget(self._build_plan_panel())
        vsplit.addWidget(self._build_thinking_panel())
        vsplit.setStretchFactor(0, 2)
        vsplit.setStretchFactor(1, 3)
        vsplit.setSizes([240, 460])
        return vsplit

    def _build_thinking_panel(self) -> QWidget:
        """思考过程（原左栏中部，随日志移走后调到右栏原日志的位置）。"""
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        header = QHBoxLayout()
        header.addWidget(QLabel("思考过程"))
        lay.addLayout(header)
        self.think_view = QPlainTextEdit()
        self.think_view.setReadOnly(True)
        font = QFont("Menlo")
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.think_view.setFont(font)
        # 防止单次超长流式推理导致内存无限增长（每轮运行前会 clear()）。
        self.think_view.setMaximumBlockCount(10000)
        lay.addWidget(self.think_view)
        return box

    def _build_plan_panel(self) -> QWidget:
        """执行计划：任务目标 + 步骤清单 + 每项完成情况，正在进行的步骤高亮。"""
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        plan_header = QHBoxLayout()
        plan_header.setSpacing(8)
        plan_header.addWidget(QLabel("执行计划"))
        # 进度摘要：`2/5 步完成 · 进行中：部署`
        self.plan_progress_label = QLabel("")
        self.plan_progress_label.setStyleSheet("color:#6b7280;")
        plan_header.addWidget(self.plan_progress_label)
        plan_header.addStretch(1)
        self.plan_toggle_btn = QPushButton("收起")
        self.plan_toggle_btn.setFixedWidth(60)
        self.plan_toggle_btn.setCheckable(True)
        plan_header.addWidget(self.plan_toggle_btn)
        lay.addLayout(plan_header)

        self.plan_view = QTextEdit()
        self.plan_view.setReadOnly(True)
        plan_font = QFont("Menlo")
        plan_font.setStyleHint(QFont.StyleHint.Monospace)
        self.plan_view.setFont(plan_font)
        # 与 QPlainTextEdit 时代一致的上限：QTextDocument 同样支持。
        self.plan_view.document().setMaximumBlockCount(300)
        self.plan_view.setPlaceholderText(
            "（尚无计划 — Agent 开始任务前会先规划）\n"
            "计划制定后，这里会列出每一步及其完成情况，并标记当前正在执行的步骤。"
        )
        lay.addWidget(self.plan_view)
        return box

    def _build_log_panel(self) -> QWidget:
        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(0, 0, 0, 0)
        rlay.setSpacing(4)
        log_header = QHBoxLayout()
        log_header.addWidget(QLabel("日志"))
        self.log_clear_btn = QPushButton("清空")
        self.log_clear_btn.setFixedWidth(60)
        log_header.addStretch(1)
        log_header.addWidget(self.log_clear_btn)
        rlay.addLayout(log_header)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        font = QFont("Menlo")
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.log_view.setFont(font)
        self.log_view.setMaximumBlockCount(5000)  # 限制日志行数，防止内存无限增长
        rlay.addWidget(self.log_view, 1)
        return right

    # ------------------------------------------------------------------
    # wiring
    # ------------------------------------------------------------------
    def _wire_signals(self) -> None:
        self.config_btn.clicked.connect(self._on_open_config)
        self.log_btn.clicked.connect(self._on_open_log)
        self.fetch_btn.clicked.connect(self._on_fetch_models)
        self.connect_btn.clicked.connect(self._on_apply_config)
        self.workspace_btn.clicked.connect(self._on_pick_workspace)
        self.knowledge_base_btn.clicked.connect(self._on_pick_knowledge_base)
        self.sync_knowledge_btn.clicked.connect(self._on_sync_knowledge)
        self.kb_sync_btn.clicked.connect(self._on_sync_knowledge)
        self.clear_btn.clicked.connect(self._on_clear_chat)
        self.log_clear_btn.clicked.connect(self.log_view.clear)
        self.send_btn.clicked.connect(self._on_send)
        self.send_editor.send_requested.connect(self._on_send)
        self.stop_btn.clicked.connect(self._on_stop)

        b = self._bridge
        b.content.connect(self._on_content)
        b.reasoning.connect(self._on_reasoning)
        b.tool_start.connect(self._on_tool_start)
        b.tool_end.connect(self._on_tool_end)
        b.log.connect(self._on_log)
        b.usage.connect(self._on_usage)
        b.speed.connect(self._on_speed)
        b.skill.connect(self._on_skill_suggested)
        b.plan.connect(self._on_plan_updated)
        self.plan_toggle_btn.toggled.connect(self._on_plan_toggle)
        b.finished.connect(self._on_finished)
        b.error.connect(self._on_error)
        b.confirm_request.connect(self._on_confirm_request)
        b.ask_request.connect(self._on_ask_request)

    # ------------------------------------------------------------------
    # config <-> widgets
    # ------------------------------------------------------------------
    def _apply_config_to_widgets(self) -> None:
        self.base_url_edit.setText(self.config.base_url)
        self.api_key_edit.setText(self.config.api_key)
        if self.config.model:
            self.model_combo.setEditText(self.config.model)
        self.temp_edit.setText(str(self.config.temperature))
        self.max_tokens_edit.setText(str(self.config.max_tokens))
        self.top_p_edit.setText(str(self.config.top_p))
        self.min_p_edit.setText(str(self.config.min_p))
        self.top_k_edit.setText(str(self.config.top_k))
        self.repeat_penalty_edit.setText(str(self.config.repetition_penalty))
        self.thinking_check.setChecked(bool(getattr(self.config, "local_llm_enable_thinking", True)))
        self.kv_reuse_check.setChecked(bool(getattr(self.config, "local_llm_kv_reuse", True)))
        self.kv_quant_check.setChecked(bool(getattr(self.config, "local_llm_kv_quantize", True)))
        self.workspace_edit.setText(self.config.workspace)
        self.knowledge_base_edit.setText(self.config.knowledge_base)
        # 主模型模式（本地 / OpenAI 兼容接口）+ 本地类型与路径
        self.main_mode_combo.setCurrentText(
            "本地模式" if self.config.llm_provider == "local" else "OpenAI兼容接口")
        self.main_type_combo.setCurrentText(
            _BACKEND_TO_LLM_TYPE.get(self.config.local_llm_backend, "MLX-LM"))
        self.local_llm_model_edit.setText(self.config.local_llm_model)
        self.main_mmproj_edit.setText(
            getattr(self.config, "local_llm_mmproj_path", ""))
        self.main_tokenizer_edit.setText(
            getattr(self.config, "local_llm_tokenizer_path", ""))
        # 辅助模型（本地专用）
        self.aux_enabled_check.setChecked(bool(getattr(self.config, "aux_llm_enabled", False)))
        self.aux_type_combo.setCurrentText(
            _BACKEND_TO_LLM_TYPE.get(self.config.aux_local_llm_backend, "MLX-LM"))
        self.aux_local_llm_model_edit.setText(self.config.aux_local_llm_model)
        self.aux_mmproj_edit.setText(
            getattr(self.config, "aux_local_llm_mmproj_path", ""))
        self.aux_tokenizer_edit.setText(
            getattr(self.config, "aux_local_llm_tokenizer_path", ""))
        self._sync_second_path(
            self.aux_second_w, self.aux_type_combo, self.aux_second_label,
            self.aux_mmproj_edit, self.aux_mmproj_pick,
            self.aux_tokenizer_edit, self.aux_tokenizer_pick)
        self.aux_temp_edit.setText(str(self.config.aux_temperature))
        self.aux_max_tokens_edit.setText(str(self.config.aux_max_tokens))
        self.aux_top_p_edit.setText(str(self.config.aux_top_p))
        self.aux_min_p_edit.setText(str(self.config.aux_min_p))
        self.aux_top_k_edit.setText(str(self.config.aux_top_k))
        self.aux_repeat_penalty_edit.setText(str(self.config.aux_repetition_penalty))
        # 兜底与 AgentConfig 字段默认值一致：辅助模型默认关闭思考模式。
        self.aux_thinking_check.setChecked(bool(getattr(self.config, "aux_local_llm_enable_thinking", False)))
        self.aux_kv_reuse_check.setChecked(bool(getattr(self.config, "aux_local_llm_kv_reuse", False)))
        self.aux_kv_quant_check.setChecked(bool(getattr(self.config, "aux_local_llm_kv_quantize", True)))
        # Embedding / Reranker：实现下拉 + 模型名 + 本地路径
        self.embedding_impl_combo.setCurrentText(_provider_backend_to_impl(
            self.config.embedding_provider, self.config.local_embedding_backend))
        self.embedding_model_name_edit.setText(self.config.rag_embedding_model)
        self.embedding_path_edit.setText(self.config.local_embedding_model)
        # PDF 解析引擎（docling | marker | ovis）：内部值 → 显示标签。
        self.pdf_engine_combo.setCurrentText(
            _PDF_VALUE_TO_DISPLAY.get(
                getattr(self.config, "rag_pdf_engine", "docling"), "docling"))
        # 并发实例（rag_docling_pool_size，docling 转换器池）：夹在 1-6（下拉框只有这六项）。
        _dl_pool = int(getattr(self.config, "rag_docling_pool_size", 1) or 1)
        self.parse_concurrency_combo.setCurrentText(str(min(max(_dl_pool, 1), 6)))
        # OCR并发（rag_ocr_pool_size，RapidOCR 引擎池）：同样夹在 1-6。
        _ocr_pool = int(getattr(self.config, "rag_ocr_pool_size", 1) or 1)
        self.ocr_concurrency_combo.setCurrentText(str(min(max(_ocr_pool, 1), 6)))
        # OCR 模型（OvisOCR2，#133）：仅类型 + 路径；模型名用 config 默认值
        self.ocr_impl_combo.setCurrentText(
            getattr(self.config, "rag_ocr_engine", "mlx-vlm") or "mlx-vlm")
        self.ocr_path_edit.setText(
            getattr(self.config, "rag_ocr_model_path", ""))
        self.ocr_mmproj_edit.setText(
            getattr(self.config, "rag_ocr_mmproj_path", ""))
        self.ocr_tokenizer_edit.setText(
            getattr(self.config, "rag_ocr_tokenizer_path", ""))
        self._sync_second_path(
            self.ocr_second_w, self.ocr_impl_combo, self.ocr_second_label,
            self.ocr_mmproj_edit, self.ocr_mmproj_pick,
            self.ocr_tokenizer_edit, self.ocr_tokenizer_pick)
        self.reranker_impl_combo.setCurrentText(_provider_backend_to_impl(
            self.config.reranker_provider, self.config.local_reranker_backend))
        self.reranker_model_name_edit.setText(self.config.rag_rerank_model)
        self.reranker_path_edit.setText(self.config.local_reranker_model)
        # TopN：reranker 之后保留的结果条数（rag_search_top_k）。
        self.reranker_topn_edit.setText(str(getattr(self.config, "rag_search_top_k", 5)))
        # 切片参数：语义切片开关 / 大小 / 重叠（token）。重叠以显式 token 为准
        # （关闭百分比驱动），使「重叠」输入直接生效，并随切片大小一起同步到所有
        # 分片环节。
        self.semantic_chunk_chk.setChecked(
            bool(getattr(self.config, "rag_chunk_semantic", True)))
        self.chunk_size_edit.setText(str(getattr(self.config, "rag_chunk_size", 800)))
        self.chunk_overlap_edit.setText(str(getattr(self.config, "rag_chunk_overlap", 50)))
        self.matryoshka_chk.setChecked(
            bool(getattr(self.config, "rag_matryoshka_enabled", True)))
        _dims_load = str(getattr(self.config, "rag_embedding_dims", 256))
        if _dims_load not in ("128", "256", "512"):
            _dims_load = "256"
        self.embedding_dims_combo.setCurrentText(_dims_load)
        # Embedding / Reranker 的「模型名 ↔ 本地路径」显隐：fastembed 显示模型名、
        # 隐藏路径；本地实现反之。按当前下拉值一次性同步初始可见性。
        self._sync_impl_fields(self.embedding_impl_combo,
                               self.embedding_model_name_edit,
                               self.embedding_path_edit)
        self._sync_impl_fields(self.reranker_impl_combo,
                               self.reranker_model_name_edit,
                               self.reranker_path_edit)
        # 同步主模型模式切换的可见性
        self._on_main_mode_changed(self.main_mode_combo.currentText())

    def _sync_impl_fields(self, combo, name_edit, path_edit) -> None:
        """按实现下拉显隐「模型名 / 本地路径」：fastembed 显示模型名、隐藏路径；
        本地实现（mlx/onnx/gguf）显示路径、隐藏模型名。

        本地模式下「模型名」用不到又允许编辑，曾经让用户把 reranker 名拼错成
        -mx-fp8 导致加载失败；隐藏它从根上消除这类手填错误（见技能检索排查）。
        """
        is_fastembed = (combo.currentText().strip().lower() == "fastembed")
        name_edit.setVisible(is_fastembed)
        path_edit.setVisible(not is_fastembed)

    def _gather_config_from_widgets(self) -> None:
        self.config.base_url = self.base_url_edit.text().strip()
        self.config.api_key = self.api_key_edit.text().strip()
        self.config.model = self.model_combo.currentText().strip()
        # 非法数值不能静默回退默认：旧实现 try/except 后直接覆盖，用户输入
        # 任何垃圾（如全角数字、粘贴带了单位 "1.0s"）都会被悄悄换成默认值，
        # 用户以为自己的设置已生效。回退时把对应输入框复位为实际生效值并在
        # 状态栏点名，让回退可见。
        invalid: list = []

        def _num(widget, attr: str, default: float, cast, label: str) -> None:
            raw = widget.text().strip()
            try:
                value = cast(raw) if raw else default
            except ValueError:
                value = default
                widget.setText(str(default))
                invalid.append(f"{label}={raw!r}")
            setattr(self.config, attr, value)

        _num(self.temp_edit, "temperature", 0.7, float, "温度")
        _num(self.max_tokens_edit, "max_tokens", 32768, int, "最大Token")
        # 钳制：负值在下游有破坏性（temperature 为负被多数 API 拒绝）。钳到合法
        # 下限并记录到 invalid 列表（与非法值回退共用状态栏提示）。
        # （request_timeout 不再暴露为 UI 项，保留默认 120s，故此项无钳制。）
        if self.config.temperature < 0:
            self.config.temperature = 0.0
            self.temp_edit.setText("0.0")
            invalid.append("温度<0 → 0")
        # Keep max_context_tokens in sync with max_tokens: the shrink threshold
        # should track the model's actual context window. If the user has
        # explicitly set a different max_context_tokens via config.json, only
        # override when it was smaller than the new max_tokens (i.e. the old
        # 4096 budget that's now too small).
        if self.config.max_context_tokens < self.config.max_tokens:
            self.config.max_context_tokens = self.config.max_tokens
        _num(self.top_p_edit, "top_p", 0.95, float, "top_p")
        _num(self.min_p_edit, "min_p", 0.05, float, "min_p")
        _num(self.top_k_edit, "top_k", 20, int, "top_k")
        _num(self.repeat_penalty_edit, "repetition_penalty", 1.0, float, "重复惩罚")
        self.config.local_llm_enable_thinking = self.thinking_check.isChecked()
        self.config.local_llm_kv_reuse = self.kv_reuse_check.isChecked()
        self.config.local_llm_kv_quantize = self.kv_quant_check.isChecked()
        # 辅助模型参数（本地专用）
        _num(self.aux_temp_edit, "aux_temperature", 0.7, float, "辅助温度")
        _num(self.aux_max_tokens_edit, "aux_max_tokens", 32768, int, "辅助最大Token")
        _num(self.aux_top_p_edit, "aux_top_p", 0.95, float, "辅助top_p")
        _num(self.aux_min_p_edit, "aux_min_p", 0.05, float, "辅助min_p")
        _num(self.aux_top_k_edit, "aux_top_k", 20, int, "辅助top_k")
        _num(self.aux_repeat_penalty_edit, "aux_repetition_penalty", 1.0, float, "辅助重复惩罚")
        self.config.aux_local_llm_enable_thinking = self.aux_thinking_check.isChecked()
        self.config.aux_local_llm_kv_reuse = self.aux_kv_reuse_check.isChecked()
        self.config.aux_local_llm_kv_quantize = self.aux_kv_quant_check.isChecked()
        if invalid:
            # 状态栏提示一次；不打断流程（回退值是安全的默认）。
            self.status_label.setText(
                f"⚠ 非法配置已回退默认值：{'、'.join(invalid)}（输入框已复位）"
            )
            self._append_log(f"[cfg] invalid values fell back to defaults: {'; '.join(invalid)}")
        ws = self.workspace_edit.text().strip()
        if ws:
            self.config.workspace = ws
        self.config.ensure_workspace()
        kb = self.knowledge_base_edit.text().strip()
        if not kb:
            # Default to <workspace>/knowledge_base so RAG tools work out of
            # the box even when the user hasn't explicitly set a path.
            kb = os.path.join(self.config.workspace, "knowledge_base")
        self.config.knowledge_base = kb
        # 主模型模式（本地 / OpenAI 兼容接口）+ 本地类型与路径
        self.config.llm_provider = (
            "local" if self.main_mode_combo.currentText() == "本地模式" else "api")
        self.config.local_llm_backend = _LLM_TYPE_TO_BACKEND.get(
            self.main_type_combo.currentText(), "mlx")
        self.config.local_llm_model = self.local_llm_model_edit.text().strip()
        self.config.local_llm_mmproj_path = self.main_mmproj_edit.text().strip()
        self.config.local_llm_tokenizer_path = self.main_tokenizer_edit.text().strip()
        # 辅助模型（本地专用）
        self.config.aux_llm_enabled = self.aux_enabled_check.isChecked()
        self.config.aux_llm_provider = "local"
        self.config.aux_local_llm_backend = _LLM_TYPE_TO_BACKEND.get(
            self.aux_type_combo.currentText(), "mlx")
        self.config.aux_local_llm_model = self.aux_local_llm_model_edit.text().strip()
        self.config.aux_local_llm_mmproj_path = self.aux_mmproj_edit.text().strip()
        self.config.aux_local_llm_tokenizer_path = self.aux_tokenizer_edit.text().strip()
        # Embedding / Reranker：实现下拉 → (provider, backend)；模型名与本地路径分别落位
        emb_impl = self.embedding_impl_combo.currentText().strip()
        self.config.embedding_provider, self.config.local_embedding_backend = \
            _impl_to_provider_backend(emb_impl)
        # 仅 fastembed 模式写模型名（网络模型 id）；本地模式由「路径」决定模型，
        # 隐藏的模型名字段不参与落位，避免旧值覆盖配置（见 _sync_impl_fields）。
        if emb_impl == "fastembed":
            name = self.embedding_model_name_edit.text().strip()
            if name:
                self.config.rag_embedding_model = name
        self.config.local_embedding_model = self.embedding_path_edit.text().strip()
        rk_impl = self.reranker_impl_combo.currentText().strip()
        self.config.reranker_provider, self.config.local_reranker_backend = \
            _impl_to_provider_backend(rk_impl)
        if rk_impl == "fastembed":
            rname = self.reranker_model_name_edit.text().strip()
            if rname:
                self.config.rag_rerank_model = rname
        self.config.local_reranker_model = self.reranker_path_edit.text().strip()
        # TopN：reranker 之后保留的结果条数（rag_search_top_k）。先走统一数值校验
        # （非法值回退 5 并在状态栏提示），再夹紧到 1~10，与 rag_search 检索时的
        # clamp 上限一致，避免用户误填后检索被静默截断到 10。
        _num(self.reranker_topn_edit, "rag_search_top_k", 5, int, "TopN")
        self.config.rag_search_top_k = max(1, min(int(self.config.rag_search_top_k), 10))
        # 切片参数：语义切片开关 / 大小 / 重叠（单位 token）。重叠以显式 token 为
        # 准（关闭百分比驱动），使「重叠」输入直接生效，并随切片大小同步到所有
        # 文本分片环节（递归 / 语义 / 混合分片 + _clamp_chunk_budget 预算收口），
        # 避免脱节。取消「语义切片」时，chunk_documents 直接走确定性递归分片，
        # 不加载任何嵌入模型。
        self.config.rag_chunk_semantic = self.semantic_chunk_chk.isChecked()
        _num(self.chunk_size_edit, "rag_chunk_size", 800, int, "切片大小")
        _num(self.chunk_overlap_edit, "rag_chunk_overlap", 50, int, "重叠")
        self.config.rag_matryoshka_enabled = self.matryoshka_chk.isChecked()
        _dims_save = self.embedding_dims_combo.currentText().strip()
        self.config.rag_embedding_dims = (
            int(_dims_save) if _dims_save in ("128", "256", "512") else 256)
        self.config.rag_chunk_overlap_percent = 0.0
        # PDF 解析引擎（docling | marker | ovis）：显示标签 → 内部值。
        self.config.rag_pdf_engine = _PDF_DISPLAY_TO_VALUE.get(
            self.pdf_engine_combo.currentText().strip(), "docling")
        # 并发实例（rag_docling_pool_size，docling 转换器池）：下拉只有 1-6，仍夹紧兜底。
        try:
            _dl_pool = int(self.parse_concurrency_combo.currentText())
        except (TypeError, ValueError):
            _dl_pool = 1
        self.config.rag_docling_pool_size = min(max(_dl_pool, 1), 6)
        # OCR并发（rag_ocr_pool_size，RapidOCR 引擎池）：同样夹紧 1-6。
        try:
            _ocr_pool = int(self.ocr_concurrency_combo.currentText())
        except (TypeError, ValueError):
            _ocr_pool = 1
        self.config.rag_ocr_pool_size = min(max(_ocr_pool, 1), 6)
        # OCR 模型（OvisOCR2，#133）：实现/模型名/本地路径
        self.config.rag_ocr_engine = self.ocr_impl_combo.currentText().strip() or "mlx-vlm"
        # 模型名不暴露给 UI：保持 config 默认值（OvisOCR2-0.8B-MLX-4bit），
        # 避免用户手填拼错导致 OCR 模型加载失败。
        self.config.rag_ocr_model_path = self.ocr_path_edit.text().strip()
        self.config.rag_ocr_mmproj_path = self.ocr_mmproj_edit.text().strip()
        self.config.rag_ocr_tokenizer_path = self.ocr_tokenizer_edit.text().strip()

    def _rebuild_agent(self, rebuild_aux: bool = True) -> None:
        # 常驻 worker 持有旧 agent 引用：先让它退出循环，否则重建后用户再发
        # 消息会打进已 close 的旧 agent（MCP/httpx 已关）。应用配置只在空闲时
        # 可点（_set_controls_busy），所以线程此刻必然阻塞在 queue.get，
        # shutdown 的哨兵会让它立刻退出。
        old_worker = getattr(self, "_worker", None)
        if old_worker is not None:
            try:
                old_worker.shutdown()
            except Exception:
                pass
            if self._is_worker_running(old_worker):
                old_worker.wait(2000)
                # wait 返回 False 表示 2s 内未结束（agent 正卡在一个长工具
                # 调用里）。旧实现直接置 None：QThread 对象的 Python 引用消失，
                # 但底层线程仍在跑，下次「应用配置」又 spawn 一个——积累成
                # 孤儿线程列表，每个都持旧 agent 句柄。收进 _orphan_workers，
                # closeEvent 时再统一 wait（见 __init__ 注释）。
                if old_worker.isRunning():
                    self._orphan_workers.append(old_worker)
            self._worker = None
        self._gather_config_from_widgets()
        # 配置可能改了 PDF 引擎 / OvisOCR2 模型——重新注入 RAG 层（与 __init__
        # 的同款调用保持一致，否则改设置要重启应用才生效）。
        try:
            from agent.rag import engine as _rag_engine_mod
            _rag_engine_mod.set_pdf_engine(
                getattr(self.config, "rag_pdf_engine", "docling"))
            _rag_engine_mod.set_ovis_model_config(
                getattr(self.config, "rag_ocr_engine", "mlx-vlm"),
                getattr(self.config, "rag_ocr_model_name", "OvisOCR2-0.8B-MLX-4bit"),
                getattr(self.config, "rag_ocr_model_path", ""),
            )
            from agent.rag.docling_parser import set_ocr_scale_config
            set_ocr_scale_config(
                base=getattr(self.config, "rag_ocr_scale_base", 2.0),
                dynamic=getattr(self.config, "rag_ocr_scale_dynamic", False),
                table=getattr(self.config, "rag_ocr_scale_table", 2.5),
            )
            # OCR 引擎池 + docling 转换器池（UI「OCR并发」/「并发实例」）：此前主应用
            # 路径漏了 set_ocr_config，导致改 OCR 并发数只能靠重启生效；这里补齐，
            # 与 __init__ / 转换工具前置注入保持一致。
            from agent.rag import parsers as _rag_parsers
            _rag_parsers.set_ocr_config(self.config)
            from agent.rag.docling_parser import set_docling_pool_size
            set_docling_pool_size(getattr(self.config, "rag_docling_pool_size", 1))
        except Exception:
            pass
        # 重建会换掉主模型，先把主按钮复位为「加载模型」未就绪态，待后台预加载
        # 成功后再置绿（仅当按钮已创建时操作）。仅重载主模型时（rebuild_aux=False
        # ）保留辅助模型：aux client 与已加载权重都不动，按钮态也保持不变。
        if getattr(self, "load_main_btn", None) is not None:
            self._mark_model_unloaded("main")
            if rebuild_aux:
                self._mark_model_unloaded("aux")
        # Shut down the old agent's resources (MCP subprocesses, httpx client)
        # before replacing it — otherwise each "应用配置" click leaks a set of
        # MCP server subprocesses and an open httpx connection.
        old_agent = getattr(self, "_agent", None)
        if old_agent is not None:
            try:
                old_agent.close()
            except Exception:
                pass
            try:
                old_agent.tools.shutdown()
            except Exception:
                pass
        old_llm = getattr(self, "_llm", None)
        if old_llm is not None:
            try:
                old_llm.close()
            except Exception:
                pass
        # 辅助模型同样需要在重建前关闭，否则每次「加载/应用」都泄漏一份权重。
        # 仅重载主模型时（rebuild_aux=False）保留 aux——不关权重、不置空引用。
        old_aux = getattr(self, "_aux_llm", None)
        if old_aux is not None and rebuild_aux:
            try:
                old_aux.close()
            except Exception:
                pass
        # 构造失败时必须把两个引用都清空：旧的 agent 已经被 tools.shutdown()
        # （MCP 子进程已杀、RAG 引擎已关），旧的 llm 也 close() 了。若把引用
        # 留着，用户照常点「发送」会得到一个半死的 agent —— MCP 工具全部
        # Unknown tool、RAG 工具静默消失、LLM 调用抛
        # "Cannot send a request, as the client has been closed."。
        self._agent = None
        self._llm = None
        if rebuild_aux:
            self._aux_llm = None
        try:
            # 主 LLM：按 config.llm_provider 选择网络 API 或本地加载（工厂路由）。
            from ..local.factory import (
                build_llm_client, build_aux_llm_client, configure_local_models,
            )
            configure_local_models(self.config)  # 同步 embedding/reranker provider
            new_llm = build_llm_client(self.config)
            new_agent = Agent(llm=new_llm, config=self.config, callbacks=self._callbacks)
            new_agent.max_iterations = self.config.max_iterations
            # 辅助模型（本地）：用于事实抽取等轻量子任务。加载失败不影响主流程。
            # 未勾选「启用」时完全不构建——启动/应用配置/发送触发重建都不误启动
            # （client 本身惰性加载权重，但草稿解码接线也会一并跳过）。
            if rebuild_aux:
                chk = getattr(self, "aux_enabled_check", None)
                aux_enabled = bool(chk.isChecked()) if chk is not None else bool(
                    getattr(self.config, "aux_llm_enabled", False))
                if not aux_enabled:
                    logger.info("辅助模型未启用（启用勾选框未勾选），跳过构建")
                    new_aux = None
                else:
                    try:
                        new_aux = build_aux_llm_client(self.config)
                    except Exception as aux_err:
                        logger.warning("辅助模型加载失败（不影响主流程）: %s", aux_err)
                        new_aux = None
                new_agent.aux_llm = new_aux
                self._aux_llm = new_aux
                # 主模型复用已加载的辅助模型当投机解码草稿（不另起权重）。
                if new_aux is not None:
                    new_llm.set_draft_source(new_aux)
            else:
                # 仅重载主模型：复用已加载的辅助模型 client（含权重）与按钮态，
                # 不销毁、不重置；新主 client 指向同一个 aux，草稿解码继续可用。
                new_agent.aux_llm = self._aux_llm
                if self._aux_llm is not None:
                    new_llm.set_draft_source(self._aux_llm)
        except Exception as e:
            self._on_error(f"无法创建 agent: {e}")
            # 保持 None：_on_send 会给出明确提示，而不是用一个坏掉的 agent。
            return
        self._llm = new_llm
        self._agent = new_agent
        # 重建成功 → 强杀标志复位：旧实现一次强杀后该标志恒为 True，之后
        # 每次退出都跳过 tools.shutdown()/agent.close()，泄漏 MCP 子进程与
        # httpx 连接池（关闭路径被永久短路）。新 agent 是全新实例，锁无残留。
        self._worker_terminated = False

    # ------------------------------------------------------------------
    # config actions
    # ------------------------------------------------------------------
    def _on_apply_config(self) -> None:
        # Guard: if the agent is currently running, rebuilding would shut down
        # the old agent's tools/MCP/httpx mid-execution, causing the running
        # AgentWorker to crash when its next tool call hits destroyed tools.
        # The user must wait for the current turn to finish (or stop it first).
        if self._busy:
            QMessageBox.warning(
                self, "正在运行",
                "Agent 正在运行，请先点击「终止对话」停止当前任务，再应用配置。",
            )
            return
        # 先收集控件再保存：旧实现先 save 再 rebuild（rebuild 内部才 gather），
        # 新填的配置不写盘，重启后回旧值。
        self._gather_config_from_widgets()
        self.config.save(CONFIG_PATH)
        self._rebuild_agent()
        self._append_log(f"[cfg] applied: {self.config.base_url} model={self.config.model}")
        self.status_label.setText("配置已应用")

    def _on_pick_workspace(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择工作目录", self.config.workspace)
        if d:
            self.workspace_edit.setText(d)

    def _on_pick_knowledge_base(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择知识库目录", self.config.knowledge_base or self.config.workspace)
        if d:
            self.knowledge_base_edit.setText(d)

    @staticmethod
    def _is_worker_running(worker) -> bool:
        """``worker.isRunning()`` that tolerates an already-deleted QThread.

        Workers are cleaned up with ``deleteLater``, so a stale reference can
        point at a QThread whose C++ object is gone — calling ``isRunning()``
        on it raises ``RuntimeError: wrapped C/C++ object has been deleted``
        and would take down the click handler.
        """
        if worker is None:
            return False
        try:
            return bool(worker.isRunning())
        except RuntimeError:
            return False

    # ------------------------------------------------------------------
    # 向量去重（清理内容完全相同的重复向量，保留每组最早入库的一份）
    # ------------------------------------------------------------------
    def _on_dedup_vectors(self) -> None:
        if self._is_worker_running(self._dedup_worker):
            logger.warning("向量去重正在进行，忽略重复请求")
            return
        if self._is_worker_running(self._sync_worker):
            QMessageBox.information(self, "提示", "知识同步正在进行，请先完成或停止同步再执行向量去重。")
            return
        ret = QMessageBox.question(
            self, "向量去重",
            "将扫描向量库中内容完全相同的向量，每组只保留最早入库的一份，"
            "其余删除。\n\n用于清理不同目录下同一本书等原因造成的重复向量，"
            "降低检索噪声。\n\n此操作不可撤销，确定继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ret != QMessageBox.StandardButton.Yes:
            return
        self._dedup_worker = VectorDedupWorker(self.config)
        self._dedup_worker.dedup_log.connect(self._on_dedup_log)
        self._dedup_worker.dedup_progress.connect(self._on_dedup_progress)
        self._dedup_worker.dedup_finished.connect(self._on_dedup_finished)
        self._dedup_worker.dedup_failed.connect(self._on_dedup_failed)
        # deleteLater 必须挂 QThread 内建 finished（run() 返回后触发），
        # 挂在自定义信号上会在 run() 执行途中销毁线程对象。
        self._dedup_worker.finished.connect(self._dedup_worker.deleteLater)
        self.dedup_vectors_btn.setEnabled(False)
        self._set_kb_sync_state(enabled=False)
        self.progress.setRange(0, 0)      # busy 态（总数未知）
        self.progress.setFormat("向量去重扫描中…")
        self._dedup_worker.start()
        logger.info("向量去重已启动: kb=%s", self.config.knowledge_base)

    def _on_dedup_log(self, line: str) -> None:
        self._append_log(line)

    def _on_dedup_progress(self, cur: int, total: int, msg: str) -> None:
        if total > 0:
            self.progress.setRange(0, total)
            self.progress.setValue(cur)
        self.status_label.setText(msg)

    def _on_dedup_finished(self, stats: dict) -> None:
        self.dedup_vectors_btn.setEnabled(True)
        self._set_kb_sync_state(enabled=True)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        # 清引用：worker 已挂 deleteLater，C++ 对象随后被销毁，继续持有
        # 会让 closeEvent 里的 isRunning() 抛 RuntimeError。
        self._dedup_worker = None
        scanned = int(stats.get("scanned", 0))
        groups = int(stats.get("groups", 0))
        removed = int(stats.get("removed", 0))
        self.status_label.setText("向量去重完成")
        logger.info("向量去重完成: 扫描 %d 条, %d 组重复, 删除 %d 条",
                    scanned, groups, removed)
        if removed:
            QMessageBox.information(
                self, "向量去重完成",
                f"扫描 {scanned} 条向量，发现 {groups} 组内容完全相同的重复，\n"
                f"已删除 {removed} 条（每组保留最早入库的一份）。")
        else:
            QMessageBox.information(
                self, "向量去重完成",
                f"扫描 {scanned} 条向量，未发现内容完全相同的重复。")

    def _on_dedup_failed(self, error: str) -> None:
        self.dedup_vectors_btn.setEnabled(True)
        self._set_kb_sync_state(enabled=True)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        self._dedup_worker = None
        self.status_label.setText("向量去重失败")
        logger.warning("向量去重失败: %s", error[:300])
        QMessageBox.critical(self, "向量去重失败", error[:500])

    def _set_kb_sync_state(self, text=None, enabled=None) -> None:
        """刷新「开始同步」按钮状态：顶栏快捷按钮 + 设置区按钮两处保持一致。

        两处默认文案不同（顶栏「知识同步」/ 设置区「开始同步」），故恢复时
        各自回到自己的默认标签；进行中的临时文案（检查依赖…/同步中…）一致。
        按钮尚未创建时（初始化早期）安全跳过。
        """
        for btn, default in ((getattr(self, "sync_knowledge_btn", None), "开始同步"),
                             (getattr(self, "kb_sync_btn", None), "知识同步")):
            if btn is None:
                continue
            if enabled is not None:
                btn.setEnabled(enabled)
            btn.setText(default if text is None else text)

    def _on_sync_knowledge(self, force: bool = False) -> None:
        if self._is_worker_running(self._sync_worker):
            logger.warning("Sync already running, ignoring duplicate request")
            return
        if self._is_worker_running(self._deps_worker):
            logger.warning("Deps check already running, ignoring duplicate request")
            return
        # 与 agent 运行互斥：ingest 与 agent 的 rag_search 共享同一 LanceDB
        # 向量表，并发时一边写一边读会撞上「table modified during query」
        # 的 RuntimeError；engine 的 _ingest_lock 只串行化 ingest 之间，挡
        # 不住 agent 的检索。agent 繁忙时直接拒绝并提示先终止对话。
        if self._busy:
            QMessageBox.warning(
                self, "Agent 运行中",
                "Agent 正在执行（可能正在检索知识库）。请先「终止对话」后再"
                "同步，避免 ingest 与检索同时操作向量表。")
            return
        kb = self.knowledge_base_edit.text().strip()
        ws = self.workspace_edit.text().strip() or os.path.expanduser("~/.base-agent/workspace")
        if not kb:
            QMessageBox.warning(self, "缺少知识库", "请先选择知识库目录")
            return
        # 让知识同步区的 Embedding / Reranker 选择即时生效：把当前下拉与路径
        # 收敛进 config 并同步到 vector_store 的 provider 开关（否则需先点
        # 「应用配置/加载模型」才能让新建向量库用上所选实现）。
        try:
            self._gather_config_from_widgets()
            from ..local.factory import configure_local_models
            configure_local_models(self.config)
        except Exception as e:
            logger.warning("同步前应用 Embedding/Reranker 设置失败: %s", e)
        logger.info("Knowledge sync triggered: kb=%s workspace=%s force=%s", kb, ws, force)
        self._sync_active = True
        # Step 1: check dependencies before starting sync
        self._set_kb_sync_state("检查依赖…", False)
        # Enable stop-sync during the deps-check phase too so the user can
        # abort the whole pipeline before sync even starts.
        self.stop_sync_btn.setEnabled(True)
        self.status_label.setText("正在检查依赖…")
        self.progress.setRange(0, 0)
        self._pending_sync_ws = ws
        self._pending_sync_kb = kb
        self._pending_sync_force = force
        logger.info("UI: starting sync — kb=%s force=%s", kb, force)
        self._append_log("[rag] ────────── 开始知识库同步 ──────────")
        self._deps_worker = InstallDepsWorker(kb)
        self._deps_worker.progress.connect(self._on_deps_progress)
        self._deps_worker.deps_finished.connect(self._on_deps_finished)
        # deleteLater 必须挂到 QThread 内建的 finished（run() 返回后触发），
        # 挂在自定义信号上会在 run() 执行途中销毁线程对象。
        self._deps_worker.finished.connect(self._deps_worker.deleteLater)
        # worker 引用也只在 built-in finished（run() 已返回）时清理。
        # deps_finished 是 run() 内部 emit 的自定义信号，槽里把
        # self._deps_worker 置 None 会让最后的 Python 引用归零 → sip 立刻
        # 析构**仍在运行**的 C++ QThread → qFatal → SIGABRT（2026-10-26
        # 崩溃报告实证：PyObject_SetAttr → subtype_dealloc → QThread::~QThread）。
        self._deps_worker.finished.connect(self._on_deps_thread_finished)
        self._deps_worker.start()

    def _on_deps_thread_finished(self) -> None:
        """QThread 内建 finished（run() 已返回）——此刻清 worker 引用才安全。"""
        self._deps_worker = None

    def _on_deps_progress(self, msg: str) -> None:
        self.status_label.setText(msg)
        self._append_log(f"[deps] {msg}")

    def _on_deps_finished(self, report) -> None:
        # 注意：这里**不能**置 self._deps_worker = None——deps_finished 是
        # run() 内部 emit 的，此刻 C++ 线程多半还活着；引用由
        # _on_deps_thread_finished（built-in finished）统一清理。
        # 检查用户是否在依赖检查期间点击了停止同步
        if self._deps_cancelled:
            self._deps_cancelled = False
            logger.info("Deps check finished but user cancelled — aborting sync")
            # 必须复位：_sync_active 是 _set_controls_busy 的开关，漏复位会让
            # 同步按钮从此永久脱离 busy 管理（不随 agent 忙闲翻转）。下面两个
            # 提前 return 是同一条路径，一并处理。
            self._sync_active = False
            self._set_kb_sync_state(enabled=True)
            self.stop_sync_btn.setEnabled(False)
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.status_label.setText("已取消同步")
            self._append_log("[rag] 已取消同步")
            return
        if report is not None and report.has_blocking:
            # Required deps still missing after auto-install attempt
            logger.warning("Deps check: blocking dependencies missing, aborting sync")
            self._sync_active = False
            self._set_kb_sync_state(enabled=True)
            self.stop_sync_btn.setEnabled(False)
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.status_label.setText("依赖缺失，无法同步")
            QMessageBox.critical(
                self, "依赖缺失",
                f"以下必要依赖未能自动安装，请手动安装后重试：\n\n{report.summary()}"
            )
            return
        if report is None:
            # worker 的 None 只有「用户取消」一种正常语义（上面已处理）；
            # ensure_dependencies 抛异常时也 emit None。依赖状态未知/缺失时
            # 绝不能照常启动 RAG 同步——那会在缺 lancedb/fastembed 的环境里
            # 跑引擎，产生难懂的二次异常并吞掉真正的失败原因。
            logger.error("Deps check failed with an exception — aborting sync")
            self._sync_active = False
            self._set_kb_sync_state(enabled=True)
            self.stop_sync_btn.setEnabled(False)
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.status_label.setText("依赖检查失败")
            QMessageBox.critical(
                self, "依赖检查失败",
                "依赖检查过程中出现异常，无法确认解析依赖是否可用。\n"
                "请查看日志面板中的错误信息，修复后重试。"
            )
            return

        # Step 2: start sync
        # pending 快照一次性消费：本方法只会由 deps worker（_on_sync_knowledge
        # 启动）触发，但残留的旧值若不被清掉，后续路径读到的是过期意图。
        ws = getattr(self, "_pending_sync_ws", "")
        kb = getattr(self, "_pending_sync_kb", "")
        force = getattr(self, "_pending_sync_force", False)
        self._pending_sync_ws = ""
        self._pending_sync_kb = ""
        self._pending_sync_force = False
        if not ws or not kb:
            self._set_kb_sync_state(enabled=True)
            self.stop_sync_btn.setEnabled(False)
            # 防御分支同样要复位不确定态进度条——_on_sync_knowledge 已把它设
            # 为 setRange(0,0)，不恢复的话进度条会永久转圈。
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.status_label.setText("未选择知识库目录")
            return
        self._set_kb_sync_state("同步中…")
        self.status_label.setText("正在同步知识库…")
        self.progress.setRange(0, 0)
        # stop_sync_btn stays enabled (it was enabled in _on_sync_knowledge)
        # so the user can cancel mid-sync.
        self._append_log("[rag] 依赖检查完成，启动同步引擎...")
        # 同步期间临时禁用 QSignalLogHandler 的 RAG 日志过滤，
        # 避免 QSignalLogHandler 和 _poll_sync_logs 同时操作 log_view
        # 导致的并发崩溃。RAG 日志在同步期间完全由 _log_buffer + QTimer 处理。
        if hasattr(self, '_log_handler') and self._log_handler is not None:
            self._log_handler._sync_active = True
        # 同步开始前临时卸载「同步前已加载」的本地主/辅模型，释放显存给 RAG 入库
        # （解析/清洗/切片/向量化是大内存操作，且只用到 embedding/reranker，不依赖
        # 对话 LLM）。记住卸载状态，待同步结束后原样重载（含主模型预热）。
        # API 模式主模型无本地权重，不处理。
        _local_mode = getattr(self.config, "llm_provider", "local") == "local"
        self._sync_unloaded_main = bool(_local_mode and getattr(self, "_llm", None) is not None)
        # 辅助模型：只要已加载就照常临时卸载（同步要吃内存）；但「同步结束后
        # 自动重载」的标记只在「启用」勾选时置位——用户未启用/已取消勾选时，
        # 同步一结束就把辅助模型悄悄拉回来属于误启动。
        chk = getattr(self, "aux_enabled_check", None)
        _aux_enabled = bool(chk.isChecked()) if chk is not None else bool(
            getattr(self.config, "aux_llm_enabled", False))
        _aux_was_loaded = getattr(self, "_aux_llm", None) is not None
        self._sync_unloaded_aux = bool(_aux_enabled and _aux_was_loaded)
        if self._sync_unloaded_main:
            self._on_unload_main_model()
        if _aux_was_loaded:
            self._on_unload_aux_model()
        self._append_log("[rag] 已临时卸载对话模型以释放资源，同步结束后将自动重载")
        self._sync_worker = SyncKnowledgeWorker(
            ws, kb,
            embedding_model=self.config.rag_embedding_model,
            force=force,
            config=self.config,
        )
        self._sync_worker.sync_finished.connect(self._on_sync_finished)
        self._sync_worker.sync_failed.connect(self._on_sync_failed)
        self._sync_worker.sync_progress.connect(self._on_sync_progress)
        self._sync_worker.finished.connect(self._on_sync_thread_finished)
        self._sync_worker.finished.connect(self._sync_worker.deleteLater)
        self._sync_worker.start()
        # 启动日志轮询定时器：每 100ms 消费 _log_buffer，实时显示同步日志
        self._log_poll_timer.start()

    def _on_sync_finished(self, stats: dict) -> None:
        logger.info("UI: _on_sync_finished called")
        self._sync_active = False
        self._log_poll_timer.stop()
        # 恢复 QSignalLogHandler 的 RAG 日志过滤
        if hasattr(self, '_log_handler') and self._log_handler is not None:
            self._log_handler._sync_active = False
        # 最后消费一次 _log_buffer，确保残余日志被显示
        self._poll_sync_logs()
        try:
            self._set_kb_sync_state(enabled=True)
            self.stop_sync_btn.setEnabled(False)
            # Fully reset the progress bar: both range AND value. Without
            # setValue(0), a stale value from the last _on_sync_progress
            # call can persist (clamped to the new max), making the bar
            # look "stuck" at 100%.
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            # 输出诊断信息：确认信号已正确接收
            if hasattr(self, '_log_handler') and self._log_handler is not None:
                logger.info("UI: sync_finished received, emit_count=%d emit_errors=%d",
                            self._log_handler._emit_count, self._log_handler._emit_errors)
            # 这里**不能**置 self._sync_worker = None：sync_finished 是 run()
            # 内部 emit 的自定义信号，此刻 C++ 线程还在收尾（释放 docling
            # converter 等），清掉最后的 Python 引用会立即 sip 析构运行中的
            # QThread → qFatal → SIGABRT（2026-10-26 崩溃报告实证）。引用由
            # _on_sync_thread_finished（built-in finished，run() 已返回）清理。
            # 刷新 agent 的 RAG 引擎表句柄，让下一次搜索看到新写入的数据。
            # LanceDB 表对象在打开时获取版本快照，同步后不重新打开会看到旧数据。
            if self._agent is not None:
                try:
                    # Agent.reload_rag 除了刷新引擎句柄，还会失效系统提示里的
                    # 知识库状态缓存，否则同步后注入的仍是旧 chunks 数。
                    self._agent.reload_rag()
                except Exception as e:
                    logger.warning("UI: reload_rag after sync: %s", e)
            # 防御：stats 可能不是 dict（极端情况下为 None）
            if not isinstance(stats, dict):
                stats = {}
            if stats.get("cancelled"):
                self.status_label.setText("同步已取消")
                self._append_log("[rag] 同步已用户取消")
                return
            self.status_label.setText("知识库同步完成")
            files = stats.get("files_found", 0)
            extracted = stats.get("files_extracted", 0)
            skipped = stats.get("files_skipped", 0)
            deleted = stats.get("files_deleted", 0)
            chunks = stats.get("chunks", 0)
            errors = stats.get("errors", [])
            # Show incremental stats so the user sees what was actually processed.
            if extracted == 0 and skipped > 0:
                summary = f"知识库已是最新（跳过 {skipped} 个未修改文件）\n\n"
            else:
                summary = (
                    f"知识库同步完成！\n\n"
                    f"扫描文件：{files} 个\n"
                    f"新增/更新：{extracted} 个   跳过：{skipped} 个\n"
                )
                if deleted > 0:
                    summary += f"已清理删除文件：{deleted} 个\n"
                summary += f"向量切片：{chunks} 个\n"
            msg = summary
            if errors:
                msg += f"\n⚠ {len(errors)} 个文件提取失败：\n"
                for e in errors[:5]:
                    msg += f"  - {e}\n"
                if len(errors) > 5:
                    msg += f"  …（还有 {len(errors) - 5} 个）\n"
            msg += "\n后续对话将自动参考知识库内容进行回答。"
            QMessageBox.information(self, "同步完成", msg)
        except Exception as e:
            logger.error("UI: _on_sync_finished error: %s", e, exc_info=True)
            # 确保按钮状态恢复，即使弹窗失败
            try:
                self._set_kb_sync_state(enabled=True)
                self.stop_sync_btn.setEnabled(False)
                self.progress.setRange(0, 1)
                self.progress.setValue(0)
                self.status_label.setText("同步完成（显示结果时出错）")
                # 引用清理不在异常路径做（同上），由 _on_sync_thread_finished 收尾。
            except Exception:
                pass

    def _on_sync_thread_finished(self) -> None:
        """兜底检查：QThread.finished 信号在 run() 返回后自动发射。

        如果 sync_finished / sync_failed 信号已正常处理，此时按钮应该已经
        恢复为"同步知识"状态。如果按钮仍处于同步相关状态，说明同步结束信号
        未被正确处理，需要强制恢复 UI 状态以避免用户界面永久卡住。

        注意：不检查 self._sync_worker is not None，因为 _on_sync_finished
        可能在异常路径中已将其设为 None，但按钮状态未恢复。
        """
        self._log_poll_timer.stop()
        if hasattr(self, '_log_handler') and self._log_handler is not None:
            self._log_handler._sync_active = False
        self._poll_sync_logs()
        btn_text = self.sync_knowledge_btn.text()
        if btn_text in ("同步中…", "检查依赖…", "正在同步知识库…"):
            logger.warning(
                "UI: sync thread finished but button still in '%s' state. "
                "sync_finished/sync_failed signal may have been lost. "
                "Forcing UI reset.",
                btn_text,
            )
            try:
                self._set_kb_sync_state(enabled=True)
                self.stop_sync_btn.setEnabled(False)
                self.progress.setRange(0, 1)
                self.progress.setValue(0)
                self.status_label.setText("知识库同步完成（状态已自动恢复）")
            except Exception:
                pass
        # 同步结束后，把「同步前已加载」的本地主/辅模型原样重载（释放的显存回收），
        # 主模型重载后会自动预热（预填系统提示 KV + warm_up 编译）。仅当该模型仍
        # 处于未加载态时才重载——若用户在同步期间手动重载过，则不重复操作。
        _reload_main = getattr(self, "_sync_unloaded_main", False) and getattr(self, "_llm", None) is None
        # 重载辅助模型前再查一次「启用」勾选的当前状态：用户可能在同步期间
        # （或停止同步前）把勾选取消——以勾选框为准，未启用绝不自动拉起。
        chk = getattr(self, "aux_enabled_check", None)
        _aux_enabled_now = bool(chk.isChecked()) if chk is not None else bool(
            getattr(self.config, "aux_llm_enabled", False))
        _reload_aux = (_aux_enabled_now
                       and getattr(self, "_sync_unloaded_aux", False)
                       and getattr(self, "_aux_llm", None) is None)
        if _reload_aux or _reload_main:
            self.status_label.setText("同步完成，正在重新加载对话模型…")
            self._append_log("[rag] 同步结束，开始重新加载对话模型（含主模型预热）")
        if _reload_aux:
            try:
                self._on_load_aux_model()
            except Exception as _e:
                logger.warning("同步后重载辅助模型失败: %s", _e)
        if _reload_main:
            try:
                self._on_load_main_model()
            except Exception as _e:
                logger.warning("同步后重载主模型失败: %s", _e)
        self._sync_unloaded_main = False
        self._sync_unloaded_aux = False
        # 清理 worker 引用，防止后续误判
        if self._sync_worker is not None:
            self._sync_worker = None

    def _on_sync_failed(self, error: str) -> None:
        logger.info("UI: _on_sync_failed called: %s", error[:200])
        self._sync_active = False
        self._log_poll_timer.stop()
        if hasattr(self, '_log_handler') and self._log_handler is not None:
            self._log_handler._sync_active = False
        self._poll_sync_logs()
        if hasattr(self, '_log_handler') and self._log_handler is not None:
            logger.info("UI: sync_failed received, emit_count=%d emit_errors=%d",
                        self._log_handler._emit_count, self._log_handler._emit_errors)
        try:
            self._set_kb_sync_state(enabled=True)
            self.stop_sync_btn.setEnabled(False)
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.status_label.setText("知识库同步失败")
            # 不在此处清引用（sync_failed 是 run() 内部 emit 的自定义信号），
            # 由 _on_sync_thread_finished（built-in finished）统一清理。
            QMessageBox.critical(self, "同步失败", f"知识库同步失败：\n{error}")
        except Exception as e:
            logger.error("UI: _on_sync_failed error: %s", e, exc_info=True)
            try:
                self._set_kb_sync_state(enabled=True)
                self.stop_sync_btn.setEnabled(False)
                self.progress.setRange(0, 1)
                self.progress.setValue(0)
            except Exception:
                pass

    def _on_fetch_models(self) -> None:
        base_url = self.base_url_edit.text().strip()
        api_key = self.api_key_edit.text().strip()
        if not base_url:
            QMessageBox.warning(self, "缺少配置", "请先填写 Base URL")
            return
        # 并发守卫：self._fetch_worker 是旧 QThread 唯一的 Python 强引用。
        # 直接覆盖会让引用计数归零 → sip 立刻析构**仍在运行**的 C++ QThread，
        # Qt6 下是 qFatal 级 abort（线程还阻塞在 30s 超时的 list_models 里）。
        # 触发路径：点「获取模型」→ 发一条消息 → 该轮结束把 fetch_btn 重新
        # enable → 再点一次。（sync/deps 都有 _is_worker_running 守卫。）
        if self._is_worker_running(self._fetch_worker):
            logger.warning("Fetch models already running, ignoring duplicate request")
            return
        self.fetch_btn.setEnabled(False)
        self.status_label.setText("正在获取模型列表…")
        self.progress.setRange(0, 0)
        self._fetch_worker = FetchModelsWorker(base_url, api_key)
        self._fetch_worker.fetched.connect(self._on_models_fetched)
        self._fetch_worker.failed.connect(self._on_fetch_failed)
        self._fetch_worker.finished.connect(self._fetch_worker.deleteLater)
        self._fetch_worker.start()

    def _on_models_fetched(self, models: list) -> None:
        # 必须清空引用：worker 已挂 deleteLater，C++ 对象随后被销毁，继续持有
        # 会让 closeEvent 里的 isRunning() 抛 RuntimeError（PyQt 默认 qFatal
        # → 进程 abort）。典型触发：获取模型成功后关闭窗口。
        self._fetch_worker = None
        self.fetch_btn.setEnabled(True)
        self.progress.setRange(0, 1)
        current = self.model_combo.currentText().strip()
        self.model_combo.clear()
        self.model_combo.addItems(models)
        if current in models:
            self.model_combo.setCurrentText(current)
        elif models:
            self.model_combo.setCurrentIndex(0)
        self.status_label.setText(f"获取到 {len(models)} 个模型")

    def _on_fetch_failed(self, msg: str) -> None:
        self._fetch_worker = None
        self.fetch_btn.setEnabled(True)
        self.progress.setRange(0, 1)
        self.status_label.setText("获取模型失败")
        QMessageBox.warning(self, "获取模型失败", msg)

    # ------------------------------------------------------------------
    # chat send / stop
    # ------------------------------------------------------------------
    def _on_send(self) -> None:
        if self._busy:
            # 旧实现静默丢弃：用户按了 Enter 却毫无反应，像「输入框坏了」。
            # Agent 运行中不能发新消息（输入框可编辑但回车无效），但「终止对话」
            # 按钮此刻是启用的——卡住/跑太久时直接点它即可，不必干等。
            self.status_label.setText("正在运行中…（如需中断请点「终止对话」）")
            return
        if getattr(self, "_sync_active", False) or self._is_worker_running(
                self._sync_worker):
            # 知识同步进行中：此时主/辅模型已被卸载（self._agent 被置 None），
            # 往下走会在**主线程**里重建 Agent（加载模型 + 拉 MCP 子进程 +
            # 新建第二个 RAGEngine），既卡死 UI 又让 agent 侧引擎与 sync
            # worker 并发读写同一张 LanceDB 表（"table modified during
            # query"）。明确拒绝并说明原因。
            self.status_label.setText("知识同步进行中，暂不可发送消息")
            return
        text = self.send_editor.toPlainText().strip()
        if not text:
            return
        if self._agent is None:
            self._rebuild_agent()
        if self._agent is None:
            QMessageBox.warning(self, "未就绪", "Agent 未初始化，请先应用配置。")
            return

        self._append_user(text)
        self.send_editor.clear()
        self._busy = True
        # 流式状态必须在每回合开始时统一复位：上一次「强制终止」可能发生在
        # <summary> 已开未闭的时刻（_streaming_assistant/stripper 残留），
        # 不复位的话下一回合的整条回复会被吞进 stripper 缓冲区——用户看到
        # 空回复。比只在 _force_stop_agent 里复位更稳（它是兜底而非唯一入口）。
        self._streaming_assistant = False
        self._summary_stripper = _SummaryStripper()
        self._set_controls_busy(True)
        self.think_view.clear()
        # 上一回合（尤其被强杀的）可能残留未落盘的缓冲
        self._pending_content.clear()
        self._pending_reasoning.clear()
        self.speed_label.setText("0.0 tok/s")
        self._callbacks.reset()
        worker = self._worker
        if worker is None or not worker.set_input(text):
            # 首次发送，或常驻线程已被强杀 → （重）建一条。**不要**每回合新建：
            # MLX KV 缓存按创建线程隔离，换线程 = 每回合全量预填充（约 6.5s 的
            # 差距）。先入队再 start，线程一跑起来就有任务可取。
            worker = AgentWorker(self._agent, self._callbacks)
            self._worker = worker
            worker.set_input(text)
            worker.start()

    def _on_stop(self) -> None:
        # 终止当前 agent 对话（不再处理同步取消，同步取消由 _on_stop_sync 负责）
        self._callbacks.cancel()
        self.status_label.setText("已请求终止对话（将在当前工具完成后生效）")
        # 兜底：协作式取消只在工具边界生效，遇到不检查取消标志的阻塞点
        # （MCP tools/call 的 300s 等待、长 shell 子进程）就永远不返回。
        # 旧实现里 _busy 只在 _on_finished 复位 → 发送/清空/应用配置/同步
        # 全部被禁用，只能杀进程。这里挂一个看门狗强制收尾。
        if getattr(self, "_stop_watchdog", None) is None:
            self._stop_watchdog = QTimer(self)
            self._stop_watchdog.setSingleShot(True)
            self._stop_watchdog.timeout.connect(self._force_stop_agent)
        # 只在未激活时启动：QTimer.start() 会重置倒计时，连点「终止」可以把
        # 30s 强杀无限推迟。
        if not self._stop_watchdog.isActive():
            self._stop_watchdog.start(_STOP_FORCE_AFTER_MS)

    def _force_stop_agent(self) -> None:
        """看门狗：协作式取消超时后强制结束工作线程并复位 UI。"""
        if not self._busy:
            return
        worker = self._worker
        if worker is not None and self._is_worker_running(worker):
            logger.warning("UI: 终止请求超时，强制结束 agent 工作线程")
            self._append_log("[ui] 终止请求超时，强制结束工作线程")
            try:
                # 断的是 bridge（worker 自身没有信号，disconnect() 是 no-op）
                worker.disconnect_bridge()
            except (TypeError, RuntimeError):
                pass
            try:
                worker.terminate()
                if worker.wait(1000):
                    self._worker = None          # 确认已结束才放手引用
                else:
                    # wait 超时：线程仍在跑。丢掉最后一个 Python 引用会让
                    # sip 析构 QThread → qFatal "Destroyed while thread is
                    # still running"（MCP tools/call 300s 阻塞完全可能命中）。
                    # 挂到孤儿列表保留引用，进程退出时再统一 wait。
                    self._orphan_workers.append(worker)
                    self._worker = None
                    self._worker_terminated = True
            except Exception as e:
                logger.warning("UI: force stop error: %s", e)
                self._orphan_workers.append(worker)
                self._worker = None
                self._worker_terminated = True
        # 流式状态复位 + 缓冲落盘：强杀可能发生在 <summary> 已开未闭的时刻，
        # 不复位的话下一回合整条回复会被吞进 stripper 缓冲区（空回复 bug）；
        # 缓冲区里最后 ≤40ms 的文本不落盘会静默丢失。
        self._flush_stream(final=True)
        self._streaming_assistant = False
        self._summary_stripper = _SummaryStripper()
        # 本轮只生成了一半：缓存末尾对应「被截断的输出」，历史里却是已落盘的
        # 部分文本——两者不再是前缀关系，必须丢弃缓存，否则下一轮会错位续算。
        try:
            if self._agent is not None:
                self._agent.invalidate_kv_cache()
        except Exception as e:
            logger.debug("UI: invalidate_kv_cache on force stop failed: %s", e)
        self._busy = False
        self._set_controls_busy(False)
        self.status_label.setText("已强制终止（工具进程可能仍在后台运行）")

    def _on_stop_sync(self) -> None:
        """停止知识库同步：在当前文件处理完成后协作式取消。

        sync_worker.cancel() 设置 RAGEngine 内部的取消标志，
        ingest 循环在处理完当前文件后检查该标志并退出。
        """
        # 用 _is_worker_running() 而非直接 isRunning()：worker 挂了
        # deleteLater，C++ 对象可能已销毁，直接调用会抛 RuntimeError。
        if self._is_worker_running(self._sync_worker):
            logger.info("User requested sync cancellation")
            self._sync_worker.cancel()
            self.stop_sync_btn.setEnabled(False)  # 防止重复点击
            self.status_label.setText("已请求停止同步（将在当前文件完成后生效）")
            self._append_log("[rag] 用户请求停止同步")
            return
        # deps 检查阶段：记录取消意图，deps 完成后不启动同步。
        if self._is_worker_running(self._deps_worker):
            logger.info("User requested stop during deps check — will abort after deps")
            self._deps_worker.cancel()
            self._deps_cancelled = True
            self.stop_sync_btn.setEnabled(False)
            self.status_label.setText("正在取消…")
            self._append_log("[rag] 用户请求停止同步（依赖检查完成后取消）")
            return

    def _on_sync_progress(self, done: int, total: int, current: str) -> None:
        """同步进度反馈。日志刷新由 _poll_sync_logs 定时器独立处理。"""
        try:
            if done < 0:
                return
            if total > 0:
                self.progress.setRange(0, total)
                self.progress.setValue(done)
                pct = done * 100 // total
                self.status_label.setText(f"同步中 {done}/{total} ({pct}%) — {current}")
            else:
                self.progress.setRange(0, 0)
                self.status_label.setText(f"同步中 — {current}")
        except Exception as e:
            logger.debug("UI: _on_sync_progress error: %s", e)

    def _poll_sync_logs(self) -> None:
        """定时轮询 sync_worker._log_buffer，把新日志追加到 log_view。

        绕过 Qt queued signal 机制——ONNX Runtime 在 embedding 期间持有
        GIL，sync_progress 信号虽然是 queued 但主线程可能无法及时处理。
        用 QTimer 直接在主线程事件循环中轮询 buffer，只要 GIL 短暂释放
        （_embed_chunks 中的 sleep(0.01)），定时器回调就能执行。

        注意：本方法只负责日志区域的输出，不更新底部状态栏。
        状态栏由 _on_sync_progress 根据进度回调单独更新，避免重复。
        """
        try:
            worker = self._sync_worker
            if worker is None:
                return
            with worker._log_buffer_lock:
                if not worker._log_buffer:
                    return
                buf = worker._log_buffer
                worker._log_buffer = []
            # 批量插入：用 QTextCursor 一次性插入所有行，比逐行
            # appendPlainText 快得多（后者每次触发一次重绘）。
            cursor = self.log_view.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertText("\n" + "\n".join(buf))
            self._scroll_to_bottom(self.log_view)
        except Exception as e:
            logger.debug("UI: _poll_sync_logs error: %s", e)

    def _set_controls_busy(self, busy: bool) -> None:
        self.send_btn.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)
        self.connect_btn.setEnabled(not busy)
        self.fetch_btn.setEnabled(not busy)
        self.clear_btn.setEnabled(not busy)
        self.workspace_btn.setEnabled(not busy)
        self.knowledge_base_btn.setEnabled(not busy)
        # 知识库同步控件**不能**随 agent 忙闲翻转：sync_knowledge_btn 的使能
        # 由 _on_sync_knowledge / _on_sync_finished / deps 流程独立管理（运行
        # 中禁用、结束才恢复）。旧实现这里一律置 enabled=not busy，agent 一
        # 跑就把同步按钮变回可点，用户点了之后与 agent 检索抢同一向量表。
        # 进度条也不归这里管（同步用 0..N 范围，agent 用不定范围 busy 态）。
        if not getattr(self, "_sync_active", False):
            self._set_kb_sync_state(enabled=not busy)
            self.progress.setRange(0, 0 if busy else 1)

    # ------------------------------------------------------------------
    # streaming slots
    # ------------------------------------------------------------------
    def _append_user(self, text: str) -> None:
        self._append_chat(f'<p><b class="u">你</b>: {self._escape(text)}</p>')

    def _on_content(self, text: str) -> None:
        text = _sanitize_stream_text(text)
        if not text:
            return
        # 协议标签 <summary>…</summary> 是给内核的机器标记，不显示给用户
        text = self._summary_stripper.feed(text)
        if not text:
            return
        if not self._streaming_assistant:
            self._append_chat('<p><b class="a">助手</b>: </p>')
            self._streaming_assistant = True
            # 不在这里重置 _summary_stripper：每回合的重置已经在 _on_send 做过了。
            # 旧实现多此一举，而且时机更晚——首块以 "<summ" 结尾时，这里会把
            # feed() 刚建立的跨 chunk carry 状态整个丢掉，下一个 chunk 的 "ary>"
            # 就会被当正文显示出来（用户看到乱入的 ary>）。
        # 合并 chunk 再写入：100 tok/s 时逐块 insertText 会让 QTextDocument
        # 每秒做数百次「取光标 → 移到末尾 → 插入 → 写回光标 → 滚到底」，
        # 每次都触发一次全视口重绘。这里按 ~25fps 缓冲刷新。
        self._pending_content.append(text)
        if self._stream_timer is None:
            self._stream_timer = QTimer(self)
            self._stream_timer.setInterval(_STREAM_FLUSH_MS)
            self._stream_timer.timeout.connect(self._flush_stream)
        # 只在未运行时启动：QTimer.start() 会重置倒计时，若每个 chunk 都调
        # 一次，chunk 间隔 < 40ms 时定时器永远不会到期 —— 表现为「流式输出
        # 完全不渲染，最后一次性全部出现」。
        if not self._stream_timer.isActive():
            self._stream_timer.start()

    def _flush_stream(self, final: bool = False) -> None:
        """把缓冲的流式文本一次性写入 chat_view。

        ``final=True`` 表示回合结束：此时才把标签剥离器的残留吐出来
        （模型漏写 ``</summary>`` 时那些字符已不可能是完整标签，应当显示）。
        中途不能 flush —— 标签常跨多个 chunk 到达，提前吐会把 ``<summ``
        这样的前缀当正文输出。
        """
        if final:
            residue = self._summary_stripper.flush()
            if residue:
                self._pending_content.append(residue)
        timer = getattr(self, "_stream_timer", None)
        if timer is not None:
            timer.stop()
        # 推理流缓冲一并落盘（与正文同一个定时器节奏）
        self._flush_reasoning()
        buf = getattr(self, "_pending_content", None)
        if not buf:
            return
        text = "".join(buf)
        buf.clear()
        # 先记录「用户是否在底部」：插入内容本身会把 value < maximum，事后
        # 判断会把每个在底部的用户都误判成「已上翻」。
        sb = self.chat_view.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 2
        cursor = self.chat_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        # 用 insertText 而非 insertHtml：HTML 的空白折叠规则会丢掉 chunk 之间的
        # 前导/后随空格（"Hello" + " World" 会被渲染成 "HelloWorld"）。
        # 临时光标插入，不 setTextCursor：后者会覆盖用户的光标与选区。
        cursor.insertText(text)
        self._scroll_to_bottom(self.chat_view, force=at_bottom)

    def _on_reasoning(self, text: str) -> None:
        text = _sanitize_stream_text(text)
        if not text:
            return
        # 与正文同节奏合并刷新：推理模型每秒可达数百个 chunk，逐块
        # 「取光标 → 移末尾 → insert → 写回 → 滚动」会让 think_view 每秒
        # 做数百次全视口重绘。复用 _stream_timer，与 content 一起 flush。
        self._pending_reasoning.append(text)
        if self._stream_timer is None:
            self._stream_timer = QTimer(self)
            self._stream_timer.setInterval(_STREAM_FLUSH_MS)
            self._stream_timer.timeout.connect(self._flush_stream)
        if not self._stream_timer.isActive():
            self._stream_timer.start()

    def _flush_reasoning(self) -> None:
        buf = getattr(self, "_pending_reasoning", None)
        if not buf:
            return
        text = "".join(buf)
        buf.clear()
        # 先记录「用户是否在底部」（插入会改变 maximum，事后判断失真）。
        sb = self.think_view.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 2
        # Move the widget cursor to the end before inserting. Without this,
        # insertPlainText writes at whatever position the cursor happens to be
        # (e.g. after the user clicked the view to read earlier output), which
        # interleaves new text with existing lines -> overlap.
        # 用**临时**光标插入：setTextCursor 会覆盖用户的光标与选区。
        cursor = self.think_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self._scroll_to_bottom(self.think_view, force=at_bottom)

    def _on_tool_start(self, name: str, args: Any) -> None:
        import json as _json
        # 工具行必须排在流式正文之后 —— 落盘缓冲，避免交错的换行顺序。
        # 注意**非** final：标签剥离器的状态跨工具边界保留（<summary> 偶发
        # 跨 chunk 到达时，final flush 会把 `<summ`/`ary>` 字面碎片当正文
        # 吐进聊天区；回合开始/结束才复位 stripper，见 _on_send/_on_finished）。
        self._flush_stream()
        self._streaming_assistant = False  # close any streaming assistant block
        try:
            pretty = _json.dumps(args, ensure_ascii=False)
        except Exception:
            pretty = str(args)
        # 截断入参：file_write / 大 grep 一次能把 MB 级 JSON 永久写进
        # QTextDocument（无上限），聊天区和日志区一起膨胀。
        shown = pretty if len(pretty) < 400 else pretty[:400] + " …"
        self._append_chat(f'<p class="t">🔧 <b>{self._escape(name)}</b>({self._escape(shown)})…</p>')
        self._append_log(f"[tool-start] {name} {shown}")

    def _on_tool_end(self, name: str, result: Any) -> None:
        out = result.to_message() if hasattr(result, "to_message") else str(result)
        snippet = out if len(out) < 400 else out[:400] + " …"
        cls = "t" if getattr(result, "success", True) else "sys"
        self._append_chat(f'<p class="{cls}">↳ <b>{self._escape(name)}</b> → {self._escape(snippet)}</p>')
        self._append_log(f"[tool-end] {name} success={getattr(result,'success',True)} :: {snippet}")

    def _on_log(self, line: str) -> None:
        self._append_log(line)

    def _on_usage(self, usage: dict) -> None:
        # Don't overwrite _total_tokens here — agent.run() already maintains
        # a running total that includes estimated reasoning tokens when the API
        # underreports. _on_speed (called immediately after) carries the
        # authoritative total. Overwriting here would cause the displayed total
        # to jump between API total (no estimate) and agent total (with estimate).
        pass

    def _on_speed(self, total_tokens: int, speed: float) -> None:
        self._total_tokens = total_tokens
        self.token_label.setText(f"总计 {total_tokens} tokens")
        self.speed_label.setText(f"{speed:.1f} tok/s")

    def _on_skill_suggested(self, skill) -> None:
        """Cache the skill suggestion for showing AFTER the run finishes.

        Showing a modal dialog here would block the UI event loop during
        streaming, preventing content/reasoning chunks from rendering.
        """
        if not isinstance(skill, Skill):
            return
        if skill.name == "_suggested":
            self._pending_skill = skill

    def _on_plan_updated(self, plan) -> None:
        """把计划快照渲染到右栏「执行计划」面板。

        只做纯文本渲染，不弹窗、不打断流式输出 —— 计划的价值在于「随时可
        见」，而不是在每个步骤变更时抢用户的注意力。

        必须用**信号携带的** plan 快照渲染（``_plan_html`` 纯函数）：旧实现
        转手读 ``self._agent.session.render_plan()`` —— 工作线程的
        update_plan 工具会原地改 step dict，跨线程读共享可变状态可能渲染出
        半新半旧的中间态；且 agent 为 None 时会把面板错误清空。
        """
        try:
            html = _plan_html(plan if isinstance(plan, dict) else None)
        except Exception:
            html = ""
        if not html:
            self.plan_view.clear()
            self.plan_progress_label.setText("")
            return
        self.plan_view.setHtml(html)
        self.plan_progress_label.setText(_plan_progress_text(plan))
        self._scroll_to_bottom(self.plan_view)

    def _on_plan_toggle(self, checked: bool) -> None:
        """收起/展开计划面板（默认展开）。"""
        self.plan_view.setVisible(not checked)
        self.plan_toggle_btn.setText("展开" if checked else "收起")

    def _show_pending_skill_dialog(self) -> None:
        """Show the skill-suggestion dialog (called after _on_finished)."""
        skill = self._pending_skill
        self._pending_skill = None
        if skill is None or self._agent is None:
            return
        reply = QMessageBox.question(
            self, "固化为技能",
            f"检测到你多次执行类似任务：\n{skill.prompt[:200]}\n\n是否保存为技能以便下次直接执行？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply == QMessageBox.StandardButton.Yes:
            name, ok = QInputDialog.getText(self, "技能名称", "技能名称：", text="my-skill")
            if ok and name.strip():
                self._agent.skills.create_skill(name.strip(), skill.keywords, skill.prompt)
                # Reload so the worker thread's SkillManager picks up the new
                # skill immediately — otherwise match() in _system_prompt would
                # still see the old (pre-save) in-memory cache and keep
                # re-suggesting the same intent every turn.
                self._agent.skills.reload()
                self._append_log(f"[skill] saved '{name}'")

    def _on_finished(self) -> None:
        if not self._busy:  # guard against double-call from _on_error + bridge.finished
            return
        # 正常收尾 → 撤掉「终止」看门狗，否则它会在 30s 后把已经空闲的 UI
        # 强行判成超时。
        wd = getattr(self, "_stop_watchdog", None)
        if wd is not None:
            wd.stop()
        # 收尾前把还没刷出去的流式文本落盘，否则最后几个 chunk 会丢。
        self._flush_stream(final=True)
        self._busy = False
        self._streaming_assistant = False
        self._set_controls_busy(False)
        self.status_label.setText("就绪")
        # 常驻 worker：回合结束但线程不退出——它阻塞在下一条消息的 queue.get
        # 上，正是这样才保证所有回合都在同一线程执行（MLX KV 缓存按线程隔离，
        # 见 AgentWorker 类注释）。所以这里不 join、不置 None；旧实现 wait(2000)
        # 是因为每回合新建线程、run() 已返回，那时才能立刻 join。
        self._show_pending_skill_dialog()

    def _on_error(self, msg: str) -> None:
        self._append_chat(f'<p class="sys">⚠ {self._escape(msg)}</p>')
        self._append_log(f"[error] {msg}")
        # bridge.finished signal will call _on_finished from the worker's finally block

    # ------------------------------------------------------------------
    # blocking dialogs (run in UI thread, unblock worker)
    # ------------------------------------------------------------------
    def _on_confirm_request(self, message: str) -> None:
        # 兜底保护：确认消息过长会把 Yes/No 按钮顶出屏幕，这里截断到合理长度。
        if len(message) > 800:
            message = message[:800] + "…\n\n（内容过长已截断）"
        reply = QMessageBox.question(self, "确认操作", message,
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        self._callbacks.set_confirm_result(reply == QMessageBox.StandardButton.Yes)

    def _on_ask_request(self, prompt: str) -> None:
        text, ok = QInputDialog.getText(self, "需要你的输入", prompt)
        self._callbacks.set_ask_result(text if ok else None)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _setup_log_bridge(self) -> None:
        """将 RAG/sync 模块的 logger 输出桥接到 UI 的 log_view。

        通过 QSignalLogHandler 把 Python logging 消息安全地转发到
        UI 线程，让用户可以在日志面板中实时看到知识同步的详细进度。

        实现方式：将 handler 注册到 root logger，通过 Filter 只放行
        匹配 _RAG_LOG_PREFIXES 的日志记录。这样做的好处是：
        1. 不依赖 Python logging 层级传播（某些运行时环境下子 logger
           的 propagate 可能被意外关闭）
        2. 后续动态创建的 logger（如 RAG 模块的延迟导入）也会被覆盖
        3. 线程安全：QSignalLogHandler 通过 pyqtSignal 将日志从工作
           线程安全地转发到 UI 线程
        """
        global _LOG_HANDLER
        # 防止重复添加：先移除旧的 handler。
        if _LOG_HANDLER is not None:
            root = logging.getLogger()
            try:
                root.removeHandler(_LOG_HANDLER)
            except Exception:
                pass
            # Remove from _handlerList in-place (see QSignalLogHandler.close
            # for the slice-assignment rationale) then close the old handler.
            try:
                import logging as _logging
                _logging._handlerList[:] = [
                    wr for wr in _logging._handlerList
                    if wr() is not _LOG_HANDLER
                ]
            except Exception:
                pass
            try:
                _LOG_HANDLER.close()
            except Exception:
                pass
        self._log_handler = QSignalLogHandler()
        self._log_handler.message_received.connect(self._append_log)
        self._log_handler.setLevel(logging.DEBUG)

        # 添加 Filter：只放行 RAG 相关模块的日志
        class _RagLogFilter(logging.Filter):
            def filter(self, record):
                return any(
                    record.name.startswith(p) for p in _RAG_LOG_PREFIXES
                )

        self._log_handler.addFilter(_RagLogFilter())

        # 将 handler 挂到 root logger，配合 Filter 实现精确过滤。
        # 同时将相关父 logger 的 level 设为 DEBUG，确保子 logger 的
        # INFO 级别日志能通过有效级别检查。
        root = logging.getLogger()
        root.addHandler(self._log_handler)
        for prefix in _RAG_LOG_PREFIXES:
            logging.getLogger(prefix).setLevel(logging.DEBUG)
        # 同时设置已知子 logger 的 level，防止子 logger 从 root 继承
        # WARNING(30) 导致 INFO(20) 日志在传播到 handler 前被拦截。
        for _name in _RAG_SUB_LOGGERS:
            logging.getLogger(_name).setLevel(logging.DEBUG)

        # 保存到模块级变量，供 closeEvent 清理使用。
        _LOG_HANDLER = self._log_handler
        # 验证日志桥接：这条消息应同时出现在终端和 UI 日志面板中。
        # 如果终端能看到但 UI 面板没有，说明信号槽或 Filter 有问题。
        # 如果 UI 面板能看到，说明桥接完全正常。
        logger.info("Log bridge: RAG/sync logs forwarded to UI panel")
        # 直接追加一条测试消息到 UI 日志面板（绕过 logger 桥接），
        # 验证 log_view 本身可以正常显示文本。
        # 日志桥接已就绪，不输出提示信息到日志区域

    def _append_chat(self, html: str) -> None:
        self.chat_view.append(html)
        self._scroll_to_bottom(self.chat_view)

    def _append_log(self, line: str) -> None:
        self.log_view.appendPlainText(line)
        self._scroll_to_bottom(self.log_view)

    @staticmethod
    def _scroll_to_bottom(widget, force: bool = False) -> None:
        """Pin the widget's viewport to the bottom so the latest content is
        always visible.

        直接把垂直滚动条置到最大值即可吸底 —— **绝不** ``setTextCursor(End)``：
        旧实现「光标移到末尾 + ensureCursorVisible + 滚动条置底」三连里，前两步
        会覆盖用户的光标与选区，用户正在选中文本准备复制时会被反复打断。
        滚动条置底达到同样的视觉效果且零副作用。

        默认只在**已经吸底**时跟随：用户上翻阅读时停止自动跟随，拉回底部
        即恢复（与主流终端行为一致）。``force=True`` 供流式写入方使用：它们
        在插入**前**记录的 at-bottom 状态（插入本身会抬高 maximum，事后判断
        会把在底部的用户误判成上翻）。
        """
        sb = widget.verticalScrollBar()
        if not force and sb.value() < sb.maximum() - 2:
            return  # 用户正在上翻，不打断
        sb.setValue(sb.maximum())

    def _on_clear_chat(self) -> None:
        if self._busy:
            self.status_label.setText("正在运行中…（等待当前回合结束）")
            return
        self.chat_view.clear()
        self.think_view.clear()
        self.plan_view.clear()
        self._streaming_assistant = False
        self._pending_content.clear()
        self._pending_reasoning.clear()
        self._total_tokens = 0
        self.token_label.setText("总计 0 tokens")
        if self._agent is not None:
            self._agent.reset()

    @staticmethod
    def _escape(text: str) -> str:
        import html as _html
        return _html.escape(text).replace("\n", "<br>")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        """Clean up background resources before the window closes.

        All QThread workers must be stopped here. If a worker thread is still
        running when the window's widgets are destroyed, the worker's signals
        fire into deleted C++ objects → ``RuntimeError: wrapped C/C++ object
        has been deleted``. For each worker we:
        1. Disconnect signals (so emit() becomes a no-op even if the thread
           sneaks in one last emit before we wait()).
        2. Request cooperative cancel (if the worker supports it).
        3. wait() with a timeout; if it doesn't finish, terminate() as a last
           resort — unsafe, but safer than the crash.
        """
        # --- 先关配置/日志窗口：它们会把各自面板移回隐藏宿主。否则主窗口销毁时
        # 会连带销毁作为子窗口的对话框，而面板是对话框的子控件 —— 主窗口仍持有
        # 的引用会变成指向已释放 C++ 对象的悬空引用。
        for _name in ("_config_dialog", "_log_window", "_ocr_test_dialog"):
            try:
                _dlg = getattr(self, _name, None)
                if _dlg is not None:
                    _dlg.close()
            except Exception:
                pass
        # --- 停止日志轮询定时器
        try:
            self._log_poll_timer.stop()
        except Exception:
            pass
        # --- AgentWorker: cancel the agent run and join the thread.
        # Without this, closing the window during an agent run leaves the
        # worker thread alive; its bridge.finished/error/content signals
        # will emit into the already-destroyed Bridge → RuntimeError.
        if self._worker is not None:
            try:
                # 断的是 bridge（worker 自身没有信号，disconnect() 是 no-op，
                # 见 AgentWorker.disconnect_bridge 注释）
                self._worker.disconnect_bridge()
            except (TypeError, RuntimeError):
                pass
            # Cooperative cancel: set the _cancelled flag so the agent loop
            # exits after the current tool/chunk. This is faster than
            # terminate() and avoids corrupting tool state mid-execution.
            try:
                self._callbacks.cancel()
            except Exception:
                pass
            # 常驻线程空闲时阻塞在 queue.get：入队哨兵让它立刻退出循环，
            # 否则下面的 wait(3000) 必然超时、被迫走 terminate 强杀路径
            # （跳过 tools.shutdown，还留一个 _worker_terminated 标志）。
            try:
                self._worker.shutdown()
            except Exception:
                pass
            if self._is_worker_running(self._worker):
                self._worker.wait(3000)
                if self._is_worker_running(self._worker):
                    logger.warning("UI: agent worker did not finish in 3s, terminating")
                    self._worker.terminate()
                    self._worker.wait(2000)
                    # terminate() 不释放任何 Python/Qt 锁：被杀的线程可能正
                    # 持有 ToolRegistry._retriever_lock / MCPClient._lock /
                    # RAGEngine._store_lock。之后若再走 tools.shutdown() 会在
                    # `with self._lock:` 处永久阻塞 → 关窗口挂死。
                    # 因此标记一下，下面跳过所有需要取锁的释放动作。
                    self._worker_terminated = True
            self._worker = None
        # --- FetchModelsWorker: a quick HTTP call, usually finishes fast.
        if self._fetch_worker is not None:
            try:
                self._fetch_worker.disconnect()
            except (TypeError, RuntimeError):
                pass
            if self._is_worker_running(self._fetch_worker):
                self._fetch_worker.wait(2000)
                if self._is_worker_running(self._fetch_worker):
                    self._fetch_worker.terminate()
                    self._fetch_worker.wait(1000)
            self._fetch_worker = None
        # --- SyncKnowledgeWorker: cooperative cancel (finishes current file).
        if self._sync_worker is not None:
            try:
                self._sync_worker.disconnect()
            except (TypeError, RuntimeError):
                pass
            if self._is_worker_running(self._sync_worker):
                self._sync_worker.cancel()
                self._sync_worker.wait(3000)
                if self._is_worker_running(self._sync_worker):
                    logger.warning("UI: sync worker did not finish in 3s, terminating")
                    self._sync_worker.terminate()
                    self._sync_worker.wait(2000)
            self._sync_worker = None
        # --- VectorDedupWorker: no cancel mechanism (scan+delete is fast),
        # just wait briefly so its signals don't fire into destroyed widgets.
        if self._dedup_worker is not None:
            try:
                self._dedup_worker.disconnect()
            except (TypeError, RuntimeError):
                pass
            if self._is_worker_running(self._dedup_worker):
                self._dedup_worker.wait(3000)
                if self._is_worker_running(self._dedup_worker):
                    logger.warning("UI: dedup worker did not finish in 3s, terminating")
                    self._dedup_worker.terminate()
                    self._dedup_worker.wait(2000)
            self._dedup_worker = None
        # --- InstallDepsWorker: no cancel mechanism, just wait.
        if self._deps_worker is not None:
            try:
                self._deps_worker.disconnect()
            except (TypeError, RuntimeError):
                pass
            if self._is_worker_running(self._deps_worker):
                self._deps_worker.wait(2000)
                if self._is_worker_running(self._deps_worker):
                    self._deps_worker.terminate()
                    self._deps_worker.wait(1000)
            self._deps_worker = None
        # --- 强杀残留的孤儿 worker：wait 超时保住的引用，退出前再等一次。
        # 多数此时已经跑完（只是当时没等及）；还在跑的 daemon 线程随进程退出
        # 由 OS 回收——但绝不能在这里丢引用（sip 析构运行中的 QThread → qFatal）。
        for orphan in getattr(self, "_orphan_workers", []):
            try:
                orphan.wait(500)
            except RuntimeError:
                pass
        self._orphan_workers = []
        # --- Release agent resources: MCP subprocesses, httpx client, RAG.
        # 线程被 terminate() 强杀时，它可能正持有这些子系统内部的锁，此处
        # 任何取锁操作都会永久阻塞 → 直接跳过，交给进程退出时由 OS 回收
        # （MCP 子进程是 start_new_session，shell_run 子进程有 atexit 钩子）。
        if self._agent is not None and not getattr(self, "_worker_terminated", False):
            try:
                self._agent.tools.shutdown()
            except Exception as e:
                logger.warning("UI: shutdown error: %s", e)
            # agent.close() 释放 fact-extraction 专用的 LLMClient（独立
            # httpx 连接池）。_rebuild_agent() 会做，但窗口直接关闭时不会走
            # 那条路径，连接池会随进程残留。
            try:
                self._agent.close()
            except Exception as e:
                logger.warning("UI: agent close error: %s", e)
        if self._llm is not None:
            try:
                self._llm.close()
            except Exception as e:
                logger.warning("UI: llm close error: %s", e)
        # 辅助模型（投机解码用）同样要关：旧实现只关 _llm，本地辅助权重
        # （~1.6GB）在关窗口时泄漏——进程退出前一直占着。
        # 先停闲时逐章摘要调度器：它可能在收尾时临时加载/使用辅助引擎，
        # 必须在 aux close 之前停止，避免竞态（stop 只置位，进行中的一章
        # 生成最多再跑几十秒即自行退出，不影响后续 close）。
        sched = getattr(self, "_summary_scheduler", None)
        if sched is not None:
            try:
                sched.stop()
            except Exception as e:
                logger.warning("UI: summary scheduler stop error: %s", e)
        if getattr(self, "_aux_llm", None) is not None:
            try:
                self._aux_llm.close()
            except Exception as e:
                logger.warning("UI: aux llm close error: %s", e)
            self._aux_llm = None

        # --- Remove the QSignalLogHandler from the root logger.
        # Without this, the handler (which holds a reference to the QPlainTextEdit
        # via its signal slot) keeps a C++ object reference alive after the widget
        # is destroyed. Any subsequent log call will emit message_received into
        # freed memory → RuntimeError.
        # 用**本窗口自己的** handler，而不是模块级全局：若创建过第二个窗口
        # （或窗口重建），先关的窗口会误删后建窗口的 handler。
        handler = getattr(self, "_log_handler", None) or _LOG_HANDLER
        if handler is not None:
            root = logging.getLogger()
            try:
                root.removeHandler(handler)
            except Exception:
                pass
            # Remove the handler from logging._handlerList IN-PLACE.
            #
            # logging.shutdown() is registered via atexit and its signature is
            #   def shutdown(handlerList=_handlerList):
            # The default argument binds to the list object at function
            # definition time.  If we replace _handlerList with a new list
            # (_handlerList = [...]), the default argument still points to the
            # OLD list, so shutdown() would still see the deleted handler.
            # Using slice assignment (_handlerList[:] = [...]) mutates the
            # list IN-PLACE, so the default argument sees the updated list.
            #
            # Without this, atexit shutdown() calls getattr(h, 'flushOnClose')
            # on a QSignalLogHandler whose Qt C++ object is already deleted,
            # raising RuntimeError.
            import logging as _logging
            try:
                _logging._handlerList[:] = [
                    wr for wr in _logging._handlerList
                    if wr() is not handler
                ]
            except Exception:
                pass
            try:
                handler.close()
            except Exception:
                pass
            # 模块级全局必须置空，否则 handler 及其 Qt 包装常驻进程。
            if handler is _LOG_HANDLER:
                globals()["_LOG_HANDLER"] = None
            self._log_handler = None
        super().closeEvent(event)


def run() -> None:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover
    run()
