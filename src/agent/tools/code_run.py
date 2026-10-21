"""code_run tool: execute arbitrary Python in a sandboxed namespace.

Executes with the workspace as cwd. File writes outside the workspace are
blocked by guarded open wrappers (builtins.open + 进程级 io.open / os.open
线程局部包装，后者覆盖 pathlib.Path.write_text 等走 io.open 的路径），while
still allowing pip installs and network access (the whole point of code_run).
A timeout prevents infinite loops from hanging the agent.

写入护栏只是**提示性的**，不是沙箱：``os.remove`` / ``shutil`` /
``subprocess`` 等非 open 系 API 不受约束，``__import__`` 也可绕过。真正的
隔离只能靠容器或独立进程。

因此**风险分级**（``tools/risk.py``）才是第一道防线：凡是可能改写文件系统
拓扑的调用（``os.symlink/os.rename/os.link/os.replace/shutil.move`` /
``pathlib.Path`` / ``io.open`` / ``os.open`` …）与工作区之外的路径字面量
同现，就必须弹人工确认。写护栏在这里只负责兜住「忘了弹确认」的漏网情况，
而不是唯一依赖。

Timeout handling: Python threads cannot be forcefully killed, but we use a
threading.Event-based cooperative cancellation to signal the worker thread
to stop as soon as possible. For truly stuck code (infinite C extension loop),
the daemon thread lingers until process exit, but we track leaked threads and
log a warning so the user is aware.
"""
from __future__ import annotations

import io
import logging
import os
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Dict, List, Optional

from ..config import AgentConfig
from .base import Tool, ToolRegistry, ToolResult
from .risk import is_risky_code

logger = logging.getLogger(__name__)

# Default timeout (seconds). Long enough for pip installs and network calls,
# short enough that a stuck script doesn't freeze the agent.
_DEFAULT_TIMEOUT = 30
# timeout 上下限：模型可能传 0 / 负数（会导致 join(timeout=0) 立即误判超时），
# 也可能传个天文数字把 agent 挂死。
_MIN_TIMEOUT = 1.0
_MAX_TIMEOUT = 600.0

# stdout/stderr 每路保留的最大字符数。旧实现用无限 StringIO——一段
# `while True: print(...)` 在超时窗口内能产出数 GB 字符串。
_MAX_STREAM_CHARS = 200_000

# Track leaked daemon threads from timed-out code_run calls.
# We can't kill them, but we can warn the user and track the count.
# Cleaned up when the process exits (daemon threads).
#
# Cap the tracked list: a truly stuck thread (tight C loop / blocking I/O
# without timeout) lives until process exit and holds a strong reference to
# its namespace/StringIO via its closure.  Tracking every such thread forever
# would grow this list without bound across many timeouts.  We only need a
# bounded sample for the warning, not an exhaustive registry.
_leaked_threads: List[threading.Thread] = []
_leaked_threads_lock = threading.Lock()
_MAX_TRACKED_LEAKED = 10

# ---------------------------------------------------------------------------
# io.open / os.open 写护栏（2026-10-21 修复护栏绕过）
#
# 旧护栏只替换 exec 命名空间里的 builtins.open，但 ``io.open`` 与 ``os.open``
# 是**模块属性**——用户代码 ``import io; io.open("/etc/hosts", "w")`` 从
# sys.modules 拿真模块，命名空间替换完全管不到；``pathlib.Path.write_text``
# 内部走的正是 io.open。三条路都能零确认、无护栏地写工作区之外。
#
# 实现约束：模块属性只能进程级包装，因此包装函数通过**线程局部**标志判断
# 当前调用是否来自 code_run worker——只有该线程过护栏，其它线程（并发执行
# 的另一场 code_run、UI 线程的正常文件 IO）原样直通，行为不变。
# ---------------------------------------------------------------------------
_IO_OS_GUARD_TLS = threading.local()
_io_os_guard_installed = False
_io_os_guard_lock = threading.Lock()


def _mode_is_write(mode) -> bool:
    """判断 open() 模式串或 os.open flags 位图是否包含写语义。"""
    if isinstance(mode, int):  # os.open 的 flags
        return bool(mode & (os.O_WRONLY | os.O_RDWR | os.O_CREAT
                            | os.O_APPEND | os.O_TRUNC))
    return any(m in str(mode) for m in ("w", "a", "x", "+"))


def _make_write_guard(ws: str, real_open, reject_int_fd: bool = True):
    """构造「工作区外禁写」护栏（builtins.open / io.open / os.open 共用）。

    ``real_open`` 是被包装的原函数：builtins.open 与 io.open 是同一对象，
    os.open 签名不同（flags 位图），由 ``_mode_is_write`` 统一判定。

    ``reject_int_fd``：builtins.open 的命名空间护栏拒绝 int fd（旧行为，
    防止 ``open(3)`` 绕过路径检查）；io.open/os.open 的线程局部护栏必须
    **放行** int fd——``subprocess.Popen`` 内部就是
    ``io.open(os.pipe()[0], 'rb')`` 包装管道描述符，拒了它 code_run 里
    所有 subprocess 调用直接崩。可用的 int fd 只能来自本进程已创建的
    管道/socket/继承的 0/1/2，不构成工作区外写入通道。
    """

    def _guard(file, mode="r", *a, **kw):
        if isinstance(file, int):
            if reject_int_fd:
                raise PermissionError(
                    "File descriptor access is not allowed in code_run")
            return real_open(file, mode, *a, **kw)
        if not isinstance(file, (str, os.PathLike)):
            raise TypeError(
                f"open() expects a path (str/PathLike), got {type(file).__name__}"
            )
        # 用 realpath 而不是 abspath：workspace 里的 symlink 指向外部文件时
        # （如 os.symlink("/etc/cron.d/x", "pwn") 后 open("pwn","w")），
        # abspath 仍落在 workspace 前缀内，防护被绕过。
        path = os.path.realpath(file)
        if _mode_is_write(mode):
            if not (path == ws or path.startswith(ws + os.sep)):
                raise PermissionError(f"Refusing to write outside workspace: {file}")
        return real_open(file, mode, *a, **kw)

    return _guard


def _install_io_os_write_guard() -> None:
    """进程级安装一次 io.open / os.open 包装（幂等，线程安全）。"""
    global _io_os_guard_installed
    if _io_os_guard_installed:
        return
    with _io_os_guard_lock:
        if _io_os_guard_installed:
            return
        _real_io_open = io.open
        _real_os_open = os.open

        def _guarded_io_open(file, mode="r", *a, **kw):
            guard = getattr(_IO_OS_GUARD_TLS, "guard", None)
            if guard is not None:
                return guard(file, mode, *a, **kw)
            return _real_io_open(file, mode, *a, **kw)

        def _guarded_os_open(path, flags, mode=0o777, *a, **kw):
            guard = getattr(_IO_OS_GUARD_TLS, "guard", None)
            if guard is not None:
                return guard(path, flags, mode, *a, **kw)
            return _real_os_open(path, flags, mode, *a, **kw)

        for fn in (_guarded_io_open, _guarded_os_open):
            fn._code_run_guard = True  # type: ignore[attr-defined]
        io.open = _guarded_io_open          # type: ignore[assignment]
        os.open = _guarded_os_open          # type: ignore[assignment]
        _io_os_guard_installed = True
        # 注意：不包装 os.fdopen——fd 只能来自已护栏的 os.open / socket /
        # 继承的 0/1/2，路径级检查对它无意义；fdopen(1, "w") 是合法网络惯用法。

# ---------------------------------------------------------------------------
# 进程级「真实 stdout/stderr」缓存（延迟恢复用）
# ---------------------------------------------------------------------------

class _BoundedIO(io.StringIO):
    """有界 StringIO：超出上限后丢弃后续写入（只标记一次）。

    模块级定义：``_snapshot_real_streams`` 需要识别「当前全局流是否为
    code_run 遗留缓冲」，run() 内定义的类无法被模块级函数引用。
    """

    def __init__(self, limit: int = _MAX_STREAM_CHARS):
        super().__init__()
        self._limit = limit
        self._truncated = False

    def write(self, s) -> int:
        # 必须显式检查 closed：截断后的快速路径先于 io.StringIO.write 的
        # closed 检查执行，close() 对已截断的缓冲完全不生效——超时/取消
        # 路径靠 close 让泄漏线程的下一次 print 撞 ValueError 自行退出，
        # 绕过即意味着泄漏线程永不退出（往真实 stdout 无限写，解释器退出
        # 时持 stdout 锁 → 进程 abort）。
        if self.closed:
            raise ValueError("I/O operation on closed file")
        # 截断后返回 **len(s)** 而不是 0：sys.stdout.write 的契约是「已写入
        # 字符数」，返回 0 会让 print() 认为写入失败并重试/走异常路径
        # （CPython 的 print 对 0 返回值不重试，但第三方代码与 io 协议
        # 检查会误解为「什么都没写」）。
        if self._truncated:
            return len(s) if s else 0
        if self.tell() + len(s) > self._limit:
            super().write("\n… [输出超限，已截断]")
            self._truncated = True
            return len(s)
        return super().write(s)


# ---------------------------------------------------------------------------
# 进程级「真实 stdout/stderr」缓存（延迟恢复用）
#
# 超时/取消路径必须把 sys.stdout **保持**在已关闭的 _BoundedIO 上一小段
# 时间，泄漏线程的下一次 print 才会撞上 ValueError 自行退出（见 run() 内
# 注释）。恢复被延迟期间，若紧接着又启动一场 code_run，它快照到的
# _prev_stdout 会是上一场已关闭的缓冲——直接恢复回去会把全局 stdout
# 永久钉死在死流上。这里在模块级记录最近一次确认的「非 code_run 缓冲」
# 流（可能是 pytest 捕获层 / 真实终端），所有恢复动作统一恢复到它。
# ---------------------------------------------------------------------------
_LAST_REAL_STDOUT: Optional[Any] = None
_LAST_REAL_STDERR: Optional[Any] = None
_REAL_STREAMS_LOCK = threading.Lock()
# 泄漏线程在缓冲区关闭后撞 ValueError 的宽限窗口（秒）。print 循环每秒
# 百万次量级，50ms 足够让它在窗口内死掉；窗口过长会让主进程其他线程的
# print 暂时打到已关闭的缓冲上（抛 ValueError）。
_LEAK_RESTORE_DELAY = 0.05


def _snapshot_real_streams() -> tuple:
    """返回本场 code_run 应记为「外部原始流」的 (stdout, stderr)。

    当前 stdout 已是上一场遗留的已关闭 code_run 缓冲时，绝不能把它当
    原始流缓存/恢复——否则全局 stdout 永久指向死流。此时沿用在模块级
    缓存的真实流。
    """
    global _LAST_REAL_STDOUT, _LAST_REAL_STDERR
    with _REAL_STREAMS_LOCK:
        cur_out, cur_err = sys.stdout, sys.stderr
        out_is_stale = isinstance(cur_out, _BoundedIO) and cur_out.closed
        err_is_stale = isinstance(cur_err, _BoundedIO) and cur_err.closed
        if not out_is_stale:
            _LAST_REAL_STDOUT = cur_out
        if not err_is_stale:
            _LAST_REAL_STDERR = cur_err
        return (
            _LAST_REAL_STDOUT if out_is_stale else cur_out,
            _LAST_REAL_STDERR if err_is_stale else cur_err,
        )


def _schedule_stream_restore(buffers: tuple, delay: float = _LEAK_RESTORE_DELAY) -> None:
    """延迟恢复全局 stdout/stderr（超时/取消路径专用）。

    恢复前检查全局流是否仍指向**本场**缓冲：后续场次已换上新缓冲时不动
    （它们会恢复到自己快照的真实流，幂等）。
    """
    out_buf, err_buf = buffers

    def _restore() -> None:
        time.sleep(delay)
        with _REAL_STREAMS_LOCK:
            try:
                if sys.stdout is out_buf:
                    sys.stdout = _LAST_REAL_STDOUT or sys.stdout
            except Exception:
                pass
            try:
                if sys.stderr is err_buf:
                    sys.stderr = _LAST_REAL_STDERR or sys.stderr
            except Exception:
                pass

    threading.Thread(target=_restore, daemon=True,
                     name="code-run-restore").start()


class CodeRunTool(Tool):
    name = "code_run"
    description = (
        "Execute Python code (multi-line). stdout/stderr captured. "
        "Can import and pip install packages, call APIs, write files in workspace. "
        "This is the capability core: anything not covered by a dedicated tool can "
        "be done here. For system commands / CLI tools prefer shell_run "
        "(or call it from here via subprocess). "
        f"Killed after {int(_DEFAULT_TIMEOUT)}s unless overridden."
    )
    destructive = True
    parameters = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python source to execute."},
            "timeout": {"type": "number", "description": f"Timeout in seconds (default {int(_DEFAULT_TIMEOUT)})."},
        },
        "required": ["code"],
    }

    config: AgentConfig
    registry: ToolRegistry

    def bind(self, config: AgentConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry

    def should_confirm(self, args: Dict[str, Any]) -> bool:
        """风险分级确认（见 tools/risk.py 模块注释）。

        code_run 是能力核心、几乎每轮都用，destructive=True 的一律确认把
        工作流切成碎片。改为静态风险分级：sudo / 管道下载执行 /
        git push --force / 破坏性调用（rmtree/remove/chmod/subprocess 等）
        作用到工作区之外路径才弹确认；其余直接执行（写护栏仍会拦截工作区
        外的 open 写入，超时与取消机制不变）。
        """
        if getattr(self.config, "always_confirm_code", False):
            return True
        return is_risky_code(str((args or {}).get("code", "")),
                             self.config.workspace)

    def _cancel_checker(self):
        """协作式取消探针（UI「终止对话」）。拿不到 callbacks 时恒 False。"""
        cb = getattr(self.registry, "callbacks", None)
        fn = getattr(cb, "is_cancelled", None)
        return fn if callable(fn) else (lambda: False)

    def run(self, code: str, timeout: Optional[float] = None) -> ToolResult:
        import builtins as _b

        # Tolerate non-string `code` (some models emit a number or a list).
        # exec() requires str/bytes; coercing here avoids a confusing TypeError
        # that the model would have to debug blind.
        if not isinstance(code, str):
            code = str(code)

        if timeout is None:
            timeout = _DEFAULT_TIMEOUT
        try:
            timeout = float(timeout)
        except (TypeError, ValueError):
            timeout = float(_DEFAULT_TIMEOUT)
        timeout = max(_MIN_TIMEOUT, min(timeout, _MAX_TIMEOUT))

        ws = os.path.realpath(self.config.workspace)
        _real_open = _b.open
        # 进程级 io.open / os.open 包装（只装一次）；本线程的护栏经 TLS 生效，
        # 见 _install_io_os_write_guard 注释。int fd 放行（subprocess 管道）。
        _install_io_os_write_guard()
        # 命名空间 builtins.open 护栏：拒绝 int fd（旧行为，防 open(3) 绕过）
        _guard_open = _make_write_guard(ws, _real_open, reject_int_fd=True)
        # 线程局部护栏（io.open / os.open 用）：int fd 放行
        _write_guard = _make_write_guard(ws, _real_open, reject_int_fd=False)

        namespace: dict = {"__name__": "__code_run__", "__file__": "<code_run>"}

        out = _BoundedIO()
        err = _BoundedIO()

        # 主线程侧的进程级状态快照（cwd / stdout / stderr），join 后恢复。
        _prev_cwd = os.getcwd()
        # stdout/stderr 不能裸快照：上一场超时遗留的恢复窗口内（约 50ms）
        # 全局流可能是**已关闭**的 _BoundedIO，恢复回去会把全局 stdout 永久
        # 钉死在死流上（泄漏线程永远撞不到 ValueError，解释器退出时持锁
        # abort）。走 _snapshot_real_streams 拿到可安全恢复的真实流。
        _prev_stdout, _prev_stderr = _snapshot_real_streams()

        # Cooperative cancellation event: set by the main thread on timeout,
        # checked by user code via an injected `_cancelled()` function so
        # well-behaved scripts can exit early. For uncooperative code (tight
        # C loop, blocking I/O without timeout), the thread will linger.
        _cancel_evt = threading.Event()

        # Result holder for the worker thread.
        result: dict = {"exc": None, "done": False}

        def _worker():
            # 线程局部护栏：只对本 worker 线程生效——io.open / os.open 的
            # 进程级包装据此决定是否拦截（见 _install_io_os_write_guard）。
            # 必须在 worker 线程内设置：TLS 是线程绑定的，主线程设了不算。
            _IO_OS_GUARD_TLS.guard = _write_guard
            try:
                # os.chdir / sys.stdout 替换是**进程级**全局状态，不是线程级的：
                # 执行期间其它线程的相对路径与 print 都会受影响。这里接受这段
                # 时间的短暂影响（相对路径语义是工具契约的一部分）。
                #
                # 恢复动作**全部**由主线程在 join 之后完成（见下方注释），
                # worker 自己绝不恢复——尤其不能碰 sys.stdout：超时泄漏的线程
                # 会在未来任意时刻退出，若它在 finally 里把 stdout 恢复成
                # 「自己启动时的值」，就可能覆盖掉下一次 code_run 刚换上的
                # StringIO，把那次执行的输出整个吞掉。
                os.chdir(ws)
                # Inject cancellation checker into the sandbox namespace so
                # user code can call `_cancelled()` to poll and exit early.
                namespace["__builtins__"] = {**vars(_b), "open": _guard_open}
                namespace["_cancelled"] = _cancel_evt.is_set
                sys.stdout, sys.stderr = out, err
                try:
                    exec(compile(code, "<code_run>", "exec"), namespace)
                finally:
                    result["done"] = True
            except PermissionError as e:
                result["exc"] = e
            except SystemExit as e:
                result["exc"] = SystemExit(f"SystemExit: {e.code}")
            except BaseException as e:
                result["exc"] = e
            finally:
                result["done"] = True

        # Run exec in a daemon thread so a timeout can return control to the
        # agent. The thread cannot be forcefully killed in Python, but as a
        # daemon it won't block process exit.
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        # 轮询式 join：旧实现一次性 join(timeout) 在用户点「终止对话」后仍要
        # 跑满整个 timeout 才返回（harness 虽立刻合成失败结果，但这个线程
        # 继续占 CPU、且 sys.stdout 仍被替换成 StringIO 直到 join 返回）。
        # 0.2s 粒度轮询取消标志，命中立即收尾。
        deadline = time.monotonic() + timeout
        cancelled = False
        while t.is_alive():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            t.join(timeout=min(0.2, remaining))
            if not t.is_alive():
                break
            if self._cancel_checker()():
                cancelled = True
                break

        # cwd 由主线程统一立即恢复（恢复权只属于主线程，见 _worker 注释）。
        # stdout/stderr 的恢复分两种时序：
        #   - 正常完成（worker 已 join）：立即恢复；
        #   - 超时/取消（泄漏线程仍在跑）：必须**延迟**恢复——旧实现先恢复
        #     全局流再 close 缓冲，而泄漏线程的 print() 每次都**动态**查找
        #     sys.stdout，拿到的是已恢复的真实 stdout，永远不会碰已关闭的
        #     缓冲 → 线程永不退出，解释器退出时它正持有 stdout 缓冲锁 →
        #     Fatal Python error: _enter_buffered_busy（进程 abort）。
        #     先 close 再延迟恢复，泄漏线程下一次 print 即撞 ValueError
        #     自行退出；50ms 窗口后全局流恢复为真实流。
        try:
            os.chdir(_prev_cwd)
        except OSError as e:
            logger.warning("code_run: failed to restore process cwd: %s", e)

        def _partial_output() -> str:
            partial = ""
            try:
                partial = (out.getvalue() or "") + (
                    ("\n[stderr]\n" + err.getvalue()) if err.getvalue() else "")
            except Exception:
                partial = ""
            return partial

        def _close_buffers() -> None:
            # getvalue 在 close 之后会抛 ValueError，必须先读再关。
            for buf in (out, err):
                try:
                    buf.close()
                except Exception:
                    pass

        if cancelled:
            partial = _partial_output()
            _cancel_evt.set()
            _close_buffers()
            # 泄漏线程仍在跑：延迟恢复全局流，让它撞 ValueError 退出。
            _schedule_stream_restore((out, err))
            msg = "用户取消了对话，代码执行被中断"
            if partial.strip():
                msg += f"\n[取消前已产生的输出]\n{partial[-3000:]}"
            return ToolResult(False, error=msg)

        if not result["done"]:
            # Thread is still running (infinite loop or slow code).
            # Signal cooperative cancellation so the worker can exit when
            # it next checks `_cancelled()`. Track the leaked thread.
            _cancel_evt.set()
            #
            # 缓冲区必须 close（而不是留着让泄漏线程继续写）：
            # 全局 stdout/stderr 此刻仍指向这两个缓冲，close 后泄漏线程的
            # 下一次写入抛 ValueError: I/O operation on closed file，线程
            # 自行退出（既止住输出堆积，也顺带回收线程与其命名空间）。
            # 全局流的恢复由 _schedule_stream_restore 延迟完成。
            partial = _partial_output()
            _close_buffers()
            _schedule_stream_restore((out, err))
            with _leaked_threads_lock:
                # Prune completed threads, then append the new one, capped.
                _leaked_threads[:] = [lt for lt in _leaked_threads if lt.is_alive()]
                _leaked_threads.append(t)
                if len(_leaked_threads) > _MAX_TRACKED_LEAKED:
                    _leaked_threads[:] = _leaked_threads[-_MAX_TRACKED_LEAKED:]
                leaked_count = len(_leaked_threads)
            if leaked_count > 1:
                logger.warning(
                    "code_run: daemon thread leaked due to timeout (total tracked: %d). "
                    "Stuck threads are cleaned up on process exit.",
                    leaked_count,
                )
            msg = f"Code execution timed out after {timeout:.0f}s"
            if partial.strip():
                tail = partial[-3000:]
                msg += f"\n[超时前已产生的输出]\n{tail}"
            return ToolResult(False, error=msg)

        # Worker completed within timeout — check if the thread we just
        # joined is still in the leaked list and remove it.
        with _leaked_threads_lock:
            if t in _leaked_threads:
                _leaked_threads.remove(t)

        # 正常完成：worker 已 join，泄漏不可能发生，立即恢复全局流。
        try:
            sys.stdout, sys.stderr = _prev_stdout, _prev_stderr
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("code_run: failed to restore process stdout: %s", e)

        exc = result["exc"]
        if exc is not None:
            # 统一带 traceback：PermissionError（写护栏拦截）/SystemExit 旧实现
            # 只给裸字符串，模型看不到出错代码行，与其他异常类型不一致、难调试。
            tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
            detail = "".join(tb)
            if isinstance(exc, (PermissionError, SystemExit)):
                # 护栏/退出类异常消息本身就是结论，放最前面再附栈。
                return ToolResult(False, error=f"{exc}\n{detail}")
            return ToolResult(False, error=detail)

        output = out.getvalue()
        if err.getvalue():
            output += ("\n" if output else "") + "[stderr]\n" + err.getvalue()
        return ToolResult(True, output=output or "(no output)")

