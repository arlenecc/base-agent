"""shell_run tool: execute shell/terminal commands in the workspace.

Runs the command with the workspace as cwd, captures stdout+stderr, and
enforces a timeout so a hung command cannot block the agent indefinitely.

确认策略（tools/risk.py）：默认直接执行；仅 sudo/系统电源控制/进程杀伤/
磁盘操作/管道下载执行/git push --force/写工作区之外路径才弹人工确认
（`config.always_confirm_code=True` 可恢复逐次确认的旧行为）。
"""
from __future__ import annotations

import atexit
import collections
import logging
import os
import signal
import subprocess
import threading
import time
from typing import Any, Dict

from ..config import AgentConfig
from .base import Tool, ToolRegistry, ToolResult
from .risk import is_risky_command

logger = logging.getLogger(__name__)

# Default timeout (seconds) for shell commands. Can be overridden per-call via
# the `timeout` argument. Long enough for pip installs, short enough that a
# hung command doesn't freeze the agent.
_DEFAULT_TIMEOUT = 120
# 上下限钳制：模型可能传 99999（agent 线程会阻塞 27 小时且无法被「终止对话」
# 打断，因为协作式取消只在工具边界轮询），也可能传 0/负数（立刻判定超时并
# kill 掉刚启动的命令）。与 code_run 的 _MIN/_MAX_TIMEOUT 同策略。
_MIN_TIMEOUT = 1.0
_MAX_TIMEOUT = 600.0

# 单条命令 stdout/stderr 各保留的最大字符数。旧实现用 capture_output 全量
# 读入内存——一条 `cat 大日志` / `find /` 就能把进程内存打满。
_MAX_OUTPUT_CHARS = 200_000

# 超时后给进程组留出优雅退出的时间，之后再 SIGKILL。
_TERM_GRACE_SECONDS = 3.0


class ShellRunTool(Tool):
    name = "shell_run"
    description = (
        "Run a shell command in the workspace, return stdout+stderr. "
        "For CLI tools, git, pip, build scripts, etc. "
        "This is code_run's system-side extension: use it whenever a task needs "
        "OS-level commands rather than Python logic."
    )
    destructive = True
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute (may include pipes, redirects, etc.).",
            },
            "timeout": {
                "type": "number",
                "description": f"Timeout in seconds (default {_DEFAULT_TIMEOUT}).",
            },
        },
        "required": ["command"],
    }

    config: AgentConfig
    registry: ToolRegistry

    def bind(self, config: AgentConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry

    def should_confirm(self, args: Dict[str, Any]) -> bool:
        """风险分级确认（见 tools/risk.py 模块注释）。

        旧实现 destructive=True 一律弹确认——shell_run 是 code_run 对系统
        命令的延伸，高频调用把工作流切成碎片。改为静态风险分级：读系统
        文件、工作区内写等直接执行；sudo/系统电源控制/进程杀伤/磁盘操作/
        管道下载执行/git push --force/写工作区之外路径才弹确认。
        `config.always_confirm_code=True` 可恢复旧的逐次确认行为。
        """
        if getattr(self.config, "always_confirm_code", False):
            return True
        return is_risky_command(str((args or {}).get("command", "")),
                                self.config.workspace)

    def _cancel_checker(self):
        """协作式取消探针（UI「终止对话」）。拿不到 callbacks 时恒 False。"""
        cb = getattr(self.registry, "callbacks", None)
        fn = getattr(cb, "is_cancelled", None)
        return fn if callable(fn) else (lambda: False)

    def run(self, command: str, timeout: float = None) -> ToolResult:
        # Tolerate non-string command (model might emit a list or number).
        if not isinstance(command, str):
            command = str(command)

        if timeout is None:
            timeout = _DEFAULT_TIMEOUT
        try:
            timeout = float(timeout)
        except (TypeError, ValueError):
            timeout = float(_DEFAULT_TIMEOUT)
        # 钳制到 [_MIN_TIMEOUT, _MAX_TIMEOUT]：过大阻塞 agent 线程无法取消，
        # 过小（0/负数）会立刻 kill 掉刚启动的命令。
        timeout = max(_MIN_TIMEOUT, min(timeout, _MAX_TIMEOUT))

        ws = os.path.abspath(self.config.workspace)
        try:
            returncode, out, err, timed_out, cancelled = _run_capture(
                command, cwd=ws, timeout=timeout,
                should_cancel=self._cancel_checker(),
            )
        except Exception as e:
            return ToolResult(False, error=f"{type(e).__name__}: {e}")

        if cancelled:
            return ToolResult(
                False,
                error="用户取消了对话，命令被中断",
                output=_combine(out, err),
            )

        if timed_out:
            return ToolResult(
                False,
                error=f"Command timed out after {timeout:.0f}s: {command[:200]}",
                output=_combine(out, err),
            )

        # Combine stdout and stderr so the model sees everything.
        output = _combine(out, err)

        if returncode == 0:
            return ToolResult(True, output=output or "(no output)")
        # Non-zero exit: report as failure but still include output so the
        # model can diagnose what went wrong.
        # returncode 为 None = 连 SIGKILL 都没能回收（极端情况），别给模型
        # 一个无法解读的 "Exit code None"。
        detail = f"Exit code {returncode}" if returncode is not None else \
            "进程未能被回收（可能仍在运行或被信号终止）"
        return ToolResult(
            False,
            output=output,
            error=detail,
        )


def _combine(out: str, err: str) -> str:
    if err:
        return out + ("\n" if out else "") + "[stderr]\n" + err
    return out


class _Collector:
    """Bounded stdout/stderr collector that keeps the head *and* the tail.

    超长输出里最有价值的信息往往在末尾（报错摘要、最后几行结果）。
    只保留开头会让模型看不到命令为什么失败。这里用 deque 保留尾部、
    单独字符串保留头部，中间用省略标记衔接。
    """

    _TAIL_RATIO = 0.25  # 尾部至少保留 25% 的预算

    def __init__(self, limit: int = _MAX_OUTPUT_CHARS):
        self.limit = limit
        self._head_budget = limit - int(limit * self._TAIL_RATIO)
        self._head = []
        self._head_size = 0
        self._tail = collections.deque()
        self._tail_size = 0
        self.truncated = False

    def _drop_tail_overflow(self) -> None:
        while self._tail_size > int(self.limit * self._TAIL_RATIO) and self._tail:
            dropped = self._tail.popleft()
            self._tail_size -= len(dropped)

    def feed(self, text: str) -> None:
        if not text:
            return
        if not self.truncated:
            if self._head_size + len(text) <= self._head_budget:
                self._head.append(text)
                self._head_size += len(text)
                return
            # 头部预算用尽 → 进入尾部模式
            keep = self._head_budget - self._head_size
            if keep > 0:
                self._head.append(text[:keep])
                rest = text[keep:]
            else:
                rest = text
            self.truncated = True
            self._feed_tail(rest)
            return
        self._feed_tail(text)

    def _feed_tail(self, text: str) -> None:
        if not text:
            return
        self._tail.append(text)
        self._tail_size += len(text)
        self._drop_tail_overflow()

    def value(self) -> str:
        text = "".join(self._head)
        if self.truncated:
            text += f"\n… [输出超过 {self.limit} 字符，中间已截断] …"
        text += "".join(self._tail)
        return text


def _reader(stream, collector: _Collector) -> None:
    """Read a pipe in a thread, appending into a bounded collector."""
    try:
        while True:
            chunk = stream.read(8192)
            if not chunk:
                break
            collector.feed(chunk)
    except Exception:  # pragma: no cover - 管道可能在关闭时被中断
        pass
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _kill_process_group(proc: subprocess.Popen) -> None:
    """SIGTERM then SIGKILL the whole process group.

    ``shell=True`` spawns ``sh -c …``; killing only that PID leaves any
    grandchild (``tail -f``, a dev server, a pip subprocess) running forever
    as an orphan.  We start the child in its own session and signal the entire
    group so nothing survives a timeout.

    只在**组内还有我们自己的进程**时发信号：组主退出后 pgid 会被 OS 回收
    复用，无差别 killpg 有可能误杀无关进程组（旧实现无条件发 SIGKILL）。
    组主活着 → 组一定还存在且属于我们，此时 SIGTERM/SIGKILL 都是安全的。
    """
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError, OSError):
        pgid = None

    def _signal(sig: int) -> None:
        try:
            if pgid is not None:
                os.killpg(pgid, sig)
            else:
                proc.send_signal(sig)
        except Exception:
            pass

    if proc.poll() is not None:
        # 组主已退出。此时 pgid 可能已被 OS 回收复用——无差别 killpg 有
        # 误杀无关进程组的风险（旧实现无条件发 SIGKILL）。先用信号 0 探测
        # 组是否仍存在：仍存在说明还有成员活着（几乎必然是忽略 SIGTERM 的
        # 孙进程，且组未被复用），可以安全收割；已不存在则无事可做。
        if pgid is not None:
            try:
                os.killpg(pgid, 0)
            except (ProcessLookupError, PermissionError, OSError):
                return          # 组已空 / 不可访问 —— 绝不盲发信号
            _signal(signal.SIGKILL)
        return
    _signal(signal.SIGTERM)
    deadline = time.monotonic() + _TERM_GRACE_SECONDS
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        time.sleep(0.05)
    # 组主仍在（SIGTERM 没说服它）→ SIGKILL 整组。组主要是在宽限期内退了，
    # 走上面的探测逻辑，不会发出可能误杀复用 pgid 的信号。
    if proc.poll() is None:
        _signal(signal.SIGKILL)
    try:
        proc.wait(timeout=1.0)
    except Exception:
        pass


# 仍在运行（或已退出但未被回收）的子进程。start_new_session=True 让子进程
# 脱离主进程的进程组 —— 终端信号收不到、ToolRegistry.shutdown() 也不感知，
# 用户在命令执行期间关窗口/杀进程就会留下最多跑 _MAX_TIMEOUT 秒的孤儿。
# 这里登记一份，进程退出时统一回收。
_LIVE_PROCS: "set[subprocess.Popen]" = set()
_LIVE_PROCS_LOCK = threading.Lock()


def _register_proc(proc: subprocess.Popen) -> None:
    with _LIVE_PROCS_LOCK:
        _LIVE_PROCS.add(proc)


def _unregister_proc(proc: subprocess.Popen) -> None:
    with _LIVE_PROCS_LOCK:
        _LIVE_PROCS.discard(proc)


def kill_all_running_procs() -> int:
    """回收所有仍在跑的 shell_run 子进程（进程退出 / shutdown 时调用）。

    收敛多轮 + 逐个 discard：旧实现 ``list()`` 快照后立刻 ``clear()``，
    快照与 clear 之间新登记的进程会被静默丢弃（永不回收 → 孤儿）。
    现在每轮只快照不清集合，杀完一个 discard 一个；最多 3 轮收敛，
    处理「杀的过程中又有新进程登记」的极端竞态（正常 teardown 不会发生）。
    """
    killed = 0
    for _ in range(3):
        with _LIVE_PROCS_LOCK:
            procs = list(_LIVE_PROCS)
        if not procs:
            break
        for proc in procs:
            try:
                if proc.poll() is None:
                    _kill_process_group(proc)
                    killed += 1
            except Exception:
                pass
            finally:
                _unregister_proc(proc)
    return killed


atexit.register(kill_all_running_procs)


def _run_capture(command: str, cwd: str, timeout: float,
                 should_cancel=None):
    """Run ``command`` in its own process group, capturing bounded output.

    Returns ``(returncode, stdout, stderr, timed_out, cancelled)``.  On timeout
    the entire process group is killed (see :func:`_kill_process_group`) and
    whatever output was produced before the timeout is still returned so the
    model can see where it got stuck.  ``should_cancel``（UI「终止对话」）命中
    时同样杀组返回——旧实现用户取消后子进程要跑到自身超时（最长 600s）。
    """
    if should_cancel is None:
        should_cancel = lambda: False  # noqa: E731
    # start_new_session=True puts the shell (and everything it spawns) into a
    # new process group/session so we can reap the whole tree on timeout.
    proc = subprocess.Popen(
        command,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    _register_proc(proc)

    out = _Collector()
    err = _Collector()
    t_out = threading.Thread(target=_reader, args=(proc.stdout, out), daemon=True)
    t_err = threading.Thread(target=_reader, args=(proc.stderr, err), daemon=True)
    t_out.start()
    t_err.start()

    timed_out = False
    cancelled = False
    try:
        # 轮询式 wait：一次性 proc.wait(timeout) 在取消/超时判定上粒度太粗，
        # 且取消后子进程会一直跑到自身超时。0.2s 粒度检查取消标志。
        deadline = time.monotonic() + timeout
        while True:
            try:
                proc.wait(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                pass
            if should_cancel():
                cancelled = True
                break
            if time.monotonic() >= deadline:
                timed_out = True
                break

        if cancelled or timed_out:
            _kill_process_group(proc)
            try:
                proc.wait(timeout=_TERM_GRACE_SECONDS)
            except subprocess.TimeoutExpired:  # pragma: no cover - 极端情况
                # 不强退成功就不再 wait，进程会变成僵尸直到 GC，并打出
                # ResourceWarning: subprocess is still running。这里补一次
                # 回收，至少把僵尸收掉（returncode 仍是 None 时由调用方兜底文案）。
                try:
                    proc.kill()
                    proc.wait(timeout=1.0)
                except Exception:
                    pass

        # Give the reader threads a moment to drain whatever is left in the pipes.
        out_done = t_out.join(timeout=2.0)
        err_done = t_err.join(timeout=2.0)
        if not out_done or not err_done:
            # reader 还没读完（孙进程仍持有管道写端）：命令尾部输出会进不了
            # collector。旧实现静默丢弃——模型以为看到的是完整输出。明确标注。
            note = "\n… [子进程仍在写入，输出可能不完整]"
            if not out_done:
                out.feed(note)
            if not err_done:
                err.feed(note)
        # 只有 reader 真的结束了才关管道：孙进程仍持有管道写端时，reader 阻塞
        # 在 read() 里并占着 TextIOWrapper 的内部锁，此时 close()→flush() 会
        # **永久挂起**工具线程（旧实现无条件 close）。reader 活着就交给它自己
        # 在 EOF 时关闭——最坏是一个 daemon 线程 + 一个 fd，远好过卡死 agent。
        for stream, t in ((proc.stdout, t_out), (proc.stderr, t_err)):
            try:
                if stream and not stream.closed and not t.is_alive():
                    stream.close()
            except Exception:
                pass
    finally:
        _unregister_proc(proc)

    return proc.returncode, out.value(), err.value(), timed_out, cancelled
