"""MCP (Model Context Protocol) JSON-RPC client over stdio.

Manages a subprocess that runs an MCP server, communicates via JSON-RPC 2.0
over stdin/stdout, and exposes tools/list and tools/call.
"""
from __future__ import annotations

import json
import logging
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# tools/call 的下限超时（秒）。握手/枚举用实例的 timeout（默认 30s）足够，
# 但工具调用本身可能是构建、批量处理、爬取等长任务——共用 30s 会让长任务
# 必然超时，且服务端还在跑，结果被当迟到响应丢弃。
_TOOL_CALL_TIMEOUT = 300.0

# 连接断开后自动重连的冷却（秒）。瞬时故障（server 崩溃、写失败、reader
# 退出）不应让该 server 的工具在本次 Agent 生命周期内全部不可用；但持续
# 崩溃的 server 也不能每个工具调用都拉起一次进程。冷却期内 ensure_connected
# 直接返回 False，调用方以 "not running" 失败（错误信息里带上了原因）。
_RESTART_COOLDOWN_SECS = 30.0

# 单次 MCP 工具返回保留的最大字符数。MCP 工具是外部代码，输出体积不受我们
# 控制（例如让它 dump 一个大文件/数据库表），而这段文本会原样进入模型上下文
# ——不设上限一次调用就能把窗口撑爆。与 file_read / shell_run 的 200K 策略
# 保持一致，保留头部并明确标注截断。
_MAX_TOOL_OUTPUT_CHARS = 200_000

# 写入队列容量与入队超时。MCP server 若停止读取 stdin，管道缓冲区（64KB）
# 写满后阻塞写会**无限期挂起** agent 线程（旧实现持 _write_lock 原地 write，
# 卡死后所有后续 _call 全部排队等死）。改用专用写线程 + 有界队列后，最坏
# 情况是入队超时后抛 MCPError，agent 线程永远不再被管道堵死。
_WRITE_QUEUE_MAX = 64
_WRITE_ENQUEUE_TIMEOUT = 10.0

# ------------------------------------------------------------------
# MCP JSON-RPC wire types
# ------------------------------------------------------------------


@dataclass
class JSONRPCRequest:
    jsonrpc: str = "2.0"
    id: int = 0
    method: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JSONRPCResponse:
    jsonrpc: str = "2.0"
    id: int = 0
    result: Any = None
    error: Optional[Dict[str, Any]] = None


# ------------------------------------------------------------------
# MCP Client
# ------------------------------------------------------------------


class MCPClient:
    """Manages one MCP server subprocess and JSON-RPC communication."""

    def __init__(
        self,
        name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
    ):
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env
        self.timeout = timeout
        self._process: Optional[subprocess.Popen] = None
        self._request_id = 0
        self._lock = threading.Lock()
        # stdin 写入走专用线程 + 有界队列（见 _WRITE_QUEUE_MAX 注释）：持锁原地
        # write 会被不读 stdin 的 server 堵死。单消费者线程天然保证消息不会
        # 交错，因此不再需要写锁。
        self._write_queue: "queue.Queue[Optional[str]]" = queue.Queue(
            maxsize=_WRITE_QUEUE_MAX
        )
        self._writer_thread: Optional[threading.Thread] = None
        self._pending: Dict[int, threading.Event] = {}
        self._results: Dict[int, JSONRPCResponse] = {}
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._running = False
        # 主动停止标志（stop()）：与「连接自己断」区分开——后者下次使用前
        # 自动重连（见 ensure_connected），前者是用户/registry 的拆除意图，
        # 永不重启。
        self._stopped = False
        # 自动重连冷却截止（monotonic）：持续崩溃的 server 不会被反复拉起。
        self._restart_after = 0.0
        self._server_info: Dict[str, Any] = {}
        self._tools: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the MCP server subprocess and perform the initialize handshake."""
        if self._running:
            return
        # 显式 start = 用户意图：清掉「已主动停止」，允许重新拉起。
        self._stopped = False

        # 上一次连接的残留状态必须先清掉：
        # * _fail_pending / stop() 可能已在 _write_queue 里留下 None 哨兵，
        #   新写线程会立刻消费它并退出 → 之后所有 _call() 干等 30~300s 超时；
        # * _pending/_results 里可能躺着上一次连接的迟到条目，污染新会话。
        # 不清的话「瞬时写失败 → _fail_pending 置 _running=False → 重启」
        # 这条路永远修不好一个只是打了磕绊的连接。
        while True:
            try:
                self._write_queue.get_nowait()
            except queue.Empty:
                break
        with self._lock:
            self._pending.clear()
            self._results.clear()

        env = None
        if self.env:
            import os
            env = os.environ.copy()
            env.update(self.env)

        try:
            self._process = subprocess.Popen(
                [self.command] + self.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                bufsize=1,
                # 独立进程组：很多 MCP server 由 `npx` 启动，npx 会再 fork 出
                # 真正的 node server。不建新会话的话 terminate() 只杀掉 npx，
                # 真正的 server 会变成孤儿进程一直存活到关机。有了独立进程组，
                # stop() 可以 killpg 一次性回收整棵树。
                start_new_session=True,
            )
        except FileNotFoundError:
            raise MCPError(f"MCP server '{self.name}': command not found: {self.command}")
        except Exception as e:
            raise MCPError(f"MCP server '{self.name}': failed to start: {e}")

        self._running = True
        # Drain stderr in a background thread so the pipe buffer never fills up
        # and blocks the subprocess.
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True
        )
        self._stderr_thread.start()
        self._writer_thread = threading.Thread(
            target=self._write_loop, daemon=True, name=f"mcp-write-{self.name}"
        )
        self._writer_thread.start()
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

        # Initialize handshake
        try:
            init_resp = self._call("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "base-agent", "version": "1.0"},
            })
            self._server_info = init_resp.result or {}
            # Send initialized notification
            self._send_notification("notifications/initialized", {})
            # Fetch tools
            tools_resp = self._call("tools/list", {})
            self._tools = tools_resp.result.get("tools", []) if tools_resp.result else []
            logger.info(
                "MCP server '%s' initialized: %d tools available",
                self.name, len(self._tools),
            )
        except Exception:
            # 握手失败只做**局部清理**（不停牌 _stopped）：这次失败可能是瞬时
            # 的（模型下载中断、端口竞争），之后的 ensure_connected 还应能
            # 自动重试。stop() 是「永久拆除」语义，不能用于这里。
            self._teardown()
            raise

    def stop(self) -> None:
        """永久停止（用户 / registry 拆除意图）：之后不会自动重连。"""
        self._stopped = True
        self._teardown()

    def _teardown(self) -> None:
        """Terminate the MCP server subprocess and wait for threads.

        Cleanup is driven by ``self._process``, NOT by ``self._running``.
        The reader thread sets ``_running = False`` in its ``finally`` block
        whenever the server dies on its own (crash, stdout closed).  If we
        bailed out on ``not self._running``, that path would skip
        terminate()/wait() entirely and leave a zombie child forever — and a
        still-alive server (only its stdout closed) would become an orphan.
        """
        # Signal the reader thread to stop first. We set _running=False under
        # the lock so the reader sees it on its next loop iteration.
        with self._lock:
            self._running = False
            process = self._process
            self._process = None
        # Terminate / wait happen OUTSIDE the lock so the reader thread (which
        # may be in its finally block trying to acquire the lock) doesn't
        # deadlock waiting for us to release it.
        if process:
            self._terminate(process)

        # 管道必须在进程结束之后才关：读线程正阻塞在 stdout.readline() 时，
        # BufferedReader 的内部锁会让 close() 一直阻塞到子进程退出（实测一个
        # 挂起 60s 的 server 会把 stop() 卡住 59s），而且因为旧实现把关管道
        # 排在前面，SIGTERM 根本来不及发出去 —— 握手超时路径尤其明显。
        if process:
            if process.poll() is None:
                # 进程还活着（D 状态等）：放弃关管道也比原地卡死好。管道会在
                # 进程最终退出后随 GC 一起回收。
                logger.warning(
                    "MCP server '%s': pid %d still alive after terminate; "
                    "skipping pipe close (would block on the reader thread)",
                    self.name, process.pid,
                )
            else:
                for pipe in (process.stdin, process.stdout, process.stderr):
                    try:
                        if pipe:
                            pipe.close()
                    except Exception:
                        pass
        # Clear pending requests under the lock.
        with self._lock:
            self._process = None
            # Wake up any waiters
            for ev in self._pending.values():
                ev.set()
            self._pending.clear()
            self._results.clear()
        # 写线程可能阻塞在 stdin.write()（server 不读、管道已满）或队列 get()。
        # 先投递哨兵唤醒 get()，再 join；join 超时说明它卡在 write 里——那是
        # daemon 线程，进程退出时随之回收，不会阻塞 shutdown。
        try:
            self._write_queue.put_nowait(None)
        except queue.Full:
            pass
        writer = self._writer_thread
        self._writer_thread = None
        if writer and writer.is_alive() and writer is not threading.current_thread():
            writer.join(timeout=2)
        # 无论 writer 是否退出，都换一个**全新**的写队列：
        # 卡在 get() 里的旧 writer 会一直阻塞在那条旧队列上（daemon，无害），
        # 但它若还挂在旧队列上，重启后就会和新 writer 变成同一队列的两个
        # 消费者——旧 writer 拿到消息后往已置 None 的 stdin 写，抛错又走
        # _fail_pending 把新连接再次判死。
        self._write_queue = queue.Queue(maxsize=_WRITE_QUEUE_MAX)
        # Wait for reader threads to finish (they should exit once _running is
        # False and stdin is closed, causing readline() to return "").
        for t in (self._reader_thread, self._stderr_thread):
            if t and t.is_alive():
                t.join(timeout=3)

    def _terminate(self, process: "subprocess.Popen") -> None:
        """SIGTERM → wait → SIGKILL → wait，全程带超时，绝不无限等待。"""
        try:
            # 整棵进程组一起收：npx 之类的包装进程会把真正的 server 挂成
            # 孙进程，只 terminate 顶层 PID 会留下孤儿。
            try:
                import os
                import signal
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except Exception:
                process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                import os
                import signal
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except Exception:
                process.kill()
            # SIGKILL 后仍设超时：进程处于不可中断状态（D，常见于 NFS/
            # FUSE I/O）时 wait() 会永久挂起，进而卡死 shutdown/重建。
            # 这种情况下只能放弃等待，让 init 在进程退出后收尸。
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "MCP server '%s': pid %d unkillable (D state?), giving up wait",
                    self.name, process.pid,
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return the list of tools exposed by this MCP server."""
        return list(self._tools)

    def ensure_connected(self) -> bool:
        """连接不在时尝试自动重连（瞬时故障自愈）。

        旧实现：``_fail_pending`` / reader 退出把 ``_running`` 置 False 后，
        ``_send`` 恒抛 "not running"，该 server 的全部工具在本次 Agent
        生命周期内不可用，且用户无从得知。现在区分两种死亡：

        * ``stop()`` 主动拆除 → ``_stopped=True``，永不重启；
        * 连接自己断（server 崩溃 / stdin 写失败 / reader 退出）→ 下次
          工具调用前重启一次，带 30s 冷却（持续崩溃不反复拉起进程）。

        重启前必须 ``_teardown()``：旧 writer 线程还阻塞在旧写队列的
        ``get()`` 上，不换队列的话新旧两个 writer 会共食同一队列，旧 writer
        往已置 None 的 stdin 写 → 又把新连接判死。
        """
        with self._lock:
            if self._running:
                return True
            if self._stopped:
                return False
        now = time.monotonic()
        if now < self._restart_after:
            return False
        self._restart_after = now + _RESTART_COOLDOWN_SECS
        logger.info("MCP server '%s' 连接已断开，尝试自动重连…", self.name)
        try:
            self._teardown()
            self.start()
            logger.info("MCP server '%s' 自动重连成功", self.name)
            return True
        except Exception as e:
            logger.warning("MCP server '%s' 自动重连失败: %s", self.name, e)
            return False

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Call a tool on the MCP server and return the text content."""
        # 连接断开过（瞬时故障）→ 先自动重连再调用，避免整个会话期内该
        # server 的工具全部不可用。重连失败则继续走原路径，以明确的
        # "not running" 错误失败（好过静默丢弃）。
        self.ensure_connected()
        # tools/call 与 initialize/tools/list 不能共用同一个 30s 超时：构建、
        # 批量处理、爬取这类工具跑几分钟是常态。旧实现让它们共用初始化超时，
        # 长任务必然超时失败——而服务端还在继续跑，结果被当迟到响应丢弃。
        resp = self._call(
            "tools/call",
            {"name": name, "arguments": arguments},
            timeout=max(self.timeout, _TOOL_CALL_TIMEOUT),
        )
        if resp.error:
            raise MCPError(
                f"MCP tool '{name}' error: {resp.error.get('message', 'unknown')}"
            )
        result = resp.result
        if not isinstance(result, dict):
            return str(result)
        # MCP tools return content as a list of content blocks
        content = result.get("content", [])
        if isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        texts.append(block.get("text", ""))
                    elif block.get("type") == "resource":
                        texts.append(f"[resource: {block.get('resource', {})}]")
                    else:
                        texts.append(json.dumps(block, ensure_ascii=False))
                else:
                    texts.append(str(block))
            text = _cap_output("\n".join(texts))
        else:
            text = _cap_output(json.dumps(result, ensure_ascii=False))
        # MCP 规范：工具执行失败仍然是 JSON-RPC 成功响应，失败由 result.isError
        # 标记。只检查 resp.error 会把 "permission denied" 之类的错误正文当成
        # 成功结果喂给模型，模型会拿错误信息继续推理。
        if result.get("isError"):
            raise MCPError(f"MCP tool '{name}' failed: {text}")
        return text

    # ------------------------------------------------------------------
    # JSON-RPC internals
    # ------------------------------------------------------------------

    def _call(
        self,
        method: str,
        params: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> JSONRPCResponse:
        """Send a JSON-RPC request and wait for the response.

        ``timeout`` 缺省用初始化超时（适合握手/枚举类调用）；长任务方法应
        传入更长的值（见 call_tool）。
        """
        with self._lock:
            req_id = self._request_id
            self._request_id += 1
            ev = threading.Event()
            self._pending[req_id] = ev

        req = JSONRPCRequest(id=req_id, method=method, params=params or {})
        try:
            self._send(json.dumps(req.__dict__))
        except MCPError:
            with self._lock:
                self._pending.pop(req_id, None)
            raise

        if not ev.wait(timeout=timeout if timeout is not None else self.timeout):
            # 读线程可能刚好在此刻拿到锁、写入 _results 并 set(ev)。只 pop
            # pending 会把那条响应永久留在 _results 里（含完整响应体，
            # _results 只在 stop() 才 clear）→ 每命中一次泄漏一条。
            with self._lock:
                self._pending.pop(req_id, None)
                self._results.pop(req_id, None)
            raise MCPError(f"MCP server '{self.name}': timeout waiting for '{method}'")

        with self._lock:
            resp = self._results.pop(req_id, None)
            self._pending.pop(req_id, None)

        if resp is None:
            raise MCPError(f"MCP server '{self.name}': no response for '{method}'")
        if resp.error:
            raise MCPError(
                f"MCP server '{self.name}' RPC error: {resp.error.get('message', 'unknown')}"
            )
        return resp

    def _send_notification(self, method: str, params: Dict[str, Any]) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        msg = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        self._send(json.dumps(msg))

    def _send(self, line: str) -> None:
        """Enqueue a single JSON-RPC message line for the writer thread.

        旧实现在调用线程里持锁原地 ``stdin.write()``：server 停止读取 stdin 时，
        管道缓冲区（64KB）写满后该调用**无限期阻塞**，并且因为持有写锁，所有
        后续 _call() 全部排队等死——agent 线程彻底卡死。现在写入交给专用线程，
        调用方只做有界入队；单消费者线程天然保证消息不会交错（旧写锁的存在
        理由），最坏情况是入队超时后抛 MCPError。
        """
        if not self._running:
            raise MCPError(f"MCP server '{self.name}': not running")
        try:
            self._write_queue.put(line, timeout=_WRITE_ENQUEUE_TIMEOUT)
        except queue.Full:
            raise MCPError(
                f"MCP server '{self.name}': stdin write queue full "
                f"(server not draining input?)"
            )

    def _write_loop(self) -> None:
        """Dedicated stdin writer: the ONLY thread that writes to the pipe.

        写失败（BrokenPipe/进程已死）时立刻唤醒所有等待中的 _call()——否则
        他们要干等满超时（tools/call 是 300s）才知道请求根本没发出去。
        """
        while True:
            line = self._write_queue.get()
            if line is None:                      # stop() 的哨兵
                return
            process = self._process
            try:
                if process is None or not process.stdin:
                    raise MCPError(f"MCP server '{self.name}': not running")
                process.stdin.write(line + "\n")
                process.stdin.flush()
            except Exception as e:
                logger.debug("MCP '%s' write failed: %s", self.name, e)
                self._fail_pending("MCP server connection lost (stdin write failed)")
                return

    def _drain_stderr(self) -> None:
        """Continuously read stderr to prevent buffer deadlock."""
        if not self._process or not self._process.stderr:
            return
        try:
            for line in self._process.stderr:
                logger.debug("MCP '%s' stderr: %s", self.name, line.rstrip())
        except Exception:
            pass

    def _read_loop(self) -> None:
        """Continuously read JSON-RPC responses from the subprocess stdout."""
        try:
            # 局部持有 pipe 引用：旧实现每轮读 ``self._process``——stop() 会把
            # 它置 None，reader 已通过检查后 stop() 关闭 stdout，readline()
            # 抛 ValueError 被 except 吞掉并走 _fail_pending，把一次**正常停止**
            # 误报成「连接丢失」（还会借机触发自动重连，白拉起一个 server）。
            process = self._process
            while self._running and process and process.stdout:
                line = process.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("MCP server '%s' non-JSON line: %s", self.name, line[:200])
                    continue

                # Skip notifications (no id)
                if "id" not in obj:
                    continue

                resp = JSONRPCResponse(
                    jsonrpc=obj.get("jsonrpc", "2.0"),
                    id=obj["id"],
                    result=obj.get("result"),
                    error=obj.get("error"),
                )
                with self._lock:
                    ev = self._pending.get(resp.id)
                    if ev is None:
                        # 迟到的响应：对应的 _call() 已超时/被取消，没人会再来
                        # pop 它。旧实现无条件写入 _results，这些条目会永久
                        # 驻留（内存泄漏）。直接丢弃。
                        continue
                    self._results[resp.id] = resp
                ev.set()
        except Exception as e:
            logger.debug("MCP server '%s' reader exited: %s", self.name, e)
        finally:
            # The reader is exiting (server crashed, stdin closed, or stop()).
            # Wake up any pending callers so they don't wait the full timeout.
            # Store an error response so _call() reports a clear message instead
            # of a generic "no response".
            self._fail_pending("MCP server connection closed")

    def _fail_pending(self, reason: str) -> None:
        """唤醒所有等待中的 _call()，让它们以明确错误失败而不是干等超时。"""
        with self._lock:
            self._running = False
            for req_id, ev in self._pending.items():
                if req_id not in self._results:
                    self._results[req_id] = JSONRPCResponse(
                        id=req_id,
                        error={"message": reason, "code": -1},
                    )
                ev.set()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _cap_output(text: str) -> str:
    """把 MCP 工具输出钳制到 _MAX_TOOL_OUTPUT_CHARS，超长部分截断并标注。

    截断标记必须让模型知道「结果被裁了」，否则它会把半截输出当成完整结果
    继续推理。
    """
    if len(text) <= _MAX_TOOL_OUTPUT_CHARS:
        return text
    return (
        text[:_MAX_TOOL_OUTPUT_CHARS]
        + f"\n… [MCP 工具输出共 {len(text)} 字符，已截断到 {_MAX_TOOL_OUTPUT_CHARS}]"
    )


# ------------------------------------------------------------------
# Error type
# ------------------------------------------------------------------


class MCPError(Exception):
    """Raised when MCP communication fails."""
    pass