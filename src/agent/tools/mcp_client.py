"""MCP (Model Context Protocol) client built on ``fastmcp`` 4.x.

协议实现统一交给 fastmcp：握手/协议版本协商、tools/list、tools/call、
resources/list、resources/read、prompts/list、prompts/get、以及 stdio /
streamable-HTTP / SSE 三种传输都由 ``fastmcp.Client`` 负责，本模块只提供：

  * **同步门面** —— 内核与 ``ToolRegistry`` 是同步的 worker 线程，fastmcp 的
    ``Client`` 全异步。这里用一个常驻后台 event loop + 单一 session task 持有
    ``Client`` 上下文，同步侧经 ``concurrent.futures.Future`` 等待结果。
    ``Client.__aenter__`` 用 ``asyncio.current_task()`` 记录连接持有者，因此
    enter / exit / 所有 RPC 必须在**同一个 task** 内完成；把请求排进队列由
    session task 顺序执行正好满足这一点，也避免每次调用重连 stdio 子进程。

  * **兼容契约** —— ``MCPClient`` 的同步方法、``MCPError``、``_cap_output``、
    ``_build_env`` 的安全白名单、超时分层、重连冷却、永久停牌语义保持与旧版
    一致，``MCPTool`` / ``ToolRegistry`` 无需感知底层换成了 fastmcp。

  * **配置** —— 旧的 ``{name, command, args?, env?}``（stdio）继续可用；新增
    ``{name, url, headers?, transport?: 'http'|'sse'}``（远程 MCP server）。
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import threading
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


class MCPError(Exception):
    """Raised when MCP communication fails."""
    pass


# 只向 MCP 子进程传递「运行所必需」的最小环境，避免父进程的全部环境变量
# （含各种 API_KEY/TOKEN）被泄露给第三方 MCP server。需要的额外变量由
# server 配置里的 env 显式提供（用户已信任该 server）。
# 注意：fastmcp 的 StdioTransport 不会自动继承 os.environ（与旧实现一致），
# 必须显式传 env；这里仍然只给白名单内容。
_MCP_SAFE_ENV = (
    "PATH", "HOME", "USER", "LOGNAME", "SHELL",
    "LANG", "LC_ALL", "TERM", "TMPDIR", "TEMP", "TMP",
    "SYSTEMROOT", "SYSTEMDRIVE", "PATHEXT", "COMSPEC", "XDG_CACHE_HOME",
)

# tools/call 的下限超时（秒）。握手/枚举用实例的 timeout（默认 30s）足够，
# 但工具调用本身可能是构建、批量处理、爬取等长任务——共用 30s 会让长任务
# 必然超时。fastmcp 的 Client 有全局 timeout，这里显式按调用覆盖。
_TOOL_CALL_TIMEOUT = 300.0

# 关闭上下文的超时（秒）。挂起的 stdio server 不应让 stop() 无限期阻塞。
_CLOSE_TIMEOUT = 5.0

# 连接断开后自动重连的冷却（秒）。瞬时故障不应让该 server 的工具在本次
# Agent 生命周期内全部不可用；但持续崩溃的 server 也不能每个调用都重连。
_RESTART_COOLDOWN_SECS = 30.0

# 单次 MCP 工具返回保留的最大字符数。MCP 工具是外部代码，输出体积不受我们
# 控制，而这段文本会原样进入模型上下文——不设上限一次调用就能把窗口撑爆。
# 与 file_read / shell_run 的 200K 策略保持一致，保留头部并明确标注截断。
_MAX_TOOL_OUTPUT_CHARS = 200_000

# 兼容常量：旧实现的 stdin 写队列上限 / 入队超时。新实现不再有写线程，
# 但请求队列仍按同一上限做背压，避免工具调用堆积把后台 loop 拖垮。
_WRITE_QUEUE_MAX = 64
_WRITE_ENQUEUE_TIMEOUT = 10.0

# fastmcp 是核心依赖（见 pyproject），但导入期仍做防御：即使某个精简安装
# 缺了它，主程序也能启动，只有实际连接 MCP server 时给出明确提示。
try:
    from fastmcp import Client as _FastMCPClient
    from fastmcp.client.transports import (
        SSETransport as _SSETransport,
        StdioTransport as _StdioTransport,
        StreamableHttpTransport as _StreamableHttpTransport,
    )
    _HAS_FASTMCP = True
    _FASTMCP_IMPORT_ERROR: Optional[str] = None
except Exception as _e:  # pragma: no cover - 取决于安装
    _FastMCPClient = None  # type: ignore[assignment]
    _SSETransport = _StdioTransport = _StreamableHttpTransport = None  # type: ignore[assignment]
    _HAS_FASTMCP = False
    _FASTMCP_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"


# ------------------------------------------------------------------
# 结果转换
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


def _json_default(value: Any) -> str:
    return str(value)


def _dump_obj(obj: Any) -> str:
    """pydantic 模型 / dict / 其它对象 → JSON 字符串（不抛异常）。"""
    try:
        if hasattr(obj, "model_dump"):
            obj = obj.model_dump(by_alias=True, exclude_none=True)
        elif hasattr(obj, "__dict__") and not isinstance(obj, (dict, list)):
            obj = dict(obj.__dict__)
        return json.dumps(obj, ensure_ascii=False, default=_json_default)
    except Exception:
        return str(obj)


def _tool_to_dict(tool: Any) -> Dict[str, Any]:
    """fastmcp 的 ``mcp.types.Tool`` → MCPTool 需要的 dict（含 inputSchema）。

    ``model_dump(by_alias=True)`` 把字段 ``input_schema`` 还原成协议里的
    ``inputSchema``。任何形状异常都兜底成空 schema，MCPTool 另有兜底逻辑。
    """
    try:
        data = tool.model_dump(by_alias=True, exclude_none=True)
        if not isinstance(data, dict):
            raise TypeError("tool dump is not a dict")
    except Exception:
        data = {
            "name": str(getattr(tool, "name", "") or ""),
            "description": str(getattr(tool, "description", "") or ""),
        }
    schema = data.get("inputSchema")
    if not isinstance(schema, dict):
        data["inputSchema"] = {"type": "object", "properties": {}, "required": []}
    return data


def _block_to_text(block: Any) -> str:
    """单个 MCP 内容块 → 可读文本。

    覆盖三类：普通 ContentBlock / TextResourceContents（都有 ``.text``）、
    PromptMessage（``.role`` + ``.content``）、以及 image/audio/resource 等
    非文本块（转 JSON，保留结构信息）。
    """
    # PromptMessage：role + 内层 content（TextContent / ImageContent …）
    role = getattr(block, "role", None)
    if role is not None and hasattr(block, "content"):
        return f"{role}: {_block_to_text(block.content)}"
    if hasattr(block, "text"):
        return str(getattr(block, "text", "") or "")
    resource = getattr(block, "resource", None)
    if resource is not None and not isinstance(block, dict):
        return f"[resource: {_dump_obj(resource)}]"
    if isinstance(block, dict):
        if block.get("type") == "text":
            return str(block.get("text", "") or "")
        if block.get("type") == "resource":
            return f"[resource: {_dump_obj(block.get('resource', {}))}]"
    return _dump_obj(block)


def _render_content(result: Any) -> str:
    """fastmcp 各种结果 → 统一文本（200K 截断）。

    fastmcp 4.x 的结果形状不统一，必须逐个适配，否则会退化成把整个
    pydantic 对象 dump 成 JSON、可读性很差：
      * ``CallToolResult``      → ``.content`` 列表
      * ``read_resource``       → **直接返回** list（无 ``.content``）
      * ``GetPromptResult``     → ``.messages`` 列表（每条是 PromptMessage）
      * 结构化输出              → ``.structured_content``
    """
    blocks: Any = None
    if isinstance(result, (list, tuple)):
        blocks = result                       # read_resource
    else:
        content = getattr(result, "content", None)
        if isinstance(content, (list, tuple)):
            blocks = content                  # CallToolResult
        else:
            messages = getattr(result, "messages", None)
            if isinstance(messages, (list, tuple)):
                blocks = messages             # GetPromptResult
    if blocks is not None:
        return _cap_output("\n".join(_block_to_text(b) for b in blocks))
    structured = getattr(result, "structured_content", None)
    if structured:
        return _cap_output(_dump_obj(structured))
    return _cap_output(_dump_obj(result))


# ------------------------------------------------------------------
# 异步桥接：常驻 event loop + 单一 session task
# ------------------------------------------------------------------


class _AsyncSession:
    """在一个后台线程里长期持有 fastmcp Client 的异步上下文。

    所有 RPC 都排进 ``asyncio.Queue``，由持有上下文的那个 task 顺序执行；
    同步线程用 Future 取结果。这样 ``__aenter__`` / ``__aexit__`` / 每次调用
    都落在同一个 ``asyncio.Task`` 里，符合 fastmcp 对连接持有者的约束。
    """

    def __init__(self, client: Any, name: str, close_timeout: float = _CLOSE_TIMEOUT):
        self._client = client
        self._name = name
        self._close_timeout = close_timeout
        self._loop = asyncio.new_event_loop()
        self._queue: Optional[asyncio.Queue] = None
        self._ready = threading.Event()
        self._error: Optional[BaseException] = None
        self._closed = False
        # 所有在途请求（含正在执行的）。stop() 靠它让同步调用方立即返回，
        # 而不是等慢工具跑完（tools/call 上限 300s）。
        self._inflight: "set[concurrent.futures.Future[Any]]" = set()
        self._inflight_lock = threading.Lock()
        # 仅由 event loop 线程读写：当前正在等待的 RPC task，供 stop() 取消。
        self._current_task: Optional["asyncio.Task[Any]"] = None
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"mcp-session-{name}")

    @property
    def alive(self) -> bool:
        return (not self._closed) and self._thread.is_alive()

    def start(self, timeout: float) -> None:
        self._thread.start()
        if not self._ready.wait(max(1.0, timeout)):
            self.stop()
            raise MCPError(f"MCP server '{self._name}': initialize 握手超时")
        if self._error is not None:
            err = self._error
            self.stop()
            raise err

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._session_main())
        except BaseException as e:  # pragma: no cover - 后台线程兜底
            logger.debug("MCP '%s' event loop exited: %s", self._name, e)
        finally:
            try:
                self._loop.close()
            except Exception:
                pass

    async def _session_main(self) -> None:
        q: asyncio.Queue = asyncio.Queue()
        self._queue = q
        entered = False
        try:
            await self._client.__aenter__()
            entered = True
            self._ready.set()
            while True:
                item = await q.get()
                if item is None:
                    break
                fn, fut, call_timeout = item
                try:
                    # 用显式 Task：stop() 可以 cancel 它，让 wait_for 立刻返回，
                    # session 不必等慢工具自然结束就能退出。
                    task = asyncio.ensure_future(fn())
                    self._current_task = task
                    value = await asyncio.wait_for(task, timeout=call_timeout)
                except BaseException as e:
                    if not fut.done():
                        fut.set_exception(e)
                else:
                    if not fut.done():
                        fut.set_result(value)
                finally:
                    self._current_task = None
        except BaseException as e:
            self._error = e
            self._ready.set()
        finally:
            if entered:
                try:
                    # 退出也必须在同一 task 内；加超时防止挂起 server 卡死 stop()。
                    await asyncio.wait_for(
                        self._client.__aexit__(None, None, None),
                        timeout=self._close_timeout)
                except BaseException as e:
                    logger.debug("MCP '%s' 上下文关闭异常(忽略): %s", self._name, e)
            self._drain(q)

    def _drain(self, q: "asyncio.Queue") -> None:
        """连接关闭时叫醒所有还在等结果的同步调用方。"""
        while True:
            try:
                item = q.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is None:
                continue
            _, fut, _ = item
            if not fut.done():
                fut.set_exception(MCPError(
                    f"MCP server '{self._name}' 连接已关闭"))

    def submit(self, fn: Callable[[], Coroutine[Any, Any, Any]],
               timeout: float) -> Any:
        """从同步线程提交一次 RPC，阻塞等待结果；任何异常交由调用方收敛。"""
        if self._closed or self._queue is None or not self._loop.is_running():
            raise MCPError(f"MCP server '{self._name}' not running")
        if self._queue.qsize() >= _WRITE_QUEUE_MAX:
            raise MCPError(
                f"MCP server '{self._name}' request queue full "
                f"({_WRITE_QUEUE_MAX} pending)")
        fut: "concurrent.futures.Future[Any]" = concurrent.futures.Future()
        # 先登记再入队：避免 stop() 与「请求已被 session 取走但尚未登记」的竞态
        # 窗口里漏掉一个等待方。
        with self._inflight_lock:
            self._inflight.add(fut)
        try:
            try:
                self._loop.call_soon_threadsafe(
                    self._queue.put_nowait, (fn, fut, timeout))
            except RuntimeError as e:
                raise MCPError(f"MCP server '{self._name}' event loop closed") from e
            try:
                return fut.result(timeout=timeout + 1.0)
            except concurrent.futures.TimeoutError as e:
                raise MCPError(
                    f"MCP server '{self._name}' request timed out after {timeout:.0f}s") from e
            except concurrent.futures.CancelledError as e:
                raise MCPError(f"MCP server '{self._name}' request cancelled") from e
        finally:
            with self._inflight_lock:
                self._inflight.discard(fut)

    def stop(self) -> None:
        """关闭连接：立即叫醒所有在途调用，取消正在执行的 RPC，优雅退出。

        「终止对话」时后台可能有一个 tools/call 正跑长任务（上限 300s）。若等它
        自然结束，调用方与 stop() 都会被拖住。这里先给所有 Future 置明确错误，
        再取消当前 RPC task，让 session 立刻走到 __aexit__。
        """
        if self._closed:
            return
        self._closed = True
        with self._inflight_lock:
            pending = list(self._inflight)
        for fut in pending:
            if not fut.done():
                fut.set_exception(MCPError(
                    f"MCP server '{self._name}' 已停止，请求被取消"))
        if self._queue is not None and self._loop.is_running():
            def _shutdown() -> None:
                task = self._current_task
                if task is not None and not task.done():
                    task.cancel()
                try:
                    self._queue.put_nowait(None)
                except Exception:
                    pass
            try:
                self._loop.call_soon_threadsafe(_shutdown)
            except RuntimeError:
                pass
        try:
            self._thread.join(timeout=self._close_timeout + 1.0)
        except Exception:
            pass


# ------------------------------------------------------------------
# MCP Client（同步门面）
# ------------------------------------------------------------------


class MCPClient:
    """Manages one MCP server connection (stdio subprocess or remote HTTP/SSE).

    Public interface is intentionally identical to the previous hand-written
    JSON-RPC client, so ``ToolRegistry`` / ``MCPTool`` need no changes.
    """

    def __init__(
        self,
        name: str,
        command: str = "",
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        *,
        url: str = "",
        transport: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        server: Any = None,
    ):
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env
        self.timeout = float(timeout)
        # 远程 server 配置（新增能力）；transport ∈ {http, sse}，缺省 http。
        self.url = url or ""
        self.transport_kind = (transport or "").strip().lower()
        self.headers = headers or {}
        self.cwd = cwd
        # 直接传 fastmcp.FastMCP 实例 → 进程内连接（不拉起子进程），供测试
        # 与嵌入式使用；为 None 时按 command/url 选择 stdio 或远程传输。
        self._server = server

        self._session: Optional[_AsyncSession] = None
        self._client: Any = None
        self._tools: List[Dict[str, Any]] = []
        self._server_info: Dict[str, Any] = {}
        self._running = False
        self._stopped = False
        self._restart_after = 0.0
        # 保留 _lock：外部（UI 关闭路径）依赖它做串行化与状态保护。
        self._lock = threading.RLock()

    # -- 传输与环境 -------------------------------------------------

    def _build_env(self) -> Optional[Dict[str, str]]:
        """构造传给 MCP 子进程的最小环境变量（安全白名单 + 配置 env）。

        fastmcp 的 ``StdioTransport`` 不会自动继承 ``os.environ``，所以这里
        必须显式构造；同时绝不把父进程的 API_KEY/TOKEN 透传给第三方 server。
        """
        if self.env or _MCP_SAFE_ENV:
            env = {k: os.environ[k] for k in _MCP_SAFE_ENV if k in os.environ}
            if self.env:
                env.update(self.env)
            return env
        return None

    def _make_transport(self) -> Any:
        if not _HAS_FASTMCP:
            raise MCPError(
                "未安装 fastmcp，无法连接 MCP server。请先安装依赖："
                "pip install 'fastmcp>=4.0'"
                + (f"（导入错误：{_FASTMCP_IMPORT_ERROR}）" if _FASTMCP_IMPORT_ERROR else "")
            )
        if self._server is not None:
            return self._server  # Client 直接接受 FastMCP 实例
        if self.url:
            kind = self.transport_kind or "http"
            if kind in ("http", "streamable-http", "streamable_http", "streamablehttp"):
                return _StreamableHttpTransport(
                    url=self.url, headers=self.headers or None)
            if kind == "sse":
                return _SSETransport(url=self.url, headers=self.headers or None)
            raise MCPError(
                f"MCP server '{self.name}': 未知 transport {kind!r}，"
                "可选 'http'(streamable-HTTP) / 'sse'")
        if not self.command:
            raise MCPError(
                f"MCP server '{self.name}': 配置缺少 command（stdio）或 url（HTTP/SSE）")
        # keep_alive=False 关键：默认 True 时 fastmcp 在连接上下文退出后仍保留
        # 子进程，stop() 会留下孤儿进程。
        return _StdioTransport(
            command=self.command,
            args=list(self.args),
            env=self._build_env(),
            cwd=self.cwd,
            keep_alive=False,
        )

    # -- 生命周期 ---------------------------------------------------

    def start(self) -> None:
        """建立连接、完成 initialize 握手并枚举工具（同步阻塞）。"""
        with self._lock:
            self._start_locked(reset_cooldown=True)

    def _start_locked(self, reset_cooldown: bool) -> None:
        """实际建连逻辑（调用方已持 _lock）。

        ``reset_cooldown=False`` 供 ensure_connected() 的自动重连使用：显式
        start 才清零冷却，否则一次失败重连会立刻把冷却抹掉，导致持续崩溃的
        server 每个工具调用都重新拉起一次。
        """
        if self._running:
            return
        # 显式 start = 用户意图：清掉「已主动停止」，允许重新拉起。
        self._stopped = False
        if reset_cooldown:
            self._restart_after = 0.0
        self._teardown_locked()

        transport = self._make_transport()
        client = _FastMCPClient(
            transport,
            timeout=self.timeout,
            init_timeout=self.timeout,
        )
        session = _AsyncSession(client, self.name)
        try:
            session.start(self.timeout)
            tools = session.submit(
                lambda: client.list_tools(), timeout=self.timeout)
            self._tools = [_tool_to_dict(t) for t in (tools or [])]
            info = getattr(client, "server_info", None)
            self._server_info = _safe_dict(info) if info is not None else {}
            caps = getattr(client, "server_capabilities", None)
            if caps is not None:
                self._server_info["capabilities"] = _safe_dict(caps)
        except BaseException as e:
            session.stop()
            if isinstance(e, MCPError):
                raise MCPError(
                    f"MCP server '{self.name}': failed to start: {e}") from e
            raise MCPError(
                f"MCP server '{self.name}': failed to start: "
                f"{type(e).__name__}: {e}") from e
        self._session = session
        self._client = client
        self._running = True
        logger.info("MCP server '%s' initialized: %d tools available",
                    self.name, len(self._tools))

    def stop(self) -> None:
        """永久停牌：关闭连接、回收子进程，幂等且不抛异常。"""
        with self._lock:
            self._stopped = True
            self._running = False
            self._teardown_locked()

    def _teardown_locked(self) -> None:
        """关闭当前 session（调用方已持 _lock），异常一律吞掉。"""
        session, self._session = self._session, None
        self._client = None
        self._tools = []
        if session is not None:
            try:
                session.stop()
            except Exception as e:
                logger.debug("MCP '%s' stop 异常(忽略): %s", self.name, e)

    def ensure_connected(self) -> bool:
        """确认连接可用；断开时按冷却策略自动重连，失败吞异常返回 False。"""
        with self._lock:
            if self._stopped:
                return False
            if self._running and self._session is not None and self._session.alive:
                try:
                    if bool(self._client.is_connected()):
                        return True
                except Exception:
                    pass
            now = time.monotonic()
            if now < self._restart_after:
                return False
            self._restart_after = now + _RESTART_COOLDOWN_SECS
            self._running = False
            try:
                self._teardown_locked()
                self._start_locked(reset_cooldown=False)
                return True
            except Exception as e:
                logger.warning("MCP server '%s' 自动重连失败: %s", self.name, e)
                return False

    # -- 协议能力 ---------------------------------------------------

    @property
    def server_info(self) -> Dict[str, Any]:
        """initialize 握手返回的 serverInfo / capabilities。"""
        return dict(self._server_info)

    def list_tools(self) -> List[Dict[str, Any]]:
        """已注册工具定义（dict，含 ``inputSchema``），供 MCPTool 消费。"""
        with self._lock:
            return list(self._tools)

    def _require_session(self) -> "_AsyncSession":
        if not self.ensure_connected():
            raise MCPError(f"MCP server '{self.name}' not running")
        session = self._session
        if session is None or not session.alive:
            raise MCPError(f"MCP server '{self.name}' not running")
        return session

    def _rpc(self, coro_factory: Callable[[], Coroutine[Any, Any, Any]],
             timeout: float, what: str) -> Any:
        session = self._require_session()
        try:
            return session.submit(coro_factory, timeout=timeout)
        except MCPError:
            raise
        except BaseException as e:
            # fastmcp / mcp SDK 的 ToolError、MCPError、TimeoutError 等一律收敛成
            # 本模块的 MCPError —— ToolRegistry 只认它，MCPTool 会把 str(e) 直接
            # 喂给模型，不能泄漏底层英文异常类型。
            raise MCPError(f"{what}: {type(e).__name__}: {e}") from e

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """调用工具并返回文本内容（200K 截断；isError → MCPError）。"""
        call_timeout = max(self.timeout, _TOOL_CALL_TIMEOUT)
        result = self._rpc(
            # 在 coroutine 真正执行时才取 self._client：_rpc 里的
            # ensure_connected() 可能刚完成一次重连并替换了 client 实例。
            lambda: self._client.call_tool(
                name, arguments or {}, timeout=call_timeout,
                raise_on_error=False),
            call_timeout,
            f"MCP tool '{name}' error",
        )
        text = _render_content(result)
        # MCP 规范：工具执行失败仍是 JSON-RPC 成功响应，失败由 isError 标记。
        # 只检查传输层异常会把 "permission denied" 当成功结果喂给模型。
        if bool(getattr(result, "is_error", False)):
            raise MCPError(f"MCP tool '{name}' failed: {text}")
        return text

    # 以下是旧客户端没有、由 fastmcp 直接带来的完整协议能力。ToolRegistry
    # 目前只用到 tools，但这些入口让客户端可作为规范 MCP client 单独使用。
    def list_resources(self) -> List[Dict[str, Any]]:
        resources = self._rpc(
            lambda: self._client.list_resources(), self.timeout,
            "MCP resources/list error")
        return [_safe_dict(r) for r in (resources or [])]

    def read_resource(self, uri: str) -> str:
        result = self._rpc(
            lambda: self._client.read_resource(uri), self.timeout,
            f"MCP resources/read '{uri}' error")
        return _render_content(result)

    def list_prompts(self) -> List[Dict[str, Any]]:
        prompts = self._rpc(
            lambda: self._client.list_prompts(), self.timeout,
            "MCP prompts/list error")
        return [_safe_dict(p) for p in (prompts or [])]

    def get_prompt(self, name: str,
                   arguments: Optional[Dict[str, Any]] = None) -> str:
        result = self._rpc(
            lambda: self._client.get_prompt(name, arguments or {}),
            self.timeout, f"MCP prompts/get '{name}' error")
        return _render_content(result)


def _safe_dict(obj: Any) -> Dict[str, Any]:
    """pydantic 模型 → dict；失败时给出最小可读结构。"""
    try:
        if hasattr(obj, "model_dump"):
            data = obj.model_dump(by_alias=True, exclude_none=True)
            if isinstance(data, dict):
                return data
        if isinstance(obj, dict):
            return dict(obj)
    except Exception:
        pass
    return {"value": str(obj)}
