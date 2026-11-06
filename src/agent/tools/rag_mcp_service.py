"""RAG MCP 服务生命周期管理：在后台线程运行 streamable-HTTP server。

把 ``rag_mcp_server.build_mcp_server()`` 的 FastMCP 实例以 2025 规范的
streamable-HTTP 传输跑起来，供局域网/本机其它大模型应用对接。由 UI 配置面板
的「启动/停止」按钮驱动，状态写回 AgentConfig（rag_mcp_enabled 等）。

生命周期范式复用项目既有的后台 worker 模式（QThread / daemon 线程），这里用
普通 daemon 线程承载 uvicorn 事件循环——PyQt 主线程不应被 server.serve() 阻塞。
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


def _scope_to_host(scope: str) -> str:
    """监听范围 → 绑定地址。localhost 仅本机（127.0.0.1），lan 允许局域网（0.0.0.0）。"""
    return "0.0.0.0" if scope == "lan" else "127.0.0.1"


class RagMcpService:
    """RAG MCP 服务的启停与状态管理（线程安全）。"""

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._uvicorn: Optional[object] = None  # uvicorn.Server
        self._host: Optional[str] = None
        self._port: Optional[int] = None
        self._running = False
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        # 不加锁：与 start/stop/url 内部的 `with self._lock` 互斥，加锁会重入死锁
        # （threading.Lock 不可重入）。状态为 best-effort 读取。
        return bool(self._running and self._thread is not None and self._thread.is_alive())

    def status(self) -> str:
        """返回人类可读状态：'running' / 'stopped'。"""
        return "running" if self.is_running else "stopped"

    def url(self) -> Optional[str]:
        """返回服务根 URL（运行中有效）。"""
        with self._lock:
            if not self.is_running or self._host is None or self._port is None:
                return None
            return f"http://{self._host}:{self._port}/mcp/"

    def start(self, scope: str = "localhost", port: int = 8080) -> None:
        """启动服务（后台线程）。scope='lan' 时绑定 0.0.0.0，否则 127.0.0.1。"""
        with self._lock:
            if self.is_running:
                raise RuntimeError("RAG MCP 服务已在运行")
            host = _scope_to_host(scope)
            self._host = host
            self._port = port
            self._uvicorn = None
            self._running = True
            t = threading.Thread(
                target=self._serve, args=(host, port), daemon=True, name="rag-mcp")
            t.start()
            self._thread = t
        logger.info("RAG MCP 服务启动中: %s:%d (scope=%s)", host, port, scope)

    def _serve(self, host: str, port: int) -> None:
        try:
            from .rag_mcp_server import run_server, _HAS_FASTMCP
            if not _HAS_FASTMCP:
                logger.error("fastmcp 未安装，无法启动 RAG MCP 服务。"
                             "请先安装可选依赖：pip install fastmcp")
                with self._lock:
                    self._running = False
                return
            # 复用 rag_mcp_server.run_server：HTTP 模式下把 uvicorn.Server 回写
            # self._uvicorn，使 stop() 能优雅退出（should_exit）。
            run_server(transport="http", host=host, port=port, service=self)
        except Exception as e:  # noqa: BLE001
            logger.error("RAG MCP 服务异常退出: %s", e)
        finally:
            with self._lock:
                self._running = False
                self._uvicorn = None

    def stop(self) -> None:
        """停止服务：通知 uvicorn 优雅退出（should_exit）。"""
        with self._lock:
            server = self._uvicorn
            running = self._running
        if server is not None:
            try:
                server.should_exit = True
            except Exception as e:  # noqa: BLE001
                logger.warning("RAG MCP 停止信号发送失败: %s", e)
        if running:
            logger.info("RAG MCP 服务停止信号已发送")
