"""MCP 客户端迁移到 fastmcp 4.x 后的回归测试。

覆盖：同步门面 / 工具定义形状 / 完整协议能力 / 三种传输选择 / env 白名单 /
配置透传 / 停牌与线程清理 / stop 取消在途请求。旧的 JSON-RPC 手写实现相关
断言已移到等价语义。
"""
from __future__ import annotations

import pytest

fastmcp = pytest.importorskip("fastmcp")
from fastmcp import FastMCP  # noqa: E402

from agent.tools.mcp_client import (  # noqa: E402
    MCPClient, MCPError, _MCP_SAFE_ENV, _MAX_TOOL_OUTPUT_CHARS,
)


def _server() -> "FastMCP":
    mcp = FastMCP("mcp-client-test")

    @mcp.tool()
    def echo(msg: str, times: int = 1) -> str:
        return msg * times

    @mcp.tool()
    def huge() -> str:
        return "x" * (_MAX_TOOL_OUTPUT_CHARS + 500)

    return mcp


def test_list_tools_returns_dict_with_input_schema():
    """MCPTool 依赖 list_tools() 返回 dict 且含协议别名 inputSchema。"""
    c = MCPClient(name="t", server=_server())
    c.start()
    try:
        defs = c.list_tools()
        assert defs and all(isinstance(d, dict) for d in defs)
        by_name = {d["name"]: d for d in defs}
        schema = by_name["echo"]["inputSchema"]
        assert schema["type"] == "object"
        assert "msg" in schema["required"]
        assert "msg" in schema["properties"]
        # 返回的是浅拷贝，外部改动不污染内部缓存
        defs.clear()
        assert c.list_tools()
    finally:
        c.stop()


def test_call_tool_text_and_output_cap():
    c = MCPClient(name="t", server=_server())
    c.start()
    try:
        assert c.call_tool("echo", {"msg": "ab", "times": 2}) == "abab"
        out = c.call_tool("huge", {})
        assert len(out) < _MAX_TOOL_OUTPUT_CHARS + 200
        assert "截断" in out
    finally:
        c.stop()


def test_full_mcp_capabilities_are_exposed():
    """fastmcp 带来的 resources/prompts 入口可用（空实现返回空列表而非报错）。"""
    c = MCPClient(name="t", server=_server())
    c.start()
    try:
        assert isinstance(c.list_resources(), list)
        assert isinstance(c.list_prompts(), list)
        assert isinstance(c.server_info, dict)
        assert c.server_info, "initialize 握手信息应被填充"
        assert "capabilities" in c.server_info
    finally:
        c.stop()


def test_invalid_config_raises_mcperror():
    """缺少 command/url、未知 transport 都要给出明确 MCPError。"""
    with pytest.raises(MCPError, match="command"):
        MCPClient(name="t").start()
    with pytest.raises(MCPError, match="transport"):
        MCPClient(name="t", url="http://127.0.0.1:1/mcp/",
                  transport="grpc").start()


def test_transport_selection(monkeypatch):
    """stdio / streamable-HTTP / SSE 三种传输按配置选择，headers 透传。"""
    import agent.tools.mcp_client as m

    made = {}

    class _FakeTransport:
        def __init__(self, kind, **kw):
            made["kind"] = kind
            made["kw"] = kw

    monkeypatch.setattr(m, "_StdioTransport",
                        lambda **kw: _FakeTransport("stdio", **kw))
    monkeypatch.setattr(m, "_StreamableHttpTransport",
                        lambda **kw: _FakeTransport("http", **kw))
    monkeypatch.setattr(m, "_SSETransport",
                        lambda **kw: _FakeTransport("sse", **kw))

    c = MCPClient(name="t", command="npx", args=["-y", "pkg"], cwd="/tmp")
    c._make_transport()
    assert made["kind"] == "stdio"
    # 关键：keep_alive=False，否则 stop() 后子进程变孤儿
    assert made["kw"]["keep_alive"] is False
    assert made["kw"]["args"] == ["-y", "pkg"]
    assert made["kw"]["cwd"] == "/tmp"
    assert "env" in made["kw"]

    c = MCPClient(name="t", url="http://host/mcp/",
                  headers={"Authorization": "Bearer x"})
    c._make_transport()
    assert made["kind"] == "http"
    assert made["kw"]["headers"] == {"Authorization": "Bearer x"}

    c = MCPClient(name="t", url="http://host/sse", transport="sse")
    c._make_transport()
    assert made["kind"] == "sse"


def test_stdio_env_whitelist_keeps_secrets_out(monkeypatch):
    """只透传白名单 + 配置 env，父进程密钥不泄露给第三方 server。"""
    monkeypatch.setenv("BASE_AGENT_TEST_LEAK", "top-secret")
    monkeypatch.setenv("PATH", "/usr/bin")
    c = MCPClient(name="t", command="true", env={"INJECTED": "v"})
    env = c._build_env()
    assert env is not None
    assert env["INJECTED"] == "v"
    assert env["PATH"] == "/usr/bin"
    assert "BASE_AGENT_TEST_LEAK" not in env
    assert all(k in _MCP_SAFE_ENV or k == "INJECTED" for k in env)


def test_registry_passes_remote_config(monkeypatch):
    """ToolRegistry 把 url/transport/headers/timeout 正确透传给 MCPClient。"""
    import agent.tools.mcp_client as m
    from agent.config import AgentConfig
    from agent.tools import ToolRegistry
    from agent.agent import AgentCallbacks

    captured = {}

    class FakeClient:
        def __init__(self, **kw):
            captured.update(kw)

        def start(self):
            pass

        def list_tools(self):
            return []

        def stop(self):
            pass

    monkeypatch.setattr(m, "MCPClient", FakeClient)
    cfg = AgentConfig(mcp_servers=[{
        "name": "remote",
        "url": "http://127.0.0.1:8080/mcp/",
        "transport": "http",
        "headers": {"Authorization": "Bearer x"},
        "timeout": 7,
    }])
    reg = ToolRegistry(config=cfg, callbacks=AgentCallbacks())
    try:
        assert captured["url"] == "http://127.0.0.1:8080/mcp/"
        assert captured["transport"] == "http"
        assert captured["headers"] == {"Authorization": "Bearer x"}
        assert captured["timeout"] == 7.0
    finally:
        reg.shutdown()


def test_stop_cleans_up_session_thread():
    """stop() 回收后台 event loop 线程，不留下活动 session。"""
    c = MCPClient(name="t", server=_server())
    c.start()
    session = c._session
    assert session is not None and session.alive
    c.stop()
    assert not session.alive
    assert c._session is None
    assert c.ensure_connected() is False


def test_resources_and_prompts_render_as_text():
    """read_resource 返回纯文本、get_prompt 返回消息正文，而非整个对象 dump。

    fastmcp 的 read_resource 直接返回 list、get_prompt 返回 .messages，
    与 CallToolResult.content 形状都不同，必须逐个适配。
    """
    m = FastMCP("caps-test")

    @m.resource("res://greeting")
    def greeting() -> str:
        return "hello from resource"

    @m.prompt()
    def review(code: str) -> str:
        return f"请审查：{code}"

    c = MCPClient(name="t", server=m)
    c.start()
    try:
        assert len(c.list_resources()) == 1
        assert c.read_resource("res://greeting") == "hello from resource"
        assert "review" in {p.get("name") for p in c.list_prompts()}
        got = c.get_prompt("review", {"code": "x=1"})
        assert "请审查：x=1" in got
        assert got.startswith("user:")
    finally:
        c.stop()


def test_stop_cancels_inflight_call():
    """终止对话时有工具在跑：调用方必须立刻拿到 MCPError，不能等满 300s。"""
    import asyncio
    import threading
    import time

    m = FastMCP("inflight-test")

    @m.tool()
    async def slow(secs: float = 1.0) -> str:
        await asyncio.sleep(secs)      # 模拟真实 RPC 的异步 I/O
        return "done"

    c = MCPClient(name="t", server=m, timeout=10)
    c.start()
    holder: dict = {}

    def call():
        t0 = time.perf_counter()
        try:
            holder["result"] = c.call_tool("slow", {"secs": 3.0})
        except MCPError as e:
            holder["error"] = str(e)
        holder["elapsed"] = time.perf_counter() - t0

    th = threading.Thread(target=call, daemon=True)
    th.start()
    # 等请求确实被 session 接收，避免测到「还没提交就 stop」的假阳性
    for _ in range(200):
        if c._session is not None and c._session._inflight:
            break
        time.sleep(0.01)
    t0 = time.perf_counter()
    c.stop()
    stop_dt = time.perf_counter() - t0
    th.join(timeout=8)
    assert not th.is_alive(), "调用线程挂死"
    assert "error" in holder and "取消" in holder["error"], holder
    assert holder["elapsed"] < 2.0, f"调用方等了 {holder['elapsed']:.2f}s"
    assert stop_dt < 5.0, f"stop() 阻塞 {stop_dt:.2f}s"


def test_stdio_subprocess_roundtrip(tmp_path):
    """真实 stdio 子进程：最小 env 下握手、枚举、调用并回收。"""
    import sys

    script = tmp_path / "stdio_server.py"
    script.write_text(
        "from fastmcp import FastMCP\n"
        "mcp = FastMCP('stdio-test')\n"
        "@mcp.tool()\n"
        "def echo(msg: str) -> str:\n"
        "    return 'stdio:' + msg\n"
        "if __name__ == '__main__':\n"
        "    mcp.run(transport='stdio')\n",
        encoding="utf-8",
    )
    c = MCPClient(name="stdio", command=sys.executable,
                  args=[str(script)], timeout=20)
    c.start()
    try:
        assert "echo" in {d["name"] for d in c.list_tools()}
        assert c.call_tool("echo", {"msg": "hi"}) == "stdio:hi"
        assert c.server_info.get("name") == "stdio-test"
    finally:
        c.stop()
