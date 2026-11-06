"""RAG MCP 服务对接测试。

验证两点：
  1. 原 RAG 流程不被破坏（本文件不改动任何 RAG 逻辑，仅做封装层测试）；
  2. 6 个 MCP 工具经 MCP 协议「客户端对接」可用——in-memory 客户端验证工具接线，
     真实 streamable-HTTP 客户端验证服务可对外暴露。

fastmcp 为可选依赖；未安装时用例自动 skip。
"""
import json
import time

import pytest

try:
    from fastmcp import Client  # noqa: F401
    _HAS_FASTMCP = True
except Exception:  # pragma: no cover
    _HAS_FASTMCP = False

from agent.tools.rag_mcp_server import (
    _HAS_FASTMCP as _SERVER_HAS_FASTMCP,
    build_mcp_server,
    set_rag_engine,
)


class StubEngine:
    """轻量假引擎，仅用于验证 MCP 封装层接线，不依赖真实 RAG 依赖。"""

    def ingest(self, force=False, progress_callback=None, log_callback=None):
        if progress_callback:
            progress_callback(0.5, "stub")
        return {"upserted": 1, "skipped": 0, "chunks": 3}

    def delete_document(self, name: str) -> bool:
        return True

    def list_documents(self):
        return ["report.pdf", "book.md"]

    def get_document_outline(self, name: str):
        return {
            "doc_name": name,
            "digest": "# 目录\n- 第一章\n- 第二章",
            "chapters": "[]",
            "md_path": "/tmp/stub.md",
        }

    def search(self, query: str, top_k: int = 5, source=None):
        return [{"source": "report.pdf", "text": "相关片段内容", "score": 0.92}]

    def get_document(self, name: str):
        return "# 完整文档 markdown\n正文……"


@pytest.fixture
def stub_engine():
    eng = StubEngine()
    set_rag_engine(eng)
    yield eng
    set_rag_engine(None)


def _text(result):
    """兼容 fastmcp 不同版本 call_tool 返回结构，取出首个文本块。"""
    if hasattr(result, "content"):
        content = result.content
    else:
        content, _ = result
    return content[0].text


def _parse(result):
    return json.loads(_text(result))


@pytest.mark.skipif(not _HAS_FASTMCP, reason="fastmcp 未安装")
def test_mcp_tools_inmemory(stub_engine):
    """in-memory MCP 客户端：6 个工具接线正确、返回结构符合约定。"""
    import asyncio

    mcp = build_mcp_server()

    async def run():
        client = Client(mcp)
        async with client:
            # 1. list_document
            d = _parse(await client.call_tool("list_document", {}))
            assert d["status"] == "ok"
            assert d["data"]["count"] == 2

            # 2. parse_and_ingest_document
            d = _parse(await client.call_tool(
                "parse_and_ingest_document", {"doc_path": "/tmp/x.pdf"}))
            assert d["status"] == "ok"
            assert d["data"]["ingested"]["upserted"] == 1

            # 3. delete_document
            d = _parse(await client.call_tool(
                "delete_document", {"doc_name": "report"}))
            assert d["status"] == "ok"
            assert d["data"]["removed"] is True

            # 4. forget_context（同 delete_document）
            d = _parse(await client.call_tool(
                "forget_context", {"doc_name": "report"}))
            assert d["status"] == "ok"

            # 5. extract_metadata
            d = _parse(await client.call_tool(
                "extract_metadata", {"doc_name": "book"}))
            assert d["status"] == "ok"
            assert "目录" in d["data"]["digest"]

            # 6. hybrid_search_with_rerank
            d = _parse(await client.call_tool(
                "hybrid_search_with_rerank", {"query": "测试", "top_k": 3}))
            assert d["status"] == "ok"
            assert d["data"]["count"] == 1

            # 7. get_document
            d = _parse(await client.call_tool(
                "get_document", {"doc_name": "book"}))
            assert d["status"] == "ok"
            assert d["data"]["markdown"].startswith("# 完整文档")

    asyncio.run(run())


@pytest.mark.skipif(not _HAS_FASTMCP, reason="fastmcp 未安装")
def test_mcp_service_http(stub_engine):
    """真实 streamable-HTTP：启动服务 + 远程客户端调用，验证对外可用。"""
    import asyncio

    from agent.tools.rag_mcp_service import RagMcpService

    svc = RagMcpService()
    port = 18099
    svc.start(scope="localhost", port=port)
    # 等待端口就绪
    deadline = time.time() + 8.0
    while time.time() < deadline and not svc.is_running:
        time.sleep(0.2)
    try:
        async def run():
            client = Client(f"http://127.0.0.1:{port}/mcp/")
            async with client:
                d = _parse(await client.call_tool("list_document", {}))
                assert d["status"] == "ok"
                assert d["data"]["count"] == 2

        asyncio.run(run())
    finally:
        svc.stop()
        time.sleep(0.6)
        assert not svc.is_running


@pytest.mark.skipif(not _HAS_FASTMCP, reason="fastmcp 未安装")
def test_mcp_service_stdio():
    """stdio 接入：以子进程 stdio 传输启动服务，验证 6 项能力（7 个工具名）对外可用。

    独立于应用进程（子进程不继承同进程注入的桩引擎），通过 MONKEY_RAG_ENGINE
    环境变量触发内置桩引擎，避免真实向量库依赖。
    """
    import asyncio
    import os
    import sys

    from fastmcp.client.transports import StdioTransport

    os.environ["MONKEY_RAG_ENGINE"] = "1"
    try:
        # 显式把当前环境（含 MONKEY_RAG_ENGINE）传给子进程：StdioTransport 不自动
        # 继承父进程 os.environ，需显式 env 才能让 stdio 子进程走内置桩引擎。
        transport = StdioTransport(
            command=sys.executable,
            args=["-m", "agent.tools.rag_mcp_server", "--transport", "stdio"],
            env={**os.environ},
        )

        async def run():
            client = Client(transport)
            async with client:
                tools = await client.list_tools()
                names = {t.name for t in tools}
                # 6 项能力 → 7 个工具名（delete_document 与 forget_context 同义）
                expected = {
                    "parse_and_ingest_document", "delete_document", "forget_context",
                    "list_document", "extract_metadata",
                    "hybrid_search_with_rerank", "get_document",
                }
                assert names == expected, names
                # 调用若干工具走通（monkey 桩）
                d = _parse(await client.call_tool("list_document", {}))
                assert d["status"] == "ok"
                assert d["data"]["count"] == 1
                d = _parse(await client.call_tool(
                    "hybrid_search_with_rerank", {"query": "测试", "top_k": 3}))
                assert d["status"] == "ok"
                assert d["data"]["count"] == 1
                d = _parse(await client.call_tool("get_document", {"doc_name": "x"}))
                assert d["status"] == "ok"
                assert "monkey" in d["data"]["markdown"]

        asyncio.run(run())
    finally:
        os.environ.pop("MONKEY_RAG_ENGINE", None)
