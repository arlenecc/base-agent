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

    # -- 知识图谱工具桩（仅验证 MCP 封装层接线） -------------------------

    def kg_ingest(self, doc_name=""):
        return {"chunks": 3, "entities": 2, "relationships": 1,
                "vectors": 4, "graph": {"nodes": 2, "edges": 1}}

    def kg_add_entity(self, name, etype="UNKNOWN", description=""):
        return {"name": name, "created": True}

    def kg_add_relationship(self, source, target, relation, description=""):
        return {"id": "rid", "source": source, "target": target,
                "relation": relation, "created": True}

    def kg_get_entity(self, entity):
        if entity == "不存在":
            raise KeyError(f"实体不存在: {entity}")
        return {"name": entity, "type": "Person", "relationships": [],
                "aliases": []}

    def kg_delete_entity(self, entity):
        return {"deleted": entity, "entities": 1, "relationships": 2,
                "vectors": 3}

    def kg_get_path(self, entity_a, entity_b, algorithm="bfs"):
        if algorithm not in ("bfs", "dfs", "dijkstra"):
            raise ValueError(f"不支持的寻路算法: {algorithm}")
        return {"found": True, "algorithm": algorithm, "length": 1,
                "nodes": [{"name_key": entity_a}, {"name_key": entity_b}],
                "edges": [{"keywords": "相关"}]}

    def kg_get_neighbors(self, entity):
        return {"entity": entity, "name_key": entity, "degree": 1,
                "neighbors": [{"name_key": "苹果", "name": "苹果",
                               "type": "Organization", "description": "",
                               "via": {"relation_type": "关联",
                                       "keywords": "关联", "weight": 1}}]}

    def kg_get_subgraph(self, root, depth=1, edge_types=None):
        return {"root": root, "depth": depth,
                "nodes": [{"name_key": root, "depth": 0}], "edges": []}

    def kg_search_graph(self, query, mode="hybrid"):
        if mode not in ("local", "global", "hybrid", "mix"):
            raise ValueError(f"不支持的检索模式: {mode}")
        return [{"kind": "entity", "name": "苹果", "score": 0.9}]


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

            # 2. ingest_document
            d = _parse(await client.call_tool(
                "ingest_document", {"doc_path": "/tmp/x.pdf"}))
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

            # 8-16. 知识图谱 9 工具
            d = _parse(await client.call_tool("ingest_knowledge", {}))
            assert d["status"] == "ok"
            assert d["data"]["ingest"]["entities"] == 2
            d = _parse(await client.call_tool(
                "ingest_knowledge", {"doc_name": "book.md"}))
            assert d["status"] == "ok"
            d = _parse(await client.call_tool(
                "add_entity", {"name": "苹果", "type": "Organization"}))
            assert d["status"] == "ok" and d["data"]["created"] is True
            d = _parse(await client.call_tool(
                "add_relationship",
                {"source": "苹果", "target": "乔布斯", "relation": "创始人"}))
            assert d["status"] == "ok" and d["data"]["relation"] == "创始人"
            d = _parse(await client.call_tool("get_entity", {"entity": "苹果"}))
            assert d["status"] == "ok" and d["data"]["type"] == "Person"
            d = _parse(await client.call_tool("get_entity", {"entity": "不存在"}))
            assert d["status"] == "error"
            d = _parse(await client.call_tool(
                "get_path", {"entity_a": "苹果", "entity_b": "乔布斯",
                             "algorithm": "dijkstra"}))
            assert d["status"] == "ok" and d["data"]["found"] is True
            d = _parse(await client.call_tool(
                "get_path", {"entity_a": "苹果", "entity_b": "乔布斯",
                             "algorithm": "bogus"}))
            assert d["status"] == "error"
            d = _parse(await client.call_tool("get_neighbors", {"entity": "乔布斯"}))
            assert d["status"] == "ok" and d["data"]["degree"] == 1
            d = _parse(await client.call_tool(
                "get_subgraph", {"root": "苹果", "depth": 2,
                                 "edge_types": "创始人,领导"}))
            assert d["status"] == "ok" and d["data"]["root"] == "苹果"
            d = _parse(await client.call_tool("delete_entity", {"entity": "苹果"}))
            assert d["status"] == "ok" and d["data"]["entities"] == 1
            d = _parse(await client.call_tool(
                "search_graph", {"query": "创始人是谁", "mode": "mix"}))
            assert d["status"] == "ok" and d["data"]["count"] == 1
            d = _parse(await client.call_tool(
                "search_graph", {"query": "x", "mode": "bogus"}))
            assert d["status"] == "error"

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
        # src 布局：子进程需要 PYTHONPATH 找到 agent 包（pytest 里由 conftest
        # 或 rootdir 配置注入，裸子进程没有）。
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src_dir = os.path.join(repo_root, "src")
        env = {**os.environ}
        env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
        transport = StdioTransport(
            command=sys.executable,
            args=["-m", "agent.tools.rag_mcp_server", "--transport", "stdio"],
            env=env,
        )

        async def run():
            client = Client(transport)
            async with client:
                tools = await client.list_tools()
                names = {t.name for t in tools}
                # 6 项 RAG 能力（7 工具，delete_document/forget_context 同义）
                # + 9 个知识图谱工具
                expected = {
                    "ingest_document", "delete_document", "forget_context",
                    "list_document", "extract_metadata",
                    "hybrid_search_with_rerank", "get_document",
                    "ingest_knowledge", "add_entity", "add_relationship",
                    "get_entity", "delete_entity", "get_path", "get_neighbors",
                    "get_subgraph", "search_graph",
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
                # monkey 图谱桩走通：详情 / 邻居 / 寻路 / 子图 / 检索
                d = _parse(await client.call_tool("get_entity", {"entity": "苹果"}))
                assert d["status"] == "ok" and d["data"]["monkey"] is True
                d = _parse(await client.call_tool("get_neighbors", {"entity": "苹果"}))
                assert d["status"] == "ok" and d["data"]["degree"] == 2
                d = _parse(await client.call_tool(
                    "get_path", {"entity_a": "苹果", "entity_b": "乔布斯",
                                 "algorithm": "dfs"}))
                assert d["status"] == "ok" and d["data"]["found"] is True
                d = _parse(await client.call_tool("get_subgraph", {"root": "苹果"}))
                assert d["status"] == "ok" and len(d["data"]["nodes"]) == 3
                d = _parse(await client.call_tool(
                    "search_graph", {"query": "创始人", "mode": "local"}))
                assert d["status"] == "ok" and d["data"]["count"] == 2

        asyncio.run(run())
    finally:
        os.environ.pop("MONKEY_RAG_ENGINE", None)
