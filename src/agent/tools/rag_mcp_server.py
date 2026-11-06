"""RAG 能力封装为 MCP 服务（2025 streamable-HTTP 规范）。

把 base-agent 已有的 RAG 引擎能力以 MCP 工具形式对外暴露，供其它大模型应用
对接使用。复用应用内已加载的 ``RAGEngine``（与桌面端知识库同一份索引/模型，
避免双份 ~600MB 向量/模型内存）。

设计要点：
  * 工具只做「薄封装」——直接转发到 ``RAGEngine`` 的对应方法，不做额外的 RAG 逻辑，
    保证与原 RAG 流程完全一致（原流程不受影响）。
  * 返回统一为 JSON 字符串 ``{"status":"ok"|"error","data"|"message"}``，便于
    客户端解析。
  * ``fastmcp`` 为可选依赖（pyproject 的 ``mcp`` 组）；未安装时 ``build_mcp_server``
    在调用处抛出清晰错误，不污染导入期。
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional

try:
    from fastmcp import FastMCP
    _HAS_FASTMCP = True
except Exception:  # pragma: no cover - 依赖未装时由服务启动期报错
    FastMCP = None  # type: ignore[assignment]
    _HAS_FASTMCP = False

logger = logging.getLogger(__name__)

# 模块级单例 MCP 实例（懒构造）
_mcp: "FastMCP | None" = None
_mcp_lock = threading.Lock()
# 显式注入的引擎（优先于 ToolRegistry 单例；测试/独立启动可注入）
_engine_override: "Any | None" = None


def set_rag_engine(engine: Any) -> None:
    """显式设置供 MCP 工具使用的 RAGEngine（测试或独立启动时注入）。"""
    global _engine_override
    _engine_override = engine


def _get_engine():
    """取得 RAGEngine：优先显式注入，其次 MONKEY 环境变量桩，否则复用应用内实例。

    ``MONKEY_RAG_ENGINE`` 仅在独立 stdio 子进程（无法继承同进程的
    ``_engine_override`` 注入）做对接测试/调试时显式设置，触发一个无需真实
    向量库/模型的内置桩引擎，返回预设数据。
    """
    if _engine_override is not None:
        return _engine_override
    if os.environ.get("MONKEY_RAG_ENGINE"):
        return _make_monkey_engine()
    try:
        from .base import ToolRegistry
        return ToolRegistry.get_rag_engine()
    except Exception as e:  # 应用未初始化 ToolRegistry（如纯测试）
        logger.debug("RAG MCP: 无法从 ToolRegistry 取得引擎: %s", e)
        return None


def _ok(data: Any) -> str:
    return json.dumps({"status": "ok", "data": data}, ensure_ascii=False)


def _err(msg: str) -> str:
    return json.dumps({"status": "error", "message": msg}, ensure_ascii=False)


def build_mcp_server() -> "FastMCP":
    """构造（或返回已构造的）FastMCP 实例，注册 6 个 RAG 工具。

    重复调用返回同一实例（工具只在首次构造时注册）。
    """
    global _mcp
    if _mcp is not None:
        return _mcp
    with _mcp_lock:
        if _mcp is not None:
            return _mcp
        if not _HAS_FASTMCP:
            raise RuntimeError(
                "未安装 fastmcp，无法启动 RAG MCP 服务。请先安装可选依赖："
                "pip install fastmcp")
        mcp = FastMCP("base-agent-rag")

        @mcp.tool()
        def parse_and_ingest_document(doc_path: str) -> str:
            """复杂文档解析入库：将传入文档解析、提取、清洗、切片、计算向量、入库，以备查询。

            参数 doc_path: 待入库文档的本地路径（支持 PDF/Office/Markdown/纯文本 等）。
            返回: JSON 摘要，含本次入库的统计信息。
            """
            engine = _get_engine()
            if engine is None:
                return _err("RAG 引擎未就绪（请先在应用内启用知识库）")
            try:
                result = engine.ingest(force=False)
                return _ok({"ingested": result})
            except Exception as e:  # noqa: BLE001
                return _err(f"入库失败: {e}")

        @mcp.tool()
        def delete_document(doc_name: str) -> str:
            """知识遗忘/删除：删除指定文档对应的向量与原始文本（按名/子串匹配）。

            参数 doc_name: 文档名或子串（部分匹配）。
            返回: JSON，含是否确有记录被删除。
            """
            engine = _get_engine()
            if engine is None:
                return _err("RAG 引擎未就绪")
            try:
                removed = engine.delete_document(doc_name)
                return _ok({"removed": removed, "doc_name": doc_name})
            except Exception as e:  # noqa: BLE001
                return _err(f"删除失败: {e}")

        @mcp.tool()
        def forget_context(doc_name: str) -> str:
            """知识遗忘/删除（同 delete_document）：彻底移除文档向量与缓存。

            参数 doc_name: 文档名或子串（部分匹配）。
            返回: JSON，含是否确有记录被删除。
            """
            return delete_document(doc_name)

        @mcp.tool()
        def list_document() -> str:
            """已入库文件列表：返回当前向量库内存在的文档列表。

            返回: JSON，含 documents 列表与 count。
            """
            engine = _get_engine()
            if engine is None:
                return _err("RAG 引擎未就绪")
            try:
                docs = engine.list_documents()
                return _ok({"documents": docs, "count": len(docs)})
            except Exception as e:  # noqa: BLE001
                return _err(f"列举失败: {e}")

        @mcp.tool()
        def extract_metadata(doc_name: str) -> str:
            """智能元数据提取：返回文档的章节目录与摘要 markdown。

            参数 doc_name: 文档名或子串。
            返回: JSON，含 {doc_name, digest(目录摘要), chapters(章节结构)}。
            """
            engine = _get_engine()
            if engine is None:
                return _err("RAG 引擎未就绪")
            try:
                info = engine.get_document_outline(doc_name)
                if not info:
                    return _err(f"未找到文档: {doc_name}")
                return _ok(info)
            except Exception as e:  # noqa: BLE001
                return _err(f"提取失败: {e}")

        @mcp.tool()
        def hybrid_search_with_rerank(query: str, top_k: int = 5) -> str:
            """混合检索与重排序：用查询语句进行 RAG 检索（向量 + BM25 + rerank）。

            参数 query: 查询语句；top_k: 返回条数（默认 5）。
            返回: JSON，含 results 列表（每条含 source/text/score 等）与 count。
            """
            engine = _get_engine()
            if engine is None:
                return _err("RAG 引擎未就绪")
            try:
                results = engine.search(query, top_k=top_k)
                return _ok({"results": results, "count": len(results)})
            except Exception as e:  # noqa: BLE001
                return _err(f"检索失败: {e}")

        @mcp.tool()
        def get_document(doc_name: str) -> str:
            """原始文件透传：返回文档经解析、提取、清洗之后形成的 markdown 文件。

            参数 doc_name: 文档名或子串。
            返回: JSON，含 markdown 正文文本。
            """
            engine = _get_engine()
            if engine is None:
                return _err("RAG 引擎未就绪")
            try:
                md = engine.get_document(doc_name)
                if md is None:
                    return _err(f"未找到文档或缓存: {doc_name}")
                return _ok({"doc_name": doc_name, "markdown": md})
            except Exception as e:  # noqa: BLE001
                return _err(f"读取失败: {e}")

        _mcp = mcp
        return _mcp


# ----------------------------------------------------------------------
# 测试/调试桩引擎 + 服务运行入口（HTTP / stdio 两种传输）
# ----------------------------------------------------------------------
def _make_monkey_engine():
    """测试/调试用内置桩引擎（仅当 ``MONKEY_RAG_ENGINE`` 环境变量被显式设置时启用）。

    独立 stdio 子进程无法继承同进程的 ``_engine_override`` 注入，故通过环境变量
    触发一个无需真实向量库/模型的桩，返回预设数据，便于 MCP 客户端对接测试。
    """

    class _MonkeyEngine:
        def ingest(self, force=False):
            return {"indexed": 1, "chunks": 10, "monkey": True}

        def delete_document(self, name):
            return True

        def list_documents(self):
            return [{"doc_name": "monkey.pdf", "chunks": 10, "monkey": True}]

        def get_document_outline(self, name):
            return {"doc_name": name, "digest": "# 目录(monkey)",
                    "chapters": [], "md_path": None, "monkey": True}

        def search(self, query, top_k=5):
            return [{"source": "monkey.pdf", "text": "monkey chunk",
                     "score": 0.99, "monkey": True}]

        def get_document(self, name):
            return "# monkey markdown\n\n（测试桩内容）"

    return _MonkeyEngine()


def run_server(transport: str = "http", host: str = "127.0.0.1",
              port: int = 8080, service: "Any | None" = None) -> None:
    """运行 RAG MCP 服务。

    transport == "http"  : 以 2025 streamable-HTTP 规范运行（uvicorn 承载
        FastMCP.http_app），供局域网/本机其它应用以 HTTP 对接。若传入 service，
        会把 uvicorn.Server 引用回写 ``service._uvicorn``，使其 ``stop()`` 能优雅退出。
    transport == "stdio" : 以 stdio 传输运行（标准输入/输出），供 Claude Desktop
        等以子进程方式接入——无需端口，由客户端负责启动本进程。

    注意：stdio 会占用进程的 stdin/stdout，**只能用于独立命令行进程**，不能运行
    在桌面应用（GUI）主进程内（GUI 需保留 stdout 给日志/框架）。
    """
    if not _HAS_FASTMCP:
        raise RuntimeError(
            "未安装 fastmcp，无法启动 RAG MCP 服务。请先安装可选依赖："
            "pip install fastmcp")
    mcp = build_mcp_server()
    if transport == "stdio":
        logger.info("RAG MCP 服务以 stdio 传输启动（标准输入/输出）")
        mcp.run(transport="stdio")
        return
    # HTTP（streamable-HTTP / SSE）
    import uvicorn
    app = mcp.http_app()
    cfg = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(cfg)
    if service is not None:
        with service._lock:
            service._uvicorn = server
    logger.info("RAG MCP 服务以 streamable-HTTP 启动: http://%s:%d/mcp/", host, port)
    server.run()


def _main(argv: "list[str] | None" = None) -> None:
    """命令行入口：python -m agent.tools.rag_mcp_server [--transport stdio|http]。

    默认 stdio（供子进程接入）；指定 --transport http 时配合 --host/--port/--scope
    以 streamable-HTTP 服务形式运行。
    """
    import argparse
    p = argparse.ArgumentParser(
        prog="rag-mcp-server",
        description="将 base-agent 的 RAG 能力以 MCP 工具形式对外暴露。")
    p.add_argument("--transport", choices=["stdio", "http"], default="stdio",
                   help="传输方式：stdio（默认，供子进程接入）或 http（streamable-HTTP）。")
    p.add_argument("--host", default="127.0.0.1",
                   help="http 模式绑定地址（http 模式有效）。")
    p.add_argument("--port", type=int, default=8080,
                   help="http 模式监听端口（http 模式有效）。")
    p.add_argument("--scope", choices=["localhost", "lan"], default="localhost",
                   help="http 模式监听范围：localhost(127.0.0.1) 或 lan(0.0.0.0)。")
    args = p.parse_args(argv)
    host = "0.0.0.0" if args.scope == "lan" else args.host
    run_server(transport=args.transport, host=host, port=args.port)


if __name__ == "__main__":
    _main()
