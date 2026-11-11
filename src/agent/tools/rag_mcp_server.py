"""RAG 能力封装为 MCP 服务（2025 streamable-HTTP 规范）。

把 base-agent 已有的 RAG 引擎能力以 MCP 工具形式对外暴露，供其它大模型应用
对接使用。复用应用内已加载的 ``RAGEngine``（与桌面端知识库同一份索引/模型，
避免双份 ~600MB 向量/模型内存）。

工具清单：
  * RAG 检索/文档管理（7）：ingest_document / delete_document /
    forget_context / list_document / extract_metadata /
    hybrid_search_with_rerank / get_document；
  * 知识图谱（9，rag/kg/）：ingest_knowledge（按文档建图）/ add_entity /
    add_relationship / get_entity / delete_entity / get_path（bfs/dfs/
    dijkstra 寻路）/ get_neighbors / get_subgraph（BFS 深度 + 关系类型过滤）/
    search_graph（local/global/hybrid/mix 四种检索策略）。

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
        def ingest_document(doc_path: str) -> str:
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

        # ------------------------------------------------------------------
        # 知识图谱工具（rag/kg/）：构建、手工增删、图查询、语义检索
        # ------------------------------------------------------------------

        @mcp.tool()
        def ingest_knowledge(doc_name: str = "") -> str:
            """知识图谱构建：对指定文档（或知识库全部文档）做实体/关系抽取并建图。

            参数 doc_name: 文档名或路径子串；为空则对知识库全部切片构建。
            返回: JSON 统计（切片/实体/关系/向量数、图规模、耗时）。
            说明: 抽取方案由应用配置 rag_kg_extractor 决定（llm_raw/llm_pydantic
            需要 LLM 辅助模型已加载；gliner 为本地 NER）。
            """
            engine = _get_engine()
            if engine is None:
                return _err("RAG 引擎未就绪（请先在应用内启用知识库）")
            try:
                stats = engine.kg_ingest(doc_name=doc_name)
                return _ok({"ingest": stats})
            except Exception as e:  # noqa: BLE001
                return _err(f"知识图谱构建失败: {e}")

        @mcp.tool()
        def add_entity(name: str, type: str = "UNKNOWN",
                       description: str = "") -> str:
            """图谱加实体：写入实体节点（同名/同义写法自动合并到已有实体）。

            参数 name: 实体名；type: 实体类型（Person/Organization/Location/
            Event/Concept/Method/Content/Data/Artifact/NaturalObject/UNKNOWN）；
            description: 实体描述。
            """
            engine = _get_engine()
            if engine is None:
                return _err("RAG 引擎未就绪")
            try:
                result = engine.kg_add_entity(name, etype=type, description=description)
                return _ok(result)
            except Exception as e:  # noqa: BLE001
                return _err(f"添加实体失败: {e}")

        @mcp.tool()
        def add_relationship(source: str, target: str, relation: str,
                             description: str = "") -> str:
            """图谱加关系：在两个实体间建立（或聚合到已有）无向关系。

            参数 source/target: 两端实体名（不存在会自动建 UNKNOWN 实体）；
            relation: 关系类型/关键词；description: 关系描述。
            """
            engine = _get_engine()
            if engine is None:
                return _err("RAG 引擎未就绪")
            try:
                result = engine.kg_add_relationship(
                    source, target, relation, description=description)
                return _ok(result)
            except Exception as e:  # noqa: BLE001
                return _err(f"添加关系失败: {e}")

        @mcp.tool()
        def get_entity(entity: str) -> str:
            """实体详情：返回单个节点的属性、全部关联边（含方向）与别名。

            参数 entity: 实体 id / 名称 / 别名（任意一种写法）。
            """
            engine = _get_engine()
            if engine is None:
                return _err("RAG 引擎未就绪")
            try:
                return _ok(engine.kg_get_entity(entity))
            except KeyError as e:
                return _err(str(e).strip("'\""))
            except Exception as e:  # noqa: BLE001
                return _err(f"查询实体失败: {e}")

        @mcp.tool()
        def delete_entity(entity: str) -> str:
            """图谱删实体：删除节点及其全部关联边，并重建图快照。

            参数 entity: 实体 id / 名称 / 别名。
            """
            engine = _get_engine()
            if engine is None:
                return _err("RAG 引擎未就绪")
            try:
                return _ok(engine.kg_delete_entity(entity))
            except KeyError as e:
                return _err(str(e).strip("'\""))
            except Exception as e:  # noqa: BLE001
                return _err(f"删除实体失败: {e}")

        @mcp.tool()
        def get_path(entity_a: str, entity_b: str, algorithm: str = "bfs") -> str:
            """图谱寻路：返回两实体间路径上的全部中间节点与边的信息。

            参数 entity_a/entity_b: 两端实体；algorithm: 寻路算法，可选
            bfs（最少跳数，默认）/ dfs（深度优先）/ dijkstra（加权最短路，
            优先走关系证据最强的边）。
            """
            engine = _get_engine()
            if engine is None:
                return _err("RAG 引擎未就绪")
            try:
                return _ok(engine.kg_get_path(entity_a, entity_b, algorithm))
            except KeyError as e:
                return _err(str(e).strip("'\""))
            except Exception as e:  # noqa: BLE001
                return _err(f"寻路失败: {e}")

        @mcp.tool()
        def get_neighbors(entity: str) -> str:
            """一跳邻居：返回实体的全部邻边与相邻实体（含经由关系与权重）。

            参数 entity: 实体 id / 名称 / 别名。
            """
            engine = _get_engine()
            if engine is None:
                return _err("RAG 引擎未就绪")
            try:
                return _ok(engine.kg_get_neighbors(entity))
            except KeyError as e:
                return _err(str(e).strip("'\""))
            except Exception as e:  # noqa: BLE001
                return _err(f"查询邻居失败: {e}")

        @mcp.tool()
        def get_subgraph(root: str, depth: int = 1, edge_types: str = "") -> str:
            """子图提取：从 root 出发按 BFS 深度提取局部子图。

            参数 root: 根实体；depth: BFS 深度（默认 1 跳）；
            edge_types: 逗号分隔的关系类型/关键词子串过滤（如
            "创始人,领导"），为空不过滤。
            返回: JSON，含 nodes（附 depth 跳数）与 edges 列表。
            """
            engine = _get_engine()
            if engine is None:
                return _err("RAG 引擎未就绪")
            try:
                types = [t.strip() for t in edge_types.split(",") if t.strip()] \
                    if edge_types else None
                return _ok(engine.kg_get_subgraph(root, depth=depth,
                                                  edge_types=types))
            except KeyError as e:
                return _err(str(e).strip("'\""))
            except Exception as e:  # noqa: BLE001
                return _err(f"提取子图失败: {e}")

        @mcp.tool()
        def search_graph(query: str, mode: str = "hybrid") -> str:
            """图谱语义检索：按指定策略在知识图谱上检索。

            参数 query: 查询语句；mode: 检索策略——
            local（命中实体拉邻边与关联实体，适合问具体细节）/
            global（命中关系反查两端实体与度数，适合跨文档宏观主题/总结）/
            hybrid（实体+关系两路交织合并，细节与全貌兼顾）/ mix（实体+关系+
            原文 chunk 全做，原始片段兜底）。
            """
            engine = _get_engine()
            if engine is None:
                return _err("RAG 引擎未就绪")
            try:
                results = engine.kg_search_graph(query, mode=mode)
                return _ok({"mode": mode, "results": results,
                            "count": len(results)})
            except Exception as e:  # noqa: BLE001
                return _err(f"图谱检索失败: {e}")

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

        # -- 知识图谱桩（monkey 图谱：苹果-乔布斯-沃兹 三角） ------------

        _MONKEY_NODES = {
            "苹果": {"name_key": "苹果", "name": "苹果", "type": "Organization",
                     "description": "科技公司", "degree": 2},
            "乔布斯": {"name_key": "乔布斯", "name": "乔布斯", "type": "Person",
                       "description": "创始人", "degree": 2},
            "沃兹尼亚克": {"name_key": "沃兹尼亚克", "name": "沃兹尼亚克",
                           "type": "Person", "description": "联合创始人", "degree": 2},
        }
        _MONKEY_EDGES = [
            {"pair_key": "乔布斯\x1f苹果", "src": "苹果", "dst": "乔布斯",
             "src_name": "苹果", "dst_name": "乔布斯", "relation_type": "创始人",
             "keywords": "创始人", "description": "乔布斯创办苹果",
             "weight": 2, "from_key": "苹果", "to_key": "乔布斯"},
            {"pair_key": "乔布斯\x1f苹果", "src": "苹果", "dst": "乔布斯",
             "src_name": "苹果", "dst_name": "乔布斯", "relation_type": "创始人",
             "keywords": "创始人", "description": "乔布斯创办苹果",
             "weight": 2, "from_key": "乔布斯", "to_key": "苹果"},
            {"pair_key": "沃兹尼亚克\x1f苹果", "src": "苹果", "dst": "沃兹尼亚克",
             "src_name": "苹果", "dst_name": "沃兹尼亚克", "relation_type": "联合创始人",
             "keywords": "联合创始人", "description": "沃兹共同创办苹果",
             "weight": 1, "from_key": "苹果", "to_key": "沃兹尼亚克"},
            {"pair_key": "沃兹尼亚克\x1f苹果", "src": "苹果", "dst": "沃兹尼亚克",
             "src_name": "苹果", "dst_name": "沃兹尼亚克", "relation_type": "联合创始人",
             "keywords": "联合创始人", "description": "沃兹共同创办苹果",
             "weight": 1, "from_key": "沃兹尼亚克", "to_key": "苹果"},
        ]

        def kg_ingest(self, doc_name=""):
            return {"chunks": 2, "entities": 3, "relationships": 2, "vectors": 8,
                    "graph": {"nodes": 3, "edges": 2}, "monkey": True}

        def kg_add_entity(self, name, etype="UNKNOWN", description=""):
            return {"name": name, "created": True}

        def kg_add_relationship(self, source, target, relation, description=""):
            return {"id": "mcp-rid", "source": source, "target": target,
                    "relation": relation, "created": True}

        def kg_get_entity(self, entity):
            node = self._MONKEY_NODES.get(entity)
            if node is None:
                raise KeyError(f"实体不存在: {entity}")
            return {**node, "relationships": self._MONKEY_EDGES[:2],
                    "aliases": ["Apple"], "monkey": True}

        def kg_delete_entity(self, entity):
            if entity not in self._MONKEY_NODES:
                raise KeyError(f"实体不存在: {entity}")
            return {"deleted": entity, "entities": 1, "relationships": 2,
                    "vectors": 5}

        def kg_get_path(self, entity_a, entity_b, algorithm="bfs"):
            for e in (entity_a, entity_b):
                if e not in self._MONKEY_NODES:
                    raise KeyError(f"实体不存在: {e}")
            return {"found": True, "algorithm": algorithm, "length": 1,
                    "nodes": [self._MONKEY_NODES[entity_a],
                              self._MONKEY_NODES[entity_b]],
                    "edges": [self._MONKEY_EDGES[0]]}

        def kg_get_neighbors(self, entity):
            node = self._MONKEY_NODES.get(entity)
            if node is None:
                raise KeyError(f"实体不存在: {entity}")
            others = [n for n in self._MONKEY_NODES.values() if n["name"] != entity]
            return {"entity": entity, "name_key": node["name_key"], "degree": 2,
                    "neighbors": [{"name_key": n["name_key"], "name": n["name"],
                                   "type": n["type"], "description": n["description"],
                                   "via": {"relation_type": "关联",
                                           "keywords": "关联", "weight": 1}}
                                  for n in others]}

        def kg_get_subgraph(self, root, depth=1, edge_types=None):
            node = self._MONKEY_NODES.get(root)
            if node is None:
                raise KeyError(f"实体不存在: {root}")
            return {"root": node["name_key"], "depth": depth,
                    "nodes": [{**n, "depth": 0 if n["name"] == root else 1}
                              for n in self._MONKEY_NODES.values()],
                    "edges": self._MONKEY_EDGES}

        def kg_search_graph(self, query, mode="hybrid"):
            if mode not in ("local", "global", "hybrid", "mix"):
                raise ValueError(f"不支持的检索模式: {mode}")
            return [{"kind": "entity", "name": "苹果", "monkey": True},
                    {"kind": "relationship", "keywords": "创始人", "monkey": True}]

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
