"""RAG tool — allows the agent to search the local knowledge base."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, TYPE_CHECKING

from .base import Tool, ToolResult, ToolRegistry
from ..rag.postprocess import (
    condense_retrieved_chunks,
    build_condensed_output,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..rag.engine import RAGEngine


class RagSearchTool(Tool):
    """Search the local knowledge base for relevant information."""

    name = "rag_search"
    description = "搜索本地知识库。查询文档内容、参考资料时使用；可用 source 限定在某本书/文档内检索。"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "检索关键词/短语。用用户原始语言，提取问题核心概念，勿翻译或音译。",
            },
            "top_k": {
                "type": "integer",
                "description": "返回结果数，默认 3，范围 1-10。",
            },
            "source": {
                "type": "string",
                "description": "可选。限定检索的书名/文档名关键词。",
            },
        },
        "required": ["query"],
    }
    destructive = False

    def __init__(self, engine: "RAGEngine", aux_llm_getter: Any = None):
        self._engine = engine
        # 辅助模型客户端惰性获取器：由 Agent 在运行时注入（aux 可能被卸载/重建），
        # 工具自身不该持有模型引用。为 None 或取到 None 时跳过精简，回退原始结果。
        self._aux_llm_getter = aux_llm_getter

    def bind(self, config: Any, registry: ToolRegistry) -> None:
        pass

    def run(self, query: str = "", top_k: int = 3, source: str = "") -> ToolResult:
        # 模型可能传 null（JSON null → None），None.strip() 会抛异常并被
        # registry 兜底误报成 "Invalid arguments"，模型会原样重试。这里统一
        # 归一化成空串再判断。
        query = (query or "").strip()
        if not query:
            return ToolResult(success=False, error="请提供搜索查询词")

        # 限制 top_k 上限，防止模型传入过大值导致返回海量内容撑爆上下文窗口。
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = 3
        top_k = max(1, min(top_k, 10))

        source = (source or "").strip()
        logger.info(
            "🔍 RAG 工具调用: rag_search(query=%r, top_k=%d, source=%r)",
            query[:80], top_k, source[:80],
        )
        try:
            # 先取重排后的原始分片（含完整 text，未截断），供辅助模型通读归纳。
            chunks = self._engine.search(query, top_k=top_k, source=source or None)
            aux = self._aux_llm_getter() if self._aux_llm_getter else None
            if chunks and aux is not None:
                try:
                    condensed = condense_retrieved_chunks(query, chunks, aux, max_chars=2000)
                except Exception as e:  # 精简异常不应让检索整体失败
                    logger.warning("RAG 辅助模型精简异常，回退原始结果: %s", e)
                    condensed = ""
                if condensed:
                    output = build_condensed_output(query, chunks, condensed)
                    logger.info(
                        "✅ RAG 检索 %d 条 → 辅助模型精简为 %d 字注入主模型",
                        len(chunks), len(condensed),
                    )
                    return ToolResult(success=True, output=output)
            # 回退：辅助模型不可用 / 精简失败 / 结果为空时，维持原格式化行为。
            results = self._engine.search_formatted(
                query, top_k=top_k, source=source or None,
            )
            return ToolResult(success=True, output=results)
        except Exception as e:
            logger.error("❌ RAG 搜索失败: %s", e, exc_info=True)
            return ToolResult(success=False, error=f"知识库搜索失败: {e}")


class RagOutlineTool(Tool):
    """Return a document's condensed outline (目录结构) for Meta-context RAG.

    当用户提到具体书名/文档名时，先调用本工具获取该文档的完整目录结构
    （标题层级），让模型先掌握全文结构，再规划需要深入检索哪些章节的细节。
    """

    name = "rag_outline"
    description = (
        "获取某文档/书籍的目录结构（章节标题）。用户提到具体书名时先调用，"
        "了解全文结构后再用 rag_search 检索章节细节。不带 book_name 时列出所有文档。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "book_name": {
                "type": "string",
                "description": "书名/文档名关键词（支持部分匹配）",
            },
        },
        "required": [],
    }
    destructive = False

    def __init__(self, engine: "RAGEngine"):
        self._engine = engine

    def bind(self, config: Any, registry: ToolRegistry) -> None:
        pass

    def run(self, book_name: str = "", **kwargs) -> ToolResult:
        book_name = (book_name or "").strip()
        if not book_name:
            # 未指定书名时，列出所有已建立缩略版本的文档。
            docs = self._engine.list_documents()
            if not docs:
                return ToolResult(success=False, error="知识库中暂无已建立缩略版本的文档")
            return ToolResult(
                success=True,
                output="知识库中已建立缩略版本的文档：\n" + "\n".join(f"  - {d}" for d in docs),
            )

        logger.info("📖 RAG 工具调用: rag_outline(book_name=%r)", book_name[:80])
        try:
            doc = self._engine.get_document_outline(book_name)
            if doc is None:
                return ToolResult(
                    success=False,
                    error=f"未找到文档「{book_name}」的缩略版本。可先调用 rag_outline（不带参数）查看可用文档列表。",
                )
            digest = doc.get("digest", "")
            if not digest:
                return ToolResult(success=False, error=f"文档「{doc.get('doc_name', book_name)}」暂无缩略版本")
            return ToolResult(success=True, output=digest)
        except Exception as e:
            logger.error("❌ RAG 缩略版本获取失败: %s", e, exc_info=True)
            return ToolResult(success=False, error=f"获取文档缩略版本失败: {e}")


class RagStatusTool(Tool):
    """Check the status of the local knowledge base."""

    name = "rag_status"
    description = (
        "查看本地知识库的状态，包括已索引的文档数量、来源文件列表等。"
    )
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    destructive = False

    def __init__(self, engine: "RAGEngine"):
        self._engine = engine

    def bind(self, config: Any, registry: ToolRegistry) -> None:
        pass

    def run(self, **kwargs) -> ToolResult:
        try:
            status = self._engine.status()
            lines = [
                f"知识库状态:",
                f"  知识库目录: {status['knowledge_base'] or '（未设置）'}",
                f"  RAG 目录: {status['rag_dir']}",
                f"  已存储切片数: {status['chunks_stored']}",
                f"  已索引文件数: {len(status['sources'])}",
            ]
            if status["sources"]:
                sources = status["sources"]
                # 大知识库可能索引了成百上千个文件，全量列出既挤占上下文也
                # 没有决策价值——截断并提示总数。
                _MAX_SOURCES = 50
                lines.append(f"  已索引文件: 共 {len(sources)} 个")
                for s in sources[:_MAX_SOURCES]:
                    lines.append(f"    - {os.path.basename(s)}")
                if len(sources) > _MAX_SOURCES:
                    lines.append(f"    … （其余 {len(sources) - _MAX_SOURCES} 个省略）")
            return ToolResult(success=True, output="\n".join(lines))
        except Exception as e:
            return ToolResult(success=False, error=f"获取知识库状态失败: {e}")


class RagIngestTool(Tool):
    """Re-ingest the knowledge base directory."""

    name = "rag_ingest"
    description = (
        "重新索引知识库目录中的所有文档。当知识库中添加了新文件后使用此工具。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "force": {
                "type": "boolean",
                "description": "是否强制重新处理所有文件（默认 false，跳过已处理的文件）",
            },
        },
        "required": [],
    }
    destructive = False

    def __init__(self, engine: "RAGEngine"):
        self._engine = engine

    def bind(self, config: Any, registry: ToolRegistry) -> None:
        pass

    def run(self, force: bool = False, **kwargs) -> ToolResult:
        # 严格布尔解析：模型（或 JSON 序列化）可能把 force 传成字符串
        # "false"/"0"——旧实现直接用，非空字符串都是真值 → 意外触发全量重建。
        if isinstance(force, str):
            force = force.strip().lower() in ("1", "true", "yes", "on")
        elif force is None:
            force = False
        else:
            force = bool(force)
        logger.info("📥 RAG 工具调用: rag_ingest(force=%s)", force)
        try:
            stats = self._engine.ingest(force=force)
            if "error" in stats:
                logger.warning("rag_ingest returned error: %s", stats["error"])
                return ToolResult(success=False, error=stats["error"])
            lines = [
                "知识库索引完成:",
                f"  扫描文件数: {stats.get('files_found', 0)}",
                f"  新提取文件: {stats.get('files_extracted', 0)}",
                f"  跳过(已处理): {stats.get('files_skipped', 0)}",
                f"  总字符数: {stats.get('total_chars', 0)}",
                f"  生成切片数: {stats.get('chunks', 0)}",
            ]
            errors = stats.get("errors", [])
            if errors:
                lines.append(f"  错误: {len(errors)} 个")
            return ToolResult(success=True, output="\n".join(lines))
        except Exception as e:
            logger.error("rag_ingest failed: %s", e, exc_info=True)
            return ToolResult(success=False, error=f"知识库索引失败: {e}")