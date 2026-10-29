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

# 工具输出体积兜底（与 mcp_client / tools.memory 同策略）：引擎侧未对内部分片
# 截断时，一次检索就能把上下文撑爆。
_MAX_TOOL_OUTPUT_CHARS = 200_000


def _cap_output(text: str) -> str:
    if len(text) <= _MAX_TOOL_OUTPUT_CHARS:
        return text
    return (text[:_MAX_TOOL_OUTPUT_CHARS]
            + f"\n… [输出超过 {_MAX_TOOL_OUTPUT_CHARS} 字符，已截断]")

if TYPE_CHECKING:
    from ..rag.engine import RAGEngine


class RagSearchTool(Tool):
    """Search the local knowledge base for relevant information."""

    name = "rag_search"
    description = (
        "在本地知识库中检索**资料内容**。仅当问题依赖用户的资料时使用"
        "（如查某主题的原文、某本书的细节）；「知识库里有什么/你能做什么」"
        "这类问题不适用——前者用 rag_outline（不带参数）列文档，后者直接回答。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "检索词：提取资料内容相关的概念/关键词短语，用用户原始语言，"
                    "勿翻译或音译。不要传用户的问句原文（如「知识库里有什么」），"
                    "那检索不到任何东西。"
                ),
            },
            "top_k": {
                "type": "integer",
                "description": "返回结果数，默认 5，范围 1-10。",
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
        # 保存 config 引用：默认 top_k 与辅助模型精简阈值都做成可配置项，
        # 运行时从这里读（bind 在工具注册时被调用，早于任何 run）。
        self._config = config

    def run(self, query: str = "", top_k: Any = None, source: str = "") -> ToolResult:
        # 模型可能传 null（JSON null → None），None.strip() 会抛异常并被
        # registry 兜底误报成 "Invalid arguments"，模型会原样重试。这里统一
        # 归一化成空串再判断。
        query = (query or "").strip()
        if not query:
            return ToolResult(success=False, error="请提供搜索查询词")

        # 默认 top_k 从配置读取（rag_search_top_k，默认 5）；模型未传/传 null
        # 时走配置默认，传了非法值也回退到配置默认。_config 由 bind() 注入，
        # 直接构造（测试/旁路）时可能缺失，一律走 getattr 默认值。
        cfg = getattr(self, "_config", None)
        default_top_k = int(getattr(cfg, "rag_search_top_k", 5) or 5)
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = default_top_k
        # 限制 top_k 上限，防止模型传入过大值导致返回海量内容撑爆上下文窗口。
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
            condensed = ""
            if chunks and aux is not None:
                total_chars = sum(len(r.get("text") or "") for r in chunks)
                min_chars = int(getattr(cfg, "rag_condense_min_chars", 4000) or 4000)
                max_in = int(getattr(cfg, "rag_condense_max_chars", 12000) or 12000)
                # 仅当检索总量落在 [min, max] 区间才调用辅助模型归纳精简：
                #   * 低于 min → 原样注入。内容本来就少，精简只会丢细节，
                #     还白耗一次辅助模型调用（慢）；
                #   * 高于 max → 也不送辅助模型（超出其舒适上下文，效果差），
                #     交由下方格式化路径按相关度截断。
                if total_chars < min_chars:
                    logger.info(
                        "  ├─ 检索总量 %d 字 < 精简下限 %d 字 → 跳过辅助模型归纳（原样注入）",
                        total_chars, min_chars,
                    )
                elif total_chars > max_in:
                    logger.info(
                        "  ├─ 检索总量 %d 字 > 精简上限 %d 字 → 跳过辅助模型归纳（格式化截断）",
                        total_chars, max_in,
                    )
                else:
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
            # 回退：辅助模型不可用 / 总量不在精简区间 / 精简失败时，维持原格式化行为。
            results = self._engine.search_formatted(
                query, top_k=top_k, source=source or None,
            )
            return ToolResult(success=True, output=_cap_output(results))
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
        "知识库文档总览。用户问「知识库里有什么/有哪些资料/文档」时，"
        "**不带 book_name** 调用即可列出全部文档；用户提到具体书名时带上书名"
        "调用，先掌握该文档的目录结构，再用 rag_search 检索章节细节。"
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
            # 千级文档全量列出会挤爆上下文：截断并提示总数。
            _MAX_DOCS = 100
            lines = [f"知识库中已建立缩略版本的文档（共 {len(docs)} 个）："]
            lines.extend(f"  - {d}" for d in docs[:_MAX_DOCS])
            if len(docs) > _MAX_DOCS:
                lines.append(f"  … （其余 {len(docs) - _MAX_DOCS} 个省略）")
            return ToolResult(success=True, output=_cap_output("\n".join(lines)))

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
            # 附带后台闲时生成的全书摘要（若有）：目录结构 + 全书摘要一起给，
            # 模型无需逐章 rag_search 即可把握全书脉络。
            book_summary = ""
            try:
                book_summary = self._engine.get_book_summary(
                    doc.get("doc_name", book_name))
            except Exception:
                book_summary = ""
            if book_summary:
                digest = (f"{digest}\n\n【全书摘要】\n{book_summary}")
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

    def __init__(self, engine: "RAGEngine", registry: "ToolRegistry" = None):
        self._engine = engine
        self._registry = registry

    def bind(self, config: Any, registry: ToolRegistry) -> None:
        self._registry = registry

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

        # 取消桥：ingest 是同步全量管线（大库数分钟），旧实现在 agent 线程里
        # 跑且循环内无取消检查——用户点「终止对话」也要等它整个跑完。这里用
        # progress_callback 轮询「终止对话」标志，命中即 engine.cancel()，
        # 引擎在文件/批次边界协作式退出（已入库数据不丢、manifest 一致）。
        def _progress(done: int, total: int, name: str) -> None:
            cb = getattr(self._registry, "callbacks", None) if self._registry else None
            fn = getattr(cb, "is_cancelled", None)
            if callable(fn) and fn():
                self._engine.cancel()

        # 仅当引擎签名接受 progress_callback 时才传（测试替身/旧版引擎可能
        # 不接受该参数，强传会 TypeError）。
        import inspect as _inspect
        _ingest_kwargs: Dict[str, Any] = {"force": force}
        try:
            if "progress_callback" in _inspect.signature(self._engine.ingest).parameters:
                _ingest_kwargs["progress_callback"] = _progress
        except (TypeError, ValueError):
            pass

        try:
            stats = self._engine.ingest(**_ingest_kwargs)
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