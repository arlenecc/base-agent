"""RAG tool — allows the agent to search the local knowledge base."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, TYPE_CHECKING

from .base import Tool, ToolResult, ToolRegistry, _FanoutTool
from ..core_opinion import simplify_for_scoring
from ..rag.postprocess import (
    condense_retrieved_chunks,
    build_condensed_output,
)

logger = logging.getLogger(__name__)

# 工具输出体积兜底（与 mcp_client / tools.memory 同策略）：引擎侧未对内部分片
# 截断时，一次检索就能把上下文撑爆。
_MAX_TOOL_OUTPUT_CHARS = 200_000
# 检索即展示模式下，单切片压缩后的硬上限（字符）。核心句抽取已把
# 长切片压到 800 字以内；这里只兜底无标点/单句超长文本。
_MAX_PRESENTATION_CHUNK_CHARS = 1500
# 知识图谱结果：实体/关系是摘要形态，体积可控，**始终**进上下文。
_GRAPH_KINDS = ("kg_entity", "kg_relation")
_ENGINE_RELEASED_ERROR = (
    "知识库引擎已释放（应用关闭或配置重建中），本次调用不可用；"
    "不要重试同一调用，等待新会话或重新同步知识库。"
)

_DEFAULT_SEARCH_DESCRIPTION = (
    "在本地知识库中检索**资料内容**。仅当问题依赖用户的资料时使用"
    "（如查某主题的原文、某本书的细节）；「知识库里有什么/你能做什么」"
    "这类问题不适用——前者用 rag_tools(op='outline') 列文档，后者直接回答。"
    "检索会把**所有召回切片**抽取核心观点句后注入上下文；完整原文仍以气泡"
    "展示给用户。请综合所有切片作答，不要声称看不到其余内容。"
)
_READING_SEARCH_DESCRIPTION = (
    "在本地知识库中检索**资料内容**（检索阅读模式）。仅当问题依赖用户的资料时"
    "使用；「知识库里有什么/你能做什么」这类问题不适用——前者用 "
    "rag_tools(op='outline') 列文档，后者直接回答。"
    "检索会把**所有召回切片**抽取核心观点句后注入上下文，请综合所有切片"
    "总结/回答；完整原文以可点击的「参考切片」气泡附在回复末尾。不要复述"
    "整片原文，也不要猜测或编造未给出的内容。"
)
_DEFAULT_RAG_TOOLS_DESCRIPTION = (
    "Unified local knowledge-base operations. op one of:\n"
    "- search(query, top_k?, source?): retrieve material content; every "
    "returned chunk is reduced to core opinion sentences and injected into "
    "your context; full text stays available to the user as reference chips "
    "(do not claim you cannot see the other chunks)\n"
    "- outline(book_name?): list all docs (no arg) or a document's outline\n"
    "- status(): knowledge-base status (chunks, sources)\n"
    "- ingest(force?): re-index the knowledge-base directory"
)
_READING_RAG_TOOLS_DESCRIPTION = (
    "Unified local knowledge-base operations (retrieval-reading mode). op one of:\n"
    "- search(query, top_k?, source?): retrieve long-document chunks. Every "
    "returned chunk is reduced to core opinion sentences and injected into "
    "your context for summarizing/answering; full text stays available to the "
    "user as clickable reference chips. Do not restate the whole chunk, and "
    "never guess or invent content that was not provided.\n"
    "- outline(book_name?): list all docs (no arg) or a document's outline\n"
    "- status(): knowledge-base status (chunks, sources)\n"
    "- ingest(force?): re-index the knowledge-base directory"
)


def _reading_enabled(engine: Any, config: Any) -> bool:
    """阅读模式以引擎实际管线为准，config 仅在引擎未提供属性时兜底。

    兜底也按 rag_mode 判断：极速检索模式 = 阅读模式；旧的独立开关字段不再
    作为是否进入阅读管线的依据。
    """
    mode = getattr(engine, "reading_mode", None)
    if mode is not None:
        return bool(mode)
    return (str(getattr(config, "rag_mode", "vector") or "vector")
            .strip().lower() == "fast")


def _is_graph_item(item: Dict[str, Any]) -> bool:
    return item.get("kind") in _GRAPH_KINDS


def _chunk_attachments(items: list) -> list:
    """召回切片 → UI 气泡附件（同文档同序号去重，保留重排后的顺序）。

    所有切片的核心观点句都已注入上下文，因此统一标记 ``used=True``
    （UI 显示「已参考」）；完整原文仍可点开查看。
    """
    out: list = []
    seen = set()
    for r in items:
        key = (r.get("source") or "", r.get("chunk_index"))
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "kind": "chunk",
            "source": r.get("source") or "",
            "chunk_index": r.get("chunk_index"),
            "score": float(r.get("score") or 0.0),
            "text": r.get("text") or "",
            "used": True,
        })
    return out


def _compact_chunk_text(item: Dict[str, Any]) -> str:
    """单个切片 → 核心观点句（默认 ≤800 字），再兜底硬截断。"""
    text = simplify_for_scoring(str(item.get("text") or ""))
    if len(text) > _MAX_PRESENTATION_CHUNK_CHARS:
        text = text[:_MAX_PRESENTATION_CHUNK_CHARS] + "…"
    return text


def _format_chunk_blocks(doc_chunks: list) -> list:
    """把多个切片格式化成「来源/相关度 + 核心观点句」块。"""
    lines: list = []
    for i, item in enumerate(doc_chunks, 1):
        src = os.path.basename(item.get("source") or "") or "文档"
        try:
            score = float(item.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        lines.append(f"### [{i}] 来源: {src} (相关度: {score:.4f})")
        lines.append(_compact_chunk_text(item))
        lines.append("")
    return lines


def _format_presentation_output(graph_items: list, doc_chunks: list) -> str:
    """检索即展示：图谱条目全量 + 所有切片的核心观点句全量注入。"""
    lines: list = []
    if graph_items:
        lines.append(f"知识图谱命中 {len(graph_items)} 条实体/关系：")
        lines.extend(str(it.get("text") or "") for it in graph_items)
    if doc_chunks:
        lines.append(
            f"知识库检索到 {len(doc_chunks)} 个相关切片；以下每个切片的"
            "核心观点句均已注入上下文，请综合所有切片作答。完整原文已作为"
            "参考切片气泡展示给用户，不要声称看不到其余内容。")
        lines.extend(_format_chunk_blocks(doc_chunks))
    if not lines:
        lines.append("（知识库中未找到相关内容。）")
    return "\n".join(lines).strip()


def _cap_output(text: str) -> str:
    if len(text) <= _MAX_TOOL_OUTPUT_CHARS:
        return text
    return (text[:_MAX_TOOL_OUTPUT_CHARS]
            + f"\n… [输出超过 {_MAX_TOOL_OUTPUT_CHARS} 字符，已截断]")


def _as_text(value: Any) -> Any:
    """把模型可能传来的标量归一化成去空白字符串；容器等非法类型返回 None。

    本地小模型常把 query/book_name 传成 int/list/dict/null。旧实现
    ``(value or "").strip()`` 对非字符串直接抛 AttributeError，被 registry
    兜底成「工具内部错误」，模型会原样重试形成循环。
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        return str(value).strip()
    return None

if TYPE_CHECKING:
    from ..rag.engine import RAGEngine


class RagSearchTool(Tool):
    """Search the local knowledge base for relevant information."""

    name = "rag_search"
    description = _DEFAULT_SEARCH_DESCRIPTION
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
        # 工具描述随模式切换：阅读模式/展示模式下所有切片抽取核心观点句
        # 后注入，完整原文出气泡；描述必须与实际数据分流一致，否则模型会
        # 编造未提供的切片正文。
        self.description = (
            _READING_SEARCH_DESCRIPTION
            if _reading_enabled(self._engine, config)
            else _DEFAULT_SEARCH_DESCRIPTION)

    def run(self, query: str = "", top_k: Any = None, source: str = "") -> ToolResult:
        # 参数校验优先于引擎可用性：空参/非法参数无需触碰引擎，且必须拿到
        # 反循环指引（本地小模型退化连发空参是实测过的失败模式）。
        # 模型可能传 null（JSON null → None），None.strip() 会抛异常并被
        # registry 兜底误报成 "Invalid arguments"，模型会原样重试。这里统一
        # 归一化成空串再判断。
        query = _as_text(query)
        if query is None:
            return ToolResult(
                False,
                error=("rag_tools(op='search') 的 query 必须是字符串检索词。合法参数："
                       "query(字符串) / top_k(1-10 整数) / source(可选字符串)。"
                       "不要重试同样的错误参数。"),
            )
        if not query:
            # 本地小模型退化时可能不带参数连发：错误信息必须自带反循环指引，
            # 否则模型只会原样重试空参。
            return ToolResult(
                success=False,
                error=(
                    "rag_tools(op=\"search\") 缺少必需参数 query（搜索查询词）。"
                    "不要重复调用空参的 rag_tools(op=\"search\")——先停下来：若 rag_tools(op=\"outline\") 已返回【全书摘要】或目录里已有章节摘要，直接据此作答；"
                    "确需细节，用一句中文查询词（如 \"薄伽梵歌 主要思想\"）再调用一次；仍拿不到就如实说明。"
                ),
            )

        if self._engine is None:
            return ToolResult(False, error=_ENGINE_RELEASED_ERROR)
        # 默认 top_k 从配置读取（rag_search_top_k，默认 5）；模型未传/传 null
        # 时走配置默认，传了非法值也回退到配置默认。_config 由 bind() 注入，
        # 直接构造（测试/旁路）时可能缺失，一律走 getattr 默认值。
        cfg = getattr(self, "_config", None)
        # 以引擎实际管线为准：config 与引擎若短暂不一致（配置已改但引擎尚未
        # 重建），按 config 分流会把阅读模式的长切片当普通召回注入上下文。
        reading_mode = _reading_enabled(self._engine, cfg)
        try:
            default_top_k = int(
                getattr(cfg, "rag_reading_top_n", 3) if reading_mode
                else getattr(cfg, "rag_search_top_k", 5))
        except (TypeError, ValueError):
            default_top_k = 3 if reading_mode else 5  # 配置值非法不能崩工具
        try:
            top_k = int(top_k)
        except (TypeError, ValueError, OverflowError):
            top_k = default_top_k
        # 限制 top_k 上限，防止模型传入过大值导致返回海量内容撑爆上下文窗口。
        top_k = max(1, min(top_k, 10))
        if reading_mode:
            # 检索阅读模式的「精选 topN」是配置项，优先于模型传入的 top_k。
            try:
                top_k = max(1, min(
                    int(getattr(cfg, "rag_reading_top_n", 3) or 3), 30))
            except (TypeError, ValueError):
                top_k = 3

        source = _as_text(source)
        if source is None:
            source = ""  # 非法 source 退化为不限定来源，不打断检索
        logger.info(
            "🔍 RAG 工具调用: rag_search(query=%r, top_k=%d, source=%r)",
            query[:80], top_k, source[:80],
        )
        try:
            # 先取重排后的原始分片（含完整 text，未截断），供辅助模型通读归纳。
            chunks = self._engine.search(query, top_k=top_k, source=source or None)
            # 检索阅读模式：所有召回切片都抽取核心观点句后注入，保证覆盖面；
            # 完整原文仍作为可点击气泡附在回复末尾。
            if reading_mode:
                attachments = _chunk_attachments(chunks) if chunks else []
                doc_chunks = [r for r in chunks if not _is_graph_item(r)]
                graph_items = [r for r in chunks if _is_graph_item(r)]
                if chunks:
                    output = (
                        f"检索阅读模式已召回 {len(attachments)} 个长文档切片。"
                        "以下每个切片的核心观点句均已注入上下文，请综合所有"
                        "切片总结或回答；完整原文已作为可点击的「参考切片」"
                        "气泡附在回复末尾，可提示用户点开阅读。不要复述整片"
                        "原文，也不要猜测或编造未给出的内容。")
                    extra = [str(it.get("text") or "") for it in graph_items]
                    extra.extend(_format_chunk_blocks(doc_chunks))
                    output += "\n\n" + "\n".join(extra)
                else:
                    output = (
                        "检索阅读模式未召回匹配切片。请提示用户换一个更具体的"
                        "问题或先同步/导入对应文档，不要编造内容。")
                logger.info(
                    "✅ 阅读模式 RAG 检索 %d 条 → 上下文注入全部 %d 片核心句，"
                    "气泡 %d 条", len(chunks), len(doc_chunks), len(attachments))
                return ToolResult(
                    success=True, output=_cap_output(output),
                    attachments=attachments)
            # 检索即展示（Retrieval-as-Presentation）：图谱条目全量 + 所有
            # 原文切片的核心观点句全量注入；完整原文作为气泡展示。
            if bool(getattr(cfg, "rag_presentation_mode", True)):
                if chunks:
                    graph_items = [r for r in chunks if _is_graph_item(r)]
                    doc_chunks = [r for r in chunks if not _is_graph_item(r)]
                    output = _format_presentation_output(
                        graph_items, doc_chunks)
                    logger.info(
                        "✅ RAG 检索 %d 条 → 上下文注入 图谱 %d / 原文 %d 片"
                        "核心句，气泡展示 %d 条",
                        len(chunks), len(graph_items), len(doc_chunks),
                        len(doc_chunks),
                    )
                    return ToolResult(
                        success=True, output=_cap_output(output),
                        attachments=_chunk_attachments(doc_chunks))
                # 无召回：沿用原格式化路径的「未找到」提示。
                results = self._engine.search_formatted(
                    query, top_k=top_k, source=source or None,
                )
                return ToolResult(success=True, output=_cap_output(results))
            condensed = ""
            use_core = bool(getattr(cfg, "rag_condense_use_core_sentences", True))
            if chunks and use_core:
                # 默认路径：本地核心句压缩，无字数门槛、无输入上限（12000 字以上
                # 同样逐切片抽取），不依赖辅助模型。逐切片压缩后拼成精简结果。
                parts: list[str] = []
                for i, r in enumerate(chunks, 1):
                    txt = str(r.get("text") or "")
                    core = simplify_for_scoring(txt, min_chars=0)
                    if not core:
                        continue
                    src = r.get("source") or ""
                    head = f"[片段 {i}]" + (f" 来源: {os.path.basename(src)}" if src else "")
                    parts.append(f"{head}\n{core}")
                condensed = "\n\n".join(parts)
                logger.info(
                    "  ├─ 核心句压缩（本地，无字数门槛）: %d 条 → %d 字",
                    len(chunks), len(condensed),
                )
            elif chunks:
                # 备用路径：开关关闭时保留原辅助模型摘要（含 [min,max] 区间约束）。
                aux = self._aux_llm_getter() if self._aux_llm_getter else None
                if aux is not None:
                    total_chars = sum(len(r.get("text") or "") for r in chunks)
                    min_chars = int(getattr(cfg, "rag_condense_min_chars", 4000) or 4000)
                    max_in = int(getattr(cfg, "rag_condense_max_chars", 12000) or 12000)
                    if total_chars < min_chars or total_chars > max_in:
                        logger.info(
                            "  ├─ 检索总量 %d 字不在 [%d,%d] → 跳过辅助模型归纳",
                            total_chars, min_chars, max_in,
                        )
                    else:
                        try:
                            condensed = condense_retrieved_chunks(
                                query, chunks, aux, max_chars=2000)
                        except Exception as e:  # 精简异常不应让检索整体失败
                            logger.warning("RAG 辅助模型精简异常，回退原始结果: %s", e)
                            condensed = ""
            if condensed:
                method = "核心句压缩" if use_core else "辅助模型归纳"
                output = build_condensed_output(query, chunks, condensed, method=method)
                logger.info(
                    "✅ RAG 检索 %d 条 → %s为 %d 字注入主模型",
                    len(chunks), method, len(condensed),
                )
                return ToolResult(success=True, output=output)
            # 回退：浓缩未产出 / 辅助模型不可用 / 总量不在精简区间时，维持原格式化行为。
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
        "调用，先掌握该文档的目录结构，再用 rag_tools(op='search') 检索章节细节。"
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
        book_name = _as_text(book_name)
        if book_name is None:
            return ToolResult(
                False,
                error=("rag_tools(op='outline') 的 book_name 必须是字符串；不带参数表示"
                       "列出全部文档。不要重试同样的错误参数。"),
            )
        if self._engine is None:
            return ToolResult(False, error=_ENGINE_RELEASED_ERROR)
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
            # 决策指引放在工具结果里（比系统提示更靠近决策点）：用户问句没点名
            # 文档时，模型常卡在「不知道对哪本书 outline」而反复复述意图。
            if len(docs) == 1:
                lines.append(
                    "\n只有这一篇：直接对它执行先总后分"
                    "（rag_tools(op='outline', book_name=…) 拿目录/摘要 → 按需 rag_tools(op='search')）。")
            else:
                lines.append(
                    "\n用户问题没点名文档时：挑文档名/主题与问题最相关的一篇"
                    "直接执行先总后分，并在回答开头注明「基于《XX》」；确实"
                    "无法判断才用 ask_user 请用户选择。")
            return ToolResult(success=True, output=_cap_output("\n".join(lines)))

        logger.info("📖 RAG 工具调用: rag_outline(book_name=%r)", book_name[:80])
        try:
            doc = self._engine.get_document_outline(book_name)
            if doc is None:
                return ToolResult(
                    success=False,
                    error=f"未找到文档「{book_name}」的缩略版本。可先调用 rag_tools(op='outline') 查看可用文档列表。",
                )
            digest = doc.get("digest", "")
            if not digest:
                return ToolResult(success=False, error=f"文档「{doc.get('doc_name', book_name)}」暂无缩略版本")
            # 附带后台闲时生成的全书摘要（若有）：目录结构 + 全书摘要一起给，
            # 模型无需逐章 rag_tools(op='search') 即可把握全书脉络。
            book_summary = ""
            try:
                book_summary = self._engine.get_book_summary(
                    doc.get("doc_name", book_name))
            except Exception:
                book_summary = ""
            if book_summary:
                digest = f"{digest}\n\n【全书摘要】\n{book_summary}"
            # 收敛指引无条件附加：无【全书摘要】时模型同样可能逐章刷 rag_search，
            # 旧实现只在有摘要时才拼，导致「先总后分」约束在冷库期完全缺失。
            digest += (
                "\n\n（问整体内容（主要思想/讲了什么）直接据此作答，"
                "无需再 rag_tools(op='search')；确需细节只检索目标章节——"
                "本轮 rag_tools(op='search') 至多 3 次，超过会被内核拦截。）")
            # 带参分支同样要过体积兜底：大书目录 + 全书摘要可能非常长。
            return ToolResult(success=True, output=_cap_output(digest))
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
        if self._engine is None:
            return ToolResult(False, error=_ENGINE_RELEASED_ERROR)
        try:
            status = self._engine.status()
            lines = [
                f"知识库状态:",
                f"  知识库目录: {status['knowledge_base'] or '（未设置）'}",
                f"  RAG 目录: {status['rag_dir']}",
                f"  已存储切片数: {status['chunks_stored']}",
                f"  已索引文件数: {len(status['sources'])}",
            ]
            if status.get("reading_mode"):
                lines.append(
                    "  运行模式: 检索阅读模式（所有切片核心句注入上下文，"
                    "完整原文仅气泡展示）")
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
    # 重建知识库会清除并重新索引全部文档，是破坏性操作，需人工确认。
    destructive = True

    def __init__(self, engine: "RAGEngine", registry: "ToolRegistry" = None):
        self._engine = engine
        self._registry = registry

    def bind(self, config: Any, registry: ToolRegistry) -> None:
        self._registry = registry

    def run(self, force: bool = False, **kwargs) -> ToolResult:
        if self._engine is None:
            return ToolResult(False, error=_ENGINE_RELEASED_ERROR)
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
                # 只报数量时用户/模型无法判断是「可忽略的坏文件」还是
                # 「manifest 写盘失败」这类会导致下次全量重来的严重问题。
                for e in errors[:3]:
                    lines.append(f"    - {str(e)[:200]}")
                if len(errors) > 3:
                    lines.append(f"    … （其余 {len(errors) - 3} 个省略）")
            # 模型自行同步后，volatile_context 的 30s 状态缓存仍在显示旧的
            # chunks 数或「Configured but empty」→ 通知上层立即失效。
            changed_cb = getattr(self._registry, "on_rag_data_changed", None)
            if callable(changed_cb):
                try:
                    changed_cb()
                except Exception as e:  # 缓存失效失败不能让一次成功的同步变失败
                    logger.warning("rag_ingest 状态缓存失效回调异常: %s", e)
            return ToolResult(success=True, output="\n".join(lines))
        except Exception as e:
            logger.error("rag_ingest failed: %s", e, exc_info=True)
            return ToolResult(success=False, error=f"知识库索引失败: {e}")


class RagToolsTool(_FanoutTool):
    """RAG 统一门面：op 分发到 search / outline / status / ingest。"""

    # op 级参数隔离：status 无参，ingest 只认 force，outline 只认 book_name。
    # 子工具 ingest/outline/status 都带 **kwargs，若不隔离会静默吞掉误传的
    # query/source 等参数，模型会以为检索生效了。
    _op_params = {
        "search": {"query", "top_k", "source"},
        "outline": {"book_name"},
        "status": set(),
        "ingest": {"force"},
    }

    name = "rag_tools"
    description = _DEFAULT_RAG_TOOLS_DESCRIPTION
    parameters = {
        "type": "object",
        "properties": {
            "op": {"type": "string", "enum": ["search", "outline", "status", "ingest"]},
            "query": {"type": "string", "description": "检索词（search）"},
            "top_k": {"type": "integer", "description": "返回结果数，默认 5，范围 1-10（search）"},
            "source": {"type": "string", "description": "限定书名/文档名关键词（search）"},
            "book_name": {"type": "string", "description": "书名/文档名关键词（outline）"},
            "force": {"type": "boolean", "description": "强制重新处理所有文件（ingest，默认 false）"},
        },
        "required": ["op"],
    }

    def __init__(self, engine: "RAGEngine", aux_llm_getter: Any = None):
        self._sub_tools = {
            "search": RagSearchTool(engine, aux_llm_getter=aux_llm_getter),
            "outline": RagOutlineTool(engine),
            "status": RagStatusTool(engine),
            "ingest": RagIngestTool(engine),
        }

    def bind(self, config: Any, registry: ToolRegistry) -> None:
        super().bind(config, registry)
        self.description = (
            _READING_RAG_TOOLS_DESCRIPTION
            if _reading_enabled(
                getattr(self._sub_tools.get("search"), "_engine", None), config)
            else _DEFAULT_RAG_TOOLS_DESCRIPTION)