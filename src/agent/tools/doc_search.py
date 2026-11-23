"""doc_search_tool：问句直接召回长文档切片（ColBERT + LanceDB）。

与 ``search_tool``（公网网页聚合）不同，这个工具面向**用户自己的长文档**：
先把 Markdown/TXT 入库（标题优先的语义切片 + ColBERT 残差量化），
之后用问句做 BM25 + 中心向量 RRF 粗筛，再用 MaxSim 精排返回 topN。

典型调用：
    doc_search_tool(op="index", path="/path/to/book.md")
    doc_search_tool(op="search", query="asyncio 的事件循环怎么工作？", top_k=5)
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .base import Tool, ToolResult
from ..core_opinion import simplify_for_scoring
from ..rag.colbert_store import (
    ColBertDocStore, ColBertStoreError, get_store_for_workspace,
    _MAX_COARSE, _MAX_TOP_K,
)
from ..rag.colbert_runtime import ColBertUnavailable


_MAX_READING_CHUNK_CHARS = 1500


def _get_store(config: Any) -> ColBertDocStore:
    workspace = os.path.abspath(getattr(config, "workspace", "") or ".")
    return get_store_for_workspace(
        workspace, model_dir=getattr(config, "rag_reading_model_dir", "") or "")


class DocSearchTool(Tool):
    name = "doc_search_tool"
    description = (
        "Long-document retrieval over the user's own files (ColBERT late "
        "interaction + LanceDB). Index Markdown/TXT first with op='index', "
        "then ask a question with op='search'. Returns the best matching "
        "chunks with real doc_id/title/content. Use this instead of stuffing "
        "a whole book into context."
    )
    _READING_DESCRIPTION = (
        "Long-document retrieval over the knowledge base (ColBERT + LanceDB). "
        "Retrieval-reading mode is on: the knowledge base is already indexed "
        "by the sync pipeline, so use op='search' only. Every returned chunk is "
        "reduced to core opinion sentences and injected into your context for "
        "summarizing/answering; full text stays available as clickable chips "
        "for the user. op='index'/'delete' are disabled in this mode — use "
        "rag_tools(op='ingest') and the knowledge-base sync instead."
    )
    destructive = False
    parameters = {
        "type": "object",
        "properties": {
            "op": {
                "type": "string",
                "enum": ["index", "search", "list", "delete", "stats"],
                "description": "Operation to perform.",
            },
            "path": {
                "type": "string",
                "description": "File or directory for op='index' "
                               "(.md/.markdown/.txt/.rst).",
            },
            "text": {
                "type": "string",
                "description": "Raw document text for op='index' when no path.",
            },
            "title": {
                "type": "string",
                "description": "Document title/name for raw text or delete.",
            },
            "doc_id": {
                "type": "string",
                "description": "Exact chunk doc_id for op='delete'.",
            },
            "query": {
                "type": "string",
                "description": "Question for op='search'.",
            },
            "top_k": {
                "type": "integer",
                "default": 5,
                "description": "Final MaxSim hits to return (1-30).",
            },
            "coarse_k": {
                "type": "integer",
                "default": 20,
                "description": "BM25 + center-vector candidates before MaxSim "
                               "(top_k-100).",
            },
            "include_content": {
                "type": "boolean",
                "default": True,
                "description": "Include a content preview in search results.",
            },
            "max_content_chars": {
                "type": "integer",
                "default": 1500,
                "description": "Preview length per hit.",
            },
            "overwrite": {
                "type": "boolean",
                "default": True,
                "description": "Replace existing chunks of the same source.",
            },
            "quant_bits": {
                "type": "integer",
                "enum": [2, 4],
                "description": "Residual quantization bits; omit to use config.",
            },
        },
        "required": ["op"],
    }

    def _reading_active(self) -> bool:
        """是否处于检索阅读模式：优先看共享 RAG 引擎的实际管线。"""
        reg = getattr(self, "registry", None)
        get_engine = getattr(reg, "get_rag_engine", None)
        engine = None
        if callable(get_engine):
            try:
                engine = get_engine()
            except Exception:
                engine = None
        mode = getattr(engine, "reading_mode", None)
        if mode is not None:
            return bool(mode)
        config = getattr(self, "config", None)
        return (str(getattr(config, "rag_mode", "vector") or "vector")
                .strip().lower() == "fast")

    def bind(self, config: Any, registry: Any) -> None:
        self.config = config
        self.registry = registry
        # 阅读模式下入库/删除由知识库同步统一负责（manifest 去重 + 三表一致），
        # 工具描述必须同步收窄，否则模型会先尝试 index 再被拒绝。
        if self._reading_active():
            self.description = self._READING_DESCRIPTION
        else:
            self.description = type(self).description

    def run(self, op: str = "search", path: str = "", text: str = "",
            title: str = "", doc_id: str = "", query: str = "",
            top_k: int = 5, coarse_k: int = 20, include_content: bool = True,
            max_content_chars: int = 1500, overwrite: bool = True,
            quant_bits: int = 0) -> ToolResult:
        try:
            op = str(op or "search").strip().lower()
            reading = self._reading_active()
            if reading and op in ("index", "delete"):
                # 与阅读模式共用同一份三表索引：绕过 manifest 的写入/删除会让
                # 「已同步」记录与索引内容错位（删掉的不会回来、加的没有记录）。
                return ToolResult(
                    False,
                    error=(
                        "检索阅读模式已启用：长文档索引由知识库同步统一构建。"
                        "doc_search_tool 的 index/delete 会绕过 manifest 与三表"
                        "一致性维护，请改用 rag_tools(op=\"ingest\") 或主窗口"
                        "「同步知识库」；删除文件请从知识库目录移除后重新同步。"))
            store = _get_store(self.config)
            if reading and op == "search":
                return self._search_reading(store, query, top_k, coarse_k)
            if op == "index":
                return self._index(store, path, text, title, overwrite,
                                   quant_bits)
            if op == "search":
                return self._search(store, query, top_k, coarse_k,
                                    include_content, max_content_chars)
            if op == "list":
                rows = store.list_documents()
                if not rows:
                    return ToolResult(True, output="Long-doc index is empty.")
                lines = [f"Indexed documents ({len(rows)}):"]
                lines += [f"- {r['doc_id']}  {r['title']}" for r in rows]
                return ToolResult(True, output="\n".join(lines))
            if op == "delete":
                removed = store.delete(doc_id=doc_id, source=path,
                                       title=title)
                return ToolResult(
                    True, output=f"Deleted {removed} chunk(s). "
                                 f"stats={store.stats()}")
            if op == "stats":
                return ToolResult(True, output=str(store.stats()))
            return ToolResult(False, error=f"unknown op: {op}")
        except (ColBertUnavailable, ColBertStoreError) as e:
            return ToolResult(False, error=str(e))
        except Exception as e:
            return ToolResult(False, error=f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------------ ops
    def _index(self, store: ColBertDocStore, path: str, text: str,
               title: str, overwrite: bool, quant_bits: int) -> ToolResult:
        configured = getattr(self.config, "rag_reading_quant_bits", 4)
        quant_bits = int(quant_bits or configured or 4)
        if quant_bits not in (2, 4):
            return ToolResult(False, error="quant_bits must be 2 or 4")
        if path:
            result = store.index_path(path, overwrite=overwrite,
                                      quant_bits=quant_bits)
            return ToolResult(
                True,
                output=(f"Indexed {result['files']} file(s), "
                        f"{result['chunks']} chunk(s). "
                        f"stats={store.stats()}"),
            )
        if not (text or "").strip():
            return ToolResult(False, error="index requires path or text")
        title = (title or "Untitled").strip()
        chunks = store.index_text(
            title, text, source=title, overwrite=overwrite,
            quant_bits=quant_bits)
        return ToolResult(
            True,
            output=f"Indexed {chunks} chunk(s) from '{title}'. "
                   f"stats={store.stats()}",
        )

    def _search_reading(self, store: ColBertDocStore, query: str,
                        top_k: int, coarse_k: int) -> ToolResult:
        """阅读模式检索：所有切片抽取核心观点句后注入，完整原文只做气泡附件。

        阅读模式的「精选 topN / 粗筛候选 / ANN」由配置窗口统一控制，
        与 rag_tools 保持一致，因此忽略模型传入的 top_k/coarse_k。
        """
        query = (query or "").strip()
        if not query:
            return ToolResult(False, error="search requires query")
        try:
            top_k = max(1, min(int(
                getattr(self.config, "rag_reading_top_n", 3) or 3),
                _MAX_TOP_K))
        except (TypeError, ValueError):
            top_k = 3
        try:
            coarse_k = max(top_k, min(int(
                getattr(self.config, "rag_reading_coarse_k", 50) or 50),
                _MAX_COARSE))
        except (TypeError, ValueError):
            coarse_k = max(top_k, 50)
        try:
            min_score = float(getattr(
                self.config, "rag_reading_min_score", 0.61) or 0.0)
        except (TypeError, ValueError):
            min_score = 0.0
        hits = store.search(
            query, top_k=top_k, coarse_k=coarse_k,
            include_content=True, max_content_chars=10_000_000,
            use_ann=bool(getattr(
                self.config, "rag_reading_use_ann", False)),
            min_score=min_score)
        attachments: list = []
        seen = set()
        for h in hits:
            key = (h.get("source") or "", h.get("chunk_index"))
            if key in seen:
                continue
            seen.add(key)
            attachments.append({
                "kind": "chunk",
                "source": h.get("source") or h.get("title") or "",
                "chunk_index": h.get("chunk_index", 0),
                "score": float(h.get("score") or 0.0),
                "text": h.get("content") or "",
                # 所有切片的核心观点句都会注入上下文
                "used": True,
            })
        if not attachments:
            return ToolResult(
                True,
                output=(f'No long-document chunk matched "{query}" in '
                        f'retrieval-reading mode. Ask the user to sync the '
                        f'knowledge base or rephrase the query; do not make up '
                        f'content.'))
        blocks: list = []
        for i, att in enumerate(attachments, 1):
            text = simplify_for_scoring(str(att.get("text") or ""))
            if len(text) > _MAX_READING_CHUNK_CHARS:
                text = text[:_MAX_READING_CHUNK_CHARS] + "…"
            src = os.path.basename(att["source"]) or "文档"
            blocks.append(
                f"### [{i}] 来源: {src} (相关度: {att['score']:.4f})\n{text}")
        output = (
            f"检索阅读模式已召回 {len(attachments)} 个长文档切片。"
            "以下每个切片的核心观点句均已注入上下文，请综合所有切片总结或"
            "回答；完整原文已作为可点击的「参考切片」气泡附在回复末尾。"
            "不要复述整片原文，也不要猜测或编造未给出的内容。"
            "\n\n" + "\n\n".join(blocks))
        return ToolResult(
            True, output=output, attachments=attachments)

    def _search(self, store: ColBertDocStore, query: str, top_k: int,
                coarse_k: int, include_content: bool,
                max_content_chars: int) -> ToolResult:
        query = (query or "").strip()
        if not query:
            return ToolResult(False, error="search requires query")
        top_k = max(1, min(int(top_k or 5), _MAX_TOP_K))
        coarse_k = max(top_k, min(int(coarse_k or 20), _MAX_COARSE))
        hits = store.search(
            query, top_k=top_k, coarse_k=coarse_k,
            include_content=bool(include_content),
            max_content_chars=max(200, int(max_content_chars or 1500)))
        if not hits:
            return ToolResult(
                True,
                output=(f'No long-document chunk matched "{query}". '
                        f'If the index is empty, call op="index" first.'),
            )
        lines = [
            f'Long-doc search "{query}" — {len(hits)} hit(s) '
            f'(coarse={coarse_k}, MaxSim sorted)'
        ]
        for i, hit in enumerate(hits, 1):
            lines.append(
                f"[{i}] score={hit['score']:.4f} | tokens={hit['token_count']}"
                f" | {hit['title']}")
            lines.append(f"    doc_id: {hit['doc_id']}")
            if hit.get("content"):
                content = hit["content"].replace("\n", "\n    ")
                lines.append("    " + content)
        return ToolResult(True, output="\n".join(lines))

    def should_confirm(self, args: Dict[str, Any]) -> bool:
        return str(args.get("op", "")).lower() == "delete"
