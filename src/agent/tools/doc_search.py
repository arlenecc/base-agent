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
from ..rag.colbert_store import (
    ColBertDocStore, ColBertStoreError, get_store_for_workspace,
    _MAX_COARSE, _MAX_TOP_K,
)
from ..rag.colbert_runtime import ColBertUnavailable


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

    def bind(self, config: Any, registry: Any) -> None:
        self.config = config
        self.registry = registry

    def run(self, op: str = "search", path: str = "", text: str = "",
            title: str = "", doc_id: str = "", query: str = "",
            top_k: int = 5, coarse_k: int = 20, include_content: bool = True,
            max_content_chars: int = 1500, overwrite: bool = True,
            quant_bits: int = 0) -> ToolResult:
        try:
            store = _get_store(self.config)
            op = str(op or "search").strip().lower()
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
