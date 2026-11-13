"""KG 检索器（spec 第七/八步）。

流程：
1. LLM 抽取 hl（高层主题）/ ll（低层实体）关键词（无辅助模型时回退原始 query）；
2. 一个 batch 同时计算三组查询向量（task=query）：ll→实体、hl→关系、原文→chunk；
3. 三路召回：实体 top40 / 关系 top40 / chunk top20，余弦 <0.2 丢弃；
4. 图谱扩展：实体一跳邻边、关系两端实体与度数（rank+weight 排序）；
5. 交错去重：实体按名、关系按无向三元组、chunk 按 id（查询直召 chunk 优先）；
6. token 预算：实体 6000 / 关系 8000 / 总 30000；反查 chunk 每实体/关系最多 5 片；
7. 五种策略（local/global/hybrid/mix/naive）产出同形结果 dict，交给
   RAGEngine 共用的 reranker 精排。
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from .builder import load_or_rebuild_graph, kg_dir_for  # noqa: F401
from .embedding import KGEmbedder
from .extractor import KGExtractor
from .storage import KGStore

logger = logging.getLogger(__name__)

# token 预算（spec 第七步）
ENTITY_TOKEN_BUDGET = 6000
RELATION_TOKEN_BUDGET = 8000
TOTAL_TOKEN_BUDGET = 30000
TOP_K_ENTITIES = 40
TOP_K_RELATIONS = 40
TOP_K_CHUNKS = 20
MAX_CHUNKS_PER_BACKREF = 5
MIN_COSINE = 0.2

VALID_STRATEGIES = ("local", "global", "hybrid", "mix", "naive")


class KGRetriever:
    """知识图谱检索（由 RAGEngine.search 在 strategy != naive 时调用）。"""

    def __init__(
        self,
        rag_dir: str,
        store,                     # rag.vector_store.VectorStore
        extractor: KGExtractor,
        estimate_tokens=None,      # chunker.estimate_tokens
    ):
        self._rag_dir = rag_dir
        self._store = store
        self._extractor = extractor
        self._estimate_tokens = estimate_tokens or (lambda s: max(1, len(s) // 2))
        self._kg_store: Optional[KGStore] = None

    def close(self) -> None:
        if self._kg_store is not None:
            self._kg_store.close()
            self._kg_store = None

    # ------------------------------------------------------------------

    def _ensure_kg_store(self) -> KGStore:
        if self._kg_store is None:
            self._kg_store = KGStore(kg_dir_for(self._rag_dir))
        return self._kg_store

    def graph_ready(self) -> bool:
        """图谱是否可用（有向量且有图节点）——决定非 naive 策略能否启用。"""
        try:
            counts = self._ensure_kg_store().counts()
            return counts["entities"] > 0 and counts["vectors"] > 0
        except Exception as e:
            logger.warning("KG: 检查图谱可用性失败: %s", e)
            return False

    def search(
        self,
        query: str,
        strategy: str,
        chunk_recall=None,         # Callable[[str, int], List[dict]] 原文 chunk 召回
        rerank=None,               # Callable[[str, List[dict], int], List[dict]]
    ) -> List[Dict[str, Any]]:
        """主入口。返回与 engine.search 同形的结果列表（多 ``kind`` 字段）。"""
        strategy = strategy if strategy in VALID_STRATEGIES else "naive"
        if strategy == "naive":
            return []
        graph = load_or_rebuild_graph(self._rag_dir, self._store)
        if graph is None or graph.is_empty:
            logger.info("KG: 知识图谱未构建，%s 策略回退 naive", strategy)
            return []
        kg = self._ensure_kg_store()
        embedder = KGEmbedder(self._store.embedding_function)

        t0 = time.monotonic()
        # 1) 关键词抽取
        kws = self._extractor.extract_keywords(query)
        hl, ll = kws["high_level_keywords"], kws["low_level_keywords"]
        ll_text = "，".join(ll) or query
        hl_text = "，".join(hl) or query
        logger.info("KG: 关键词 hl=%s ll=%s (耗时 %.2fs)",
                    hl[:6], ll[:6], time.monotonic() - t0)

        # 2) 三组查询向量一个 batch 算完
        q_vecs = embedder.embed_query([ll_text, hl_text, query])
        ll_vec, hl_vec, q_vec = q_vecs[0], q_vecs[1], q_vecs[2]

        # 3) 三路召回
        ent_vectors = kg.iter_vectors("entity")
        rel_vectors = kg.iter_vectors("relationship")
        ent_hits = embedder.search_vectors(ll_vec, ent_vectors,
                                           TOP_K_ENTITIES, MIN_COSINE)
        rel_hits = embedder.search_vectors(hl_vec, rel_vectors,
                                           TOP_K_RELATIONS, MIN_COSINE)
        entities = self._load_entities(kg, ent_hits)
        relations = self._load_relations(kg, rel_hits)
        logger.info("KG: 召回 实体=%d 关系=%d (余弦>=%.2f)",
                    len(entities), len(relations), MIN_COSINE)

        # 4) 图谱扩展 + 策略裁剪 + 去重
        if strategy == "local":
            kg_entities, kg_relations = self._strategy_local(graph, entities)
        elif strategy == "global":
            kg_entities, kg_relations = self._strategy_global(graph, relations)
        elif strategy == "hybrid":
            e1, r1 = self._strategy_local(graph, entities)
            e2, r2 = self._strategy_global(graph, relations)
            kg_entities, kg_relations = e1 + e2, r1 + r2
        else:  # mix：实体 + 关系 + 原文 chunk 全做
            e1, r1 = self._strategy_local(graph, entities)
            e2, r2 = self._strategy_global(graph, relations)
            kg_entities, kg_relations = e1 + e2, r1 + r2

        # 5) 交错去重（实体按名 / 关系按无向三元组），token 预算截断
        kg_entities = self._dedup_entities(kg_entities)
        kg_relations = self._dedup_relations(kg_relations)
        kg_entities = self._truncate_entities(kg_entities)
        kg_relations = self._truncate_relations(kg_relations)

        results: List[Dict[str, Any]] = []
        # mix 模式：原文 chunk 召回 + 实体/关系反查 chunk（交错去重，查询直召优先）
        if strategy == "mix" and chunk_recall is not None:
            direct = chunk_recall(query, TOP_K_CHUNKS)
            backref = self._backref_chunks(kg, kg_entities, kg_relations)
            results.extend(self._dedup_chunks(direct, backref))
        for ent in kg_entities:
            results.append({
                "text": self._format_entity(ent),
                "source": (ent.get("file_paths") or ["知识图谱"])[0],
                "score": ent.get("similarity", 0.0),
                "kind": "kg_entity",
            })
        for rel in kg_relations:
            results.append({
                "text": self._format_relation(rel),
                "source": (rel.get("file_paths") or ["知识图谱"])[0],
                "score": rel.get("similarity", 0.0),
                "kind": "kg_relation",
            })
        # 6) token 总预算（KG + chunks ≤ 30000）
        results = self._apply_total_budget(results, query)
        logger.info("KG: %s 策略产出 %d 条 (实体 %d / 关系 %d, 耗时 %.2fs)",
                    strategy, len(results), len(kg_entities), len(kg_relations),
                    time.monotonic() - t0)
        # 7) 共用 reranker 精排
        if rerank is not None and results:
            results = rerank(query, results, top_k=len(results))
        return results

    # ------------------------------------------------------------------
    # 策略实现
    # ------------------------------------------------------------------

    def _strategy_local(self, graph, entities):
        """local：命中实体拉邻边与关联实体（问具体细节）。"""
        out_e, out_r = [], []
        for ent in entities:
            out_e.append(ent)
            for edge in graph.neighbor_edges(ent["name_key"]):
                rel = self._graph_edge_to_rel(edge, ent["name_key"])
                if rel:
                    out_r.append(rel)
                for other_key in (edge.get("src"), edge.get("dst")):
                    if other_key and other_key != ent["name_key"]:
                        node = graph.get_node(other_key)
                        if node:
                            out_e.append(self._graph_node_to_ent(node, other_key))
        return out_e, out_r

    def _strategy_global(self, graph, relations):
        """global：命中关系反查两端实体及度数（跨文档宏观主题）。"""
        out_e, out_r = [], []
        for rel in relations:
            out_r.append(rel)
            pk = rel.get("pair_key")
            edge = graph.get_edge_by_pair(pk) if pk else None
            keys = ([edge["src"], edge["dst"]] if edge
                    else [rel.get("src_entity"), rel.get("dst_entity")])
            for k in keys:
                if not k:
                    continue
                node = graph.get_node(k)
                if node:
                    out_e.append(self._graph_node_to_ent(node, k))
        return out_e, out_r

    # ------------------------------------------------------------------
    # 载入 / 转换
    # ------------------------------------------------------------------

    def _load_entities(self, kg, hits) -> List[Dict[str, Any]]:
        out = []
        for h in hits:
            ent = kg.get_entity(h["target_id"])
            if not ent:
                continue
            ent["similarity"] = h["similarity"]
            out.append(ent)
        return out

    def _load_relations(self, kg, hits) -> List[Dict[str, Any]]:
        out = []
        for h in hits:
            rel = kg.get_relationship(h["target_id"])
            if not rel:
                continue
            rel["similarity"] = h["similarity"]
            out.append(rel)
        # 向量相似度 → rank+weight 降序（spec：按 rank(度数)和 weight 排）
        out.sort(key=lambda r: (r.get("raw_weight", 1), r.get("similarity", 0.0)),
                 reverse=True)
        return out

    def _graph_node_to_ent(self, node: Dict[str, Any], name_key: str) -> Dict[str, Any]:
        ent = {
            "name": node.get("name") or name_key, "name_key": name_key,
            "type": node.get("type") or "UNKNOWN",
            "description": node.get("description") or "",
            "source_ids": node.get("source_ids") or [],
            "degree": node.get("degree", 0),
        }
        ent["file_paths"] = []
        return ent

    def _graph_edge_to_rel(self, edge: Dict[str, Any], anchor_key: str) -> Optional[Dict[str, Any]]:
        return {
            "src_entity": edge.get("src_name") or edge.get("src"),
            "dst_entity": edge.get("dst_name") or edge.get("dst"),
            "pair_key": edge.get("pair_key"),
            "relation_type": edge.get("relation_type") or "",
            "keywords": edge.get("keywords") or "",
            "description": edge.get("description") or "",
            "raw_weight": edge.get("weight", 1),
            "source_ids": edge.get("source_ids") or [],
            "file_paths": edge.get("file_paths") or [],
        }

    # ------------------------------------------------------------------
    # 去重 / 截断
    # ------------------------------------------------------------------

    @staticmethod
    def _dedup_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """实体按名去重：合并 source_ids、相似度取最大，文本字段保留
        相似度更高（或原为空）的版本。"""
        best: Dict[str, Dict[str, Any]] = {}
        for ent in entities:
            key = (ent.get("name_key")
                   or str(ent.get("name") or "").lower())
            cur = best.get(key)
            if cur is None:
                best[key] = dict(ent)
                continue
            cur["source_ids"] = list(dict.fromkeys(
                list(cur.get("source_ids") or []) + list(ent.get("source_ids") or [])))
            cur_sim = cur.get("similarity", 0)
            ent_sim = ent.get("similarity", 0)
            cur["similarity"] = max(cur_sim, ent_sim)
            if ent.get("description") and (
                    not cur.get("description") or ent_sim >= cur_sim):
                for k in ("name", "type", "description", "degree", "file_paths"):
                    if ent.get(k):
                        cur[k] = ent[k]
        return list(best.values())

    @staticmethod
    def _dedup_relations(relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """关系按无向三元组（pair_key）去重，weight 取最大。"""
        best: Dict[str, Dict[str, Any]] = {}
        for rel in relations:
            pk = (rel.get("pair_key")
                  or "\x1f".join(sorted([str(rel.get("src_entity", "")).lower(),
                                         str(rel.get("dst_entity", "")).lower()])))
            rel = {**rel, "pair_key": pk}
            cur = best.get(pk)
            if cur is None:
                best[pk] = rel
            else:
                cur["raw_weight"] = max(cur.get("raw_weight", 1),
                                        rel.get("raw_weight", 1))
                cur["similarity"] = max(cur.get("similarity", 0),
                                        rel.get("similarity", 0))
        return list(best.values())

    def _truncate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按向量相似度排序 + 6000 token 预算截断。"""
        entities = sorted(entities,
                          key=lambda e: e.get("similarity", 0), reverse=True)
        out, budget = [], ENTITY_TOKEN_BUDGET
        for ent in entities:
            cost = self._estimate_tokens(self._format_entity(ent))
            if cost > budget:
                break
            out.append(ent)
            budget -= cost
        return out

    def _truncate_relations(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按 rank+weight 排序 + 8000 token 预算截断。"""
        relations = sorted(relations,
                           key=lambda r: (r.get("raw_weight", 1),
                                          r.get("similarity", 0)), reverse=True)
        out, budget = [], RELATION_TOKEN_BUDGET
        for rel in relations:
            cost = self._estimate_tokens(self._format_relation(rel))
            if cost > budget:
                break
            out.append(rel)
            budget -= cost
        return out

    def _apply_total_budget(self, results: List[Dict[str, Any]],
                            query: str) -> List[Dict[str, Any]]:
        """整体预算 30000（system+query+KG+chunks；system/query 预留 2000）。"""
        budget = TOTAL_TOKEN_BUDGET - 2000 - self._estimate_tokens(query)
        out = []
        for r in results:
            cost = self._estimate_tokens(r.get("text") or "")
            if cost > budget:
                break
            out.append(r)
            budget -= cost
        return out

    # ------------------------------------------------------------------
    # 反查 chunk（归因边 → 溯源）
    # ------------------------------------------------------------------

    def _backref_chunks(self, kg, entities, relations) -> List[Dict[str, Any]]:
        """召回实体/关系对应的源 chunk（每条最多 5 片，按 id 去重）。"""
        ids: List[str] = []
        for ent in entities[:10]:
            for cid in (ent.get("source_ids") or [])[:MAX_CHUNKS_PER_BACKREF]:
                if cid not in ids:
                    ids.append(cid)
        for rel in relations[:10]:
            for cid in (rel.get("source_ids") or [])[:MAX_CHUNKS_PER_BACKREF]:
                if cid not in ids:
                    ids.append(cid)
        if not ids:
            return []
        chunks = self._store.get_chunks_by_ids(ids)
        # 带上 id / chunk_index：检索即展示模式下这些切片要变成可点击气泡，
        # 阅读器需要按（source, chunk_index）翻页（缺字段会退化成「第 ? 片」）。
        return [{"id": c.get("id") or "",
                 "text": c.get("text") or "",
                 "source": c.get("source") or "",
                 "chunk_index": c.get("chunk_index"),
                 "score": 0.0, "kind": "kg_backref_chunk"}
                for c in chunks]

    @staticmethod
    def _dedup_chunks(direct: List[Dict[str, Any]],
                      backref: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """chunk 按 id 去重，查询语句直召的 chunk 优先（spec 第七步）。"""
        seen = set()
        out = []

        def _key(c):
            return c.get("id") or f"{c.get('source')}#{c.get('chunk_index')}"

        for c in direct:
            k = _key(c)
            if k in seen:
                continue
            seen.add(k)
            out.append({**c, "kind": "chunk"})
        for c in backref:
            k = _key(c)
            if k in seen:
                continue
            seen.add(k)
            out.append(c)
        return out

    # ------------------------------------------------------------------
    # 格式化（给 LLM 的 context 块）
    # ------------------------------------------------------------------

    @staticmethod
    def _format_entity(ent: Dict[str, Any]) -> str:
        deg = ent.get("degree")
        deg_s = f"，度数 {deg}" if isinstance(deg, int) else ""
        return (f"[实体] {ent.get('name')}（类型: {ent.get('type')}{deg_s}）\n"
                f"{ent.get('description') or ''}".strip())

    @staticmethod
    def _format_relation(rel: Dict[str, Any]) -> str:
        kws = rel.get("keywords") or rel.get("relation_type") or "相关"
        return (f"[关系] {rel.get('src_entity')} ——{kws}—— {rel.get('dst_entity')}"
                f"（证据权重 {rel.get('raw_weight', 1)}）\n"
                f"{rel.get('description') or ''}".strip())
