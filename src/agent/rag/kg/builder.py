"""知识图谱构建流程（spec 第六/八/九步）。

入口 :func:`build_knowledge_graph`：
1. 遍历向量库全部切片（VectorStore 新增 ``iter_all_chunks``）；
2. 每片用辅助模型抽取实体/关系，写入 entities/relationships 表
   （同名实体/无向同对关系聚合，多描述交给 LLM 用
   ``summarize_entity_descriptions`` prompt 归纳）；
3. 全部抽取完成后做向量化（entity/relationship 文本 → 128 维截断 →
   L2 重归一化 → int8 入 kg_vectors；无向关系正反序两个 vector_id）；
4. 从库重建内存图并保存 JSON 快照（构建中途每 ``SNAPSHOT_EVERY`` 片
   落一次快照，避免频繁 IO；结束后落最终快照）。

进度/取消：``progress_callback(done, total, message)``；``cancel_event`` 置位后
在下一个切片前退出（已完成部分保留，可重跑续建——聚合天然幂等）。
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .embedding import KGEmbedder
from .extractor import KGExtractor
from .graph import KnowledgeGraph
from .storage import KGStore, normalize_name, pair_key

logger = logging.getLogger(__name__)

SNAPSHOT_EVERY = 50          # 每 N 片落一次快照
_SUMMARY_MIN_DESCS = 2       # 描述数达到该值才触发 LLM 归纳


def kg_dir_for(rag_dir: str) -> str:
    import os
    return os.path.join(rag_dir, "kg")


def snapshot_path_for(rag_dir: str) -> str:
    import os
    return os.path.join(kg_dir_for(rag_dir), "graph_snapshot.json")


class KGCancellation(Exception):
    pass


class KGBuilder:
    """驱动一次「抽取 → 入库 → 向量化 → 建图 → 快照」的完整构建。"""

    def __init__(
        self,
        rag_dir: str,
        store,                    # rag.vector_store.VectorStore（读 chunks + embedding fn）
        extractor: KGExtractor,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_event=None,
        log_callback: Optional[Callable[[str], None]] = None,
    ):
        self._rag_dir = rag_dir
        self._store = store
        self._extractor = extractor
        self._progress = progress_callback or (lambda d, t, m: None)
        self._cancel_event = cancel_event
        self._log = log_callback or (lambda m: None)
        self._kg_store = KGStore(kg_dir_for(rag_dir))
        self._embedder = KGEmbedder(store.embedding_function)

    def close(self) -> None:
        self._kg_store.close()

    # ------------------------------------------------------------------

    def build(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """执行完整构建。返回统计 dict。"""
        t0 = time.monotonic()
        chunks = list(self._store.iter_all_chunks())
        if limit:
            chunks = chunks[:limit]
        total = len(chunks)
        self._log(f"[kg] 开始构建知识图谱: {total} 个切片, "
                  f"实体/关系抽取模型={'辅助模型' if self._extractor.available else '（未配置，跳过抽取）'}")
        self._progress(0, total, "知识图谱构建开始")
        if not self._extractor.available:
            reason = getattr(self._extractor, "unavailable_reason", "") or ""
            raise RuntimeError(reason or "抽取器不可用：知识图谱构建无法继续")

        stats = {"chunks": 0, "entities": 0, "relationships": 0, "errors": 0}
        snapshot_path = snapshot_path_for(self._rag_dir)
        graph = KnowledgeGraph()
        graph.load_snapshot(snapshot_path)  # 增量续建：已有快照则热身内存图

        for done, chunk in enumerate(chunks, 1):
            if self._cancel_event is not None and self._cancel_event.is_set():
                self._log("[kg] 收到取消信号，构建中止（已完成部分已入库）")
                graph.save_snapshot(snapshot_path)
                raise KGCancellation()
            text = (chunk.get("text") or "").strip()
            chunk_id = chunk.get("id") or ""
            source = chunk.get("source") or ""
            if not text or not chunk_id:
                continue
            try:
                entities, relations = self._extractor.extract_entities_relations(
                    text, section_context=f"来源文档: {source}")
                self._commit_extraction(entities, relations, chunk_id, source)
                stats["entities"] += len(entities)
                stats["relationships"] += len(relations)
                stats["chunks"] += 1
            except KGCancellation:
                raise
            except Exception as e:  # 单片失败不中断整体
                stats["errors"] += 1
                logger.warning("KG: 切片 %s 抽取失败: %s", chunk_id, e)
                self._log(f"[kg] ⚠ 切片 {chunk_id} 抽取失败: {e}")
            self._progress(done, total, f"知识图谱抽取 {done}/{total}")
            if done % SNAPSHOT_EVERY == 0:
                self._refresh_graph(graph)
                graph.save_snapshot(snapshot_path)

        # 全部抽取完成 → 向量化 → 建图 → 最终快照（spec 第六步）
        self._log("[kg] 抽取完成，开始实体/关系向量化...")
        self._progress(total, total, "知识图谱向量化")
        n_vec = self._vectorize()
        self._log(f"[kg] 向量化完成: {n_vec} 条向量")
        self._refresh_graph(graph)
        graph.save_snapshot(snapshot_path)
        stats["vectors"] = n_vec
        stats["graph"] = graph.stats()
        stats["elapsed"] = round(time.monotonic() - t0, 1)
        self._log(f"[kg] ✅ 知识图谱构建完成: {stats}")
        self._progress(total, total, "知识图谱构建完成")
        return stats

    # ------------------------------------------------------------------

    def _commit_extraction(self, entities, relations, chunk_id: str,
                           source: str) -> None:
        """单片抽取结果入库（聚合 + 描述归纳）。"""
        # 先实体：upsert 返回 (id, created)；多条描述时触发 LLM 归纳。
        entity_names = []
        for ent in entities:
            name = ent.name.strip()
            if not name:
                continue
            entity_names.append(name)
            eid, _created = self._kg_store.upsert_entity(
                name, ent.type, ent.description, chunk_id, file_path=source)
            self._maybe_summarize_entity(name)
        # 关系：引用了不存在的实体 → 现场建 UNKNOWN 节点（spec 第六步）。
        for rel in relations:
            src, dst = rel.src.strip(), rel.dst.strip()
            if not src or not dst:
                continue
            for name, known in ((src, src in entity_names), (dst, dst in entity_names)):
                if not known:
                    self._kg_store.upsert_entity(
                        name, "UNKNOWN", "", chunk_id, file_path=source)
            self._kg_store.upsert_relationship(
                src, dst, rel.keywords, rel.description, chunk_id, file_path=source)
            self._maybe_summarize_relationship(src, dst)

    def _maybe_summarize_entity(self, name: str) -> None:
        row = self._kg_store.get_entity_by_key(normalize_name(name))
        if not row:
            return
        descs = self._raw_descriptions("entities", row["id"])
        if len(descs) < _SUMMARY_MIN_DESCS:
            return
        merged = self._extractor.summarize_descriptions(row["name"], descs)
        if merged:
            with self._kg_store._lock:
                self._kg_store._conn.execute(
                    "UPDATE entities SET description=? WHERE id=?", (merged, row["id"]))
                self._kg_store._conn.commit()

    def _maybe_summarize_relationship(self, src: str, dst: str) -> None:
        row = self._kg_store.get_relationship_by_pair(pair_key(src, dst))
        if not row:
            return
        descs = self._raw_descriptions("relationships", row["id"])
        if len(descs) < _SUMMARY_MIN_DESCS:
            return
        display = f"{row['src_entity']} → {row['dst_entity']}"
        merged = self._extractor.summarize_descriptions(display, descs)
        if merged:
            with self._kg_store._lock:
                self._kg_store._conn.execute(
                    "UPDATE relationships SET description=? WHERE id=?", (merged, row["id"]))
                self._kg_store._conn.commit()

    def _raw_descriptions(self, table: str, row_id: str) -> List[str]:
        from .storage import _json_loads
        col = "descriptions"
        with self._kg_store._lock:
            row = self._kg_store._conn.execute(
                f"SELECT {col} FROM {table} WHERE id=?", (row_id,)).fetchone()
        return _json_loads(row[0], []) if row else []

    # ------------------------------------------------------------------

    def _vectorize(self) -> int:
        """全部待向量化行 → int8 向量入库（增量：content_hash 未变则跳过）。"""
        kg = self._kg_store
        count = 0
        # 实体
        ent_texts, ent_keys = [], []
        for e in kg.iter_entities():
            text = KGEmbedder.entity_text(e["name"], e["description"])
            vid = f"ent:{e['id']}"
            h = KGEmbedder.content_hash(text)
            if kg.get_vector_hash(vid) == h:
                continue
            ent_texts.append(text)
            ent_keys.append((vid, "entity", e["id"], h))
        # 关系（正反序两个 id）
        rel_texts, rel_keys = [], []
        for r in kg.iter_relationships():
            for reverse in (False, True):
                text = KGEmbedder.relationship_text(
                    r["src_entity"], r["dst_entity"], r["keywords"],
                    r["description"], reverse=reverse)
                vid = f"rel:{r['id']}:{'rev' if reverse else 'fwd'}"
                h = KGEmbedder.content_hash(text)
                if kg.get_vector_hash(vid) == h:
                    continue
                rel_texts.append(text)
                rel_keys.append((vid, "relationship", r["id"], h))
        for texts, keys in ((ent_texts, ent_keys), (rel_texts, rel_keys)):
            if not texts:
                continue
            vecs = self._embedder.embed_for_index(texts)
            for (vid, ttype, tid, h), vec in zip(keys, vecs):
                kg.put_vector(vid, ttype, tid, h,
                              np.asarray(vec, dtype=np.int8).tobytes())
                count += 1
        return count

    def _refresh_graph(self, graph: KnowledgeGraph) -> None:
        graph.rebuild_from_store(self._kg_store)


# ---------------------------------------------------------------------------
# 供 engine / UI 使用的轻量入口
# ---------------------------------------------------------------------------

def load_or_rebuild_graph(rag_dir: str, store) -> Optional[KnowledgeGraph]:
    """检索期加载图：优先快照，缺失/损坏时从 KGStore 重建（内存图，不落盘）。"""
    graph = KnowledgeGraph()
    snap = snapshot_path_for(rag_dir)
    if graph.load_snapshot(snap):
        return graph
    kg_store = KGStore(kg_dir_for(rag_dir))
    try:
        if kg_store.counts()["entities"] == 0:
            return None
        graph.rebuild_from_store(kg_store)
        return graph
    finally:
        kg_store.close()
