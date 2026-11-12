"""知识图谱构建流程（spec 第六/八/九步）。

入口 :func:`build_knowledge_graph`：
1. 遍历向量库全部切片（VectorStore 新增 ``iter_all_chunks``）；
2. 每片用辅助模型抽取实体/关系，写入 entities/relationships 表
   （同名实体/无向同对关系聚合，多描述交给 LLM 用
   ``summarize_entity_descriptions`` prompt 归纳）；**入库前先做实体消歧**：
   同一实体的不同写法（简称/全称/括号注释/机构后缀/全半角）解析到已有实体，
   证据并入已有行而不新增实体，关系端点改写为规范名（entity_resolver.py）；
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

from .embedding import KGEmbedder
from .entity_resolver import EntityResolver
from .extractor import KGExtractor
from .graph import KnowledgeGraph
from .storage import KGStore, normalize_name, pair_key

logger = logging.getLogger(__name__)

SNAPSHOT_EVERY = 50          # 每 N 片落一次快照
_SUMMARY_MIN_DESCS = 2       # 描述数达到该值才触发 LLM 归纳
# 单片抽取硬上限（与 prompts.py 的约束一致）：模型不守约时在此兜底，
# 描述超长按 100 字截断，避免单条描述把后续上下文挤爆。
MAX_ENTITIES_PER_CHUNK = 20
MAX_RELATIONS_PER_CHUNK = 50
MAX_DESCRIPTION_CHARS = 100


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
        resolver: Optional[EntityResolver] = None,
    ):
        self._rag_dir = rag_dir
        self._store = store
        self._extractor = extractor
        self._progress = progress_callback or (lambda d, t, m: None)
        self._cancel_event = cancel_event

        def _emit_log(msg: str) -> None:
            # 终端也打一份：UI 日志窗口可能没打开，用户从终端启动时只能靠
            # logger 看到进度（否则构建期间表现为「完全没有任何信息」）。
            logger.info("%s", msg)
            if log_callback is not None:
                log_callback(msg)

        self._log = _emit_log
        self._kg_store = KGStore(kg_dir_for(rag_dir))
        self._embedder = KGEmbedder(store.embedding_function)
        # 实体消歧（入库前解析到已有实体，不重复建行）。
        self._resolver = resolver or EntityResolver(self._kg_store)
        # 描述归纳延后：抽取循环里只登记，全部抽取结束后统一归纳。抽取期间若
        # 夹杂 summarize 调用（kv_group="aux"）会 reset GGUF/ONNX 的底层上下文、
        # 切换 MLX 缓存，下一片抽取就得重新预填充 system+任务说明——稳定前缀
        # 复用失效。延后同时消掉了「每多一条描述就重归纳一遍」的重复调用。
        self._defer_summaries = False
        self._pending_entities: Dict[str, str] = {}      # name_key → 展示名
        self._pending_relations: Dict[str, tuple] = {}   # pair_key → (src, dst)

    def close(self) -> None:
        self._kg_store.close()

    # ------------------------------------------------------------------

    def build(self, limit: Optional[int] = None,
              doc_filter: Optional[str] = None,
              rebuild: bool = False) -> Dict[str, Any]:
        """执行完整构建。返回统计 dict。

        doc_filter 非空时只抽取 source 路径包含该子串的文档切片；此时先
        **清掉这些文档的旧图谱证据再重抽**（幂等：同文档重复 ingest 不叠加，
        文档改版后旧 chunk 悬空的证据也随之清除）。
        rebuild=True 时清空全部图谱数据（实体/关系/向量）后全量重建——
        用于切换抽取方案后的重建，或旧数据被污染后的自愈。
        """
        t0 = time.monotonic()
        if rebuild:
            self._log("[kg] 重建模式：清空已有图谱数据（实体/关系/向量）...")
            self._kg_store.clear()
        chunks = list(self._store.iter_all_chunks())
        if doc_filter:
            chunks = [c for c in chunks
                      if doc_filter in (c.get("source") or "")]
            # 按文档重建：清掉这些文档的旧证据（前缀 = chunk_id 去掉 _idx，
            # 同一路径 hash 的历史 chunk 全部覆盖，含文档改版后已消失的行）。
            prefixes = sorted({(c.get("id") or "").rsplit("_", 1)[0]
                               for c in chunks if c.get("id")})
            if prefixes:
                purged = self._kg_store.purge_document(prefixes)
                if purged.get("entities") or purged.get("relationships"):
                    self._log(f"[kg] 已清理该文档旧图谱证据: "
                              f"实体 {purged['entities']} / 关系 {purged['relationships']}")
        if limit:
            chunks = chunks[:limit]
        total = len(chunks)
        if doc_filter and total == 0:
            raise ValueError(
                f"知识库中未找到匹配 {doc_filter!r} 的文档切片"
                "（请确认文档已同步入库）")
        self._log(f"[kg] 开始构建知识图谱: {total} 个切片, "
                  f"实体/关系抽取模型={'辅助模型' if self._extractor.available else '（未配置，跳过抽取）'}")
        self._progress(0, total, "知识图谱构建开始")
        if not self._extractor.available:
            reason = getattr(self._extractor, "unavailable_reason", "") or ""
            raise RuntimeError(reason or "抽取器不可用：知识图谱构建无法继续")

        stats = {"chunks": 0, "entities": 0, "relationships": 0, "errors": 0,
                 "merged_entities": 0}
        snapshot_path = snapshot_path_for(self._rag_dir)
        graph = KnowledgeGraph()
        if not rebuild:
            # 增量续建：已有快照则热身内存图（rebuild 时库已清空，
            # 热身旧快照会在取消时把旧图存回去，与空库不一致）。
            graph.load_snapshot(snapshot_path)

        # 存量整理：入库前把历史遗留的重复实体合并掉（不新增实体），
        # 并补全别名索引，保证后续 resolve 全部走 O(1) 索引。
        self._log("[kg] 实体消歧：扫描已有实体，合并同义重复项...")
        cons = self._resolver.consolidate()
        if cons.get("merged"):
            self._log(f"[kg] 实体消歧完成：扫描 {cons['scanned']} 个实体，"
                      f"合并 {cons['merged']} 个重复实体")

        # 抽取阶段：描述归纳只登记不执行（见 __init__ 中的说明），保证整轮
        # 抽取的 LLM 调用都是「同一 system+任务说明 + 不同切片正文」的稳定
        # 前缀，后端只需 prefill 变化的正文段。
        self._defer_summaries = True
        try:
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
                    created, merged = self._commit_extraction(
                        entities, relations, chunk_id, source)
                    stats["entities"] += created
                    stats["merged_entities"] += merged
                    stats["relationships"] += len(relations)
                    stats["chunks"] += 1
                except KGCancellation:
                    raise
                except Exception as e:  # 单片失败不中断整体
                    stats["errors"] += 1
                    logger.warning("KG: 切片 %s 抽取失败: %s", chunk_id, e)
                    self._log(f"[kg] ⚠ 切片 {chunk_id} 抽取失败: {e}")
                self._progress(done, total, f"知识图谱抽取 {done}/{total}")
                if done % 20 == 0 or done == total:
                    self._log(f"[kg] 抽取进度 {done}/{total}"
                              f"（新增实体 {stats['entities']}，"
                              f"关系 {stats['relationships']}）")
                if done % SNAPSHOT_EVERY == 0:
                    self._refresh_graph(graph)
                    graph.save_snapshot(snapshot_path)
            # 抽取全部结束 → 统一归纳多描述实体/关系（此时才允许打断前缀）
            self._flush_pending_summaries()
        finally:
            self._defer_summaries = False
            self._pending_entities.clear()
            self._pending_relations.clear()

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
                           source: str) -> tuple:
        """单片抽取结果入库（实体消歧合并 + 聚合 + 描述归纳）。

        返回 ``(新建实体数, 合并到已有实体的次数)``。实体写入前一律先经
        消歧器解析：同义写法命中已有实体则合并入库，关系端点统一用规范名，
        因此同一实体在库中只会有一行。
        """
        name_map: Dict[str, str] = {}
        created = 0
        merged = 0
        # 兜底执行 prompt 的数量/长度约束（模型不守约时防止洪泛）：
        # 上限取自抽取器（配置项 rag_kg_max_*），与 prompt 渲染用的是同一份值。
        max_ent = int(getattr(self._extractor, "max_entities",
                              MAX_ENTITIES_PER_CHUNK))
        max_rel = int(getattr(self._extractor, "max_relations",
                              MAX_RELATIONS_PER_CHUNK))
        max_desc = int(getattr(self._extractor, "max_desc_chars",
                               MAX_DESCRIPTION_CHARS))
        entities = list(entities)[:max_ent]
        relations = list(relations)[:max_rel]
        for ent in entities:
            name = (ent.name or "").strip()
            if not name:
                continue
            canonical, is_new, was_merged = self._resolver.merge_or_create(
                name, ent.type, (ent.description or "")[:max_desc],
                chunk_id, source)
            name_map[name] = canonical
            created += 1 if is_new else 0
            merged += 1 if was_merged else 0
            self._summarize_or_queue_entity(canonical)
        # 关系：引用了未抽取的实体 → 同样先消歧（命中则复用已有实体），
        # 未命中才现场建 UNKNOWN 节点（spec 第六步）。
        for rel in relations:
            src, dst = (rel.src or "").strip(), (rel.dst or "").strip()
            if not src or not dst:
                continue
            src_c = self._canonical_endpoint(name_map, src, chunk_id, source)
            dst_c = self._canonical_endpoint(name_map, dst, chunk_id, source)
            if normalize_name(src_c) == normalize_name(dst_c):
                # 消歧合并后两端归一为同一实体，自环关系无意义。
                continue
            self._kg_store.upsert_relationship(
                src_c, dst_c, rel.keywords,
                (rel.description or "")[:max_desc], chunk_id,
                file_path=source)
            self._summarize_or_queue_relationship(src_c, dst_c)
        return created, merged

    def _canonical_endpoint(self, name_map: Dict[str, str], name: str,
                            chunk_id: str, source: str) -> str:
        """关系端点的规范名（同片内已解析过的直接复用，避免重复消歧）。"""
        if name in name_map:
            return name_map[name]
        canonical, _is_new, _merged = self._resolver.merge_or_create(
            name, "UNKNOWN", "", chunk_id, source)
        name_map[name] = canonical
        return canonical

    # -- 描述归纳（抽取期延后） -------------------------------------------

    def _summarize_or_queue_entity(self, name: str) -> None:
        """抽取期只登记待归纳实体，其余场景立即归纳。"""
        if self._defer_summaries:
            self._pending_entities.setdefault(normalize_name(name), name)
            return
        self._maybe_summarize_entity(name)

    def _summarize_or_queue_relationship(self, src: str, dst: str) -> None:
        if self._defer_summaries:
            self._pending_relations.setdefault(pair_key(src, dst), (src, dst))
            return
        self._maybe_summarize_relationship(src, dst)

    def _flush_pending_summaries(self) -> None:
        """统一归纳抽取期登记的实体/关系（每个只归纳一次）。"""
        items = ([(True, name) for name in self._pending_entities.values()]
                 + [(False, pair) for pair in self._pending_relations.values()])
        self._pending_entities.clear()
        self._pending_relations.clear()
        if not items:
            return
        self._log(f"[kg] 抽取完成，开始归纳多描述实体/关系（待检查 {len(items)} 项）...")
        for i, (is_entity, payload) in enumerate(items, 1):
            if self._cancel_event is not None and self._cancel_event.is_set():
                self._log("[kg] 收到取消信号，描述归纳中止（已完成部分保留）")
                return
            if is_entity:
                try:
                    self._maybe_summarize_entity(payload)
                except Exception as e:  # 归纳是增强项，失败不影响已入库数据
                    logger.warning("KG: 实体描述归纳失败（%s）: %s", payload, e)
            else:
                try:
                    self._maybe_summarize_relationship(*payload)
                except Exception as e:
                    logger.warning("KG: 关系描述归纳失败（%s）: %s", payload, e)
            if i % 10 == 0 or i == len(items):
                self._log(f"[kg] 描述归纳进度 {i}/{len(items)}")

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
        return vectorize_pending_rows(self._kg_store, self._embedder)

    def _refresh_graph(self, graph: KnowledgeGraph) -> None:
        graph.rebuild_from_store(self._kg_store)


def vectorize_pending_rows(kg_store: "KGStore", embedder: "KGEmbedder") -> int:
    """增量向量化 KGStore 中未向量化/内容变化的实体与关系行。

    独立成模块级函数：构建流程与 MCP 手工增删实体/关系后的补向量化共用。
    """
    import numpy as np

    kg = kg_store
    count = 0
    # 实体
    ent_texts, ent_keys = [], []
    for e in kg.iter_entities():
        text = KGEmbedder.entity_text(e["name"], e["description"])
        vid = f"ent:{e['id']}"
        h = embedder.content_hash(text)
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
            h = embedder.content_hash(text)
            if kg.get_vector_hash(vid) == h:
                continue
            rel_texts.append(text)
            rel_keys.append((vid, "relationship", r["id"], h))
    for texts, keys in ((ent_texts, ent_keys), (rel_texts, rel_keys)):
        if not texts:
            continue
        vecs = embedder.embed_for_index(texts)
        for (vid, ttype, tid, h), vec in zip(keys, vecs):
            kg.put_vector(vid, ttype, tid, h,
                          np.asarray(vec, dtype=np.int8).tobytes())
            count += 1
    return count


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
