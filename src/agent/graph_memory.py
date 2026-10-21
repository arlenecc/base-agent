"""Knowledge-graph long-term memory: entities, relations, observations.

Replaces the flat fact-list LongTermMemory with a structured graph that
mirrors the Memory MCP server model:

- Entity: a named node (person, project, concept, ...) with a type and
  a list of observations (free-text facts attached to the entity).
- Relation: a directed, labeled edge between two entities
  (source --[label]--> target).

All data is persisted in a single JSON file.  Observations are also
embedded via FastEmbed (nomic-embed-text-v1.5-Q) and indexed in LanceDB
so the agent can do semantic recall instead of keyword matching.

The graph is intentionally simple — no graph database, no query language.
Just dicts, lists, and a vector index.  This keeps the runtime dependency
surface flat while still giving the agent structured, searchable memory.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from .atomic_io import atomic_write_json, backup_corrupt
from .memory import _register_atexit_flush

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 规模上限（防无上限增长）
#
# 旧实现里实体 / observation 只增不减（去重仅按文本相等），而每次写入都全量
# 序列化整个图谱 → JSON 体积与内存随时间单调增长，写放大 O(N)。后台事实抽取
# 线程每轮都在追加，一个长跑 workspace 很容易攒到几万条 observation。
#
# 上限按「最近优先」淘汰：observation 保留最新的 N 条，实体按 created_at
# 淘汰最旧的。被淘汰的条目同步从向量索引里删掉，避免「JSON 删了、向量还在」
# 导致已删事实仍被 search() 召回。
# ---------------------------------------------------------------------------
_MAX_OBS_PER_ENTITY = 200
_MAX_ENTITIES = 5_000
_MAX_RELATIONS = 20_000
# 落盘最小间隔：与 memory.py 的 _JsonStore 同一策略，把连续写入合并成一次。
_SAVE_MIN_INTERVAL = 0.5

# ---------------------------------------------------------------------------
# Data model (plain dicts; no dataclasses to keep JSON round-trip trivial)
# ---------------------------------------------------------------------------
#
#   entity  = {name, type, observations: [str], created_at}
#   relation = {source: name, target: name, label: str}
#
# The graph is keyed by entity *name* (case-insensitive).  Names must be
# unique — creating an entity whose name already exists merges observations.
# ---------------------------------------------------------------------------


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _norm_name(name: str) -> str:
    """Normalise an entity name for case-insensitive dedup."""
    return name.strip()


def _normalize_vector(vec: List[float]) -> List[float]:
    """Normalize a vector to unit length.

    LanceDB's default L2 distance only maps to cosine similarity
    (``1 - d²/2``) when vectors are unit-normalized.  FastEmbed outputs are
    NOT normalized, so raw L2 distances are huge (~300+), making the
    ``1 - d²/2`` similarity collapse to 0 for every pair.  Normalize both on
    write and on query.
    """
    norm = 0.0
    for x in vec:
        norm += x * x
    norm = norm ** 0.5
    if norm == 0:
        return list(vec)
    return [x / norm for x in vec]


def _hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Graph store
# ---------------------------------------------------------------------------


class GraphMemoryStore:
    """Thread-safe knowledge-graph store with LanceDB semantic search.

    Persistence:
        <path>              — the graph JSON (entities + relations)
        <path>.vectors/     — LanceDB table for observation embeddings

    The vector index is *optional*: if LanceDB or FastEmbed is unavailable
    (e.g. dependencies not installed, model download failed), the store
    still works — ``search`` just falls back to keyword matching.
    """

    def __init__(self, path: str, embedding_model: str = "nomic-ai/nomic-embed-text-v1.5-Q"):
        self._path = path
        self._vector_dir = path + ".vectors"
        self._embedding_model_name = embedding_model
        self._lock = threading.Lock()
        self._vec_lock = threading.Lock()  # protects vector-index lazy init
        self._backfill_lock = threading.Lock()  # 回填判重 + 串行化（见 _ensure_vectors）
        # _pending_index 的专用锁：抽取守护线程与工具线程会并发入队/取出，
        # 旧实现无锁 swap+extend，极端交错下丢重试项 →「图里有、向量库没有」。
        self._pending_lock = threading.Lock()
        self._entities: Dict[str, dict] = {}  # name → entity
        self._relations: List[dict] = []
        # 关系判重索引（source.lower, target.lower, label.lower) → set：
        # create_relation 旧实现每次全表线性扫描判重并持全局锁，关系密集时
        # 写入退化为 O(N²)（上限 _MAX_RELATIONS=20000）。列表只在 prune /
        # delete / clear / load 这些**整体重排**的路径上变更，届时重建索引。
        self._relation_keys: set = set()
        self._db = None
        self._table = None
        self._ef = None
        self._vec_ready = False
        # 向量索引只探测一次；lancedb/fastembed 缺失时不反复重试 connect+import。
        self._vec_tried = False
        self._backfilled = False
        # 索引失败的 (entity, obs) 待重试队列：ONNX/LanceDB 瞬时失败不再造成
        # 「JSON 图里有、向量库里没有」的永久检索缺口 —— 下次写入或检索时重试。
        # 有上限：嵌入持续失败（模型下载不了）时队列只增不减，会把每条
        # observation 的文本驻留内存。超限丢最旧并告警。
        self._pending_index: List[Tuple[str, str]] = []
        self._pending_index_max = 500
        # 防抖写状态 + 淘汰后的向量清理队列（见 _prune_locked）
        self._dirty = False
        self._last_save = 0.0
        self._prune_cleanup: List[Tuple[str, str]] = []   # (entity, observation)
        self._prune_entities: List[str] = []
        self._load()
        # 防抖写必须挂退出刷新，否则进程退出时窗口内的改动静默丢失。
        _register_atexit_flush(self)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 结构校验：手改坏/旧版本/写一半崩溃的 JSON 只要「能解析」
            # 就会带着畸形结构进入内存——随后 _prune_locked / snapshot /
            # list_relations 对 None/缺字段记录做下标访问，把整个记忆系统
            # 打炸（不只是坏那条）。与 entities 一样，relations 也必须归一化。
            if not isinstance(data, dict):
                raise ValueError("graph root is not an object")
            raw_entities = data.get("entities", {})
            if not isinstance(raw_entities, dict):
                raise ValueError("graph entities is not an object")
            entities: Dict[str, dict] = {}
            for n, e in raw_entities.items():
                if not isinstance(n, str) or not isinstance(e, dict):
                    continue                      # 跳过畸形实体
                e.setdefault("name", n)
                e.setdefault("type", "")
                if not isinstance(e.get("observations"), list):
                    e["observations"] = []
                entities[n.lower()] = e
            raw_relations = data.get("relations", [])
            if not isinstance(raw_relations, list):
                raise ValueError("graph relations is not a list")
            relations: List[dict] = []
            for r in raw_relations:
                if not isinstance(r, dict):
                    continue                      # 跳过畸形关系
                src = str(r.get("source", "") or "").strip()
                tgt = str(r.get("target", "") or "").strip()
                lbl = str(r.get("label", "") or "").strip()
                if not (src and tgt and lbl):
                    continue                      # 缺字段的关系没有语义
                relations.append({
                    "source": src,
                    "target": tgt,
                    "label": lbl,
                    "created": str(r.get("created", "") or ""),
                })
            self._entities = entities
            self._relations = relations
            self._rebuild_relation_keys_locked()
        except (OSError, json.JSONDecodeError, ValueError) as e:
            logger.warning("graph_memory: failed to load %s: %s", self._path, e)
            backup_corrupt(self._path)
            self._entities = {}
            self._relations = []
            self._relation_keys = set()

    def _rebuild_relation_keys_locked(self) -> None:
        """重建关系判重索引。必须在 ``_lock`` 内调用。

        只在 load / prune / delete_entity / clear 这些整体重排路径上调用；
        create_relation 走 O(1) 增量维护，不做全表重建。
        """
        self._relation_keys = {
            (r["source"].lower(), r["target"].lower(), r["label"].lower())
            for r in self._relations
        }

    @staticmethod
    def _relation_key(source: str, target: str, label: str) -> tuple:
        return (source.lower(), target.lower(), label.lower())

    def _save(self, force: bool = False) -> None:
        """Atomic write (debounced unless ``force``). Must be called under _lock.

        旧实现每次写入都全量序列化整个图谱：observation 只增不减，长跑后一次
        add_observations 就要重写几 MB JSON（写放大 O(N)）。这里加最小间隔
        合并写，与 memory.py 的 _JsonStore 同一策略；``flush()`` / atexit 走
        ``force=True`` 保证退出前落盘。
        """
        self._dirty = True
        now = time.monotonic()
        if not force and (now - self._last_save) < _SAVE_MIN_INTERVAL:
            return
        try:
            payload = {
                "entities": self._entities,
                "relations": self._relations,
            }
            atomic_write_json(self._path, payload)
            self._dirty = False
            self._last_save = now
        except OSError as e:
            logger.warning("graph_memory: failed to save %s: %s", self._path, e)

    def flush(self, force: bool = True) -> None:
        """Force pending changes to disk (退出前 / 显式同步时调用）。"""
        with self._lock:
            if self._dirty:
                self._save(force=force)

    # ------------------------------------------------------------------
    # 规模裁剪（防无上限增长）
    # ------------------------------------------------------------------

    def _prune_locked(self) -> None:
        """把图谱压回上限内。必须在 ``_lock`` 内调用。

        淘汰的 observation / 实体记进 ``_prune_cleanup`` / ``_prune_entities``，
        由调用方在**锁外**执行向量清理（``_deindex_observations`` 内部要物化
        全表，锁内执行会阻塞所有图读取）。
        """
        # 1) 每实体 observation 上限：保留最新的（最新的事实最有用）。
        for key, e in self._entities.items():
            obs = e.get("observations") or []
            if len(obs) > _MAX_OBS_PER_ENTITY:
                dropped = obs[: len(obs) - _MAX_OBS_PER_ENTITY]
                e["observations"] = obs[len(obs) - _MAX_OBS_PER_ENTITY:]
                name = e.get("name") or key
                for o in dropped:
                    self._prune_cleanup.append((name, o))
                logger.info(
                    "graph_memory: pruned %d oldest observations of %r "
                    "(cap=%d)", len(dropped), name, _MAX_OBS_PER_ENTITY)
        # 2) 实体总数上限：按 created_at 淘汰最旧的（无时间戳的排最后）。
        if len(self._entities) > _MAX_ENTITIES:
            ordered = sorted(
                self._entities.items(),
                key=lambda kv: (str(kv[1].get("created_at") or ""), kv[0]),
            )
            excess = len(self._entities) - _MAX_ENTITIES
            for key, e in ordered[:excess]:
                self._prune_entities.append(e.get("name") or key)
                del self._entities[key]
            # 关系里的悬边一并清掉，否则会出现指向不存在实体的边。
            if self._prune_entities:
                dead = {n.lower() for n in self._prune_entities}
                self._relations = [
                    r for r in self._relations
                    if r["source"].lower() not in dead and r["target"].lower() not in dead
                ]
                self._rebuild_relation_keys_locked()
            logger.info(
                "graph_memory: evicted %d oldest entities (cap=%d)",
                excess, _MAX_ENTITIES)
        # 3) 关系总数上限：丢最旧的。
        if len(self._relations) > _MAX_RELATIONS:
            drop = len(self._relations) - _MAX_RELATIONS
            del self._relations[:drop]
            self._rebuild_relation_keys_locked()
            logger.info("graph_memory: pruned %d oldest relations (cap=%d)",
                        drop, _MAX_RELATIONS)

    def _cleanup_pruned(self) -> None:
        """锁外执行淘汰条目的向量清理（幂等、失败只记日志）。"""
        while True:
            with self._lock:
                if not (self._prune_cleanup or self._prune_entities):
                    return
                obs_pairs = self._prune_cleanup
                entities = self._prune_entities
                self._prune_cleanup = []
                self._prune_entities = []
            for name in entities:
                try:
                    self._deindex_observations(name)
                except Exception as e:  # pragma: no cover - 尽力而为
                    logger.debug("graph_memory: deindex %r failed: %s", name, e)
            for name, obs in obs_pairs:
                try:
                    self._deindex_observation(name, obs)
                except Exception as e:  # pragma: no cover
                    logger.debug("graph_memory: deindex obs of %r failed: %s", name, e)

    # ------------------------------------------------------------------
    # vector index (lazy init)
    # ------------------------------------------------------------------

    def _ensure_vectors(self) -> None:
        """Lazy-init LanceDB + FastEmbed. Non-fatal on failure.

        Uses a separate _vec_lock with double-checked locking so that
        concurrent callers (e.g. agent thread + tool thread) don't
        race into loading the ONNX model twice.

        On first successful init, backfills any observations already present
        in the JSON graph but missing from the vector table (so facts stored
        before the vector index existed become searchable too).
        """
        if self._vec_ready or self._vec_tried:
            return
        with self._vec_lock:
            if self._vec_ready or self._vec_tried:  # double-check
                return
            try:
                import lancedb
                from .rag.vector_store import get_or_create_embedding_function
                os.makedirs(self._vector_dir, exist_ok=True)
                self._db = lancedb.connect(self._vector_dir)
                # Share the process-wide ONNX model with RAG's VectorStore
                # to avoid loading a second ~137MB copy.
                self._ef = get_or_create_embedding_function(self._embedding_model_name)
                if "observations" in self._db.table_names():
                    self._table = self._db.open_table("observations")
                self._vec_ready = True
            except Exception as e:
                # Non-fatal: semantic search just falls back to keyword.
                logger.debug("graph_memory: vector index unavailable: %s", e)
                self._vec_ready = False
                # 已 connect 成功但后续（EF 获取 / open_table）失败时，
                # self._db 是悬挂引用——旧实现留着它走到 keyword 分支，
                # 连接在 close() 前一直开着。这里回滚，避免句柄泄漏。
                if self._db is not None:
                    try:
                        self._db = None
                    except Exception:
                        pass
                self._table = None
                self._ef = None
            finally:
                # 只尝试一次：lancedb/fastembed 缺失的环境下，旧的
                # 「失败不记忆」会让每次 search() 都重新 connect + import
                # （并且每次都把同样那条 debug 日志刷出来）。
                self._vec_tried = True
        # Backfill outside the lock (embedding is slow). Only once per instance.
        #
        # 两个坑都在这里：
        #  1) 标志必须在**成功后**才置位。放在前面时，回填中途失败（模型下载
        #     中断、某批 embedding 抛异常）也会被记成「已回填」，于是永远不
        #     重试 → JSON 里有、向量库里没有的永久检索缺口。
        #  2) 判重也必须在锁内，否则两个线程同时通过检查、各自回填一遍 →
        #     同一条 observation 的重复向量行（memory_search 在并发白名单里，
        #     首次调用极易并发进入）。
        if self._vec_ready and not getattr(self, "_backfilled", False):
            with self._backfill_lock:
                if self._backfilled:
                    return
                try:
                    self._backfill_vectors()
                except Exception as e:
                    logger.debug("graph_memory: backfill failed: %s", e)
                    return          # 失败不置位 → 下次还会再试
                self._backfilled = True

    def _backfill_vectors(self) -> None:
        """Index any observations present in the graph JSON but missing from
        the vector table (e.g. facts stored before the vector index existed)."""
        # Snapshot entities under lock.
        with self._lock:
            pairs = [
                (e["name"], obs)
                for e in self._entities.values()
                for obs in e.get("observations", [])
            ]
        if not pairs:
            return
        table = self._ensure_table()
        # Determine which ids are already indexed.
        existing_ids: set = set()
        if table is not None:
            try:
                # 列投影下推到存储层：表里有 768 维向量列，全表物化会把它们
                # 全部读进内存（与 skill_retriever / vector_store 同一修法）。
                rows = (
                    table.search()
                    .limit(max(table.count_rows(), 1))
                    .select(["id"])
                    .to_arrow()
                    .to_pylist()
                )
                existing_ids = {r.get("id", "") for r in rows}
            except Exception:
                try:
                    existing_ids = {r.get("id", "") for r in table.to_arrow().to_pylist()}
                except Exception:
                    existing_ids = set()
        # Collect missing (entity, observation) pairs.
        missing = [
            (name, obs)
            for name, obs in pairs
            if _hash(name + "|" + obs) not in existing_ids
        ]
        if not missing:
            return
        # Embed + insert missing observations in batches of 100.
        texts = [obs for _, obs in missing]
        for i in range(0, len(missing), 100):
            batch = missing[i:i + 100]
            batch_texts = [obs for _, obs in batch]
            try:
                vecs = self._ef.embed_documents(batch_texts)
            except Exception as e:
                logger.warning("graph_memory: backfill embed failed: %s", e)
                return
            records = [
                {
                    "id": _hash(name + "|" + obs),
                    "entity": name,
                    "text": obs,
                    "vector": _normalize_vector(vec),
                }
                for (name, obs), vec in zip(batch, vecs)
            ]
            with self._vec_lock:
                if table is None:
                    self._table = self._db.create_table("observations", records)
                    table = self._table
                else:
                    table.add(records)
        logger.info("graph_memory: backfilled %d observations into vector index", len(missing))

    def _ensure_table(self):
        if self._table is not None:
            return self._table
        if self._db is not None and "observations" in self._db.table_names():
            self._table = self._db.open_table("observations")
        return self._table

    def _deindex_observations(self, entity_name: str) -> None:
        """Remove all vectors for an entity (e.g. after entity deletion)."""
        if not self._vec_ready:
            return
        try:
            table = self._ensure_table()
            if table is None:
                return
            # JSON 里实体按 name.lower() 建键，但向量行的 entity 列存的是
            # **创建时的原始大小写**。用 "postgres" 删以 "PostgreSQL" 建的
            # 实体时，字面量过滤匹配不上 → JSON 删掉了、向量还在，已删实体
            # 仍会被 search() 召回。这里先按归一化后的名字解析出实际存过的
            # 所有拼写，再逐个删。
            #
            # 单引号仍要转义：实体名来自 LLM，不转义会让过滤表达式畸形或被注入。
            target = (entity_name or "").strip().lower()
            spellings = set()
            try:
                # 只投影 entity 列（同 _backfill_vectors：避免物化向量列）。
                rows = (
                    table.search()
                    .limit(max(table.count_rows(), 1))
                    .select(["entity"])
                    .to_arrow()
                    .to_pylist()
                )
                for r in rows:
                    ent = str(r.get("entity") or "")
                    if ent.strip().lower() == target:
                        spellings.add(ent)
            except Exception:
                spellings = set()
            for ent in (spellings or {entity_name}):
                table.delete(f"entity = '{ent.replace(chr(39), chr(39) * 2)}'")
        except Exception:
            pass

    def _deindex_observation(self, entity_name: str, observation: str) -> None:
        """Remove a single observation's vector row (after it was pruned).

        与 ``_deindex_observations`` 分开：淘汰是按 observation 粒度做的，
        不能因为一条超限就把整个实体的向量都删掉。
        """
        if not self._vec_ready:
            return
        try:
            table = self._ensure_table()
            if table is None:
                return
            row_id = _hash((entity_name or "") + "|" + (observation or ""))
            table.delete(f"id = '{row_id}'")
        except Exception as e:
            logger.debug("graph_memory: deindex observation failed: %s", e)

    def _index_observations_batch(self, entity_name: str, observations: List[str]) -> None:
        """Embed and upsert multiple observations in ONE batch.

        Called OUTSIDE self._lock.  Batching matters: each individual
        embed_query() costs ~10-50ms of ONNX compute, so adding N facts
        from a fact-extraction pass would take N× that.  embed_documents()
        processes the whole list in a single batched ONNX forward pass.

        Lazily initialises the vector index on first use (``_ensure_vectors``)
        — previously it bailed out if the index wasn't already ready, which
        meant the very first ``add_observations`` (before any ``search``)
        silently skipped indexing and the vector table was never built.
        """
        if not observations:
            return
        self._ensure_vectors()
        if not self._vec_ready:
            return
        # 先重试上次失败的条目。必须按各自实体名分组重试 —— 直接并进当前
        # batch 会把 A 实体的 observation 记到 B 实体名下。重试再失败会重新
        # 入队（本方法的 except 分支），下次继续。
        # 取出操作必须持 _pending_lock：与入队方（另一个线程的 except 分支）
        # 无锁交错时会丢条目。
        with self._pending_lock:
            pending, self._pending_index = self._pending_index, []
        if pending:
            grouped: Dict[str, List[str]] = {}
            for ent, obs in pending:
                grouped.setdefault(ent, []).append(obs)
            for ent, obs_list in grouped.items():
                self._index_observations_batch(ent, obs_list)
        try:
            vecs = self._ef.embed_documents(observations)
            if len(vecs) != len(observations):
                # embedding 返回数量不足（模型异常/截断）时不能静默用 zip 截断
                # ——被丢掉的内容已写进 JSON 图谱，会造成「图里有、向量库没有」
                # 的永久性检索缺口。这里宁可整批失败（外层有兜底），也不留
                # 半截索引。
                raise RuntimeError(
                    f"embedding returned {len(vecs)} vectors for "
                    f"{len(observations)} observations"
                )
            records = [
                {
                    "id": _hash(entity_name + "|" + obs),
                    "entity": entity_name,
                    "text": obs,
                    "vector": _normalize_vector(vec),
                }
                for obs, vec in zip(observations, vecs)
            ]
            with self._vec_lock:
                table = self._ensure_table()
                if table is None:
                    self._table = self._db.create_table("observations", records)
                    return
                for rec in records:
                    try:
                        table.delete(f"id = '{rec['id']}'")
                    except Exception:
                        pass
                table.add(records)
        except Exception as e:
            logger.debug("graph_memory: index_observations_batch failed: %s", e)
            # 入队待重试：下次写入或 search() 时补上，不留「图里有、向量库没有」的缺口。
            # 持锁入队 + 有界：并发下不丢条目，嵌入持续失败也不会无限驻留文本。
            with self._pending_lock:
                self._pending_index.extend((entity_name, o) for o in observations)
                if len(self._pending_index) > self._pending_index_max:
                    dropped = len(self._pending_index) - self._pending_index_max
                    del self._pending_index[:dropped]
                    logger.warning(
                        "graph_memory: pending index queue overflow, dropped %d oldest", dropped
                    )

    # ------------------------------------------------------------------
    # CRUD: entities
    # ------------------------------------------------------------------

    def create_entity(self, name: str, entity_type: str = "", observations: Optional[List[str]] = None) -> dict:
        """Create an entity, or merge observations if it already exists."""
        name = _norm_name(name)
        to_index: List[str] = []  # collected under lock, indexed outside
        with self._lock:
            key = name.lower()
            if key in self._entities:
                e = self._entities[key]
                if entity_type and not e.get("type"):
                    e["type"] = entity_type
                if observations:
                    for obs in observations:
                        obs = obs.strip()
                        if obs and obs not in e["observations"]:
                            e["observations"].append(obs)
                            to_index.append(obs)
            else:
                e = {
                    "name": name,
                    "type": entity_type,
                    "observations": [o.strip() for o in (observations or []) if o.strip()],
                    "created_at": _now(),
                }
                self._entities[key] = e
                to_index = list(e["observations"])
            self._prune_locked()
            self._save()
            result = dict(e)
        # Index outside lock (batched: one ONNX pass for all observations)
        if to_index:
            self._index_observations_batch(name, to_index)
        self._cleanup_pruned()
        return result

    def delete_entity(self, name: str) -> bool:
        """Delete an entity and all its relations."""
        name = _norm_name(name)
        with self._lock:
            key = name.lower()
            if key not in self._entities:
                return False
            del self._entities[key]
            self._relations = [
                r for r in self._relations
                if r["source"].lower() != key and r["target"].lower() != key
            ]
            self._rebuild_relation_keys_locked()
            self._save(force=True)
        # 向量删除移到锁外：_deindex_observations 内部是全表物化 + 逐条 delete，
        # 锁内执行会阻塞所有图读取（同 _index_observations_batch 的设计理由）。
        self._deindex_observations(name)
        return True

    def get_entity(self, name: str) -> Optional[dict]:
        key = _norm_name(name).lower()
        with self._lock:
            e = self._entities.get(key)
            return dict(e) if e else None

    def list_entities(self) -> List[dict]:
        with self._lock:
            return [dict(e) for e in self._entities.values()]

    # ------------------------------------------------------------------
    # CRUD: observations
    # ------------------------------------------------------------------

    def add_observations(self, name: str, observations: List[str]) -> int:
        """Add observations to an entity. Creates the entity if missing."""
        name = _norm_name(name)
        cleaned = [o.strip() for o in observations if o and o.strip()]
        if not cleaned:
            return 0
        to_index: List[str] = []
        with self._lock:
            key = name.lower()
            if key not in self._entities:
                # create entity lazily so callers don't need a separate step
                self._entities[key] = {
                    "name": name,
                    "type": "",
                    "observations": [],
                    "created_at": _now(),
                }
            e = self._entities[key]
            for obs in cleaned:
                if obs not in e["observations"]:
                    e["observations"].append(obs)
                    to_index.append(obs)
            self._prune_locked()
            if to_index or self._prune_cleanup:
                self._save()
        # Index outside lock (batched: one ONNX pass for all observations)
        if to_index:
            self._index_observations_batch(name, to_index)
        self._cleanup_pruned()
        return len(to_index)

    # ------------------------------------------------------------------
    # CRUD: relations
    # ------------------------------------------------------------------

    def create_relation(self, source: str, target: str, label: str) -> bool:
        """Create a directed edge source --[label]--> target."""
        source = _norm_name(source)
        target = _norm_name(target)
        label = label.strip()
        if not source or not target or not label:
            return False
        result_ok = False
        with self._lock:
            # Ensure both entities exist
            for n in (source, target):
                k = n.lower()
                if k not in self._entities:
                    self._entities[k] = {
                        "name": n,
                        "type": "",
                        "observations": [],
                        "created_at": _now(),
                    }
            # Dedup: same source+target+label = same relation（O(1) 索引判定，
            # 旧实现每次全表线性扫描并持全局锁，关系密集时写入退化 O(N²)）
            rk = self._relation_key(source, target, label)
            if rk in self._relation_keys:
                return False
            self._relations.append({
                "source": source,
                "target": target,
                "label": label,
            })
            self._relation_keys.add(rk)
            self._prune_locked()
            self._save()
            result_ok = True
        self._cleanup_pruned()
        return result_ok

    def list_relations(self) -> List[dict]:
        with self._lock:
            return [dict(r) for r in self._relations]

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 8) -> List[dict]:
        """Semantic search over observations.

        Returns a list of {entity, text, score} dicts sorted by relevance.
        Falls back to keyword substring match when the vector index is
        unavailable.
        """
        if not query.strip():
            return []
        self._ensure_vectors()
        if self._vec_ready and self._ef is not None:
            # 检索前先把积压的待重试条目补上 —— 语义检索不应因一次瞬时
            # 失败而永远漏掉某条 observation。
            with self._pending_lock:
                has_pending = bool(self._pending_index)
            if has_pending:
                self._flush_pending_index()
            return self._semantic_search(query, top_k)
        return self._keyword_search(query, top_k)

    def _flush_pending_index(self) -> None:
        with self._pending_lock:
            pending, self._pending_index = self._pending_index, []
        grouped: Dict[str, List[str]] = {}
        for ent, obs in pending:
            grouped.setdefault(ent, []).append(obs)
        for ent, obs_list in grouped.items():
            self._index_observations_batch(ent, obs_list)

    def _semantic_search(self, query: str, top_k: int) -> List[dict]:
        try:
            table = self._ensure_table()
            if table is None:
                return self._keyword_search(query, top_k)
            qvec = _normalize_vector(self._ef.embed_query(query))
            results = table.search(qvec).limit(top_k).to_list()
            out = []
            for r in results:
                dist = float(r.get("_distance", 1.0))
                score = max(0.0, 1.0 - (dist * dist) / 2.0)
                # 相似度阈值：过滤语义上不相关的结果（否则无关查询也会
                # 返回低分命中，破坏"无匹配返回空"的语义）。
                if score < 0.5:
                    continue
                out.append({
                    "entity": r.get("entity", ""),
                    "text": r.get("text", ""),
                    "score": round(score, 4),
                })
            return out
        except Exception as e:
            logger.debug("graph_memory: semantic search failed: %s", e)
            return self._keyword_search(query, top_k)

    def _keyword_search(self, query: str, top_k: int = 8) -> List[dict]:
        """Keyword substring fallback.

        ``top_k`` **必须**生效：这是没有装 rag 可选依赖时的默认路径，若不限制
        条数，一次宽泛查询会把全部 observation 灌进上下文（其他检索工具都做了
        钳制，这里不能例外）。
        """
        q = query.lower()
        out = []
        with self._lock:
            for e in self._entities.values():
                for obs in e["observations"]:
                    if q in obs.lower():
                        out.append({"entity": e["name"], "text": obs, "score": 0.5})
                        if len(out) >= top_k:
                            return out
        return out

    # ------------------------------------------------------------------
    # snapshot (for prompt injection)
    # ------------------------------------------------------------------

    def snapshot(self, max_items: int = 10) -> str:
        """Compact text snapshot of the most salient graph facts.

        Used for prompt injection: returns a short, readable digest of
        entities/observations/relations, capped at ``max_items`` items
        so it doesn't bloat the system prompt.
        """
        with self._lock:
            entities = list(self._entities.values())
            relations = list(self._relations)
        if not entities and not relations:
            return ""
        lines = []
        # Prioritise entities with the most observations
        entities.sort(key=lambda e: len(e.get("observations", [])), reverse=True)
        shown = 0
        for e in entities:
            if shown >= max_items:
                break
            obs = e.get("observations", [])
            if not obs:
                continue
            typ = f" ({e['type']})" if e.get("type") else ""
            lines.append(f"- {e['name']}{typ}: {'; '.join(obs[:3])}")
            shown += 1
        # A few relations
        for r in relations[:max_items // 2]:
            lines.append(f"- {r['source']} --[{r['label']}]--> {r['target']}")
        return "\n".join(lines) if lines else ""

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def clear(self) -> None:
        with self._lock:
            self._entities = {}
            self._relations = []
            self._relation_keys = set()
            self._save(force=True)
        # Drop vector table
        if self._db is not None:
            try:
                if "observations" in self._db.table_names():
                    self._db.drop_table("observations")
            except Exception:
                pass
        self._table = None
        # 必须一并清空待重试队列：此前嵌入失败入队的条目对应的 JSON 实体已删，
        # clear 后任何一次 add/search 都会把它们写回新表 → 用户删掉的事实
        # 又被语义检索召回（「删除复活」的向量库变体）。
        with self._pending_lock:
            self._pending_index.clear()

    def close(self) -> None:
        """Release resources.

        The FastEmbed ONNX model is a process-wide shared singleton
        (get_or_create_embedding_function) — we only drop our reference,
        we don't release the session, because RAG or another caller may
        still be using it.  The LanceDB connection is private and safe
        to close.
        """
        # 真正关闭 LanceDB 连接：每次「应用配置」重建 Agent 都会走这里，
        # 只把引用置 None 会泄漏连接及其对象存储缓存。
        if self._db is not None:
            try:
                close_fn = getattr(self._db, "close", None)
                if callable(close_fn):
                    close_fn()
            except Exception as e:
                logger.debug("graph_memory: closing LanceDB connection failed: %s", e)
        self._db = None
        self._table = None
        self._ef = None
        self._vec_ready = False
        # 允许重新初始化：否则 close 后 _ensure_vectors 会提前返回，
        # 后续（重建的）store 永远拿不到向量能力。
        self._vec_tried = False
        # 一并重置回填标志：close 复活的实例向量表重新打开，若 _backfilled
        # 仍为 True，已有 JSON observation 不会再回填 → 语义检索静默残缺。
        self._backfilled = False
        # 关闭前把防抖窗口内尚未落盘的改动写出去（_save 默认只标脏）。
        try:
            self.flush()
        except Exception as e:  # pragma: no cover - 退出路径尽力而为
            logger.debug("graph_memory: flush on close failed: %s", e)


# ---------------------------------------------------------------------------
# LLM-based extraction: pull entities / relations / observations from text
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM = """\
Extract durable facts from the conversation as a knowledge graph. Return ONLY JSON:
{
  "entities": [{"name": "name", "type": "person|project|concept|tool|place|other", "observations": ["fact"]}],
  "relations": [{"source": "name", "target": "name", "label": "verb phrase"}]
}

Rules:
- Only concrete, durable facts about the user, their projects, preferences,
  decisions, or domain knowledge. Skip chitchat, transient state, file contents
  and raw tool output.
- Skip the assistant's own tools/capabilities (no entities for "rag_search", etc.).
- Observations are short self-contained sentences still useful weeks later.
- Canonical entity names ("PostgreSQL", not "postgres db").
- Write observations in the same language as the conversation (mismatched
  language makes later semantic recall fail).
- Extract at most 5 entities and 3 observations each — keep only the most
  durable. Exceeding this pollutes memory and inflates every later prompt.
- If nothing to persist, return {"entities": [], "relations": []}.
- Valid JSON only, no markdown fences or commentary.\
"""

_EXTRACT_USER = """\
Extract knowledge-graph facts from this conversation fragment (≤5 entities).

--- Conversation ---
{conversation}
--- End ---
"""


def extract_facts_via_llm(
    llm: Any,
    conversation: str,
    max_chars: int = 4000,
) -> dict:
    """Call the LLM to extract entities/relations/observations from text.

    Returns a dict with 'entities' and 'relations' lists.  On any error
    (parse failure, LLM error) returns an empty graph — callers should
    treat extraction as best-effort, never blocking.

    Args:
        llm: an object with chat_stream(messages, tools, temperature)
             (same interface as LLMClient).
        conversation: the conversation text to extract from.
        max_chars: truncate overly long inputs to stay within the LLM's
                   context budget.
    """
    if not conversation.strip():
        return {"entities": [], "relations": []}

    # Truncate to a reasonable window so we don't blow the context budget.
    text = conversation[:max_chars]

    messages = [
        {"role": "system", "content": _EXTRACT_SYSTEM},
        {"role": "user", "content": _EXTRACT_USER.format(conversation=text)},
    ]
    try:
        out: List[str] = []
        for ev in llm.chat_stream(messages, tools=None, temperature=0.1,
                                  kv_group="aux"):
            if ev.type == "content":
                out.append(ev.content)
            elif ev.type == "done":
                break
        raw = "".join(out).strip()
        # Tolerate markdown-fenced JSON from models that ignore instructions.
        if raw.startswith("```"):
            # Strip ```json ... ``` wrapper
            lines = raw.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw = "\n".join(lines).strip()
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {"entities": [], "relations": []}
        # 结构归一化：LLM 常把实体返成字符串、observations 混入非 str、relation
        # 缺字段。旧实现只 setdefault 就返回，下游 ent.get()/o.strip() 抛
        # AttributeError → 被 agent 的 except 吞掉，**整轮实体/关系全部丢弃**。
        # 这里逐条容错（与 skills.py 的单条容错一致），坏条目跳过、好条目保留。
        entities = []
        for ent in data.get("entities", []):
            if not isinstance(ent, dict):
                continue
            name = str(ent.get("name", "") or "").strip()
            if not name:
                continue
            raw_obs = ent.get("observations", [])
            if isinstance(raw_obs, str):
                raw_obs = [raw_obs]
            elif not isinstance(raw_obs, (list, tuple)):
                raw_obs = []
            obs = [str(o).strip() for o in raw_obs if o is not None and str(o).strip()]
            entities.append({
                "name": name,
                "type": str(ent.get("type", "") or ""),
                "observations": obs,
            })
        relations = []
        for rel in data.get("relations", []):
            if not isinstance(rel, dict):
                continue
            src = str(rel.get("source", "") or "").strip()
            tgt = str(rel.get("target", "") or "").strip()
            lbl = str(rel.get("label", "") or "").strip()
            if src and tgt and lbl:
                relations.append({"source": src, "target": tgt, "label": lbl})
        return {"entities": entities, "relations": relations}
    except Exception as e:
        logger.debug("graph_memory: extract_facts failed: %s", e)
        return {"entities": [], "relations": []}
