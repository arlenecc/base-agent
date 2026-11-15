"""In-process memory stores: short-term work memory and long-term memory."""
from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import time
import weakref
from typing import Dict, List, Optional

from .atomic_io import atomic_write_json, backup_corrupt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 进程级 atexit 刷新注册表
#
# 旧实现里每个 _JsonStore 实例在 __init__ 中调用 atexit.register(...)，但从不
# unregister。atexit 内部持有回调的强引用，因此每创建一个 store（LongTermMemory
# 重建、workspace 切换…）都会永久新增一条回调及其闭包引用，进程生命周期内单调
# 增长 —— 典型的缓慢内存泄漏。
#
# 现在改为：只注册一个全局 atexit 回调，遍历 WeakSet 中仍存活的 store。store
# 被 GC 后自动从 WeakSet 消失，回调数量恒为 1，且不会阻止 store 被回收。
# ---------------------------------------------------------------------------

_LIVE_STORES: "weakref.WeakSet" = weakref.WeakSet()
_LIVE_STORES_LOCK = threading.Lock()
_ATEXIT_REGISTERED = False


def _flush_all_stores() -> None:
    for store in list(_LIVE_STORES):
        try:
            store.flush(force=True)
        except Exception:  # pragma: no cover - 退出路径，尽力而为
            pass


def _register_atexit_flush(store) -> None:
    """把任何带 ``flush(force=True)`` 的 store 挂到全局退出刷新表上。

    不只服务 ``_JsonStore``：``GraphMemoryStore`` 同样用了防抖写，退出前
    必须落盘（否则防抖窗口内的改动静默丢失）。
    """
    global _ATEXIT_REGISTERED
    with _LIVE_STORES_LOCK:
        _LIVE_STORES.add(store)
        if not _ATEXIT_REGISTERED:
            atexit.register(_flush_all_stores)
            _ATEXIT_REGISTERED = True


class _JsonStore:
    """A tiny thread-safe JSON key/value file store.

    Uses a delayed-write (debounce) strategy: set() / delete() only mark
    _dirty = True.  The next _flush() call (or the next set() after 0.5s)
    actually writes to disk.  This avoids O(N²) I/O when the agent calls
    work_memory.set() in a tight loop.
    """

    # Minimum interval between two _save() calls (seconds).
    _FLUSH_INTERVAL = 0.5

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._data: dict = {}
        self._dirty = False
        self._last_save = 0.0
        self._load()
        # Register for the process-level atexit flush (one global callback for
        # all stores — see _register_atexit_flush for why we don't call
        # atexit.register() per instance).
        _register_atexit_flush(self)

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("memory: failed to load %s: %s", self.path, e)
                backup_corrupt(self.path)
                self._data = {}

    def reload(self) -> None:
        """Re-read the file from disk, discarding the in-memory cache.

        Used after an external write (e.g. another thread / process saved the
        same store file) so this instance picks up the latest data.

        必须先把未落盘的脏数据强制写回：防抖策略（_FLUSH_INTERVAL=0.5s）下，
        刚 set() 的内容可能还只在内存里。旧实现直接 _load() 用磁盘内容整体
        覆盖 _data——「set 后 0.5s 内 reload」会把那次 set 静默丢弃。实际触发
        场景：UI 在 create_skill() 后立刻 reload()（main_window），若距上一轮
        record_request 的写入不足 0.5s，用户刚固化的技能直接丢失。
        """
        with self._lock:
            if self._dirty:
                self._save()
            self._load()

    def _save(self) -> None:
        """Atomic write. Must be called under _lock."""
        try:
            atomic_write_json(self.path, self._data)
            self._dirty = False
            self._last_save = time.monotonic()
        except OSError as e:
            logger.warning("memory: failed to save %s: %s", self.path, e)

    def _flush(self) -> None:
        """Flush pending changes to disk if dirty and interval elapsed."""
        if not self._dirty:
            return
        if time.monotonic() - self._last_save >= self._FLUSH_INTERVAL:
            self._save()

    def flush(self, force: bool = False) -> None:
        """Force-write pending changes to disk.

        When ``force=True`` the debounce interval is bypassed — used by the
        atexit hook and by callers that need the data on disk immediately
        (e.g. before spawning a subprocess that reads the same store file).
        """
        with self._lock:
            if self._dirty and (force or time.monotonic() - self._last_save >= self._FLUSH_INTERVAL):
                self._save()

    def set(self, key: str, value) -> None:
        with self._lock:
            self._data[key] = value
            self._dirty = True
            self._flush()

    def get(self, key: str, default=None):
        with self._lock:
            return self._data.get(key, default)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)
            self._dirty = True
            self._flush()

    def all(self) -> dict:
        with self._lock:
            return dict(self._data)

    def replace(self, data: dict) -> None:
        with self._lock:
            self._data = dict(data)
            self._dirty = True
            # replace is always a big change — save immediately.
            self._save()


class WorkMemory:
    """A working notepad the agent reads/writes during a task."""

    def __init__(self, path: str):
        self._store = _JsonStore(path)

    def set(self, key: str, value: str) -> None:
        self._store.set(key, value)

    def get(self, key: str) -> Optional[str]:
        return self._store.get(key)

    def list(self) -> Dict[str, str]:
        return {k: v for k, v in self._store.all().items()}

    def clear(self) -> None:
        self._store.replace({})

    def flush(self, force: bool = False) -> None:
        """把防抖窗口里未落盘的写入立即写盘（淘汰缓存实例前调用）。

        ``force=True`` 绕过 0.5s 防抖间隔——淘汰路径必须用 True，否则
        刚 set() 的内容会被静默丢弃（旧实现没透传 force，注释里写的
        「先落盘再丢弃」实际做不到）。
        """
        self._store.flush(force=force)


class LongTermMemory:
    """Knowledge-graph long-term memory backed by GraphMemoryStore.

    Replaces the old flat fact-list with a structured graph of entities,
    observations, and relations.  Observations are embedded via FastEmbed
    (nomic-embed-text-v1.5-Q) and indexed in LanceDB for semantic recall.

    The public API stays compatible with the old one (add / add_many /
    search / all / clear) so callers that treat memory as a string list
    keep working — each fact is stored as a single observation on a
    generic "General" entity.  The graph CRUD methods are exposed via
    the ``graph`` attribute (a GraphMemoryStore).
    """

    def __init__(self, path: str, embedding_model: str = "nomic-ai/nomic-embed-text-v1.5-Q"):
        from .graph_memory import GraphMemoryStore
        # path is like .../long_memory.json; the graph file is the same
        # path but with .json → .graph.json so old flat-memory files are
        # not clobbered.
        # 用切片而不是 replace：路径中更早的位置若也含 ".json"（如
        # /data.json/long_memory.json），replace 会把它们一起改掉，得到一个
        # 错误的目录路径。
        graph_path = path[:-5] + ".graph.json" if path.endswith(".json") else path + ".graph"
        self._graph = GraphMemoryStore(path=graph_path, embedding_model=embedding_model)
        self._legacy_store = _JsonStore(path)
        # Migrate old flat facts into the graph on first use.
        self._migrate_legacy_facts()

    @property
    def graph(self) -> "GraphMemoryStore":
        """Direct access to the underlying knowledge-graph store."""
        return self._graph

    def _migrate_legacy_facts(self) -> None:
        """One-time migration: old long_memory.json had {"facts": [...]}."""
        facts = self._legacy_store.get("facts")
        if not isinstance(facts, list) or not facts:
            return
        # Only migrate facts not already in the graph.
        existing = {o for e in self._graph.list_entities() for o in e.get("observations", [])}
        new = [f for f in facts if f and f not in existing]
        if new:
            self._graph.add_observations("General", new)
            logger.info("long_memory: migrated %d legacy facts into graph", len(new))
        # Clear the old file so we don't re-migrate.
        self._legacy_store.set("facts", [])

    # -- compatible API (thin wrappers over the graph) --

    def add(self, fact: str) -> None:
        """Add a single fact as an observation on the General entity."""
        self._graph.add_observations("General", [fact])

    def add_many(self, facts: List[str]) -> None:
        """Add many facts as observations on the General entity."""
        self._graph.add_observations("General", facts)

    def all(self) -> List[str]:
        """Return all observations across all entities (flat list, legacy)."""
        out = []
        for e in self._graph.list_entities():
            out.extend(e.get("observations", []))
        return out

    def search(self, query: str, top_k: int = 8) -> List[str]:
        """Semantic search over observations. Returns observation texts."""
        results = self._graph.search(query, top_k=top_k)
        return [r["text"] for r in results]

    def clear(self) -> None:
        self._graph.clear()

    def close(self) -> None:
        self._graph.close()

    # -- snapshot for prompt injection --

    def snapshot(self, max_items: int = 10) -> str:
        """Compact text digest for prompt injection."""
        return self._graph.snapshot(max_items=max_items)
