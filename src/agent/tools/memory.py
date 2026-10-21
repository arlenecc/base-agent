"""work_memory and knowledge-graph memory tools.

WorkMemory stays a simple key-value scratchpad (unchanged).

LongTermMemory is now a knowledge graph (entities + relations + observations)
backed by GraphMemoryStore with LanceDB semantic search.  Three tools expose
the graph to the agent:

  - work_memory         — short-term scratchpad (set/get/list/clear)
  - memory_graph        — CRUD on the knowledge graph
  - memory_search       — semantic recall over observations
"""
from __future__ import annotations

import collections
import json
import logging
import os
import threading
from typing import Any, Dict, List

from ..config import AgentConfig
from ..memory import LongTermMemory, WorkMemory
from .base import Tool, ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Singleton WorkMemory — avoids atexit-handler accumulation.
#
# Each WorkMemory wraps a _JsonStore that registers an atexit flush closure
# on construction.  If we created a new instance on every call, each tool
# invocation would append one more closure to the atexit registry, growing
# without bound for the lifetime of the process (memory leak).  The instance
# is cached per workspace path on the module level so it survives registry
# rebuilds (work memory data must persist across "应用配置" anyway).
# ---------------------------------------------------------------------------

_work_memory_by_path: "collections.OrderedDict" = collections.OrderedDict()
_work_memory_lock = threading.Lock()
# 缓存上限：GUI 固定一个 workspace 时永远只命中第 1 条；但测试套件每个
# 用例用唯一 tmp_path，无上限的缓存会把数百个 WorkMemory（各持一份 JSON
# 内存副本）驻留整个进程生命周期。按最近使用淘汰，超限时把最久未用的那
# 份先 flush 再丢弃（避免丢防抖窗口里的未落盘写入）。
_WORK_MEMORY_CACHE_MAX = 8


def _work_memory(config: AgentConfig) -> WorkMemory:
    path = os.path.join(config.workspace, ".agent", "work_memory.json")
    # move_to_end 与锁内的 popitem 同为 OrderedDict 变更操作，必须同锁：
    # 旧实现在锁外 move_to_end，与另一个线程的淘汰路径并发修改 →
    # RuntimeError（dict changed size during iteration 一类）或条目错乱。
    with _work_memory_lock:
        wm = _work_memory_by_path.get(path)
        if wm is not None:
            _work_memory_by_path.move_to_end(path)   # 最近使用
            return wm
        wm = WorkMemory(path=path)
        _work_memory_by_path[path] = wm
        if len(_work_memory_by_path) > _WORK_MEMORY_CACHE_MAX:
            _old_path, old_wm = _work_memory_by_path.popitem(last=False)
            try:
                # force=True：绕过 0.5s 防抖间隔，把窗口里的写入真正落盘
                # 再丢弃（旧实现漏传 force，淘汰即丢数据）。
                old_wm.flush(force=True)
            except Exception:  # pragma: no cover - best-effort
                pass
    return wm


# ---------------------------------------------------------------------------
# Singleton LongTermMemory — avoids leaking ONNX/LanceDB resources.
#
# Each LongTermMemory wraps a GraphMemoryStore that lazily loads a FastEmbed
# ONNX model (~137MB).  If we created a new instance on every call, every
# tool invocation and every fact extraction would leak ~137MB of C++ heap
# that Python GC can't reclaim promptly.  The singleton is cached on the
# ToolRegistry so shutdown() can release it.
# ---------------------------------------------------------------------------

_LONG_MEMORY_SINGLETON_ATTR = "_long_memory_instance"


def _get_long_memory(registry: "ToolRegistry", config: AgentConfig) -> LongTermMemory:
    """Return the singleton LongTermMemory, creating it on first access."""
    existing = getattr(registry, _LONG_MEMORY_SINGLETON_ATTR, None)
    if existing is not None:
        return existing
    # 首次创建是重活（懒加载 ~137MB ONNX + LanceDB 连接），agent 线程与 UI
    # 线程可能并发首次调用：不加锁会各建一个实例，后写者覆盖属性，先建的
    # 连接/模型泄漏且 shutdown() 收不回。与 _work_memory 相同的双重检查模式。
    with registry._singleton_lock:
        existing = getattr(registry, _LONG_MEMORY_SINGLETON_ATTR, None)
        if existing is not None:
            return existing
        emb = getattr(config, "rag_embedding_model", "nomic-ai/nomic-embed-text-v1.5-Q")
        lm = LongTermMemory(
            path=os.path.join(config.workspace, ".agent", "long_memory.json"),
            embedding_model=emb,
        )
        setattr(registry, _LONG_MEMORY_SINGLETON_ATTR, lm)
        return lm


def _long_memory(config: AgentConfig) -> LongTermMemory:
    """Deprecated: creates an UNMANAGED LongTermMemory (caller must close()).

    保留仅为兼容旧调用方；正常路径一律用 ``_get_long_memory``——单例才能被
    ``ToolRegistry.shutdown()`` 回收（每个漏 close 的实例泄漏 ~137MB ONNX）。
    """
    # 只警告一次：agent.py 的 _system_prompt 每轮都调这个函数，旧实现每次
    # 一条 warning，日志直接被洪水刷掉。
    global _long_memory_warned
    if not _long_memory_warned:
        _long_memory_warned = True
        logger.warning("tools.memory._long_memory is deprecated; use _get_long_memory")
    emb = getattr(config, "rag_embedding_model", "nomic-ai/nomic-embed-text-v1.5-Q")
    return LongTermMemory(
        path=os.path.join(config.workspace, ".agent", "long_memory.json"),
        embedding_model=emb,
    )


_long_memory_warned = False

# 工具输出体积兜底：与 mcp_client 的 _cap_output 同策略（200K + 截断标记）。
_MAX_TOOL_OUTPUT_CHARS = 200_000


def _cap_output(text: str) -> str:
    """把工具输出钳到 _MAX_TOOL_OUTPUT_CHARS，超长部分截断并明确标注。"""
    if len(text) <= _MAX_TOOL_OUTPUT_CHARS:
        return text
    return (text[:_MAX_TOOL_OUTPUT_CHARS]
            + f"\n… [输出超过 {_MAX_TOOL_OUTPUT_CHARS} 字符，已截断]")


# ---------------------------------------------------------------------------
# Work memory tool (unchanged)
# ---------------------------------------------------------------------------


class WorkMemoryTool(Tool):
    name = "work_memory"
    description = "Task scratchpad. op: set(key,value), get(key), list, clear."
    parameters = {
        "type": "object",
        "properties": {
            "op": {"type": "string", "enum": ["set", "get", "list", "clear"]},
            "key": {"type": "string"},
            "value": {"type": "string"},
        },
        "required": ["op"],
    }

    config: AgentConfig
    registry: ToolRegistry

    def bind(self, config: AgentConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry

    def run(self, op: str, key: str = "", value: str = "") -> ToolResult:
        wm = _work_memory(self.config)
        if op == "set":
            if not key:
                return ToolResult(False, error="op 'set' requires 'key'")
            wm.set(key, value)
            return ToolResult(True, output=f"set {key}")
        if op == "get":
            if not key:
                return ToolResult(False, error="op 'get' requires 'key'")
            v = wm.get(key)
            return ToolResult(True, output=v if v is not None else "")
        if op == "list":
            return ToolResult(True, output=json.dumps(wm.list(), ensure_ascii=False))
        if op == "clear":
            wm.clear()
            return ToolResult(True, output="cleared")
        return ToolResult(False, error=f"unknown op: {op}")


# ---------------------------------------------------------------------------
# Knowledge-graph CRUD tool
# ---------------------------------------------------------------------------


class MemoryGraphTool(Tool):
    """CRUD operations on the long-term knowledge graph.

    Operations:
      - create_entity:  create/merge an entity (args: name, type?, observations?)
      - delete_entity:  remove an entity and its relations (arg: name)
      - add_observations: add facts to an entity (args: name, observations)
      - create_relation: create a directed edge (args: source, target, label)
      - list_entities:  list all entities (no extra args)
      - get_entity:     get one entity's full record (arg: name)
      - snapshot:       compact digest for the prompt (no extra args)
      - clear:          wipe the entire graph
    """
    name = "memory_graph"
    description = (
        "Long-term knowledge graph CRUD. op: create_entity(name,type?,observations?), "
        "delete_entity(name), add_observations(name,observations), "
        "create_relation(source,target,label), list_entities, get_entity(name), "
        "snapshot, clear."
    )
    parameters = {
        "type": "object",
        "properties": {
            "op": {
                "type": "string",
                "enum": [
                    "create_entity", "delete_entity", "add_observations",
                    "create_relation", "list_entities", "get_entity",
                    "snapshot", "clear",
                ],
            },
            "name": {"type": "string"},
            "entity_type": {"type": "string", "description": "person/project/concept/tool/place/other"},
            "observations": {"type": "array", "items": {"type": "string"}},
            "source": {"type": "string"},
            "target": {"type": "string"},
            "label": {"type": "string"},
        },
        "required": ["op"],
    }

    config: AgentConfig
    registry: ToolRegistry

    def bind(self, config: AgentConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry

    def should_confirm(self, args: Dict[str, Any]) -> bool:
        """clear / delete_entity 破坏整个图谱或整块记忆，必须弹确认。

        默认策略只看 ``destructive`` 类属性；这两个 op 与文件读写无关，
        旧实现不弹确认，模型一次误调用（或幻觉参数）就清空长期记忆。
        """
        return str((args or {}).get("op", "")).strip().lower() in (
            "clear", "delete_entity",
        )

    def run(self, op: str, name: str = "", entity_type: str = "",
            observations: List[str] = None, source: str = "",
            target: str = "", label: str = "") -> ToolResult:
        lm = _get_long_memory(self.registry, self.config)
        g = lm.graph
        if op == "create_entity":
            if not name:
                return ToolResult(False, error="op 'create_entity' requires 'name'")
            e = g.create_entity(name, entity_type=entity_type, observations=observations or [])
            return ToolResult(True, output=json.dumps(e, ensure_ascii=False))
        if op == "delete_entity":
            if not name:
                return ToolResult(False, error="op 'delete_entity' requires 'name'")
            ok = g.delete_entity(name)
            return ToolResult(True, output="deleted" if ok else "not found")
        if op == "add_observations":
            if not name or not observations:
                return ToolResult(False, error="op 'add_observations' requires 'name' and 'observations'")
            # observations 必须是字符串列表：模型传字符串时旧实现按字符迭代，
            # 生成几十条逐字符 observation 污染图谱。
            if isinstance(observations, str) or not isinstance(observations, (list, tuple)):
                return ToolResult(
                    False,
                    error="'observations' must be a list of strings, "
                          f"got {type(observations).__name__}",
                )
            observations = [str(o) for o in observations if o and str(o).strip()]
            if not observations:
                return ToolResult(False, error="'observations' has no non-empty entry")
            added = g.add_observations(name, observations)
            return ToolResult(True, output=f"added {added} observation(s)")
        if op == "create_relation":
            if not source or not target or not label:
                return ToolResult(False, error="op 'create_relation' requires 'source', 'target', 'label'")
            ok = g.create_relation(source, target, label)
            return ToolResult(True, output="created" if ok else "duplicate or invalid")
        if op == "list_entities":
            entities = g.list_entities()
            # Compact: name + type + observation count.
            # 图谱可能积累数百实体，全量塞进上下文会挤掉真正要处理的内容；
            # 截断并提示用 get_entity 精确查询。
            _MAX_LIST = 200
            summary = [
                f"{e['name']} ({e.get('type','?')}) [{len(e.get('observations',[]))} obs]"
                for e in entities[:_MAX_LIST]
            ]
            if len(entities) > _MAX_LIST:
                summary.append(f"… (共 {len(entities)} 个实体，仅显示前 {_MAX_LIST} 个；"
                               f"用 get_entity 查询具体实体)")
            return ToolResult(True, output=_cap_output(json.dumps(summary, ensure_ascii=False)))
        if op == "get_entity":
            if not name:
                return ToolResult(False, error="op 'get_entity' requires 'name'")
            e = g.get_entity(name)
            # 单个实体的 observations 是累积写入的，几十次 add 后可达数十 KB，
            # 直接全量返回会把上下文撑爆。按条数 + 总长双重截断。
            if isinstance(e, dict) and isinstance(e.get("observations"), list):
                _MAX_OBS = 50
                _MAX_OBS_CHARS = 4000
                kept, used = [], 0
                for o in e["observations"]:
                    if len(kept) >= _MAX_OBS or used >= _MAX_OBS_CHARS:
                        break
                    s = str(o)
                    if len(s) > 400:
                        s = s[:400] + "…"
                    kept.append(s)
                    used += len(s)
                dropped = len(e["observations"]) - len(kept)
                if dropped > 0:
                    kept.append(f"… (其余 {dropped} 条观察已省略)")
                e = dict(e)
                e["observations"] = kept
            return ToolResult(True, output=_cap_output(
                json.dumps(e, ensure_ascii=False)) if e else "not found")
        if op == "snapshot":
            snap = g.snapshot(max_items=10)
            return ToolResult(True, output=snap or "(empty)")
        if op == "clear":
            g.clear()
            return ToolResult(True, output="cleared")
        return ToolResult(False, error=f"unknown op: {op}")


# ---------------------------------------------------------------------------
# Semantic search tool
# ---------------------------------------------------------------------------


class MemorySearchTool(Tool):
    """Semantic search over long-term memory observations."""
    name = "memory_search"
    description = "Semantic search over long-term memory observations. Returns matching facts with entity and relevance score."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to recall — use the user's own natural phrasing; never translate or transliterate",
            },
            "top_k": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    }

    config: AgentConfig
    registry: ToolRegistry

    def bind(self, config: AgentConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry

    def run(self, query: str, top_k: int = 5) -> ToolResult:
        # 模型可能传 null 或超大 top_k：None 会让 embed 失败，大 top_k 会把
        # 全部观察灌进上下文。归一化 + 钳制（与 rag_search 一致）。
        query = (query or "").strip()
        if not query:
            return ToolResult(False, error="请提供要回忆的内容（query）")
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = 5
        top_k = max(1, min(top_k, 20))
        lm = _get_long_memory(self.registry, self.config)
        results = lm.graph.search(query, top_k=top_k)
        # 统一体积兜底：单个观察可达数十 KB，20 条巨块全塞进上下文会挤爆
        # 窗口（与 mcp_client 的 200K 策略一致）。
        return ToolResult(True, output=_cap_output(json.dumps(results, ensure_ascii=False)))
