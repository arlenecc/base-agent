"""内存知识图谱 + 磁盘快照缓存（spec 第六步，networkx 实现）。

需求文档以 Swift 的 SwiftGraph/Codable 描述本层；Python 侧映射为：
* 内存图 —— networkx ``nx.Graph``（无向图）：节点/边属性直接挂在图上，
  度数、邻接遍历等由 networkx 提供，无需自维护邻接索引；
* Codable 快照 —— JSON 序列化文件（原子写），启动时直接加载重建图，
  免掉查库重建；数据更新时内存增量更新，构建批次结束/空闲时重存快照。

图节点/边带来源 chunk 信息（归因边）：反查 chunk、度数排序（rank）与
溯源都以此为基础。weight = max(raw_weight 累加, 去重来源 chunk 数)。

快照版本 2：内部结构改为 networkx（节点/边属性布局不变，仍是
nodes[name_key] / edges[pair_key] 两个属性字典）；旧 v1 快照拒载，
调用方回退查库重建（rebuild_from_store 幂等，代价可接受）。
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional

import networkx as nx

from ...atomic_io import atomic_write_text

logger = logging.getLogger(__name__)

SNAPSHOT_VERSION = 2

_PAIR_SEP = "\x1f"


def _pair_key(a: str, b: str) -> str:
    return _PAIR_SEP.join(sorted([a, b]))


class KnowledgeGraph:
    """内存图（networkx）：节点 name_key / 边 pair_key（无向规范化）。"""

    def __init__(self) -> None:
        self._g = nx.Graph()
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # 构建（增量）
    # ------------------------------------------------------------------

    def upsert_node(self, name_key: str, name: str, etype: str,
                    description: str, source_ids: List[str]) -> None:
        with self._lock:
            node = self._g.nodes.get(name_key)
            if node is None:
                self._g.add_node(name_key, name=name, type=etype or "UNKNOWN",
                                 description=description or "",
                                 source_ids=list(dict.fromkeys(source_ids)),
                                 degree=0)
            else:
                srcs = node.get("source_ids") or []
                for s in source_ids:
                    if s not in srcs:
                        srcs.append(s)
                if etype and (node.get("type") in ("", "UNKNOWN")):
                    node["type"] = etype
                if description:
                    node["description"] = description
                node["source_ids"] = srcs
            node = self._g.nodes[name_key]
            node["degree"] = self._g.degree(name_key)

    def upsert_edge(self, src_key: str, dst_key: str, src_name: str, dst_name: str,
                    relation_type: str, keywords: str, description: str,
                    raw_weight: int, source_ids: List[str], file_paths: List[str]) -> None:
        """加入/更新一条无向边；weight = max(累加 weight, 去重 chunk 数)。"""
        pk = _pair_key(src_key, dst_key)
        with self._lock:
            weight = max(int(raw_weight or 0), len(set(source_ids))) or 1
            data = self._g.get_edge_data(src_key, dst_key)
            if data is None:
                # 首次插入：边的 src/dst 记录首次方向（快照/检索沿用）。
                self._g.add_edge(src_key, dst_key, pair_key=pk, src=src_key,
                                 dst=dst_key, src_name=src_name, dst_name=dst_name,
                                 relation_type=relation_type or "",
                                 keywords=keywords or "",
                                 description=description or "",
                                 weight=weight,
                                 source_ids=list(dict.fromkeys(source_ids)),
                                 file_paths=list(dict.fromkeys(file_paths)))
            else:
                # 聚合行的字段以 DB 行为准（构建时整体重建优先走 rebuild）。
                data["weight"] = max(data.get("weight", 1), weight)
                for s in source_ids:
                    if s not in data["source_ids"]:
                        data["source_ids"].append(s)
                for f in file_paths:
                    if f not in data["file_paths"]:
                        data["file_paths"].append(f)
            # networkx 自动维护邻接与度数，这里仅刷新节点度数缓存属性。
            for k in (src_key, dst_key):
                node = self._g.nodes.get(k)
                if node is not None:
                    node["degree"] = self._g.degree(k)

    def rebuild_from_store(self, store) -> None:
        """从 KGStore 全量重建内存图（首次构建 / purge 后 / 无快照时）。"""
        with self._lock:
            self._g.clear()
            for e in store.iter_entities():
                self.upsert_node(e["name_key"], e["name"], e["type"],
                                 e["description"], e["source_ids"])
            for r in store.iter_relationships():
                self.upsert_edge(
                    r["src_entity"], r["dst_entity"], r["src_entity"], r["dst_entity"],
                    r["relation_type"], r["keywords"], r["description"],
                    # weight = max(累加, 去重 chunk 数) —— upsert_edge 内部计算
                    r["raw_weight"], r["source_ids"], r.get("file_paths") or [])

    # ------------------------------------------------------------------
    # 查询辅助
    # ------------------------------------------------------------------

    def get_node(self, name_key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            node = self._g.nodes.get(name_key)
            return dict(node) if node else None

    def get_edge_by_pair(self, pk: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            a, _, b = pk.partition(_PAIR_SEP)
            if not a or not b:
                return None
            data = self._g.get_edge_data(a, b)
            return dict(data) if data else None

    def neighbor_edges(self, name_key: str) -> List[Dict[str, Any]]:
        """某实体的一跳邻边（weight 降序，spec：rank+weight 排序）。"""
        with self._lock:
            if name_key not in self._g:
                return []
            edges = [dict(d) for _u, _v, d in self._g.edges(name_key, data=True)]
        edges.sort(key=lambda e: e.get("weight", 1), reverse=True)
        return edges

    def sorted_edges(self) -> List[Dict[str, Any]]:
        """全部边，按 weight 降序（global 策略召回用）。"""
        with self._lock:
            edges = [dict(d) for _u, _v, d in self._g.edges(data=True)]
        edges.sort(key=lambda e: e.get("weight", 1), reverse=True)
        return edges

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return self._g.number_of_nodes() == 0

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {"nodes": self._g.number_of_nodes(),
                    "edges": self._g.number_of_edges()}

    # ------------------------------------------------------------------
    # 快照缓存（Codable → JSON）
    # ------------------------------------------------------------------

    def _snapshot_payload(self) -> Dict[str, Any]:
        """导出为 nodes[name_key] / edges[pair_key] 两个属性字典（v2 布局）。"""
        nodes = {k: dict(d) for k, d in self._g.nodes(data=True)}
        edges: Dict[str, Dict[str, Any]] = {}
        for _u, _v, d in self._g.edges(data=True):
            edges[d["pair_key"]] = dict(d)
        return nodes, edges

    def to_snapshot(self) -> str:
        with self._lock:
            nodes, edges = self._snapshot_payload()
            payload = {
                "version": SNAPSHOT_VERSION,
                "saved_at": time.time(),
                "nodes": nodes,
                "edges": edges,
            }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def save_snapshot(self, path: str) -> None:
        """原子写快照文件（构建批次结束 / 空闲落盘点调用）。"""
        try:
            atomic_write_text(path, self.to_snapshot())
            logger.debug("KG: 快照已保存 %s", path)
        except OSError as e:
            logger.warning("KG: 快照保存失败 %s: %s", path, e)

    def load_snapshot(self, path: str) -> bool:
        """从快照重建图；文件缺失/损坏/旧版本返回 False（调用方回退查库重建）。"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, ValueError):
            return False
        if not isinstance(payload, dict) or payload.get("version") != SNAPSHOT_VERSION:
            logger.debug("KG: 快照版本不匹配（期望 v%d），回退查库重建",
                         SNAPSHOT_VERSION)
            return False
        nodes = payload.get("nodes")
        edges = payload.get("edges")
        if not isinstance(nodes, dict) or not isinstance(edges, dict):
            return False
        with self._lock:
            self._g.clear()
            for k, attrs in nodes.items():
                self._g.add_node(k, **dict(attrs))
            for pk, attrs in edges.items():
                src, dst = attrs.get("src"), attrs.get("dst")
                if not src or not dst:
                    continue  # 防御畸形快照行
                data = dict(attrs)
                data.setdefault("pair_key", pk)
                self._g.add_edge(src, dst, **data)
            for k in self._g.nodes:
                self._g.nodes[k]["degree"] = self._g.degree(k)
        return True
