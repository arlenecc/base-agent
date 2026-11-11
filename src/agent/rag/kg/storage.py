"""知识图谱持久化存储（SQLite）。

三张表（对应 spec 第一~三步）：
* ``entities``        —— id/name/type/description/properties/source_ids；
* ``relationships``   —— id/src/dst/relation_type/description/weight/properties/
                         source_ids（无向边按 pair_key 合并，一行 = 一对实体的
                         全部证据聚合）；
* ``kg_vectors``      —— vector_id/target_type/target_id/embedding(int8 blob)；
* ``entity_aliases``  —— 实体消歧别名索引：别名键 → 实体 id（同一实体的
                         不同写法/简称统一命中同一行，避免重复入库）。

关键设计：
* DB 行是「证据聚合行」：同一对实体的多条抽取按无向 pair_key 合并进一行，
  ``raw_weight`` 每多一条抽取记录 +1，``source_ids`` 去重累积。weight 的最终
  取值 = max(raw_weight, 去重 chunk 数)，在图谱构建（graph.py）时计算——
  purge 后按剩余证据重算，不保留旧标量（spec 第四步）。
* 实体向量按 ``ent:{id}``，关系向量按 ``rel:{id}:fwd`` / ``rel:{id}:rev``
  （无向边正反序两个 id，spec 第五步）。
* ``content_hash`` 记录被嵌入文本的指纹，增量构建时跳过未变化的行。
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    name_key    TEXT NOT NULL UNIQUE,
    type        TEXT NOT NULL DEFAULT 'UNKNOWN',
    description TEXT NOT NULL DEFAULT '',
    properties  TEXT NOT NULL DEFAULT '{}',
    source_ids  TEXT NOT NULL DEFAULT '[]',
    descriptions TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS relationships (
    id            TEXT PRIMARY KEY,
    src_entity    TEXT NOT NULL,
    dst_entity    TEXT NOT NULL,
    pair_key      TEXT NOT NULL UNIQUE,
    relation_type TEXT NOT NULL DEFAULT '',
    keywords      TEXT NOT NULL DEFAULT '',
    description   TEXT NOT NULL DEFAULT '',
    raw_weight    INTEGER NOT NULL DEFAULT 0,
    properties    TEXT NOT NULL DEFAULT '{}',
    source_ids    TEXT NOT NULL DEFAULT '[]',
    file_paths    TEXT NOT NULL DEFAULT '[]',
    descriptions  TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS kg_vectors (
    vector_id    TEXT PRIMARY KEY,
    target_type  TEXT NOT NULL,
    target_id    TEXT NOT NULL,
    content_hash TEXT NOT NULL DEFAULT '',
    embedding    BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS entity_aliases (
    alias_key  TEXT PRIMARY KEY,
    alias      TEXT NOT NULL,
    entity_id  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rel_src ON relationships(src_entity);
CREATE INDEX IF NOT EXISTS idx_rel_dst ON relationships(dst_entity);
CREATE INDEX IF NOT EXISTS idx_vec_target ON kg_vectors(target_type, target_id);
CREATE INDEX IF NOT EXISTS idx_alias_entity ON entity_aliases(entity_id);
"""


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _json_loads(text: str, default: Any) -> Any:
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return default


def normalize_name(name: str) -> str:
    """实体名规范化键：去空白 + lower（与抽取 prompt 的 title case 约定配合）。"""
    return " ".join(str(name or "").split()).lower()


def pair_key(src: str, dst: str) -> str:
    """无向边的规范化键（正反序同键，spec：关系是无向的）。"""
    a, b = normalize_name(src), normalize_name(dst)
    lo, hi = (a, b) if a <= b else (b, a)
    return f"{lo}\x1f{hi}"


class KGStore:
    """KG 三张表的 SQLite 封装（线程安全：模块级锁按 db 路径键控）。"""

    def __init__(self, kg_dir: str):
        self._dir = kg_dir
        os.makedirs(kg_dir, exist_ok=True)
        self._db_path = os.path.join(kg_dir, "kg.sqlite")
        self._lock = threading.Lock()
        with self._lock:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    @property
    def db_path(self) -> str:
        return self._db_path

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # entities
    # ------------------------------------------------------------------

    def upsert_entity(
        self,
        name: str,
        etype: str,
        description: str,
        source_id: str,
        file_path: str = "",
        merged_description: Optional[str] = None,
    ) -> Tuple[str, bool]:
        """插入/聚合一个实体证据，返回 (entity_id, created)。

        同名（name_key 相同）实体聚合：source_ids 去重累积、descriptions 去重
        累积；type 取首个非 UNKNOWN 值；description 可由调用方传入 LLM 归纳
        结果覆盖（merged_description）。
        """
        key = normalize_name(name)
        if not key:
            raise ValueError("empty entity name")
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, type, descriptions, source_ids FROM entities WHERE name_key=?",
                (key,))
            row = cur.fetchone()
            if row is None:
                eid = str(uuid.uuid4())
                props = {"file_paths": [file_path] if file_path else []}
                self._conn.execute(
                    "INSERT INTO entities (id, name, name_key, type, description,"
                    " properties, source_ids, descriptions) VALUES (?,?,?,?,?,?,?,?)",
                    (eid, name.strip(), key, etype or "UNKNOWN", description or "",
                     _json_dumps(props), _json_dumps([source_id] if source_id else []),
                     _json_dumps([description] if description else [])))
                self._conn.commit()
                return eid, True
            eid, old_type, old_descs_json, old_src_json = row
            srcs = _json_loads(old_src_json, [])
            if source_id and source_id not in srcs:
                srcs.append(source_id)
            descs = _json_loads(old_descs_json, [])
            if description and description not in descs:
                descs.append(description)
            new_type = old_type if (old_type and old_type != "UNKNOWN") else (etype or old_type or "UNKNOWN")
            desc = merged_description if merged_description is not None else " ".join(descs)[:8000]
            self._conn.execute(
                "UPDATE entities SET type=?, description=?, source_ids=?, descriptions=? WHERE id=?",
                (new_type, desc, _json_dumps(srcs), _json_dumps(descs), eid))
            self._conn.commit()
            return eid, False

    def get_entity_by_key(self, name_key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, name, name_key, type, description, properties, source_ids"
                " FROM entities WHERE name_key=?", (name_key,)).fetchone()
        if row is None:
            return None
        return self._entity_row(row)

    @staticmethod
    def _entity_row(row) -> Dict[str, Any]:
        return {
            "id": row[0], "name": row[1], "name_key": row[2], "type": row[3],
            "description": row[4],
            "properties": _json_loads(row[5], {}),
            "source_ids": _json_loads(row[6], []),
        }

    # ------------------------------------------------------------------
    # 实体消歧：别名索引 + 合并
    # ------------------------------------------------------------------

    def find_entity_by_alias(self, alias_key: str) -> Optional[Dict[str, Any]]:
        """按别名键查实体（消歧用：同一实体的不同写法命中同一行）。"""
        key = (alias_key or "").strip()
        if not key:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT entity_id FROM entity_aliases WHERE alias_key=?",
                (key,)).fetchone()
            if row is None:
                return None
            ent = self._conn.execute(
                "SELECT id, name, name_key, type, description, properties, source_ids"
                " FROM entities WHERE id=?", (row[0],)).fetchone()
        return self._entity_row(ent) if ent else None

    def add_alias(self, alias_key: str, alias: str, entity_id: str) -> bool:
        """注册别名键 → 实体。

        已存在且指向同一实体返回 True；指向其他实体返回 False（不覆盖，
        交由消歧合并流程处理）；新写入返回 True。
        """
        key = (alias_key or "").strip()
        if not key or not entity_id:
            return False
        with self._lock:
            row = self._conn.execute(
                "SELECT entity_id FROM entity_aliases WHERE alias_key=?",
                (key,)).fetchone()
            if row is not None:
                return row[0] == entity_id
            self._conn.execute(
                "INSERT OR REPLACE INTO entity_aliases (alias_key, alias, entity_id)"
                " VALUES (?,?,?)", (key, alias or key, entity_id))
            self._conn.commit()
            return True

    def add_aliases(self, items: Iterable[Tuple[str, str, str]]) -> int:
        """批量注册别名（单事务）：``(alias_key, alias, entity_id)``。

        已存在的键不覆盖（``INSERT OR IGNORE``），返回新写入行数。构建期
        实体量大，逐条 commit 会成为瓶颈，故合并为一次事务。
        """
        rows = []
        for alias_key, alias, entity_id in items:
            key = (alias_key or "").strip()
            if key and entity_id:
                rows.append((key, alias or key, entity_id))
        if not rows:
            return 0
        with self._lock:
            before = self._conn.total_changes
            self._conn.executemany(
                "INSERT OR IGNORE INTO entity_aliases"
                " (alias_key, alias, entity_id) VALUES (?,?,?)", rows)
            self._conn.commit()
            return self._conn.total_changes - before

    def aliases_of(self, entity_id: str) -> List[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT alias FROM entity_aliases WHERE entity_id=?",
                (entity_id,)).fetchall()
        return [r[0] for r in rows]

    def merge_entities(self, source_id: str, target_id: str) -> Dict[str, int]:
        """把 source 实体整体合并进 target（source 行删除）。

        实体消歧的核心动作（**不新增实体行**）：
        1. 证据聚合：source_ids / descriptions / properties.file_paths 并入 target，
           type 以 target 的非 UNKNOWN 类型优先；
        2. 别名转移：source 的别名键与其 name_key 全部改指 target；
        3. 关系重写：以 source 为端点的关系换成 target 展示名并重算 pair_key；
           与已有关系撞键时合并为一行（raw_weight 相加、证据并集），
           合并后成为自环的关系直接删除；
        4. 清理：source 实体行、被删关系的向量、source 实体向量。
        """
        stats = {"entities": 0, "relationships": 0, "vectors": 0, "rewritten": 0}
        if not source_id or not target_id or source_id == target_id:
            return stats
        sel = ("SELECT id, name, name_key, type, description, properties,"
               " source_ids, descriptions FROM entities WHERE id=?")
        with self._lock:
            src = self._conn.execute(sel, (source_id,)).fetchone()
            tgt = self._conn.execute(sel, (target_id,)).fetchone()
            if src is None or tgt is None:
                return stats
            src_name, src_key = src[1], src[2]
            tgt_name, tgt_type = tgt[1], tgt[3]
            # 1) 实体行证据聚合
            srcs = list(dict.fromkeys(_json_loads(tgt[6], [])
                                      + _json_loads(src[6], [])))
            descs = list(dict.fromkeys(_json_loads(tgt[7], [])
                                       + _json_loads(src[7], [])))
            new_type = (tgt_type if tgt_type and tgt_type != "UNKNOWN"
                        else (src[3] or "UNKNOWN"))
            props = _json_loads(tgt[5], {}) or {}
            for k, v in (_json_loads(src[5], {}) or {}).items():
                if k == "file_paths" and isinstance(v, list):
                    props[k] = list(dict.fromkeys(list(props.get(k) or []) + v))
                else:
                    props.setdefault(k, v)
            self._conn.execute(
                "UPDATE entities SET type=?, description=?, properties=?,"
                " source_ids=?, descriptions=? WHERE id=?",
                (new_type, " ".join(descs)[:8000], _json_dumps(props),
                 _json_dumps(srcs), _json_dumps(descs), target_id))
            # 2) 别名转移
            self._conn.execute(
                "UPDATE entity_aliases SET entity_id=? WHERE entity_id=?",
                (target_id, source_id))
            self._conn.execute(
                "INSERT OR REPLACE INTO entity_aliases (alias_key, alias, entity_id)"
                " VALUES (?,?,?)", (src_key, src_name, target_id))
            # 3) 关系端点重写（display 名 + pair_key 双条件，兼容历史数据）
            rows = self._conn.execute(
                "SELECT id, src_entity, dst_entity, pair_key FROM relationships"
                " WHERE src_entity=? OR dst_entity=? OR pair_key LIKE ? OR pair_key LIKE ?",
                (src_name, src_name, f"{src_key}\x1f%", f"%\x1f{src_key}")).fetchall()
            for rid, s_disp, d_disp, pk in rows:
                s_is_src = normalize_name(s_disp) == src_key
                d_is_src = normalize_name(d_disp) == src_key
                if not (s_is_src or d_is_src):
                    continue
                new_s = tgt_name if s_is_src else s_disp
                new_d = tgt_name if d_is_src else d_disp
                if normalize_name(new_s) == normalize_name(new_d):
                    # 两端合并为同一实体 → 自环无意义，删除
                    stats["vectors"] += self._delete_relationship_locked(rid)
                    stats["relationships"] += 1
                    continue
                new_pk = pair_key(new_s, new_d)
                if new_pk == pk:
                    self._conn.execute(
                        "UPDATE relationships SET src_entity=?, dst_entity=? WHERE id=?",
                        (new_s, new_d, rid))
                    stats["rewritten"] += 1
                    continue
                dup = self._conn.execute(
                    "SELECT id FROM relationships WHERE pair_key=? AND id<>?",
                    (new_pk, rid)).fetchone()
                if dup is not None:
                    self._merge_relationship_locked(rid, dup[0])
                    stats["vectors"] += self._delete_relationship_locked(rid)
                    stats["relationships"] += 1
                else:
                    self._conn.execute(
                        "UPDATE relationships SET src_entity=?, dst_entity=?,"
                        " pair_key=? WHERE id=?", (new_s, new_d, new_pk, rid))
                    stats["rewritten"] += 1
            # 4) 删除 source 实体行与向量
            self._conn.execute("DELETE FROM entities WHERE id=?", (source_id,))
            stats["vectors"] += self._delete_vectors_locked("entity", source_id)
            stats["entities"] = 1
            self._conn.commit()
        return stats

    def _delete_relationship_locked(self, rid: str) -> int:
        """删除一条关系及其向量（调用方已持锁）。返回删除的向量数。"""
        self._conn.execute("DELETE FROM relationships WHERE id=?", (rid,))
        return self._delete_vectors_locked("relationship", rid)

    def _delete_vectors_locked(self, target_type: str, target_id: str) -> int:
        cur = self._conn.execute(
            "DELETE FROM kg_vectors WHERE target_type=? AND target_id=?",
            (target_type, target_id))
        return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0

    def _merge_relationship_locked(self, source_rid: str, target_rid: str) -> None:
        """把 source 关系行合并进 target 行（调用方已持锁）。"""
        sel = ("SELECT keywords, descriptions, raw_weight, source_ids, file_paths"
               " FROM relationships WHERE id=?")
        src = self._conn.execute(sel, (source_rid,)).fetchone()
        tgt = self._conn.execute(sel, (target_rid,)).fetchone()
        if src is None or tgt is None:
            return
        kws = [k.strip() for k in (tgt[0] or "").split(",") if k.strip()]
        known = {k.lower() for k in kws}
        for k in [k.strip() for k in (src[0] or "").split(",") if k.strip()]:
            if k.lower() not in known:
                kws.append(k)
                known.add(k.lower())
        descs = list(dict.fromkeys(_json_loads(tgt[1], [])
                                   + _json_loads(src[1], [])))
        srcs = list(dict.fromkeys(_json_loads(tgt[3], [])
                                  + _json_loads(src[3], [])))
        fps = list(dict.fromkeys(_json_loads(tgt[4], [])
                                 + _json_loads(src[4], [])))
        self._conn.execute(
            "UPDATE relationships SET keywords=?, description=?, raw_weight=?,"
            " source_ids=?, file_paths=?, descriptions=? WHERE id=?",
            (",".join(kws), " ".join(descs)[:8000],
             int(tgt[2] or 0) + int(src[2] or 0),
             _json_dumps(srcs), _json_dumps(fps), _json_dumps(descs), target_rid))

    # ------------------------------------------------------------------
    # relationships（无向合并）
    # ------------------------------------------------------------------

    def upsert_relationship(
        self,
        src_name: str,
        dst_name: str,
        keywords: str,
        description: str,
        source_id: str,
        file_path: str = "",
        merged_description: Optional[str] = None,
    ) -> Tuple[str, bool]:
        """插入/聚合一条关系证据，返回 (relationship_id, created)。

        无向 pair_key 相同即同一行：raw_weight +1（spec：每多一个 chunk 抽到
        加一）、keywords 并集、source_ids/file_paths/descriptions 去重累积。
        """
        pk = pair_key(src_name, dst_name)
        a, b = pk.split("\x1f")
        # 保持首次出现方向的展示名，由 name_key 反查；新对则按本次方向。
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, src_entity, dst_entity, keywords, descriptions,"
                " raw_weight, source_ids, file_paths FROM relationships WHERE pair_key=?",
                (pk,))
            row = cur.fetchone()
            if row is None:
                rid = str(uuid.uuid4())
                src_disp, dst_disp = self._display_names(a, b, src_name, dst_name)
                self._conn.execute(
                    "INSERT INTO relationships (id, src_entity, dst_entity, pair_key,"
                    " relation_type, keywords, description, raw_weight, properties,"
                    " source_ids, file_paths, descriptions)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (rid, src_disp, dst_disp, pk, (keywords or "").split(",")[0].strip(),
                     keywords or "", description or "", 1, "{}",
                     _json_dumps([source_id] if source_id else []),
                     _json_dumps([file_path] if file_path else []),
                     _json_dumps([description] if description else [])))
                self._conn.commit()
                return rid, True
            rid, src_disp, dst_disp, old_kw, old_descs, old_w, old_src, old_fp = row
            srcs = _json_loads(old_src, [])
            if source_id and source_id not in srcs:
                srcs.append(source_id)
            fps = _json_loads(old_fp, [])
            if file_path and file_path not in fps:
                fps.append(file_path)
            descs = _json_loads(old_descs, [])
            if description and description not in descs:
                descs.append(description)
            kws = [k.strip() for k in (old_kw or "").split(",") if k.strip()]
            for k in [k.strip() for k in (keywords or "").split(",") if k.strip()]:
                if k.lower() not in {x.lower() for x in kws}:
                    kws.append(k)
            desc = merged_description if merged_description is not None else " ".join(descs)[:8000]
            self._conn.execute(
                "UPDATE relationships SET keywords=?, description=?, raw_weight=?,"
                " source_ids=?, file_paths=?, descriptions=? WHERE id=?",
                (",".join(kws), desc, old_w + 1,
                 _json_dumps(srcs), _json_dumps(fps), _json_dumps(descs), rid))
            self._conn.commit()
            return rid, False

    def _display_names(self, a_key: str, b_key: str, src_name: str, dst_name: str):
        """pair 键顺序 → 展示名。已存在实体用库内展示名，否则用本次传入名。"""
        out = []
        for k, fallback in ((a_key, src_name), (b_key, dst_name)):
            row = self._conn.execute(
                "SELECT name FROM entities WHERE name_key=?", (k,)).fetchone()
            out.append(row[0] if row else fallback)
        return out[0], out[1]

    def get_relationship_by_pair(self, pk: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, src_entity, dst_entity, pair_key, relation_type, keywords,"
                " description, raw_weight, properties, source_ids, file_paths"
                " FROM relationships WHERE pair_key=?", (pk,)).fetchone()
        if row is None:
            return None
        return self._rel_row(row)

    @staticmethod
    def _rel_row(row) -> Dict[str, Any]:
        return {
            "id": row[0], "src_entity": row[1], "dst_entity": row[2],
            "pair_key": row[3], "relation_type": row[4], "keywords": row[5],
            "description": row[6], "raw_weight": row[7],
            "properties": _json_loads(row[8], {}),
            "source_ids": _json_loads(row[9], []),
            "file_paths": _json_loads(row[10], []),
        }

    # ------------------------------------------------------------------
    # 迭代（图谱构建 / 向量化用）
    # ------------------------------------------------------------------

    def iter_entities(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, name, name_key, type, description, properties, source_ids"
                " FROM entities").fetchall()
        return [self._entity_row(r) for r in rows]

    def iter_relationships(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, src_entity, dst_entity, pair_key, relation_type, keywords,"
                " description, raw_weight, properties, source_ids, file_paths"
                " FROM relationships").fetchall()
        return [self._rel_row(r) for r in rows]

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, name, name_key, type, description, properties, source_ids"
                " FROM entities WHERE id=?", (entity_id,)).fetchone()
        return self._entity_row(row) if row else None

    def get_relationship(self, rel_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, src_entity, dst_entity, pair_key, relation_type, keywords,"
                " description, raw_weight, properties, source_ids, file_paths"
                " FROM relationships WHERE id=?", (rel_id,)).fetchone()
        return self._rel_row(row) if row else None

    def relationships_of_entity(self, name_key: str) -> List[Dict[str, Any]]:
        """某实体的全部关联边（一跳邻边，图谱反查用）。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, src_entity, dst_entity, pair_key, relation_type, keywords,"
                " description, raw_weight, properties, source_ids, file_paths"
                " FROM relationships WHERE src_entity=? OR dst_entity=?",
                (name_key, name_key)).fetchall()
        return [self._rel_row(r) for r in rows]

    def delete_entity_cascade(self, entity_id: str) -> Dict[str, int]:
        """删除实体及其全部关联边（MCP delete_entity 工具用）。

        步骤：删该实体的全部关系行与关系向量 → 删别名 → 删实体向量 →
        删实体行。返回 ``{"entities": 0|1, "relationships": n, "vectors": n}``；
        实体不存在时返回全零。
        """
        stats = {"entities": 0, "relationships": 0, "vectors": 0}
        if not entity_id:
            return stats
        with self._lock:
            row = self._conn.execute(
                "SELECT name, name_key FROM entities WHERE id=?",
                (entity_id,)).fetchone()
            if row is None:
                return stats
            name, key = row
            # 关联边：展示名 / name_key / pair_key 三种历史形态都要匹配
            rel_ids = self._conn.execute(
                "SELECT id FROM relationships WHERE src_entity=? OR dst_entity=?"
                " OR pair_key LIKE ? OR pair_key LIKE ?",
                (name, name, f"{key}\x1f%", f"%\x1f{key}")).fetchall()
            for (rid,) in rel_ids:
                stats["vectors"] += self._delete_relationship_locked(rid)
                stats["relationships"] += 1
            stats["vectors"] += self._delete_vectors_locked("entity", entity_id)
            self._conn.execute(
                "DELETE FROM entity_aliases WHERE entity_id=?", (entity_id,))
            self._conn.execute("DELETE FROM entities WHERE id=?", (entity_id,))
            stats["entities"] = 1
            self._conn.commit()
        return stats

    # ------------------------------------------------------------------
    # 向量虚拟表
    # ------------------------------------------------------------------

    def get_vector(self, vector_id: str) -> Optional[bytes]:
        with self._lock:
            row = self._conn.execute(
                "SELECT embedding FROM kg_vectors WHERE vector_id=?",
                (vector_id,)).fetchone()
        return row[0] if row else None

    def get_vector_hash(self, vector_id: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT content_hash FROM kg_vectors WHERE vector_id=?",
                (vector_id,)).fetchone()
        return row[0] if row else None

    def put_vector(self, vector_id: str, target_type: str, target_id: str,
                   content_hash: str, embedding: bytes) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO kg_vectors"
                " (vector_id, target_type, target_id, content_hash, embedding)"
                " VALUES (?,?,?,?,?)",
                (vector_id, target_type, target_id, content_hash, embedding))
            self._conn.commit()

    def iter_vectors(self, target_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """全部向量（检索时一次性载入内存做余弦，KG 规模在万级、毫秒可扫完）。"""
        with self._lock:
            if target_type:
                rows = self._conn.execute(
                    "SELECT vector_id, target_type, target_id, embedding"
                    " FROM kg_vectors WHERE target_type=?", (target_type,)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT vector_id, target_type, target_id, embedding"
                    " FROM kg_vectors").fetchall()
        return [
            {"vector_id": r[0], "target_type": r[1], "target_id": r[2],
             "embedding": r[3]}
            for r in rows
        ]

    # ------------------------------------------------------------------
    # purge（知识遗忘路径）
    # ------------------------------------------------------------------

    def purge_document(self, chunk_id_prefixes: Iterable[str]) -> Dict[str, int]:
        """删除某文档的全部证据并重算 weight（spec 第四步）。

        chunk id 形如 ``{hash}_{idx}``，hash = 文档路径指纹前缀。对所有以给定
        前缀开头的 source_id：从各行的 source_ids 中剔除；剔除后为空的行整体
        删除；关系的 raw_weight 不保留旧标量，直接按剩余证据重算
        （= 剩余去重 chunk 数）。返回删除统计。
        """
        prefixes = tuple(p for p in chunk_id_prefixes if p)
        if not prefixes:
            return {"entities": 0, "relationships": 0, "vectors": 0}
        stats = {"entities": 0, "relationships": 0, "vectors": 0}
        with self._lock:
            # --- entities ---
            rows = self._conn.execute(
                "SELECT id, source_ids FROM entities").fetchall()
            for eid, src_json in rows:
                srcs = _json_loads(src_json, [])
                kept = [s for s in srcs if not s.startswith(prefixes)]
                if len(kept) == len(srcs):
                    continue
                if kept:
                    self._conn.execute(
                        "UPDATE entities SET source_ids=? WHERE id=?",
                        (_json_dumps(kept), eid))
                else:
                    self._conn.execute("DELETE FROM entities WHERE id=?", (eid,))
                    self._conn.execute(
                        "DELETE FROM entity_aliases WHERE entity_id=?", (eid,))
                    stats["entities"] += 1
            # --- relationships ---
            rows = self._conn.execute(
                "SELECT id, source_ids FROM relationships").fetchall()
            for rid, src_json in rows:
                srcs = _json_loads(src_json, [])
                kept = [s for s in srcs if not s.startswith(prefixes)]
                if len(kept) == len(srcs):
                    continue
                if kept:
                    # weight 随证据向下重算：不保留旧 raw_weight 标量。
                    self._conn.execute(
                        "UPDATE relationships SET source_ids=?, raw_weight=? WHERE id=?",
                        (_json_dumps(kept), len(set(kept)), rid))
                else:
                    self._conn.execute("DELETE FROM relationships WHERE id=?", (rid,))
                    stats["relationships"] += 1
            # --- 向量清理（孤儿向量 + 被删行）---
            vec_rows = self._conn.execute(
                "SELECT kv.vector_id, kv.target_type, kv.target_id FROM kg_vectors kv"
                " LEFT JOIN entities e ON kv.target_type='entity' AND e.id=kv.target_id"
                " LEFT JOIN relationships r ON kv.target_type='relationship' AND r.id=kv.target_id"
            ).fetchall()
            for vid, ttype, tid in vec_rows:
                exists = False
                if ttype == "entity":
                    exists = self._conn.execute(
                        "SELECT 1 FROM entities WHERE id=?", (tid,)).fetchone() is not None
                elif ttype == "relationship":
                    exists = self._conn.execute(
                        "SELECT 1 FROM relationships WHERE id=?", (tid,)).fetchone() is not None
                if not exists:
                    self._conn.execute("DELETE FROM kg_vectors WHERE vector_id=?", (vid,))
                    stats["vectors"] += 1
            self._conn.commit()
        return stats

    def clear(self) -> None:
        """清空三张表（强制重建知识图谱用）。"""
        with self._lock:
            self._conn.executescript(
                "DELETE FROM kg_vectors; DELETE FROM relationships;"
                " DELETE FROM entities; DELETE FROM entity_aliases;")
            self._conn.commit()

    def counts(self) -> Dict[str, int]:
        with self._lock:
            e = self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            r = self._conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
            v = self._conn.execute("SELECT COUNT(*) FROM kg_vectors").fetchone()[0]
        return {"entities": e, "relationships": r, "vectors": v}
