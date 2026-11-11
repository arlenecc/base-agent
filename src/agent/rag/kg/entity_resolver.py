"""实体消歧与合并（三元组入库前的规范化环节）。

目标：同一实体的不同写法（简称/全称/大小写/标点/全半角/括号注释/机构后缀）
在入库时识别为同一实体，**不新增实体行**，而是把证据合并到已有实体上，
关系端点同步改写——避免图谱被同名变体割裂成多个孤立节点。

消歧规则（确定性、无 LLM 调用，可离线运行）：
1. 精确规范化键（``normalize_name``：小写 + 空白折叠）；
2. 别名索引（``entity_aliases`` 表，历史合并结果与新注册变体都能命中）；
3. 规范形（``canonical_forms``）比对：NFKC 全半角归一、去标点空白、
   剥括号注释（内外两侧各成候选）、反复剥离机构后缀
   （公司/集团/Corp/Inc/Ltd...）；
4. 类型兼容守卫：两侧类型都是已知且不同（如 Person vs Organization）时
   不合并，避免把「苹果（水果）」并进「苹果公司」。

合并动作由 :meth:`KGStore.merge_entities` 完成：证据聚合 + 别名转移 +
关系端点重写（撞键关系合并、自环删除）+ 孤儿向量清理。

两类使用场景：
* 增量入库（``KGBuilder._commit_extraction``）：每个抽取实体先 :meth:`resolve`，
  命中则合并入库并返回规范展示名，关系端点用规范名写入；
* 存量整理（``KGBuilder.build`` 开始处 :meth:`consolidate`）：扫描全库，
  把历史遗留的重复实体一次性合并，并补全别名索引。
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from .storage import KGStore, normalize_name

logger = logging.getLogger(__name__)

# 机构/组织类后缀（消歧时剥离后再比对）。按长度降序匹配，避免「有限公司」
# 被「公司」抢先截断。
_ORG_SUFFIXES = (
    "股份有限公司", "有限责任公司", "有限公司", "株式会社", "集团", "控股",
    "公司", "corporation", "incorporated", "limited", "company", "holdings",
    "corp", "inc", "ltd", "llc", "gmbh", "plc",
)
# 括号注释："苹果（Apple）" / "Apple (Inc.)"
_PAREN_RE = re.compile(r"[（(【\[][^（()）【\[\]】]*[)）】\]]")
# 标点/空白（保留字母数字与汉字）
_STRIP_RE = re.compile(r"[\s\-_·・.,，。、:：;；'\"“”‘’/\\|!！?？&＆+＋#*～~]+")

# 合并时视为「无类型」，可与任意类型兼容
_ANY_TYPE = {"", "UNKNOWN", "OTHER"}


def _compact(text: str) -> str:
    """去标点与空白（保留大小写已由 normalize_name 统一为小写）。"""
    return _STRIP_RE.sub("", text or "")


def _strip_org_suffix(text: str) -> str:
    """反复剥离机构后缀（保留至少 2 个字符，避免剥空）。"""
    out = text or ""
    changed = True
    while changed:
        changed = False
        for suffix in _ORG_SUFFIXES:
            if out.endswith(suffix) and len(out) > len(suffix) + 1:
                out = out[: -len(suffix)]
                changed = True
                break
    return out


def canonical_forms(name: str) -> List[str]:
    """实体名的全部规范形候选（首个为精确规范化键，其余为变体键）。"""
    base = normalize_name(name)
    if not base:
        return []
    nfkc = unicodedata.normalize("NFKC", base)
    parts: List[str] = [nfkc]
    # 括号内外各自作为候选：「苹果（Apple）」→ 苹果 / apple
    outer = _PAREN_RE.sub(" ", nfkc)
    if outer.strip():
        parts.append(outer)
    for m in _PAREN_RE.finditer(nfkc):
        inner = m.group(0)[1:-1].strip()
        if inner:
            parts.append(inner)
    forms: List[str] = []

    def _add(form: str) -> None:
        form = (form or "").strip()
        if form and form not in forms:
            forms.append(form)

    _add(base)
    for part in parts:
        _add(part)
        compact = _compact(part)
        _add(compact)
        _add(_strip_org_suffix(compact))
    return forms


def types_compatible(a: str, b: str) -> bool:
    """类型兼容判断：任一侧未知/Other 即兼容，否则要求同类型。"""
    ta = str(a or "").strip().upper()
    tb = str(b or "").strip().upper()
    if ta in _ANY_TYPE or tb in _ANY_TYPE:
        return True
    return ta == tb


def _keep_priority(ent: Dict[str, Any]) -> Tuple[int, int, int]:
    """合并时保留哪个实体：证据多 > 类型已明确 > 描述长。"""
    return (len(ent.get("source_ids") or []),
            0 if str(ent.get("type") or "").upper() in _ANY_TYPE else 1,
            len(ent.get("description") or ""))


class EntityResolver:
    """实体消歧器：把抽取名解析到已有规范实体，必要时合并。"""

    def __init__(self, store: KGStore, enabled: bool = True):
        self._store = store
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    # 单实体解析
    # ------------------------------------------------------------------

    def resolve(self, name: str, etype: str = "") -> Optional[Dict[str, Any]]:
        """返回已存在的同一实体（无则 None）。"""
        if not self._enabled:
            return None
        key = normalize_name(name)
        if not key:
            return None
        ent = self._store.get_entity_by_key(key)
        if ent is not None:
            return ent
        for form in canonical_forms(name):
            # 精确键已在上面查过，但别名索引仍可能命中（变体注册）。
            ent = self._store.find_entity_by_alias(form)
            if ent is not None and types_compatible(ent.get("type"), etype):
                return ent
            ent = self._store.get_entity_by_key(form)
            if ent is not None and types_compatible(ent.get("type"), etype):
                return ent
        return None

    def merge_or_create(
        self, name: str, etype: str, description: str,
        chunk_id: str, source: str = "",
    ) -> Tuple[str, bool, bool]:
        """入库一个抽取实体，返回 ``(规范展示名, 是否新建, 是否合并)``。

        命中已有实体时**不插入新行**：把证据聚合到已有实体（source_ids /
        descriptions 去重累积）并注册本次写法为别名；未命中则正常插入。
        """
        name = (name or "").strip()
        if not name:
            raise ValueError("empty entity name")
        existing = self.resolve(name, etype)
        if existing is not None:
            canonical = existing["name"]
            eid, _created = self._store.upsert_entity(
                canonical, etype or existing.get("type") or "UNKNOWN",
                description, chunk_id, file_path=source)
            merged = normalize_name(canonical) != normalize_name(name)
            self._register_forms(eid, canonical, name)
            return canonical, False, merged
        eid, created = self._store.upsert_entity(
            name, etype, description, chunk_id, file_path=source)
        self._register_forms(eid, name)
        return name, created, False

    def _register_forms(self, entity_id: str, *names: str) -> None:
        items = []
        for name in names:
            for form in canonical_forms(name):
                items.append((form, name, entity_id))
        self._store.add_aliases(items)

    # ------------------------------------------------------------------
    # 存量整理
    # ------------------------------------------------------------------

    def consolidate(self) -> Dict[str, int]:
        """全库扫描：按规范形分组合并重复实体，并补全别名索引。

        返回 ``{"scanned": 实体总数, "merged": 合并掉的实体数}``。
        """
        stats = {"scanned": 0, "merged": 0}
        if not self._enabled:
            return stats
        entities = self._store.iter_entities()
        stats["scanned"] = len(entities)
        if len(entities) < 2:
            self._register_all(entities)
            return stats

        groups: Dict[str, List[Dict[str, Any]]] = {}
        for ent in entities:
            for form in canonical_forms(ent["name"]):
                groups.setdefault(form, []).append(ent)

        merged_ids: set = set()
        for form in sorted(groups):
            members = [m for m in groups[form] if m["id"] not in merged_ids]
            if len(members) < 2:
                continue
            unique = {m["id"]: m for m in members}
            ordered = sorted(unique.values(), key=_keep_priority, reverse=True)
            target = ordered[0]
            if self._store.get_entity(target["id"]) is None:
                continue
            for other in ordered[1:]:
                if other["id"] in merged_ids:
                    continue
                target_row = self._store.get_entity(target["id"])
                if target_row is None:
                    break
                if not types_compatible(target_row.get("type"), other.get("type")):
                    continue
                result = self._store.merge_entities(other["id"], target["id"])
                if result.get("entities"):
                    merged_ids.add(other["id"])
                    stats["merged"] += 1
                    logger.info("KG: 实体合并 %r → %r", other["name"], target["name"])
        self._register_all(self._store.iter_entities())
        return stats

    def _register_all(self, entities: List[Dict[str, Any]]) -> None:
        items = []
        for ent in entities:
            for form in canonical_forms(ent["name"]):
                items.append((form, ent["name"], ent["id"]))
        self._store.add_aliases(items)


# ---------------------------------------------------------------------------
# 存量语义向量消歧（配置面板「实体消歧」按钮）
# ---------------------------------------------------------------------------

# 语义消歧相似度阈值：实体名+描述向量余弦 ≥ 该值才视为同一实体。
# 取高阈值（0.92）宁可漏合不误合——错误合并会破坏图谱结构。
DEFAULT_SEMANTIC_THRESHOLD = 0.92

# 相似度分块计算大小：每次 blocks×N 行矩阵，控制内存（O(n²) 但分块）。
_SIM_BLOCK = 512


def semantic_consolidate(
    store: "KGStore",
    embedding_function: Any = None,
    threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    progress_callback: Optional[Any] = None,
    cancel_event: Any = None,
) -> Dict[str, int]:
    """对库内现有实体做语义向量消歧：向量相似度对齐 + 合并（不新增实体）。

    与增量入库的规则消歧（:meth:`EntityResolver`）互补：规则消歧只覆盖
    确定性的写法变体（后缀/括号/全半角），语义消歧用实体向量余弦相似度
    找出「名称写法完全不同但语义同一」的实体（如「OpenAI」与「开放人工
    智能公司」）进一步对齐合并。

    流程：
    1. 先补齐缺失向量（手工新增实体可能尚未向量化，复用增量向量化入口）；
    2. 从 kg_vectors 载入全部实体向量（name+description 文本向量）；
    3. 分块余弦相似度，收集 ≥ threshold 的候选对（同对取最高相似度）；
    4. 按相似度降序合并：证据多/类型明确者为保留方，类型冲突不合并
       （与规则消歧同一守卫），合并走的仍是 :meth:`KGStore.merge_entities`
       ——证据聚合、别名转移、关系端点改写、向量清理全部复用。

    参数 progress_callback(done, total, message)；cancel_event 置位后抛
    :class:`KGCancellation`。返回统计 dict。
    """
    from .builder import KGCancellation, vectorize_pending_rows
    from .embedding import KGEmbedder

    import numpy as np

    progress_callback = progress_callback or (lambda d, t, m: None)
    stats = {"scanned": 0, "merged": 0, "threshold": float(threshold)}
    if embedding_function is not None:
        # 补齐缺失向量（实体 + 关系），保证后续比对覆盖全部实体。
        vectorize_pending_rows(store, KGEmbedder(embedding_function))
    entities = list(store.iter_entities())
    stats["scanned"] = len(entities)
    if len(entities) < 2:
        return stats

    # 载入实体向量（缺向量的实体跳过，不参与本轮比对）
    ids: List[str] = []
    rows: List[np.ndarray] = []
    for e in entities:
        blob = store.get_vector(f"ent:{e['id']}")
        if blob is None:
            continue
        vec = np.frombuffer(blob, dtype=np.int8).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm <= 0:
            continue
        rows.append(vec / norm)
        ids.append(e["id"])
    if len(ids) < 2:
        return stats
    mat = np.stack(rows)
    n = len(ids)

    # 分块余弦相似度 → 候选对（i<j，sim≥threshold）
    pairs: List[tuple] = []
    for start in range(0, n, _SIM_BLOCK):
        if cancel_event is not None and cancel_event.is_set():
            raise KGCancellation()
        block = mat[start:start + _SIM_BLOCK]
        sim = block @ mat.T  # (b, n)
        for bi in range(sim.shape[0]):
            i = start + bi
            hits = np.nonzero(sim[bi] >= threshold)[0]
            for j in hits:
                j = int(j)
                if j <= i:
                    continue
                pairs.append((float(sim[bi][j]), i, j))
        progress_callback(min(start + _SIM_BLOCK, n), n,
                          f"实体向量比对 {min(start + _SIM_BLOCK, n)}/{n}")
    pairs.sort(key=lambda p: p[0], reverse=True)

    # 合并：并查集把 pair 落到根实体，避免 A→B→C 链式重复合并
    parent: Dict[str, str] = {}

    def _root(eid: str) -> str:
        while eid in parent:
            eid = parent[eid]
            if eid not in parent:
                break
        return eid

    for _sim, i, j in pairs:
        if cancel_event is not None and cancel_event.is_set():
            # 已完成部分已入库；重新调用可续跑（合并幂等）。
            raise KGCancellation()
        ra, rb = _root(ids[i]), _root(ids[j])
        if ra == rb:
            continue
        ea = store.get_entity(ra)
        eb = store.get_entity(rb)
        if ea is None or eb is None:
            continue  # 快照过期等异常：跳过该对
        if not types_compatible(ea.get("type"), eb.get("type")):
            continue
        target, other = ((ea, eb) if _keep_priority(ea) >= _keep_priority(eb)
                         else (eb, ea))
        result = store.merge_entities(other["id"], target["id"])
        if result.get("entities"):
            parent[other["id"]] = target["id"]
            stats["merged"] += 1
            logger.info("KG: 语义消歧合并 %.4f %r → %r",
                        _sim, other["name"], target["name"])
    if stats["merged"] and embedding_function is not None:
        # 合并改变了实体 description → content_hash 变化，刷新向量。
        vectorize_pending_rows(store, KGEmbedder(embedding_function))
    return stats
