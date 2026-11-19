"""多知识分区联合检索：决策模型路由 + 跨分区加权 RRF 融合。

流程（查询期，多分区且启用 ≥2 时生效）：

1. **路由**：Meteor 决策模型对每个分区名（领域标签）与问句打相关性分
   （0~1），取 >threshold 的分区按分降序；全部低于阈值时保底最高分 1 个；
   模型不可用 / 仅 1 个启用分区 → 跳过路由（零额外开销）。
2. **各分区粗筛**：调用方按模式自行检索（vector/fast = 各库混合检索，
   decision = 各库 BM25 粗筛 coarse_k 条）。
3. **跨库融合**：路由得分 min-max 归一化为库权重 → weighted RRF
   （复用 vector_store._rrf_fuse，key=(分区, source, chunk_index)）。
4. **判定**（decision 模式，先融合后判定）：融合池取前 ``FUSION_POOL``
   条 → 核心句提取（复用 decision_search.extract_core_sentences）→
   判定模型一次批量打分 → 取前 ``MULTI_DECISION_TOP`` 返回。

多分区联合时由引擎侧屏蔽 reranker（use_reranker=False），避免 N 库 ×
cross-encoder 的长停顿。
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .decision_search import extract_core_sentences
from .vector_store import _rrf_fuse

logger = logging.getLogger(__name__)

# 路由判定阈值：分区名与问句的相关性得分（0~1）超过它才入选。
ROUTE_THRESHOLD = 0.5
# 多库联合（decision 模式）：融合池大小（各库粗筛候选加权 RRF 后保留的条数）。
FUSION_POOL = 20
# 多库联合（decision 模式）：判定后返回的相关切片数。
MULTI_DECISION_TOP = 3


def route_partitions(
    query: str,
    partitions: Sequence[Dict[str, Any]],
    model: Optional[Any],
    threshold: float = ROUTE_THRESHOLD,
    hit_probe: Optional[Any] = None,
) -> List[Tuple[Dict[str, Any], float]]:
    """决策模型路由：返回 [(分区, 路由得分)] 按分降序。

    - 仅 1 个分区或模型不可用 → 返回全部分区（得分 1.0，等权），零开销；
    - 得分 >threshold 的入选；全部低于阈值 → 两级保底：
      1. ``hit_probe``（各分区 BM25 命中探针）有命中的分区入选——Meteor 对
         短领域名的区分度有限，BM25 命中是客观的相关性证据；
      2. 仍无命中 → 保底 Meteor 最高分 1 个（保证检索永远有落点）。
    """
    if not partitions:
        return []
    if len(partitions) == 1 or model is None:
        return [(p, 1.0) for p in partitions]
    names = [str(p.get("name") or "") for p in partitions]
    try:
        scores = model.score(query, names)
    except Exception as e:
        logger.warning("  └─ ⚠ 分区路由打分失败，退化为全分区等权检索: %s", e)
        return [(p, 1.0) for p in partitions]
    scored = list(zip(partitions, [float(s) for s in scores]))
    selected = [(p, s) for p, s in scored if s > threshold]
    if not selected:
        if hit_probe is not None:
            try:
                with_hits = [p for p in partitions if hit_probe(p)]
            except Exception as e:
                logger.warning("  └─ ⚠ 分区命中探针失败: %s", e)
                with_hits = []
            if with_hits:
                logger.info(
                    "  ├─ 分区路由: Meteor 全部低于阈值 %.2f（区分度不足），"
                    "按 BM25 命中证据路由 %d 库: %s",
                    threshold, len(with_hits),
                    ", ".join(str(p.get("name")) for p in with_hits))
                return [(p, threshold) for p in with_hits]
        best = max(scored, key=lambda ps: ps[1])
        logger.info("  ├─ 分区路由: 全部低于阈值 %.2f，保底最高分分区 %r (%.3f)",
                    threshold, best[0].get("name"), best[1])
        selected = [best]
    selected.sort(key=lambda ps: ps[1], reverse=True)
    logger.info("  ├─ 分区路由: %s", ", ".join(
        f"{p.get('name')}={s:.2f}" for p, s in selected))
    return selected


def fuse_partition_results(
    per_partition_results: Dict[str, List[Dict[str, Any]]],
    route_scores: Dict[str, float],
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """跨分区加权 RRF 融合。

    ``per_partition_results``: {分区名: 该库检索候选列表（已按相关度降序）}；
    ``route_scores``: {分区名: 路由得分}。路由得分 min-max 归一化到 [0.3, 1.0]
    作为各库 RRF 权重（下限 0.3 保底，避免低分库完全失声）；融合后每条候选
    附 ``partition`` 字段，按融合分降序返回前 ``top_k`` 条。
    """
    if not per_partition_results:
        return []
    scores = {name: max(0.0, float(route_scores.get(name, 1.0)))
              for name in per_partition_results}
    mx = max(scores.values()) if scores else 1.0
    mn = min(scores.values()) if scores else 0.0
    span = (mx - mn) or 1.0
    weights = {name: 0.3 + 0.7 * (s - mn) / span for name, s in scores.items()}
    logger.info("  ├─ 跨库融合权重: %s", ", ".join(
        f"{name}={w:.2f}" for name, w in weights.items()))

    rankings: List[List[Dict[str, Any]]] = []
    names: List[str] = []
    for name, results in per_partition_results.items():
        if results:
            names.append(name)
            rankings.append(results)
    if not rankings:
        return []

    def _key(item: Dict[str, Any]) -> Tuple[str, str, int]:
        # 跨库 key 必须含分区名：同名文件/同 chunk_index 在不同库是不同候选
        return (item.get("_partition", ""),
                item.get("source") or "",
                int(item.get("chunk_index") or 0))

    fused = _rrf_fuse(rankings, _key, weights=[weights[n] for n in names])
    out: List[Dict[str, Any]] = []
    for item, score in fused:
        rec = dict(item)
        rec["partition"] = item.get("_partition", "")
        rec["fusion_score"] = round(float(score), 4)
        rec.pop("_partition", None)
        out.append(rec)
    return out[:top_k] if top_k else out


def multi_decision_search(
    engine: Any,
    query: str,
    partitions: Sequence[Tuple[Dict[str, Any], float]],
    model: Optional[Any],
    coarse_k: int = 50,
    use_core_sentences: bool = True,
    core_top: int = 3,
    where: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """多库联合判定管线（decision 模式）：先融合后判定。

    1. 各分区 BM25 粗筛 ``coarse_k`` 条（store.search_fts，``where`` 为可选
       定向检索过滤）；
    2. 全部候选按路由得分加权 RRF 融合，取前 ``FUSION_POOL`` 条；
    3. 融合候选提取核心句（截短输入），判定模型**一次批量**打分；
    4. 取判定分最高 ``MULTI_DECISION_TOP`` 条返回。

    ``engine`` 需暴露 ``_get_store(partition)`` 与 ``_partition_slug(name)``。
    """
    t0 = time.monotonic()
    per_partition: Dict[str, List[Dict[str, Any]]] = {}
    route_scores: Dict[str, float] = {}
    store_names: Dict[str, str] = {}
    t_coarse_start = time.monotonic()
    for p, score in partitions:
        name = str(p.get("name") or "")
        slug = engine._partition_slug(name)
        store = engine._get_store(name)
        store_names[name] = slug
        route_scores[name] = float(score)
        try:
            cands = store.search_fts(query, top_k=coarse_k, where=where) or []
        except Exception as e:
            logger.warning("  └─ ⚠ 分区 %r BM25 粗筛失败: %s", name, e)
            cands = []
        tagged = []
        for c in cands:
            rec = dict(c)
            rec["_partition"] = name
            tagged.append(rec)
        per_partition[name] = tagged
    t_coarse = time.monotonic() - t_coarse_start
    total_cands = sum(len(v) for v in per_partition.values())
    if not total_cands:
        logger.info("  └─ 多库联合检索: 各分区粗筛均无结果 (耗时 %.3fs)", t_coarse)
        return []

    # ---- 分区配额：每库取头部 ceil(FUSION_POOL/n) 条进融合池 ----
    # 直接对全量候选（每库 50 条）做加权 RRF 会被高权库垄断：权重 1.0 vs 0.3
    # 时，高权库 rank N (1/(60+N)) 直到 N≈144 才低于低权库 rank1 (0.3/61)，
    # 融合池 20 全被一个库占满，判定阶段看不到其它分区的候选。配额制保证
    # 每个路由命中的分区都有代表性候选进入判定。
    active = [n for n, v in per_partition.items() if v]
    quota = max(1, FUSION_POOL // len(active)) if active else 0
    for n in active:
        per_partition[n] = per_partition[n][:quota]

    # ---- 跨库加权 RRF 融合，取融合池 ----
    t1 = time.monotonic()
    fused = fuse_partition_results(per_partition, route_scores,
                                   top_k=FUSION_POOL)
    t_fuse = time.monotonic() - t1

    # ---- 核心句 + 判定模型一次批量评判 ----
    t2 = time.monotonic()
    judged_texts: List[str] = []
    used_model = model is not None
    decision_scores: List[float] = []
    for cand in fused:
        chunk_text = cand.get("text") or ""
        if use_core_sentences:
            core_text, _ = extract_core_sentences(chunk_text, query,
                                                  top_n=core_top)
            judged_texts.append(core_text or chunk_text.strip())
        else:
            judged_texts.append(chunk_text.strip())
    if model is not None:
        try:
            decision_scores = model.score(query, judged_texts)
        except Exception as e:
            logger.warning("  └─ ⚠ 多库判定模型批量评判失败，回退融合分排序: %s", e)
            model = None
            used_model = False
    if model is None:
        # 回退：融合分归一化（0~1 口径）
        mx = max((c.get("fusion_score") or 0.0) for c in fused) or 1.0
        decision_scores = [
            min(1.0, (c.get("fusion_score") or 0.0) / mx) for c in fused]
    t_decision = time.monotonic() - t2

    # ---- 排序取最终 TopN ----
    t3 = time.monotonic()
    ranked = sorted(range(len(fused)),
                    key=lambda i: decision_scores[i], reverse=True)
    results: List[Dict[str, Any]] = []
    for rank, i in enumerate(ranked[:MULTI_DECISION_TOP], 1):
        cand = fused[i]
        cand.pop("_partition", None)
        cand["score"] = round(float(decision_scores[i]), 4)
        cand["core_sentences"] = judged_texts[i]
        cand["decision_mode"] = True
        cand["multi_partition"] = True
        results.append(cand)
        logger.info("      [%d] %s@%s (判定分: %.4f)", rank,
                    cand.get("partition", "?"),
                    os.path.basename(cand.get("source") or ""),
                    decision_scores[i])
    t_rank = time.monotonic() - t3
    logger.info(
        "    └─ ⏱ 多库联合判定: 路由 %d 库 | 粗筛 %d 条 %.3fs | 融合 %.3fs "
        "| 判定 %d 片 %.3fs%s | 总计 %.3fs",
        len(partitions), total_cands, t_coarse, t_fuse, len(fused),
        t_decision, "（模型）" if used_model else "（回退融合分）",
        time.monotonic() - t0)
    return results
