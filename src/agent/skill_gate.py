"""技能加载前判定：决策模型按用户问句与对话状态挑选/校验技能。

链路：检索（keyword / semantic / 自动提示）→ **判定** → 加载使用。

- 检索返回多个技能：决策模型对「用户问句 + 对话状态」与各技能描述打分，
  只有相关分最高的技能允许加载；请求加载其它技能会被拒绝并指明应加载哪个。
- 检索仅返回一个技能：同样打分，低于阈值判定为不合适，**禁止加载使用**。
- 最高分也低于阈值：检索结果均不匹配，禁止加载任何技能。
- 决策模型不可用 / 本回合没有检索记录 → 放行（保持原行为，零额外开销）。

候选状态记录在 ``ToolRegistry._skill_gate_state``（按回合键控：上一回合的
检索记录不会误伤本回合的加载）。
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 技能与「用户问句 + 对话状态」的相关分阈值：低于它判定为不合适。
# 校准（真实 Meteor）：匹配查询-技能对 0.62~0.93；无关对 0.01~0.22。
# 统一作用于：多候选择优、加载前校验、Active skill 注入、自动提示摘要注入。
SKILL_SUITABILITY_THRESHOLD = 0.55
# 请求文本里携带的「对话状态」：最近已落地轮次的一行摘要条数。
_CONVERSATION_STATE_LINES = 2
# 单个技能描述参与打分的最大字符数（过长的 SKILL.md 正文不参与）。
_MAX_DESC_CHARS = 220

_STATE_ATTR = "_skill_gate_state"


def _turn_of(registry: Any) -> int:
    sess = getattr(registry, "session", None)
    try:
        return int(getattr(sess, "turn_index", 0) or 0)
    except Exception:
        return 0


def _norm_candidate(c: Dict[str, Any]) -> Dict[str, str]:
    return {
        "name": str(c.get("name") or "").strip(),
        "dir": str(c.get("dir") or c.get("path") or "").strip(),
        "path": str(c.get("path") or c.get("dir") or "").strip(),
        "description": str(c.get("description") or "").strip()[:_MAX_DESC_CHARS],
        "tags": str(c.get("tags") or "").strip(),
    }


def clear_state(registry: Any) -> None:
    """清空检索记录（每个**用户回合**开始时调用）。

    不能用 Session.turn_index 判断新旧：它是内核每轮计数，一次用户请求内
    会递增多次，会把同一回合内的「先检索后加载」误判为跨回合。
    """
    try:
        delattr(registry, _STATE_ATTR)
    except AttributeError:
        pass


def record_candidates(registry: Any, query: str, candidates: List[Dict[str, Any]]) -> None:
    """记录本回合检索返回的技能候选（供加载前判定）。

    ``candidates`` 为空也记录（``searched=True``）：本回合检索过且没有结果时，
    加载任何技能都缺乏依据，应被判定拦截。
    """
    norm = [_norm_candidate(c) for c in candidates if (c.get("name") or c.get("dir")
                                                       or c.get("path"))]
    setattr(registry, _STATE_ATTR, {
        "turn": _turn_of(registry),
        "query": str(query or "").strip(),
        "candidates": norm,
        "searched": True,
    })


def record_hint(registry: Any, query: str, candidate: Dict[str, Any]) -> None:
    """把自动提示（系统提示里的 Relevant skill）并入本用户回合候选，去重。"""
    state = getattr(registry, _STATE_ATTR, None)
    cand = _norm_candidate(candidate)
    if not isinstance(state, dict) or not state.get("searched"):
        record_candidates(registry, query, [cand])
        return
    existing = {_norm_candidate(c)["dir"] or _norm_candidate(c)["name"]
                for c in state.get("candidates") or []}
    key = cand["dir"] or cand["name"]
    if key and key not in existing:
        state["candidates"].append(cand)


def _request_text(state: Dict[str, Any], registry: Any) -> str:
    """判定输入：用户问句 + 最近对话状态（摘要行）。"""
    sess = getattr(registry, "session", None)
    parts: List[str] = []
    try:
        turns = list(getattr(sess, "turns", []) or [])
    except Exception:
        turns = []
    if turns:
        last = turns[-1]
        prompt = str(getattr(last, "prompt", "") or "").strip()
        if prompt:
            parts.append(prompt)
        digest = [t.digest_line() for t in turns[-1 - _CONVERSATION_STATE_LINES:-1]]
        if digest:
            parts.append("对话进展：" + "；".join(digest))
    if not parts:
        parts.append(str(state.get("query") or ""))
    return "\n".join(p for p in parts if p)


def _candidate_text(c: Dict[str, Any]) -> str:
    text = str(c.get("name") or "")
    desc = str(c.get("description") or "")
    tags = str(c.get("tags") or "")
    if desc:
        text = f"{text}：{desc}"
    if tags:
        text = f"{text}（{tags}）"
    return text


def _identifiers(c: Dict[str, Any]) -> set:
    out = set()
    for key in ("dir", "path", "name"):
        v = str(c.get(key) or "").strip()
        if not v:
            continue
        out.add(v)
        out.add(os.path.normpath(v))
        base = os.path.basename(os.path.normpath(v))
        if base:
            out.add(base)
        try:
            out.add(os.path.realpath(v))
        except Exception:
            pass
    return out


def _match_requested(candidates: List[Dict[str, Any]], requested: str) -> Optional[int]:
    want = set()
    v = str(requested or "").strip()
    if not v:
        return None
    want.add(v)
    want.add(os.path.normpath(v))
    base = os.path.basename(os.path.normpath(v))
    if base:
        want.add(base)
    for i, c in enumerate(candidates):
        ids = _identifiers(c)
        if want & ids:
            return i
    return None


def _score(model: Any, request: str, texts: List[str]) -> Optional[List[float]]:
    if not texts:
        return None
    try:
        return [float(s) for s in model.score(request, texts)]
    except Exception as e:
        logger.warning("技能判定打分失败，放行: %s", e)
        return None


def judge_skill_load(registry: Any, config: Any,
                     requested_path: str) -> Tuple[bool, str]:
    """加载前判定：返回 (是否允许, 拒绝原因)。

    - 无本回合检索记录 / 模型不可用 → 允许（原行为）；
    - 本回合检索无结果 → 禁止加载（无依据）；
    - 请求的技能不在本回合检索结果中 → 禁止加载（只能加载检索返回的技能）；
    - 多候选：仅最高分技能允许；请求其它技能 → 拒绝并指明应加载项；
    - 单候选或最高分：低于阈值 → 拒绝（技能与任务不匹配，禁止加载使用）。
    """
    state = getattr(registry, _STATE_ATTR, None)
    if not isinstance(state, dict) or not state.get("searched"):
        return True, ""
    from . import plan_review
    model = plan_review.load_plan_model(config)
    if model is None:
        return True, ""
    cands = state.get("candidates") or []
    if not cands:
        return False, (
            "技能判定未通过：本回合技能检索没有返回任何结果，禁止加载使用技能。"
            "请直接按常规工具流程处理当前任务，或换一组关键词重新检索。")
    idx = _match_requested(cands, requested_path)
    if idx is None:
        names = ", ".join(str(c.get("name") or "?") for c in cands[:5])
        return False, (
            f"技能判定未通过：请求加载的技能不在本回合检索结果中。"
            f"只能加载检索返回的技能：{names}。")
    request = _request_text(state, registry)
    scores = _score(model, request, [_candidate_text(c) for c in cands])
    if scores is None:
        return True, ""
    best_i = max(range(len(cands)), key=lambda i: scores[i])
    best_score = scores[best_i]
    best = cands[best_i]
    logger.info("技能判定：候选 %d 个，最高 %r=%.2f，请求 %r=%.2f",
                len(cands), best.get("name"), best_score,
                cands[idx].get("name"), scores[idx])
    if best_score < SKILL_SUITABILITY_THRESHOLD:
        return False, (
            f"技能判定未通过：检索到的技能与当前任务均不匹配"
            f"（最高相关分 {best_score:.2f} < {SKILL_SUITABILITY_THRESHOLD:.2f}）。"
            "禁止加载使用任何技能，请直接按常规工具流程处理当前任务。")
    if idx != best_i:
        target = best.get("dir") or best.get("path") or best.get("name")
        return False, (
            f"技能判定未通过：决策模型判定应使用技能 {best.get('name')!r}"
            f"（相关分 {best_score:.2f}），而不是 {cands[idx].get('name')!r}"
            f"（相关分 {scores[idx]:.2f}）。请改为加载 {target}。")
    return True, ""


def judge_skill_injection(registry: Any, config: Any, name: str,
                          description: str) -> Tuple[bool, str]:
    """自动注入（Active skill）前的单技能判定：不合适则禁止使用。"""
    from . import plan_review
    model = plan_review.load_plan_model(config)
    if model is None:
        return True, ""
    state = getattr(registry, _STATE_ATTR, None)
    request = _request_text(
        state if isinstance(state, dict) else {"query": ""}, registry)
    if not request:
        return True, ""
    scores = _score(model, request, [f"{name}：{description}"] if description else [name])
    if scores is None:
        return True, ""
    if scores[0] < SKILL_SUITABILITY_THRESHOLD:
        return False, (
            f"技能判定未通过：{name!r} 与当前任务不匹配"
            f"（相关分 {scores[0]:.2f} < {SKILL_SUITABILITY_THRESHOLD:.2f}），"
            "已禁止使用。")
    return True, ""
