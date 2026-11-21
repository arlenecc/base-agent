"""计划评审与收尾拦截：决策模型在计划生命周期上的两个判定点。

1. **计划评审（计划步骤合理性）**：``update_plan(op='create')`` 建计划时，
   决策模型对「目标 × 每个步骤标题」打相关分（0~1）；平均分低于阈值 →
   拒绝该计划并给出低相关步骤清单，让模型重新完善计划（同一目标最多
   拒绝 ``PLAN_REVIEW_MAX_REJECTS`` 次，防止 create↔拒绝 死循环）。
2. **收尾拦截（计划是否最终正确完成）**：agent 判定 DONE 准备收尾时，
   决策模型对「用户请求 × 证据（已完成步骤 title+note，无计划则用最终
   答复）」打相关分；平均分低于阈值 → 判定未真正完成（假装收尾 / 信息
   不足直答），反馈经 ``__plan_boundary__`` 注入并强制续跑，最多
   ``COMPLETION_MAX_FORCE`` 次。

决策模型不可用（``get_decision_model`` 返回 None）→ 两个拦截点都静默
跳过，行为与原方式完全一致。
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 计划评审：步骤与目标的平均相关分低于该值 → 拒绝计划（退回模型完善）。
PLAN_REVIEW_THRESHOLD = 0.45
# 计划评审：单步相关分低于该值 → 在反馈中点名「相关性偏低」。
PLAN_STEP_LOW = 0.30
# 计划评审：同一目标最多拒绝次数（第 3 次 create 直接放行）。
PLAN_REVIEW_MAX_REJECTS = 2
# 收尾拦截：请求与证据的平均相关分低于该值 → 判定未完成。
# 校准（真实 Meteor 实测）：代表性证据（done 步骤 title+note，含任务关键词）
# 0.46~0.88；无关证据 0.008~0.19（「完成了」类空话 0.188）。0.40 两侧留有
# 余量；过严（0.45+）会误拦省略关键词的简短真证据（实测 0.356 一例）。
COMPLETION_THRESHOLD = 0.40
# 收尾拦截：同一用户轮次内最多强制续跑次数。
COMPLETION_MAX_FORCE = 2

# sess.working 里记录「同一目标已被拒绝次数」的键（按目标键控：
# 换目标/新任务自动重新计数，不依赖 working 是否按轮清空）。
PLAN_REVIEW_REJECTS_KEY = "__plan_review_rejects__"


def load_plan_model(config: Any) -> Optional[Any]:
    """按配置加载决策模型（进程级单例）；不可用 → None（拦截点跳过）。"""
    path = str(getattr(config, "rag_decision_model_path", "") or "")
    try:
        from .rag.decision_model import get_decision_model
        return get_decision_model("mlx-vlm", path)
    except Exception as e:
        logger.debug("决策模型加载失败，计划评审/收尾拦截跳过: %s", e)
        return None


def _mean(scores: Iterable[float]) -> float:
    scores = list(scores)
    return sum(scores) / len(scores) if scores else 1.0


def review_plan_steps(
    model: Optional[Any],
    goal: str,
    steps: List[dict],
) -> Tuple[bool, str]:
    """评审计划步骤合理性。返回 (是否通过, 拒绝反馈)。

    - 模型不可用 / 单步计划 / 打分失败 → 通过（按原方式）；
    - 步骤标题与目标的平均相关分 ≥ 阈值 → 通过；
    - 否则拒绝，反馈列出相关性明显偏低的步骤并要求重新规划。
    """
    titles = [str(s.get("title") or "").strip() for s in steps]
    titles = [t for t in titles if t]
    if model is None or not titles:
        return True, ""
    try:
        scores = [float(s) for s in model.score(goal, titles)]
    except Exception as e:
        logger.warning("计划评审打分失败，放行: %s", e)
        return True, ""
    mean = _mean(scores)
    if mean >= PLAN_REVIEW_THRESHOLD:
        return True, ""
    lines = [
        f"计划评审未通过：{len(titles)} 个步骤与目标的平均相关性 "
        f"{mean:.2f} 低于 {PLAN_REVIEW_THRESHOLD:.2f}。"
    ]
    low = [(t, s) for t, s in zip(titles, scores) if s < PLAN_STEP_LOW]
    if low:
        lines.append("相关性明显偏低的步骤：")
        lines.extend(f"  · {t}（相关分 {s:.2f}）" for t, s in low[:5])
    lines.append(
        "请调用 update_plan(op='create') 重新规划：步骤要直接服务于目标、"
        "写明每步的具体动作与产出，删除与目标无关或含义模糊的步骤。")
    return False, "\n".join(lines)


def check_plan_completion(
    model: Optional[Any],
    request: str,
    evidence_texts: List[str],
) -> Tuple[bool, str]:
    """收尾拦截：判定任务是否真正完成。返回 (是否完成, 续跑反馈)。

    - 模型不可用 / 无请求或证据 / 打分失败 → 完成（按原方式收尾）；
    - 证据与请求的平均相关分 ≥ 阈值 → 完成；
    - 否则判定未完成（假装收尾 / 信息不足直答），反馈列出证据不足的
      部分，驱动模型继续补充工具调用后再收尾。
    """
    texts = [str(t or "").strip() for t in evidence_texts if (t or "").strip()]
    if model is None or not texts or not str(request or "").strip():
        return True, ""
    try:
        scores = [float(s) for s in model.score(request, texts)]
    except Exception as e:
        logger.warning("完成度判定打分失败，放行收尾: %s", e)
        return True, ""
    mean = _mean(scores)
    if mean >= COMPLETION_THRESHOLD:
        return True, ""
    lines = [
        f"收尾拦截：决策模型判定任务尚未真正完成（证据与请求的平均相关性 "
        f"{mean:.2f} < {COMPLETION_THRESHOLD:.2f}）。"
    ]
    weak = [(t, s) for t, s in zip(texts, scores) if s < COMPLETION_THRESHOLD]
    if weak:
        lines.append("证据不足的部分：")
        lines.extend(f"  · {t[:60]}（相关分 {s:.2f}）" for t, s in weak[:5])
    lines.append(
        "请继续执行 / 补充必要的检索或工具调用，取得足够依据后再收尾；"
        "不要在没有依据的情况下直接回答。若确实无法完成，"
        "请明确说明缺失的部分并如实告知用户。")
    return False, "\n".join(lines)
