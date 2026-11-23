"""核心观点句抽取：两路融合、去重、顺序与打分前简化。"""
from __future__ import annotations

from agent.core_opinion import (
    DEFAULT_MAX_CHARS,
    DEFAULT_TOP_N,
    extract_core_opinion_sentences,
    simplify_for_scoring,
    simplify_texts_for_scoring,
)
from agent.plan_review import check_plan_completion


def _make_doc(n: int = 80) -> str:
    parts = []
    for i in range(n):
        if i == 0:
            parts.append("本报告讨论知识库检索系统的整体设计。")
        elif i == n // 2:
            parts.append("因此，本文的核心结论是：必须优先保证数据质量。")
        elif i == n - 1:
            parts.append("综上所述，检索质量取决于数据质量与评估方法。")
        else:
            parts.append(f"第{i}句补充说明系统在测试环境中的一般表现。")
    return "".join(parts)


def test_extract_fuses_two_methods_and_keeps_document_order():
    doc = _make_doc(120)
    core = extract_core_opinion_sentences(doc)

    assert 0 < len(core) <= DEFAULT_TOP_N * 2
    assert len(core) == len(set(core)), "融合后必须去重"
    # 首尾与观点句都必须覆盖（位置信号 + 观点词/TextRank 共同保证）
    assert "本报告讨论知识库检索系统的整体设计。" in core
    assert "综上所述，检索质量取决于数据质量与评估方法。" in core
    assert any("核心结论" in s for s in core)
    # 输出必须保持原文顺序，方便模型理解上下文
    positions = [doc.index(s) for s in core]
    assert positions == sorted(positions)


def test_short_text_is_not_simplified():
    short = "这是一段很短的文本。它只有两句话。"
    assert simplify_for_scoring(short) == short


def test_long_text_is_simplified_and_bounded():
    doc = _make_doc(120)
    simplified = simplify_for_scoring(doc)
    assert len(simplified) < len(doc)
    assert len(simplified) <= DEFAULT_MAX_CHARS
    assert "核心结论" in simplified
    batch = simplify_texts_for_scoring([doc, "短文本。"])
    assert len(batch) == 2
    assert len(batch[0]) < len(doc)
    assert batch[1] == "短文本。"


def test_single_long_sentence_is_truncated_to_cap():
    doc = "长" * 1000 + "。"
    simplified = simplify_for_scoring(doc)
    assert len(simplified) == DEFAULT_MAX_CHARS
    assert simplified.endswith("…")


def test_numeric_and_constraint_facts_survive_compression():
    """中段数字/约束句即使不在 TopN，也必须被保真席位保留。"""
    parts = ["总则：本制度适用于公司全体员工。"]
    parts.extend(f"第{i}条补充说明一般审批流程与注意事项。" for i in range(1, 30))
    parts.append("差旅住宿标准：一线城市每晚上限 500 元，其他城市每晚上限 350 元。")
    parts.append("出差交通应优先选择高铁二等座或飞机经济舱，未经审批不得乘坐商务座。")
    parts.extend(f"第{i}条补充说明其他管理要求。" for i in range(30, 60))
    doc = "".join(parts)

    out = simplify_for_scoring(doc, top_n=10)
    assert len(out) <= DEFAULT_MAX_CHARS
    assert "500 元" in out and "350 元" in out
    assert "二等座" in out and "经济舱" in out


def test_general_numeric_formats_are_preserved():
    """端口、超时、配置值、版本、区间、百分比等通用格式都应保真。"""
    parts = ["配置说明：本系统支持多种运行参数。"]
    parts.extend(f"第{i}项为常规说明，不涉及关键参数。" for i in range(1, 40))
    parts.append("服务默认监听端口为 8080，请求超时时间为 30 秒。")
    parts.append("配置文件要求 max_tokens=32768，版本号 v2.1.0。")
    parts.append("批处理间隔范围是 1~5 分钟，重试比例为百分之三十。")
    parts.extend(f"第{i}项为其他补充说明。" for i in range(40, 80))
    doc = "".join(parts)

    out = simplify_for_scoring(doc, top_n=10)
    assert len(out) <= DEFAULT_MAX_CHARS
    for token in ("8080", "30 秒", "32768", "v2.1.0", "1~5", "百分之三十"):
        assert token in out


def test_plan_completion_simplifies_evidence_before_scoring():
    class _RecordingModel:
        def __init__(self):
            self.texts = []

        def score(self, query, texts):
            self.texts = list(texts)
            return [0.9] * len(texts)

    doc = _make_doc(120)
    model = _RecordingModel()
    ok, feedback = check_plan_completion(model, "总结系统设计", [doc])

    assert ok and not feedback
    assert model.texts, "决策模型必须被调用"
    assert len(model.texts[0]) < len(doc), "长证据应先简化再打分"
    assert len(model.texts[0]) <= DEFAULT_MAX_CHARS
    assert "核心结论" in model.texts[0]
