"""核心观点句抽取：多信号启发式 + TextRank 观点词加权，两路 Top10 融合。

用途：在把长文本交给决策模型打分/评估前，先压成少量代表性句子，降低
前向开销；两路各取 10 句再合并去重（默认最多 20 句），并按得分装箱
限制在 800 字以内；同时对数字、单位、比例、区间、日期、版本、配置值、
阈值/期限等通用数字/约束信号额外加分并保留保真席位，避免压缩把关键
约束挤掉。兼顾「位置/观点词/中心度」与「全局图重要性」两种视角，避免
单一方法漏掉核心句。

注意：RAG 召回切片判定（``rag/decision_search.py``）用的是「查询句 vs
切片内句子」的字符级 BM25，只取 3~5 句、速度优先，**不走本模块**；
本模块面向的是需要覆盖面、且与具体查询无关的长文本简化。
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 两路各取 10 句，合并去重后最多 20 句。
DEFAULT_TOP_N = 10
# 文本短于该长度不简化：短文本直接送模型更完整，也避免无谓开销。
DEFAULT_MIN_CHARS = 600
# 简化后的总字数硬上限：超出时按句子得分取舍，确保送模型的输入可控。
DEFAULT_MAX_CHARS = 800
# TextRank 图节点上限：超长文本等距抽样 + 观点句补样，防止 O(n²) 爆炸。
MAX_GRAPH_SENTENCES = 600
# 相似度矩阵的 bigram 特征上限：避免超长句把稠密矩阵撑爆内存。
MAX_GRAPH_FEATURES = 8192

# 观点指示词：情态动词 / 认知动词 / 逻辑关联词。
OPINION_WORDS = (
    "认为", "必须", "应该", "核心", "本质", "实际上", "意味着", "导致",
    "因此", "所以", "综上所述", "指出", "强调", "不可忽视",
)

# 多信号权重（Score = Σ w_i * signal_i）。
W_POSITION = 0.35
W_LEXICON = 0.25
W_CENTRALITY = 0.25
W_LENGTH = 0.15
# 数字/实体保真加分：技术参数、制度约束中的金额、时限、比例、阈值、版本、
# 端口、配置值等往往落在数字句里，纯位置/观点词信号容易挤掉。
W_FACT = 0.25
# 保真席位：两路 TopN 之外，额外纳入数字句/约束句。
FACT_GUARANTEE_TOP = 8
# 其中给「无数值的约束句」（不得/需审批/优先选择等）保留的席位，
# 避免它们被大量数字句挤掉。
FACT_CONSTRAINT_SEATS = 2

# TextRank 参数。
TEXTRANK_DAMPING = 0.85
TEXTRANK_ITERS = 20
TEXTRANK_INIT_OPINION = 2.0
TEXTRANK_INIT_NORMAL = 0.5

_MIN_SENT_CHARS = 2
_SENT_SPLIT_RE = re.compile(r"[。！？；!?;]+|[.](?=\s|$)|[\n]{2,}")
# 字符级 2-gram 前先去掉空白与标点，避免标点主导相似度。
_CLEAN_RE = re.compile(
    r"[\s，。！？；：、,.!?;:'\"“”‘’()（）\[\]【】<>《》—\-_/\\|]+")

# 通用数字/实体模式：不绑定具体业务词，覆盖中英文数字、单位、比例、
# 区间、日期时间、版本号、配置键值、条款编号和等级标识。
_CN_NUM = r"[一二三四五六七八九十百千万两零〇]+"
_FACT_NUM = rf"(?:\d+(?:\.\d+)?|{_CN_NUM})"
_FACT_UNIT = (
    r"%|％|元|万元|亿元|万|亿|天|日|个工作日|工作日|个月|月|年|小时|分钟|"
    r"秒|毫秒|ms|s|次|人|名|位|晚|公里|千米|km|米|厘米|毫米|kg|g|L|ml|"
    r"GB|MB|KB|TB|台|份|周|季度|级|档|℃|°C|W|V|A|Hz|"
    r"核|线程|并发|维|层"
)
_FACT_PATTERNS = (
    re.compile(rf"{_FACT_NUM}\s*(?:{_FACT_UNIT})", re.I),
    re.compile(rf"{_CN_NUM}分之[一二三四五六七八九十百千万零〇\d]+"),
    re.compile(r"\d+(?:\.\d+)?\s*(?:-|~|—|至|到)\s*\d+(?:\.\d+)?"),
    re.compile(r"\d{4}\s*[-/年]\s*\d{1,2}(?:\s*[-/月]\s*\d{1,2})?(?:\s*[日号])?"),
    re.compile(r"\d{1,2}\s*[:：]\s*\d{2}"),
    re.compile(r"\bv?\d+(?:\.\d+){1,3}\b", re.I),
    re.compile(r"\b[A-Za-z_][\w.]*\s*[=:]\s*-?\d+(?:\.\d+)?\b"),
    re.compile(r"[A-Za-z甲乙丙丁戊一二三四五六七八九十]+级"),
    re.compile(r"(?<![\w.])\d{3,}(?![\w.])"),
)
# 无数值的通用约束词：阈值、期限、审批、范围、优先级等。
_CONSTRAINT_TERMS = (
    "不超过", "不得超过", "不低于", "不少于", "至少", "最多", "上限", "下限",
    "限额", "额度", "阈值", "标准", "比例", "期限", "时限", "有效期",
    "截止", "提前", "以内", "以上", "以下", "每次", "每月", "每周", "每年",
    "每日", "每季度", "按天", "按月", "年度", "季度", "不得", "必须",
    "应当", "只能在", "仅限", "优先选择", "需要审批", "需审批", "未经审批",
)


def _split_sentences(text: str) -> List[str]:
    """按句末标点切句；优先复用 doc_chunker 的实现，失败时本地兜底。"""
    try:
        from .rag.doc_chunker import _split_sentences as _split
        return [u.text.strip() for u in _split(text or "") if u.text.strip()]
    except Exception:  # pragma: no cover - 仅依赖异常兜底
        return [p.strip() for p in _SENT_SPLIT_RE.split(text or "")
                if p.strip()]


def _clean(text: str) -> str:
    return _CLEAN_RE.sub("", text or "").lower()


def _bigrams(text: str) -> Counter:
    """字符级 2-gram（拉丁文同样按字符切，保持统一口径）。"""
    s = _clean(text)
    if len(s) < 2:
        return Counter([s]) if s else Counter()
    return Counter(s[i:i + 2] for i in range(len(s) - 1))


def _norm(counter: Counter) -> float:
    return math.sqrt(sum(v * v for v in counter.values()))


def _cosine(a: Counter, b: Counter, norm_a: float, norm_b: float) -> float:
    if not a or not b or norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    return dot / (norm_a * norm_b)


def _position_score(index: int, total: int) -> float:
    """首句 1.0、末句 0.8、中间 0.2（论述文本的首尾效应）。"""
    if total <= 1:
        return 1.0
    if index == 0:
        return 1.0
    if index == total - 1:
        return 0.8
    return 0.2


def _length_score(sentence: str) -> float:
    """15~60 字满分；过短/过长按比例打折，最低 0.1。"""
    length = len(sentence)
    if 15 <= length <= 60:
        return 1.0
    if length < 15:
        return max(0.1, length / 15.0)
    return max(0.1, 60.0 / length)


def _fact_level(sentence: str) -> int:
    """2=数字/实体句；1=无数值约束句；0=普通句。"""
    text = sentence or ""
    if any(pattern.search(text) for pattern in _FACT_PATTERNS):
        return 2
    if any(term in text for term in _CONSTRAINT_TERMS):
        return 1
    return 0


def _constraint_hits(sentence: str) -> int:
    """命中的不同约束词数量：约束越密集，句子越可能是关键规则。"""
    text = sentence or ""
    return sum(1 for term in _CONSTRAINT_TERMS if term in text)


def _fact_score(sentence: str) -> float:
    """数字/实体保真信号：数字句 1.0，无数值约束句 0.6~0.9。"""
    level = _fact_level(sentence)
    if level == 2:
        return 1.0
    if level == 1:
        return 0.6 + 0.1 * min(3, _constraint_hits(sentence))
    return 0.0


def _opinion_density(sentences: Sequence[str]) -> List[float]:
    """观点词频率 = 命中次数 / 句子有效字数。"""
    out: List[float] = []
    for sentence in sentences:
        hits = sum(sentence.count(word) for word in OPINION_WORDS)
        length = max(1, len(_clean(sentence)))
        out.append(hits / float(length))
    return out


def _normalize(values: Sequence[float]) -> List[float]:
    top = max(values) if values else 0.0
    if top <= 0.0:
        return [0.0] * len(values)
    return [value / top for value in values]


def _similarity_matrix(counters: Sequence[Counter]) -> np.ndarray:
    """句子两两余弦相似度矩阵（n×n，对角线置 0）。

    特征维数按文档频率截断到 ``MAX_GRAPH_FEATURES``：超长句/超长文档下
    稠密矩阵不会无限膨胀；被截掉的都是低频 bigram，对句子间相似度影响小。
    """
    n = len(counters)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    df: Counter = Counter()
    for counter in counters:
        df.update(counter.keys())
    vocab = {gram: i for i, (gram, _freq)
             in enumerate(df.most_common(MAX_GRAPH_FEATURES))}
    matrix = np.zeros((n, len(vocab)), dtype=np.float32)
    for i, counter in enumerate(counters):
        for gram, value in counter.items():
            slot = vocab.get(gram)
            if slot is not None:
                matrix[i, slot] = value
    norms = np.linalg.norm(matrix, axis=1)
    denom = np.outer(norms, norms) + 1e-9
    sim = (matrix @ matrix.T) / denom
    np.fill_diagonal(sim, 0.0)
    return sim.astype(np.float64)


def _heuristic_scores(sentences: Sequence[str]) -> List[float]:
    """多信号启发式逐句打分（位置 + 观点词 + 中心度 + 长度）。"""
    n = len(sentences)
    if n == 0:
        return []
    densities = _opinion_density(sentences)
    lex = _normalize(densities)
    doc_grams = _bigrams("".join(sentences))
    doc_norm = _norm(doc_grams)
    scores: List[float] = []
    for i, sentence in enumerate(sentences):
        grams = _bigrams(sentence)
        central = _cosine(grams, doc_grams, _norm(grams), doc_norm)
        score = (W_POSITION * _position_score(i, n)
                 + W_LEXICON * lex[i]
                 + W_CENTRALITY * central
                 + W_LENGTH * _length_score(sentence)
                 + W_FACT * _fact_score(sentence))
        scores.append(score)
    return scores


def _top_indices(scores: Sequence[float], top_n: int) -> List[int]:
    """按分数取 TopN，返回按原文顺序排列的下标。"""
    order = sorted(range(len(scores)),
                   key=lambda i: scores[i], reverse=True)[:top_n]
    return sorted(order)


def _textrank_top(sentences: Sequence[str], top_n: int) -> List[int]:
    """TextRank + 观点词加权初始分，返回按原文顺序排列的 TopN 下标。"""
    n = len(sentences)
    if n == 0:
        return []
    densities = _opinion_density(sentences)
    flags = [density > 0.0 for density in densities]

    # 节点过多时等距抽样，并补入观点词最强的句子，避免漏掉核心观点。
    indices = list(range(n))
    if n > MAX_GRAPH_SENTENCES:
        step = n / float(MAX_GRAPH_SENTENCES)
        picked = {0, n - 1}
        for i in range(MAX_GRAPH_SENTENCES):
            picked.add(min(n - 1, int(i * step)))
        rich = sorted(range(n),
                      key=lambda i: (flags[i], densities[i]), reverse=True)
        picked.update(rich[:top_n])
        indices = sorted(picked)

    totals = len(indices)
    if totals <= 1:
        return indices[:top_n]

    sim = _similarity_matrix([_bigrams(sentences[i]) for i in indices])
    weights = sim.copy()
    np.fill_diagonal(weights, 0.0)
    row_sums = weights.sum(axis=1)
    dangling = row_sums <= 0.0
    weights[~dangling] /= row_sums[~dangling, None]
    weights[dangling] = 1.0 / totals

    init = np.asarray(
        [TEXTRANK_INIT_OPINION if flags[i] else TEXTRANK_INIT_NORMAL
         for i in indices], dtype=np.float64)
    init /= init.sum()
    scores = init.copy()
    for _ in range(TEXTRANK_ITERS):
        scores = ((1.0 - TEXTRANK_DAMPING) / totals
                  + TEXTRANK_DAMPING * (weights.T @ scores))

    ranked = np.argsort(-scores)[:top_n]
    return sorted(indices[int(i)] for i in ranked)


def _select_core_indices(text: str, top_n: int,
                         ) -> Tuple[List[str], List[int], List[float]]:
    """返回 (全部句子, 融合后的核心句下标, 启发式分数)。"""
    sentences = [s for s in _split_sentences(text or "")
                 if len(_clean(s)) >= _MIN_SENT_CHARS]
    if not sentences:
        return [], [], []
    scores = _heuristic_scores(sentences)
    # 句数不足两路容量时，两路都会覆盖全部句子，直接返回，省掉图计算。
    if len(sentences) <= top_n * 2:
        return sentences, list(range(len(sentences))), scores

    candidates = (set(_top_indices(scores, top_n))
                  | set(_textrank_top(sentences, top_n)))
    # 数字/实体保真席位：数字句优先，同时给无数值约束句保留固定席位，
    # 避免「不得/需审批/优先选择」这类规则被大量数字句挤掉。
    numeric = [i for i, s in enumerate(sentences) if _fact_level(s) == 2]
    constraint = [i for i, s in enumerate(sentences) if _fact_level(s) == 1]
    numeric.sort(key=lambda i: scores[i], reverse=True)
    constraint.sort(key=lambda i: scores[i], reverse=True)
    guaranteed = numeric[:FACT_GUARANTEE_TOP - FACT_CONSTRAINT_SEATS]
    guaranteed.extend(constraint[:FACT_CONSTRAINT_SEATS])
    if len(guaranteed) < FACT_GUARANTEE_TOP:
        rest = [i for i in numeric + constraint if i not in guaranteed]
        rest.sort(key=lambda i: scores[i], reverse=True)
        guaranteed.extend(rest[:FACT_GUARANTEE_TOP - len(guaranteed)])
    candidates.update(guaranteed)

    merged: List[int] = []
    seen: set = set()
    for i in sorted(candidates):
        key = _clean(sentences[i])
        if key and key not in seen:
            seen.add(key)
            merged.append(i)
    return sentences, merged, scores


def extract_core_opinion_sentences(text: str,
                                   top_n: int = DEFAULT_TOP_N) -> List[str]:
    """抽取核心观点句：两路各取 top_n（默认 10），合并去重后按原文顺序返回。"""
    sentences, indices, _scores = _select_core_indices(text, top_n)
    return [sentences[i] for i in indices]


def _fit_budget(sentences: Sequence[str], indices: Sequence[int],
                scores: Sequence[float], max_chars: int) -> List[str]:
    """按句子得分取舍，保证输出总字数不超过 ``max_chars``。

    整包已在上限内时直接返回；否则按启发式得分从高到低装箱，最后按原文
    顺序输出。单句就超上限时截断该句（保证硬上限）。
    """
    if max_chars <= 0 or not indices:
        return [sentences[i] for i in indices]
    total = (sum(len(sentences[i]) for i in indices)
             + max(0, len(indices) - 1))
    if total <= max_chars:
        return [sentences[i] for i in indices]

    order = sorted(indices, key=lambda i: scores[i], reverse=True)
    picked: List[int] = []
    used = 0
    for i in order:
        add = len(sentences[i]) + (1 if picked else 0)
        if used + add <= max_chars:
            picked.append(i)
            used += add
    if not picked:
        sentence = sentences[order[0]]
        room = max(1, max_chars - 1)
        return [sentence[:room].rstrip() + "…"]
    picked.sort()
    return [sentences[i] for i in picked]


def simplify_for_scoring(text: str, top_n: int = DEFAULT_TOP_N,
                         min_chars: int = DEFAULT_MIN_CHARS,
                         max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """把长文本压成核心观点句；短文本或压缩后反而更长时原样返回。"""
    raw = (text or "").strip()
    if len(raw) <= min_chars:
        return raw
    sentences, indices, scores = _select_core_indices(raw, top_n)
    if not indices:
        return raw
    selected = _fit_budget(sentences, indices, scores, max_chars)
    simplified = "\n".join(selected).strip()
    if not simplified or len(simplified) >= len(raw):
        return raw
    logger.debug("核心观点句压缩: %d → %d 字（%d 句）",
                 len(raw), len(simplified), len(selected))
    return simplified


def simplify_texts_for_scoring(texts: Sequence[str],
                               top_n: int = DEFAULT_TOP_N,
                               min_chars: int = DEFAULT_MIN_CHARS,
                               max_chars: int = DEFAULT_MAX_CHARS) -> List[str]:
    """批量版本：供决策模型打分前统一简化候选文本。"""
    return [simplify_for_scoring(t, top_n=top_n, min_chars=min_chars,
                                 max_chars=max_chars)
            for t in texts]
