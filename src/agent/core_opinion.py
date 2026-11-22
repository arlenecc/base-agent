"""核心观点句抽取：多信号启发式 + TextRank 观点词加权，两路 Top10 融合。

用途：在把长文本交给决策模型打分/评估前，先压成少量代表性句子，降低
前向开销；两路各取 10 句再合并去重（默认最多 20 句），兼顾
「位置/观点词/中心度」与
「全局图重要性」两种视角，避免单一方法漏掉核心句。

注意：RAG 召回切片判定（``rag/decision_search.py``）用的是「查询句 vs
切片内句子」的字符级 BM25，只取 3~5 句、速度优先，**不走本模块**；
本模块面向的是需要覆盖面、且与具体查询无关的长文本简化。
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import List, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# 两路各取 10 句，合并去重后最多 20 句。
DEFAULT_TOP_N = 10
# 文本短于该长度不简化：短文本直接送模型更完整，也避免无谓开销。
DEFAULT_MIN_CHARS = 600
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


def _heuristic_top(sentences: Sequence[str], top_n: int) -> List[int]:
    """多信号启发式打分，返回按原文顺序排列的 TopN 下标。"""
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
                 + W_LENGTH * _length_score(sentence))
        scores.append(score)
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)[:top_n]
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


def extract_core_opinion_sentences(text: str,
                                   top_n: int = DEFAULT_TOP_N) -> List[str]:
    """抽取核心观点句：两路各取 top_n（默认 10），合并去重后按原文顺序返回。"""
    sentences = [s for s in _split_sentences(text or "")
                 if len(_clean(s)) >= _MIN_SENT_CHARS]
    if not sentences:
        return []
    # 句数不足两路容量时，两路都会覆盖全部句子，直接返回，省掉图计算。
    if len(sentences) <= top_n * 2:
        return sentences

    merged: List[int] = []
    seen: set = set()
    for i in sorted(set(_heuristic_top(sentences, top_n))
                    | set(_textrank_top(sentences, top_n))):
        key = _clean(sentences[i])
        if key and key not in seen:
            seen.add(key)
            merged.append(i)
    return [sentences[i] for i in merged]


def simplify_for_scoring(text: str, top_n: int = DEFAULT_TOP_N,
                         min_chars: int = DEFAULT_MIN_CHARS) -> str:
    """把长文本压成核心观点句；短文本或压缩后反而更长时原样返回。"""
    raw = (text or "").strip()
    if len(raw) <= min_chars:
        return raw
    sentences = extract_core_opinion_sentences(raw, top_n=top_n)
    if not sentences:
        return raw
    simplified = "\n".join(sentences).strip()
    if not simplified or len(simplified) >= len(raw):
        return raw
    logger.debug("核心观点句压缩: %d → %d 字（%d 句）",
                 len(raw), len(simplified), len(sentences))
    return simplified


def simplify_texts_for_scoring(texts: Sequence[str],
                               top_n: int = DEFAULT_TOP_N,
                               min_chars: int = DEFAULT_MIN_CHARS) -> List[str]:
    """批量版本：供决策模型打分前统一简化候选文本。"""
    return [simplify_for_scoring(t, top_n=top_n, min_chars=min_chars)
            for t in texts]
