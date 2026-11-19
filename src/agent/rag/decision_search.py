"""决策判定模式的检索召回路径。

流程（不走向量检索与 reranker）：

1. **BM25 粗筛**：普通向量库切片表的全文索引（LanceDB FTS，jieba token 级
   BM25）召回 top ``coarse_k``（20/50/100）个原文切片；
2. **核心句选取**：字符级 BM25（3 字滑窗 N-gram、K1=1.6、b=0.4，自实现——
   LanceDB 0.40.0 的 FTS 不支持自定义 k1/b，且句级小语料也不适合建索引）
   计算问句与切片内每个句子的分数，取最高 3/5 句并按原文出现顺序拼接，
   代表该切片参与判定（关闭「使用核心句判定」时送整片）；
3. **决策判定**：本地判定模型逐片打分（0~1），按分数排序取 TopN 相关切片。

模型不可用 / 单片判定全失败时回退：按字符级 BM25 句分聚合排序返回粗筛
topN，并告警——检索不崩、只是精度降级。
"""
from __future__ import annotations

import logging
import math
import os
import re
import time
from collections import Counter
from typing import Dict, List, Sequence, Tuple

from .doc_chunker import _split_sentences

logger = logging.getLogger(__name__)

# 字符级 BM25 参数（需求指定）：3 字滑窗、K1=1.6、b=0.4。
CHAR_NGRAM_SIZE = 3
BM25_K1 = 1.6
BM25_B = 0.4

# 核心句最少保留字数：纯标点/空白句没有判定价值。
_MIN_SENT_CHARS = 2

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def _char_ngrams(text: str, n: int = CHAR_NGRAM_SIZE) -> List[str]:
    """字符级 N-gram：CJK（含日文假名/韩文）连续串做 n 字滑窗，拉丁文保词。"""
    grams: List[str] = []
    buf: List[str] = []
    for ch in text or "":
        if _CJK_RE.match(ch):
            if buf:
                grams.extend(_latin_grams(buf))
                buf = []
            grams.append(ch)
        else:
            buf.append(ch)
    if buf:
        grams.extend(_latin_grams(buf))
    return _finalize(grams, n)


def _latin_grams(chars: List[str]) -> List[str]:
    """把拉丁/数字字符序列按词切出（连续字母数字合并，标点丢弃）。"""
    words: List[str] = []
    cur = []
    for ch in chars:
        if ch.isalnum():
            cur.append(ch)
        elif cur:
            words.append("".join(cur).lower())
            cur = []
    if cur:
        words.append("".join(cur).lower())
    return words


def _finalize(tokens: List[str], n: int) -> List[str]:
    """对 CJK 单字流做 n-gram 滑窗，拉丁词原样保留。"""
    out: List[str] = []
    run: List[str] = []
    for tok in tokens:
        if len(tok) == 1 and _CJK_RE.match(tok):
            run.append(tok)
        else:
            if len(run) >= n:
                out.extend("".join(run[i:i + n])
                           for i in range(len(run) - n + 1))
            elif run:
                out.append("".join(run))
            run = []
            out.append(tok)
    if len(run) >= n:
        out.extend("".join(run[i:i + n]) for i in range(len(run) - n + 1))
    elif run:
        out.append("".join(run))
    return out


class CharBM25:
    """字符级 BM25（Okapi）：语料 = 候选切片的全部分句。

    IDF 用 Robertson-Sparck Jones 平滑（BM25 论文推荐）：
    ``log((N - df + 0.5) / (df + 0.5) + 1)``——df=N 时非负，小语料稳定。
    句数 < 2 时退化为纯查询覆盖率打分，避免单文档 IDF 全为 0 的抖动。
    """

    def __init__(self, sentences: Sequence[str],
                 k1: float = BM25_K1, b: float = BM25_B):
        self.k1 = float(k1)
        self.b = float(b)
        self._doc_counts: List[Counter] = [Counter(_char_ngrams(s))
                                           for s in sentences]
        self._doc_lens = [sum(c.values()) for c in self._doc_counts]
        self._n = len(sentences)
        self._avgdl = (sum(self._doc_lens) / self._n) if self._n else 0.0
        self._df: Counter = Counter()
        for c in self._doc_counts:
            self._df.update(c.keys())

    def idf(self, gram: str) -> float:
        df = self._df.get(gram, 0)
        return math.log((self._n - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query_grams: Sequence[str], index: int) -> float:
        """第 index 个句子对 query_grams 的 BM25 分。"""
        counts = self._doc_counts[index]
        if not counts:
            return 0.0
        dl = self._doc_lens[index]
        norm = self.k1 * (1.0 - self.b + self.b * dl / self._avgdl)
        total = 0.0
        for gram in set(query_grams):
            tf = counts.get(gram, 0)
            if tf <= 0:
                continue
            total += self.idf(gram) * (
                tf * (self.k1 + 1.0)) / (tf + norm)
        return total

    def coverage(self, query_grams: Sequence[str], index: int) -> float:
        """小语料回退分：query gram 在句中的命中占比。"""
        counts = self._doc_counts[index]
        if not query_grams:
            return 0.0
        hit = sum(1 for g in set(query_grams) if counts.get(g, 0) > 0)
        return hit / len(set(query_grams))


def extract_core_sentences(chunk_text: str, query: str, top_n: int = 3,
                           ) -> Tuple[str, List[Dict[str, float]]]:
    """字符级 BM25 选核心句：返回 (按原文顺序拼接的核心句, 句分明细)。

    短切片（句数 <= top_n）不裁剪，直接返回整片（保留原顺序）。
    """
    sentences = [u.text.strip() for u in _split_sentences(chunk_text or "")]
    sentences = [s for s in sentences if len(s) >= _MIN_SENT_CHARS]
    if not sentences:
        return (chunk_text or "").strip(), []
    query_grams = _char_ngrams(query or "")
    if len(sentences) <= top_n:
        return "\n".join(sentences), []
    scorer = CharBM25(sentences)
    scores: List[float] = []
    for i in range(len(sentences)):
        s = scorer.score(query_grams, i)
        if s <= 0.0:
            # 小语料 IDF 可能全 0（df=N）：回退覆盖率，保底区分度。
            s = 0.01 * scorer.coverage(query_grams, i)
        scores.append(s)
    order = sorted(range(len(sentences)), key=lambda i: scores[i],
                   reverse=True)[:top_n]
    # 按原文出现顺序重排（需求：核心句按在切片内出现的顺序排列）。
    order.sort()
    detail = [{"index": i, "score": scores[i]} for i in order]
    return "\n".join(sentences[i] for i in order), detail


def decision_search(store, query: str, cfg, model,
                    where: str = None) -> List[Dict[str, object]]:
    """决策判定检索主流程。

    ``store`` 为普通向量库（VectorStore，用 search_fts 粗筛）；``cfg`` 是
    引擎侧已夹紧的决策参数对象（dot-access：coarse_k / use_core_sentences /
    core_top / top_n）；``model`` 为 DecisionModel 或 None（None/不可用时
    回退句分聚合排序）；``where`` 为可选 SQL 过滤（定向检索 source LIKE）。
    """
    t0 = time.monotonic()
    coarse_k = int(getattr(cfg, "coarse_k", 50))
    use_core = bool(getattr(cfg, "use_core_sentences", True))
    core_top = int(getattr(cfg, "core_top", 3))
    top_n = int(getattr(cfg, "top_n", 5))

    # ---- 阶段 1：BM25 粗筛（token 级 FTS）----
    candidates = store.search_fts(query, top_k=coarse_k, where=where) or []
    t_coarse = time.monotonic() - t0
    if not candidates:
        logger.info("  └─ 决策判定模式: BM25 粗筛无结果 (耗时 %.3fs)", t_coarse)
        return []

    # ---- 阶段 2：核心句选取（字符级 BM25）----
    t1 = time.monotonic()
    judged_texts: List[str] = []
    core_flags: List[bool] = []
    for cand in candidates:
        chunk_text = cand.get("text") or ""
        if use_core:
            core_text, _detail = extract_core_sentences(
                chunk_text, query, top_n=core_top)
            judged_texts.append(core_text)
            core_flags.append(bool(core_text))
        else:
            judged_texts.append(chunk_text.strip())
            core_flags.append(False)
    t_core = time.monotonic() - t1

    # ---- 阶段 3：决策模型逐片判定 ----
    t2 = time.monotonic()
    used_model = model is not None
    decision_scores: List[float] = []
    if model is not None:
        try:
            decision_scores = model.score(query, judged_texts)
        except Exception as e:
            logger.warning("  └─ ⚠ 决策模型批量判定失败，回退字符 BM25 句分排序: %s", e)
            model = None
            used_model = False
    if model is None:
        # 回退排序：每片取其核心句的字符级 BM25 最大句分（无核心句用 0）。
        decision_scores = []
        query_grams = _char_ngrams(query or "")
        for cand in candidates:
            sentences = [u.text.strip()
                         for u in _split_sentences(cand.get("text") or "")]
            sentences = [s for s in sentences if len(s) >= _MIN_SENT_CHARS]
            best = 0.0
            if sentences:
                scorer = CharBM25(sentences)
                for i in range(len(sentences)):
                    best = max(best, scorer.score(query_grams, i))
            decision_scores.append(best)
        decision_scores = _normalize_scores(decision_scores)
    t_decision = time.monotonic() - t2

    # ---- 阶段 4：排序取 TopN ----
    t3 = time.monotonic()
    ranked = sorted(range(len(candidates)),
                    key=lambda i: decision_scores[i], reverse=True)[:top_n]
    results: List[Dict[str, object]] = []
    for rank, i in enumerate(ranked, 1):
        cand = candidates[i]
        results.append({
            "text": cand.get("text") or "",
            "source": cand.get("source") or "",
            "chunk_index": cand.get("chunk_index", 0),
            "score": round(float(decision_scores[i]), 4),
            "fts_score": cand.get("fts_score", 0.0),
            "core_sentences": judged_texts[i] if core_flags[i] else "",
            "decision_mode": True,
        })
        src_name = os.path.basename(cand.get("source") or "")
        preview = (judged_texts[i] or cand.get("text") or "")[:60].replace("\n", " ")
        logger.info("      [%d] %s (判定分: %.4f) | %s...", rank, src_name,
                    decision_scores[i], preview)
    t_rank = time.monotonic() - t3
    t_total = time.monotonic() - t0
    logger.info(
        "    └─ ⏱ 决策判定检索: 粗筛 %d 片 %.3fs | 核心句 %.3fs | 判定 %.3fs%s "
        "| 排序 %.3fs | 总计 %.3fs",
        len(candidates), t_coarse, t_core, t_decision,
        "（模型）" if used_model else "（回退 BM25）", t_rank, t_total)
    return results


def _normalize_scores(scores: Sequence[float]) -> List[float]:
    """回退路径的分数归一（原始 BM25 分无上界 → 0~1，对齐模型分口径）。"""
    if not scores:
        return []
    mx = max(scores)
    if mx <= 0:
        return [0.0] * len(scores)
    return [s / mx for s in scores]
