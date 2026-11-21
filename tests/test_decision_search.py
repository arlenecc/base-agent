"""决策判定模式：字符级 BM25 + 核心句 + 完整检索流程测试。"""
import math

import pytest

from agent.rag.decision_search import (
    CharBM25,
    _char_ngrams,
    _normalize_scores,
    decision_search,
    extract_core_sentences,
)


# ---------------------------------------------------------------------------
# 字符级 N-gram
# ---------------------------------------------------------------------------

def test_char_ngrams_cjk_sliding_window():
    grams = _char_ngrams("知识库检索")
    assert "知识库" in grams and "识库检" in grams and "库检索" in grams
    # 连续 4 个 CJK 字 → 2 个 3-gram，不会有 4 字 gram
    assert all(len(g) == 3 for g in grams)


def test_char_ngrams_short_cjk_kept_whole():
    # 不足 n 的 CJK 串保留整串，短查询不至于无特征
    assert _char_ngrams("检索") == ["检索"]


def test_char_ngrams_latin_words_kept():
    grams = _char_ngrams("BM25 search auth")
    assert "bm25" in grams and "search" in grams and "auth" in grams


def test_char_ngrams_mixed():
    grams = _char_ngrams("OAuth2认证流程")
    assert "oauth2" in grams and "认证流" in grams


def test_char_ngrams_empty():
    assert _char_ngrams("") == []
    assert _char_ngrams(None) == []


# ---------------------------------------------------------------------------
# 字符级 BM25 数值正确性
# ---------------------------------------------------------------------------

def test_char_bm25_scores_matching_sentence_higher():
    sents = ["向量检索使用embedding模型计算相似度。",
             "BM25是经典的关键词检索算法。",
             "今天天气很好，适合出门散步。"]
    scorer = CharBM25(sents)
    q = _char_ngrams("关键词检索算法")
    scores = [scorer.score(q, i) for i in range(3)]
    assert scores[1] > scores[0] > scores[2] or scores[1] == max(scores)
    assert scores[1] > 0.0


def test_char_bm25_idf_smoothed_nonnegative():
    # df = N（gram 出现在所有句子）时 Robertson 平滑保证非负
    sents = ["重复重复重复", "重复重复重复", "重复重复重复"]
    scorer = CharBM25(sents)
    assert scorer.idf("重复重") >= 0.0


def test_char_bm25_single_sentence_coverage_fallback():
    # 单句语料：BM25 分可能为 0（IDF=0），coverage 应有区分度
    scorer = CharBM25(["知识库支持向量检索。"])
    q = _char_ngrams("向量检索")
    assert scorer.coverage(q, 0) == 1.0


def test_normalize_scores():
    assert _normalize_scores([1.0, 2.0, 4.0]) == [0.25, 0.5, 1.0]
    assert _normalize_scores([0.0, 0.0]) == [0.0, 0.0]
    assert _normalize_scores([]) == []


# ---------------------------------------------------------------------------
# 核心句选取
# ---------------------------------------------------------------------------

_CHUNK = (
    "知识库系统支持多种检索方式。今天天气不错，适合出门散步放松心情。"
    "向量检索依赖embedding模型。晚饭吃了火锅，非常美味。"
    "BM25关键词检索对专名和书名效果好。周末计划去爬山。"
)


def test_extract_core_sentences_keeps_original_order():
    core, detail = extract_core_sentences(_CHUNK, "向量检索 embedding", top_n=3)
    assert core, "应选出核心句"
    # 按原文出现顺序：向量检索句必须在天气句之后（若都被选中）
    positions = [_CHUNK.index(seg.split("\n")[0]) for seg in core.split("\n")]
    assert positions == sorted(positions), "核心句必须按原文顺序排列"
    assert len(core.split("\n")) == 3


def test_extract_core_sentences_picks_relevant_sentences():
    core, _ = extract_core_sentences(_CHUNK, "BM25 关键词检索", top_n=3)
    assert "BM25" in core, "与查询最相关的句子应被选中"


def test_extract_core_sentences_short_chunk_untouched():
    short = "只有一个句子。"
    core, detail = extract_core_sentences(short, "句子", top_n=3)
    assert core == short and detail == []


def test_extract_core_sentences_empty_chunk():
    core, detail = extract_core_sentences("", "查询", top_n=3)
    assert core == "" and detail == []


# ---------------------------------------------------------------------------
# decision_search 完整流程（Fake 模型 + Fake store 注入）
# ---------------------------------------------------------------------------

class FakeDecisionModel:
    """score 型 Fake：切片文本里含『相关词』给 0.9，否则 0.1。"""

    backend = "fake"

    def score(self, query, texts):
        return [0.9 if "相关词" in t else 0.1 for t in texts]


class FakeStore:
    """VectorStore.search_fts 的最小 Fake。"""

    def __init__(self, chunks):
        self._chunks = chunks
        self.calls = []

    def count(self):
        return len(self._chunks)

    def search_fts(self, query, top_k=5, where=None):
        self.calls.append({"query": query, "top_k": top_k, "where": where})
        return list(self._chunks[:top_k])


class _Cfg:
    def __init__(self, coarse_k=50, use_core=True, core_top=3, top_n=5):
        self.coarse_k = coarse_k
        self.use_core_sentences = use_core
        self.core_top = core_top
        self.top_n = top_n


def _chunks():
    return [
        {"text": "这是相关词切片。包含相关词的正文内容。补一句无关 filler。",
         "source": "/docs/a.md", "chunk_index": 0, "fts_score": 12.0},
        {"text": "完全无关的切片。天气很好，出去散步。吃饭睡觉打豆豆。",
         "source": "/docs/b.md", "chunk_index": 1, "fts_score": 8.0},
        {"text": "又一个相关词所在切片。这里有相关词出现两次：相关词。",
         "source": "/docs/c.md", "chunk_index": 2, "fts_score": 5.0},
    ]


def test_decision_search_full_flow_with_model():
    store = FakeStore(_chunks())
    results = decision_search(store, "相关词 检索", _Cfg(), FakeDecisionModel())
    assert store.calls and store.calls[0]["top_k"] == 50
    assert len(results) == 3
    # 判定分排序：相关切片排前
    assert results[0]["score"] >= results[-1]["score"]
    assert results[0]["decision_mode"] is True
    assert "相关词" in results[0]["core_sentences"]
    # 两个相关切片分数为 0.9
    top_scores = sorted([r["score"] for r in results], reverse=True)
    assert top_scores[0] == 0.9 and top_scores[2] == 0.1


def test_decision_search_top_n_limits_results():
    store = FakeStore(_chunks())
    results = decision_search(store, "相关词", _Cfg(top_n=2), FakeDecisionModel())
    assert len(results) == 2


def test_decision_search_coarse_k_limits_candidates():
    store = FakeStore(_chunks())
    decision_search(store, "相关词", _Cfg(coarse_k=2), FakeDecisionModel())
    assert store.calls[0]["top_k"] == 2


def test_decision_search_without_core_sentences_judges_whole_chunk():
    store = FakeStore(_chunks())
    results = decision_search(store, "相关词", _Cfg(use_core=False),
                              FakeDecisionModel())
    # 整片判定：含相关词的片照样拿高分，但 core_sentences 为空
    assert all(r["core_sentences"] == "" for r in results)
    assert results[0]["score"] >= 0.9


def test_decision_search_empty_candidates():
    store = FakeStore([])
    results = decision_search(store, "相关词", _Cfg(), FakeDecisionModel())
    assert results == []


def test_decision_search_passes_where_to_store():
    """定向检索（source 过滤）：where SQL 必须透传给 search_fts。"""
    store = FakeStore(_chunks())
    decision_search(store, "相关词", _Cfg(), FakeDecisionModel(),
                    where="source LIKE '%a.md%' ESCAPE '\\'")
    assert store.calls[0]["where"] == "source LIKE '%a.md%' ESCAPE '\\'"


def test_engine_decision_mode_source_filter(monkeypatch):
    """引擎决策分支：source 参数生成与向量路径同款的 LIKE 过滤。"""
    eng = _bare_engine()
    store = FakeStore(_chunks())
    eng._get_store = lambda: store
    monkeypatch.setattr(
        "agent.rag.decision_model.get_decision_model",
        lambda backend, path: FakeDecisionModel())
    eng.search("相关词", top_k=5, source="/docs/a.md")
    where = store.calls[0]["where"]
    assert where is not None and "source LIKE" in where and "a.md" in where
    # 无 source 时不过滤
    eng.search("相关词", top_k=5)
    assert store.calls[1]["where"] is None


def test_decision_search_fallback_when_model_none():
    """模型不可用：回退字符 BM25 句分聚合排序，不崩、有结果。"""
    store = FakeStore(_chunks())
    results = decision_search(store, "相关词", _Cfg(), None)
    assert len(results) == 3
    assert all(0.0 <= r["score"] <= 1.0 for r in results)
    # 含相关词的切片应排更前
    related = [r for r in results if "相关词" in r["text"]]
    unrelated = [r for r in results if "相关词" not in r["text"]]
    assert related and unrelated
    assert related[0]["score"] >= unrelated[0]["score"]


def test_decision_search_model_batch_failure_falls_back(monkeypatch):
    """模型 score 抛异常：捕获回退排序，不向上抛。"""
    store = FakeStore(_chunks())

    class BoomModel:
        backend = "boom"

        def score(self, query, texts):
            raise RuntimeError("模型炸了")

    results = decision_search(store, "相关词", _Cfg(), BoomModel())
    assert len(results) == 3


# ---------------------------------------------------------------------------
# 引擎分发（decision 分支 + reading 优先级）
# ---------------------------------------------------------------------------

def _bare_engine(rag_mode="decision", reading_mode=False):
    from agent.rag.engine import RAGEngine
    eng = object.__new__(RAGEngine)
    eng._rag_mode = rag_mode
    eng._reading_mode = reading_mode
    eng._decision_backend = "mlx-vlm"
    eng._decision_model_path = ""
    eng._decision_coarse_k = 50
    eng._decision_use_core = True
    eng._decision_core_top = 3
    eng._decision_top_n = 5
    # 多分区相关属性（object.__new__ 绕过 __init__，需手工置空 = 单库模式）
    eng._partitions = []
    eng._stores = {}
    eng._active_partition = None
    # 知识图谱策略（search 入口会读取；naive = 不走向量图谱）
    eng._kg_strategy = "naive"
    return eng


def test_engine_search_dispatches_to_decision_mode(monkeypatch):
    eng = _bare_engine()
    store = FakeStore(_chunks())
    eng._get_store = lambda: store
    # 注入 fake 模型工厂：不加载真实 MLX 权重
    import agent.rag.engine as engine_mod
    monkeypatch.setattr(
        "agent.rag.decision_model.get_decision_model",
        lambda backend, path: FakeDecisionModel())
    results = eng.search("相关词 检索", top_k=5)
    assert results and results[0]["decision_mode"] is True


def test_engine_reading_mode_takes_priority_over_decision(monkeypatch):
    eng = _bare_engine(reading_mode=True)

    def _no_decision_store():
        raise AssertionError("reading 模式优先，不应触碰普通向量库")

    eng._get_store = _no_decision_store

    class EmptyReadingStore:
        def stats(self):
            return {"documents": 0}  # ColBERT 库为空 → reading 提前返回

    eng._get_reading_store = lambda force=False: EmptyReadingStore()
    # reading 分支先行：不触碰普通向量库（否则 _get_store 直接 AssertionError）
    assert eng.search("相关词") == []


def test_engine_decision_params_clamped():
    eng = _bare_engine()
    eng._decision_coarse_k = 33
    eng._decision_core_top = 7
    eng._decision_top_n = 99
    # __init__ 里的夹紧逻辑单测不了 object.__new__，改为直接验证模块级行为：
    # 构造真实引擎（tmp 工作区）并验证非法参数被夹紧。
    import tempfile, os
    from agent.rag.engine import RAGEngine
    with tempfile.TemporaryDirectory() as td:
        e = RAGEngine(
            workspace=td, embedding_model="",
            rag_mode="decision", rag_decision_coarse_k=33,
            rag_decision_core_top=7, rag_decision_top_n=99,
            rag_decision_model_backend="bad-backend",
        )
        assert e._decision_coarse_k == 50
        assert e._decision_core_top == 3
        assert e._decision_top_n == 5
        assert e._decision_backend == "bad-backend"  # 后端在加载时才校验
        e.close()


def test_engine_rag_mode_invalid_falls_back_vector():
    import tempfile
    from agent.rag.engine import RAGEngine
    with tempfile.TemporaryDirectory() as td:
        e = RAGEngine(workspace=td, embedding_model="", rag_mode="nonsense")
        assert e._rag_mode == "vector"
        e.close()


def test_engine_pipeline_signature_unchanged_in_decision_mode():
    """决策模式入库管线与 normal 一致，签名沿用不触发重建。"""
    import tempfile
    from agent.rag.engine import RAGEngine
    with tempfile.TemporaryDirectory() as td:
        e = RAGEngine(workspace=td, embedding_model="", rag_mode="decision")
        assert e._pipeline_signature() == "normal"
        e2 = RAGEngine(workspace=td, embedding_model="")
        assert e2._pipeline_signature() == e._pipeline_signature()
        e.close()
        e2.close()
