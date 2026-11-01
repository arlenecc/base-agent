# -*- coding: utf-8 -*-
"""RAG 全流程召回基准（纳入测试套件，作为回归断言）。

通过真实 ``VectorStore.search_with_rerank`` 管线验证检索各功能正常、召回准确且不遗漏：

  1) 标准路径：每篇文档都能被其「主题 + 唯一标记」查询召回（Recall@1/@5 = 100%）。
  2) 二值粗筛路径（``rag_binary_prefilter_enabled=True``）：召回必须与标准路径一致
     —— 该优化不得引入任何遗漏（核心诉求「不遗漏真正需要的切片」）。
  3) 定向检索（``source LIKE`` 过滤）：仅返回指定来源，不跨 source 泄漏。
  4) BM25 关键词检索：唯一标记词可被关键词通道命中原文。

使用**零中心嵌入**（语义聚类 + 每文档唯一标记），使余弦相似度与汉明距离语义均成立，
从而同时真实检验标准路径与二值粗筛路径的召回。``VectorStore`` 默认不含 reranker
实例 → ``search_with_rerank`` 走 RRF 兜底排序，召回判定确定性、可复现。
"""
import hashlib

import numpy as np
import pytest

from agent.rag.vector_store import VectorStore, set_binary_prefilter


K = 20           # 主题(聚类)数
DOCS_PER_K = 5   # 每主题文档数

_rng = np.random.default_rng(11)
CENTERS = _rng.normal(size=(K, 128))
CENTERS /= np.linalg.norm(CENTERS, axis=1, keepdims=True)
TOPIC = [f"主题{i:02d}" for i in range(K)]


class SignedEmbedding:
    """零中心嵌入：主题决定聚类中心，唯一标记词叠加确定性偏移 → 同文档查询近 1。

    全正分量的假嵌入会让二值量化退化（所有位恒为 1），故此处用零中心嵌入，
    使汉明距离与余弦相似度都具备真实语义，才能同时验证两段式粗筛的召回。
    """

    def __call__(self, input):
        out = []
        for text in input:
            c = 0
            for i, w in enumerate(TOPIC):
                if w in text:
                    c = i
                    break
            uw = ""
            if "UW_" in text:
                uw = text[text.index("UW_"):].split()[0]
            v = CENTERS[c].copy()
            if uw:
                h = hashlib.md5(uw.encode()).digest()
                off = (np.frombuffer(h, "u1").astype(np.float32) / 255.0 - 0.5)
                off = np.tile(off, 128 // 16)  # 16 → 128 维
                v = v + 0.18 * off
            v /= np.linalg.norm(v)
            out.append(v.tolist())
        return out

    def embed_query(self, t):
        return self.__call__([t])[0]

    def name(self):
        return "signed-fake"


def _build_corpus():
    """构造多主题语料：每主题若干文档，每文档带唯一标记（用于精确召回验证）。"""
    docs, facts = [], []  # facts: (query, expected_source)
    for c in range(K):
        for d in range(DOCS_PER_K):
            did = c * DOCS_PER_K + d
            uw = f"UW_{did:04d}"
            src = f"/kb/主题{c:02d}_文档{d}.md"
            text = (f"{TOPIC[c]} 这是关于本主题的说明性内容段落，"
                    f"用于检索验证的唯一标记 {uw} 出现在此处。其余内容描述"
                    f"主题{TOPIC[c]} 的背景知识与要点。")
            docs.append({"source": src, "chunk_index": 0, "text": text})
            facts.append((f"{TOPIC[c]} {uw}", src))
    return docs, facts


@pytest.fixture
def recall_store(tmp_path):
    """零中心嵌入 + 已入库的多主题语料，返回 (store, facts)。"""
    ef = SignedEmbedding()
    set_binary_prefilter(False, 200)  # 默认关闭粗筛，由用例显式开启
    store = VectorStore(persist_dir=str(tmp_path), embedding_function=ef)
    docs, facts = _build_corpus()
    store.add(docs)
    return store, facts


def _recall(store, facts, top_k=5):
    """逐条事实查询，返回 (Recall@1, Recall@5) 比例。"""
    at1 = at5 = 0
    for q, exp_src in facts:
        res = store.search_with_rerank(q, top_k=top_k)
        got = [r["source"] for r in res]
        if got and got[0] == exp_src:
            at1 += 1
        if exp_src in got:
            at5 += 1
    n = len(facts)
    return at1 / n, at5 / n


def test_standard_recall_no_omission(recall_store):
    """标准路径：每篇文档都能被其查询召回（Recall@1/@5 = 100%，不遗漏）。"""
    store, facts = recall_store
    set_binary_prefilter(False, 200)
    r1, r5 = _recall(store, facts)
    assert r1 == 1.0, f"标准路径 Recall@1={r1*100:.1f}%，存在遗漏"
    assert r5 == 1.0, f"标准路径 Recall@5={r5*100:.1f}%，存在遗漏"


def test_binary_prefilter_recall_no_omission(recall_store):
    """二值粗筛路径召回必须与标准路径一致（优化不得引入遗漏）。"""
    store, facts = recall_store
    set_binary_prefilter(True, 200)
    r1, r5 = _recall(store, facts)
    set_binary_prefilter(False, 200)
    assert r1 == 1.0, f"二值粗筛 Recall@1={r1*100:.1f}%，引入遗漏"
    assert r5 == 1.0, f"二值粗筛 Recall@5={r5*100:.1f}%，引入遗漏"


def test_targeted_retrieval_filters_source(recall_store):
    """定向检索（source LIKE）仅返回指定来源，不跨 source 泄漏。"""
    store, facts = recall_store
    set_binary_prefilter(False, 200)
    target_src = facts[0][1]
    q0 = facts[0][0]
    res = store.search_with_rerank(
        q0, top_k=10, where=f"source LIKE '%{target_src}%'")
    assert res, "定向检索应返回结果"
    assert all(r["source"] == target_src for r in res), "定向检索泄漏到其它来源"


def test_bm25_keyword_hit(recall_store):
    """BM25 关键词通道：唯一标记词查询能命中原文。"""
    store, facts = recall_store
    set_binary_prefilter(False, 200)
    uw = "UW_0000"
    res = store.search_with_rerank(uw, top_k=5)
    assert any(uw in (r.get("text") or "") for r in res), "BM25 关键词未命中原文"
