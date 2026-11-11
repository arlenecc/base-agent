"""知识图谱模块（rag/kg/）单元测试。

覆盖 spec 核心行为：
* 存储层：实体/关系聚合（无向 pair 合并、weight=证据 chunk 计数）、
  purge 按证据重算 weight、向量虚拟表；
* 抽取输出解析（entity/relation 行、<|#|> 分隔、无向去重）与关键词 JSON；
* 内存图 + JSON 快照（Codable 对应物）往返；
* embedding 文本构造（name\ndesc / keywords\tsrc\ntgt\ndesc 正反序）与
  int8 余弦检索（阈值 0.2）；
* 检索器：三路召回、交错去重、token 预算、local/global/mix 策略。

不依赖 LanceDB / 真模型：embedding 用可控的假 ef。
"""
import json
import os
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.rag.kg.storage import KGStore, normalize_name, pair_key
from agent.rag.kg.graph import KnowledgeGraph
from agent.rag.kg.embedding import KGEmbedder
from agent.rag.kg.extractor import KGExtractor, parse_extraction_output
from agent.rag.kg.builder import KGBuilder, kg_dir_for, snapshot_path_for
from agent.rag.kg.retriever import KGRetriever


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def kg_store(tmp_path):
    store = KGStore(str(tmp_path / "kg"))
    yield store
    store.close()


@pytest.fixture()
def rag_dir(tmp_path):
    d = tmp_path / "rag"
    d.mkdir()
    return str(d)


class FakeEF:
    """可控假 embedding：关键词 → one-hot 单位向量（float32 list）。

    ``name()`` 返回非 Matryoshka 模型名，避免触发 128 维截断逻辑。
    """

    DIMS = 8

    def name(self):
        return "mock-embedding"

    def embed_task(self, texts, task):
        out = []
        for t in texts:
            v = [0.0] * self.DIMS
            low = t.lower()
            keys = ["苹果", "乔布斯", " iphone", "公司", "创始", "科技"]
            for i, k in enumerate(keys):
                if k.strip() and k.strip() in low:
                    v[i % self.DIMS] = 1.0
            if not any(v):
                v[0] = 0.1  # 无关文本 → 与所有单位向量低相似
            out.append(v)
        return out


class CallableEF(FakeEF):
    """可调用版 FakeEF：VectorStore.add 直接 ``ef(texts)``，需 ``__call__``。"""

    def __call__(self, texts):
        return self.embed_task(list(texts), "doc")


class FakeLLM:
    """假辅助模型：按 prompt 特征返回固定输出，记录调用。"""

    def __init__(self, extract_output="", keywords='{"high_level_keywords": ["科技公司"], "low_level_keywords": ["苹果", "乔布斯"]}', summary="归纳后的描述"):
        self.extract_output = extract_output
        self.keywords = keywords
        self.summary = summary
        self.calls = []

    def batch_chat(self, messages_list, *args, **kwargs):
        self.calls.append(messages_list)
        text = messages_list[0][-1]["content"] if messages_list else ""
        if "high_level_keywords" in text:
            return [self.keywords]
        if "Description List" in text:
            return [self.summary]
        return [self.extract_output]


def _int8(vec):
    """float → 与 vector_store 一致的 int8 表示（L2 归一化 ×127）。"""
    a = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(a))
    if n == 0:
        return np.zeros(len(a), dtype=np.int8)
    return np.round(a / n * 127).astype(np.int8)


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------

class TestKGStore:
    def test_entity_upsert_merges_by_name_key(self, kg_store):
        e1, c1 = kg_store.upsert_entity("Apple", "Organization", "一家科技公司", "c1")
        e2, c2 = kg_store.upsert_entity("apple", "Organization", "消费电子公司", "c2")
        assert c1 and not c2 and e1 == e2  # 大小写不敏感聚合
        row = kg_store.get_entity_by_key(normalize_name("APPLE"))
        assert row["id"] == e1
        assert set(row["source_ids"]) == {"c1", "c2"}
        assert row["type"] == "Organization"

    def test_relationship_undirected_merge(self, kg_store):
        r1, c1 = kg_store.upsert_relationship("苹果", "乔布斯", "创始人", "乔布斯创办了苹果", "c1", "a.md")
        r2, c2 = kg_store.upsert_relationship("乔布斯", "苹果", "联合创始,创业", "两人共同创办", "c1", "a.md")
        r3, _ = kg_store.upsert_relationship("苹果", "乔布斯", "创始人", "再次证据", "c2", "b.md")
        assert c1 and not c2 and r1 == r2 == r3
        pk = pair_key("苹果", "乔布斯")
        row = kg_store.get_relationship_by_pair(pk)
        assert row is not None
        # raw_weight：每多一条抽取记录 +1（3 条）；
        # 去重来源 chunk 数 = 2 → 图谱 weight = max(3, 2) = 3。
        assert row["raw_weight"] == 3
        assert set(row["source_ids"]) == {"c1", "c2"}
        assert "创始人" in row["keywords"] and "联合创始" in row["keywords"]
        assert set(row["file_paths"]) == {"a.md", "b.md"}

    def test_purge_recomputes_weight_from_evidence(self, kg_store):
        kg_store.upsert_relationship("A", "B", "相关", "d1", "doc1_0", "1.md")
        kg_store.upsert_relationship("A", "B", "相关", "d2", "doc1_1", "1.md")
        kg_store.upsert_relationship("A", "B", "相关", "d3", "doc2_0", "2.md")
        row = kg_store.get_relationship_by_pair(pair_key("A", "B"))
        assert row["raw_weight"] == 3
        stats = kg_store.purge_document(["doc1"])
        assert stats["relationships"] == 0  # 还有 doc2 证据，行保留
        row = kg_store.get_relationship_by_pair(pair_key("A", "B"))
        assert row["raw_weight"] == 1  # 按剩余证据重算，不保留旧标量 3
        assert row["source_ids"] == ["doc2_0"]
        # 全部证据删除 → 行删除
        stats = kg_store.purge_document(["doc2"])
        assert stats["relationships"] == 1
        assert kg_store.get_relationship_by_pair(pair_key("A", "B")) is None

    def test_purge_cleans_orphan_vectors(self, kg_store):
        eid, _ = kg_store.upsert_entity("X", "Other", "d", "c1")
        kg_store.put_vector(f"ent:{eid}", "entity", eid, "h", b"\x01\x02")
        kg_store.purge_document(["c1"])
        assert kg_store.get_vector(f"ent:{eid}") is None

    def test_vector_table_roundtrip(self, kg_store):
        kg_store.put_vector("ent:e1", "entity", "e1", "h1", b"\x01\x02\x03")
        assert kg_store.get_vector_hash("ent:e1") == "h1"
        assert kg_store.get_vector("ent:e1") == b"\x01\x02\x03"
        assert kg_store.iter_vectors("entity")[0]["vector_id"] == "ent:e1"


# ---------------------------------------------------------------------------
# extractor
# ---------------------------------------------------------------------------

RAW_EXTRACT = """entity<|#|>Apple<|#|>Organization<|#|>一家科技公司
entity<|#|>Steve Jobs<|#|>Person<|#|>苹果联合创始人
entity<|#|>Apple<|#|>Organization<|#|>重复实体应去重
relation<|#|>Apple<|#|>Steve Jobs<|#|>创始人,联合创始<|#|>乔布斯与沃兹尼亚克共同创办苹果
relation<|#|>Steve Jobs<|#|>Apple<|#|>创始人<|#|>无向重复应去重
<|COMPLETE|>
"""


class TestExtractorParsing:
    def test_parse_extraction_output(self):
        entities, relations = parse_extraction_output(RAW_EXTRACT)
        assert len(entities) == 3  # 同名实体不在此层去重（聚合在存储层）
        assert entities[0].name == "Apple"
        assert entities[0].type == "Organization"
        assert len(relations) == 1  # 无向去重
        assert relations[0].src == "Apple" and relations[0].dst == "Steve Jobs"
        assert "创始人" in relations[0].keywords

    def test_parse_thinking_and_garbage(self):
        raw = "<think>推理内容</think>\n上文无关\n" + RAW_EXTRACT
        entities, relations = parse_extraction_output(raw)
        assert len(entities) == 3 and len(relations) == 1

    def test_unknown_entity_type_maps_to_other(self):
        entities, _ = parse_extraction_output("entity<|#|>X<|#|>MagicType<|#|>desc")
        assert entities[0].type == "Other"

    def test_extract_keywords_parses_json(self):
        llm = FakeLLM()
        ex = KGExtractor(llm)
        kws = ex.extract_keywords("苹果公司的创始人是谁？")
        assert kws["high_level_keywords"] == ["科技公司"]
        assert "苹果" in kws["low_level_keywords"]

    def test_extract_keywords_fallback_on_bad_output(self):
        ex = KGExtractor(FakeLLM(keywords="不是JSON"))
        kws = ex.extract_keywords("查询原文")
        assert kws["low_level_keywords"] == ["查询原文"]

    def test_no_client_degrades(self):
        ex = KGExtractor(None)
        assert not ex.available
        kws = ex.extract_keywords("q")
        assert kws["low_level_keywords"] == ["q"]
        assert ex.extract_entities_relations("文本") == ([], [])


# ---------------------------------------------------------------------------
# graph + snapshot
# ---------------------------------------------------------------------------

class TestGraph:
    def _seed(self, kg_store):
        kg_store.upsert_entity("苹果", "Organization", "科技公司", "c1")
        kg_store.upsert_entity("乔布斯", "Person", "创始人", "c1")
        kg_store.upsert_entity("沃兹尼亚克", "Person", "联合创始人", "c2")
        kg_store.upsert_relationship("苹果", "乔布斯", "创始人", "乔布斯创办苹果", "c1")
        kg_store.upsert_relationship("苹果", "乔布斯", "创始人", "补充证据", "c2")
        kg_store.upsert_relationship("苹果", "沃兹尼亚克", "联合创始人", "沃兹共同创办", "c2")

    def test_rebuild_and_weight(self, kg_store):
        self._seed(kg_store)
        g = KnowledgeGraph()
        g.rebuild_from_store(kg_store)
        assert g.stats() == {"nodes": 3, "edges": 2}
        apple = g.get_node(normalize_name("苹果"))
        assert apple["degree"] == 2  # 两条边
        edge = g.get_edge_by_pair(pair_key("苹果", "乔布斯"))
        assert edge["weight"] == 2  # raw_weight=2, unique chunks=2 → max=2

    def test_neighbor_edges_sorted_by_weight(self, kg_store):
        self._seed(kg_store)
        g = KnowledgeGraph()
        g.rebuild_from_store(kg_store)
        edges = g.neighbor_edges(normalize_name("苹果"))
        assert len(edges) == 2
        assert all(e["weight"] <= edges[0]["weight"] for e in edges)

    def test_snapshot_roundtrip(self, kg_store, tmp_path):
        self._seed(kg_store)
        g = KnowledgeGraph()
        g.rebuild_from_store(kg_store)
        snap = str(tmp_path / "snap.json")
        g.save_snapshot(snap)
        g2 = KnowledgeGraph()
        assert g2.load_snapshot(snap)
        assert g2.stats() == g.stats()
        assert g2.get_node(normalize_name("乔布斯"))["type"] == "Person"
        # 损坏快照 → False（回退查库重建）
        with open(snap, "w") as f:
            f.write("not json")
        assert not KnowledgeGraph().load_snapshot(snap)


# ---------------------------------------------------------------------------
# embedding
# ---------------------------------------------------------------------------

class TestKGEmbedder:
    def test_entity_text_format(self):
        assert KGEmbedder.entity_text("苹果", "科技公司") == "苹果\n科技公司"

    def test_relationship_text_forward_and_reverse(self):
        fwd = KGEmbedder.relationship_text("苹果", "乔布斯", "创始人", "创办", reverse=False)
        rev = KGEmbedder.relationship_text("苹果", "乔布斯", "创始人", "创办", reverse=True)
        assert fwd == "创始人\t苹果\n乔布斯\n创办"
        assert rev == "创始人\t乔布斯\n苹果\n创办"
        assert fwd != rev  # 正反序两个向量 id

    def test_cosine_int8_identical_vectors(self):
        v = _int8([1.0, 2.0, 3.0])
        assert KGEmbedder.cosine_int8(v.tobytes(), v.tobytes()) == pytest.approx(1.0, abs=1e-3)

    def test_cosine_int8_orthogonal(self):
        a = _int8([1.0, 0.0])
        b = _int8([0.0, 1.0])
        assert KGEmbedder.cosine_int8(a.tobytes(), b.tobytes()) == pytest.approx(0.0, abs=1e-3)

    def test_search_vectors_threshold(self):
        emb = KGEmbedder(FakeEF())
        q = _int8([1.0, 0.0, 0.0]).tobytes()
        cands = [
            {"vector_id": "a", "target_id": "a", "embedding": _int8([1.0, 0.1, 0.0]).tobytes()},
            {"vector_id": "b", "target_id": "b", "embedding": _int8([0.0, 1.0, 0.0]).tobytes()},
        ]
        hits = emb.search_vectors(q, cands, top_k=5, min_similarity=0.2)
        assert [h["vector_id"] for h in hits] == ["a"]  # 余弦 <0.2 丢弃

    def test_int8_roundtrip_via_prepare(self):
        from agent.rag.vector_store import _prepare_vector_for_index
        vec = [0.3, -0.4, 0.5, 0.6]
        out = _prepare_vector_for_index(vec, False, 0)
        arr = np.asarray(out, dtype=np.int8)
        # 重归一化后量化：单位向量 ×127
        n = float(np.linalg.norm(np.asarray(vec, dtype=np.float32)))
        expect = np.round(np.asarray(vec, dtype=np.float32) / n * 127).astype(np.int8)
        assert np.array_equal(arr, expect)


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------

EXTRACT_1 = """entity<|#|>苹果<|#|>Organization<|#|>一家科技公司
entity<|#|>乔布斯<|#|>Person<|#|>苹果创始人
relation<|#|>苹果<|#|>乔布斯<|#|>创始人<|#|>乔布斯创办了苹果公司
<|COMPLETE|>"""

EXTRACT_2 = """entity<|#|>苹果<|#|>Organization<|#|>总部位于库比蒂诺
relation<|#|>苹果<|#|>乔布斯<|#|>创始人<|#|>1976年创办
<|COMPLETE|>"""


class FakeStore:
    """假 VectorStore：固定切片 + FakeEF，供 KGBuilder/KGRetriever 使用。"""

    def __init__(self, chunks):
        self._chunks = chunks
        self.embedding_function = FakeEF()

    def iter_all_chunks(self):
        return list(self._chunks)

    def get_chunks_by_ids(self, ids):
        return [c for c in self._chunks if c["id"] in set(ids)]


class TestBuilder:
    def test_build_pipeline(self, rag_dir, tmp_path):
        store = FakeStore([
            {"id": "c1_0", "text": "苹果公司由乔布斯创立。", "source": "a.md", "chunk_index": 0},
            {"id": "c2_0", "text": "苹果总部位于库比蒂诺。", "source": "b.md", "chunk_index": 0},
        ])
        llm = FakeLLM(extract_output=EXTRACT_1)
        ex = KGExtractor(llm)
        builder = KGBuilder(rag_dir, store, ex)
        try:
            stats = builder.build()
        finally:
            builder.close()
        assert stats["chunks"] == 2
        assert stats["errors"] == 0
        # 第二片用不同抽取输出
        # （FakeLLM 固定输出——这里两次抽取输出相同，验证聚合即可）
        kg = KGStore(kg_dir_for(rag_dir))
        try:
            counts = kg.counts()
            assert counts["entities"] >= 2
            assert counts["relationships"] == 1
            assert counts["vectors"] >= 1 + 2  # 实体向量 + 关系正反序两个
            ent = kg.get_entity_by_key(normalize_name("苹果"))
            assert set(ent["source_ids"]) == {"c1_0", "c2_0"}
            rel = kg.get_relationship_by_pair(pair_key("苹果", "乔布斯"))
            assert rel["raw_weight"] == 2
        finally:
            kg.close()
        # 快照存在且可加载
        g = KnowledgeGraph()
        assert g.load_snapshot(snapshot_path_for(rag_dir))
        assert g.stats()["nodes"] >= 2

    def test_build_requires_aux_llm(self, rag_dir):
        store = FakeStore([{"id": "c", "text": "t", "source": "s", "chunk_index": 0}])
        builder = KGBuilder(rag_dir, store, KGExtractor(None))
        try:
            with pytest.raises(RuntimeError):
                builder.build()
        finally:
            builder.close()

    def test_build_cancellation(self, rag_dir):
        store = FakeStore([{"id": f"c{i}", "text": "t", "source": "s", "chunk_index": 0}
                           for i in range(5)])
        cancel = threading.Event()
        cancel.set()
        builder = KGBuilder(rag_dir, store, KGExtractor(FakeLLM(extract_output=EXTRACT_1)),
                            cancel_event=cancel)
        try:
            with pytest.raises(KGCancellation):
                builder.build()
        finally:
            builder.close()


from agent.rag.kg.builder import KGCancellation  # noqa: E402


# ---------------------------------------------------------------------------
# retriever
# ---------------------------------------------------------------------------

class TestRetriever:
    def _prepare(self, rag_dir, strategy_ctx=None):
        """构建含 2 实体 + 1 关系的 KG，返回 (retriever, store)。"""
        store = FakeStore([
            {"id": "c1_0", "text": "苹果公司由乔布斯创立， iPhone 是其产品。",
             "source": "a.md", "chunk_index": 0},
        ])
        builder = KGBuilder(rag_dir, store, KGExtractor(FakeLLM(extract_output=EXTRACT_1)))
        try:
            builder.build()
        finally:
            builder.close()
        ex = KGExtractor(FakeLLM())  # 查询期：返回 hl/ll 关键词
        ret = KGRetriever(rag_dir, store, ex, estimate_tokens=lambda s: max(1, len(s) // 2))
        return ret, store

    def test_graph_ready(self, rag_dir):
        ret, _ = self._prepare(rag_dir)
        try:
            assert ret.graph_ready()
        finally:
            ret.close()
        # 未构建 → False
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            ret2 = KGRetriever(d, FakeStore([]), KGExtractor(None))
            try:
                assert not ret2.graph_ready()
            finally:
                ret2.close()

    def test_local_strategy_returns_entities_and_relations(self, rag_dir):
        ret, _ = self._prepare(rag_dir)
        try:
            # FakeLLM 的 ll 关键词（苹果/乔布斯）命中实体向量（name+desc 含苹果/乔布斯）
            results = ret.search("苹果公司的创始人是谁？", "local")
            kinds = {r["kind"] for r in results}
            assert "kg_entity" in kinds
            assert "kg_relation" in kinds  # local：实体一跳邻边
        finally:
            ret.close()

    def test_global_strategy(self, rag_dir):
        ret, _ = self._prepare(rag_dir)
        try:
            results = ret.search("科技公司之间的创始关系", "global")
            kinds = {r["kind"] for r in results}
            assert "kg_relation" in kinds
        finally:
            ret.close()

    def test_mix_includes_chunks_with_direct_first(self, rag_dir):
        ret, store = self._prepare(rag_dir)
        try:
            direct = [{"id": "d1", "text": "直召切片", "source": "x.md",
                       "chunk_index": 0, "score": 0.9}]
            results = ret.search("苹果 iPhone", "mix", chunk_recall=lambda q, k: direct)
            kinds = {r["kind"] for r in results}
            assert "kg_entity" in kinds and "kg_relation" in kinds and "chunk" in kinds
            # 查询直召 chunk 排在最前（交错去重、直召优先）
            assert results[0]["kind"] == "chunk"
        finally:
            ret.close()

    def test_naive_returns_empty(self, rag_dir):
        ret, _ = self._prepare(rag_dir)
        try:
            assert ret.search("q", "naive") == []
        finally:
            ret.close()

    def test_no_graph_falls_back(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            ret = KGRetriever(d, FakeStore([]), KGExtractor(None))
            try:
                assert ret.search("q", "local") == []
            finally:
                ret.close()

    def test_dedup_entities_by_name(self, rag_dir):
        ret, _ = self._prepare(rag_dir)
        try:
            e1 = {"name": "苹果", "name_key": "苹果", "type": "Organization",
                  "description": "d1", "source_ids": ["c1"], "similarity": 0.9}
            e2 = {"name": "苹果", "name_key": "苹果", "type": "Organization",
                  "description": "d2", "source_ids": ["c2"], "similarity": 0.5}
            out = ret._dedup_entities([e1, e2])
            assert len(out) == 1
            assert set(out[0]["source_ids"]) == {"c1", "c2"}
            assert out[0]["similarity"] == 0.9
        finally:
            ret.close()

    def test_dedup_relations_by_undirected_pair(self, rag_dir):
        ret, _ = self._prepare(rag_dir)
        try:
            r1 = {"src_entity": "苹果", "dst_entity": "乔布斯", "raw_weight": 2,
                  "similarity": 0.8}
            r2 = {"src_entity": "乔布斯", "dst_entity": "苹果", "raw_weight": 3,
                  "similarity": 0.5}
            out = ret._dedup_relations([r1, r2])
            assert len(out) == 1
            assert out[0]["raw_weight"] == 3  # weight 取最大
        finally:
            ret.close()

    def test_token_budget_truncation(self, rag_dir):
        ret, _ = self._prepare(rag_dir)
        try:
            ents = [{"name": f"实体{i}", "name_key": f"实体{i}", "type": "Other",
                     "description": "长" * 200, "source_ids": [],
                     "similarity": 1.0 - i * 0.01}
                    for i in range(100)]
            out = ret._truncate_entities(ents)
            total = sum(ret._estimate_tokens(ret._format_entity(e)) for e in out)
            assert total <= 6000
            assert len(out) < 100
        finally:
            ret.close()

    def test_backref_chunks_capped(self, rag_dir):
        ret, store = self._prepare(rag_dir)
        try:
            chunks = [{"id": f"c{i}", "text": "t", "source": "s", "chunk_index": i}
                      for i in range(20)]
            st = FakeStore(chunks)
            ret._store = st
            ents = [{"name": f"e{i}", "source_ids": [f"c{j}" for j in range(10)]}
                    for i in range(10)]
            got = ret._backref_chunks(None, ents, [])
            assert len(got) <= 10 * 5  # 每实体最多 5 片
        finally:
            ret.close()


# ---------------------------------------------------------------------------
# engine 集成（不依赖 LanceDB 之外的真模型）
# ---------------------------------------------------------------------------

class TestEngineKGIntegration:
    def test_strategy_fallback_and_status(self, tmp_path):
        from agent.rag.engine import RAGEngine
        eng = RAGEngine(workspace=str(tmp_path),
                        knowledge_base=str(tmp_path / "kb"),
                        embedding_function=FakeEF())
        try:
            assert eng.status()["kg_strategy"] == "naive"
            assert eng.status()["kg_ready"] is False
            eng.set_kg_strategy("local")
            # 图谱未构建 → 自动回退 naive；空库 → 无结果，不崩溃
            assert eng.search("苹果公司", top_k=5) == []
            assert eng.status()["kg_strategy"] == "local"
            assert eng.kg_ready() is False
        finally:
            eng.close()

    def test_aux_llm_proxy_semantics(self):
        from agent.tools.base import _AuxLLMProxy
        state = {"client": None}

        def getter():
            return state["client"]

        proxy = _AuxLLMProxy(getter)
        assert not proxy.available and not bool(proxy)
        assert proxy.batch_chat([[{"role": "user", "content": "hi"}]]) == [""]

        class _C:
            def batch_chat(self, msgs, **kw):
                return ["ok"]

        state["client"] = _C()
        assert proxy.available and bool(proxy)
        assert proxy.batch_chat([[{"role": "user", "content": "hi"}]]) == ["ok"]

    def test_extractor_respects_proxy_availability(self):
        # 未加载辅助模型时 available=False → 构建入口直接拒绝
        from agent.tools.base import _AuxLLMProxy
        ex = KGExtractor(_AuxLLMProxy(lambda: None))
        assert not ex.available


# ---------------------------------------------------------------------------
# 三种抽取方案（llm_raw / llm_pydantic / gliner）+ 工厂
# ---------------------------------------------------------------------------

from agent.rag.kg.extractor import (  # noqa: E402
    PydanticLLMExtractor, GLiNERExtractor, create_extractor,
)

PYDANTIC_EXTRACT_JSON = """```json
{"entities": [
   {"name": "苹果", "type": "Organization", "description": "一家科技公司"},
   {"name": "乔布斯", "type": "company", "description": "苹果创始人"}],
 "relations": [
   {"src": "苹果", "dst": "乔布斯", "keywords": "创始人", "description": "乔布斯创办苹果"}]}
```"""

PYDANTIC_BAD_JSON = "抱歉，我无法完成该任务。"


class TestPydanticExtractor:
    def test_json_output_parsed(self):
        ex = PydanticLLMExtractor(FakeLLM(extract_output=PYDANTIC_EXTRACT_JSON))
        ents, rels = ex.extract_entities_relations("苹果由乔布斯创立", "ctx")
        assert [e.name for e in ents] == ["苹果", "乔布斯"]
        assert ents[0].type == "Organization"
        assert len(rels) == 1
        assert rels[0].src == "苹果" and rels[0].dst == "乔布斯"

    def test_unknown_type_normalized(self):
        # pydantic validator：非法实体类型归一化为 Other
        ex = PydanticLLMExtractor(FakeLLM(extract_output=PYDANTIC_EXTRACT_JSON))
        ents, _ = ex.extract_entities_relations("t", "")
        assert ents[1].type == "Other"  # "company" 不在类型表内

    def test_invalid_json_returns_empty(self):
        ex = PydanticLLMExtractor(FakeLLM(extract_output=PYDANTIC_BAD_JSON))
        ents, rels = ex.extract_entities_relations("t", "")
        assert ents == [] and rels == []

    def test_missing_fields_rejected(self):
        # 缺 description 等可选字段可过；缺 name 的实体被拒绝
        bad = '{"entities": [{"type": "Person"}], "relations": []}'
        ex = PydanticLLMExtractor(FakeLLM(extract_output=bad))
        ents, rels = ex.extract_entities_relations("t", "")
        assert ents == [] and rels == []

    def test_available_semantics(self):
        ex = PydanticLLMExtractor(None)
        assert not ex.available  # 先取 available（内部记录原因），再读原因
        assert "辅助模型" in ex.unavailable_reason
        assert PydanticLLMExtractor(FakeLLM(extract_output="{}")).available


def _seed_graph(kg_store):
    """种入 3 实体 + 3 关系证据（苹果-乔布斯 2 条、苹果-沃兹 1 条）。"""
    kg_store.upsert_entity("苹果", "Organization", "科技公司", "c1")
    kg_store.upsert_entity("乔布斯", "Person", "创始人", "c1")
    kg_store.upsert_entity("沃兹尼亚克", "Person", "联合创始人", "c2")
    kg_store.upsert_relationship("苹果", "乔布斯", "创始人", "乔布斯创办苹果", "c1")
    kg_store.upsert_relationship("苹果", "乔布斯", "创始人", "补充证据", "c2")
    kg_store.upsert_relationship("苹果", "沃兹尼亚克", "联合创始人", "沃兹共同创办", "c2")


class TestGLiNERExtractor:
    def test_unavailable_without_package(self):
        # gliner 包未安装 → available=False，原因含安装指引（已装则通过）
        ex = GLiNERExtractor()
        try:
            import gliner  # noqa: F401
            assert ex.available
        except ImportError:
            assert not ex.available
            assert "pip install gliner" in ex.unavailable_reason

    def test_extract_without_model_raises(self):
        # 模型懒加载：抽取时才触发加载；未安装包 → RuntimeError（带指引）
        ex = GLiNERExtractor()
        try:
            import gliner  # noqa: F401
            pytest.skip("gliner 已安装，跳过未安装路径测试")
        except ImportError:
            with pytest.raises(RuntimeError, match="gliner"):
                ex.extract_entities_relations("苹果由乔布斯创立", "")

    def test_keywords_fallback_without_model(self):
        # 查询期 GLiNER 不可用 → 回退 ll=[query]（实体检索仍有输入）
        ex = GLiNERExtractor()
        try:
            import gliner  # noqa: F401
            pytest.skip("gliner 已安装，跳过未安装路径测试")
        except ImportError:
            kw = ex.extract_keywords("苹果公司的创始人")
            assert kw == {"high_level_keywords": [],
                          "low_level_keywords": ["苹果公司的创始人"]}

    def test_summarize_joins_without_llm(self):
        ex = GLiNERExtractor()
        merged = ex.summarize_descriptions("苹果", ["描述A", "描述B", "描述A"])
        assert "描述A" in merged and "描述B" in merged
        assert merged.count("描述A") == 1  # 去重拼接


class TestExtractorFactory:
    def test_scheme_mapping(self):
        assert create_extractor("llm_raw").scheme == "llm_raw"
        assert isinstance(create_extractor("llm_raw"), KGExtractor)
        assert create_extractor("llm_pydantic").scheme == "llm_pydantic"
        assert isinstance(create_extractor("llm_pydantic"), PydanticLLMExtractor)
        assert create_extractor("gliner").scheme == "gliner"
        assert isinstance(create_extractor("gliner"), GLiNERExtractor)

    def test_unknown_falls_back_to_llm_raw(self):
        assert create_extractor("bogus").scheme == "llm_raw"
        assert create_extractor("").scheme == "llm_raw"

    def test_llm_client_passthrough(self):
        llm = FakeLLM(extract_output=EXTRACT_1)
        ex = create_extractor("llm_raw", llm)
        ents, _ = ex.extract_entities_relations("t", "ctx")
        assert len(ents) == 2


class TestGraphNetworkx:
    def test_snapshot_v1_rejected(self, kg_store, tmp_path):
        # v1（自实现图时代）快照 → 拒载，回退查库重建
        _seed_graph(kg_store)
        snap = str(tmp_path / "snap.json")
        with open(snap, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "nodes": {}, "edges": {}}, f)
        g = KnowledgeGraph()
        assert not g.load_snapshot(snap)
        g.rebuild_from_store(kg_store)
        assert g.stats() == {"nodes": 3, "edges": 2}

    def test_edge_lookup_either_orientation(self, kg_store):
        _seed_graph(kg_store)
        g = KnowledgeGraph()
        g.rebuild_from_store(kg_store)
        pk = pair_key("乔布斯", "苹果")  # 反向 pair_key 也能命中同一条无向边
        assert g.get_edge_by_pair(pk) is not None
        assert g.get_edge_by_pair(pk)["weight"] == 2

    def test_degree_updates_on_edge_upsert(self):
        g = KnowledgeGraph()
        g.upsert_node("a", "A", "Person", "", [])
        g.upsert_node("b", "B", "Person", "", [])
        g.upsert_node("c", "C", "Person", "", [])
        g.upsert_edge("a", "b", "A", "B", "", "", "", 1, ["c1"], [])
        assert g.get_node("a")["degree"] == 1
        g.upsert_edge("a", "c", "A", "C", "", "", "", 1, ["c1"], [])
        assert g.get_node("a")["degree"] == 2
        assert len(g.neighbor_edges("a")) == 2

    def test_networkx_internal_graph(self):
        import networkx as nx
        g = KnowledgeGraph()
        g.upsert_node("a", "A", "Person", "d", ["c1"])
        g.upsert_edge("a", "b", "A", "B", "r", "kw", "desc", 3, ["c1", "c2"], ["f"])
        assert isinstance(g._g, nx.Graph)
        assert not g._g.is_directed()
        assert g._g.edges["a", "b"]["weight"] == 3  # max(3, 2 chunks)


class TestEngineKgTools:
    """MCP 工具路径（engine.kg_*）真实引擎测试：入库 → 查询 → 增删 → 检索。"""

    EXTRACT_A = (
        "entity<|#|>苹果公司<|#|>Organization<|#|>科技公司\n"
        "entity<|#|>乔布斯<|#|>Person<|#|>创始人\n"
        "relation<|#|>苹果公司<|#|>乔布斯<|#|>创始人<|#|>乔布斯创办苹果\n"
        "relation<|#|>苹果公司<|#|>沃兹尼亚克<|#|>联合创始人<|#|>沃兹共同创办\n"
        "entity<|#|>沃兹尼亚克<|#|>Person<|#|>联合创始人\n")

    def _make_engine(self, tmp_path):
        from agent.rag.engine import RAGEngine
        eng = RAGEngine(workspace=str(tmp_path / "ws"),
                        knowledge_base=str(tmp_path / "kb"),
                        embedding_function=CallableEF())
        eng.set_aux_llm(FakeLLM(extract_output=self.EXTRACT_A))
        return eng

    @staticmethod
    def _seed(eng, docs=(("a.md", "苹果公司由乔布斯创立。"),
                         ("b.md", "苹果总部位于库比蒂诺。"))):
        eng._get_store().add([
            {"text": t, "source": s, "chunk_index": 0} for s, t in docs])

    def test_ingest_all_and_query_tools(self, tmp_path):
        eng = self._make_engine(tmp_path)
        try:
            self._seed(eng)
            stats = eng.kg_ingest()
            assert stats["chunks"] == 2 and stats["graph"]["nodes"] >= 2
            # get_entity：属性 + 关联边 + 方向
            ent = eng.kg_get_entity("苹果公司")
            assert ent["type"] == "Organization"
            assert any(r["relation_type"] == "创始人"
                       for r in ent["relationships"])
            # 邻居
            nb = eng.kg_get_neighbors("乔布斯")
            assert nb["degree"] == 1
            assert nb["neighbors"][0]["name"] == "苹果公司"
            # 寻路：三种算法
            for algo in ("bfs", "dfs", "dijkstra"):
                p = eng.kg_get_path("乔布斯", "沃兹尼亚克", algo)
                assert p["found"] is True
                assert {n["name"] for n in p["nodes"]} >= {"乔布斯", "沃兹尼亚克"}
            # 子图：depth=1 + edge_types 过滤
            sub = eng.kg_get_subgraph("苹果公司", depth=1, edge_types=["创始人"])
            assert sub["nodes"][0]["name"] == "苹果公司"
            assert all("创始人" in (e["relation_type"] + e["keywords"]).lower()
                       for e in sub["edges"])
            # search_graph：非法模式报错；合法模式返回列表
            with pytest.raises(ValueError):
                eng.kg_search_graph("q", mode="naive")
            results = eng.kg_search_graph("苹果公司 创始人", mode="mix")
            assert isinstance(results, list)
        finally:
            eng.close()

    def test_ingest_doc_filter(self, tmp_path):
        eng = self._make_engine(tmp_path)
        try:
            self._seed(eng)
            stats = eng.kg_ingest(doc_name="a.md")
            assert stats["chunks"] == 1
            with pytest.raises(ValueError):
                eng.kg_ingest(doc_name="不存在的文档")
        finally:
            eng.close()

    def test_ingest_requires_available_extractor(self, tmp_path):
        from agent.rag.engine import RAGEngine
        eng = RAGEngine(workspace=str(tmp_path / "ws"),
                        knowledge_base=str(tmp_path / "kb"),
                        embedding_function=CallableEF())
        try:
            self._seed(eng)
            with pytest.raises(RuntimeError):
                eng.kg_ingest()  # 未注入辅助模型 → 抽取器不可用
        finally:
            eng.close()

    def test_add_entity_merges_variant(self, tmp_path):
        eng = self._make_engine(tmp_path)
        try:
            self._seed(eng)
            eng.kg_ingest()
            from agent.rag.kg.builder import kg_dir_for
            from agent.rag.kg.storage import KGStore
            before = KGStore(kg_dir_for(eng._rag_dir)).counts()["entities"]
            out = eng.kg_add_entity("苹果", "Organization", "手工补充")
            assert out["created"] is False  # 变体写法合并进已有实体
            after = KGStore(kg_dir_for(eng._rag_dir)).counts()["entities"]
            assert after == before  # 不新增实体行
        finally:
            eng.close()

    def test_add_relationship_auto_creates_endpoints(self, tmp_path):
        eng = self._make_engine(tmp_path)
        try:
            self._seed(eng)
            eng.kg_ingest()
            out = eng.kg_add_relationship("库比蒂诺", "苹果公司", "位于",
                                          description="总部所在地")
            assert out["created"] is True
            ent = eng.kg_get_entity("库比蒂诺")  # 不存在 → 现场建 UNKNOWN
            assert ent["type"] == "UNKNOWN"
            nb = eng.kg_get_neighbors("苹果公司")
            assert any(n["name"] == "库比蒂诺" for n in nb["neighbors"])
        finally:
            eng.close()

    def test_delete_entity_cascades(self, tmp_path):
        eng = self._make_engine(tmp_path)
        try:
            self._seed(eng)
            eng.kg_ingest()
            out = eng.kg_delete_entity("苹果公司")
            assert out["deleted"] == "苹果公司"
            with pytest.raises(KeyError):
                eng.kg_get_entity("苹果公司")
            # 关联边一并删除：乔布斯度数归零
            nb = eng.kg_get_neighbors("乔布斯")
            assert nb["degree"] == 0
        finally:
            eng.close()

    def test_path_and_entity_errors(self, tmp_path):
        eng = self._make_engine(tmp_path)
        try:
            self._seed(eng)
            eng.kg_ingest()
            with pytest.raises(KeyError):
                eng.kg_get_path("苹果公司", "不存在实体")
            with pytest.raises(ValueError):
                eng.kg_get_path("苹果公司", "乔布斯", algorithm="astar")
            with pytest.raises(KeyError):
                eng.kg_get_entity("幽灵")
        finally:
            eng.close()


class TestEngineExtractorScheme:
    def test_status_and_retriever_scheme(self, tmp_path):
        from agent.rag.engine import RAGEngine
        eng = RAGEngine(workspace=str(tmp_path),
                        knowledge_base=str(tmp_path / "kb"),
                        embedding_function=FakeEF(),
                        kg_strategy="local",
                        kg_extractor="gliner")
        try:
            assert eng.status()["kg_extractor"] == "gliner"
            ret = eng._get_kg_retriever()
            assert ret._extractor.scheme == "gliner"
            # 即时切换方案 → 检索器 extractor 跟着换
            eng.set_kg_extractor("llm_pydantic")
            assert eng._get_kg_retriever()._extractor.scheme == "llm_pydantic"
            assert isinstance(eng._get_kg_retriever()._extractor, PydanticLLMExtractor)
        finally:
            eng.close()

    def test_default_scheme_is_llm_raw(self, tmp_path):
        from agent.rag.engine import RAGEngine
        eng = RAGEngine(workspace=str(tmp_path),
                        knowledge_base=str(tmp_path / "kb"),
                        embedding_function=FakeEF())
        try:
            assert eng.status()["kg_extractor"] == "llm_raw"
        finally:
            eng.close()


# ---------------------------------------------------------------------------
# 实体消歧与合并（入库阶段）
# ---------------------------------------------------------------------------

from agent.rag.kg.entity_resolver import (  # noqa: E402
    EntityResolver, canonical_forms, types_compatible,
    semantic_consolidate,
)


class TestCanonicalForms:
    def test_case_and_space(self):
        assert normalize_name("  Apple   Inc ") in canonical_forms("  Apple   Inc ")

    def test_org_suffix_stripped(self):
        forms = canonical_forms("苹果公司")
        assert "苹果" in forms

    def test_english_suffix_stripped(self):
        assert "apple" in canonical_forms("Apple Inc.")
        assert "apple" in canonical_forms("apple corporation")

    def test_parenthetical_both_sides(self):
        forms = canonical_forms("苹果（Apple）")
        assert "苹果（apple）" in forms
        assert "苹果" in forms and "apple" in forms

    def test_fullwidth_normalized(self):
        # 全角字母数字 → NFKC 半角
        assert "iphone15" in canonical_forms("ｉＰｈｏｎｅ　１５")

    def test_types_compatible(self):
        assert types_compatible("Person", "Person")
        assert types_compatible("UNKNOWN", "Organization")
        assert types_compatible("", "Concept")
        assert not types_compatible("Person", "Organization")


class TestMergeEntities:
    def test_merge_aggregates_evidence(self, kg_store):
        a, _ = kg_store.upsert_entity("苹果", "Organization", "科技公司", "c1")
        b, _ = kg_store.upsert_entity("苹果公司", "Organization", "消费电子巨头", "c2")
        kg_store.upsert_entity("苹果", "Organization", "补充描述", "c3")
        stats = kg_store.merge_entities(b, a)
        assert stats["entities"] == 1
        row = kg_store.get_entity_by_key(normalize_name("苹果"))
        assert set(row["source_ids"]) == {"c1", "c2", "c3"}
        assert kg_store.get_entity_by_key(normalize_name("苹果公司")) is None
        # 被合并实体名成为别名
        assert kg_store.find_entity_by_alias("苹果公司")["id"] == a

    def test_merge_rewrites_relationship_endpoints(self, kg_store):
        a, _ = kg_store.upsert_entity("苹果", "Organization", "", "c1")
        b, _ = kg_store.upsert_entity("苹果公司", "Organization", "", "c2")
        kg_store.upsert_entity("乔布斯", "Person", "", "c1")
        rid, _ = kg_store.upsert_relationship("苹果公司", "乔布斯", "创始人", "创办", "c2")
        kg_store.merge_entities(b, a)
        rel = kg_store.get_relationship(rid)
        assert rel is not None
        # 端点改写为规范名（db 行按 pair_key 排序存两端展示名）
        assert {rel["src_entity"], rel["dst_entity"]} == {"苹果", "乔布斯"}
        assert rel["pair_key"] == pair_key("苹果", "乔布斯")

    def test_merge_collides_into_single_relationship(self, kg_store):
        a, _ = kg_store.upsert_entity("苹果", "Organization", "", "c1")
        b, _ = kg_store.upsert_entity("苹果公司", "Organization", "", "c2")
        kg_store.upsert_entity("乔布斯", "Person", "", "c1")
        r1, _ = kg_store.upsert_relationship("苹果", "乔布斯", "创始人", "创办", "c1")
        r2, _ = kg_store.upsert_relationship("苹果公司", "乔布斯", "领导", "领导苹果", "c2")
        stats = kg_store.merge_entities(b, a)
        assert stats["relationships"] == 1
        rel = kg_store.get_relationship_by_pair(pair_key("苹果", "乔布斯"))
        assert rel["id"] == r1  # 保留已有行
        assert rel["raw_weight"] == 2  # 两条证据相加
        assert "创始人" in rel["keywords"] and "领导" in rel["keywords"]
        assert set(rel["source_ids"]) == {"c1", "c2"}
        assert kg_store.get_relationship(r2) is None

    def test_merge_drops_self_loop(self, kg_store):
        a, _ = kg_store.upsert_entity("苹果", "Organization", "", "c1")
        b, _ = kg_store.upsert_entity("苹果公司", "Organization", "", "c2")
        rid, _ = kg_store.upsert_relationship("苹果公司", "苹果", "同义", "同一实体", "c2")
        stats = kg_store.merge_entities(b, a)
        assert stats["relationships"] == 1
        assert kg_store.get_relationship(rid) is None

    def test_merge_deletes_orphan_vectors(self, kg_store):
        a, _ = kg_store.upsert_entity("苹果", "Organization", "", "c1")
        b, _ = kg_store.upsert_entity("苹果公司", "Organization", "", "c2")
        kg_store.put_vector(f"ent:{b}", "entity", b, "h", b"\x01")
        kg_store.merge_entities(b, a)
        assert kg_store.get_vector(f"ent:{b}") is None
        assert kg_store.counts()["vectors"] == 0

    def test_purge_cleans_aliases(self, kg_store):
        a, _ = kg_store.upsert_entity("苹果", "Organization", "", "doc1_0")
        kg_store.add_alias("apple", "Apple", a)
        kg_store.purge_document(["doc1_"])
        assert kg_store.find_entity_by_alias("apple") is None


class TestEntityResolver:
    def test_variant_resolves_and_does_not_insert(self, kg_store):
        resolver = EntityResolver(kg_store)
        name1, new1, _ = resolver.merge_or_create("苹果公司", "Organization", "科技公司", "c1")
        name2, new2, merged = resolver.merge_or_create("苹果", "Organization", "消费电子", "c2")
        assert new1 and not new2 and merged
        assert name1 == name2 == "苹果公司"  # 不新增实体行，统一到已有实体
        assert kg_store.counts()["entities"] == 1
        row = kg_store.get_entity_by_key(normalize_name("苹果公司"))
        assert set(row["source_ids"]) == {"c1", "c2"}

    def test_alias_hit_for_later_variant(self, kg_store):
        resolver = EntityResolver(kg_store)
        resolver.merge_or_create("苹果公司", "Organization", "", "c1")
        # 括号注释写法也能命中
        name, is_new, merged = resolver.merge_or_create(
            "苹果（Apple）", "Organization", "", "c2")
        assert not is_new and merged and name == "苹果公司"

    def test_type_guard_prevents_wrong_merge(self, kg_store):
        resolver = EntityResolver(kg_store)
        resolver.merge_or_create("苹果", "Organization", "", "c1")
        # 变体名相同但类型冲突（如「苹果」水果 vs 「苹果公司」）→ 独立实体
        name, is_new, _merged = resolver.merge_or_create("苹果公司", "Person", "", "c2")
        assert is_new
        assert kg_store.counts()["entities"] == 2
        assert name == "苹果公司"

    def test_resolve_returns_none_when_disabled(self, kg_store):
        store_resolver = EntityResolver(kg_store, enabled=False)
        store_resolver.merge_or_create("苹果", "Organization", "", "c1")
        store_resolver.merge_or_create("苹果公司", "Organization", "", "c2")
        assert kg_store.counts()["entities"] == 2

    def test_consolidate_merges_existing_duplicates(self, kg_store):
        # 模拟历史数据：同名变体已分别入库（未经消歧）
        kg_store.upsert_entity("苹果", "Organization", "描述A", "c1")
        kg_store.upsert_entity("苹果公司", "Organization", "描述B", "c2")
        kg_store.upsert_entity("乔布斯", "Person", "", "c1")
        kg_store.upsert_relationship("苹果", "乔布斯", "创始人", "创办", "c1")
        kg_store.upsert_relationship("苹果公司", "乔布斯", "领导", "领导", "c2")
        stats = EntityResolver(kg_store).consolidate()
        assert stats["merged"] == 1
        assert kg_store.counts()["entities"] == 2  # 苹果 + 乔布斯
        # 归并后两个写法都能解析到同一实体
        kept = kg_store.find_entity_by_alias("苹果")
        assert kept is not None and kept["id"] == kg_store.find_entity_by_alias("苹果公司")["id"]
        # 关系只剩一条且证据合并
        rels = kg_store.iter_relationships()
        assert len(rels) == 1 and rels[0]["raw_weight"] == 2

    def test_consolidate_registers_aliases(self, kg_store):
        kg_store.upsert_entity("苹果公司", "Organization", "", "c1")
        EntityResolver(kg_store).consolidate()
        assert kg_store.find_entity_by_alias("苹果") is not None


class TestBuilderDisambiguation:
    def test_build_merges_variant_entities(self, rag_dir):
        extract_output = (
            "entity<|#|>苹果公司<|#|>Organization<|#|>科技公司\n"
            "entity<|#|>乔布斯<|#|>Person<|#|>创始人\n"
            "relation<|#|>苹果公司<|#|>乔布斯<|#|>创始人<|#|>乔布斯创办苹果\n")
        store = FakeStore([
            {"id": "c1_0", "text": "苹果公司由乔布斯创立。", "source": "a.md", "chunk_index": 0},
        ])
        builder = KGBuilder(rag_dir, store, KGExtractor(FakeLLM(extract_output=extract_output)))
        try:
            stats = builder.build()
        finally:
            builder.close()
        assert stats["entities"] == 2
        kg = KGStore(kg_dir_for(rag_dir))
        try:
            # 两个实体入库、关系端点写成规范名（后续片段的变体写法会命中同一行）
            assert kg.counts()["entities"] == 2
            assert kg.get_relationship_by_pair(pair_key("苹果公司", "乔布斯")) is not None
        finally:
            kg.close()

    def test_build_reuses_entity_across_chunks(self, rag_dir):
        store = FakeStore([
            {"id": "c1_0", "text": "苹果公司由乔布斯创立。", "source": "a.md", "chunk_index": 0},
            {"id": "c2_0", "text": "苹果总部位于库比蒂诺。", "source": "b.md", "chunk_index": 0},
        ])
        # 第一片抽到全称、第二片抽到简称
        outputs = [
            "entity<|#|>苹果公司<|#|>Organization<|#|>科技公司\n"
            "entity<|#|>乔布斯<|#|>Person<|#|>创始人\n"
            "relation<|#|>苹果公司<|#|>乔布斯<|#|>创始人<|#|>创办\n",
            "entity<|#|>苹果<|#|>Organization<|#|>总部\n",
        ]

        class SeqLLM(FakeLLM):
            def __init__(self):
                super().__init__()
                self._i = 0

            def batch_chat(self, messages_list, *args, **kwargs):
                text = messages_list[0][-1]["content"] if messages_list else ""
                if "high_level_keywords" in text:
                    return [self.keywords]
                self._i += 1
                return [outputs[min(self._i - 1, len(outputs) - 1)]]

        builder = KGBuilder(rag_dir, store, KGExtractor(SeqLLM()))
        try:
            stats = builder.build()
        finally:
            builder.close()
        assert stats["merged_entities"] == 1
        kg = KGStore(kg_dir_for(rag_dir))
        try:
            assert kg.counts()["entities"] == 2  # 苹果（含苹果公司） + 乔布斯
            ent = kg.get_entity_by_key(normalize_name("苹果公司"))
            assert set(ent["source_ids"]) == {"c1_0", "c2_0"}
        finally:
            kg.close()


# ---------------------------------------------------------------------------
# 语义向量消歧（后期手动触发：对库内现有实体做向量对齐与合并）
# ---------------------------------------------------------------------------

class TestSemanticConsolidate:
    """semantic_consolidate：实体向量余弦 ≥ 阈值 → 并查集聚类 → merge_entities。

    FakeEF 关键词向量：苹果(0)/乔布斯(1)/公司(3)/创始(4)/科技(5)。
    「苹果公司\n苹果 科技」与「苹果科技公司\n苹果 公司」关键词集合相同
    → 余弦 1.0 ≥ 0.92 必合并；「苹果公司饮品」与「苹果公司」仅部分重叠
    （2/√6 ≈ 0.816 < 0.92）不合并。
    """

    @staticmethod
    def _vectorize(store):
        from agent.rag.kg.builder import vectorize_pending_rows
        vectorize_pending_rows(store, KGEmbedder(CallableEF()))

    def test_semantic_merge_rewrites_relationships(self, kg_store):
        a, _ = kg_store.upsert_entity("苹果公司", "Organization", "苹果 科技", "c1")
        b, _ = kg_store.upsert_entity("苹果科技公司", "Organization", "苹果 公司", "c2")
        kg_store.upsert_entity("乔布斯", "Person", "创始", "c1")
        kg_store.upsert_relationship("苹果公司", "乔布斯", "创始人", "创办", "c1")
        kg_store.upsert_relationship("苹果科技公司", "乔布斯", "领导", "执掌", "c2")
        self._vectorize(kg_store)
        stats = semantic_consolidate(kg_store, CallableEF())
        assert stats["scanned"] == 3
        assert stats["merged"] == 1
        assert stats["threshold"] == pytest.approx(0.92)
        assert kg_store.counts()["entities"] == 2
        # 两个写法都解析到同一实体（保留方按主名查、被合并方按别名查）
        kept_a = kg_store.get_entity_by_key(normalize_name("苹果公司"))
        kept_b = kg_store.find_entity_by_alias("苹果科技公司")
        assert kept_a is not None and kept_b is not None
        assert kept_a["id"] == kept_b["id"]
        # 关系端点改写并聚合为一条（raw_weight = 2 条证据）
        rels = kg_store.iter_relationships()
        assert len(rels) == 1 and rels[0]["raw_weight"] == 2

    def test_type_guard_prevents_semantic_merge(self, kg_store):
        # 向量几乎相同但类型冲突 → 与规则消歧同一守卫，不合并
        kg_store.upsert_entity("苹果公司", "Organization", "苹果 科技", "c1")
        kg_store.upsert_entity("苹果科技公司", "Product", "苹果 公司", "c2")
        self._vectorize(kg_store)
        stats = semantic_consolidate(kg_store, CallableEF())
        assert stats["merged"] == 0
        assert kg_store.counts()["entities"] == 2

    def test_below_threshold_not_merged(self, kg_store):
        # 部分关键词重叠（cos ≈ 0.816 < 0.92）：宁漏合不误合
        kg_store.upsert_entity("苹果公司", "Organization", "科技", "c1")
        kg_store.upsert_entity("苹果公司饮品", "Organization", "", "c2")
        self._vectorize(kg_store)
        stats = semantic_consolidate(kg_store, CallableEF())
        assert stats["merged"] == 0
        assert kg_store.counts()["entities"] == 2

    def test_custom_threshold(self, kg_store):
        # 阈值降到 0.8 → 部分重叠对也被合并
        kg_store.upsert_entity("苹果公司", "Organization", "科技", "c1")
        kg_store.upsert_entity("苹果公司饮品", "Organization", "", "c2")
        self._vectorize(kg_store)
        stats = semantic_consolidate(kg_store, CallableEF(), threshold=0.8)
        assert stats["merged"] == 1

    def test_missing_vectors_skipped(self, kg_store):
        # 无向量（embedding_function=None 且未预置）：只扫描不比对
        kg_store.upsert_entity("苹果公司", "Organization", "科技", "c1")
        kg_store.upsert_entity("苹果科技公司", "Organization", "苹果", "c2")
        stats = semantic_consolidate(kg_store, None)
        assert stats["scanned"] == 2 and stats["merged"] == 0

    def test_fewer_than_two_entities_returns_early(self, kg_store):
        kg_store.upsert_entity("苹果公司", "Organization", "科技", "c1")
        self._vectorize(kg_store)
        stats = semantic_consolidate(kg_store, CallableEF())
        assert stats == {"scanned": 1, "merged": 0, "threshold": 0.92}

    def test_cancellation(self, kg_store):
        from agent.rag.kg.builder import KGCancellation
        kg_store.upsert_entity("苹果公司", "Organization", "苹果 科技", "c1")
        kg_store.upsert_entity("苹果科技公司", "Organization", "苹果 公司", "c2")
        self._vectorize(kg_store)
        cancel = threading.Event()
        cancel.set()
        with pytest.raises(KGCancellation):
            semantic_consolidate(kg_store, CallableEF(), cancel_event=cancel)

    def test_idempotent_rerun(self, kg_store):
        kg_store.upsert_entity("苹果公司", "Organization", "苹果 科技", "c1")
        kg_store.upsert_entity("苹果科技公司", "Organization", "苹果 公司", "c2")
        self._vectorize(kg_store)
        first = semantic_consolidate(kg_store, CallableEF())
        assert first["merged"] == 1
        # 重跑：已合并实体只剩一行，幂等
        second = semantic_consolidate(kg_store, CallableEF())
        assert second["merged"] == 0
        assert kg_store.counts()["entities"] == 1


class TestEngineSemanticConsolidate:
    """engine.kg_semantic_consolidate 集成：构建后消歧入口 + 图快照重建。"""

    def _make_engine(self, tmp_path):
        from agent.rag.engine import RAGEngine
        eng = RAGEngine(workspace=str(tmp_path / "ws"),
                        knowledge_base=str(tmp_path / "kb"),
                        embedding_function=CallableEF())
        eng.set_aux_llm(FakeLLM(extract_output=TestEngineKgTools.EXTRACT_A))
        return eng

    _seed = staticmethod(TestEngineKgTools._seed)

    def test_consolidate_after_ingest(self, tmp_path):
        eng = self._make_engine(tmp_path)
        try:
            self._seed(eng)
            ingested = eng.kg_ingest()
            assert ingested["entities"] == 3
            stats = eng.kg_semantic_consolidate()
            assert set(stats) == {"scanned", "merged", "threshold"}
            assert stats["scanned"] == 3
            # 现有实体向量互不相似（均 < 0.92）→ 不误合并
            assert stats["merged"] == 0
            # 幂等：重复执行结果一致
            assert eng.kg_semantic_consolidate() == stats
        finally:
            eng.close()

    def test_consolidate_merges_manual_variant(self, tmp_path):
        eng = self._make_engine(tmp_path)
        try:
            self._seed(eng)
            eng.kg_ingest()
            # 手工加一个与「苹果公司」向量相同（关键词集合一致）的变体实体
            # ——规则消歧覆盖不到的深层语义对齐场景
            eng._kg_store_open()  # 确保目录存在
            from agent.rag.kg.builder import kg_dir_for
            from agent.rag.kg.storage import KGStore
            kg = KGStore(kg_dir_for(eng._rag_dir))
            try:
                kg.upsert_entity("苹果科技公司", "Organization", "苹果 科技 公司", "manual_0")
            finally:
                kg.close()
            stats = eng.kg_semantic_consolidate()
            assert stats["merged"] == 1
            kg = KGStore(kg_dir_for(eng._rag_dir))
            try:
                ids = {kg.get_entity_by_key(normalize_name("苹果公司"))["id"],
                       kg.find_entity_by_alias("苹果科技公司")["id"]}
                assert len(ids) == 1
            finally:
                kg.close()
        finally:
            eng.close()
