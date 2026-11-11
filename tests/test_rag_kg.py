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
