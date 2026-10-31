"""二值量化粗筛（两段式检索第一阶段，2026-10-31）的测试。

覆盖：
- 二值化辅助函数：sign 量化 + 位打包（每 8 维 1 字节）、字节数换算；
- 入库：``vector_bin`` 列与 int8 向量 / 原文同行存储且数值一致；
- 旧表自动迁移：缺 ``vector_bin`` 的旧库打开时 drop 重建补列（从已存量化
  向量派生，无需重嵌入），已有 ``text_fts`` 原样带回（列集合必须与 schema
  一致，否则 create_table 直接抛 "different number of columns"）；
- 检索：开启后 ``search_with_rerank`` 走「XOR+Popcount 汉明粗筛 → 候选集内
  余弦 + BM25 → rerank」两段式；关闭 / 粗筛扫描失败时透明回退原流程。
"""
from __future__ import annotations

import hashlib

import lancedb
import numpy as np
import pyarrow as pa
import pytest

from agent.config import AgentConfig
from agent.rag import vector_store as vs
from agent.rag.vector_store import (
    _BIN_COLUMN,
    _FTS_COLUMN,
    _TABLE_NAME,
    VectorStore,
    _bin_bytes_for_dim,
    _binarize_quantized,
    _hash_source,
    _tokenize_for_fts,
    set_binary_prefilter,
)


class SignedFakeEmbeddingFunction:
    """零中心确定性假嵌入（有正有负分量）。

    conftest 的 FakeEmbeddingFunction 分量全为正（md5 字节 / 255 ∈ [0,1]），
    sign 二值化后所有行都是全 1、汉明距离恒 0，粗筛退化成随机抽样——无法
    用于两段式检索测试。本实现把 md5 字节映射到 [-1, 1]，sign 分布约各半，
    二值量化才有区分度（真实 embedding 模型即为零中心分布）。
    """

    def __call__(self, input):
        out = []
        for text in input:
            h = hashlib.md5(text.encode("utf-8")).digest()
            out.append([(h[i % len(h)] - 127.5) / 127.5 for i in range(128)])
        return out

    def embed_query(self, text: str):
        return self.__call__([text])[0]

    def name(self):
        return "signed-fake-embedding"


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def test_bin_bytes_for_dim():
    assert _bin_bytes_for_dim(128) == 16
    assert _bin_bytes_for_dim(256) == 32
    assert _bin_bytes_for_dim(129) == 17      # 非 8 倍数向上取整
    assert _bin_bytes_for_dim(8) == 1
    assert _bin_bytes_for_dim(0) == 0


def test_binarize_quantized_sign_and_packing():
    # sign 量化：>0 → 1，否则 → 0（含 0.0）；packbits 后高位在前。
    vec = [0.5, -0.5, 0.0, 0.25, -0.1, 0.9, -0.7, 0.3]
    packed = _binarize_quantized(vec)
    assert len(packed) == 1
    # 期望位串 1,0,0,1,0,1,0,1 = 0b10010101
    assert packed[0] == 0b10010101


def test_binarize_quantized_padding_bits_are_zero():
    # 非 8 倍数维度：尾部补零位（查询侧同样补齐，XOR 不产生汉明距离贡献）。
    packed = _binarize_quantized([1.0] * 3)
    assert len(packed) == 1
    assert packed[0] == 0b11100000


# ---------------------------------------------------------------------------
# 入库：vector_bin 列
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return VectorStore(persist_dir=str(tmp_path / "vectors"),
                       embedding_function=SignedFakeEmbeddingFunction())


def _add(store, source, texts):
    store.add([
        {"source": source, "chunk_index": i, "text": t}
        for i, t in enumerate(texts)
    ])


def test_add_populates_vector_bin(store):
    _add(store, "a.pdf", ["机器学习内容", "深度学习内容", "普通内容"])
    rows = store._ensure_table().to_arrow().to_pylist()
    assert rows, "应有数据行"
    for r in rows:
        assert _BIN_COLUMN in r
        # 与量化向量派生一致（新增行与迁移行同源）
        assert r[_BIN_COLUMN] == _binarize_quantized(r["vector"])
        # fake 嵌入 128 维 → 16 字节
        assert len(r[_BIN_COLUMN]) == _bin_bytes_for_dim(len(r["vector"]))


# ---------------------------------------------------------------------------
# 旧表自动迁移
# ---------------------------------------------------------------------------

def _make_old_table(db_dir: str, texts, with_fts: bool):
    """直接用 lancedb 建一张『升级前』的旧 schema 表（无 vector_bin）。"""
    ef = SignedFakeEmbeddingFunction()
    helper_dir = db_dir + "_helper"
    st_dummy = VectorStore(persist_dir=helper_dir, embedding_function=ef)
    fields = [
        pa.field("id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float16(), 128)),
        pa.field("source", pa.string()),
        pa.field("chunk_index", pa.int64()),
    ]
    if with_fts:
        fields.append(pa.field(_FTS_COLUMN, pa.string()))
    recs = []
    for i, t in enumerate(texts):
        prepared = st_dummy._prepare_stored_vector(ef([t])[0])
        rec = {"id": f"{_hash_source(t)}_{i}", "text": t, "vector": prepared,
               "source": "old.pdf", "chunk_index": i}
        if with_fts:
            rec[_FTS_COLUMN] = _tokenize_for_fts(t)
        recs.append(rec)
    db = lancedb.connect(db_dir)
    db.create_table(_TABLE_NAME, recs, schema=pa.schema(fields))
    return recs


def test_migration_adds_vector_bin_keeps_fts(tmp_path):
    """只缺 vector_bin 的旧表：迁移补列，已有 text_fts 原样带回。"""
    d = str(tmp_path / "vdb")
    old = _make_old_table(d, ["阿尔法文档内容", "贝塔文档内容"], with_fts=True)
    st = VectorStore(persist_dir=d,
                     embedding_function=SignedFakeEmbeddingFunction())
    st._ensure_initialized()
    schema_names = st._ensure_table().schema.names
    assert _BIN_COLUMN in schema_names
    assert _FTS_COLUMN in schema_names
    rows = st._ensure_table().to_arrow().to_pylist()
    assert len(rows) == len(old)
    by_id = {r["id"]: r for r in old}
    for r in rows:
        assert r[_BIN_COLUMN] == _binarize_quantized(r["vector"])
        # text_fts 原样带回（非空、与旧值一致）
        assert r[_FTS_COLUMN] == by_id[r["id"]][_FTS_COLUMN]


def test_migration_adds_both_columns(tmp_path):
    """更老的表（缺 text_fts + vector_bin）：一次迁移补齐两列。"""
    d = str(tmp_path / "vdb2")
    _make_old_table(d, ["伽马文档内容"], with_fts=False)
    st = VectorStore(persist_dir=d,
                     embedding_function=SignedFakeEmbeddingFunction())
    st._ensure_initialized()
    schema_names = st._ensure_table().schema.names
    assert _BIN_COLUMN in schema_names
    assert _FTS_COLUMN in schema_names
    rows = st._ensure_table().to_arrow().to_pylist()
    assert rows[0][_BIN_COLUMN] == _binarize_quantized(rows[0]["vector"])
    assert rows[0][_FTS_COLUMN] == _tokenize_for_fts(rows[0]["text"])


def test_two_stage_on_migrated_table(tmp_path):
    """迁移后的旧库开启粗筛可正常两段式检索（精确文本命中 top1）。"""
    d = str(tmp_path / "vdb3")
    _make_old_table(d, ["阿尔法文档内容", "贝塔文档内容", "伽马文档内容"],
                    with_fts=True)
    st = VectorStore(persist_dir=d,
                     embedding_function=SignedFakeEmbeddingFunction())
    st._ensure_initialized()
    set_binary_prefilter(True, 200)
    results = st.search_with_rerank("贝塔文档内容", top_k=2)
    assert results
    assert results[0]["text"] == "贝塔文档内容"


# ---------------------------------------------------------------------------
# 两段式检索
# ---------------------------------------------------------------------------

def test_two_stage_disabled_by_default(store):
    _add(store, "a.pdf", ["机器学习内容", "深度学习内容"])
    table = store._ensure_table()
    assert not store._binary_prefilter_ready(table)
    results = store.search_with_rerank("机器学习内容", top_k=2)
    assert results
    assert results[0]["text"] == "机器学习内容"


def test_two_stage_enabled_finds_exact_match(store, caplog):
    _add(store, "a.pdf", ["机器学习内容甲", "机器学习内容乙", "无关内容丙"])
    set_binary_prefilter(True, 200)
    table = store._ensure_table()
    assert store._binary_prefilter_ready(table)
    with caplog.at_level("INFO", logger="agent.rag.vector_store"):
        results = store.search_with_rerank("机器学习内容乙", top_k=3)
    assert results
    assert results[0]["text"] == "机器学习内容乙"
    # 确认走的确实是两段式路径（而非静默回退）
    assert any("二值粗筛" in rec.message for rec in caplog.records)


def test_two_stage_matches_original_flow_top1(store):
    """精确文本查询（fake 嵌入同文同向量）下两种模式 top1 一致。"""
    texts = ["机器学习是人工智能的核心分支", "深度学习使用多层神经网络"]
    _add(store, "a.pdf", texts)
    set_binary_prefilter(False)
    off = store.search_with_rerank("深度学习使用多层神经网络", top_k=3)
    set_binary_prefilter(True, 200)
    on = store.search_with_rerank("深度学习使用多层神经网络", top_k=3)
    assert off and on
    assert off[0]["text"] == on[0]["text"] == "深度学习使用多层神经网络"


def test_two_stage_respects_where(store):
    texts = [f"第{i}段机器学习内容" for i in range(6)]
    _add(store, "/kb/a.pdf", texts)
    _add(store, "/kb/b.pdf", texts)
    set_binary_prefilter(True, 200)
    results = store.search_with_rerank(
        "第1段机器学习内容", top_k=4,
        where="source LIKE '%a.pdf%'")
    # 内存 source 过滤：结果必须全部落在被过滤的 source 上
    assert results, "应返回结果"
    assert all(r["source"] == "/kb/a.pdf" for r in results)


def test_two_stage_where_ram_filter_finds_only_matching_source(store):
    """定向检索（source LIKE）走内存过滤：b.pdf 的独有内容不应被 a.pdf 查询命中。"""
    _add(store, "/kb/a.pdf", ["阿尔法主题独有内容甲", "阿尔法主题独有内容乙"])
    _add(store, "/kb/b.pdf", ["贝塔主题独有内容丙", "贝塔主题独有内容丁"])
    set_binary_prefilter(True, 200)
    results = store.search_with_rerank(
        "贝塔主题独有内容", top_k=4, where="source LIKE '%a.pdf%'")
    # a.pdf 里根本没有『贝塔』内容 → 过滤后无结果（而非跨 source 泄漏召回）
    assert all(r["source"] == "/kb/a.pdf" for r in results)
    assert all("阿尔法" in r["text"] for r in results)


def test_bin_cache_invalidated_after_add(store):
    """写入后内存矩阵缓存必须失效重建，否则新增切片会被粗筛遗漏。"""
    _add(store, "a.pdf", ["机器学习内容甲", "机器学习内容乙"])
    set_binary_prefilter(True, 200)
    assert store.search_with_rerank("机器学习内容甲", top_k=2)
    # 新增一条独有内容的切片
    _add(store, "a.pdf", ["新增的锌钛合金专题切片"])
    # 不经过 add 的显式无效化会漏掉该切片；这里验证写后重建命中
    results = store.search_with_rerank("锌钛合金专题", top_k=2)
    assert any("锌钛合金" in r["text"] for r in results)


def test_two_stage_where_escaped_format_uses_ram_filter(store):
    """生产环境 where 带 ESCAPE '\\' 后缀，仍应走内存 source 过滤（不回退 LanceDB）。"""
    _add(store, "/kb/a.pdf", ["阿尔法主题独有内容甲"])
    _add(store, "/kb/b.pdf", ["贝塔主题独有内容乙"])
    set_binary_prefilter(True, 200)
    results = store.search_with_rerank(
        "阿尔法主题独有内容", top_k=4,
        where="source LIKE '%a.pdf%' ESCAPE '\\'")
    assert results
    assert all(r["source"] == "/kb/a.pdf" for r in results)


def test_two_stage_degrades_on_scan_failure(store, monkeypatch):
    """粗筛列扫描失败时透明回退原流程，检索结果不劣化。"""
    _add(store, "a.pdf", ["机器学习内容甲", "机器学习内容乙"])
    set_binary_prefilter(True, 200)
    table = store._ensure_table()

    def _boom(*args, **kwargs):
        raise RuntimeError("scan exploded")

    orig_search = table.search

    def flaky_search(*args, **kwargs):
        # 阶段一扫描（无查询向量）失败；带查询向量的原流程 ANN 正常。
        if not args and not kwargs:
            return _boom()
        return orig_search(*args, **kwargs)

    monkeypatch.setattr(table, "search", flaky_search)
    results = store.search_with_rerank("机器学习内容乙", top_k=2)
    assert results
    assert results[0]["text"] == "机器学习内容乙"


def test_two_stage_scale_recall(store):
    """规模 + 真实筛选：3000 行、粗筛只留 top 50，精确匹配必须经两段式回到
    候选首位。防止二值量化退化（如全正分量向量使所有行汉明距离同为 0、粗筛
    退化为随机抽样）的实现在小库上侥幸通过。
    """
    n = 3000
    _add(store, "big.pdf",
         [f"文档{i}号机器学习实践指南第{i}章" for i in range(n)])
    set_binary_prefilter(True, 50)
    table = store._ensure_table()
    q = "文档2500号机器学习实践指南第2500章"
    vec_c, _fts_c = store._search_two_stage(table, q, 12, None)
    assert vec_c, "两段式应产出向量候选"
    assert vec_c[0]["text"] == q
    assert vec_c[0]["score"] > 0.99   # 精确文本 → 余弦 ≈ 1.0
    # 端到端：最终结果（rerank / RRF 兜底后）仍包含精确匹配
    results = store.search_with_rerank(q, top_k=3)
    assert any(r["text"] == q for r in results)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

def test_config_defaults_off():
    cfg = AgentConfig()
    assert cfg.rag_binary_prefilter_enabled is False
    assert cfg.rag_binary_prefilter_top == 200


@pytest.mark.parametrize("raw,expected", [
    ({"rag_binary_prefilter_enabled": "true", "rag_binary_prefilter_top": "150"}, (True, 150)),
    ({"rag_binary_prefilter_enabled": True, "rag_binary_prefilter_top": 64}, (True, 64)),
])
def test_config_coerces_string_types(tmp_path, raw, expected):
    import json
    p = tmp_path / "c.json"
    p.write_text(json.dumps(raw))
    cfg = AgentConfig.load(str(p))
    assert (cfg.rag_binary_prefilter_enabled, cfg.rag_binary_prefilter_top) == expected


def test_set_binary_prefilter_and_reset():
    set_binary_prefilter(True, 100)
    assert vs._BINARY_PREFILTER_ENABLED is True
    assert vs._BINARY_PREFILTER_TOP == 100
    # 非法 / 非正 top 回退默认
    set_binary_prefilter(True, 0)
    assert vs._BINARY_PREFILTER_TOP == 200
    set_binary_prefilter(True, "abc")
    assert vs._BINARY_PREFILTER_TOP == 200
    # conftest 的 _reset_global_model_providers 夹具前后都会复位；这里手动验证
    vs.reset_local_providers()
    assert vs._BINARY_PREFILTER_ENABLED is False
    assert vs._BINARY_PREFILTER_TOP == 200


def test_popcount_lut_matches_builtin():
    lut = vs._POPCOUNT_LUT
    for i in (0, 1, 255, 0b1011, 0b11111111):
        assert int(lut[i]) == bin(i).count("1")
    # numpy 位运算路径与 Python 参考一致（小规模随机向量）
    rng = np.random.default_rng(42)
    a = rng.integers(0, 256, size=(17, 16), dtype=np.uint8)
    b = rng.integers(0, 256, size=16, dtype=np.uint8)
    ham = lut[np.bitwise_xor(a, b)].sum(axis=1)
    ref = [sum(bin(int(x)).count("1") for x in (row ^ b)) for row in a]
    assert list(ham) == ref
