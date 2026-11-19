"""defer 异步填充（文本先行入库 + 后台回填 embedding）的测试。

覆盖：defer 写入占位行（embedded=false，BM25 立即可用）、检索过滤
（向量通道/两段式粗筛不返回 pending 行）、后台回填线程补齐向量并
重建索引、崩溃恢复（start_backfill 续填）、迁移旧行 embedded 默认 True。
"""
from __future__ import annotations

import time

import pytest


@pytest.fixture
def store(tmp_path, vector_store_factory):
    return vector_store_factory(str(tmp_path / "vectors"))


def _chunks(source, texts):
    return [{"source": source, "chunk_index": i, "text": t}
            for i, t in enumerate(texts)]


def test_defer_add_writes_pending_rows_and_fts_ready(store):
    """defer 写入：占位行入库、BM25 立即可用、向量通道暂无结果。"""
    texts = ["异步填充的第一段文本内容", "异步填充的第二段文本内容"]
    n = store.add_streaming(iter(_chunks("/kb/a.md", texts)),
                            batch_size=2, defer_embeddings=True)
    assert n == 2
    assert store.count_pending() == 2
    rows = store._ensure_table().to_arrow().to_pylist()
    assert all(r["embedded"] is False for r in rows)
    assert all(all(v == 0.0 for v in r["vector"]) for r in rows)
    assert all(all(b == 0 for b in r["vector_bin"]) for r in rows)
    # BM25 立即可用（文本与分词列已入库）
    fts = store.search_fts("异步填充", top_k=5)
    assert len(fts) == 2
    # 向量通道过滤 pending 行 → 空
    assert store.search("异步填充", top_k=5) == []


def test_backfill_fills_vectors_and_search_recovers(store):
    """回填线程：占位行全部填充后向量检索恢复。"""
    texts = ["后台回填测试文本甲", "后台回填测试文本乙"]
    store.add_streaming(iter(_chunks("/kb/b.md", texts)),
                        batch_size=2, defer_embeddings=True)
    assert store.start_backfill() is True
    thread = store._backfill_thread
    thread.join(timeout=30)
    assert not thread.is_alive()
    assert store.count_pending() == 0
    rows = store._ensure_table().to_arrow().to_pylist()
    assert len(rows) == 2
    assert all(r["embedded"] is True for r in rows)
    assert all(any(v != 0.0 for v in r["vector"]) for r in rows)
    # 向量检索恢复：查询词命中已填充切片（结果 dict 不含 embedded 列，
    # 通过行表验证标志）
    hits = store.search("后台回填", top_k=5)
    assert len(hits) == 2
    assert all(r["embedded"] is True for r in rows)
    # 再启动：无 pending → 幂等跳过
    assert store.start_backfill() is False


def test_search_filters_pending_after_partial_fill(store):
    """混合状态检索：已填充来源可召回，pending 来源不混入假相似度。"""
    store.add_streaming(iter(_chunks("/kb/done.md", ["已填充的旧内容甲"])),
                        batch_size=1, defer_embeddings=True)
    store.start_backfill()
    store._backfill_thread.join(timeout=30)
    # 新来源以 defer 模式追加（pending 占位）
    store.add_streaming(iter(_chunks("/kb/new.md", ["待填充的新内容乙"])),
                        batch_size=1, defer_embeddings=True)
    assert store.count_pending() == 1
    hits = store.search("内容", top_k=5)
    assert {h["source"] for h in hits} == {"/kb/done.md"}
    # 两段式粗筛同样只覆盖已填充行（不因全零向量产生假候选）
    table = store._ensure_table()
    two = store._search_two_stage(table, "内容", 10, None)
    if two is not None:
        vec_cands, fts_cands = two
        assert {c["source"] for c in vec_cands} <= {"/kb/done.md"}


def test_backfill_survives_stop_and_resumes(store):
    """停止回填后 pending 保留，再次启动续填（崩溃恢复语义）。"""
    texts = ["恢复测试文本一", "恢复测试文本二"]
    store.add_streaming(iter(_chunks("/kb/c.md", texts)),
                        batch_size=2, defer_embeddings=True)
    # 未启动线程时 stop 是无操作；启动后立即置 stop（协作式退出）
    store.start_backfill()
    store.stop_backfill(join_timeout=0.5)
    # 无论线程是否已填完，start_backfill 都能续填到 0 pending
    deadline = time.monotonic() + 30
    while store.count_pending() > 0 and time.monotonic() < deadline:
        if (store._backfill_thread is None
                or not store._backfill_thread.is_alive()):
            store.start_backfill()
        if store._backfill_thread is not None:
            store._backfill_thread.join(timeout=5)
    assert store.count_pending() == 0
    assert len(store.search("恢复测试", top_k=5)) == 2


def test_sync_add_sets_embedded_true(store):
    """同步嵌入路径（含迁移兜底语义）：所有正式行 embedded=True。"""
    store.add(_chunks("/kb/e.md", ["正常同步嵌入的文本"]))
    rows = store._ensure_table().to_arrow().to_pylist()
    assert rows and all(r["embedded"] is True for r in rows)
    assert store.count_pending() == 0


def test_engine_recovery_backfills_pending(tmp_path, rag_engine_factory):
    """引擎级崩溃恢复：上个进程 defer 入库后退出，新引擎实例自动续填。

    回归场景：start_backfill 依赖的 _backfill_* 属性曾在懒初始化
    _initialize_locked 里才创建，engine.__init__ 的恢复钩子对全新
    store 调用时抛 AttributeError 被吞，pending 永远不续填。
    """
    eng1 = rag_engine_factory(str(tmp_path), embed_background=False)
    store1 = eng1._get_store()
    store1.add_streaming(
        iter([{"source": "/kb/x.md", "chunk_index": 0,
               "text": "引擎恢复测试文本"}]),
        batch_size=1, defer_embeddings=True)
    assert store1.has_pending()
    eng1.close()

    eng2 = rag_engine_factory(str(tmp_path), embed_background=True)
    try:
        store2 = eng2._get_store()
        deadline = time.monotonic() + 60
        while store2.has_pending() and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not store2.has_pending()
        hits = store2.search("引擎恢复", top_k=3)
        assert len(hits) == 1
    finally:
        eng2.close()
