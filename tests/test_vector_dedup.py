"""向量去重（2026-10-22 用户定版）的测试。

VectorStore.deduplicate_vectors：按归一化文本哈希分组，每组保留
order_key 最小的一份（引擎侧 = markdown 缓存 mtime → 入库时间代理），
其余按 id 删除；RAGEngine.dedup_vectors 提供该 order_key。
"""
from __future__ import annotations

import os

import pytest


# ---------------------------------------------------------------------------
# VectorStore.deduplicate_vectors（假嵌入，store 层）
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path, vector_store_factory):
    return vector_store_factory(str(tmp_path / "vectors"))


def _add(store, source, texts):
    store.add([
        {"source": source, "chunk_index": i, "text": t}
        for i, t in enumerate(texts)
    ])


def test_dedup_removes_exact_duplicates_keeping_earliest(store):
    texts = ["第一段重复内容", "第二段重复内容", "独有内容"]
    _add(store, "/kb/a/book.pdf", texts)
    _add(store, "/kb/b/book.pdf", texts)          # 三条与 a 完全相同
    # order_key：source 字典序小者胜出（模拟「入库早」）
    result = store.deduplicate_vectors(
        order_key=lambda r: (r.get("source") or "", r.get("chunk_index") or 0))
    assert result["scanned"] == 6
    assert result["groups"] == 3                  # 三组重复
    assert result["removed"] == 3                 # b 的三条被删
    rows = store._ensure_table().to_arrow().to_pylist()
    assert {r["source"] for r in rows} == {"/kb/a/book.pdf"}
    assert sorted(r["text"] for r in rows) == sorted(texts)


def test_dedup_is_idempotent(store):
    texts = ["重复甲", "重复乙"]
    _add(store, "/kb/a.pdf", texts)
    _add(store, "/kb/b.pdf", texts)
    first = store.deduplicate_vectors()
    assert first["removed"] == 2
    second = store.deduplicate_vectors()
    assert second["removed"] == 0                 # 幂等：无重复可清


def test_dedup_normalizes_whitespace(store):
    # 归一化（strip + 内部空白折叠）后相同即算重复
    _add(store, "/kb/a.pdf", ["同一   段\v文字  "])
    _add(store, "/kb/b.pdf", ["同一 段 文字"])
    result = store.deduplicate_vectors()
    assert result["removed"] == 1


def test_dedup_keeps_one_of_three_copies(store):
    for i in range(3):
        _add(store, f"/kb/copy{i}.pdf", ["三段完全相同的内容"])
    result = store.deduplicate_vectors()
    assert result["groups"] == 1
    assert result["removed"] == 2
    rows = store._ensure_table().to_arrow().to_pylist()
    assert len(rows) == 1
    # order_key 缺省：字典序最小的 source 胜出
    assert rows[0]["source"] == "/kb/copy0.pdf"


def test_dedup_within_same_source(store):
    # 同一文档内的完全重复段落（如重复的空白页/样板页）同样只留一份
    _add(store, "/kb/a.pdf", ["注释", "正文一", "注释"])
    result = store.deduplicate_vectors()
    assert result["removed"] == 1
    rows = store._ensure_table().to_arrow().to_pylist()
    assert sorted(r["text"] for r in rows) == ["正文一", "注释"]


def test_dedup_empty_store(store):
    assert store.deduplicate_vectors() == {
        "scanned": 0, "groups": 0, "removed": 0}


def test_dedup_no_duplicates_noop(store):
    _add(store, "/kb/a.pdf", ["甲", "乙"])
    _add(store, "/kb/b.pdf", ["丙", "丁"])
    result = store.deduplicate_vectors()
    assert result == {"scanned": 4, "groups": 0, "removed": 0}


def test_dedup_progress_callback(store):
    for i in range(5):
        _add(store, f"/kb/f{i}.pdf", [f"内容{i}", f"内容{i}"])   # 5 组重复
    calls = []
    store.deduplicate_vectors(progress_callback=lambda c, t, m: calls.append((c, t)))
    assert calls
    assert calls[-1] == (10, 10)                  # 扫描完成：cur == total


# ---------------------------------------------------------------------------
# RAGEngine.dedup_vectors（order_key = markdown 缓存 mtime 代理）
# ---------------------------------------------------------------------------

@pytest.fixture
def dedup_engine(tmp_path, rag_engine_factory):
    kb = tmp_path / "kb"
    kb.mkdir()
    eng = rag_engine_factory(workspace=str(tmp_path / "ws"),
                             knowledge_base=str(kb))
    return eng, kb


def _touch_md(eng, kb, rel, mtime):
    """给某源文件造一个 markdown 缓存文件并设 mtime（入库时间代理）。"""
    md_path = os.path.join(eng._markdown_dir,
                           __import__("agent.rag.engine", fromlist=["_safe_filename"])
                           ._safe_filename(rel) + ".md")
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# cache\n")
    os.utime(md_path, (mtime, mtime))
    return md_path


def test_engine_dedup_keeps_earliest_md_mtime(dedup_engine):
    eng, kb = dedup_engine
    old_src, new_src = str(kb / "旧目录" / "book.pdf"), str(kb / "新目录" / "book.pdf")
    store = eng._get_store()
    for src in (old_src, new_src):
        store.add([{"source": src, "chunk_index": i, "text": f"相同内容{i}"}
                   for i in range(2)])
    # 旧目录的缓存 mtime 更早（2024），新目录更晚（2026）
    _touch_md(eng, kb, os.path.join("旧目录", "book.pdf"), 1700000000)
    _touch_md(eng, kb, os.path.join("新目录", "book.pdf"), 1760000000)
    result = eng.dedup_vectors()
    assert result["removed"] == 2                 # 新目录（入库晚）的被删
    rows = store._ensure_table().to_arrow().to_pylist()
    assert {r["source"] for r in rows} == {old_src}


def test_engine_dedup_without_cache_falls_back_to_source_order(dedup_engine):
    """无 markdown 缓存的源（缓存被清/文档已删）→ mtime=0，按 source 字典序。"""
    eng, kb = dedup_engine
    a, b = str(kb / "a.pdf"), str(kb / "b.pdf")
    store = eng._get_store()
    for src in (b, a):                            # 故意反序添加
        store.add([{"source": src, "chunk_index": 0, "text": "同一内容"}])
    result = eng.dedup_vectors()
    assert result["removed"] == 1
    rows = store._ensure_table().to_arrow().to_pylist()
    assert rows[0]["source"] == a                 # 字典序小者胜出


# ---------------------------------------------------------------------------
# VectorDedupWorker（UI 线程壳）——直接同步调 run()，不起真线程
# ---------------------------------------------------------------------------

def test_vector_dedup_worker_run_emits_stats(qtbot, tmp_path, monkeypatch):
    """回归：run() 内引用的 DEFAULT_EMBEDDING_MODEL 曾漏导入，点按钮即
    NameError（GUI 启动不触发，只在 worker 运行时炸，测试也没拦住）。
    这里同步调 run()，任何未解析名字都会暴露。"""
    from agent.config import AgentConfig
    from agent.ui.main_window import VectorDedupWorker

    captured = {}

    class FakeEngine:
        def __init__(self, **kw):
            captured["init"] = kw

        def dedup_vectors(self, progress_callback=None):
            if progress_callback is not None:
                progress_callback(5, 10, "已扫描 5/10 条向量")
            return {"scanned": 10, "groups": 2, "removed": 3}

    # run() 里是函数内局部 import，monkeypatch 模块属性即可生效
    monkeypatch.setattr("agent.rag.engine.RAGEngine", FakeEngine)
    cfg = AgentConfig()
    cfg.workspace = str(tmp_path)
    cfg.knowledge_base = str(tmp_path / "kb")
    worker = VectorDedupWorker(cfg)
    got = {}
    worker.dedup_finished.connect(lambda s: got.update(stats=s))
    worker.dedup_progress.connect(lambda c, t, m: got.update(prog=(c, t, m)))
    worker.run()
    assert got["stats"] == {"scanned": 10, "groups": 2, "removed": 3}
    assert got["prog"] == (5, 10, "已扫描 5/10 条向量")
    # embedding_model 回退值必须解析成非空（DEFAULT_EMBEDDING_MODEL）
    assert captured["init"]["embedding_model"]
    assert captured["init"]["workspace"] == str(tmp_path)
