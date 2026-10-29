"""Unit tests for the incremental-sync manifest logic in RAGEngine.

These tests don't require the embedding model — they verify the manifest
load/save, file-signature comparison, and the partitioning logic that
decides which files to process vs skip. The embedding/vector-store layer
is exercised by test_rag_e2e.py and test_rag_full_pipeline.py.
"""
from __future__ import annotations

import json
import os
import time

import pytest

from agent.rag.engine import (
    _content_hash,
    _file_signature,
    _load_manifest,
    _manifest_path,
    _save_manifest,
)


# ---------------------------------------------------------------------------
# manifest helpers
# ---------------------------------------------------------------------------

def test_manifest_path_is_inside_rag_dir(tmp_path):
    p = _manifest_path(str(tmp_path / "rag"))
    assert p.endswith("manifest.json")
    assert os.path.dirname(p).endswith("rag")


def test_load_manifest_returns_empty_when_missing(tmp_path):
    assert _load_manifest(str(tmp_path / "rag")) == {}


def test_load_manifest_returns_empty_when_corrupted(tmp_path):
    rag_dir = tmp_path / "rag"
    rag_dir.mkdir()
    (rag_dir / "manifest.json").write_text("not valid json{{{")
    assert _load_manifest(str(rag_dir)) == {}


def test_load_manifest_returns_empty_when_wrong_shape(tmp_path):
    """A manifest without the 'files' key must return {} not the raw dict."""
    rag_dir = tmp_path / "rag"
    rag_dir.mkdir()
    (rag_dir / "manifest.json").write_text(json.dumps({"wrong_key": {}}))
    assert _load_manifest(str(rag_dir)) == {}


def test_save_and_load_manifest_roundtrip(tmp_path):
    rag_dir = str(tmp_path / "rag")
    files = {
        "/kb/a.txt": {"mtime": 1000.0, "size": 100, "content_hash": "abc", "chunk_count": 3},
        "/kb/b.md": {"mtime": 2000.0, "size": 200, "content_hash": "def", "chunk_count": 5},
    }
    _save_manifest(rag_dir, files)
    loaded = _load_manifest(rag_dir)
    assert loaded == files


def test_save_manifest_is_atomic_no_tmp_leftover(tmp_path):
    """After a successful save, no .tmp file should remain."""
    rag_dir = str(tmp_path / "rag")
    _save_manifest(rag_dir, {})
    assert not os.path.exists(_manifest_path(rag_dir) + ".tmp")


# ---------------------------------------------------------------------------
# file signature & content hash
# ---------------------------------------------------------------------------

def test_file_signature_captures_mtime_and_size(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("hello world")
    sig = _file_signature(str(f))
    assert "mtime" in sig
    assert "size" in sig
    assert sig["size"] == 11


def test_file_signature_changes_when_content_modified(tmp_path):
    """Modifying a file must produce a different signature (mtime changes)."""
    f = tmp_path / "doc.txt"
    f.write_text("original content here")
    sig1 = _file_signature(str(f))
    # Ensure mtime changes even on fast filesystems with coarse mtime resolution.
    time.sleep(0.05)
    f.write_text("modified content here!!")
    os.utime(str(f), None)  # force mtime update
    sig2 = _file_signature(str(f))
    assert sig1 != sig2


def test_content_hash_is_stable():
    """Same input must produce the same hash."""
    h1 = _content_hash("hello world")
    h2 = _content_hash("hello world")
    assert h1 == h2


def test_content_hash_differs_for_different_text():
    assert _content_hash("foo") != _content_hash("bar")


def test_content_hash_is_hex_and_truncated():
    h = _content_hash("test content")
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# incremental ingest partitioning (using a fake VectorStore to avoid model load)
# ---------------------------------------------------------------------------

class _FakeStore:
    """Minimal VectorStore stub for testing ingest logic without loading the
    embedding model. Records all add_streaming calls and delete_by_source calls."""

    def __init__(self):
        self.records = []
        self.deleted_sources = []
        self.deleted_documents = []
        self.deleted_md_paths = []
        self._count = 0

    def add_streaming(self, chunk_iter, batch_size=100, on_batch=None, is_cancelled=None):
        added = 0
        for chunk in chunk_iter:
            self.records.append(chunk)
            added += 1
        self._count += added
        if on_batch:
            on_batch(added, added)
        return added

    def delete_by_source(self, source):
        self.deleted_sources.append(source)
        return 1

    def delete_document(self, doc_name):
        self.deleted_documents.append(doc_name)

    def get_document_by_md_path(self, md_path):
        return None

    def delete_document_by_md_path(self, md_path):
        self.deleted_md_paths.append(md_path)
        return True

    def count(self):
        return self._count

    def list_sources(self):
        return sorted({r.get("source", "") for r in self.records if r.get("source")})

    def clear(self):
        self.records = []
        self._count = 0


def _make_engine(tmp_path, fake_store=None):
    """Build a RAGEngine with a fake embedding function so no model loads."""
    from agent.rag.engine import RAGEngine

    kb = tmp_path / "kb"
    kb.mkdir()
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    engine = RAGEngine(workspace=str(ws), knowledge_base=str(kb))
    if fake_store is not None:
        engine._store = fake_store
    # Create a .txt file in the KB（正文需超过入库最小字数门槛，避免被当作碎片丢弃）
    (kb / "a.txt").write_text(
        "first document content with enough text to be chunked and embedded into "
        "the vector store so that retrieval tests can validate incremental sync "
        "behavior correctly across multiple ingest passes")
    return engine, kb


def test_first_ingest_processes_all_files(tmp_path):
    """First ingest (no manifest) must process all files — none skipped."""
    store = _FakeStore()
    engine, kb = _make_engine(tmp_path, store)
    stats = engine.ingest(force=False)

    assert stats["files_found"] == 1
    assert stats["files_extracted"] == 1
    assert stats["files_skipped"] == 0
    # Manifest must be saved after ingest
    manifest = _load_manifest(engine._rag_dir)
    assert len(manifest) == 1
    entry = list(manifest.values())[0]
    assert "mtime" in entry and "size" in entry and "content_hash" in entry


def test_second_ingest_skips_unchanged_files(tmp_path):
    """Second ingest with force=False must skip all unchanged files — no
    re-parse, no re-chunk, no re-embed, no upsert."""
    store = _FakeStore()
    engine, kb = _make_engine(tmp_path, store)

    # First ingest processes everything
    engine.ingest(force=False)
    assert store.count() > 0  # chunks written
    count_after_first = store.count()

    # Second ingest — all files unchanged
    stats = engine.ingest(force=False)
    assert stats["files_found"] == 1
    assert stats["files_skipped"] == 1
    assert stats["files_extracted"] == 0
    # No new chunks added (no re-embed)
    assert store.count() == count_after_first


def test_force_reingest_processes_all_files(tmp_path):
    """force=True must ignore the manifest and reprocess all files."""
    store = _FakeStore()
    engine, kb = _make_engine(tmp_path, store)

    engine.ingest(force=False)  # first, populates manifest
    # Add a second file so there are 2 total
    (kb / "b.txt").write_text(
        "second document content for chunking purposes here that must be long "
        "enough to pass the minimum ingest length threshold used by the pipeline")

    stats = engine.ingest(force=True)
    assert stats["files_found"] == 2
    assert stats["files_extracted"] == 2
    assert stats["files_skipped"] == 0


def test_modified_file_is_reprocessed(tmp_path):
    """A file whose content changed (mtime/size differ) must be reprocessed."""
    store = _FakeStore()
    engine, kb = _make_engine(tmp_path, store)
    engine.ingest(force=False)

    # Modify the file
    f = kb / "a.txt"
    time.sleep(0.05)
    f.write_text(
        "completely different content that should trigger reprocessing because "
        "the new text is long enough to exceed the minimum ingest length and "
        "force the pipeline to re-chunk and re-embed the modified document")
    os.utime(str(f), None)

    stats = engine.ingest(force=False)
    assert stats["files_extracted"] == 1, "modified file must be reprocessed"
    assert stats["files_skipped"] == 0


def test_new_file_is_processed_on_incremental_sync(tmp_path):
    """Adding a new file between syncs must process only the new file."""
    store = _FakeStore()
    engine, kb = _make_engine(tmp_path, store)
    engine.ingest(force=False)

    # Add a new file
    (kb / "b.txt").write_text(
        "new document added after first sync that contains sufficient text to "
        "exceed the minimum chunk length requirement enforced before ingestion")

    stats = engine.ingest(force=False)
    assert stats["files_found"] == 2
    assert stats["files_extracted"] == 1  # only the new file
    assert stats["files_skipped"] == 1    # the old file is unchanged


def test_deleted_file_is_cleaned_up(tmp_path):
    """A file removed from the KB must have its vectors deleted and manifest
    entry removed on the next incremental sync."""
    store = _FakeStore()
    engine, kb = _make_engine(tmp_path, store)
    engine.ingest(force=False)

    # Record the source path
    deleted_source = str(kb / "a.txt")

    # Remove the file
    os.remove(deleted_source)

    stats = engine.ingest(force=False)
    assert stats["files_deleted"] == 1, "deleted file must be cleaned up"
    assert deleted_source in store.deleted_sources, "vectors must be deleted"
    # 目录概要（documents 元数据表）也必须同步清理（用 basename 作为 doc_id）。
    assert "a.txt" in store.deleted_documents, "document digest must be deleted"

    # Manifest must no longer contain the deleted file
    manifest = _load_manifest(engine._rag_dir)
    assert deleted_source not in manifest


def test_deleted_file_markdown_cache_is_removed(tmp_path):
    """删除清理必须连 Markdown 缓存一起删。

    旧实现只删向量和元数据，而紧接着的
    ``ensure_digests_for_cached_documents`` 会发现「有缓存文件、无 digest
    记录」，于是用旧缓存重新生成一条 —— 日志打了「🗑 已清理」，但
    rag_outline 仍列出该文档，且 rag/documents/ 下的 .md 永久堆积。
    """
    store = _FakeStore()
    engine, kb = _make_engine(tmp_path, store)
    engine.ingest(force=False)

    md_dir = engine.markdown_dir
    cached = [f for f in os.listdir(md_dir) if f.endswith(".md")]
    assert cached, "首次同步应生成 Markdown 缓存"

    os.remove(str(kb / "a.txt"))
    engine.ingest(force=False)

    assert store.deleted_md_paths, "必须按 md_path 删除文档元数据"
    remaining = [f for f in os.listdir(md_dir) if f.endswith(".md")]
    assert "a.txt" not in " ".join(remaining)
    # 缓存文件数必须减少，而不是「删了又被 Step 7 写回来」
    assert len(remaining) == len(cached) - 1, (
        f"缓存未被清理: {cached} -> {remaining}"
    )


def test_content_change_with_preserved_mtime_is_re_synced(tmp_path):
    """mtime+size 都没变但内容变了（cp -p / 解压 / git checkout）必须重新同步。

    只看 mtime+size 会**永久**漏同步：每次都被判为「未修改」而跳过。
    """
    store = _FakeStore()
    engine, kb = _make_engine(tmp_path, store)
    engine.ingest(force=False)
    before = list(store.records)
    assert before, "首次同步应写入切片"

    target = kb / "a.txt"
    old_stat = os.stat(target)
    target.write_text("完全不同的新内容 " * 20, encoding="utf-8")
    # 伪造「mtime 与 size 均未变」：内容换了，签名没变
    os.utime(target, (old_stat.st_atime, old_stat.st_mtime))

    stats = engine.ingest(force=False)
    assert stats["files_extracted"] >= 1, "内容变了却被判为未修改而跳过"


def test_tiny_document_is_discarded_as_junk(tmp_path):
    """正文 ≤100 字的碎片文档（抽取失败残渣 / 图片附件上的零散信息）不应切片入库。

    与「空文件」不同：它能成功解析出一些文本，但内容过短无检索价值——
    入库前直接丢弃整个文件（不建 digest、不切片、不写向量），避免垃圾切片
    污染检索命中。正常长度的文档仍应正常入库。
    """
    store = _FakeStore()
    engine, kb = _make_engine(tmp_path, store)
    # KB 里再加一个 <100 字的微型文档（碎片垃圾）
    (kb / "junk.txt").write_text("图片上的零星说明文字。")
    engine.ingest(force=False)

    sources = {r["source"] for r in store.records}
    assert any("a.txt" in s for s in sources), "正常文档应入库"
    assert not any("junk.txt" in s for s in sources), "≤100 字碎片不应切片入库"


def test_manifest_survives_corruption(tmp_path):
    """If the manifest file is corrupted, ingest must treat it as empty and
    process all files (rather than crashing)."""
    store = _FakeStore()
    engine, kb = _make_engine(tmp_path, store)
    # Write a corrupted manifest before ingest
    rag_dir = engine._rag_dir
    os.makedirs(rag_dir, exist_ok=True)
    (open(os.path.join(rag_dir, "manifest.json"), "w")).write("corrupted{{{invalid json")

    stats = engine.ingest(force=False)
    assert stats["files_extracted"] == 1  # processed because manifest treated as empty
    assert stats["files_skipped"] == 0
    # After successful ingest, manifest must be valid
    manifest = _load_manifest(engine._rag_dir)
    assert len(manifest) == 1


def test_clear_removes_manifest(tmp_path):
    """clear() must delete the manifest so the next ingest starts fresh."""
    store = _FakeStore()
    engine, kb = _make_engine(tmp_path, store)
    engine.ingest(force=False)
    assert os.path.exists(_manifest_path(engine._rag_dir))

    engine.clear()
    assert not os.path.exists(_manifest_path(engine._rag_dir))


def test_empty_kb_cleans_up_deleted_files(tmp_path):
    """If all files are removed from the KB, ingest must clean up all old
    vectors and manifest entries."""
    store = _FakeStore()
    engine, kb = _make_engine(tmp_path, store)
    engine.ingest(force=False)
    assert store.count() > 0

    # Remove all files
    for f in kb.iterdir():
        if f.is_file():
            os.remove(str(f))

    stats = engine.ingest(force=False)
    assert stats["files_found"] == 0
    assert stats["files_deleted"] == 1
    # Manifest must be empty
    manifest = _load_manifest(engine._rag_dir)
    assert len(manifest) == 0


# ---------------------------------------------------------------------------
# Regression: agent's VectorStore must see table created by sync worker
# ---------------------------------------------------------------------------

def test_vector_store_rechecks_table_after_external_creation(tmp_path, vector_store_factory):
    """Regression: VectorStore must re-check table existence on every call.

    Previously _ensure_table() cached "table doesn't exist" via _table_checked
    flag. When SyncKnowledgeWorker (separate RAGEngine) later created the
    table, the agent's VectorStore still returned None forever → empty search
    results despite data being on disk.
    """
    persist = str(tmp_path / "vectors")

    # Agent's VectorStore, created at app startup BEFORE any sync.
    agent_store = vector_store_factory(persist)
    agent_store._ensure_initialized()
    # Table doesn't exist yet → _ensure_table returns None
    assert agent_store._ensure_table() is None

    # SyncKnowledgeWorker creates a SEPARATE VectorStore pointing to the same
    # dir and writes data (creates the table on disk).
    sync_store = vector_store_factory(persist)
    sync_store._ensure_initialized()
    sync_store.add([{
        "source": "doc1.md",
        "chunk_index": 0,
        "text": "hello world",
    }])
    assert sync_store.count() == 1
    sync_store.close()

    # Agent's VectorStore must now see the table (re-check, not cached None).
    table = agent_store._ensure_table()
    assert table is not None, "agent_store should see table created by sync_store"
    assert agent_store.count() == 1
    # Search must return results, not empty.
    results = agent_store.search("hello", top_k=5)
    assert len(results) >= 1
    assert results[0]["text"] == "hello world"
    agent_store.close()


def test_rag_engine_search_sees_data_written_by_another_engine(tmp_path, rag_engine_factory):
    """End-to-end regression: two RAGEngine instances pointing to the same
    workspace. The first (agent's, created at startup) must find data written
    by the second (SyncKnowledgeWorker) without needing a restart.
    """
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "note.txt").write_text(
        "LanceDB is a columnar vector database designed for efficient similarity "
        "search and AI applications at scale with strong performance")

    # Agent's engine, created at startup (table doesn't exist yet).
    agent_engine = rag_engine_factory(str(tmp_path), str(kb))
    # Trigger lazy init by calling search (returns empty, table not yet created).
    results_before = agent_engine.search("database", top_k=3)
    assert results_before == []

    # SyncKnowledgeWorker's engine writes data.
    sync_engine = rag_engine_factory(str(tmp_path), str(kb))
    sync_engine.ingest(force=False)
    sync_engine.close()

    # Agent's engine must now find the data WITHOUT restart.
    results_after = agent_engine.search("database", top_k=3)
    assert len(results_after) >= 1
    assert "LanceDB" in results_after[0]["text"]
    agent_engine.close()


def test_incremental_sync_visible_to_existing_engine(tmp_path, rag_engine_factory):
    """增量同步后，已存在的 agent 引擎应立即看到新数据，无需重启。

    场景：agent 引擎先同步 doc1 并搜索（打开表），然后 sync 引擎追加 doc2，
    agent 引擎立即搜索 doc2 内容应能命中。验证 LanceDB 的表对象不缓存数据，
    agent 持有的旧表对象能看到新写入的行。
    """
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "doc1.txt").write_text(
        "First document about Python basics covering variables, functions, loops, "
        "classes, and object-oriented programming concepts in depth")

    # Agent 的引擎：首次同步 + 触发一次搜索（让 VectorStore 打开表）
    agent_engine = rag_engine_factory(str(tmp_path), str(kb))
    agent_engine.ingest(force=False)
    r1 = agent_engine.search("Python", top_k=3)
    assert len(r1) >= 1, "首次同步后应能搜到 doc1"

    # 添加新文件
    (kb / "doc2.txt").write_text(
        "Second document about JavaScript and web development covering events, the "
        "DOM, async patterns, closures, and popular frontend frameworks")

    # SyncKnowledgeWorker 的引擎：增量同步
    sync_engine = rag_engine_factory(str(tmp_path), str(kb))
    stats = sync_engine.ingest(force=False)
    assert stats["files_extracted"] == 1, "应增量处理 doc2"
    assert stats["files_skipped"] == 1, "应跳过 doc1"
    sync_engine.close()

    # 模拟 UI _on_sync_finished 中的 reload_rag() 调用：
    # 刷新 agent 引擎的表句柄，让下一次搜索看到新写入的数据。
    # LanceDB 表对象在打开时获取版本快照，不刷新会看到旧数据。
    agent_engine.reload()

    # Agent 的引擎立即搜索新内容（不重启、不重建）
    r2 = agent_engine.search("JavaScript", top_k=3)
    assert len(r2) >= 1, "增量同步后应立即搜到 doc2"
    assert "JavaScript" in r2[0]["text"]

    # 旧内容仍可搜到
    r3 = agent_engine.search("Python", top_k=3)
    assert len(r3) >= 1, "旧内容仍应可搜到"
    agent_engine.close()
