"""内容去重（B 方案，2026-10-22 用户定版）的测试。

两条防线：
1. 预扫描去重：同一本书在不同子目录/改名，只解析入库第一个副本，
   其余记 duplicate_of 跳过（采样指纹判别，≤4MB 全量 / 大文件头中尾
   三段 64KB）；
2. 同步收尾按 content_hash 清理：存量重复副本（去重功能上线前已各自
   入库的）删向量 + documents 表记录 + 标记 duplicate_of；owner 缺失/
   解析失败的僵尸副本提升为正常条目（下轮重新入库）。

不依赖真实嵌入模型：cleanup 单测用假 store；集成测试用 conftest 的
rag_engine_factory（FakeEmbeddingFunction）。
"""
from __future__ import annotations

import os

import pytest

from agent.rag.engine import _load_manifest


# ---------------------------------------------------------------------------
# _dedup_manifest_by_content 单元测试（假 store）
# ---------------------------------------------------------------------------

class _FakeStore:
    def __init__(self):
        self.deleted_sources = []
        self.deleted_docs = []

    def delete_by_source(self, source):
        self.deleted_sources.append(source)
        return 1

    def delete_document(self, doc_name):
        self.deleted_docs.append(doc_name)


def _make_engine(tmp_path):
    from agent.rag.engine import RAGEngine
    return RAGEngine(
        workspace=str(tmp_path / "ws"),
        knowledge_base=str(tmp_path / "kb"),
        embedding_function=lambda x: [[0.0] * 8 for _ in x],
    )


def _entry(ch="h1", **kw):
    e = {"mtime": 1.0, "size": 10, "source_hash": "s1",
         "content_hash": ch, "chunk_count": 2}
    e.update(kw)
    return e


def test_dedup_marks_and_deletes_duplicate(tmp_path):
    eng = _make_engine(tmp_path)
    fake = _FakeStore()
    eng._get_store = lambda: fake
    kb = str(tmp_path / "kb")
    a, b = os.path.join(kb, "a", "book.pdf"), os.path.join(kb, "b", "book.pdf")
    m = {a: _entry(), b: _entry()}
    removed = eng._dedup_manifest_by_content(m)
    assert removed == 1
    # owner = 字典序最小（a/book.pdf），且不带 duplicate_of
    assert "duplicate_of" not in m[a]
    assert m[b]["duplicate_of"] == a
    assert fake.deleted_sources == [b]
    # documents 表兼容删除（rel_path + basename 两种 id）
    assert os.path.join("b", "book.pdf") in fake.deleted_docs
    assert "book.pdf" in fake.deleted_docs


def test_dedup_is_idempotent(tmp_path):
    eng = _make_engine(tmp_path)
    fake = _FakeStore()
    eng._get_store = lambda: fake
    kb = str(tmp_path / "kb")
    a, b = os.path.join(kb, "a.pdf"), os.path.join(kb, "b.pdf")
    m = {a: _entry(), b: _entry()}
    assert eng._dedup_manifest_by_content(m) == 1
    fake.deleted_sources.clear()
    assert eng._dedup_manifest_by_content(m) == 0     # 已标记，不重复删
    assert fake.deleted_sources == []


def test_dedup_promotes_orphan_duplicate(tmp_path):
    """owner 没了 → 曾真实入库的副本摘标记转正（向量还在，内容继续可检索）。"""
    eng = _make_engine(tmp_path)
    eng._get_store = lambda: _FakeStore()
    kb = str(tmp_path / "kb")
    b = os.path.join(kb, "b", "book.pdf")
    m = {b: _entry(duplicate_of=os.path.join(kb, "a", "book.pdf"))}
    removed = eng._dedup_manifest_by_content(m)
    assert removed == 0
    assert "duplicate_of" not in m[b]        # 已摘标记
    assert m[b]["content_hash"] == "h1"      # 条目保留（向量还在）


def test_dedup_drops_never_ingested_orphan(tmp_path):
    """向量已被清理的副本（chunk_count=0）+ owner 没了 → 整条删除，
    下轮同步重新完整入库（只摘标记会被当「未修改」跳过，内容永远回不来）。

    判据是 chunk_count（向量是否还在）而非 content_hash：清理副本时向量被
    删、content_hash 仍保留，只看 content_hash 会漏判这种情况。
    """
    eng = _make_engine(tmp_path)
    eng._get_store = lambda: _FakeStore()
    kb = str(tmp_path / "kb")
    b = os.path.join(kb, "b", "book.pdf")
    m = {b: _entry(chunk_count=0, duplicate_of=os.path.join(kb, "a", "book.pdf"))}
    eng._dedup_manifest_by_content(m)
    assert b not in m


def test_dedup_promotes_when_owner_has_no_content_hash(tmp_path):
    """owner 解析失败（无 content_hash）→ 有向量的副本摘标记，不跟着消失。"""
    eng = _make_engine(tmp_path)
    eng._get_store = lambda: _FakeStore()
    kb = str(tmp_path / "kb")
    a, b = os.path.join(kb, "a.pdf"), os.path.join(kb, "b.pdf")
    m = {a: _entry(content_hash=None), b: _entry(duplicate_of=a)}
    eng._dedup_manifest_by_content(m)
    assert "duplicate_of" not in m[b]


def test_dedup_noop_for_unique_content(tmp_path):
    eng = _make_engine(tmp_path)
    eng._get_store = lambda: _FakeStore()
    kb = str(tmp_path / "kb")
    m = {os.path.join(kb, "a.pdf"): _entry("h1"),
         os.path.join(kb, "b.pdf"): _entry("h2")}
    assert eng._dedup_manifest_by_content(m) == 0


def test_dedup_cleans_marked_entry_with_leftover_vectors(tmp_path):
    """已标记副本但还带着旧向量（force 场景/历史双份入库）→ 也要清。"""
    eng = _make_engine(tmp_path)
    fake = _FakeStore()
    eng._get_store = lambda: fake
    kb = str(tmp_path / "kb")
    a, b = os.path.join(kb, "a.pdf"), os.path.join(kb, "b.pdf")
    m = {a: _entry(chunk_count=3),
         b: _entry(chunk_count=3, duplicate_of=a)}
    assert eng._dedup_manifest_by_content(m) == 1
    assert fake.deleted_sources == [b]
    assert m[b]["chunk_count"] == 0
    fake.deleted_sources.clear()
    assert eng._dedup_manifest_by_content(m) == 0     # 幂等


# ---------------------------------------------------------------------------
# 集成测试：完整 ingest（FakeEmbeddingFunction）
# ---------------------------------------------------------------------------

@pytest.fixture
def dedup_env(tmp_path, rag_engine_factory):
    kb = tmp_path / "kb"
    (kb / "a").mkdir(parents=True)
    (kb / "b").mkdir(parents=True)
    ws = tmp_path / "ws"
    eng = rag_engine_factory(workspace=str(ws), knowledge_base=str(kb))
    return eng, kb, ws


def test_same_book_in_two_dirs_ingested_once(dedup_env):
    eng, kb, ws = dedup_env
    content = "# 同一本书\n\n" + "这是用于去重测试的正文内容。" * 50
    a, b = kb / "a" / "book.md", kb / "b" / "book.md"
    a.write_text(content, encoding="utf-8")
    b.write_text(content, encoding="utf-8")
    stats = eng.ingest()
    assert stats["files_duplicated"] == 1
    manifest = _load_manifest(eng._rag_dir)
    ea, eb = manifest[str(a)], manifest[str(b)]
    # 字典序最小者为 owner（all_files 已排序，确定性）
    assert "duplicate_of" not in ea
    assert eb["duplicate_of"] == str(a)
    # 检索只召回 owner 一个来源（无双倍噪声）
    results = eng.search("去重测试")
    assert results
    assert {r["source"] for r in results} == {str(a)}


def test_duplicate_promoted_after_owner_deleted(dedup_env):
    eng, kb, ws = dedup_env
    content = "# 同一本书\n\n" + "删除主文件后副本应转正入库。" * 50
    a, b = kb / "a" / "book.md", kb / "b" / "book.md"
    a.write_text(content, encoding="utf-8")
    b.write_text(content, encoding="utf-8")
    eng.ingest()
    a.unlink()                                # 删掉 owner
    stats = eng.ingest()
    # 副本提升：重新解析入库（不再被记成副本）
    assert stats["files_duplicated"] == 0
    manifest = _load_manifest(eng._rag_dir)
    assert "duplicate_of" not in manifest[str(b)]
    results = eng.search("副本转正")
    assert results
    assert {r["source"] for r in results} == {str(b)}


def test_copy_into_new_folder_not_reparsed(dedup_env):
    """用户实测场景（2026-10-22）：已入库的文件，新建文件夹拷贝一份进去，
    再次同步时**不重新解析**——新副本把 owner 让给「已在库」的文件。

    根因：seen_fp 原先边遍历边播种，拷贝路径排序在原文件之前时会抢先成为
    owner 被解析。现在已入库指纹预播种，新副本必让位。
    """
    eng, kb, ws = dedup_env
    content = "# 已在库的书\n\n" + "拷贝到新文件夹后不应再次解析入库。" * 50
    a = kb / "book.md"
    a.write_text(content, encoding="utf-8")
    eng.ingest()                              # 首次入库
    store = eng._get_store()
    before = store._ensure_table().count_rows()
    # 新建文件夹，拷贝一份进去（新路径、新 mtime）
    newdir = kb / "a-copy"
    newdir.mkdir()
    b = newdir / "book.md"
    b.write_text(content, encoding="utf-8")
    stats = eng.ingest()
    # 新副本被记成重复，零解析零入库
    assert stats["files_duplicated"] == 1
    assert stats["files_extracted"] == 0
    assert store._ensure_table().count_rows() == before
    manifest = _load_manifest(eng._rag_dir)
    assert manifest[str(b)]["duplicate_of"] == str(a)
    assert "duplicate_of" not in manifest[str(a)]
    # 检索仍只召回原路径一个来源
    results = eng.search("再次解析")
    assert {r["source"] for r in results} == {str(a)}


def test_copy_into_new_folder_not_reparsed_even_with_force(dedup_env):
    """force（强制全量）下同样不重复解析副本：owner 让给已在库文件，
    只重解析 owner 本身。"""
    eng, kb, ws = dedup_env
    content = "# 强制全量\n\n" + "force 模式下副本也不应被单独解析。" * 50
    a = kb / "book.md"
    a.write_text(content, encoding="utf-8")
    eng.ingest()
    newdir = kb / "a-copy2"
    newdir.mkdir()
    b = newdir / "book.md"
    b.write_text(content, encoding="utf-8")
    stats = eng.ingest(force=True)
    assert stats["files_duplicated"] == 1     # 副本被去重
    assert stats["files_extracted"] == 1      # 只有 owner 被重解析
    manifest = _load_manifest(eng._rag_dir)
    assert manifest[str(b)]["duplicate_of"] == str(a)
    # 无双份向量：owner 的旧向量被替换，副本从未入库
    results = eng.search("单独解析")
    assert {r["source"] for r in results} == {str(a)}


def test_legacy_duplicate_vectors_cleaned_by_content_hash(dedup_env):
    """模拟存量：两个副本都已入库（两套向量）→ 收尾清理删副本的向量。"""
    eng, kb, ws = dedup_env
    content = "# 存量重复\n\n" + "上线去重功能之前两个目录都各自入了库。" * 50
    a, b = kb / "a" / "book.md", kb / "b" / "book.md"
    a.write_text(content, encoding="utf-8")
    b.write_text(content, encoding="utf-8")
    # 手动构造「存量」manifest：两条条目同 content_hash、都无 duplicate_of。
    # source_hash 用真实指纹——否则再 ingest 时会被判「已修改」重新解析，
    # 测不到「已标记副本走跳过路径」这条路径。
    from agent.rag.engine import (_content_hash, _file_signature,
                                  _save_manifest, _source_fingerprint)
    ch = _content_hash(content)
    manifest = {}
    for p in (a, b):
        sig = _file_signature(str(p))
        manifest[str(p)] = {
            **sig,
            "source_hash": _source_fingerprint(str(p), sig["size"]),
            "content_hash": ch,
            "chunk_count": 3,
        }
    # 给两个来源都灌入向量（模拟上线前的双份入库）
    store = eng._get_store()
    for p in (a, b):
        store.add([
            {"source": str(p), "chunk_index": i, "text": f"{content[:40]} #{i}"}
            for i in range(3)
        ])
    assert store._ensure_table().count_rows() == 6
    removed = eng._dedup_manifest_by_content(manifest)
    assert removed == 1
    assert store._ensure_table().count_rows() == 3   # 副本的 3 条向量已删
    assert manifest[str(b)]["duplicate_of"] == str(a)
    # 落盘后的 manifest 保持标记；再跑一次 ingest 不会重新解析副本
    _save_manifest(eng._rag_dir, manifest)
    stats = eng.ingest()
    assert stats["files_duplicated"] == 0           # 已标记副本走跳过路径
    assert store._ensure_table().count_rows() == 3
