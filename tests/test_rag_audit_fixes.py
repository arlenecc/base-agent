"""2026-10-23 全链路审计修复的回归测试。

1. docling 转换器池：`release_all()` 期间有实例在租时不能把 _created 整池
   归零，否则空闲自动释放永久失效（4.6GB 常驻）。
2. `_cleanup_deleted_files`：删除失败的文件必须保留 manifest 条目（下次
   重试），不能被 _save_manifest(removed=…) 无条件 pop。
3. 内容去重：副本向量已被清理（chunk_count=0）而 owner 消失时，条目必须
   整条删除以便下轮重新入库；且无向量的副本不能当 owner 吞掉新文件。
4. ingest 结束无条件释放解析单例（agent 的 rag_ingest 路径不调 close）。
5. chunker 的 `_GapLocator`：与 `_gap_between` 结果一致且带游标前进。
"""
from __future__ import annotations

import os

import pytest


# ---------------------------------------------------------------------------
# 1. docling 转换器池引用计数
# ---------------------------------------------------------------------------

def test_pool_release_all_keeps_on_loan_instances_counted():
    from agent.rag.docling_parser import _ConverterPool

    pool = _ConverterPool(lambda: object(), max_size=2)
    # 借出一个实例
    with pool.lease(exclusive=False) as conv:
        assert conv is not None
        assert pool._created == 1
        # 此时整池释放（同步结束/取消路径）：只丢空闲实例，租借中的仍计数
        pool.release_all()
        assert pool._created == 1          # 不能归零
        assert pool._free == []
    # 归还后进入空闲栈，_created 仍为 1 → reaper 能看见它
    assert pool._created == 1
    assert len(pool._free) == 1


def test_pool_reap_still_works_after_release_all():
    """回归：旧实现 release_all 后 _created=0 → reap_if_idle 永久短路。"""
    import time

    from agent.rag.docling_parser import _ConverterPool

    pool = _ConverterPool(lambda: object(), max_size=2)
    with pool.lease(exclusive=False):
        pool.release_all()
    # 空闲超时后必须能释放（_last_used 回拨模拟 90s 空闲）
    pool._last_used = time.monotonic() - 999
    assert pool.reap_if_idle(90.0) == 1
    assert not pool.has_instances()


# ---------------------------------------------------------------------------
# 2. 删除失败的文件保留 manifest 条目
# ---------------------------------------------------------------------------

def test_cleanup_deleted_files_keeps_failed_entries(tmp_path, rag_engine_factory):
    eng = rag_engine_factory(workspace=str(tmp_path / "ws"),
                             knowledge_base=str(tmp_path / "kb"))

    class BoomStore:
        def __init__(self):
            self.deleted = []

        def delete_by_source(self, source):
            self.deleted.append(source)
            raise RuntimeError("lancedb exploded")

        def delete_document(self, *a, **kw):
            raise RuntimeError("lancedb exploded")

        def delete_document_by_md_path(self, *a, **kw):
            raise RuntimeError("lancedb exploded")

    boom = BoomStore()
    eng._get_store = lambda: boom
    gone = str(tmp_path / "kb" / "gone.pdf")
    old = {gone: {"mtime": 1.0, "size": 2, "source_hash": "s",
                  "content_hash": "c", "chunk_count": 1}}
    new = dict(old)
    from agent.config import AgentConfig
    eng._knowledge_base = str(tmp_path / "kb")
    stats = {"files_deleted": 0}
    import threading
    removed = eng._cleanup_deleted_files([], stats, threading.Lock(),
                                         old_manifest=old, new_manifest=new)
    # 删除失败 → 不进 removed（manifest 条目保留，下次同步重试）
    assert removed == set()
    assert gone in new


# ---------------------------------------------------------------------------
# 3. 去重：僵尸副本与 owner 选择
# ---------------------------------------------------------------------------

def _engine(tmp_path, rag_engine_factory):
    kb = tmp_path / "kb"
    kb.mkdir(exist_ok=True)
    return rag_engine_factory(workspace=str(tmp_path / "ws"),
                              knowledge_base=str(kb)), kb


def test_dup_without_vectors_promoted_when_owner_gone(tmp_path, rag_engine_factory):
    """副本向量已被清理（chunk_count=0）+ owner 消失 → 整条删除以便下轮重入。"""
    eng, kb = _engine(tmp_path, rag_engine_factory)
    owner = os.path.join(str(kb), "a.pdf")
    dup = os.path.join(str(kb), "b.pdf")
    m = {
        dup: {"mtime": 1.0, "size": 2, "source_hash": "s1",
              "content_hash": "h1", "chunk_count": 0, "duplicate_of": owner},
    }
    os.makedirs(os.path.dirname(dup), exist_ok=True)
    open(dup, "wb").write(b"x")            # 副本仍在磁盘（owner 不在）
    eng._dedup_manifest_by_content(m)
    assert dup not in m                    # 整条删除 → 下轮重新入库


def test_dup_with_vectors_kept_as_orphan(tmp_path, rag_engine_factory):
    """副本向量还在（chunk_count>0）+ owner 消失 → 摘标记转正，内容继续可检索。"""
    eng, kb = _engine(tmp_path, rag_engine_factory)
    owner = os.path.join(str(kb), "a.pdf")
    dup = os.path.join(str(kb), "b.pdf")
    m = {
        dup: {"mtime": 1.0, "size": 2, "source_hash": "s1",
              "content_hash": "h1", "chunk_count": 3, "duplicate_of": owner},
    }
    open(dup, "wb").write(b"x")
    eng._dedup_manifest_by_content(m)
    assert dup in m
    assert "duplicate_of" not in m[dup]


# ---------------------------------------------------------------------------
# 4. ingest 结束释放解析单例
# ---------------------------------------------------------------------------

def test_ingest_releases_parsers(tmp_path, rag_engine_factory):
    eng, kb = _engine(tmp_path, rag_engine_factory)
    (kb / "doc.md").write_text("# 标题\n\n正文内容。" * 20, encoding="utf-8")
    calls = []
    orig = type(eng).release_parsers

    def spy(self, *a, **kw):
        calls.append(1)
        return orig(self, *a, **kw)

    eng.release_parsers = lambda *a, **kw: spy(eng, *a, **kw)
    eng.ingest()
    assert len(calls) == 1                 # 正常结束也释放


def test_ingest_releases_parsers_on_error(tmp_path, rag_engine_factory):
    eng, kb = _engine(tmp_path, rag_engine_factory)
    calls = []
    orig = type(eng).release_parsers
    eng.release_parsers = lambda *a, **kw: (calls.append(1), orig(eng, *a, **kw))[1]
    # 强制 _ingest_locked 抛异常，验证 finally 里仍释放
    def boom(*a, **kw):
        raise RuntimeError("kaboom")
    eng._ingest_locked = boom
    with pytest.raises(RuntimeError):
        eng.ingest()
    assert calls


# ---------------------------------------------------------------------------
# 5. chunker 的 _GapLocator
# ---------------------------------------------------------------------------

def test_gap_locator_matches_gap_between():
    from agent.rag.chunker import _GapLocator, _gap_between

    src = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
    left, right = "第一段内容。", "第二段内容。"
    assert _GapLocator(src).gap(left, right, "|") == _gap_between(
        src, left, right, fallback="|")


def test_gap_locator_advances_cursor():
    from agent.rag.chunker import _GapLocator

    src = "AAA\n\nBBB\n\nCCC"
    loc = _GapLocator(src)
    assert loc.gap("AAA", "BBB", "|") == "\n\n"
    assert loc._pos > 0                    # 游标已前进（不再从头 find）
    assert loc.gap("BBB", "CCC", "|") == "\n\n"


def test_gap_locator_falls_back_when_not_found():
    from agent.rag.chunker import _GapLocator

    loc = _GapLocator("只有一段")
    assert loc.gap("没有的", "也没有", "|") == "|"
