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
# 1. docling 单实例 + 转换串行化（2026-10-23 回退多实例池后的契约）
# ---------------------------------------------------------------------------

def test_docling_converts_are_serialized(tmp_path, monkeypatch):
    """回归：`extract_markdown` 必须全程持 `_converter_lock` 串行 convert。

    背景：DocumentConverter 非线程安全，多实例池方案（2026-10-22）把 16GB
    机器打到 20+GB 系统崩溃，已回退为「单实例 + 全程持锁串行」。本测试锁住
    串行契约——并发调用不得出现两个 convert 重叠。
    """
    import threading
    import time

    from agent.rag import docling_parser as dp

    active = 0
    peak = 0
    lk = threading.Lock()

    class _FakeConv:
        def convert(self, path):
            nonlocal active, peak
            with lk:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            with lk:
                active -= 1
            raise RuntimeError("stop-after-overlap-check")

    monkeypatch.setattr(dp, "is_docling_available", lambda: True)
    monkeypatch.setattr(dp, "_get_converter", lambda: _FakeConv())
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    errs = []

    def _run():
        try:
            dp.extract_markdown(str(pdf))
        except Exception as e:  # convert 抛错会被吞并返回 None
            errs.append(e)

    ts = [threading.Thread(target=_run) for _ in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=5)
    assert peak == 1                            # 三个调用严格串行
    assert errs == []                           # 异常被吞，回退链继续


def test_singleton_release_clears_converter(monkeypatch):
    """release_converter() 必须真正清掉单例引用（否则 ~4.6GB 常驻）。"""
    from agent.rag import docling_parser as dp

    monkeypatch.setattr(dp, "is_docling_available", lambda: True)
    monkeypatch.setattr(dp, "_get_converter",
                        lambda: "fake-converter-sentinel")
    # 直接验证释放语义：置位后 release_converter 清空
    dp._converter = "fake-converter-sentinel"
    dp.release_converter()
    assert dp._converter is None


def test_ingest_does_not_inject_concurrency_config(rag_engine_factory, tmp_path):
    """回归：多实例池的并发配置注入已移除（rag_parse_* 字段不存在）。"""
    from agent.config import AgentConfig
    cfg = AgentConfig()
    assert not hasattr(cfg, "rag_parse_parallel")
    assert not hasattr(cfg, "rag_parse_large_mb")
    from agent.rag import docling_parser as dp
    assert not hasattr(dp, "set_docling_concurrency")
    assert not hasattr(dp, "_ConverterPool")


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


# ---------------------------------------------------------------------------
# 6. 回退链熔断 + 释放（2026-10-23：防主引擎系统性失败时逐棒加载全驻留）
# ---------------------------------------------------------------------------

def _patch_engines(monkeypatch, behaviors, released):
    """按引擎名注入行为（函数或异常），并记录释放调用。"""
    from agent.rag import docling_parser, engine as eng_mod
    from agent.rag import marker_parser, ovis_ocr_parser

    # 纯文本兜底也桩掉：测试文件是假 PDF，PyMuPDF 打开会抛无关异常
    monkeypatch.setattr(eng_mod, "extract_text", lambda p: "plain-text")

    def _mk(name, fn):
        def _call(path):
            if isinstance(fn, Exception):
                raise fn
            return fn(path)
        return _call

    monkeypatch.setattr(docling_parser, "extract_markdown",
                        _mk("docling", behaviors.get("docling", lambda p: None)))
    monkeypatch.setattr(marker_parser, "extract_markdown",
                        _mk("marker", behaviors.get("marker", lambda p: None)))
    monkeypatch.setattr(ovis_ocr_parser, "extract_markdown",
                        _mk("ovis", behaviors.get("ovis", lambda p: None)))
    for mod, fname in ((docling_parser, "release_converter"),
                       (marker_parser, "release_converter"),
                       (ovis_ocr_parser, "release_model")):
        def _rel(_fname=fname):
            released.append(_fname)
        monkeypatch.setattr(mod, fname, _rel)


def test_engine_breaker_trips_after_limit_and_releases(tmp_path, monkeypatch):
    from agent.rag import engine as eng_mod

    released = []
    _patch_engines(monkeypatch,
                   {"docling": RuntimeError("boom")}, released)
    eng_mod.reset_engine_fail_streak()
    f = tmp_path / "x.pdf"
    f.write_bytes(b"%PDF-1.4")
    # 前 3 次：docling 每次抛 → 依次尝试 marker/ovis（都返回 None）→ 纯文本
    for _ in range(3):
        md, is_md = eng_mod._extract_with_docling(str(f))
        assert is_md is False
    # 第 3 次失败后 docling 熔断并释放
    assert released == ["release_converter"]
    # 第 4 次：docling 被跳过（不再尝试），直接 marker→ovis→纯文本
    released.clear()
    md, is_md = eng_mod._extract_with_docling(str(f))
    assert is_md is False
    assert eng_mod._engine_fail_streak["docling"] >= 3


def test_engine_none_return_does_not_trip_breaker(tmp_path, monkeypatch):
    """返回 None（格式不适用）不得熔断——否则 txt 语料会把能解析 PDF 的
    引擎也熔掉，后续 PDF 被悄悄降级成纯文本。"""
    from agent.rag import engine as eng_mod

    released = []
    _patch_engines(monkeypatch, {}, released)   # 三个引擎全部返回 None
    eng_mod.reset_engine_fail_streak()
    f = tmp_path / "a.txt"
    f.write_text("纯文本内容", encoding="utf-8")
    for _ in range(10):
        md, is_md = eng_mod._extract_with_docling(str(f))
        assert is_md is False
    assert released == []                        # 从未熔断/释放
    assert not any(v >= 3 for v in eng_mod._engine_fail_streak.values())


def test_engine_success_resets_streak(monkeypatch):
    from agent.rag import engine as eng_mod

    _patch_engines(monkeypatch, {}, [])
    eng_mod.reset_engine_fail_streak()
    eng_mod._engine_fail_streak["docling"] = 2     # 模拟已失败 2 次
    monkeypatch.setattr("agent.rag.docling_parser.extract_markdown",
                        lambda p: "# ok")
    md, is_md = eng_mod._extract_with_docling("/kb/z.pdf")
    assert is_md is True and md == "# ok"
    assert eng_mod._engine_fail_streak["docling"] == 0   # 成功即复位


def test_reset_engine_fail_streak_clears():
    from agent.rag import engine as eng_mod

    eng_mod._engine_fail_streak = {"docling": 9}
    eng_mod.reset_engine_fail_streak()
    assert eng_mod._engine_fail_streak == {}
