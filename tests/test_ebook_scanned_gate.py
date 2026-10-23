"""扫描版 EPUB 串行闸门测试（2026-10-23 用户定版）。

约束（用户原话）：整体 EPUB 解析仍是 2 路并发——一个扫描版 + 一个文字版
可以同时跑；但**不能有两个扫描版同时解析**（OCR 段串行）。

实现：`ebook_parser._SCANNED_EPUB_GATE`（Semaphore(1)）只闸「扫描版探测
之后的 OCR」段；探测在锁外（毫秒级），文字版 EPUB 完全不碰这把锁。
"""
from __future__ import annotations

import os
import threading
import time

import pytest

from agent.rag import ebook_parser as eb


class _FakeBook:
    """最小 ebooklib 替身：无 spine/toc → 文字版路径快速走完返回 None。"""

    toc = []
    spine = []

    def get_metadata(self, *a, **kw):
        return []

    def get_item_with_id(self, idref):
        return None

    def get_items(self):
        return []


def _patch_base(monkeypatch):
    monkeypatch.setattr(eb, "_read_epub", lambda path: _FakeBook())
    monkeypatch.setattr(eb, "is_pandoc_available", lambda: True)


def test_two_scanned_epubs_are_serialized(tmp_path, monkeypatch):
    """两个扫描版不得同时解析：OCR 段峰值并发必须为 1。"""
    active = 0
    peak = 0
    lk = threading.Lock()

    def _slow_scan(path, book=None):
        nonlocal active, peak
        with lk:
            active += 1
            peak = max(peak, active)
        time.sleep(0.15)
        with lk:
            active -= 1
        return "扫描文本"

    _patch_base(monkeypatch)
    monkeypatch.setattr(eb, "_detect_scanned_epub", lambda path, book=None: True)
    monkeypatch.setattr(eb, "extract_scanned_epub", _slow_scan)

    f1, f2 = tmp_path / "a.epub", tmp_path / "b.epub"
    for f in (f1, f2):
        f.write_bytes(b"PK\x03\x04fake")
    results = {}
    ts = [threading.Thread(target=lambda p=f: results.update(
        {str(p): eb.extract_ebook(str(p))})) for f in (f1, f2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=10)
    assert peak == 1                                   # 严格串行
    assert len(results) == 2
    assert all(v and v.get("chapters") == ["扫描文本"] for v in results.values())


def test_text_epub_not_blocked_by_scanned(tmp_path, monkeypatch):
    """文字版不得被扫描版的闸门堵住：扫描版还在跑时文字版就已返回。"""
    scan_started = threading.Event()
    release_scan = threading.Event()

    def _block_scan(path, book=None):
        scan_started.set()
        release_scan.wait(timeout=10)
        return "扫描文本"

    _patch_base(monkeypatch)
    monkeypatch.setattr(eb, "extract_scanned_epub", _block_scan)
    monkeypatch.setattr(
        eb, "_detect_scanned_epub",
        lambda path, book=None: "scanned" in os.path.basename(path))

    scanned = tmp_path / "scanned.epub"
    text = tmp_path / "text.epub"
    for f in (scanned, text):
        f.write_bytes(b"PK\x03\x04fake")

    out = {}
    t_scan = threading.Thread(
        target=lambda: out.update(scan=eb.extract_ebook(str(scanned))))
    t_text = threading.Thread(
        target=lambda: out.update(text=eb.extract_ebook(str(text))))
    t_scan.start()
    assert scan_started.wait(timeout=5)      # 扫描版已进入 OCR 段并持闸
    t_text.start()
    t_text.join(timeout=5)
    assert not t_text.is_alive()             # 文字版未被闸门堵住
    assert out.get("text") is None           # 假书无章节 → None（正常路径）
    release_scan.set()
    t_scan.join(timeout=5)
    assert out["scan"]["chapters"] == ["扫描文本"]


def test_two_text_epubs_run_concurrently(tmp_path, monkeypatch):
    """两个文字版互不串行（2 路并发保持）。"""
    active = 0
    peak = 0
    lk = threading.Lock()

    def _slow_text(path, book=None):
        nonlocal active, peak
        with lk:
            active += 1
            peak = max(peak, active)
        time.sleep(0.1)
        with lk:
            active -= 1
        return None                            # 文字版不走 OCR 段

    _patch_base(monkeypatch)
    monkeypatch.setattr(eb, "_detect_scanned_epub", lambda path, book=None: False)
    monkeypatch.setattr(eb, "extract_scanned_epub", _slow_text)

    f1, f2 = tmp_path / "t1.epub", tmp_path / "t2.epub"
    for f in (f1, f2):
        f.write_bytes(b"PK\x03\x04fake")
    ts = [threading.Thread(target=lambda p=f: eb.extract_ebook(str(p)))
          for f in (f1, f2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=10)
    # 文字版路径根本不调用 extract_scanned_epub（探测在锁外且为 False）
    assert peak == 0


def test_gate_released_on_scanned_failure(tmp_path, monkeypatch):
    """OCR 段抛异常时闸门必须释放（否则后续扫描版永久堵死）。"""
    def _boom(path, book=None):
        raise RuntimeError("ocr exploded")

    _patch_base(monkeypatch)
    monkeypatch.setattr(eb, "_detect_scanned_epub", lambda path, book=None: True)
    monkeypatch.setattr(eb, "extract_scanned_epub", _boom)

    f = tmp_path / "bad.epub"
    f.write_bytes(b"PK\x03\x04fake")
    # 异常被吞 → 落回常规路径（假书无章节）→ None，不抛出
    assert eb.extract_ebook(str(f)) is None
    # 闸门已释放：能立刻无阻塞拿到
    assert eb._SCANNED_EPUB_GATE.acquire(blocking=False)
    eb._SCANNED_EPUB_GATE.release()
