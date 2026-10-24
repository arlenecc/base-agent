"""同步日志的分阶段耗时：每文件一行 + 全局阶段汇总（2026-10-24）。

背景：同步日志此前只有「文件处理完成（N 切片）」，看不出解析/清洗/切片/
向量化/入库哪段是卡点。向量化与入库是**按 200 切片批量跨文件**做的，
因此批次耗时需要按切片数摊回各文件——本文件锁定这套行为。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from conftest import FakeEmbeddingFunction  # noqa: E402


def _make_kb(tmp_path, n_files: int = 3):
    kb = tmp_path / "kb"
    kb.mkdir()
    for i in range(n_files):
        body = " ".join(
            f"第{i}章第{s}段：这是用于同步耗时验证的正文内容，包含足够长度以便切分。" * 6
            for s in range(30)
        )
        (kb / f"book_{i}.txt").write_text(body)
    return kb


def test_per_file_timing_line_logged(tmp_path):
    """每个文件入库完成后，日志里必须有一行含全部阶段的耗时。"""
    from agent.rag.engine import RAGEngine
    kb = _make_kb(tmp_path)
    lines: list = []

    eng = RAGEngine(
        workspace=str(tmp_path / "ws"),
        knowledge_base=str(kb),
        embedding_function=FakeEmbeddingFunction(),
    )
    try:
        eng.ingest(force=False, log_callback=lambda m: lines.append(str(m)))
    finally:
        eng.close()

    for i in range(3):
        name = f"book_{i}.txt"
        row = next((l for l in lines if "⏱" in l and name in l), None)
        assert row is not None, f"缺少 {name} 的分阶段耗时日志行"
        for stage in ("解析", "清洗", "切片", "向量化", "入库", "合计"):
            assert stage in row
        assert "切片" in row          # 切片数量也应在行内


def test_stage_summary_line_logged(tmp_path):
    """整轮同步结束后必须输出阶段耗时汇总（定位整体卡点）。"""
    from agent.rag.engine import RAGEngine
    kb = _make_kb(tmp_path, n_files=2)
    lines: list = []

    eng = RAGEngine(
        workspace=str(tmp_path / "ws"),
        knowledge_base=str(kb),
        embedding_function=FakeEmbeddingFunction(),
    )
    try:
        eng.ingest(force=False, log_callback=lambda m: lines.append(str(m)))
    finally:
        eng.close()

    summary = next((l for l in lines if "阶段耗时汇总" in l), None)
    assert summary is not None
    for stage in ("解析", "清洗", "切片", "向量化", "入库"):
        assert stage in summary


def test_on_batch_timing_callback_compat():
    """_call_on_batch 对两参数/三参数回调都要兼容（新回调才拿得到耗时）。"""
    from agent.rag.vector_store import _call_on_batch
    got2 = {}
    got3 = {}

    def cb2(n, total):
        got2["n"] = n
        got2["total"] = total

    def cb3(n, total, timing):
        got3.update(n=n, total=total, timing=timing)

    _call_on_batch(cb2, 5, 200, 1.5, 0.2)
    assert got2 == {"n": 5, "total": 200}

    _call_on_batch(cb3, 5, 200, 1.5, 0.2)
    assert got3["n"] == 5 and got3["total"] == 200
    assert got3["timing"] == {"embed": 1.5, "store": 0.2}

    _call_on_batch(None, 1, 1, 0.0, 0.0)   # 无回调不报错
