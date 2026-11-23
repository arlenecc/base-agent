"""每 4 小时 raw_session 增量事实抽取 + 辅助/主模型降级测试。"""
import os
import sys
import zipfile
from types import SimpleNamespace

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from agent.memory_v2 import (
    MemoryLayout, pending_fact_sessions, mark_fact_session_done,
)


def test_pending_sessions_md_and_zip(tmp_path):
    """当天 md 与归档 zip 都能被增量扫描，游标推进后不重复。"""
    lay = MemoryLayout(str(tmp_path))
    lay.ensure()
    open(lay.full_path("2026-11-23"), "w", encoding="utf-8").write(
        "## [T1] 2026-11-23 10:00:00\n### USER\n我喜欢 Python。\n"
        "### ASSISTANT_RESPONSE\n已记录。\n### TOOLS\n噪音")
    with zipfile.ZipFile(lay.zip_path("2026-11-22"), "w") as zf:
        zf.writestr("1122.md", "### USER\n用 Milvus。\n### ASSISTANT_RESPONSE\n好的。")

    items = pending_fact_sessions(lay)
    keys = {it["key"] for it in items}
    assert "2026-11-23.md" in keys
    assert any(k.startswith("2026-11-22.zip:") for k in keys)
    assert all("噪音" not in it["text"] for it in items)  # TOOLS 段已剥离

    mark_fact_session_done(lay, "2026-11-23.md")
    assert "2026-11-23.md" not in {it["key"] for it in pending_fact_sessions(lay)}


def test_config_fields():
    """4 小时任务的配置开关与周期字段存在且默认正确。"""
    from agent.config import AgentConfig
    cfg = AgentConfig()
    assert cfg.memory_fact_extract_enabled is True
    assert cfg.memory_fact_extract_interval_hours == 4
    assert cfg.memory_turn_fact_extract_enabled is False
    assert "memory_fact_extract_enabled" in AgentConfig._BOOL_FIELDS
    assert "memory_fact_extract_interval_hours" in AgentConfig._INT_FIELDS
    assert "memory_turn_fact_extract_enabled" in AgentConfig._BOOL_FIELDS
