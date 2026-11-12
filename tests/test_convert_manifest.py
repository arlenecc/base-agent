"""转换工具的文件级增量记录（ConvertManifest）测试。

场景对齐实际需求：批量转换中途崩溃/被终止后，重跑只处理未完成的文件；
源文件改动、输出被删、上次失败都要重新转换。
"""
import json
import os

from agent.ui.convert_tool_dialog import (
    _STATE_FILENAME,
    ConvertManifest,
)


def _mk(path, text="hello"):
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_skip_after_success(tmp_path):
    src = _mk(tmp_path / "a.txt")
    out = _mk(tmp_path / "a.md")
    m = ConvertManifest(str(tmp_path))
    assert not m.is_done("a.txt", src, out)
    m.mark_ok("a.txt", src, out)
    assert m.is_done("a.txt", src, out)


def test_record_persists_across_instances(tmp_path):
    src = _mk(tmp_path / "b.txt")
    out = _mk(tmp_path / "b.md")
    ConvertManifest(str(tmp_path)).mark_ok("b.txt", src, out)
    # 模拟重启：新实例从磁盘读取记录
    assert ConvertManifest(str(tmp_path)).is_done("b.txt", src, out)


def test_source_change_invalidates(tmp_path):
    src = _mk(tmp_path / "c.txt", "v1")
    out = _mk(tmp_path / "c.md")
    m = ConvertManifest(str(tmp_path))
    m.mark_ok("c.txt", src, out)
    # 源文件内容/大小变化 → 必须重新转换
    with open(src, "w", encoding="utf-8") as f:
        f.write("v2-longer-content")
    os.utime(src, None)
    assert not m.is_done("c.txt", src, out)


def test_output_deleted_invalidates(tmp_path):
    src = _mk(tmp_path / "d.txt")
    out = _mk(tmp_path / "d.md")
    m = ConvertManifest(str(tmp_path))
    m.mark_ok("d.txt", src, out)
    os.remove(out)
    assert not m.is_done("d.txt", src, out)


def test_failed_record_retries(tmp_path):
    src = _mk(tmp_path / "e.txt")
    out = _mk(tmp_path / "e.md")
    m = ConvertManifest(str(tmp_path))
    m.mark_failed("e.txt", src, "OCR 失败")
    assert not m.is_done("e.txt", src, out)


def test_empty_content_failure_retries(tmp_path):
    src = _mk(tmp_path / "f.txt")
    out = _mk(tmp_path / "f.md")
    m = ConvertManifest(str(tmp_path))
    m.mark_failed("f.txt", src, "提取内容为空")
    assert not m.is_done("f.txt", src, out)


def test_corrupt_state_file_is_ignored(tmp_path):
    (tmp_path / _STATE_FILENAME).write_text("{not-json", encoding="utf-8")
    m = ConvertManifest(str(tmp_path))          # 不应抛异常
    src = _mk(tmp_path / "g.txt")
    out = _mk(tmp_path / "g.md")
    assert not m.is_done("g.txt", src, out)
    m.mark_ok("g.txt", src, out)                # 损坏后仍能正常重建记录
    assert ConvertManifest(str(tmp_path)).is_done("g.txt", src, out)


def test_state_file_is_valid_json(tmp_path):
    src = _mk(tmp_path / "h.txt")
    out = _mk(tmp_path / "h.md")
    ConvertManifest(str(tmp_path)).mark_ok("h.txt", src, out)
    data = json.loads((tmp_path / _STATE_FILENAME).read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert data["files"]["h.txt"]["ok"] is True
    assert data["files"]["h.txt"]["out_size"] == os.path.getsize(out)
