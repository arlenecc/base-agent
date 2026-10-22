"""Tests for the docling unified parser (graceful fallback behaviour)."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _fresh_converter_pool():
    """实例池会跨测试缓存旧工厂（monkeypatch 的假工厂）的实例——每个测试
    前后清空，避免上一个测试的假 converter 泄漏到下一个测试的断言里。"""
    from agent.rag import docling_parser as dp

    dp._pool.release_all()
    yield
    dp._pool.release_all()


def test_extract_markdown_returns_none_when_docling_missing(monkeypatch):
    """Without docling, extract_markdown returns None (caller falls back)."""
    from agent.rag import docling_parser as dp

    monkeypatch.setattr(dp, "is_docling_available", lambda: False)
    assert dp.extract_markdown("anything.pdf") is None


def test_extract_markdown_skips_unsupported_extension(monkeypatch):
    """Non-docling formats (txt/md/csv) return None."""
    from agent.rag import docling_parser as dp

    monkeypatch.setattr(dp, "is_docling_available", lambda: True)
    assert dp.extract_markdown("notes.txt") is None
    assert dp.extract_markdown("data.csv") is None


def test_is_supported_extensions():
    from agent.rag import docling_parser as dp

    assert dp.is_supported(".pdf")
    assert dp.is_supported(".docx")
    assert dp.is_supported(".xlsx")
    assert dp.is_supported(".pptx")
    # .epub 按设计**不**走 docling：ebook_parser（pandoc）秒级完成，
    # docling 处理 epub 要先构建 ~3-4GB converter，完全重复且慢得多。
    assert not dp.is_supported(".epub")
    assert not dp.is_supported(".txt")
    assert not dp.is_supported(".csv")


def test_extract_markdown_returns_none_when_converter_unavailable(monkeypatch, tmp_path):
    """converter 构建失败（返回 None）时 extract_markdown 返回 None（回退）。"""
    from agent.rag import docling_parser as dp

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")

    monkeypatch.setattr(dp, "is_docling_available", lambda: True)
    monkeypatch.setattr(dp, "_get_converter", lambda: None)
    assert dp.extract_markdown(str(pdf)) is None


def test_extract_markdown_converts_via_docling(monkeypatch, tmp_path):
    """When docling is available and the file is text-based, return Markdown."""
    from agent.rag import docling_parser as dp

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")

    class _FakeDoc:
        def export_to_markdown(self):
            return "# Title\n\nSome content."

    class _FakeResult:
        document = _FakeDoc()

    class _FakeConverter:
        def convert(self, path):
            return _FakeResult()

    monkeypatch.setattr(dp, "is_docling_available", lambda: True)
    monkeypatch.setattr(dp, "_get_converter", lambda: _FakeConverter())

    md = dp.extract_markdown(str(pdf))
    assert md == "# Title\n\nSome content."


def test_extract_markdown_converter_init_failure(monkeypatch, tmp_path):
    """If converter init fails, return None (fallback)."""
    from agent.rag import docling_parser as dp

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")

    monkeypatch.setattr(dp, "is_docling_available", lambda: True)
    monkeypatch.setattr(dp, "_get_converter", lambda: None)

    assert dp.extract_markdown(str(pdf)) is None


def test_extract_markdown_empty_output_returns_none(monkeypatch, tmp_path):
    """docling 返回空白 Markdown 时按失败处理（回退轻量解析器）。"""
    from agent.rag import docling_parser as dp

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")

    class _FakeDoc:
        def export_to_markdown(self):
            return "   \n  "

    class _FakeResult:
        document = _FakeDoc()

    class _FakeConverter:
        def convert(self, path):
            return _FakeResult()

    monkeypatch.setattr(dp, "is_docling_available", lambda: True)
    monkeypatch.setattr(dp, "_get_converter", lambda: _FakeConverter())
    assert dp.extract_markdown(str(pdf)) is None
