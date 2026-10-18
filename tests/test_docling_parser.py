"""Tests for the docling unified parser (graceful fallback behaviour).

PDF 现在由 docling 全权接管（文本 / 扫描件 / 图片重统一走 docling，其 OCR
后端默认 ocrmac）。旧实现里「扫描件 / 图片重 PDF 转交 PyMuPDF + RapidOCR」
的分流已移除——见 rag/docling_parser.py。
"""
from __future__ import annotations

import pytest


def _fake_converter(markdown: str):
    """构造一个返回固定 Markdown 的假 DocumentConverter。"""

    class _FakeDoc:
        def export_to_markdown(self):
            return markdown

    class _FakeResult:
        document = _FakeDoc()

    class _FakeConverter:
        def convert(self, path):
            return _FakeResult()

    return _FakeConverter


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
    assert dp.is_supported(".epub")
    assert not dp.is_supported(".txt")
    assert not dp.is_supported(".csv")


def test_extract_markdown_scan_pdf_handled_by_docling(monkeypatch, tmp_path):
    """扫描件 PDF 不再转交 PyMuPDF，而是交给 docling（含 ocrmac OCR）。"""
    from agent.rag import docling_parser as dp

    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")

    monkeypatch.setattr(dp, "is_docling_available", lambda: True)
    monkeypatch.setattr(dp, "_get_converter", lambda: _fake_converter("# OCR'd scan content")())
    # 不再有「扫描件 → 返回 None 转交 PyMuPDF」的分流
    assert dp.extract_markdown(str(pdf)) == "# OCR'd scan content"


def test_extract_markdown_converts_via_docling(monkeypatch, tmp_path):
    """When docling is available, return Markdown (text or scanned alike)."""
    from agent.rag import docling_parser as dp

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")

    monkeypatch.setattr(dp, "is_docling_available", lambda: True)
    monkeypatch.setattr(dp, "_get_converter", lambda: _fake_converter("# Title\n\nSome content.")())

    md = dp.extract_markdown(str(pdf))
    assert md == "# Title\n\nSome content."


def test_extract_markdown_converter_init_failure(monkeypatch, tmp_path):
    """If converter init fails, return None (fallback to PyMuPDF path)."""
    from agent.rag import docling_parser as dp

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")

    monkeypatch.setattr(dp, "is_docling_available", lambda: True)
    monkeypatch.setattr(dp, "_get_converter", lambda: None)

    assert dp.extract_markdown(str(pdf)) is None
