"""Tests for the per-page OCR test path (``rag/ocr_tester.py``).

Only the routing / dispatch logic is exercised — every real backend
(ocrmac / RapidOCR / OvisOCR2 VLM) is monkey-patched out so the suite runs
without models, ONNX runtimes or a GPU.
"""
import sys

import pytest

sys.path.insert(0, "src")

from agent.rag import ocr_tester as ot  # noqa: E402


@pytest.fixture
def png(tmp_path):
    from PIL import Image
    p = tmp_path / "page.png"
    Image.new("RGB", (240, 120), "white").save(p)
    return str(p)


# ---------------------------------------------------------------------------
# 文件判定
# ---------------------------------------------------------------------------

def test_supported_suffixes(tmp_path):
    assert ot.is_supported("a.pdf")
    assert ot.is_supported("a.PNG")
    assert ot.is_supported("scan.tiff")
    assert not ot.is_supported("a.docx")
    assert ot.is_pdf("a.pdf") and not ot.is_pdf("a.png")


def test_probe_pages_image(png):
    assert ot.probe_pages(png) == 1


def test_engine_status_has_all_engines():
    status = ot.engine_status()
    assert set(status) == set(ot.ENGINE_IDS)
    # 混合模式依赖 Vision + VLM：任一不可用则它不可用
    if status[ot.APPLE] or status[ot.OVIS]:
        assert status[ot.HYBRID]
    else:
        assert status[ot.HYBRID] == ""


def test_rejects_bad_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        ot.recognize_file(str(tmp_path / "nope.pdf"), ot.APPLE)
    txt = tmp_path / "a.txt"
    txt.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError):
        ot.recognize_file(str(txt), ot.APPLE)
    with pytest.raises(ValueError):
        ot.recognize_file(str(txt), "no-such-engine")


# ---------------------------------------------------------------------------
# 图片分支
# ---------------------------------------------------------------------------

def test_image_goes_through_normalization(png, monkeypatch):
    """图片必须走统一归一化（preprocess_for_ocr）后再送引擎。"""
    calls = []
    monkeypatch.setattr(ot, "_normalize",
                        lambda img, pre, bin_: calls.append((pre, bin_)) or img)
    monkeypatch.setattr(ot, "_plain_engine_text",
                        lambda img, eng, no, name: ("recognized", ""))
    res = ot.recognize_file(png, ot.APPLE)
    assert len(res) == 1 and res[0].markdown == "recognized"
    assert calls == [(True, True)]       # 默认开启预处理与二值化


def test_image_ovis_uses_vlm(png, monkeypatch):
    import agent.rag.ovis_ocr_parser as ovis

    monkeypatch.setattr(ovis, "_vlm_ocr_pil", lambda img, out_dir="": "# md")
    res = ot.recognize_file(png, ot.OVIS)
    assert res[0].markdown == "# md" and res[0].error == ""


def test_image_hybrid_rejected(png, monkeypatch):
    """混合模式依赖页面表格检测，图片直接给出可解释的提示。"""
    monkeypatch.setattr(ot, "_plain_engine_text",
                        lambda *a, **k: pytest.fail("不应调用引擎"))
    res = ot.recognize_file(png, ot.HYBRID)
    assert res[0].markdown == "" and "仅支持 PDF" in res[0].error


# ---------------------------------------------------------------------------
# PDF 分支
# ---------------------------------------------------------------------------

@pytest.fixture
def scanned_pdf(tmp_path):
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "mix.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "text layer " * 20, fontsize=12)
    doc.new_page()                       # 空白页：无文本层也无图像
    doc.save(str(path))
    doc.close()
    return str(path)


def test_pdf_text_layer_extracted_directly(scanned_pdf, monkeypatch):
    """文本型页用 PyMuPDF4LLM 转 Markdown（不跑 OCR）。"""
    monkeypatch.setattr(ot, "_plain_engine_text",
                        lambda *a, **k: pytest.fail("文本页不应调用 OCR"))
    res = ot.recognize_file(scanned_pdf, ot.RAPID)
    assert res[0].kind == "text"
    # 原文被 PyMuPDF4LLM 保留（库不可用时回退 get_text 同样保留）
    assert "text layer" in res[0].markdown
    assert res[1].kind == "blank" and res[1].markdown == ""


def test_pdf_scanned_page_uses_engine(tmp_path, monkeypatch):
    """扫描页（无文本层但有图像）才真正走引擎。"""
    fitz = pytest.importorskip("fitz")
    from PIL import Image
    scan = tmp_path / "scan.png"
    Image.new("RGB", (120, 60), "white").save(scan)
    path = tmp_path / "mix2.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 100), "text layer " * 20, fontsize=12)
    doc.new_page().insert_image(fitz.Rect(0, 0, 200, 200), filename=str(scan))
    doc.save(str(path))
    doc.close()

    seen = []
    monkeypatch.setattr(ot, "_render_page_bgr", lambda page: "IMG")
    monkeypatch.setattr(ot, "_normalize", lambda img, p, b: img)
    monkeypatch.setattr(ot, "_plain_engine_text",
                        lambda img, eng, no, name: seen.append((img, eng)) or ("ocr", ""))
    res = ot.recognize_file(str(path), ot.APPLE)
    assert seen == [("IMG", ot.APPLE)]
    assert res[1].kind == "image" and res[1].markdown == "ocr"


def test_cancel_stops_before_next_page(scanned_pdf, monkeypatch):
    monkeypatch.setattr(ot, "_plain_engine_text", lambda *a, **k: ("x", ""))
    res = ot.recognize_file(scanned_pdf, ot.RAPID, cancelled=lambda: True)
    assert res == []
