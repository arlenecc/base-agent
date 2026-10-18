"""docling PDF OCR 后端选择 + 扫描路由移除的测试。

背景：
- PDF 改由 docling 全权接管，其 OCR 后端默认 ocrmac（Apple Vision），
  不可用时回退 RapidOCR（见 rag/docling_parser._choose_ocr_options）。
- 旧的 PyMuPDF 扫描件 / 图片重检测函数（_is_scan_pdf / _pdf_is_image_heavy）
  已删除——docling 现在统一处理所有 PDF。
"""
import builtins
import inspect
import sys


def _ocrmac_importable() -> bool:
    try:
        import ocrmac  # noqa: F401

        return True
    except ImportError:
        return False


def test_choose_ocr_options_prefers_ocrmac(monkeypatch):
    """macOS 且 ocrmac 可装 → OcrMacOptions（中文语种 + accurate）；否则回退 RapidOcrOptions。"""
    from agent.rag import docling_parser as dp

    # 保证 ocrmac 真实可装（否则本测试机器的行为就是回退分支）
    opts = dp._choose_ocr_options()
    if sys.platform == "darwin" and _ocrmac_importable():
        assert type(opts).__name__ == "OcrMacOptions", (
            f"macOS 应优先 ocrmac，实际 {type(opts).__name__}"
        )
        # 中文为主（简+繁），兼容英文
        assert opts.lang == ["zh-Hans", "zh-Hant", "en-US"]
        # recognition 固定 accurate：macOS fast 档不支持中文，用 fast 处理
        # 中文 PDF 会直接抛 ValueError（ocrmac 校验 language_preference）。
        assert opts.recognition == "accurate"
        assert opts.framework == "vision"
    else:
        assert type(opts).__name__ == "RapidOcrOptions"


def test_choose_ocr_options_falls_back_when_ocrmac_missing(monkeypatch):
    """ocrmac 不可用时必须回退 RapidOcrOptions（不崩、不静默 None）。"""
    from agent.rag import docling_parser as dp

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "ocrmac" or name.startswith("ocrmac."):
            raise ImportError("simulated: ocrmac not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    opts = dp._choose_ocr_options()
    assert type(opts).__name__ == "RapidOcrOptions"


def test_scan_routing_functions_removed():
    """走 docling 全权接管后，旧的 PyMuPDF 扫描检测函数应已删除（不再直接 import fitz）。"""
    from agent.rag import docling_parser as dp

    assert not hasattr(dp, "_is_scan_pdf")
    assert not hasattr(dp, "_pdf_is_image_heavy")
    # extract_markdown 不应再引用这两个被删函数
    src = inspect.getsource(dp.extract_markdown)
    assert "_is_scan_pdf" not in src
    assert "_pdf_is_image_heavy" not in src
