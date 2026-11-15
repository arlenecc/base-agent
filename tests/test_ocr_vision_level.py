"""Apple Vision OCR 识别级别（accurate / fast）配置与注入测试。

不加载真实 ocrmac / Vision：用假模块断言传给 OCR 的 recognition_level，
并验证 docling 侧与兜底路径共享同一份配置。
"""
from __future__ import annotations

import json
import sys
import types

sys.path.insert(0, "src")

from agent.config import AgentConfig
from agent.rag import docling_parser, ocr_engine


def test_vision_level_normalization():
    try:
        assert ocr_engine.set_ocrmac_recognition_level("fast") == "fast"
        assert ocr_engine.set_ocrmac_recognition_level("FAST") == "fast"
        # 非法值回退精确模式，不能把 True/None 之类传给 Vision
        assert ocr_engine.set_ocrmac_recognition_level("precise") == "accurate"
        assert ocr_engine.set_ocrmac_recognition_level(None) == "accurate"
    finally:
        ocr_engine.set_ocrmac_recognition_level("accurate")


def test_ocr_with_ocrmac_passes_configured_level(monkeypatch):
    calls = []

    class _FakeOCR:
        def __init__(self, image, language_preference=None,
                     recognition_level=None):
            calls.append((language_preference, recognition_level))

        def recognize(self):
            return [("识别结果", 1.0, (0, 0, 1, 1))]

    fake_pkg = types.ModuleType("ocrmac")
    fake_sub = types.ModuleType("ocrmac.ocrmac")
    fake_sub.OCR = _FakeOCR
    fake_pkg.ocrmac = fake_sub
    monkeypatch.setitem(sys.modules, "ocrmac", fake_pkg)
    monkeypatch.setitem(sys.modules, "ocrmac.ocrmac", fake_sub)
    monkeypatch.setattr(ocr_engine, "ocrmac_available", lambda: True)

    import numpy as np

    try:
        ocr_engine.set_ocrmac_recognition_level("fast")
        text = ocr_engine.ocr_with_ocrmac(np.zeros((8, 8, 3), dtype=np.uint8))
        assert text == "识别结果"
        assert calls[-1] == (["zh-Hans", "en-US"], "fast")
    finally:
        ocr_engine.set_ocrmac_recognition_level("accurate")


def test_docling_and_fallback_share_vision_level():
    try:
        docling_parser.set_ocr_scale_config(
            base=2.0, dynamic=False, table=2.5, vision_level="fast")
        assert docling_parser._ocr_vision_level == "fast"
        assert ocr_engine._OCRMAC_RECOGNITION_LEVEL == "fast"
    finally:
        docling_parser.set_ocr_scale_config(
            base=2.0, dynamic=False, table=2.5, vision_level="accurate")


def test_config_migrates_invalid_vision_level(tmp_path):
    path = tmp_path / "config.json"

    path.write_text(json.dumps({"rag_ocr_vision_level": "FAST"}),
                    encoding="utf-8")
    assert AgentConfig.load(str(path)).rag_ocr_vision_level == "fast"

    path.write_text(json.dumps({"rag_ocr_vision_level": "nonsense"}),
                    encoding="utf-8")
    assert AgentConfig.load(str(path)).rag_ocr_vision_level == "accurate"

    path.write_text(json.dumps({}), encoding="utf-8")
    assert AgentConfig.load(str(path)).rag_ocr_vision_level == "accurate"
