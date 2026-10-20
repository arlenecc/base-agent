"""Tests for OCR engine construction (CoreML EP injection) and pre-processing.

These avoid invoking the real RapidOCR model load / real cv2 resize so they
run on any platform (including CI on Intel x86).  The CoreML EP *injection*
mechanism (monkey-patching RapidOCR's OrtInferSession EP list) is exercised
directly against the real class method.
"""
import sys
from unittest import mock

import pytest

sys.path.insert(0, "src")

from agent.rag import ocr_engine as oe  # noqa: E402


class _Run:
    """Stand-in for subprocess.CompletedProcess (real attribute, not class attr)."""

    def __init__(self, stdout):
        self.stdout = stdout


class _FakeSession:
    """Minimal stand-in for rapidocr's OrtInferSession (no real model load)."""

    def _check_cuda(self):
        return False

    def _check_dml(self):
        return False


# ---------------------------------------------------------------------------
# CoreML EP injection mechanism
# ---------------------------------------------------------------------------


def _ep_list_mechanism(cls):
    """返回该 rapidocr 布局使用的注入机制：'method' 或 'factory'。"""
    return "method" if hasattr(cls, "_get_ep_list") else "factory"


def _patched_marker(cls):
    """当前注入路径上的「已 patch」标记（两种布局通用）。"""
    if hasattr(cls, "_get_ep_list"):
        return getattr(cls._get_ep_list, "_coreml_patched", False)
    mod = sys.modules.get(getattr(cls, "__module__", ""))
    return bool(mod) and getattr(
        getattr(mod, "InferenceSession", None), "_coreml_patched", False
    )


def test_patch_injects_coreml_ep_first():
    cls = oe._ort_infer_session_class()
    assert cls is not None, "rapidocr OrtInferSession 类未找到"
    try:
        ok = oe._patch_coreml_ep(cls)
        assert ok, "patch 未生效"
        if hasattr(cls, "_get_ep_list"):
            ep_list = cls._get_ep_list(_FakeSession())
            assert ep_list, "EP 列表为空"
            first = ep_list[0]
            first_name = first[0] if isinstance(first, (list, tuple)) else first
            assert first_name == oe._COREML_EP, "CoreML 必须排在最前"
            assert any(
                (ep[0] if isinstance(ep, (list, tuple)) else ep) == oe._CPU_EP
                for ep in ep_list
            ), "CPU 必须保留作回退"
        else:
            # rapidocr 1.2.x 布局：EP 列表在 __init__ 内联构建，
            # 注入点是模块级 InferenceSession 工厂（行为由
            # test_factory_patch_injects_coreml 单独验证）。
            assert _patched_marker(cls), "factory 未被标记为已 patch"
    finally:
        oe._unpatch_coreml_ep(cls)
        assert not _patched_marker(cls), "unpatch 后必须恢复原始 EP 列表/工厂"


def test_patch_is_idempotent():
    """重复 patch 不重复包装（两种布局都不能套两层）。"""
    cls = oe._ort_infer_session_class()
    assert cls is not None
    try:
        assert oe._patch_coreml_ep(cls) is True
        assert oe._patch_coreml_ep(cls) is True
        if hasattr(cls, "_get_ep_list"):
            # 套两层会把 CoreML 插两次——幂等性必须保证只插一次
            eps = cls._get_ep_list(_FakeSession())
            coreml_count = sum(
                1 for ep in eps
                if (ep[0] if isinstance(ep, (list, tuple)) else ep) == oe._COREML_EP
            )
            assert coreml_count == 1, "CoreML 被重复注入"
    finally:
        oe._unpatch_coreml_ep(cls)


def test_factory_patch_injects_coreml(monkeypatch):
    """factory 路径（rapidocr 1.2.x 布局）：验证 providers 注入与恢复。

    用一个注册进 sys.modules 的假模块驱动 ``_session_module``，
    不触碰真实 rapidocr / onnxruntime。
    """
    import types

    mod = types.ModuleType("rapidocr_fake_infer_engine")
    calls = []

    class _FakeIS:
        def __init__(self, *a, **kw):
            calls.append(kw)

    mod.InferenceSession = _FakeIS
    monkeypatch.setitem(sys.modules, "rapidocr_fake_infer_engine", mod)

    class _FakeSessionCls:
        pass

    _FakeSessionCls.__module__ = "rapidocr_fake_infer_engine"

    try:
        assert oe._patch_inference_session_factory(_FakeSessionCls) is True
        # 幂等
        assert oe._patch_inference_session_factory(_FakeSessionCls) is True
        mod.InferenceSession("model.onnx", providers=[("CPUExecutionProvider", {})])
        providers = calls[-1]["providers"]
        assert providers[0][0] == oe._COREML_EP, "CoreML 必须排在最前"
        assert any(p[0] == oe._CPU_EP for p in providers), "CPU 必须保留作回退"
    finally:
        oe._unpatch_coreml_ep(_FakeSessionCls)
        assert mod.InferenceSession is _FakeIS, "unpatch 后必须恢复原工厂"


def test_factory_patch_rejects_foreign_module():
    """非 rapidocr 模块的类不得被 patch（避免误伤无关代码）。"""
    import types

    class _Local:
        pass

    assert oe._patch_inference_session_factory(_Local) is False
    assert oe._unpatch_coreml_ep(_Local) is None


def test_create_ocr_engine_injects_coreml_on_apple_silicon(monkeypatch):
    """Apple Silicon + ort>=1.31 → 引擎构造时 CoreML EP 被注入到 EP 列表。"""
    monkeypatch.setattr(oe, "_apple_silicon_cache", True)
    monkeypatch.setattr(oe, "ort_version", lambda: (1, 31, 0))

    captured = {}

    class FakeRapidOCR:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr("rapidocr_onnxruntime.RapidOCR", FakeRapidOCR)
    cls = oe._ort_infer_session_class()
    oe._patch_coreml_ep(cls)  # 模拟注入
    try:
        eng = oe.create_ocr_engine(use_coreml=True)
        assert isinstance(eng, FakeRapidOCR)
        # det_limit_side_len 应被我们放大（默认 736 → 1024），提升小字识别
        assert captured.get("det_limit_side_len") == oe._DET_LIMIT_SIDE_LEN
        assert oe._DET_LIMIT_SIDE_LEN > 736
    finally:
        oe._unpatch_coreml_ep(cls)


def test_create_ocr_engine_no_coreml_when_disabled(monkeypatch):
    """use_coreml=False → 不注入、不报错，且仍带 det_limit_side_len 参数。"""
    captured = {}

    class FakeRapidOCR:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr("rapidocr_onnxruntime.RapidOCR", FakeRapidOCR)
    cls = oe._ort_infer_session_class()
    if hasattr(cls, "_get_ep_list"):
        was_patched = getattr(cls._get_ep_list, "_coreml_patched", False)
    else:
        mod = sys.modules.get(getattr(cls, "__module__", ""))
        was_patched = getattr(
            getattr(mod, "InferenceSession", None), "_coreml_patched", False
        )
    eng = oe.create_ocr_engine(use_coreml=False)
    assert isinstance(eng, FakeRapidOCR)
    assert captured.get("det_limit_side_len") == oe._DET_LIMIT_SIDE_LEN
    # 关闭时不得注入 CoreML 到全局类/工厂
    if hasattr(cls, "_get_ep_list"):
        now_patched = getattr(cls._get_ep_list, "_coreml_patched", False)
    else:
        mod = sys.modules.get(getattr(cls, "__module__", ""))
        now_patched = getattr(
            getattr(mod, "InferenceSession", None), "_coreml_patched", False
        )
    assert now_patched == was_patched


def test_create_ocr_engine_survives_kwargs_incompatibility(monkeypatch):
    """rapidocr 1.2.x 布局：det_* kwargs 触发上游 KeyError 时必须降级建成。

    复现真实缺陷：只传 det_limit_side_len 而不传 det_model_path 时，
    rapidocr 1.2.3 的 UpdateParameters 抛 KeyError('model_path')，
    旧实现让 OCR 引擎永远建不起来（表现为「提示未安装」）。
    """
    attempts = []

    class FakeRapidOCR12x:
        """模拟 rapidocr 1.2.x：缺 det_model_path 就 KeyError。"""

        def __init__(self, **kw):
            attempts.append(kw)
            if any(k.startswith("det_") for k in kw) and "det_model_path" not in kw:
                raise KeyError("model_path")

    monkeypatch.setattr("rapidocr_onnxruntime.RapidOCR", FakeRapidOCR12x)
    eng = oe.create_ocr_engine(use_coreml=False)
    assert isinstance(eng, FakeRapidOCR12x)
    # 第一次带完整调优参数失败，第二次补 det_model_path 成功
    assert len(attempts) >= 2
    assert attempts[-1].get("det_model_path") == ""
    assert attempts[-1].get("det_limit_side_len") == oe._DET_LIMIT_SIDE_LEN


def test_create_ocr_engine_falls_back_to_bare(monkeypatch):
    """kwargs 全不兼容时退到裸构造——OCR 可用性优先于分辨率调优。"""

    class FakeRapidOCRStrict:
        def __init__(self, **kw):
            if kw:
                raise TypeError("unexpected keyword argument")

    monkeypatch.setattr("rapidocr_onnxruntime.RapidOCR", FakeRapidOCRStrict)
    eng = oe.create_ocr_engine(use_coreml=False)
    assert isinstance(eng, FakeRapidOCRStrict)


def test_coreml_reason_refuses_on_non_apple():
    usable, reason = oe.coreml_reason()
    # 在 CI / Intel 机器上应直接拒绝（不要求 onnxruntime 版本）
    if not oe.is_apple_silicon():
        assert usable is False
        assert "Apple" in reason or "非" in reason


# ---------------------------------------------------------------------------
# Apple Silicon detection
# ---------------------------------------------------------------------------


def test_is_apple_silicon_current():
    # 原生 arm64 → True
    with mock.patch.object(oe.platform, "system", return_value="Darwin"), \
         mock.patch.object(oe.platform, "machine", return_value="arm64"):
        assert oe._detect_apple_silicon() is True

    # Rosetta：machine 报 x86_64，但 sysctl 真实 CPU 是 Apple M1
    with mock.patch.object(oe.platform, "system", return_value="Darwin"), \
         mock.patch.object(oe.platform, "machine", return_value="x86_64"), \
         mock.patch.object(oe.subprocess, "run", return_value=_Run("Apple M1")):
        assert oe._detect_apple_silicon() is True

    # 真 Intel：sysctl 不含 Apple
    with mock.patch.object(oe.platform, "system", return_value="Darwin"), \
         mock.patch.object(oe.platform, "machine", return_value="x86_64"), \
         mock.patch.object(oe.subprocess, "run", return_value=_Run("Intel(R) Core(TM) i5")):
        assert oe._detect_apple_silicon() is False

    # 非 Darwin
    with mock.patch.object(oe.platform, "system", return_value="Linux"):
        assert oe._detect_apple_silicon() is False


# ---------------------------------------------------------------------------
# Pre-processing robustness (best-effort; must never raise)
# ---------------------------------------------------------------------------


def test_preprocess_returns_input_on_none():
    assert oe.preprocess_for_ocr(None) is None


def test_preprocess_safe_on_arbitrary_input():
    # 任意非图像输入不得抛错，原样返回
    obj = "not an image"
    assert oe.preprocess_for_ocr(obj) is obj


def test_preprocess_handles_2d_and_4ch():
    import numpy as np
    gray = np.zeros((10, 10), dtype=np.uint8)
    rgba = np.zeros((10, 10, 4), dtype=np.uint8)
    # 这些路径在 cv2 可用时走转换；cv2 损坏时走降级返回原图——都不应抛错
    r1 = oe.preprocess_for_ocr(gray)
    r2 = oe.preprocess_for_ocr(rgba)
    assert r1 is not None
    assert r2 is not None


def test_render_dpi_capped_and_long_side_bounded(monkeypatch):
    """渲染 dpi 不超过 _MAX_RENDER_DPI，且长边反推后位图不超 _MAX_LONG_SIDE。

    旧语义是 clamp 到 [150, 400]（短边凑 1600px，不足 150 时抬到 150）。现改为
    「短边目标 1600 与长边上限 1500 取更严格者 + 硬上限 150」：竖版大开本不再
    被下限抬回 150 去渲染 8.9MP 再缩掉，而是按长边反推一次渲染到位。走 OCR 的
    都是扫描件（位图嵌入），实测 60/100/150dpi 归一到长边 1500 后识别结果一致。
    """

    class Rect:
        def __init__(self, w, h):
            self.width, self.height = w, h

    class Page:
        def __init__(self, w, h):
            self.rect = Rect(w, h)

    # 小页面：为凑够短边会解出极高 dpi → 必须被硬上限封住
    assert oe.render_dpi_for_page(Page(72, 72)) == oe._MAX_RENDER_DPI

    # 超大页：按长边反推，不再被下限抬到 150（抬上去只是渲染一张要被缩掉的图）
    tiny_dpi = oe.render_dpi_for_page(Page(72 * 100, 72 * 100))
    assert 0 < tiny_dpi < oe._MAX_RENDER_DPI

    # 任意页面：dpi 恒 ≤ 上限，且按该 dpi 渲染后长边不超过 _MAX_LONG_SIDE
    for w, h in ((1151, 1788), (595, 842), (612, 792), (2000, 3000), (300, 400)):
        dpi = oe.render_dpi_for_page(Page(w, h))
        assert dpi <= oe._MAX_RENDER_DPI
        long_px = max(w, h) / 72.0 * dpi
        assert long_px <= oe._MAX_LONG_SIDE + 1, (w, h, dpi, long_px)
