"""Tests for document parsers (OCR engine pool lifecycle)."""
from __future__ import annotations

import sys
import threading
import time
import types

import pytest

from agent.rag import parsers


def _reset_ocr_state():
    """Reset the module-level OCR engine pool so tests start clean."""
    with parsers._OCR_ENGINE_COND:
        parsers._OCR_ENGINE_IDLE.clear()
        parsers._OCR_ENGINE_LIVE = 0
        parsers._OCR_ENGINE_GEN = 0
        parsers._OCR_RETRY_AFTER = 0.0
        parsers._OCR_POOL_SIZE = 2


def _fake_rapidocr_module():
    """Build a fake rapidocr_onnxruntime (1.x) module exposing RapidOCR()."""
    mod = types.ModuleType("rapidocr_onnxruntime")

    class _FakeRapidOCR:
        instances = 0

        # RapidOCR accepts config kwargs (create_ocr_engine passes
        # det_limit_side_len / det_limit_type).
        def __init__(self, **kwargs):
            _FakeRapidOCR.instances += 1

        def __call__(self, *args, **kwargs):
            return (None, None)

    mod.RapidOCR = _FakeRapidOCR
    return mod


def _fake_rapidocr3_module():
    """Build a fake rapidocr (3.x) module exposing RapidOCR(params=...)."""
    mod = types.ModuleType("rapidocr")

    class _FakeRapidOCR:
        instances = 0

        def __init__(self, params=None, config_path=None):
            _FakeRapidOCR.instances += 1

        def __call__(self, *args, **kwargs):
            return (None, None)

    mod.RapidOCR = _FakeRapidOCR
    return mod


def _install_fake_ocr(monkeypatch):
    """同时伪造 3.x 与 1.x 两个 rapidocr 模块。

    生产环境装有 rapidocr 3.x 时，只伪造 1.x 模块会让 _build_ocr_engine
    走到真实的 3.x 引擎（加载模型、跑预热）——测试因此既慢又不确定。
    这里把两个入口都换成 fake，无论环境装的是哪一代都能走同一路径。
    """
    mod3 = _fake_rapidocr3_module()
    mod1 = _fake_rapidocr_module()
    monkeypatch.setitem(sys.modules, "rapidocr", mod3)
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", mod1)
    return mod3


# ---------------------------------------------------------------------------
# 引擎池（2026-10-23：2 实例纯 CPU 并发）
# ---------------------------------------------------------------------------

def test_pool_hands_out_distinct_engines(monkeypatch):
    """池内可同时借出多个**不同**实例（并发 OCR 的前提）。"""
    _reset_ocr_state()
    mod = _install_fake_ocr(monkeypatch)

    e1, g1 = parsers._acquire_ocr_engine()
    e2, g2 = parsers._acquire_ocr_engine()
    assert e1 is not None and e2 is not None
    assert e1 is not e2                      # 两个独立实例
    assert mod.RapidOCR.instances == 2
    parsers._return_ocr_engine(e1, g1)
    parsers._return_ocr_engine(e2, g2)


def test_pool_exhaustion_times_out(monkeypatch):
    """池被占满后借用超时返回 None（不永久挂起）。"""
    _reset_ocr_state()
    _install_fake_ocr(monkeypatch)

    held = [parsers._acquire_ocr_engine() for _ in range(2)]
    assert all(e is not None for e, _ in held)
    t0 = time.monotonic()
    engine, _gen = parsers._acquire_ocr_engine(timeout=0.2)
    assert engine is None                    # 超时 → None
    assert time.monotonic() - t0 < 2.0       # 确实没等更久
    for e, g in held:
        parsers._return_ocr_engine(e, g)


def test_return_wakes_waiter(monkeypatch):
    """归还要唤醒等待中的借用者（条件变量，不是轮询）。"""
    _reset_ocr_state()
    _install_fake_ocr(monkeypatch)

    e1, g1 = parsers._acquire_ocr_engine()
    e2, g2 = parsers._acquire_ocr_engine()      # 池占满
    got = {}

    def _borrow():
        got["engine"], got["gen"] = parsers._acquire_ocr_engine(timeout=5)

    t = threading.Thread(target=_borrow)
    t.start()
    time.sleep(0.1)                             # 确保已在条件变量上等待
    parsers._return_ocr_engine(e1, g1)          # 归还 → 等待者立刻拿到
    t.join(timeout=5)
    assert got.get("engine") is e1              # 复用同一实例（空闲栈）
    parsers._return_ocr_engine(got["engine"], got["gen"])
    parsers._return_ocr_engine(e2, g2)


def test_release_drops_idle_only_and_recreates(monkeypatch):
    """release 只丢空闲实例；借出中的归还时按代数淘汰（不缩水池）。"""
    _reset_ocr_state()
    mod = _install_fake_ocr(monkeypatch)

    e1, g1 = parsers._acquire_ocr_engine()
    e2, g2 = parsers._acquire_ocr_engine()
    assert mod.RapidOCR.instances == 2
    parsers.release_ocr_engine()             # 两个都在借出中 → 无可释放
    assert parsers._OCR_ENGINE_IDLE == []
    assert parsers._OCR_ENGINE_LIVE == 2     # 计数不被整池归零
    # 借出中的实例归还 → 旧代淘汰（不回流、计数减一）
    parsers._return_ocr_engine(e1, g1)
    assert parsers._OCR_ENGINE_IDLE == []
    assert parsers._OCR_ENGINE_LIVE == 1
    # 下一次借用新建一个（池未满）
    e3, g3 = parsers._acquire_ocr_engine()
    assert e3 is not None and e3 is not e1
    assert mod.RapidOCR.instances == 3
    parsers._return_ocr_engine(e2, g2)
    parsers._return_ocr_engine(e3, g3)


def test_release_ocr_engine_when_not_loaded(monkeypatch):
    """Releasing when the pool was never used is a safe no-op."""
    _reset_ocr_state()
    parsers.release_ocr_engine()
    assert parsers._OCR_ENGINE_IDLE == []
    assert parsers._OCR_ENGINE_LIVE == 0


def test_acquire_returns_none_when_missing(monkeypatch):
    """Missing rapidocr dependency returns None (caller skips OCR)."""
    _reset_ocr_state()
    monkeypatch.delitem(sys.modules, "rapidocr", raising=False)
    monkeypatch.delitem(sys.modules, "rapidocr_onnxruntime", raising=False)
    import builtins
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name in ("rapidocr", "rapidocr_onnxruntime"):
            raise ImportError("no rapidocr")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    engine, _gen = parsers._acquire_ocr_engine()
    assert engine is None
    # 失败不占池位（下次还能重试）
    assert parsers._OCR_ENGINE_LIVE == 0


def test_pool_builds_cpu_ep_even_when_coreml_requested(monkeypatch):
    """引擎池固定纯 CPU EP——即使配置里请求了 CoreML（并发翻倍爬升）。"""
    _reset_ocr_state()
    mod3 = _fake_rapidocr3_module()
    monkeypatch.setitem(sys.modules, "rapidocr", mod3)
    monkeypatch.setattr(parsers, "_is_apple_silicon", lambda: True)
    monkeypatch.setattr(parsers, "_OCR_CONFIG",
                        types.SimpleNamespace(rag_ocr_coreml=True))
    seen = {}
    _orig_init = mod3.RapidOCR.__init__

    def _spy_init(self, params=None, config_path=None):
        seen["params"] = params
        _orig_init(self, params=params, config_path=config_path)

    monkeypatch.setattr(mod3.RapidOCR, "__init__", _spy_init)
    engine, _gen = parsers._acquire_ocr_engine()
    assert engine is not None
    assert seen["params"] is None             # 没有注入 CoreML 开关
    parsers._return_ocr_engine(engine, _gen)


def test_set_ocr_pool_size_shrink_drops_idle(monkeypatch):
    """调小池大小立刻丢弃多出来的空闲实例。"""
    _reset_ocr_state()
    _install_fake_ocr(monkeypatch)
    engines = [parsers._acquire_ocr_engine() for _ in range(2)]
    for e, g in engines:
        parsers._return_ocr_engine(e, g)
    assert len(parsers._OCR_ENGINE_IDLE) == 2
    parsers.set_ocr_pool_size(1)
    assert len(parsers._OCR_ENGINE_IDLE) == 1  # 多出的被丢弃
    assert parsers._OCR_POOL_SIZE == 1
    parsers.set_ocr_pool_size(2)


def test_set_ocr_config_injects_pool_size(monkeypatch):
    """set_ocr_config 把 rag_ocr_pool_size 注入引擎池。"""
    _reset_ocr_state()
    parsers.set_ocr_config(types.SimpleNamespace(
        rag_ocr_coreml=False, rag_ocr_pool_size=3))
    assert parsers._OCR_POOL_SIZE == 3
    parsers.set_ocr_pool_size(2)


def test_build_ocr_engine_prefers_rapidocr3(monkeypatch):
    """3.x 可用时优先构造它（params 传 use_coreml），不碰 1.x 兼容层。"""
    mod3 = _fake_rapidocr3_module()
    mod1 = _fake_rapidocr_module()
    monkeypatch.setitem(sys.modules, "rapidocr", mod3)
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", mod1)
    monkeypatch.setattr(parsers, "_is_apple_silicon", lambda: True)

    eng = parsers._build_ocr_engine()
    assert mod3.RapidOCR.instances == 1
    assert mod1.RapidOCR.instances == 0
    assert eng is not None


def test_build_ocr_engine_falls_back_to_legacy(monkeypatch):
    """3.x 缺失时回退 1.x（ocr_engine.create_ocr_engine 注入 CoreML）。"""
    monkeypatch.delitem(sys.modules, "rapidocr", raising=False)
    mod1 = _fake_rapidocr_module()
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", mod1)

    import builtins
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "rapidocr":
            raise ImportError("no rapidocr 3.x")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    eng = parsers._build_ocr_engine()
    assert eng is not None
    assert mod1.RapidOCR.instances == 1


def test_ocr_image_cached_supports_rapidocr3_output(monkeypatch):
    """3.x 的 RapidOCROutput（.txts）与 1.x 的 (result, elapse) 都能解析。"""

    class _Out3x:
        txts = ["第一行", "", "第三行"]

    class _Eng3x:
        def __call__(self, img):
            return _Out3x()

    class _Eng1x:
        def __call__(self, img):
            return ([["box", "旧版行", 0.9]], 0.1)

    assert parsers._ocr_image_cached(_Eng3x(), b"img", 1, "f.pdf") == "第一行\n第三行"
    assert parsers._ocr_image_cached(_Eng1x(), b"img", 1, "f.pdf") == "旧版行"


def test_pixmap_is_blank():
    """近纯色位图判 blank；含内容的位图不判。"""

    class _Pix:
        def __init__(self, samples):
            self.samples = samples

    assert parsers._pixmap_is_blank(_Pix(b"\xff" * 10000)) is True
    assert parsers._pixmap_is_blank(_Pix(b"")) is True
    mixed = bytearray(b"\xff" * 10000)
    mixed[5000] = 0
    assert parsers._pixmap_is_blank(_Pix(bytes(mixed))) is False


def test_ocr_recycle_pages_constant():
    """回收周期是主线 #130 实测定标值（改动需有实测依据）。"""
    assert parsers._OCR_RECYCLE_PAGES == 50
