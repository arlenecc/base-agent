"""RapidOCR engine construction + image pre-processing.

Two things live here:

1. **Execution provider selection** — RapidOCR runs its three ONNX models
   (det / cls / rec) through ``onnxruntime.InferenceSession``.  Its bundled
   :class:`OrtInferSession` hardcodes the EP list to CUDA / DirectML / CPU —
   **CoreML is never considered**, so on Apple Silicon every OCR run burns
   CPU only while the Neural Engine and GPU sit idle.  We patch the EP list
   to put ``CoreMLExecutionProvider`` first when (and only when) the machine
   really is Apple Silicon and ``onnxruntime`` is new enough.

2. **Pre-processing / normalisation** — scanned pages vary wildly in DPI,
   contrast and noise.  Feeding the raw render to the detector loses small
   glyphs (the detector downsamples the short side to 736px by default).
   We normalise scale and enhance contrast before inference.

Everything is best-effort: any failure falls back to the stock behaviour
(CPU EP / raw image) instead of breaking ingestion.
"""
from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
import threading
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Apple Silicon detection
# ---------------------------------------------------------------------------

# CoreML EP requires onnxruntime >= 1.31.0 (below that the CoreML EP is either
# absent or too buggy to trust for these models).
_ORT_MIN_VERSION_FOR_COREML: Tuple[int, int, int] = (1, 31, 0)

_COREML_EP = "CoreMLExecutionProvider"
_CPU_EP = "CPUExecutionProvider"

# Compute units "ALL" lets CoreML schedule on CPU + GPU + ANE (Neural Engine),
# which is what actually makes OCR several times faster on M-series Macs.
# ModelFormat MLProgram is the modern CoreML format (coremltools >= 7).
_COREML_EP_OPTIONS: Dict[str, Any] = {
    "ModelFormat": "MLProgram",
    "MLComputeUnits": "ALL",
}

_apple_silicon_cache: Optional[bool] = None
_apple_silicon_lock = threading.Lock()


def is_apple_silicon() -> bool:
    """True when running on an Apple M-series Mac.

    ``platform.machine()`` reports ``arm64`` for native arm64 builds, but a
    Rosetta-translated interpreter reports ``x86_64`` on an M-series machine —
    so it alone is not enough.  We fall back to ``sysctl`` which always reports
    the *real* CPU ("Apple M1 Pro", "Apple M3 Max", …).
    """
    global _apple_silicon_cache
    if _apple_silicon_cache is not None:
        return _apple_silicon_cache
    with _apple_silicon_lock:
        if _apple_silicon_cache is not None:
            return _apple_silicon_cache
        _apple_silicon_cache = _detect_apple_silicon()
        return _apple_silicon_cache


def _detect_apple_silicon() -> bool:
    if platform.system() != "Darwin":
        return False
    if platform.machine() in ("arm64", "aarch64"):
        return True
    # Rosetta: machine() lies, ask the kernel for the real CPU name.
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=3,
        )
        brand = (out.stdout or "").strip()
    except Exception:  # pragma: no cover - sysctl missing/timeout
        brand = ""
    if re.match(r"^Apple\s+M\d", brand):
        return True
    # Very old Apple Silicon report "Apple processor"/virtualized CPUs.
    return bool(re.match(r"^Apple\s+", brand)) and "Intel" not in brand


def ort_version() -> Optional[Tuple[int, int, int]]:
    """Installed onnxruntime version as a tuple, or None if unavailable."""
    try:
        import onnxruntime as ort
    except Exception:
        return None
    raw = str(getattr(ort, "__version__", "") or "")
    nums: List[int] = []
    for part in raw.split("."):
        m = re.match(r"^(\d+)", part.strip())
        if not m:
            break
        nums.append(int(m.group(1)))
    while len(nums) < 3:
        nums.append(0)
    return (nums[0], nums[1], nums[2])


def coreml_reason() -> Tuple[bool, str]:
    """Explain whether CoreML EP *can* be used, and why (not).

    Returns ``(usable, human_readable_reason)``.  Purely diagnostic — no
    side effects, safe to call for logging/UI.
    """
    if not is_apple_silicon():
        return False, f"非 Apple Silicon（{platform.system()} {platform.machine()}），跳过 CoreML EP"
    ver = ort_version()
    if ver is None:
        return False, "onnxruntime 未安装"
    if ver < _ORT_MIN_VERSION_FOR_COREML:
        return False, (
            f"onnxruntime {'.'.join(str(v) for v in ver)} < "
            f"{'.'.join(str(v) for v in _ORT_MIN_VERSION_FOR_COREML)}，"
            f"请升级: pip install -U 'onnxruntime>={'.'.join(str(v) for v in _ORT_MIN_VERSION_FOR_COREML)}'"
        )
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
    except Exception as e:
        return False, f"无法读取 providers: {e}"
    if _COREML_EP not in providers:
        return False, f"当前 onnxruntime 未编译 CoreML EP（可用: {providers}）"
    return True, f"Apple Silicon + onnxruntime {'.'.join(str(v) for v in ver)} + CoreML EP 可用"


# ---------------------------------------------------------------------------
# CoreML EP injection
# ---------------------------------------------------------------------------

_patch_lock = threading.Lock()


def _ort_infer_session_class():
    """Locate RapidOCR's ``OrtInferSession`` across package layouts.

    Different rapidocr_onnxruntime releases place it in different modules:

    * 1.2.x（当前最新版）→ ``rapidocr_onnxruntime.utils``，且 EP 列表在
      ``__init__`` 里内联构建，**没有** ``_get_ep_list`` 方法；
    * 其它发布 → 各推理引擎子模块，多数暴露 ``_get_ep_list``。

    只按「模块里有 OrtInferSession」判定，不要求 ``_get_ep_list`` ——
    旧实现额外要求该方法，导致在 1.2.3 上永远返回 None，CoreML EP
    静默失效（日志只打「找不到 OrtInferSession」）。
    """
    candidates = (
        "rapidocr_onnxruntime.utils",                          # 1.2.x（实际布局）
        "rapidocr_onnxruntime.utils.infer_engine",
        "rapidocr_onnxruntime.utils.ort_infer_engine",
        "rapidocr_onnxruntime.inference_engine.ort_inference_engine",
    )
    import importlib
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        cls = getattr(mod, "OrtInferSession", None)
        if isinstance(cls, type):
            return cls
    return None


def _patch_coreml_ep(session_cls) -> bool:
    """Prepend CoreMLExecutionProvider to the EP list RapidOCR builds.

    两种布局各有一条注入路径：

    1. **有 ``_get_ep_list`` 的版本** —— 包装该方法，让 CoreML 排最前、
       CPU 留最后。ONNX Runtime 会自动把 CoreML 不支持的子图回退到 CPU，
       因此不支持的算子只会降速而不是报错。
    2. **只有 ``__init__`` 内联构建 EP 列表的版本**（如 rapidocr 1.2.x）
       —— 它通过模块级 ``InferenceSession(...)`` 工厂建会话，包装该工厂、
       在 ``providers`` 参数里注入 CoreML 即可，语义与上面一致。
    """
    if _patch_ep_list_method(session_cls):
        return True
    return _patch_inference_session_factory(session_cls)


def _patch_ep_list_method(session_cls) -> bool:
    """路径 1：包装 ``_get_ep_list``（暴露该方法的版本）。"""
    original = getattr(session_cls, "_get_ep_list", None)
    if original is None:
        return False
    if getattr(original, "_coreml_patched", False):
        return True

    def _get_ep_list(self):  # type: ignore[no-untyped-def]
        try:
            ep_list = list(original(self))
        except Exception:
            ep_list = [(_CPU_EP, {})]
        # Drop any pre-existing CoreML entry so ordering is deterministic.
        ep_list = [ep for ep in ep_list
                   if (ep[0] if isinstance(ep, (list, tuple)) else ep) != _COREML_EP]
        ep_list.insert(0, (_COREML_EP, dict(_COREML_EP_OPTIONS)))
        return ep_list

    _get_ep_list._coreml_patched = True          # type: ignore[attr-defined]
    _get_ep_list._coreml_original = original     # type: ignore[attr-defined]
    session_cls._get_ep_list = _get_ep_list      # type: ignore[assignment]
    return True


def _session_module(session_cls):
    """返回 ``OrtInferSession`` 定义所在模块（已 import），失败返回 None。"""
    mod_name = getattr(session_cls, "__module__", None) or ""
    if not mod_name.startswith("rapidocr"):
        return None
    import sys
    return sys.modules.get(mod_name)


def _patch_inference_session_factory(session_cls) -> bool:
    """路径 2：包装 ``OrtInferSession`` 模块里的 ``InferenceSession`` 工厂。

    rapidocr 1.2.x 的 ``__init__`` 直接调模块级
    ``InferenceSession(model_path, sess_options=..., providers=EP_list)``。
    替换该全局引用即可在 providers 里注入 CoreML；``__init__`` 里的其它
    逻辑（CUDA 探测、sess_opt）不受影响。
    """
    mod = _session_module(session_cls)
    if mod is None:
        return False
    original_is = getattr(mod, "InferenceSession", None)
    if original_is is None:
        return False
    if getattr(original_is, "_coreml_patched", False):
        return True

    def _make_session(*args, **kwargs):  # type: ignore[no-untyped-def]
        providers = list(kwargs.get("providers") or [])
        providers = [p for p in providers
                     if (p[0] if isinstance(p, (list, tuple)) else p) != _COREML_EP]
        providers.insert(0, (_COREML_EP, dict(_COREML_EP_OPTIONS)))
        kwargs["providers"] = providers
        return original_is(*args, **kwargs)

    _make_session._coreml_patched = True          # type: ignore[attr-defined]
    _make_session._coreml_original = original_is  # type: ignore[attr-defined]
    mod.InferenceSession = _make_session          # type: ignore[assignment]
    return True


def _unpatch_coreml_ep(session_cls) -> None:
    """Restore the stock EP list (used when CoreML session creation fails).

    两条注入路径都要能恢复：``_get_ep_list`` 包装与
    ``InferenceSession`` 工厂包装。
    """
    patched = getattr(session_cls, "_get_ep_list", None)
    original = getattr(patched, "_coreml_original", None)
    if original is not None:
        session_cls._get_ep_list = original      # type: ignore[assignment]
        return
    mod = _session_module(session_cls)
    if mod is None:
        return
    cur_is = getattr(mod, "InferenceSession", None)
    orig_is = getattr(cur_is, "_coreml_original", None)
    if orig_is is not None:
        mod.InferenceSession = orig_is           # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Engine construction
# ---------------------------------------------------------------------------

# Detector input resolution.  RapidOCR's default (736, limit_type="min")
# downsamples an A4 page rendered at 200dpi to ~736px short side — small
# glyphs get destroyed.  1024 keeps the text legible at a modest cost.
_DET_LIMIT_SIDE_LEN = 1024


def _engine_kwargs(**overrides) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "det_limit_side_len": _DET_LIMIT_SIDE_LEN,
        "det_limit_type": "min",
    }
    kwargs.update(overrides)
    return kwargs


def _build_rapid_ocr(engine_kwargs: Dict[str, Any]):
    """构造 RapidOCR 引擎，兼容各版本的 kwargs 体系。

    rapidocr 1.2.x 的 ``UpdateParameters`` 在收到**任何** ``det_*``
    kwargs 时都会无条件读 ``det_dict['model_path']`` —— 只传
    ``det_limit_side_len`` 而不传 ``det_model_path`` 会直接
    ``KeyError: 'model_path'``（上游缺陷），引擎根本建不起来，表现为
    「明明装了 rapidocr 却提示未安装」。旧版（1.1.x）的 argparse 体系
    则不认 ``det_model_path`` 这个多余参数。

    因此按顺序降级重试，任一成功即返回：
    1. 完整 kwargs（保留 1024 分辨率调优）；
    2. 补 ``det_model_path=""`` —— 空值让 1.2.x 回退到自带模型路径；
    3. 裸构造 —— 放弃调优也要保证 OCR 可用。
    """
    from rapidocr_onnxruntime import RapidOCR

    with_model_path = dict(engine_kwargs)
    with_model_path.setdefault("det_model_path", "")
    attempts = [engine_kwargs, with_model_path, {}]

    last_exc: Optional[BaseException] = None
    for kw in attempts:
        try:
            return RapidOCR(**kw)
        except (KeyError, TypeError) as e:
            # kwargs 体系不兼容 → 换下一组更保守的参数。
            # 其它异常（模型下载失败等）直接上抛，重试没有意义。
            last_exc = e
            logger.debug("OCR: RapidOCR(**%s) 失败，降级重试: %s", sorted(kw), e)
    assert last_exc is not None
    raise last_exc


def create_ocr_engine(use_coreml: bool = True, **kwargs):
    """Build a RapidOCR engine, preferring CoreML EP on Apple Silicon.

    Falls back to the stock CPU engine if CoreML is unavailable *or* if the
    CoreML session fails to build (some ops are unsupported on certain ORT
    builds) — the patch is rolled back before retrying so a later construction
    is not poisoned by the failed attempt.
    """
    engine_kwargs = _engine_kwargs(**kwargs)

    def _build():
        return _build_rapid_ocr(engine_kwargs)

    if not use_coreml:
        logger.info("OCR: CoreML EP 已按配置关闭，使用默认 EP")
        return _build()

    usable, reason = coreml_reason()
    if not usable:
        logger.info("OCR: 未启用 CoreML EP — %s", reason)
        return _build()

    with _patch_lock:
        session_cls = _ort_infer_session_class()
        if session_cls is None:
            logger.warning(
                "OCR: 找不到 rapidocr 的 OrtInferSession，无法注入 CoreML EP，回退默认 EP"
            )
            return _build()
        if not _patch_coreml_ep(session_cls):
            return _build()
        try:
            engine = _build()
            logger.info("OCR: 已启用 CoreML EP（MLComputeUnits=ALL，CPU+GPU+ANE）— %s", reason)
            return engine
        except Exception as e:
            # CoreML session creation failed (unsupported op / MLProgram
            # unavailable).  Roll the patch back and retry on CPU so OCR still
            # works — a failed EP must never break ingestion.
            _unpatch_coreml_ep(session_cls)
            logger.warning("OCR: CoreML EP 初始化失败，回退 CPU EP: %s", e)
            return _build()


# ---------------------------------------------------------------------------
# Image pre-processing / normalisation
# ---------------------------------------------------------------------------

# 送入引擎前的图像尺寸闸门：
#   _TARGET_SHORT_SIDE — 渲染 DPI 的求解目标（短边 px）
#   _MIN_SHORT_SIDE    — 短边低于此值则放大（小字才认得出）
#   _MAX_LONG_SIDE     — 长边硬上限，超过即按比例缩小（真正的归一化闸门）
#
# _MAX_LONG_SIDE 4000 → 1500（实测修订）：决定单页像素总量的是**长边**，而
# 竖版扫描件正是在这里失控——本书渲染出 2399×3725（8.9MP），长边 3725 < 4000，
# 旧阈值完全不触发，图像原样进引擎（短边 2399 也落在 900~4000 的盲区里）。
# 长边限 1500 后为 966×1500（1.45MP，像素 -84%）：60 页 CoreML OCR 的 RSS
# 增长 4.08GB → 3.37GB（-17%）、峰值 4.59 → 3.88GB、速度 1.73 → 1.65 s/页；
# 识别结果 23329 vs 23412 字符（差 0.35%，逐字重合 0.991~1.000）——质量基本无损。
# 注意短边会被连带压到 ~1000px（A4 等效约 128dpi），小字号文档可能掉字；
# 若实测到掉质量，调回 2000 / 2600 即可（本书上 2600 是逐字重合 1.000）。
_TARGET_SHORT_SIDE = 1600
_MAX_LONG_SIDE = 1500
_MIN_SHORT_SIDE = 900
# 渲染 DPI 硬上限。走 OCR 的页面最终都会归一到长边 _MAX_LONG_SIDE，渲染 dpi
# 只决定中间重采样率：超过上限就是在渲染一张马上要被缩掉的大位图（浪费内存
# 与渲染时间），低于上限则只是少一次超采样（实测对扫描件质量无差别）。
_MAX_RENDER_DPI = 150


def render_dpi_for_page(page, target_short_side: int = _TARGET_SHORT_SIDE) -> int:
    """Pick a render DPI that lands the page at its normalised size directly.

    PDFs ship at wildly different physical sizes (A4 vs letter vs a tiny
    cropped figure).  Rendering everything at a fixed 200dpi makes the
    detector see very different glyph sizes.  Solving for DPI first means the
    detector always works on roughly the same scale.

    两个目标取更严格者：短边凑到 ``_TARGET_SHORT_SIDE``，**且**长边不超过
    ``_MAX_LONG_SIDE``。旧实现只看短边，于是竖版大开本被解出 100dpi 后又被
    ``max(150, ...)`` 抬回 150，渲染出 2399×3725（8.9MP）再缩到 1500——白
    渲染 6 倍像素。按长边反推后一次渲染就落在 1500，省掉这一步浪费。

    DPI 只作为「不超过 150」的上限约束，不要求等于 150：走 OCR 的页面都是
    无文字层的扫描件（位图嵌入），渲染 dpi 只决定重采样率，最终信息量由长边
    1500 决定。实测同一页 60/100/150dpi 渲染后归一到 1500，识别结果完全一致
    （字符数相同、逐字重合 0.997~1.000）。
    """
    try:
        rect = page.rect
        width_in = max(float(rect.width), 1e-6) / 72.0
        height_in = max(float(rect.height), 1e-6) / 72.0
        short_in = min(width_in, height_in)
        long_in = max(width_in, height_in)
        dpi = int(round(target_short_side / short_in))
        # 长边约束：超过上限就按长边反推，一次渲染直接落在目标尺寸。
        dpi = min(dpi, int(round(_MAX_LONG_SIDE / long_in)))
    except Exception:
        return _MAX_RENDER_DPI
    # 硬上限 150：小页面（物理尺寸小）为凑够像素会被解出很高的 dpi，渲染出
    # 远大于最终尺寸的位图（实测浪费 2-4 倍渲染开销与位图内存）。封顶即可，
    # 不做下限抬升——长的那一边由上面的长边约束负责。
    return max(1, min(dpi, _MAX_RENDER_DPI))


def preprocess_for_ocr(
    img: "Any",
    enhance: bool = True,
    binarize: bool = True,
) -> "Any":
    """Normalise + enhance a BGR image before OCR.

    Steps (all best-effort; any failure returns the input untouched):

    1. **Scale normalisation** — short side < ``_MIN_SHORT_SIDE`` is upscaled,
       long side > ``_MAX_LONG_SIDE`` is downscaled, both with **Lanczos**
       (``INTER_LANCZOS4``): 缩小用 Lanczos 保留笔画边缘，比 INTER_AREA 的字形
       更完整（对中文这种笔画密集的字尤其明显）。
    2. **Grayscale + dark-theme inversion + CLAHE** — predominantly dark pages
       (dark mode e-books, night-mode scans) are inverted to white-background
       first; adaptive histogram equalisation then lifts faint scans and evens
       out lighting gradients that break text detection. 后续所有处理都在
       **灰度**上进行。
    3. **Unsharp mask** — restores edge definition lost to the resample.
    4. **Binarisation** (default on) — 自适应阈值（高斯加权、blockSize=31、
       C=10）把灰度压成黑白。必须放在锐化**之后**：先增强边缘再二值化，字形
       才完整；反过来锐化作用在二值图上没有意义。
       关闭它请用 ``binarize=False``（截图 / 彩色插图这类非文档图像二值化会
       丢信息）。

    Returns a **BGR** ``ndarray``（内容是灰度/二值，但通道数保持 3 —— RapidOCR
    把 3 通道 ndarray 当 BGR 处理、不做转换）。
    """
    if img is None:
        return img
    try:
        import cv2
        import numpy as np
    except Exception:
        return img
    try:
        arr = np.asarray(img)
        if arr.ndim == 2:
            work = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            work = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            work = arr
        else:
            return img
        if work.size == 0:
            return img

        # ---- 1. scale normalisation ----
        h, w = work.shape[:2]
        short, long_ = min(h, w), max(h, w)
        scale = 1.0
        if short < _MIN_SHORT_SIDE:
            scale = _MIN_SHORT_SIDE / float(short)
        if long_ * scale > _MAX_LONG_SIDE:
            scale = _MAX_LONG_SIDE / float(long_)
        if abs(scale - 1.0) > 0.01:
            # Lanczos：缩小（超出长边上限）时保留笔画边缘，放大（短边不足）
            # 时也比 CUBIC 更少振铃。代价是比 INTER_AREA 慢一些，相对每页
            # 1~2s 的 OCR 可忽略。
            work = cv2.resize(work, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_LANCZOS4)

        if not enhance:
            return work

        # ---- 2. grayscale + 暗色主题反色 + CLAHE ----
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        # 2a. 暗色主题自动反色（#100-#101）：RapidOCR 的 det/rec 模型在白底
        # 黑字、高对比度样本上训练，黑底白字/灰字的暗色主题页（电子书暗色
        # 模式、夜间模式扫描）几乎检不出任何文本框——「一大片页面检测不到
        # 文字」的根因。页面整体偏暗（p50 < 100 且 p95 < 160，即 95% 像素
        # 都是暗色调，符合「深色背景 + 少量浅色文字」的特征）时反色为白底
        # 深字。插图占比大的正常页 p95 会远超 160，不会误触发。
        # 反色之后再进 CLAHE——CLAHE 的局部对比度增强同时覆盖了低对比度
        # 页面的拉伸需求（与主线 _ocr_normalize 的两步分位数拉伸等价）。
        try:
            import numpy as np
            p50, p95 = np.percentile(gray, (50, 95))
            if p50 < 100 and p95 < 160:
                gray = 255 - gray
        except Exception:
            pass
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        except Exception:
            # CLAHE unavailable on very old OpenCV builds.
            pass

        # ---- 3. unsharp mask ----
        # 必须在二值化**之前**：锐化增强的是灰度边缘，作用在二值图上无意义。
        try:
            blurred = cv2.GaussianBlur(gray, (0, 0), 1.2)
            sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        except Exception:
            sharpened = gray

        # ---- 4. 二值化 ----
        if binarize:
            # 自适应阈值：高斯加权、blockSize=31、C=10。对扫描件（白底黑字、
            # 光照不均）能显著拉开字与纸的对比；截图/彩色插图请传
            # binarize=False。
            try:
                sharpened = cv2.adaptiveThreshold(
                    sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 31, 10,
                )
            except Exception:
                pass

        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    except Exception as e:
        logger.debug("OCR 预处理失败，使用原图: %s", e)
        return img


# ---------------------------------------------------------------------------
# Apple Vision（ocrmac）：中文印刷体的识别质量高于 RapidOCR（尤其竖排、
# 复杂版式），ANE 推理不占 CPU/内存。作为全项目 OCR 的首选后端，任何失败
# 回退 RapidOCR。回退契约：ocrmac 不可用或调用异常返回 None；识别成功但
# 页面上确实无文字返回空串。
# ---------------------------------------------------------------------------
_OCRMAC_STATE: Optional[bool] = None
_OCRMAC_LOCK = threading.Lock()
# 语言优先级：中文在前。ocrmac 默认 ['en-US', 'es-ES', ...] 不含中文，
# 对中文页面会大面积漏识别。英文保留在列表里以混排（变量名/引用）兼容。
_OCRMAC_LANGS = ["zh-Hans", "en-US"]


def ocrmac_available() -> bool:
    """ocrmac（Apple Vision 框架封装）是否可用；懒探测一次并缓存。"""
    global _OCRMAC_STATE
    if _OCRMAC_STATE is not None:
        return _OCRMAC_STATE
    with _OCRMAC_LOCK:
        if _OCRMAC_STATE is None:
            ok = False
            if platform.system() == "Darwin":
                try:
                    import ocrmac  # noqa: F401
                    ok = True
                except Exception:
                    ok = False
            _OCRMAC_STATE = ok
            logger.info("ocrmac (Apple Vision) 可用性: %s", ok)
    return _OCRMAC_STATE


def ocr_with_ocrmac(img: "Any") -> Optional[str]:
    """用 Apple Vision 识别一张图片。

    ``img`` 接受 BGR/gray ndarray 或 PNG/JPEG bytes（先经 :func:`decode_image`）。
    返回识别文本（空串 = 页面上确实没有字）；ocrmac 不可用或调用失败返回
    ``None``——调用方据此回退 RapidOCR。
    """
    if not ocrmac_available():
        return None
    try:
        import numpy as np
        from ocrmac import ocrmac as _om
        from PIL import Image
        if isinstance(img, (bytes, bytearray)):
            decoded = decode_image(bytes(img))
            if decoded is None:
                return None
            img = decoded
        arr = np.asarray(img)
        if arr.ndim == 3 and arr.shape[2] == 3:
            pil = Image.fromarray(arr[:, :, ::-1])   # BGR → RGB
        elif arr.ndim == 2:
            pil = Image.fromarray(arr)
        else:
            return None
        # ocrmac 的 API：OCR(image, framework, recognition_level, language_
        # preference, ...)——参数名是 recognition_level，没有 LEVEL_* 常量。
        # 传错会 TypeError → 被下面的 except 吞掉静默回退，这里参数名必须对。
        lines = []
        for text, _conf, _box in _om.OCR(
                pil, language_preference=_OCRMAC_LANGS,
                recognition_level="accurate").recognize():
            t = (text or "").strip()
            if t:
                lines.append(t)
        return "\n".join(lines)
    except Exception as e:
        logger.debug("ocrmac 识别失败，将回退 RapidOCR: %s", e)
        return None


def decode_image(img_bytes: bytes) -> "Any":
    """PNG/JPEG bytes → BGR ndarray (``None`` when decoding fails)."""
    try:
        import cv2
        import numpy as np
        buf = np.frombuffer(img_bytes, dtype=np.uint8)
        arr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return arr
    except Exception as e:
        logger.debug("OCR 图像解码失败: %s", e)
        return None
