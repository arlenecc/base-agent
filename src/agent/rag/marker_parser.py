"""PDF/文档解析 via marker-pdf（与 docling 并行的第二套统一解析引擎）。

marker（datalab-to/marker）把 PDF/图片转成高质量 Markdown，核心是
surya OCR/布局模型 + pdftext 文本层直读：文本页走文本层，图片页/低置信页
自动回退 OCR——与 docling 的分工类似但实现栈不同（torch/surya vs
onnx/docling-ibm-models）。

本模块接口与 :mod:`agent.rag.docling_parser` 完全对齐，方便引擎层并行
试用与对比：
  * ``extract_markdown(filepath)`` — 转 Markdown；不可用/不适用返回 None
  * ``is_docling_available`` 对应 ``is_marker_available``
  * ``release_converter`` / ``reset_unavailable`` 语义一致

注意：marker 与 docling 的依赖在同一解释器里存在已知冲突
（marker 拉 transformers>=5.x，docling-ibm-models 声明 <5.9；实测两者
仍都能转换，仅告警。打包/分发时两者是否共存需在 build 层面验证）。
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 与 docling_parser 同款约束：探测可用性只能用 find_spec，绝不能真实
# import marker——marker 会连带加载 torch/surya 全家桶（~1-2GB），而这条
# 探测在 PDF/epub 解析入口每文件都会走到。#131 离线环境变量同样适用
# （marker 的模型经 huggingface_hub 下载，已缓存则离线可用）。

# marker 2.x 原生支持的输入格式。docx/pptx/xlsx 交给 docling/按格式解析器
# （marker 只认 PDF/图片/epub/mobi 等文档位图类输入）。
_MARKER_EXTENSIONS = {".pdf", ".epub", ".mobi", ".azw3", ".azw", ".png", ".jpg", ".jpeg", ".webp"}

_converter = None
_converter_lock = threading.Lock()
_converter_unavailable = False
# marker 不做多实例：surya 系模型 ~3GB/实例，两实例 ~6GB 起步，16GB 机器
# 扛不住。docling 侧同样只保留单实例（2026-10-23 回退，两实例常驻 ~9GB +
# 两路工作集把 16GB 机器打到系统崩溃），保持单实例独占串行。


def release_converter() -> None:
    """Drop the shared marker converter so surya/torch models can be freed."""
    global _converter
    with _converter_lock:
        _converter = None


def reset_unavailable() -> None:
    """清掉「marker 不可用」的永久禁用标志（同 docling_parser.reset_unavailable）。"""
    global _converter_unavailable
    with _converter_lock:
        _converter_unavailable = False


def is_marker_available() -> bool:
    """Return True if marker is installed.

    用 ``find_spec`` 探测而**不是真实 import**：import marker 会连带加载
    torch/surya 全家桶（~1-2GB），与 docling 的 find_spec 修复（#131）同理。
    """
    global _converter_unavailable
    if _converter_unavailable:
        return False
    try:
        import importlib.util
        return importlib.util.find_spec("marker") is not None
    except Exception:
        return False


def is_supported(ext: str) -> bool:
    return ext.lower() in _MARKER_EXTENSIONS


def _get_converter():
    """Lazily build and cache the marker PdfConverter singleton.

    marker 模型（surya detection/recognition/layout + texify）首次加载会从
    HF 下载（~1-2GB），之后常驻内存约 3GB+。与 docling 同样的并发策略：
    进程级单例 + 同一把锁串行化转换（PdfConverter 未声明线程安全）。
    """
    global _converter, _converter_unavailable
    if _converter is not None:
        return _converter
    if _converter_unavailable:
        return None
    with _converter_lock:
        if _converter is not None:
            return _converter
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict

            artifact_dict = create_model_dict()
            # pdftext_workers=1：默认 4 会在 macOS GUI/冻结环境里 fork 出的
            # worker 进程崩掉（"A worker process died while extracting ..."）。
            # 单进程损失少量文本提取吞吐，换稳定；OCR 仍是批处理不受影响。
            # DPI 保持默认（布局 96 / OCR 192）。
            _converter = PdfConverter(
                artifact_dict=artifact_dict,
                config={"pdftext_workers": 1},
            )
            logger.info("marker PdfConverter initialized (global singleton)")
            return _converter
        except ImportError:
            logger.info("marker not installed; falling back to other parsers")
            _converter_unavailable = True
            return None
        except Exception as e:
            # 与 docling 相同策略：只对缺失永久禁用，瞬时错误（模型下载中断、
            # OOM）不置位，下次重试。
            logger.warning("failed to init marker converter (will retry): %s", e)
            return None


def extract_markdown(filepath: str) -> Optional[str]:
    """Convert a document to Markdown using marker.

    Returns the Markdown text, or ``None`` if marker is unavailable or the
    format is not handled (callers should fall back to docling/per-format
    parsers).
    """
    if not is_marker_available():
        return None

    path = Path(filepath)
    suffix = path.suffix.lower()
    if suffix not in _MARKER_EXTENSIONS:
        return None

    converter = _get_converter()
    if converter is None:
        return None

    try:
        with _converter_lock:
            # marker 2.x：PdfConverter 是可调用对象（无 .convert 方法）。
            rendered = converter(str(path))
            from marker.output import text_from_rendered
            text, ext, _images = text_from_rendered(rendered)
        del rendered
        if not text or not text.strip():
            logger.warning("marker extracted empty markdown from %s", path.name)
            return None
        return text
    except Exception as e:
        # Never let a marker failure abort ingestion — caller falls back.
        logger.warning("marker conversion failed for %s: %s", path.name, e)
        return None
