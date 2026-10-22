"""Unified document parsing via docling.

docling is a single, high-quality document-understanding engine that converts
Word, Excel, PowerPoint, PDF and EPUB into structured Markdown (with layout
awareness, tables, headings, and image OCR).  This module wraps it behind the
same ``extract_text``-style interface so the RAG engine can use one parser for
all office/document formats.

Design:
  * ``extract_markdown(filepath)`` — convert any supported document to Markdown
    via docling.  Returns ``None`` when docling is unavailable (callers fall
    back to the per-format parsers in :mod:`agent.rag.parsers`).
  * Image-based PDF pages are OCR'd by ocrmac (Apple Vision, zh-Hans first)
    when available, RapidOCR otherwise; for pure-scan PDFs without a text
    layer we route through the shared OCR entry in :mod:`agent.rag.parsers`
    (same ocrmac-first policy).
"""
from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, List, Optional

# 强制 HuggingFace 离线（#131）：docling convert 首次运行会向 huggingface.co
# 检查模型更新，网络不通时 connect() 要等完整 TCP 超时 + hub 客户端重试，
# 实测把整个同步卡死 20+ 分钟（2026-10-20 抓到 worker 阻塞在 __connect，
# 上层是 pydantic 模型构造，OCR 的 VN 线程同时挂起）。docling 所需模型
# （layout-heron / TableFormer）早已缓存在 ~/.cache/huggingface，离线模式
# 直接走缓存；未缓存的全新环境会立刻报错 → 走轻量解析器回退，不会挂死。
# setdefault：用户显式设置的值优先。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# 保险：若 huggingface_hub 在本模块之前已被导入（其 constants 已固化
# HF_HUB_OFFLINE=False），直接改写常量兜底。部分 hf 模块在 import 时绑定了
# 旧值，改不干净——根治靠 agent/__init__.py 的最早设置，这里只是兜底。
try:
    import huggingface_hub.constants as _hf_constants
    if not _hf_constants.HF_HUB_OFFLINE:
        _hf_constants.HF_HUB_OFFLINE = True
except Exception:
    pass

from .ocr_scale_policy import (
    analyze_pdf as _analyze_ocr_scale,
    make_scale_policy_pipeline_cls,
    set_current_policy as _set_scale_policy,
    set_scale_config as _set_scale_cfg,
)

logger = logging.getLogger(__name__)

# OCR 渲染 scale 配置（#132，经 set_ocr_scale_config 注入）。默认固定
# 2.0、动态策略关闭——位图内存 ∝ scale²，2.0 相比旧默认 3.0 省 56%。
#
# 动态策略关闭时（默认）：**完全不挂载**策略管线、不做预分析、不包装
# OCR 模型——scale 只在 converter 构建时设进 OcrMacOptions 一次，默认
# 路径零额外开销。动态开启时才构建带 ScalePolicyOcrModel 的管线。
_ocr_scale_base = 2.0
_ocr_scale_dynamic = False
_ocr_scale_table = 2.5


def set_ocr_scale_config(base: float = 2.0, dynamic: bool = False,
                         table: float = 2.5) -> None:
    """注入 OCR scale 配置（UI 保存设置 / 应用配置时调用）。

    base   固定 scale（rag_ocr_scale_base）
    dynamic 动态策略开关（rag_ocr_scale_dynamic，默认 False）
    table  动态策略下表格页的 scale（rag_ocr_scale_table）

    任一值变化时释放已构建的 converter 单例——scale / 动态开关在 converter
    构建时固化（OcrMacOptions 构造 + pipeline_cls 选择），不重建的话改配置
    要重启应用才生效。
    """
    global _ocr_scale_base, _ocr_scale_dynamic, _ocr_scale_table
    new_base = float(base) if base else 2.0
    new_dynamic = bool(dynamic)
    new_table = float(table) if table else 2.5
    changed = (new_base != _ocr_scale_base
               or new_dynamic != _ocr_scale_dynamic
               or new_table != _ocr_scale_table)
    _ocr_scale_base = new_base
    _ocr_scale_dynamic = new_dynamic
    _ocr_scale_table = new_table
    _set_scale_cfg(new_dynamic, new_base, new_table)
    if changed and _pool is not None and _pool.has_instances():
        logger.info(
            "OCR scale 配置变化（base=%.1f dynamic=%s table=%.1f），"
            "释放 converter 以重建",
            new_base, new_dynamic, new_table,
        )
        release_converter()

# Formats docling can handle natively.  Text/markdown/csv/html are handled by
# the lightweight parsers (docling is overkill and slower for them).
# 不含 .epub：EPUB 有专门的 ebook_parser（pandoc → Markdown + 章节切块，
# 秒级完成）；docling 处理 epub 要先构建 ~3-4GB 的 DocumentConverter
# （布局模型 + TableFormer，torch），与 ebook_parser 完全重复且慢得多。
# 2026-10 实测：同步中 docling converter 构建是 2 秒内 +3.7GB 内存突增的
# 元凶。.doc/.xls 等 legacy 格式由调用方提前拦截（docling 同样不支持）。
_DOCLING_EXTENSIONS = {".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt", ".pdf"}

# Converter 实例池：docling 模型很重（布局/表格 ~1.6GB/实例），DocumentConverter
# 非线程安全（不能单实例并发 convert），但**多实例各自转换是安全的**。
#
# 大小分流（2026-10-22 用户定版：实际内存压力与 PDF 大小相关）：
# - 小 PDF（< _large_file_bytes，默认 20MB）：≤ _max_parallel 路并行转换——
#   20MB 以下大概率是文本 PDF（文本层内存开销小），即便是扫描件页数也不
#   多，并行安全；
# - 大 PDF（≥ 阈值）：独占全部槽位——大页渲染位图 ∝ scale²，两份大书并行
#   就能撑爆内存；
# - 排队的大文件优先：有大文件等待时小文件不再新增并行（防饿死）。
_max_parallel = 2
_large_file_bytes = 20 * 1024 * 1024
# 「docling 不可用」永久禁用标志（ImportError 时置位，reset_unavailable 复位）。
# 池化改造时曾误删此模块级定义 → is_docling_available()/ _get_converter() 一
# 调用就 NameError，且被 engine 回退链的 except Exception 静默吞掉，表现为
# 所有 PDF 静默降级到 marker（2026-10-22 实测修复）。
_converter_unavailable = False


def set_docling_concurrency(max_parallel: int, large_file_mb: float) -> None:
    """注入转换并发配置（同步开始前由 worker 从 config 调用）。

    max_parallel  转换器实例上限 = 小文件并行转换上限（rag_parse_parallel）
    large_file_mb 大文件阈值：≥ 此大小的 PDF 独占全部实例（rag_parse_large_mb）
    """
    global _max_parallel, _large_file_bytes
    _max_parallel = max(1, int(max_parallel or 1))
    _large_file_bytes = max(0.0, float(large_file_mb or 0)) * 1024 * 1024
    if _pool is not None:
        _pool.resize(_max_parallel)


class _ConverterPool:
    """docling 转换器实例池：小文件并行、大文件独占、惰性建实例。

    旧实现是「模块级单例 + 一把锁全程持有」——PDF 转换实际串行（这正是
    并发提不上去的原因：单实例不线程安全，无法并发 convert）。改为实例池
    后小文件可真并行（各自独立实例），大文件独占保内存。
    """

    def __init__(self, factory, max_size: int = 1) -> None:
        self._factory = factory
        self._max = max(1, int(max_size))
        self._cond = threading.Condition()
        self._free: List[Any] = []      # 空闲实例栈
        self._created = 0               # 存活实例数（含借出）
        self._busy = 0                  # 占用槽位数（独占时 = _max）
        self._active_large = False      # 大文件转换中
        self._large_waiters = 0         # 排队中的大文件数
        self._last_used = 0.0

    # -- 配置 ---------------------------------------------------------------

    def resize(self, max_size: int) -> None:
        with self._cond:
            self._max = max(1, int(max_size))
            self._cond.notify_all()

    def has_instances(self) -> bool:
        with self._cond:
            return self._created > 0

    # -- 租约 -----------------------------------------------------------------

    @contextlib.contextmanager
    def lease(self, exclusive: bool):
        """租一个转换器实例：``with _pool.lease(excl) as conv:``。

        factory 返回 None → yield None（调用方本次回退）；factory 抛异常 →
        先归还槽位再上抛（不留死槽位）。
        """
        conv = None
        with self._cond:
            if exclusive:
                self._large_waiters += 1
                try:
                    while self._busy > 0 or self._active_large:
                        self._cond.wait()
                    self._active_large = True
                    self._busy = self._max        # 独占全部槽位
                finally:
                    self._large_waiters -= 1
            else:
                while (self._active_large or self._large_waiters > 0
                       or self._busy >= self._max):
                    self._cond.wait()
                self._busy += 1
            try:
                conv = self._acquire_instance()
            except BaseException:
                self._give_back_slots(exclusive)
                raise
            self._last_used = time.monotonic()
        try:
            yield conv
        finally:
            with self._cond:
                if conv is not None:
                    self._free.append(conv)
                self._give_back_slots(exclusive)
                self._last_used = time.monotonic()

    def _give_back_slots(self, exclusive: bool) -> None:
        self._busy = 0 if exclusive else max(0, self._busy - 1)
        if exclusive:
            self._active_large = False
        self._cond.notify_all()

    def _acquire_instance(self):
        if self._free:
            return self._free.pop()
        conv = self._factory()
        if conv is not None:
            self._created += 1
        return conv

    # -- 空闲释放 ---------------------------------------------------------------

    def reap_if_idle(self, min_idle: float) -> int:
        """空闲超时则丢掉全部空闲实例，返回释放个数；有活跃转换时不动。"""
        with self._cond:
            if (self._busy or self._created == 0 or self._last_used <= 0.0
                    or time.monotonic() - self._last_used < min_idle):
                return 0
            freed = len(self._free)
            self._free = []
            self._created = 0
            self._cond.notify_all()
            return freed

    def release_all(self) -> None:
        """立刻丢掉全部空闲实例（engine 同步结束后的显式释放）。"""
        with self._cond:
            self._free = []
            self._created = 0
            self._cond.notify_all()


# ---- 空闲自动释放（#131）----
# 2026-10-20 实测：同步第 3 分钟排到第一个 PDF/office 文件时构建 converter
# （import torch + docling-layout-heron 权重 + TableFormer + 首次推理激活）
# 一次性 +4.6GB，此后常驻到同步结束才被 engine 的 release_converter 释放。
# 对「大部分是 txt/epub、只夹了一两个 PDF」的语料，这 4GB+ 要白占几十分钟。
# 空闲超过 _IDLE_RELEASE_SECS 且无活跃转换时由后台线程整池释放；下一个
# docling 文件到来时工厂会重建（构建 ~2s，代价远小于长时间白占 4GB+）。
_IDLE_RELEASE_SECS = 90.0
_IDLE_CHECK_INTERVAL = 15.0
_reaper_started = False
# reaper 启动标志的专用锁：绝不能用转换锁——它在 converter.convert()
# 期间被长时间持有（整份文档转换数分钟），_start_idle_reaper 在工厂成功
# 路径上被调用，抢这把锁会让刚构建完 converter 的 worker 卡在另一个
# worker 的整个转换期间（2026-10-20 实测：正是这个等待让 py-spy 抓到
# Thread-4 卡在 _start_idle_reaper，掩盖了真正的 convert 挂死）。
_reaper_start_lock = threading.Lock()


def _start_idle_reaper() -> None:
    """启动后台空闲释放线程（进程内仅启动一次）。"""
    global _reaper_started
    if _reaper_started:
        return
    with _reaper_start_lock:
        if _reaper_started:
            return
        _reaper_started = True

    def _reap() -> None:
        while True:
            time.sleep(_IDLE_CHECK_INTERVAL)
            freed = _pool.reap_if_idle(_IDLE_RELEASE_SECS)
            if freed:
                logger.info(
                    "docling converter 空闲超时，释放 %d 个实例的布局/表格"
                    "模型（数 GB）", freed,
                )

    threading.Thread(
        target=_reap, name="docling-idle-reaper", daemon=True
    ).start()


def release_converter() -> None:
    """Drop the shared docling converter so its models can be freed.

    Mirrors ``parsers.release_ocr_engine()``: ingestion runs in short bursts
    (a sync job), after which the converter's models would otherwise stay
    resident for the whole process lifetime.
    """
    _pool.release_all()


def reset_unavailable() -> None:
    """清掉「docling 不可用」的永久禁用标志。

    ``_converter_unavailable`` 在任何异常（不只是 ImportError）下都会被置位
    且没有复位路径。典型后果：进程早期跑过一次 ingest（docling 未装 → 禁用），
    随后依赖检查自动装上了 docling，之后的同步仍然全程回退到按格式解析
    （PDF 表格/版面质量下降），且毫无提示。依赖装好后调用本函数即可恢复。
    """
    global _converter_unavailable
    _converter_unavailable = False


def is_docling_available() -> bool:
    """Return True if docling can be imported.

    用 ``find_spec`` 探测而**不是真实 import**：import docling 会连带加载
    torch 全家桶（~1-2GB）。这条探测在每个 PDF/EPUB 的解析入口都会走到，
    为探测而加载 torch 纯属浪费（2026-10 实测：同步中 2 秒内 +3.7GB 的
    元凶之一）。
    """
    global _converter_unavailable
    if _converter_unavailable:
        return False
    try:
        import importlib.util
        return importlib.util.find_spec("docling") is not None
    except Exception:
        return False


def is_supported(ext: str) -> bool:
    return ext.lower() in _DOCLING_EXTENSIONS


def _get_converter():
    """构建一个**新的** docling DocumentConverter 实例（池负责复用/释放）。

    旧实现是模块级单例（本函数内缓存 + 全程持锁串行 convert）；改为实例池
    后本函数只负责构建单个实例，缓存/并发/空闲释放策略全在 _ConverterPool。
    保留函数名——tests/test_docling_parser.py 以它为 monkeypatch 接缝。

    管线配置（版本兼容，全部包在 try/except 中，任何失败都回退默认构造）：
    accelerator = MPS（torch Metal）+ 4 线程。2026-10-20 A/B：布局/表格
    模型走 MPS 比 CPU 快 ~35% 且峰值内存更低，#116 的 36GB 失控（MPS 缓存
    只增不放）由三道闸兜住：convert 后 torch.mps.empty_cache() 归还缓存、
    document_timeout 硬超时、MLX 缓存上限。注意不要设
    PYTORCH_MPS_HIGH_WATERMARK_RATIO（docling 内部 low=1.4 会直接报错）。
    """
    global _converter_unavailable
    if _converter_unavailable:
        return None
    try:
        from docling.document_converter import DocumentConverter

        converter = None
        try:
            from docling.datamodel.accelerator_options import (
                AcceleratorDevice,
                AcceleratorOptions,
            )
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import PdfFormatOption

            pdf_opts = PdfPipelineOptions()
            # 加速器：MPS（torch Metal）。2026-10-20 60 页真实扫描书 A/B：
            # 热转换快 ~35%（106.5s vs 160.9s）、峰值内存 2.58GB 平坦 vs
            # CPU 版 3.94→4.80GB 持续增长；且布局模型移到 GPU 后 CPU 核
            # 让给 RapidOCR/嵌入（原来日志里子批调优器频繁"降档避让"
            # 就是这个争抢）。#116 的 36GB 失控由 empty_cache 护栏 +
            # document_timeout + 峰值监控三道闸兜住（不再依赖回避 MPS）。
            # 注意：不要设 PYTORCH_MPS_HIGH_WATERMARK_RATIO——docling 的
            # transformers 引擎内部设 low=1.4，高水位低于它会直接报错。
            pdf_opts.accelerator_options = AcceleratorOptions(
                num_threads=4, device=AcceleratorDevice.MPS
            )
            # 单文档硬超时（#131）：convert 内部可能永久挂死（实测 2026-10-20
            # 同步中 convert 卡 20+ 分钟无 CPU 活动，疑 Apple Vision OCR 回调
            # 不返回）。docling 原生支持 document_timeout，超时抛错 →
            # extract_markdown 捕获 → 回退轻量解析器，同步不中断。
            # 3600s：图片书逐页 OCR（Apple Vision）+ 布局 + TableFormer
            # 较慢（数百页大书约 10-20 分钟），超时只是防永久挂死的最后
            # 防线，不能误杀慢而正常的转换。
            try:
                pdf_opts.document_timeout = 3600.0
            except Exception:
                pass  # 老版本无此字段
            # OCR 后端：Apple Vision（ocrmac）优先——中文印刷体识别质量
            # 高于 RapidOCR，ANE 推理不占 CPU/内存；不可用（非 macOS /
            # 未安装 ocrmac）回退 RapidOCR。lang 必须显式给中文：docling
            # 的 OcrMacOptions 默认 ['en-US', 'es-ES', ...] 不含中文，
            # 对中文扫描页会大面积漏识别。
            from .ocr_engine import ocrmac_available
            if ocrmac_available():
                try:
                    from docling.datamodel.pipeline_options import OcrMacOptions
                    # OCR 渲染 scale（#132）：可配置（rag_ocr_scale_base，
                    # 默认 2.0，原 3.0）。整页位图内存 ∝ scale²，
                    # 3.0→2.0 省 56%——实测 25GB 尖峰的主要来源。动态
                    # 策略（默认关）开启时表格页会在逐页渲染前提到
                    # rag_ocr_scale_table（见 extract_markdown 的策略注入）。
                    pdf_opts.ocr_options = OcrMacOptions(
                        lang=["zh-Hans", "en-US"], recognition="accurate",
                        scale=_ocr_scale_base,
                    )
                    pdf_opts.do_ocr = True
                    logger.info("docling OCR 后端: Apple Vision (ocrmac, zh-Hans 优先)")
                except Exception as e:
                    # docling 版本差异导致 OcrMacOptions 不可用 → 回退。
                    logger.info("OcrMacOptions 不可用（%s），回退 RapidOCR", e)
            if getattr(pdf_opts, "ocr_options", None) is None:
                try:
                    from docling.datamodel.pipeline_options import RapidOcrOptions
                    try:
                        pdf_opts.ocr_options = RapidOcrOptions(
                            scale=_ocr_scale_base)
                    except Exception:
                        # 老版本 RapidOcrOptions 无 scale 参数
                        pdf_opts.ocr_options = RapidOcrOptions()
                    pdf_opts.do_ocr = True
                    logger.info("docling OCR 后端: RapidOCR")
                except Exception:
                    pass
            # OCR scale 动态策略（#132）：**仅动态开关开启时**才挂载策略
            # 管线（包装 OCR 模型逐页改写 scale）。默认关闭——标准管线 +
            # 构建时固化 scale，默认路径零额外开销（无包装、无预分析）。
            fmt_kwargs: dict = {"pipeline_options": pdf_opts}
            if _ocr_scale_dynamic:
                policy_pipeline_cls = make_scale_policy_pipeline_cls()
                if policy_pipeline_cls is not None:
                    fmt_kwargs["pipeline_cls"] = policy_pipeline_cls
                    logger.info(
                        "OCR scale 动态策略启用（表格页 %.1f / 其余 %.1f）",
                        _ocr_scale_table, _ocr_scale_base,
                    )
            converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(**fmt_kwargs)}
            )
            logger.info("docling DocumentConverter 已构建（布局/表格走 MPS，OCR 走 Apple Vision/ANE，scale=%.1f）", _ocr_scale_base)
        except Exception as e:
            # docling 版本差异（2.13x 前后多次移动模块路径）：拿不到
            # 降配入口就用默认构造——正确性不受影响，只是 macOS 上
            # MPS 缓存可能再次推高内存。
            logger.info("docling 管线降配不可用（%s），使用默认配置", e)
            converter = DocumentConverter()
        _start_idle_reaper()
        return converter
    except ImportError:
        logger.info("docling not installed; falling back to per-format parsers")
        _converter_unavailable = True
        return None
    except Exception as e:
        # 只在**缺失**时永久禁用。瞬时错误（模型下载中断、磁盘抖动、
        # OOM）绝不能置位 —— 那会让本进程后续所有同步都静默降级到轻量
        # 解析器，即使原因早已消失了也不自愈。
        logger.warning("failed to init docling converter (will retry): %s", e)
        return None


# 实例池（工厂 lambda 延迟绑定 _get_converter——测试 monkeypatch 它时
# 池会在下次建实例时拿到假工厂）。
_pool = _ConverterPool(lambda: _get_converter(), max_size=_max_parallel)


def extract_markdown(filepath: str) -> Optional[str]:
    """Convert a document to Markdown using docling.

    Returns the Markdown text, or ``None`` if docling is unavailable or the
    format is not handled by docling (callers should fall back to the
    per-format parsers).
    """
    if not is_docling_available():
        return None

    path = Path(filepath)
    suffix = path.suffix.lower()
    if suffix not in _DOCLING_EXTENSIONS:
        return None

    # Image-based PDF handling: docling OCRs image pages via RapidOCR, but a
    # 图片型 PDF（扫描件/满页背景图的电子书）**不再拦截**：设计上所有 PDF
    # 统一走 docling（内部 OCR 优先 Apple Vision/ocrmac，输出版面良好的
    # Markdown）；PyMuPDF+RapidOCR 只是 docling 不可用/转换失败时的兜底。
    # 旧的「扫描件/图片重 → 转投 PyMuPDF」拦截（#116）已按用户决策撤销——
    # 当时 30GB+/40-70min 的失控根因（MPS 加速器、无超时、无内存约束）都已
    # 修复：accelerator 强制 CPU、document_timeout 硬超时、MLX/缓存上限。

    # 大小分流（2026-10-22 用户定版）：大 PDF 独占全部实例防内存叠加；小
    # PDF 并行转换（≤ rag_parse_parallel 路实例）。动态 scale 策略会改写
    # 全局 _scale_policy（单个模块级变量），并发转换会互相踩——开着动态
    # 策略时一律独占串行（默认关，不影响并行）。
    try:
        _large = (suffix == ".pdf"
                  and path.stat().st_size >= _large_file_bytes)
    except OSError:
        _large = False
    exclusive = _large or _ocr_scale_dynamic

    # OCR scale 动态策略（#132，可配置）：仅 rag_ocr_scale_dynamic 开启时
    # convert 前预分析 PDF（文本层/原生 DPI/表格启发式），表格页逐页渲染
    # 提到 rag_ocr_scale_table，其余用 rag_ocr_scale_base。默认关闭——此处
    # 直接跳过，convert 前零开销（scale 已在 converter 构建时固化）。
    _scale_policy = (_analyze_ocr_scale(path)
                     if suffix == ".pdf" and _ocr_scale_dynamic else None)

    # 看门狗心跳：图片书走 docling 全量管线耗时数分钟到十几分钟且期间无任何
    # 输出，UI 只能看到「解析文件: xxx」一条日志挂着不动，会被误判为卡死
    # （#131）。守护线程每 60s 报一次已耗时；convert 返回/异常时自动停止。
    # 2026-10-21：MPS 布局 + ANE OCR + scale 2.0 后单本书实测 ~6-10 分钟，
    # 文案改 10-20 分钟（留足大书/复杂版面余量）。
    _convert_done = threading.Event()

    def _heartbeat() -> None:
        t0 = time.monotonic()
        while not _convert_done.wait(60.0):
            logger.info(
                "docling 仍在转换 %s（已 %.0f 分钟，图片书 10-20 分钟属正常）",
                path.name, (time.monotonic() - t0) / 60.0,
            )

    threading.Thread(target=_heartbeat, name="docling-heartbeat", daemon=True).start()

    try:
        # 实例池：小文件各自租独立实例并行转换（DocumentConverter 非线程
        # 安全，单实例并发会状态污染/崩溃——旧单例全程持锁正是 PDF 解析
        # 并发提不上去的原因）；大文件独占全部槽位防内存叠加。
        _set_scale_policy(_scale_policy)
        try:
            with _pool.lease(exclusive) as converter:
                if converter is None:
                    return None
                result = converter.convert(str(path))
                doc = result.document
                markdown = doc.export_to_markdown()
        finally:
            _set_scale_policy(None)
        # 及时释放大对象引用，避免整棵文档树在切片前一直常驻。
        del result, doc
        if not markdown or not markdown.strip():
            logger.warning("docling extracted empty markdown from %s", path.name)
            return None
        return markdown
    except Exception as e:
        # Never let a docling failure abort ingestion — caller falls back.
        logger.warning("docling conversion failed for %s: %s", path.name, e)
        return None
    finally:
        _convert_done.set()
        # MPS 缓存护栏（#131）：torch 的 MPS caching allocator 只借不还，
        # 逐页激活碎片会在长书转换间累积（#116 的 36GB 根因）。每次转换
        # 结束立刻把空闲缓冲归还系统——60 页 A/B 实测峰值平坦（2.58GB）。
        try:
            import torch
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass


