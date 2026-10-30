"""闲时后台逐章摘要：结合章节目录与原书 Markdown，用辅助引擎逐章生成摘要。

功能（2026-10-29 需求）：
  * 闲时后台自动为知识库里每本书逐章生成 **50-100 字**的章节摘要；
  * 全部章节完成后，结合章节目录结构把各章摘要汇总成 **全书摘要**；
  * 逐章处理过程中发现 documents 表里的目录结构与实际 Markdown 标题不符时，
    顺便以实际结构补充完善目录（写回 documents 表）；
  * **状态维护 + 增量**：进度落盘 ``{workspace}/rag/chapter_summaries.json``，
    已提取过的章节绝不重复提取；原书内容变化（mtime/大小变化）自动重置该书；
  * **辅助引擎策略**：无论辅助引擎「启用」与否都执行本任务——
      - 辅助引擎已加载可用（UI 持有实例）→ 直接复用，**每 15 分钟一章**
        （不独占引擎资源：一次只生成一章，生成期与 agent 对话共用同一
        ``_gen_lock`` 天然串行）；
      - 未加载/不可用 → **临时加载**，生成一章后立即 ``close()`` 卸载释放，
        **每 5 分钟一章**。

调度模型：自带 daemon 线程，每 ``_TICK_SECONDS`` 醒来一次。只有当注入的
``is_idle()`` 为真（UI 侧：agent 非忙、非同步、无相关 worker 在跑）且到达
下次执行时间才处理一章。全部工作都在「一次一章」的粒度上，单次失败只推迟
（冷却期），不影响主流程。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..atomic_io import atomic_write_json, backup_corrupt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 复用已加载的辅助引擎：每 15 分钟一章（用户明确要求：严格控制频次、不独占
# 辅助引擎资源）。临时加载模式：每 5 分钟一章（用完即卸载，无常驻占用）。
_INTERVAL_SHARED_AUX = 15 * 60
_INTERVAL_TEMP_AUX = 5 * 60
# 调度线程唤醒周期：轻量检查（空闲守卫 + 到期判断），不触发任何模型/IO 重活。
_TICK_SECONDS = 30.0
# 单次失败（辅助模型异常 / 临时加载失败 / 输出为空）后的冷却期：避免每 30s
# 热循环重试同一章把日志刷爆。
_FAILURE_COOLDOWN = 5 * 60
# 全文摘要（不再截断章节正文）：按辅助引擎上下文窗口决定「整章直摘」还是
# 「map-reduce 分块摘要再汇总」。
#  * 上下文窗口探测不到时的兜底假设（token）：保守取 8k，保证不溢出本地模型。
#  * 输出预算 + 系统/模板开销预留：实际可用于章节正文输入的 token 数 =
#    ctx_tokens - _SUMMARY_OUT_TOKENS - _SUMMARY_SYS_TOKENS。
#  * token→char 用中文保守系数（中文约 1.6 char/token，留余量取 1.8）。
_DEFAULT_AUX_CTX = 8_192
_SUMMARY_OUT_TOKENS = 130        # 章摘要输出上限（50-100 字 ≈ 60-110 token）
_SUMMARY_SYS_TOKENS = 200        # 系统提示 + chat 模板固定开销预留
_CHARS_PER_TOKEN = 1.8
# map-reduce 分块下限：单块至少这么长才有独立摘要意义（否则直接并入前块）。
_MIN_BLOCK_CHARS = 400
# 章节摘要策略：正文超过此阈值（字）强制走 map-reduce，避免单段摘要因篇章过长而
# 质量下降（保证摘要效果）；不超过则整章直摘。
_MAP_REDUCE_THRESHOLD = 4000
# map-reduce 每块目标字数：每 3000 字一个 map；并发 batching 由 aux.batch_chat 的
# batch_size（默认 5）封顶，同一时刻最多 5 条序列在跑，约束峰值 KV 显存。
_MAP_BLOCK_CHARS = 3000
# 全书摘要输入（目录 + 各章摘要）上限：通常远小于章正文，直接整段送。
_BOOK_INPUT_CHARS = 24_000
# 章摘要字数约束（同时写进 prompt 与输出校验：超长截断到 ~1.5 倍上限，过短
# 视为失败不落盘，下轮重试）。
_CHAPTER_SUMMARY_MIN = 20
_CHAPTER_SUMMARY_MAX = 160

_CHAPTER_SYSTEM = (
    "你是严谨的书籍章节摘要助手。根据给定的章节标题与正文，输出一段 50-100 字的"
    "中文内容摘要。\n"
    "要求：\n"
    "1. 只概括该章实际内容：核心主题、关键事实/论点/情节，不臆造、不评论；\n"
    "2. 输出纯文本摘要本体，不要任何前言、标题、引号或总结性客套话；\n"
    "3. 控制在 50-100 字。"
)

_BOOK_SYSTEM = (
    "你是严谨的书籍摘要助手。下面给出一本书的章节目录与各章摘要，"
    "请综合生成一段 100-300 字的**全书摘要**（整体主题、结构与核心内容脉络）。\n"
    "要求：输出纯文本摘要本体，不要前言、标题或客套话。"
)

# 状态文件名（位于 {workspace}/rag/ 下，与 manifest.json 同级）。
_STATE_FILENAME = "chapter_summaries.json"
_STATE_VERSION = 1


# ---------------------------------------------------------------------------
# 状态持久化（单写者：调度线程；UI/工具只读）
# ---------------------------------------------------------------------------

def state_path(workspace: str) -> str:
    return os.path.join(workspace, "rag", _STATE_FILENAME)


def load_summaries_state(workspace: str) -> Dict[str, Any]:
    """读取进度状态（损坏时备份原文件并返回空状态——摘要可再生，无损）。"""
    path = state_path(workspace)
    if not os.path.exists(path):
        return {"version": _STATE_VERSION, "books": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or not isinstance(data.get("books"), dict):
            raise ValueError("bad state root")
        return data
    except Exception as e:
        logger.warning("逐章摘要状态文件损坏，已备份并重建: %s (%s)", path, e)
        backup_corrupt(path)
        return {"version": _STATE_VERSION, "books": {}}


def _save_summaries_state(workspace: str, state: Dict[str, Any]) -> None:
    state["version"] = _STATE_VERSION
    state["updated_at"] = time.time()
    atomic_write_json(state_path(workspace), state)


class ChapterSummaryScheduler:
    """闲时后台逐章摘要调度器（自带 daemon 线程，UI 只需 start/stop）。

    参数均为注入的 callable，避免持有 UI/引擎/辅助引擎的强引用：
      * ``get_config``       → 当前 AgentConfig（辅助模型配置实时读取）；
      * ``get_engine``       → RAGEngine 或 None（UI 的 agent 存在时复用其引擎，
        避免重复打开 LanceDB；agent 不存在时返回 None，本轮跳过——**绝不**
        在后台隐式构造引擎触发 embedding 权重加载）；
      * ``get_shared_aux``   → 已加载的辅助 LLM client 或 None（UI 的
        ``_aux_llm``；返回 None 时走临时加载路径）；
      * ``is_idle``          → True 时才允许干活（UI 侧聚合
        非 _busy / 非 _sync_active / 无同步相关 worker）。
    """

    def __init__(
        self,
        get_config: Callable[[], Any],
        get_engine: Callable[[], Any],
        get_shared_aux: Callable[[], Any],
        is_idle: Callable[[], bool],
    ):
        self._get_config = get_config
        self._get_engine = get_engine
        self._get_shared_aux = get_shared_aux
        self._is_idle = is_idle
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._next_due = 0.0          # monotonic；0 = 立即可跑首次
        self._state_lock = threading.Lock()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="rag-chapter-summary")
        self._thread.start()
        logger.info("逐章摘要后台调度器已启动（闲时运行；已加载辅助引擎 15 分钟/章，"
                    "临时加载 5 分钟/章）")

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.wait(_TICK_SECONDS):
            try:
                if not self._is_idle():
                    continue
                now = time.monotonic()
                if now < self._next_due:
                    continue
                interval = self._process_next()
                if interval is not None:
                    self._next_due = time.monotonic() + interval
                # 无活可干 → 保持 tick 节奏，不推进 next_due。
            except Exception as e:
                logger.warning("逐章摘要后台任务异常（冷却 %.0f 秒后重试）: %s",
                               _FAILURE_COOLDOWN, e, exc_info=True)
                self._next_due = time.monotonic() + _FAILURE_COOLDOWN

    # ------------------------------------------------------------------
    # 单步工作：处理一个未完成章节（或收尾全书摘要）
    # ------------------------------------------------------------------

    def _process_next(self) -> Optional[float]:
        """找到第一本有未完成章节的书并处理一章。

        返回本次应采用的下次间隔（秒）；无活可干返回 None。
        只做轻量探测（stat + 状态表）定位目标，仅在真正处理时才读 Markdown。
        """
        engine = self._get_engine()
        if engine is None:
            return None
        cfg = self._get_config()
        workspace = getattr(cfg, "workspace", "") or ""
        if not workspace:
            return None
        # 从未同步过的库（无向量目录）绝不触发存储层初始化——避免后台隐式
        # 加载 embedding 权重。同步过一次后向量目录必然存在。
        if not os.path.isdir(os.path.join(workspace, "rag", "vectors")):
            return None

        docs = sorted(engine.list_documents() or [])
        if not docs:
            return None

        with self._state_lock:
            state = load_summaries_state(workspace)
            books: Dict[str, Any] = state.setdefault("books", {})

            for doc_name in docs:
                info = engine.get_document_outline(doc_name)
                if not info:
                    continue
                md_path = info.get("md_path") or ""
                if not md_path or not os.path.isfile(md_path):
                    continue
                book = books.get(doc_name)
                total = _chapter_count_from_toc(info.get("chapters"))
                if book is None or total <= 0:
                    # 新书：需要读 Markdown 提取章节（同时完成目录核对/修复）。
                    interval = self._process_book(engine, cfg, doc_name, md_path,
                                                  info, books)
                    if interval is not None:
                        _save_summaries_state(workspace, state)
                        return interval
                    continue
                # 已有进度：mtime/大小变化 → 原书更新，重置该书（摘要全部重做）。
                # 指纹用 list（与 JSON 往返后的存储类型一致，tuple 会永不匹配）。
                try:
                    st = os.stat(md_path)
                    fingerprint = [int(st.st_mtime), st.st_size]
                except OSError:
                    continue
                if book.get("md_fingerprint") not in (None, fingerprint):
                    logger.info("逐章摘要: 检测到《%s》内容已更新，重置该书进度", doc_name)
                    books.pop(doc_name, None)
                    interval = self._process_book(engine, cfg, doc_name, md_path,
                                                  info, books)
                    if interval is not None:
                        _save_summaries_state(workspace, state)
                        return interval
                    continue
                done = {c.get("idx") for c in book.get("chapters", [])}
                if len(done) < total:
                    interval = self._process_book(engine, cfg, doc_name, md_path,
                                                  info, books)
                    if interval is not None:
                        _save_summaries_state(workspace, state)
                        return interval
                    continue
                # 章节全部完成 → 收尾全书摘要（一次）。
                if not book.get("book_summary"):
                    aux, mode = self._acquire_aux()
                    try:
                        summary = self._generate_book_summary(
                            aux, json.loads(info.get("chapters") or "[]"),
                            book.get("chapters", []))
                    finally:
                        self._release_aux(aux, mode)
                    if not summary:
                        return _FAILURE_COOLDOWN
                    book["book_summary"] = summary
                    book["book_summary_at"] = time.time()
                    logger.info("逐章摘要: 《%s》全书摘要已生成 (%d 字)",
                                doc_name, len(summary))
                    _save_summaries_state(workspace, state)
                    return _interval_for(mode)
            return None

    # ------------------------------------------------------------------
    # 处理一本书的一个章节
    # ------------------------------------------------------------------

    def _process_book(
        self,
        engine: Any,
        cfg: Any,
        doc_name: str,
        md_path: str,
        info: Dict[str, Any],
        books: Dict[str, Any],
    ) -> Optional[float]:
        """读取 Markdown → 核对/修复目录 → 摘要第一个未完成章节。

        返回下次间隔；该书当前无事可做（提取不到章节）返回 None。
        """
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                markdown = f.read()
        except OSError as e:
            logger.warning("逐章摘要: 读取《%s》缓存 Markdown 失败: %s", doc_name, e)
            return None
        if not markdown.strip():
            return None

        from .document_outline import extract_chapters
        chapters = extract_chapters(markdown)
        if not chapters:
            # documents 表有目录但原文提取不到章节：markdown 可能损坏/格式异常。
            # 静默返回会令该书永久「待处理却做不了」且无日志——明确告警一次。
            if _safe_json_list(info.get("chapters")):
                logger.warning(
                    "逐章摘要: 《%s》documents 表有目录但原文提取不到章节"
                    "（markdown 可能损坏或格式异常），跳过该书", doc_name)
            return None

        book = books.setdefault(doc_name, {})
        try:
            st = os.stat(md_path)
            # 用 list 而非 tuple：状态会经 JSON 落盘（tuple → list），两侧
            # 类型一致才能正确比对（否则每次都误判「内容已更新」而重置进度）。
            book["md_fingerprint"] = [int(st.st_mtime), st.st_size]
        except OSError:
            pass
        book.setdefault("content_hash",
                        hashlib.sha256(markdown.encode("utf-8")).hexdigest()[:16])

        # ---- 目录结构核对 / 顺便修复 ------------------------------------
        # documents 表里的目录来自同步时的一次性提取；实际 Markdown 才是真源。
        # 两者标题序列不一致（缺章/改名/层级漂移）→ 以实际为准写回。
        toc_actual = [{"level": c.level, "title": c.title} for c in chapters]
        toc_stored = _safe_json_list(info.get("chapters"))
        if not book.get("toc_fixed") and _toc_differs(toc_stored, toc_actual):
            try:
                engine.build_and_store_digest(
                    doc_name, markdown="", md_path=md_path,
                    digest=info.get("digest") or "",
                    chapters=toc_actual,
                )
                book["toc_fixed"] = True
                # 目录结构变化（章节增删 / 顺序调整）会让「按 idx 记录」的逐章
                # 摘要错位到错误章节 → 已有摘要作废，全书摘要随之失效需重做。
                # 仅标题改名（数量与顺序不变）不影响 idx 对齐，可保留摘要。
                if book.get("chapters") and _toc_structurally_changed(
                        toc_stored, toc_actual):
                    logger.info(
                        "逐章摘要: 《%s》目录结构变化导致章节 idx 错位，"
                        "重置该书已生成的逐章摘要重做", doc_name)
                    book["chapters"] = []
                    book["book_summary"] = ""
                    book["book_summary_at"] = 0
                logger.info(
                    "逐章摘要: 《%s》目录结构与原书不符，已按实际标题修正 "
                    "(表内 %d 章 → 实际 %d 章)",
                    doc_name, len(toc_stored), len(toc_actual))
            except Exception as e:
                logger.warning("逐章摘要: 《%s》目录修复失败（不阻塞摘要）: %s",
                               doc_name, e)

        # ---- 逐章摘要（增量） -------------------------------------------
        done = {c.get("idx") for c in book.get("chapters", [])}
        target = next((i for i in range(len(chapters)) if i not in done), None)
        if target is None:
            return None  # 全部完成（全书摘要由调用方收尾）

        ch = chapters[target]
        aux, mode = self._acquire_aux()
        try:
            summary = self._generate_chapter_summary(aux, ch)
        finally:
            self._release_aux(aux, mode)
        if not summary:
            logger.warning("逐章摘要: 《%s》第 %d 章摘要生成失败（冷却重试）",
                           doc_name, target)
            return _FAILURE_COOLDOWN
        book.setdefault("chapters", []).append({
            "idx": target,
            "title": ch.title,
            "level": ch.level,
            "summary": summary,
            "done_at": time.time(),
        })
        logger.info("逐章摘要: 《%s》第 %d/%d 章「%s」摘要完成 (%d 字, 辅助引擎=%s)",
                    doc_name, target + 1, len(chapters), ch.title[:30],
                    len(summary), mode)
        return _interval_for(mode)

    # ------------------------------------------------------------------
    # 辅助引擎获取 / 释放
    # ------------------------------------------------------------------

    def _acquire_aux(self) -> Tuple[Any, str]:
        """优先复用已加载实例（shared 模式）；不可用则临时加载（temp 模式）。"""
        shared = None
        try:
            shared = self._get_shared_aux()
        except Exception:
            shared = None
        if shared is not None:
            return shared, "shared"
        from ..local.factory import build_aux_llm_client
        client = build_aux_llm_client(self._get_config())
        logger.info("逐章摘要: 辅助引擎未加载，已临时加载（用完即卸载）")
        return client, "temp"

    @staticmethod
    def _release_aux(aux: Any, mode: str) -> None:
        """temp 模式用完立即卸载释放；shared 模式绝不关闭（归 UI 管理）。"""
        if mode != "temp" or aux is None:
            return
        try:
            aux.close()
            logger.info("逐章摘要: 临时辅助引擎已卸载释放")
        except Exception as e:
            logger.warning("逐章摘要: 临时辅助引擎卸载失败: %s", e)

    # ------------------------------------------------------------------
    # LLM 调用
    # ------------------------------------------------------------------

    @staticmethod
    def _llm_text(aux: Any, system: str, user: str) -> str:
        """流式调用辅助模型并收集纯文本（与 postprocess._stream_collect 同款）。"""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        out: List[str] = []
        for ev in aux.chat_stream(messages, tools=None, temperature=0.2,
                                  kv_group="aux", enable_thinking=False):
            if ev.type == "content":
                out.append(ev.content)
            elif ev.type == "done":
                break
        return "".join(out).strip()

    # ------------------------------------------------------------------
    # 辅助引擎上下文窗口探测
    # ------------------------------------------------------------------

    @staticmethod
    def _aux_context_tokens(aux: Any) -> int:
        """探测辅助引擎上下文窗口（token），探不到用兜底假设。

        优先读显式属性；本地客户端（local/llm.py）未直接暴露 context_length，
        但存了 ``_max_tokens``——GGUF 路径据此推导 n_ctx（``max(8192,
        min(_max_tokens+1024, 131072))``），MLX 路径 _max_tokens 即生成上限、
        上下文通常更大，用同一推导式作为保守估计足够。
        """
        if aux is None:
            return _DEFAULT_AUX_CTX
        for attr in ("context_length", "context_window", "n_ctx",
                     "max_context_length"):
            v = getattr(aux, attr, None)
            if isinstance(v, int) and v > 0:
                return v
        mt = getattr(aux, "_max_tokens", None)
        if isinstance(mt, int) and mt > 0:
            return max(8192, min(mt + 1024, 131072))
        return _DEFAULT_AUX_CTX

    def _generate_chapter_summary(self, aux: Any, chapter: Any) -> str:
        """单章**全文**摘要（不截断章节正文，保证质量）。

        策略（保证摘要效果）：
          * 章节正文 ≤ 4000 字 → 整章直摘（单段摘要，强制关思考）；
          * 章节正文 > 4000 字 → 强制 **map-reduce**：每 3000 字一个 map 各自
            摘要，再用 batching（并发上限 5）并行生成，最后汇总成 50-100 字章摘要。
        块摘要与最终摘要都走同一套字数校验（过短判失败、过长截断）。
        """
        text = (getattr(chapter, "text", "") or "").strip()
        if not text:
            return ""

        ctx = self._aux_context_tokens(aux)
        budget_tokens = max(256, ctx - _SUMMARY_OUT_TOKENS - _SUMMARY_SYS_TOKENS)
        budget_chars = max(800, int(budget_tokens * _CHARS_PER_TOKEN))
        # map 块大小：每 3000 字一个 map，且不超过模型单次输入预算（极端小上下文
        # 模型保底，避免单块超出窗口）。
        map_block = min(_MAP_BLOCK_CHARS, budget_chars)

        # 正文超过 4000 字（保证摘要效果）或超长超出模型单次预算 → 强制 map-reduce。
        if len(text) > _MAP_REDUCE_THRESHOLD or len(text) > budget_chars:
            blocks = self._split_blocks(text, map_block)
            block_summaries = self._map_summarize_blocks(
                aux, chapter.title, chapter.level, blocks)
            if not block_summaries:
                return ""
            if len(block_summaries) == 1:
                return block_summaries[0]
            return self._summarize_blocks_combined(
                aux, chapter.title, chapter.level, block_summaries)

        # 短章节：整章直摘（单段摘要，强制关思考）。
        return self._summarize_block(aux, chapter.title, chapter.level, text)

    @staticmethod
    def _split_blocks(text: str, budget_chars: int) -> List[str]:
        """把全文切成若干 ≤ budget_chars 的块（按段落边界，落单超长段硬切）。"""
        # 优先按空行分段；若正文无空行则用换行；都无则整段。
        paras = [p for p in text.split("\n\n") if p.strip()]
        if len(paras) <= 1:
            paras = [p for p in text.split("\n") if p.strip()]
        if not paras:
            return [text]
        blocks: List[str] = []
        buf = ""
        for p in paras:
            # 单个段落本身就超过预算：按句/按字符硬切。
            if len(p) > budget_chars:
                hard = ChapterSummaryScheduler._hard_split(p, budget_chars)
                for h in hard:
                    if buf:
                        blocks.append(buf)
                        buf = ""
                    blocks.append(h)
                continue
            if buf and len(buf) + len(p) + 2 > budget_chars:
                blocks.append(buf)
                buf = p
            else:
                buf = (buf + "\n\n" + p) if buf else p
        if buf:
            blocks.append(buf)
        # 丢弃过短尾块（并入上一块，避免无效小块摘要）。
        merged: List[str] = []
        for b in blocks:
            if merged and len(b) < _MIN_BLOCK_CHARS:
                merged[-1] = merged[-1] + "\n\n" + b
            else:
                merged.append(b)
        return merged

    @staticmethod
    def _hard_split(text: str, budget_chars: int) -> List[str]:
        """超长单段硬切：尽量在句末（。！？；）切，否则按字符。"""
        out: List[str] = []
        cur = ""
        for ch in text:
            cur += ch
            if len(cur) >= budget_chars and ch in "。！？；\n":
                out.append(cur)
                cur = ""
        if cur:
            out.append(cur)
        return out

    def _summarize_block(self, aux: Any, title: str, level: int, text: str) -> str:
        """单个文本块摘要（50-100 字）；过短判失败、过长截断。"""
        user = (f"章节标题：{title}\n"
                f"标题层级：h{level}\n\n"
                f"待概括文本：\n{text}")
        summary = _strip_wrappers(self._llm_text(aux, _CHAPTER_SYSTEM, user))
        if len(summary) < _CHAPTER_SUMMARY_MIN:
            return ""
        if len(summary) > _CHAPTER_SUMMARY_MAX:
            summary = summary[:_CHAPTER_SUMMARY_MAX].rstrip() + "…"
        return summary

    def _map_summarize_blocks(
        self,
        aux: Any,
        title: str,
        level: int,
        blocks: List[str],
    ) -> List[str]:
        """map 阶段：各文本块摘要。

        优先走辅助模型的 **Batching 模式**（`aux.batch_chat` 连续批处理多分片，
        单模型共享预填充、并行解码，远快于逐条串行）；辅助客户端不支持批量时
        退回逐条串行。返回非空摘要列表（保序）。
        """
        if not blocks:
            return []
        # 与各块对应的 messages（与 _summarize_block 同款 system/user）。
        messages_list = []
        for i, blk in enumerate(blocks):
            user = (f"章节标题：{title}\n标题层级：h{level}\n\n"
                    f"待概括文本：\n{blk}")
            messages_list.append([
                {"role": "system", "content": _CHAPTER_SYSTEM},
                {"role": "user", "content": user},
            ])

        if hasattr(aux, "batch_chat"):
            try:
                raw = aux.batch_chat(
                    messages_list, temperature=0.2, max_tokens=200,
                    enable_thinking=False)
            except Exception as e:  # 批量失败 → 退回逐条串行，保证健壮
                logger.warning("逐章摘要: batch_chat 失败，退回串行: %s", e)
                raw = [self._summarize_block(aux, title, level, b) for b in blocks]
        else:
            raw = [self._summarize_block(aux, title, level, b) for b in blocks]

        summaries: List[str] = []
        for i, s in enumerate(raw):
            s = _strip_wrappers(s or "").strip()
            if len(s) < _CHAPTER_SUMMARY_MIN:
                s = ""
            elif len(s) > _CHAPTER_SUMMARY_MAX:
                s = s[:_CHAPTER_SUMMARY_MAX].rstrip() + "…"
            if s:
                summaries.append(s)
            else:
                logger.warning(
                    "逐章摘要: 第%d/%d段摘要为空（失败），跳过",
                    i + 1, len(blocks))
        return summaries

    def _summarize_blocks_combined(
        self,
        aux: Any,
        title: str,
        level: int,
        block_summaries: List[str],
    ) -> str:
        """map 阶段各块摘要 → 最终章摘要（50-100 字，覆盖全章脉络）。"""
        joined = "\n".join(f"[{i + 1}] {s}"
                           for i, s in enumerate(block_summaries))
        user = (f"章节标题：{title}\n标题层级：h{level}\n\n"
                f"下面是该章各分段摘要，请综合成**一段** 50-100 字的整章摘要"
                f"（保留核心主题与关键脉络，不要罗列分段）：\n{joined}")
        summary = _strip_wrappers(self._llm_text(aux, _CHAPTER_SYSTEM, user))
        if len(summary) < _CHAPTER_SUMMARY_MIN:
            # 汇总失败不致命：退回各块摘要拼接（仍是有意义的整章概览）。
            return "。".join(block_summaries)[:_CHAPTER_SUMMARY_MAX]
        if len(summary) > _CHAPTER_SUMMARY_MAX:
            summary = summary[:_CHAPTER_SUMMARY_MAX].rstrip() + "…"
        return summary

    def _generate_book_summary(
        self,
        aux: Any,
        toc: List[Dict[str, Any]],
        chapter_summaries: List[Dict[str, Any]],
    ) -> str:
        """全书摘要：目录结构 + 各章摘要 → 100-300 字。"""
        by_idx = {c.get("idx"): c for c in chapter_summaries}
        parts: List[str] = []
        for i, entry in enumerate(toc):
            s = by_idx.get(i, {}).get("summary", "")
            title = entry.get("title", f"章节{i + 1}")
            if s:
                parts.append(f"- {title}：{s}")
            else:
                parts.append(f"- {title}")
        doc = "\n".join(parts)
        # 同样尊重上下文窗口：全书摘要输入（目录+各章摘要）可能很长，按其
        # 预算截断，避免小窗口辅助引擎溢出。
        budget_tokens = max(256, self._aux_context_tokens(aux)
                            - _SUMMARY_OUT_TOKENS - _SUMMARY_SYS_TOKENS)
        budget_chars = max(800, int(budget_tokens * _CHARS_PER_TOKEN))
        cap = min(_BOOK_INPUT_CHARS, budget_chars)
        if len(doc) > cap:
            doc = doc[:cap].rstrip() + "\n…（章节过多，已截断）"
        return _strip_wrappers(self._llm_text(aux, _BOOK_SYSTEM, doc))


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _interval_for(mode: str) -> float:
    return _INTERVAL_SHARED_AUX if mode == "shared" else _INTERVAL_TEMP_AUX


def _safe_json_list(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return raw
    try:
        data = json.loads(raw or "[]")
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _chapter_count_from_toc(raw: Any) -> int:
    toc = _safe_json_list(raw)
    return len(toc) if toc else 0


def _toc_differs(stored: List[Dict[str, Any]],
                 actual: List[Dict[str, Any]]) -> bool:
    """目录结构是否需要修复：标题序列（含层级）不一致即视为有问题。"""
    if len(stored) != len(actual):
        return True
    for s, a in zip(stored, actual):
        if (s.get("title") or "") != (a.get("title") or ""):
            return True
        if (s.get("level") or 0) != (a.get("level") or 0):
            return True
    return False


def _toc_structurally_changed(stored: List[Dict[str, Any]],
                             actual: List[Dict[str, Any]]) -> bool:
    """目录「数量或层级顺序」是否变化（决定逐章摘要 idx 是否错位）。

    仅标题改名（数量与层级顺序不变）不影响按位置(idx)记录摘要的对齐，不算；
    章节增删或层级顺序调整会让 idx→章节 映射错位，必须作废旧摘要重做。
    """
    if len(stored) != len(actual):
        return True
    for s, a in zip(stored, actual):
        if (s.get("level") or 0) != (a.get("level") or 0):
            return True
    return False


def _strip_wrappers(text: str) -> str:
    """去掉模型输出常见的引号/前后缀包装。"""
    t = (text or "").strip()
    if len(t) >= 2 and t[0] in "\"“”'「『" and t[-1] in "\"“”'」』":
        t = t[1:-1].strip()
    for prefix in ("摘要：", "摘要:", "本章摘要：", "本章摘要:"):
        if t.startswith(prefix):
            t = t[len(prefix):].strip()
    return t
