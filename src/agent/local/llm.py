"""本地 LLM 后端：MLX / ONNX / GGUF。

对外暴露的 ``LocalLLMClient`` 与 ``llm_client.LLMClient`` **同接口**：
``chat_stream(messages, tools, temperature) -> Iterator[StreamEvent]`` +
``close()``。内核（kernel/loop.py）与 Agent 只依赖这个鸭子类型，因此主模型
可以在「网络 API」与「应用内本地加载」之间切换而不改任何调用方。

三种后端：
    mlx   — mlx-lm：``load`` + ``stream_generate``，采样参数走
             ``make_sampler`` / ``make_logits_processors``。
    gguf  — llama-cpp-python：``create_chat_completion(stream=True)``，
             返回的就是 OpenAI 兼容 chunk，转换成本模块的 StreamEvent。
    onnx — onnxruntime 直接跑导出的 decoder（带 KV cache），采样用 numpy
            自己实现（temperature / top_p / top_k / repetition penalty）。
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# StreamEvent 数据类与 LLMClient 同源，保证内核解析一致。
from ..llm_client import StreamEvent  # noqa: E402


def _kv_diag(group, cached, prompt, reuse, suffix, ttft, total,
             common, thread_mismatch, first_cached, first_prompt,
             div_cached, div_prompt):
    """诊断：每轮 KV 复用情况写到 /tmp/base-agent-kv.log，排查「时快时慢」。

    UI 日志桥接的 Filter 只放行 RAG 模块，且 root 级别是 WARNING，故 KV 行
    到不了界面/主日志。单独落盘，事后可直接读取各轮复用与首字延迟。
    """
    try:
        import datetime as _dt
        with open(os.path.join("/tmp", "base-agent-kv.log"), "a",
                  encoding="utf-8") as _f:
            _f.write(
                "%s group=%-4s cached=%d prompt=%d reuse=%d suffix=%d "
                "common=%d thread_mismatch=%s ttft=%.2fs total=%.2fs\n"
                "    cached[0:16]=%r\n    prompt[0:16]=%r\n"
                "    >>> fork@%d cached:~%r\n    >>> fork@%d prompt:~%r\n" % (
                    _dt.datetime.now().strftime("%H:%M:%S"), group, cached,
                    prompt, reuse, suffix, common, thread_mismatch, ttft,
                    total, first_cached, first_prompt, common, div_cached,
                    common, div_prompt))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 工具调用解析：把模型输出的文本转成 OpenAI 风格 tool_calls
# ---------------------------------------------------------------------------

_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE
)
# Qwen XML: <function=name><parameter=k>v</parameter>...</function>
_QWEEN_FUNC_RE = re.compile(
    r"<function=([^>]+)>(.*?)</function>", re.DOTALL | re.IGNORECASE
)
_QWEEN_PARAM_RE = re.compile(
    r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL | re.IGNORECASE
)
# Hermes / 通用 JSON: {"name": ..., "arguments": {...}}（arguments 可为字符串）
_JSON_CALL_RE = re.compile(r"\{[^{}]*\"name\"\s*:.*?\}", re.DOTALL)
# 残留的孤立 think 标签字面量（模型偶发不配对输出）——从正文里剔除。
_STRAY_THINK_RE = re.compile(r"<think>|</think>")


def _new_tool_call_id(n: int) -> str:
    return f"local_call_{n}"


# ---------------------------------------------------------------------------
# 双视图历史：raw_content → 忠实于生成的 prompt 序列
# ---------------------------------------------------------------------------

# assistant 轮次在 ChatML 里的起始标记。模板在它之后预置 think 片段
# （enable_thinking=True 时 "<think>\n"，False 时空 think 块）。
_ASSISTANT_MARK = "<|im_start|>assistant\n"

# 跨轮 KV 缓存的 token 上限。缓存随会话线性增长（每层每 token 都要留 KV），
# 超过就整体重建，避免长会话把内存吃穿。
_KV_MAX_TOKENS = 32768


def _assistant_prompt_prefix(prompt: str) -> str:
    """取模板在最后一个 assistant 起始标记后预置的片段。

    这段属于「生成序列」的头部：生成时它就在 prompt 里（不是模型吐出来的），
    所以重建历史时必须一并带上，否则前缀会差这几个 token 而对不上。
    """
    idx = prompt.rfind(_ASSISTANT_MARK)
    if idx < 0:
        return ""
    return prompt[idx + len(_ASSISTANT_MARK):]


def _rebuild_raw_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把带 ``raw_content`` 的 assistant 消息重建为「忠实于生成」的形状。

    Qwen 模板有个暗门：``message.reasoning_content`` 是**字符串**时跳过自动
    拆分（否则它会把 content 里的 ``<think>…</think>`` 摘走）；而历史 assistant
    分支直出 content。于是把 raw 放进 content、再塞一个字符串 reasoning_content，
    模板就会原样吐出 raw——含 think 标签与工具调用 XML，与生成时的 token 序列
    逐 token 一致（真实 tokenizer 实测 2600/2600 对齐）。

    没有 raw_content 的消息（HTTP 后端历史、user/tool 消息）原样返回，
    因此对 HTTP 后端零影响。
    """
    if not messages:
        return messages
    out: List[Dict[str, Any]] = []
    changed = False
    for m in messages:
        raw = m.get("raw_content") if isinstance(m, dict) else None
        if (isinstance(m, dict) and m.get("role") == "assistant"
                and isinstance(raw, str) and raw):
            nm = dict(m)
            nm["content"] = raw
            nm["reasoning_content"] = ""
            nm.pop("raw_content", None)
            out.append(nm)
            changed = True
        else:
            out.append(m)
    return out if changed else messages


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """从模型输出里抽取工具调用，返回 OpenAI 风格的 tool_calls 列表。

    支持三种常见格式：
      1. Qwen XML   ``<tool_call><function=name><parameter=k>v</parameter></function></tool_call>``
      2. Hermes JSON ``<tool_call>{"name":...,"arguments":{...}}</tool_call>``
      3. 裸 JSON     ``{"name":...,"arguments":{...}}`` 或 ``[{...}]``
    """
    calls: List[Dict[str, Any]] = []
    seen_xml = False

    # 1) Qwen XML
    for block in _TOOL_CALL_BLOCK_RE.findall(text):
        for fn_name, fn_body in _QWEEN_FUNC_RE.findall(block):
            fn_name = fn_name.strip()
            args: Dict[str, Any] = {}
            for pk, pv in _QWEEN_PARAM_RE.findall(fn_body):
                args[pk.strip()] = pv.strip()
            calls.append({
                "id": _new_tool_call_id(len(calls)),
                "type": "function",
                "function": {"name": fn_name, "arguments": json.dumps(args, ensure_ascii=False)},
            })
            seen_xml = True

    # 2) Hermes JSON 嵌在 <tool_call> 里
    for block in _TOOL_CALL_BLOCK_RE.findall(text):
        if seen_xml and _QWEEN_FUNC_RE.search(block):
            continue
        try:
            obj = json.loads(block.strip())
            calls.append(_dict_to_tool_call(obj, len(calls)))
        except (json.JSONDecodeError, TypeError):
            pass

    # 3) 裸 JSON（不在任何 <tool_call> 标签内也要抓）
    if not calls:
        for m in _JSON_CALL_RE.finditer(text):
            try:
                obj = json.loads(m.group(0))
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(obj, dict) and "name" in obj:
                calls.append(_dict_to_tool_call(obj, len(calls)))

    return calls


def _dict_to_tool_call(obj: Any, idx: int) -> Dict[str, Any]:
    """把 {'name':..., 'arguments':...} 规整成 OpenAI 风格 tool_call 槽位。"""
    name = str(obj.get("name", ""))
    raw = obj.get("arguments", {})
    if isinstance(raw, str):
        try:
            args = raw if raw.strip().startswith(("{", "[")) else json.dumps(
                {"value": raw}, ensure_ascii=False)
        except Exception:
            args = raw
    else:
        args = json.dumps(raw, ensure_ascii=False)
    return {
        "id": _new_tool_call_id(idx),
        "type": "function",
        "function": {"name": name, "arguments": args},
    }


# ---------------------------------------------------------------------------
# think 标签增量拆分：把流式 delta 分为 reasoning / content
# ---------------------------------------------------------------------------

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_TAG_PREFIX_LEN = max(len(_THINK_OPEN), len(_THINK_CLOSE)) - 1


class _ThinkSplitter:
    """增量地把文本流拆成 (kind, segment) 对，处理跨 chunk 的 ``<`` 标签边界。

    流式 delta 可能把标签（``<`` / ``</``）切碎（例如 ``<``、``<t``、``<thi``…）。
    每个 feed 只「安全地」消费到最后一个**不足以构成半截标签**的位置，剩余尾部
    留在 buffer 等下一个 delta——避免把半截标签当 content 发出、也避免漏判边界。
    """

    def __init__(self, start_in_think: bool = False) -> None:
        # Qwen3 系模板在 add_generation_prompt 时把 <think> 预填进生成提示，
        # 模型输出直接以推理正文开始、只含闭合标签——需以「已在 think 内」起步。
        self._in_think = bool(start_in_think)
        self._buf = ""

    def feed(self, delta: str) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        self._buf += delta
        i = 0
        n = len(self._buf)
        while i < n:
            tag = _THINK_CLOSE if self._in_think else _THINK_OPEN
            kind = "reasoning" if self._in_think else "content"
            # 完整标签落在当前位置 → 切换状态并跳过标签本身
            if self._buf.startswith(tag, i):
                self._in_think = not self._in_think
                i += len(tag)
                continue
            k = self._buf.find(tag, i)
            if k != -1:
                # 前方有完整标签：先把 [i, k) 作为当前 kind 发出，跳到标签处
                seg = self._buf[i:k]
                if seg:
                    out.append((kind, seg))
                i = k
                continue
            # 前方无完整标签：只消费到安全边界，保留可能的半截标签尾部给下次
            safe = n - (len(tag) - 1)
            if safe <= i:
                break
            seg = self._buf[i:safe]
            if seg:
                out.append((kind, seg))
            i = safe
            break
        self._buf = self._buf[i:]
        return out

    def flush(self) -> List[Tuple[str, str]]:
        buf, self._buf = self._buf, ""
        if not buf:
            return []
        return [("reasoning" if self._in_think else "content", buf)]


# ---------------------------------------------------------------------------
# 工具调用块增量剔除：把正文流里的 <tool_call>…


_TOOL_OPEN = "<" + "tool_call>"
_TOOL_CLOSE = "<" + "/tool_call>"


class _ToolCallSplitter:
    """增量剔除 content 流里的工具调用块（tool_call 标签对内）。"""

    def __init__(self) -> None:
        self._in_tool = False
        self._buf = ""

    def feed(self, delta: str) -> List[str]:
        out: List[str] = []
        self._buf += delta
        i, n = 0, len(self._buf)
        while i < n:
            tag = _TOOL_CLOSE if self._in_tool else _TOOL_OPEN
            if self._buf.startswith(tag, i):
                self._in_tool = not self._in_tool
                i += len(tag)
                continue
            k = self._buf.find(tag, i)
            if k != -1:
                if not self._in_tool:
                    seg = self._buf[i:k]
                    if seg:
                        out.append(seg)
                i = k
                continue
            safe = n - (len(tag) - 1)
            if safe <= i:
                break
            if not self._in_tool:
                seg = self._buf[i:safe]
                if seg:
                    out.append(seg)
            i = safe
            break
        self._buf = self._buf[i:]
        return out

    def flush(self) -> str:
        buf, self._buf = self._buf, ""
        if not buf or self._in_tool:
            return ""
        return buf


class _SummarySpanRouter:
    """识别 content 流里的 summary 协议区间并分流。

    yield ("content", text) —— 非摘要正文 + **完整标签区间**（content 通道
    保留完整 ``<summary>…</summary>`` 给内核 l2_summary / has_summary_tag）；
    yield ("summary", inner) —— 摘要**内文**（另发一份 reasoning 事件，让
    「模型自摘要」显示在思考面板）。
    跨 chunk 的部分标签按「消费到安全边界 + 尾部重扫」处理，任意切分都不会
    错切或泄漏。
    """

    _OPEN = "<summary>"
    _CLOSE = "</summary>"

    def __init__(self) -> None:
        self._inside = False
        self._span_buf = ""      # 跨 feed 累计的完整区间（含标签）
        self._buf = ""

    def feed(self, delta: str):
        self._buf += delta
        i, n = 0, len(self._buf)
        while i < n:
            if self._inside:
                j = self._buf.find(self._CLOSE, i)
                if j >= 0:
                    self._span_buf += self._buf[i:j + len(self._CLOSE)]
                    yield ("summary", self._buf[i:j])
                    yield ("content", self._span_buf)
                    self._span_buf = ""
                    i = j + len(self._CLOSE)
                    self._inside = False
                    continue
                safe = n - (len(self._CLOSE) - 1)
                if safe <= i:
                    break
                self._span_buf += self._buf[i:safe]
                yield ("summary", self._buf[i:safe])    # 内文流式外发
                i = safe
                break
            if self._buf.startswith(self._CLOSE, i):
                i += len(self._CLOSE)
                continue
            j = self._buf.find(self._OPEN, i)
            jc = self._buf.find(self._CLOSE, i)
            if jc != -1 and (j == -1 or jc < j):
                yield ("content", self._buf[i:jc])
                i = jc + len(self._CLOSE)
                continue
            if j != -1:
                yield ("content", self._buf[i:j])
                self._span_buf = self._OPEN
                self._inside = True
                i = j + len(self._OPEN)
                continue
            # 无完整标签：消费到两种标签共用的安全边界，尾部留待重扫
            safe = n - (max(len(self._OPEN), len(self._CLOSE)) - 1)
            if safe <= i:
                break
            yield ("content", self._buf[i:safe])
            i = safe
            break
        self._buf = self._buf[i:]

    def flush(self):
        buf, self._buf = self._buf, ""
        if buf:
            # 回合结束仍未闭合 → 不是协议标签，按正文还给 content 通道
            yield ("content", (self._span_buf + buf) if self._inside else buf)
            self._span_buf = ""


# ---------------------------------------------------------------------------
# 本地 LLM 客户端
# ---------------------------------------------------------------------------


class LocalLLMClient:
    """本地 LLM 客户端（LLMClient 的并行可选项）。

    默认后端 mlx-lm（Apple Silicon 原生）。构造时只存参数，模型懒加载于
    首次 ``chat_stream``（加载 ~GB 级权重不该阻塞 UI 线程）。
    """

    def __init__(
        self,
        path: str,
        backend: str = "mlx",
        max_tokens: int = 32768,
        temperature: float = 0.7,
        top_p: float = 0.95,
        min_p: float = 0.05,
        top_k: int = 20,
        repetition_penalty: float = 1.0,
        dtype: str = "float16",
        enable_thinking: bool = True,
        num_draft_tokens: int = 0,
        prefill_step_size: int = 0,
    ):
        self._path = path
        self._backend = (backend or "mlx").strip().lower()
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)
        self._top_p = float(top_p)
        self._min_p = float(min_p)
        self._top_k = int(top_k)
        self._repetition_penalty = float(repetition_penalty)
        self._dtype = dtype
        # Qwen3 系模板的 enable_thinking：False 时模板注入空 think 块，模型
        # 直接作答（首字延迟大幅降低，复杂推理质量换速度）。
        self._enable_thinking = bool(enable_thinking)
        self._num_draft_tokens = int(num_draft_tokens)
        self._prefill_step_size = int(prefill_step_size)
        # 投机解码草稿源：指向已加载的辅助模型 client（复用其权重，不另起草稿）。
        # 惰性取值——真正生成时才取 draft_source._model（此时必然已加载），避免
        # 「加载时 _model 尚为 None」的时序坑；None 表示不启用投机解码。
        self._draft_source = None
        self._model = None
        self._tokenizer = None
        self._session = None
        self._load_lock = threading.Lock()
        # 串行化同实例的生成调用：MLX/GGUF 模型对同一实例并发 generate 不安全。
        # 主模型与辅助模型是各自独立的 LocalLLMClient 实例，因此这把锁互不阻塞
        # 二者，只防止辅助模型被「RAG 精简线程」与「事实抽取线程」等并发调用。
        self._gen_lock = threading.Lock()
        # 跨轮 KV 缓存（仅 MLX 路径）：多轮对话复用 system+历史 的 KV 状态，
        # 每轮只预填充本轮新增 token（对齐 Ollama 的 prompt cache 行为）。
        self._mlx_cache = None
        self._cache_token_ids: List[int] = []
        # 创建该缓存的线程 ID。MLX 的 GPU stream 是**线程局部**的：cache 里的
        # 数组绑定创建线程的 stream，换线程复用会直接
        # RuntimeError: There is no Stream(gpu, N) in current thread。
        self._kv_thread: Optional[int] = None
        self._loaded = False
        # ONNX 后端的 KV 缓存（past_key_values.*）：__init__ 必须显式初始化——
        # 旧实现只在 _stream_onnx 循环内懒创建，首轮读 self._past 直接
        # AttributeError，整个 ONNX 后端不可用。
        self._past: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    def _ensure_model(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            if self._backend == "mlx":
                self._load_mlx()
            elif self._backend == "gguf":
                self._load_gguf()
            elif self._backend == "onnx":
                self._load_onnx()
            else:
                raise ValueError(f"未知 LLM 后端: {self._backend!r}")
            # 模型（重）加载后旧 KV 缓存全部失效
            self._mlx_cache = None
            self._cache_token_ids = []
            self._loaded = True

    def _load_mlx(self) -> None:
        if not os.path.isdir(self._path):
            raise FileNotFoundError(f"本地 LLM 模型目录不存在: {self._path}")
        try:
            from mlx_lm import load
        except Exception as e:
            raise RuntimeError(
                "MLX LLM 后端不可用：未安装 mlx-lm。\n"
                "安装：pip install mlx-lm\n"
                f"原始错误: {type(e).__name__}: {e}"
            ) from e
        self._model, self._tokenizer = load(self._path, trust_remote_code=True)

    def _load_gguf(self) -> None:
        gguf = self._path
        if os.path.isdir(gguf):
            for f in sorted(os.listdir(gguf)):
                if f.endswith(".gguf"):
                    gguf = os.path.join(gguf, f)
                    break
        try:
            from llama_cpp import Llama
        except Exception as e:
            raise RuntimeError(
                "GGUF LLM 后端不可用：未安装 llama-cpp-python。\n"
                f"原始错误: {type(e).__name__}: {e}"
            ) from e
        # n_ctx 必须由 max_tokens 推导：旧实现硬编码 4096，而 max_tokens 默认
        # 32768——llama.cpp 的生成受上下文长度限制，长上下文场景下生成早停/
        # 报错，max_tokens 形同虚设且静默发生。预留 1k 给 prompt 余量，
        # 下限 8192（短 max_tokens 时也有基本的多轮空间）。
        n_ctx = max(8192, min(self._max_tokens + 1024, 131072))
        self._model = Llama(
            model_path=gguf, n_ctx=n_ctx, n_gpu_layers=-1, verbose=False
        )

    def _load_onnx(self) -> None:
        onnx_path = self._path
        if os.path.isdir(self._path):
            cand = [os.path.join(self._path, n) for n in ("model.onnx", "model_quantized.onnx")]
            onnx_path = next((c for c in cand if os.path.exists(c)), "")
        if not onnx_path or not os.path.exists(onnx_path):
            raise FileNotFoundError(f"未找到 LLM ONNX 权重: {self._path}")
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except Exception as e:
            raise RuntimeError(
                "ONNX LLM 后端不可用：需要 onnxruntime 与 transformers。\n"
                f"原始错误: {type(e).__name__}: {e}"
            ) from e
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            onnx_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._path if os.path.isdir(self._path) else os.path.dirname(self._path)
        )

    # ------------------------------------------------------------------
    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        # KV 缓存调用组：主对话用 "main"（持久、跨轮复用，避免每句全量预填充）；
        # 辅助调用（知识图谱抽取 / RAG 重排 / 技能检索 / 工具描述 / 预热）用
        # "aux"（一次性本地缓存，不写实例状态）。两者共享同一模型权重但缓存
        # 隔离，aux 调用再也不会冲掉主对话的 KV 缓存。
        kv_group: str = "main",
    ) -> Iterator[StreamEvent]:
        self._ensure_model()
        # 双视图历史的第二视图只有 MLX 后端读得懂；gguf/onnx 后端会把
        # messages 原样交给底层引擎，多一个键有风险 → 先剥掉。
        if self._backend != "mlx":
            messages = [
                {k: v for k, v in m.items() if k != "raw_content"}
                if isinstance(m, dict) else m
                for m in messages
            ]
        # 串行化同实例的生成（MLX/GGUF 并发不安全）；主/辅实例各自独立互不阻塞。
        with self._gen_lock:
            if self._backend == "mlx":
                yield from self._stream_mlx(messages, tools, temperature, kv_group)
            elif self._backend == "gguf":
                yield from self._stream_gguf(messages, tools, temperature, kv_group)
            else:
                yield from self._stream_onnx(messages, tools, temperature, kv_group)

    # ---- 通用：把 think 分段 + 收集全文（用于工具调用解析）----
    def _stream_with_producer(self, deltas: Iterator[str],
                              start_in_think: bool = False,
                              raw_getter: Optional[Callable[[], str]] = None):
        """消费 (text_delta) 迭代器，发出 content/reasoning，最后解析工具调用。

        分流规则（本地模型把思考/工具调用/摘要都以文本写在正文里）：
          * think 标签对 → reasoning（思考过程面板）；
          * tool_call 标签对 → 抑制，不进正文（由 parse_tool_calls 从完整文本
            另行解析成 tool_calls，避免原始标记重复显示到对话区）；
          * 残留的孤立 think 字面量 → 从正文剔除；
          * summary 标签对 → **保留**在 content 里（内核 l2_summary /
            has_summary_tag 需要它；UI 侧 _SummaryStripper 负责展示时剥离）。
        """
        think = _ThinkSplitter(start_in_think=start_in_think)
        tool = _ToolCallSplitter()
        summary = _SummarySpanRouter()
        full: List[str] = []

        def emit_content(seg: str):
            # 正文通道：剔工具调用块（summary 完整标签保留给内核，由 UI 侧
            # _SummaryStripper 负责展示时剥离）。
            # summary 路由器只作**旁路观察者**：从同一输入流独立提取摘要内文
            # 发往思考面板，其 content 输出丢弃——避免三个状态机串行缓冲时
            # 的边界组合问题。
            for piece in tool.feed(seg):
                if not piece:
                    continue
                piece = _STRAY_THINK_RE.sub("", piece)
                if piece:
                    yield StreamEvent("content", content=piece)
            for kind, piece in summary.feed(seg):
                if kind == "summary" and piece:
                    yield StreamEvent("reasoning", content=piece)

        for delta in deltas:
            full.append(delta)
            for kind, seg in think.feed(delta):
                if not seg:
                    continue
                if kind == "reasoning":
                    yield StreamEvent("reasoning", content=seg)
                else:
                    yield from emit_content(seg)
        for kind, seg in think.flush():
            if not seg:
                continue
            if kind == "reasoning":
                yield StreamEvent("reasoning", content=seg)
            else:
                yield from emit_content(seg)
        residue = tool.flush()
        if residue:
            residue = _STRAY_THINK_RE.sub("", residue)
            if residue:
                yield StreamEvent("content", content=residue)
        for kind, piece in summary.flush():
            if kind == "summary" and piece:
                yield StreamEvent("reasoning", content=piece)
        text = "".join(full)
        calls = parse_tool_calls(text)
        raw = ""
        if raw_getter is not None:
            try:
                raw = raw_getter() or ""
            except Exception as e:  # 防御：原始输出拿不到不该中断收尾
                logger.debug("LLM: raw_content unavailable: %s", e)
        yield StreamEvent("done", tool_calls=calls, usage={}, raw_content=raw)

    # ---- MLX ----
    def _stream_mlx(self, messages, tools, temperature, kv_group: str = "main"):
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler, make_logits_processors

        tok = self._tokenizer
        hf = getattr(tok, "tokenizer", tok)
        # 双视图历史：本地后端读 raw_content，把 assistant 消息还原成「生成时
        # 的逐 token 序列」，跨轮 KV 前缀才对得上。HTTP 后端的历史没有该字段，
        # messages 原样不动（零影响）。
        messages = _rebuild_raw_messages(messages)
        try:
            prompt = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                tools=tools if tools else None,
                enable_thinking=self._enable_thinking,
            )
        except Exception:
            prompt = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        # 模板把 <think> 预填进生成提示时（enable_thinking 默认开），模型输出
        # 直接以推理正文开始、只含闭合标签——按「已在 think 内」起步，否则
        # 推理会全部落进正文（对话区）。
        start_in_think = prompt.rstrip().endswith(_THINK_OPEN)

        sampler = make_sampler(
            temperature if temperature > 0 else 0.0,
            self._top_p, self._min_p, 1, self._top_k,
        )
        logits_processors = make_logits_processors(
            repetition_penalty=self._repetition_penalty or None,
            repetition_context_size=20,
        ) if self._repetition_penalty and self._repetition_penalty != 1.0 else []

        # ---- 跨轮 KV 缓存复用（对齐 Ollama 的 prompt cache）----
        # 缓存只创建一次（实例级），之后每轮传入同一个 cache，只把**本轮新增
        # 的 token**喂给 stream_generate，mlx 从缓存 offset 续算——实测同一份
        # 上下文首字 3.18s → 0.19s。
        #
        # 命中判据是「缓存的 token 序列必须严格是新 prompt 的前缀」，两种形态：
        #   ① 缓存更长/相等（缓存是前缀，新 prompt 在后追加）——常态，直接续算；
        #   ② 新 prompt 更短（回退、上下文收缩后重试）——必须把缓存裁到公共
        #      前缀，裁不动就整体重建。
        # 中途分叉一律整体重建：绝不拿错位的缓存去续算（会输出乱码）。
        from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

        prompt_ids = tok.encode(prompt)
        # KV 缓存按「调用组」隔离：主对话(kv_group="main")用持久缓存跨轮复用；
        # 辅助调用(知识图谱抽取/RAG 重排/技能检索/工具描述等)走一次性本地缓存，
        # 绝不污染主对话的实例级缓存——否则辅助调用会改写 self._mlx_cache /
        # _kv_thread / _cache_token_ids，导致下一轮主对话被判「跨线程 / 前缀不符」
        # 而整体重建，退回每句全量预填充（约 20s）。
        persist = (kv_group == "main")
        cache = self._mlx_cache if persist else None
        cached_ids = self._cache_token_ids if persist else []
        reuse = 0
        thread_mismatch = [False]
        # 线程隔离：MLX 的 GPU stream 是线程局部的，cache 只能在创建它的线程
        # 上复用。UI 每次发送都是新的 agent worker 线程（辅助模型的事实抽取/
        # RAG 精简又是另外的线程），直接拿旧线程的 cache 会抛
        # "There is no Stream(gpu, N) in current thread" —— 表现为生成中断、
        # reasoning 已流出但正文为空。宁可重建（全量预填充）也不要崩溃。
        # （仅对持久主缓存做线程校验；aux 本地缓存无所谓。）
        if persist and cache is not None and self._kv_thread != threading.get_ident():
            if cached_ids:
                logger.debug("LLM: KV cache 创建于其它线程，丢弃重建")
            cache = None
            thread_mismatch[0] = True
        over_limit = False
        common = 0
        if cache is None or not cached_ids:
            cache = None
        elif len(cached_ids) > _KV_MAX_TOKENS:      # 长会话内存护栏
            cache = None
            over_limit = True
        else:
            lim = min(len(cached_ids), len(prompt_ids))
            while common < lim and cached_ids[common] == prompt_ids[common]:
                common += 1
            if common >= len(cached_ids):
                # 形态①：缓存是严格前缀 → 只喂新增部分
                reuse = len(cached_ids)
            elif common >= len(prompt_ids) and len(prompt_ids) > 0:
                # 形态②：新 prompt 是缓存的前缀 → 需要裁剪掉多余部分
                need = len(cached_ids) - common
                if need > 0 and trim_prompt_cache(cache, need) == need:
                    reuse = common
                else:
                    cache = None      # 不可裁剪 → 重建
            else:
                # 形态③：中途分叉。分叉点通常在 user 消息的动态包装处
                # （WORKING MEMORY / 锚点注入每轮都变），但公共前缀——几乎总是
                # **整个 system prompt + 早期历史**——的 KV 仍然有效：裁到分叉
                # 点复用，只重算分叉之后的一小段。全量丢弃的话，4500 token 的
                # system prompt 每次调用都要重算一遍。
                need = len(cached_ids) - common
                if (common > 0 and need > 0
                        and trim_prompt_cache(cache, need) == need):
                    reuse = common
                    logger.debug(
                        "LLM: KV cache 分叉 @%d/%d → 裁剪复用公共前缀"
                        "（suffix %d token）", common, len(cached_ids),
                        len(prompt_ids) - common)
                else:
                    cache = None
        if cache is None and cached_ids:
            # 整体重建的原因必须 info 级可见——「中间突然很慢」时看日志即可
            # 定位是护栏命中还是分叉裁剪失败（而非盲目怀疑工具调用）。
            reason = ("缓存创建于其它线程" if thread_mismatch[0]
                      else "超 _KV_MAX_TOKENS 长会话护栏" if over_limit
                      else "分叉/回退裁剪失败（Qwen3.5 ArraysCache 不可裁剪）")
            logger.info(
                "LLM: KV 缓存整体重建（%s）: cached=%d, prompt=%d, fork@%d",
                reason, len(cached_ids), len(prompt_ids), common)
        if cache is None:
            cache = make_prompt_cache(self._model)
            reuse = 0
        # 兜底：喂给 stream_generate 的 suffix **绝不能为空**。prompt 为空时
        # mlx-lm 会抛 "Either input_embeddings or prompt (or both) must be
        # provided"，整轮连同工具调用一起报废。
        # 触发场景比想象中常见：本轮被 rollback 后重发同一段历史（工具参数
        # 校验失败、上下文收缩重试都会 rollback），此时缓存恰好已完全覆盖
        # prompt，reuse == len(prompt_ids)。退回一个 token 让它至少有事可做。
        if prompt_ids and reuse >= len(prompt_ids):
            reuse = len(prompt_ids) - 1
        if not prompt_ids:
            raise ValueError(
                "渲染后的 prompt 为空（chat template 未产出任何 token）")
        suffix_ids = prompt_ids[reuse:]
        if persist:
            # 主对话组命中/复用必须 info 级可见：用户问「预热到底生效没」时，
            # UI 日志里这行直接给出答案（reuse≈cached 即命中；重建另有
            # 「KV 缓存整体重建」的 info 行说明原因）。aux 组高频调用保持 debug。
            logger.info(
                "MLX KV cache: cached=%d prompt=%d reuse=%d suffix=%d group=%s",
                len(cached_ids), len(prompt_ids), reuse, len(suffix_ids), kv_group)
        else:
            logger.debug(
                "MLX KV cache: cached=%d prompt=%d reuse=%d suffix=%d group=%s",
                len(cached_ids), len(prompt_ids), reuse, len(suffix_ids), kv_group)
        # 仅主对话持久化 KV 缓存；aux 调用用一次性本地缓存，不写实例状态，
        # 避免覆盖主对话缓存导致下一轮全量重 prefill。
        if persist:
            self._mlx_cache = cache
            self._kv_thread = threading.get_ident()

        # 模板在 assistant 起始标记后预置的片段（<think>\n 或空 think 块）。它
        # 属于「生成序列」的头部而非模型输出，重建历史时必须一并带上，否则
        # 前缀会差这几个 token 而对不上。
        gen_prefix = _assistant_prompt_prefix(prompt)
        gen_ref: List[List[int]] = []

        def _produce():
            gen_ids: List[int] = []
            gen_ref.append(gen_ids)
            _t0 = time.time()
            _first_ts: List[Optional[float]] = [None]
            # 投机解码：草稿源指向已加载的辅助模型（复用权重，不另起草稿）。惰性取
            # aux._model（生成时必然已加载）；取不到 / 后端非 mlx / 未配置则不启用。
            # mlx-lm 的 stream_generate 仅在 draft_model 非空时把 num_draft_tokens
            # 透传给 speculative_generate_step，故 draft 关闭时不会误传该参数。
            draft_model = None
            draft_lock = None
            ds = self._draft_source
            if (self._backend == "mlx" and ds is not None
                    and self._num_draft_tokens > 0):
                dm = getattr(ds, "_model", None)
                if dm is not None:
                    # 投机解码要求主/草稿缓存都「可裁剪」；但 Qwen3.5 等模型的线性
                    # 注意力层用 ArraysCache（不可裁剪），与 mlx-lm 的可裁剪要求
                    # 互斥——mlx-lm 最新即 0.32.2 对此类模型不可用。探测缓存可裁剪
                    # 性，不可则静默降级为普通生成（prefill_step_size 仍生效），避免
                    # ValueError / IndexError 崩溃。
                    from mlx_lm.models.cache import can_trim_prompt_cache
                    if (can_trim_prompt_cache(cache)
                            and can_trim_prompt_cache(dm.make_cache())):
                        draft_model = dm
                        draft_lock = getattr(ds, "_gen_lock", None)
                    else:
                        logger.debug(
                            "LLM: prompt cache 不可裁剪，关闭投机解码"
                            "（降级为普通生成）")
            try:
                # 持有 aux 的生成锁：防止主模型把 aux 当草稿做前向时，aux 自身正在
                # 别处生成（权重前向非线程安全）。本仓库本地模式下二者在 worker
                # 线程内串行，这里仅为兜底。
                if draft_lock is not None:
                    draft_lock.acquire()
                _sg_kwargs: Dict[str, Any] = {}
                # prefill 批大小（mlx-lm 的 prefill_step_size）：加快长 prompt 的
                # 预填充；0 表示用 mlx-lm 默认值（普通 2048 / 投机 512）。
                if self._prefill_step_size and self._prefill_step_size > 0:
                    _sg_kwargs["prefill_step_size"] = self._prefill_step_size
                # 草稿模型需独立 KV 缓存，且与主缓存【拼接】后整体传入
                # stream_generate（mlx 按 len(model.layers) 切片：前半主缓存、
                # 后半草稿缓存）。草稿缓存不持久化、每轮新建——草稿模型每轮重新
                # prefill 当前 prompt，旧 KV 无意义；结构须与主/草稿模型 make_cache
                # 完全一致（含线性注意力层的 ArraysCache），故用 make_prompt_cache。
                prompt_cache = cache
                if draft_model is not None:
                    _sg_kwargs["draft_model"] = draft_model
                    _sg_kwargs["num_draft_tokens"] = self._num_draft_tokens
                    from mlx_lm.models.cache import make_prompt_cache as _mpc
                    draft_cache = _mpc(draft_model)
                    prompt_cache = cache + draft_cache
                for resp in stream_generate(
                    self._model, tok, prompt=suffix_ids,
                    max_tokens=self._max_tokens, sampler=sampler,
                    logits_processors=logits_processors,
                    prompt_cache=prompt_cache, **_sg_kwargs,
                ):
                    text = getattr(resp, "text", "") or ""
                    if text:
                        if _first_ts[0] is None:
                            _first_ts[0] = time.time() - _t0
                        yield text
                    t = getattr(resp, "token", None)
                    if t is not None:
                        try:
                            gen_ids.append(
                                int(t.item()) if hasattr(t, "item") else int(t))
                        except (ValueError, TypeError):
                            pass
            finally:
                if draft_lock is not None:
                    draft_lock.release()
                _total = time.time() - _t0
                _ttft = _first_ts[0] if _first_ts[0] is not None else _total
                # 诊断：重算公共前缀长度 + 解码两端前几个 token，定位 reuse=0 的根因
                # （缓存对象丢失 vs prompt 前缀对不上）。
                _lim = min(len(cached_ids), len(prompt_ids))
                _common = 0
                while _common < _lim and cached_ids[_common] == prompt_ids[_common]:
                    _common += 1
                try:
                    _fc = tok.decode(cached_ids[:16])
                except Exception:
                    _fc = str(cached_ids[:16])
                try:
                    _fp = tok.decode(prompt_ids[:16])
                except Exception:
                    _fp = str(prompt_ids[:16])
                _lo = max(0, _common - 8)
                _hi = min(_lim, _common + 60)
                try:
                    _div_c = tok.decode(cached_ids[_lo:_hi])
                except Exception:
                    _div_c = str(cached_ids[_lo:_hi])
                try:
                    _div_p = tok.decode(prompt_ids[_lo:_hi])
                except Exception:
                    _div_p = str(prompt_ids[_lo:_hi])
                _kv_diag(kv_group, len(cached_ids), len(prompt_ids), reuse,
                         len(suffix_ids), _ttft, _total, _common,
                         thread_mismatch[0], _fc, _fp, _div_c, _div_p)
                # 无论生成到哪一步（含中途取消）都如实记录缓存对应的 token
                # 序列：命中的话缓存现在覆盖到整个 prompt + 已生成部分。
                # 仅主对话持久化；aux 调用不写，避免污染主对话缓存。
                if persist:
                    self._cache_token_ids = list(prompt_ids) + gen_ids

        def _raw_getter() -> str:
            ids = gen_ref[0] if gen_ref else []
            if not ids:
                return ""
            try:
                return gen_prefix + tok.decode(ids)
            except Exception:
                return ""

        # 本地后端也要报 usage：旧实现三个后端全部发空 dict——HTTP 后端有真实
        # final_usage，本地模型的 tok/s 显示与任何依赖 usage 的计量全部失效。
        # prompt_tokens 用整段 prompt 的 token 数（含 KV 复用部分，语义是
        # 「本轮的输入规模」），completion_tokens 用实际生成数。
        for ev in self._stream_with_producer(
                _produce(), start_in_think=start_in_think, raw_getter=_raw_getter):
            if ev.type == "done":
                n_gen = len(gen_ref[0]) if gen_ref else 0
                ev.usage = {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": n_gen,
                    "total_tokens": len(prompt_ids) + n_gen,
                }
            yield ev

    # ---- GGUF ----
    def _stream_gguf(self, messages, tools, temperature, kv_group: str = "main"):
        create = getattr(self._model, "create_chat_completion", None)
        if create is None:
            raise RuntimeError("llama-cpp-python 版本过旧，缺少 create_chat_completion")
        kwargs: Dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self._max_tokens,
            "stream": True,
        }
        if self._top_p:
            kwargs["top_p"] = self._top_p
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        splitter = _ThinkSplitter()
        full: List[str] = []
        calls_acc: Dict[int, Dict[str, Any]] = {}
        for chunk in create(**kwargs):
            choices = (chunk.get("choices") or []) if isinstance(chunk, dict) else []
            if not choices:
                continue
            delta = choices[0].get("delta", {}) or {}
            rc = delta.get("reasoning_content")
            if rc:
                yield StreamEvent("reasoning", content=rc)
            ct = delta.get("content")
            if ct:
                full.append(ct)
                for kind, seg in splitter.feed(ct):
                    if seg:
                        yield StreamEvent(kind, content=seg)
            for tc in delta.get("tool_calls", []) or []:
                idx = int(tc.get("index", 0) or 0)
                slot = calls_acc.setdefault(idx, {
                    "id": f"local_call_{idx}", "type": "function",
                    "function": {"name": "", "arguments": ""},
                })
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function", {}) or {}
                if fn.get("name"):
                    slot["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]
        for kind, seg in splitter.flush():
            if seg:
                yield StreamEvent(kind, content=seg)
        if not calls_acc:
            text = "".join(full)
            calls_acc_list = parse_tool_calls(text)
        else:
            calls_acc_list = [calls_acc[k] for k in sorted(calls_acc)]
        yield StreamEvent("done", tool_calls=calls_acc_list, usage={})

    # ---- ONNX ----
    def _stream_onnx(self, messages, tools, temperature, kv_group: str = "main"):
        import numpy as np
        tok = self._tokenizer
        hf = getattr(tok, "tokenizer", tok)
        prompt = hf.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        ids = hf(prompt, return_tensors="np", add_special_tokens=False)["input_ids"][0]
        # 通用 KV-cache decoder 循环：输入 input_ids + attention_mask +
        # past_key_values，输出 logits + present_key_values。没有 ONNX LLM 模型
        # 可测，仅实现标准布局；遇到不同导出格式会给出明确报错。
        # KV 前缀续算（与 MLX 同源）：past 只有在「新 prompt 是旧 prompt 的
        # 延长」时可复用，且只需喂**新增的那一段**。旧实现每轮都喂整段新
        # prompt 却带着上一轮的 past → 历史被重复计入（第二轮起生成与
        # prompt 错位）；且 past 只增不减，会话越长 KV 张量越大（无界）。
        ids_list = [int(x) for x in ids]
        prev_ids = getattr(self, "_onnx_prompt_ids", None)
        if (prev_ids and len(ids_list) > len(prev_ids)
                and ids_list[:len(prev_ids)] == prev_ids):
            feed_ids = ids_list[len(prev_ids):]
        else:
            # 序列不再对齐（换了会话/回滚/提示词改写）→ 丢弃旧 KV
            feed_ids = ids_list
            self._past = {}
        self._onnx_prompt_ids = ids_list
        input_ids = np.array([feed_ids], dtype=np.int64)
        full: List[str] = []
        generated: List[int] = []
        # 同 MLX：模板可能已把 <think> 预填进生成提示（见 _stream_mlx 注释）。
        splitter = _ThinkSplitter(
            start_in_think=prompt.rstrip().endswith(_THINK_OPEN))
        # past / present 输入名在会话生命周期内不变，循环外探测一次即可
        # （旧实现每轮 get_inputs()，纯浪费）。
        kv_inputs = [i.name for i in self._session.get_inputs()
                     if i.name.startswith("past_key_values")]
        # EOS 归一：tokenizer 可能给 None（无 EOS）或 list（多个 EOS）。
        # 旧写法 `next_id in (hf.eos_token_id,)` 在 None/list 下永假——
        # 生成只能靠 max_tokens 截断，尾部接无关 token。
        _eos = hf.eos_token_id
        _eos_ids = (set(_eos) if isinstance(_eos, (list, tuple))
                    else ({_eos} if _eos is not None else set()))
        for _ in range(self._max_tokens):
            feeds = {"input_ids": input_ids}
            # 探测 past 输入名（不同导出差异大，尽力而为）
            if kv_inputs:
                feeds.update({n: self._past[n] for n in kv_inputs if n in self._past})
            # 输出必须按 **name** 建键：旧实现 dict(zip(get_outputs(), run()))
            # 的 key 是 NodeArg 对象，outs.get("logits") 恒落空、退化成取第一个
            # 输出——若导出把 present.* 排在 logits 前，采样直接拿到 KV 张量。
            outs = {o.name: v for o, v in
                    zip(self._session.get_outputs(), self._session.run(None, feeds))}
            logits = outs.get("logits") or next(iter(outs.values()))
            next_id = self._sample(np.asarray(logits)[0, -1])
            if next_id in _eos_ids:
                break
            # 解码增量 token
            piece = hf.decode([next_id])
            full.append(piece)
            for kind, seg in splitter.feed(piece):
                if seg:
                    yield StreamEvent(kind, content=seg)
            generated.append(next_id)
            input_ids = np.array([[next_id]], dtype=np.int64)
            # present → 下一轮 past
            for o in self._session.get_outputs():
                if o.name.startswith("present"):
                    self._past[o.name.replace("present", "past_key_values", 1)] = outs[o.name]
        for kind, seg in splitter.flush():
            if seg:
                yield StreamEvent(kind, content=seg)
        # 本地后端补 usage（见 _stream_mlx 同款注释）
        n_prompt = int(len(ids))
        yield StreamEvent("done", tool_calls=parse_tool_calls("".join(full)),
                          usage={"prompt_tokens": n_prompt,
                                 "completion_tokens": len(generated),
                                 "total_tokens": n_prompt + len(generated)})

    def _sample(self, logits_np: "np.ndarray") -> int:
        import numpy as np
        t = self._temperature
        if t <= 0:
            return int(np.argmax(logits_np))
        # 温度缩放 + top_k + top_p（numpy 自实现）
        logits = logits_np.astype(np.float64) / max(t, 1e-6)
        if self._top_k > 0:
            kth = np.sort(logits)[-self._top_k]
            logits[logits < kth] = -np.inf
        logits -= logits.max()
        exp = np.exp(logits)
        probs = exp / exp.sum()
        if self._top_p < 1.0:
            order = np.argsort(-probs)
            cum = np.cumsum(probs[order])
            keep = order[cum <= self._top_p]
            mask = np.zeros_like(probs, dtype=bool)
            mask[keep] = True
            probs = np.where(mask, probs, 0.0)
            s = probs.sum()
            if s > 0:
                probs = probs / s
        return int(np.random.choice(len(probs), p=probs))

    # ------------------------------------------------------------------
    def name(self) -> str:
        base = os.path.basename(self._path.rstrip("/")) or self._path
        return f"local:{self._backend}:{base}"

    def preload_system_kv(self, system_prompt: str, tools=None) -> None:
        """在生成线程内预填系统提示 KV，使首句只需预填短用户消息。

        必须在**真实生成所在的线程**调用（MLX 缓存按创建线程的 GPU stream 隔离）：
        缓存创建线程与首句生成线程一致，首句才不会因「跨线程」被判丢弃、退回
        全量预填充。只做前向、不解码，因此缓存精确等于系统提示 token 序列；
        首句 prompt = 系统提示 + 用户消息 时 form ① 命中（need=0，无需裁剪）。
        仅 MLX 后端有效；失败不致命（退化成首句全量预填充）。
        """
        if self._backend != "mlx":
            return
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        with self._gen_lock:
            self._ensure_model()
            if self._model is None:
                return
            # 同线程已预填过则跳过，避免重复前向。
            if self._mlx_cache is not None and self._kv_thread == threading.get_ident():
                return
            try:
                tok = self._tokenizer
                messages = [{"role": "system", "content": system_prompt}]
                try:
                    prompt = tok.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False,
                        tools=tools if tools else None,
                        enable_thinking=self._enable_thinking,
                    )
                except Exception:
                    prompt = tok.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False,
                    )
                prompt_ids = tok.encode(prompt)
                cache = make_prompt_cache(self._model)
                # 仅前向、不解码：把系统提示的 KV 写进 cache，不追加生成 token。
                out = self._model(mx.expand_dims(mx.array(prompt_ids), 0), cache)
                if isinstance(out, tuple) and len(out) >= 2:
                    cache = out[1]
                # MLX 是惰性求值：上面的前向只构建了计算图，不 mx.eval 就一个
                # token 都不会真正算——KV 全是懒数组，首句 stream_generate 第一次
                # 求值时才把 1400+ token 的 prefill 整体补算（实测 preload 0.0s、
                # 首句 TTFT 4.5s 里含全部 prefill），预热完全白做。求值 logits 会
                # 级联算出所有层的 KV（cache 里的 k/v 都是 logits 的依赖中间量），
                # 把这笔成本在启动空闲期一次性付清。
                first = out[0] if isinstance(out, tuple) and len(out) > 0 else out
                if first is not None:
                    mx.eval(first)
                self._mlx_cache = cache
                self._cache_token_ids = list(prompt_ids)
                self._kv_thread = threading.get_ident()
                logger.info("LLM: 系统提示 KV 已预热（%d token，线程 %d）",
                            len(prompt_ids), self._kv_thread)
            except Exception as e:
                logger.warning("系统提示 KV 预热失败(忽略): %s", e)
                self._mlx_cache = None
                self._cache_token_ids = []
                self._kv_thread = None

    def set_draft_source(self, client: "Optional[LocalLLMClient]") -> None:
        """把辅助模型 client 设为投机解码的草稿源（复用其已加载权重，不另起草稿）。

        仅主模型 client 调用；辅助模型 client 自身不应设。传入 None 关闭投机解码
        （退化为普通自回归生成）。线程安全：本方法只改一个引用，真正的 draft_model
        在 ``_stream_mlx`` 生成时惰性取值。
        """
        self._draft_source = client

    def invalidate_kv_cache(self) -> None:
        """丢弃跨轮 KV 缓存。

        缓存在这些时刻必须失效，否则会拿「对应旧历史」的 KV 去续算新 prompt
        （前缀对不上时会被检测到并重建，但对得上的**错位**情形会静默生成乱码）：
          * 会话重置 / 新建会话；
          * 上下文收缩改写了历史（L3 折叠把旧轮次换成摘要）；
          * 用户中途取消生成（本轮只生成了一半，历史与缓存中段不一致）。
        """
        if self._mlx_cache is not None or self._cache_token_ids:
            logger.info("LLM: KV cache invalidated")
        self._mlx_cache = None
        self._cache_token_ids = []
        self._kv_thread = None
        # ONNX 的 past_key_values 同样必须失效（会话重置/取消后不能拿旧 KV 续算）
        self._past = {}
        # ONNX 侧记录「past 对应的 prompt」，一并失效（见 _stream_onnx）
        self._onnx_prompt_ids = None

    def close(self) -> None:
        # 释放大块权重：引用置空，等待 GC。会话级 ORT session 亦可析构。
        self.invalidate_kv_cache()
        self._model = None
        self._session = None
        self._tokenizer = None
        self._loaded = False
        # MLX 有独立的 Metal 缓冲池：权重 del 后只是回到池里、不立刻还给系统——
        # 必须显式 gc.collect() + mx.clear_cache() 才能真正回收统一内存。
        # 否则「同步前卸载对话模型」形同虚设：引用清空了，但 ~3GB 权重仍在物理
        # 内存，16GB 机器上同步峰值照旧冲高。只 MLX 后端需要这一步；ONNX 后端
        # 靠 session=None + gc 即可，clear_cache 对 ORT 无副作用也不必要。
        # 已加载的其它 MLX 模型（嵌入/重排）仍被引用，clear_cache 只回收「已无
        # 引用」的缓冲，不会误伤它们。
        if self._backend == "mlx":
            try:
                import gc
                import mlx.core as mx
                gc.collect()
                mx.clear_cache()
            except Exception:
                pass
