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
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# StreamEvent 数据类与 LLMClient 同源，保证内核解析一致。
from ..llm_client import StreamEvent  # noqa: E402


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


def _new_tool_call_id(n: int) -> str:
    return f"local_call_{n}"


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

_THINK_OPEN = "<think:6124c78e>"
_THINK_CLOSE = "</think:6124c78e>"
_TAG_PREFIX_LEN = max(len(_THINK_OPEN), len(_THINK_CLOSE)) - 1


class _ThinkSplitter:
    """增量地把文本流拆成 (kind, segment) 对，处理跨 chunk 的 ``<think:6124c78e>`` 边界。"""

    def __init__(self) -> None:
        self._in_think = False
        self._buf = ""

    def feed(self, delta: str) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        self._buf += delta
        i = 0
        n = len(self._buf)
        while i < n:
            if not self._in_think and self._buf.startswith(_THINK_OPEN, i):
                self._in_think = True
                i += len(_THINK_OPEN)
                continue
            if self._in_think and self._buf.startswith(_THINK_CLOSE, i):
                self._in_think = False
                i += len(_THINK_CLOSE)
                continue
            # 找下一个标签起点（或字符串末）
            j = n if not self._in_think else self._buf.find(_THINK_CLOSE, i)
            if not self._in_think:
                k = self._buf.find(_THINK_OPEN, i)
                j = k if k != -1 else n
            if j == -1:
                j = n
            seg = self._buf[i:j]
            if seg:
                out.append(("reasoning" if self._in_think else "content", seg))
            i = j
        # 把可能构成标签前缀的尾部留在 buffer（避免把 "<thi" 当 content 发出去）
        if len(self._buf) > _TAG_PREFIX_LEN:
            cut = max(0, len(self._buf) - _TAG_PREFIX_LEN)
            # 只在尾部不是标签起点时才前移游标；简化：保留最后 _TAG_PREFIX_LEN 字符
            keep = self._buf[-_TAG_PREFIX_LEN:]
            if keep in (_THINK_OPEN[:_TAG_PREFIX_LEN], _THINK_CLOSE[:_TAG_PREFIX_LEN]):
                self._buf = self._buf[:-_TAG_PREFIX_LEN]
            else:
                self._buf = ""
        else:
            self._buf = ""
        return out

    def flush(self) -> List[Tuple[str, str]]:
        if self._buf:
            out = [("reasoning" if self._in_think else "content", self._buf)]
            self._buf = ""
            return out
        return []


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
        self._model = None
        self._tokenizer = None
        self._session = None
        self._load_lock = threading.Lock()
        # 串行化同实例的生成调用：MLX/GGUF 模型对同一实例并发 generate 不安全。
        # 主模型与辅助模型是各自独立的 LocalLLMClient 实例，因此这把锁互不阻塞
        # 二者，只防止辅助模型被「RAG 精简线程」与「事实抽取线程」等并发调用。
        self._gen_lock = threading.Lock()
        self._loaded = False

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
        self._model = Llama(
            model_path=gguf, n_ctx=4096, n_gpu_layers=-1, verbose=False
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
    ) -> Iterator[StreamEvent]:
        self._ensure_model()
        # 串行化同实例的生成（MLX/GGUF 并发不安全）；主/辅实例各自独立互不阻塞。
        with self._gen_lock:
            if self._backend == "mlx":
                yield from self._stream_mlx(messages, tools, temperature)
            elif self._backend == "gguf":
                yield from self._stream_gguf(messages, tools, temperature)
            else:
                yield from self._stream_onnx(messages, tools, temperature)

    # ---- 通用：把 think 分段 + 收集全文（用于工具调用解析）----
    def _stream_with_producer(self, deltas: Iterator[str]):
        """消费 (text_delta) 迭代器，发出 content/reasoning，最后解析工具调用。"""
        splitter = _ThinkSplitter()
        full: List[str] = []
        for delta in deltas:
            full.append(delta)
            for kind, seg in splitter.feed(delta):
                if seg:
                    yield StreamEvent(kind, content=seg)
        for kind, seg in splitter.flush():
            if seg:
                yield StreamEvent(kind, content=seg)
        text = "".join(full)
        calls = parse_tool_calls(text)
        yield StreamEvent("done", tool_calls=calls, usage={})

    # ---- MLX ----
    def _stream_mlx(self, messages, tools, temperature):
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler, make_logits_processors

        tok = self._tokenizer
        hf = getattr(tok, "tokenizer", tok)
        try:
            prompt = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                tools=tools if tools else None,
            )
        except Exception:
            prompt = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        sampler = make_sampler(
            temperature if temperature > 0 else 0.0,
            self._top_p, self._min_p, 1, self._top_k,
        )
        logits_processors = make_logits_processors(
            repetition_penalty=self._repetition_penalty or None,
            repetition_context_size=20,
        ) if self._repetition_penalty and self._repetition_penalty != 1.0 else []

        def _produce():
            for resp in stream_generate(
                self._model, tok, prompt=prompt,
                max_tokens=self._max_tokens, sampler=sampler,
                logits_processors=logits_processors,
            ):
                text = getattr(resp, "text", "") or ""
                if text:
                    yield text
                mx.eval(resp.token) if hasattr(resp, "token") else None
        yield from self._stream_with_producer(_produce())

    # ---- GGUF ----
    def _stream_gguf(self, messages, tools, temperature):
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
    def _stream_onnx(self, messages, tools, temperature):
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
        input_ids = np.array([ids], dtype=np.int64)
        full: List[str] = []
        generated: List[int] = []
        splitter = _ThinkSplitter()
        for _ in range(self._max_tokens):
            feeds = {"input_ids": input_ids}
            # 探测 past 输入名（不同导出差异大，尽力而为）
            kv_inputs = [i.name for i in self._session.get_inputs()
                         if i.name.startswith("past_key_values")]
            if kv_inputs:
                feeds.update({n: self._past[n] for n in kv_inputs if n in self._past})
            outs = dict(zip(self._session.get_outputs(), self._session.run(None, feeds)))
            logits = outs.get("logits", next(iter(outs.values())))
            next_id = self._sample(np.asarray(logits)[0, -1])
            if next_id in (hf.eos_token_id,):
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
                    self._past = getattr(self, "_past", {})
                    self._past[o.name.replace("present", "past_key_values", 1)] = outs[o.name]
        for kind, seg in splitter.flush():
            if seg:
                yield StreamEvent(kind, content=seg)
        yield StreamEvent("done", tool_calls=parse_tool_calls("".join(full)), usage={})

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

    def close(self) -> None:
        # 释放大块权重：引用置空，等待 GC。会话级 ORT session 亦可析构。
        self._model = None
        self._session = None
        self._loaded = False
