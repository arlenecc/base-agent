"""OpenAI-compatible streaming chat client.

Supports:
- listing models (GET /v1/models)
- streaming chat completions with tool_calls and reasoning_content (deepseek-style)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

import httpx


def _strip_raw_content(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """剥掉双视图历史的第二视图（raw_content）后再发往远端。

    ``raw_content`` 只服务本地 MLX 后端（用它重建 KV 前缀）。远端 API 不认识
    这个键：严格实现会 400，宽松实现也会白烧一份 token（raw 通常比正文长）。
    绝大多数历史本来就没有该字段 → 原样返回，零拷贝开销。
    """
    if not any(isinstance(m, dict) and "raw_content" in m for m in messages):
        return messages
    return [
        {k: v for k, v in m.items() if k != "raw_content"} if isinstance(m, dict) else m
        for m in messages
    ]


@dataclass
class StreamEvent:
    """A single parsed streaming event.

    type:
        "content"   - a delta of the visible assistant reply (content field)
        "reasoning" - a delta of the model's reasoning (reasoning_content field)
        "done"      - emitted once at the end; carries accumulated tool_calls + usage
    """

    type: str
    content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    # 模型**原始输出**（含 think 标签与工具调用 XML），只在 done 事件携带。
    # 仅本地 MLX 后端填它：后端要用它把历史 assistant 消息重建回「生成时的
    # 逐 token 序列」，跨轮 KV 缓存的前缀才对得上（见 local/llm.py）。
    # HTTP 后端恒为 ""，历史沿用旧形状（只有 content），零影响。
    raw_content: str = ""


class LLMError(RuntimeError):
    pass


def _tool_call_index(tc: Dict[str, Any], state: List[int]) -> int:
    """归一化 tool_call 的 ``index`` 为 int。

    少数兼容服务把 index 发成字符串（或并发多工具时省略）。原样当 key 会让
    两个后果二选一：同一次回复的多个 tool_call 因 key 不同而被合并进同一
    slot，或末尾 ``sorted(tool_acc)`` 抛 ``TypeError: '<' not supported
    between instances of 'str' and 'int'``。统一转 int，转不了按 0 归并。

    省略 index 时（``state`` = [next_seq, last_slot]）：
    * 带 ``id`` 或 ``function.name`` 的 delta 视为**新调用** → 分配递增
      index（旧实现一律归 0，同一 delta 里多个无 index 调用的 arguments
      被拼进同一 slot，产出非法 JSON）；
    * 只有 ``function.arguments`` 的 delta 视为**参数片段** → 归并到上一
      个 slot（连续流式输出的常态）。
    """
    raw = tc.get("index")
    if raw is None or raw == "":
        fn = tc.get("function", {}) or {}
        if tc.get("id") or fn.get("name"):
            idx = state[0]
            state[0] += 1
        else:
            idx = state[1]
        state[1] = idx
        return idx
    try:
        idx = int(raw)
    except (TypeError, ValueError):
        idx = 0
    state[1] = idx
    return idx


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 120.0,
        max_tokens: int = 32768,
        top_p: float = 1.0,
        min_p: float = 0.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.min_p = min_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        # Use a structured timeout so that streaming reads have a per-chunk
        # deadline.  A plain `timeout=float` applies only to the *whole*
        # request (and, for streaming, once the first byte arrives, an idle
        # server that stalls mid-generation would block `iter_lines()`
        # FOREVER — the "reply stops halfway and never resumes" symptom).
        # `read` here is the max gap between chunks; local models can take
        # tens of seconds on a hard reasoning step, so we use a generous but
        # finite bound to fail fast instead of hanging the UI.
        if isinstance(timeout, httpx.Timeout):
            self._http: httpx.Client = httpx.Client(timeout=timeout)
        else:
            read_timeout = max(timeout, 60.0)
            self._http: httpx.Client = httpx.Client(
                timeout=httpx.Timeout(connect=timeout, read=read_timeout,
                                      write=timeout, pool=timeout)
            )

    # ------------------------------------------------------------------
    # set http (used by tests; also lets the UI inject a configured client)
    # ------------------------------------------------------------------
    def _set_http(self, client: Any) -> None:  # pragma: no cover - test helper
        self._http = client

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    # ------------------------------------------------------------------
    # list models
    # ------------------------------------------------------------------
    def list_models(self) -> List[str]:
        url = f"{self.base_url}/models"
        resp = self._http.get(url, headers=self._headers())
        try:
            if resp.status_code >= 400:
                raise LLMError(f"list_models failed: HTTP {resp.status_code} {resp.text[:200]}")
            data = resp.json()
        finally:
            resp.close()
        items = data.get("data", []) if isinstance(data, dict) else data
        ids = [it["id"] for it in items if isinstance(it, dict) and it.get("id")]
        return sorted(ids)

    # ------------------------------------------------------------------
    # chat (streaming)
    # ------------------------------------------------------------------
    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
    ) -> Iterator[StreamEvent]:
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": _strip_raw_content(messages),
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.min_p > 0:
            payload["min_p"] = self.min_p
        if self.top_k > 0:
            payload["top_k"] = self.top_k
        if self.repetition_penalty != 1.0:
            payload["repetition_penalty"] = self.repetition_penalty
        if tools:
            payload["tools"] = tools

        tool_acc: Dict[int, Dict[str, Any]] = {}
        final_usage: Dict[str, int] = {}
        # 无 index 兼容路径的状态：[next_seq, last_slot]，见 _tool_call_index
        _tc_state: List[int] = [0, 0]

        try:
            with self._http.stream("POST", url, headers=self._headers(),
                                   content=json.dumps(payload)) as resp:
                if resp.status_code >= 400:
                    try:
                        body = resp.read().decode("utf-8", errors="replace")
                    except Exception:
                        body = ""
                    raise LLMError(
                        f"chat_stream failed: HTTP {resp.status_code} {body[:300]}")

                for raw in resp.iter_lines():
                    if not raw:
                        continue
                    line = (raw.decode("utf-8", errors="replace")
                            if isinstance(raw, (bytes, bytearray)) else raw)
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    if obj.get("usage"):
                        final_usage = obj["usage"]

                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta", {}) or {}

                    rc = delta.get("reasoning_content")
                    if rc:
                        yield StreamEvent("reasoning", content=rc)

                    ct = delta.get("content")
                    if ct:
                        yield StreamEvent("content", content=ct)

                    for tc in delta.get("tool_calls", []) or []:
                        idx = _tool_call_index(tc, _tc_state)
                        slot = tool_acc.setdefault(
                            idx, {"id": "", "type": "function",
                                  "function": {"name": "", "arguments": ""}})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        if tc.get("type"):
                            slot["type"] = tc["type"]
                        fn = tc.get("function", {}) or {}
                        if fn.get("name"):
                            slot["function"]["name"] += fn["name"]
                        if fn.get("arguments"):
                            slot["function"]["arguments"] += fn["arguments"]
        except httpx.HTTPError as e:
            # 网络中断（ReadTimeout / ConnectError / RemoteProtocolError）在
            # iter_lines 中途抛出的是 httpx.* 而不是 LLMError——调用方只捕
            # LLMError 时会漏判真实断网，错误冒泡成难懂的原始异常。统一归一，
            # 保留原文。
            raise LLMError(f"chat_stream 网络错误: {type(e).__name__}: {e}") from e

        # sorted() 要求 key 可比：_tool_call_index 已保证全为 int。
        yield StreamEvent("done",
                          tool_calls=[tool_acc[k] for k in sorted(tool_acc)],
                          usage=final_usage)

    # ------------------------------------------------------------------
    def close(self) -> None:
        try:
            self._http.close()
        except Exception:
            pass
