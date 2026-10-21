"""Web tools: web_scan (fetch + extract text) and webexec_js (browser control)."""
from __future__ import annotations

import ipaddress
import json
import os
import socket
from typing import Any, List
from urllib.parse import urljoin, urlparse

import httpx

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover - bs4 optional at runtime
    BeautifulSoup = None  # type: ignore

from ..config import AgentConfig
from .base import Tool, ToolRegistry, ToolResult

# 单次抓取最多读入的字节数。httpx 的非流式 get() 会把整个响应体读进内存，
# 一个几 GB 的文件/流式接口就能把进程打满。
_MAX_FETCH_BYTES = 8 * 1024 * 1024
# 交给 BeautifulSoup 的字符上限：建完整 DOM 的内存峰值可达响应体的数十倍。
# 上面只限制了下载量，解析量还得单独限一道。
_MAX_PARSE_CHARS = 2 * 1024 * 1024
# 手动跟随的重定向上限（与 httpx 默认一致）。
_MAX_REDIRECTS = 5


def _is_blocked_host(host: str) -> bool:
    """重定向目标主机是否落在应拦截的网段（SSRF 防护）。

    follow_redirects=True 只约束协议不跨（http→file），管不了跨网络边界：
    公网 URL 可被 302 引到云元数据（169.254.169.254）、loopback、内网段。
    这里对**重定向目标**做主机解析 + 网段判定；初始 URL 是用户/模型显式
    给的，仍只校验 scheme（本地调试扫 localhost 是合法需求）。
    """
    if not host:
        return True
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception:
        return True  # 解析不了的主机不当跳转目标
    for info in infos:
        ip_str = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str.split("%")[0])
        except ValueError:
            return True
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            return True
    return False


def _strip_blocks_linear(text: str) -> str:
    """线性剥掉 ``<script>`` / ``<style>`` 块（bs4 缺失时的回退）。

    纯 ``str.find`` 扫描，任何输入 O(n)——正则回退（``<script.*?</script>``
    配 re.S）在超长且标签不闭合的 HTML 上会大量回溯（ReDoS）。
    """
    out: List[str] = []
    rest = text
    while True:
        # 找最早出现的 script / style 开标签
        idxs = [(rest.lower().find(f"<{tag}"), tag) for tag in ("script", "style")]
        idxs = [(i, t) for i, t in idxs if i >= 0]
        if not idxs:
            out.append(rest)
            break
        i, tag = min(idxs)
        out.append(rest[:i])
        close = rest.lower().find(f"</{tag}>", i)
        if close < 0:
            # 未闭合：丢弃剩余全部（正是正则回溯最坏的输入）
            break
        rest = rest[close + len(f"</{tag}>"):]
    return "".join(out)


def _truncate_marked(text: str, limit: int) -> str:
    """截断并**明确标注**——静默硬切会让模型把半截输出当成全文下结论。"""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [内容超过 {limit} 字符，已截断]"


def _read_capped(client, url: str) -> str:
    """GET ``url`` and return the body text, capped at ``_MAX_FETCH_BYTES``.

    **手动跟随重定向**（逐跳校验）：旧实现用 ``follow_redirects=True``，
    httpx 只禁跨协议跳转（→file://），管不了跨网络边界——公网 URL 可被
    302 引到云元数据 / loopback / 内网段（SSRF）。现在每跳都重新校验协议
    与目标网段（见 ``_is_blocked_host``）。

    流式读取优先：``client.get()`` 是非流式的——整个响应体会先完整读进内存，
    才轮到我们截断，对一个 2GB 的直链内存峰值仍然是 2GB。``client.stream()``
    让我们读够上限就关掉连接。测试替身/极简适配器可能只实现了 ``get()``，
    那条路径退化为「get 后截断」。
    """
    current = url
    for _hop in range(_MAX_REDIRECTS + 1):
        body, redirect = _fetch_once(client, current)
        if redirect is None:
            return body
        if not redirect.lower().startswith(("http://", "https://")):
            return "__HTTP_ERROR__redirect to non-http(s) blocked"
        if _is_blocked_host(urlparse(redirect).hostname or ""):
            return "__HTTP_ERROR__redirect to private/loopback address blocked"
        current = redirect
    return "__HTTP_ERROR__too many redirects"


def _fetch_once(client, url: str):
    """单次请求（不自动跟随重定向）。

    Returns ``(body_text, None)`` 或 ``(None, next_url)``（3xx 跳转）。
    """
    if hasattr(client, "stream"):
        chunks: list = []
        total = 0
        truncated = False
        try:
            with client.stream("GET", url, follow_redirects=False) as resp:
                if 300 <= resp.status_code < 400:
                    loc = resp.headers.get("location")
                    if loc:
                        return None, urljoin(url, loc)
                    return f"__HTTP_ERROR__{resp.status_code}", None
                if resp.status_code >= 400:
                    return f"__HTTP_ERROR__{resp.status_code}", None
                for block in resp.iter_bytes(chunk_size=65536):
                    if not block:
                        continue
                    remaining = _MAX_FETCH_BYTES - total
                    if remaining <= 0:
                        truncated = True
                        break
                    if len(block) > remaining:
                        chunks.append(block[:remaining])
                        total += remaining
                        truncated = True
                        break
                    chunks.append(block)
                    total += len(block)
                encoding = getattr(resp, "encoding", None) or "utf-8"
            text = b"".join(chunks).decode(encoding, errors="replace")
            del chunks
            if truncated:
                text += f"\n… [响应超过 {_MAX_FETCH_BYTES // (1024 * 1024)} MB，已截断]"
            return text, None
        except AttributeError:
            # 仅当 client 没有真正的 stream 实现（测试替身/残缺实现）才回退。
            pass
        except Exception:
            # 真实传输失败（超时/断流/连接错误）必须向上抛：否则会对同一 URL
            # 再发一次完整请求（最坏 2×30s 且重复下载），且已读内容被整段丢弃。
            raise

    resp = client.get(url, follow_redirects=False)
    try:
        if 300 <= resp.status_code < 400:
            loc = resp.headers.get("location")
            if loc:
                return None, urljoin(url, loc)
            return f"__HTTP_ERROR__{resp.status_code}", None
        if resp.status_code >= 400:
            return f"__HTTP_ERROR__{resp.status_code}", None
        # 优先按字节裁剪（真实 httpx 的 .content），拿不到字节时退化为
        # .text（更轻量的响应/测试替身只实现 text）。
        raw = getattr(resp, "content", None)
        if isinstance(raw, (bytes, bytearray)):
            truncated = len(raw) > _MAX_FETCH_BYTES
            text = raw[:_MAX_FETCH_BYTES].decode(
                getattr(resp, "encoding", None) or "utf-8", errors="replace")
            del raw
        else:
            text = resp.text or ""
            truncated = len(text) > _MAX_FETCH_BYTES
            text = text[:_MAX_FETCH_BYTES]
    finally:
        resp.close()

    if truncated:
        text += f"\n… [响应超过 {_MAX_FETCH_BYTES // (1024 * 1024)} MB，已截断]"
    return text, None


class WebScanTool(Tool):
    name = "web_scan"
    description = "Fetch a URL and return its visible text content (HTML stripped, scripts removed)."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "raw": {"type": "boolean", "description": "Return raw HTML instead of extracted text (default false)."},
        },
        "required": ["url"],
    }

    config: AgentConfig
    registry: ToolRegistry
    _http: httpx.Client  # shared connection pool (lazy-init per instance)

    def bind(self, config: AgentConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry
        # follow_redirects=False：重定向由 _read_capped **手动**跟随并逐跳
        # 校验目标网段（SSRF 防护），httpx 的自动跟随只禁跨协议、不管跨网络。
        self._http = httpx.Client(timeout=30.0, follow_redirects=False,
                                  headers={"User-Agent": "base-agent/0.1"})

    def run(self, url: str, raw: bool = False) -> ToolResult:
        # 协议白名单：url 由模型提供，不加限制的话 file:// / gopher:// 等
        # 可被用来读取本地文件或探测内网。
        if not url.lower().startswith(("http://", "https://")):
            return ToolResult(False, error=f"Only http/https URLs are supported: {url[:80]}")
        http = getattr(self, "_http", None)
        if http is None:
            return ToolResult(False, error="web_scan 未绑定（registry.bind 未执行）")
        try:
            text = _read_capped(http, url)
        except httpx.TimeoutException as e:
            # 传输层错误透出 httpx 原文（如 "[Errno 61] Connection refused"）
            # 对模型难以据此改参数。分类给出可操作提示。
            return ToolResult(
                False,
                error=f"Request failed (network unreachable or timed out after 30s): "
                      f"{type(e).__name__}: {e}. Check the URL/host and retry, "
                      f"or use shell_run(curl) to diagnose connectivity.",
            )
        except httpx.HTTPError as e:
            return ToolResult(False, error=f"Request failed: {type(e).__name__}: {e}")
        except Exception as e:
            return ToolResult(False, error=f"Request failed: {type(e).__name__}: {e}")
        if text.startswith("__HTTP_ERROR__"):
            return ToolResult(False, error=f"HTTP {text[len('__HTTP_ERROR__'):]}")
        if raw:
            return ToolResult(True, output=_truncate_marked(text, 20000))
        if BeautifulSoup is not None:
            # 建完整 DOM 的内存峰值可达响应体的数十倍、耗时数秒。
            # _MAX_FETCH_BYTES 只限制了下载，这里再限制解析量。
            soup = BeautifulSoup(text[:_MAX_PARSE_CHARS], "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            content = soup.get_text(separator="\n")
        else:
            # bs4 缺失时的线性降噪回退：**不用正则**处理 script/style 块——
            # `<script.*?</script>` 配 re.S 在超长且标签不闭合的输入上会
            # 大量回溯（ReDoS，实测秒级到分钟级，阻塞 agent 线程）。改用
            # str.find 线性扫描，任何输入都是 O(n)。
            content = _strip_blocks_linear(text[:_MAX_PARSE_CHARS])
            import re
            content = re.sub(r"<[^>]+>", " ", content)
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        return ToolResult(True, output=_truncate_marked("\n".join(lines), 20000))


class WebExecJsTool(Tool):
    name = "webexec_js"
    description = (
        "Control a browser via a Playwright-style bridge: navigate to a URL and "
        "evaluate JavaScript. Requires the agent's browser_endpoint to be set; "
        "otherwise returns guidance on enabling it."
    )
    # Browser-scoped: does not touch the local filesystem or system. The agent
    # is expected to ask the user (via ask_user) before irreversible web actions.
    destructive = False
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to navigate to (optional if only evaluating)."},
            "script": {"type": "string", "description": "JavaScript to evaluate in the page."},
            "action": {"type": "string", "description": "Optional action: 'click','type','screenshot'."},
            "selector": {"type": "string"},
            "value": {"type": "string"},
        },
        "required": ["script"],
    }

    config: AgentConfig
    registry: ToolRegistry
    _http: httpx.Client  # shared connection pool (lazy-init per instance)

    def bind(self, config: AgentConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry
        self._http = httpx.Client(timeout=60.0)

    def run(self, script: str, url: str = "", action: str = "",
            selector: str = "", value: str = "") -> ToolResult:
        endpoint = getattr(self.config, "browser_endpoint", "") or ""
        if not endpoint:
            return ToolResult(False, error=(
                "Browser bridge not configured. Set config.browser_endpoint to a "
                "Playwright HTTP bridge (e.g. http://localhost:9222) to use webexec_js."
            ))
        payload = {"url": url, "script": script, "action": action,
                   "selector": selector, "value": value}
        try:
            resp = self._http.post(endpoint.rstrip("/") + "/exec", json=payload)
        except Exception as e:
            return ToolResult(False, error=f"Browser bridge error: {e}")
        try:
            if resp.status_code >= 400:
                return ToolResult(False, error=f"HTTP {resp.status_code}: {resp.text[:500]}")
            # content-type 声称 JSON 而 body 畸形时 resp.json() 抛
            # JSONDecodeError（ValueError 子类）——旧实现只 catch 了外面的
            # 请求异常，这个异常会逃逸成无上下文的 "JSONDecodeError: ..."。
            try:
                data = (resp.json() if resp.headers.get("content-type", "")
                        .startswith("application/json") else {"result": resp.text})
            except ValueError:
                data = {"result": resp.text[:2000],
                        "note": "bridge 返回了非 JSON 内容，已按原文截断返回"}
        finally:
            resp.close()
        return ToolResult(True, output=json.dumps(data, ensure_ascii=False)[:8000])
