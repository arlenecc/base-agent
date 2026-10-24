"""Web tools: web_scan (fetch + extract text) and webexec_js (browser control)."""
from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
import time
from typing import Any, List, Optional, Tuple
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
# ---- web_scan 增强：动态页 / API 数据源 / iframe（2026-10-24）----
# 现代站点大量内容是 JS 异步渲染的（Azure Search UI、React SPA 等），
# 直接抓 HTML 只会得到 "Loading jobs please wait..." 这类空壳。web_scan
# 在不引入新工具、不引浏览器的前提下做三层兜底：
#   1) JSON 直采：响应体本身就是 JSON（模型给了 API 端点）→ 结构化摘要；
#   2) iframe 嵌套：抓子框架内容并拼回（有深度/数量上限）；
#   3) 端点发现：页面看着是「JS 渲染空壳」时，从内联脚本与同源 .js 里挖
#      候选 API 端点，逐个试探，命中即返回其数据。
_MAX_IFRAMES = 3            # 最多深入几个 iframe
_MAX_JS_FILES = 6           # 最多下载几个同源 .js 挖端点
_MAX_API_CANDIDATES = 8     # 最多试探几个候选 URL（含参数变体）
_MAX_SUB_FETCHES = 10       # 单次 web_scan 的子请求总上限
_DISCOVERY_DEADLINE_S = 25.0  # 发现阶段的总时间预算（含子请求）
_MIN_VISIBLE_CHARS = 200    # 可见文本少于此长度视为「空壳」
# 「空壳」特征串（加载提示/反爬挑战页）。
_LOADER_MARKERS = (
    "loading jobs please wait", "please wait while loading",
    "just a moment", "enable javascript", "checking your browser",
)
# 从脚本里挖端点：绝对 URL 或以 / 开头、含 api/ashx/asmx/svc/json/search 的路径。
# 注意 ``/`` 后的字符类必须用 ``*`` 不是 ``+``：``/api/...`` 里 ``/`` 与 ``api``
# 之间没有字符，``+`` 会永远匹配不上（实测踩过）。
_ENDPOINT_RE = re.compile(
    r"""["']((?:https?://[^"'\s]{4,160})|(?:/[A-Za-z0-9_\-./]*"""
    r"""(?:api|ashx|asmx|svc|json|search)[A-Za-z0-9_\-./?=&%]*))["']""",
    re.IGNORECASE,
)
# 明确不是数据端点的静态资源后缀。
_STATIC_ASSET_RE = re.compile(
    r"\.(?:png|jpe?g|gif|svg|css|js|woff2?|ttf|eot|ico|mp4|webp)(?:\?|$)",
    re.IGNORECASE,
)


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


def _visible_text(html: str) -> str:
    """HTML → 可见文本（剥 script/style/noscript，去空行）。bs4 缺失时线性回退。"""
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html[:_MAX_PARSE_CHARS], "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        content = soup.get_text(separator="\n")
    else:
        # bs4 缺失时的线性降噪回退：**不用正则**处理 script/style 块——
        # `<script.*?</script>` 配 re.S 在超长且标签不闭合的输入上会大量回溯
        # （ReDoS）。改用 str.find 线性扫描，任何输入都是 O(n)。
        content = _strip_blocks_linear(html[:_MAX_PARSE_CHARS])
        content = re.sub(r"<[^>]+>", " ", content)
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    return "\n".join(lines)


def _looks_js_rendered(text: str) -> bool:
    """可见文本是否薄到像「JS 渲染空壳」（或含加载提示串）。"""
    t = (text or "").strip().lower()
    if len(t) < _MIN_VISIBLE_CHARS:
        return True
    return any(marker in t for marker in _LOADER_MARKERS)


def _summarize_json(obj: Any, max_items: int = 10,
                    max_chars: int = 6000) -> str:
    """把任意 JSON 响应压成「模型可读」的结构化摘要（不原样倾倒几十万字符）。"""
    items: List[Any] = []
    lines: List[str] = []
    key: Optional[str] = None
    if isinstance(obj, dict):
        lines.append("JSON 对象，顶层键: " + ", ".join(list(obj)[:20]))
        for k in ("Results", "results", "items", "data", "value", "jobs"):
            v = obj.get(k)
            if isinstance(v, list) and v:
                items, key = v, k
                break
        if key is None:
            for k, v in obj.items():
                if isinstance(v, list) and v and k not in ("Facets", "facets"):
                    items, key = v, k
                    break
        if key is not None:
            cnt = obj.get("Count") or obj.get("count") or obj.get("total")
            lines.append(f"列表字段「{key}」: {len(items)} 项"
                         + (f"（总数 {cnt}）" if cnt else ""))
    elif isinstance(obj, list):
        items = obj
        lines.append(f"JSON 数组: {len(items)} 项")
    for it in items[:max_items]:
        if isinstance(it, dict):
            # Azure Search 之类的响应把实体嵌在内层（如 Document），自动展开
            pick = {k: v for k, v in it.items()
                    if isinstance(v, (str, int, float, bool)) and not k.startswith("@")}
            if len(pick) <= 2:
                inner = it.get("Document") or it.get("document") or it.get("doc")
                if isinstance(inner, dict):
                    pick = {k: v for k, v in inner.items()
                            if isinstance(v, (str, int, float, bool))}
            pick = {k: (v if not isinstance(v, str) or len(v) < 160 else v[:157] + "…")
                    for k, v in list(pick.items())[:14]}
            lines.append("- " + json.dumps(pick, ensure_ascii=False)[:400])
        else:
            lines.append(f"- {str(it)[:240]}")
    if len(items) > max_items:
        lines.append(f"… （其余 {len(items) - max_items} 项从略）")
    out = "\n".join(lines)
    if len(out) > max_chars:
        out = out[:max_chars] + "\n… [摘要超过字符上限，已截断]"
    return out or "（JSON 无可用字段）"


def _try_parse_json(text: str) -> Optional[Any]:
    """响应体是 JSON 时返回解析结果，否则 None（截断/HTML 都安全返回 None）。"""
    s = (text or "").lstrip()
    if not s or s[0] not in "{[":
        return None
    try:
        return json.loads(s[:_MAX_PARSE_CHARS])
    except (ValueError, RecursionError):
        return None


def _extract_iframe_urls(html: str, base_url: str) -> List[str]:
    """挖 <iframe src>（绝对化 + 去重 + 只留 http(s)）。"""
    urls: List[str] = []
    seen: set = set()
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html[:_MAX_PARSE_CHARS], "html.parser")
        srcs = [f.get("src") for f in soup.find_all("iframe")]
    else:
        srcs = re.findall(r"<iframe[^>]+src=[\"']([^\"']+)[\"']", html, re.I)
    for src in srcs:
        if not src or not isinstance(src, str):
            continue
        abs_url = urljoin(base_url, src.strip())
        p = urlparse(abs_url)
        if p.scheme not in ("http", "https") or not p.netloc:
            continue
        if abs_url in seen:
            continue
        seen.add(abs_url)
        urls.append(abs_url)
    return urls


def _same_origin(a: str, b: str) -> bool:
    pa, pb = urlparse(a), urlparse(b)
    return (pa.scheme, pa.netloc) == (pb.scheme, pb.netloc)


def _sub_fetch_allowed(page_url: str, target: str) -> bool:
    """子请求（iframe / 候选端点）目标是否放行。

    同源直接放行——用户已经选择访问该站，抓它自己的子框架/接口不扩大攻击面；
    跨源才做私网/回环/链路本地判定（防公网页把请求引向内网）。

    透明代理/VPN 环境下所有域名都解析到 fake-ip 段（198.18.x.x），严格判定
    会把一切误判成内网（实测本机就是这样，导致所有候选端点被跳过）——同源
    豁免让这类环境可用，同时不放松对真正跨网目标的防护。
    """
    if _same_origin(target, page_url):
        return True
    return not _is_blocked_host(urlparse(target).hostname or "")


def _scan_endpoints(blob: str, page_url: str) -> List[str]:
    """从脚本文本里挖候选数据端点（去重、去静态资源、同源优先）。"""
    out: List[str] = []
    seen: set = set()
    for raw in _ENDPOINT_RE.findall(blob or ""):
        if isinstance(raw, tuple):
            raw = raw[0]
        cand = raw.strip()
        if not cand or _STATIC_ASSET_RE.search(cand):
            continue
        abs_url = cand if cand.startswith("http") else urljoin(page_url, cand)
        p = urlparse(abs_url)
        if p.scheme not in ("http", "https") or not p.netloc:
            continue
        if abs_url in seen or abs_url == page_url:
            continue
        seen.add(abs_url)
        out.append(abs_url)
    # 同源的排前面（跨域端点大概率不是本站数据）
    out.sort(key=lambda u: (not _same_origin(u, page_url),))
    return out


def _mining_literals(blob: str, limit: int = 4) -> List[str]:
    """从脚本文本里挖「短标识符字面量」当查询参数值候选。

    用途：端点多形如 ``/api/X?Param=``（页面用 ``+变量`` 拼值），值候选只能
    从脚本自身找——如 Azure Search UI 的 settings 里 ``module:"job"``，
    ``job`` 正是该站搜索接口的可用值。

    优先取 **语义键的值**（module/type/category/mode/kind/entity/source/feed），
    再退回泛化的短词扫描；后者在特性检测脚本里大量出现
    （ontouchstart/touch…），价值低，只作补充。
    """
    out: List[str] = []
    seen: set = set()
    for key in ("module", "type", "category", "mode", "kind", "entity",
                "source", "feed", "searchtype"):
        for m in re.findall(rf'{key}\s*:\s*["\']([a-z][a-z0-9_\-]{{1,20}})["\']',
                            blob or "", re.I):
            if m not in seen:
                seen.add(m)
                out.append(m)
            if len(out) >= limit:
                return out
    for m in re.findall(r"""["']([a-z][a-z0-9_]{2,15})["']""", blob or ""):
        if m in seen or m in ("true", "false", "null", "none", "undefined"):
            continue
        seen.add(m)
        out.append(m)
        if len(out) >= limit:
            break
    return out


class WebScanTool(Tool):
    name = "web_scan"
    description = (
        "Fetch a URL and return its usable content. Handles three shapes: "
        "(1) normal HTML → visible text; (2) a JSON API endpoint → structured "
        "summary of the payload; (3) JS-rendered/empty-shell pages → follows "
        "iframes and auto-discovers the page's own XHR/JSON endpoints (from "
        "inline and same-origin scripts) and returns their data when found. "
        "Use raw=true to get the original HTML."
    )
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

        # ---- 形状 1：响应体本身就是 JSON（API 类数据源）----
        # 模型直接给 JSON 端点时，旧实现会把它当 HTML 剥成一行乱码。
        payload = _try_parse_json(text)
        if payload is not None:
            return ToolResult(
                True,
                output=_truncate_marked(
                    f"[JSON 响应] {url}\n{_summarize_json(payload)}", 20000))

        # ---- 形状 2/3：HTML ----
        visible = _visible_text(text)
        sections: List[str] = [visible] if visible else []
        notes: List[str] = []

        # iframe 嵌套：子框架内容拼回（站内常用 iframe 嵌搜索/列表组件）。
        for ifr in _extract_iframe_urls(text, url)[:_MAX_IFRAMES]:
            if not _sub_fetch_allowed(url, ifr):
                notes.append(f"[iframe 已跳过（内网/回环目标）] {ifr}")
                continue
            try:
                sub = _read_capped(http, ifr)
            except Exception as e:
                notes.append(f"[iframe 抓取失败] {ifr}: {type(e).__name__}")
                continue
            if sub.startswith("__HTTP_ERROR__"):
                continue
            sub_visible = _visible_text(sub)
            if sub_visible.strip():
                sections.append(f"[iframe: {ifr}]\n{sub_visible}")

        # ---- 形状 3：JS 渲染空壳 → 自动发现站点自己的数据端点 ----
        if _looks_js_rendered(visible):
            sections, notes = self._discover_and_fetch(
                http, url, text, sections, notes)

        if notes:
            sections.append("\n".join(notes))
        out = "\n\n".join(s for s in sections if s.strip())
        if not out.strip():
            # 仍然空：把已发现的候选端点回给模型（比一句「无内容」有用）。
            cands = _scan_endpoints(text, url)
            hint = ("页面未提取到可见文本（可能是纯 JS 渲染或需要登录）。"
                    "可尝试 raw=true 查看原始 HTML。")
            if cands:
                hint += " 页面脚本中发现的候选数据端点:\n" + "\n".join(
                    f"  - {c}" for c in cands[:_MAX_API_CANDIDATES])
            return ToolResult(True, output=_truncate_marked(hint, 20000))
        return ToolResult(True, output=_truncate_marked(out, 20000))

    # ------------------------------------------------------------------
    # 动态页兜底：从页面脚本里挖站点自己的数据端点并试探
    # ------------------------------------------------------------------

    def _discover_and_fetch(self, http, url: str, html: str,
                            sections: List[str], notes: List[str]):
        """空壳页的端点发现 + 试探。

        顺序：内联脚本 → 同源 .js（有数量/体积上限）。候选逐个 GET，命中
        「有实质内容的 JSON」即采用并停止（带总时间与总子请求预算）。
        """
        deadline = time.monotonic() + _DISCOVERY_DEADLINE_S
        budget = {"sub": _MAX_SUB_FETCHES}

        def _sub_fetch(u: str) -> Optional[str]:
            if budget["sub"] <= 0 or time.monotonic() > deadline:
                return None
            budget["sub"] -= 1
            try:
                return _read_capped(http, u)
            except Exception:
                return None

        # 1) 内联脚本（整页 HTML 里的脚本段）
        inline = "\n".join(
            m for m in re.findall(r"<script[^>]*>(.*?)</script>", html,
                                  re.S | re.I))
        candidates = _scan_endpoints(inline, url)

        # 2) 同源 .js 文件（挖得更深，如 azure-search-ui-settings.min.js）
        #    注意：这里**不能**过滤 .js——端点字符串恰恰藏在 .js 里（静态资源
        #    过滤只用于端点候选本身，不用于「要下载来挖端点的脚本」）。
        #    排序：名字像配置/接口的脚本优先（settings/config/api/search/app），
        #    端点声明最可能在那里；jquery 之类库脚本排最后。
        js_urls = []
        for m in re.findall(r"<script[^>]+src=[\"']([^\"']+)[\"']", html, re.I):
            abs_u = urljoin(url, m.strip())
            if _same_origin(abs_u, url):
                js_urls.append(abs_u)

        def _js_priority(u: str) -> int:
            name = u.rsplit("/", 1)[-1].lower()
            if re.search(r"setting|config|option|api|search|main|app|bundle", name):
                return 0
            return 1

        js_urls.sort(key=_js_priority)
        # (endpoint, 来源脚本文本) —— 参数值候选从同一脚本里挖
        found: List[Tuple[str, str]] = [(c, inline) for c in candidates]
        js_blobs: List[str] = []
        for ju in js_urls[:_MAX_JS_FILES]:
            blob = _sub_fetch(ju)
            if blob and not blob.startswith("__HTTP_ERROR__"):
                js_blobs.append(blob)
                for c in _scan_endpoints(blob, url):
                    if c not in [u for u, _ in found]:
                        found.append((c, blob))
            if time.monotonic() > deadline:
                break

        # 数据类端点优先（search/list/query/job/post/get…）；页面型候选
        # （.aspx/.html/.php…）排最后——它们是导航链接不是数据接口。
        _PAGE_EXT_RE = re.compile(r"\.(?:aspx?|html?|php|jsp|cfm)(?:\?|$)",
                                  re.I)

        def _data_ish(u: str) -> int:
            name = u.rsplit("/", 1)[-1].lower()
            if _PAGE_EXT_RE.search(name):
                return 2
            if re.search(r"search|list|query|find|job|post|data|feed", name):
                return 0
            return 1

        found.sort(key=lambda pair: _data_ish(pair[0]))

        # 查询片段挖掘：端点路径与查询串常**分开放在不同脚本**里
        # （如 settings js 只给 "/api/SearchIndex/GetSearch"，" ?SearchText="
        # 写在另一个 js 的拼接处，且常嵌在更长字符串里）。从所有脚本文本中
        # 提取 ``?Param=`` 形状的片段，供裸路径端点补全。
        query_frags: List[str] = []
        for blob in [inline] + list(js_blobs):
            for m in re.findall(r"\?([A-Za-z_][A-Za-z0-9_]{1,30})=", blob or ""):
                frag = f"?{m}="
                if frag not in query_frags:
                    query_frags.append(frag)
        # 查询片段排序：参数名像「检索词」的优先（SearchText/q/query/text/
        # keyword/filter/category…），ip/utm/token 之类靠后——前者才是列表
        # 数据接口的入参。
        def _frag_priority(f: str) -> int:
            name = f.lstrip("?").split("=")[0].lower()
            if re.search(r"search|query|^q$|keyword|text|filter|category|"
                         r"type|term|find", name):
                return 0
            if re.search(r"^ip|utm|token|key|lang|locale|v$|cb$", name):
                return 2
            return 1

        query_frags.sort(key=_frag_priority)
        # 值字面量也跨脚本汇总：查询参数名在一个 js、它的可用值（module:"job"）
        # 常在另一个 js 里。
        all_blobs = "\n".join([inline] + list(js_blobs))

        tried: List[str] = []
        for cand, src_blob in found:
            if len(tried) >= _MAX_API_CANDIDATES or time.monotonic() > deadline:
                break
            if not _sub_fetch_allowed(url, cand):
                notes.append(f"[端点已跳过（内网/回环目标）] {cand}")
                continue
            # 端点常是「裸路径」或 ``?Param=`` 前缀（值由页面用 +变量 拼）。
            # 每个候选最多 3 个变体（裸 / 通配 / 最佳字面量），避免一个坏候选
            # 吃光总预算。
            variants = [cand]
            if "?" in cand:
                base_part, q = cand.split("?", 1)
                if q.endswith("=") or "=&" in q or q == "":
                    prefix = q if q.endswith("=") else (q + "=" if q else "")
                    variants.append(f"{base_part}?{prefix}*")
                    for lit in _mining_literals(all_blobs, limit=1):
                        variants.append(f"{base_part}?{prefix}{lit}")
            elif query_frags:
                frag = query_frags[0]
                variants.append(cand + frag + "*")
                for lit in _mining_literals(all_blobs, limit=1):
                    variants.append(cand + frag + lit)
            for attempt in variants:
                if len(tried) >= _MAX_API_CANDIDATES or time.monotonic() > deadline:
                    break
                tried.append(attempt)
                body = _sub_fetch(attempt)
                if not body or body.startswith("__HTTP_ERROR__"):
                    continue
                data = _try_parse_json(body)
                if data is None:
                    continue
                summary = _summarize_json(data)
                # 「有实质内容」：至少解析出列表项或非空对象
                substantive = (isinstance(data, list) and data) or (
                    isinstance(data, dict) and any(
                        isinstance(v, list) and v for v in data.values()))
                if substantive and summary.strip() not in ("", "（JSON 无可用字段）"):
                    sections.append(
                        f"[自动发现的数据端点] {attempt}\n{summary}")
                    notes.append(
                        f"[提示] 页面为 JS 动态渲染，以上数据来自其脚本中发现的接口 {attempt}")
                    return sections, notes

        if tried:
            notes.append("[提示] 页面为 JS 动态渲染，已试探以下脚本中的数据端点但未取得可用数据: "
                         + ", ".join(tried))
        else:
            notes.append("[提示] 页面为 JS 动态渲染，未在脚本中发现数据端点")
        return sections, notes


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
