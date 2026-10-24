"""web_scan 增强（2026-10-24）：JSON API 直采 / 动态页端点发现 / iframe 嵌套。

目标：不新增工具、不引浏览器，让 web_scan 尽量不返回空。三类能力：
  1. 响应体是 JSON（API 类数据源）→ 结构化摘要；
  2. JS 渲染空壳 → 从内联/同源脚本挖站点自己的数据端点并试探；
  3. iframe 嵌套 → 子框架内容拼回。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.tools import web as web_mod  # noqa: E402


class _Resp:
    def __init__(self, text="", status=200, headers=None):
        self.status_code = status
        self.text = text
        self.headers = headers or {"Content-Type": "text/html"}

    def iter_bytes(self, chunk_size=65536):
        return iter([self.text.encode("utf-8")])

    @property
    def content(self):
        return self.text.encode("utf-8")

    @property
    def encoding(self):
        return "utf-8"

    def close(self):
        pass


class _StreamCtx:
    def __init__(self, resp):
        self._resp = resp

    def __enter__(self):
        return self._resp

    def __exit__(self, *exc):
        return False


def _install_client(monkeypatch, routes):
    """按 URL 路由返回预置内容的假 httpx client（打桩 stream/get）。"""
    def _fake_stream(self, method, url, **kw):
        text = routes.get(url, "")
        return _StreamCtx(_Resp(text))

    def _fake_get(self, url, **kw):
        return _Resp(routes.get(url, ""))

    monkeypatch.setattr(web_mod.httpx.Client, "stream", _fake_stream)
    monkeypatch.setattr(web_mod.httpx.Client, "get", _fake_get)


def _tool(config):
    t = web_mod.WebScanTool()
    t.bind(config, object())
    return t


# ---------------------------------------------------------------------------
# 1) JSON API 直采
# ---------------------------------------------------------------------------

def test_web_scan_returns_json_summary_for_api_endpoint(monkeypatch, config):
    payload = {"Count": 3, "Results": [
        {"position_Subject": "Primary Teacher", "employer_Name": "SD28"},
        {"position_Subject": "ECE", "employer_Name": "Nuu-chah-nulth"},
        {"position_Subject": "HVAC Technician", "employer_Name": "SD37"},
    ]}
    _install_client(monkeypatch, {
        "https://api.example.com/jobs": json.dumps(payload)})
    r = _tool(config).run("https://api.example.com/jobs")
    assert r.success
    assert "[JSON 响应]" in r.output
    assert "Primary Teacher" in r.output
    assert "ECE" in r.output
    assert "总数 3" in r.output


def test_web_scan_json_array_endpoint(monkeypatch, config):
    payload = [{"id": 1, "name": "alpha"}, {"id": 2, "name": "beta"}]
    _install_client(monkeypatch, {
        "https://api.example.com/list": json.dumps(payload)})
    r = _tool(config).run("https://api.example.com/list")
    assert r.success
    assert "JSON 数组: 2 项" in r.output
    assert "alpha" in r.output


# ---------------------------------------------------------------------------
# 2) 动态页端点发现（还原 applytoeducation 的场景）
# ---------------------------------------------------------------------------

_DYNAMIC_HTML = """<html><head>
<script src="/jscripts/azure-search-ui-settings.min.js"></script>
<script src="/jscripts/azure-search-ui.min.js"></script>
</head><body>
<div>Loading jobs please wait...</div>
</body></html>"""

_SETTINGS_JS = ('var settings={api_request:{module:"job",'
                'search_submit:base+"/api/SearchIndex/GetSearch"}};')

_MAIN_JS = ('function f(){url=settings.api_request.search_submit'
            '+"/api/SearchIndex/GetSearch?SearchText="+module;}')

_API_JSON = json.dumps({
    "Count": 2,
    "Results": [
        {"Document": {"position_Subject": "Primary Teacher (Term)",
                      "employer_Name": "Salt Spring Centre School",
                      "posted_On": "2026-10-24T00:00"}},
        {"Document": {"position_Subject": "KG Teacher Needed",
                      "employer_Name": "Khalsa Schools",
                      "posted_On": "2026-10-24T00:00"}},
    ],
})


def test_web_scan_discovers_js_rendered_data_endpoint(monkeypatch, config):
    """空壳页 → 自动挖出站点数据端点并返回其内容（不新增工具）。"""
    routes = {
        "https://jobs.example.com/AttSearch.aspx": _DYNAMIC_HTML,
        "https://jobs.example.com/jscripts/azure-search-ui-settings.min.js": _SETTINGS_JS,
        "https://jobs.example.com/jscripts/azure-search-ui.min.js": _MAIN_JS,
        "https://jobs.example.com/api/SearchIndex/GetSearch?SearchText=job": _API_JSON,
        # 其它变体（* / 空值）返回空列表 → 不命中，继续尝试
        "https://jobs.example.com/api/SearchIndex/GetSearch?SearchText=*": "{}",
    }
    _install_client(monkeypatch, routes)
    r = _tool(config).run("https://jobs.example.com/AttSearch.aspx")
    assert r.success
    assert "[自动发现的数据端点]" in r.output
    assert "GetSearch" in r.output
    assert "Primary Teacher (Term)" in r.output   # Document 内层被展开
    assert "KG Teacher Needed" in r.output


def test_web_scan_no_candidates_returns_hint_not_empty(monkeypatch, config):
    """挖不到端点也要给可操作提示（而不是空输出）。"""
    _install_client(monkeypatch, {
        "https://jobs.example.com/blank": "<html><body>Loading…</body></html>"})
    r = _tool(config).run("https://jobs.example.com/blank")
    assert r.success
    assert "JS" in r.output or "动态渲染" in r.output
    assert r.output.strip() != ""


# ---------------------------------------------------------------------------
# 3) iframe 嵌套
# ---------------------------------------------------------------------------

def test_web_scan_includes_iframe_content(monkeypatch, config):
    page = ('<html><body><p>外壳文本</p>'
            '<iframe src="https://jobs.example.com/embed/list"></iframe>'
            '</body></html>')
    iframe_html = "<html><body><p>iframe 里的职位列表内容</p></body></html>"
    _install_client(monkeypatch, {
        "https://jobs.example.com/page": page,
        "https://jobs.example.com/embed/list": iframe_html,
    })
    r = _tool(config).run("https://jobs.example.com/page")
    assert r.success
    assert "外壳文本" in r.output
    assert "iframe 里的职位列表内容" in r.output
    assert "[iframe:" in r.output


# ---------------------------------------------------------------------------
# 4) 既有行为不回退 + 无新工具
# ---------------------------------------------------------------------------

def test_web_scan_still_extracts_plain_html(monkeypatch, config):
    html = "<html><body><h1>Hi</h1><p>Hello world</p><script>bad()</script></body></html>"
    _install_client(monkeypatch, {"https://example.com": html})
    r = _tool(config).run("https://example.com")
    assert r.success
    assert "Hi" in r.output and "Hello world" in r.output
    assert "bad()" not in r.output


def test_web_scan_raw_mode_unchanged(monkeypatch, config):
    html = "<html><body><p>x</p></body></html>"
    _install_client(monkeypatch, {"https://example.com": html})
    r = _tool(config).run("https://example.com", raw=True)
    assert r.success and "<html>" in r.output


def test_web_scan_rejects_non_http(config):
    r = _tool(config).run("file:///etc/passwd")
    assert not r.success and "http" in r.error.lower()


def test_endpoint_regex_matches_bare_api_path():
    """``/`` 与 ``api`` 之间无字符的正则回归（旧写法用 + 永远匹配不上）。"""
    from agent.tools.web import _scan_endpoints
    s = 'x+"/api/SearchIndex/GetSearch",y:"/api/v1/items"'
    hits = _scan_endpoints(s, "https://jobs.example.com/p")
    assert "https://jobs.example.com/api/SearchIndex/GetSearch" in hits
    assert "https://jobs.example.com/api/v1/items" in hits


def test_sub_fetch_allowed_same_origin_without_dns(config):
    """同源子请求不做私网判定（透明代理环境下所有域名都解析到 fake-ip）。"""
    from agent.tools.web import _sub_fetch_allowed
    page = "https://jobs.example.com/AttSearch.aspx"
    assert _sub_fetch_allowed(page, "https://jobs.example.com/api/x")
    assert not _sub_fetch_allowed(page, "https://169.254.169.254/latest/meta-data")
