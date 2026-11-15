"""search_tool: 引擎适配、去重/过滤/排序与并发聚合。"""
from __future__ import annotations

import json

from agent.tools import search as sm
from agent.tools.search import SearchTool


# ---------------------------------------------------------------------------
# 引擎解析
# ---------------------------------------------------------------------------


def test_parse_bing_unwraps_redirect_and_skips_ads():
    html = """
    <ol id="b_results">
      <li class="b_algo">
        <h2><a href="https://example.com/a">Bing A</a></h2>
        <div class="b_caption"><p>Bing snippet A</p></div>
      </li>
      <li class="b_algo">
        <h2><a href="https://www.bing.com/ck/a?u=a1aHR0cHM6Ly9leGFtcGxlLmNvbS9i">Bing B</a></h2>
      </li>
      <li class="b_algo b_ad"><h2><a href="https://ads.example.com/">AD</a></h2></li>
    </ol>
    """
    results = sm._parse_bing(html, "https://www.bing.com/search?q=x")
    assert [r.title for r in results] == ["Bing A", "Bing B"]
    assert results[0].snippet == "Bing snippet A"
    assert results[1].url == "https://example.com/b"


def test_parse_ddg_unwraps_uddg():
    html = """
    <div class="result results_links web-result">
      <h2 class="result__title">
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fddg">DDG Title</a>
      </h2>
      <a class="result__snippet">DDG snippet</a>
    </div>
    """
    results = sm._parse_ddg(html, "https://html.duckduckgo.com/html/")
    assert len(results) == 1
    assert results[0].url == "https://example.com/ddg"
    assert results[0].title == "DDG Title"
    assert results[0].snippet == "DDG snippet"


def test_parse_google_wml_and_fallback():
    html = """
    <div class="zMzFAb">
      <a class="fuLhoc" href="/url?q=https://example.com/g&amp;sa=U">
        <span class="CVA68e">Google Title</span>
      </a>
      <div class="taTFJ"><span class="FrIlee">Google snippet</span></div>
    </div>
    """
    results = sm._parse_google(html, "https://www.google.com/wml/search?q=x")
    assert len(results) == 1
    assert results[0].url == "https://example.com/g"
    assert results[0].title == "Google Title"

    fallback = '<a href="/url?q=https://example.com/f">Fallback Title</a>'
    results = sm._parse_google(fallback, "https://www.google.com/wml/search")
    assert results and results[0].url == "https://example.com/f"


def test_parse_brave_embedded_js_payload():
    html = """
    <script>
    kit.start(app, element, {
      node_ids: [0, 1],
      data: [{type:"data", data: {body: {response: {web: {results: [
        {title:"Brave Title", url:"https://example.com/brave",
         description:"Brave snippet", page_age:null}
      ]}}}}}]
    });
    </script>
    """
    results = sm._parse_brave(html, "https://search.brave.com/search")
    assert len(results) == 1
    assert results[0].url == "https://example.com/brave"
    assert results[0].snippet == "Brave snippet"


def test_parse_yahoo_unwraps_ru():
    html = """
    <div class="algo-sr">
      <div class="compTitle">
        <a href="https://r.search.yahoo.com/_ylt=X/RU=https%3a%2f%2fexample.com%2fy/RK=2/RS=x-">
          <h3>Yahoo Title</h3>
        </a>
      </div>
      <div class="compText">Yahoo snippet</div>
    </div>
    """
    results = sm._parse_yahoo(html, "https://search.yahoo.com/search?p=x")
    assert len(results) == 1
    assert results[0].url == "https://example.com/y"
    assert results[0].title == "Yahoo Title"


def test_parse_mojeek():
    html = """
    <ul class="results-standard">
      <li>
        <a class="ob" href="https://example.com/m"></a>
        <h2><a href="https://example.com/m">Mojeek Title</a></h2>
        <p class="s">Mojeek snippet</p>
      </li>
    </ul>
    """
    results = sm._parse_mojeek(html, "https://www.mojeek.com/search?q=x")
    assert len(results) == 1
    assert results[0].url == "https://example.com/m"
    assert results[0].snippet == "Mojeek snippet"


def test_parse_baidu_json():
    payload = json.dumps({
        "feed": {"entry": [
            {"title": "Baidu Title", "url": "https://example.com/bd",
             "abs": "Baidu snippet"},
            {"title": "", "url": "https://example.com/empty"},
        ]}
    })
    results = sm._parse_baidu(payload, "https://www.baidu.com/s")
    assert len(results) == 1
    assert results[0].title == "Baidu Title"
    assert results[0].snippet == "Baidu snippet"


def test_parse_sogou():
    html = """
    <div class="vrwrap">
      <h3 class="vr-title"><a href="https://example.com/s">Sogou Title</a></h3>
      <div class="fz-mid space-txt">Sogou snippet</div>
    </div>
    """
    results = sm._parse_sogou(html, "https://www.sogou.com/web?query=x")
    assert len(results) == 1
    assert results[0].url == "https://example.com/s"
    assert results[0].snippet == "Sogou snippet"


def test_parse_360_prefers_data_mdurl():
    html = """
    <li class="res-list">
      <h3 class="res-title">
        <a href="#" data-mdurl="https://example.com/q">360 Title</a>
      </h3>
      <p class="res-desc">360 snippet</p>
    </li>
    """
    results = sm._parse_360(html, "https://www.so.com/s?q=x")
    assert len(results) == 1
    assert results[0].url == "https://example.com/q"
    assert results[0].snippet == "360 snippet"


def test_js_object_to_json_handles_unquoted_keys_and_js_literals():
    src = ("{type:'data', ok:true, n:null, yes:!0, no:!1, nothing:void 0, "
           "list:[1,2], nested:{a:'b'}}")
    data = json.loads(sm._js_object_to_json(src))
    assert data == {"type": "data", "ok": True, "n": None,
                    "yes": True, "no": False, "nothing": None,
                    "list": [1, 2], "nested": {"a": "b"}}


# ---------------------------------------------------------------------------
# 去重 / 过滤 / 排序
# ---------------------------------------------------------------------------


def test_merge_ignores_scheme_www_fragment_tracking():
    runs = [
        sm._EngineRun("bing", [sm._RawResult(
            "Same", "https://www.example.com/a/?utm_source=x#top", "", 1)]),
        sm._EngineRun("google", [sm._RawResult(
            "Same", "http://example.com/a", "", 2)]),
    ]
    merged = sm._merge_results(runs)
    assert len(merged) == 1
    assert merged[0].engines == {"bing", "google"}


def test_filter_drops_engine_self_links_and_non_pages():
    runs = [
        sm._EngineRun("bing", [
            sm._RawResult("Search", "https://www.bing.com/search?q=x", "", 1),
            sm._RawResult("JS", "javascript:void(0)", "", 2),
            sm._RawResult("Image", "https://cdn.example.com/a.png", "", 3),
            sm._RawResult("", "https://example.com/empty-title", "", 4),
            sm._RawResult("Good", "https://example.com/good", "", 5),
        ]),
    ]
    merged = sm._merge_results(runs)
    assert [r.url for r in merged] == ["https://example.com/good"]


def test_filter_keeps_360_baike_but_drops_vertical_onebox():
    runs = [
        sm._EngineRun("360", [
            sm._RawResult("Baike", "https://baike.so.com/doc/1", "", 1),
            sm._RawResult("Translate", "https://fanyi.so.com/?src=onebox", "", 2),
            sm._RawResult("News", "http://news.so.com/ns?q=x", "", 3),
        ]),
    ]
    merged = sm._merge_results(runs)
    assert [r.url for r in merged] == ["https://baike.so.com/doc/1"]


def test_ranking_prefers_multi_engine_agreement():
    q = sm._Query.build("test")
    single = sm._Result("Single", "https://example.com/single",
                        engines={"mojeek"}, ranks=[("mojeek", 1)])
    multi = sm._Result("Multi", "https://example.com/multi",
                       engines={"bing", "google"},
                       ranks=[("bing", 2), ("google", 2)])
    single.score = sm._score(q, single)
    multi.score = sm._score(q, multi)
    assert multi.score > single.score


def test_ranking_lets_lower_weight_engine_top_result_through():
    """旧 RRF(1/(60+rank)) 会让 Bing 第 10 名压过 Sogou 第 1 名。"""
    q = sm._Query.build("test")
    bing_last = sm._Result("Bing 10", "https://example.com/b10",
                           engines={"bing"}, ranks=[("bing", 10)])
    sogou_first = sm._Result("Sogou 1", "https://example.com/s1",
                             engines={"sogou"}, ranks=[("sogou", 1)])
    assert sm._score(q, sogou_first) > sm._score(q, bing_last)


def test_clean_url_strips_tracking_but_keeps_functional_params():
    dirty = ("https://example.com/a?utm_source=x&id=7&msockid=abc"
             "&lang=zh#section")
    clean = sm._clean_url(dirty)
    assert "utm_source" not in clean
    assert "msockid" not in clean
    assert "id=7" in clean and "lang=zh" in clean
    assert clean.endswith("#section")


def test_kernel_dispatches_search_tool(monkeypatch, config):
    """内核 → 工具注册表 → search_tool → 结果回灌的完整链路。"""
    import json

    from agent.agent import Agent
    from agent.llm_client import StreamEvent
    from agent.kernel import ExitReason

    def fake_run(spec, *args):
        return sm._EngineRun(spec.name, [
            sm._RawResult("Kernel Result", "https://example.com/k", "snip", 1),
        ])

    monkeypatch.setattr(sm, "_run_engine", fake_run)

    class Noop:
        def is_cancelled(self):
            return False

        def on_content(self, text):
            pass

        def on_reasoning(self, text):
            pass

        def on_tool_start(self, name, args):
            pass

        def on_tool_end(self, name, result):
            pass

        def on_log(self, line):
            pass

        def on_usage(self, usage):
            pass

        def on_token_speed(self, total, speed):
            pass

        def on_finished(self):
            pass

        def confirm(self, message):
            return True

        def ask_user(self, prompt):
            return "ok"

    class MockLLM:
        def __init__(self):
            self.calls = []

        def chat_stream(self, messages, tools=None, temperature=0.7, **kw):
            self.calls.append(messages)
            if len(self.calls) == 1:
                tc = [{"id": "c1", "function": {
                    "name": "search_tool",
                    "arguments": json.dumps({"query": "x", "num_results": 3}),
                }}]
                return iter([StreamEvent("done", tool_calls=tc, usage={})])
            return iter([
                StreamEvent("content", "搜索完成"),
                StreamEvent("done", usage={}),
            ])

    llm = MockLLM()
    agent = Agent(llm=llm, config=config, callbacks=Noop())
    agent._auto_continue = False
    agent._distill_min_turns = 999

    result = agent.kernel.run("搜索一下 x")

    assert result.reason is ExitReason.DONE
    assert "搜索完成" in result.text
    second = "\n".join(str(m.get("content", "")) for m in llm.calls[1])
    assert "https://example.com/k" in second, "工具结果必须回灌给模型"


def test_engine_cooldown_helpers():
    try:
        sm._clear_cooldown("bing")
        assert sm._cooldown_remaining("bing") == 0
        sm._set_cooldown("bing", "HTTP 429")
        assert sm._cooldown_remaining("bing") > 0
        sm._clear_cooldown("bing")
        assert sm._cooldown_remaining("bing") == 0
        # 普通错误（如 500）不进入冷却。
        sm._set_cooldown("bing", "HTTP 500")
        assert sm._cooldown_remaining("bing") == 0
    finally:
        sm._clear_cooldown("bing")


def test_select_engines_by_language():
    cn, _ = sm._select_engines(sm._Query.build("人工智能最新进展"), None)
    en, _ = sm._select_engines(sm._Query.build("latest ai news"), None)
    assert {"baidu", "sogou", "360"} <= set(cn)
    assert {"brave", "yahoo", "mojeek"} <= set(en)


# ---------------------------------------------------------------------------
# 工具聚合
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 备用引擎池
# ---------------------------------------------------------------------------


def test_backup_pool_has_50_unique_specs_sorted_by_latency():
    assert len(sm._BACKUP_SPECS) == 50
    names = [s.name for s in sm._BACKUP_SPECS]
    assert len(names) == len(set(names))
    latencies = [s.latency for s in sm._BACKUP_SPECS]
    assert latencies == sorted(latencies)
    assert {"general", "news", "reference", "academic", "code"} <= {
        s.category for s in sm._BACKUP_SPECS}


def test_relevance_filter_rejects_off_topic_results():
    q = sm._Query.build("best coffee beans 2026")
    assert not sm._relevance_ok(q, "Best Buy | Official Online Store", "")
    assert not sm._relevance_ok(q, "BEST Definition & Meaning", "")
    assert sm._relevance_ok(q, "Best Coffee Beans of 2026: Expert Picks", "")
    cn = sm._Query.build("新能源汽车 销量")
    assert not sm._relevance_ok(cn, "新（汉语汉字）_百度百科", "")
    assert sm._relevance_ok(cn, "10月份新能源汽车销量占比达64.3%", "")


def test_backup_preset_runs_fast_general_sources(monkeypatch, config):
    tool = SearchTool()
    tool.config = config
    called = []

    def fake_run(spec, query, timeout, proxy):
        called.append(spec.name)
        return sm._EngineRun(spec.name, [sm._RawResult(
            "Backup Result", "https://example.com/backup", "x", 1)])

    monkeypatch.setattr(sm, "_run_engine", fake_run)
    result = tool.run("x", engines=["backup"], num_results=3)
    assert result.success
    assert set(called) == set(sm._BACKUP_PRESET_EN)


def test_backup_selection_presets_and_alias():
    q = sm._Query.build("python asyncio")
    backup, _ = sm._select_engines(q, ["backup"])
    assert backup == sm._BACKUP_PRESET_EN
    everything, _ = sm._select_engines(q, ["all"])
    assert set(sm._EN_ENGINES) <= set(everything)
    assert set(sm._AUTO_FALLBACK_EN) <= set(everything)
    aliased, unknown = sm._select_engines(q, ["wikipedia", "hn"])
    assert aliased == ["wikipedia_en", "hn_algolia"] and not unknown


def test_parse_json_spec_mediawiki_and_crossref():
    mediawiki = sm._JsonSpec(
        items="query.search", title="title", snippet="snippet",
        url_template="https://en.wikipedia.org/wiki/{title}")
    text = json.dumps({"query": {"search": [
        {"title": "Python (programming language)",
         "snippet": "General-purpose programming language"},
    ]}})
    res = sm._parse_json_spec(text, mediawiki)
    assert res[0].title.startswith("Python")
    assert res[0].url.startswith("https://en.wikipedia.org/wiki/Python")
    assert "General-purpose" in res[0].snippet

    crossref = sm._JsonSpec(
        items="message.items", title="title.0", url="DOI",
        url_prefix="https://doi.org/", snippet="abstract")
    text = json.dumps({"message": {"items": [{
        "title": ["A Paper"], "DOI": "10.1/xyz", "abstract": "An abstract",
    }]}})
    res = sm._parse_json_spec(text, crossref)
    assert res[0].url == "https://doi.org/10.1/xyz"


def test_parse_rss_and_atom():
    rss = ("<rss><channel><item><title>RSS Title</title>"
           "<link>https://example.com/rss</link>"
           "<description>RSS snippet</description></item></channel></rss>")
    res = sm._parse_rss(rss, "https://example.com")
    assert res[0].title == "RSS Title"
    assert res[0].url == "https://example.com/rss"

    atom = ("<feed xmlns='http://www.w3.org/2005/Atom'><entry>"
            "<title>Atom Title</title><link href='https://example.com/atom'/>"
            "<summary>Atom snippet</summary></entry></feed>")
    res = sm._parse_rss(atom, "https://example.com")
    assert res[0].url == "https://example.com/atom"
    assert res[0].snippet == "Atom snippet"


def test_parse_generic_html_filters_nav_and_cleans_mwmbl_title():
    html = """
    <div><a href="https://example.com/good">A genuinely useful result</a>
         <p>Result snippet</p></div>
    <a href="https://example.com/privacy">Privacy Policy</a>
    <a href="https://example.com/news/">Today's news</a>
    <a href="https://docs.python.org/3/library/asyncio.html">
      docs.python.org 3 library asyncio.html Mwmbl Real Mwmbl Title</a>
    """
    res = sm._parse_generic_html(html, "https://search.example.com/")
    titles = [r.title for r in res]
    assert "A genuinely useful result" in titles
    assert "Privacy Policy" not in titles
    assert "Today's news" not in titles
    mwmbl = [r for r in res if "Real" in r.title]
    assert mwmbl and mwmbl[0].title == "Real Mwmbl Title"


def test_parse_openaire_nested_shape():
    payload = json.dumps({"response": {"results": {"result": [{
        "header": {"dri:objIdentifier": {"$": "id-1"}},
        "metadata": {"oaf:entity": {"oaf:result": {
            "title": {"$": "OpenAIRE Title"},
            "description": {"$": "OpenAIRE abstract"},
            "pid": [{"@classid": "doi", "$": "10.1/oa"}],
        }}},
    }]}}})
    res = sm._parse_openaire(payload, "https://api.openaire.eu")
    assert res[0].title == "OpenAIRE Title"
    assert res[0].url == "https://doi.org/10.1/oa"
    assert res[0].snippet == "OpenAIRE abstract"


def test_auto_fallback_when_primary_engines_cooling(monkeypatch, config):
    tool = SearchTool()
    tool.config = config

    def fake_run(spec, query, timeout, proxy):
        return sm._EngineRun(spec.name, [sm._RawResult(
            "Backup Result", "https://example.com/backup",
            "python asyncio tutorial", 1)])

    monkeypatch.setattr(sm, "_run_engine", fake_run)
    for name in sm._EN_ENGINES:
        sm._set_cooldown(name, "HTTP 429")
    try:
        result = tool.run("python asyncio", num_results=3)
    finally:
        for name in sm._EN_ENGINES:
            sm._clear_cooldown(name)

    assert result.success
    assert "https://example.com/backup" in result.output
    assert "cooling:" in result.output
    assert "rightdao" in result.output or "wiby" in result.output


def test_search_tool_aggregates_concurrently_and_reports_failures(monkeypatch, config):
    tool = SearchTool()
    tool.config = config
    seen = []

    def fake_run(spec, query, timeout, proxy):
        seen.append(spec.name)
        if spec.name == "bing":
            return sm._EngineRun("bing", [
                sm._RawResult("One", "https://example.com/one", "s1", 1),
            ])
        if spec.name == "google":
            return sm._EngineRun("google", [
                sm._RawResult("One", "https://example.com/one?utm_source=g",
                              "s2", 1),
                sm._RawResult("One Two", "https://example.com/two", "s3", 2),
            ])
        return sm._EngineRun(spec.name, error="boom")

    monkeypatch.setattr(sm, "_run_engine", fake_run)
    result = tool.run("one", num_results=5,
                      engines=["bing", "google", "yahoo"])

    assert result.success
    assert "example.com/one" in result.output
    assert "example.com/two" in result.output
    assert "failed: yahoo(boom)" in result.output
    assert "untrusted web content" in result.output.lower()
    assert sorted(seen) == ["bing", "google", "yahoo"]


def test_search_tool_skips_cooling_engines(monkeypatch, config):
    tool = SearchTool()
    tool.config = config
    called = []

    def fake_run(spec, query, timeout, proxy):
        called.append(spec.name)
        return sm._EngineRun(spec.name, [
            sm._RawResult("Live", "https://example.com/live", "", 1),
        ])

    monkeypatch.setattr(sm, "_run_engine", fake_run)
    sm._set_cooldown("bing", "HTTP 429")
    try:
        result = tool.run("x", engines=["bing", "google"])
    finally:
        sm._clear_cooldown("bing")

    assert result.success
    assert called == ["google"]
    assert "cooling: bing(" in result.output

    # 请求的引擎全部在冷却中 → 明确报错，不空转。
    sm._set_cooldown("bing", "HTTP 429")
    try:
        result = tool.run("x", engines=["bing"])
    finally:
        sm._clear_cooldown("bing")
    assert not result.success
    assert "冷却" in result.error


def test_search_tool_reports_unknown_engine_and_keeps_valid(monkeypatch, config):
    tool = SearchTool()
    tool.config = config
    monkeypatch.setattr(sm, "_run_engine", lambda spec, *a: sm._EngineRun(
        spec.name, [sm._RawResult("R", "https://example.com/r", "", 1)]))
    result = tool.run("x", engines=["bing", "nope"])
    assert result.success
    assert "ignored unknown engines: nope" in result.output


def test_search_tool_all_engines_fail(monkeypatch, config):
    tool = SearchTool()
    tool.config = config
    monkeypatch.setattr(sm, "_run_engine", lambda *a, **k: sm._EngineRun(
        a[0].name, error="blocked"))
    result = tool.run("anything", engines=["bing", "google"])
    assert not result.success
    assert "无结果" in result.error


def test_search_tool_rejects_empty_query(config):
    tool = SearchTool()
    tool.config = config
    result = tool.run("   ")
    assert not result.success
    assert "query" in result.error


def test_search_tool_is_registered(config):
    from agent.agent import AgentCallbacks
    from agent.tools import ToolRegistry

    reg = ToolRegistry(config=config, callbacks=AgentCallbacks())
    assert "search_tool" in reg.names()
    schema = reg.get("search_tool").schema()
    assert schema["function"]["name"] == "search_tool"
    assert "query" in schema["function"]["parameters"]["required"]


def test_system_prompt_documents_proactive_search_and_real_urls():
    from agent.agent import SYSTEM_PROMPT

    assert "web_search" in SYSTEM_PROMPT
    assert "知识库没有相关内容" in SYSTEM_PROMPT
    assert "web_scan(url=...)" in SYSTEM_PROMPT
    assert "真实结果" in SYSTEM_PROMPT
    assert "禁止编造、拼接" in SYSTEM_PROMPT


def test_web_search_alias_routes_to_search_tool(monkeypatch, config):
    from agent.agent import AgentCallbacks
    from agent.tools import ToolRegistry

    def fake_run(spec, query, timeout, proxy):
        return sm._EngineRun(spec.name, [
            sm._RawResult("Alias Result", "https://example.com/alias", "x", 1),
        ])

    monkeypatch.setattr(sm, "_run_engine", fake_run)
    reg = ToolRegistry(config=config, callbacks=AgentCallbacks())
    assert "web_search" not in reg.names(), "兼容别名不应出现在模型 schema"
    assert "web_search" not in {
        s["function"]["name"] for s in reg.schemas()}
    assert reg.get("web_search") is not None
    result = reg.execute("web_search", {"query": "x", "num_results": 3})
    assert result.success
    assert "https://example.com/alias" in result.output


def test_registry_executes_search_tool(monkeypatch, config):
    from agent.agent import AgentCallbacks
    from agent.tools import ToolRegistry

    def fake_run(spec, query, timeout, proxy):
        return sm._EngineRun(spec.name, [
            sm._RawResult("Result", "https://example.com/r", "snippet", 1),
        ])

    monkeypatch.setattr(sm, "_run_engine", fake_run)
    reg = ToolRegistry(config=config, callbacks=AgentCallbacks())
    result = reg.execute("search_tool", {"query": "x", "num_results": 3})
    assert result.success
    assert "https://example.com/r" in result.output


def test_browser_ua_pool_is_desktop_only():
    """移动 UA 会让 360/搜狗/百度返回移动版 DOM，桌面选择器全部落空。"""
    for ua in sm._BROWSER_UAS:
        low = ua.lower()
        assert "mobile" not in low and "iphone" not in low and "android" not in low


def test_request_builders_apply_time_range():
    q = sm._Query.build("ai", "week")
    assert sm._build_ddg(q, "UA").data["df"] == "w"
    assert "btf=w" in sm._build_yahoo(q, "UA").url
    assert "tf=pw" in sm._build_brave(q, "UA").url
    assert "gpc=" in sm._build_baidu(q, "UA").url
    assert "adv_t=w" in sm._build_360(q, "UA").url
    assert "s_from=inttime_week" in sm._build_sogou(q, "UA").url


def test_request_builders_shape():
    cn = sm._Query.build("你好")
    en = sm._Query.build("hello")

    ddg = sm._build_ddg(cn, "UA")
    assert ddg.method == "POST" and ddg.data["kl"] == "cn-zh"

    baidu = sm._build_baidu(cn, "UA")
    assert "tn=json" in baidu.url

    bing = sm._build_bing(en, "UA")
    assert "q=hello" in bing.url and "setlang=en" in bing.url

    google = sm._build_google(en, "UA")
    assert "hl=en" in google.url
