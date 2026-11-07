"""Tests for the built-in tool set."""
from __future__ import annotations

import os

import httpx
import pytest

from agent.agent import AgentCallbacks
from agent.config import AgentConfig
from agent.tools import ToolRegistry


def _reg(config, callbacks=None):
    return ToolRegistry(config=config, callbacks=callbacks or AgentCallbacks())


# ---------------------------------------------------------------------------
# file_read
# ---------------------------------------------------------------------------

def test_file_read_returns_contents(config):
    p = os.path.join(config.workspace, "a.txt")
    with open(p, "w") as f:
        f.write("hello world")
    reg = _reg(config)
    res = reg.execute("file_read", {"path": "a.txt"})
    assert res.success
    assert res.output == "hello world"


def test_file_read_missing_file_returns_error(config):
    reg = _reg(config)
    res = reg.execute("file_read", {"path": "nope.txt"})
    assert not res.success
    assert "not exist" in res.error.lower() or "no such" in res.error.lower()


def test_file_read_rejects_escape(config):
    reg = _reg(config)
    res = reg.execute("file_read", {"path": "../../etc/passwd"})
    assert not res.success
    assert "workspace" in res.error.lower() or "outside" in res.error.lower()


def test_file_write_rejects_symlink_escape(config, tmp_path):
    """A symlink inside the workspace pointing outside must not allow writes
    to escape the workspace. _resolve must use realpath to resolve symlinks."""
    import os
    outside = os.path.join(os.path.dirname(config.workspace), "outside_target.txt")
    if os.path.exists(outside):
        os.remove(outside)
    link_path = os.path.join(config.workspace, "escape_link")
    if not os.path.exists(link_path):
        os.symlink(outside, link_path)
    reg = _reg(config)
    res = reg.execute("file_write", {"path": "escape_link", "content": "escaped"})
    assert not res.success, "write via symlink to outside must be rejected"
    assert not os.path.exists(outside), "target outside workspace must not be created"


def test_file_write_atomic_no_tmp_leftover(config):
    """file_write must use atomic write (temp + os.replace); after a successful
    write no .tmp file should remain in the workspace."""
    reg = _reg(config)
    reg.execute("file_write", {"path": "atom.txt", "content": "data"})
    assert not os.path.exists(os.path.join(config.workspace, "atom.txt.tmp"))


def test_file_modify_atomic_no_tmp_leftover(config):
    """file_modify must use atomic write; no .tmp leftover after success."""
    with open(os.path.join(config.workspace, "mod.txt"), "w") as f:
        f.write("hello")
    reg = _reg(config)
    reg.execute("file_modify", {"path": "mod.txt", "old": "hello", "new": "world"})
    with open(os.path.join(config.workspace, "mod.txt")) as f:
        assert f.read() == "world"
    assert not os.path.exists(os.path.join(config.workspace, "mod.txt.tmp"))


# ---------------------------------------------------------------------------
# file_write
# ---------------------------------------------------------------------------

def test_file_write_creates_file(config):
    reg = _reg(config)
    res = reg.execute("file_write", {"path": "out.txt", "content": "abc"})
    assert res.success
    with open(os.path.join(config.workspace, "out.txt")) as f:
        assert f.read() == "abc"


def test_file_write_overwrites_existing(config):
    with open(os.path.join(config.workspace, "out.txt"), "w") as f:
        f.write("old")
    reg = _reg(config)
    reg.execute("file_write", {"path": "out.txt", "content": "new"})
    with open(os.path.join(config.workspace, "out.txt")) as f:
        assert f.read() == "new"


def test_file_write_creates_subdirs(config):
    reg = _reg(config)
    res = reg.execute("file_write", {"path": "sub/dir/x.txt", "content": "y"})
    assert res.success
    with open(os.path.join(config.workspace, "sub", "dir", "x.txt")) as f:
        assert f.read() == "y"


# ---------------------------------------------------------------------------
# file_modify
# ---------------------------------------------------------------------------

def test_file_modify_replaces_text(config):
    with open(os.path.join(config.workspace, "m.txt"), "w") as f:
        f.write("foo bar foo")
    reg = _reg(config)
    res = reg.execute("file_modify", {"path": "m.txt", "old": "bar", "new": "baz"})
    assert res.success
    with open(os.path.join(config.workspace, "m.txt")) as f:
        assert f.read() == "foo baz foo"


def test_file_modify_missing_old_string(config):
    with open(os.path.join(config.workspace, "m.txt"), "w") as f:
        f.write("foo")
    reg = _reg(config)
    res = reg.execute("file_modify", {"path": "m.txt", "old": "zzz", "new": "yy"})
    assert not res.success


# ---------------------------------------------------------------------------
# code_run
# ---------------------------------------------------------------------------

def test_code_run_executes_and_captures_stdout(config):
    reg = _reg(config)
    res = reg.execute("code_run", {"code": "print(1+1)\nimport math; print(math.sqrt(16))"})
    assert res.success
    assert "2" in res.output
    assert "4.0" in res.output


def test_code_run_can_write_files_in_workspace(config):
    reg = _reg(config)
    res = reg.execute("code_run", {"code": "open('crr.txt','w').write('done')"})
    assert res.success
    with open(os.path.join(config.workspace, "crr.txt")) as f:
        assert f.read() == "done"


def test_code_run_returns_error_on_exception(config):
    reg = _reg(config)
    res = reg.execute("code_run", {"code": "raise ValueError('boom')"})
    assert not res.success
    assert "boom" in res.error


def test_code_run_cannot_escape_workspace(config):
    reg = _reg(config)
    outside = os.path.join(os.path.dirname(config.workspace), "escaped.txt")
    if os.path.exists(outside):
        os.remove(outside)
    code = f"open({outside!r},'w').write('x')"
    res = reg.execute("code_run", {"code": code})
    assert not res.success
    assert not os.path.exists(outside)


def test_code_run_timeout_kills_infinite_loop(config):
    """An infinite loop in code_run must be killed by the timeout, not hang
    the agent forever."""
    reg = _reg(config)
    res = reg.execute("code_run", {"code": "while True:\n    pass", "timeout": 2})
    assert not res.success
    assert "timeout" in res.error.lower() or "timed out" in res.error.lower()


def test_code_run_can_use_subprocess(config):
    """code_run must allow subprocess so the model can pip install or run
    shell commands from within Python."""
    reg = _reg(config)
    res = reg.execute("code_run", {"code": "import subprocess; print(subprocess.check_output(['echo','hello']).decode().strip())"})
    assert res.success
    assert "hello" in res.output


def test_code_run_pip_install_and_import(config):
    """The model should be able to pip install a small package and import it."""
    reg = _reg(config)
    code = (
        "import subprocess, sys\n"
        "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'six'])\n"
        "import six\n"
        "print('six version:', six.__version__)"
    )
    res = reg.execute("code_run", {"code": code})
    assert res.success
    assert "six version:" in res.output


# ---------------------------------------------------------------------------
# shell_run
# ---------------------------------------------------------------------------

def test_shell_run_executes_command(config):
    reg = _reg(config)
    res = reg.execute("shell_run", {"command": "echo hello world"})
    assert res.success
    assert "hello world" in res.output


def test_shell_run_captures_stderr(config):
    reg = _reg(config)
    res = reg.execute("shell_run", {"command": "echo out; echo err >&2"})
    assert res.success
    assert "out" in res.output
    assert "err" in res.output


def test_shell_run_nonzero_exit_reported(config):
    reg = _reg(config)
    res = reg.execute("shell_run", {"command": "exit 3"})
    assert not res.success
    assert "3" in res.error or "3" in res.output


def test_shell_run_timeout(config):
    reg = _reg(config)
    res = reg.execute("shell_run", {"command": "sleep 30", "timeout": 2})
    assert not res.success
    assert "timeout" in res.error.lower() or "timed out" in res.error.lower()


def test_shell_run_uses_workspace_cwd(config):
    reg = _reg(config)
    res = reg.execute("shell_run", {"command": "pwd"})
    assert res.success
    assert os.path.abspath(config.workspace) in res.output or config.workspace in res.output


def test_shell_run_risk_graded_confirmation(config):
    """shell_run 按风险分级确认（tools/risk.py）：良性命令直接执行，
    sudo/系统控制/写工作区外路径才弹确认。"""
    reg = _reg(config)
    assert reg.get("shell_run").destructive is True
    assert reg.get("shell_run").should_confirm({"command": "ls"}) is False
    assert reg.get("shell_run").should_confirm({"command": "cat /etc/hosts"}) is False
    assert reg.get("shell_run").should_confirm({"command": "sudo rm -rf /tmp/x"}) is True
    assert reg.get("shell_run").should_confirm(
        {"command": "echo x > /etc/hosts"}) is True


# ---------------------------------------------------------------------------
# web_scan
# ---------------------------------------------------------------------------

def test_web_scan_extracts_text(monkeypatch, config):
    from agent.tools import web as web_mod

    HTML = "<html><body><h1>Hi</h1><p>Hello world</p><script>bad()</script></body></html>"

    class FakeResp:
        status_code = 200
        text = HTML
        headers = {"Content-Type": "text/html"}

        def raise_for_status(self):
            pass

        def close(self):
            pass

    class FakeStreamResp(FakeResp):
        # 流式路径需要 context manager + iter_bytes。
        encoding = None

        def __init__(self):
            self._chunks = [HTML.encode("utf-8")]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def iter_bytes(self, chunk_size=65536):
            return iter(self._chunks)

    # web_scan 优先走流式（client.stream），拿不到才退回 get()。
    # 两条路径都要打桩，否则有真实网络的机器会发真请求。
    def _fake_stream(self, method, url, **kw):
        return FakeStreamResp()

    monkeypatch.setattr(httpx.Client, "stream", _fake_stream)
    monkeypatch.setattr(httpx.Client, "get", lambda self, url, **kw: FakeResp())
    reg = _reg(config)
    res = reg.execute("web_scan", {"url": "https://example.com"})
    assert res.success
    assert "Hi" in res.output
    assert "Hello world" in res.output
    assert "bad()" not in res.output  # script stripped


def test_web_scan_falls_back_to_get_when_stream_unsupported(monkeypatch, config):
    """仅实现 get() 的客户端（极简适配器/旧替身）也要能工作。"""

    class FakeResp:
        status_code = 200
        text = "<html><body><p>via-get</p></body></html>"
        headers = {"Content-Type": "text/html"}

        def raise_for_status(self):
            pass

        def close(self):
            pass

    class GetOnlyClient(httpx.Client):
        # 模拟「没有 stream」的客户端：hasattr 检查会走 get() 分支。
        stream = None  # noqa: 安排成属性即可被 hasattr 识别为缺失

    monkeypatch.setattr(httpx.Client, "get", lambda self, url, **kw: FakeResp())
    monkeypatch.delattr(httpx.Client, "stream", raising=False)
    reg = _reg(config)
    res = reg.execute("web_scan", {"url": "https://example.com"})
    assert res.success
    assert "via-get" in res.output


# ---------------------------------------------------------------------------
# ask_user
# ---------------------------------------------------------------------------

def test_ask_user_invokes_callback(config, recording_callbacks):
    recording_callbacks._ask_return = "user-said"
    reg = _reg(config, recording_callbacks)
    res = reg.execute("ask_user", {"prompt": "Which one?"})
    assert res.success
    assert res.output == "user-said"
    assert recording_callbacks.asks == ["Which one?"]


# ---------------------------------------------------------------------------
# work_memory
# ---------------------------------------------------------------------------

def test_work_memory_set_get_list_clear(config):
    reg = _reg(config)
    assert reg.execute("work_memory", {"op": "list"}).output.strip() == "{}"

    assert reg.execute("work_memory", {"op": "set", "key": "goal", "value": "fix bug"}).success
    assert reg.execute("work_memory", {"op": "get", "key": "goal"}).output == "fix bug"

    listing = reg.execute("work_memory", {"op": "list"}).output
    assert "goal" in listing and "fix bug" in listing

    reg.execute("work_memory", {"op": "clear"})
    assert reg.execute("work_memory", {"op": "list"}).output.strip() == "{}"


# ---------------------------------------------------------------------------
# memory_graph / memory_search
# ---------------------------------------------------------------------------

def test_memory_graph_create_and_search(config):
    reg = _reg(config)
    # add_observations creates the entity + observations
    res = reg.execute("memory_graph", {
        "op": "add_observations",
        "name": "TestUser",
        "observations": ["prefers dark mode", "uses python"],
    })
    assert res.success
    # search should find the observation
    res2 = reg.execute("memory_search", {"query": "python", "top_k": 5})
    assert res2.success
    assert "python" in res2.output.lower()
    # cleanup
    reg.execute("memory_graph", {"op": "clear"})


# ---------------------------------------------------------------------------
# destructive flag & confirmation
# ---------------------------------------------------------------------------

def test_destructive_flags(config):
    """Only tools that can affect the system/external world require confirmation.

    Workspace file ops are locked to the workspace, so they are non-destructive.
    code_run and shell_run can execute arbitrary system-affecting code, so they
    are destructive.
    """
    reg = _reg(config)
    assert reg.get("file_read").destructive is False
    assert reg.get("file_write").destructive is False
    assert reg.get("file_modify").destructive is False
    assert reg.get("code_run").destructive is True
    assert reg.get("shell_run").destructive is True
    assert reg.get("web_scan").destructive is False
    # webexec_js is only registered when browser_endpoint is configured
    webexec = reg.get("webexec_js")
    if webexec is not None:
        assert webexec.destructive is False
    assert reg.get("ask_user").destructive is False
    assert reg.get("work_memory").destructive is False
    assert reg.get("memory_graph").destructive is False
    assert reg.get("memory_search").destructive is False


def test_should_confirm_default_follows_destructive(config):
    reg = _reg(config)
    # code_run：良性代码直接执行（风险分级，见 tools/risk.py）
    assert reg.get("code_run").should_confirm({"code": "print(1)"}) is False
    assert reg.get("code_run").should_confirm(
        {"code": "os.system('sudo x')"}) is True
    assert reg.get("file_write").should_confirm({"path": "a", "content": "b"}) is False
    assert reg.get("file_read").should_confirm({"path": "a"}) is False


def test_unknown_tool_returns_error(config):
    reg = _reg(config)
    res = reg.execute("nope", {})
    assert not res.success
    assert "unknown" in res.error.lower()


def test_registry_exposes_function_schemas(config):
    reg = _reg(config)
    schemas = reg.schemas()
    names = {s["function"]["name"] for s in schemas}
    expected = {
        "code_run", "shell_run", "file_read", "file_write", "file_modify",
        "web_scan", "ask_user",
        "work_memory", "memory_graph", "memory_search",
    }
    assert expected <= names


# ---------------------------------------------------------------------------
# RAG tool registration
# ---------------------------------------------------------------------------

def test_rag_tools_registered_with_default_knowledge_base(config):
    """Empty knowledge_base → defaults to <workspace>/knowledge_base,
    RAG tools should still be registered."""
    config.knowledge_base = ""
    reg = _reg(config)
    names = {s["function"]["name"] for s in reg.schemas()}
    assert "rag_search" in names
    assert "rag_status" in names
    assert "rag_ingest" in names
    reg.shutdown()


def test_rag_tools_registered_with_knowledge_base(config, tmp_path):
    """knowledge_base configured → RAG tools should be registered and
    exposed to the LLM via schemas(). Also verifies that construction does
    NOT trigger ingest (which would block the UI main thread)."""
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "doc.txt").write_text("test content")
    config.knowledge_base = str(kb)

    reg = _reg(config)
    names = {s["function"]["name"] for s in reg.schemas()}
    assert "rag_search" in names, "rag_search must be registered when knowledge_base is set"
    assert "rag_status" in names
    assert "rag_ingest" in names

    # Verify engine is available but no ingest was performed (lazy init).
    engine = reg.get_rag_engine()
    assert engine is not None
    # count() triggers _ensure_initialized which loads FastEmbed — skip in
    # unit test to avoid model download. Just verify the engine exists.

    reg.shutdown()


def test_rag_outline_no_args_guides_next_step():
    """无参 rag_outline 列出文档后必须自带下一步指引。

    2026-11-03 用户反馈：问「主要思想是什么」（没点名书），模型卡在
    「不知道对哪本 outline」反复复述意图——工具结果要直接给出决策规则。
    """
    from agent.tools.rag_tool import RagOutlineTool

    class _Eng:
        def __init__(self, docs):
            self._docs = docs

        def list_documents(self):
            return list(self._docs)

    single = RagOutlineTool(_Eng(["神之所欲.epub"])).run()
    assert single.success
    assert "只有这一篇" in single.output
    assert "先总后分" in single.output

    multi = RagOutlineTool(_Eng(["a.epub", "b.pdf"])).run()
    assert multi.success
    assert "挑文档名/主题与问题最相关的一篇" in multi.output
    assert "ask_user" in multi.output


def test_rag_search_empty_query_error_discourages_retry():
    """空参 rag_search 的错误信息必须自带反循环指引。

    2026-11-03 用户实测（1.2.8 版本）：模型空参连发 14 次全失败——
    「请提供搜索查询词」不含任何跳出循环的信息，模型只会原样重试。
    错误信息要指明：停止重复、可用已有结果作答、正确用法示例。
    """
    from agent.tools.rag_tool import RagSearchTool

    res = RagSearchTool(engine=None).run(query="")
    assert not res.success
    # 指出缺什么
    assert "query" in res.error
    # 反循环：停止重复检索
    assert "不要重复调用" in res.error
    # 出路：已有摘要可直接作答 / 给出正确示例
    assert "全书摘要" in res.error
    assert "如" in res.error and "主要思想" in res.error


def test_rag_outline_with_summary_discourages_extra_search():
    """outline 附全书摘要时必须明示：问整体内容直接作答，无需再 search。"""
    from agent.tools.rag_tool import RagOutlineTool

    class _Eng:
        def list_documents(self):
            return ["神之所欲.epub"]

        def get_document_outline(self, name):
            return {"doc_name": name, "digest": "第一章 神……"}

        def get_book_summary(self, name):
            return "本书主张……"

    res = RagOutlineTool(_Eng()).run("神之所欲")
    assert res.success
    assert "【全书摘要】" in res.output
    assert "无需再 rag_search" in res.output
    assert "至多 3 次" in res.output


# ---------------------------------------------------------------------------
# shell_run: timeout must not leave orphaned processes behind
# ---------------------------------------------------------------------------

def test_shell_run_timeout_kills_the_process_group(config):
    """A timed-out command must not leave its children running.

    `shell=True` spawns `sh -c …`; killing only that PID used to orphan
    grandchildren (a `sleep`, a dev server, a pip subprocess) that then lived
    until logout.
    """
    import subprocess
    import time

    reg = _reg(config)
    marker = os.path.join(config.workspace, "orphan.marker")
    # `sh -c 'sleep 30 & ...'` — the sleep is a grandchild of the shell.
    res = reg.execute("shell_run", {
        "command": f"sleep 30; echo done > {marker}",
        "timeout": 1,
    })
    assert not res.success
    assert "timed out" in res.error.lower()

    time.sleep(1.5)
    assert not os.path.exists(marker), "orphaned grandchild was still running"
    # No `sleep 30` should survive the timeout.
    out = subprocess.run(["pgrep", "-f", "sleep 30"], capture_output=True, text=True)
    assert not out.stdout.strip(), f"orphan process survived: {out.stdout!r}"
    reg.shutdown()


def test_shell_run_caps_huge_output(config):
    """A command that floods stdout must not dump it all into the reply."""
    reg = _reg(config)
    res = reg.execute("shell_run", {
        "command": "python3 -c \"print('x' * 1000)\" ; "
                   "python3 -c \"import sys; sys.stdout.write('y' * 500000)\"",
        "timeout": 60,
    })
    assert res.success
    assert len(res.output) < 300_000, f"output not capped: {len(res.output)} chars"
    reg.shutdown()


def test_web_scan_rejects_non_http_schemes(config):
    """`url` comes from the model — file:// and friends must be refused."""
    reg = _reg(config)
    res = reg.execute("web_scan", {"url": "file:///etc/passwd"})
    assert not res.success
    assert "http" in res.error.lower()
    reg.shutdown()


# ---------------------------------------------------------------------------
# file_ops robustness (round 3 review)
# ---------------------------------------------------------------------------

def test_file_modify_binary_file_returns_clear_error(config):
    """二进制文件替换文本应报「不是 UTF-8」，而不是裸 UnicodeDecodeError。"""
    p = os.path.join(config.workspace, "img.bin")
    with open(p, "wb") as f:
        f.write(bytes(range(256)) * 64)  # 无效 UTF-8
    reg = _reg(config)
    res = reg.execute("file_modify", {"path": "img.bin", "old": "x", "new": "y"})
    assert not res.success
    assert "utf-8" in res.error.lower()
    reg.shutdown()


def test_file_modify_huge_file_is_rejected(config):
    """超大文件全量读入内存做替换会被拒绝（给出指引而不是 OOM）。"""
    from agent.tools import file_ops as fo

    p = os.path.join(config.workspace, "big.log")
    # 用 monkeypatch 把阈值压小，避免真的写 10MB。
    import unittest.mock as mock
    reg = _reg(config)
    with open(p, "w") as f:
        f.write("hello world\n" * 100)
    with mock.patch.object(fo, "_MAX_MODIFY_BYTES", 100):
        res = reg.execute("file_modify", {"path": "big.log", "old": "hello", "new": "bye"})
    assert not res.success
    assert "too large" in res.error.lower()
    assert "file_write" in res.error or "sed" in res.error
    reg.shutdown()


def test_file_write_preserves_mode_and_default_0644(config):
    """mkstemp 是 0600——file_write 不应悄悄把目标文件变成仅属主可读。"""
    p1 = os.path.join(config.workspace, "mode_new.txt")
    p2 = os.path.join(config.workspace, "mode_exist.txt")
    with open(p2, "w") as f:
        f.write("old")
    os.chmod(p2, 0o640)

    reg = _reg(config)
    assert reg.execute("file_write", {"path": "mode_new.txt", "content": "x"}).success
    assert reg.execute("file_write", {"path": "mode_exist.txt", "content": "y"}).success
    reg.shutdown()

    # 新文件按 umask 得到常规权限（非 0600）；已有文件继承原权限。
    new_mode = os.stat(p1).st_mode & 0o777
    assert new_mode != 0o600, f"new file got mkstemp's 0600: {oct(new_mode)}"
    assert os.stat(p2).st_mode & 0o777 == 0o640


def test_memory_search_clamps_top_k(config):
    """top_k=100000 不能把全部观察灌进上下文。"""
    reg = _reg(config)
    reg.execute("memory_graph", {"op": "create_entity",
                                 "name": "E", "observations": ["fact1", "fact2"]})
    res = reg.execute("memory_search", {"query": "fact", "top_k": 100000})
    assert res.success
    # 钳制到 20：JSON 里最多 20 条结果。
    import json as _json
    results = _json.loads(res.output)
    assert len(results) <= 20
    reg.shutdown()


def test_memory_search_rejects_blank_query(config):
    reg = _reg(config)
    res = reg.execute("memory_search", {"query": "   "})
    assert not res.success
    reg.shutdown()


def test_mcp_client_caps_tool_output():
    """MCP 工具输出无上限会撑爆模型上下文，必须钳制并标注。"""
    from agent.tools.mcp_client import _cap_output, _MAX_TOOL_OUTPUT_CHARS

    assert _cap_output("short") == "short"
    big = "x" * (_MAX_TOOL_OUTPUT_CHARS + 1234)
    capped = _cap_output(big)
    assert len(capped) < _MAX_TOOL_OUTPUT_CHARS + 200
    assert "截断" in capped
    assert "1234" in capped  # 标注里写明原始长度


def test_mcp_call_after_stop_raises_clean_error():
    """stop() 之后协议调用必须快速抛 MCPError，而不是 AttributeError/挂起。"""
    from agent.tools.mcp_client import MCPClient, MCPError

    c = MCPClient(name="t", command="true")
    c.stop()                              # 永久停牌
    with pytest.raises(MCPError):
        c.call_tool("anything", {})


def test_mcp_call_tool_reports_iserror_as_failure():
    """MCP 规范里工具失败是 result.isError=true，必须报错而不是当成功结果。"""
    from fastmcp import FastMCP
    from agent.tools.mcp_client import MCPClient, MCPError

    mcp = FastMCP("mcp-iserror")

    @mcp.tool()
    def boom() -> str:
        raise ValueError("permission denied")

    c = MCPClient(name="t", server=mcp)
    c.start()
    try:
        with pytest.raises(MCPError, match="permission denied"):
            c.call_tool("boom", {})
    finally:
        c.stop()


def test_mcp_stop_does_not_block_on_hung_server(tmp_path):
    """server 挂起时 stop() 不能卡在管道 close() 上（读线程阻塞在 readline）。"""
    import sys
    import time
    from agent.tools.mcp_client import MCPClient

    script = tmp_path / "hang.py"
    script.write_text("import time\ntime.sleep(30)\n")
    c = MCPClient(name="hang", command=sys.executable, args=[str(script)], timeout=1)
    t0 = time.time()
    try:
        c.start()
    except Exception:
        pass  # 握手必然超时
    c.stop()
    elapsed = time.time() - t0
    assert elapsed < 12, f"stop() 被挂起的 server 卡住 {elapsed:.1f}s（旧实现会等到子进程退出）"


def test_mcp_tool_rejects_missing_required_args():
    """MCP 工具必须本地校验 required，否则缺参会原样发给远端。"""
    from agent.tools.mcp_tool import MCPTool

    class FakeClient:
        def __init__(self):
            self.calls = []

        def call_tool(self, name, arguments):
            self.calls.append((name, arguments))
            return "ok"

    client = FakeClient()
    t = MCPTool(client, {
        "name": "echo",
        "description": "echo",
        "inputSchema": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    })
    res = t.run()
    assert not res.success
    assert "message" in res.error and "Invalid arguments" in res.error
    assert client.calls == [], "缺参时不该真的打到远端"

    ok = t.run(message="hi")
    assert ok.success and ok.output == "ok"
    assert client.calls == [("echo", {"message": "hi"})]


# ---------------------------------------------------------------------------
# execute() argument validation vs internal errors (round 4)
# ---------------------------------------------------------------------------

def test_execute_reports_bad_args_as_invalid_arguments(config):
    """模型漏传/多传参数 → "Invalid arguments"（模型可据此修正调用）。"""
    reg = _reg(config)

    # 缺 required 参数。
    res = reg.execute("file_read", {})
    assert not res.success
    assert "Invalid arguments" in res.error
    assert "path" in res.error

    # 传了签名里不存在的参数。
    res = reg.execute("file_read", {"path": "a.txt", "bogus": 1})
    assert not res.success
    assert "Invalid arguments" in res.error

    # args 不是 dict。
    res = reg.execute("file_read", ["not", "a", "dict"])
    assert not res.success
    assert "Invalid arguments" in res.error
    reg.shutdown()


def test_execute_distinguishes_internal_typeerror(config):
    """工具内部抛 TypeError 不能再被误报成 "Invalid arguments"。

    旧实现一律捕 TypeError 报参数错误，模型会反复重试同样的调用而不是
    意识到工具本身坏了。
    """
    from agent.tools.base import Tool

    class BuggyTool(Tool):
        name = "buggy_tool"
        description = "raises TypeError internally"
        parameters = {"type": "object", "properties": {"x": {"type": "string"}},
                      "required": ["x"]}

        def run(self, x: str) -> ToolResult:
            # 与参数无关的内部 TypeError（len() 不接受 int）。
            len(42)
            return ToolResult(True, output="never")

    reg = _reg(config)
    reg._tools["buggy_tool"] = BuggyTool()
    res = reg.execute("buggy_tool", {"x": "hello"})
    assert not res.success
    assert "Invalid arguments" not in res.error
    assert "internally" in res.error
    assert "object of type 'int' has no len()" in res.error
    reg.shutdown()


# ---------------------------------------------------------------------------
# code_run stdout restore ownership (round 4)
# ---------------------------------------------------------------------------

def test_code_run_leaked_thread_does_not_hijack_stdout(config):
    """超时泄漏的线程退出后不得污染进程级 sys.stdout。

    旧实现 worker 在 finally 里恢复 stdout——泄漏线程在未来的任意时刻退出时，
    若期间又有一次 code_run 在执行（已换上自己的 StringIO），那次执行的输出
    会被这次恢复吞掉。
    """
    import sys as _sys
    import time as _time

    reg = _reg(config)
    # 在任何 code_run 之前记录真实 stdout：超时路径会把全局流保持在已关闭
    # 缓冲上 ~50ms（等泄漏线程撞 ValueError 退出）才恢复，res1 返回后立即
    # 读 sys.stdout 拿到的是已关闭缓冲，不是可比对外部流。
    real_stdout = _sys.stdout
    # 第一次调用：启动一个后台线程持续向 stdout 写，然后主脚本超时。
    # 该泄漏线程的 finally（旧实现）会在它退出时恢复 stdout。
    res1 = reg.execute("code_run", {
        "code": (
            "import threading, time, sys\n"
            "def spin():\n"
            "    while not _cancelled():\n"
            "        try:\n"
            "            print('x')\n"
            "        except Exception:\n"
            "            pass\n"
            "        time.sleep(0.01)\n"
            "threading.Thread(target=spin, daemon=True).start()\n"
            "time.sleep(30)\n"
        ),
        "timeout": 1,
    })
    assert not res1.success and "timed out" in res1.error.lower()

    try:
        # 等延迟恢复完成（50ms 窗口 + 余量），让后续断言比较的是外部流。
        _time.sleep(0.2)
        # 第二次调用：验证其输出完整进入自己的 StringIO（未被泄漏线程吞掉）。
        res2 = reg.execute("code_run", {
            "code": "print('second-run-ok')",
            "timeout": 10,
        })
        assert res2.success
        assert "second-run-ok" in res2.output
        # 全局 stdout 始终是真 stdout（未被任何 StringIO 劫持）。
        assert _sys.stdout is real_stdout
    finally:
        # 唤醒泄漏线程，让它尽快退出，避免污染后续测试。
        reg.execute("code_run", {"code": "_cancelled()", "timeout": 5})
        _time.sleep(0.05)
    reg.shutdown()


def test_code_run_timeout_closes_buffers(config):
    """超时后必须关闭泄漏线程的 StringIO。

    print() 每次都动态查找 sys.stdout，而主线程超时后已把它恢复成真实
    stdout —— 不 close 的话泄漏线程会一直往**真实** stdout 写（不受
    _BoundedIO 截断、也不进工具结果）。close() 之后它的下一次写入抛
    ValueError，线程自行退出。
    """
    reg = ToolRegistry(config=config, callbacks=AgentCallbacks())
    res = reg.execute("code_run", {
        "code": "while True:\n    print('noise')\n",
        "timeout": 1,
    })
    assert not res.success and "timed out" in res.error.lower()
    # 缓冲区已关闭 → 泄漏线程再 print 会抛异常并退出（不再污染 stdout）
    from agent.tools import code_run as code_run_mod
    with code_run_mod._leaked_threads_lock:
        leaked = [t for t in code_run_mod._leaked_threads if t.is_alive()]
    reg.shutdown()


def test_skill_save_marks_index_stale(config):
    """skill_save 必须让检索器重新同步索引。

    旧实现调用 scanner.scan(force=True)，会把 _last_scan/_last_mtime 刷新掉，
    于是 ensure_indexed() 的 _should_scan() 恒为 False → 刚保存的技能永远
    不会被嵌入，skill_semantic_search 检索不到它。
    """
    reg = ToolRegistry(config=config, callbacks=AgentCallbacks())
    retriever = reg.get_skill_retriever()
    assert retriever is not None
    # 先把索引建起来（"用过一次之后就坏"的前提）
    try:
        retriever.ensure_indexed()
    except Exception:
        pytest.skip("技能向量索引不可用（缺少 lancedb/fastembed）")
    retriever._dirty = False

    res = reg.execute("skill_save", {
        "name": "indexed-skill", "description": "测试索引同步",
        "steps": "1. do\n2. done",
    })
    assert res.success, res.error
    assert retriever._dirty is True, "skill_save 没有把索引标记为脏"
    reg.shutdown()


def test_skill_search_normalizes_bad_top_k(config):
    """模型传 null / 非数字的 top_k 不能抛裸异常。"""
    reg = ToolRegistry(config=config, callbacks=AgentCallbacks())
    for bad in (None, "abc", -5, 9999, 3.7):
        res = reg.execute("skill_search", {"op": "list", "top_k": bad})
        assert res.success, f"top_k={bad!r} 应被归一化，而不是失败"
    reg.shutdown()
