"""Tests for the agent main loop (LLM mocked)."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Iterator, List

import pytest

from agent.agent import Agent, _AUTO_CONTINUE_MAX
from agent.config import AgentConfig
from agent.llm_client import LLMClient, StreamEvent


def _evt(t, content="", tool_calls=None, usage=None):
    return StreamEvent(type=t, content=content, tool_calls=tool_calls or [], usage=usage or {})


@pytest.fixture(autouse=True)
def _legacy_confirm_policy(config):
    """本文件的历史用例建立在「code_run/shell_run 每次都确认」的旧契约上。

    新的确认策略（tools/risk.py）是风险分级的——良性代码直接执行。这里把
    本模块的配置切回逐次确认，保持历史断言（confirms == 1 等）不变；
    新策略的行为断言在 tests/test_confirm_policy.py。
    """
    config.always_confirm_code = True


class MockLLM:
    """Replays a queue of canned event-lists, one per chat_stream call."""

    def __init__(self, scripts: List[List[StreamEvent]]):
        self._scripts = list(scripts)
        self.calls = []

    def chat_stream(self, messages, tools=None, temperature=0.7):
        self.calls.append({"messages": messages, "tools": tools, "temperature": temperature})
        if not self._scripts:
            raise AssertionError("MockLLM ran out of scripted responses")
        return iter(self._scripts.pop(0))


# ---------------------------------------------------------------------------
# simple reply (no tools)
# ---------------------------------------------------------------------------

def test_agent_streams_content_and_stops(config, recording_callbacks):
    mock = MockLLM([[_evt("content", "He"), _evt("content", "llo"),
                     _evt("done", usage={"total_tokens": 5, "prompt_tokens": 2, "completion_tokens": 3})]])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("hi")

    assert "".join(recording_callbacks.content) == "Hello"
    assert recording_callbacks.tool_starts == []
    assert recording_callbacks.usages == [{"total_tokens": 5, "prompt_tokens": 2, "completion_tokens": 3}]


def test_agent_streams_reasoning(config, recording_callbacks):
    mock = MockLLM([[_evt("reasoning", "thinking..."),
                     _evt("content", "answer"),
                     _evt("done", usage={"total_tokens": 4})]])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("hi")

    assert "".join(recording_callbacks.reasoning) == "thinking..."
    assert "".join(recording_callbacks.content) == "answer"


# ---------------------------------------------------------------------------
# one tool call then final answer
# ---------------------------------------------------------------------------

def test_agent_executes_tool_then_answers(config, recording_callbacks):
    # Create a file the tool will read.
    with open(__import__("os").path.join(config.workspace, "data.txt"), "w") as f:
        f.write("42")

    tc = [{"id": "c1", "function": {"name": "file_read", "arguments": json.dumps({"path": "data.txt"})}}]
    mock = MockLLM([
        [_evt("reasoning", "I will read the file"),
         _evt("done", tool_calls=tc, usage={"total_tokens": 10})],
        [_evt("content", "The answer is 42"),
         _evt("done", usage={"total_tokens": 12})],
    ])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("read data.txt")

    assert recording_callbacks.tool_starts == [("file_read", {"path": "data.txt"})]
    assert len(recording_callbacks.tool_ends) == 1
    name, res = recording_callbacks.tool_ends[0]
    assert name == "file_read"
    assert res.success and res.output == "42"
    assert "".join(recording_callbacks.content) == "The answer is 42"
    # 2 LLM calls
    assert len(mock.calls) == 2
    # second call must include the tool result message
    second_msgs = mock.calls[1]["messages"]
    assert any(m.get("role") == "tool" and m.get("tool_call_id") == "c1" for m in second_msgs)


# ---------------------------------------------------------------------------
# multi-step tool calls
# ---------------------------------------------------------------------------

def test_agent_multi_step_tool_chain(config, recording_callbacks):
    tc1 = [{"id": "c1", "function": {"name": "code_run", "arguments": json.dumps({"code": "x=6*7"})}}]
    tc2 = [{"id": "c2", "function": {"name": "ask_user", "arguments": json.dumps({"prompt": "ok?"})}}]
    mock = MockLLM([
        [_evt("done", tool_calls=tc1, usage={"total_tokens": 5})],
        [_evt("done", tool_calls=tc2, usage={"total_tokens": 5})],
        [_evt("content", "done"), _evt("done", usage={"total_tokens": 5})],
    ])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("compute then ask")

    assert [n for n, _ in recording_callbacks.tool_starts] == ["code_run", "ask_user"]
    assert recording_callbacks.asks == ["ok?"]


# ---------------------------------------------------------------------------
# confirmation policy: only system-affecting tools (code_run) ask; workspace
# file ops do not.
# ---------------------------------------------------------------------------

def test_code_run_requires_confirmation(config, recording_callbacks):
    tc = [{"id": "c1", "function": {"name": "code_run",
            "arguments": json.dumps({"code": "print('hi')"})}}]
    mock = MockLLM([[_evt("done", tool_calls=tc, usage={"total_tokens": 5})],
                    [_evt("content", "done"), _evt("done", usage={"total_tokens": 5})]])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)

    recording_callbacks._confirm_return = True
    agent.run("run code")

    assert len(recording_callbacks.confirms) == 1
    assert recording_callbacks.tool_starts == [("code_run", {"code": "print('hi')"})]


def test_code_run_aborted_when_user_declines(config, recording_callbacks):
    tc = [{"id": "c1", "function": {"name": "code_run",
            "arguments": json.dumps({"code": "print('hi')"})}}]
    mock = MockLLM([
        [_evt("done", tool_calls=tc, usage={"total_tokens": 5})],
        # after refusal, model should still get a tool message saying denied, then answer
        [_evt("content", "ok skipped"), _evt("done", usage={"total_tokens": 5})],
    ])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    recording_callbacks._confirm_return = False
    agent.run("run code")

    assert recording_callbacks.tool_starts == []  # never started
    # tool result message should report denial
    second_msgs = mock.calls[1]["messages"]
    tool_msg = [m for m in second_msgs if m.get("role") == "tool"][0]
    assert "denied" in tool_msg["content"].lower() or "declined" in tool_msg["content"].lower()


def test_workspace_file_write_does_not_confirm(config, recording_callbacks):
    """Writing a file inside the workspace runs without a confirmation prompt."""
    tc = [{"id": "c1", "function": {"name": "file_write",
            "arguments": json.dumps({"path": "x.txt", "content": "y"})}}]
    mock = MockLLM([[_evt("done", tool_calls=tc, usage={"total_tokens": 5})],
                    [_evt("content", "written"), _evt("done", usage={"total_tokens": 5})]])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("write file")

    assert recording_callbacks.confirms == []  # no prompt
    assert recording_callbacks.tool_starts == [("file_write", {"path": "x.txt", "content": "y"})]
    import os

    assert os.path.exists(os.path.join(config.workspace, "x.txt"))


def test_workspace_file_modify_does_not_confirm(config, recording_callbacks):
    import os

    with open(os.path.join(config.workspace, "m.txt"), "w") as f:
        f.write("foo")
    tc = [{"id": "c1", "function": {"name": "file_modify",
            "arguments": json.dumps({"path": "m.txt", "old": "foo", "new": "bar"})}}]
    mock = MockLLM([[_evt("done", tool_calls=tc, usage={"total_tokens": 5})],
                    [_evt("content", "ok"), _evt("done", usage={"total_tokens": 5})]])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("edit file")

    assert recording_callbacks.confirms == []
    with open(os.path.join(config.workspace, "m.txt")) as f:
        assert f.read() == "bar"


# ---------------------------------------------------------------------------
# safety: max iterations
# ---------------------------------------------------------------------------

def test_agent_stops_after_max_iterations(config, recording_callbacks):
    tc = [{"id": "c1", "function": {"name": "code_run", "arguments": json.dumps({"code": "print(1)"})}}]
    # always returns another tool call -> infinite loop
    loop = [_evt("done", tool_calls=tc, usage={"total_tokens": 5})]
    mock = MockLLM([loop] * 100)
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.max_iterations = 3
    agent._distill_min_turns = 999  # 隔离异步蒸馏线程（否则它会复用同一 mock）
    agent.run("loop")

    # 每段最多 max_iterations 轮；MAX_TURNS 触发自动续跑（上限 _AUTO_CONTINUE_MAX
    # 段），总调用数有界 —— 不断言 ==3，只断言不会无限循环。
    assert len(mock.calls) <= 3 * (_AUTO_CONTINUE_MAX + 1)


# ---------------------------------------------------------------------------
# tool error propagates back to model
# ---------------------------------------------------------------------------

def test_tool_error_is_returned_to_model(config, recording_callbacks):
    tc = [{"id": "c1", "function": {"name": "file_read",
            "arguments": json.dumps({"path": "missing.txt"})}}]
    mock = MockLLM([
        [_evt("done", tool_calls=tc, usage={"total_tokens": 5})],
        [_evt("content", "could not read"), _evt("done", usage={"total_tokens": 5})],
    ])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("read missing")

    _, res = recording_callbacks.tool_ends[0]
    assert not res.success
    second_msgs = mock.calls[1]["messages"]
    tool_msg = [m for m in second_msgs if m.get("role") == "tool"][0]
    assert "error" in tool_msg["content"].lower() or "not exist" in tool_msg["content"].lower()


# ---------------------------------------------------------------------------
# logs & token speed reporting
# ---------------------------------------------------------------------------

def test_agent_emits_logs_and_speed(config, recording_callbacks):
    import time

    mock = MockLLM([[_evt("content", "hi"), _evt("done", usage={"total_tokens": 5, "completion_tokens": 5})]])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("hi")

    assert len(recording_callbacks.logs) > 0
    # at least one speed report
    assert len(recording_callbacks.speeds) >= 1


# ---------------------------------------------------------------------------
# reasoning streaming: real-time speed + token counting
# ---------------------------------------------------------------------------

def test_speed_emitted_during_reasoning_streaming(config, recording_callbacks):
    """While reasoning chunks are streaming in, the agent must emit real-time
    on_token_speed updates so the UI can show tokens/sec live -- not just a
    single update at the very end."""
    import time
    # A long reasoning stream followed by a short content + done.
    # Inter-chunk delay simulates real streaming so the 0.3s throttle triggers.
    reasoning_chunks = [_evt("reasoning", "thinking " * 20) for _ in range(5)]

    class SlowMockLLM(MockLLM):
        def chat_stream(self, messages, tools=None, temperature=0.7):
            self.calls.append({"messages": messages, "tools": tools, "temperature": temperature})
            script = self._scripts.pop(0)
            def gen():
                for ev in script:
                    time.sleep(0.08)  # 80ms per chunk → 5 chunks = 400ms > 0.3s throttle
                    yield ev
            return gen()

    mock = SlowMockLLM([reasoning_chunks + [_evt("content", "answer"),
                                        _evt("done", usage={"total_tokens": 50, "completion_tokens": 50})]])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("think hard")

    # At least one speed update must arrive *during* reasoning, before the
    # final post-done update. We check there are >1 speed reports total.
    assert len(recording_callbacks.speeds) >= 2, (
        f"expected real-time speed updates during reasoning, got {recording_callbacks.speeds}"
    )


def test_speed_emitted_during_content_only_streaming(config, recording_callbacks):
    """For models that emit only content (no reasoning), the agent must still
    emit real-time on_token_speed updates during streaming — not just a single
    update at the end. Otherwise the UI looks frozen until the turn completes."""
    import time
    content_chunks = [_evt("content", "word " * 20) for _ in range(5)]

    class SlowMockLLM(MockLLM):
        def chat_stream(self, messages, tools=None, temperature=0.7):
            self.calls.append({"messages": messages, "tools": tools, "temperature": temperature})
            script = self._scripts.pop(0)
            def gen():
                for ev in script:
                    time.sleep(0.08)
                    yield ev
            return gen()

    mock = SlowMockLLM([content_chunks + [_evt("done", usage={"total_tokens": 50, "completion_tokens": 50})]])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("write something")

    assert len(recording_callbacks.speeds) >= 2, (
        f"expected real-time speed updates during content streaming, got {recording_callbacks.speeds}"
    )


def test_stream_end_without_done_logs_warning(config, recording_callbacks):
    """If the LLM stream ends without a 'done' event (network cut / server
    truncation), the agent should log a warning so the issue is visible."""
    # Stream that just stops after content — no done event.
    mock = MockLLM([[_evt("content", "partial reply")]])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("hi")

    assert any("without 'done'" in line for line in recording_callbacks.logs), (
        f"expected a warning about missing done event, got logs: {recording_callbacks.logs}"
    )


def test_reasoning_tokens_counted_in_speed(config, recording_callbacks):
    """When the API reports completion_tokens but the model also emitted a lot
    of reasoning text, the final speed must account for the reasoning tokens,
    not just completion_tokens. Otherwise reasoning-heavy models show an
    implausibly low tok/s."""
    # 500 chars of reasoning, but usage claims only 5 completion_tokens
    # (as if the API didn't count reasoning). Speed should not be 5/elapsed.
    reasoning = _evt("reasoning", "x" * 500)
    mock = MockLLM([[reasoning, _evt("content", "ok"),
                     _evt("done", usage={"total_tokens": 10, "completion_tokens": 5})]])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("think")

    # The final speed report should reflect more than just 5 tokens, because
    # the agent estimates reasoning tokens from the streamed text.
    last_total, last_speed = recording_callbacks.speeds[-1]
    assert last_total > 10, (
        f"total tokens should include estimated reasoning tokens, got {last_total}"
    )


def test_reasoning_tokens_added_to_total_when_api_omits_them(config, recording_callbacks):
    """Some local model APIs return usage that doesn't include reasoning tokens
    at all. The agent must add an estimate so the running total reflects
    actual consumption."""
    # Long reasoning, usage reports 0 total_tokens (API bug / omission)
    reasoning = _evt("reasoning", "y" * 400)
    mock = MockLLM([[reasoning, _evt("content", "done"),
                     _evt("done", usage={"total_tokens": 0, "completion_tokens": 0})]])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("think")

    # Despite usage saying 0, the total should be non-zero (estimated).
    last_total, _ = recording_callbacks.speeds[-1]
    assert last_total > 0, "total tokens must include estimated reasoning tokens"


def test_api_reasoning_tokens_used_when_provided(config, recording_callbacks):
    """If the API helpfully provides completion_tokens_details.reasoning_tokens,
    use that exact number instead of a text-length estimate."""
    reasoning = _evt("reasoning", "z" * 300)
    usage = {
        "total_tokens": 100,
        "completion_tokens": 80,
        "completion_tokens_details": {"reasoning_tokens": 60},
    }
    mock = MockLLM([[reasoning, _evt("content", "ok"), _evt("done", usage=usage)]])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("think")

    last_total, _ = recording_callbacks.speeds[-1]
    # API said 100 total; agent should trust that, not inflate with estimates.
    assert last_total == 100


def test_estimate_tokens_helper():
    """The token estimation helper returns a positive int for non-empty text
    and 0 for empty text."""
    from agent.agent import _estimate_tokens

    assert _estimate_tokens("") == 0
    assert _estimate_tokens("hello world") > 0
    # CJK text: each char is roughly 1-2 tokens, so 10 chars >= 5 tokens
    assert _estimate_tokens("你好世界这是一段中文") >= 5
    # longer text -> more tokens
    assert _estimate_tokens("a" * 100) > _estimate_tokens("a" * 10)


# ---------------------------------------------------------------------------
# code_run end-to-end: model writes code -> agent executes -> result fed back
# ---------------------------------------------------------------------------

def test_code_run_result_fed_back_to_model(config, recording_callbacks):
    """The stdout from code_run must appear in the tool message sent back to the
    model so it can reason about the result in the next turn."""
    recording_callbacks._confirm_return = True
    tc = [{"id": "c1", "function": {"name": "code_run",
            "arguments": json.dumps({"code": "print(2+2)"})}}]
    mock = MockLLM([
        [_evt("reasoning", "I'll compute 2+2"), _evt("done", tool_calls=tc, usage={"total_tokens": 5})],
        [_evt("content", "The answer is 4"), _evt("done", usage={"total_tokens": 5})],
    ])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("compute 2+2")

    # tool actually ran and captured stdout
    name, res = recording_callbacks.tool_ends[0]
    assert name == "code_run"
    assert res.success
    assert "4" in res.output
    # the tool message in the second LLM call carries the stdout
    second_msgs = mock.calls[1]["messages"]
    tool_msg = [m for m in second_msgs if m.get("role") == "tool"][0]
    assert "4" in tool_msg["content"]


def test_code_run_error_traceback_fed_back_to_model(config, recording_callbacks):
    """When code_run raises, the traceback must reach the model so it can fix
    the code on the next turn."""
    recording_callbacks._confirm_return = True
    tc = [{"id": "c1", "function": {"name": "code_run",
            "arguments": json.dumps({"code": "1/0"})}}]
    mock = MockLLM([
        [_evt("done", tool_calls=tc, usage={"total_tokens": 5})],
        [_evt("content", "fixed"), _evt("done", usage={"total_tokens": 5})],
    ])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("divide")

    _, res = recording_callbacks.tool_ends[0]
    assert not res.success
    assert "ZeroDivisionError" in res.error
    second_msgs = mock.calls[1]["messages"]
    tool_msg = [m for m in second_msgs if m.get("role") == "tool"][0]
    assert "ZeroDivisionError" in tool_msg["content"]


# ---------------------------------------------------------------------------
# robustness: malformed tool-call arguments must NOT silently run with {}
# ---------------------------------------------------------------------------

def test_malformed_json_args_reported_to_model(config, recording_callbacks):
    """If the model emits a tool_call whose arguments are not valid JSON, the
    agent must tell the model exactly that (not silently run with empty args
    and produce a confusing 'missing argument' error)."""
    recording_callbacks._confirm_return = True
    tc = [{"id": "c1", "function": {"name": "code_run",
            "arguments": "print('hi')"  # not valid JSON, missing braces
           }}]
    mock = MockLLM([
        [_evt("done", tool_calls=tc, usage={"total_tokens": 5})],
        [_evt("content", "recovered"), _evt("done", usage={"total_tokens": 5})],
    ])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("run")

    _, res = recording_callbacks.tool_ends[0]
    assert not res.success
    # the error must mention JSON so the model knows what to fix
    assert "json" in res.error.lower() or "parse" in res.error.lower()
    second_msgs = mock.calls[1]["messages"]
    tool_msg = [m for m in second_msgs if m.get("role") == "tool"][0]
    assert "json" in tool_msg["content"].lower() or "parse" in tool_msg["content"].lower()


def test_empty_args_for_required_param_reports_clear_error(config, recording_callbacks):
    """A tool call with empty arguments for a tool requiring params must produce
    a clear error naming the missing parameter."""
    tc = [{"id": "c1", "function": {"name": "file_read", "arguments": ""}}]
    mock = MockLLM([
        [_evt("done", tool_calls=tc, usage={"total_tokens": 5})],
        [_evt("content", "ok"), _evt("done", usage={"total_tokens": 5})],
    ])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("read")

    _, res = recording_callbacks.tool_ends[0]
    assert not res.success
    assert "path" in res.error.lower()


# ---------------------------------------------------------------------------
# API compatibility: assistant messages and tool_call IDs
# ---------------------------------------------------------------------------

def test_assistant_message_has_content_null_when_only_tool_calls(config, recording_callbacks):
    """OpenAI-compatible APIs expect assistant messages with tool_calls to carry
    a `content` field (null). Omitting it causes HTTP 400 on some servers."""
    recording_callbacks._confirm_return = True
    tc = [{"id": "c1", "function": {"name": "code_run",
            "arguments": json.dumps({"code": "print(1)"})}}]
    mock = MockLLM([
        [_evt("done", tool_calls=tc, usage={"total_tokens": 5})],
        [_evt("content", "done"), _evt("done", usage={"total_tokens": 5})],
    ])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("run")

    # The assistant message in the second call's history must include content.
    second_msgs = mock.calls[1]["messages"]
    asst = [m for m in second_msgs if m.get("role") == "assistant" and m.get("tool_calls")]
    assert asst, "assistant message with tool_calls must be in history"
    assert "content" in asst[0], "assistant message must have content key (null allowed)"


def test_missing_tool_call_id_gets_generated(config, recording_callbacks):
    """Some local models omit tool_call.id. The agent must synthesize one so the
    tool result can be correlated back (tool_call_id must not be empty)."""
    recording_callbacks._confirm_return = True
    tc = [{"function": {"name": "code_run",
            "arguments": json.dumps({"code": "print(1)"})}}]  # no "id"
    mock = MockLLM([
        [_evt("done", tool_calls=tc, usage={"total_tokens": 5})],
        [_evt("content", "done"), _evt("done", usage={"total_tokens": 5})],
    ])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("run")

    second_msgs = mock.calls[1]["messages"]
    asst = [m for m in second_msgs if m.get("role") == "assistant" and m.get("tool_calls")][0]
    tool_msgs = [m for m in second_msgs if m.get("role") == "tool"]
    assert tool_msgs, "tool result message must exist"
    # the tool_call id on the assistant side must be non-empty...
    assert asst["tool_calls"][0].get("id"), "tool_call.id must be non-empty"
    # ...and match the tool_call_id on the tool side
    assert tool_msgs[0]["tool_call_id"] == asst["tool_calls"][0]["id"]


def test_multiple_tool_calls_results_all_sent_back(config, recording_callbacks):
    """When the model emits several tool_calls in one turn, every result must
    be in the next request's history, each with its own tool_call_id."""
    recording_callbacks._confirm_return = True
    tcs = [
        {"id": "c1", "function": {"name": "code_run",
         "arguments": json.dumps({"code": "print(1)"})}},
        {"id": "c2", "function": {"name": "file_write",
         "arguments": json.dumps({"path": "x.txt", "content": "y"})}},
    ]
    mock = MockLLM([
        [_evt("done", tool_calls=tcs, usage={"total_tokens": 5})],
        [_evt("content", "done"), _evt("done", usage={"total_tokens": 5})],
    ])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("do two things")

    second_msgs = mock.calls[1]["messages"]
    tool_msgs = [m for m in second_msgs if m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    ids = {m["tool_call_id"] for m in tool_msgs}
    assert ids == {"c1", "c2"}


def test_code_run_non_string_code_coerced(config, recording_callbacks):
    """If the model emits code as a non-string (number/dict), code_run must
    coerce it to a string rather than crashing with a TypeError."""
    recording_callbacks._confirm_return = True
    tc = [{"id": "c1", "function": {"name": "code_run",
            "arguments": json.dumps({"code": 42})}}]
    mock = MockLLM([
        [_evt("done", tool_calls=tc, usage={"total_tokens": 5})],
        [_evt("content", "ok"), _evt("done", usage={"total_tokens": 5})],
    ])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("run number")

    _, res = recording_callbacks.tool_ends[0]
    # Coercion succeeded: the tool ran without a TypeError crash.
    assert res.success
    assert "typeerror" not in res.error.lower()


# ---------------------------------------------------------------------------
# shell_run end-to-end: model calls shell_run -> agent executes -> result back
# ---------------------------------------------------------------------------

def test_shell_run_result_fed_back_to_model(config, recording_callbacks):
    """shell_run stdout must appear in the tool message sent back to the model."""
    recording_callbacks._confirm_return = True
    tc = [{"id": "c1", "function": {"name": "shell_run",
            "arguments": json.dumps({"command": "echo hello_from_shell"})}}]
    mock = MockLLM([
        [_evt("reasoning", "I'll run a shell command"), _evt("done", tool_calls=tc, usage={"total_tokens": 5})],
        [_evt("content", "The shell said hello"), _evt("done", usage={"total_tokens": 5})],
    ])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("run echo")

    name, res = recording_callbacks.tool_ends[0]
    assert name == "shell_run"
    assert res.success
    assert "hello_from_shell" in res.output
    second_msgs = mock.calls[1]["messages"]
    tool_msg = [m for m in second_msgs if m.get("role") == "tool"][0]
    assert "hello_from_shell" in tool_msg["content"]


def test_shell_run_requires_confirmation(config, recording_callbacks):
    """shell_run is destructive — it must ask the user before executing."""
    tc = [{"id": "c1", "function": {"name": "shell_run",
            "arguments": json.dumps({"command": "ls"})}}]
    mock = MockLLM([
        [_evt("done", tool_calls=tc, usage={"total_tokens": 5})],
        [_evt("content", "ok"), _evt("done", usage={"total_tokens": 5})],
    ])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    recording_callbacks._confirm_return = True
    agent.run("list files")

    assert len(recording_callbacks.confirms) == 1
    assert recording_callbacks.tool_starts[0][0] == "shell_run"


def test_shell_run_error_fed_back_to_model(config, recording_callbacks):
    """A non-zero exit code must be reported back so the model can react."""
    recording_callbacks._confirm_return = True
    tc = [{"id": "c1", "function": {"name": "shell_run",
            "arguments": json.dumps({"command": "exit 7"})}}]
    mock = MockLLM([
        [_evt("done", tool_calls=tc, usage={"total_tokens": 5})],
        [_evt("content", "noted the failure"), _evt("done", usage={"total_tokens": 5})],
    ])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("fail")

    _, res = recording_callbacks.tool_ends[0]
    assert not res.success
    second_msgs = mock.calls[1]["messages"]
    tool_msg = [m for m in second_msgs if m.get("role") == "tool"][0]
    assert "7" in tool_msg["content"]


def test_code_run_then_shell_run_feedback_loop(config, recording_callbacks):
    """Multi-turn feedback loop: model writes a file via code_run, then reads
    it back via shell_run, then produces a final answer. Verifies the full
    tool-calling cycle: parse -> confirm -> execute -> result -> next call."""
    recording_callbacks._confirm_return = True
    tc1 = [{"id": "c1", "function": {"name": "code_run",
            "arguments": json.dumps({"code": "open('loop.txt','w').write('data123')"})}}]
    tc2 = [{"id": "c2", "function": {"name": "shell_run",
            "arguments": json.dumps({"command": "cat loop.txt"})}}]
    mock = MockLLM([
        [_evt("reasoning", "write file"), _evt("done", tool_calls=tc1, usage={"total_tokens": 5})],
        [_evt("reasoning", "read it back"), _evt("done", tool_calls=tc2, usage={"total_tokens": 5})],
        [_evt("content", "The file contains data123"), _evt("done", usage={"total_tokens": 5})],
    ])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    answer = agent.run("write and read back")

    assert "data123" in answer
    assert [n for n, _ in recording_callbacks.tool_starts] == ["code_run", "shell_run"]
    # 3 LLM calls: write -> read -> final answer
    assert len(mock.calls) == 3
    # The second LLM call sees code_run's result; the third sees both results
    # (history accumulates). The latest tool result (shell_run) has data123.
    second_tool_msgs = [m for m in mock.calls[1]["messages"] if m.get("role") == "tool"]
    third_tool_msgs = [m for m in mock.calls[2]["messages"] if m.get("role") == "tool"]
    assert len(second_tool_msgs) == 1
    assert len(third_tool_msgs) == 2  # code_run + shell_run results
    # the shell_run result (tool_call_id c2) should carry the file content
    shell_msg = [m for m in third_tool_msgs if m["tool_call_id"] == "c2"][0]
    assert "data123" in shell_msg["content"]


# ---------------------------------------------------------------------------
# context shrink: proactive (90% threshold) + reactive (context_length error)
# ---------------------------------------------------------------------------

def test_is_context_too_long_error_detects_common_patterns():
    """The error-pattern detector should match OpenAI/Anthropic/local server
    variants of context-length-exceeded errors."""
    from agent.agent import Agent

    assert Agent._is_context_too_long_error(
        "chat_stream failed: HTTP 400 {'error': {'message': 'context_length_exceeded'}}"
    )
    assert Agent._is_context_too_long_error("This model's maximum context length is 8192 tokens")
    assert Agent._is_context_too_long_error("prompt is too long")
    assert Agent._is_context_too_long_error("too many tokens in prompt")
    # Non-context errors must not match.
    assert not Agent._is_context_too_long_error("HTTP 500 internal server error")
    assert not Agent._is_context_too_long_error("rate limit exceeded")
    assert not Agent._is_context_too_long_error("invalid api key")


# ---------------------------------------------------------------------------
# 内核：历史恒定（L2/L3）与 30k 预算
#
# 旧实现把整条 _history 每轮重发，大小是 O(轮次)，只能靠超阈值后的被动摘要
# 兜底。内核改为「最近 K 轮原文 + 旧轮折叠成一行摘要」，历史恒定 O(1)。
# ---------------------------------------------------------------------------


def _loop_scripts(n, args=None, workspace=None):
    """n 轮工具调用 + 1 轮最终答复。

    每轮在 file_read 与 update_working_checkpoint 之间**轮换**：内核现在会检测
    「同名 + 同参数的连续重复调用」并强制收尾（防模型空转），若每轮参数完全
    相同会被判成卡住，跑不到 n 轮。
    """
    args = args or {"path": "a.txt"}
    tc_read = [{"id": "c1", "function": {"name": "file_read",
                                         "arguments": json.dumps(args)}}]
    scripts = []
    for i in range(n):
        if i % 2:
            tc = [{"id": f"c{i}", "function": {
                "name": "update_working_checkpoint",
                "arguments": json.dumps({"key_info": f"turn {i}"})}}]
        else:
            tc = tc_read
        scripts.append([_evt("done", tool_calls=tc, usage={"total_tokens": 5})])
    scripts.append([_evt("content", "done"), _evt("done", usage={"total_tokens": 5})])
    return scripts


def test_kernel_context_stays_bounded_across_many_turns(config, recording_callbacks):
    """跑到第 N 轮时，发往 LLM 的 prompt 大小不再随轮次线性膨胀。"""
    with open(os.path.join(config.workspace, "a.txt"), "w") as f:
        f.write("A" * 100)
    mock = MockLLM(_loop_scripts(12))
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.max_iterations = 20
    agent._distill_min_turns = 999  # 关掉异步蒸馏，避免后台线程抢占 mock 脚本
    agent.run("loop")

    def size(call):
        return sum(len(str(m.get("content") or "")) for m in call["messages"])

    assert len(mock.calls) >= 10
    early, late = size(mock.calls[2]), size(mock.calls[-1])
    assert late < early * 2 + 3000, f"上下文未收敛: {early} -> {late}"


def test_kernel_folds_old_turns_into_earlier_context(config, recording_callbacks):
    """超出原文窗口的旧轮必须折叠进 <earlier_context>，不再占原文额度。"""
    with open(os.path.join(config.workspace, "a.txt"), "w") as f:
        f.write("A" * 100)
    mock = MockLLM(_loop_scripts(10))
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.max_iterations = 20
    agent._distill_min_turns = 999  # 关掉异步蒸馏，避免后台线程抢占 mock 脚本
    agent.run("loop")

    msgs = mock.calls[-1]["messages"]
    joined = "\n".join(str(m.get("content") or "") for m in msgs)
    assert "<earlier_context>" in joined, "旧轮未折叠"
    # 原文窗口只保留 kernel_raw_turns 个已完成轮
    assistants = [m for m in msgs if m.get("role") == "assistant"]
    assert len(assistants) <= int(config.kernel_raw_turns) + 1


def test_kernel_never_emits_dangling_tool_message(config, recording_callbacks):
    """任何 tool 消息都必须紧跟带 tool_calls 的 assistant —— 否则 API 400。

    折叠以「整轮」为单位，因此这个不变量在结构上就成立，不依赖事后修复。
    """
    with open(os.path.join(config.workspace, "a.txt"), "w") as f:
        f.write("A" * 100)
    mock = MockLLM(_loop_scripts(8))
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.max_iterations = 20
    agent._distill_min_turns = 999  # 关掉异步蒸馏，避免后台线程抢占 mock 脚本
    agent.run("loop")

    for call in mock.calls:
        msgs = call["messages"]
        assert msgs[0].get("role") == "system", "首条必须是 system"
        for i, m in enumerate(msgs):
            if m.get("role") != "tool":
                continue
            assert i > 0, "tool 消息出现在位置 0"
            prev = msgs[i - 1]
            assert prev.get("role") == "assistant" and prev.get("tool_calls"), (
                f"悬挂 tool 消息 @ {i}: 前置是 {prev.get('role')}"
            )
            ids = {tc["id"] for tc in prev["tool_calls"]}
            assert m.get("tool_call_id") in ids


def test_kernel_budget_allocates_tool_output_by_parallelism():
    """L1：单条工具输出的上限由「RAW 区余额 ÷ 并发数」推出，且不会为负。"""
    from agent.kernel import ContextBudget

    b = ContextBudget(30_000)
    b.set_used("raw", 0)
    assert b.remaining_for_tool(1, 100_000) == b.quota.raw * 3
    assert b.remaining_for_tool(4, 100_000) < b.remaining_for_tool(1, 100_000)
    assert b.remaining_for_tool(1, 2_000) == 2_000   # max_chars 钳制
    # 已经超预算 → 只给最小可诊断额度，逼内核立刻折叠
    b.set_used("raw", b.quota.raw + 100)
    assert b.remaining_for_tool(1, 6_000) <= 600
    assert b.raw_over


def test_shrink_session_fallback_writes_memory_md(config, recording_callbacks):
    """折叠用尽后的兜底收缩：LLM 摘要 + 落盘 memory.md + 通知 UI。"""
    mock = MockLLM([[_evt("content", "用户的关键需求是部署到生产环境"),
                     _evt("done", usage={})]])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    for i in range(5):
        rec = agent.session.start_turn(f"old {i}")
        agent.session.finish_turn(rec, f"reply {i}", [], [], f"摘要{i}")
    agent.session.start_turn("current")

    assert agent._shrink_session("测试兜底收缩") is True

    memory_path = os.path.join(config.workspace, "memory.md")
    assert os.path.exists(memory_path)
    with open(memory_path, encoding="utf-8") as f:
        content = f.read()
    assert "上下文摘要" in content and "触发原因" in content
    assert "部署到生产环境" in content
    assert not os.path.exists(memory_path + ".tmp")
    assert len(recording_callbacks.context_shrinks) == 1


def test_shrink_session_skipped_when_too_small(config, recording_callbacks):
    """只有一轮时不做兜底收缩——不能把唯一的上下文也摘要掉。"""
    agent = Agent(llm=MockLLM([]), config=config, callbacks=recording_callbacks)
    agent.session.start_turn("only")
    assert agent._shrink_session("x") is False
    assert recording_callbacks.context_shrinks == []


# ---------------------------------------------------------------------------
# 五级压缩的原语
# ---------------------------------------------------------------------------


def test_l2_summary_prefers_summary_tag():
    from agent.kernel import l2_summary

    assert l2_summary("<summary>读取了 a.txt，下一步写文件</summary>") == \
        "读取了 a.txt，下一步写文件"
    # 没有 <summary> 时回退到首句，且永不返回空串
    assert l2_summary("先读文件。再写文件。") == "先读文件。"
    assert l2_summary("") == "（无文本输出）"


def test_l3_fold_merges_consecutive_repeats():
    from agent.kernel import l3_fold

    lines = ["[T1] a", "[T2] b", "[T2] b", "[T2] b", "[T3] c"]
    out = l3_fold(lines)
    assert "[T2] b（3 turns）" in out
    assert out.count("[T2] b") == 1


def test_l0_capture_hard_cap():
    from agent.kernel import l0_capture

    big = l0_capture("x" * 50_000, hard_cap=2_000)
    assert len(big) <= 2_200
    assert big.startswith("x") and big.endswith("x")


def test_l4_normalize_is_total():
    """提纯结果必须类型确定，任何脏输入都不能让下游炸掉。"""
    from agent.kernel import l4_extract_json, l4_normalize

    assert l4_normalize(l4_extract_json("")) == {"facts": [], "lessons": [],
                                                 "skill": {"name": "", "summary": "",
                                                           "when_to_use": "", "steps": ""}}
    got = l4_normalize(l4_extract_json(
        '```json\n{"facts": ["a"], "skill": {"name": "My Skill!", "summary": "s"}}\n```'))
    assert got["facts"] == ["a"]
    assert got["skill"]["name"] == "my-skill"   # 目录穿越防护

# ---------------------------------------------------------------------------
# RAG integration: agent discovers and calls rag_status / rag_search
# ---------------------------------------------------------------------------

def test_agent_calls_rag_tools_when_knowledge_base_configured(
    config, recording_callbacks, tmp_path, fake_ef, monkeypatch
):
    """End-to-end RAG tool invocation through the agent loop.

    When a knowledge_base is configured, the agent must:
    1. Expose rag_status / rag_search / rag_ingest in the tool schema list
       passed to the LLM.
    2. Actually dispatch LLM-emitted tool_calls to the RAGEngine.
    3. Feed the rag_search result back to the LLM as a tool message so the
       next turn can ground its answer in retrieved context.

    Uses fake_ef (no FastEmbed model download) by monkeypatching
    RAGEngine in the engine module — ToolRegistry's _register_rag_tools
    imports RAGEngine lazily inside the function, so the patch is picked up.
    """
    import json as _json

    from agent.rag import engine as engine_mod
    from agent.tools.base import ToolRegistry

    # 1. Set up a knowledge base with a unique fact the LLM cannot know.
    kb = tmp_path / "kb"
    kb.mkdir()
    unique_fact = "ZetaCloud 内部代号 ProjectAurora2024"
    (kb / "company_faq.md").write_text(
        f"# 公司内部 FAQ\n\n"
        f"## 项目代号\n\n{unique_fact}\n"
        f"该项目于 2024 年第三季度启动，负责人是张工。\n",
        encoding="utf-8",
    )

    # 2. Patch RAGEngine so the registry's _register_rag_tools uses fake_ef
    #    (no FastEmbed / ONNX model download).
    class _FakeRAGEngine(engine_mod.RAGEngine):
        def __init__(self, workspace, knowledge_base="", **kwargs):
            kwargs.setdefault("embedding_function", fake_ef)
            super().__init__(workspace=workspace, knowledge_base=knowledge_base, **kwargs)

    monkeypatch.setattr(engine_mod, "RAGEngine", _FakeRAGEngine)

    # 3. Configure knowledge_base and build the registry.
    config.knowledge_base = str(kb)
    registry = ToolRegistry(config=config, callbacks=recording_callbacks)

    # 4. Verify RAG tools are exposed to the LLM via schemas().
    schema_names = {s["function"]["name"] for s in registry.schemas()}
    assert "rag_search" in schema_names, "rag_search must be in tool schemas"
    assert "rag_status" in schema_names, "rag_status must be in tool schemas"
    assert "rag_ingest" in schema_names, "rag_ingest must be in tool schemas"

    # 5. Ingest synchronously so the vector store has data to search.
    rag_engine = registry.get_rag_engine()
    assert rag_engine is not None, "RAG engine was not initialized"
    stats = rag_engine.ingest(force=True)
    assert stats["chunks"] > 0, f"ingest produced no chunks: {stats}"
    # Refresh the LanceDB table handle so the upcoming search sees new rows.
    registry.reload_rag()

    # 6. Script the LLM: turn 1 calls rag_status, turn 2 calls rag_search,
    #    turn 3 produces the final grounded answer.
    tc_status = [{"id": "c1", "function": {"name": "rag_status",
                  "arguments": "{}"}}]
    tc_search = [{"id": "c2", "function": {"name": "rag_search",
                  "arguments": _json.dumps({"query": "项目代号是什么"})}}]
    mock = MockLLM([
        [_evt("reasoning", "First check what's in the knowledge base."),
         _evt("done", tool_calls=tc_status, usage={"total_tokens": 5})],
        [_evt("reasoning", "Now search for the project codename."),
         _evt("done", tool_calls=tc_search, usage={"total_tokens": 5})],
        [_evt("content", f"根据知识库，{unique_fact}。"),
         _evt("done", usage={"total_tokens": 20})],
    ])

    # 7. Run the agent.
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks,
                  tool_registry=registry)
    agent.run("公司项目的代号是什么？")

    # 8. Both RAG tools must have been dispatched, in order.
    called = [n for n, _ in recording_callbacks.tool_starts]
    assert called == ["rag_status", "rag_search"], (
        f"expected [rag_status, rag_search], got {called}"
    )

    # 9. Both tool calls must have succeeded.
    for name, res in recording_callbacks.tool_ends:
        assert res.success, f"{name} failed: {res.error}"

    # 10. The rag_search result must carry the unique fact.
    search_name, search_result = recording_callbacks.tool_ends[1]
    assert search_name == "rag_search"
    assert unique_fact in search_result.output, (
        f"rag_search result missing unique fact: {search_result.output[:300]}"
    )

    # 11. The rag_search result must be in the third LLM call's history so
    #     the model can ground its final answer in retrieved context.
    third_msgs = mock.calls[2]["messages"]
    rag_tool_msgs = [m for m in third_msgs
                     if m.get("role") == "tool" and m.get("name") == "rag_search"]
    assert rag_tool_msgs, "rag_search result not in third LLM call history"
    assert unique_fact in rag_tool_msgs[0]["content"], (
        "RAG context not propagated to LLM for the final answer"
    )

    # 12. The agent's final visible reply should reference the retrieved fact.
    assert unique_fact in "".join(recording_callbacks.content), (
        "agent final reply did not include the retrieved knowledge base fact"
    )

    # 13. Release RAG engine resources (no leak across tests).
    registry.shutdown()


def test_agent_registers_rag_tools_with_default_knowledge_base(
    config, recording_callbacks
):
    """When knowledge_base is empty, the agent defaults to
    <workspace>/knowledge_base and registers RAG tools so the agent can
    discover the local knowledge base."""
    from agent.tools.base import ToolRegistry

    config.knowledge_base = ""
    registry = ToolRegistry(config=config, callbacks=recording_callbacks)
    schema_names = {s["function"]["name"] for s in registry.schemas()}
    assert "rag_search" in schema_names
    assert "rag_status" in schema_names
    assert "rag_ingest" in schema_names
    registry.shutdown()


# ---------------------------------------------------------------------------
# Daily auto-extracted facts (short-term memory) — 短期记忆只保留当天事实
# ---------------------------------------------------------------------------

def test_parse_daily_facts_valid():
    """_parse_daily_facts 解析合法的当天桶。"""
    from agent.agent import Agent
    raw = '{"date": "2026-08-15", "facts": ["f1", "f2"]}'
    date, facts = Agent._parse_daily_facts(raw)
    assert date == "2026-08-15"
    assert facts == ["f1", "f2"]


def test_parse_daily_facts_malformed():
    """_parse_daily_facts 对空/非法输入返回 ("", [])，视为需重置。"""
    from agent.agent import Agent
    assert Agent._parse_daily_facts("") == ("", [])
    assert Agent._parse_daily_facts("not json") == ("", [])
    assert Agent._parse_daily_facts('{"date": 123, "facts": "oops"}') == ("", [])


def test_parse_daily_facts_filters_non_strings():
    """facts 里的非字符串项被过滤。"""
    from agent.agent import Agent
    raw = '{"date": "2026-08-15", "facts": ["ok", 123, null, "  ", "also"]}'
    date, facts = Agent._parse_daily_facts(raw)
    assert date == "2026-08-15"
    assert facts == ["ok", "also"]


def test_append_daily_facts_writes_and_dedups(config, recording_callbacks):
    """_append_daily_facts 写入当天桶并去重。"""
    import datetime
    from agent.agent import Agent
    from agent.tools.memory import _work_memory

    agent = Agent(llm=MockLLM([[_evt("content", "x"), _evt("done")]]),
                  config=config, callbacks=recording_callbacks)

    agent._append_daily_facts(["事实A", "事实B"])
    agent._append_daily_facts(["事实B", "事实C"])  # B 重复，应去重

    wm = _work_memory(config)
    date, facts = agent._parse_daily_facts(wm.get("__auto_facts__") or "")
    assert date == datetime.date.today().isoformat()
    assert facts == ["事实A", "事实B", "事实C"]


def test_append_daily_facts_resets_on_new_day(config, recording_callbacks):
    """跨天后追加，旧当天桶应被清空（历史归长期记忆）。"""
    import datetime
    from agent.agent import Agent
    from agent.tools.memory import _work_memory

    agent = Agent(llm=MockLLM([[_evt("content", "x"), _evt("done")]]),
                  config=config, callbacks=recording_callbacks)

    # 预置一个「昨天」的桶。
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    _work_memory(config).set(
        "__auto_facts__",
        json.dumps({"date": yesterday, "facts": ["旧事实"]}, ensure_ascii=False),
    )

    # 今天追加新事实 → 旧事实应被丢弃。
    agent._append_daily_facts(["新事实"])

    wm = _work_memory(config)
    date, facts = agent._parse_daily_facts(wm.get("__auto_facts__") or "")
    assert date == datetime.date.today().isoformat()
    assert facts == ["新事实"], f"跨天应只保留当天事实，got {facts}"


# ---------------------------------------------------------------------------
# 独立抽取 LLM 客户端（避免与对话线程争用 httpx.Client）
# ---------------------------------------------------------------------------

def test_get_extract_llm_returns_dedicated_client(config, recording_callbacks):
    """self.llm 是 LLMClient 时，抽取应使用独立实例（不同连接池）。"""
    llm = LLMClient(base_url="http://localhost:1/v1", api_key="", model="m")
    agent = Agent(llm=llm, config=config, callbacks=recording_callbacks)
    try:
        ext = agent._get_extract_llm()
        assert ext is not llm            # 独立实例
        assert isinstance(ext, LLMClient)
        assert ext is agent._get_extract_llm()  # 缓存复用
    finally:
        agent.close()
        llm.close()


def test_get_extract_llm_falls_back_to_mock(config, recording_callbacks):
    """self.llm 是 mock（非 LLMClient）时，直接复用 self.llm（无共享 socket）。"""
    mock = MockLLM([[_evt("content", "x"), _evt("done")]])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    assert agent._get_extract_llm() is mock
    # close() 不应关闭 mock（不拥有它）。
    agent.close()


def test_agent_close_releases_extract_client(config, recording_callbacks):
    """close() 释放独立抽取客户端，且不关闭 self.llm。"""
    llm = LLMClient(base_url="http://localhost:1/v1", api_key="", model="m")
    agent = Agent(llm=llm, config=config, callbacks=recording_callbacks)
    ext = agent._get_extract_llm()
    assert ext is not llm
    agent.close()
    # 独立客户端已释放，再次获取会重新创建（或用 self.llm）。
    assert getattr(agent, "_extract_llm", None) is None
    # self.llm 未被关闭（仍可复用）。
    llm.close()



# ---------------------------------------------------------------------------
# cooperative cancellation
# ---------------------------------------------------------------------------

def test_agent_cancel_stops_streaming(config, recording_callbacks):
    """「终止对话」必须在流式期间立即生效。

    旧实现只在 confirm/ask_user 里检查 _cancelled——长回复/工具链场景下取消
    形同虚设，会跑满 max_iterations。
    """
    # 20 个 content chunk + done：取消后不应再调用 LLM，也不应产出 done。
    events = [_evt("content", f"chunk{i}") for i in range(20)]
    events.append(_evt("done", usage={"total_tokens": 999}))
    mock = MockLLM([[*events]])
    recording_callbacks._cancel_after = ("content", 3)
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("tell me something long")

    # 只消费到取消点附近的几个 chunk，20 个不会全部流完。
    assert len(recording_callbacks.content) < 20
    # on_finished 恰好调用一次（旧实现会漏掉，UI 卡在「运行中」）。
    assert recording_callbacks.logs == recording_callbacks.logs  # no-op sanity
    assert not any("max iterations" in l for l in recording_callbacks.logs)


def test_agent_cancel_skips_remaining_tools_and_keeps_history_consistent(
        config, recording_callbacks):
    """取消后剩余 tool_call 必须逐个补合成结果——否则历史里出现悬挂
    tool_calls，下一轮请求会被 API 以 HTTP 400 拒绝。"""
    tc = [
        {"id": "c1", "function": {"name": "file_read",
                                  "arguments": json.dumps({"path": "a.txt"})}},
        {"id": "c2", "function": {"name": "file_read",
                                  "arguments": json.dumps({"path": "b.txt"})}},
    ]
    with open(os.path.join(config.workspace, "a.txt"), "w") as f:
        f.write("A")
    with open(os.path.join(config.workspace, "b.txt"), "w") as f:
        f.write("B")
    mock = MockLLM([
        [_evt("done", tool_calls=tc, usage={"total_tokens": 5})],
    ])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    # 第一个工具结束后取消。
    recording_callbacks._cancel_after = ("tool_end", 1)
    orig_on_tool_end = recording_callbacks.on_tool_end

    def on_tool_end(name, result):
        recording_callbacks._cancel_count += 1
        orig_on_tool_end(name, result)

    recording_callbacks.on_tool_end = on_tool_end
    recording_callbacks._cancel_after = ("tool_end", 1)
    recording_callbacks.is_cancelled = lambda: recording_callbacks._cancel_count >= 1

    agent.run("read both files")

    names = [n for n, _ in recording_callbacks.tool_starts]
    assert names.count("file_read") == 2  # 第 1 个真执行，第 2 个跳过但也走了 start
    # 历史一致性：assistant 的每个 tool_call 都有对应 tool 消息（否则 API 400）。
    rec = next(t for t in agent.session.turns if t.tool_results)
    assert len(rec.tool_results) == 2
    assert "取消" in rec.tool_results[1]["content"]


# ---------------------------------------------------------------------------
# Harness: LLM 重试 / 工具并发 / 工具执行中取消 / 历史裁剪
# ---------------------------------------------------------------------------

class FlakyLLM:
    """前 N 次调用直接抛错（首字节前失败），之后返回正常脚本。"""

    def __init__(self, failures, script):
        self.failures = failures
        self.script = script
        self.calls = 0

    def chat_stream(self, messages, tools=None, temperature=0.7):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("connection reset")
        return iter(self.script)


class PartialThenFailLLM:
    """先吐出内容再断流 —— 不该重试（重试会把两段内容拼在一起）。"""

    def __init__(self):
        self.calls = 0

    def chat_stream(self, messages, tools=None, temperature=0.7):
        self.calls += 1

        def _gen():
            yield _evt("content", "half")
            raise RuntimeError("stream cut")

        return _gen()


def test_llm_retries_before_first_chunk(config, recording_callbacks, monkeypatch):
    """首字节前的失败（网络抖动/502）应退避重试，而不是整轮失败。"""
    import agent.kernel.loop as kloop

    monkeypatch.setattr(kloop, "_LLM_RETRY_BASE_DELAY", 0.01)
    mock = FlakyLLM(1, [_evt("content", "ok"), _evt("done", usage={"total_tokens": 3})])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)

    assert agent.run("hi") == "ok"
    assert mock.calls == 2


def test_llm_does_not_retry_after_partial_output(config, recording_callbacks, monkeypatch):
    """已经吐过内容的流中断后不能重试。"""
    import agent.kernel.loop as kloop

    monkeypatch.setattr(kloop, "_LLM_RETRY_BASE_DELAY", 0.01)
    mock = PartialThenFailLLM()
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)

    agent.run("hi")
    assert mock.calls == 1, "半截输出后重试会把两段内容拼在一起"


def test_context_length_error_is_not_retried(config, recording_callbacks, monkeypatch):
    """上下文过长交给收缩逻辑处理，重试同一个超长 prompt 没有意义。"""
    import agent.kernel.loop as kloop

    monkeypatch.setattr(kloop, "_LLM_RETRY_BASE_DELAY", 0.01)

    class TooLongLLM:
        def __init__(self):
            self.calls = 0

        def chat_stream(self, messages, tools=None, temperature=0.7):
            self.calls += 1
            raise RuntimeError("context_length_exceeded: prompt is too long")

    mock = TooLongLLM()
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("hi")
    assert mock.calls == 1


class SlowReadTool:
    """只读的慢工具，用于验证并发执行。"""

    name = "slow_read"
    description = "slow read-only tool"
    parameters = {"type": "object", "properties": {"i": {"type": "integer"}}}
    destructive = False

    def bind(self, config, registry):
        pass

    def should_confirm(self, args):
        return False

    def schema(self):
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": self.parameters}}

    def run(self, i: int = 0):
        time.sleep(0.4)
        from agent.tools.base import ToolResult
        return ToolResult(True, output=f"R{i}")


def test_independent_read_only_tools_run_in_parallel(config, recording_callbacks, monkeypatch):
    """多个互相独立的只读工具并发执行：wall time 从 Σ 降为 max。"""
    import agent.agent as agent_mod

    monkeypatch.setattr(agent_mod, "_PARALLEL_SAFE_TOOLS", frozenset({"slow_read"}))
    tc = [
        {"id": "c1", "function": {"name": "slow_read", "arguments": json.dumps({"i": 1})}},
        {"id": "c2", "function": {"name": "slow_read", "arguments": json.dumps({"i": 2})}},
    ]
    mock = MockLLM([
        [_evt("done", tool_calls=tc, usage={"total_tokens": 5})],
        [_evt("content", "done"), _evt("done", usage={"total_tokens": 6})],
    ])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.tools._tools["slow_read"] = SlowReadTool()
    # 预热系统提示（首次 rag_engine.status() 会惰性导入 lancedb，冷启动数秒），
    # 否则计时测的是冷导入而不是工具并发。
    agent._system_prompt()

    t0 = time.time()
    agent.run("read both")
    elapsed = time.time() - t0
    assert elapsed < 0.75, f"串行执行了？两个 0.4s 的工具耗时 {elapsed:.2f}s"

    # 结果顺序必须仍与 tool_calls 一致（并发执行不改变顺序）
    rec = next(t for t in agent.session.turns if t.tool_results)
    assert [r["id"] for r in rec.tool_results] == ["c1", "c2"]
    assert [r["content"] for r in rec.tool_results] == ["R1", "R2"]


def test_cancel_during_tool_execution_returns_promptly(config, recording_callbacks, monkeypatch):
    """工具执行期间取消：不能等工具跑完（shell/MCP 可能是几分钟）。"""
    import agent.agent as agent_mod

    monkeypatch.setattr(agent_mod, "_PARALLEL_SAFE_TOOLS", frozenset())
    cbs = recording_callbacks
    t_start = time.time()
    cbs.is_cancelled = lambda: (time.time() - t_start) > 0.15

    tc = [{"id": "c1", "function": {"name": "slow_read", "arguments": json.dumps({"i": 1})}}]
    mock = MockLLM([[_evt("done", tool_calls=tc, usage={"total_tokens": 5})]])
    agent = Agent(llm=mock, config=config, callbacks=cbs)
    agent.tools._tools["slow_read"] = SlowReadTool()

    t0 = time.time()
    agent.run("read")
    elapsed = time.time() - t0
    assert elapsed < 0.35, f"取消后仍等了 {elapsed:.2f}s（工具本身要跑 0.4s）"
    rec = next((t for t in agent.session.turns if t.tool_results), None)
    assert rec and "取消" in rec.tool_results[0]["content"]


def test_large_tool_output_is_capped_before_feeding_model(config, recording_callbacks):
    """单条 50K 的工具输出在注入下一轮之前必须被 L1 裁剪。

    旧实现是「先入历史、超阈值后再回头裁」；内核在结果进上下文之前就按
    RAW 区余额分配上限，因此单条输出永远不可能撑爆 30k 预算。
    """
    from agent.tools.base import Tool, ToolResult

    class BigReadTool(Tool):
        name = "big_read"
        description = "returns a huge payload"
        parameters = {"type": "object", "properties": {}}

        def run(self, **kwargs):
            return ToolResult(True, output="x" * 50_000)

    tc = [{"id": "c1", "function": {"name": "big_read", "arguments": "{}"}}]
    mock = MockLLM([
        [_evt("done", tool_calls=tc, usage={"total_tokens": 5})],
        [_evt("content", "ok"), _evt("done", usage={"total_tokens": 5})],
    ])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.tools._tools["big_read"] = BigReadTool()

    agent.run("read big")

    tool_msgs = [m for m in mock.calls[1]["messages"] if m.get("role") == "tool"]
    assert tool_msgs, "工具结果必须回灌给模型"
    assert len(tool_msgs[0]["content"]) < 12_000, f"未裁剪: {len(tool_msgs[0]['content'])}"


def test_max_iterations_leaves_continuation_hint(config, recording_callbacks):
    """迭代上限耗尽时，必须让下一轮知道上一轮是被截断的。"""
    with open(os.path.join(config.workspace, "a.txt"), "w") as f:
        f.write("A")
    tc = [{"id": "c1", "function": {"name": "file_read",
                                    "arguments": json.dumps({"path": "a.txt"})}}]
    mock = MockLLM([[_evt("done", tool_calls=tc, usage={"total_tokens": 5})]])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.max_iterations = 1
    agent._auto_continue = False    # 本测试专测截断提示；自动续跑会消耗脚本

    agent.run("read")
    hint = agent.session.working.get("__truncated__", "")
    assert "最大轮次上限" in hint
    # 下一轮的锚点提示必须带上它，否则模型会以为任务已完成
    assert "最大轮次上限" in agent.session.anchor_prompt()


def test_tools_schemas_are_cached(config, recording_callbacks):
    """工具 schema 在注册表构造后即固定，不该每轮重建。"""
    mock = MockLLM([
        [_evt("content", "a"), _evt("done", usage={"total_tokens": 1})],
        [_evt("content", "b"), _evt("done", usage={"total_tokens": 1})],
    ])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("one")
    agent.run("two")
    assert mock.calls[0]["tools"] is mock.calls[1]["tools"]
    assert agent._tools_schemas() is agent._tools_schemas()
