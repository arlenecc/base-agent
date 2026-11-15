"""本地模型文本工具调用解析 + 内核空转兜底。"""
from __future__ import annotations

from agent.local.llm import parse_tool_calls


def _names_args(calls):
    return [(c["function"]["name"], c["function"]["arguments"]) for c in calls]


def test_parse_qwen_xml():
    text = (
        "<tool_call><function=file_tools>"
        "<parameter=op>read</parameter><parameter=path>a.txt</parameter>"
        "</function></tool_call>"
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "file_tools"
    assert '"op": "read"' in calls[0]["function"]["arguments"]


def test_parse_function_without_tool_call_wrapper():
    text = ("<function=file_tools><parameter=op>list</parameter>"
            "</function>")
    calls = parse_tool_calls(text)
    assert len(calls) == 1 and calls[0]["function"]["name"] == "file_tools"


def test_parse_hermes_json_with_literal_newlines():
    text = (
        '<tool_call>{"name": "file_tools", "arguments": '
        '{"op": "write", "path": "a.txt", "content": "line1\nline2"}}'
        "</tool_call>"
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "file_tools"


def test_parse_unclosed_tool_call():
    text = ('<tool_call>{"name": "file_tools", "arguments": '
            '{"op": "read", "path": "a.txt"}}')
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "file_tools"


def test_parse_json_array_and_aliases():
    text = ('<tool_call>[{"tool": "file_tools", "args": {"op": "list"}}, '
            '{"name": "update_plan", "arguments": {"op": "clear"}}]</tool_call>')
    calls = parse_tool_calls(text)
    assert [c["function"]["name"] for c in calls] == ["file_tools", "update_plan"]


def test_parse_bare_tool_tag_with_known_names():
    text = '<update_plan>{"op": "create", "goal": "g", "steps": [{"title": "s"}]}</update_plan>'
    calls = parse_tool_calls(text, ["update_plan", "file_tools"])
    assert len(calls) == 1 and calls[0]["function"]["name"] == "update_plan"


def test_parse_bare_json_with_nested_arguments_and_prose():
    """裸 JSON + 嵌套 arguments：旧正则按第一个 } 截断，调用会被丢掉。"""
    text = ('好的，我来读取：{"name": "file_tools", "arguments": '
            '{"op": "write", "path": "a.txt", "meta": {"deep": true}}} 请稍等')
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "file_tools"
    assert '"deep": true' in calls[0]["function"]["arguments"]


def test_parse_openai_message_shapes():
    text = ('{"tool_calls": [{"function": {"name": "file_tools", '
            '"arguments": {"op": "list"}}}]}')
    calls = parse_tool_calls(text)
    assert len(calls) == 1 and calls[0]["function"]["name"] == "file_tools"

    text2 = '{"function": "file_tools", "parameters": {"op": "list"}}'
    calls2 = parse_tool_calls(text2)
    assert len(calls2) == 1 and '"op": "list"' in calls2[0]["function"]["arguments"]


def test_bare_json_example_data_is_not_a_tool_call():
    """普通答案里的示例 JSON（没有 arguments）不得被误判成工具调用。"""
    assert parse_tool_calls('表单字段 {"name": "Alice", "age": 30}') == []


def test_no_tool_event_flags_unparsed_tool_markup():
    from agent.kernel import LLMResponse, KernelLoop

    loop = KernelLoop.__new__(KernelLoop)
    loop._tool_parse_retry = 0
    loop._empty_retry = 0
    loop.log = lambda *_a, **_k: None

    resp = LLMResponse(
        content="", reasoning="<tool_call>{bad json}</tool_call>",
        tool_calls=[], raw_content='<tool_call>{"name": "file_tools"}</tool_call>',
    )
    assert loop._no_tool_event(resp) == "retry_tool"
    assert loop._no_tool_event(resp) == "retry_tool"
    # 超过上限后给可见错误，而不是继续按空回复重试
    assert loop._no_tool_event(resp) == "done_tool_parse"


def test_no_tool_event_keeps_long_prose_with_tool_example():
    """有实质正文（讲解工具调用写法）时不重试，避免打断正常回答。"""
    from agent.kernel import LLMResponse, KernelLoop

    loop = KernelLoop.__new__(KernelLoop)
    loop._tool_parse_retry = 0
    loop._empty_retry = 0
    loop.log = lambda *_a, **_k: None

    resp = LLMResponse(
        content=("工具调用的标准写法是：外层 <tool_call> 标签，内部放 "
                 '{"name": "file_tools", "arguments": {"op": "read"}}。'
                 "参数里的换行必须转义，否则整段 JSON 解析会失败。" * 2),
        reasoning="", tool_calls=[], raw_content="")
    assert loop._no_tool_event(resp) == "done"
