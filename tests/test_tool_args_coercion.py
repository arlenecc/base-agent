"""工具参数类型强转 + 约束解码重试钩子测试。

重点：正常调用零影响（不影响普通对话），仅在参数错误时触发增强。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from agent.config import AgentConfig
from agent.tools.base import ToolRegistry, _coerce_args_by_schema
from agent.tools.file_ops import FileListTool, FileSearchTool


class _NoopCallbacks:
    def is_cancelled(self):
        return False

    def __getattr__(self, n):
        return lambda *a, **k: None


def test_coerce_string_to_bool_and_int():
    """字符串 "false"/"50" 按 schema 转成 bool/int。"""
    fixed, log = _coerce_args_by_schema(
        FileListTool(), {"path": ".", "recursive": "false"})
    assert fixed["recursive"] is False
    assert "recursive" in log

    fixed2, log2 = _coerce_args_by_schema(
        FileSearchTool(),
        {"query": "TODO", "case_sensitive": "false", "max_results": "50"})
    assert fixed2["case_sensitive"] is False
    assert fixed2["max_results"] == 50
    assert isinstance(fixed2["max_results"], int)


def test_coerce_ignores_invalid_and_string_fields():
    """无法推断的值保持原样；string 字段不做反向转换。"""
    fixed, log = _coerce_args_by_schema(
        FileListTool(), {"path": ".", "recursive": "maybe"})
    assert fixed["recursive"] == "maybe"
    assert not log

    fixed2, _ = _coerce_args_by_schema(
        FileSearchTool(), {"query": "TODO", "max_results": "abc"})
    assert fixed2["max_results"] == "abc"


def test_retry_hook_only_on_argument_error():
    """正常调用不触发钩子；参数错误时才触发。"""
    cfg = AgentConfig()
    cfg.workspace = "/tmp/test_tool_args_ws"
    reg = ToolRegistry(cfg, _NoopCallbacks())
    calls = []

    def fake(name, args, error, schema):
        calls.append(name)
        return None

    reg.set_constrained_retry(fake)
    reg.execute("file_list", {"path": ".", "recursive": False})
    assert not calls                      # 正常调用零触发
    reg.execute("start_long_term_update", {"reason": "x"})  # 缺必填参数
    assert calls == ["start_long_term_update"]
    reg.shutdown()


def test_retry_hook_can_fix_arguments():
    """钩子返回修正参数时，工具用修正参数重新执行（而非沿用原错误）。"""
    cfg = AgentConfig()
    cfg.workspace = "/tmp/test_tool_args_ws"
    reg = ToolRegistry(cfg, _NoopCallbacks())
    called = []

    def fix(name, args, error, schema):
        called.append(args)
        return {"focus": "fixed by constrained decode"}

    reg.set_constrained_retry(fix)
    # 传一个非法参数触发签名预检失败；钩子返回合法的 focus 参数
    result = reg.execute("start_long_term_update", {"bad_param": "x"})
    assert called == [{"bad_param": "x"}]       # 钩子收到了原始错误参数
    assert "Invalid arguments" not in (result.error or "")  # 未回退到原错误
    reg.shutdown()
