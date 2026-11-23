"""评估：工具调用参数输出的准确性 & 约束解码的可行性与效果。

用法（需本地 MLX 模型）：
    python tests/bench_tool_constrained.py [模型路径]

评估内容：
  A. 基线：当前自由生成下，模型输出的工具调用参数是否合法（JSON 可解析、
     工具名正确、必填参数齐全）——用项目真实 parse_tool_calls + 签名预检。
  B. 约束可行性：Outlines 能否用「自由文本 + 仅 tool_call 内 JSON」的分段
     语法约束，做到只约束参数、不污染普通对话（不影响普通回复文本）。
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from agent.local.llm import LocalLLMClient, parse_tool_calls
from agent.tools.base import ToolRegistry
from agent.config import AgentConfig

DEFAULT_MODEL = os.path.join(
    ROOT, "models", "qwen3.5-4b-agentic-coder-v4-4bit-mlx")

# 典型工具调用场景（用户指令 → 期望调用的工具）
CASES = [
    ("读文件", "帮我读一下 memory/global_mem.txt 的内容", "file_tools"),
    ("列目录", "列出工作区根目录下有哪些文件", "file_tools"),
    ("检索知识库", "在知识库里搜一下机器学习的相关内容", "rag_tools"),
    ("更新计划", "帮我制定一个三步计划：分析数据、建模、写报告", "update_plan"),
]

TOOL_SCHEMA_HINT = """可用工具（用 <tool_call>{"name":工具名,"arguments":{参数}}</tool_call> 调用）：
- file_tools: op(read/write/list/...), path, content...
- rag_tools: op(search/outline/...), query, source...
- update_plan: op(create/patch/...), goal, steps...
"""


def assess_call(text: str, expected: str, schemas: list) -> dict:
    """用项目解析器评估工具调用合法性。"""
    calls = parse_tool_calls(text, [s["function"]["name"] for s in schemas])
    if not calls:
        return {"parsed": False, "name_ok": False, "json_ok": False,
                "has_required": False, "calls": 0}
    c = calls[0]
    name = c["function"]["name"]
    args = c["function"]["arguments"]
    try:
        parsed = json.loads(args) if isinstance(args, str) else args
        json_ok = isinstance(parsed, dict)
    except Exception:
        json_ok = False
        parsed = {}
    # 必填参数：从 schema 里找 required
    schema = next((s for s in schemas if s["function"]["name"] == name), None)
    required = (schema or {}).get("function", {}).get(
        "parameters", {}).get("required", [])
    has_required = all(k in parsed for k in required)
    return {"parsed": True, "name_ok": name == expected,
            "json_ok": json_ok, "has_required": has_required,
            "calls": len(calls), "name": name, "args": str(args)[:150]}


def main() -> None:
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    print(f"模型: {model_path}")
    client = LocalLLMClient(path=model_path, backend="mlx", max_tokens=512,
                            temperature=0.0, enable_thinking=False)

    # 真实工具 schema（只取几个用于评估）
    cfg = AgentConfig()
    cfg.workspace = "/tmp/bench_ws"
    class _Noop:
        def is_cancelled(self): return False
        def __getattr__(self, n): return (lambda *a, **k: None)
    reg = ToolRegistry(cfg, _Noop())
    all_schemas = reg.schemas()
    keep = {"file_tools", "rag_tools", "update_plan"}
    schemas = [s for s in all_schemas if s["function"]["name"] in keep]
    print(f"工具 schema 数: {len(schemas)}")
    reg.shutdown()

    print("\n=== A. 基线：自由生成的工具参数准确性 ===")
    print(f"{'场景':<10}{'耗时':>8}{'解析':>6}{'名对':>6}{'JSON':>6}{'必填':>6}")
    results = []
    for name, prompt, expected in CASES:
        messages = [
            {"role": "system", "content": TOOL_SCHEMA_HINT},
            {"role": "user", "content": prompt},
        ]
        # 用自由生成（带 tools schema，走真实 chat 模板）
        out = []
        done_calls = None
        t0 = time.perf_counter()
        try:
            for ev in client.chat_stream(
                    messages, tools=schemas, temperature=0.0,
                    enable_thinking=False, kv_group="aux", max_tokens=256):
                if ev.type == "content":
                    out.append(ev.content)
                elif ev.type == "done":
                    done_calls = getattr(ev, "tool_calls", None)
                    break
        except Exception as e:
            print(f"  {name} 生成异常: {e}")
            continue
        elapsed = time.perf_counter() - t0
        raw = "".join(out)
        # 工具块被 _ToolCallSplitter 从 content 抑制，done 事件带已解析的 tool_calls
        if done_calls:
            c = done_calls[0]
            name = c.get("function", {}).get("name", "")
            args_raw = c.get("function", {}).get("arguments", "{}")
            try:
                parsed_args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                json_ok = isinstance(parsed_args, dict)
            except Exception:
                parsed_args, json_ok = {}, False
            schema = next((s for s in schemas if s["function"]["name"] == name), None)
            required = (schema or {}).get("function", {}).get(
                "parameters", {}).get("required", [])
            a = {"parsed": True, "name_ok": name == expected,
                 "json_ok": json_ok,
                 "has_required": all(k in parsed_args for k in required),
                 "calls": len(done_calls), "name": name,
                 "args": str(args_raw)[:150]}
        else:
            a = assess_call(raw, expected, schemas)
        results.append((name, elapsed, a))
        print(f"{name:<10}{elapsed:>7.1f}s{str(a['parsed']):>6}"
              f"{str(a['name_ok']):>6}{str(a['json_ok']):>6}"
              f"{str(a['has_required']):>6}")
        print(f"    原文: {raw[:160]!r}")

    ok = sum(1 for _, _, a in results if a["parsed"] and a["name_ok"]
             and a["json_ok"] and a["has_required"])
    print(f"\n基线合格率: {ok}/{len(results)}")

    print("\n=== B. 约束可行性：分段语法（自由前缀 + tool_call 内 JSON）===")
    # 验证 Outlines CFG 能否表达「任意文本 + 约束 JSON 块」
    try:
        import outlines
        cfg_def = r'''
        root ::= prefix toolcall suffix?
        prefix ::= [^{]*
        toolcall ::= "<tool_call>{" json "}</tool_call>"
        json ::= [^{}]+
        suffix ::= [^{]*
        '''
        g = outlines.cfg(cfg_def)
        print("  分段 CFG 构建: OK")
        # 用真实 client 试跑
        out = client.chat_constrained(
            [{"role": "user", "content":
              "读一下 a.txt。用 <tool_call>{\"name\":\"file_tools\",\"arguments\":{\"op\":\"read\",\"path\":\"a.txt\"}}</tool_call> 格式"}],
            schema=g, max_tokens=128)
        print(f"  约束输出: {out[:200]!r}")
    except Exception as e:
        print(f"  分段 CFG 失败: {type(e).__name__}: {e}")

    client.close()


if __name__ == "__main__":
    main()
