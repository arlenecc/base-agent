"""对比测试：知识图谱抽取用 Outlines 约束解码 vs 自由生成。

用法（需本地 MLX 模型，耗时数分钟）：
    python tests/bench_outlines_kg.py [模型路径]

对比维度：
  * 格式合法率（输出是否为合法 JSON、是否通过 pydantic 校验）
  * 抽取效果（实体数 / 关系数 / 是否非空）
  * 速度（端到端秒数、每秒token 近似）
同一 client、同一 prompt、同一文本，各跑一次；输出汇总表。

注意：这是基准脚本（非 pytest 测试），需手动运行并肉眼对比结果。
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from agent.llm_client import StreamEvent
from agent.local.llm import LocalLLMClient
from agent.rag.kg import prompts
from agent.rag.kg.extractor import _build_pydantic_models

DEFAULT_MODEL = os.path.join(
    ROOT, "models", "qwen3.5-4b-agentic-coder-v4-4bit-mlx")

SAMPLES = [
    ("简单", "小明在北京工作，他喜欢Python编程。"),
    ("中等", (
        "华为公司由任正非创立于1987年，总部位于深圳。"
        "华为的主要业务包括通信设备和智能手机，其竞争对手有小米公司和苹果公司。"
        "2023年华为发布了Mate 60手机，搭载了麒麟9000S芯片。")),
    ("复杂", (
        "张三是清华大学计算机系的教授，他研究方向是自然语言处理。"
        "2022年张三和李四大共同创办了智源科技公司，位于北京中关村。"
        "智源科技主要开发大语言模型，其产品包括悟道系列。"
        "李四曾在谷歌工作，负责Transformer相关研究。"
        "清华大学与智源科技在2023年联合成立了人工智能联合实验室。")),
]


def build_messages(text: str) -> list:
    sys_p = prompts.with_limits(
        prompts.PYDANTIC_EXTRACT_SYSTEM, max_entities=20,
        max_relations=50, max_desc_chars=100)
    task = prompts.with_limits(
        prompts.PYDANTIC_EXTRACT_TASK, max_entities=20,
        max_relations=50, max_desc_chars=100).format(
            language="中文", section_context="", input_text=text)
    return [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": task},
    ]


def free_generate(client: LocalLLMClient, messages: list,
                  max_tokens: int = 2048) -> tuple:
    """现有路径：自由生成 → 事后解析。

    max_tokens 默认 2048（生产 KG 抽取用 4096），与约束解码同预算对比；
    脚本首轮曾用 512，发现自由生成因 markdown 包裹+空格冗余被截断。
    """
    out: list = []
    t0 = time.perf_counter()
    for ev in client.chat_stream(
            messages, tools=None, temperature=0.0,
            enable_thinking=False, kv_group="aux", max_tokens=max_tokens):
        if ev.type == "content":
            out.append(ev.content)
        elif ev.type == "done":
            break
    elapsed = time.perf_counter() - t0
    return "".join(out).strip(), elapsed


def constrained_generate(client: LocalLLMClient, messages: list, schema) -> tuple:
    """新路径：Outlines 约束解码。"""
    t0 = time.perf_counter()
    out = client.chat_constrained(messages, schema=schema, max_tokens=2048)
    elapsed = time.perf_counter() - t0
    return (out if isinstance(out, str) else json.dumps(
        out, ensure_ascii=False, default=str)), elapsed


def validate(raw: str, schema) -> tuple:
    """返回 (是否合法JSON, 是否通过pydantic, 实体数, 关系数)。

    用项目实际的 ``_parse_json_object``（能剥离 markdown 代码块等），与生产
    路径一致，避免基准脚本自己的粗糙解析低估自由生成的效果。
    """
    from agent.rag.kg.extractor import _parse_json_object
    data = _parse_json_object(raw)
    if not isinstance(data, dict):
        return False, False, 0, 0
    try:
        parsed = schema.model_validate(data)
        return True, True, len(parsed.entities), len(parsed.relations)
    except Exception:
        ents = len(data.get("entities", [])) if isinstance(data, dict) else 0
        rels = len(data.get("relations", [])) if isinstance(data, dict) else 0
        return True, False, ents, rels


def main() -> None:
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    print(f"模型: {model_path}")
    client = LocalLLMClient(path=model_path, backend="mlx", max_tokens=512,
                            temperature=0.0, enable_thinking=False)
    schema = _build_pydantic_models()
    rows = []
    for name, text in SAMPLES:
        msgs = build_messages(text)
        raw_free, t_free = free_generate(client, msgs)
        raw_con, t_con = constrained_generate(client, msgs, schema)
        v_free = validate(raw_free, schema)
        v_con = validate(raw_con, schema)
        rows.append((name, t_free, t_con, v_free, v_con))
        print(f"\n--- {name} ---")
        print(f"自由生成: {t_free:.1f}s | 合法={v_free[0]} 校验={v_free[1]} "
              f"实体={v_free[2]} 关系={v_free[3]}")
        print(f"约束解码: {t_con:.1f}s | 合法={v_con[0]} 校验={v_con[1]} "
              f"实体={v_con[2]} 关系={v_con[3]}")
        print(f"自由生成原文前200字: {raw_free[:200]!r}")
        print(f"约束解码原文前200字: {raw_con[:200]!r}")

    print("\n=== 汇总 ===")
    print(f"{'样本':<6}{'自由耗时':>10}{'约束耗时':>10}{'速度比':>8}"
          f"{'自由实体/关系':>16}{'约束实体/关系':>16}{'校验通过':>12}")
    for name, tf, tc, vf, vc in rows:
        ratio = tc / tf if tf > 0 else 0
        print(f"{name:<6}{tf:>9.1f}s{tc:>9.1f}s{ratio:>8.2f}"
              f"{f'{vf[2]}/{vf[3]}':>16}{f'{vc[2]}/{vc[3]}':>16}"
              f"{f'{vf[1]}→{vc[1]}':>12}")
    client.close()


if __name__ == "__main__":
    main()
