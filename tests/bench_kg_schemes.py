"""对比测试：llm_raw（自实现行解析） vs llm_pydantic（JSON+pydantic）抽取方案。

用法（需本地 MLX 模型，耗时数分钟）：
    python tests/bench_kg_schemes.py [模型路径]

两个方案用同一 client、同一文本、同一 token 预算（4096，生产配置），
各自跑一次（含各自的补漏轮逻辑），对比：
  * 实体数 / 关系数 / 抽取质量（字段完整度、描述长度、端点一致性）
  * 端到端耗时
  * 结果合法性（解析器是否成功、是否触发补漏）
"""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from agent.local.llm import LocalLLMClient
from agent.rag.kg.extractor import create_extractor

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


def assess(entities, relations) -> dict:
    """评估抽取质量。"""
    names = {e.name for e in entities}
    endpoints_ok = all(
        r.src in names and r.dst in names for r in relations)
    desc_lens = [len(e.description or "") for e in entities]
    return {
        "entities": len(entities),
        "relations": len(relations),
        "endpoints_ok": endpoints_ok,
        "avg_desc": round(sum(desc_lens) / len(desc_lens), 1) if desc_lens else 0,
        "empty_names": sum(1 for e in entities if not e.name.strip()),
    }


def main() -> None:
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    print(f"模型: {model_path}")
    client = LocalLLMClient(path=model_path, backend="mlx", max_tokens=4096,
                            temperature=0.1, enable_thinking=False)
    schemes = [
        ("llm_raw", create_extractor(
            "llm_raw", client, max_entities=20, max_relations=50,
            max_desc_chars=100, glean=True)),
        ("llm_pydantic", create_extractor(
            "llm_pydantic", client, max_entities=20, max_relations=50,
            max_desc_chars=100, glean=True)),
    ]
    rows = []
    for name, text in SAMPLES:
        print(f"\n{'='*60}\n样本【{name}】")
        for scheme, extractor in schemes:
            t0 = time.perf_counter()
            try:
                entities, relations = extractor.extract_entities_relations(text)
                elapsed = time.perf_counter() - t0
                q = assess(entities, relations)
                rows.append((name, scheme, elapsed, q))
                print(f"  {scheme:<14} {elapsed:>6.1f}s | "
                      f"实体={q['entities']} 关系={q['relations']} "
                      f"端点一致={q['endpoints_ok']} 平均描述={q['avg_desc']}字")
                for e in entities[:5]:
                    print(f"      · {e.name} [{e.type}] {e.description[:50]}")
            except Exception as e:
                print(f"  {scheme:<14} 异常: {e}")

    print(f"\n{'='*60}\n=== 汇总 ===")
    print(f"{'样本':<6}{'方案':<14}{'耗时':>8}{'实体':>6}{'关系':>6}"
          f"{'端点一致':>10}{'平均描述':>10}")
    for name, scheme, elapsed, q in rows:
        print(f"{name:<6}{scheme:<14}{elapsed:>7.1f}s{q['entities']:>6}"
              f"{q['relations']:>6}{str(q['endpoints_ok']):>10}"
              f"{q['avg_desc']:>9}字")

    # 速度对比（按样本名去重）
    print(f"\n=== 速度对比（llm_pydantic / llm_raw）===")
    seen = set()
    for name, _, _, _ in rows:
        if name in seen:
            continue
        seen.add(name)
        raw = next((r for r in rows if r[0] == name and r[1] == "llm_raw"), None)
        pyd = next((r for r in rows if r[0] == name and r[1] == "llm_pydantic"), None)
        if raw and pyd and raw[2] > 0:
            print(f"  {name}: {pyd[2]/raw[2]:.2f}x "
                  f"(raw {raw[2]:.1f}s → pydantic {pyd[2]:.1f}s)")
    client.close()


if __name__ == "__main__":
    main()
