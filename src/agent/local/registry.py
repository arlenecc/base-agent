"""本地模型注册表：把「用途 + 用户配置」解析成具体的模型目录/文件。

仓库在 ``models/`` 下已自带三套 MLX 权重（见 .gitignore，权重不入库，需自行
放置）：
    embedding  →  models/jina-embeddings-v5-text-nano-mlx-int4
    reranker   →  models/jina-reranker-v2-base-multilingual-onnx
    llm        →  models/qwen3.5-4b-agentic-coder-v4-4bit-mlx

``resolve_model_path(kind, value)`` 的解析规则：
    value 为空        → 使用上述默认目录
    value 是已存在路径 → 原样返回
    value 是目录名     → 视为 ``models/<value>``
    其它              → 原样返回（交给后端报错，信息更明确）
"""
from __future__ import annotations

import os
from typing import Dict, Optional

# 应用内自带权重的默认相对路径（相对项目根）
_DEFAULTS: Dict[str, str] = {
    "embedding": "models/jina-embeddings-v5-text-nano-mlx-int4",
    "reranker": "models/jina-reranker-v2-base-multilingual-onnx",
    "llm": "models/qwen3.5-4b-agentic-coder-v4-4bit-mlx",
    "aux": "models/Qwen3.5-2B-MLX-4bit",
    # vlm 无自带默认，必须显式指定
}


def _project_root() -> str:
    """项目根目录（base-agent）：含 ``pyproject.toml`` 的那一级。

    从本文件（src/agent/local/registry.py）向上逐级找 ``pyproject.toml``；
    这是默认模型路径（``<root>/models/...``）的基准。早期实现按固定层级
    「上溯四级」，把根算到了 git 仓库的上一级（base-agent 的父目录），导致
    空路径默认值解析到不存在的 ``<父>/models/...``——默认模型静默失效。
    改为按标志物查找，布局变动也不怕；找不到时回退到上溯三级。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    cur = here
    for _ in range(8):
        if os.path.isfile(os.path.join(cur, "pyproject.toml")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    # 回退：src/agent/local → base-agent（三级）
    return os.path.dirname(os.path.dirname(os.path.dirname(here)))


def resolve_model_path(kind: str, value: str = "", project_root: Optional[str] = None) -> str:
    """解析某用途的模型路径；不存在时返回默认（可能仍不存在 → 后端给出明确报错）。

    搜索顺序（价值的是「用户把权重放在哪」）：
      1. value 本身是已存在路径（绝对/相对）→ 原样返回；
      2. ``<project_root>/models/<value>``（仓库自带权重的位置）；
      3. ``<project_root>/<value>``；
      4. ``~/models/<value>``（用户级模型目录——权重搬出项目后最常放的地方）。
    都找不到 → 原样返回 value，让后端报「目录不存在」（信息更明确）。
    """
    root = project_root or _project_root()

    def _exists(p: str) -> bool:
        return os.path.isdir(p) or os.path.isfile(p)

    if value:
        if _exists(value):
            return value
        under = os.path.join(root, "models", value)
        if _exists(under):
            return under
        # 当作相对项目根的路径再试一次
        rel = os.path.join(root, value)
        if _exists(rel):
            return rel
        # 用户级模型目录：权重搬出项目（<project>/models 不存在）后的落点
        home_models = os.path.join(os.path.expanduser("~"), "models", value)
        if _exists(home_models):
            return home_models
        # 未找到：返回原始值，让后端报「目录不存在」
        return value

    default = _DEFAULTS.get(kind, "")
    if default:
        return os.path.join(root, default)
    return default


def default_model_path(kind: str, project_root: Optional[str] = None) -> str:
    return resolve_model_path(kind, "", project_root)


# 旋转式（RoPE，config 含 n_positions）模型的外推上下文上限
_ROPE_CONTEXT = 8192


def resolve_model_max_length(model_path: str, default: int = 8192) -> int:
    """按模型位置编码类型夹紧 embedding / reranker 的最大序列长度。

    规则：
      - 旋转式（RoPE，config 含 ``n_positions``）→ 8192。这类模型靠 RoPE
        外推，``max_position_embeddings`` 往往是训练时的较小值（nomic 写的是
        2048），但 ``n_positions`` 才是真实可用上下文；盲信前者会把 nomic
        钳到 2048，损失 8K 长文档能力。
      - 绝对位置嵌入（如 xlm-roberta 的 bge-m3）→ 取 ``max_position_embeddings``
        （bge-m3 为 8194），这是该架构的硬上限，超出会索引越界。
      - 都读不到 → 回退到 default（8192）。
    """
    import json
    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(cfg_path):
        return default
    try:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return default
    if cfg.get("n_positions"):
        return _ROPE_CONTEXT
    mpe = cfg.get("max_position_embeddings")
    if mpe:
        return int(mpe)
    return default

