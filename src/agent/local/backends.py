"""本地模型后端标识的单一来源（归一化 + 合法性）。

背景：配置里的 ``*_backend`` 字段可能被**手工编辑**、被旧版本写入、或被外部
工具（LM Studio 等）写成 UI 显示名——例如 ``mlx-lm``（配置窗口下拉的标签）而
不是代码认的 ``mlx``。旧行为：``LocalLLMClient`` 直接抛
``未知 LLM 后端: 'mlx-lm'`` → 主模型加载失败、每句对话都报错（用户侧表现为
「莫名抽风」）。现在统一在这里归一化，config / llm / factory 共用一张表。
"""
from __future__ import annotations

from typing import Optional

# 合法后端（LocalLLMClient._ensure_model 的分支值）
VALID_BACKENDS = ("mlx", "gguf", "onnx")

# 别名 → 规范值。键统一小写、去空格/连字符差异。
_ALIASES = {
    # MLX
    "mlx": "mlx",
    "mlxlm": "mlx", "mlx-lm": "mlx", "mlx_lm": "mlx", "mlx-lm(4bit)": "mlx",
    "mlxvlm": "mlx", "mlx-vlm": "mlx", "mlx_vlm": "mlx",
    # GGUF（llama.cpp）
    "gguf": "gguf",
    "llamacpp": "gguf", "llama_cpp": "gguf", "llama.cpp": "gguf",
    "llama-cpp": "gguf", "llama-cpp-python": "gguf",
    # ONNX
    "onnx": "onnx",
    "onnxruntime": "onnx", "onnx-runtime": "onnx",
}


def normalize_backend(value: Optional[str], default: str = "mlx") -> Optional[str]:
    """把任意写法的后端标识归一化为 ``mlx`` / ``gguf`` / ``onnx``。

    空值 → ``default``；无法识别 → 返回 None（调用方负责报错，错误信息里
    带上合法值列表）。
    """
    if value is None:
        return default
    key = str(value).strip().lower().replace(" ", "")
    if not key:
        return default
    if key in _ALIASES:
        return _ALIASES[key]
    # 再试一次去掉连字符/下划线的形式（mlx-lm → mlxlm）
    squashed = key.replace("-", "").replace("_", "").replace(".", "")
    return _ALIASES.get(squashed)
