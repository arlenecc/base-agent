"""本地模型加载层：应用内直接加载并运行模型，不走网络 API。

三种后端（config 里可切换）：

    mlx   — Apple Silicon 原生。LLM 走 ``mlx-lm``，VLM 走 ``mlx-vlm``，
            encoder 类模型（embedding / reranker）走本包内的 MLX 实现
            （``mlx_nomic``）或 ``mlx-embeddings``。
    onnx  — ``onnxruntime`` + ``transformers`` + ``numpy``，跨平台。
    gguf  — ``llama-cpp-python``，跨平台（Metal / CUDA / Vulkan）。

对外只暴露三个工厂函数（``factory``）：

    build_llm_client(config)            → 与 ``LLMClient`` 同接口（chat_stream）
    build_embedding_function(config)    → 与 FastEmbedEmbeddingFunction 同接口
    build_reranker(config)              → 与 fastembed TextCrossEncoder 同接口

接口刻意与现有实现**鸭子类型兼容**：调用方（kernel / VectorStore / Agent）
不需要任何改动就能在主模型与本地模型之间切换。
"""
from __future__ import annotations

from .registry import resolve_model_path, default_model_path
from .factory import (
    build_llm_client,
    build_aux_llm_client,
    build_embedding_function,
    build_reranker,
    build_vlm,
    configure_local_models,
)
from .llm import LocalLLMClient
from .vlm import LocalVLMClient
from .embedding import LocalEmbeddingFunction
from .reranker import LocalReranker

__all__ = [
    "resolve_model_path",
    "default_model_path",
    "build_llm_client",
    "build_aux_llm_client",
    "build_embedding_function",
    "build_reranker",
    "build_vlm",
    "configure_local_models",
    "LocalLLMClient",
    "LocalVLMClient",
    "LocalEmbeddingFunction",
    "LocalReranker",
]
