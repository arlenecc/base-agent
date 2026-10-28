"""本地模型工厂：把 ``AgentConfig`` 映射到具体的 LLM / Embedding / Reranker / VLM。

设计要点：
- 每个用途都是一个「provider」开关：``api``（默认，走原有网络实现）或
  ``local``（应用内直接加载本地权重，不走网络）。
- embedding / reranker 默认 ``local``（仓库自带 nomic / bge 的 MLX 权重），
  主 LLM 默认 ``api``（本地 qwen 是并行可选项）。
- 返回对象刻意与原有实现**鸭子类型兼容**：``LocalLLMClient`` ↔ ``LLMClient``
  （``chat_stream`` / ``close``），``LocalEmbeddingFunction`` ↔
  ``FastEmbedEmbeddingFunction``，``LocalReranker`` ↔ fastembed
  ``TextCrossEncoder``。上层（kernel / VectorStore / Agent）零改动切换。
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _provider(value: Any, default: str) -> str:
    v = (str(value or "").strip().lower())
    return v if v in ("api", "local") else default


def _backend(value: Any, default: str) -> str:
    return (str(value or "").strip().lower()) or default


# ---------------------------------------------------------------------------
# LLM（主 agent 引擎）
# ---------------------------------------------------------------------------

def build_llm_client(config) -> Any:
    """返回与 ``LLMClient`` 同接口的客户端（网络 API 或本地模型）。"""
    provider = _provider(getattr(config, "llm_provider", "api"), "api")
    if provider == "local":
        from ..local.llm import LocalLLMClient
        from ..local import registry
        path = registry.resolve_model_path("llm", getattr(config, "local_llm_model", ""))
        backend = _backend(getattr(config, "local_llm_backend", "mlx"), "mlx")
        logger.info("本地 LLM 引擎：%s（backend=%s）", path, backend)
        return LocalLLMClient(
            path=path, backend=backend,
            max_tokens=getattr(config, "max_tokens", 32768),
            temperature=getattr(config, "temperature", 0.7),
            top_p=getattr(config, "top_p", 0.95),
            min_p=getattr(config, "min_p", 0.05),
            top_k=getattr(config, "top_k", 20),
            repetition_penalty=getattr(config, "repetition_penalty", 1.0),
            enable_thinking=bool(getattr(config, "local_llm_enable_thinking", True)),
            num_draft_tokens=getattr(config, "local_llm_num_draft_tokens", 0),
            prefill_step_size=getattr(config, "local_llm_prefill_step_size", 0),
            # None（配置被写成 null）必须按默认 True 处理：bool(None)=False 会把
            # 跨轮 KV 复用静默关掉。LocalLLMClient 内部也会再归一化一次。
            kv_reuse=(True if getattr(config, "local_llm_kv_reuse", True) is None
                      else bool(getattr(config, "local_llm_kv_reuse", True))),
            # KV cache INT8 量化：主、辅助模型都启用（省显存，质量近乎无损）。
            kv_quant=(True if getattr(config, "local_llm_kv_quantize", True) is None
                      else bool(getattr(config, "local_llm_kv_quantize", True))),
            # gguf 视觉塔（mmproj）与 onnx 分词器路径：可选，留空走默认。
            mmproj=getattr(config, "local_llm_mmproj_path", "") or "",
            tokenizer_path=getattr(config, "local_llm_tokenizer_path", "") or "",
        )
    # 默认：网络 API
    from ..llm_client import LLMClient
    return LLMClient(
        base_url=getattr(config, "base_url", "https://api.openai.com/v1"),
        api_key=getattr(config, "api_key", ""),
        model=getattr(config, "model", "gpt-4o-mini"),
        timeout=getattr(config, "request_timeout", 120.0),
        max_tokens=getattr(config, "max_tokens", 32768),
        top_p=getattr(config, "top_p", 0.95),
        min_p=getattr(config, "min_p", 0.05),
        top_k=getattr(config, "top_k", 20),
        repetition_penalty=getattr(config, "repetition_penalty", 1.0),
    )


def build_aux_llm_client(config) -> Any:
    """辅助 LLM 客户端（本地专用，用于事实抽取等轻量子任务）。

    与 :func:`build_llm_client` 同接口，但读 ``aux_*`` 配置项、默认模型走
    ``models/Qwen3.5-2B-MLX-4bit``。辅助任务始终本地加载，无 OpenAI 模式。
    """
    from ..local.llm import LocalLLMClient
    from ..local import registry
    path = registry.resolve_model_path("aux", getattr(config, "aux_local_llm_model", ""))
    backend = _backend(getattr(config, "aux_local_llm_backend", "mlx"), "mlx")
    logger.info("辅助 LLM 引擎：%s（backend=%s）", path, backend)
    return LocalLLMClient(
        path=path, backend=backend,
        max_tokens=getattr(config, "aux_max_tokens", 32768),
        temperature=getattr(config, "aux_temperature", 0.7),
        top_p=getattr(config, "aux_top_p", 0.95),
        min_p=getattr(config, "aux_min_p", 0.05),
        top_k=getattr(config, "aux_top_k", 20),
        repetition_penalty=getattr(config, "aux_repetition_penalty", 1.0),
        # 兜底值必须与 AgentConfig 的字段默认值一致（辅助模型默认关闭思考）。
        enable_thinking=bool(getattr(config, "aux_local_llm_enable_thinking", False)),
        # 辅助模型 KV 复用：默认关闭（辅助 prompt 每轮都变，复用通常无意义），
        # 但作为可配项开放给用户按需开启（与主模型同款开关）。
        kv_reuse=(False if getattr(config, "aux_local_llm_kv_reuse", None) is None
                  else bool(getattr(config, "aux_local_llm_kv_reuse", True))),
        # 辅助模型 KV INT8 量化（用户需求：主/辅助都省显存），与主模型同款可配。
        kv_quant=(True if getattr(config, "aux_local_llm_kv_quantize", None) is None
                  else bool(getattr(config, "aux_local_llm_kv_quantize", True))),
        # gguf 视觉塔（mmproj）与 onnx 分词器路径：可选，留空走默认。
        mmproj=getattr(config, "aux_local_llm_mmproj_path", "") or "",
        tokenizer_path=getattr(config, "aux_local_llm_tokenizer_path", "") or "",
    )


# ---------------------------------------------------------------------------
# Embedding / Reranker（RAG 向量库）
# ---------------------------------------------------------------------------

def configure_local_models(config) -> None:
    """把 config 里的 provider 选择同步到 vector_store 的模块级开关。

    在 ``main_window`` 重建 agent / 初始化 RAG 引擎前调用一次即可，之后
    ``get_or_create_embedding_function`` / ``VectorStore._get_reranker`` 会按开关
    选择 fastembed 或本地模型。幂等。
    """
    from ..rag import vector_store as vs
    # MLX 默认 cache_limit 是统一内存的 ~75%：聊天/嵌入的空闲 Metal buffer
    # 会一路涨到上限才归还系统（16GB 机器上 ~12GB）。按配置压到指定上限，空闲即还。
    try:
        from ..rag.engine import configure_mlx_memory_limits
        _cache_mb = int(getattr(config, "mlx_cache_limit_mb", 2000))
        configure_mlx_memory_limits(_cache_mb * 1_000_000)
    except Exception:  # mlx 未装 / 非 macOS：跳过
        pass
    vs.set_embedding_provider(
        _provider(getattr(config, "embedding_provider", "local"), "local"),
        path=getattr(config, "local_embedding_model", ""),
        backend=_backend(getattr(config, "local_embedding_backend", "mlx"), "mlx"),
    )
    # Matryoshka 维度裁剪：开关（rag_matryoshka_enabled）+ 维度（rag_embedding_dims）。
    # 对所有支持 Matryoshka 的模型（nomic-embed-text 系列、jina-embeddings-v3 等）生效。
    vs.set_matryoshka_enabled(bool(getattr(config, "rag_matryoshka_enabled", True)))
    vs.set_matryoshka_dims(int(getattr(config, "rag_embedding_dims", 256)))
    vs.set_reranker_provider(
        _provider(getattr(config, "reranker_provider", "local"), "local"),
        path=getattr(config, "local_reranker_model", ""),
        backend=_backend(getattr(config, "local_reranker_backend", "mlx"), "mlx"),
    )


def build_embedding_function(config):
    """返回与 ``FastEmbedEmbeddingFunction`` 同接口的 embedding 函数。"""
    from ..rag import vector_store as vs
    configure_local_models(config)
    # 走 vector_store 的带锁单例路径（避免重复加载 ~GB 级权重）
    model_name = getattr(config, "rag_embedding_model", "nomic-ai/nomic-embed-text-v1.5-Q")
    return vs.get_or_create_embedding_function(model_name)


def build_reranker(config):
    """返回与 fastembed ``TextCrossEncoder`` 同接口（``.rerank(query, docs)``）的精排器。"""
    from ..rag import vector_store as vs
    configure_local_models(config)
    return vs.get_or_create_reranker()


# ---------------------------------------------------------------------------
# VLM（多模态，可选能力）
# ---------------------------------------------------------------------------

def build_vlm(config) -> Optional[Any]:
    """返回本地 VLM 客户端；未配置本地 VLM 时返回 None。"""
    provider = _provider(getattr(config, "vlm_provider", "api"), "api")
    if provider != "local":
        return None
    from ..local.vlm import LocalVLMClient
    from ..local import registry
    path = registry.resolve_model_path("vlm", getattr(config, "local_vlm_model", ""))
    backend = _backend(getattr(config, "local_vlm_backend", "mlx"), "mlx")
    return LocalVLMClient(path=path, backend=backend)
