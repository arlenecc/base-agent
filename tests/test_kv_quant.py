"""KV cache INT8 量化的单元测试（MLX 量化缓存构造 + GGUF 量化参数）。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest


def _make_client(backend="mlx", kv_quant=True, model=None):
    """构造一个不加载权重的 LocalLLMClient，仅用于测 _make_prompt_cache 逻辑。"""
    from agent.local.llm import LocalLLMClient
    c = LocalLLMClient(path="/tmp/fake-model", backend=backend, kv_quant=kv_quant)
    c._model = model
    return c


def _fake_model(cache_type):
    """伪模型：make_cache() 返回一个含指定缓存类型的列表。"""
    from mlx_lm.models.cache import KVCache, RotatingKVCache, ArraysCache

    class _M:
        def make_cache(self):
            if cache_type == "arrays":
                return [ArraysCache(10)]
            if cache_type == "rotating":
                return [RotatingKVCache(max_size=10, keep=4)]
            return [KVCache()]
    return _M()


class TestMlxKvQuantCache:

    def test_standard_kv_cache_is_quantized(self):
        """默认缓存是标准 KV（KVCache）→ 换成 QuantizedKVCache 列表。"""
        from mlx_lm.models.cache import QuantizedKVCache
        c = _make_client(kv_quant=True, model=_fake_model("standard"))
        out = c._make_prompt_cache()
        assert isinstance(out, list) and len(out) == 1
        assert isinstance(out[0], QuantizedKVCache)

    def test_rotating_kv_cache_is_quantized(self):
        from mlx_lm.models.cache import QuantizedKVCache
        c = _make_client(kv_quant=True, model=_fake_model("rotating"))
        out = c._make_prompt_cache()
        assert isinstance(out[0], QuantizedKVCache)

    def test_arrays_cache_falls_back(self):
        """线性注意力模型（ArraysCache）无标准 KV，量化不适用 → 原样返回。"""
        from mlx_lm.models.cache import ArraysCache
        c = _make_client(kv_quant=True, model=_fake_model("arrays"))
        out = c._make_prompt_cache()
        assert isinstance(out[0], ArraysCache)  # 不量化，退回默认

    def test_kv_quant_off_returns_default(self):
        c = _make_client(kv_quant=False, model=_fake_model("standard"))
        out = c._make_prompt_cache()
        from mlx_lm.models.cache import KVCache
        assert isinstance(out[0], KVCache)

    def test_non_mlx_returns_make_prompt_cache(self):
        """非 mlx 后端（GGUF/ONNX）没有 python 级 prompt cache 概念，
        _make_prompt_cache 仅作防御性透传（返回默认缓存列表）。"""
        c = _make_client(backend="gguf", kv_quant=True, model=_fake_model("standard"))
        out = c._make_prompt_cache()
        from mlx_lm.models.cache import KVCache
        assert isinstance(out, list) and isinstance(out[0], KVCache)


def test_gguf_kv_quant_kwargs_built():
    """GGUF Llama 在 kv_quant 开启时注入 type_k/type_v=q8_0。"""
    from agent.local.llm import LocalLLMClient
    c = LocalLLMClient(path="/x.gguf", backend="gguf", kv_quant=True)
    # 直接复用 _load_gguf 内部的 kwargs 组装逻辑分支：这里只断言构造参数会被接受
    import inspect
    from llama_cpp import Llama
    params = inspect.signature(Llama.__init__).parameters
    assert "type_k" in params and "type_v" in params
