"""嵌入模型「量化版 + 强制 CPU-only」的回归测试。

锁定两个属性（用户明确要求确认的两点）：

1. **量化版**：默认模型是 ``nomic-ai/nomic-embed-text-v1.5-Q``，它在 FastEmbed
   注册表里映射到 ``onnx/model_quantized.onnx``（int8，~137MB / 768 维），
   而不是 fp32 的 ``onnx/model.onnx``（~548MB）。config 还会把显式写成非量化
   名的配置迁移到 ``-Q``。
2. **强制 CPU-only**：构造 TextEmbedding / TextCrossEncoder 时显式传
   ``providers=["CPUExecutionProvider"]``。此前依赖 FastEmbed 的默认回退
   （``cuda=Device.AUTO`` 只在环境没有 CUDAExecutionProvider 时才落到 CPU），
   装了 onnxruntime-gpu 的机器上嵌入会静默切到 CUDA——与 OCR/解析全程 CPU
   的设计不符，也让同一向量库在不同机器上数值行为漂移。
"""
import inspect
import json
import sys

import pytest

sys.path.insert(0, "src")

from agent.rag import vector_store as vs
from agent.rag.vector_store import (
    DEFAULT_EMBEDDING_MODEL,
    FastEmbedEmbeddingFunction,
    _cpu_only_kwargs,
)


QUANTIZED_SUFFIX = "-Q"


class TestQuantizedModel:
    def test_default_model_is_quantized(self):
        assert DEFAULT_EMBEDDING_MODEL == "nomic-ai/nomic-embed-text-v1.5-Q"
        assert DEFAULT_EMBEDDING_MODEL.endswith(QUANTIZED_SUFFIX)

    def test_quantized_maps_to_quantized_onnx_file(self):
        """FastEmbed 注册表里 -Q 必须指向 model_quantized.onnx。"""
        from fastembed import TextEmbedding

        table = {m["model"]: m for m in TextEmbedding.list_supported_models()}
        entry = table["nomic-ai/nomic-embed-text-v1.5-Q"]
        assert entry["model_file"] == "onnx/model_quantized.onnx"
        assert entry["dim"] == 768
        # int8 量化的体积约为 fp32 的 1/4（0.13GB vs 0.52GB）
        fp32 = table["nomic-ai/nomic-embed-text-v1.5"]
        assert entry["size_in_GB"] < fp32["size_in_GB"] / 2

    def test_config_migrates_non_quantized(self, tmp_path):
        """用户配置里写非量化名也要被迁移到 -Q（体积 4 倍差）。"""
        from agent.config import AgentConfig

        p = tmp_path / "c.json"
        p.write_text(json.dumps({
            "base_url": "http://x",
            "api_key": "",
            "model": "m",
            "rag_embedding_model": "nomic-ai/nomic-embed-text-v1.5",
        }), encoding="utf-8")
        cfg = AgentConfig.load(str(p))
        assert cfg.rag_embedding_model == "nomic-ai/nomic-embed-text-v1.5-Q"


class TestCpuOnlyForced:
    def test_cpu_only_kwargs_for_supporting_class(self):
        """支持 providers 的类必须拿到 CPU-only 参数。"""
        from fastembed import TextEmbedding

        kwargs = _cpu_only_kwargs(TextEmbedding)
        assert kwargs == {"providers": ["CPUExecutionProvider"]}

    def test_cpu_only_kwargs_for_reranker(self):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        kwargs = _cpu_only_kwargs(TextCrossEncoder)
        assert kwargs == {"providers": ["CPUExecutionProvider"]}

    def test_cpu_only_kwargs_empty_for_unsupported_class(self):
        """签名里没有 providers 的类返回 {}（旧版 fastembed 不会因多余参数崩）。"""

        class _NoProviders:
            def __init__(self, model_name):
                pass

        assert _cpu_only_kwargs(_NoProviders) == {}

    def test_embedding_function_passes_providers(self, monkeypatch):
        """构造 TextEmbedding 时确实传了 providers=["CPUExecutionProvider"]。"""
        captured = {}

        class _FakeTextEmbedding:
            # 签名必须显式声明 providers——_cpu_only_kwargs 靠签名探测决策，
            # 只写 **kwargs 的假类会被判定为「不支持」而正确地不传。
            def __init__(self, model_name, cache_dir=None, threads=None,
                         providers=None, cuda=None, **kwargs):
                captured.update({
                    "model_name": model_name,
                    "cache_dir": cache_dir,
                    "threads": threads,
                    "providers": providers,
                    "cuda": cuda,
                })

            def embed_query(self, text):
                return [0.0] * 768

        monkeypatch.setattr(vs, "TextEmbedding", _FakeTextEmbedding,
                            raising=False)
        monkeypatch.setattr(vs, "_MODEL_LOAD_LOCK",
                            __import__("threading").Lock())
        # 直接驱动 _ensure_model：伪造 import 来源
        import builtins
        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "fastembed":
                mod = type(sys)("fastembed")
                mod.TextEmbedding = _FakeTextEmbedding
                return mod
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        ef = FastEmbedEmbeddingFunction(model_name=DEFAULT_EMBEDDING_MODEL)
        ef._ensure_model()

        assert captured.get("providers") == ["CPUExecutionProvider"]
        assert captured.get("model_name") == DEFAULT_EMBEDDING_MODEL
        # 不能把 cuda 留在 AUTO：那正是 GPU 机器上静默漂移的根源
        assert "cuda" not in captured or captured.get("cuda") != "auto"
