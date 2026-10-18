"""嵌入模型与重排序模型「量化版 + 强制 CPU-only」的回归测试。

锁定两组属性（用户明确要求确认的两点，嵌入与 reranker 各一组）：

**嵌入（nomic-embed-text）**
1. 量化版：默认 ``nomic-ai/nomic-embed-text-v1.5-Q``，在 FastEmbed 注册表里
   映射到 ``onnx/model_quantized.onnx``（int8，~137MB / 768 维），而不是
   fp32 的 ``onnx/model.onnx``（~548MB）。config 还会把显式写成非量化名的
   配置迁移到 ``-Q``。
2. CPU-only：构造 TextEmbedding / TextCrossEncoder 时显式传
   ``providers=["CPUExecutionProvider"]``。此前依赖 FastEmbed 的默认回退
   （``cuda=Device.AUTO`` 只在环境没有 CUDAExecutionProvider 时才落到 CPU），
   装了 onnxruntime-gpu 的机器上嵌入会静默切到 CUDA——与 OCR/解析全程 CPU
   的设计不符，也让同一向量库在不同机器上数值行为漂移。

**重排序（bge-reranker-v2-m3）**
3. 量化版：默认 ``BAAI/bge-reranker-v2-m3-int8``（~570MB），而非 fp32 的
   ``BAAI/bge-reranker-v2-m3``（2.27GB）。四个默认值站点（config / engine /
   tools.base / ui.main_window）必须与 deps.RERANK_MODEL_ID、
   vector_store.DEFAULT_RERANK_MODEL 一致——曾出现 deps 预下载并探测 int8、
   运行时却按 fp32 名加载的分裂：预下载白下，首次检索再触发一次 2.27GB
   下载，常驻内存 4 倍。
4. CPU-only：同上，显式强制。
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


# ---------------------------------------------------------------------------
# 重排序模型：量化版 + CPU-only
# ---------------------------------------------------------------------------


RERANK_INT8 = "BAAI/bge-reranker-v2-m3-int8"
RERANK_FP32 = "BAAI/bge-reranker-v2-m3"


class TestRerankerQuantized:
    def test_default_is_int8_across_all_sites(self):
        """五个站点必须一致指向 int8（曾分裂：deps 探测 int8、运行时加载 fp32）。"""
        from agent.config import AgentConfig
        from agent.rag import deps
        from agent.rag.engine import RAGEngine
        from agent.rag.vector_store import DEFAULT_RERANK_MODEL
        import inspect

        cfg = AgentConfig(workspace="/tmp")
        assert cfg.rag_rerank_model == RERANK_INT8
        assert DEFAULT_RERANK_MODEL == RERANK_INT8
        assert deps.RERANK_MODEL_ID == RERANK_INT8
        # engine 的签名默认值
        sig = inspect.signature(RAGEngine.__init__)
        assert sig.parameters["rerank_model"].default == RERANK_INT8
        # UI / 工具层的 getattr 兜底值
        import agent.ui.main_window as mw
        import agent.tools.base as tb

        for src in (mw, tb):
            text = inspect.getsource(src)
            assert RERANK_FP32 not in text, (
                f"{src.__name__} 仍硬编码 fp32 全量 reranker 名（2.27GB）"
            )

    def test_int8_maps_to_quantized_onnx_file(self):
        """FastEmbed 注册表里 int8 必须指向 model_int8.onnx。"""
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        table = {m["model"]: m for m in TextCrossEncoder.list_supported_models()}
        int8 = table[RERANK_INT8]
        fp32 = table[RERANK_FP32]
        assert int8["model_file"] == "onnx/model_int8.onnx"
        assert fp32["model_file"] == "onnx/model.onnx"
        # int8 体积应不足 fp32 的 1/3（0.57GB vs 2.27GB）
        assert int8["size_in_GB"] < fp32["size_in_GB"] / 3

    def test_config_migrates_fp32_name(self, tmp_path):
        """旧配置持久化的 fp32 名必须迁到 int8，否则继续每次加载 2.27GB。"""
        from agent.config import AgentConfig

        p = tmp_path / "c.json"
        p.write_text(json.dumps({
            "base_url": "http://x", "api_key": "", "model": "m",
            "rag_rerank_model": RERANK_FP32,
        }), encoding="utf-8")
        assert AgentConfig.load(str(p)).rag_rerank_model == RERANK_INT8

    def test_config_keeps_custom_rerank_model(self, tmp_path):
        """用户显式选的其它 reranker 不被迁移覆盖。"""
        from agent.config import AgentConfig

        p = tmp_path / "c.json"
        p.write_text(json.dumps({
            "base_url": "http://x", "api_key": "", "model": "m",
            "rag_rerank_model": "BAAI/bge-reranker-base",
        }), encoding="utf-8")
        assert AgentConfig.load(str(p)).rag_rerank_model == "BAAI/bge-reranker-base"

    def test_deps_probes_int8_file(self):
        """缓存探测的是 int8 文件（预下载与运行时必须同一个模型）。"""
        from agent.rag import deps

        assert deps._RERANK_REQUIRED_FILES == ["onnx/model_int8.onnx"]


class TestRerankerCpuOnly:
    def test_cpu_only_kwargs_for_reranker(self):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        assert _cpu_only_kwargs(TextCrossEncoder) == {
            "providers": ["CPUExecutionProvider"]
        }

    def test_vectorstore_constructs_reranker_with_providers(self, monkeypatch):
        """VectorStore 构造 TextCrossEncoder 时确实传了 CPU-only。"""
        import threading

        captured = {}

        class _FakeCrossEncoder:
            def __init__(self, model_name, cache_dir=None, threads=None,
                         providers=None, cuda=None, lazy_load=False, **kw):
                captured.update({
                    "model_name": model_name,
                    "providers": providers,
                    "cuda": cuda,
                })

            def rerank(self, query, documents, **kw):
                return [0.5 for _ in documents]

        monkeypatch.setattr(vs, "_cpu_only_kwargs",
                            lambda cls: {"providers": ["CPUExecutionProvider"]})
        # 直接驱动 _get_reranker 的加载线程体：伪造 import 来源
        import builtins
        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "fastembed.rerank.cross_encoder":
                mod = type(sys)("fastembed.rerank.cross_encoder")
                mod.TextCrossEncoder = _FakeCrossEncoder
                return mod
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        store = vs.VectorStore(persist_dir="/tmp/rr_cpu_test",
                               rerank_model=RERANK_INT8)
        try:
            store._get_reranker()
        finally:
            store.close()

        assert captured.get("model_name") == RERANK_INT8
        assert captured.get("providers") == ["CPUExecutionProvider"]
        # 不能把 cuda 留在 AUTO：那正是 GPU 机器上静默漂移的根源
        assert captured.get("cuda") != "auto"

    def test_loaded_reranker_uses_cpu_and_int8_file(self):
        """真实加载：providers 只有 CPU，且加载的是 model_int8.onnx。"""
        from agent.rag.vector_store import DEFAULT_RERANK_MODEL, VectorStore

        store = VectorStore(persist_dir="/tmp/rr_real_test",
                            rerank_model=DEFAULT_RERANK_MODEL)
        try:
            rr = store._get_reranker()
            assert rr is not None
            assert rr.model_name == RERANK_INT8
            session = getattr(getattr(rr, "model", None), "model", None)
            if session is not None and hasattr(session, "get_providers"):
                assert session.get_providers() == ["CPUExecutionProvider"]
            if hasattr(session, "_model_path"):
                assert session._model_path.endswith("model_int8.onnx")
        finally:
            store.close()


# ---------------------------------------------------------------------------
# deps 预下载路径：CPU-only + 缓存探测
# ---------------------------------------------------------------------------


class TestDepsCpuOnly:
    def test_embedding_download_passes_providers(self, monkeypatch):
        """deps._download_embedding_model 也强制 CPU-only（不是 lazy_load）。"""
        from agent.rag import deps

        captured = {}

        class _FakeTE:
            def __init__(self, model_name, cache_dir=None, threads=None,
                         providers=None, cuda=None, **kw):
                captured.update({"providers": providers, "cuda": cuda})

        monkeypatch.setitem(sys.modules, "fastembed",
                            type(sys)("fastembed"))
        import builtins
        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "fastembed":
                mod = type(sys)("fastembed")
                mod.TextEmbedding = _FakeTE
                return mod
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        ok, _ = deps._download_embedding_model()
        assert ok
        assert captured.get("providers") == ["CPUExecutionProvider"]
        assert captured.get("cuda") != "auto"

    def test_reranker_download_passes_providers(self, monkeypatch):
        from agent.rag import deps

        captured = {}

        class _FakeCE:
            def __init__(self, model_name, cache_dir=None, threads=None,
                         providers=None, cuda=None, lazy_load=False, **kw):
                captured.update({"providers": providers, "cuda": cuda})

        import builtins
        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name in ("fastembed.rerank.cross_encoder", "fastembed"):
                mod = type(sys)(name)
                mod.TextCrossEncoder = _FakeCE
                return mod
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        ok, _ = deps._download_reranker_model()
        assert ok
        assert captured.get("providers") == ["CPUExecutionProvider"]
        assert captured.get("cuda") != "auto"


class TestModelCacheProbe:
    """缓存探测：LFS 大文件按 lfs_sha256 命名，旧实现只查 blob_id 导致永远误报。"""

    def test_lfs_blob_named_by_lfs_sha256(self, tmp_path):
        from agent.rag import deps

        model_dir = tmp_path / "models--org--repo"
        (model_dir / "blobs").mkdir(parents=True)
        (model_dir / "trees").mkdir(parents=True)
        # LFS 大文件：blob 文件名是 lfs_sha256，不是 blob_id
        lfs_sha = "a" * 64
        blob_id = "b" * 64
        (model_dir / "blobs" / lfs_sha).write_bytes(b"weights")
        (model_dir / "trees" / "t1.json").write_text(json.dumps({
            "files": {
                "onnx/model_int8.onnx": {
                    "size": 7, "blob_id": blob_id, "lfs_sha256": lfs_sha,
                },
            },
        }), encoding="utf-8")

        assert deps._model_file_cached(str(model_dir), "onnx/model_int8.onnx")
        # 只写 blob_id 而没有 lfs_sha256 对应文件时（普通小文件）也要能命中
        small_id = "c" * 64
        (model_dir / "blobs" / small_id).write_bytes(b"cfg")
        (model_dir / "trees" / "t1.json").write_text(json.dumps({
            "files": {
                "config.json": {"size": 3, "blob_id": small_id},
                "onnx/model_int8.onnx": {
                    "size": 7, "blob_id": blob_id, "lfs_sha256": lfs_sha,
                },
            },
        }), encoding="utf-8")
        assert deps._model_file_cached(str(model_dir), "config.json")

    def test_incomplete_download_not_counted(self, tmp_path):
        """只有 .incomplete 残留时不算缓存（避免把半成品当已下载）。"""
        from agent.rag import deps

        model_dir = tmp_path / "models--org--repo"
        (model_dir / "blobs").mkdir(parents=True)
        (model_dir / "trees").mkdir(parents=True)
        lfs_sha = "d" * 64
        # 中断的下载留下 .incomplete，完整 blob 不存在
        (model_dir / "blobs" / (lfs_sha + ".incomplete")).write_bytes(b"partial")
        (model_dir / "trees" / "t1.json").write_text(json.dumps({
            "files": {"onnx/model_int8.onnx": {"blob_id": "e" * 64,
                                               "lfs_sha256": lfs_sha}},
        }), encoding="utf-8")
        assert not deps._model_file_cached(str(model_dir), "onnx/model_int8.onnx")

    def test_missing_file_not_counted(self, tmp_path):
        from agent.rag import deps

        model_dir = tmp_path / "models--org--repo"
        (model_dir / "blobs").mkdir(parents=True)
        (model_dir / "trees").mkdir(parents=True)
        (model_dir / "trees" / "t1.json").write_text(json.dumps({
            "files": {"other.onnx": {"blob_id": "f" * 64}},
        }), encoding="utf-8")
        assert not deps._model_file_cached(str(model_dir), "onnx/model_int8.onnx")

    def test_embedding_required_files_all_downloadable(self):
        """清单里不能有 fastembed 从不下载的文件（那会让探测永远失败）。"""
        from fastembed import TextEmbedding

        from agent.rag import deps

        registry = {m["model"]: m for m in TextEmbedding.list_supported_models()}
        entry = registry[deps.FASTEMBED_MODEL_NAME]
        # fastembed 声明的必需文件：model_file + additional_files
        needed = [entry["model_file"]] + list(entry.get("additional_files") or [])
        # 我们清单里的 onnx 权重必须与 fastembed 声明的一致
        assert "onnx/model_quantized.onnx" in deps._EMBEDDING_REQUIRED_FILES
        assert "onnx/model_quantized.onnx" in needed
        # vocab.txt 是 BERT 式遗留项：fastembed 的 additional_files 为空，
        # 不会下载它，列进清单 = 探测永远 False
        assert "vocab.txt" not in deps._EMBEDDING_REQUIRED_FILES
