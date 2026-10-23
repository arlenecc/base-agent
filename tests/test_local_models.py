"""本地模型加载层（MLX/ONNX/GGUF）的功能测试。

重模型（embedding/reranker/llm 的权重）在 ``models/`` 下，.gitignore 不入库。
本测试在对应目录存在时才真跑权重；否则跳过，避免 CI 拉取 GB 级文件。
"""
import os
import sys
import types

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from agent.local.llm import parse_tool_calls, _ThinkSplitter  # noqa: E402
from agent.local import registry  # noqa: E402

MODELS = os.path.join(ROOT, "models")
NOMIC = os.path.join(MODELS, "nomic-embed-text-v1.5-mlx")
BGE = os.path.join(MODELS, "bge-reranker-v2-m3-mlx-mxfp8")
QWEN = os.path.join(MODELS, "qwen3.5-4b-agentic-coder-v4-4bit-mlx")

os.environ.setdefault("HF_HUB_OFFLINE", "1")


# ---------------------------------------------------------------------------
# 工具调用解析（不依赖权重）
# ---------------------------------------------------------------------------

def test_parse_tool_calls_qwen_xml():
    text = (
        '<tool_call>\n'
        '<function=web_search>\n'
        '<parameter=query>北京天气</parameter>\n'
        '</function>\n'
        '</tool_call>'
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    c = calls[0]
    assert c["type"] == "function"
    assert c["function"]["name"] == "web_search"
    assert '"北京天气"' in c["function"]["arguments"]


def test_parse_tool_calls_hermes_json():
    text = '<tool_call>{"name": "get_weather", "arguments": {"city": "Shanghai"}}</tool_call>'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert "Shanghai" in calls[0]["function"]["arguments"]


def test_parse_tool_calls_none():
    assert parse_tool_calls("这是一段普通回复，没有工具调用") == []


def test_think_splitter_incremental():
    sp = _ThinkSplitter()
    out = []
    for delta in ["先看问题", "<think>让我想想", "中间推理", "</think>最终答案"]:
        for kind, seg in sp.feed(delta):
            out.append((kind, seg))
    out += sp.flush()
    kinds = [k for k, _ in out]
    assert "reasoning" in kinds
    assert "content" in kinds
    content_segs = [s for k, s in out if k == "content"]
    reason_segs = [s for k, s in out if k == "reasoning"]
    # 最终答案应出现在某个 content 段；中间推理应出现在某个 reasoning 段
    assert any("最终答案" in s for s in content_segs)
    assert any("中间推理" in s for s in reason_segs)


# ---------------------------------------------------------------------------
# registry 路径解析（不依赖权重）
# ---------------------------------------------------------------------------

def test_registry_default_paths():
    assert registry.resolve_model_path("embedding").endswith(
        "models/nomic-embed-text-v1.5-mlx")
    assert registry.resolve_model_path("reranker").endswith(
        "models/bge-reranker-v2-m3-mlx-mxfp8")
    assert registry.resolve_model_path("llm").endswith(
        "models/qwen3.5-4b-agentic-coder-v4-4bit-mlx")


def test_registry_explicit_path_returns_as_is():
    assert registry.resolve_model_path("embedding", "/tmp/foo") == "/tmp/foo"


def test_resolve_model_max_length_clamps_by_position_type():
    # 旋转式（nomic 有 n_positions）→ 8192；绝对位置（bge-m3 无 n_positions）→ max_position_embeddings
    assert registry.resolve_model_max_length(NOMIC) == 8192
    assert registry.resolve_model_max_length(BGE) == 8194
    # 无 config.json → 回退默认 8192
    assert registry.resolve_model_max_length("/nonexistent/path") == 8192


@pytest.mark.skipif(not os.path.isdir(NOMIC), reason="nomic MLX 权重缺失")
def test_embedding_autoclamps_max_length():
    from agent.local.embedding import LocalEmbeddingFunction
    ef = LocalEmbeddingFunction(NOMIC, backend="mlx")
    assert ef._max_length == 8192  # 旋转式 → 8192


@pytest.mark.skipif(not os.path.isdir(BGE), reason="bge MLX 权重缺失")
def test_reranker_autoclamps_max_length():
    from agent.local.reranker import LocalReranker
    rk = LocalReranker(BGE, backend="mlx")
    assert rk._max_length == 8194  # 绝对位置嵌入 → max_position_embeddings


@pytest.mark.skipif(not os.path.isdir(NOMIC), reason="nomic MLX 权重缺失")
def test_embedding_extrapolation_no_quality_loss():
    """MLX@8192 不应因外推掉质量：短文本下与 MLX@2048 余弦≈1.0，且与 fastembed 同空间。"""
    import numpy as np
    from fastembed import TextEmbedding
    from agent.local.embedding import LocalEmbeddingFunction
    text = "向量检索是一种基于稠密向量相似度的信息检索方法。" * 8  # ~190 token，远小于 2048
    ef = LocalEmbeddingFunction(NOMIC, backend="mlx")

    def norm(v):
        v = np.asarray(v, dtype=np.float64)
        return v / np.linalg.norm(v)

    ef._max_length = 2048
    v2048 = norm(ef.embed_query(text))
    ef._max_length = 8192
    v8192 = norm(ef.embed_query(text))
    # 短文本无截断 → 两种 max_length 应给出完全相同的向量
    assert float(np.dot(v2048, v8192)) > 0.999

    # 与 fastembed 官方 ONNX 参照同空间
    fe = TextEmbedding(model_name="nomic-ai/nomic-embed-text-v1.5-Q",
                      cache_dir=os.path.expanduser("~/.cache/fastembed"))
    vfe = norm(list(fe.query_embed(text))[0])
    assert float(np.dot(vfe, v8192)) > 0.9


# ---------------------------------------------------------------------------
# Embedding（需 nomic 权重）
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.isdir(NOMIC), reason="nomic MLX 权重缺失")
def test_local_embedding_normalized_and_diverse():
    from agent.local.embedding import LocalEmbeddingFunction
    ef = LocalEmbeddingFunction(NOMIC, backend="mlx")
    docs = ef([
        "search_document: 番茄炒蛋的做法",
        "search_document: 量子计算的基本原理",
    ])
    assert len(docs) == 2 and len(docs[0]) == 768
    norms = [float(np.linalg.norm(d)) for d in docs]
    assert all(abs(n - 1.0) < 1e-3 for n in norms)
    # 两个不同主题应明显可分（余弦不应接近 1.0）
    cos = float(np.dot(docs[0], docs[1]))
    assert cos < 0.95


@pytest.mark.skipif(not os.path.isdir(NOMIC), reason="nomic MLX 权重缺失")
def test_local_embedding_matches_fastembed_space():
    """本地 MLX 与既有 fastembed(int8) 索引在同一向量空间（余弦≈0.95+）。

    注意：和 fastembed 对齐时，query 用 ``embed_query``（内部加 search_query:
    前缀），document 用 ``__call__``（加 search_document: 前缀），与 fastembed
    的 query_embed/embed 一一对应——不能直接把已带前缀的文本再喂给本地函数
    （会双重前缀）。
    """
    from agent.local.embedding import LocalEmbeddingFunction
    from fastembed import TextEmbedding
    queries = ["如何做番茄炒蛋", "量子计算机原理"]
    fe = TextEmbedding(model_name="nomic-ai/nomic-embed-text-v1.5-Q",
                       cache_dir=os.path.expanduser("~/.cache/fastembed"))
    fev = np.array([list(fe.query_embed(t))[0].tolist() for t in queries])
    fev /= np.linalg.norm(fev, axis=1, keepdims=True)
    ef = LocalEmbeddingFunction(NOMIC, backend="mlx")
    v = np.array([ef.embed_query(q) for q in queries])
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    per = np.sum(fev * v, axis=1)
    assert per.min() > 0.9, f"本地与 fastembed 向量空间偏差过大: {per}"


# ---------------------------------------------------------------------------
# Reranker（需 bge 权重）
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.isdir(BGE), reason="bge MLX 权重缺失")
def test_local_reranker_ranks_relevant_first():
    from agent.local.reranker import LocalReranker
    rk = LocalReranker(BGE, backend="mlx")
    scores = rk.rerank(
        "如何做番茄炒蛋",
        ["番茄炒蛋需要鸡蛋和番茄", "量子计算原理介绍", "番茄炒蛋的详细步骤"],
    )
    assert len(scores) == 3
    # 无关主题（量子）应排最低
    assert scores[1] == min(scores)
    # 两个相关项应高于无关项
    assert scores[0] > scores[1] and scores[2] > scores[1]


# ---------------------------------------------------------------------------
# Factory 路由（不加载权重）
# ---------------------------------------------------------------------------

def test_factory_llm_api_returns_llmclient():
    from agent.config import AgentConfig
    from agent.local.factory import build_llm_client
    cfg = AgentConfig()
    cfg.llm_provider = "api"
    client = build_llm_client(cfg)
    from agent.llm_client import LLMClient
    assert isinstance(client, LLMClient)


def test_factory_llm_local_returns_localllmclient():
    from agent.config import AgentConfig
    from agent.local.factory import build_llm_client
    cfg = AgentConfig()
    cfg.llm_provider = "local"
    client = build_llm_client(cfg)
    from agent.local.llm import LocalLLMClient
    assert isinstance(client, LocalLLMClient)
    assert client._path.endswith("models/qwen3.5-4b-agentic-coder-v4-4bit-mlx")


# ---------------------------------------------------------------------------
# close() 必须真正回收 MLX 统一内存（#2026-10-23：同步前卸载对话模型却没降内存）
# ---------------------------------------------------------------------------

def _fake_mlx(monkeypatch):
    """注入一个轻量 fake mlx.core，使 close() 在不装真 mlx 时也能跑通并可见调用。"""
    calls = {"clear_cache": 0}

    class _Metal:
        @staticmethod
        def clear_cache():
            calls["clear_cache"] += 1

    class _Core:
        metal = _Metal
        clear_cache = _Metal.clear_cache   # mlx.core.clear_cache 是 metal.clear_cache 的别名

    monkeypatch.setitem(sys.modules, "mlx", type(sys)("mlx"))
    monkeypatch.setitem(sys.modules, "mlx.core", _Core)
    return calls


def test_llm_close_reclaims_mlx_memory(monkeypatch):
    """MLX 后端 close() 必须摘引用 + 调 mx.clear_cache，否则权重滞留 Metal 池。"""
    from agent.local.llm import LocalLLMClient
    calls = _fake_mlx(monkeypatch)
    c = LocalLLMClient(
        "/tmp/does-not-exist-mlx", backend="mlx",
        max_tokens=256, temperature=0.7,
    )
    c._model = object()      # 占位权重，模拟已加载
    c._tokenizer = object()
    c._loaded = True
    c.close()
    assert c._model is None
    assert c._tokenizer is None
    assert c._loaded is False
    assert calls["clear_cache"] == 1   # 关键：必须触发回收


def test_llm_close_no_clear_cache_for_non_mlx(monkeypatch):
    """非 MLX 后端（如 onnx）不应调 mx.clear_cache（避免误伤/无谓调用）。"""
    from agent.local.llm import LocalLLMClient
    calls = _fake_mlx(monkeypatch)
    c = LocalLLMClient(
        "/tmp/x", backend="onnx", max_tokens=256, temperature=0.7,
    )
    c._session = object()
    c._loaded = True
    c.close()
    assert c._session is None
    assert c._loaded is False
    assert calls["clear_cache"] == 0


# ---------------------------------------------------------------------------
# GGUF 加载路径校验（2026-10-23：加载失败时给出明确错误）
# ---------------------------------------------------------------------------

def _fake_llama_cpp(monkeypatch):
    """注入假 llama_cpp.Llama：记录构造 kwargs，不加载真实权重。"""
    seen: dict = {}

    class _Llama:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    mod = types.ModuleType("llama_cpp")
    mod.Llama = _Llama
    monkeypatch.setitem(sys.modules, "llama_cpp", mod)
    return seen


def test_gguf_missing_path_raises_clear_error():
    """路径不存在时应报 FileNotFoundError（而不是 llama.cpp 的黑盒报错）。"""
    from agent.local.llm import LocalLLMClient
    c = LocalLLMClient("/nonexistent/model.gguf", backend="gguf",
                       max_tokens=4096)
    try:
        c._ensure_model()
    except FileNotFoundError as e:
        assert "未找到 GGUF 权重" in str(e)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_gguf_dir_without_gguf_raises_clear_error(tmp_path):
    """目录里没有 .gguf 文件时应报明确错误（含目录提示）。"""
    from agent.local.llm import LocalLLMClient
    (tmp_path / "config.json").write_text("{}")
    c = LocalLLMClient(str(tmp_path), backend="gguf", max_tokens=4096)
    try:
        c._ensure_model()
    except FileNotFoundError as e:
        assert "未找到 GGUF 权重" in str(e)
        assert "目录下没有 .gguf 文件" in str(e)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_gguf_picks_first_gguf_in_dir(monkeypatch, tmp_path):
    """目录含多个 .gguf 时取字典序第一个，并透传给 llama.cpp。"""
    from agent.local.llm import LocalLLMClient
    (tmp_path / "a.gguf").write_bytes(b"fake")
    (tmp_path / "b.gguf").write_bytes(b"fake")
    seen = _fake_llama_cpp(monkeypatch)
    c = LocalLLMClient(str(tmp_path), backend="gguf", max_tokens=8192)
    c._ensure_model()
    assert seen["model_path"] == str(tmp_path / "a.gguf")
    # n_ctx 由 max_tokens 推导（预留 1k prompt 余量）
    assert seen["n_ctx"] == 8192 + 1024


def test_gguf_expands_user_path(monkeypatch, tmp_path):
    """配置里填 ~/xxx.gguf 应展开为用户主目录（旧实现不展开直接失败）。"""
    from agent.local.llm import LocalLLMClient
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "model.gguf").write_bytes(b"fake")
    seen = _fake_llama_cpp(monkeypatch)
    c = LocalLLMClient("~/model.gguf", backend="gguf", max_tokens=8192)
    c._ensure_model()
    assert seen["model_path"] == str(tmp_path / "model.gguf")
    assert c._loaded is True


@pytest.mark.skipif(not os.path.isdir(QWEN), reason="qwen MLX 权重缺失")
def test_factory_llm_local_streams():
    from agent.config import AgentConfig
    from agent.local.factory import build_llm_client
    cfg = AgentConfig()
    cfg.llm_provider = "local"
    cfg.local_llm_model = QWEN
    client = build_llm_client(cfg)
    events = list(client.chat_stream(
        [{"role": "user", "content": "用一句话解释向量检索"}], tools=None,
        temperature=0.3))
    assert any(e.type == "content" and e.content for e in events)
    assert any(e.type == "done" for e in events)
    client.close()


# ---------------------------------------------------------------------------
# 跨轮 KV / 前缀复用开关（2026-10-24：GGUF 对齐 MLX）
# ---------------------------------------------------------------------------

def test_kv_reuse_default_on_and_config_field():
    """开关默认开，且 config 暴露该字段。"""
    from agent.config import AgentConfig
    from agent.local.llm import LocalLLMClient
    assert AgentConfig().local_llm_kv_reuse is True
    assert LocalLLMClient("/tmp/x", backend="gguf")._kv_reuse is True


def test_gguf_kv_reuse_disabled_uses_legacy_path(monkeypatch):
    """开关关闭 → GGUF 主对话也走旧路径（每轮整轮生成，不复用前缀）。"""
    from agent.local.llm import LocalLLMClient
    c = LocalLLMClient("/tmp/x", backend="gguf", kv_reuse=False)
    c._loaded = True      # 跳过真实权重加载
    calls = {"legacy": 0}

    def fake_legacy(self, messages, tools, temperature):
        calls["legacy"] += 1
        return iter([])

    def boom_reuse(self, messages, tools, temperature):
        raise AssertionError("开关关闭时不该走复用路径")

    monkeypatch.setattr(LocalLLMClient, "_stream_gguf_legacy", fake_legacy)
    monkeypatch.setattr(LocalLLMClient, "_stream_gguf_reuse", boom_reuse)
    list(c.chat_stream([{"role": "user", "content": "hi"}], temperature=0.7))
    assert calls["legacy"] == 1


def test_gguf_kv_reuse_enabled_tries_reuse_then_falls_back(monkeypatch):
    """开关打开时先试复用；复用路径抛错则清状态后退回旧路径（不降级功能）。"""
    from agent.local.llm import LocalLLMClient
    c = LocalLLMClient("/tmp/x", backend="gguf", kv_reuse=True)
    c._loaded = True
    reset_calls = {"n": 0}

    class FakeLlama:
        def reset(self):
            reset_calls["n"] += 1
    c._model = FakeLlama()
    c._gguf_fmt = object()     # 有 formatter 才会进入复用分支
    c._gguf_ids = [1, 2, 3]
    calls = {"legacy": 0}

    def boom_reuse(self, messages, tools, temperature):
        raise RuntimeError("boom")

    def fake_legacy(self, messages, tools, temperature):
        calls["legacy"] += 1
        return iter([])

    monkeypatch.setattr(LocalLLMClient, "_stream_gguf_reuse", boom_reuse)
    monkeypatch.setattr(LocalLLMClient, "_stream_gguf_legacy", fake_legacy)
    list(c.chat_stream([{"role": "user", "content": "hi"}], temperature=0.7))
    assert calls["legacy"] == 1
    assert reset_calls["n"] >= 1      # 退回前 reset
    assert c._gguf_ids == []          # 状态清空


def test_gguf_invalidate_kv_cache_resets_context():
    """GGUF 的 KV 在 C++ context 里，失效必须显式 reset + 清空已 eval 序列。"""
    from agent.local.llm import LocalLLMClient
    c = LocalLLMClient("/tmp/x", backend="gguf")
    calls = {"reset": 0}

    class FakeLlama:
        def reset(self):
            calls["reset"] += 1
    c._model = FakeLlama()
    c._gguf_ids = [1, 2, 3]
    c.invalidate_kv_cache()
    assert calls["reset"] == 1
    assert c._gguf_ids == []


def test_gguf_render_none_without_formatter():
    """formatter 不可用时渲染返回 (None, None) → 调用方退回旧路径。"""
    from agent.local.llm import LocalLLMClient
    c = LocalLLMClient("/tmp/x", backend="gguf")
    c._gguf_fmt = None
    c._gguf_fmt_sys = None
    assert c._gguf_render([{"role": "user", "content": "hi"}]) == (None, None)


def test_factory_passes_kv_reuse_and_aux_disabled():
    """主模型按 config 传开关；辅助模型恒为 False（prompt 每轮都变）。"""
    from agent.config import AgentConfig
    from agent.local.factory import build_llm_client, build_aux_llm_client
    cfg = AgentConfig()
    cfg.llm_provider = "local"
    cfg.local_llm_kv_reuse = False
    main = build_llm_client(cfg)
    assert main._kv_reuse is False
    aux = build_aux_llm_client(cfg)
    assert aux._kv_reuse is False
