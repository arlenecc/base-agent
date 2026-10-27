"""Shared pytest fixtures."""
import os
import sys
import json
import hashlib
import tempfile
from pathlib import Path

import pytest

# Make src importable without installing the package.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Fake embedding function — deterministic, no network, no model download.
# Used by all RAG tests via the `fake_ef` fixture.
# ---------------------------------------------------------------------------

class FakeEmbeddingFunction:
    """Deterministic embedding function for testing.

    Generates a 128-dimensional vector from the MD5 hash of the text,
    repeated to fill the vector. This gives stable, reproducible embeddings
    without requiring any API or model download.
    """

    def __call__(self, input):
        results = []
        for text in input:
            h = hashlib.md5(text.encode("utf-8")).digest()
            # Repeat hash bytes to fill 128 floats, normalize to [0, 1]
            vec = []
            for i in range(128):
                vec.append(h[i % len(h)] / 255.0)
            results.append(vec)
        return results

    def embed_query(self, text: str):
        """Embed a single query string. Returns one vector."""
        return self.__call__([text])[0]

    def name(self):
        return "fake-embedding"


@pytest.fixture(scope="session")
def fake_ef():
    """Return a FakeEmbeddingFunction instance."""
    return FakeEmbeddingFunction()


@pytest.fixture(scope="session")
def rag_engine_factory(fake_ef):
    """Factory that creates RAGEngine instances with the fake embedding function."""
    def _make(workspace, knowledge_base="", **kwargs):
        from agent.rag.engine import RAGEngine
        return RAGEngine(
            workspace=workspace,
            knowledge_base=knowledge_base,
            embedding_function=fake_ef,
            **kwargs,
        )
    return _make


@pytest.fixture(scope="session")
def vector_store_factory(fake_ef):
    """Factory that creates VectorStore instances with the fake embedding function."""
    def _make(persist_dir, **kwargs):
        from agent.rag.vector_store import VectorStore
        return VectorStore(
            persist_dir=persist_dir,
            embedding_function=fake_ef,
            **kwargs,
        )
    return _make


@pytest.fixture(autouse=True)
def _reset_global_model_providers():
    """每个测试前后复位 vector_store 的进程级 embedding/reranker provider 开关。

    不 reset 的话，任何调用过 ``configure_local_models`` 的测试（如构造
    MainWindow 的 test_bridge）会把开关切到 ``local`` 并泄漏到后续测试——
    显式使用 fastembed 的用例会拿到 LocalReranker/FastEmbed 之外的实现而失败。
    """
    from agent.rag import vector_store as _vs
    _vs.reset_local_providers()
    yield
    _vs.reset_local_providers()


@pytest.fixture(autouse=True)
def _reset_chunk_tokenizer():
    """每个测试前后解绑切片分词器（chunker 的进程级全局状态）。

    RAGEngine 构造时会 ``chunker.configure_tokenizer(模型目录)`` 绑定真实分词器。
    不复位的话，任何构造过 RAGEngine 的用例都会把「真实 token 计数」泄漏给后续
    用例——依赖启发式估算数值的断言（切片预算、超长句拆分）会随机失败。
    """
    from agent.rag import chunker as _chunker
    _chunker.configure_tokenizer(None)
    yield
    _chunker.configure_tokenizer(None)


@pytest.fixture()
def tmp_workspace(tmp_path):
    """An isolated working directory used as the agent's filesystem root."""
    return tmp_path


@pytest.fixture()
def config(tmp_workspace):
    from agent.config import AgentConfig

    cfg = AgentConfig()
    cfg.workspace = str(tmp_workspace)
    cfg.base_url = "https://api.example.com/v1"
    cfg.api_key = "test-key"
    cfg.model = "gpt-test"
    return cfg


@pytest.fixture()
def recording_callbacks():
    """An AgentCallbacks implementation that records every call for assertions."""
    from agent.agent import AgentCallbacks

    class Rec(AgentCallbacks):
        def __init__(self):
            self.content = []
            self.reasoning = []
            self.logs = []
            self.tool_starts = []
            self.tool_ends = []
            self.usages = []
            self.speeds = []
            self.confirms = []
            self.asks = []
            self.context_shrinks = []  # list of (summary, reason)
            self._confirm_return = True
            self._ask_return = "ok"
            # 协作式取消测试钩子：设为 True 后 run() 应尽快退出。
            self._cancel_after = None  # e.g. ("content", 2) → 收到第 2 个 content 后取消
            self._cancel_count = 0

        def is_cancelled(self):
            if self._cancel_after is None:
                return False
            kind, n = self._cancel_after
            count = self._cancel_count
            if kind == "content":
                return count >= n
            return False

        def on_content(self, text):
            self.content.append(text)
            if self._cancel_after is not None and self._cancel_after[0] == "content":
                self._cancel_count += 1

        def on_reasoning(self, text):
            self.reasoning.append(text)

        def on_tool_start(self, name, args):
            self.tool_starts.append((name, args))

        def on_tool_end(self, name, result):
            self.tool_ends.append((name, result))

        def on_log(self, line):
            self.logs.append(line)

        def on_usage(self, usage):
            self.usages.append(usage)

        def on_token_speed(self, tokens, speed):
            self.speeds.append((tokens, speed))

        def on_context_shrunk(self, summary, reason):
            self.context_shrinks.append((summary, reason))

        def confirm(self, message):
            self.confirms.append(message)
            return self._confirm_return

        def ask_user(self, prompt):
            self.asks.append(prompt)
            return self._ask_return

    return Rec()
