"""embedding 指纹校验的测试（换 embedding 模型后旧库防护）。

换 embedding 模型后打开旧向量库：
  * 指纹一致 → 正常使用；
  * 指纹不一致（换模型）→ add/search 明确报错并指明清空重建 / 切回原模型，
    而不是 LanceDB 的 Arrow 字节长度错误，或（同维度时）静默混用污染检索；
  * 老库无指纹 → 采用当前模型为基线（历史模型无从得知）；
  * 维度兜底：被采用的老库若维度不同，写/查时用 schema 的 list_size 拦下；
  * clear() 删除指纹 → 换模型后可直接重建。
"""
from __future__ import annotations

import hashlib

import pytest


class _FakeEF:
    """确定性假 embedding：维度可配、name() 指纹可配。"""

    def __init__(self, name: str = "model-a", dim: int = 128):
        self._name = name
        self._dim = dim

    def __call__(self, input):
        out = []
        for text in input:
            h = hashlib.md5(text.encode("utf-8")).digest()
            out.append([h[i % len(h)] / 255.0 for i in range(self._dim)])
        return out

    def embed_query(self, text: str):
        return self.__call__([text])[0]

    def name(self):
        return self._name


def _make(dirpath, ef):
    from agent.rag.vector_store import VectorStore

    return VectorStore(persist_dir=str(dirpath), embedding_function=ef)


def _chunks(source, texts):
    return [{"source": source, "chunk_index": i, "text": t}
            for i, t in enumerate(texts)]


def test_fingerprint_recorded_on_create(tmp_path):
    """建表即写入指纹（模型名 + 维度）。"""
    d = tmp_path / "vectors"
    store = _make(d, _FakeEF("model-a", 128))
    store.add(_chunks("/kb/a.pdf", ["内容一", "内容二"]))
    assert store._read_stored_fingerprint() == ("model-a", 128)


def test_reopen_with_same_model_ok(tmp_path):
    """同模型重新打开：读写不受影响。"""
    d = tmp_path / "vectors"
    _make(d, _FakeEF("model-a")).add(_chunks("/kb/a.pdf", ["内容"]))
    store2 = _make(d, _FakeEF("model-a"))
    assert store2.search("内容", top_k=3)
    store2.add(_chunks("/kb/b.pdf", ["新内容"]))       # 不抛


def test_reopen_with_different_model_raises(tmp_path):
    """换模型（同维度）后 add/search 明确报错，不静默混用。"""
    d = tmp_path / "vectors"
    _make(d, _FakeEF("model-a")).add(_chunks("/kb/a.pdf", ["内容"]))
    store2 = _make(d, _FakeEF("model-b"))              # 同为 128 维
    with pytest.raises(RuntimeError, match="model-a"):
        store2.search("内容", top_k=3)
    with pytest.raises(RuntimeError, match="model-b"):
        store2.add(_chunks("/kb/c.pdf", ["新内容"]))
    # 错误信息给出可执行的出路
    with pytest.raises(RuntimeError, match="清空知识库"):
        store2.add(_chunks("/kb/c.pdf", ["新内容"]))


def test_legacy_lib_adopts_current_model(tmp_path):
    """老库（无指纹表）：采用当前模型为基线并回写，之后正常使用。"""
    d = tmp_path / "vectors"
    _make(d, _FakeEF("model-a")).add(_chunks("/kb/a.pdf", ["内容"]))
    store2 = _make(d, _FakeEF("model-a"))
    store2._ensure_initialized()
    store2._db.drop_table("_embedding_meta")           # 模拟旧版本建的库
    assert store2.search("内容", top_k=3)              # 不抛
    assert store2._read_stored_fingerprint() == ("model-a", 128)


def test_dim_mismatch_caught_after_adoption(tmp_path):
    """老库被采用后维度不同：schema 维度兜底拦截，报可读错误。"""
    d = tmp_path / "vectors"
    _make(d, _FakeEF("model-a", dim=128)).add(_chunks("/kb/a.pdf", ["内容"]))
    store2 = _make(d, _FakeEF("model-a", dim=64))      # 同名模型、维度不同
    store2._ensure_initialized()
    store2._db.drop_table("_embedding_meta")
    store3 = _make(d, _FakeEF("model-a", dim=64))      # 重新打开 → 采用 → 维度拦截
    with pytest.raises(RuntimeError, match="64 维"):
        store3.add(_chunks("/kb/b.pdf", ["新内容"]))
    with pytest.raises(RuntimeError, match="128 维"):
        store3.search("内容", top_k=3)


def test_clear_allows_model_switch(tmp_path):
    """clear() 删除指纹：清空后换模型重建不被旧指纹挡住。"""
    d = tmp_path / "vectors"
    _make(d, _FakeEF("model-a")).add(_chunks("/kb/a.pdf", ["内容"]))
    store = _make(d, _FakeEF("model-a"))
    store.clear()
    assert store._read_stored_fingerprint() is None
    _make(d, _FakeEF("model-b")).add(_chunks("/kb/a.pdf", ["新内容"]))
    assert _make(d, _FakeEF("model-b")).search("新内容", top_k=3)
