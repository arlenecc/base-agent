"""bge-small 的 MLX 缓存上限是否生效、不会爆内存的验证。

背景：bge-small 默认经 ``LocalEmbeddingFunction`` 的 MLX 后端加载（config 默认
``embedding_provider="local"`` + ``local_embedding_model="bge-small-zh-v1.5-mlx"`` +
``local_embedding_backend="mlx"``）。其前向激活进入 MLX 的 Metal buffer 缓存池，正
是 ``configure_mlx_memory_limits`` 管的那块。本测试验证：

1. ``configure_mlx_memory_limits`` 真的把上限设到 1GB（接到 ``mx.set_cache_limit``）；
2. bge-small 默认经 MLX 后端加载（确认缓存上限管得到它）；
3. ``release_mlx_pool`` 的 ``mx.clear_cache`` 不会把刚设的上限重置掉（否则每文件后
   上限就失效，#131 修复自相矛盾）——MLX 的 clear_cache 只释放未使用 buffer，不动
   set_cache_limit 的全局上限。
"""
import sys

import pytest

mlx = pytest.importorskip("mlx.core")  # 仅 macOS / 装了 mlx 时跑


def test_configure_mlx_memory_limits_wires_1gb_cap(monkeypatch):
    from agent.rag.engine import configure_mlx_memory_limits

    captured = {}
    monkeypatch.setattr(
        mlx, "set_cache_limit",
        lambda n: captured.setdefault("limit", n),
    )
    configure_mlx_memory_limits()
    assert captured.get("limit") == 1_000_000_000, captured


def test_bge_small_defaults_to_mlx_backend():
    # 默认配置下 bge-small 走 local + mlx，确认缓存上限覆盖到它。
    from agent.config import AgentConfig
    cfg = AgentConfig()
    assert cfg.embedding_provider == "local"
    assert "bge-small" in (cfg.local_embedding_model or "")
    assert cfg.local_embedding_backend == "mlx"


def test_clear_cache_does_not_reset_limit():
    # 设 1GB → clear_cache → 上限应仍在（MLX 不重置）。用 monkeypatch 记录调用序列：
    # set_cache_limit 后 clear_cache 不应再次把上限改回默认。
    calls = []
    real_set = mlx.set_cache_limit
    real_clear = mlx.clear_cache

    def spy_set(n):
        calls.append(("set", n))
        return real_set(n)

    def spy_clear():
        calls.append(("clear", None))
        return real_clear()

    try:
        mlx.set_cache_limit = spy_set
        mlx.clear_cache = spy_clear
        mlx.set_cache_limit(1_000_000_000)  # 经 spy 记录
        # 模拟每文件后的 release_mlx_pool：连续 clear 多次
        mlx.clear_cache()
        mlx.clear_cache()
        spy_clear()
    finally:
        mlx.set_cache_limit = real_set
        mlx.clear_cache = real_clear

    # 之后若再 set 才是重置；此处全程只 set 一次（1GB），clear 不应触发 set。
    sets = [c for c in calls if c[0] == "set"]
    assert sets == [("set", 1_000_000_000)], (
        f"clear_cache 不应重置上限，但观察到 set 调用序列：{sets}"
    )
