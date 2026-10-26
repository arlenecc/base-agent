"""本地 reranker（cross-encoder）回归测试。

覆盖项目两种内置 bge reranker 的可选方案，均经 ``LocalReranker`` 同接口：

- ``bge-reranker-v2-m3-mlx-mxfp8``（backend="mlx"，Apple Silicon MLX，mxfp8 量化）
- ``bge-reranker-base-ONNX``（backend="onnx"，强制 CPU-only，量化 onnx）

``LocalReranker`` 已支持 mlx / onnx / gguf 三后端，并通过
``config.reranker_provider`` / ``local_reranker_model`` / ``local_reranker_backend``
切换——这两条路径就是「把 bge 接进 reranker 作为可选方案」的落地形态。

模型权重或对应依赖（mlx_lm / onnxruntime）缺失时整组跳过（与本地权重测试
一致策略）。
"""
import os
import pytest

_ROOT = os.path.dirname(__file__)
_MLX_DIR = os.path.join(_ROOT, "..", "models", "bge-reranker-v2-m3-mlx-mxfp8")
_ONNX_DIR = os.path.join(_ROOT, "..", "models", "bge-reranker-base-ONNX")


def _mlx_available():
    try:
        import mlx_lm  # noqa: F401
        return os.path.isdir(_MLX_DIR)
    except Exception:
        return False


def _onnx_available():
    try:
        import onnxruntime  # noqa: F401
        return os.path.isdir(_ONNX_DIR)
    except Exception:
        return False


@pytest.mark.skipif(not _mlx_available(), reason="mlx_lm 或 MLX reranker 目录不可用")
class TestLocalRerankerMlx:
    def test_mlx_reranker_scores_and_orders(self):
        from agent.local.reranker import LocalReranker

        rr = LocalReranker(_MLX_DIR, backend="mlx")
        q = "如何用 Python 解析 JSON 文件"
        docs = [
            "Python 标准库的 json 模块提供 loads/dumps 来解析与序列化 JSON 字符串",  # 相关
            "红烧肉要先用冷水下锅焯水去腥，再用小火慢炖收汁才软糯",                  # 不相关
        ]
        scores = rr.rerank(q, docs)
        assert len(scores) == 2
        # 相关文档分数应高于不相关文档（两模型均为越大越相关）
        assert scores[0] > scores[1]


@pytest.mark.skipif(not _onnx_available(), reason="onnxruntime 或 ONNX reranker 目录不可用")
class TestLocalRerankerOnnx:
    def test_onnx_reranker_forces_cpu_only(self):
        """ONNX 后端必须强制 CPUExecutionProvider（不占 GPU，与 MLX 路径互不争抢）。"""
        from agent.local.reranker import LocalReranker

        rr = LocalReranker(_ONNX_DIR, backend="onnx")
        rr.rerank("查询占位", ["文档占位"])  # 触发加载
        # 加载后 session 的 provider 列表应只有 CPU
        assert rr._session.get_providers() == ["CPUExecutionProvider"]

    def test_onnx_reranker_scores_and_orders(self):
        from agent.local.reranker import LocalReranker

        rr = LocalReranker(_ONNX_DIR, backend="onnx")
        q = "如何用 Python 解析 JSON 文件"
        docs = [
            "Python 标准库的 json 模块提供 loads/dumps 来解析与序列化 JSON 字符串",  # 相关
            "红烧肉要先用冷水下锅焯水去腥，再用小火慢炖收汁才软糯",                  # 不相关
        ]
        scores = rr.rerank(q, docs)
        assert len(scores) == 2
        assert scores[0] > scores[1]

    def test_onnx_reranker_via_provider_path(self):
        """经真实 provider 路径（reranker_provider=local + 切 onnx 后端）也能产出。"""
        from agent.local import registry
        from agent.rag import vector_store as vs

        path = registry.resolve_model_path("reranker", "bge-reranker-base-ONNX")
        vs.set_reranker_provider("local", "bge-reranker-base-ONNX", "onnx")
        try:
            rr = vs.get_or_create_reranker()
            assert rr.name() == "local-rerank:onnx:bge-reranker-base-ONNX"
            scores = rr.rerank("机器学习为什么需要 GPU", [
                "深度学习依赖 GPU 进行大规模矩阵运算，显著加速反向传播",  # 相关
                "唐朝是中国历史上最强盛的朝代之一",                       # 不相关
            ])
            assert len(scores) == 2 and scores[0] > scores[1]
        finally:
            vs.reset_local_providers()
