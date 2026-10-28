"""端到端验证：jina-embeddings-v3 全链路（切片 → embedding → Matryoshka 裁剪
→ 重新归一化 → int8 量化 → 入库 → 查询侧同等变换 → 检索效果）。

仅当 jina MLX 权重存在时运行；用注入式 LocalEmbeddingFunction 直接驱动
VectorStore，验证每一阶段都正确且检索有效。
"""
import os
import shutil
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
JINA_MLX = os.path.join(MODELS, "jina-embeddings-v3-mlx-int4")

from agent.rag import vector_store as vs_mod
from agent.rag.chunker import chunk_documents
from agent.local.embedding import LocalEmbeddingFunction
from agent.rag.vector_store import VectorStore


# 多主题语料：前三个各约 1 个切片，python 长文强制切成多个切片。
_RECIPE = (
    "番茄炒蛋的做法：先将鸡蛋打散加少许盐，热锅下油，倒入蛋液快速滑散盛出；"
    "另起锅爆香葱花，下番茄块翻炒出汁，加糖与盐调味，最后倒回鸡蛋翻炒均匀即可。"
    "关键在于蛋液要热油快炒保持嫩滑，番茄需炒出红油才够味。"
)
_QUANTUM = (
    "量子计算机利用量子比特的叠加与纠缠进行并行计算，相比经典计算机在特定问题"
    "上具备指数级加速。Shor 算法可高效分解大整数，Grover 算法可平方级加速无序搜索。"
    "实现上依赖超导、离子阱等物理体系，当前仍受退相干与错误率限制，需量子纠错支撑。"
)
_BASKETBALL = (
    "篮球比赛每队五人，目标将球投入对方篮筐得分。进攻方需在 24 秒内完成投篮，"
    "走步、两次运球等违例会被判罚。三分线外命中得 3 分，线内得 2 分，罚球每球 1 分。"
    "防守端通过抢断、盖帽与篮板限制对手，最终比分高者获胜。"
)
_PYTHON = (
    "装饰器是编程语言里一种非常优雅且实用的高级特性，它允许我们在不改动目标函数"
    "或类原有代码的前提下，为其动态地附加新的行为。它的本质是一个高阶函数，接收"
    "一个可调用对象，返回另一个可调用对象。日常开发中最常见的写法，是用特殊的符号"
    "把装饰器贴在目标定义的上一行，从而声明式地增强功能。举例来说，一个用来统计耗"
    "时的装饰器，可以在函数开始执行之前记下时间点，在结束之后计算并输出经过的毫秒"
    "数；一个用来记录日志的装饰器，则可以统一打印出函数的入参、返回值与异常信息。"
    "装饰器之所以能够实现这种「无侵入式增强」，核心秘密在于闭包：被返回的内部函数"
    "会保留对外部作用域变量的引用，于是状态可以跨越多次调用而持续存在。对于需要接"
    "收参数的装饰器，还要在外面再多包裹一层工厂函数，先接收参数，再返回真正的装饰"
    "器本体。除了函数装饰器，还有类装饰器，它通过实现特定的调用方法来包装实例，常"
    "被用于对象的注册、单例控制或权限校验。无论哪种装饰器，都应当在内层包装函数上"
    "保留原函数的名称、文档与参数签名，否则在反射、调试与文档生成时都会指向错误的"
    "内层函数，给排查带来困难。在主流的网站开发框架中，路由的注册、身份的鉴权、接"
    "口的限流以及结果的缓存，几乎全都由装饰器来实现，既解耦又声明式。理解并熟练运"
    "用装饰器，是写出更地道、更易维护代码的必经之路。生成器是另一种延迟计算的关键"
    "结构，它用产出语句逐个返回数值，从而节省内存并天然支持无限序列。配合内置的迭"
    "代工具与生成器表达式，可以轻松写出流式的数据处理管线。上下文管理器则专注于资"
    "源的生命周期，确保在正常结束或发生异常时都能正确地释放文件句柄、网络连接或锁。"
    "异步编程以协程表达并发，配合事件循环在高并发的输入输出场景下显著提升吞吐。类"
    "型注解让代码更容易被静态检查与重构，虽然它并不在运行期强制校验，却大幅改善了"
    "可读性与工具支持。虚拟环境用于隔离不同项目的依赖，避免版本之间的冲突；而标准"
    "的打包与安装方式，已经成为社区事实上的共同约定。单元测试借助夹具与参数化，能"
    "一次性覆盖大量边界条件；覆盖率报告则帮助暴露尚未被测试的分支。性能优化永远应"
    "当先测量再动手，精准定位最耗时的热点函数，避免凭直觉盲目优化。多线程由于全局"
    "锁的存在更适合输入输出密集的任务，而计算密集的任务则应交给多进程或原生扩展。"
    "综上所述，掌握装饰器仅仅是入门，真正地把生成器、上下文管理、异步、类型与测试"
    "串成一个完整体系，才算具备了扎实的工程化能力，也才能在阅读大型项目源码时游刃"
    "有余，快速定位问题并给出稳妥的改进方案。"
)


@pytest.mark.skipif(not os.path.isdir(JINA_MLX), reason="jina MLX 权重缺失")
def test_e2e_jina_pipeline_retrieval():
    saved_dims = vs_mod._MATRYOSHKA_DIMS
    saved_enabled = vs_mod._MATRYOSHKA_ENABLED
    tmp = tempfile.mkdtemp()
    vs = None
    try:
        vs_mod.set_matryoshka_enabled(True)
        vs_mod.set_matryoshka_dims(256)

        # ---- 1) 切片 ----
        documents = [
            {"source": "recipe.txt", "text": _RECIPE},
            {"source": "quantum.txt", "text": _QUANTUM},
            {"source": "basketball.txt", "text": _BASKETBALL},
            {"source": "python.txt", "text": _PYTHON},
        ]
        chunks = chunk_documents(
            documents, chunk_size=800, chunk_overlap=50, use_semantic=False
        )
        assert len(chunks) > 0
        py_chunks = [c for c in chunks if c["source"] == "python.txt"]
        # 长文应被切成多片（验证切片确实按 token 预算拆分）
        print(f"[切片] 总切片数={len(chunks)}; python 长文切片数={len(py_chunks)}")
        assert len(py_chunks) >= 2, "python 长文未按预算切成多片"

        # ---- 2~5) embedding → 裁剪 → 重归一化 → int8（逐片显式验证） ----
        ef = LocalEmbeddingFunction(JINA_MLX, backend="mlx")
        assert vs_mod._is_matryoshka_model(ef.name()), ef.name()
        sample = chunks[0]["text"]
        raw = ef.embed_documents([sample])[0]
        assert len(raw) == 1024, f"原始维度应为 1024，实为 {len(raw)}"
        assert abs(np.linalg.norm(raw) - 1.0) < 1e-3, "模型输出应已 L2 归一化"
        stored = vs_mod._prepare_vector_for_index(raw, True, 256)
        assert len(stored) == 256, f"裁剪后应为 256 维，实为 {len(stored)}"
        assert all(isinstance(x, int) and -127 <= x <= 127 for x in stored), "应为 int8 网格"
        cont = np.array(stored, dtype=float) / 127.0
        assert abs(np.linalg.norm(cont) - 1.0) < 1e-2, "量化后还原应重归一化到单位向量"
        # 与查询侧用同一函数 → 同空间
        q_raw = ef.embed_query("如何做番茄炒蛋")
        q_stored = vs_mod._prepare_vector_for_index(q_raw, True, 256)
        assert len(q_stored) == 256
        print(f"[变换] 原始 1024 维归一化 → 裁剪 256 维 → int8 量化; "
              f"doc 模长={np.linalg.norm(cont):.4f}")

        # ---- 6) 入库（注入 jina ef，Matryoshka 开启 / 256） ----
        vs = VectorStore(tmp, embedding_function=ef)
        n = vs.add_streaming(iter(chunks), batch_size=50)
        assert n == len(chunks)
        table = vs._ensure_table()
        assert table is not None
        assert table.count_rows() == len(chunks)
        # 向量列维度 = 256（建表依据首条 record）
        first_vec = table.to_pandas().iloc[0]["vector"]
        assert len(first_vec) == 256, f"入库向量维度应为 256，实为 {len(first_vec)}"
        print(f"[入库] 写入 {n} 条; 入库向量维度={len(first_vec)}")

        # ---- 7) 查询侧同等变换 + 检索效果 ----
        # 查询向量维度同样 256（search 内部用 _prepare_stored_vector）
        results = vs.search("番茄炒蛋的具体做法和步骤是什么", top_k=5)
        assert len(results) > 0
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), "返回结果应按相似度降序"
        top = results[0]
        print(f"[查询] top1 source={top['source']} score={top['score']:.4f}")
        # 相关主题（recipe）应排第一，且相似度高于任一不相干主题
        recipe_score = next(r["score"] for r in results if r["source"] == "recipe.txt")
        other_scores = [r["score"] for r in results if r["source"] != "recipe.txt"]
        assert top["source"] == "recipe.txt", results
        assert recipe_score > max(other_scores), (
            f"recipe 相似度 {recipe_score:.4f} 应高于其它 {max(other_scores):.4f}"
        )

        # 反向验证：python 查询应召回 python 切片
        res2 = vs.search("Python 装饰器的作用和写法是什么", top_k=3)
        assert res2[0]["source"] == "python.txt", res2
        print(f"[查询·反向] top1 source={res2[0]['source']} score={res2[0]['score']:.4f}")

        print("[结论] 全链路（切片→embedding→裁剪→重归一化→int8→入库"
              "→查询侧同等变换）正确，检索有效。")
    finally:
        if vs is not None:
            try:
                vs.close()
            except Exception:
                pass
        vs_mod.set_matryoshka_dims(saved_dims)
        vs_mod.set_matryoshka_enabled(saved_enabled)
        shutil.rmtree(tmp, ignore_errors=True)
