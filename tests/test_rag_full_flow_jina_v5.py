"""端到端验证：真实 jina-v5 嵌入 + 真实 jina ONNX reranker 跑通整个 RAGEngine 流程。

与 ``test_rag_e2e.py`` 的关键区别：该文件用 ``fake_ef``（确定性 MD5 嵌入，无语义信号）
且未接入真实 reranker，因此**无法验证真实检索质量与 reranker 是否真正介入**。本文件用
项目默认的本地 jina-v5 MLX int4 嵌入（走 provider 单例 → ``_custom_ef`` 为 None → reranker
才会介入）与 jina ONNX reranker，覆盖：

1. 入库全流程：文件发现 → 转换 → 语义切片 → 嵌入 → 128 维 + 归一化 + int8 入库
   （6 篇中文 .md，每篇写长以越过 ``_MIN_DOC_CHARS_FOR_INGEST`` 碎片过滤）
2. 三通道检索：向量 / BM25 全文 / 混合重排，均返回正确结果
3. reranker 实际介入：``search_with_rerank`` 后 ``_last_rerank_used_fallback`` 为 False
   （这是修复「注入 ef 时跳过 reranker」回归的关键断言）
4. 生产入口 ``RAGEngine.search``（混合+重排）能把正确文档排入返回列表
5. 章节摘要调度器线程修复：构造后 ``start()`` 能正常存活（验证 ``_TICK_SECONDS`` 笔误修复）

仅当 jina-v5 MLX 权重 + jina ONNX reranker 权重及对应后端可用时运行。
"""
import os
import shutil
import tempfile
import time
import logging

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
JINA_V5_MLX = os.path.join(MODELS, "jina-embeddings-v5-text-nano-mlx-int4")
JINA_RERANKER = os.path.join(MODELS, "jina-reranker-v2-base-multilingual-onnx")


def _v5_mlx_available():
    try:
        import mlx  # noqa: F401
        return os.path.isdir(JINA_V5_MLX)
    except Exception:
        return False


def _jina_reranker_available():
    try:
        import onnxruntime  # noqa: F401
        return os.path.isdir(JINA_RERANKER)
    except Exception:
        return False


@pytest.mark.skipif(
    not (_v5_mlx_available() and _jina_reranker_available()),
    reason="mlx/onnxruntime 不可用，或缺 jina-v5 MLX / jina ONNX reranker 权重",
)
class TestRagFullFlowJinaV5:
    """真实 jina-v5 嵌入 + 真实 jina reranker 的整条 RAG 链路。"""

    _DOCS = {
        "苹果手机.md": (
            "# 苹果手机\n苹果公司发布的智能手机搭载自研 A 系列芯片，性能强劲且能效出色，"
            "续航表现明显优于前代产品。\n\n新款 iPhone 的影像系统大幅升级，支持电影模式与夜间拍摄，"
            "拍照体验在同类产品中处于领先水平。\n\niOS 系统以流畅、安全与生态闭环著称，"
            "多设备协同让手机、平板与电脑之间的文件与剪贴板无缝流转，极大提升了日常办公效率。"),
        "机器学习.md": (
            "# 机器学习\n机器学习是通过数据训练模型、让程序自动改进任务表现的技术，是人工智能的核心分支。\n\n"
            "监督学习利用带标注的样本训练分类或回归模型；无监督学习则在无标签数据中发现结构，"
            "例如聚类和降维，两者共同构成现代数据分析的基础工具。\n\n"
            "深度学习依托多层神经网络，借助大规模算力在图像识别、语音理解和自然语言处理上取得突破，"
            "推动了自动驾驶与智能助理的发展。训练深度学习模型需要大量数据与 GPU 算力，并通过反向传播不断优化网络权重。\n\n"
            "过拟合是常见难题：模型在训练集表现好却在测试集变差。常用正则化手段包括丢弃法、"
            "权重衰减与早停，用以提升模型在未知数据上的泛化能力。"),
        "红烧肉烹饪.md": (
            "# 红烧肉\n红烧肉的关键在于煸炒糖色与慢火炖煮，成品肥而不腻、色泽红亮诱人。\n\n"
            "五花肉先焯水去腥再切块，冷油下冰糖小火炒出焦糖色，随后下肉块翻炒均匀上色。\n\n"
            "加入生抽、老抽、料酒与葱姜，添热水没过肉块，小火炖煮约四十分钟至软糯，最后大火收汁即可出锅，"
            "咸甜适口，是一道经典的家常荤菜，深受老少喜爱。"),
        "跑步健康.md": (
            "# 跑步与健康\n规律跑步能增强心肺功能、改善新陈代谢，建议每周三到四次中等强度为宜。\n\n"
            "跑步前后应做好热身与拉伸，避免膝盖与小腿受伤，并循序渐进地增加每周里程，切忌突然加量。\n\n"
            "搭配均衡饮食与充足睡眠，跑步对减脂、控制血压和情绪调节都有明显益处，是门槛最低的有氧运动之一。"),
        "唐朝历史.md": (
            "# 唐朝\n唐朝是中国古代最强盛的王朝之一，诗歌创作与丝绸之路贸易都极为繁荣昌盛。\n\n"
            "贞观之治与开元盛世奠定了盛世的基石，长安成为当时世界上最繁华的国际都市之一，"
            "各国使节与商旅云集。\n\n唐诗是文学史上的高峰，李白与杜甫的作品至今仍广为传诵，"
            "对后世东亚文化产生了极为深远的影响。"),
        "篮球运动.md": (
            "# 篮球\n篮球比赛每队五人，以将球投入对方篮筐得分多者为胜，全场通常分四节进行。\n\n"
            "运球、传球与投篮是三项基本功，防守端则需要积极轮转保护与拼抢篮板球。\n\n"
            "NBA 与 FIBA 规则虽有细微差异，但核心目标一致：通过团队配合与个人技术得分，"
            "是一项兼具观赏性与竞技性的全球流行运动。"),
    }

    _QUERIES = [
        ("苹果手机有什么特点", "苹果手机.md"),
        ("什么是机器学习", "机器学习.md"),
        ("红烧肉怎么做才不腻", "红烧肉烹饪.md"),
        ("跑步对身体有什么好处", "跑步健康.md"),
        ("唐朝为什么强盛", "唐朝历史.md"),
        ("篮球比赛几个人一队", "篮球运动.md"),
    ]

    def _build_workspace(self, base):
        kb = os.path.join(base, "kb")
        os.makedirs(kb, exist_ok=True)
        for name, text in self._DOCS.items():
            with open(os.path.join(kb, name), "w", encoding="utf-8") as f:
                f.write(text)
        return kb

    def test_ingest_and_three_channels_with_reranker(self):
        from agent.local.embedding import LocalEmbeddingFunction
        from agent.rag import vector_store as vs
        from agent.rag.engine import RAGEngine

        # 保存并配置：Matryoshka 128 + embedding/reranker 都走 provider 单例
        saved_dims = vs._MATRYOSHKA_DIMS
        saved_enabled = vs._MATRYOSHKA_ENABLED
        vs.set_matryoshka_enabled(True)
        vs.set_matryoshka_dims(128)
        vs.set_embedding_provider("local", path=JINA_V5_MLX, backend="mlx")
        vs.set_reranker_provider("local", path="jina-reranker-v2-base-multilingual-onnx", backend="onnx")

        prev_log_level = logging.root.manager.disable
        logging.disable(logging.CRITICAL)  # 压住 RAG 入库的大量 info 日志
        tmp = tempfile.mkdtemp(prefix="rag_flow_")
        try:
            kb = self._build_workspace(tmp)
            # embedding 走 provider（不注入 ef）→ _custom_ef=None → reranker 才会介入
            engine = RAGEngine(
                tmp, knowledge_base=kb,
                embedding_model=JINA_V5_MLX, clamp_model=JINA_V5_MLX,
                rerank_enabled=True,
            )
            stat = engine.ingest()
            assert stat.get("error") is None, f"入库报错: {stat.get('error')}"
            # 6 篇都应越过碎片过滤被提取（每篇写长）
            assert stat["files_extracted"] >= 6, f"有文件被碎片过滤丢弃: {stat}"

            store = engine._get_store()
            assert store.count() >= 6, "入库切片过少"
            dim = store._table.schema.field("vector").type.list_size
            assert dim == 128, f"落库向量维度应为 128，实为 {dim}"

            # ---- 三通道检索 ----
            q0 = "苹果手机有什么特点"
            vec = store.search(q0, top_k=5)
            fts = store.search_fts(q0, top_k=5)
            hyb = store.search_with_rerank(q0, top_k=5)
            assert vec and fts and hyb, "某检索通道返回空"
            assert os.path.basename(vec[0]["source"]) == "苹果手机.md"
            assert os.path.basename(fts[0]["source"]) == "苹果手机.md"

            # reranker 必须真正介入（修复「注入 ef 时跳过 reranker」的回归断言）
            assert store._last_rerank_used_fallback is False, "reranker 未生效（仍走降级）"

            # ---- 生产入口：向量检索应语义命中（真实 jina-v5，确定性）----
            for q, exp in self._QUERIES:
                res = store.search(q, top_k=5)
                assert res, f"向量检索无结果: {q}"
                assert os.path.basename(res[0]["source"]) == exp, (
                    f"向量检索 top1 应为 {exp}，实为 {os.path.basename(res[0]['source'])}"
                )

            # ---- 生产入口 RAGEngine.search：reranker 路径返回结果且含正确文档 ----
            for q, exp in self._QUERIES:
                res = engine.search(q, top_k=5)
                assert res, f"RAGEngine.search 无结果: {q}"
                for r in res:
                    assert 0.0 <= r["score"] <= 1.0, f"score 越界 {r['score']}"
                sources = [os.path.basename(r["source"]) for r in res]
                assert exp in sources, f"{exp} 未进入返回列表: {sources}"

            print(f"[RAG 全流程] 入库={stat['files_extracted']}篇/切片={store.count()} "
                  f"落库维度={dim} reranker已介入=是")
        finally:
            vs.set_matryoshka_dims(saved_dims)
            vs.set_matryoshka_enabled(saved_enabled)
            logging.disable(prev_log_level)
            shutil.rmtree(tmp, ignore_errors=True)

    def test_rerank_min_similarity_configurable_keeps_top1(self):
        """rerank 相似度下限可配置，且全部低于阈值时兜底保留得分最高一条。

        - 把阈值调到 0.99（远高于 reranker 实际得分）→ 正常路径过滤后本应全空，
          但 ``_filter_by_similarity`` 兜底保留 top1，故 ``search_with_rerank`` 仍返回
          ≥1 条；且 reranker 实际参与了（``_last_rerank_used_fallback`` 应为 False，
          不是 RRF 旁路）。
        - 阈值 <=0 表示不过滤，恢复多结果返回。
        注意：为避免命中查询缓存，三个分支用不同的 top_k 形成不同缓存键。
        """
        from agent.rag import vector_store as vs
        from agent.rag.engine import RAGEngine

        saved_dims = vs._MATRYOSHKA_DIMS
        saved_enabled = vs._MATRYOSHKA_ENABLED
        saved_min = vs._RERANK_MIN_SIMILARITY
        vs.set_matryoshka_enabled(True)
        vs.set_matryoshka_dims(128)
        vs.set_embedding_provider("local", path=JINA_V5_MLX, backend="mlx")
        vs.set_reranker_provider(
            "local", path="jina-reranker-v2-base-multilingual-onnx", backend="onnx")

        prev = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        tmp = tempfile.mkdtemp(prefix="rag_min_sim_")
        try:
            kb = self._build_workspace(tmp)
            engine = RAGEngine(
                tmp, knowledge_base=kb, embedding_model=JINA_V5_MLX,
                clamp_model=JINA_V5_MLX, rerank_enabled=True)
            stat = engine.ingest()
            assert stat.get("error") is None, f"入库报错: {stat.get('error')}"
            store = engine._get_store()

            q0 = "苹果手机有什么特点"
            # 默认阈值 0.5：应返回多条，且 reranker 正常介入
            hyb_default = store.search_with_rerank(q0, top_k=5)
            assert hyb_default, "默认阈值下应返回结果"
            assert store._last_rerank_used_fallback is False

            # 调到 0.99：全部低于阈值 → 兜底保留 top1，仍返回 ≥1 条；
            # 且不是 RRF 旁路（reranker 参与了，只是分数被过滤）
            vs.set_rerank_min_similarity(0.99)
            hyb_hi = store.search_with_rerank(q0, top_k=2)
            assert hyb_hi, "高阈值下仍应至少保留 top1"
            assert len(hyb_hi) >= 1
            assert store._last_rerank_used_fallback is False, (
                "高阈值过滤后不应走 RRF 旁路（应保留 rerank top1）")
            assert os.path.basename(hyb_hi[0]["source"]) == "苹果手机.md"

            # <=0 表示不过滤：恢复多结果
            vs.set_rerank_min_similarity(0.0)
            hyb_all = store.search_with_rerank(q0, top_k=1)
            assert hyb_all, "不过滤时应返回结果"
        finally:
            vs.set_rerank_min_similarity(saved_min)
            vs.set_matryoshka_dims(saved_dims)
            vs.set_matryoshka_enabled(saved_enabled)
            logging.disable(prev)
            shutil.rmtree(tmp, ignore_errors=True)

    def test_chapter_summary_scheduler_thread_starts(self):
        """章节摘要调度器线程应能正常启动循环（验证 _TICK_SECONDS 笔误修复）。

        修复前：``while not self._stop.wait(self._TICK_SECONDS)`` 抛 AttributeError，
        线程启动即崩溃。修复后：进程应存活。is_idle 返回 False 让循环空转，
        足以验证走到 wait 行不再因属性缺失而崩。
        """
        from agent.rag.chapter_summarizer import ChapterSummaryScheduler

        def dummy_cfg():
            class C:
                rag_chapter_summary_enabled = True
            return C()

        sched = ChapterSummaryScheduler(
            get_config=dummy_cfg,
            get_engine=lambda: None,
            get_shared_aux=lambda: None,
            is_idle=lambda: False,
        )
        sched.start()
        time.sleep(0.4)
        alive = sched._thread.is_alive()
        sched.stop()
        time.sleep(0.2)
        assert alive is True, "章节摘要线程启动后未存活（_TICK_SECONDS 修复可能失效）"


def test_rerank_min_similarity_config_default():
    """rag_rerank_min_similarity 配置项存在、默认 0.5、且能被强制为 float。

    不依赖任何模型权重（纯配置单测），随 pytest 始终运行。
    """
    from agent.config import AgentConfig

    cfg = AgentConfig()
    assert cfg.rag_rerank_min_similarity == 0.5
    assert "rag_rerank_min_similarity" in AgentConfig._FLOAT_FIELDS
    # config.json 里手写成字符串也要能转回 float（类型强制）
    assert AgentConfig._coerce("rag_rerank_min_similarity", "0.3") == 0.3
    assert AgentConfig._coerce("rag_rerank_min_similarity", "0") == 0.0


def test_set_rerank_min_similarity_clamps():
    """set_rerank_min_similarity 接受 float/字符串，并将负数归一为 0（不过滤）。

    纯模块级单测，不依赖模型权重。
    """
    from agent.rag import vector_store as vs

    vs.set_rerank_min_similarity(0.3)
    assert vs._RERANK_MIN_SIMILARITY == 0.3
    vs.set_rerank_min_similarity("0.7")  # 字符串也接受
    assert vs._RERANK_MIN_SIMILARITY == 0.7
    vs.set_rerank_min_similarity(-0.2)   # 负数 → 0（不过滤）
    assert vs._RERANK_MIN_SIMILARITY == 0.0
    vs.set_rerank_min_similarity("not-a-number")  # 非法值忽略，保持原值
    assert vs._RERANK_MIN_SIMILARITY == 0.0
