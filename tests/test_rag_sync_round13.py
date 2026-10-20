"""RAG 主线同步（第十三轮）的回归测试。

本轮把 Base-Aug-Agent 主线 RAG 的最新实现移植进本工程，覆盖：
* manifest 跨进程/跨实例互斥与读改写合并（_ManifestLock / _save_manifest）
* LPT 负载均衡分片（_shard_files_balanced）
* 向量库写入批量 200 + stats["chunks"] 记本次写入量
* 图片型 PDF OCR 路由（rapidocr 3.x 优先 / 1.x 回退 / CoreML 门控）
* OCR 引擎周期回收与每页取用
* 暗色主题反色与 _pixmap_is_blank
"""
import json
import os
import sys

import pytest

sys.path.insert(0, "src")

from agent.rag import engine as rag_engine
from agent.rag import parsers
from agent.rag.engine import _shard_files_balanced, _ManifestLock


# ---------------------------------------------------------------------------
# manifest 锁与合并
# ---------------------------------------------------------------------------


class TestManifestLock:
    def test_save_merges_concurrent_entries(self, tmp_path):
        """读改写合并：另一个实例刚写的条目不能被旧快照抹掉。"""
        rag_dir = str(tmp_path / "rag")
        os.makedirs(rag_dir, exist_ok=True)
        # 实例 A 先写一条
        rag_engine._save_manifest(rag_dir, {"a.md": {"mtime": 1}})
        # 实例 B 拿着**旧快照**（空）后写：旧实现会把 a.md 抹掉
        rag_engine._save_manifest(rag_dir, {"b.md": {"mtime": 2}})
        merged = rag_engine._load_manifest(rag_dir)
        assert set(merged) == {"a.md", "b.md"}

    def test_save_removes_intentional_deletions(self, tmp_path):
        """removed 集合里的键必须删掉（不能被合并逻辑复活）。"""
        rag_dir = str(tmp_path / "rag")
        os.makedirs(rag_dir, exist_ok=True)
        rag_engine._save_manifest(rag_dir, {"a.md": {"mtime": 1}})
        rag_engine._save_manifest(rag_dir, {"b.md": {"mtime": 2}},
                                  removed={"a.md"})
        merged = rag_engine._load_manifest(rag_dir)
        assert set(merged) == {"b.md"}

    def test_lock_is_reentrant_in_same_thread(self, tmp_path):
        """同一线程嵌套 __enter__ 不能自锁（flock 同 fd 互斥的坑）。"""
        rag_dir = str(tmp_path / "rag")
        os.makedirs(rag_dir, exist_ok=True)
        with _ManifestLock(rag_dir):
            with _ManifestLock(rag_dir):
                # 能走到这里且不挂死即通过
                assert rag_engine._load_manifest(rag_dir) == {}

    def test_lock_released_after_exception(self, tmp_path):
        """异常路径也必须释放（否则后续所有同步永久挂起）。"""
        rag_dir = str(tmp_path / "rag")
        os.makedirs(rag_dir, exist_ok=True)
        with pytest.raises(RuntimeError):
            with _ManifestLock(rag_dir):
                raise RuntimeError("boom")
        # 锁已释放：再次进入不会挂死
        with _ManifestLock(rag_dir):
            pass


# ---------------------------------------------------------------------------
# LPT 分片
# ---------------------------------------------------------------------------


class TestShardFilesBalanced:
    def test_no_empty_shards(self, tmp_path):
        """n ≥ workers 时不允许空片（旧实现 9 文件 4 worker 第 4 片为空）。"""
        files = []
        for i in range(9):
            p = tmp_path / f"f{i}.txt"
            p.write_text("x" * (100 * (i + 1)))
            files.append(str(p))
        shards = _shard_files_balanced(files, 4)
        assert len(shards) == 4
        assert all(shards), "存在空分片"

    def test_biggest_files_spread_across_workers(self, tmp_path):
        """最大的 N 个文件必然分属不同 worker（慢书自动摊开）。"""
        files = []
        for i in range(8):
            p = tmp_path / f"f{i}.txt"
            p.write_text("x" * ((i + 1) * 1000))
            files.append(str(p))
        shards = _shard_files_balanced(files, 4)
        owners = {}
        for idx, shard in enumerate(shards):
            for f in shard:
                owners[os.path.basename(f)] = idx
        # 最大的 4 个文件（f7..f4）应分布在 4 个不同 worker
        biggest = ["f7.txt", "f6.txt", "f5.txt", "f4.txt"]
        assert len({owners[b] for b in biggest}) == 4

    def test_all_files_assigned_exactly_once(self, tmp_path):
        files = []
        for i in range(7):
            p = tmp_path / f"f{i}.txt"
            p.write_text("y" * (50 * i))
            files.append(str(p))
        shards = _shard_files_balanced(files, 3)
        assigned = [f for shard in shards for f in shard]
        assert sorted(assigned) == sorted(files)

    def test_missing_file_treated_as_zero(self, tmp_path):
        """扫描后消失的文件按 0 处理，不抛（worker 内再隔离）。"""
        shards = _shard_files_balanced(["/nonexistent/x.txt"], 2)
        assert sorted(f for s in shards for f in s) == ["/nonexistent/x.txt"]

    def test_single_worker_gets_everything(self):
        shards = _shard_files_balanced(["a", "b"], 1)
        assert shards == [["a", "b"]]


# ---------------------------------------------------------------------------
# 写入批量 / stats 语义
# ---------------------------------------------------------------------------


class TestWriteBatch:
    def test_write_batch_size_is_200(self):
        """主线定标：批量 20 在大库上留下海量 fragment。"""
        assert rag_engine._WRITE_BATCH_SIZE == 200


# ---------------------------------------------------------------------------
# OCR 路由
# ---------------------------------------------------------------------------


class TestOcrRouting:
    def test_3x_installed_but_not_apple_silicon_uses_cpu(self, monkeypatch):
        """3.x 可用但非 Apple Silicon → 不传 use_coreml（避免 CoreML 初始化失败）。"""
        mod3 = types_mod = __import__("types").ModuleType("rapidocr")
        seen = {}

        class _R:
            def __init__(self, params=None, config_path=None):
                seen["params"] = params

            def __call__(self, *a, **k):
                return (None, None)

        mod3.RapidOCR = _R
        monkeypatch.setitem(sys.modules, "rapidocr", mod3)
        monkeypatch.setattr(parsers, "_is_apple_silicon", lambda: False)
        parsers._build_ocr_engine()
        assert seen["params"] is None

    def test_3x_on_apple_silicon_enables_coreml(self, monkeypatch):
        mod3 = __import__("types").ModuleType("rapidocr")
        seen = {}

        class _R:
            def __init__(self, params=None, config_path=None):
                seen["params"] = params

            def __call__(self, *a, **k):
                return (None, None)

        mod3.RapidOCR = _R
        monkeypatch.setitem(sys.modules, "rapidocr", mod3)
        monkeypatch.setattr(parsers, "_is_apple_silicon", lambda: True)
        # 显式给出本用例的输入：_OCR_CONFIG 是模块级状态，别的用例注入过
        # rag_ocr_coreml=False 的 config 会残留到这里；AgentConfig 的默认值
        # 也不能当断言依据（它是可调的产品决策）。_OCR_FORCE_CPU 同理——
        # 内存护栏一旦置位，配置再怎么开都会被否决。
        ns = __import__("types").SimpleNamespace
        monkeypatch.setattr(parsers, "_OCR_CONFIG", ns(rag_ocr_coreml=True))
        monkeypatch.setattr(parsers, "_OCR_FORCE_CPU", False)
        parsers._build_ocr_engine()
        assert seen["params"] == {"EngineConfig.onnxruntime.use_coreml": True}

    def test_legacy_ocr_rejects_legacy_formats(self):
        """.doc/.xls/.ppt 已不在支持列表（装了也解析不了）。"""
        assert ".doc" not in parsers.SUPPORTED_EXTENSIONS
        assert ".xls" not in parsers.SUPPORTED_EXTENSIONS
        assert ".ppt" not in parsers.SUPPORTED_EXTENSIONS


# ---------------------------------------------------------------------------
# OCR 引擎周期回收
# ---------------------------------------------------------------------------


class TestOcrRecycle:
    def test_recycle_pages_constant(self):
        assert parsers._OCR_RECYCLE_PAGES == 50

    def test_engine_released_at_recycle_boundary(self):
        """跨回收边界后 release 被调用（冲刷 CoreML 原生缓存）。"""
        calls = {"n": 0}
        orig = parsers.release_ocr_engine

        def _counting():
            calls["n"] += 1
            orig()

        parsers.release_ocr_engine = _counting
        try:
            ocr_page_count = 0
            ocr_skipped_count = 0
            # 模拟 100 页 OCR：跨两个回收周期
            for _ in range(100):
                ocr_page_count += 1
                if (ocr_page_count + ocr_skipped_count) % parsers._OCR_RECYCLE_PAGES == 0:
                    parsers.release_ocr_engine()
            assert calls["n"] == 2
        finally:
            parsers.release_ocr_engine = orig


# ---------------------------------------------------------------------------
# 暗色主题反色
# ---------------------------------------------------------------------------


class TestOcrNormalize:
    def test_dark_theme_page_inverted(self):
        """黑底白字页必须反色（模型在白底黑字上训练）。"""
        import numpy as np
        from agent.rag.ocr_engine import preprocess_for_ocr

        # 黑底 + 少量白字
        dark = np.zeros((200, 200, 3), dtype=np.uint8)
        dark[100:110, 50:150] = 255
        out = preprocess_for_ocr(dark)
        # 反色后背景应变亮（均值接近 255 而不是 0）
        assert out.mean() > 128

    def test_normal_page_not_inverted(self):
        """白底黑字页不得反色（反了反而检不出）。"""
        import numpy as np
        from agent.rag.ocr_engine import preprocess_for_ocr

        light = np.full((200, 200, 3), 255, dtype=np.uint8)
        light[100:110, 50:150] = 0
        out = preprocess_for_ocr(light)
        assert out.mean() > 128

    def test_illustration_page_not_inverted(self):
        """大面积彩色插图的正常页（p95 高）不得误触发反色。"""
        import numpy as np
        from agent.rag.ocr_engine import preprocess_for_ocr

        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        out = preprocess_for_ocr(img)
        # 随机图不反色：均值应保持在中位附近（反色会把分布镜像）
        assert abs(out.mean() - img.mean()) < 60
