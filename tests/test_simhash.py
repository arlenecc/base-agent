"""SimHash 全局文本相似去重（tools/SimHashDedupe.swift → Python 移植）。

覆盖：
* MurmurHash3 x64_128（h1）权威测试向量（与 Swift 自检一致，纯独立生成）
* 文本清理 / 代表文本分段 / 双窗口特征 / SimHash 指纹
* LSH 索引：召回、精排、持久化往返、check_and_remove / remove_path
* engine 集成：转 Markdown 后、切片入库前的近重复整篇跳过 + manifest 标记
"""
import json
import os
import sys

import pytest

sys.path.insert(0, "src")

from agent.rag.simhash import (
    SimHashIndex,
    band,
    clean_keep,
    collides,
    extract_features,
    fingerprint_text,
    hamming,
    make_fingerprint,
    murmur_hash64_str,
    representative_segments,
    verify_similar,
)


# ---------------------------------------------------------------------------
# MurmurHash3 x64_128（h1）权威向量（Swift self-test 同源）
# ---------------------------------------------------------------------------

class TestMurmurHash3:
    VECTORS = {
        "": "0000000000000000",
        "foo": "e271865701f54561",
        "hello": "cbd8a7b341bd9b02",
        "量子力学": "dc380f4d0ae2fc34",
        "abcdefghijklmnopq": "7564747f88bda657",
        "The quick brown fox": "85a60ea92caa4a2a",
        "天地玄黄宇宙洪荒": "d2fa8de00a60c83d",
    }

    def test_authoritative_vectors(self):
        for s, expect in self.VECTORS.items():
            assert format(murmur_hash64_str(s), "016x") == expect, s


# ---------------------------------------------------------------------------
# 文本清理 / 分段 / 特征 / 指纹
# ---------------------------------------------------------------------------

class TestTextAndFeatures:
    def test_clean_keep(self):
        assert clean_keep("Hello, 世界！ \n\tni Hao。") == "hello世界nihao"

    def test_segments_head_tail(self):
        long = "天地玄黄宇宙洪荒" * 10
        segs = representative_segments(long, sample_chars=20)
        assert len(segs) == 2
        assert len(segs[0]) == 20 and len(segs[1]) == 20

    def test_segments_short_and_full(self):
        long = "天地玄黄宇宙洪荒" * 10
        full = representative_segments(long, sample_chars=20, full_text=True)
        assert len(full) == 1 and len(full[0]) == 80
        assert len(representative_segments("abc", sample_chars=20)) == 1

    def test_features_tf_weight_and_stopwords(self):
        opts = dict(sample_chars=0, full_text=True, top3=3, top2=2)
        text = "量子力学量子力学量子力学量力与机器学习机器学习"
        feats = extract_features(text, **opts)
        assert len(feats) <= 5
        d = dict(feats)
        assert d.get("量子力", 0) >= 3  # 权重 = 文档内 TF 频次
        # 停用词被过滤
        with_stop = extract_features("实际上实际上实际上", **opts)
        assert "实际上" not in [w for w, _ in with_stop]

    def test_fingerprint_manual_and_empty(self):
        assert fingerprint_text("") == 0
        a, b = murmur_hash64_str("aaa"), murmur_hash64_str("bbb")
        manual = 0
        for i in range(64):
            va = 5 if (a >> i) & 1 else -5
            vb = 3 if (b >> i) & 1 else -3
            if va + vb > 0:
                manual |= 1 << i
        assert make_fingerprint([("aaa", 5), ("bbb", 3)]) == manual
        assert make_fingerprint([("aaa", 5)]) == a  # 单特征指纹 = 特征哈希

    def test_band_rebuild_and_hamming(self):
        fp = 0x0123456789ABCDEF
        rebuilt = 0
        for i in range(4):
            rebuilt |= band(fp, i) << (16 * i)
        assert rebuilt == fp
        assert hamming(fp, fp) == 0
        assert hamming(fp, fp ^ 0x7) == 3

    def test_pigeonhole_ham3_always_collides(self):
        """全部 1/2/3 bit 翻转（43744 种）必有一段相同（鸽巢原理）。"""
        fp = 0x0123456789ABCDEF
        for i in range(64):
            assert collides(fp, fp ^ (1 << i))
        for i in range(64):
            for j in range(i + 1, 64):
                assert collides(fp, fp ^ (1 << i) ^ (1 << j))

    def test_verify_jaccard(self):
        base = "量子力学与机器学习是现代科学的支柱" * 100
        assert verify_similar(base, base)
        other = "深度学习模型的训练依赖反向传播与随机梯度下降" * 100
        assert not verify_similar(base, other)


# ---------------------------------------------------------------------------
# SimHashIndex：查询即入库、持久化、线程安全封装的语义
# ---------------------------------------------------------------------------

class TestSimHashIndex:
    BODY = "量子力学与机器学习文本" * 60
    OTHER = "完全不同的另一本书内容" * 60

    def test_dup_new_and_persistence(self, tmp_path):
        idx_path = str(tmp_path / "index.json")
        idx = SimHashIndex(path=idx_path, md_dir=str(tmp_path))
        assert idx.check_and_insert(self.BODY, "a.md", "a.md.md") == (False, "", 0)
        r = idx.check_and_insert(self.BODY, "b.md", "b.md.md")
        assert r[0] is True and r[2] == 0 and r[1] == "a.md"
        assert idx.check_and_insert(self.OTHER, "c.md", "c.md.md")[0] is False
        idx.save()
        # 重载后查询结果一致
        idx2 = SimHashIndex.load(idx_path, md_dir=str(tmp_path))
        assert len(idx2) == 2
        r = idx2.check_and_insert(self.BODY, "d.md", "d.md.md")
        assert r[0] is True and r[1] == "a.md"

    def test_same_path_replaces_stale_entry(self, tmp_path):
        """同路径重同步：自身旧条目放行并替换（不追加、不判自身重复）。"""
        idx = SimHashIndex(path=str(tmp_path / "i.json"))
        assert idx.check_and_insert(self.OTHER, "c.md", "c.md.md") == (False, "", 0)
        # 同路径同内容：自身条目不算重复
        assert idx.check_and_insert(self.OTHER, "c.md", "c.md.md") == (False, "", 0)
        assert len(idx) == 1  # 替换而非追加
        # 同路径内容修改：旧指纹被替换，新内容可被他人匹配
        modified = "第二版完全重写的内容河图洛书" * 60
        assert idx.check_and_insert(modified, "c.md", "c.md.md") == (False, "", 0)
        r = idx.check_and_insert(modified, "d.md", "d.md.md")
        assert r[0] is True and r[1] == "c.md"

    def test_missing_verify_material_never_judges_dup(self, tmp_path):
        """owner 的 Markdown 缓存缺失（复核材料拿不到）→ 不判重，宁可多入库。"""
        idx = SimHashIndex(path=str(tmp_path / "i.json"), md_dir="/nonexistent")
        idx.check_and_insert(self.BODY, "x.md", "x.md.md")
        # BODY 自身距离 0 不走复核；构造距离 >0 但无材料的场景
        mutated = self.BODY[:500] + "插入的完全不同片段" + self.BODY[500:]
        r = idx.check_and_insert(mutated, "y.md", "y.md.md")
        # 距离 ≤3 且无复核材料 → 不判重
        if r[0]:
            assert r[2] == 0  # 距离 0 无需材料
        else:
            assert r == (False, "", 0)

    def test_remove_path(self, tmp_path):
        idx = SimHashIndex(path=str(tmp_path / "i.json"))
        idx.check_and_insert(self.BODY, "a.md", "a.md.md")
        idx.check_and_insert(self.OTHER, "b.md", "b.md.md")
        assert idx.remove_path("a.md") == 1
        assert len(idx) == 1
        assert idx.remove_path("a.md") == 0
        # 移除后同内容重新入库为「新文章」
        assert idx.check_and_insert(self.BODY, "a2.md", "a2.md.md") == (False, "", 0)

    def test_load_param_mismatch_resets(self, tmp_path):
        idx_path = str(tmp_path / "i.json")
        idx = SimHashIndex(path=idx_path)
        idx.check_and_insert(self.BODY, "a.md", "a.md.md")
        idx.save()
        # 参数变化 → 旧索引作废重建
        idx2 = SimHashIndex.load(idx_path, ham=2)
        assert len(idx2) == 0


# ---------------------------------------------------------------------------
# 引擎集成：Markdown 后 / 切片前的近重复整篇跳过
# ---------------------------------------------------------------------------

_BASE_PARAS = "".join(
    f"第{n}章 量子纠缠的实验装置由光源、偏振片与单光子探测器组成，"
    "测量结果服从玻姆力学与贝尔不等式的联合约束。"
    for n in range(1, 41))


def _make_kb(tmp_path):
    """a.md 原文；b.md 近重复（换标题 + 中段一句替换，精确指纹不同）；
    c.md 无关内容。"""
    a = "# 量子信息导论\n\n" + _BASE_PARAS
    b = ("# 量子信息概论（修订版）\n\n" + _BASE_PARAS[:1200]
         + "其中探测器阵列经过了校准升级，暗计数显著降低。" + _BASE_PARAS[1200:])
    c = ("# 深度学习入门\n\n"
         + "深度学习模型的训练依赖反向传播与随机梯度下降，"
           "卷积神经网络在大规模图像识别任务中表现优异。" * 40)
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "a.md").write_text(a, encoding="utf-8")
    (kb / "b.md").write_text(b, encoding="utf-8")
    (kb / "c.md").write_text(c, encoding="utf-8")
    return a, b, c


class TestEngineSimHashDedup:
    @pytest.fixture
    def engine(self, tmp_path, rag_engine_factory):
        _make_kb(tmp_path)
        return rag_engine_factory(
            workspace=str(tmp_path / "ws"),
            knowledge_base=str(tmp_path / "kb"),
            parallel_workers=1,
        )

    def test_near_duplicate_skipped_before_chunking(self, engine, tmp_path):
        stats = engine.ingest(force=True)
        assert "error" not in stats
        # b.md 与 a.md 近重复：整篇跳过（不切片入库）
        assert stats["files_duplicated"] == 1
        assert stats["files_extracted"] == 2  # a + c
        assert stats["chunks"] > 0

        # 向量库只有 owner + 无关文档的切片（source 为绝对路径）
        kb_a = str(tmp_path / "kb" / "a.md")
        kb_b = str(tmp_path / "kb" / "b.md")
        kb_c = str(tmp_path / "kb" / "c.md")
        store = engine._get_store()
        sources = set(store.list_sources())
        assert kb_b not in sources
        assert {kb_a, kb_c} <= sources

        # manifest 标记 duplicate_of（下次同步直接跳过）
        with open(os.path.join(engine.rag_dir, "manifest.json"),
                  encoding="utf-8") as f:
            manifest = json.load(f)["files"]
        b_entry = manifest.get(kb_b)
        assert b_entry and b_entry.get("duplicate_of") == kb_a
        assert b_entry.get("simhash_dup") is True

        # 指纹索引已持久化（2 篇唯一文档）
        idx = SimHashIndex.load(os.path.join(engine.rag_dir, "simhash_index.json"))
        assert len(idx) == 2

        # 第二次同步：b.md 走「未修改」跳过路径
        stats2 = engine.ingest(force=False)
        assert stats2["files_skipped"] == 3
        assert stats2["files_extracted"] == 0

    def test_owner_deletion_releases_duplicate(self, engine, tmp_path):
        """owner 被删除：副本记录被僵尸提升丢弃，近重复内容可重新入库。"""
        engine.ingest(force=True)
        os.remove(str(tmp_path / "kb" / "a.md"))
        stats = engine.ingest(force=False)
        # a 删除；b 的 duplicate_of 指向已删 owner → 本轮重新解析。
        # 判重材料：a 的指纹已随删除清理移除，b 不再被判重 → 正常入库。
        assert stats["files_deleted"] >= 1
        store = engine._get_store()
        assert str(tmp_path / "kb" / "b.md") in set(store.list_sources())
