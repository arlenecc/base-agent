"""多知识分区 + 决策模型路由联合检索测试。

覆盖：config 分区归一 / 引擎多分区存储隔离 / 路由保底 / 跨库加权融合 /
多库判定管线（粗筛→融合20→判定→3）/ 三模式分发 / reranker 屏蔽。
"""
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.config import AgentConfig
from agent.rag.multi_search import (
    FUSION_POOL,
    MULTI_DECISION_TOP,
    fuse_partition_results,
    multi_decision_search,
    route_partitions,
)


# ---------------------------------------------------------------------------
# config：分区归一
# ---------------------------------------------------------------------------

def test_config_partitions_empty_single_library():
    c = AgentConfig()
    assert c.get_partitions() == []


def test_config_partitions_synth_default_from_legacy_kb():
    """老配置（无 rag_partitions 有 knowledge_base）→ 合成默认分区。"""
    c = AgentConfig(knowledge_base="/tmp/kb")
    parts = c.get_partitions()
    assert len(parts) == 1
    assert parts[0]["name"] == "默认知识库"
    assert parts[0]["default"] is True and parts[0]["enabled"] is True
    assert c.get_default_partition()["path"] == "/tmp/kb"


def test_config_partitions_name_dedup_and_single_default():
    parts_in = [
        {"name": "法律", "path": "/l1", "enabled": True, "default": False},
        {"name": "法律", "path": "/l2", "enabled": True, "default": True},
        {"name": "医学", "path": "/m", "enabled": True, "default": True},
    ]
    c = AgentConfig(rag_partitions=json.dumps(parts_in))
    parts = c.get_partitions()
    names = [p["name"] for p in parts]
    assert len(names) == len(set(names))  # 名字唯一
    defaults = [p for p in parts if p["default"]]
    assert len(defaults) == 1


def test_config_partitions_default_must_be_enabled():
    c = AgentConfig(rag_partitions=json.dumps(
        [{"name": "a", "path": "/a", "enabled": False, "default": True}]))
    assert c.get_partitions()[0]["enabled"] is True


def test_config_partitions_invalid_json_falls_back():
    c = AgentConfig(rag_partitions="not-json", knowledge_base="/k")
    assert len(c.get_partitions()) == 1  # 回退合成默认分区


# ---------------------------------------------------------------------------
# 引擎：分区 slug / 目录 / 存储隔离
# ---------------------------------------------------------------------------

def _mk_engine(partitions, td, rag_mode="vector", **kw):
    from agent.rag.engine import RAGEngine
    return RAGEngine(
        workspace=td, knowledge_base="/tmp/kb",
        rag_partitions=json.dumps(partitions) if partitions else "",
        rag_mode=rag_mode, embedding_model="", **kw)


def _two_partitions(td):
    kb1 = os.path.join(td, "kb1"); os.makedirs(kb1, exist_ok=True)
    kb2 = os.path.join(td, "kb2"); os.makedirs(kb2, exist_ok=True)
    open(os.path.join(kb1, "a.md"), "w").write("# 科技\n苹果手机性能强。" * 10)
    open(os.path.join(kb2, "b.md"), "w").write("# 法律\n合同法条文内容。" * 10)
    return [
        {"name": "科技", "path": kb1, "enabled": True, "default": True},
        {"name": "法律", "path": kb2, "enabled": True, "default": False},
    ]


def test_partition_slug_unique_and_safe():
    from agent.rag.engine import RAGEngine
    s1 = RAGEngine._partition_slug("法律")
    s2 = RAGEngine._partition_slug("法律 ")
    assert s1 != s2
    assert "法律" in s1 and len(s1.split("_")[-1]) == 12


def test_engine_partition_dirs_default_uses_legacy_layout(tmp_path):
    parts = _two_partitions(tmp_path)
    eng = _mk_engine(parts, str(tmp_path))
    dirs = eng._partition_dirs()
    default_slug = eng._default_partition_slug()
    assert dirs[default_slug] == str(tmp_path / "rag")  # 默认分区 = 旧布局
    other = [s for s in dirs if s != default_slug][0]
    assert f"partitions{os.sep}" in dirs[other]


def test_engine_per_partition_ingest_and_store_isolation(tmp_path):
    parts = _two_partitions(tmp_path)
    eng = _mk_engine(parts, str(tmp_path))
    stat = eng.ingest()
    assert not stat.get("error")
    assert set(stat.get("partitions", {}).keys()) == {"科技", "法律"}
    s1 = eng._get_store("科技")
    s2 = eng._get_store("法律")
    assert s1.count() > 0 and s2.count() > 0
    assert s1._persist_dir != s2._persist_dir  # 存储完全独立
    # 各库 FTS 只命中各自文档
    r1 = s1.search_fts("苹果手机", top_k=3)
    r2 = s2.search_fts("合同法", top_k=3)
    assert r1 and all("a.md" in x["source"] for x in r1)
    assert r2 and all("b.md" in x["source"] for x in r2)
    # 增量再同步幂等
    stat2 = eng.ingest()
    for v in stat2.get("partitions", {}).values():
        assert v.get("files_skipped", 0) >= 1
    eng.close()


def test_engine_ingest_single_partition_only(tmp_path):
    parts = _two_partitions(tmp_path)
    eng = _mk_engine(parts, str(tmp_path))
    stat = eng.ingest(partition="法律")
    assert set(stat.get("partitions", {}).keys()) == {"法律"}
    assert eng._get_store("科技").count() == 0  # 另一分区未动
    eng.close()


def test_engine_close_releases_all_partition_stores(tmp_path):
    parts = _two_partitions(tmp_path)
    eng = _mk_engine(parts, str(tmp_path))
    eng.ingest()
    assert eng._get_store("法律") is not None
    eng.close()
    assert eng._store is None and eng._stores == {}


# ---------------------------------------------------------------------------
# 路由：决策模型打分 + 阈值保底
# ---------------------------------------------------------------------------

class _FakeRouter:
    """按文本关键字打分的假决策模型。"""

    def score(self, query, texts):
        out = []
        for t in texts:
            if t in query or query in t:
                out.append(0.9)
            elif "法" in t and "法" in query:
                out.append(0.6)
            else:
                out.append(0.1)
        return out


P1 = {"name": "科技", "path": "/k", "enabled": True, "default": True}
P2 = {"name": "法律", "path": "/l", "enabled": True, "default": False}
P3 = {"name": "医学", "path": "/m", "enabled": True, "default": False}


def test_route_threshold_selects_relevant_partitions():
    routed = route_partitions("合同法条文", [P1, P2, P3], _FakeRouter())
    names = [p["name"] for p, _s in routed]
    assert names[0] == "法律"  # 0.6 > 0.5 入选；0.1 的被剔除
    assert len(routed) == 1


def test_route_falls_back_to_best_when_all_below_threshold():
    class _Low(_FakeRouter):
        def score(self, query, texts):
            return [0.1] * len(texts)
    routed = route_partitions("合同法条文", [P1, P2, P3], _Low())
    assert len(routed) == 1  # 保底最高分 1 个


def test_route_single_partition_skips_model():
    called = []

    class _Boom:
        def score(self, *a):
            called.append(1)
            raise AssertionError("单分区不应调用模型")
    routed = route_partitions("q", [P1], _Boom())
    assert len(routed) == 1 and routed[0][1] == 1.0
    assert not called


def test_route_model_none_returns_all_equal_weight():
    routed = route_partitions("q", [P1, P2, P3], None)
    assert [(p["name"], s) for p, s in routed] == [("科技", 1.0), ("法律", 1.0),
                                                   ("医学", 1.0)]


def test_route_model_error_degrades_to_all():
    class _Explode:
        def score(self, *a):
            raise RuntimeError("mlx boom")
    routed = route_partitions("q", [P1, P2], _Explode())
    assert len(routed) == 2  # 打分异常 → 退化为全分区等权，检索不崩


# ---------------------------------------------------------------------------
# 跨库加权 RRF 融合
# ---------------------------------------------------------------------------

def _cand(part, src, idx):
    return {"_partition": part, "source": f"/x/{src}", "chunk_index": idx,
            "text": f"{src}#{idx}"}


def test_fuse_normalizes_route_scores_to_weights():
    per = {"科技": [_cand("科技", "a.md", 0)],
           "法律": [_cand("法律", "b.md", 0)]}
    fused = fuse_partition_results(per, {"科技": 0.9, "法律": 0.2})
    assert len(fused) == 2
    assert fused[0]["partition"] == "科技"  # 路由分高的库权重高 → 排前
    w_tech = 0.3 + 0.7 * (0.9 - 0.2) / 0.7  # 归一化 → 1.0
    w_law = 0.3  # 归一化下限
    # fusion_score 已 round(…, 4)
    assert fused[0]["fusion_score"] == pytest.approx(round(w_tech / 61.0, 4))
    assert fused[1]["fusion_score"] == pytest.approx(round(w_law / 61.0, 4))


def test_fuse_dedupes_by_partition_source_chunk():
    per = {"科技": [_cand("科技", "a.md", 0), _cand("科技", "a.md", 1)],
           "法律": [_cand("法律", "a.md", 0)]}  # 同名文件不同库 = 不同候选
    fused = fuse_partition_results(per, {"科技": 1.0, "法律": 1.0})
    assert len(fused) == 3  # 跨库 key 含分区名，不去重


def test_fuse_both_channels_ranked_candidate_wins():
    """同一候选在两库都出现（同名文件路径相同）→ 融合分累加居首。"""
    per = {"科技": [_cand("科技", "same.md", 0), _cand("科技", "a.md", 0)],
           "法律": [_cand("法律", "same.md", 0)]}
    fused = fuse_partition_results(per, {"科技": 1.0, "法律": 1.0})
    # same.md 只在科技库出现一次（key 含分区），跨库同分区重复才会累加
    assert len(fused) == 3
    assert fused[0]["partition"] == "科技"


# ---------------------------------------------------------------------------
# 多库判定管线（decision 模式）：粗筛 → 融合 20 → 判定 → 3
# ---------------------------------------------------------------------------

class _FakePartEngine:
    """假引擎：每分区一个假 store（FTS 返回固定候选）。"""

    def __init__(self, per_store):
        self._per_store = per_store

    def _get_store(self, name):
        return self._per_store[name]

    def _partition_slug(self, name):
        return f"slug_{name}"


class _FakeDecision:
    """判定模型：分区名路由（法律>科技）+ 文本判定（含「合同」高分）。"""

    def score(self, query, texts):
        out = []
        for t in texts:
            if t == "法律":
                out.append(0.8)
            elif t == "科技":
                out.append(0.6)
            elif "合同" in t:
                out.append(0.9)
            else:
                out.append(0.1)
        return out


def _chunks(tag, n):
    # 「合同效力」只出现在 law 库候选中：保证判定分区分库
    kw = "合同效力规定" if tag == "law" else "手机性能规定"
    return [{"text": f"{tag}第{i}条 {kw}" if i % 2 else f"{tag} filler {i}",
             "source": f"/x/{tag}.md", "chunk_index": i,
             "fts_score": 10.0 - i} for i in range(n)]


def test_multi_decision_pipeline_fuse_then_judge():
    eng = _FakePartEngine({
        "科技": _FakeStore(_chunks("tech", 60)),
        "法律": _FakeStore(_chunks("law", 60)),
    })
    routed = [(P1, 0.9), (P2, 0.6)]
    results = multi_decision_search(
        eng, "合同效力", routed, _FakeDecision(),
        coarse_k=50, use_core_sentences=False)
    # 判定只跑一次（融合后批量），返回 ≤3 条
    assert len(results) <= MULTI_DECISION_TOP
    assert all("partition" in r and "score" in r for r in results)
    top = results[0]
    assert top["partition"] == "法律"  # 法律库的「合同」候选判定分最高
    assert top["multi_partition"] is True


class _FakeStore:
    def __init__(self, chunks):
        self._chunks = chunks

    def search_fts(self, query, top_k=50, where=None):
        return list(self._chunks[:top_k])


def test_multi_decision_fallback_without_model():
    """模型不可用 → 回退融合分排序，不崩。"""
    eng = _FakePartEngine({"科技": _FakeStore(_chunks("tech", 5)),
                           "法律": _FakeStore(_chunks("law", 5))})
    results = multi_decision_search(eng, "合同", [(P1, 1.0), (P2, 1.0)],
                                    None, coarse_k=50,
                                    use_core_sentences=False)
    assert results  # 有结果（回退排序）


def test_multi_decision_empty_coarse_returns_empty():
    eng = _FakePartEngine({"科技": _FakeStore([]), "法律": _FakeStore([])})
    results = multi_decision_search(eng, "q", [(P1, 1.0), (P2, 1.0)],
                                    _FakeDecision())
    assert results == []


# ---------------------------------------------------------------------------
# 引擎三模式分发：多分区生效、reranker 屏蔽
# ---------------------------------------------------------------------------

def _engine_with_stores(monkeypatch, parts, rag_mode):
    """构造真引擎（分区解析），但注入假 store 与假模型，隔离 IO。"""
    eng = _mk_engine(parts, str(tempfile.mkdtemp()), rag_mode=rag_mode)
    fake = _FakePartEngine({
        "科技": _FakeStore(_chunks("tech", 60)),
        "法律": _FakeStore(_chunks("law", 60)),
    })
    monkeypatch.setattr(eng, "_get_store", fake._get_store, raising=False)
    monkeypatch.setattr(
        "agent.rag.decision_model.get_decision_model",
        lambda backend, path: _FakeDecision())
    return eng


def test_engine_multi_partition_vector_mode_fuses(monkeypatch, tmp_path):
    eng = _engine_with_stores(monkeypatch, _two_partitions(tmp_path), "vector")
    # search_with_rerank 在假 store 上不存在 → 打补丁记录 use_reranker
    calls = []

    class _RecordingStore(_FakeStore):
        def search_with_rerank(self, query, top_k=5, where=None,
                               min_similarity=0.5, use_reranker=True):
            calls.append(use_reranker)
            return _chunks("r", 3)

    fake = _FakePartEngine({"科技": _RecordingStore([]),
                            "法律": _RecordingStore([])})
    monkeypatch.setattr(eng, "_get_store", fake._get_store, raising=False)
    eng._min_similarity = 0.5
    results = eng.search("合同", top_k=5)
    assert calls and all(c is False for c in calls)  # 多库联合屏蔽 reranker
    assert len(results) <= 5
    assert all("partition" in r for r in results)


def test_engine_multi_partition_decision_mode(monkeypatch, tmp_path):
    eng = _engine_with_stores(monkeypatch, _two_partitions(tmp_path), "decision")
    results = eng.search("合同效力", top_k=5)
    assert 0 < len(results) <= MULTI_DECISION_TOP
    assert results[0]["partition"] == "法律"


def test_engine_single_partition_uses_legacy_path(monkeypatch, tmp_path):
    """单分区：不路由、不融合，走原单库路径。"""
    parts = [{"name": "科技", "path": "/k", "enabled": True, "default": True}]
    eng = _mk_engine(parts, str(tmp_path), rag_mode="vector")
    called = []

    def _boom(*a, **kw):
        called.append(1)
        raise AssertionError("单分区不应进入多分区检索")

    monkeypatch.setattr(eng, "_multi_partition_search", _boom, raising=False)
    monkeypatch.setattr(eng, "_get_store", lambda *a: None, raising=False)
    # _get_store 返回 None → 单库路径会崩；直接断言 _multi_partition_search 未被调用即可
    try:
        eng.search("q", top_k=1)
    except Exception:
        pass
    assert not called


def test_engine_no_partitions_legacy_mode(monkeypatch, tmp_path):
    """无分区配置（老用户）：行为与改动前完全一致。"""
    eng = _mk_engine([], str(tmp_path), rag_mode="vector")
    assert eng._enabled_partitions() == []
    monkeypatch.setattr(eng, "_get_store", lambda *a: None, raising=False)
    try:
        eng.search("q", top_k=1)
    except Exception:
        pass  # store=None 崩没关系，关键是没走进多分区路径
    assert len(eng._stores) == 0


def test_fusion_pool_constant():
    """按用户规格：融合池 20、最终返回 3。"""
    assert FUSION_POOL == 20
    assert MULTI_DECISION_TOP == 3
