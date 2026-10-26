"""_SubBatchTuner 自适应子批控制器的回归测试。

重点覆盖快模型（bge-small 等）单子批耗时被计时器量化成 0 的情形：旧实现逐子批
``max(dt, 1e-6)`` 兜底会在 dt 临界处制造评分抖动，使调优器误判「变慢」一路下探到
最小子批 2。修复后按窗口累计真实 dt、时间基准稳定后才闭合窗口。
"""
import pytest

from agent.rag.vector_store import _SubBatchTuner, _SUBBATCH_MIN, _SUBBATCH_MAX


def _drive(tuner, dt_fn, rounds=600, L=500):
    """模拟嵌入循环：反复 propose() → report()，覆盖预热/爬升/监控全阶段。

    窗口需累计 _SUBBATCH_MAX_WINDOW(32) 个子批才闭合，故轮数要远大于 32，
    否则连首个基线窗口都跑不完（子批会卡在起点 16 造成假阴性）。
    """
    k = 0
    for _ in range(rounds):
        size = tuner.propose()
        tuner.report(size, L, dt_fn(size, k))
        k += 1
    return tuner.propose()


def test_fast_model_flat_throughput_rests_near_start():
    # 计算受限（吞吐平坦）：dt ∝ 条数 × 最长序列²。bge-small 在 Apple Silicon 上
    # 单子批亚毫秒、偶发被量化成 0。修复后比率估计稳定，应停在起点附近而非沉到 2。
    def dt_fast(size, k):
        if k % 4 == 0:
            return 0.0
        return 1e-6 * size  # 亚毫秒，接近旧兜底阈值 1e-6

    final = _drive(_SubBatchTuner(), dt_fast)
    assert final != _SUBBATCH_MIN, f"快模型子批沉到最小 {final}，应停在合理区间"
    assert final >= 16, f"快模型子批被误压到 {final}"


def test_fast_model_overhead_bound_climbs_not_sinks():
    # 开销受限（越大批越快）：dt 为常数，与条数无关。应确保爬升而非下探。
    def dt_overhead(_size, k):
        if k % 4 == 0:
            return 0.0
        return 1e-6

    final = _drive(_SubBatchTuner(), dt_overhead)
    assert final != _SUBBATCH_MIN, f"快模型子批沉到最小 {final}"
    assert final >= 32, f"开销受限应爬升，却停在 {final}"


def test_slow_model_unchanged_window_behavior():
    # 慢模型：单子批 50ms，累计 2 子批即超 20ms 阈值，窗口闭合行为与修复前一致。
    # 吞吐平坦 → 停在起点。
    def dt_slow(size, _k):
        return 0.05 * size

    final = _drive(_SubBatchTuner(), dt_slow)
    assert final == 16, f"慢模型行为异常：{final}"


def test_subbatch_never_below_min():
    # 极端噪声：dt 在 0 与极小值间跳变，任何情形都不应跌破最小子批。
    def dt_noisy(size, k):
        vals = [0.0, 1e-7, 2e-7, 0.0]
        return vals[k % len(vals)] * size

    final = _drive(_SubBatchTuner(), dt_noisy)
    assert final >= _SUBBATCH_MIN
    assert final <= _SUBBATCH_MAX


def _drive_with_rss(tuner, dt_fn, rss_gb, rounds=600, L=500):
    """复刻 ``_embed_chunks`` 的真实循环：report() 后按相对基线判定是否 force_down。"""
    k = 0
    for _ in range(rounds):
        size = tuner.propose()
        tuner.report(size, L, dt_fn(size, k))
        if tuner.rss_over_limit(rss_gb):
            tuner.force_down()
        k += 1
    return tuner.propose()


def test_rss_relative_baseline_keeps_size_on_docling_heavy_sync():
    # docling 2 实例 ~7-9GB + MLX LLM + LanceDB 把稳定基线推到 ~9GB，远超旧绝对墙
    # 8GB。旧逻辑：每个子批 rss>8 → force_down 钉死在 2，且缩小子批降不了 docling/LLM
    # 占的 RSS，永久回不到墙下。新逻辑（相对基线）：9GB 即基线、不触发，平坦吞吐应
    # 停在 16，不再被钉死在最小子批。
    steady_rss = 9.0  # GB，docling 重型同步的典型稳定占用

    def dt_flat(size, k):
        # 吞吐平坦：dt ∝ 条数（计算受限），bge-small 快模型。
        return 0.0 if k % 4 == 0 else 1e-6 * size

    final = _drive_with_rss(_SubBatchTuner(), dt_flat, steady_rss)
    assert final != _SUBBATCH_MIN, f"docling 重型同步下子批仍被 RSS 墙钉死在 {final}"
    assert final >= 16, f"子批被误压到 {final}（应停在 16 附近）"


def test_rss_old_absolute_logic_would_have_sunk():
    # 对照：用旧「绝对 8GB」判定复刻，证明正是它把子批钉死在 2 —— 反证新修复必要。
    tuner = _SubBatchTuner()
    k = 0
    for _ in range(600):
        size = tuner.propose()
        tuner.report(size, 500, 0.0 if k % 4 == 0 else 1e-6 * size)
        if 9.0 > 8.0:  # 旧绝对墙
            tuner.force_down()
        k += 1
    assert tuner.propose() == _SUBBATCH_MIN, "旧绝对墙未把子批钉死在最小 2（对照失败）"


def test_rss_over_limit_semantics():
    # 相对基线判定：首测记基线（高水位只升不降），仅高出「基线+余量(2.5)」且越过绝对
    # 地板(13) 才触发；docling/LLM 常驻波动（基线内）不触发。
    t = _SubBatchTuner()
    assert t.rss_over_limit(9.0) is False    # 记基线，不触发
    assert t.rss_over_limit(9.0) is False    # 同基线
    assert t.rss_over_limit(11.0) is False   # 9+2.5=11.5 < 绝对地板 13
    assert t.rss_over_limit(14.0) is True    # 14 > 绝对地板 13 → 触发（基线随之升 14）
    assert t.rss_over_limit(13.0) is False   # 现基线 14, effective=max(13,16.5)=16.5
    assert t.rss_over_limit(17.0) is True    # >16.5 触发
    assert t.rss_over_limit(9.0) is False    # 回落到基线内不触发（基线不降）

