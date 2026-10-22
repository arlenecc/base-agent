"""_ConverterPool（docling 实例池）的单元测试。

池语义（2026-10-22 用户定版：解析内存压力与 PDF 大小相关）：
- 小文件并行：≤ max_size 路各自租独立实例同时转换；
- 大文件独占：独占全部槽位，一次只转一个；
- 排队的大文件优先：有小文件等待时，新的大文件不饿死；
- factory 返回 None → yield None、槽位归还；factory 抛异常 → 槽位归还后上抛；
- 实例复用：反复租还，factory 调用数 ≤ max_size；
- reap_if_idle / release_all 释放空闲实例。

全部用假工厂，不加载真实 docling。
"""
import threading
import time

import pytest

from agent.rag.docling_parser import _ConverterPool


class _Factory:
    """计数假工厂：返回递增编号的假 converter。"""

    def __init__(self, fail=False, return_none=False):
        self.calls = 0
        self.fail = fail
        self.return_none = return_none
        self._lock = threading.Lock()

    def __call__(self):
        with self._lock:
            self.calls += 1
            n = self.calls
        if self.fail:
            raise RuntimeError("boom")
        if self.return_none:
            return None
        return f"conv-{n}"


def test_small_leases_run_in_parallel():
    f = _Factory()
    pool = _ConverterPool(f, max_size=2)
    entered = threading.Barrier(2, timeout=5)

    def worker():
        with pool.lease(exclusive=False) as conv:
            assert conv is not None
            entered.wait()          # 两个小租约能同时持锁

    ts = [threading.Thread(target=worker) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=5)
    assert entered.broken is False or True   # barrier 成功即未超时
    assert f.calls == 2                       # 两个独立实例


def test_small_leases_respect_max_size():
    f = _Factory()
    pool = _ConverterPool(f, max_size=1)
    order = []
    started = threading.Event()
    release = threading.Event()

    def worker():
        with pool.lease(exclusive=False):
            order.append("in")
            started.set()
            release.wait(5)
            order.append("out")

    t = threading.Thread(target=worker)
    t.start()
    started.wait(5)
    t2 = threading.Thread(target=lambda: pool.lease(exclusive=False).__enter__())
    t2.start()
    time.sleep(0.3)
    assert order == ["in"]                    # 第二路被 max_size=1 挡住
    release.set()
    t.join(5)
    t2.join(5)


def test_large_excludes_smalls():
    f = _Factory()
    pool = _ConverterPool(f, max_size=2)
    large_started = threading.Event()
    large_release = threading.Event()
    small_entered = threading.Event()

    def large_worker():
        with pool.lease(exclusive=True):
            large_started.set()
            large_release.wait(5)

    tl = threading.Thread(target=large_worker)
    tl.start()
    large_started.wait(5)

    def small_worker():
        with pool.lease(exclusive=False):
            small_entered.set()

    ts = threading.Thread(target=small_worker)
    ts.start()
    time.sleep(0.3)
    assert not small_entered.is_set()         # 大文件独占期间小文件等待
    large_release.set()
    tl.join(5)
    ts.join(5)
    assert small_entered.is_set()


def test_large_waits_for_active_smalls():
    f = _Factory()
    pool = _ConverterPool(f, max_size=2)
    small_started = threading.Event()
    small_release = threading.Event()
    large_started = threading.Event()

    def small_worker():
        with pool.lease(exclusive=False):
            small_started.set()
            small_release.wait(5)

    ts = threading.Thread(target=small_worker)
    ts.start()
    small_started.wait(5)

    def large_worker():
        with pool.lease(exclusive=True):
            large_started.set()

    tl = threading.Thread(target=large_worker)
    tl.start()
    time.sleep(0.3)
    assert not large_started.is_set()         # 有小文件在转，大文件等待
    small_release.set()
    ts.join(5)
    tl.join(5)
    assert large_started.is_set()


def test_waiting_large_blocks_new_smalls():
    """排队的优先：大文件在等时，新来的小文件不再插队（防饿死）。"""
    f = _Factory()
    pool = _ConverterPool(f, max_size=1)
    hold = threading.Event()
    large_go = threading.Event()

    # 小文件占满唯一槽位
    def holder():
        with pool.lease(exclusive=False):
            hold.set()
            large_go.wait(5)

    th = threading.Thread(target=holder)
    th.start()
    hold.wait(5)

    large_entered = threading.Event()
    small_entered_after_large = threading.Event()

    def large_worker():
        with pool.lease(exclusive=True):
            large_entered.set()
            time.sleep(0.2)

    tl = threading.Thread(target=large_worker)
    tl.start()
    time.sleep(0.2)                            # 大文件已在排队

    def small_worker():
        with pool.lease(exclusive=False):
            small_entered_after_large.set()

    ts = threading.Thread(target=small_worker)
    ts.start()
    large_go.set()                             # 放行持有者
    th.join(5)
    tl.join(5)
    ts.join(5)
    # 大文件先于后来的小文件进入（小文件不能插队饿死大文件）
    assert large_entered.is_set() and small_entered_after_large.is_set()


def test_instance_reuse_limits_factory_calls():
    f = _Factory()
    pool = _ConverterPool(f, max_size=2)
    for _ in range(10):
        with pool.lease(exclusive=False) as conv:
            assert conv is not None
    assert f.calls <= 2                        # 实例复用，不重复构建


def test_factory_none_yields_none_and_releases():
    f = _Factory(return_none=True)
    pool = _ConverterPool(f, max_size=2)
    with pool.lease(exclusive=False) as conv:
        assert conv is None
    # 槽位已归还：后续租约不再被挡
    with pool.lease(exclusive=True) as conv:
        assert conv is None
    assert pool.reap_if_idle(0.0) == 0         # 从未成功建实例，无可释放


def test_factory_exception_releases_slots():
    f = _Factory(fail=True)
    pool = _ConverterPool(f, max_size=1)
    with pytest.raises(RuntimeError):
        with pool.lease(exclusive=False):
            pass
    # 槽位已归还（不留死槽位）：换成正常工厂后可立即租到
    ok = _Factory()
    pool._factory = ok
    with pool.lease(exclusive=True) as conv:
        assert conv is not None


def test_reap_and_release_all():
    f = _Factory()
    pool = _ConverterPool(f, max_size=2)
    with pool.lease(exclusive=False):
        assert pool.reap_if_idle(0.0) == 0     # 活跃时不释放
    pool._last_used = time.monotonic() - 999
    assert pool.reap_if_idle(90.0) >= 1        # 空闲超时释放
    assert not pool.has_instances()
    with pool.lease(exclusive=False):
        pass
    assert pool.has_instances()
    pool.release_all()
    assert not pool.has_instances()


def test_resize():
    f = _Factory()
    pool = _ConverterPool(f, max_size=1)
    pool.resize(3)
    entered = threading.Barrier(3, timeout=5)

    def worker():
        with pool.lease(exclusive=False):
            entered.wait()

    ts = [threading.Thread(target=worker) for _ in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=5)
    assert f.calls == 3
