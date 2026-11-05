"""Context budget: the 30k hard ceiling, split into fixed quotas.

The kernel never asks "is the context too big?" after the fact — it *allocates*
first and lets every compression level spend against its quota. This is what
turns the context from "shrink when it explodes" (old behaviour) into
"structurally bounded" (GenericAgent behaviour).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# 密度常数偏保守：宁可高估触发折叠，也不要低估后静默突破 30k。
# 散文 ASCII ≈4 chars/token，但本仓库塞进上下文的多是代码/JSON/工具 schema，
# 实测接近 3 chars/token；统一取 3.5。CJK 多数分词器 ≈1~1.2 chars/token。
_ASCII_CHARS_PER_TOKEN = 3.5
_CJK_CHARS_PER_TOKEN = 1.3

# CJK 码点 → None 的删除表（str.translate 在 C 层统计，比逐字符 Python 循环
# 快一个数量级）。estimate_tokens 在热路径上被高频调用：每个流式 chunk 一次
# （_ask 的 live_est），外加 _windows 每轮最多 4 次 _render_raw（各 ~40K 字符）。
_CJK_DELETE_TABLE = {
    cp: None
    for cp in list(range(0x4E00, 0xA000))      # CJK 统一表意文字
    + list(range(0x3400, 0x4DC0))              # CJK 扩展 A
    + list(range(0x3000, 0x3100))              # CJK 符号 + 假名
    + list(range(0xFF00, 0xFFF0))              # 全角
}


def estimate_tokens(text: str) -> int:
    """Rough token count. ASCII fast path keeps the hot loop cheap."""
    if not text:
        return 0
    if text.isascii():
        return max(1, int(len(text) / _ASCII_CHARS_PER_TOKEN))
    # translate() 删除的是 CJK 字符 → 用总长相减得到 CJK 数量。
    cjk = len(text) - len(text.translate(_CJK_DELETE_TABLE))
    other = len(text) - cjk
    return max(1, int(cjk / _CJK_CHARS_PER_TOKEN + other / _ASCII_CHARS_PER_TOKEN))


@dataclass
class Quota:
    """各区配额（token）。加起来 = total。"""

    system: int        # 系统提示 + 记忆索引 + 工作记忆 checkpoint
    earlier: int       # L3 折叠摘要区
    raw: int           # 最近 K 轮原文（含当前轮 L1 裁剪后的工具输出）
    tools: int         # 工具 schema
    reserve: int       # 模型输出 + 估算误差


class ContextBudget:
    """把 30k 切成固定配额，并回答「还剩多少可以给工具输出」。

    配额按比例从 ``total`` 派生，因此换模型（更大/更小窗口）时无需改代码。
    """

    # 比例（合计 1.0）
    # 2026-11-05：system 由 0.13（30k 下 3900）提到 1/6（30k 下 5000）。
    # 起因：L0 META-SOP + L1 全局索引常驻 system prompt，压缩前实测 4553 token
    # 会被旧配额判成超支，由 overspent() 从 RAW 扣回 653 token。同日又对
    # SYSTEM_PROMPT 做了提炼（-1354 token，现合计约 3.2K）；5000 配额保留，
    # 给 L1 索引与后续 SOP 增长留余量。多出的份额从 RESERVE 扣，
    # RAW / TOOLS / EARLIER 保持原值：不加剧折叠、不压近期原文预算。
    _SYSTEM_RATIO = 1 / 6
    _RATIO = {
        "system": _SYSTEM_RATIO,
        "earlier": 0.09,
        "raw": 0.40,
        "tools": 0.17,
        "reserve": 0.21 - (_SYSTEM_RATIO - 0.13),
    }

    def __init__(self, total: int = 30_000):
        self.total = max(4_000, int(total))
        r = self._RATIO
        self.quota = Quota(
            # round 而非 int：1/6 的浮点表示乘 30000 会是 4999.999999999999，
            # 直接 int() 会静默少 1 token（用户要的是恰好 5000）。
            system=round(self.total * r["system"]),
            earlier=int(self.total * r["earlier"]),
            raw=int(self.total * r["raw"]),
            tools=int(self.total * r["tools"]),
            reserve=int(self.total * r["reserve"]),
        )
        # 取整尾差（最多 1 token）统一补给 reserve，保证各区配额之和精确
        # 等于 total —— Quota 的契约是「加起来 = total」。
        q = self.quota
        q.reserve += self.total - (q.system + q.earlier + q.raw + q.tools + q.reserve)
        # 实测值（每轮刷新），用于判定是否越界
        self._used = {"system": 0, "earlier": 0, "raw": 0, "tools": 0}

    # -- 记账 ----------------------------------------------------------
    def set_used(self, section: str, tokens: int) -> None:
        if section in self._used:
            self._used[section] = int(tokens)

    def used(self, section: Optional[str] = None) -> int:
        if section:
            return self._used.get(section, 0)
        return sum(self._used.values())

    def over(self, section: str) -> bool:
        return self._used.get(section, 0) > getattr(self.quota, section, 0)

    @property
    def raw_over(self) -> bool:
        return self.remaining_for_raw() <= 0

    @property
    def total_over(self) -> bool:
        """整体越界（含 reserve 被吃掉）。"""
        return self.used() > self.total - self.quota.reserve

    def overspent(self) -> int:
        """除 RAW 外其它区合计超支的 token 数 —— 要从 RAW 里扣回来。"""
        return sum(
            max(0, self._used.get(k, 0) - getattr(self.quota, k, 0))
            for k in ("system", "earlier", "tools")
        )

    # -- 派生 ----------------------------------------------------------
    def remaining_for_raw(self) -> int:
        """RAW 区剩余额度（可能为负，此时必须折叠）。

        system / earlier / tools 任一区超支都要从这里扣，否则「30k 硬顶」
        只是记账：工具 schema 涨到 8k、系统提示涨到 6k 时 RAW 仍然拿满额，
        实际发送量早就越过 30k 而 raw_over 永远为假。
        """
        return self.quota.raw - self._used.get("raw", 0) - self.overspent()

    def remaining_for_tool(self, n_calls: int = 1, max_chars: int = 6_000) -> int:
        """L1：本轮每个工具结果可用的**字符**上限。

        RAW 区剩余额度按并发调用数平分后换算成字符数（粗算 3 chars/token，
        兼顾中英混排），再被 ``max_chars`` 钳住。这样并发 4 个工具时每个
        不会各自拿到满额。
        """
        n = max(1, int(n_calls))
        left_tokens = self.remaining_for_raw()
        if left_tokens <= 0:
            # 已经超预算：只给最小可诊断额度，逼内核立即折叠。
            return min(max_chars, 600)
        return max(600, min(max_chars, int(left_tokens * 3 / n)))

    def chars_for(self, tokens: int) -> int:
        """token → 字符的粗换算（用于给压缩器设置裁剪长度）。"""
        return max(200, int(tokens * 3))

    def snapshot(self) -> str:
        u = self._used
        return (
            f"budget {self.used()}/{self.total} "
            f"(sys {u['system']}/{self.quota.system}, "
            f"earlier {u['earlier']}/{self.quota.earlier}, "
            f"raw {u['raw']}/{self.quota.raw}, "
            f"tools {u['tools']}/{self.quota.tools})"
        )
