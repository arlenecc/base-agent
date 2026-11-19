"""SimHash(64) + 4 表 LSH 全局文本相似去重。

移植自 tools/SimHashDedupe.swift（Swift 命令行工具 → Python 库内联使用），
指纹算法与判定参数与原实现一一对应：

指纹生成：
    1) 代表文本：默认头/尾各 sample_chars（20000）字；清理只保留汉字
       （4E00-9FFF / 3400-4DBF / 20000-2A6DF / F900-FAFF）与英文小写。
    2) 双窗口扫描：k=3 统计 TF 取前 300，k=2 取前 200（同频按字典序，
       保证确定性）；过滤停用词 + 虚字边界碎片，特征权重 = 文档内频次。
    3) 每特征 MurmurHash3 x64_128（取 h1）→ 64 维逐位 ±权重累加，
       >0 记 1 → 64 位 SimHash 指纹。

LSH 索引与查询：
    指纹切 4 段各 16 位建 4 张倒排表；海明距离 ≤3 按鸽巢原理必有一段
    相同 → 必定召回。查询：4 表并集粗排 → XOR+popcount 精排 →
    3-gram Jaccard 内容复核（≥0.5 确认真重复）→ 重复不入库；
    否则判新文章，指纹与 4 段键值写入索引。

接入点（engine._ingest_locked）：解析/提取/清洗、转 Markdown 完成后，
切片入库之前；判重的文档整篇跳过（不切片、不嵌入、不入库）。
索引按知识分区持久化在 ``<rag_dir>/simhash_index.json``，随分区隔离。

与 Swift 版的差异（工程内使用所限）：
    - 未做文档级 DF 停用词抽样统计（原 --df-cut 批处理优化）：假阳性由
      内容复核兜底，假阴性不影响正确性；
    - 索引按源文件路径唯一（同路径重同步时替换旧指纹，而不是追加）。
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 参数（与 Swift Options 默认值一致）
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_CHARS = 20000   # 头/尾各取多少字（共 2×）
DEFAULT_TOP3 = 300             # k=3 高频特征数
DEFAULT_TOP2 = 200             # k=2 高频特征数
DEFAULT_HAM = 3                # 重复判定的海明距离上限（4 段 LSH 保证 ≤3 必召回）
DEFAULT_VERIFY_JACCARD = 0.5   # 内容复核 3-gram Jaccard 下限
_HASH_SEED = 0
_IDX_VERSION = 1

_M64 = 0xFFFFFFFFFFFFFFFF  # UInt64 掩码

# 出现在词首/词尾时几乎必然是跨词边界的碎片（CoreWordKit.weakBoundaryChars）
_WEAK_BOUNDARY_CHARS = frozenset(
    "的了着过得地是在和与或也就都而之以于其这那一不很更最可能会要将被把从对为所由"
    "并且但则者矣焉乎哉又再还才只已曾就很更最其之而且但则乃若即便却竟偏"
)

# 项目早期精选的中文功能词（CoreWordKit.legacyBuiltinStopwords）
_LEGACY_STOPWORDS = frozenset("""
我们 你们 他们 它们 咱们 自己 这个 那个 这些 那些 这种 那种 什么 怎么 怎样 为何 哪里
因为 所以 但是 然而 虽然 如果 假如 即使 尽管 而且 并且 或者 还是 以及 于是 因此 由于
关于 对于 通过 根据 按照 为了 除了 由于 关于 并且 况且 何况 与其 不如 不过 只是
可以 能够 应该 必须 需要 可能 似乎 其实 基本 主要 一般 一定 非常 十分 相当 比较
更加 逐渐 一直 已经 正在 曾经 仍然 依然 忽然 突然 显然 当然 自然 确实 实际上
进行 成为 作为 出现 存在 发生 产生 具有 包括 使用 采用 提供 实现 表示 认为 发现
一个 一种 一些 一次 一下 一点 一时 第一 第二 第三 这样 那样 如此 其中 之后 之前
以后 以前 现在 当时 同时 目前 开始 左右 上下 方面 情况 问题 时候 时间 地方 部分
没有 不是 就是 还有 也是 都是 一种 是一样的 为主 可见 看出 体现 甚至 改变 更多
先生 太太 小姐 地说 地说 起来 一样 到了 到了这种 到了那个 知道 我们 他们 什么
这一 那一 之一 之一 之中 之间 之内 之外 一般 一切 一切 所有 任何 每个 各个
这是一种 是一种 就可以了 就可以了 的时候 就是说 也就是说 事实上 实际上 总的 来说
""".split())


def _load_full_stopwords() -> frozenset:
    """加载包内汇总停用词表（tools/stopwords_full.txt 的编译期嵌入副本）。"""
    path = os.path.join(os.path.dirname(__file__), "stopwords_full.txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            words = {line.strip() for line in f if line.strip()}
    except OSError:  # pragma: no cover - 资源缺失时退回内置精选表
        return _LEGACY_STOPWORDS
    return frozenset(words)


_STOPWORDS: Optional[frozenset] = None


def get_stopwords() -> frozenset:
    global _STOPWORDS
    if _STOPWORDS is None:
        _STOPWORDS = _LEGACY_STOPWORDS | _load_full_stopwords()
    return _STOPWORDS


# ---------------------------------------------------------------------------
# MurmurHash3 x64_128（取 128 位输出的低 64 位 h1）— 与 Swift 版逐位一致
# ---------------------------------------------------------------------------

def murmur_hash64(data: bytes, seed: int = _HASH_SEED) -> int:
    c1 = 0x87C37B91114253D5
    c2 = 0x4CF5AD432745937F
    h1 = seed & _M64
    h2 = seed & _M64
    blocks = len(data) // 16
    for b in range(blocks):
        off = b * 16
        k1 = int.from_bytes(data[off:off + 8], "little")
        k2 = int.from_bytes(data[off + 8:off + 16], "little")
        k1 = (k1 * c1) & _M64
        k1 = ((k1 << 31) | (k1 >> 33)) & _M64
        k1 = (k1 * c2) & _M64
        h1 ^= k1
        h1 = ((h1 << 27) | (h1 >> 37)) & _M64
        h1 = (h1 + h2) & _M64
        h1 = (h1 * 5 + 0x52DCE729) & _M64
        k2 = (k2 * c2) & _M64
        k2 = ((k2 << 33) | (k2 >> 31)) & _M64
        k2 = (k2 * c1) & _M64
        h2 ^= k2
        h2 = ((h2 << 31) | (h2 >> 33)) & _M64
        h2 = (h2 + h1) & _M64
        h2 = (h2 * 5 + 0x38495AB5) & _M64
    off = blocks * 16
    tail = len(data) - off
    if tail > 0:
        if tail >= 9:
            k2 = int.from_bytes(data[off + 8:off + tail], "little")
            k2 = (k2 * c2) & _M64
            k2 = ((k2 << 33) | (k2 >> 31)) & _M64
            k2 = (k2 * c1) & _M64
            h2 ^= k2
        if tail >= 1:
            k1 = int.from_bytes(data[off:off + min(tail, 8)], "little")
            k1 = (k1 * c1) & _M64
            k1 = ((k1 << 31) | (k1 >> 33)) & _M64
            k1 = (k1 * c2) & _M64
            h1 ^= k1
    h1 ^= len(data)
    h2 ^= len(data)
    h1 = (h1 + h2) & _M64
    h2 = (h2 + h1) & _M64

    def _fmix64(k: int) -> int:
        k ^= k >> 33
        k = (k * 0xFF51AFD7ED558CCD) & _M64
        k ^= k >> 33
        k = (k * 0xC4CEB9FE1A85EC53) & _M64
        k ^= k >> 33
        return k

    h1 = _fmix64(h1)
    h2 = _fmix64(h2)
    h1 = (h1 + h2) & _M64
    h2 = (h2 + h1) & _M64
    return h1


def murmur_hash64_str(s: str, seed: int = _HASH_SEED) -> int:
    return murmur_hash64(s.encode("utf-8"), seed=seed)


# ---------------------------------------------------------------------------
# 文本清理与代表文本
# ---------------------------------------------------------------------------

def _kept(ch: str) -> Optional[str]:
    """汉字保留（4 个 CJK 区块）、A-Z 转小写、a-z 保留，其余剔除。"""
    v = ord(ch)
    if (0x4E00 <= v <= 0x9FFF or 0x3400 <= v <= 0x4DBF
            or 0x20000 <= v <= 0x2A6DF or 0xF900 <= v <= 0xFAFF):
        return ch
    if 0x41 <= v <= 0x5A:
        return chr(v + 32)
    if 0x61 <= v <= 0x7A:
        return ch
    return None


def clean_keep(text: str) -> str:
    if not text:
        return ""
    out = []
    for c in text:
        k = _kept(c)
        if k is not None:
            out.append(k)
    return "".join(out)


def _kept_indices_total(text: str) -> int:
    return sum(1 for c in text if _kept(c) is not None)


def representative_segments(text: str,
                            sample_chars: int = DEFAULT_SAMPLE_CHARS,
                            full_text: bool = False) -> List[str]:
    """代表文本分段：全文/短文本 → 1 段；否则头 sample + 尾 sample 两段。

    分段返回，避免首尾拼接处产生虚假跨段 k-gram；内存只占 2×sample_chars。
    """
    if full_text or sample_chars <= 0:
        return [clean_keep(text)]
    total = _kept_indices_total(text)
    if total <= 2 * sample_chars:
        return [clean_keep(text)]
    head: List[str] = []
    tail: List[str] = []
    idx = 0
    for c in text:
        if _kept(c) is None:
            continue
        if idx < sample_chars:
            head.append(c)
        if idx >= total - sample_chars:
            tail.append(c)
        idx += 1
    return ["".join(head), "".join(tail)]


# ---------------------------------------------------------------------------
# 特征工程（双窗口 + 停用词/虚字边界过滤 + topN，权重 = TF）
# ---------------------------------------------------------------------------

def _grams(segs: List[str], k: int, stops: frozenset,
           use_stopwords: bool = True) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for seg in segs:
        if len(seg) < k:
            continue
        for i in range(len(seg) - k + 1):
            g = seg[i:i + k]
            if use_stopwords:
                if g in stops:
                    continue
                if all(c in _WEAK_BOUNDARY_CHARS for c in g):
                    continue
            counts[g] = counts.get(g, 0) + 1
    return counts


def _top(counts: Dict[str, int], n: int) -> List[Tuple[str, int]]:
    """按频次降序取 topN（同频次按字典序，保证确定性）。"""
    if n <= 0 or not counts:
        return []
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ordered[:n]


def extract_features(text: str,
                     stops: Optional[frozenset] = None,
                     sample_chars: int = DEFAULT_SAMPLE_CHARS,
                     full_text: bool = False,
                     top3: int = DEFAULT_TOP3,
                     top2: int = DEFAULT_TOP2) -> List[Tuple[str, int]]:
    """最终特征集合 = k=3 top300 ∪ k=2 top200（shingles），权重 = 文档内 TF。"""
    if stops is None:
        stops = get_stopwords()
    segs = representative_segments(text, sample_chars=sample_chars,
                                   full_text=full_text)
    c3 = _grams(segs, 3, stops)
    c2 = _grams(segs, 2, stops)
    return _top(c3, top3) + _top(c2, top2)


# ---------------------------------------------------------------------------
# SimHash 指纹、海明距离与 LSH 分段
# ---------------------------------------------------------------------------

def make_fingerprint(features: List[Tuple[str, int]],
                     seed: int = _HASH_SEED) -> int:
    v = [0] * 64
    for word, weight in features:
        h = murmur_hash64_str(word, seed=seed)
        for i in range(64):
            v[i] += weight if (h >> i) & 1 else -weight
    fp = 0
    for i in range(64):
        if v[i] > 0:
            fp |= 1 << i
    return fp


def hamming(a: int, b: int) -> int:
    return bin((a ^ b) & _M64).count("1")


def band(fp: int, b: int) -> int:
    """第 b 段（0..3）的 16 位值：b=0 取最低 16 位（对应 Swift 表1=1-16 位）。"""
    return (fp >> (16 * b)) & 0xFFFF


def collides(a: int, b: int) -> bool:
    return any(band(a, i) == band(b, i) for i in range(4))


# ---------------------------------------------------------------------------
# 内容复核：头尾代表文本的 3-gram Jaccard
# ---------------------------------------------------------------------------

def _trigrams(text: str, sample_chars: int = DEFAULT_SAMPLE_CHARS) -> set:
    s: set = set()
    for seg in representative_segments(text, sample_chars=sample_chars,
                                       full_text=sample_chars <= 0):
        if len(seg) < 3:
            continue
        for i in range(len(seg) - 2):
            s.add(seg[i:i + 3])
    return s


def verify_similar(a_text: str, b_text: str,
                   threshold: float = DEFAULT_VERIFY_JACCARD,
                   sample_chars: int = DEFAULT_SAMPLE_CHARS) -> bool:
    """SimHash 距离 1..ham 时做内容复核，剔除指纹碰撞误判。"""
    a = _trigrams(a_text, sample_chars)
    b = _trigrams(b_text, sample_chars)
    inter = len(a & b)
    union = len(a) + len(b) - inter
    if union <= 0:
        return False
    return inter / union >= threshold


# ---------------------------------------------------------------------------
# LSH 索引（持久化 + 线程安全的查询即入库）
# ---------------------------------------------------------------------------

class SimHashIndex:
    """SimHash 指纹的 4×16 位 LSH 索引，JSON 持久化。

    线程安全：``check_and_insert`` 是「查询 + 插入」的原子操作——同步管线
    的多个 worker 线程并发调用时不会竞态（两份相同内容同时进来只有一个
    判为新并入库）。
    """

    def __init__(self,
                 path: str = "",
                 ham: int = DEFAULT_HAM,
                 verify_jaccard: float = DEFAULT_VERIFY_JACCARD,
                 sample_chars: int = DEFAULT_SAMPLE_CHARS,
                 md_dir: str = "") -> None:
        self.path = path
        self.ham = max(0, min(64, int(ham)))
        self.verify_jaccard = float(verify_jaccard)
        self.sample_chars = int(sample_chars)
        self.md_dir = md_dir          # 内容复核读取 owner markdown 的目录
        self.docs: List[Dict[str, object]] = []   # {path, md, fp(16进制), size}
        self.tables: List[Dict[str, List[int]]] = [
            {} for _ in range(4)]     # 段键值 → 文档 ID 列表
        self.dirty = False
        self._lock = threading.RLock()

    # -- 参数指纹：参数变化时旧索引整体作废（指纹不可比） --

    def _params(self) -> str:
        return (f"v={_IDX_VERSION}|sample={self.sample_chars}"
                f"|top3={DEFAULT_TOP3}|top2={DEFAULT_TOP2}"
                f"|ham={self.ham}|jaccard={self.verify_jaccard}"
                f"|seed={_HASH_SEED}")

    # -- 持久化 --

    @classmethod
    def load(cls, path: str, md_dir: str = "", **params) -> "SimHashIndex":
        idx = cls(path=path, md_dir=md_dir, **params)
        if not path or not os.path.isfile(path):
            return idx
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            logger.warning("simhash 索引读取失败（按空索引继续）: %s: %s", path, e)
            return idx
        try:
            if (data.get("version") != _IDX_VERSION
                    or data.get("params") != idx._params()):
                logger.info("simhash 索引参数不匹配（旧索引作废重建）: %s", path)
                return idx
            idx.docs = list(data.get("docs") or [])
            tables = data.get("tables") or []
            idx.tables = [
                {str(k): list(v) for k, v in (tables[b] or {}).items()
                 if isinstance(v, list)}
                for b in range(4)
            ]
            logger.info("simhash 索引已加载: %d 篇文档 (%s)", len(idx.docs), path)
        except Exception as e:  # pragma: no cover - 结构损坏时按空索引继续
            logger.warning("simhash 索引结构异常（按空索引继续）: %s: %s", path, e)
            idx.docs = []
            idx.tables = [{} for _ in range(4)]
        return idx

    def save(self) -> None:
        if not self.path:
            return
        payload = {
            "version": _IDX_VERSION,
            "params": self._params(),
            "docs": self.docs,
            "tables": self.tables,
        }
        tmp = self.path + ".tmp"
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp, self.path)
            self.dirty = False
        except OSError as e:
            logger.warning("simhash 索引写入失败: %s: %s", self.path, e)

    def __len__(self) -> int:
        return len(self.docs)

    # -- 索引操作（内部调用须持锁） --

    def _fp_of(self, doc_id: int) -> int:
        return int(self.docs[doc_id]["fp"], 16)

    def _candidates(self, fp: int) -> set:
        out: set = set()
        for b in range(4):
            ids = self.tables[b].get(str(band(fp, b)))
            if ids:
                out.update(ids)
        return out

    def _insert(self, fp: int, path: str, md_name: str = "", size: int = 0) -> int:
        self._remove_path_unlocked(path)  # 同路径唯一：重同步时替换旧指纹
        doc_id = len(self.docs)
        self.docs.append({"path": path, "md": md_name,
                          "fp": format(fp, "x"), "size": int(size)})
        for b in range(4):
            key = str(band(fp, b))
            self.tables[b].setdefault(key, []).append(doc_id)
        self.dirty = True
        return doc_id

    def _remove_path_unlocked(self, path: str) -> int:
        gone = [i for i, d in enumerate(self.docs) if d.get("path") == path]
        if not gone:
            return 0
        gone_set = set(gone)
        # 重建紧凑 ID（表里存的是文档 ID，直接删 ID 会有空洞）
        remap: Dict[int, int] = {}
        new_docs: List[Dict[str, object]] = []
        for i, d in enumerate(self.docs):
            if i not in gone_set:
                remap[i] = len(new_docs)
                new_docs.append(d)
        self.docs = new_docs
        for b in range(4):
            rebuilt: Dict[str, List[int]] = {}
            for key, ids in self.tables[b].items():
                kept = [remap[i] for i in ids if i in remap]
                if kept:
                    rebuilt[key] = kept
            self.tables[b] = rebuilt
        self.dirty = True
        return len(gone)

    def remove_path(self, path: str) -> int:
        """源文件被删除/清理时移除其指纹条目（返回移除的条目数）。"""
        with self._lock:
            return self._remove_path_unlocked(path)

    def _read_owner_text(self, doc_id: int) -> Optional[str]:
        """读 owner 的 Markdown 缓存作复核材料；缺失返回 None（不判重）。"""
        md_name = str(self.docs[doc_id].get("md") or "")
        if not md_name or not self.md_dir:
            return None
        try:
            with open(os.path.join(self.md_dir, md_name), "r",
                      encoding="utf-8") as f:
                return f.read()
        except OSError:
            return None

    # -- 主入口 --

    def check_and_insert(self, text: str, path: str,
                         md_name: str = "", size: int = 0) -> Tuple[bool, str, int]:
        """原子「查询 + 入库」：判重则返回 (True, owner, dist)；新文章写入
        索引并返回 (False, "", 0)。

        - 距离 ≤ ham 的候选先做 3-gram Jaccard 内容复核，复核不通过视为
          指纹碰撞误判（内容其实不同 → 继续判新）；
        - owner 的 Markdown 缓存缺失（复核材料拿不到）时不判重——宁可
          多入库一份也不丢内容；
        - 距离 0 的自身旧条目（上次同步中断遗留的同路径指纹）不算重复。
        """
        feats = extract_features(text, sample_chars=self.sample_chars)
        fp = make_fingerprint(feats)
        with self._lock:
            best: Optional[Tuple[str, int]] = None
            for doc_id in self._candidates(fp):
                d = hamming(fp, self._fp_of(doc_id))
                if d > self.ham:
                    continue
                owner_path = str(self.docs[doc_id].get("path") or "")
                if owner_path == path:
                    continue  # 自身旧条目（中断遗留）：放行并替换
                if d > 0:
                    owner_text = self._read_owner_text(doc_id)
                    if owner_text is None:
                        logger.info(
                            "simhash 复核材料缺失（owner 缓存不存在），不判重: %s",
                            owner_path)
                        continue
                    if not verify_similar(text, owner_text,
                                          threshold=self.verify_jaccard,
                                          sample_chars=self.sample_chars):
                        continue  # 指纹碰撞误判：内容其实不同
                if best is None or d < best[1]:
                    best = (owner_path, d)
            if best is not None:
                return (True, best[0], best[1])
            self._insert(fp, path, md_name=md_name, size=size)
            return (False, "", 0)


# ---------------------------------------------------------------------------
# 指纹便捷入口（测试 / 外部查询用）
# ---------------------------------------------------------------------------

def fingerprint_text(text: str, **kwargs) -> int:
    """文本 → 64 位 SimHash 指纹（一次完成特征提取 + 加权累加）。"""
    return make_fingerprint(extract_features(text, **kwargs))
