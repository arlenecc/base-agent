// ClusterBooks.swift
//
// 批量文档自动领域分类：文档特征 → MinHash(128) → LSH(32 bands × 4) → 精确 Jaccard
// → 加权稀疏图 → Leiden 社区发现（支持目标簇数 + γ 二分搜索）→ 簇命名 → 归类文档。
//
// 文档特征可选两套方案（--scheme）：
//   core-words    方案一：N-gram + NPMI + 左右熵提取的核心词；
//   char-shingles 方案二：清理文本（汉字 + 小写字母）后做字符级 4-shingle，
//                 去重、过滤停用词、bottom-k 采样，得到最终特征集合。
// 两套方案共用同一条 MinHash / LSH / Jaccard / Leiden 流水线，便于直接对比。
//
// 算法实现见 CoreWordKit.swift（复用第一个工具的 N-gram + NPMI + 左右熵提取器）。
// 纯 Foundation，零第三方依赖。
//
// 编译：
//   swiftc -O -o /tmp/bookcluster tools/CoreWordKit.swift tools/ClusterBooks.swift
//
// 典型用法（默认 dry-run，不移动任何文件）：
//   /tmp/bookcluster /Volumes/SSD/books-md --limit 500 --max-chars 20000 --report /tmp/report.json
//   /tmp/bookcluster /Volumes/SSD/books-md --apply          # 确认后真正移动
//   /tmp/bookcluster --undo /Volumes/SSD/books-md/.bookcluster_manifest.json
//   /tmp/bookcluster --self-test
//
// ---------------------------------------------------------------------------
// 设计要点
// - MinHash：128 个独立哈希函数 h_i(x) = (a_i·x + b_i) mod (2^61-1)，
//   (a_i,b_i) 由确定性 SplitMix64 生成，保证跨运行可复现；每个函数对词集合
//   只保留最小哈希值 → 128 维整数签名。
// - LSH：签名切 32 段，每段 4 维作为一个 key 分桶，同桶文档对成为候选；
//   段更短 ⇒ 碰撞概率 b^(1/r) 更低、召回阈值更宽松（更易召回中等相似文档），
//   代价是候选对变多，由精确 Jaccard 兜底。桶过大时跳过（--max-bucket-size），
//   避免 O(n²) 爆炸；候选对用 UInt64 编码去重。
// - 相似度：候选对用「词 ID 排序数组」双指针求精确 Jaccard；sim > 阈值才连边，
//   sim 作为边权。文档 = 节点，相似关系 = 边，得到加权无向稀疏图。
// - Leiden：局部移动（加权模块度）→ 连通性细化（refinement，从 singleton 重新
//   聚合）→ 图聚合，多层迭代直到划分稳定；孤立节点自动单独成簇。
// - 目标簇数：γ 越大惩罚项越大、社区越碎（簇数越多）。据此在 [γmin, γmax]
//   上做对数二分，用「簇数-目标」的符号收缩区间，保留历史最优点；
//   簇数对 γ 非严格单调（离散 + 局部最优），故用容差 + 最优点兜底。
// - 簇名：簇内所有文档核心词按「文档频次优先、累计词频次之」合并排序取 top3。
// - 安全：默认 dry-run；--apply 才移动；每次移动写 manifest，--undo 可完整回退。
// ---------------------------------------------------------------------------

import Foundation
import Dispatch

// MARK: - 常量

private let signatureDim = 128
private let bandCount = 32
private let bandRows = 4
private let mersenne61: UInt64 = (1 << 61) - 1

// 32 × 4 必须恰好覆盖 128 维签名，否则末段越界/漏维。
private let bandLayoutIsValid = (bandCount * bandRows == signatureDim) && (bandRows > 0)

// MARK: - 哈希基础

/// 确定性 PRNG（SplitMix64），用于生成 128 组独立哈希函数参数。
struct SplitMix64 {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state = state &+ 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// 字符串 → 64 位稳定哈希（FNV-1a）。
func fnv1a64(_ text: String) -> UInt64 {
    var h: UInt64 = 0xCBF2_9CE4_8422_2325
    for byte in text.utf8 {
        h ^= UInt64(byte)
        h = h &* 0x100_0000_01B3
    }
    return h
}

@inline(__always)
func mod61(_ x: UInt64) -> UInt64 {
    var v = (x & mersenne61) &+ (x >> 61)
    if v >= mersenne61 { v &-= mersenne61 }
    return v
}

/// (a*b) mod (2^61-1)：用 multipliedFullWidth 避免 128 位溢出，
/// 再利用 2^61 ≡ 1 ⇒ 2^64 ≡ 8 (mod prime) 快速约简。
@inline(__always)
func mulMod61(_ a: UInt64, _ b: UInt64) -> UInt64 {
    let m = a.multipliedFullWidth(by: b)
    return mod61(mod61(m.low) &+ mod61(m.high &* 8))
}

/// 128 组 (a,b)，固定 seed 保证不同运行、不同输入顺序得到同一套函数。
struct HashFamily {
    let a: [UInt64]
    let b: [UInt64]

    init(count: Int = signatureDim, seed: UInt64 = 0x42A1_9C3D_7E55_1B07) {
        var rng = SplitMix64(seed: seed)
        var aa: [UInt64] = []
        var bb: [UInt64] = []
        aa.reserveCapacity(count)
        bb.reserveCapacity(count)
        for _ in 0..<count {
            // a ∈ [1, prime-1]，b ∈ [0, prime-1]
            aa.append(mod61(rng.next()) | 1)
            bb.append(mod61(rng.next()))
        }
        a = aa
        b = bb
    }

    @inline(__always)
    func value(_ i: Int, x: UInt64) -> UInt64 {
        mod61(mulMod61(a[i], x) &+ b[i])
    }
}

// MARK: - MinHash

enum MinHash {
    /// 对一组词生成 128 维签名：每个哈希函数只保留最小哈希值。
    /// 空集合返回全 prime（不会被任何 LSH 桶当成有效相似信号）。
    static func signature(_ words: [String], family: HashFamily) -> [UInt64] {
        var sig = [UInt64](repeating: mersenne61, count: signatureDim)
        guard !words.isEmpty else { return sig }
        // 词去重后取稳定哈希
        var xs: [UInt64] = []
        xs.reserveCapacity(words.count)
        var seen = Set<UInt64>()
        for w in words {
            let x = mod61(fnv1a64(w))
            if seen.insert(x).inserted { xs.append(x) }
        }
        for x in xs {
            for i in 0..<signatureDim {
                let h = family.value(i, x: x)
                if h < sig[i] { sig[i] = h }
            }
        }
        return sig
    }

    /// 从已算好的特征哈希直接生成签名（方案二用，避免对每个 shingle 重复 FNV）。
    static func signatureFromHashes(_ xs: [UInt64], family: HashFamily) -> [UInt64] {
        var sig = [UInt64](repeating: mersenne61, count: signatureDim)
        guard !xs.isEmpty else { return sig }
        for x in xs {
            for i in 0..<signatureDim {
                let h = family.value(i, x: x)
                if h < sig[i] { sig[i] = h }
            }
        }
        return sig
    }

    /// 由签名估计的 Jaccard（相等分量占比），仅用于诊断/自检。
    static func estimatedJaccard(_ a: [UInt64], _ b: [UInt64]) -> Double {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var same = 0
        for i in 0..<a.count where a[i] == b[i] { same += 1 }
        return Double(same) / Double(a.count)
    }
}

// MARK: - LSH

enum LSH {
    /// 16 段 × 8 维，每段作为 key 分桶，返回去重的候选对（编码成 UInt64）。
    /// - 超过 maxBucketSize 的桶跳过（常见词签名造成的巨桶，噪声且会 O(n²)）。
    static func candidatePairs(_ signatures: [[UInt64]], maxBucketSize: Int,
                               maxPairs: Int) -> (pairs: [UInt64], skippedBuckets: Int) {
        guard bandLayoutIsValid else { return ([], 0) }
        var buckets: [[UInt64]: [Int]] = [:]
        for (idx, sig) in signatures.enumerated() {
            guard sig.count == signatureDim else { continue }
            // 空词集合的签名全是 prime 哨兵：它们彼此并不相似，
            // 若不跳过，成千上万本无核心词的小书会挤进同一个巨桶。
            if sig.allSatisfy({ $0 == mersenne61 }) { continue }
            for band in 0..<bandCount {
                let start = band * bandRows
                let key = Array(sig[start..<(start + bandRows)])
                buckets[key, default: []].append(idx)
            }
        }
        var pairs = Set<UInt64>()
        var skipped = 0
        for (_, members) in buckets {
            if members.count < 2 { continue }
            if members.count > maxBucketSize { skipped += 1; continue }
            for i in 0..<members.count {
                for j in (i + 1)..<members.count {
                    let lo = UInt64(min(members[i], members[j]))
                    let hi = UInt64(max(members[i], members[j]))
                    pairs.insert((lo << 32) | hi)
                    if pairs.count >= maxPairs { return (Array(pairs), skipped) }
                }
            }
        }
        return (Array(pairs), skipped)
    }

    @inline(__always)
    static func decode(_ pair: UInt64) -> (Int, Int) {
        (Int(pair >> 32), Int(pair & 0xFFFF_FFFF))
    }
}

// MARK: - 文档特征方案

/// 文档特征的两套方案，共用同一条 MinHash → LSH → Jaccard → Leiden 流水线。
enum FeatureScheme: String, Codable {
    /// 方案一：N-gram + NPMI + 左右熵提取的核心词（CoreWordKit）。
    case coreWords = "core-words"
    /// 方案二：清理后的纯文字做字符级 k-shingle（默认 k=4）。
    case charShingles = "char-shingles"
}

/// 方案二：文本清理 + 字符级 shingle + 停用词过滤 + bottom-k 采样。
///
/// 与方案一不同，字符 4-gram 的数量与文本长度同阶（3 万字 → 约 3 万个片段），
/// 全量 6.6 万本直接保留全部片段会占用数十 GB，且 MinHash 计算不可行。
/// 因此按「固定哈希值最小的 K 个」做确定性 bottom-k sketch（k-mins），
/// 这是 MinHash 的标准等价近似：两集合的 Jaccard 在 sketch 上仍可无偏估计。
/// K=0 表示不裁剪（仅适合小样本）。
enum CharShingleFeatures {
    /// 只保留汉字与英文字母（转小写）。标点、空格、换行、数字、其它符号一律剔除。
    static func clean(_ text: String, maxChars: Int) -> [Character] {
        var out: [Character] = []
        for scalar in text.unicodeScalars {
            let v = scalar.value
            if (v >= 0x4E00 && v <= 0x9FFF)
                || (v >= 0x3400 && v <= 0x4DBF)
                || (v >= 0x20000 && v <= 0x2A6DF)
                || (v >= 0xF900 && v <= 0xFAFF) {
                out.append(Character(scalar))
            } else if v >= 0x41 && v <= 0x5A {
                if let lower = UnicodeScalar(v + 32) { out.append(Character(lower)) }
            } else if v >= 0x61 && v <= 0x7A {
                out.append(Character(scalar))
            }
            if maxChars > 0 && out.count >= maxChars { break }
        }
        return out
    }

    /// 从头到尾滑动窗口取长度 k 的子串，Set 自动去重。
    static func shingles(_ chars: [Character], k: Int) -> Set<String> {
        guard k > 0, chars.count >= k else { return [] }
        var result = Set<String>()
        result.reserveCapacity(min(chars.count - k + 1, 65_536))
        for i in 0...(chars.count - k) {
            result.insert(String(chars[i..<(i + k)]))
        }
        return result
    }

    /// 纯 ASCII 的高频功能片段（清理后无空格，故为连续字母串），
    /// 这些 4-gram 几乎出现在所有英文书里，属于字符级「停用词」。
    static let asciiStopShingles: Set<String> = [
        "thei", "ethe", "andh", "ofth", "toth", "inth", "fort", "wit", "tion", "atio",
        "ment", "that", "this", "with", "from", "have", "whic", "here", "ther", "ould",
        "ight", "ough", "thes", "othe", "nthe", "ethe", "dthe", "sthe", "ethe", "ofth",
        "ingt", "sand", "dand", "eand", "hand", "sthi", "stha", "swit", "sfro", "eof",
    ]

    /// 过滤停用词：
    ///  (a) shingle 整体命中内置/自定义停用词表；
    ///  (b) 4 个字符全是虚字（「的了是在」这类纯功能片段）；
    ///  (c) 纯 ASCII 且命中英文功能片段表。
    static func filterStopwords(_ shingles: Set<String>, k: Int,
                                extra: Set<String>) -> Set<String> {
        let wordStops = WordFilters.builtinStopwords.filter { $0.count == k }
        var out = Set<String>()
        out.reserveCapacity(shingles.count)
        for s in shingles {
            if wordStops.contains(s) || extra.contains(s) { continue }
            if s.unicodeScalars.allSatisfy({ scalar in
                WordFilters.weakBoundaryChars.contains(Character(scalar))
            }) { continue }
            if s.utf8.allSatisfy({ $0 >= 97 && $0 <= 122 }) && asciiStopShingles.contains(s) {
                continue
            }
            out.insert(s)
        }
        return out
    }

    /// bottom-k：按稳定哈希取最小 K 个，返回 (文本, 哈希) 对，哈希升序便于 Jaccard 双指针。
    static func bottomK(_ shingles: Set<String>, limit: Int)
        -> (words: [String], hashes: [UInt64]) {
        let scored: [(String, UInt64)] = shingles.map { ($0, mod61(fnv1a64($0))) }
            .sorted { $0.1 == $1.1 ? $0.0 < $1.0 : $0.1 < $1.1 }
        let kept = limit > 0 ? Array(scored.prefix(limit)) : scored
        return (kept.map(\.0), kept.map(\.1))
    }

    /// 对单个 shingle 判定是否为停用片段（命中停用词 / 全虚字 / 英文功能片段）。
    static func isStop(_ s: String, k: Int, extra: Set<String>) -> Bool {
        if WordFilters.builtinStopwords.contains(s) || extra.contains(s) { return true }
        if s.unicodeScalars.allSatisfy({ WordFilters.weakBoundaryChars.contains(Character($0)) }) {
            return true
        }
        if s.utf8.allSatisfy({ $0 >= 97 && $0 <= 122 }) && asciiStopShingles.contains(s) {
            return true
        }
        return false
    }

    /// 文档级停用词：剔除出现在 ≥ cut 比例文档中的高频 shingle。
    ///
    /// 这些片段（如「一个什么」「这样才能」）几乎每本书都有，毫无区分度，
    /// 相当于 TF-IDF 里 DF 极高的语料级停用词。DF 统计用排序数组而非字典：
    /// 全量 1.3 亿特征只占约 1 GB，且排序后线性扫描即可数出高频段。
    /// 同时从命名集合（words）中剔除对应片段，避免簇名被通用片段占据。
    /// 返回被剔除的唯一 shingle 数。
    static func filterCommonShingles(_ docs: inout [BookDocument], cut: Double) -> Int {
        guard cut > 0, !docs.isEmpty else { return 0 }
        let cutoff = max(2, Int((Double(docs.count) * cut).rounded(.up)))
        var all = [UInt64]()
        all.reserveCapacity(docs.reduce(0) { $0 + $1.featureIDs.count })
        for d in docs { all.append(contentsOf: d.featureIDs) }
        guard !all.isEmpty else { return 0 }
        all.sort()
        var common = Set<UInt64>()
        var i = 0
        while i < all.count {
            var j = i
            while j < all.count && all[j] == all[i] { j += 1 }
            if j - i >= cutoff { common.insert(all[i]) }
            i = j
        }
        guard !common.isEmpty else { return 0 }
        for idx in docs.indices {
            docs[idx].featureIDs.removeAll { common.contains($0) }
            docs[idx].words.removeAll { common.contains(mod61(fnv1a64($0))) }
            docs[idx].freqs = [Int](repeating: 1, count: docs[idx].words.count)
        }
        return common.count
    }

    /// 一步到位：清理 → 4-gram 词频统计 → 停用词过滤 →
    ///   - `hashes`：唯一 shingle 按哈希最小取 K 个（bottom-k 散点，用于相似度，规模可控）；
    ///   - `labels`：唯一 shingle 按**词频 TF 降序**取前 L 个（密集、主题相关，用于簇名拼接）。
    /// 两者分离：bottom-k 散点之间几乎无相邻重叠，无法拼回连续片段；而高频密集 shingle
    /// 来自同一段落/主题，跨书共享时重叠多，assembleFragments 能拼出较长可读片段。
    static func extract(_ text: String, maxChars: Int, k: Int, limit: Int, labelLimit: Int,
                        useStopwords: Bool, extraStopwords: Set<String>)
        -> (labels: [String], hashes: [UInt64], rawShingles: Int, chars: Int) {
        let chars = clean(text, maxChars: maxChars)
        // 词频统计：每个 4-gram 的全文出现次数（不去重计数，TF 才能区分主题高频片段）。
        var tf: [String: Int] = [:]
        if chars.count >= k {
            for i in 0...(chars.count - k) {
                tf[String(chars[i..<(i + k)]), default: 0] += 1
            }
        }
        let rawCount = tf.count
        let kept: [(String, Int)] = tf.compactMap { (s, c) in
            (useStopwords && isStop(s, k: k, extra: extraStopwords)) ? nil : (s, c)
        }
        // 命名集合：TF 降序取前 labelLimit（密集、主题相关）。
        let labels = kept
            .sorted { $0.1 != $1.1 ? $0.1 > $1.1 : $0.0 < $1.0 }
            .prefix(labelLimit)
            .map(\.0)
        // 相似度集合：唯一 shingle 哈希升序取前 limit（bottom-k）。
        var scored: [(s: String, h: UInt64)] = []
        scored.reserveCapacity(kept.count)
        for (s, _) in kept { scored.append((s: s, h: mod61(fnv1a64(s)))) }
        scored.sort { $0.h == $1.h ? $0.s < $1.s : $0.h < $1.h }
        let picked = limit > 0 ? Array(scored.prefix(limit)) : scored
        let hashes = picked.map { $0.h }
        return (labels, hashes, rawCount, chars.count)
    }
}

// MARK: - 文档与词表

struct BookDocument {
    let path: String
    let name: String
    var words: [String]            // 可读特征（核心词 或 shingle 文本），用于簇命名
    var freqs: [Int]               // 特征权重（核心词频次；shingle 统一为 1）
    var shingleHashes: [UInt64]?   // 方案二：已排序的 shingle 哈希（缓存可复用）
    var rawShingles: Int = 0       // 方案二：裁剪前原始 4-gram 去重数
    var featureIDs: [UInt64] = []  // 排序去重的特征哈希，用于精确 Jaccard
    var signature: [UInt64] = []
}

/// 词 → 全局 ID；用 Int32 排序数组存每本书的词集合，比 Set<String> 省一个数量级内存。
final class WordTable {
    private var ids: [String: Int32] = [:]
    func id(_ word: String) -> Int32 {
        if let existing = ids[word] { return existing }
        let newID = Int32(ids.count)
        ids[word] = newID
        return newID
    }
    var count: Int { ids.count }
}

/// 精确 Jaccard：两个已排序、去重的 UInt64 特征数组，双指针求交集。
/// 方案一传全局词 ID（转 UInt64），方案二在 --shingle-exact 下传 shingle 哈希。
@inline(__always)
func exactJaccard(_ a: [UInt64], _ b: [UInt64]) -> Double {
    if a.isEmpty || b.isEmpty { return 0 }
    var i = 0, j = 0, inter = 0
    while i < a.count && j < b.count {
        if a[i] == b[j] { inter += 1; i += 1; j += 1 }
        else if a[i] < b[j] { i += 1 }
        else { j += 1 }
    }
    let union = a.count + b.count - inter
    return union > 0 ? Double(inter) / Double(union) : 0
}

/// k-mins（bottom-k sketch）的 Jaccard 无偏估计。
///
/// 两个集合各保留哈希最小的 k 个元素后，直接对两份 sketch 求 Jaccard 会
/// 系统性低估（并集里较大的哈希被各自截掉了）。正确做法：以
/// t = min(max(A_k), max(B_k)) 为公共阈值，只统计哈希 ≤ t 的部分——
/// 该范围内两集合的元素都完整保留在 sketch 里，交并比即为原 Jaccard 的无偏估计。
@inline(__always)
func minsJaccard(_ a: [UInt64], _ b: [UInt64]) -> Double {
    if a.isEmpty || b.isEmpty { return 0 }
    let t = min(a[a.count - 1], b[b.count - 1])
    var i = 0, j = 0, inter = 0, unionCount = 0
    // 一侧越过阈值/耗尽时，另一侧剩余 ≤t 的元素仍属于公共阈值内的并集。
    while (i < a.count && a[i] <= t) || (j < b.count && b[j] <= t) {
        let aValid = i < a.count && a[i] <= t
        let bValid = j < b.count && b[j] <= t
        if !bValid {
            unionCount += 1; i += 1
        } else if !aValid {
            unionCount += 1; j += 1
        } else if a[i] == b[j] {
            inter += 1; unionCount += 1; i += 1; j += 1
        } else if a[i] < b[j] {
            unionCount += 1; i += 1
        } else {
            unionCount += 1; j += 1
        }
    }
    return unionCount > 0 ? Double(inter) / Double(unionCount) : 0
}

// MARK: - 加权图

struct WeightedGraph {
    let n: Int
    /// 邻接表：每条无向边在两端各存一次；自环（社区内边权）单独记在 loop 里。
    var adj: [[(Int, Double)]]
    var loops: [Double]
    var degree: [Double]
    var twoM: Double

    init(n: Int, edges: [(Int, Int, Double)]) {
        self.n = n
        var a = [[(Int, Double)]](repeating: [], count: n)
        var lp = [Double](repeating: 0, count: n)
        var deg = [Double](repeating: 0, count: n)
        var m2 = 0.0
        for (u, v, w) in edges {
            guard w > 0 else { continue }
            if u == v {
                lp[u] += w
                deg[u] += 2 * w
                m2 += 2 * w
            } else {
                a[u].append((v, w))
                a[v].append((u, w))
                deg[u] += w
                deg[v] += w
                m2 += 2 * w
            }
        }
        adj = a
        loops = lp
        degree = deg
        twoM = m2
    }

    var edgeCount: Int {
        adj.reduce(0) { $0 + $1.count } / 2 + loops.filter { $0 > 0 }.count
    }
}

// MARK: - Leiden 社区发现

/// Leiden 风格的三阶段多层实现：局部移动（加权模块度）→ 连通性细化 → 图聚合。
/// 细化阶段把每个社区内部从 singleton 重新局部聚合，是 Leiden 相对 Louvain
/// 的关键改进：保证最终社区内部连通，避免把不相连的团塞进同一个簇。
enum Leiden {
    struct Result {
        /// 每个原始节点 → 社区 ID（0..<communityCount）
        let assignment: [Int]
        let communityCount: Int
        let levels: Int
    }

    /// 局部移动：把节点移到使加权模块度增益最大的邻居社区。
    /// ΔQ ∝ k(i,c) - γ·k_i·tot(c)/(2m)，tot(c) 为移除 i 之后的社区总度。
    private static func localMoving(_ g: WeightedGraph, initial: [Int]?,
                                    resolution: Double, maxIter: Int) -> [Int] {
        let n = g.n
        var comm = initial ?? Array(0..<n)
        let twoM = max(g.twoM, 1e-12)
        var tot = [Double](repeating: 0, count: n)
        for i in 0..<n { tot[comm[i]] += g.degree[i] }

        for _ in 0..<maxIter {
            var changed = false
            for i in 0..<n {
                let old = comm[i]
                let ki = g.degree[i]
                if ki <= 0 { continue }               // 孤立节点保持自己的社区
                tot[old] -= ki

                // 统计 i 到各邻居社区的边权和
                var wmap: [Int: Double] = [:]
                for (v, w) in g.adj[i] {
                    wmap[comm[v], default: 0] += w
                }

                var best = old
                var bestGain = (wmap[old] ?? 0) - resolution * ki * tot[old] / twoM
                for (c, w) in wmap where c != old {
                    let gain = w - resolution * ki * tot[c] / twoM
                    if gain > bestGain + 1e-12 { bestGain = gain; best = c }
                }
                comm[i] = best
                tot[best] += ki
                if best != old { changed = true }
            }
            if !changed { break }
        }
        return comm
    }

    /// 压缩社区编号到连续区间，返回 (映射, 社区数)。
    private static func renumber(_ comm: [Int]) -> ([Int], Int) {
        var map: [Int: Int] = [:]
        var out = [Int](repeating: 0, count: comm.count)
        for (i, c) in comm.enumerated() {
            if let v = map[c] { out[i] = v }
            else { let v = map.count; map[c] = v; out[i] = v }
        }
        return (out, map.count)
    }

    /// 由社区划分构建聚合图：跨社区边权累加，社区内边权记为该超节点的自环。
    private static func aggregate(_ g: WeightedGraph, comm: [Int], communities: Int)
        -> WeightedGraph {
        var weights: [UInt64: Double] = [:]
        @inline(__always) func key(_ a: Int, _ b: Int) -> UInt64 {
            let lo = UInt64(min(a, b)), hi = UInt64(max(a, b))
            return (lo << 32) | hi
        }
        for u in 0..<g.n {
            for (v, w) in g.adj[u] where u < v {
                let cu = comm[u], cv = comm[v]
                weights[key(cu, cv), default: 0] += w
            }
            if g.loops[u] > 0 { weights[key(comm[u], comm[u]), default: 0] += g.loops[u] }
        }
        var edges: [(Int, Int, Double)] = []
        edges.reserveCapacity(weights.count)
        for (k, w) in weights { edges.append((Int(k >> 32), Int(k & 0xFFFF_FFFF), w)) }
        return WeightedGraph(n: communities, edges: edges)
    }

    /// 细化阶段：对某个社区的节点集合，从 singleton 重新做「只允许组内合并」
    /// 的局部移动。局部社区编号是 0..<k-1，返回结果按传入顺序对齐。
    /// 这是 Leiden 相对 Louvain 的关键改进：不连通的团不会被塞进同一个社区。
    private static func refineGroup(_ g: WeightedGraph, members: [Int],
                                    resolution: Double, maxIter: Int) -> [Int] {
        let k = members.count
        if k <= 1 { return [0] }
        var localIndex: [Int: Int] = [:]
        for (idx, node) in members.enumerated() { localIndex[node] = idx }
        var comm = Array(0..<k)
        var tot = [Double](repeating: 0, count: k)
        for (idx, node) in members.enumerated() { tot[idx] = g.degree[node] }
        let twoM = max(g.twoM, 1e-12)

        for _ in 0..<maxIter {
            var changed = false
            for (idx, node) in members.enumerated() {
                let old = comm[idx]
                let ki = g.degree[node]
                if ki <= 0 { continue }
                tot[old] -= ki
                var wmap: [Int: Double] = [:]
                for (v, w) in g.adj[node] {
                    guard let li = localIndex[v] else { continue }
                    wmap[comm[li], default: 0] += w
                }
                var best = old
                var bestGain = (wmap[old] ?? 0) - resolution * ki * tot[old] / twoM
                for (c, w) in wmap where c != old {
                    let gain = w - resolution * ki * tot[c] / twoM
                    if gain > bestGain + 1e-12 { bestGain = gain; best = c }
                }
                comm[idx] = best
                tot[best] += ki
                if best != old { changed = true }
            }
            if !changed { break }
        }
        return comm
    }

    /// 主流程：多层「局部移动 → 连通性细化 → 聚合」迭代。
    static func cluster(_ graph: WeightedGraph, resolution: Double = 1.0,
                        maxIter: Int = 20, maxLevels: Int = 12) -> Result {
        let n = graph.n
        if graph.twoM <= 0 {
            return Result(assignment: Array(0..<n), communityCount: n, levels: 0)
        }

        var current = graph
        // membership[l] = 当前图第 l 个超节点包含的原始节点
        var membership: [[Int]] = (0..<n).map { [$0] }
        var levels = 0

        for _ in 0..<maxLevels {
            let nodeCount = current.n
            // 阶段 1：局部移动（每个超节点初始单独成社区）
            let moved = localMoving(current, initial: nil, resolution: resolution,
                                    maxIter: maxIter)
            let (base, baseCount) = renumber(moved)

            // 阶段 2：细化，每个 base 社区内从 singleton 重新聚合
            var groups = [[Int]](repeating: [], count: baseCount)
            for node in 0..<nodeCount { groups[base[node]].append(node) }
            var refined = [Int](repeating: 0, count: nodeCount)
            var nextID = 0
            for group in groups where !group.isEmpty {
                let local = refineGroup(current, members: group,
                                        resolution: resolution, maxIter: maxIter)
                var remap: [Int: Int] = [:]
                for (idx, node) in group.enumerated() {
                    let c = local[idx]
                    if let v = remap[c] { refined[node] = v }
                    else { remap[c] = nextID; refined[node] = nextID; nextID += 1 }
                }
            }
            let (partition, communityCount) = renumber(refined)

            // 收敛：本层没有任何合并 → 划分稳定，停止（membership 保持上一层）
            if communityCount == nodeCount {
                break
            }

            // 阶段 3：聚合成超节点，进入下一层
            var nextMembership = [[Int]](repeating: [], count: communityCount)
            for (node, c) in partition.enumerated() {
                nextMembership[c].append(contentsOf: membership[node])
            }
            membership = nextMembership
            current = aggregate(current, comm: partition, communities: communityCount)
            levels += 1
            if communityCount <= 1 { break }
        }

        var assignment = [Int](repeating: 0, count: n)
        for (c, members) in membership.enumerated() {
            for node in members { assignment[node] = c }
        }
        return Result(assignment: assignment, communityCount: membership.count, levels: levels)
    }
}

// MARK: - 分辨率二分搜索

/// 目标簇数的统计口径。`total` 直接对齐 Leiden 输出的 communityCount；
/// `nonisolated` 只统计有边相连的文档形成的社区（排除固定不变的孤立簇）。
enum TargetCountMode: String, Codable {
    case total
    case nonisolated
}

struct SearchTrial: Codable {
    let gamma: Double
    let communities: Int
    let levels: Int
}

struct ResolutionSearchOutcome {
    let result: Leiden.Result
    let gamma: Double
    let trials: [SearchTrial]
    let effectiveTarget: Int
    let mode: TargetCountMode
    let reached: Bool
    let bestError: Int
    let isolatedFloor: Int
    /// γ→0 时每个有边连通分量各成一簇，加上孤立文档，构成簇数硬下界。
    let achievableMinClusters: Int
    let expandedBounds: Bool
}

/// 通过二分搜索分辨率 γ 逼近用户指定的簇数量。
///
/// 依据：模块度增益 ΔQ ∝ k(i,c) − γ·k_i·Σtot(c)/(2m)，γ 越大「合并」越不划算，
/// 社区越碎、簇数越多，故可用「簇数 − 目标」的符号收缩 γ 区间。
/// γ 跨数量级起作用，用对数区间的几何中点二分，比线性二分收敛更稳。
///
/// 两个现实约束：
/// - 集群数是离散值且 Leiden 是局部最优启发式，未必严格单调 ⇒ 记录每个探测点，
///   最终返回「历史误差最小」的 γ，而不是最后一次二分结果；
/// - 无边的孤立文档永远单独成簇（ki=0，局部移动跳过），构成簇数下界，
///   目标低于该下界时不可能达到，会明确标记 reached=false 并取最接近值。
enum ResolutionSearch {
    struct Config {
        var target: Int
        var mode: TargetCountMode = .total
        var tolerance: Int = 0
        var minGamma: Double = 1e-4
        var maxGamma: Double = 1e3
        var maxSteps: Int = 14
        var maxIter: Int = 20
        var maxLevels: Int = 12
        /// 自适应扩张 γ 区间的最大次数（每侧）。0 = 只用给定区间。
        var maxExpansions: Int = 3
    }

    /// 孤立文档数：无边的节点永远单独成簇，与 γ 无关。
    static func isolatedFloor(_ graph: WeightedGraph) -> Int {
        var count = 0
        for i in 0..<graph.n where graph.degree[i] <= 0 { count += 1 }
        return count
    }

    /// γ→0 时每个有边连通分量各自合成一个社区 ⇒ 簇数硬下界 = 孤立数 + 连通分量数。
    /// 并查集线性计算，不额外跑 Leiden。
    static func achievableMinClusters(_ graph: WeightedGraph, isolated: Int) -> Int {
        var parent = Array(0..<graph.n)
        func find(_ x: Int) -> Int {
            var r = x
            while parent[r] != r { r = parent[r] }
            var i = x
            while parent[i] != i { let next = parent[i]; parent[i] = r; i = next }
            return r
        }
        for u in 0..<graph.n {
            for (v, _) in graph.adj[u] where u < v {
                let a = find(u), b = find(v)
                if a != b { parent[a] = b }
            }
        }
        var components = Set<Int>()
        for i in 0..<graph.n where graph.degree[i] > 0 { components.insert(find(i)) }
        return isolated + components.count
    }

    static func run(_ graph: WeightedGraph, config: Config,
                    logger: ((String) -> Void)? = nil) -> ResolutionSearchOutcome {
        let floor = isolatedFloor(graph)
        let minClusters = achievableMinClusters(graph, isolated: floor)
        let effectiveTarget = config.mode == .nonisolated
            ? config.target + floor
            : config.target
        var loGamma = max(1e-12, min(config.minGamma, config.maxGamma))
        var hiGamma = max(loGamma * 1.000001, config.maxGamma)

        var cache: [Double: Leiden.Result] = [:]
        var trials: [SearchTrial] = []
        var bestGamma = loGamma
        var bestResult: Leiden.Result?
        var bestError = Int.max

        func evaluate(_ gamma: Double) -> Leiden.Result {
            if let hit = cache[gamma] { return hit }
            let result = Leiden.cluster(graph, resolution: gamma,
                                        maxIter: config.maxIter, maxLevels: config.maxLevels)
            cache[gamma] = result
            trials.append(SearchTrial(gamma: gamma,
                                      communities: result.communityCount,
                                      levels: result.levels))
            let error = abs(result.communityCount - effectiveTarget)
            if error < bestError {
                bestError = error
                bestGamma = gamma
                bestResult = result
            }
            logger?(String(format: "[search] γ=%.6g 簇=%d 目标=%d 误差=%d",
                        gamma, result.communityCount, effectiveTarget, error))
            return result
        }

        // 先探两端，再按需向两侧扩张 γ 区间，直到目标落入可达范围。
        // 硬下界 minClusters 达不到时及时停手，避免在极小 γ 上白跑多次 Leiden。
        var low = evaluate(loGamma)
        var high = evaluate(hiGamma)
        let tolerance = max(0, config.tolerance)
        var expanded = false

        if effectiveTarget < low.communityCount && low.communityCount > minClusters {
            for _ in 0..<max(0, config.maxExpansions) {
                let next = loGamma / 100
                if next < 1e-12 || next >= loGamma { break }
                loGamma = next
                expanded = true
                low = evaluate(loGamma)
                if low.communityCount <= effectiveTarget || low.communityCount <= minClusters { break }
            }
        }
        if effectiveTarget > high.communityCount {
            for _ in 0..<max(0, config.maxExpansions) {
                let next = hiGamma * 100
                if next > 1e12 || next <= hiGamma { break }
                hiGamma = next
                expanded = true
                high = evaluate(hiGamma)
                if high.communityCount >= effectiveTarget || high.communityCount >= graph.n { break }
            }
        }

        var lo = log(loGamma)
        var hi = log(hiGamma)
        let targetBetween = min(low.communityCount, high.communityCount) <= effectiveTarget
            && max(low.communityCount, high.communityCount) >= effectiveTarget

        if targetBetween && bestError > tolerance {
            for _ in 0..<max(1, config.maxSteps) {
                if hi - lo < 1e-9 { break }
                let mid = exp((lo + hi) / 2)
                let result = evaluate(mid)
                if abs(result.communityCount - effectiveTarget) <= tolerance { break }
                if result.communityCount > effectiveTarget {
                    hi = log(mid)          // 簇太多 → 降低 γ 让社区合并
                } else {
                    lo = log(mid)          // 簇太少 → 提高 γ 让社区分裂
                }
            }
        }

        // 簇数是离散值，γ 附近可能出现整段跳变（如 2141→2871），
        // 纯二分会停在跳变边界。在最优点对数邻域再探几枪，专门找跨越跳变的 γ。
        // 目标低于硬下界时不可能改善，直接跳过，避免白跑 Leiden。
        if bestError > tolerance && effectiveTarget >= minClusters {
            let base = log(bestGamma)
            for delta in [0.05, -0.05, 0.1, -0.1, 0.2, -0.2, 0.4, -0.4] {
                let candidate = exp(base + delta)
                if cache[candidate] != nil { continue }
                _ = evaluate(candidate)
                if bestError <= tolerance { break }
            }
        }

        func outcome(_ result: Leiden.Result) -> ResolutionSearchOutcome {
            ResolutionSearchOutcome(result: result, gamma: bestGamma,
                                    trials: trials, effectiveTarget: effectiveTarget,
                                    mode: config.mode, reached: bestError <= tolerance,
                                    bestError: bestError, isolatedFloor: floor,
                                    achievableMinClusters: minClusters,
                                    expandedBounds: expanded)
        }
        if let result = bestResult { return outcome(result) }
        return outcome(Leiden.cluster(graph, resolution: bestGamma))
    }
}

// MARK: - 缓存

struct CacheEntry: Codable {
    let signature: String
    let words: [String]
    let freqs: [Int]
    /// 方案二额外存的 shingle 哈希（已 bottom-k、升序），避免重新清洗全文。
    var hashes: [UInt64]?
    /// 裁剪前的 shingle 原始数量，用于报告对照。
    var rawShingles: Int?
}

struct WordCache: Codable {
    let version: Int
    let params: String
    var entries: [String: CacheEntry]
}

// MARK: - 簇命名

enum ClusterNaming {
    /// 把高频 shingle 按「完整 k-1 字符重叠」拼回连续片段。
    /// 例如 ["量子纠缠","子纠缠论"] 重叠 "子纠缠" → "量子纠缠论"。
    /// 只接受完整重叠（重叠 = k-1）以保证拼接出的仍是原文连续片段；
    /// 重叠不足时会留下多个短碎片，符合「仍是碎片」的预期，但可读性更好。
    static func assembleFragments(_ shingles: [String], k: Int) -> [String] {
        guard k > 1 else { return shingles }
        var frags = shingles.filter { $0.count == k }
        // 命名场景下碎片数量可能上百，O(n^3) 合并需限规模：只保留最长的 200 个。
        if frags.count > 120 {
            frags = Array(frags.sorted { $0.count > $1.count || ($0.count == $1.count && $0 < $1) }
                              .prefix(120))
        }
        guard frags.count > 1 else { return frags }
        var changed = true
        while changed {
            changed = false
            outer: for i in 0..<frags.count {
                for j in (0..<frags.count).filter({ $0 != i }) {
                    let a = frags[i], b = frags[j]
                    if a.suffix(k - 1) == b.prefix(k - 1) {
                        frags[i] = a + b.dropFirst(k - 1)
                        frags.remove(at: j)
                        changed = true
                        break outer
                    }
                }
            }
        }
        // 片段间若存在包含关系，去掉短的那个，避免冗余。
        var kept = frags
        kept.sort { $0.count > $1.count }
        var filtered: [String] = []
        for f in kept {
            if !filtered.contains(where: { $0.contains(f) && $0.count > f.count }) {
                filtered.append(f)
            }
        }
        return filtered
    }

    /// 簇内所有文档的核心词合并排序：文档频次（DF）优先，累计词频次之，
    /// 再按词字面保证确定性；取前三作为簇名。
    static func topWords(_ members: [Int], docs: [BookDocument], count: Int = 3,
                         scheme: FeatureScheme = .coreWords, shingleK: Int = 4) -> [String] {
        if scheme == .charShingles {
            return shingleNames(members, docs: docs, k: shingleK, count: count)
        }
        var df: [String: Int] = [:]
        var weight: [String: Int] = [:]
        for m in members {
            for (idx, w) in docs[m].words.enumerated() {
                df[w, default: 0] += 1
                weight[w, default: 0] += idx < docs[m].freqs.count ? docs[m].freqs[idx] : 1
            }
        }
        return df.keys.sorted {
            if df[$0] != df[$1] { return df[$0]! > df[$1]! }
            if weight[$0] != weight[$1] { return weight[$0]! > weight[$1]! }
            return $0 < $1
        }.prefix(count).map { $0 }
    }

    /// 方案二命名：取簇内高频 shingle（DF 排序前 300），拼接还原为较长连续片段。
    static func shingleNames(_ members: [Int], docs: [BookDocument], k: Int,
                             count: Int = 3) -> [String] {
        var df: [String: Int] = [:]
        for m in members {
            for w in docs[m].words where w.count == k {
                df[w, default: 0] += 1
            }
        }
        let top = df.keys.sorted { df[$0]! > df[$1]! }.prefix(120).map { $0 }
        let frags = assembleFragments(top, k: k)
        return frags
            .sorted { $0.count > $1.count || ($0.count == $1.count && $0 < $1) }
            .prefix(count)
            .map { String($0.prefix(20)) }
    }

    static func clusterName(_ members: [Int], docs: [BookDocument],
                            scheme: FeatureScheme = .coreWords, shingleK: Int = 4) -> String {
        let words = topWords(members, docs: docs, count: 3, scheme: scheme, shingleK: shingleK)
        let raw = words.isEmpty ? "未命名" : words.joined(separator: "-")
        let cleaned = sanitize(raw)
        return cleaned.isEmpty ? "未命名" : cleaned
    }

    /// 去掉文件名非法字符、控制字符，限制长度。
    static func sanitize(_ text: String) -> String {
        var out = ""
        for ch in text {
            if ch == "/" || ch == ":" || ch == "\\" || ch == "*" || ch == "?" ||
                ch == "\"" || ch == "<" || ch == ">" || ch == "|" ||
                ch == "\n" || ch == "\r" || ch == "\t" || ch.isWhitespace { continue }
            out.append(ch)
            if out.count >= 60 { break }
        }
        // 保留首尾下划线（--isolated-name 常用 _ 前缀让目录排在前面），
        // 只去掉可能产生「隐藏文件/相对路径」语义的点与连字符。
        return out.trimmingCharacters(in: CharacterSet(charactersIn: ".-"))
    }
}

// MARK: - 报告 / manifest

struct ClusterInfo: Codable {
    let id: Int
    let name: String
    let size: Int
    let topWords: [String]
    let isolated: Bool
    let sampleFiles: [String]
}

struct RunReport: Codable {
    let inputDir: String
    let outRoot: String
    let documentCount: Int
    let wordVocabulary: Int
    let candidatePairs: Int
    let skippedBuckets: Int
    let edges: Int
    let communities: Int
    let isolatedDocuments: Int
    let forceClusters: Int
    let leidenLevels: Int
    let elapsedSeconds: Double
    let applied: Bool
    let clusters: [ClusterInfo]
    // LSH 布局与分辨率搜索信息
    let lshBands: Int
    let lshRowsPerBand: Int
    let resolution: Double
    let targetClusters: Int
    let targetMode: String
    let targetReached: Bool
    let targetError: Int
    let resolutionTrials: [SearchTrial]
    // 文档特征方案信息
    let featureScheme: String
    let shingleK: Int
    let shingleLimit: Int
    let shingleStopwords: Bool
    let shingleJaccard: String
    let shingleDfCut: Double
    let wordDfCut: Double
    let wordDfFiltered: Int
    let dfFilteredShingles: Int
    let avgRawShingles: Int
    let avgKeptFeatures: Int
}

struct MoveRecord: Codable {
    let src: String
    let dst: String
    let community: Int
}

struct MoveManifest: Codable {
    let version: Int
    let createdAt: String
    let outRoot: String
    let moves: [MoveRecord]
}

// MARK: - 参数

struct ClusterOptions {
    var inputDir = "/Volumes/SSD/books-md"
    var outRoot = ""
    var apply = false
    var undoPath = ""
    var limit = 0
    var maxChars = 0
    var topWords = 200
    // 文档特征方案
    var scheme: FeatureScheme = .coreWords
    var shingleK = 4
    var shingleLimit = 2048
    var shingleLabelLimit = 768
    var shingleStopwords = true
    /// 文档级停用词阈值：出现在 ≥ 该比例文档中的 shingle 视为语料级停用词剔除。0=关闭。
    var shingleDfCut = 0.5
    /// 方案一文档级停用词阈值：出现在 ≥ 该比例文档中的核心词剔除（防泛化词）。0=关闭。
    var wordDfCut = 0.5
    var shingleExact = false
    var stopwordsPath = ""
    var jobs = ProcessInfo.processInfo.activeProcessorCount
    var cachePath = ""
    var noCache = false
    var threshold = 0.2
    var maxBucketSize = 200
    var maxPairs = 2_000_000
    var resolution = 1.0
    var targetClusters = 0
    var targetMode: TargetCountMode = .total
    var targetTolerance = 0
    var resolutionMin = 1e-4
    var resolutionMax = 1e3
    var searchSteps = 14
    /// K 簇硬合并目标：Leiden 后把簇强行合并到该数（0=关闭）。默认 200。
    var forceClusters = 200
    /// 锚簇吸收的最低签名相似度：低于该值的簇不并入锚簇，改走均衡兜底（0=只要有重叠即吸收）。
    var anchorSimMin = 0.05
    /// 超大簇二次聚类阈值：硬合并后 size > 该值的簇用簇内边重跑 Leiden 拆分（0=关闭）。
    var splitMax = 5000
    /// 超大簇拆分使用的 Leiden 分辨率。
    var splitGamma = 5.0
    var isolatedName = "_未分类"
    var isolatedEachFolder = false
    var manifestPath = ""
    var reportPath = ""
    var selfTest = false
    var help = false
}

private let usage = """
用法: bookcluster [目录] [选项]     （默认目录 /Volumes/SSD/books-md）

文档特征方案:
  --scheme NAME           core-words=核心词方案 / char-shingles=字符4-gram方案 (默认 core-words)
  --shingle-k N           字符 shingle 窗口长度 (默认 4)
  --shingle-limit K       每本保留哈希最小的 K 个 shingle，0=不裁剪 (默认 2048)
  --shingle-label-limit L 每本保留词频最高的 L 个 shingle 用于簇名拼接 (默认 768)
  --shingle-df-cut X      文档级停用词：出现在 ≥X 比例文档的 shingle 剔除，0=关闭 (默认 0.5)
  --no-shingle-stopwords  关闭 shingle 停用词过滤
  --shingle-exact         用 sketch 的朴素精确 Jaccard（会低估，仅作对照）
  --stopwords PATH        追加自定义停用词文件（每行一词）
  --top-words K           方案一每本保留的核心词数 (默认 200)
  --word-df-cut X         方案一文档级停用词：出现在 ≥X 比例文档的核心词剔除，0=关闭 (默认 0.5)

通用提取:
  --max-chars N           每本书只取前 N 个汉字/字符，0=全量 (默认 0)
  --limit N               只处理前 N 个文件，0=全部 (默认 0)
  --jobs N                并发提取线程数 (默认 CPU 核数)
  --cache PATH            特征缓存 JSON (默认 <输入目录>/.bookcluster_cache.json)
  --no-cache              忽略已有缓存

相似度与图:
  --threshold X           Jaccard 建边阈值，严格大于 (默认 0.2)
  --max-bucket-size N     LSH 单桶文档数上限，超过跳过 (默认 200)
  --max-pairs N           候选对上限 (默认 2000000)
  --resolution X          Leiden 模块度分辨率 γ (默认 1.0；指定 --target-clusters 时忽略)

目标簇数（二分搜索 γ）:
  --target-clusters N     期望簇数，0=关闭搜索直接用 --resolution (默认 0)
  --target-mode MODE      total=含孤立簇 / nonisolated=只算有边文档的社区 (默认 total)
  --target-tolerance K    允许的簇数误差，命中即提前停止 (默认 0)
  --resolution-min X      γ 搜索下界 (默认 0.0001)
  --resolution-max X      γ 搜索上界 (默认 1000)
  --search-steps N        二分最大探测次数 (默认 14)
  --force-clusters K      硬合并到 K 簇：Leiden 后按簇间相似度层级贪心合并，
                          孤立簇并入签名最相似的簇，无相似信息时均衡合并 (默认 200，0=关闭)
  --anchor-sim-min X      锚簇吸收的最低签名相似度，弱信号簇改走均衡兜底 (默认 0.05)
  --split-max N           硬合并后超过 N 本的簇用簇内边重跑 Leiden 拆分 (默认 5000，0=关闭)
  --split-gamma X         超大簇拆分的 Leiden 分辨率 (默认 5.0)

输出与归类:
  --out-root PATH         簇文件夹创建位置 (默认 = 输入目录)
  --isolated-name NAME    孤立文档统一目录名 (默认 _未分类)
  --isolated-each-folder  孤立文档各自单独建文件夹（默认关闭）
  --report PATH           聚类报告 JSON
  --manifest PATH         manifest 路径 (默认 <out-root>/.bookcluster_manifest.json)
  --apply                 真正创建文件夹并移动文件（默认 dry-run，只预览）
  --undo PATH             按 manifest 撤销上一次移动
  --self-test             运行内置自检
  -h, --help              显示本帮助
"""

enum ClusterCLI {
    static func parse(_ argv: [String]) -> ClusterOptions {
        var o = ClusterOptions()
        var positional: [String] = []
        var i = 0
        while i < argv.count {
            let arg = argv[i]
            func value() -> String? {
                guard i + 1 < argv.count else { return nil }
                i += 1
                return argv[i]
            }
            switch arg {
            case "--apply": o.apply = true
            case "--undo": o.undoPath = value() ?? ""
            case "--limit": o.limit = Int(value() ?? "") ?? 0
            case "--max-chars": o.maxChars = Int(value() ?? "") ?? 0
            case "--top-words": o.topWords = Int(value() ?? "") ?? 200
            case "--scheme": o.scheme = FeatureScheme(rawValue: value() ?? "") ?? .coreWords
            case "--shingle-k": o.shingleK = max(1, Int(value() ?? "") ?? 4)
            case "--shingle-limit": o.shingleLimit = max(0, Int(value() ?? "") ?? 2048)
            case "--shingle-label-limit": o.shingleLabelLimit = max(0, Int(value() ?? "") ?? 768)
            case "--shingle-df-cut": o.shingleDfCut = Double(value() ?? "") ?? 0.5
            case "--word-df-cut": o.wordDfCut = Double(value() ?? "") ?? 0.5
            case "--no-shingle-stopwords": o.shingleStopwords = false
            case "--shingle-exact": o.shingleExact = true
            case "--stopwords": o.stopwordsPath = value() ?? ""
            case "--jobs": o.jobs = max(1, Int(value() ?? "") ?? o.jobs)
            case "--cache": o.cachePath = value() ?? ""
            case "--no-cache": o.noCache = true
            case "--threshold": o.threshold = Double(value() ?? "") ?? 0.2
            case "--max-bucket-size": o.maxBucketSize = Int(value() ?? "") ?? 200
            case "--max-pairs": o.maxPairs = Int(value() ?? "") ?? 2_000_000
            case "--resolution": o.resolution = Double(value() ?? "") ?? 1.0
            case "--target-clusters": o.targetClusters = max(0, Int(value() ?? "") ?? 0)
            case "--target-mode":
                o.targetMode = TargetCountMode(rawValue: value() ?? "") ?? .total
            case "--target-tolerance": o.targetTolerance = max(0, Int(value() ?? "") ?? 0)
            case "--resolution-min": o.resolutionMin = Double(value() ?? "") ?? 1e-4
            case "--resolution-max": o.resolutionMax = Double(value() ?? "") ?? 1e3
            case "--search-steps": o.searchSteps = max(1, Int(value() ?? "") ?? 14)
            case "--force-clusters": o.forceClusters = max(0, Int(value() ?? "") ?? 200)
            case "--anchor-sim-min": o.anchorSimMin = Double(value() ?? "") ?? 0.05
            case "--split-max": o.splitMax = max(0, Int(value() ?? "") ?? 5000)
            case "--split-gamma": o.splitGamma = Double(value() ?? "") ?? 5.0
            case "--out-root": o.outRoot = value() ?? ""
            case "--isolated-name": o.isolatedName = value() ?? "_未分类"
            case "--isolated-each-folder": o.isolatedEachFolder = true
            case "--report": o.reportPath = value() ?? ""
            case "--manifest": o.manifestPath = value() ?? ""
            case "--self-test": o.selfTest = true
            case "-h", "--help": o.help = true
            default:
                if arg.hasPrefix("-") {
                    FileHandle.standardError.write(Data("未知选项: \(arg)\n".utf8))
                } else { positional.append(arg) }
            }
            i += 1
        }
        if let first = positional.first { o.inputDir = first }
        if o.outRoot.isEmpty { o.outRoot = o.inputDir }
        // 缓存/manifest 放在输入目录（隐藏文件，枚举时自动跳过），
        // 避免 dry-run 为了写缓存而提前创建 out-root 簇目录。
        if o.cachePath.isEmpty {
            o.cachePath = (o.inputDir as NSString).appendingPathComponent(".bookcluster_cache.json")
        }
        if o.manifestPath.isEmpty {
            o.manifestPath = (o.inputDir as NSString).appendingPathComponent(".bookcluster_manifest.json")
        }
        return o
    }
}

// MARK: - 主流程

enum BookClusterEngine {
    static func log(_ text: String) {
        FileHandle.standardError.write(Data((text + "\n").utf8))
    }

    /// 枚举输入目录顶层文本文件（不递归，避免把已生成的簇文件夹重新当书扫进来）。
    static func enumerate(_ root: String, limit: Int) -> [String] {
        let fm = FileManager.default
        guard let names = try? fm.contentsOfDirectory(atPath: root) else { return [] }
        var files: [String] = []
        for name in names where !name.hasPrefix(".") {
            let path = (root as NSString).appendingPathComponent(name)
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: path, isDirectory: &isDir), !isDir.boolValue else { continue }
            guard textSuffixes.contains((name as NSString).pathExtension.lowercased()) else { continue }
            files.append(path)
        }
        files.sort { ($0 as NSString).lastPathComponent < ($1 as NSString).lastPathComponent }
        return limit > 0 ? Array(files.prefix(limit)) : files
    }

    static func fileStamp(_ path: String) -> String {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: path),
              let size = attrs[.size] as? NSNumber,
              let date = attrs[.modificationDate] as? Date else { return "unknown" }
        return "\(size.int64Value)-\(Int(date.timeIntervalSince1970))"
    }

    /// 缓存命名空间：方案与所有影响特征的参数都参与，避免两套方案串味。
    static func cacheParams(_ options: ClusterOptions) -> String {
        let base: String
        switch options.scheme {
        case .coreWords:
            base = "scheme=core-words|top=\(options.topWords)|maxChars=\(options.maxChars)|kit=3"
                + "|sw=\(options.stopwordsPath)"
        case .charShingles:
            base = "scheme=char-shingles|k=\(options.shingleK)|limit=\(options.shingleLimit)"
                + "|labels=\(options.shingleLabelLimit)"
                + "|maxChars=\(options.maxChars)|stop=\(options.shingleStopwords)"
                + "|sw=\(options.stopwordsPath)"
        }
        // 内置停用词表内容参与命名空间，词表更新后旧缓存自动失效
        return base + "|swv=\(WordFilters.stopwordsVersion)"
    }

    static func extractAll(_ files: [String], options: ClusterOptions)
        -> (docs: [BookDocument], cache: WordCache) {
        let params = cacheParams(options)
        var cached = WordCache(version: 2, params: params, entries: [:])
        if !options.noCache,
           let data = try? Data(contentsOf: URL(fileURLWithPath: options.cachePath)),
           let decoded = try? JSONDecoder().decode(WordCache.self, from: data),
           decoded.version == 2, decoded.params == params {
            cached = decoded
        }

        var extractOptions = ExtractOptions()
        extractOptions.top = options.topWords
        extractOptions.maxChars = options.maxChars
        extractOptions.stopwordsPath = options.stopwordsPath
        let extraStopwords = WordFilters.loadExtra(options.stopwordsPath)

        var results = [BookDocument?](repeating: nil, count: files.count)
        let lock = NSLock()
        var pending: [Int] = []
        for (idx, path) in files.enumerated() {
            let stamp = fileStamp(path)
            if let entry = cached.entries[path], entry.signature == stamp {
                var doc = BookDocument(path: path,
                                       name: (path as NSString).lastPathComponent,
                                       words: entry.words, freqs: entry.freqs)
                doc.shingleHashes = entry.hashes
                doc.rawShingles = entry.rawShingles ?? 0
                results[idx] = doc
            } else {
                pending.append(idx)
            }
        }
        log("[cluster] 方案=\(options.scheme.rawValue) 文件 \(files.count)，"
            + "缓存命中 \(files.count - pending.count)，待提取 \(pending.count)")

        if !pending.isEmpty {
            let progressLock = NSLock()
            var done = 0
            DispatchQueue.concurrentPerform(iterations: pending.count) { slot in
                let idx = pending[slot]
                let path = files[idx]
                let name = (path as NSString).lastPathComponent
                var doc: BookDocument
                var entry: CacheEntry
                do {
                    let text = try CorpusLoader.load(path, encoding: "auto")
                    switch options.scheme {
                    case .coreWords:
                        let result = try CoreWordExtractor.extract(text: text, options: extractOptions)
                        doc = BookDocument(path: path, name: name,
                                           words: result.words.map(\.word),
                                           freqs: result.words.map(\.freq))
                        entry = CacheEntry(signature: fileStamp(path), words: doc.words,
                                           freqs: doc.freqs, hashes: nil, rawShingles: nil)
                    case .charShingles:
                        let s = CharShingleFeatures.extract(
                            text, maxChars: options.maxChars, k: options.shingleK,
                            limit: options.shingleLimit,
                            labelLimit: options.shingleLabelLimit,
                            useStopwords: options.shingleStopwords,
                            extraStopwords: extraStopwords)
                        doc = BookDocument(path: path, name: name, words: s.labels,
                                           freqs: [Int](repeating: 1, count: s.labels.count))
                        doc.shingleHashes = s.hashes
                        doc.rawShingles = s.rawShingles
                        entry = CacheEntry(signature: fileStamp(path), words: s.labels,
                                           freqs: doc.freqs, hashes: s.hashes,
                                           rawShingles: s.rawShingles)
                    }
                } catch {
                    log("[cluster] 提取失败 \(path): \(error)")
                    doc = BookDocument(path: path, name: name, words: [], freqs: [])
                    entry = CacheEntry(signature: fileStamp(path), words: [], freqs: [],
                                       hashes: nil, rawShingles: 0)
                }
                lock.lock()
                results[idx] = doc
                cached.entries[path] = entry
                lock.unlock()
                progressLock.lock()
                done += 1
                if done % 200 == 0 || done == pending.count { log("[cluster] 已提取 \(done)/\(pending.count)") }
                progressLock.unlock()
            }
            if !options.noCache { saveCache(cached, path: options.cachePath) }
        }

        return (results.map { $0 ?? BookDocument(path: "", name: "", words: [], freqs: []) }, cached)
    }

    static func ensureParent(_ path: String) {
        let parent = (path as NSString).deletingLastPathComponent
        try? FileManager.default.createDirectory(atPath: parent, withIntermediateDirectories: true)
    }

    static func saveCache(_ cache: WordCache, path: String) {
        guard let data = try? JSONEncoder().encode(cache) else { return }
        ensureParent(path)
        try? data.write(to: URL(fileURLWithPath: path), options: .atomic)
    }

    static func writeJSON<T: Encodable>(_ value: T, path: String) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(value)
        ensureParent(path)
        try data.write(to: URL(fileURLWithPath: path), options: .atomic)
    }

    /// 方案一文档级停用词：剔除出现在 ≥ cut 比例文档中的核心词。
    ///
    /// 核心词提取（N-gram+NPMI+左右熵）已过滤虚词，但「中国经济」「文化发展」
    /// 这类跨主题通用词仍可能漏进来——它们几乎每本书都有，毫无区分度。
    /// DF 阈值默认 0.5：只删出现在一半以上文档的词；专业词（如「区块链」
    /// 「量子纠缠」）通常只出现在百分之几的书里，DF 远低于阈值，不会被误伤。
    /// words/freqs/featureIDs 同步过滤，签名按字符串域重算（仅受影响文档）。
    /// 返回被剔除的全局词数。
    static func filterCommonWords(_ docs: inout [BookDocument], cut: Double,
                                  table: WordTable, family: HashFamily) -> Int {
        guard cut > 0, !docs.isEmpty else { return 0 }
        let cutoff = max(2, Int((Double(docs.count) * cut).rounded(.up)))
        var all = [UInt64]()
        all.reserveCapacity(docs.reduce(0) { $0 + $1.featureIDs.count })
        for d in docs { all.append(contentsOf: d.featureIDs) }
        guard !all.isEmpty else { return 0 }
        all.sort()
        var common = Set<UInt64>()
        var i = 0
        while i < all.count {
            var j = i
            while j < all.count && all[j] == all[i] { j += 1 }
            if j - i >= cutoff { common.insert(all[i]) }
            i = j
        }
        guard !common.isEmpty else { return 0 }
        for idx in docs.indices {
            // 快速跳过不含被删词的文档（避免无谓的签名重算）
            guard docs[idx].featureIDs.contains(where: { common.contains($0) }) else { continue }
            var words: [String] = []
            var freqs: [Int] = []
            words.reserveCapacity(docs[idx].words.count)
            for (wi, w) in docs[idx].words.enumerated() {
                if common.contains(UInt64(table.id(w))) { continue }
                words.append(w)
                freqs.append(wi < docs[idx].freqs.count ? docs[idx].freqs[wi] : 1)
            }
            docs[idx].words = words
            docs[idx].freqs = freqs
            var ids = Set<UInt64>()
            for w in words { ids.insert(UInt64(table.id(w))) }
            docs[idx].featureIDs = ids.sorted()
            docs[idx].signature = MinHash.signature(words, family: family)
        }
        return common.count
    }

    /// --force-clusters：把 Leiden 结果硬合并到 K 簇（层级贪心）。
    ///
    /// 三阶段：
    ///  1) Kruskal 式：簇间边（文档边映射到簇对，取最大相似度）按相似度降序合并；
    ///  2) 锚簇吸收：仍有剩余时，把每个非锚簇并入签名最相似的锚簇（簇签名 =
    ///     成员签名逐维取 min，可估计簇并集的 Jaccard），相似度为 0 的暂缓；
    ///  3) 均衡兜底：相似度全为 0 的孤立簇之间无任何语义信息，按「每次合并
    ///     当前最小的两个簇」均衡合并到目标数，避免单一大簇吞掉全部孤立。
    /// 簇数已 ≤ K 或 K 非法时原样返回。
    static func forceMerge(_ groups: [Int: [Int]], docs: [BookDocument],
                           edges: [(Int, Int, Double)], target: Int,
                           anchorLimit: Int = 128, anchorSimMin: Double = 0.0,
                           log: (String) -> Void) -> [Int: [Int]] {
        guard target > 0, groups.count > target else { return groups }
        let startCount = groups.count

        var parent: [Int: Int] = [:]
        for cid in groups.keys { parent[cid] = cid }
        func find(_ x: Int) -> Int {
            var r = x
            while parent[r] != r { r = parent[r]! }
            var c = x
            while parent[c] != c { let n = parent[c]!; parent[c] = r; c = n }
            return r
        }
        var size: [Int: Int] = [:]
        var sig: [Int: [UInt64]] = [:]
        var nodeCluster = [Int](repeating: -1, count: docs.count)
        for (cid, members) in groups {
            size[cid] = members.count
            for m in members { nodeCluster[m] = cid }
            var s = [UInt64](repeating: UInt64.max, count: docs[members[0]].signature.count)
            for m in members {
                let ms = docs[m].signature
                for i in 0..<s.count where i < ms.count { s[i] = min(s[i], ms[i]) }
            }
            sig[cid] = s
        }

        var count = groups.count
        var kruskalMerges = 0

        // 阶段 1：簇间边贪心合并
        var bestEdge: [Int: Double] = [:]
        for (u, v, w) in edges {
            let cu = nodeCluster[u], cv = nodeCluster[v]
            guard cu >= 0, cv >= 0, cu != cv else { continue }
            let a = find(cu), b = find(cv)
            guard a != b else { continue }
            let key = a < b ? a * 1_000_000 + b : b * 1_000_000 + a
            if w > (bestEdge[key] ?? -1) { bestEdge[key] = w }
        }
        let pairEdges: [(w: Double, a: Int, b: Int)] = bestEdge.map {
            (w: $0.value, a: $0.key / 1_000_000, b: $0.key % 1_000_000)
        }.sorted { $0.w == $1.w ? $0.a < $1.a : $0.w > $1.w }
        for (_, a, b) in pairEdges {
            guard count > target else { break }
            let ra = find(a), rb = find(b)
            guard ra != rb else { continue }
            parent[rb] = ra
            size[ra]! += size[rb]!
            let sa = sig[ra]!, sb = sig[rb]!
            sig[ra] = zip(sa, sb).map { min($0, $1) }
            sig[rb] = nil
            count -= 1
            kruskalMerges += 1
        }

        // 阶段 2：非锚簇并入签名最相似的锚簇
        var anchorAbsorbed = 0
        if count > target {
            var active: [Int] = []
            var seen = Set<Int>()
            // 字典遍历无序：排序保证 tie-break 与结果可复现
            for cid in groups.keys.sorted() {
                let r = find(cid)
                if seen.insert(r).inserted { active.append(r) }
            }
            let anchors = active.sorted {
                size[$0]! != size[$1]! ? size[$0]! > size[$1]! : $0 < $1
            }.prefix(max(1, min(anchorLimit, active.count)))
            let anchorList = Array(anchors)
            let anchorSet = Set(anchorList)
            func sigSim(_ a: [UInt64], _ b: [UInt64]) -> Double {
                guard a.count == b.count, !a.isEmpty else { return 0 }
                var same = 0
                for i in 0..<a.count where a[i] == b[i] { same += 1 }
                return Double(same) / Double(a.count)
            }
            for cid in active where !anchorSet.contains(cid) {
                guard count > target else { break }
                var bestA = -1
                var bestW = 0.0
                for a in anchorList {
                    let s = sigSim(sig[cid]!, sig[a]!)
                    if s > bestW { bestW = s; bestA = a }
                }
                guard bestA >= 0, bestW > anchorSimMin else { continue }
                parent[cid] = bestA
                size[bestA]! += size[cid]!
                let sa = sig[bestA]!, sb = sig[cid]!
                sig[bestA] = zip(sa, sb).map { min($0, $1) }
                sig[cid] = nil
                count -= 1
                anchorAbsorbed += 1
            }
        }

        // 阶段 3：均衡兜底——每轮合并当前最小的两个簇。
        // 全量可达数万簇，逐轮全排序是 O(C^2 log C)；改为维护升序数组，
        // 合并结果二分定位重插，总体 O(C^2) 且常数小（数组 move 为 memmove）。
        var balancedMerges = 0
        if count > target {
            var seenRoot = Set<Int>()
            var roots = groups.keys.sorted().map { find($0) }.filter { seenRoot.insert($0).inserted }
            roots = roots.filter { sig[$0] != nil }
            roots.sort { size[$0]! != size[$1]! ? size[$0]! < size[$1]! : $0 < $1 }
            func insertBySize(_ x: Int) {
                var lo = 0, hi = roots.count
                while lo < hi {
                    let mid = (lo + hi) / 2
                    let s = size[roots[mid]]!
                    if s < size[x]! || (s == size[x]! && roots[mid] < x) { lo = mid + 1 } else { hi = mid }
                }
                roots.insert(x, at: lo)
            }
            while roots.count > target {
                let a = roots.removeFirst()
                let b = roots.removeFirst()
                parent[a] = b
                size[b]! += size[a]!
                let sb = sig[b]!, sa2 = sig[a]!
                sig[b] = zip(sb, sa2).map { min($0, $1) }
                sig[a] = nil
                insertBySize(b)
                balancedMerges += 1
            }
            count = roots.count
        }

        var merged: [Int: [Int]] = [:]
        for (cid, members) in groups.sorted(by: { $0.key < $1.key }) {
            merged[find(cid), default: []].append(contentsOf: members)
        }
        log("[cluster] force-clusters：\(startCount) 簇 → \(merged.count) 簇 "
            + "（簇间边贪心 \(kruskalMerges) + 锚簇吸收 \(anchorAbsorbed) + 均衡兜底 \(balancedMerges)）")
        return merged
    }

    /// 超大簇二次聚类：对硬合并后 size > max 的簇，取簇内边子图重跑 Leiden（高 γ）拆分，
    /// 递归直到所有簇 ≤ max、或簇内无可拆边（如全由孤立文档堆积而成——无边文档
    /// 无法提供拆分依据，只能作为 singleton 落地）。返回重编号后的 groups。
    static func splitOversized(_ groups: [Int: [Int]], edges: [(Int, Int, Double)],
                               max: Int, gamma: Double, log: (String) -> Void) -> [Int: [Int]] {
        guard max > 0, !groups.isEmpty else { return groups }
        var pending = Array(groups.values).sorted { $0.count > $1.count }
        var result: [[Int]] = []
        var splitRounds = 0
        while !pending.isEmpty {
            let members = pending.removeFirst()
            guard members.count > max else { result.append(members); continue }
            let local = Dictionary(uniqueKeysWithValues: members.enumerated().map { ($1, $0) })
            var subEdges: [(Int, Int, Double)] = []
            for (u, v, w) in edges {
                guard let lu = local[u], let lv = local[v] else { continue }
                subEdges.append((lu, lv, w))
            }
            guard !subEdges.isEmpty else { result.append(members); continue }
            let g = WeightedGraph(n: members.count, edges: subEdges)
            let r = Leiden.cluster(g, resolution: gamma)
            guard r.communityCount > 1 else { result.append(members); continue }
            var sub: [Int: [Int]] = [:]
            for (node, c) in r.assignment.enumerated() { sub[c, default: []].append(members[node]) }
            splitRounds += 1
            for grp in sub.values.sorted(by: { $0.count > $1.count }) { pending.append(grp) }
        }
        var out: [Int: [Int]] = [:]
        for (i, grp) in result.enumerated() { out[i] = grp }
        log("[cluster] 超大簇二次聚类：拆分 \(splitRounds) 次 → \(out.count) 簇（max=\(max) γ=\(gamma)）")
        return out
    }

    /// 撤销：把 manifest 里仍在 dst 的文件移回 src。
    static func undo(_ path: String) -> Int32 {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let manifest = try? JSONDecoder().decode(MoveManifest.self, from: data) else {
            log("无法读取 manifest: \(path)")
            return 1
        }
        let fm = FileManager.default
        var restored = 0
        for record in manifest.moves.reversed() {
            guard fm.fileExists(atPath: record.dst) else { continue }
            if fm.fileExists(atPath: record.src) {
                log("[undo] 源路径已存在，跳过: \(record.src)")
                continue
            }
            do {
                try fm.moveItem(atPath: record.dst, toPath: record.src)
                restored += 1
            } catch {
                log("[undo] 失败 \(record.dst): \(error)")
            }
        }
        log("[undo] 已恢复 \(restored)/\(manifest.moves.count) 个文件；簇文件夹未自动删除，请按需清理")
        return 0
    }

    static func run(_ options: ClusterOptions) -> Int32 {
        if options.help { print(usage); return 0 }
        if options.selfTest { return SelfTest.run() }
        if !options.undoPath.isEmpty { return undo(options.undoPath) }

        let started = Date()
        let fm = FileManager.default
        guard fm.fileExists(atPath: options.inputDir) else {
            log("输入目录不存在: \(options.inputDir)")
            return 1
        }

        // 1) 枚举 + 提取核心词
        let files = enumerate(options.inputDir, limit: options.limit)
        if files.isEmpty { log("未找到文本文件"); return 1 }
        let (docs, _) = extractAll(files, options: options)

        // 2) 特征 ID + MinHash 签名（两套方案共用同一套哈希族与 LSH）
        let table = WordTable()
        let family = HashFamily()
        var prepared: [BookDocument] = []
        prepared.reserveCapacity(docs.count)
        var rawShingleSum = 0
        for var doc in docs {
            switch options.scheme {
            case .coreWords:
                var ids = Set<UInt64>()
                for w in doc.words { ids.insert(UInt64(table.id(w))) }
                doc.featureIDs = ids.sorted()
                doc.signature = MinHash.signature(doc.words, family: family)
            case .charShingles:
                // shingle 哈希已是排序好的 61 位值；签名推迟到 DF 过滤之后统一生成，
                // 避免全量重算两遍。
                let hashes = doc.shingleHashes ?? doc.words.map { mod61(fnv1a64($0)) }.sorted()
                doc.featureIDs = hashes
                rawShingleSum += doc.rawShingles
            }
            prepared.append(doc)
        }

        // 2.4) 方案一文档级停用词：剔除出现于绝大多数文档的泛化核心词（DF 过滤）。
        var wordDfFiltered = 0
        if options.scheme == .coreWords && options.wordDfCut > 0 {
            wordDfFiltered = BookClusterEngine.filterCommonWords(
                &prepared, cut: options.wordDfCut, table: table, family: family)
            if wordDfFiltered > 0 {
                log("[cluster] 文档级停用词：DF ≥ \(options.wordDfCut) 的核心词 "
                    + "\(wordDfFiltered) 个已剔除（专业词 DF 远低于阈值，不受影响）")
            }
        }

        // 2.5) 方案二文档级停用词：剔除出现于绝大多数文档的高频 shingle（DF 过滤）。
        var dfFiltered = 0
        if options.scheme == .charShingles && options.shingleDfCut > 0 {
            dfFiltered = CharShingleFeatures.filterCommonShingles(&prepared, cut: options.shingleDfCut)
            if dfFiltered > 0 {
                log("[cluster] 文档级停用词：DF ≥ \(options.shingleDfCut) 的 shingle "
                    + "\(dfFiltered) 个已剔除（特征与命名集合同步过滤）")
            }
        }

        // 2.6) 方案二统一生成签名（DF 过滤后只算一遍）
        if options.scheme == .charShingles {
            for idx in prepared.indices {
                prepared[idx].signature = MinHash.signatureFromHashes(
                    prepared[idx].featureIDs, family: family)
            }
        }
        let signatures = prepared.map { $0.signature }

        let nDocs = max(1, prepared.count)
        let avgRawShingles = rawShingleSum / nDocs
        let avgKeptFeatures = prepared.reduce(0) { $0 + $1.featureIDs.count } / nDocs
        if options.scheme == .charShingles {
            log("[cluster] shingle 平均原始 \(avgRawShingles) 个 → 保留 \(avgKeptFeatures) 个，"
                + "签名 \(signatures.count) 份")
        } else {
            log("[cluster] 核心词词表 \(table.count) 个，签名 \(signatures.count) 份")
        }

        // 3) LSH 候选对
        let (pairs, skipped) = LSH.candidatePairs(signatures, maxBucketSize: options.maxBucketSize,
                                                  maxPairs: options.maxPairs)
        log("[cluster] LSH 候选对 \(pairs.count)，跳过巨桶 \(skipped) 个")

        // 4) 精确 Jaccard → 加权边
        var edges: [(Int, Int, Double)] = []
        edges.reserveCapacity(min(pairs.count, 500_000))
        for pair in pairs {
            let (a, b) = LSH.decode(pair)
            guard a < prepared.count, b < prepared.count else { continue }
            let sim = (options.scheme == .charShingles && !options.shingleExact)
                ? minsJaccard(prepared[a].featureIDs, prepared[b].featureIDs)
                : exactJaccard(prepared[a].featureIDs, prepared[b].featureIDs)
            if sim > options.threshold { edges.append((a, b, sim)) }
        }
        if options.scheme == .charShingles {
            log("[cluster] Jaccard 估计 = \(options.shingleExact ? "sketch 朴素精确（低估）" : "k-mins 无偏估计")")
        }
        log("[cluster] 相似边 \(edges.count) 条 (sim > \(options.threshold))")

        // 5) Leiden 社区发现（可选 γ 二分搜索以逼近目标簇数）
        let graph = WeightedGraph(n: prepared.count, edges: edges)
        let leiden: Leiden.Result
        var usedResolution = options.resolution
        var trials: [SearchTrial] = []
        var targetReached = false
        var targetError = 0
        if options.targetClusters > 0 {
            var config = ResolutionSearch.Config(target: options.targetClusters,
                                                 mode: options.targetMode)
            config.tolerance = options.targetTolerance
            config.minGamma = options.resolutionMin
            config.maxGamma = options.resolutionMax
            config.maxSteps = options.searchSteps
            log("[cluster] 目标簇数 \(options.targetClusters)（\(options.targetMode.rawValue)），"
                + "二分搜索 γ ∈ [\(options.resolutionMin), \(options.resolutionMax)]")
            let search = ResolutionSearch.run(graph, config: config, logger: log)
            leiden = search.result
            usedResolution = search.gamma
            trials = search.trials
            targetReached = search.reached
            targetError = search.bestError
            let note: String
            if search.reached {
                note = "✓达标"
            } else if search.effectiveTarget < search.achievableMinClusters {
                note = "（目标低于可达下界 \(search.achievableMinClusters)："
                    + "孤立 \(search.isolatedFloor) + 有边连通分量；可放宽 --threshold 增加连边）"
            } else {
                note = "（簇数为离散值，已取最接近）"
            }
            log(String(format: "[cluster] 搜索完成：γ=%.6g 实际簇=%d 目标=%d 误差=%d %@",
                       search.gamma, leiden.communityCount, search.effectiveTarget,
                       search.bestError, note))
        } else {
            leiden = Leiden.cluster(graph, resolution: options.resolution)
        }
        var groups: [Int: [Int]] = [:]
        for (node, c) in leiden.assignment.enumerated() { groups[c, default: []].append(node) }
        let connected = Set(edges.flatMap { [$0.0, $0.1] })
        func isIsolatedSingle(_ members: [Int]) -> Bool {
            members.count == 1 && !connected.contains(members[0])
        }
        var isolated = groups.values.filter { isIsolatedSingle($0) }.count
        log("[cluster] Leiden 得到 \(groups.count) 个簇（\(leiden.levels) 层），孤立文档 \(isolated)")

        // 5.5) --force-clusters：K 簇硬合并（层级贪心，孤立簇并入最近簇）
        if options.forceClusters > 0 {
            if groups.count <= options.forceClusters {
                log("[cluster] force-clusters：当前 \(groups.count) 簇 ≤ 目标 \(options.forceClusters)，"
                    + "无法拆分，保持不变")
            } else {
                groups = forceMerge(groups, docs: prepared, edges: edges,
                                    target: options.forceClusters,
                                    anchorSimMin: options.anchorSimMin, log: log)
                if options.splitMax > 0 {
                    groups = splitOversized(groups, edges: edges, max: options.splitMax,
                                            gamma: options.splitGamma, log: log)
                }
                isolated = groups.values.filter { isIsolatedSingle($0) }.count
                log("[cluster] 硬合并后 \(groups.count) 个簇，孤立文档 \(isolated)")
            }
        }

        // 6) 簇命名 + 报告
        var usedNames = Set<String>()
        var infos: [ClusterInfo] = []
        var moves: [MoveRecord] = []
        let sortedIDs = groups.keys.sorted()
        for id in sortedIDs {
            guard let members = groups[id] else { continue }
            let single = isIsolatedSingle(members)
            var name: String
            if single && !options.isolatedEachFolder {
                name = ClusterNaming.sanitize(options.isolatedName)
            } else {
                name = ClusterNaming.clusterName(members, docs: prepared,
                                                 scheme: options.scheme,
                                                 shingleK: options.shingleK)
            }
            // 重名簇加后缀，避免不同簇覆盖到同一文件夹
            var unique = name
            var suffix = 2
            while usedNames.contains(unique) && !(single && !options.isolatedEachFolder) {
                unique = "\(name)-\(suffix)"
                suffix += 1
            }
            usedNames.insert(unique)
            let dir = (options.outRoot as NSString).appendingPathComponent(unique)

            let info = ClusterInfo(id: id, name: unique, size: members.count,
                                   topWords: ClusterNaming.topWords(members, docs: prepared,
                                                                  scheme: options.scheme,
                                                                  shingleK: options.shingleK),
                                   isolated: single,
                                   sampleFiles: members.prefix(5).map { prepared[$0].name })
            infos.append(info)

            for m in members {
                let src = prepared[m].path
                var dst = (dir as NSString).appendingPathComponent(prepared[m].name)
                // 目标已存在同名文件：加序号，绝不覆盖
                if fm.fileExists(atPath: dst) {
                    let base = (prepared[m].name as NSString).deletingPathExtension
                    let ext = (prepared[m].name as NSString).pathExtension
                    var k = 2
                    repeat {
                        let candidate = ext.isEmpty ? "\(base)-\(k)" : "\(base)-\(k).\(ext)"
                        dst = (dir as NSString).appendingPathComponent(candidate)
                        k += 1
                    } while fm.fileExists(atPath: dst)
                }
                moves.append(MoveRecord(src: src, dst: dst, community: id))
            }
        }

        let elapsed = Date().timeIntervalSince(started)
        let report = RunReport(inputDir: options.inputDir, outRoot: options.outRoot,
                               documentCount: prepared.count, wordVocabulary: table.count,
                               candidatePairs: pairs.count, skippedBuckets: skipped,
                               edges: edges.count, communities: groups.count,
                               isolatedDocuments: isolated,
                               forceClusters: options.forceClusters,
                               leidenLevels: leiden.levels,
                               elapsedSeconds: elapsed, applied: options.apply,
                               clusters: infos.sorted { $0.size > $1.size },
                               lshBands: bandCount, lshRowsPerBand: bandRows,
                               resolution: usedResolution,
                               targetClusters: options.targetClusters,
                               targetMode: options.targetMode.rawValue,
                               targetReached: targetReached, targetError: targetError,
                               resolutionTrials: trials,
                               featureScheme: options.scheme.rawValue,
                               shingleK: options.shingleK,
                               shingleLimit: options.shingleLimit,
                               shingleStopwords: options.shingleStopwords,
                               shingleJaccard: options.scheme == .charShingles
                                   ? (options.shingleExact ? "exact-sketch" : "k-mins") : "exact",
                               shingleDfCut: options.scheme == .charShingles ? options.shingleDfCut : 0,
                               wordDfCut: options.scheme == .coreWords ? options.wordDfCut : 0,
                               wordDfFiltered: wordDfFiltered,
                               dfFilteredShingles: dfFiltered,
                               avgRawShingles: avgRawShingles,
                               avgKeptFeatures: avgKeptFeatures)
        if !options.reportPath.isEmpty {
            do { try writeJSON(report, path: options.reportPath); log("[cluster] 报告 \(options.reportPath)") }
            catch { log("[cluster] 报告写入失败: \(error)") }
        }

        // 7) 预览 / 执行
        printPreview(report: report, moves: moves, apply: options.apply)
        if !options.apply {
            log("[cluster] dry-run：未创建文件夹、未移动文件。确认后加 --apply 执行")
            return 0
        }

        var applied = 0
        var succeeded: [MoveRecord] = []
        for record in moves {
            let dir = (record.dst as NSString).deletingLastPathComponent
            do {
                try fm.createDirectory(atPath: dir, withIntermediateDirectories: true)
                try fm.moveItem(atPath: record.src, toPath: record.dst)
                succeeded.append(record)
                applied += 1
            } catch {
                log("[cluster] 移动失败 \(record.src): \(error)")
            }
        }
        let manifest = MoveManifest(version: 1,
                                    createdAt: ISO8601DateFormatter().string(from: Date()),
                                    outRoot: options.outRoot, moves: succeeded)
        do {
            try writeJSON(manifest, path: options.manifestPath)
            log("[cluster] 已移动 \(applied)/\(moves.count) 个文件；manifest: \(options.manifestPath)")
        } catch { log("[cluster] manifest 写入失败: \(error)（移动已执行，请手动记录）") }
        return 0
    }

    static func printPreview(report: RunReport, moves: [MoveRecord], apply: Bool) {
        print("\(apply ? "[APPLY]" : "[DRY-RUN]") 文档 \(report.documentCount) | "
              + "候选对 \(report.candidatePairs) | 边 \(report.edges) | "
              + "簇 \(report.communities) | 孤立 \(report.isolatedDocuments) | "
              + String(format: "%.1fs", report.elapsedSeconds))
        for cluster in report.clusters.prefix(20) {
            print("  #\(cluster.id) \(cluster.name)  \(cluster.size) 本  "
                  + "词[\(cluster.topWords.joined(separator: ","))]  "
                  + "例: \(cluster.sampleFiles.prefix(2).joined(separator: " / "))")
        }
        if report.clusters.count > 20 { print("  ... 其余 \(report.clusters.count - 20) 个簇见报告 JSON") }
        print("  计划移动 \(moves.count) 个文件")
    }
}

// MARK: - 自检

enum SelfTest {
    private static var failures: [String] = []
    private static func check(_ condition: Bool, _ message: String) {
        if !condition { failures.append(message) }
    }
    private static func approx(_ a: Double, _ b: Double, _ eps: Double = 1e-9) -> Bool {
        abs(a - b) < eps
    }

    static func run() -> Int32 {
        // 哈希基础
        check(fnv1a64("金融") == fnv1a64("金融"), "FNV 稳定")
        check(fnv1a64("金融") != fnv1a64("数据"), "FNV 区分不同词")
        check(mod61(mersenne61) == 0, "mod61(prime)=0")
        check(mulMod61(3, 5) == 15, "mulMod61 小值")
        check(mulMod61(mersenne61 - 1, 2) == mod61(2 * (mersenne61 - 1)), "mulMod61 大值约简")

        // HashFamily 确定性 + 独立性
        let f1 = HashFamily(), f2 = HashFamily()
        check(f1.a == f2.a && f1.b == f2.b, "哈希函数族可复现")
        check(Set(f1.a).count > 100, "128 组 a 参数基本互异")

        // MinHash
        let wordsA = ["金融", "数据", "平台", "支付", "互联网"]
        let sigA = MinHash.signature(wordsA, family: f1)
        let sigA2 = MinHash.signature(["互联网", "支付", "平台", "数据", "金融"], family: f1)
        check(sigA.count == signatureDim, "签名 128 维")
        check(sigA == sigA2, "词序不影响签名")
        check(approx(MinHash.estimatedJaccard(sigA, sigA), 1.0), "相同集合估计相似度=1")
        let sigB = MinHash.signature(["佛教", "菩萨", "大乘", "涅槃"], family: f1)
        check(MinHash.estimatedJaccard(sigA, sigB) < 0.5, "不同集合估计相似度低")
        let emptySig = MinHash.signature([], family: f1)
        check(emptySig.allSatisfy { $0 == mersenne61 }, "空集合签名为 prime 哨兵")

        // LSH：相同集合必定同桶；不同集合也可能偶然同桶，只验证前者被召回
        var sigs = [sigA, sigA2, sigB, sigB]
        sigs.append(MinHash.signature(["量子", "纠缠", "物理"], family: f1))
        let (pairs, skipped) = LSH.candidatePairs(sigs, maxBucketSize: 200, maxPairs: 1000)
        let decoded = Set(pairs.map { LSH.decode($0) }.map { "\($0.0)-\($0.1)" })
        check(decoded.contains("0-1"), "完全相同的两份文档成为候选对")
        check(decoded.contains("2-3"), "另一组相同文档成为候选对")
        check(skipped == 0, "无巨桶时 skipped=0")
        let (limited, _) = LSH.candidatePairs(sigs, maxBucketSize: 1, maxPairs: 1000)
        check(limited.isEmpty, "桶超限时被跳过")
        check(LSH.decode((UInt64(7) << 32) | 9) == (7, 9), "候选对编解码")
        check(bandLayoutIsValid && bandCount == 32 && bandRows == 4
                && bandCount * bandRows == signatureDim, "LSH 布局 32 bands × 4 维覆盖 128 维")

        // 方案二：文本清理 / 字符 shingle / 停用词 / bottom-k
        let cleaned = CharShingleFeatures.clean("量子纠缠，是物理！\nAlpha-Beta 123_测试 OK", maxChars: 0)
        let cleanedText = String(cleaned)
        check(cleanedText == "量子纠缠是物理alphabeta测试ok",
              "清理后只保留汉字与小写字母（实际=\(cleanedText)）")
        check(CharShingleFeatures.clean("量子纠缠是物理", maxChars: 4).count == 4,
              "maxChars 截断生效")
        let sh = CharShingleFeatures.shingles(Array("甲乙丙甲乙丙"), k: 4)
        check(sh.count == 3 && sh.contains("甲乙丙甲") && sh.contains("丙甲乙丙"),
              "4-shingle 滑窗 + 自动去重（实际 \(sh.count) 个）")
        check(CharShingleFeatures.shingles(Array("甲乙"), k: 4).isEmpty,
              "短于 k 的文本无 shingle")
        let stops = CharShingleFeatures.filterStopwords(["的了是在", "量子纠缠", "tion", "甲乙丙丁"],
                                                        k: 4, extra: ["甲乙丙丁"])
        check(!stops.contains("的了是在"), "全虚字 shingle 被过滤")
        check(!stops.contains("tion"), "英文功能片段被过滤")
        check(!stops.contains("甲乙丙丁"), "自定义停用词生效")
        check(stops.contains("量子纠缠"), "真词 shingle 不被误杀")
        let bk = CharShingleFeatures.bottomK(sh, limit: 2)
        check(bk.words.count == 2 && bk.hashes.count == 2
                && bk.hashes == bk.hashes.sorted(), "bottom-k 裁剪且哈希升序")
        let shingleResult = CharShingleFeatures.extract(
            "甲乙丙丁戊己庚辛甲乙丙丁戊己庚辛", maxChars: 0, k: 4, limit: 3, labelLimit: 5,
            useStopwords: true, extraStopwords: [])
        check(shingleResult.labels.count == 5 && shingleResult.hashes.count == 3
                && shingleResult.rawShingles > 3 && shingleResult.chars == 16,
              "方案二提取：命名集合(密集)与相似度集合(bottom-k)分离")
        check(CharShingleFeatures.isStop("的了是在", k: 4, extra: []), "isStop 全虚字")
        let shingleSig = MinHash.signatureFromHashes(shingleResult.hashes, family: f1)
        check(shingleSig.count == signatureDim && shingleSig != emptySig,
              "shingle 哈希可直接生成 MinHash 签名")
        check(approx(exactJaccard([1, 2], [1, 2]), 1.0), "UInt64 特征 Jaccard 正常")
        // k-mins 估计：两份 bottom-k sketch 有共同阈值时不低估相似度
        check(approx(minsJaccard([1, 2, 3, 4], [1, 2, 3, 9]), 0.75),
              "k-mins 估计按公共阈值计算")
        check(approx(minsJaccard([1, 2], [1, 2]), 1.0), "相同 sketch k-mins=1")
        check(minsJaccard([1, 2, 3], [7, 8, 9]) == 0, "无公共前缀 k-mins=0")
        check(minsJaccard([1, 2, 3, 4], [1, 2, 3, 9]) >
                exactJaccard([1, 2, 3, 4], [1, 2, 3, 9]),
              "k-mins 修正了 sketch 朴素 Jaccard 的低估")
        check(BookClusterEngine.cacheParams(.init()) !=
                BookClusterEngine.cacheParams({
                    var o = ClusterOptions(); o.scheme = .charShingles; return o
                }()), "两套方案缓存命名空间不同")

        // 精确 Jaccard
        check(approx(exactJaccard([1, 2, 3], [2, 3, 4]), 0.5), "精确 Jaccard 交2并4")
        check(approx(exactJaccard([1, 2], [1, 2]), 1.0), "相同集合 Jaccard=1")
        check(approx(exactJaccard([], [1, 2]), 0.0), "空集合 Jaccard=0")

        // 图
        let g = WeightedGraph(n: 4, edges: [(0, 1, 0.8), (1, 2, 0.9), (2, 3, 0.1)])
        check(g.edgeCount == 3, "图边数")
        check(approx(g.twoM, 2 * (0.8 + 0.9 + 0.1)), "图总权重 2m")

        // Leiden：两个内部紧密、之间只有弱桥的团 → 应分成 2 个社区
        var edges: [(Int, Int, Double)] = []
        for i in 0..<4 { for j in (i + 1)..<4 { edges.append((i, j, 0.9)) } }
        for i in 4..<8 { for j in (i + 1)..<8 { edges.append((i, j, 0.9)) } }
        edges.append((3, 4, 0.05))                 // 弱桥
        edges.append((8, 9, 0.4))                 // 第三个小团
        let g2 = WeightedGraph(n: 10, edges: edges)
        let r = Leiden.cluster(g2, resolution: 1.0)
        check(r.communityCount >= 2, "Leiden 至少两个社区")
        check(r.assignment[0] == r.assignment[1] && r.assignment[4] == r.assignment[5],
              "团内节点同社区")
        check(r.assignment[0] != r.assignment[4], "两个团分到不同社区")
        let isolatedGraph = WeightedGraph(n: 3, edges: [])
        let ri = Leiden.cluster(isolatedGraph)
        check(ri.communityCount == 3 && ri.assignment == [0, 1, 2], "无边的孤立节点各自成簇")
        let oneEdge = WeightedGraph(n: 3, edges: [(0, 1, 0.5)])
        let ro = Leiden.cluster(oneEdge)
        check(ro.assignment[0] == ro.assignment[1] && ro.assignment[2] != ro.assignment[0],
              "单边两端同社区，孤立节点独立")

        // 分辨率 γ 与簇数的关系：γ 越大越碎 → 簇数不少于低 γ。
        // 用 5 个内部紧密的 5-clique 加弱桥，簇数在 [1, 25] 间可调。
        var tunableEdges: [(Int, Int, Double)] = []
        for c in 0..<5 {
            for i in 0..<5 {
                for j in (i + 1)..<5 {
                    tunableEdges.append((c * 5 + i, c * 5 + j, 1.0))
                }
            }
        }
        for c in 0..<4 { tunableEdges.append((c * 5 + 4, (c + 1) * 5, 0.02)) }
        let tunable = WeightedGraph(n: 25, edges: tunableEdges)
        let lowGamma = Leiden.cluster(tunable, resolution: 0.001)
        let highGamma = Leiden.cluster(tunable, resolution: 100)
        check(highGamma.communityCount >= lowGamma.communityCount,
              "γ 增大时簇数不减少（低γ=\(lowGamma.communityCount) 高γ=\(highGamma.communityCount)）")

        // 二分搜索：多个不同目标都应收敛到「历史误差最小」的 γ。
        for target in [2, 5, 10, 25] {
            var cfg = ResolutionSearch.Config(target: target)
            cfg.minGamma = 0.001
            cfg.maxGamma = 100
            cfg.maxSteps = 14
            let outcome = ResolutionSearch.run(tunable, config: cfg)
            // 端点即命中时只探 2 次并提前退出；未命中则必须进入二分。
            check(outcome.trials.count >= 2, "目标\(target)至少探测两端")
            check(outcome.bestError == 0 || outcome.trials.count > 2,
                  "目标\(target)未命中端点时执行二分")
            // 预算 = 两端 + 最多两侧各 3 次扩张 + 二分步数 + 8 次邻域细化
            let budget = cfg.maxSteps + 2 + 2 * cfg.maxExpansions + 8
            check(outcome.trials.count <= budget, "目标\(target)探测次数受预算约束")
            check(outcome.bestError == outcome.trials.map { abs($0.communities - target) }.min(),
                  "目标\(target)返回历史误差最小点")
            check(outcome.result.communityCount == outcome.trials
                    .first { $0.gamma == outcome.gamma }?.communities,
                  "目标\(target)返回的结果与最终 γ 一致")
            check(outcome.gamma >= cfg.minGamma && outcome.gamma <= cfg.maxGamma,
                  "目标\(target)的 γ 落在搜索区间内")
        }

        // 容差提前退出 / 目标不可达时的兜底
        var tolCfg = ResolutionSearch.Config(target: 5)
        tolCfg.tolerance = 100
        let tolOutcome = ResolutionSearch.run(tunable, config: tolCfg)
        check(tolOutcome.reached && tolOutcome.trials.count <= 3, "容差命中即提前停止")
        var impossibleCfg = ResolutionSearch.Config(target: 1)
        impossibleCfg.maxSteps = 6
        let impossible = ResolutionSearch.run(oneEdge, config: impossibleCfg)
        check(impossible.isolatedFloor == 1, "孤立簇下界计算正确")
        check(impossible.achievableMinClusters == 2,
              "可达下界=孤立数+有边连通分量数")
        check(impossible.result.communityCount >= 2 && !impossible.reached,
              "目标低于可达下界时标记不可达并取最接近值")
        check(impossible.trials.count <= 2, "目标低于硬下界时不浪费邻域探测")
        var expandCfg = ResolutionSearch.Config(target: 25)
        expandCfg.minGamma = 0.001
        expandCfg.maxGamma = 0.01
        expandCfg.maxSteps = 4
        let expandedOutcome = ResolutionSearch.run(tunable, config: expandCfg)
        check(expandedOutcome.expandedBounds, "目标超出初始区间时自动扩张 γ 上界")
        let niCfg = ResolutionSearch.Config(target: 2, mode: .nonisolated)
        let niOutcome = ResolutionSearch.run(oneEdge, config: niCfg)
        check(niOutcome.effectiveTarget == 2 + niOutcome.isolatedFloor,
              "nonisolated 模式把孤立簇数加到目标上")

        // CLI 新参数解析
        let parsed = ClusterCLI.parse(["--target-clusters", "7", "--target-mode", "nonisolated",
                                       "--target-tolerance", "2", "--resolution-min", "0.01",
                                       "--resolution-max", "20", "--search-steps", "9",
                                       "--shingle-df-cut", "0.8", "--force-clusters", "60",
                                       "--word-df-cut", "0.7", "--anchor-sim-min", "0.05",
                                       "--split-max", "5000", "--split-gamma", "8"])
        check(parsed.targetClusters == 7 && parsed.targetMode == .nonisolated
                && parsed.targetTolerance == 2 && parsed.searchSteps == 9
                && approx(parsed.resolutionMin, 0.01) && approx(parsed.resolutionMax, 20)
                && approx(parsed.shingleDfCut, 0.8) && parsed.forceClusters == 60
                && approx(parsed.wordDfCut, 0.7) && approx(parsed.anchorSimMin, 0.05)
                && parsed.splitMax == 5000 && approx(parsed.splitGamma, 8.0),
              "目标簇数相关 CLI 参数解析")

        // 默认值锁定：threshold 0.2 / force 200 / anchor 0.05 / split 5000 / γ 5
        let defOpts = ClusterOptions()
        check(approx(defOpts.threshold, 0.2) && defOpts.forceClusters == 200
                && approx(defOpts.anchorSimMin, 0.05) && defOpts.splitMax == 5000
                && approx(defOpts.splitGamma, 5.0),
              "threshold/force-clusters/anchor-sim-min/split-max/split-gamma 默认值")

        // 簇命名
        var d0 = BookDocument(path: "/a.md", name: "a.md", words: ["金融", "数据", "支付", "平台"], freqs: [10, 8, 6, 4])
        var d1 = BookDocument(path: "/b.md", name: "b.md", words: ["金融", "数据", "众筹", "平台"], freqs: [9, 7, 5, 3])
        d0.featureIDs = [0, 1, 2, 3]
        d1.featureIDs = [0, 1, 4, 3]
        let docs = [d0, d1]
        let top = ClusterNaming.topWords([0, 1], docs: docs)
        check(top.count == 3, "簇名取前三词")
        check(top[0] == "金融" || top[0] == "数据" || top[0] == "平台", "前三词为高频共有词")
        check(!ClusterNaming.sanitize("金融/数据:平台*?").contains("/"), "簇名清洗非法字符")
        check(ClusterNaming.clusterName([0], docs: docs).count > 0, "单文档簇名非空")

        // 方案二命名：高频 shingle 重叠拼接还原连续片段
        check(ClusterNaming.assembleFragments(["量子纠缠", "子纠缠论"], k: 4)
                == ["量子纠缠论"], "重叠 k-1 字符拼接")
        check(ClusterNaming.assembleFragments(["量子纠缠", "子纠缠论", "纠缠论导"], k: 4)
                == ["量子纠缠论导"], "多次拼接收敛为单片段")
        check(ClusterNaming.assembleFragments(["量子纠缠", "金融数据"], k: 4)
                .count == 2, "无重叠则保留碎片")
        var s0 = BookDocument(path: "/a.md", name: "a.md",
                              words: ["量子纠缠", "子纠缠论", "纠缠论导", "理论物理"],
                              freqs: [1, 1, 1, 1])
        var s1 = BookDocument(path: "/b.md", name: "b.md",
                              words: ["量子纠缠", "子纠缠论", "纠缠体系", "理论物理"],
                              freqs: [1, 1, 1, 1])
        s0.featureIDs = [1, 2, 3, 4]
        s1.featureIDs = [1, 2, 5, 4]
        let sname = ClusterNaming.clusterName([0, 1], docs: [s0, s1],
                                              scheme: .charShingles, shingleK: 4)
        check(!sname.isEmpty && sname.count >= 3, "方案二簇名非空且含拼接片段")
        check(sname.contains("量子纠缠"), "方案二簇名还原出连续词段")

        // 文档级停用词：4 文档、cut=0.75 → cutoff=3，出现在 ≥3 文档的 shingle 被剔除
        let wA = "量子纠缠", wB = "子纠缠论", wC = "纠缠理论", wD = "论理物性"
        func h(_ w: String) -> UInt64 { mod61(fnv1a64(w)) }
        var fDocs = [
            BookDocument(path: "/1", name: "1", words: [wA, wB], freqs: [1, 1]),
            BookDocument(path: "/2", name: "2", words: [wA, wC], freqs: [1, 1]),
            BookDocument(path: "/3", name: "3", words: [wA, wD], freqs: [1, 1]),
            BookDocument(path: "/4", name: "4", words: [wB, wC], freqs: [1, 1]),
        ]
        fDocs[0].featureIDs = [h(wA), h(wB)].sorted()
        fDocs[1].featureIDs = [h(wA), h(wC)].sorted()
        fDocs[2].featureIDs = [h(wA), h(wD)].sorted()
        fDocs[3].featureIDs = [h(wB), h(wC)].sorted()
        let removed = CharShingleFeatures.filterCommonShingles(&fDocs, cut: 0.75)
        check(removed == 1, "DF 过滤剔除 1 个全局高频 shingle（实际 \(removed)）")
        check(!fDocs[0].featureIDs.contains(h(wA)) && fDocs[0].featureIDs.contains(h(wB)),
              "高频特征被剔除、低频保留")
        check(!fDocs[0].words.contains(wA), "命名集合同步剔除高频片段")
        check(fDocs[3].words.contains(wB) && fDocs[3].words.contains(wC),
              "未达阈值的文档不受影响")
        check(CharShingleFeatures.filterCommonShingles(&fDocs, cut: 0) == 0, "cut=0 关闭过滤")

        // 方案一文档级停用词：5 文档、cut=0.5 → cutoff=3，「中国经济」DF=5 被剔除，
        // 专业词（区块链/量子纠缠/文化发展）DF=2 保留
        let wt = WordTable()
        let fam = HashFamily()
        func mkWD(_ i: Int, _ ws: [String]) -> BookDocument {
            var d = BookDocument(path: "/w\(i)", name: "w\(i)", words: ws, freqs: ws.map { _ in 1 })
            var ids = Set<UInt64>()
            for w in ws { ids.insert(UInt64(wt.id(w))) }
            d.featureIDs = ids.sorted()
            d.signature = MinHash.signature(ws, family: fam)
            return d
        }
        var wDocs = [
            mkWD(0, ["中国经济", "区块链", "量子纠缠"]),
            mkWD(1, ["中国经济", "区块链"]),
            mkWD(2, ["中国经济", "量子纠缠"]),
            mkWD(3, ["中国经济", "文化发展"]),
            mkWD(4, ["中国经济", "文化发展"]),
        ]
        let removedW = BookClusterEngine.filterCommonWords(&wDocs, cut: 0.5, table: wt, family: fam)
        check(removedW == 1, "DF 过滤剔除 1 个泛化核心词（实际 \(removedW)）")
        check(wDocs[0].words == ["区块链", "量子纠缠"], "泛化词被剔除、专业词保留")
        check(wDocs[0].freqs == [1, 1], "freqs 同步过滤")
        check(wDocs[3].words.contains("文化发展"), "DF 低于阈值的专业词不受影响")
        check(wDocs[0].signature == MinHash.signature(["区块链", "量子纠缠"], family: fam),
              "签名按过滤后词表重算")
        check(wDocs[0].featureIDs
                == [UInt64(wt.id("区块链")), UInt64(wt.id("量子纠缠"))].sorted(),
              "featureIDs 同步重算")
        check(BookClusterEngine.filterCommonWords(&wDocs, cut: 0, table: wt, family: fam) == 0,
              "word cut=0 关闭过滤")

        // --force-clusters：K 簇硬合并
        func mkDoc(_ i: Int, sig: [UInt64]) -> BookDocument {
            var d = BookDocument(path: "/f\(i)", name: "f\(i)", words: ["词\(i)"], freqs: [1])
            d.signature = sig
            return d
        }
        let sA: [UInt64] = [1, 2, 3, 4]
        let sB: [UInt64] = [9, 8, 7, 6]
        let nlog: (String) -> Void = { _ in }
        // Kruskal：相似度最高的簇对先合并
        let kd = [mkDoc(0, sig: sA), mkDoc(1, sig: sA), mkDoc(2, sig: sB)]
        let kMerged = BookClusterEngine.forceMerge([0: [0], 1: [1], 2: [2]], docs: kd,
                                                   edges: [(0, 1, 0.9), (0, 2, 0.5), (1, 2, 0.6)],
                                                   target: 2, log: nlog)
        check(kMerged.count == 2 && kMerged.values.contains { Set($0) == [0, 1] },
              "force 合并按簇间边相似度降序（0.9 先并）")
        // 锚簇吸收：签名相似的簇并入锚簇；无相似信息的均衡合并
        let ad = [mkDoc(0, sig: sA), mkDoc(1, sig: sA), mkDoc(2, sig: sA), mkDoc(3, sig: sB)]
        let aMerged = BookClusterEngine.forceMerge([0: [0], 1: [1], 2: [2], 3: [3]],
                                                   docs: ad, edges: [],
                                                   target: 2, anchorLimit: 2, log: nlog)
        check(aMerged.count == 2, "锚吸收+均衡合并收敛到目标簇数")
        check(aMerged.values.contains { $0.contains(0) && $0.contains(2) },
              "签名相似的簇被锚簇吸收（doc2 并入 doc0 所在簇）")
        check(aMerged.values.allSatisfy { $0.contains(0) ? !$0.contains(3) : true },
              "签名无关的簇不被误吸")
        // 均衡兜底：全零相似时最小簇两两合并
        let bd = [mkDoc(0, sig: sA), mkDoc(1, sig: sB), mkDoc(2, sig: sA), mkDoc(3, sig: sB)]
        let bMerged = BookClusterEngine.forceMerge(
            [0: [0], 1: [1], 2: [2], 3: [3]], docs: bd, edges: [],
            target: 2, anchorLimit: 1, log: nlog)
        check(bMerged.count == 2 && bMerged.values.allSatisfy { $0.count == 2 },
              "全零相似时按最小簇均衡合并")
        // target ≥ 当前簇数时不拆分
        let cMerged = BookClusterEngine.forceMerge([0: [0], 1: [1]], docs: bd, edges: [],
                                                   target: 5, log: nlog)
        check(cMerged.count == 2, "目标大于当前簇数时保持不变")
        // 锚簇吸收门槛：弱信号（sim=0.25 < 0.5）不并入锚簇，强信号（sim=1.0）仍吸收
        let sP: [UInt64] = [1, 99, 98, 97]           // 与 sA 仅 1/4 维相同 → sim 0.25
        let gd = [mkDoc(0, sig: sA), mkDoc(1, sig: sA), mkDoc(2, sig: sP), mkDoc(3, sig: sB)]
        let g0 = BookClusterEngine.forceMerge([0: [0], 1: [1], 2: [2], 3: [3]],
                                              docs: gd, edges: [],
                                              target: 2, anchorLimit: 2,
                                              anchorSimMin: 0, log: nlog)
        check(g0.values.contains { $0.contains(0) && $0.contains(2) },
              "无门槛时弱信号簇也并入锚簇")
        let gMerged = BookClusterEngine.forceMerge([0: [0], 1: [1], 2: [2], 3: [3]],
                                                   docs: gd, edges: [],
                                                   target: 2, anchorLimit: 2,
                                                   anchorSimMin: 0.5, log: nlog)
        check(gMerged.count == 2 && !gMerged.values.contains { $0.contains(0) && $0.contains(2) },
              "吸收门槛阻止弱信号簇并入锚簇（改走均衡兜底）")
        // 超大簇二次聚类：两个 5 节点团 + 弱桥构成 10 节点单簇 → 拆成 ≤6 的子簇
        var sd: [BookDocument] = []
        for i in 0..<10 { sd.append(BookDocument(path: "/s\(i)", name: "s\(i)", words: [], freqs: [])) }
        var sEdges: [(Int, Int, Double)] = []
        for c in 0..<2 { for i in 0..<5 { for j in (i + 1)..<5 { sEdges.append((c * 5 + i, c * 5 + j, 0.9)) } } }
        sEdges.append((4, 5, 0.1))
        let sp = BookClusterEngine.splitOversized([0: Array(0..<10)], edges: sEdges,
                                                  max: 6, gamma: 1.0, log: nlog)
        check(sp.count >= 2 && sp.values.allSatisfy { $0.count <= 6 },
              "超大簇拆分为 ≤max 的子簇（实际 \(sp.count) 簇）")
        check(sp.values.flatMap { $0 }.sorted() == Array(0..<10), "拆分不丢文档")
        let sp2 = BookClusterEngine.splitOversized([0: Array(0..<3)], edges: sEdges,
                                                   max: 10, gamma: 1.0, log: nlog)
        check(sp2.count == 1, "低于阈值的簇不拆分")

        // 端到端小样例 + dry-run 不移动 + apply/undo
        do {
            let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("cluster-selftest-\(UUID().uuidString)")
            let root = tmp.appendingPathComponent("books")
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: tmp) }
            // 每个主题词配 8 种不同左右邻字 → 左右熵=3，能被核心词提取器命中；
            // 同类两本书用同一批词（Jaccard≈1），异类用完全不同的词（Jaccard≈0）。
            func makeBook(_ words: [String]) -> String {
                let left = Array("甲乙丙丁戊己庚辛")
                let right = Array("壬癸子丑寅卯辰巳")
                var lines: [String] = []
                for w in words {
                    for i in 0..<8 { lines.append("\(left[i])\(w)\(right[(i + 3) % 8])") }
                }
                return lines.joined(separator: "\n")
            }
            let topicA = makeBook(["金融数据", "支付平台", "互联网征信", "众筹融资", "银行贷款"])
            let topicB = makeBook(["佛教大乘", "菩萨涅槃", "唯识般若", "密教修行", "如来缘起"])
            try topicA.write(to: root.appendingPathComponent("a1.md"), atomically: true, encoding: .utf8)
            try topicA.write(to: root.appendingPathComponent("a2.md"), atomically: true, encoding: .utf8)
            try topicB.write(to: root.appendingPathComponent("b1.md"), atomically: true, encoding: .utf8)
            try topicB.write(to: root.appendingPathComponent("b2.md"), atomically: true, encoding: .utf8)
            try "完全无关的零散文字内容避免成词".write(to: root.appendingPathComponent("c.md"),
                                                  atomically: true, encoding: .utf8)

            var opts = ClusterOptions()
            opts.inputDir = root.path
            opts.outRoot = tmp.appendingPathComponent("out").path
            opts.cachePath = tmp.appendingPathComponent("cache.json").path
            opts.reportPath = tmp.appendingPathComponent("report.json").path
            opts.manifestPath = tmp.appendingPathComponent("manifest.json").path
            opts.maxChars = 10000
            opts.threshold = 0.3
            opts.jobs = 2
            _ = BookClusterEngine.run(opts)
            check(FileManager.default.fileExists(atPath: opts.reportPath), "dry-run 生成报告")
            check(FileManager.default.fileExists(atPath: root.appendingPathComponent("a1.md").path),
                  "dry-run 不移动源文件")
            if let data = try? Data(contentsOf: URL(fileURLWithPath: opts.reportPath)),
               let rep = try? JSONDecoder().decode(RunReport.self, from: data) {
                check(rep.documentCount == 5, "报告文档数=5")
                check(rep.applied == false, "报告标记 dry-run")
                check(rep.edges > 0, "同类文档之间建了边")
                check(rep.lshBands == 32 && rep.lshRowsPerBand == 4, "报告记录 LSH 32×4")
                check(rep.targetClusters == 0 && rep.resolutionTrials.isEmpty,
                      "未启用目标搜索时报告无轨迹")
                check(rep.forceClusters == 200 && rep.communities <= 200,
                      "默认 force-clusters 200（小样例簇数少不触发合并）")
            } else { failures.append("报告无法解码") }

            // 端到端目标簇数搜索：报告应带轨迹与最终 γ
            var targetOpts = opts
            targetOpts.targetClusters = 2
            targetOpts.targetMode = .nonisolated
            targetOpts.searchSteps = 6
            targetOpts.reportPath = tmp.appendingPathComponent("target-report.json").path
            _ = BookClusterEngine.run(targetOpts)
            if let data = try? Data(contentsOf: URL(fileURLWithPath: targetOpts.reportPath)),
               let rep2 = try? JSONDecoder().decode(RunReport.self, from: data) {
                check(rep2.targetClusters == 2 && rep2.targetMode == "nonisolated",
                      "报告记录目标簇数与口径")
                check(!rep2.resolutionTrials.isEmpty && rep2.resolution > 0,
                      "报告记录二分搜索轨迹与最终 γ")
                check(rep2.communities == rep2.clusters.count, "报告簇数与簇列表一致")
            } else { failures.append("目标簇数报告无法解码") }

            // 方案二端到端：字符 4-shingle，独立缓存，同类文档仍应建边
            var shOpts = opts
            shOpts.scheme = .charShingles
            shOpts.shingleLimit = 256
            // 小样例（4 文件）下 cutoff=2 会把所有共享 shingle 视为文档级停用词，
            // 端到端样例关闭 DF 过滤；DF 行为由下方单元测试单独覆盖。
            shOpts.shingleDfCut = 0
            shOpts.cachePath = tmp.appendingPathComponent("shingle-cache.json").path
            shOpts.reportPath = tmp.appendingPathComponent("shingle-report.json").path
            shOpts.threshold = 0.3
            _ = BookClusterEngine.run(shOpts)
            if let data = try? Data(contentsOf: URL(fileURLWithPath: shOpts.reportPath)),
               let rep3 = try? JSONDecoder().decode(RunReport.self, from: data) {
                check(rep3.featureScheme == "char-shingles" && rep3.shingleK == 4,
                      "报告记录方案二与 k=4")
                check(rep3.shingleDfCut == 0 && rep3.dfFilteredShingles == 0,
                      "报告记录 DF 过滤配置（端到端样例关闭）")
                check(rep3.avgRawShingles > 0 && rep3.avgKeptFeatures > 0,
                      "报告记录 shingle 裁剪前后规模")
                check(rep3.edges > 0, "方案二同类文档之间建了边")
                check(FileManager.default.fileExists(atPath: shOpts.cachePath),
                      "方案二独立缓存写入")
            } else { failures.append("方案二报告无法解码") }
            // 缓存复用：二次运行应全部命中
            _ = BookClusterEngine.run(shOpts)
            check(FileManager.default.fileExists(atPath: root.appendingPathComponent("a1.md").path),
                  "方案二 dry-run 不移动源文件")

            // --force-clusters 端到端：硬合并到 2 簇
            var fcOpts = opts
            fcOpts.forceClusters = 2
            fcOpts.reportPath = tmp.appendingPathComponent("force-report.json").path
            _ = BookClusterEngine.run(fcOpts)
            if let data = try? Data(contentsOf: URL(fileURLWithPath: fcOpts.reportPath)),
               let rep4 = try? JSONDecoder().decode(RunReport.self, from: data) {
                check(rep4.forceClusters == 2 && rep4.communities <= 2,
                      "force-clusters 硬合并到 ≤K 簇（实际 \(rep4.communities)）")
                check(rep4.clusters.count == rep4.communities, "硬合并后簇列表与簇数一致")
            } else { failures.append("force-clusters 报告无法解码") }

            // apply + undo 往返
            opts.apply = true
            _ = BookClusterEngine.run(opts)
            let manifestExists = FileManager.default.fileExists(atPath: opts.manifestPath)
            check(manifestExists, "apply 写 manifest")
            if manifestExists {
                let movedCount = (try? FileManager.default.contentsOfDirectory(atPath: opts.outRoot))?.count ?? 0
                check(movedCount > 0, "apply 创建了簇文件夹")
                var undoOpts = ClusterOptions()
                undoOpts.undoPath = opts.manifestPath
                check(BookClusterEngine.run(undoOpts) == 0, "undo 执行成功")
                check(FileManager.default.fileExists(atPath: root.appendingPathComponent("a1.md").path),
                      "undo 把文件移回原位")
            }
        } catch {
            failures.append("端到端异常: \(error)")
        }

        if failures.isEmpty {
            print("ClusterBooks self-test: all checks passed")
            return 0
        }
        for message in failures { FileHandle.standardError.write(Data("FAIL: \(message)\n".utf8)) }
        FileHandle.standardError.write(Data("\(failures.count) check(s) failed\n".utf8))
        return 1
    }
}

// MARK: - 入口

@main enum ClusterBooksMain {
    static func main() {
        exit(BookClusterEngine.run(ClusterCLI.parse(Array(CommandLine.arguments.dropFirst()))))
    }
}
