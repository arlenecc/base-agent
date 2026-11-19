// DedupeFiles.swift — 第三个工具：文本文件去重
//
// 两道流程：
//  第一道（完全相同）：文件字节流双 FNV-1a 128 位指纹 + 大小，同组保留路径排序第一个。
//  第二道（近重复）：代表文本（默认首部+尾部各取一半，共 --max-chars 字符；
//      --full-text 或 --max-chars 0 则用全文）→ 清理（剔标点/空格/换行，
//      汉字保留、英文转小写）→ 3-shingle 集合（滑窗去重）→ 停用词过滤 →
//      文档级停用词（DF ≥ --df-cut 剔除）→ MinHash 128 维签名 →
//      LSH 8 bands × 16 rows 召回 → 桶内粗筛：签名相同位数/128 > --sim（默认 0.85）
//      → 连通分量内贪心选代表：每个被删文档都与保留代表直接相似。
//
// dry-run 只报告；--apply 把重复文件移入 out-root 的 --duplicates-name 文件夹（不删除，
// 重名加序号），写 manifest 支持 --undo。
//
// 编译：swiftc -O -o /tmp/bookdedupe tools/CoreWordKit.swift tools/DedupeFiles.swift
// （复用 CoreWordKit 的 WordFilters 停用词表）

import Foundation
import Dispatch

// MARK: - 常量

private let ddSignatureDim = 128
private let ddBandCount = 8
private let ddBandRows = 16
private let ddMersenne61: UInt64 = (1 << 61) - 1
private let ddBandLayoutValid = (ddBandCount * ddBandRows == ddSignatureDim) && (ddBandRows > 0)

// MARK: - 哈希基础

private struct DDSplitMix64 {
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

private func ddfnv1aBytes(_ bytes: [UInt8], seed: UInt64) -> UInt64 {
    var h = seed
    for b in bytes {
        h ^= UInt64(b)
        h = h &* 0x100_0000_01B3
    }
    return h
}

private func ddfnv1aData(_ data: Data, seed: UInt64) -> UInt64 {
    var h = seed
    for b in data {
        h ^= UInt64(b)
        h = h &* 0x100_0000_01B3
    }
    return h
}

private func ddfnv1a64(_ text: String) -> UInt64 {
    var h: UInt64 = 0xCBF2_9CE4_8422_2325
    for byte in text.utf8 {
        h ^= UInt64(byte)
        h = h &* 0x100_0000_01B3
    }
    return h
}

@inline(__always)
private func ddMod61(_ x: UInt64) -> UInt64 {
    var v = (x & ddMersenne61) &+ (x >> 61)
    if v >= ddMersenne61 { v &-= ddMersenne61 }
    return v
}

/// 128 组独立仿射哈希函数（a*x+b mod 2^61-1）。
private struct DDFamily {
    let a: [UInt64]
    let b: [UInt64]

    init(seed: UInt64 = 0x51DE_D1CE_0DD1_7E5D) {
        var rng = DDSplitMix64(seed: seed)
        var aa: [UInt64] = []
        var bb: [UInt64] = []
        aa.reserveCapacity(ddSignatureDim)
        bb.reserveCapacity(ddSignatureDim)
        for _ in 0..<ddSignatureDim {
            aa.append(ddMod61(rng.next()) | 1)
            bb.append(ddMod61(rng.next()))
        }
        a = aa
        b = bb
    }

    @inline(__always)
    func value(_ i: Int, x: UInt64) -> UInt64 {
        let m = a[i].multipliedFullWidth(by: x)
        var low = m.low
        let high = m.high
        // 2^64 ≡ 8 (mod 2^61-1)
        low = low &+ (high &<< 3)
        if low < high &<< 3 { low &+= 8 }
        return ddMod61(low &+ b[i])
    }
}

/// MinHash：集合元素哈希在 128 个独立函数下各取最小值。
private enum DDMinHash {
    static func signature(fromHashes xs: [UInt64], family: DDFamily) -> [UInt64] {
        var sig = [UInt64](repeating: ddMersenne61, count: ddSignatureDim)
        guard !xs.isEmpty else { return sig }
        for x in xs {
            for i in 0..<ddSignatureDim {
                let h = family.value(i, x: x)
                if h < sig[i] { sig[i] = h }
            }
        }
        return sig
    }
}

// MARK: - 文本处理

private enum DDText {
    /// 单字符清理：汉字保留、英文字母转小写、其余（标点/空白/数字等）剔除。
    private static func kept(_ scalar: Unicode.Scalar) -> Character? {
        let v = scalar.value
        if (v >= 0x4E00 && v <= 0x9FFF)
            || (v >= 0x3400 && v <= 0x4DBF)
            || (v >= 0x20000 && v <= 0x2A6DF)
            || (v >= 0xF900 && v <= 0xFAFF) {
            return Character(scalar)
        } else if v >= 0x41 && v <= 0x5A {
            return UnicodeScalar(v + 32).map { Character($0) }
        } else if v >= 0x61 && v <= 0x7A {
            return Character(scalar)
        }
        return nil
    }

    /// 生成代表文本分段：
    ///  - fullText=true 或 maxChars<=0：全文（1 段）；
    ///  - 否则清理后全文 ≤ maxChars 时直接用全文（1 段），
    ///    否则取「前 maxChars/2 + 后 maxChars-maxChars/2」（2 段）。
    /// 分段返回便于分别生成 shingle，避免首/尾拼接处产生虚假跨段 k-gram。
    /// 两遍线性扫描（先计数再构造），无需把整本大书常驻内存。
    static func segments(_ text: String, maxChars: Int, fullText: Bool = false) -> [[Character]] {
        var total = 0
        for scalar in text.unicodeScalars where kept(scalar) != nil { total += 1 }
        let whole = fullText || maxChars <= 0 || total <= maxChars
        if whole {
            var out: [Character] = []
            out.reserveCapacity(total)
            for scalar in text.unicodeScalars {
                if let c = kept(scalar) { out.append(c) }
            }
            return [out]
        }
        let headLen = maxChars / 2
        let tailLen = maxChars - headLen
        let tailStart = total - tailLen
        var head: [Character] = []
        var tail: [Character] = []
        head.reserveCapacity(headLen)
        tail.reserveCapacity(tailLen)
        var idx = 0
        for scalar in text.unicodeScalars {
            guard let c = kept(scalar) else { continue }
            if idx < headLen { head.append(c) }
            if idx >= tailStart { tail.append(c) }
            idx += 1
        }
        return [head, tail]
    }

    /// 便捷包装（自检/兼容）：返回拼接后的代表文本。
    static func clean(_ text: String, maxChars: Int, fullText: Bool = false) -> [Character] {
        return segments(text, maxChars: maxChars, fullText: fullText).flatMap { $0 }
    }

    /// 滑窗提取长度 k 的子串，Set 自动去重。
    static func shingles(_ chars: [Character], k: Int) -> Set<String> {
        guard k > 0, chars.count >= k else { return [] }
        var result = Set<String>()
        result.reserveCapacity(min(chars.count - k + 1, 1 << 18))
        for i in 0...(chars.count - k) {
            result.insert(String(chars[i..<(i + k)]))
        }
        return result
    }

    /// 纯 ASCII 高频 3-gram：英文功能片段停用词。
    static let ascii3Stops: Set<String> = [
        "the", "and", "ing", "her", "ere", "ent", "tha", "ith", "hat", "for", "ted",
        "his", "thi", "ver", "was", "are", "not", "all", "ont", "you", "ave", "ell",
    ]

    /// 停用判定：命中词表 / 全虚字 / 英文功能 3-gram。
    static func isStop(_ s: String, extra: Set<String>) -> Bool {
        if WordFilters.builtinStopwords.contains(s) || extra.contains(s) { return true }
        if s.unicodeScalars.allSatisfy({ WordFilters.weakBoundaryChars.contains(Character($0)) }) {
            return true
        }
        if s.utf8.allSatisfy({ $0 >= 97 && $0 <= 122 }) && ascii3Stops.contains(s) { return true }
        return false
    }
}

// MARK: - 数据结构

private struct DDCacheEntry: Codable {
    let signature: String      // size-mtime 文件指纹（变更失效）
    let sig: [UInt64]          // DF 过滤后的 MinHash 128 维签名
    let rawShingles: Int
    let chars: Int
}

private struct DDCache: Codable {
    let version: Int
    let params: String
    var entries: [String: DDCacheEntry]
}

private struct DDMove: Codable {
    let src: String
    let dst: String
}

private struct DDManifest: Codable {
    let version: Int
    let createdAt: String
    let outRoot: String
    let moves: [DDMove]
}

private struct DDGroup: Codable {
    let representative: String
    let members: [String]
    let size: Int
    let kind: String            // "exact" / "fuzzy" / "partial"（partial 不移动）
}

private struct DDReport: Codable {
    let inputDir: String
    let outRoot: String
    let fileCount: Int
    let exactDupGroups: Int
    let exactDupFiles: Int
    let fuzzyDupGroups: Int
    let fuzzyDupFiles: Int
    let partialGroups: Int
    let partialFiles: Int
    let totalRemoved: Int
    let bytesSaved: Int64
    let avgShingles: Int
    let dfFilteredShingles: Int
    let candidatePairs: Int
    let sampleMode: String
    let sampleMaxChars: Int
    let elapsedSeconds: Double
    let applied: Bool
    let groups: [DDGroup]
}

private struct DDFile {
    let path: String
    let name: String
    var size: Int64 = 0
    var fp1: UInt64 = 0
    var fp2: UInt64 = 0
    var keptShingles = 0        // DF 过滤后的 shingle 数（报告统计用）
    var rawShingles = 0
    var chars = 0
    var sig: [UInt64] = []
}

// MARK: - 算法

private enum DDAlgo {
    /// 从样本统计全局高频 shingle（文档级停用词）：出现在 ≥ cut 比例文档中的哈希。
    /// 全量 shingle 数十亿无法驻留内存，用子样本统计（阈值放宽 0.8 倍补偿抽样噪声）
    /// 即可稳定识别「绝大多数文档都出现」的片段——专业词 DF 远低于阈值不受影响。
    /// 返回高频哈希集合。
    static func commonIDs(fromSample sets: [[UInt64]], cut: Double) -> Set<UInt64> {
        guard cut > 0, !sets.isEmpty else { return [] }
        let cutoff = max(2, Int((Double(sets.count) * cut * 0.8).rounded(.up)))
        var all = [UInt64]()
        all.reserveCapacity(sets.reduce(0) { $0 + $1.count })
        for s in sets { all.append(contentsOf: s) }
        guard !all.isEmpty else { return [] }
        all.sort()
        var common = Set<UInt64>()
        var i = 0
        while i < all.count {
            var j = i
            while j < all.count && all[j] == all[i] { j += 1 }
            if j - i >= cutoff { common.insert(all[i]) }
            i = j
        }
        return common
    }

    /// 粗筛相似度：128 维签名相同位数 / 128。
    @inline(__always)
    static func agreement(_ a: [UInt64], _ b: [UInt64]) -> Double {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var same = 0
        for i in 0..<a.count where a[i] == b[i] { same += 1 }
        return Double(same) / Double(a.count)
    }
}

private extension DDAlgo {
    /// LSH 分桶召回候选对：签名切 bands 段、每段 rows 维，段内全同进同桶。
    static func candidates(_ sigs: [[UInt64]], maxBucketSize: Int, maxPairs: Int)
        -> (pairs: Set<UInt64>, skipped: Int) {
        guard ddBandLayoutValid else { return ([], 0) }
        var buckets: [[UInt64]: [Int]] = [:]
        for (idx, sig) in sigs.enumerated() {
            guard sig.count == ddSignatureDim else { continue }
            // 全 prime 哨兵签名 = 空 shingle 集合，彼此不相似，跳过避免巨桶。
            if sig.allSatisfy({ $0 == ddMersenne61 }) { continue }
            for band in 0..<ddBandCount {
                let start = band * ddBandRows
                buckets[Array(sig[start..<(start + ddBandRows)]), default: []].append(idx)
            }
        }
        var pairs = Set<UInt64>()
        var skipped = 0
        for (_, members) in buckets {
            guard members.count >= 2 else { continue }
            if members.count > maxBucketSize { skipped += 1; continue }
            for i in 0..<members.count {
                for j in (i + 1)..<members.count {
                    let a = min(members[i], members[j])
                    let b = max(members[i], members[j])
                    pairs.insert(UInt64(a) << 32 | UInt64(b))
                }
            }
            if pairs.count >= maxPairs { break }
        }
        return (pairs, skipped)
    }
}

private extension DDAlgo {
    /// 连通分量内选代表：强制选分量内**最大文件**作唯一代表，保证大文件永不被删；
    /// 只删与最大文件**直接相似（共享边）**的成员——被删文件必与保留代表直接高度相似，
    /// 且绝不会比保留文件更大。与最大文件无直接相似的成员（如链末端）保留，不误删。
    /// 返回 (保留代表, 被删文档及其担保代表(=最大文件)与相似度)。
    static func greedyKeep(_ n: Int, _ pairs: [(Int, Int, Double)], sizes: [Int64]? = nil)
        -> (keepers: [Int], removed: [(dup: Int, rep: Int, sim: Double)]) {
        var adj = [[Int]](repeating: [], count: n)
        var simOf: [UInt64: Double] = [:]
        for (u, v, s) in pairs {
            let key = UInt64(min(u, v)) << 32 | UInt64(max(u, v))
            adj[u].append(v)
            adj[v].append(u)
            simOf[key] = max(simOf[key] ?? 0, s)
        }
        var visited = [Bool](repeating: false, count: n)
        var keepers: [Int] = []
        var removed: [(dup: Int, rep: Int, sim: Double)] = []
        for start in 0..<n where !visited[start] {
            var comp: [Int] = []
            var stack = [start]
            visited[start] = true
            while let x = stack.popLast() {
                comp.append(x)
                for y in adj[x] where !visited[y] {
                    visited[y] = true
                    stack.append(y)
                }
            }
            let biggest = comp.max(by: { (sizes?[$0] ?? 0) < (sizes?[$1] ?? 0) }) ?? comp[0]
            for x in comp where x != biggest {
                let key = UInt64(min(x, biggest)) << 32 | UInt64(max(x, biggest))
                if let s = simOf[key], s > 0 {
                    removed.append((x, biggest, s))   // 与最大文件直接相似 → 删除
                } else {
                    keepers.append(x)                 // 仅与中间节点相似 → 保留
                }
            }
            keepers.append(biggest)
        }
        for i in 0..<n where !visited[i] { keepers.append(i) }
        return (keepers.sorted(), removed.sorted { $0.dup < $1.dup })
    }
}

// MARK: - 主流程

enum FileDedupe {
    struct Options {
        var inputDir = ""
        var outRoot = ""            // 默认 = 输入目录
        var limit = 0               // 0 = 全部
        var maxChars = 20000        // 代表文本总长（首+尾各半），0 = 全文
        var fullText = false        // 全文计算，忽略 maxChars（--full-text）
        var shingleK = 3
        var simMin = 0.85           // 签名粗筛阈值
        var sizeRatio = 3.0         // 判重的大小比上限：超过改标「部分重复」不动，0=关闭
        var dfCut = 0.5             // 文档级停用词阈值，0 = 关闭
        var noStopwords = false
        var stopwordsPath = ""
        var maxBucketSize = 2000
        var maxPairs = 4_000_000
        var jobs = 0                // 0 = 自动
        var cachePath = ""
        var reportPath = ""
        var manifestPath = ""
        var apply = false
        var undoPath = ""
        var duplicatesName = "_重复"
        var exts = ["md", "txt"]
        var selfTest = false

        var sampleModeName: String { (fullText || maxChars <= 0) ? "full" : "headtail" }
    }

    static func log(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

    static func fileStamp(_ path: String) -> String {
        let fm = FileManager.default
        guard let attrs = try? fm.attributesOfItem(atPath: path),
              let size = attrs[.size] as? Int64,
              let mtime = attrs[.modificationDate] as? Date else { return "missing" }
        return "\(size)-\(Int(mtime.timeIntervalSince1970))"
    }

    static func ensureParent(_ path: String) {
        let parent = (path as NSString).deletingLastPathComponent
        try? FileManager.default.createDirectory(atPath: parent, withIntermediateDirectories: true)
    }

    static func writeJSON<T: Encodable>(_ value: T, path: String) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(value)
        ensureParent(path)
        try data.write(to: URL(fileURLWithPath: path), options: .atomic)
    }

    static func cacheParams(_ o: Options) -> String {
        return "k=\(o.shingleK)|mode=\(o.sampleModeName)|maxChars=\(o.maxChars)"
            + "|stop=\(!o.noStopwords)|df=\(o.dfCut)|sw=\(o.stopwordsPath)"
            + "|swv=\(WordFilters.stopwordsVersion)"
    }
}

private extension FileDedupe {
    /// 并发提取（三阶段流式，内存可控）：
    ///  1) 并发读文件 → 双指纹（第一道）+ 清理 + shingle 哈希写入临时二进制文件；
    ///     缓存命中（签名已算）直接填充。全量 shingle 数十亿无法驻留，必须落盘。
    ///  2) 抽样（约 5000 份）统计 DF，识别文档级停用词。
    ///  3) 并发从临时文件流式读取 → 剔除高频 → MinHash 签名，写回缓存。
    static func extractAll(_ files: [String], options: Options)
        -> (docs: [DDFile], cache: DDCache, dfFiltered: Int) {
        let params = cacheParams(options)
        var cached = DDCache(version: 3, params: params, entries: [:])
        if !options.cachePath.isEmpty,
           let data = try? Data(contentsOf: URL(fileURLWithPath: options.cachePath)),
           let decoded = try? JSONDecoder().decode(DDCache.self, from: data),
           decoded.version == 3, decoded.params == params {
            cached = decoded
        }

        if options.sampleModeName == "full" {
            log("[dedupe] 代表文本：全文计算（不截取）")
        } else {
            let h = options.maxChars / 2
            log("[dedupe] 代表文本：首 \(h) + 尾 \(options.maxChars - h) = \(options.maxChars) 字符")
        }
        let family = DDFamily()
        let extraStop = WordFilters.loadExtra(options.stopwordsPath)
        var results = [DDFile?](repeating: nil, count: files.count)
        let lock = NSLock()
        var done = 0
        let total = files.count
        let tmpDir = NSTemporaryDirectory() + "/dd-tmp-\(UUID().uuidString)"
        try? FileManager.default.createDirectory(atPath: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: tmpDir) }

        // 阶段 1：并发提取（指纹 + shingle 哈希落盘；缓存命中直接填签名）
        var needsDF: [Int] = []
        DispatchQueue.concurrentPerform(iterations: files.count) { i in
            let path = files[i]
            let name = (path as NSString).lastPathComponent
            var doc = DDFile(path: path, name: name, size: 0)
            defer {
                lock.lock()
                done += 1
                if done % 5000 == 0 || done == total {
                    log("[dedupe] 提取进度 \(done)/\(total)")
                }
                lock.unlock()
            }
            guard let bytes = FileManager.default.contents(atPath: path) else { return }
            doc.size = Int64(bytes.count)
            doc.fp1 = ddfnv1aData(bytes, seed: 0xCBF2_9CE4_8422_2325)
            doc.fp2 = ddfnv1aData(bytes, seed: 0xFEED_BEEF_CAFE_1234)

            if let entry = cached.entries[path], entry.signature == fileStamp(path) {
                doc.sig = entry.sig
                doc.keptShingles = entry.rawShingles
                doc.rawShingles = entry.rawShingles
                doc.chars = entry.chars
                results[i] = doc
                return
            }
            guard let text = String(data: Data(bytes), encoding: .utf8) else { return }
            let segs = DDText.segments(text, maxChars: options.maxChars, fullText: options.fullText)
            var raw = Set<String>()
            for seg in segs { raw.formUnion(DDText.shingles(seg, k: options.shingleK)) }
            let keptShingles = options.noStopwords ? raw : Set(raw.filter { !DDText.isStop($0, extra: extraStop) })
            var hashes = keptShingles.map { ddMod61(ddfnv1a64($0)) }
            hashes.sort()
            doc.rawShingles = raw.count
            doc.chars = segs.reduce(0) { $0 + $1.count }
            var buf = Data(capacity: hashes.count * 8)
            for h in hashes {
                withUnsafeBytes(of: h.littleEndian) { buf.append(contentsOf: $0) }
            }
            try? buf.write(to: URL(fileURLWithPath: tmpDir + "/\(i).bin"))
            results[i] = doc
            lock.lock()
            needsDF.append(i)
            lock.unlock()
        }
        log("[dedupe] 提取完成 \(files.count) 个文件，其中 \(needsDF.count) 个待算签名")

        // 阶段 2：抽样 DF 统计（文档级停用词）
        let stride = max(1, needsDF.count / 5000)
        var sampleSets: [[UInt64]] = []
        var k = 0
        while k < needsDF.count {
            if let data = try? Data(contentsOf: URL(fileURLWithPath: tmpDir + "/\(needsDF[k]).bin")) {
                sampleSets.append(dataAsHashes(data))
            }
            k += stride
        }
        let common = DDAlgo.commonIDs(fromSample: sampleSets, cut: options.dfCut)
        if !common.isEmpty {
            log("[dedupe] 文档级停用词：DF ≥ \(options.dfCut) 的 shingle \(common.count) 个已剔除（抽样 \(sampleSets.count) 份统计）")
        }

        // 阶段 3：并发签名（流式读临时文件 → 剔高频 → MinHash → 写缓存）
        DispatchQueue.concurrentPerform(iterations: needsDF.count) { j in
            let i = needsDF[j]
            guard var doc = results[i],
                  let data = try? Data(contentsOf: URL(fileURLWithPath: tmpDir + "/\(i).bin")) else { return }
            var hashes = dataAsHashes(data)
            if !common.isEmpty { hashes.removeAll { common.contains($0) } }
            doc.keptShingles = hashes.count
            doc.sig = DDMinHash.signature(fromHashes: hashes, family: family)
            results[i] = doc
            lock.lock()
            cached.entries[doc.path] = DDCacheEntry(signature: fileStamp(doc.path), sig: doc.sig,
                                                    rawShingles: doc.keptShingles, chars: doc.chars)
            lock.unlock()
        }
        return (results.map { $0 ?? DDFile(path: "", name: "", size: 0) }, cached, common.count)
    }

    /// Data → UInt64 数组（写入与读取同机器，直接按主机字节序拷贝）。
    static func dataAsHashes(_ data: Data) -> [UInt64] {
        let n = data.count / 8
        guard n > 0 else { return [] }
        var out = [UInt64](repeating: 0, count: n)
        out.withUnsafeMutableBytes { dst in
            data.withUnsafeBytes { src in
                _ = memcpy(dst.baseAddress, src.baseAddress, n * 8)
            }
        }
        return out
    }
}

private extension FileDedupe {
    /// 把重复文件移入 out-root 的重复文件夹；目标重名加序号，绝不覆盖。
    static func moveDuplicates(_ paths: [(src: String, size: Int64)], options: Options)
        -> (moved: Int, moves: [DDMove]) {
        let fm = FileManager.default
        let dir = (options.outRoot as NSString).appendingPathComponent(options.duplicatesName)
        try? fm.createDirectory(atPath: dir, withIntermediateDirectories: true)
        var moved = 0
        var moves: [DDMove] = []
        for (src, _) in paths {
            var dst = (dir as NSString).appendingPathComponent((src as NSString).lastPathComponent)
            if fm.fileExists(atPath: dst) {
                let base = (dst as NSString).deletingPathExtension
                let ext = (dst as NSString).pathExtension
                var suffix = 2
                repeat {
                    dst = ext.isEmpty ? "\(base)-\(suffix)" : "\(base)-\(suffix).\(ext)"
                    suffix += 1
                } while fm.fileExists(atPath: dst)
            }
            do {
                try fm.moveItem(atPath: src, toPath: dst)
                moves.append(DDMove(src: src, dst: dst))
                moved += 1
            } catch {
                log("[dedupe] 移动失败 \(src): \(error)")
            }
        }
        return (moved, moves)
    }

    static func undo(_ path: String) -> Int32 {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let manifest = try? JSONDecoder().decode(DDManifest.self, from: data) else {
            log("无法读取 manifest: \(path)")
            return 1
        }
        let fm = FileManager.default
        var restored = 0
        for record in manifest.moves.reversed() {
            guard fm.fileExists(atPath: record.dst) else { continue }
            guard !fm.fileExists(atPath: record.src) else {
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
        log("[undo] 已恢复 \(restored)/\(manifest.moves.count) 个文件；重复文件夹未自动删除")
        return 0
    }
}

extension FileDedupe {
    static func run(_ options: Options) -> Int32 {
        let start = Date()
        let fm = FileManager.default
        if !options.undoPath.isEmpty { return undo(options.undoPath) }
        guard fm.fileExists(atPath: options.inputDir) else {
            log("输入目录不存在: \(options.inputDir)")
            return 1
        }

        // 扫描文件（按扩展名过滤，跳过输出目录——仅当其与输入目录不同时）
        let outRoot = options.outRoot.isEmpty ? options.inputDir : options.outRoot
        let skipOut = outRoot != options.inputDir
        let extSet = Set(options.exts.map { $0.lowercased() })
        let enumerator = fm.enumerator(atPath: options.inputDir)
        var files: [String] = []
        while let sub = enumerator?.nextObject() as? String {
            let full = (options.inputDir as NSString).appendingPathComponent(sub)
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: full, isDirectory: &isDir), !isDir.boolValue else { continue }
            if skipOut && full.hasPrefix(outRoot + "/") { continue }
            let ext = (sub as NSString).pathExtension.lowercased()
            guard extSet.contains(ext) else { continue }
            files.append(full)
        }
        files.sort()
        if options.limit > 0, files.count > options.limit { files = Array(files.prefix(options.limit)) }
        log("[dedupe] 扫描到 \(files.count) 个文本文件")

        // 提取（指纹 + shingle 集合 + DF 过滤 + 签名，带缓存）
        let (docs, cache, dfFiltered) = extractAll(files, options: options)
        if !options.cachePath.isEmpty {
            if let data = try? JSONEncoder().encode(cache) {
                ensureParent(options.cachePath)
                try? data.write(to: URL(fileURLWithPath: options.cachePath), options: .atomic)
            }
        }
        let valid = docs.filter { !$0.path.isEmpty }

        // 第一道：完全相同（size + 双指纹）
        var byFP: [String: [Int]] = [:]
        for (i, d) in valid.enumerated() where !d.path.isEmpty {
            byFP["\(d.size)-\(d.fp1)-\(d.fp2)", default: []].append(i)
        }
        var exactRemoved: [(src: String, size: Int64)] = []
        var exactGroups: [[Int]] = []
        var exactRepPaths = Set<String>()
        for (_, members) in byFP where members.count > 1 {
            let sorted = members.sorted { valid[$0].path < valid[$1].path }
            exactRepPaths.insert(valid[sorted[0]].path)
            exactGroups.append(Array(sorted.dropFirst()))
            for m in sorted.dropFirst() { exactRemoved.append((valid[m].path, valid[m].size)) }
        }
        log("[dedupe] 第一道：完全相同 \(exactGroups.count) 组 / \(exactRemoved.count) 个文件")
        let exactSet = Set(exactRemoved.map { $0.src })

        // 第二道：近重复（签名已在提取阶段生成，这里直接召回比对）
        let remaining = valid.enumerated().filter { !exactSet.contains($1.path) }
        let sigs = remaining.map { $1.sig }
        let (pairs, skipped) = DDAlgo.candidates(sigs, maxBucketSize: options.maxBucketSize,
                                                 maxPairs: options.maxPairs)
        log("[dedupe] LSH 候选对 \(pairs.count)，跳过巨桶 \(skipped) 个")

        // 桶内粗筛：签名 agreement > 阈值判为重复。
        // 大小比保护：开头相同但文件大小差异超过 size-ratio 倍的（如合集 vs 单册）
        // 改标「部分重复」只报告不移动，防止内容量不同被误删。
        var dupPairs: [(Int, Int, Double)] = []
        var partialPairs: [(Int, Int, Double)] = []
        for p in pairs {
            let a = Int(p >> 32)
            let b = Int(p & 0xFFFF_FFFF)
            let sim = DDAlgo.agreement(sigs[a], sigs[b])
            guard sim > options.simMin else { continue }
            let idxA = remaining[a].offset
            let idxB = remaining[b].offset
            if options.sizeRatio > 0 {
                let sA = max(valid[idxA].size, 1)
                let sB = max(valid[idxB].size, 1)
                let ratio = Double(max(sA, sB)) / Double(min(sA, sB))
                if ratio >= options.sizeRatio {
                    partialPairs.append((idxA, idxB, sim))
                    continue
                }
            }
            dupPairs.append((idxA, idxB, sim))
        }
        log("[dedupe] 粗筛判定重复对 \(dupPairs.count) 个（sim > \(options.simMin)），"
            + "部分重复（大小比 ≥ \(options.sizeRatio)）\(partialPairs.count) 个对不动")

        // 贪心保留：每个被删文档与代表直接相似；代表按字节覆盖量选取
        let sizesForGreedy = valid.map { $0.size }
        let (keepersBase, removedFuzzy) = DDAlgo.greedyKeep(valid.count, dupPairs, sizes: sizesForGreedy)
        var keepers = keepersBase
        // 移动边界二次校验：
        //  1) 第一道保留的代表不得被第二道再删除（否则整组文件都会离开原位）；
        //  2) 实际移动对 (被删, 代表) 大小比 ≥ sizeRatio 时改标部分重复、二者均保留。
        var finalRemoved: [(Int, Int, Double)] = []
        for (dup, rep, sim) in removedFuzzy {
            if exactRepPaths.contains(valid[dup].path) {
                keepers.append(dup)
                continue
            }
            if options.sizeRatio > 0 {
                let sd = max(valid[rep].size, 1)
                let su = max(valid[dup].size, 1)
                if Double(max(sd, su)) / Double(min(sd, su)) >= options.sizeRatio {
                    partialPairs.append((dup, rep, sim))
                    continue
                }
            }
            finalRemoved.append((dup, rep, sim))
        }
        let removedFuzzyFinal = finalRemoved
        var fuzzyRemoved: [(src: String, size: Int64)] = []
        var fuzzyGroups: [Int: [Int]] = [:]
        for (dup, rep, _) in removedFuzzyFinal {
            fuzzyRemoved.append((valid[dup].path, valid[dup].size))
            fuzzyGroups[rep, default: []].append(dup)
        }
        log("[dedupe] 第二道：近重复 \(fuzzyGroups.count) 组 / \(fuzzyRemoved.count) 个文件"
            + "（保留代表 \(keepers.count) 个文档）")

        // 汇总
        let allRemoved = exactRemoved + fuzzyRemoved
        let bytesSaved = allRemoved.reduce(Int64(0)) { $0 + $1.size }
        let avgShingles = valid.isEmpty ? 0 : valid.reduce(0) { $0 + $1.keptShingles } / max(1, valid.count)
        var groups: [DDGroup] = []
        for (_, members) in byFP where members.count > 1 {
            let sorted = members.sorted { valid[$0].path < valid[$1].path }
            groups.append(DDGroup(representative: valid[sorted[0]].path,
                                  members: sorted.dropFirst().map { valid[$0].path },
                                  size: sorted.count, kind: "exact"))
        }
        for (rep, dups) in fuzzyGroups {
            groups.append(DDGroup(representative: valid[rep].path,
                                  members: dups.map { valid[$0].path },
                                  size: dups.count + 1, kind: "fuzzy"))
        }
        // 部分重复：大小比超限的疑似「合集 vs 单册」，只报告不移动
        var partialAdj: [Int: Set<Int>] = [:]
        for (a, b, _) in partialPairs {
            partialAdj[a, default: []].insert(b)
            partialAdj[b, default: []].insert(a)
        }
        var seenPartial = Set<Int>()
        var partialGroups: [[Int]] = []
        for a in partialAdj.keys.sorted() where !seenPartial.contains(a) {
            var comp: [Int] = []
            var stack = [a]
            while let x = stack.popLast() {
                guard !seenPartial.contains(x) else { continue }
                seenPartial.insert(x)
                comp.append(x)
                stack.append(contentsOf: partialAdj[x] ?? [])
            }
            partialGroups.append(comp)
        }
        var partialFiles = 0
        for comp in partialGroups {
            let sorted = comp.sorted { valid[$0].size > valid[$1].size }
            partialFiles += sorted.count - 1
            groups.append(DDGroup(representative: valid[sorted[0]].path,
                                  members: sorted.dropFirst().map { valid[$0].path },
                                  size: sorted.count, kind: "partial"))
        }
        groups.sort { $0.size > $1.size }

        let elapsed = Date().timeIntervalSince(start)
        let report = DDReport(inputDir: options.inputDir, outRoot: outRoot,
                              fileCount: files.count, exactDupGroups: exactGroups.count,
                              exactDupFiles: exactRemoved.count,
                              fuzzyDupGroups: fuzzyGroups.count,
                              fuzzyDupFiles: fuzzyRemoved.count,
                              partialGroups: partialGroups.count,
                              partialFiles: partialFiles,
                              totalRemoved: allRemoved.count, bytesSaved: bytesSaved,
                              avgShingles: avgShingles, dfFilteredShingles: dfFiltered,
                              candidatePairs: pairs.count,
                              sampleMode: options.sampleModeName,
                              sampleMaxChars: options.sampleModeName == "full" ? 0 : options.maxChars,
                              elapsedSeconds: elapsed,
                              applied: options.apply, groups: groups)
        if !options.reportPath.isEmpty, let data = try? JSONEncoder().encode(report) {
            ensureParent(options.reportPath)
            try? data.write(to: URL(fileURLWithPath: options.reportPath), options: .atomic)
        }

        // 预览
        for g in groups.prefix(30) {
            log("[组 \(g.size)] 代表: \((g.representative as NSString).lastPathComponent)")
            for m in g.members.prefix(4) { log("    重复: \((m as NSString).lastPathComponent)") }
        }
        if groups.count > 30 { log("…… 其余 \(groups.count - 30) 组见报告") }
        log("[dedupe] \(options.apply ? "APPLY" : "DRY-RUN") 文件 \(files.count) | "
            + "完全重复 \(exactRemoved.count) | 近重复 \(fuzzyRemoved.count) | "
            + "部分重复（不动）\(partialFiles) | "
            + "共移除 \(allRemoved.count) 个，节省 \(bytesSaved / 1024) KB | "
            + String(format: "%.1fs", elapsed))

        guard options.apply, !allRemoved.isEmpty else { return 0 }
        let (moved, moves) = moveDuplicates(allRemoved, options: Options(outRoot: outRoot, duplicatesName: options.duplicatesName))
        if !options.manifestPath.isEmpty {
            let manifest = DDManifest(version: 1,
                                      createdAt: ISO8601DateFormatter().string(from: Date()),
                                      outRoot: outRoot, moves: moves)
            try? writeJSON(manifest, path: options.manifestPath)
        }
        log("[dedupe] 已移动 \(moved)/\(moves.count) 个重复文件；manifest 已写")
        return 0
    }
}

// MARK: - 自检

extension FileDedupe {
    static func selfTest() -> Int32 {
        var failures: [String] = []
        func check(_ cond: Bool, _ name: String) {
            if cond { print("✓ \(name)") } else { print("FAIL: \(name)"); failures.append(name) }
        }

        // 清理：剔标点/空格/换行，英文转小写
        let c1 = DDText.clean("Hello, 世界！ \n\tni Hao。", maxChars: 0)
        check(String(c1) == "hello世界nihao", "清理剔标点空白并转小写")
        let c2 = DDText.clean("字字字", maxChars: 2)
        check(c2.count == 2, "maxChars 截断代表文本")

        // 代表文本：首尾各半 / 全文
        check(String(DDText.clean("ABCDEFGHIJ", maxChars: 4)) == "abij",
              "代表文本取首 2 + 尾 2 字符")
        check(String(DDText.clean("ABCDEFGHIJ", maxChars: 5)) == "abhij",
              "奇数上限首 2 + 尾 3 字符")
        check(String(DDText.clean("abc", maxChars: 10)) == "abc",
              "短于上限时返回全文且不重复")
        check(String(DDText.clean("ABCDEFGHIJ", maxChars: 4, fullText: true)) == "abcdefghij",
              "--full-text 忽略上限用全文")
        check(String(DDText.clean("ABCDEFGHIJ", maxChars: 0)) == "abcdefghij",
              "maxChars=0 等价全文")
        let segsHT = DDText.segments("ABCDEFGHIJ", maxChars: 4)
        check(segsHT.count == 2 && segsHT[0].count == 2 && segsHT[1].count == 2,
              "截取模式返回首/尾两段（无虚假跨段 shingle）")
        let segsFT = DDText.segments("ABCDEFGHIJ", maxChars: 4, fullText: true)
        check(segsFT.count == 1 && segsFT[0].count == 10, "全文模式返回单段")
        var optsHT = Options(); optsHT.maxChars = 10000
        var optsFT = Options(); optsFT.fullText = true
        check(optsHT.sampleModeName == "headtail" && optsFT.sampleModeName == "full"
              && cacheParams(optsHT).contains("mode=headtail")
              && cacheParams(optsFT).contains("mode=full"),
              "采样模式进入缓存参数，两种模式不误复用签名")

        // 3-shingle 滑窗
        let sh = DDText.shingles(Array("abcd"), k: 3)
        check(sh == Set(["abc", "bcd"]), "k=3 滑窗取子串")

        // 停用判定
        check(DDText.isStop("的了着", extra: []), "全虚字 shingle 被停用")
        check(DDText.isStop("实际上", extra: []), "词表命中被停用")
        check(DDText.isStop("the", extra: []), "英文功能 3-gram 被停用")
        check(!DDText.isStop("量子力", extra: []), "实词不被停用")

        // 文档级停用词（DF 过滤）
        let sample = [[1, 2], [1, 3], [1, 4], [5, 6]].map { $0.map { UInt64($0) } }
        let common = DDAlgo.commonIDs(fromSample: sample, cut: 0.5)
        check(common == Set([UInt64(1)]),
              "DF ≥ 0.5 的 shingle 被识别为文档级停用词")

        // 双指纹：相同内容相同、不同内容不同
        let b1 = Array("同一份内容".utf8)
        let b2 = Array("同一份内容".utf8)
        let b3 = Array("不同内容".utf8)
        check(ddfnv1aBytes(b1, seed: 1) == ddfnv1aBytes(b2, seed: 1), "相同字节指纹一致")
        check(ddfnv1aBytes(b1, seed: 1) != ddfnv1aBytes(b3, seed: 1), "不同字节指纹不同")

        // 签名与 agreement（大集合避免 MinHash 小样本噪声）
        let fam = DDFamily()
        let xsA: [UInt64] = Array(UInt64(0)..<UInt64(200))
        let xsB: [UInt64] = Array(UInt64(0)..<UInt64(220))   // Jaccard = 200/220 ≈ 0.91
        let xsC: [UInt64] = Array(UInt64(500)..<UInt64(700)) // 不相交
        let sigA = DDMinHash.signature(fromHashes: xsA, family: fam)
        let sigB = DDMinHash.signature(fromHashes: xsB, family: fam)
        let sigC = DDMinHash.signature(fromHashes: xsC, family: fam)
        check(DDAlgo.agreement(sigA, sigA) == 1.0, "同集合签名完全一致")
        check(DDAlgo.agreement(sigA, sigB) > 0.8, "相似集合签名 agreement 高")
        check(DDAlgo.agreement(sigA, sigC) < 0.4, "不相似集合签名 agreement 低")

        // LSH：8×16 布局，完全相同签名必碰撞
        let (pairsSame, _) = DDAlgo.candidates([sigA, sigA], maxBucketSize: 100, maxPairs: 1000)
        check(pairsSame.count == 1, "相同签名必进同一桶")

        // 贪心保留：链式 A-B-C → 代表 B，删 A、C（各与 B 直接相似）
        let chain = DDAlgo.greedyKeep(3, [(0, 1, 0.9), (1, 2, 0.9)], sizes: [10, 100, 10])
        check(chain.keepers == [1] && chain.removed.map { $0.dup } == [0, 2]
                && chain.removed.allSatisfy { $0.rep == 1 && $0.sim > 0.85 },
              "链式重复由中位代表担保，每个被删文档与代表直接相似")
        // 无边节点保留
        let lonely = DDAlgo.greedyKeep(3, [])
        check(lonely.keepers == [0, 1, 2] && lonely.removed.isEmpty, "无重复文档全部保留")
        // 两个独立分量各留代表
        let two = DDAlgo.greedyKeep(4, [(0, 1, 0.9), (2, 3, 0.9)])
        check(two.keepers == [0, 2] && two.removed.map { $0.dup }.sorted() == [1, 3],
              "独立分量各自保留代表")

        // 端到端：完全相同 2 个 + 近似 2 个 + 独特 1 个；dry-run → apply → undo
        do {
            let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("dedupe-selftest-\(UUID().uuidString)")
            let root = tmp.appendingPathComponent("books")
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: tmp) }
            func repeated(_ body: String, times: Int) -> String {
                Array(repeating: body, count: times).joined()
            }
            let base = repeated("这是一个关于量子力学与宇宙奥秘的科普故事，讲述科学家在实验",
                                times: 12)
                + "室里日夜探索，最终发现引力波信号的完整历程。"
            try base.write(to: root.appendingPathComponent("a1.md"), atomically: true, encoding: .utf8)
            try base.write(to: root.appendingPathComponent("a2.md"), atomically: true, encoding: .utf8)
            let near = repeated("这是一个关于量子力学与宇宙奥秘的科普故事，讲述学者在实验",
                                times: 12)
                + "室里日夜探索，最终发现引力波信号的完整历程。"
            try near.write(to: root.appendingPathComponent("b1.md"), atomically: true, encoding: .utf8)
            // b2 与 b1 近似但不同（尾部短追加），走第二道近重复路径
            try (near + "附录：略。").write(
                to: root.appendingPathComponent("b2.md"), atomically: true, encoding: .utf8)
            try "完全无关的独特内容，讲摄影构图心得。".write(
                to: root.appendingPathComponent("c.md"), atomically: true, encoding: .utf8)

            var opts = Options()
            opts.inputDir = root.path
            opts.outRoot = tmp.appendingPathComponent("out").path
            opts.cachePath = tmp.appendingPathComponent("cache.json").path
            opts.reportPath = tmp.appendingPathComponent("report.json").path
            opts.manifestPath = tmp.appendingPathComponent("manifest.json").path
            opts.dfCut = 0  // 小样例 cutoff=2 会把共享 shingle 全判为文档级停用词，关闭
            let rc = FileDedupe.run(opts)
            check(rc == 0, "去重 dry-run 执行成功")
            check(FileManager.default.fileExists(atPath: opts.reportPath), "dry-run 生成报告")
            check(FileManager.default.fileExists(atPath: root.appendingPathComponent("a2.md").path),
                  "dry-run 不移动源文件")
            if let data = try? Data(contentsOf: URL(fileURLWithPath: opts.reportPath)),
               let rep = try? JSONDecoder().decode(DDReport.self, from: data) {
                check(rep.fileCount == 5, "报告文件数=5")
                check(rep.exactDupGroups == 1 && rep.exactDupFiles == 1, "完全重复 1 组 1 个")
                check(rep.fuzzyDupGroups == 1 && rep.fuzzyDupFiles == 1, "近重复 1 组 1 个")
                check(rep.totalRemoved == 2, "共移除 2 个")
                check(rep.applied == false, "报告标记 dry-run")
            } else { failures.append("去重报告无法解码") }

            // apply：移动 a2/b2 到重复文件夹
            opts.apply = true
            check(FileDedupe.run(opts) == 0, "apply 执行成功")
            check(!FileManager.default.fileExists(atPath: root.appendingPathComponent("a2.md").path),
                  "apply 后 a2 已离开原位")
            let dupDir = opts.outRoot + "/" + opts.duplicatesName
            let movedCount = (try? FileManager.default.contentsOfDirectory(atPath: dupDir))?.count ?? 0
            check(movedCount == 2, "重复文件夹含 2 个文件")

            // undo：恢复
            var undoOpts = Options()
            undoOpts.undoPath = opts.manifestPath
            check(FileDedupe.run(undoOpts) == 0, "undo 执行成功")
            check(FileManager.default.fileExists(atPath: root.appendingPathComponent("a2.md").path)
                    && FileManager.default.fileExists(atPath: root.appendingPathComponent("b2.md").path),
                  "undo 把重复文件移回原位")
        } catch { failures.append("端到端异常: \(error)") }

        if failures.isEmpty {
            print("DedupeFiles self-test: all checks passed")
            return 0
        }
        print("DedupeFiles self-test: \(failures.count) failures")
        return 1
    }
}

// MARK: - CLI 入口

private func ddUsage() {
    let text = """
    DedupeFiles — 文本文件去重（完全相同 + 近重复）

    用法: bookdedupe <目录> [选项]

    流程:
      第一道  文件双 FNV 指纹 + 大小 → 完全相同的文件组内保留路径排序第一个
      第二道  代表文本(默认首+尾各半，共 --max-chars 字符；--full-text 用全文)
              → 清理(汉字+小写字母) → k-shingle(默认3) →
              停用词 → 文档级停用词(DF≥--df-cut) → MinHash 128 →
              LSH 8×16 召回 → 桶内签名 agreement > --sim 判重复 →
              分量内贪心保留代表(每个被删文档与代表直接相似)

    选项:
      --apply                 执行移动（默认 dry-run 只报告）
      --undo PATH             按 manifest 撤销上次移动
      --limit N               只处理前 N 个文件
      --max-chars N           代表文本长度：首部+尾部各取一半，合计 N 字符 (默认 20000)
                              N=0 等价全文
      --full-text             不截取，直接用全文计算签名（忽略 --max-chars；更慢更占内存）
      --shingle-k N           shingle 窗口长度 (默认 3)
      --sim X                 签名粗筛阈值，相同位数/128 (默认 0.85)
      --size-ratio X          判重的大小比上限：开头相同但大小差异超过 X 倍的文件
                              改标「部分重复」只报告不移动，防合集/单册误删 (默认 3，0=关闭)
      --df-cut X              文档级停用词阈值，0=关闭 (默认 0.5)
      --no-stopwords          关闭停用词过滤
      --stopwords PATH        追加自定义停用词文件
      --ext LIST              扩展名列表，逗号分隔 (默认 md,txt)
      --max-bucket-size N     LSH 单桶上限，超过跳过 (默认 2000)
      --jobs N                并发数 (默认 = CPU 核数)
      --cache PATH            提取缓存路径
      --report PATH           报告输出路径 (JSON)
      --out-root PATH         重复文件夹创建位置 (默认 = 输入目录)
      --duplicates-name NAME  重复文件夹名 (默认 _重复)
      --manifest PATH         移动清单路径（--apply 时写，供 --undo）
      --self-test             运行自检
      --help                  本帮助
    """
    print(text)
}

@main enum DedupeFilesMain {
    static func main() {
        let argv = CommandLine.arguments
        if argv.contains("--self-test") { exit(FileDedupe.selfTest()) }
        if argv.contains("--help") || argv.count < 2 { ddUsage(); exit(0) }
        var opts = FileDedupe.Options()
        var i = 1
        func value() -> String? {
            guard i + 1 < argv.count else { return nil }
            i += 1
            return argv[i]
        }
        while i < argv.count {
            let arg = argv[i]
            switch arg {
            case "--apply": opts.apply = true
            case "--undo": opts.undoPath = value() ?? ""
            case "--limit": opts.limit = Int(value() ?? "") ?? 0
            case "--max-chars": opts.maxChars = Int(value() ?? "") ?? 20000
            case "--full-text": opts.fullText = true
            case "--shingle-k": opts.shingleK = max(2, Int(value() ?? "") ?? 3)
            case "--sim": opts.simMin = Double(value() ?? "") ?? 0.85
        case "--size-ratio": opts.sizeRatio = Double(value() ?? "") ?? 3.0
            case "--df-cut": opts.dfCut = Double(value() ?? "") ?? 0.5
            case "--no-stopwords": opts.noStopwords = true
            case "--stopwords": opts.stopwordsPath = value() ?? ""
            case "--ext": opts.exts = (value() ?? "md,txt").split(separator: ",").map(String.init)
            case "--max-bucket-size": opts.maxBucketSize = max(2, Int(value() ?? "") ?? 2000)
            case "--jobs": opts.jobs = max(1, Int(value() ?? "") ?? 0)
            case "--cache": opts.cachePath = value() ?? ""
            case "--report": opts.reportPath = value() ?? ""
            case "--out-root": opts.outRoot = value() ?? ""
            case "--duplicates-name": opts.duplicatesName = value() ?? "_重复"
            case "--manifest": opts.manifestPath = value() ?? ""
            default: break
            }
            i += 1
        }
        opts.inputDir = argv[1].hasPrefix("--") ? opts.inputDir : argv[1]
        guard !opts.inputDir.isEmpty else { ddUsage(); exit(1) }
        exit(FileDedupe.run(opts))
    }
}


