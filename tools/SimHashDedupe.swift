// SimHashDedupe.swift — 第六个工具：SimHash(64) + 4 表 LSH 文本去重
//
// 指纹生成：
//   1) 代表文本：默认取文件头部、尾部各 --sample-chars 字（默认 20000，共 40000）；
//      --full-text 用全文。清理剔标点/空格/换行，汉字保留、英文转小写。
//   2) 双窗口扫描：k=3 统计 TF 并去重，过滤停用词 + 文档级高频 shingle(DF)，
//      取频次最高的前 300 个；k=2 同样扫一遍取前 200 个；两者合并为特征词集合，
//      每个特征权重 = 文档内频次。
//   3) 每个特征词经 MurmurHash3 x64_128(取 h1) 得 64 位特征哈希；
//      64 维数组逐位加权累加（1 → +权重，0 → −权重），>0 记 1、≤0 记 0
//      → 64 位 SimHash 指纹。
//
// LSH 索引与查询：
//   指纹切 4 段各 16 位（表1=1-16位 … 表4=49-64位），4 张倒排表存文档 ID。
//   海明距离 ≤3 时按鸽巢原理必有一段相同 → 必定召回。
//   新文章：算指纹 → 4 表召回候选并集(粗排) → XOR+popcount 精排 →
//   ≤ --ham 判重复不入库；> --ham 判新文章，指纹与 4 段键值写入索引。
//
// 依赖 CoreWordKit.swift 停用词表。编译：
//   swiftc -O -o /tmp/booksimhash tools/CoreWordKit.swift tools/SimHashDedupe.swift

import Foundation

// MARK: - MurmurHash3 x64_128（取 128 位输出的低 64 位 h1）

private enum Murmur {
    @inline(__always) static func rotl(_ x: UInt64, _ r: Int) -> UInt64 {
        (x << r) | (x >> (64 - r))
    }
    @inline(__always) static func fmix64(_ k: UInt64) -> UInt64 {
        var k = k
        k ^= k >> 33
        k = k &* 0xff51afd7ed558ccd
        k ^= k >> 33
        k = k &* 0xc4ceb9fe1a85ec53
        k ^= k >> 33
        return k
    }

    static func hash64(_ data: [UInt8], seed: UInt64 = 0) -> UInt64 {
        let c1: UInt64 = 0x87c37b91114253d5
        let c2: UInt64 = 0x4cf5ad432745937f
        var h1 = seed
        var h2 = seed
        let blocks = data.count / 16
        for b in 0..<blocks {
            var k1: UInt64 = 0
            var k2: UInt64 = 0
            for j in 0..<8 { k1 |= UInt64(data[b * 16 + j]) << (8 * j) }
            for j in 0..<8 { k2 |= UInt64(data[b * 16 + 8 + j]) << (8 * j) }
            k1 = k1 &* c1; k1 = rotl(k1, 31); k1 = k1 &* c2; h1 ^= k1
            h1 = rotl(h1, 27); h1 = h1 &+ h2; h1 = h1 &* 5 &+ 0x52dce729
            k2 = k2 &* c2; k2 = rotl(k2, 33); k2 = k2 &* c1; h2 ^= k2
            h2 = rotl(h2, 31); h2 = h2 &+ h1; h2 = h2 &* 5 &+ 0x38495ab5
        }
        let off = blocks * 16
        let tail = data.count - off
        if tail > 0 {
            var k1: UInt64 = 0
            var k2: UInt64 = 0
            if tail >= 9 {
                for j in 8..<tail { k2 ^= UInt64(data[off + j]) << (8 * (j - 8)) }
                k2 = k2 &* c2; k2 = rotl(k2, 33); k2 = k2 &* c1; h2 ^= k2
            }
            if tail >= 1 {
                for j in 0..<min(tail, 8) { k1 ^= UInt64(data[off + j]) << (8 * j) }
                k1 = k1 &* c1; k1 = rotl(k1, 31); k1 = k1 &* c2; h1 ^= k1
            }
        }
        h1 ^= UInt64(data.count)
        h2 ^= UInt64(data.count)
        h1 = h1 &+ h2
        h2 = h2 &+ h1
        h1 = fmix64(h1)
        h2 = fmix64(h2)
        h1 = h1 &+ h2
        h2 = h2 &+ h1
        return h1
    }

    static func hash64(_ s: String, seed: UInt64 = 0) -> UInt64 {
        hash64(Array(s.utf8), seed: seed)
    }
}

// MARK: - 文本清理与代表文本

private enum SHText {
    /// 汉字保留、A-Z 转小写、a-z 保留，其余剔除。
    static func kept(_ s: Unicode.Scalar) -> Character? {
        let v = s.value
        if (v >= 0x4E00 && v <= 0x9FFF) || (v >= 0x3400 && v <= 0x4DBF)
            || (v >= 0x20000 && v <= 0x2A6DF) || (v >= 0xF900 && v <= 0xFAFF) {
            return Character(s)
        } else if v >= 0x41 && v <= 0x5A {
            return UnicodeScalar(v + 32).map { Character($0) }
        } else if v >= 0x61 && v <= 0x7A {
            return Character(s)
        }
        return nil
    }

    static func cleanAll(_ text: String) -> [Character] {
        var out: [Character] = []
        for s in text.unicodeScalars {
            if let c = kept(s) { out.append(c) }
        }
        return out
    }

    /// 代表文本分段：全文/短文本 → 1 段；否则头 sampleChars + 尾 sampleChars 两段。
    /// 分段返回，避免首尾拼接处产生虚假跨段 k-gram；内存只占 2×sampleChars。
    static func segments(_ text: String, sampleChars: Int, fullText: Bool) -> [[Character]] {
        if fullText || sampleChars <= 0 { return [cleanAll(text)] }
        var total = 0
        for s in text.unicodeScalars where kept(s) != nil { total += 1 }
        if total <= 2 * sampleChars { return [cleanAll(text)] }
        var head: [Character] = []
        var tail: [Character] = []
        head.reserveCapacity(sampleChars)
        tail.reserveCapacity(sampleChars)
        var idx = 0
        for s in text.unicodeScalars {
            guard let c = kept(s) else { continue }
            if idx < sampleChars { head.append(c) }
            if idx >= total - sampleChars { tail.append(c) }
            idx += 1
        }
        return [head, tail]
    }
}

// MARK: - 特征工程（双窗口 + 停用词 + DF 过滤 + topN）

private enum SHFeatures {
    static let weakChars: Set<Character> = WordFilters.weakBoundaryChars

    /// 扫一遍取 k-gram 的 TF 字典（Set 去重体现在字典 key）。
    static func grams(_ segs: [[Character]], k: Int, stops: Set<String>,
                      dfStops: Set<String>, useStopwords: Bool) -> [String: Int] {
        var counts: [String: Int] = [:]
        for seg in segs {
            guard seg.count >= k else { continue }
            for i in 0...(seg.count - k) {
                let g = String(seg[i..<(i + k)])
                if useStopwords {
                    if stops.contains(g) { continue }
                    if g.allSatisfy({ weakChars.contains($0) }) { continue }
                }
                if !dfStops.isEmpty && dfStops.contains("\(k):\(g)") { continue }
                counts[g, default: 0] += 1
            }
        }
        return counts
    }

    /// 按频次降序取 topN（同频次按字典序，保证确定性），返回 (特征词, 权重=TF)。
    static func top(_ counts: [String: Int], _ n: Int) -> [(String, Int)] {
        guard n > 0 else { return [] }
        return counts.sorted {
            if $0.value != $1.value { return $0.value > $1.value }
            return $0.key < $1.key
        }.prefix(n).map { ($0.key, $0.value) }
    }

    /// 核心词特征：ExtractCoreWords 的 N-gram + NPMI + 左右熵 + 停用词/边界/子串过滤，
    /// 取 top N 个词，权重 = 文档内频次。与 shingle 模式共用同一份代表文本。
    static func coreWords(_ text: String, opts: SimHashDedupe.Options) -> [(String, Int)] {
        let segs = SHText.segments(text, sampleChars: opts.sampleChars, fullText: opts.fullText)
        let joined = segs.map { String($0) }.joined(separator: "\n")
        var eo = ExtractOptions()
        eo.top = opts.coreWordsTop
        eo.useStopwords = opts.useStopwords
        eo.stopwordsPath = opts.stopwordsPath
        guard let r = try? CoreWordExtractor.extract(text: joined, options: eo) else { return [] }
        return r.words.map { ($0.word, $0.freq) }
    }

    /// 最终特征集合 = k=3 top3 ∪ k=2 top2（shingles）；core-words 模式改用核心词提取。
    static func extract(_ text: String, opts: SimHashDedupe.Options,
                        stops: Set<String>, dfStops: Set<String>) -> [(String, Int)] {
        if opts.featureMode == "core-words" { return coreWords(text, opts: opts) }
        let segs = SHText.segments(text, sampleChars: opts.sampleChars, fullText: opts.fullText)
        let c3 = grams(segs, k: 3, stops: stops, dfStops: dfStops,
                       useStopwords: opts.useStopwords)
        let c2 = grams(segs, k: 2, stops: stops, dfStops: dfStops,
                       useStopwords: opts.useStopwords)
        return top(c3, opts.top3) + top(c2, opts.top2)
    }

    /// 抽样统计文档级停用词（DF ≥ cut 的 shingle），k=2/3 分别带前缀区分。
    static func dfStops(_ files: [String], opts: SimHashDedupe.Options,
                        stops: Set<String>) -> Set<String> {
        guard opts.dfCut > 0, !files.isEmpty else { return [] }
        // 核心词模式跳过 shingle DF：NPMI/熵与停用词已承担筛选，且词级 DF 语义不同
        if opts.featureMode == "core-words" { return [] }
        let stride = max(1, files.count / max(1, opts.dfSample))
        var sample: [String] = []
        var i = 0
        while i < files.count { sample.append(files[i]); i += stride }
        var df: [String: Int] = [:]
        for path in sample {
            guard let text = SimHashDedupe.readText(path) else { continue }
            let segs = SHText.segments(text, sampleChars: opts.sampleChars,
                                       fullText: opts.fullText)
            for k in [3, 2] {
                let g = grams(segs, k: k, stops: stops, dfStops: [],
                              useStopwords: opts.useStopwords)
                for key in g.keys { df["\(k):\(key)", default: 0] += 1 }
            }
        }
        let cutoff = max(2, Int((Double(sample.count) * opts.dfCut * 0.8).rounded(.up)))
        return Set(df.filter { $0.value >= cutoff }.map { $0.key })
    }
}

// MARK: - SimHash 指纹与 LSH 索引

private enum SHFinger {
    static func make(_ features: [(String, Int)], seed: UInt64) -> UInt64 {
        var v = [Int64](repeating: 0, count: 64)
        for (word, weight) in features {
            let h = Murmur.hash64(word, seed: seed)
            let w = Int64(weight)
            for i in 0..<64 {
                v[i] += ((h >> i) & 1 == 1) ? w : -w
            }
        }
        var fp: UInt64 = 0
        for i in 0..<64 where v[i] > 0 { fp |= (UInt64(1) << i) }
        return fp
    }

    /// 海明距离 = XOR 后 1 的个数。
    static func hamming(_ a: UInt64, _ b: UInt64) -> Int { (a ^ b).nonzeroBitCount }

    /// 第 b 段（0..3）的 16 位值：b=0 取 1-16 位（最低 16 位）。
    static func band(_ fp: UInt64, _ b: Int) -> UInt16 {
        UInt16(truncatingIfNeeded: fp >> (16 * b))
    }

    /// 四段是否至少一段相同（即 LSH 必然碰撞）。
    static func collides(_ a: UInt64, _ b: UInt64) -> Bool {
        for i in 0..<4 where band(a, i) == band(b, i) { return true }
        return false
    }
}

private struct SHDoc: Codable {
    let path: String
    let fp: String          // 十六进制
    let size: Int64
}

private struct SHIndexFile: Codable {
    let version: Int
    let params: String
    var docs: [SHDoc]
    var tables: [[String: [Int]]]
    var dfStops: [String]
}

private struct SHIndex {
    var docs: [SHDoc] = []
    var tables: [[String: [Int]]] = Array(repeating: [:], count: 4)
    var dfStops: Set<String> = []

    func fp(of id: Int) -> UInt64 { UInt64(docs[id].fp, radix: 16) ?? 0 }

    mutating func insert(fp: UInt64, path: String, size: Int64) -> Int {
        let id = docs.count
        docs.append(SHDoc(path: path, fp: String(fp, radix: 16), size: size))
        for b in 0..<4 {
            let key = String(SHFinger.band(fp, b))
            tables[b][key, default: []].append(id)
        }
        return id
    }

    /// 粗排：4 张表按各段键值召回，取文档 ID 并集。
    func candidates(_ fp: UInt64) -> Set<Int> {
        var out = Set<Int>()
        for b in 0..<4 {
            if let ids = tables[b][String(SHFinger.band(fp, b))] { out.formUnion(ids) }
        }
        return out
    }

    /// 精排：候选中海明距离 ≤ ham 且最小者。
    func nearest(_ fp: UInt64, ham: Int) -> (id: Int, dist: Int)? {
        var best: (id: Int, dist: Int)?
        for id in candidates(fp) {
            let d = SHFinger.hamming(fp, self.fp(of: id))
            if d <= ham {
                if let cur = best { if d < cur.dist { best = (id, d) } }
                else { best = (id, d) }
            }
        }
        return best
    }

    func toFile(_ params: String) -> SHIndexFile {
        SHIndexFile(version: 1, params: params, docs: docs,
                    tables: tables, dfStops: dfStops.sorted())
    }

    static func from(_ f: SHIndexFile) -> SHIndex {
        SHIndex(docs: f.docs, tables: f.tables, dfStops: Set(f.dfStops))
    }
}

// MARK: - 报告与移动清单

private struct SHDup: Codable {
    let path: String
    let hamming: Int
}

private struct SHGroup: Codable {
    let representative: String
    let duplicates: [SHDup]
}

private struct SHReport: Codable {
    let params: String
    let fileCount: Int
    let uniqueCount: Int
    let duplicateCount: Int
    let candidateHits: Int
    let exactChecks: Int
    let verifyRejected: Int
    let avgFeatures: Int
    let dfStops: Int
    let groups: [SHGroup]
    let applied: Bool
    let elapsedSeconds: Double
}

private struct SHMove: Codable { let src: String; let dst: String }
private struct SHManifest: Codable { let version: Int; let moves: [SHMove] }

// MARK: - 主流程

enum SimHashDedupe {
    struct Options {
        var sampleChars = 20000       // 头/尾各取多少字（共 2×）
        var fullText = false
        var featureMode = "shingles"     // "shingles" | "core-words"
        var coreWordsTop = 500
        var top3 = 300
        var top2 = 200
        var dfCut = 0.5
        var dfSample = 1000
        var ham = 3
        var verify = true            // 距离 1..ham 时做内容复核，剔除指纹碰撞误判
        var verifyJaccard = 0.5     // 复核文本 3-gram Jaccard 下限
        var verifySampleChars = 20000  // 复核独立于指纹：头尾各取这么多字（长文本更抗模板陷阱）
        var hashSeed: UInt64 = 0
        var useStopwords = true
        var stopwordsPath = ""
        var indexPath = ""
        var queryPath = ""
        var noInsert = false
        var apply = false
        var undoPath = ""
        var outRoot = ""
        var duplicatesName = "_重复"
        var reportPath = ""
        var exts = Set(["md", "txt"])
        var limit = 0
        var paths: [String] = []
    }

    static func log(_ s: String) {
        FileHandle.standardError.write((s + "\n").data(using: .utf8)!)
    }

    /// 4 段 × 16 位的「必碰撞」保证只在 ham ≤ 3 时成立（鸽巢原理）。
    static func warnHam(_ o: Options) {
        if o.ham > 3 {
            log("[simhash] 警告：--ham > 3 时 4 段 LSH 不再保证『距离 ≤ ham 必定碰撞』，"
                + "可能漏召回；如需更大阈值请增加 band 数或改用多路分段")
        }
    }

    static func fileSize(_ path: String) -> Int64 {
        (try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int64) ?? 0
    }

    /// 内容复核：头尾代表文本的 3-gram Jaccard。SimHash 距离 1..ham 时调用，
    /// 把「指纹碰巧接近但内容不同」的候选排除，避免误移不同的书。
    static func verifyContent(_ a: String, _ b: String, opts: Options) -> Bool {
        guard opts.verify else { return true }
        guard let ta = readText(a), let tb = readText(b) else { return false }
        func grams(_ text: String) -> Set<String> {
            var s = Set<String>()
            for seg in SHText.segments(text, sampleChars: opts.verifySampleChars,
                                       fullText: opts.verifySampleChars <= 0) {
                guard seg.count >= 3 else { continue }
                for i in 0...(seg.count - 3) { s.insert(String(seg[i..<(i + 3)])) }
            }
            return s
        }
        let A = grams(ta), B = grams(tb)
        let inter = A.intersection(B).count
        let union = A.count + B.count - inter
        guard union > 0 else { return false }
        return Double(inter) / Double(union) >= opts.verifyJaccard
    }

    static func readText(_ path: String) -> String? {
        guard let data = FileManager.default.contents(atPath: path),
              let text = String(data: data, encoding: .utf8) else { return nil }
        return text
    }

    static func paramsString(_ o: Options) -> String {
        "sample=\(o.fullText ? "full" : "\(o.sampleChars)")|feat=\(o.featureMode)"
            + "|cwtop=\(o.coreWordsTop)|top3=\(o.top3)|top2=\(o.top2)"
            + "|df=\(o.dfCut)|sw=\(!o.useStopwords)|swpath=\(o.stopwordsPath)"
            + "|seed=\(o.hashSeed)|swv=\(WordFilters.stopwordsVersion)"
    }

    static func stopSet(_ o: Options) -> Set<String> {
        guard o.useStopwords else { return [] }
        var s = WordFilters.builtinStopwords
        s.formUnion(WordFilters.loadExtra(o.stopwordsPath))
        return s
    }

    static func collect(_ inputs: [String], exts: Set<String>,
                        skipDirs: Set<String>) -> [String] {
        let fm = FileManager.default
        var out: [String] = []
        for p in inputs {
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: p, isDirectory: &isDir) else {
                log("[simhash] 路径不存在，跳过: \(p)"); continue
            }
            if isDir.boolValue {
                guard let en = fm.enumerator(atPath: p) else { continue }
                for case let sub as String in en {
                    let parts = sub.split(separator: "/").map(String.init)
                    if parts.contains(where: { $0.hasPrefix(".") }) { continue }
                    if parts.dropLast().contains(where: { skipDirs.contains($0) }) { continue }
                    guard exts.contains((sub as NSString).pathExtension.lowercased()) else { continue }
                    out.append((p as NSString).appendingPathComponent(sub))
                }
            } else { out.append(p) }
        }
        return out.sorted()
    }

    static func fingerprint(_ path: String, opts: Options,
                            stops: Set<String>, dfStops: Set<String>) -> (fp: UInt64, features: Int)? {
        guard let text = readText(path) else {
            log("[simhash] 非 UTF-8 或读取失败，跳过: \(path)"); return nil
        }
        let feats = SHFeatures.extract(text, opts: opts, stops: stops, dfStops: dfStops)
        return (SHFinger.make(feats, seed: opts.hashSeed), feats.count)
    }

    fileprivate static func loadIndex(_ opts: Options) -> SHIndex {
        guard !opts.indexPath.isEmpty,
              let data = try? Data(contentsOf: URL(fileURLWithPath: opts.indexPath)),
              let f = try? JSONDecoder().decode(SHIndexFile.self, from: data),
              f.version == 1, f.params == paramsString(opts) else { return SHIndex() }
        let idx = SHIndex.from(f)
        log("[simhash] 已加载索引：\(idx.docs.count) 篇文档，DF 停用词 \(idx.dfStops.count) 个")
        return idx
    }

    fileprivate static func saveIndex(_ idx: SHIndex, _ opts: Options) {
        guard !opts.indexPath.isEmpty else { return }
        do {
            let data = try JSONEncoder().encode(idx.toFile(paramsString(opts)))
            let url = URL(fileURLWithPath: opts.indexPath)
            try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                     withIntermediateDirectories: true)
            try data.write(to: url, options: .atomic)
        } catch { log("[simhash] 索引写入失败: \(error)") }
    }

    /// 在线查询：算指纹 → 粗排 → 精排 → 决策（重复不入库 / 新文章写 4 段键值）。
    static func query(_ opts: Options) -> Int32 {
        warnHam(opts)
        var idx = loadIndex(opts)
        let stops = stopSet(opts)
        guard let fres = fingerprint(opts.queryPath, opts: opts, stops: stops,
                                     dfStops: idx.dfStops) else { return 1 }
        let fp = fres.fp
        let cand = idx.candidates(fp)
        var best: (id: Int, dist: Int)?
        for id in cand {
            let d = SHFinger.hamming(fp, idx.fp(of: id))
            guard d <= opts.ham else { continue }
            if opts.verify && !verifyContent(opts.queryPath, idx.docs[id].path, opts: opts) {
                log("[simhash] 内容复核拦截指纹碰撞（距离 \(d)）：\(idx.docs[id].path)")
                continue
            }
            if let cur = best { if d < cur.dist { best = (id, d) } }
            else { best = (id, d) }
        }
        print("指纹 0x\(String(fp, radix: 16))  候选 \(cand.count) 篇")
        if let b = best {
            print("判定：重复/相似（海明距离 \(b.dist) ≤ \(opts.ham)）→ 代表 \(idx.docs[b.id].path)")
            return 0
        }
        print("判定：新文章（无候选距离 ≤ \(opts.ham)）")
        if !opts.noInsert {
            let id = idx.insert(fp: fp, path: opts.queryPath, size: fileSize(opts.queryPath))
            saveIndex(idx, opts)
            log("[simhash] 已写入索引（文档 ID \(id)，4 段键值已登记）")
        }
        return 0
    }

    /// 批量：逐个文件算指纹并走同一套「粗排 → 精排 → 去重决策」流程。
    static func run(_ opts: Options) -> Int32 {
        if !opts.undoPath.isEmpty { return undo(opts.undoPath) }
        if !opts.queryPath.isEmpty { return query(opts) }
        let start = Date()
        var files = collect(opts.paths, exts: opts.exts,
                            skipDirs: Set([opts.duplicatesName, "_重复"]))
        if opts.limit > 0, files.count > opts.limit { files = Array(files.prefix(opts.limit)) }
        guard !files.isEmpty else { log("[simhash] 没有可处理的文件"); return 1 }
        warnHam(opts)
        let params = paramsString(opts)
        var idx = loadIndex(opts)
        let stops = stopSet(opts)

        // 文档级停用词：索引里没有就抽样统计
        if opts.dfCut > 0 && idx.dfStops.isEmpty {
            idx.dfStops = SHFeatures.dfStops(files, opts: opts, stops: stops)
            log("[simhash] 文档级停用词：DF ≥ \(opts.dfCut) 的 shingle \(idx.dfStops.count) 个已剔除"
                + "（抽样统计）")
        }
        let featDesc = opts.featureMode == "core-words"
            ? "核心词 top\(opts.coreWordsTop)" : "shingle top3=\(opts.top3)+top2=\(opts.top2)"
        log("[simhash] 处理 \(files.count) 个文件 | 代表文本 "
            + (opts.fullText ? "全文" : "头 \(opts.sampleChars) + 尾 \(opts.sampleChars)")
            + " | 特征 \(featDesc) | 海明 ≤ \(opts.ham)")

        var groups: [SHGroup] = []
        var dupPaths: [(src: String, size: Int64)] = []
        var candidateHits = 0
        var exactChecks = 0
        var verifyRejected = 0
        var totalFeatures = 0
        var unique = 0
        for (n, path) in files.enumerated() {
            guard let fres = fingerprint(path, opts: opts, stops: stops,
                                         dfStops: idx.dfStops) else { continue }
            let fp = fres.fp
            totalFeatures += fres.features
            let cand = idx.candidates(fp)
            candidateHits += cand.count
            exactChecks += cand.count
            var best: (id: Int, dist: Int)?
            for id in cand {
                let d = SHFinger.hamming(fp, idx.fp(of: id))
                guard d <= opts.ham else { continue }
                if opts.verify && !verifyContent(path, idx.docs[id].path, opts: opts) {
                    verifyRejected += 1
                    continue
                }
                if let cur = best { if d < cur.dist { best = (id, d) } }
                else { best = (id, d) }
            }
            if let b = best {
                dupPaths.append((path, fileSize(path)))
                groups.append(SHGroup(representative: idx.docs[b.id].path,
                                      duplicates: [SHDup(path: path, hamming: b.dist)]))
            } else {
                _ = idx.insert(fp: fp, path: path, size: fileSize(path))
                unique += 1
            }
            if (n + 1) % 1000 == 0 { log("[simhash] 进度 \(n + 1)/\(files.count)") }
        }
        // 同一代表的多篇重复合并成组
        var merged: [String: [SHDup]] = [:]
        for g in groups { merged[g.representative, default: []].append(contentsOf: g.duplicates) }
        let finalGroups = merged.map { SHGroup(representative: $0.key, duplicates: $0.value) }
            .sorted { $0.duplicates.count > $1.duplicates.count }

        let report = SHReport(params: params, fileCount: files.count, uniqueCount: unique,
                              duplicateCount: dupPaths.count, candidateHits: candidateHits,
                              exactChecks: exactChecks, verifyRejected: verifyRejected,
                              avgFeatures: totalFeatures / max(1, files.count),
                              dfStops: idx.dfStops.count,
                              groups: finalGroups, applied: opts.apply,
                              elapsedSeconds: Date().timeIntervalSince(start))
        if !opts.reportPath.isEmpty {
            do {
                let data = try JSONEncoder().encode(report)
                let url = URL(fileURLWithPath: opts.reportPath)
                try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                         withIntermediateDirectories: true)
                try data.write(to: url, options: .atomic)
            } catch { log("[simhash] 报告写入失败: \(error)") }
        }
        saveIndex(idx, opts)
        log("[simhash] \(opts.apply ? "APPLY" : "DRY-RUN") 文件 \(files.count) | 唯一 \(unique)"
            + " | 重复 \(dupPaths.count) | 内容复核拦截 \(verifyRejected)"
            + " | 粗排召回 \(candidateHits) | 精排 \(exactChecks)"
            + String(format: " | %.1fs", report.elapsedSeconds))
        if opts.apply && !dupPaths.isEmpty {
            let moves = moveDuplicates(dupPaths, opts: opts)
            log("[simhash] 已移动 \(moves.count) 个重复文件")
        } else {
            for g in finalGroups.prefix(5) {
                log("  代表 \((g.representative as NSString).lastPathComponent)"
                    + " ← \(g.duplicates.count) 篇（距离 \(g.duplicates.map { $0.hamming }.prefix(3))）")
            }
        }
        return 0
    }

    fileprivate static func moveDuplicates(_ paths: [(src: String, size: Int64)],
                               opts: Options) -> [SHMove] {
        let fm = FileManager.default
        var root = opts.outRoot
        if root.isEmpty {
            let first = opts.paths.first ?? ""
            var isDir: ObjCBool = false
            root = (FileManager.default.fileExists(atPath: first, isDirectory: &isDir)
                    && isDir.boolValue) ? first : (first as NSString).deletingLastPathComponent
        }
        let dir = (root as NSString).appendingPathComponent(opts.duplicatesName)
        try? fm.createDirectory(atPath: dir, withIntermediateDirectories: true)
        var moves: [SHMove] = []
        for (src, _) in paths {
            var dst = (dir as NSString).appendingPathComponent((src as NSString).lastPathComponent)
            var n = 1
            while fm.fileExists(atPath: dst) {
                let ext = (src as NSString).pathExtension
                let base = ((src as NSString).deletingPathExtension as NSString).lastPathComponent
                dst = (dir as NSString).appendingPathComponent("\(base)_\(n).\(ext)")
                n += 1
            }
            do {
                try fm.moveItem(atPath: src, toPath: dst)
                moves.append(SHMove(src: src, dst: dst))
            } catch { log("[simhash] 移动失败 \(src): \(error)") }
        }
        if !moves.isEmpty, !opts.reportPath.isEmpty {
            let mf = SHManifest(version: 1, moves: moves)
            if let data = try? JSONEncoder().encode(mf) {
                try? data.write(to: URL(fileURLWithPath: opts.reportPath + ".manifest.json"),
                                options: .atomic)
            }
        }
        return moves
    }

    static func undo(_ path: String) -> Int32 {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let mf = try? JSONDecoder().decode(SHManifest.self, from: data) else {
            log("[simhash] 撤销清单读取失败"); return 1
        }
        var ok = 0
        for m in mf.moves {
            do { try FileManager.default.moveItem(atPath: m.dst, toPath: m.src); ok += 1 }
            catch { log("[simhash] 还原失败 \(m.dst): \(error)") }
        }
        log("[simhash] 已还原 \(ok)/\(mf.moves.count) 个文件")
        return 0
    }
}

// MARK: - 自检

extension SimHashDedupe {
    static func selfTest() -> Int32 {
        var failures: [String] = []
        func check(_ c: Bool, _ name: String) {
            if c { print("✓ \(name)") } else { print("FAIL: \(name)"); failures.append(name) }
        }
        func hx(_ s: String) -> String {
            let raw = String(Murmur.hash64(s), radix: 16)
            return String(repeating: "0", count: 16 - raw.count) + raw
        }

        // MurmurHash3 x64_128 (h1) 权威向量（纯 Python 独立复刻生成）
        check(hx("") == "0000000000000000", "MurmurHash3 空串向量")
        check(hx("foo") == "e271865701f54561", "MurmurHash3 foo 向量")
        check(hx("hello") == "cbd8a7b341bd9b02", "MurmurHash3 hello 向量")
        check(hx("量子力学") == "dc380f4d0ae2fc34", "MurmurHash3 中文向量")
        check(hx("abcdefghijklmnopq") == "7564747f88bda657", "MurmurHash3 跨 16 字节块 + 尾块")
        check(hx("The quick brown fox") == "85a60ea92caa4a2a", "MurmurHash3 长串向量")
        check(hx("天地玄黄宇宙洪荒") == "d2fa8de00a60c83d", "MurmurHash3 中文长串向量")

        // 清理与代表文本
        check(String(SHText.cleanAll("Hello, 世界！ \n\tni Hao。")) == "hello世界nihao",
              "清理剔标点空白、英文转小写")
        let long = String(repeating: "天地玄黄宇宙洪荒", count: 10)
        let segs = SHText.segments(long, sampleChars: 20, fullText: false)
        check(segs.count == 2 && segs[0].count == 20 && segs[1].count == 20,
              "代表文本 = 头 20 + 尾 20 两段")
        check(SHText.segments(long, sampleChars: 20, fullText: true).count == 1
              && SHText.segments(long, sampleChars: 20, fullText: true)[0].count == 80,
              "--full-text 返回全文单段")
        check(SHText.segments("abc", sampleChars: 20, fullText: false).count == 1,
              "短文本不重复取首尾")

        // 双窗口特征、topN、TF 权重、停用词与 DF 过滤
        var o = Options(); o.sampleChars = 0; o.fullText = true
        o.top3 = 3; o.top2 = 2; o.dfCut = 0
        let sampleText = "量子力学量子力学量子力学量力与机器学习机器学习"
        let f = SHFeatures.extract(sampleText, opts: o, stops: [], dfStops: [])
        check(f.count <= 5 && f.contains(where: { $0.0 == "量子力" }),
              "k=3 topN 与 k=2 topN 合并为特征集合")
        check(f.contains(where: { $0.0 == "量子力" && $0.1 >= 3 }),
              "特征权重 = 文档内 TF 频次")
        let withStop = SHFeatures.extract("实际上实际上实际上", opts: o,
                                          stops: WordFilters.builtinStopwords, dfStops: [])
        check(!withStop.contains(where: { $0.0 == "实际上" }), "停用词被过滤")
        let withDF = SHFeatures.extract("量子力学量子力学量子力学量力", opts: o,
                                        stops: [], dfStops: ["3:量子力"])
        check(!withDF.contains(where: { $0.0 == "量子力" }), "文档级高频 shingle 被过滤")

        // 核心词特征模式（ExtractCoreWords 流水线）
        var oc = Options(); oc.sampleChars = 0; oc.fullText = true
        oc.featureMode = "core-words"; oc.coreWordsTop = 20; oc.dfCut = 0.5
        let cw = SHFeatures.extract(String(repeating: "量子力学与机器学习", count: 60),
                                    opts: oc, stops: [], dfStops: [])
        check(cw.count <= 20 && cw.allSatisfy({ !$0.0.isEmpty && $0.1 > 0 }),
              "core-words 模式特征受 top 限制且带 TF 权重")
        check(SHFeatures.dfStops(["/nonexistent"], opts: oc, stops: []).isEmpty,
              "core-words 模式跳过 shingle DF 统计")
        check(o.featureMode == "shingles", "默认仍为 shingle 特征模式")

        // SimHash 加权累加二值化（手工构造 2 个特征）
        let a = Murmur.hash64("aaa"), b = Murmur.hash64("bbb")
        var manual: UInt64 = 0
        for i in 0..<64 {
            let va = ((a >> i) & 1 == 1) ? 5 : -5
            let vb = ((b >> i) & 1 == 1) ? 3 : -3
            if va + vb > 0 { manual |= UInt64(1) << i }
        }
        check(SHFinger.make([("aaa", 5), ("bbb", 3)], seed: 0) == manual,
              "SimHash 逐位 +w/−w 累加、>0 记 1")
        check(SHFinger.make([("aaa", 5)], seed: 0) == a, "单特征指纹 = 其特征哈希")
        check(SHFinger.make([], seed: 0) == 0, "空特征集合指纹为 0")

        // 分段、海明距离、鸽巢原理
        let fp: UInt64 = 0x0123_4567_89AB_CDEF
        var rebuilt: UInt64 = 0
        for i in 0..<4 { rebuilt |= UInt64(SHFinger.band(fp, i)) << (16 * i) }
        check(rebuilt == fp, "4 段 16 位可还原 64 位指纹")
        check(SHFinger.hamming(fp, fp) == 0 && SHFinger.hamming(fp, fp ^ 0x7) == 3,
              "海明距离 = XOR 后 1 的个数")
        // 枚举全部 1/2/3 bit 翻转（43744 种）：必有一段相同
        var allCollide = true
        func ck(_ x: UInt64) { if !SHFinger.collides(fp, x) { allCollide = false } }
        for i in 0..<64 { ck(fp ^ (UInt64(1) << i)) }
        for i in 0..<64 { for j in (i + 1)..<64 { ck(fp ^ (UInt64(1) << i) ^ (UInt64(1) << j)) } }
        for i in 0..<64 {
            for j in (i + 1)..<64 {
                for k in (j + 1)..<64 {
                    ck(fp ^ (UInt64(1) << i) ^ (UInt64(1) << j) ^ (UInt64(1) << k))
                }
            }
        }
        check(allCollide, "全部 1/2/3 bit 翻转（43744 种）均在至少一张 LSH 表碰撞")

        // LSH 索引：召回 → 精排 → 决策
        var idx = SHIndex()
        let id0 = idx.insert(fp: fp, path: "doc0", size: 10)
        check(id0 == 0 && idx.docs.count == 1, "新文章写入索引与 4 段键值")
        let variant = fp ^ 0x7    // 距离 3
        let cand = idx.candidates(variant)
        check(cand.contains(0), "海明距离 3 的变体被粗排召回")
        let near = idx.nearest(variant, ham: 3)
        check(near?.id == 0 && near?.dist == 3, "精排 XOR+popcount 得到距离 3")
        check(idx.nearest(fp ^ 0xFFFF, ham: 3) == nil, "距离 > 3 不判重复")
        _ = idx.insert(fp: fp | 0xF0, path: "doc1", size: 10)
        check(idx.docs.count == 2, "新文章入库、重复文章不入库")

        // 索引持久化往返
        let file = idx.toFile("p")
        let back = SHIndex.from(file)
        check(back.docs.count == 2 && back.nearest(variant, ham: 3)?.id == 0,
              "索引序列化/反序列化后查询结果一致")

        // 端到端：临时文件批量去重
        let tmp = NSTemporaryDirectory() + "/simhash-selftest-\(UUID().uuidString)"
        defer { try? FileManager.default.removeItem(atPath: tmp) }
        do {
            try FileManager.default.createDirectory(atPath: tmp, withIntermediateDirectories: true)
            let body = String(repeating: "量子力学与机器学习文本", count: 30)
            try body.write(toFile: tmp + "/a.md", atomically: true, encoding: .utf8)
            try body.write(toFile: tmp + "/b.md", atomically: true, encoding: .utf8)
            try body.write(toFile: tmp + "/c.md", atomically: true, encoding: .utf8)
            try String(repeating: "完全不同的另一本书内容", count: 30)
                .write(toFile: tmp + "/d.md", atomically: true, encoding: .utf8)
            var e = Options(); e.paths = [tmp]; e.exts = ["md"]; e.dfCut = 0
            e.reportPath = tmp + "/report.json"; e.indexPath = tmp + "/index.json"
            let rc = run(e)
            let rep = try? JSONDecoder().decode(SHReport.self,
                                                from: Data(contentsOf: URL(fileURLWithPath: e.reportPath)))
            check(rc == 0 && rep?.fileCount == 4, "端到端批量处理 4 个文件")
            check(rep?.duplicateCount == 2, "端到端：b/c 与 a 判重复，d 为新文章")
            check(rep?.uniqueCount == 2, "端到端：索引只保留 2 篇唯一文档")
            // 在线查询
            var q = e; q.queryPath = tmp + "/a.md"; q.noInsert = true
            let qrc = query(q)
            check(qrc == 0, "在线查询复用同一索引")
        } catch {
            check(false, "端到端临时文件: \(error)")
        }

        if failures.isEmpty {
            print("SimHashDedupe self-test: all checks passed")
            return 0
        }
        print("SimHashDedupe self-test: \(failures.count) FAILED")
        return 1
    }
}

// MARK: - CLI

private func simUsage() {
    print("""
    SimHashDedupe — SimHash(64) + 4 表 LSH 文本去重

    用法: booksimhash <目录|文件> [选项]

    指纹: 代表文本(默认头/尾各 20000 字，--full-text 用全文) → 清理 →
          k=3 滑窗去重停用词/DF 后取 top300(TF 权重) + k=2 取 top200 合并 →
          (--features core-words 可改用 NPMI/熵提取的 500 个核心词)
          每特征 MurmurHash3 64 位 → 64 维逐位 +w/−w 累加 → >0 记 1 → SimHash
    LSH:  指纹切 4×16 位建 4 张倒排表；查询召回 4 表并集 → XOR+popcount 精排
          → 距离 ≤3 判疑似 → 内容复核(3-gram Jaccard ≥0.5) 确认内容重复
          → 重复不入库，否则判新文章并写入指纹与 4 段键值

    选项:
      --sample-chars N        头/尾各取字数 (默认 20000)
      --full-text             不截取，用全文计算
      --features MODE         特征源: shingles(默认) | core-words(ExtractCoreWords)
      --core-words-top N      core-words 模式提取核心词数 (默认 500)
      --top3 N / --top2 N     k=3 / k=2 的高频特征数 (默认 300 / 200)
      --df-cut X              文档级停用词 DF 阈值，0=关闭 (默认 0.5)
      --df-sample N           DF 抽样文档数 (默认 1000)
      --ham N                 重复判定的海明距离上限 (默认 3)
      --no-verify             关闭内容复核（纯 SimHash 判定，可能误移不同的书）
      --verify-jaccard X      内容复核的 3-gram Jaccard 下限 (默认 0.5)
      --verify-sample-chars N 内容复核头/尾各取字数 (默认 20000；独立于 --sample-chars)
      --hash-seed N           MurmurHash3 种子 (默认 0)
      --no-stopwords          关闭停用词过滤
      --stopwords PATH        追加自定义停用词文件
      --index PATH            索引持久化路径（加载/更新）
      --query FILE            在线查询单篇新文章（配合 --index）
      --no-insert             查询判新时不写入索引
      --apply                 执行去重：把重复文件移入 --duplicates-name 文件夹
      --undo PATH             按 manifest 撤销移动
      --report PATH           JSON 报告
      --out-root PATH         重复文件夹位置 (默认 = 输入目录)
      --duplicates-name NAME  重复文件夹名 (默认 _重复)
      --ext LIST              目录扫描扩展名 (默认 md,txt)
      --limit N               最多处理 N 个文件
      --self-test             运行自检
      --help                  本帮助
    """)
}

@main enum SimHashDedupeMain {
    static func main() {
        let argv = CommandLine.arguments
        if argv.contains("--self-test") { exit(SimHashDedupe.selfTest()) }
        if argv.contains("--help") || argv.count < 2 { simUsage(); exit(0) }
        var o = SimHashDedupe.Options()
        var i = 1
        func value() -> String? {
            guard i + 1 < argv.count else { return nil }
            i += 1
            return argv[i]
        }
        while i < argv.count {
            let arg = argv[i]
            switch arg {
            case "--sample-chars": o.sampleChars = max(0, Int(value() ?? "") ?? 20000)
            case "--full-text": o.fullText = true
            case "--features": o.featureMode = (value() ?? "shingles") == "core-words" ? "core-words" : "shingles"
            case "--core-words-top": o.coreWordsTop = max(1, Int(value() ?? "") ?? 500)
            case "--top3": o.top3 = max(0, Int(value() ?? "") ?? 300)
            case "--top2": o.top2 = max(0, Int(value() ?? "") ?? 200)
            case "--df-cut": o.dfCut = Double(value() ?? "") ?? 0.5
            case "--df-sample": o.dfSample = max(1, Int(value() ?? "") ?? 1000)
            case "--ham": o.ham = max(0, min(64, Int(value() ?? "") ?? 3))
            case "--no-verify": o.verify = false
            case "--verify-jaccard": o.verifyJaccard = Double(value() ?? "") ?? 0.5
            case "--verify-sample-chars": o.verifySampleChars = Int(value() ?? "") ?? 20000
            case "--hash-seed": o.hashSeed = UInt64(value() ?? "") ?? 0
            case "--no-stopwords": o.useStopwords = false
            case "--stopwords": o.stopwordsPath = value() ?? ""
            case "--index": o.indexPath = value() ?? ""
            case "--query": o.queryPath = value() ?? ""
            case "--no-insert": o.noInsert = true
            case "--apply": o.apply = true
            case "--undo": o.undoPath = value() ?? ""
            case "--report": o.reportPath = value() ?? ""
            case "--out-root": o.outRoot = value() ?? ""
            case "--duplicates-name": o.duplicatesName = value() ?? "_重复"
            case "--ext": o.exts = Set((value() ?? "md,txt").split(separator: ",").map(String.init))
            case "--limit": o.limit = max(0, Int(value() ?? "") ?? 0)
            default:
                if arg.hasPrefix("-") { SimHashDedupe.log("[simhash] 未知选项 \(arg)"); simUsage(); exit(2) }
                else { o.paths.append(arg) }
            }
            i += 1
        }
        if o.queryPath.isEmpty && o.undoPath.isEmpty && o.paths.isEmpty { simUsage(); exit(2) }
        exit(SimHashDedupe.run(o))
    }
}
