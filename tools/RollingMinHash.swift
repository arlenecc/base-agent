// RollingMinHash.swift — 第五个工具：滚动哈希 + 分块 MinHash 加权平均签名
//
// 流程（严格按顺序）：
//   1) 原文按 --chunk-size 字分块（默认 2000），逐块清理：剔标点/空格/换行，
//      汉字保留、英文字母转小写，得到纯文字的 Unicode scalar 值流。
//   2) 每块用【滚动哈希】滑窗扫描：Rabin-Karp 式多项式哈希
//      h = Σ v_i·B^(k-1-i) (mod 2^64)，窗口滑动时
//      h = (h·B − v_out·B^k + v_in) mod 2^64，O(1) 更新，无需构造子串。
//      每个窗口的哈希放入 Set<UInt64> 自动去重 = 字符级 k-shingle 集合（默认 k=3）。
//   3) 过滤停用词：停用词表用同一滚动哈希公式预编码成哈希集合，O(1) 判定；
//      全虚字窗口（窗口内 k 字都属虚字表）与英文功能 3-gram 同样剔除。
//   4) 对最终集合用 128 个独立仿射哈希函数取最小值 → 每块一个 128 维 MinHash 签名。
//   5) 以每块清理后的字数（有效字数）为权重，对所有分块签名逐维加权平均
//      → 128 维全书平均向量，即文件的最终 MinHash 签名。
//
// 复用 CoreWordKit.swift 的停用词表。编译：
//   swiftc -O -o /tmp/bookminhash tools/CoreWordKit.swift tools/RollingMinHash.swift

import Foundation

// MARK: - 滚动哈希与文本清理

private enum Rolling {
    /// 多项式滚动哈希的基数（FNV-1a 大奇素数，保证 B 在 2^64 下可逆）。
    static let base: UInt64 = 0x100000001B3

    /// B^k mod 2^64，用于滑窗时减去移出字符的高次项。
    static func power(_ k: Int) -> UInt64 {
        var p: UInt64 = 1
        for _ in 0..<k { p = p &* base }
        return p
    }

    /// 单字符清理：汉字（含扩展区）保留、A-Z 转小写、a-z 保留，其余剔除。
    static func kept(_ s: Unicode.Scalar) -> UInt32? {
        let v = s.value
        if (v >= 0x4E00 && v <= 0x9FFF)
            || (v >= 0x3400 && v <= 0x4DBF)
            || (v >= 0x20000 && v <= 0x2A6DF)
            || (v >= 0xF900 && v <= 0xFAFF) {
            return v
        } else if v >= 0x41 && v <= 0x5A {
            return v + 32
        } else if v >= 0x61 && v <= 0x7A {
            return v
        }
        return nil
    }

    static func clean(_ text: String) -> [UInt32] {
        var out: [UInt32] = []
        for s in text.unicodeScalars {
            if let v = kept(s) { out.append(v) }
        }
        return out
    }

    /// 先按原文字符数分块，再逐块清理（边扫描边做，内存 O(chunk)）。
    /// 返回每块的原始字数与清理后的 scalar 值流。
    static func splitBlocks(_ text: String, chunkSize: Int) -> [(raw: Int, vals: [UInt32])] {
        var blocks: [(raw: Int, vals: [UInt32])] = []
        var cur: [UInt32] = []
        cur.reserveCapacity(chunkSize)
        var raw = 0
        for s in text.unicodeScalars {
            raw += 1
            if let v = kept(s) { cur.append(v) }
            if raw == chunkSize {
                blocks.append((chunkSize, cur))
                cur.removeAll(keepingCapacity: true)
                raw = 0
            }
        }
        if raw > 0 { blocks.append((raw, cur)) }
        return blocks
    }

    /// 初始窗口哈希：Σ v_i·B^(k-1-i)。停用词也用此公式编码。
    static func windowHash(_ vals: [UInt32], k: Int) -> UInt64 {
        var h: UInt64 = 0
        for j in 0..<min(k, vals.count) { h = h &* base &+ UInt64(vals[j]) }
        return h
    }

    /// 全部窗口的滚动哈希序列（自检用来验证增量公式；不过滤停用词）。
    static func windowSequence(_ vals: [UInt32], k: Int) -> [UInt64] {
        guard vals.count >= k else { return [] }
        var out: [UInt64] = []
        out.reserveCapacity(vals.count - k + 1)
        var h = windowHash(vals, k: k)
        out.append(h)
        let bk = power(k)
        if vals.count > k {
            for i in 1...(vals.count - k) {
                h = (h &* base) &- (UInt64(vals[i - 1]) &* bk) &+ UInt64(vals[i + k - 1])
                out.append(h)
            }
        }
        return out
    }

    /// 纯 ASCII 高频 3-gram（英文功能片段停用词）。
    static let ascii3Stops: Set<String> = [
        "the", "and", "ing", "her", "ere", "ent", "tha", "ith", "hat", "for", "ted",
        "his", "thi", "ver", "was", "are", "not", "all", "ont", "you", "ave", "ell",
    ]

    /// 停用词表 → 滚动哈希集合（只编码清理后长度恰为 k 的词，长度不符的永不命中）。
    static func stopHashes(k: Int, extraPath: String, enabled: Bool) -> Set<UInt64> {
        guard enabled else { return [] }
        var words = WordFilters.builtinStopwords
        words.formUnion(WordFilters.loadExtra(extraPath))
        if k == 3 { words.formUnion(ascii3Stops) }
        var out = Set<UInt64>()
        for w in words {
            let vals = clean(w)
            if vals.count == k { out.insert(windowHash(vals, k: k)) }
        }
        return out
    }

    /// 虚字表的 scalar 值集合，用于 O(1) 判定「窗口内全是虚字」。
    static let weakScalars: Set<UInt32> = Set(
        WordFilters.weakBoundaryChars.compactMap { $0.unicodeScalars.first?.value })

    /// 滚动滑窗提取 k-shingle 哈希集合（Set 自动去重），并过滤停用词/全虚字窗口。
    static func shingleHashes(_ vals: [UInt32], k: Int, stop: Set<UInt64>,
                              useStopwords: Bool) -> Set<UInt64> {
        guard vals.count >= k else { return [] }
        var out = Set<UInt64>()
        let bk = power(k)
        var h = windowHash(vals, k: k)

        func consider(_ idx: Int, _ hv: UInt64) {
            if useStopwords {
                if stop.contains(hv) { return }
                var allWeak = true
                for j in 0..<k {
                    if !weakScalars.contains(vals[idx + j]) { allWeak = false; break }
                }
                if allWeak { return }
            }
            out.insert(hv)
        }

        consider(0, h)
        if vals.count > k {
            for i in 1...(vals.count - k) {
                h = (h &* base) &- (UInt64(vals[i - 1]) &* bk) &+ UInt64(vals[i + k - 1])
                consider(i, h)
            }
        }
        return out
    }
}

// MARK: - MinHash 签名族与加权平均

private enum MH {
    static let two64 = 18446744073709551616.0   // 2^64，用于把 64 位哈希归一到 [0,1)

    /// splitmix64：确定性地生成 128 组独立的仿射哈希参数。
    @inline(__always) static func next(_ state: inout UInt64) -> UInt64 {
        state = state &+ 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }

    static func family(dim: Int, seed: UInt64) -> (a: [UInt64], b: [UInt64]) {
        var st = seed
        var a: [UInt64] = []
        var b: [UInt64] = []
        a.reserveCapacity(dim)
        b.reserveCapacity(dim)
        for _ in 0..<dim {
            a.append(next(&st) | 1)   // 奇数 → 在 2^64 上是置换，分布均匀
            b.append(next(&st))
        }
        return (a, b)
    }

    /// 对集合元素 x 计算 128 个 h_i(x)=a_i·x+b_i (mod 2^64)，逐维取最小值。
    /// 返回归一化到 [0,1) 的 128 维签名；空集合返回全 1（哨兵）。
    static func signature(_ set: Set<UInt64>, a: [UInt64], b: [UInt64]) -> [Double] {
        var mins = [UInt64](repeating: UInt64.max, count: a.count)
        for x in set {
            for i in 0..<a.count {
                let h = a[i] &* x &+ b[i]
                if h < mins[i] { mins[i] = h }
            }
        }
        return mins.map { Double($0) / two64 }
    }

    /// 按块的字数权重逐维加权平均；无有效块时返回全 0 向量。
    static func average(_ blocks: [(weight: Double, sig: [Double])], dim: Int) -> [Double] {
        let total = blocks.reduce(0.0) { $0 + $1.weight }
        guard total > 0 else { return [Double](repeating: 0, count: dim) }
        var acc = [Double](repeating: 0, count: dim)
        for blk in blocks {
            for i in 0..<dim { acc[i] += blk.weight * blk.sig[i] }
        }
        return acc.map { $0 / total }
    }
}

// MARK: - 报告结构

private struct BlockInfo: Codable {
    let rawChars: Int
    let chars: Int
    let shingleCount: Int
    let weight: Double
    let signature: [Double]?
}

private struct FileResult: Codable {
    let path: String
    let bytes: Int64
    let chars: Int
    let blockCount: Int
    let usedBlocks: Int
    let vector: [Double]
    let blocks: [BlockInfo]
}

private struct RMHReport: Codable {
    let chunkSize: Int
    let shingleK: Int
    let dim: Int
    let seed: UInt64
    let rollingBase: String
    let useStopwords: Bool
    let fileCount: Int
    let files: [FileResult]
}

// MARK: - 主流程

enum RollingMinHash {
    struct Options {
        var chunkSize = 2000
        var shingleK = 3
        var dim = 128
        var seed: UInt64 = 0x5EED_C0DE_1234_5678
        var useStopwords = true
        var stopwordsPath = ""
        var includeSignatures = false
        var exts = Set(["md", "txt"])
        var limit = 0
        var reportPath = ""
        var paths: [String] = []
    }

    static func log(_ s: String) {
        FileHandle.standardError.write((s + "\n").data(using: .utf8)!)
    }

    static func collect(_ inputs: [String], exts: Set<String>) -> [String] {
        let fm = FileManager.default
        var out: [String] = []
        for p in inputs {
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: p, isDirectory: &isDir) else {
                log("[rmh] 路径不存在，跳过: \(p)")
                continue
            }
            if isDir.boolValue {
                guard let en = fm.enumerator(atPath: p) else { continue }
                for case let sub as String in en {
                    let parts = sub.split(separator: "/")
                    if parts.contains(where: { $0.hasPrefix(".") }) { continue }
                    guard exts.contains((sub as NSString).pathExtension.lowercased()) else { continue }
                    out.append((p as NSString).appendingPathComponent(sub))
                }
            } else {
                out.append(p)
            }
        }
        return out.sorted()
    }

    fileprivate static func compute(path: String, opts: Options,
                                    family: (a: [UInt64], b: [UInt64]),
                                    stop: Set<UInt64>) -> FileResult? {
        guard let data = FileManager.default.contents(atPath: path),
              let text = String(data: data, encoding: .utf8) else {
            log("[rmh] 非 UTF-8 或读取失败，跳过: \(path)")
            return nil
        }
        let blocks = Rolling.splitBlocks(text, chunkSize: opts.chunkSize)
        var infos: [BlockInfo] = []
        var weighted: [(weight: Double, sig: [Double])] = []
        var totalChars = 0
        for b in blocks {
            totalChars += b.vals.count
            let set = Rolling.shingleHashes(b.vals, k: opts.shingleK, stop: stop,
                                            useStopwords: opts.useStopwords)
            guard !set.isEmpty else {
                infos.append(BlockInfo(rawChars: b.raw, chars: b.vals.count,
                                       shingleCount: 0, weight: 0, signature: nil))
                continue
            }
            let sig = MH.signature(set, a: family.a, b: family.b)
            let w = Double(b.vals.count)
            weighted.append((w, sig))
            infos.append(BlockInfo(rawChars: b.raw, chars: b.vals.count,
                                   shingleCount: set.count, weight: w,
                                   signature: opts.includeSignatures ? sig : nil))
        }
        return FileResult(path: path, bytes: Int64(data.count), chars: totalChars,
                          blockCount: blocks.count, usedBlocks: weighted.count,
                          vector: MH.average(weighted, dim: opts.dim), blocks: infos)
    }

    static func run(_ opts: Options) -> Int32 {
        var files = collect(opts.paths, exts: opts.exts)
        if opts.limit > 0, files.count > opts.limit { files = Array(files.prefix(opts.limit)) }
        guard !files.isEmpty else { log("[rmh] 没有可处理的文件"); return 1 }
        let family = MH.family(dim: opts.dim, seed: opts.seed)
        let stop = Rolling.stopHashes(k: opts.shingleK, extraPath: opts.stopwordsPath,
                                      enabled: opts.useStopwords)
        log("[rmh] 处理 \(files.count) 个文件 | 分块 \(opts.chunkSize) 字 | k=\(opts.shingleK)"
            + " | 滚动哈希 base=0x\(String(Rolling.base, radix: 16))"
            + " | MinHash \(opts.dim) 维 | 停用词哈希 \(stop.count) 个")
        var results: [FileResult] = []
        for (i, path) in files.enumerated() {
            guard let e = compute(path: path, opts: opts, family: family, stop: stop) else { continue }
            results.append(e)
            let head = e.vector.prefix(3).map { String(format: "%.6f", $0) }.joined(separator: ",")
            print("\(path)  v=[\(head),…]  blocks=\(e.usedBlocks)/\(e.blockCount)  dim=\(e.vector.count)")
            if (i + 1) % 1000 == 0 { log("[rmh] 进度 \(i + 1)/\(files.count)") }
        }
        if !opts.reportPath.isEmpty {
            let report = RMHReport(chunkSize: opts.chunkSize, shingleK: opts.shingleK,
                                   dim: opts.dim, seed: opts.seed,
                                   rollingBase: "0x" + String(Rolling.base, radix: 16),
                                   useStopwords: opts.useStopwords,
                                   fileCount: results.count, files: results)
            do {
                let data = try JSONEncoder().encode(report)
                let url = URL(fileURLWithPath: opts.reportPath)
                try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                         withIntermediateDirectories: true)
                try data.write(to: url, options: .atomic)
                log("[rmh] 报告已写 \(opts.reportPath)")
            } catch { log("[rmh] 报告写入失败: \(error)") }
        }
        log("[rmh] 完成 \(results.count) 个文件，每个得到 \(opts.dim) 维全书平均 MinHash 签名向量")
        return 0
    }
}

// MARK: - 自检

extension RollingMinHash {
    static func selfTest() -> Int32 {
        var failures: [String] = []
        func check(_ cond: Bool, _ name: String) {
            if cond { print("✓ \(name)") } else { print("FAIL: \(name)"); failures.append(name) }
        }
        func scalars(_ s: String) -> String {
            String(decoding: Rolling.clean(s).flatMap { v -> [UInt8] in
                Array(String(UnicodeScalar(v)!).utf8)
            }, as: UTF8.self)
        }
        func direct(_ vals: [UInt32], _ i: Int, _ k: Int) -> UInt64 {
            var h: UInt64 = 0
            for j in 0..<k { h = h &* Rolling.base &+ UInt64(vals[i + j]) }
            return h
        }

        // 清理
        check(scalars("Hello, 世界！ \n\tni Hao。") == "hello世界nihao",
              "清理：剔标点空白换行、汉字保留、英文转小写")
        check(Rolling.clean("ＡＢＣ123！").isEmpty, "全角字母/数字被剔除")

        // 分块：按原文字符数切，逐块清理
        let blocks = Rolling.splitBlocks("a,b!c d\nef", chunkSize: 4)
        check(blocks.count == 3 && blocks[0].raw == 4 && blocks[2].raw == 2,
              "按原文字符数分块（最后一块可短）")
        check(blocks.map { $0.vals.count } == [2, 2, 2], "逐块清理后字数")
        check(Rolling.splitBlocks("", chunkSize: 2000).isEmpty, "空文本无分块")

        // 滚动哈希增量公式 == 每个窗口从头直接计算
        let vals: [UInt32] = Rolling.clean("天地玄黄宇宙洪荒日月盈昃辰宿列张abcxyz")
        for k in [2, 3, 4, 5] {
            let seq = Rolling.windowSequence(vals, k: k)
            var ok = seq.count == vals.count - k + 1
            if ok {
                for (i, h) in seq.enumerated() where h != direct(vals, i, k) { ok = false; break }
            }
            check(ok, "滚动哈希 k=\(k)：O(1) 滑窗与逐窗口直接计算一致")
        }

        // shingle 集合去重
        let sh = Rolling.shingleHashes(Rolling.clean("abcabcabcabc"), k: 3,
                                       stop: [], useStopwords: false)
        check(sh.count == 3, "滚动哈希 3-shingle 集合自动去重")

        // 停用词哈希预编码与全虚字判定
        let stops = Rolling.stopHashes(k: 3, extraPath: "", enabled: true)
        check(stops.contains(Rolling.windowHash(Rolling.clean("实际上"), k: 3)),
              "停用词表用同一滚动哈希公式预编码")
        check(stops.contains(Rolling.windowHash(Rolling.clean("the"), k: 3)),
              "英文功能 3-gram 进入停用哈希集合")
        let weakOnly = Rolling.shingleHashes(Rolling.clean("的了着"), k: 3,
                                             stop: [], useStopwords: true)
        check(weakOnly.isEmpty, "全虚字窗口被过滤")
        let withStop = Rolling.shingleHashes(Rolling.clean("实际上"), k: 3,
                                             stop: stops, useStopwords: true)
        check(withStop.isEmpty, "命中停用词的窗口被过滤")
        let noFilter = Rolling.shingleHashes(Rolling.clean("实际上"), k: 3,
                                             stop: stops, useStopwords: false)
        check(noFilter.count == 1, "--no-stopwords 关闭过滤后窗口保留")

        // MinHash 签名
        let fam = MH.family(dim: 128, seed: 0x5EED_C0DE_1234_5678)
        let s1 = MH.signature(sh, a: fam.a, b: fam.b)
        let s2 = MH.signature(sh, a: fam.a, b: fam.b)
        check(s1.count == 128 && s1 == s2, "128 维签名确定性")
        check(MH.signature([], a: fam.a, b: fam.b).allSatisfy { $0 == 1.0 },
              "空集合返回全 1 哨兵签名")
        let s3 = MH.signature(Rolling.shingleHashes(Rolling.clean("完全不同的量子力学文本"),
                                                    k: 3, stop: [], useStopwords: false),
                              a: fam.a, b: fam.b)
        check(s1 != s3, "不同集合得到不同签名")

        // 加权平均公式
        let w1 = 3.0, w2 = 7.0
        let avg = MH.average([(w1, s1), (w2, s3)], dim: 128)
        var formulaOK = true
        for i in 0..<128 where abs(avg[i] - (w1 * s1[i] + w2 * s3[i]) / (w1 + w2)) > 1e-15 {
            formulaOK = false; break
        }
        check(formulaOK, "按块字数加权平均逐维等于 Σw·v / Σw")
        check(MH.average([], dim: 128).allSatisfy { $0 == 0 }, "无有效块返回全 0 向量")

        // 端到端：真实文件（关闭停用词，便于与字符串法交叉验证）
        let tmp = NSTemporaryDirectory() + "/rmh-selftest-\(UUID().uuidString).txt"
        defer { try? FileManager.default.removeItem(atPath: tmp) }
        do {
            try "天地玄黄宇宙洪荒日月盈昃辰宿列张".write(toFile: tmp, atomically: true,
                                                     encoding: .utf8)
            var o = Options(); o.paths = [tmp]; o.chunkSize = 5
            o.useStopwords = false; o.reportPath = ""
            let fam2 = MH.family(dim: o.dim, seed: o.seed)
            let e = compute(path: tmp, opts: o, family: fam2, stop: [])
            check(e?.blockCount == 4 && e?.usedBlocks == 3 && e?.chars == 16,
                  "端到端分块数/有效块数/字数（末块 1 字无 shingle 被跳过）")
            check(e?.vector.count == 128 && e?.vector.allSatisfy({ $0 >= 0 && $0 <= 1 }) ?? false,
                  "端到端 128 维平均向量取值在 [0,1]")
            let again = compute(path: tmp, opts: o, family: fam2, stop: [])
            check(e?.vector == again?.vector, "端到端确定性")
        } catch {
            check(false, "端到端临时文件写入")
        }

        if failures.isEmpty {
            print("RollingMinHash self-test: all checks passed")
            return 0
        }
        print("RollingMinHash self-test: \(failures.count) FAILED")
        return 1
    }
}

// MARK: - CLI

private func rmhUsage() {
    print("""
    RollingMinHash — 滚动哈希 + 分块 MinHash 加权平均签名

    用法: bookminhash <文件或目录> [更多路径...] [选项]

    流程:
      原文按 --chunk-size 字分块 → 逐块清理(剔标点/空白，汉字保留、英文转小写)
      → 滚动哈希滑窗提取 k-shingle 集合(B=0x100000001b3, mod 2^64, O(1) 更新)
      → 停用词/全虚字过滤(停用词预编码为滚动哈希)
      → 每块 128 维 MinHash 签名(128 个仿射哈希取最小值)
      → 按每块清理后字数加权平均 → 128 维全书平均向量

    选项:
      --chunk-size N          分块长度，原文字符数 (默认 2000)
      --shingle-k N           滑窗长度 (默认 3)
      --dim N                 MinHash 签名维度 (默认 128)
      --seed N                哈希族随机种子 (默认固定，保证可复现)
      --no-stopwords          关闭停用词过滤
      --stopwords PATH        追加自定义停用词文件
      --signatures            报告中附带每个分块的 128 维签名
      --report PATH           写 JSON 报告（含 128 维全书平均向量）
      --ext LIST              目录扫描扩展名，逗号分隔 (默认 md,txt)
      --limit N               最多处理 N 个文件
      --self-test             运行自检
      --help                  本帮助
    """)
}

@main enum RollingMinHashMain {
    static func main() {
        let argv = CommandLine.arguments
        if argv.contains("--self-test") { exit(RollingMinHash.selfTest()) }
        if argv.contains("--help") || argv.count < 2 { rmhUsage(); exit(0) }
        var opts = RollingMinHash.Options()
        var i = 1
        func value() -> String? {
            guard i + 1 < argv.count else { return nil }
            i += 1
            return argv[i]
        }
        while i < argv.count {
            let arg = argv[i]
            switch arg {
            case "--chunk-size": opts.chunkSize = max(1, Int(value() ?? "") ?? 2000)
            case "--shingle-k": opts.shingleK = max(2, Int(value() ?? "") ?? 3)
            case "--dim": opts.dim = min(1024, max(1, Int(value() ?? "") ?? 128))
            case "--seed": opts.seed = UInt64(value() ?? "") ?? 0x5EED_C0DE_1234_5678
            case "--no-stopwords": opts.useStopwords = false
            case "--stopwords": opts.stopwordsPath = value() ?? ""
            case "--signatures": opts.includeSignatures = true
            case "--report": opts.reportPath = value() ?? ""
            case "--ext": opts.exts = Set((value() ?? "md,txt").split(separator: ",").map(String.init))
            case "--limit": opts.limit = max(0, Int(value() ?? "") ?? 0)
            default:
                if arg.hasPrefix("-") {
                    RollingMinHash.log("[rmh] 未知选项 \(arg)")
                    rmhUsage()
                    exit(2)
                } else {
                    opts.paths.append(arg)
                }
            }
            i += 1
        }
        guard !opts.paths.isEmpty else { rmhUsage(); exit(2) }
        exit(RollingMinHash.run(opts))
    }
}
