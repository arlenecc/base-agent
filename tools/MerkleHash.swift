// MerkleHash.swift — 第四个工具：文档的 Merkle Tree 根签名
//
// 流程：
//   1) 按 Unicode 字符分片（--chunk-size，默认 2000 字），每片 UTF-8 字节 → BLAKE2b 哈希 = 叶子节点
//   2) 相邻两片哈希拼接（left || right）→ 再算一次 BLAKE2b = 中间节点
//   3) 从底层按树结构逐层向上两两聚合，奇数节点复制自身与自身配对
//   4) 最后一层仅剩两个中间节点时，拼接它们的哈希再算最后一次哈希 → Merkle Root
//      （即整本书唯一的根签名；仅一片时根 = 该片哈希，空文件根 = BLAKE2b(空)）
//
// BLAKE2b 为纯 Swift 实现（RFC 7693，12 轮压缩，输出 1..64 字节，默认 32），
// 自检用 Python hashlib.blake2b 的权威向量校验，无外部依赖。
//
// 编译（单文件需 -parse-as-library 才能用 @main）：
//   swiftc -O -parse-as-library -o /tmp/bookmerkle tools/MerkleHash.swift

import Foundation

// MARK: - BLAKE2b (RFC 7693)

private enum B2 {
    static let iv: [UInt64] = [
        0x6A09E667F3BCC908, 0xBB67AE8584CAA73B,
        0x3C6EF372FE94F82B, 0xA54FF53A5F1D36F1,
        0x510E527FADE682D1, 0x9B05688C2B3E6C1F,
        0x1F83D9ABFB41BD6B, 0x5BE0CD19137E2179,
    ]

    static let sigma: [[Int]] = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
        [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
        [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
        [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
        [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
        [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
        [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
        [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
        [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    ]

    @inline(__always) static func rotr(_ x: UInt64, _ n: Int) -> UInt64 {
        return (x >> n) | (x << (64 - n))
    }

    @inline(__always) static func g(_ v: inout [UInt64], _ a: Int, _ b: Int, _ c: Int, _ d: Int,
                                    _ x: UInt64, _ y: UInt64) {
        v[a] = v[a] &+ v[b] &+ x
        v[d] = rotr(v[d] ^ v[a], 32)
        v[c] = v[c] &+ v[d]
        v[b] = rotr(v[b] ^ v[c], 24)
        v[a] = v[a] &+ v[b] &+ y
        v[d] = rotr(v[d] ^ v[a], 16)
        v[c] = v[c] &+ v[d]
        v[b] = rotr(v[b] ^ v[c], 63)
    }

    /// 压缩一个 128 字节块；count = 处理完本块后的累计消息字节数（BLAKE2b 的 t 计数器低 64 位）。
    static func compress(_ h: inout [UInt64], block: [UInt8], count: UInt64, last: Bool) {
        var m = [UInt64](repeating: 0, count: 16)
        for i in 0..<16 {
            var w: UInt64 = 0
            for j in 0..<8 { w |= UInt64(block[i * 8 + j]) << (8 * j) }
            m[i] = w
        }
        var v = [UInt64](repeating: 0, count: 16)
        for i in 0..<8 {
            v[i] = h[i]
            v[i + 8] = iv[i]
        }
        v[12] ^= count
        if last { v[14] = ~v[14] }
        for r in 0..<12 {
            let s = sigma[r]
            g(&v, 0, 4, 8, 12, m[s[0]], m[s[1]])
            g(&v, 1, 5, 9, 13, m[s[2]], m[s[3]])
            g(&v, 2, 6, 10, 14, m[s[4]], m[s[5]])
            g(&v, 3, 7, 11, 15, m[s[6]], m[s[7]])
            g(&v, 0, 5, 10, 15, m[s[8]], m[s[9]])
            g(&v, 1, 6, 11, 12, m[s[10]], m[s[11]])
            g(&v, 2, 7, 8, 13, m[s[12]], m[s[13]])
            g(&v, 3, 4, 9, 14, m[s[14]], m[s[15]])
        }
        for i in 0..<8 { h[i] ^= v[i] ^ v[i + 8] }
    }

    /// BLAKE2b 一次性哈希。参数块 h[0] ^= 0x01010000 | digestLength（fanout=1, depth=1, 无 key）。
    static func hash(_ data: [UInt8], digestBytes: Int) -> [UInt8] {
        let nn = min(max(digestBytes, 1), 64)
        var h = iv
        h[0] ^= UInt64(0x01010000) | UInt64(nn)
        let n = data.count
        var offset = 0
        while true {
            var block = [UInt8](repeating: 0, count: 128)
            let take = min(128, n - offset)
            if take > 0 {
                for i in 0..<take { block[i] = data[offset + i] }
            }
            offset += take
            let last = offset >= n
            compress(&h, block: block, count: UInt64(offset), last: last)
            if last { break }
        }
        var out = [UInt8]()
        out.reserveCapacity(64)
        for i in 0..<8 {
            let w = h[i]
            for j in 0..<8 { out.append(UInt8(truncatingIfNeeded: (w >> (8 * j)) & 0xff)) }
        }
        return Array(out.prefix(nn))
    }

    static func hex(_ bytes: [UInt8]) -> String {
        let table = Array("0123456789abcdef".utf8)
        var out = [UInt8]()
        out.reserveCapacity(bytes.count * 2)
        for b in bytes {
            out.append(table[Int(b >> 4)])
            out.append(table[Int(b & 0x0f)])
        }
        return String(decoding: out, as: UTF8.self)
    }
}

// MARK: - Merkle Tree

private enum MerkleTree {
    /// 按 Unicode 标量（汉字/字母均计 1 字）分片，边扫描边累积 UTF-8 字节，
    /// 内存只占一个分片，无需把整本大书转成字符数组。
    static func leafHashes(_ text: String, chunkSize: Int, digestBytes: Int)
        -> (leaves: [[UInt8]], chars: Int) {
        var leaves: [[UInt8]] = []
        var buf = [UInt8]()
        buf.reserveCapacity(min(chunkSize * 3, 1 << 20))
        var cnt = 0
        var chars = 0
        for scalar in text.unicodeScalars {
            appendUTF8(scalar, to: &buf)
            cnt += 1
            chars += 1
            if cnt == chunkSize {
                leaves.append(B2.hash(buf, digestBytes: digestBytes))
                buf.removeAll(keepingCapacity: true)
                cnt = 0
            }
        }
        if cnt > 0 { leaves.append(B2.hash(buf, digestBytes: digestBytes)) }
        return (leaves, chars)
    }

    private static func appendUTF8(_ s: Unicode.Scalar, to buf: inout [UInt8]) {
        let v = s.value
        if v < 0x80 {
            buf.append(UInt8(truncatingIfNeeded: v))
        } else if v < 0x800 {
            buf.append(UInt8(truncatingIfNeeded: 0xC0 | (v >> 6)))
            buf.append(UInt8(truncatingIfNeeded: 0x80 | (v & 0x3F)))
        } else if v < 0x10000 {
            buf.append(UInt8(truncatingIfNeeded: 0xE0 | (v >> 12)))
            buf.append(UInt8(truncatingIfNeeded: 0x80 | ((v >> 6) & 0x3F)))
            buf.append(UInt8(truncatingIfNeeded: 0x80 | (v & 0x3F)))
        } else {
            buf.append(UInt8(truncatingIfNeeded: 0xF0 | (v >> 18)))
            buf.append(UInt8(truncatingIfNeeded: 0x80 | ((v >> 12) & 0x3F)))
            buf.append(UInt8(truncatingIfNeeded: 0x80 | ((v >> 6) & 0x3F)))
            buf.append(UInt8(truncatingIfNeeded: 0x80 | (v & 0x3F)))
        }
    }

    /// 逐层两两聚合：H(left || right)；某层节点数为奇数时最后一个节点复制自身配对。
    /// 空文件根 = BLAKE2b(空)；只有一片时根 = 该片哈希。
    static func build(leaves: [[UInt8]], digestBytes: Int)
        -> (root: [UInt8], levels: [[[UInt8]]]) {
        if leaves.isEmpty {
            let root = B2.hash([], digestBytes: digestBytes)
            return (root, [[root]])
        }
        var levels: [[[UInt8]]] = [leaves]
        var cur = leaves
        while cur.count > 1 {
            var next: [[UInt8]] = []
            next.reserveCapacity((cur.count + 1) / 2)
            var i = 0
            while i < cur.count {
                let left = cur[i]
                let right = (i + 1 < cur.count) ? cur[i + 1] : left
                var pair = left
                pair.append(contentsOf: right)
                next.append(B2.hash(pair, digestBytes: digestBytes))
                i += 2
            }
            levels.append(next)
            cur = next
        }
        return (cur[0], levels)
    }
}

// MARK: - 报告结构

private struct MerkleFileEntry: Codable {
    let path: String
    let bytes: Int64
    let chars: Int
    let leafCount: Int
    let levelCount: Int
    let root: String
    let leaves: [String]?
}

private struct MerkleReport: Codable {
    let chunkSize: Int
    let digestBytes: Int
    let fileCount: Int
    let uniqueRoots: Int
    let files: [MerkleFileEntry]
}

// MARK: - 主流程

enum MerkleHash {
    struct Options {
        var chunkSize = 2000
        var digestBytes = 32
        var reportPath = ""
        var includeLeaves = false
        var exts = Set(["md", "txt"])
        var limit = 0
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
                log("[merkle] 路径不存在，跳过: \(p)")
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

    fileprivate static func compute(path: String, opts: Options) -> MerkleFileEntry? {
        let fm = FileManager.default
        guard let data = fm.contents(atPath: path),
              let text = String(data: data, encoding: .utf8) else {
            log("[merkle] 非 UTF-8 或读取失败，跳过: \(path)")
            return nil
        }
        let (leaves, chars) = MerkleTree.leafHashes(text, chunkSize: opts.chunkSize,
                                                    digestBytes: opts.digestBytes)
        let (root, levels) = MerkleTree.build(leaves: leaves, digestBytes: opts.digestBytes)
        return MerkleFileEntry(
            path: path, bytes: Int64(data.count), chars: chars,
            leafCount: leaves.count, levelCount: levels.count,
            root: B2.hex(root),
            leaves: opts.includeLeaves ? leaves.map { B2.hex($0) } : nil)
    }

    static func run(_ opts: Options) -> Int32 {
        var files = collect(opts.paths, exts: opts.exts)
        if opts.limit > 0, files.count > opts.limit { files = Array(files.prefix(opts.limit)) }
        guard !files.isEmpty else {
            log("[merkle] 没有可处理的文件")
            return 1
        }
        log("[merkle] 处理 \(files.count) 个文件 | 分片 \(opts.chunkSize) 字 | BLAKE2b-\(opts.digestBytes * 8)")
        var entries: [MerkleFileEntry] = []
        for (i, path) in files.enumerated() {
            guard let e = compute(path: path, opts: opts) else { continue }
            entries.append(e)
            print("\(e.root)  \(path)")
            if (i + 1) % 1000 == 0 { log("[merkle] 进度 \(i + 1)/\(files.count)") }
        }
        let unique = Set(entries.map { $0.root }).count
        if !opts.reportPath.isEmpty {
            let report = MerkleReport(chunkSize: opts.chunkSize, digestBytes: opts.digestBytes,
                                      fileCount: entries.count, uniqueRoots: unique, files: entries)
            do {
                let data = try JSONEncoder().encode(report)
                let url = URL(fileURLWithPath: opts.reportPath)
                try? FileManager.default.createDirectory(
                    at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
                try data.write(to: url, options: .atomic)
                log("[merkle] 报告已写 \(opts.reportPath)")
            } catch {
                log("[merkle] 报告写入失败: \(error)")
            }
        }
        log("[merkle] 完成 \(entries.count) 个文件 | 唯一根签名 \(unique) 个"
            + "（根相同 ⟺ 逐片内容完全一致）")
        return 0
    }
}

// MARK: - 自检

extension MerkleHash {
    static func selfTest() -> Int32 {
        var failures: [String] = []
        func check(_ cond: Bool, _ name: String) {
            if cond { print("✓ \(name)") } else { print("FAIL: \(name)"); failures.append(name) }
        }
        func h(_ s: String, _ n: Int = 32) -> String {
            B2.hex(B2.hash(Array(s.utf8), digestBytes: n))
        }
        func root(_ text: String, _ chunk: Int, _ n: Int = 32) -> String {
            let ls = MerkleTree.leafHashes(text, chunkSize: chunk, digestBytes: n).leaves
            return B2.hex(MerkleTree.build(leaves: ls, digestBytes: n).root)
        }

        // BLAKE2b 权威向量（Python hashlib.blake2b 生成）
        let seq = (0..<1024).map { UInt8(truncatingIfNeeded: $0 % 256) }
        check(B2.hex(B2.hash([], digestBytes: 32)) ==
              "0e5751c026e543b2e8ab2eb06099daa1d1e5df47778f7787faab45cdf12fe3a8",
              "BLAKE2b-256 空串向量")
        check(B2.hex(B2.hash([], digestBytes: 64)) ==
              "786a02f742015903c6c6fd852552d272912f4740e15847618a86e217f71f5419d25e1031afee585313896444934eb04b903a685b1448b755d56f701afe9be2ce",
              "BLAKE2b-512 空串向量")
        check(h("abc") == "bddd813c634239723171ef3fee98579b94964e3bb1cb3e427262c8c068d52319",
              "BLAKE2b-256 abc 向量")
        check(h("abc", 64) ==
              "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d17d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923",
              "BLAKE2b-512 abc 向量")
        check(B2.hex(B2.hash(Array(repeating: UInt8(0x61), count: 1000), digestBytes: 32)) ==
              "e00b0ddbf1e2cdaf5c898e1a5e8826ea3a2c339bcf2a478da2e5fca9ff126672",
              "BLAKE2b-256 1000 字节（跨 128 字节块）")
        check(B2.hex(B2.hash(seq, digestBytes: 32)) ==
              "f1551feeb252c7e60bb362205bd1ac2f70b145260a91d41e8c5d0a187549a5f2",
              "BLAKE2b-256 1024 字节（恰好整块结尾）")
        check(B2.hex(B2.hash(seq, digestBytes: 64)) ==
              "6b490f42e902f61b1ee12d3c85e34152e37c94d07ab9ea577cad6a6eb4690fad38064f53a19c225703a5c52cdc9a85add71b339d327e1630ee3432b920240e8a",
              "BLAKE2b-512 1024 字节")

        // Merkle 树（与 Python 独立实现交叉验证的硬编码预期值）
        check(root("abc", 2000) == h("abc"), "单分片文件：根 = 全文 BLAKE2b")
        check(root("", 2000) == B2.hex(B2.hash([], digestBytes: 32)),
              "空文件：根 = BLAKE2b(空)")
        check(root("abcdefgh", 4) ==
              "c83c827eec71045c7379bda2ed0ccaf9c867f5535e334b07f0407303d975ec82",
              "两分片：根 = 两个叶子哈希拼接后再哈希")
        check(root("0123456789", 4) ==
              "c55b66e40273a4a29636b175885f800473561d539d055e75b67d9214fbe63e49",
              "三分片：奇数节点复制自身配对")
        check(root("天地玄黄宇宙洪荒", 3) ==
              "88979304d86bb98c4592e6946ccd72441e0e5a6f8ec51dd55f1b6aaa51baf01f",
              "中文 UTF-8 分片")
        check(root(String(repeating: "量化投资与机器学习", count: 7), 5) ==
              "da1d5f314e4e577915dde102757b57c96d12d30d504aa337db3a0edf33a99980",
              "13 个叶子多层递归聚合")

        // 树结构与确定性
        let ls = MerkleTree.leafHashes("0123456789", chunkSize: 4, digestBytes: 32).leaves
        let built = MerkleTree.build(leaves: ls, digestBytes: 32)
        check(ls.count == 3 && built.levels.count == 3 && built.levels.last?.count == 1,
              "树层级 3 → 2 → 1，最后收敛到单根")
        let selfPair = B2.hash(ls[2] + ls[2], digestBytes: 32)
        check(built.levels[1][1] == selfPair, "奇数层最后一个节点与自身拼接")
        check(root("0123456789", 4) == root("0123456789", 4), "确定性：同输入同根")
        check(root("0123456789", 4) != root("012345678X", 4), "改动任一分片即改变根")
        check(root("0123456789", 4) != root("0123456789", 3), "分片长度影响根")
        check(root("abcdefgh", 4, 64).count == 128, "BLAKE2b-512 输出 128 hex 字符")

        // 端到端：真实文件
        let tmp = NSTemporaryDirectory() + "/merkle-selftest-\(UUID().uuidString).txt"
        defer { try? FileManager.default.removeItem(atPath: tmp) }
        do {
            try "abcdefgh".write(toFile: tmp, atomically: true, encoding: .utf8)
            var o = Options(); o.paths = [tmp]; o.chunkSize = 4; o.reportPath = ""
            let e = compute(path: tmp, opts: o)
            check(e?.root == "c83c827eec71045c7379bda2ed0ccaf9c867f5535e334b07f0407303d975ec82",
                  "端到端读取文件得到相同根签名")
            check(e?.leafCount == 2 && e?.levelCount == 2 && e?.chars == 8,
                  "端到端叶子数/层数/字符数统计正确")
        } catch {
            check(false, "端到端临时文件写入")
        }

        if failures.isEmpty {
            print("MerkleHash self-test: all checks passed")
            return 0
        }
        print("MerkleHash self-test: \(failures.count) FAILED")
        return 1
    }
}

// MARK: - CLI

private func merkleUsage() {
    print("""
    MerkleHash — 文档 Merkle Tree 根签名（BLAKE2b）

    用法: bookmerkle <文件或目录> [更多路径...] [选项]

    算法:
      按 --chunk-size 字分片 → 每片 UTF-8 字节算 BLAKE2b = 叶子
      相邻两片哈希拼接后再算 BLAKE2b = 中间节点，逐层向上两两聚合
      （奇数节点复制自身配对）→ 最后两个中间节点再拼接哈希 = Merkle Root
      只有一片时根 = 该片哈希；空文件根 = BLAKE2b(空)

    选项:
      --chunk-size N          分片长度（Unicode 字符数，默认 2000）
      --digest-bytes N        BLAKE2b 输出字节数 1..64 (默认 32，即 BLAKE2b-256)
      --report PATH           写 JSON 报告（路径/大小/字符数/叶子数/层数/根）
      --leaves                报告中附带全部叶子哈希（可定位内容差异在哪片）
      --ext LIST              目录扫描的扩展名，逗号分隔 (默认 md,txt)
      --limit N               最多处理 N 个文件
      --self-test             运行自检（含 BLAKE2b 权威向量）
      --help                  本帮助
    """)
}

@main enum MerkleHashMain {
    static func main() {
        let argv = CommandLine.arguments
        if argv.contains("--self-test") { exit(MerkleHash.selfTest()) }
        if argv.contains("--help") || argv.count < 2 { merkleUsage(); exit(0) }
        var opts = MerkleHash.Options()
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
            case "--digest-bytes": opts.digestBytes = min(64, max(1, Int(value() ?? "") ?? 32))
            case "--report": opts.reportPath = value() ?? ""
            case "--leaves": opts.includeLeaves = true
            case "--ext":
                opts.exts = Set((value() ?? "md,txt").split(separator: ",").map(String.init))
            case "--limit": opts.limit = max(0, Int(value() ?? "") ?? 0)
            default:
                if arg.hasPrefix("-") {
                    MerkleHash.log("[merkle] 未知选项 \(arg)")
                    merkleUsage()
                    exit(2)
                } else {
                    opts.paths.append(arg)
                }
            }
            i += 1
        }
        guard !opts.paths.isEmpty else { merkleUsage(); exit(2) }
        exit(MerkleHash.run(opts))
    }
}
