// KeywordExtractor.swift — 无监督单句核心词提取
//  1) 清理原文：剔标点/空格/换行，汉字保留、英文转小写
//  2) 滑窗 k=2,3,4,5 生成字符级 N-gram，Set 自动去重，过滤停用词/全虚字 → 备选集
//  3) 整句原文与每个备选词分别过 BGE ONNX embedding → 余弦相似度排序取 top3
import Foundation

struct KeywordHit: Codable {
    let word: String
    let similarity: Double
}

enum KeywordExtractor {
    /// 清理：仅保留汉字（含扩展区）+ 小写英文字母。
    static func clean(_ text: String) -> [Character] {
        var out: [Character] = []
        for scalar in text.unicodeScalars {
            let v = scalar.value
            if (v >= 0x4E00 && v <= 0x9FFF) || (v >= 0x3400 && v <= 0x4DBF)
                || (v >= 0x20000 && v <= 0x2A6DF) || (v >= 0xF900 && v <= 0xFAFF) {
                out.append(Character(scalar))
            } else if v >= 0x41 && v <= 0x5A {
                if let lower = UnicodeScalar(v + 32) { out.append(Character(lower)) }
            } else if v >= 0x61 && v <= 0x7A {
                out.append(Character(scalar))
            }
        }
        return out
    }

    /// 备选集：按指定窗口 k 生成全部滑窗子串，去重、过滤停用词与全虚字窗口，字典序保证确定性。
    static func candidates(_ text: String, stops: Set<String>, weak: Set<Character>,
                           windows: [Int]) -> [String] {
        let chars = clean(text)
        var set = Set<String>()
        for k in windows {
            guard chars.count >= k else { continue }
            for i in 0...(chars.count - k) {
                let g = String(chars[i..<(i + k)])
                if stops.contains(g) { continue }
                if g.allSatisfy({ weak.contains($0) }) { continue }
                set.insert(g)
            }
        }
        return set.sorted()
    }

    static func loadStopwords(_ path: String) -> Set<String> {
        guard let text = try? String(contentsOfFile: path, encoding: .utf8) else { return [] }
        var out = Set<String>()
        text.enumerateLines { line, _ in
            let w = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if !w.isEmpty { out.insert(w) }
        }
        return out
    }

    /// 虚字单字表（与项目 CoreWordKit 的 weakBoundaryChars 一致）。
    static let weakChars: Set<Character> = Set(
        "的 了 着 过 得 地 是 在 和 与 或 也 就 都 而 之 以 于 其 这 那 一 不 很 更 最 可 能 会 要 将 被 把 从 对 为 所 由 并 且 但 则 者 矣 焉 乎 哉 又 再 还 才 只 已 曾 乃 若 即 便 却 竟 偏"
            .filter { !$0.isWhitespace })

    static func dot(_ a: [Float], _ b: [Float]) -> Double {
        var s: Double = 0
        for i in 0..<min(a.count, b.count) { s += Double(a[i]) * Double(b[i]) }
        return s
    }

    /// 对一句话提取 topN 核心词，返回 (词, 与整句余弦相似度)。
    static func extract(sentence: String, embedder: BGEEmbedder,
                        tokenizer: WordPieceTokenizer, stops: Set<String>,
                        maxLen: Int, batchSize: Int, windows: [Int],
                        rank: String = "similarity", mmrLambda: Double = 0.7,
                        candidateLimit: Int = 0) throws -> (hits: [KeywordHit], candidateCount: Int) {
        var cand = candidates(sentence, stops: stops, weak: weakChars, windows: windows)
        if candidateLimit > 0 && cand.count > candidateLimit {
            cand = Array(cand.prefix(candidateLimit))
        }
        guard !cand.isEmpty else { return ([], 0) }
        // 第 0 项为整句原文（保留标点），其后为清理后的备选词
        let texts = [sentence] + cand
        let encoded = texts.map { tokenizer.encode($0, maxLen: maxLen) }
        let vecs = try embedder.embed(batches: encoded, seqLen: maxLen, batchSize: batchSize)
        guard vecs.count == texts.count else { return ([], cand.count) }
        let q = vecs[0]
        var sims = [Double]()
        sims.reserveCapacity(cand.count)
        for i in 1..<vecs.count { sims.append(dot(q, vecs[i])) }

        let order: [Int]
        if rank == "mmr" {
            // MMR：λ·与整句相似度 − (1−λ)·与已选词的最大冗余，迭代选取，
            // 抑制 topN 全来自同一段文字的滑动错位片段。
            var selected: [Int] = []
            var remaining = Set(0..<cand.count)
            while !remaining.isEmpty {
                var best = -1
                var bestScore = -Double.infinity
                for i in remaining {
                    var maxRed = 0.0
                    for j in selected {
                        let d = dot(vecs[i + 1], vecs[j + 1])
                        if d > maxRed { maxRed = d }
                    }
                    let score = mmrLambda * sims[i] - (1 - mmrLambda) * maxRed
                    if score > bestScore || (score == bestScore && best >= 0 && i < best) {
                        bestScore = score
                        best = i
                    }
                }
                guard best >= 0 else { break }
                selected.append(best)
                remaining.remove(best)
            }
            order = selected
        } else {
            order = (0..<cand.count).sorted {
                if sims[$0] != sims[$1] { return sims[$0] > sims[$1] }
                return cand[$0] < cand[$1]
            }
        }
        let hits = order.map { KeywordHit(word: cand[$0], similarity: sims[$0]) }
        return (hits, cand.count)
    }
}
