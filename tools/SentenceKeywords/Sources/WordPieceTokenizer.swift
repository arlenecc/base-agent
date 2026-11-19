// WordPieceTokenizer.swift — BERT WordPiece 分词，复刻 tokenizer.json 的配置：
// BertNormalizer(lowercase=true, handle_chinese_chars=true) + BertPreTokenizer + WordPiece(##)
import Foundation

final class WordPieceTokenizer {
    private let vocab: [String: Int32]
    let clsId: Int64
    let sepId: Int64
    let padId: Int64
    let unkId: Int64
    private let lowercase: Bool

    /// 支持两种词表来源：
    ///  - vocab.txt：行号即 token id（bge-micro-v2）
    ///  - tokenizer.json：读 model.vocab 字典与 normalizer.lowercase（bge-small-zh-v1.5）
    init(vocabPath: String) throws {
        let v: [String: Int32]
        var doLower = true
        if vocabPath.hasSuffix(".json") {
            let data = try Data(contentsOf: URL(fileURLWithPath: vocabPath))
            guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let model = root["model"] as? [String: Any],
                  let rawVocab = model["vocab"] as? [String: Any] else {
                throw NSError(domain: "skw", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: "tokenizer.json 缺少 model.vocab"])
            }
            var m: [String: Int32] = [:]
            for (k, val) in rawVocab {
                if let n = val as? Int { m[k] = Int32(n) }
                else if let n = val as? NSNumber { m[k] = n.int32Value }
            }
            v = m
            if let norm = root["normalizer"] as? [String: Any],
               let flag = norm["lowercase"] as? Bool { doLower = flag }
        } else {
            let text = try String(contentsOfFile: vocabPath, encoding: .utf8)
            var m: [String: Int32] = [:]
            let lines = text.split(separator: "\n", omittingEmptySubsequences: false)
            for (i, line) in lines.enumerated() { m[String(line)] = Int32(i) }
            v = m
        }
        vocab = v
        lowercase = doLower
        clsId = Int64(v["[CLS]"] ?? 101)
        sepId = Int64(v["[SEP]"] ?? 102)
        padId = Int64(v["[PAD]"] ?? 0)
        unkId = Int64(v["[UNK]"] ?? 100)
    }

    var vocabSize: Int { vocab.count }

    func hasToken(_ s: String) -> Bool { vocab[s] != nil }

    /// [CLS] + WordPiece tokens + [SEP]，右补 [PAD] 到 maxLen；types 全 0。
    func encode(_ text: String, maxLen: Int) -> ([Int64], [Int64], [Int64]) {
        var ids: [Int64] = [clsId]
        for piece in preTokenize(normalize(text)) {
            for t in wordpiece(piece) { ids.append(Int64(t)) }
        }
        if ids.count > maxLen - 1 { ids = Array(ids.prefix(maxLen - 1)) }
        ids.append(sepId)
        var mask = [Int64](repeating: 1, count: ids.count)
        var types = [Int64](repeating: 0, count: ids.count)
        while ids.count < maxLen {
            ids.append(padId); mask.append(0); types.append(0)
        }
        return (ids, mask, types)
    }

    // MARK: - normalization

    private func normalize(_ s: String) -> String {
        var out = ""
        out.reserveCapacity(s.count * 2)
        for scalar in s.unicodeScalars {
            let v = scalar.value
            // clean_text：剔除控制字符（保留 \t \n \r 当空白）
            if v < 0x20 && v != 0x09 && v != 0x0A && v != 0x0D { continue }
            if v == 0x7F { continue }
            let ch = Character(scalar)
            if scalar.properties.isWhitespace {
                out.append(" ")
                continue
            }
            // handle_chinese_chars：汉字两侧加空格
            if isCJK(v) {
                out.append(" ")
                out.append(ch)
                out.append(" ")
            } else {
                let piece = lowercase ? String(ch).lowercased() : String(ch)
                for c in piece { out.append(c) }
            }
        }
        return out
    }

    private func isCJK(_ v: UInt32) -> Bool {
        (v >= 0x4E00 && v <= 0x9FFF) || (v >= 0x3400 && v <= 0x4DBF)
            || (v >= 0x20000 && v <= 0x2A6DF) || (v >= 0x2A700 && v <= 0x2B73F)
            || (v >= 0x2B740 && v <= 0x2B81F) || (v >= 0x2B820 && v <= 0x2CEAF)
            || (v >= 0xF900 && v <= 0xFAFF) || (v >= 0x2F800 && v <= 0x2FA1F)
    }

    private func preTokenize(_ s: String) -> [String] {
        var out: [String] = []
        var buf = ""
        let punct = CharacterSet.punctuationCharacters
        for ch in s {
            if ch == " " {
                if !buf.isEmpty { out.append(buf); buf = "" }
                continue
            }
            let isPunct = ch.unicodeScalars.allSatisfy { punct.contains($0) }
            let isHan = ch.unicodeScalars.count == 1 && isCJK(ch.unicodeScalars.first!.value)
            if isPunct || isHan {
                if !buf.isEmpty { out.append(buf); buf = "" }
                out.append(String(ch))
            } else {
                buf.append(ch)
            }
        }
        if !buf.isEmpty { out.append(buf) }
        return out
    }

    /// WordPiece 最长前缀匹配，词内后续片段带 ##；任一段找不到则整个词映射为 [UNK]。
    private func wordpiece(_ word: String) -> [Int32] {
        let chars = Array(word)
        var out: [Int32] = []
        var start = 0
        while start < chars.count {
            var end = chars.count
            var found: Int32?
            while start < end {
                let piece = (start > 0 ? "##" : "") + String(chars[start..<end])
                if let id = vocab[piece] { found = id; break }
                end -= 1
            }
            guard let f = found else { return [Int32(unkId)] }
            out.append(f)
            start = end
        }
        return out
    }
}
