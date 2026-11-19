// main.swift — 单句核心词提取 CLI + 100 句实测
import Foundation

// MARK: - 配置

struct Config {
    var model = "/Volumes/SSD/embedding&reranker/bge-small-zh-v1.5-ONNX/onnx/model_q4.onnx"
    var vocab = "/Volumes/SSD/embedding&reranker/bge-small-zh-v1.5-ONNX/tokenizer.json"
    var ortLib = ""
    var stopwords = ""
    var maxLen = 48
    var top = 3
    var batchSize = 64
    var threads = 4
    var windows = [2, 3, 4]
    var candidateLimit = 0
    var sentence = ""
    var inputFile = ""
    var demo = 0
    var report = ""
    var preset = "zh"
    var rank = "mmr"
    var mmrLambda = 0.7
    var probeCoreWords = false
    var selfTest = false
}

let defaultStopwords: String = {
    let sources = (#filePath as NSString).deletingLastPathComponent as NSString
    return sources.deletingLastPathComponent + "/Resources/stopwords_full.txt"
}()

func usage() {
    print("""
    SentenceKeywords — 无监督单句核心词提取（Swift + ONNX Runtime / BGE-micro-v2）

    用法: sentencekw "一句话" [选项]
          sentencekw --demo 100 [选项]        # 生成 100 句实测并统计
          sentencekw --file PATH              # 文件每行一句
          sentencekw --self-test

    流程: 清理 → k=2/3/4 滑窗 N-gram(去重/停用词，--windows 可改) → 备选集
          → 整句原文与每个备选词过 bge-small-zh ONNX embedding(mean pooling + L2)
          → MMR 去冗余排序取 top3

    选项:
      --preset NAME       模型预设: zh(bge-small-zh-v1.5，默认) | micro(bge-micro-v2)
                          设 --preset 后再给 --model/--vocab 可单独覆盖
      --rank MODE         排序: mmr(默认) | similarity
      --mmr-lambda X      MMR 相关性权重 (默认 0.7)
      --probe-corewords   探针：测试 CoreWordKit 纯统计方法在短句上的输出
      --model PATH        ONNX 模型 (默认 \(Config().model))
      --vocab PATH        vocab.txt
      --ort-lib PATH      libonnxruntime*.dylib（默认自动搜索）
      --stopwords PATH    停用词表（默认随工具附带）
      --max-len N         token 序列长度 (默认 48)
      --top N             核心词个数 (默认 3)
      --batch N           ONNX 推理批大小 (默认 64)
      --threads N         线程数 (默认 4)
      --windows LIST      滑窗长度，逗号分隔 (默认 2,3,4；可填 2,3 / 2,3,4,5 对比)
                          复现旧默认: --preset micro --rank similarity --windows 2,3
      --candidate-limit N 每句备选词上限，0=不限
      --report PATH       JSON 报告
      --help
    """)
}

func parseArgs() -> Config {
    var c = Config()
    let argv = CommandLine.arguments
    var i = 1
    func value() -> String? {
        guard i + 1 < argv.count else { return nil }
        i += 1
        return argv[i]
    }
    while i < argv.count {
        let a = argv[i]
        switch a {
        case "--model": c.model = value() ?? c.model
        case "--vocab": c.vocab = value() ?? c.vocab
        case "--ort-lib": c.ortLib = value() ?? ""
        case "--stopwords": c.stopwords = value() ?? ""
        case "--max-len": c.maxLen = max(8, Int(value() ?? "") ?? 48)
        case "--top": c.top = max(1, Int(value() ?? "") ?? 3)
        case "--batch": c.batchSize = max(1, Int(value() ?? "") ?? 64)
        case "--threads": c.threads = max(1, Int(value() ?? "") ?? 4)
        case "--windows":
            let ks = (value() ?? "2,3").split(separator: ",").compactMap { Int($0) }.filter { $0 >= 1 }
            if !ks.isEmpty { c.windows = Array(Set(ks)).sorted() }
        case "--candidate-limit": c.candidateLimit = max(0, Int(value() ?? "") ?? 0)
        case "--sentence": c.sentence = value() ?? ""
        case "--file": c.inputFile = value() ?? ""
        case "--demo": c.demo = max(0, Int(value() ?? "") ?? 100)
        case "--report": c.report = value() ?? ""
        case "--preset":
            let v = value() ?? "micro"
            c.preset = v
            if v == "zh" {
                let base = "/Volumes/SSD/embedding&reranker/bge-small-zh-v1.5-ONNX"
                c.model = base + "/onnx/model_q4.onnx"
                c.vocab = base + "/tokenizer.json"
            } else {
                c.model = "/Volumes/SSD/bge-micro-v2/onnx/model_quantized.onnx"
                c.vocab = "/Volumes/SSD/bge-micro-v2/vocab.txt"
            }
        case "--rank": c.rank = (value() ?? "similarity") == "mmr" ? "mmr" : "similarity"
        case "--mmr-lambda": c.mmrLambda = min(1, max(0, Double(value() ?? "") ?? 0.7))
        case "--probe-corewords": c.probeCoreWords = true
        case "--self-test": c.selfTest = true
        case "--help": usage(); exit(0)
        default:
            if a.hasPrefix("-") { FileHandle.standardError.write("未知选项 \(a)\n".data(using: .utf8)!); exit(2) }
            if c.sentence.isEmpty { c.sentence = a }
        }
        i += 1
    }
    if c.stopwords.isEmpty { c.stopwords = defaultStopwords }
    if c.ortLib.isEmpty { c.ortLib = findOrtDylib() ?? "" }
    return c
}

// MARK: - 报告结构

struct SentenceResult: Codable {
    let sentence: String
    let topic: String
    let candidateCount: Int
    let keywords: [KeywordHit]
    let exactHit: Bool
    let partialHit: Bool
}

struct DemoStats: Codable {
    let sentences: Int
    let avgCandidates: Double
    let top1Mean: Double
    let top2Mean: Double
    let top3Mean: Double
    let top3Median: Double
    let top3Min: Double
    let top3Max: Double
    let exactHitRate: Double
    let partialHitRate: Double
    let allCandidatesMean: Double
    let allCandidatesStd: Double
    let top1MinusMean: Double
    let avgUnkRate: Double
    let elapsedSeconds: Double
}

struct DemoReport: Codable {
    let model: String
    let windows: String
    let rank: String
    let mmrLambda: Double
    let maxLen: Int
    let top: Int
    let stats: DemoStats
    let results: [SentenceResult]
}

func writeReport<T: Encodable>(_ v: T, _ path: String) {
    guard !path.isEmpty else { return }
    do {
        let data = try JSONEncoder().encode(v)
        let url = URL(fileURLWithPath: path)
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                 withIntermediateDirectories: true)
        try data.write(to: url, options: .atomic)
        FileHandle.standardError.write("[skw] 报告已写 \(path)\n".data(using: .utf8)!)
    } catch { FileHandle.standardError.write("[skw] 报告写入失败 \(error)\n".data(using: .utf8)!) }
}

// MARK: - 100 句测试集

let topics = [
    "量子力学", "机器学习", "自然语言", "深度学习", "神经网络",
    "数据科学", "计算机视觉", "知识图谱", "语音识别", "推荐系统",
    "区块链", "云计算", "生物多样性", "气候变化", "宏观经济",
    "基因编辑", "天体物理", "古代文学", "现代艺术", "公共卫生",
]
let templates = [
    "%@是近年来发展迅速的研究领域",
    "研究人员正在深入探讨%@的核心原理",
    "这本专著系统介绍了%@的理论基础",
    "随着技术不断进步%@的应用场景持续拓展",
    "关于%@的最新成果已经发表在学术期刊上",
]

func makeSentences(_ n: Int) -> [(String, String)] {
    var out: [(String, String)] = []
    for t in topics {
        for tpl in templates {
            if out.count >= n { break }
            out.append((String(format: tpl, t), t))
        }
        if out.count >= n { break }
    }
    return out
}

// MARK: - CoreWordKit 短句探针

/// 验证「纯统计方法能否用于短句」：对 100 句分别跑默认阈值 / 放宽阈值，
/// 再与「100 句拼接成长文」对照，看统计是否因样本太短而失效。
func runCoreWordsProbe() -> Int32 {
    let items = makeSentences(100)
    print("CoreWordKit 纯统计方法在短句上的表现（\(items.count) 句，每句约 \(items[0].0.count) 字）\n")

    func summarize(_ label: String, _ make: () -> ExtractOptions) {
        var words = 0, zero = 0, ngram = 0, npmi = 0, entropy = 0, sw = 0, bd = 0
        var example: [String] = []
        for (sentence, _) in items {
            guard let r = try? CoreWordExtractor.extract(text: sentence, options: make()) else {
                zero += 1; continue
            }
            words += r.words.count
            if r.words.isEmpty { zero += 1 }
            ngram += r.stats.ngramTypes
            npmi += r.stats.npmiSurvivors
            entropy += r.stats.entropySurvivors
            sw += r.stats.stopwordRemoved
            bd += r.stats.boundaryRemoved
            if example.isEmpty { example = r.words.prefix(5).map { $0.word } }
        }
        let n = Double(items.count)
        print(String(format: "%@：N-gram 种类 %.1f | NPMI 通过 %.1f | 熵通过 %.1f | 停用词剔除 %.1f | 边界剔除 %.1f → 输出 %.2f 个词/句 | 空输出 %.0f%%",
                     label, Double(ngram)/n, Double(npmi)/n, Double(entropy)/n,
                     Double(sw)/n, Double(bd)/n, Double(words)/n, Double(zero)/n*100))
        print("   首句样例: \(example.isEmpty ? "（空）" : example.joined(separator: " / "))")
    }

    summarize("默认阈值  ") { var o = ExtractOptions(); o.top = 10; return o }
    summarize("放宽全部阈值") {
        var o = ExtractOptions()
        o.top = 10
        o.minNPMI = 0
        o.minLeftEntropy = 0
        o.minRightEntropy = 0
        o.useBoundaryFilter = false
        o.useSubstringFilter = false
        o.useStopwords = false
        return o
    }

    let long = items.map { $0.0 }.joined(separator: "\n")
    var o = ExtractOptions(); o.top = 20
    if let r = try? CoreWordExtractor.extract(text: long, options: o) {
        print(String(format: "\n对照：100 句拼成长文（%d 字）→ N-gram 种类 %d | NPMI 通过 %d | 熵通过 %d | 输出 %d 个词",
                     r.stats.totalChars, r.stats.ngramTypes, r.stats.npmiSurvivors,
                     r.stats.entropySurvivors, r.words.count))
        print("   前 8 个核心词: " + r.words.prefix(8).map { $0.word }.joined(separator: " / "))
    }
    return 0
}

// MARK: - 自检

func runSelfTest(_ cfg: Config, _ embedder: BGEEmbedder,
                 _ tokenizer: WordPieceTokenizer, _ stops: Set<String>) -> Int32 {
    var fail = 0
    func check(_ ok: Bool, _ name: String) {
        print(ok ? "✓ \(name)" : "FAIL: \(name)")
        if !ok { fail += 1 }
    }
    check(String(KeywordExtractor.clean("Hello, 世界！\n\tni Hao。")) == "hello世界nihao",
          "清理剔标点空白、英文转小写")
    let cand = KeywordExtractor.candidates("量子力学研究微观粒子", stops: stops,
                                           weak: KeywordExtractor.weakChars,
                                           windows: [2, 3])
    let defCfg = Config()
    check(defCfg.preset == "zh" && defCfg.rank == "mmr" && defCfg.windows == [2, 3, 4],
          "默认配置为 zh + MMR + 窗口 2,3,4")
    check(cand.contains("量子力") && cand.contains("力学") && !cand.contains("量子力学"),
          "指定窗口 [2,3] 时只生成 2、3 字子串")
    check(cand.allSatisfy({ $0.count == 2 || $0.count == 3 }), "备选词长度在指定窗口内")
    let cand2 = KeywordExtractor.candidates("量子力学研究微观粒子", stops: stops,
                                            weak: KeywordExtractor.weakChars, windows: [2])
    check(!cand2.isEmpty && cand2.allSatisfy({ $0.count == 2 }), "单窗口 k=2 生效")
    let cand234 = KeywordExtractor.candidates("量子力学研究微观粒子", stops: stops,
                                              weak: KeywordExtractor.weakChars,
                                              windows: [2, 3, 4])
    check(cand234.contains("量子力学"), "可配置窗口 2,3,4 生成 4-gram")
    check(Set(cand).count == cand.count, "备选集自动去重")
    check(tokenizer.hasToken("[CLS]") && tokenizer.hasToken("学"), "vocab 含 CLS 与汉字")
    let enc = tokenizer.encode("量子力学", maxLen: cfg.maxLen)
    check(enc.0.first == tokenizer.clsId && enc.0.count == cfg.maxLen
          && enc.1.filter({ $0 == 1 }).count == 6, "分词带 CLS/SEP 并 padding 到 maxLen")
    do {
        let v = try embedder.embed(batches: [tokenizer.encode("量子力学", maxLen: cfg.maxLen)],
                                   seqLen: cfg.maxLen, batchSize: 1)
        check(v[0].count == 384 || v[0].count == 512,
              "embedding 维度与模型一致（当前 \(v[0].count)）")
        let norm = sqrt(v[0].reduce(Float(0)) { $0 + $1 * $1 })
        check(abs(norm - 1) < 1e-3, "向量已 L2 归一化")
        let v2 = try embedder.embed(batches: [tokenizer.encode("量子力学", maxLen: cfg.maxLen)],
                                    seqLen: cfg.maxLen, batchSize: 1)
        check(v[0] == v2[0], "embedding 确定性")
        check(abs(KeywordExtractor.dot(v[0], v2[0]) - 1) < 1e-3, "自身余弦相似度 = 1")
        let trio = [
            "The cat sits on the mat",
            "A cat is sitting on a mat",
            "Quantum physics studies subatomic particles",
        ].map { tokenizer.encode($0, maxLen: cfg.maxLen) }
        let t = try embedder.embed(batches: trio, seqLen: cfg.maxLen, batchSize: 3)
        let near = KeywordExtractor.dot(t[0], t[1])
        let far = KeywordExtractor.dot(t[0], t[2])
        print(String(format: "  [参考] 英文近义 %.3f vs 无关 %.3f", near, far))
        check(near > far + 0.1, "embedding 有语义区分力（英文 sanity check）")
    } catch { check(false, "embedding 推理: \(error)") }
    print(fail == 0 ? "SentenceKeywords self-test: all checks passed"
                    : "SentenceKeywords self-test: \(fail) FAILED")
    return fail == 0 ? 0 : 1
}

// MARK: - 主流程

let cfg = parseArgs()
if cfg.probeCoreWords { exit(runCoreWordsProbe()) }
guard !cfg.ortLib.isEmpty else {
    FileHandle.standardError.write("[skw] 找不到 libonnxruntime dylib，请用 --ort-lib 指定\n".data(using: .utf8)!)
    exit(2)
}
let stops = KeywordExtractor.loadStopwords(cfg.stopwords)
let tokenizer: WordPieceTokenizer
let embedder: BGEEmbedder
do {
    tokenizer = try WordPieceTokenizer(vocabPath: cfg.vocab)
    embedder = try BGEEmbedder(modelPath: cfg.model, dylibPath: cfg.ortLib, threads: cfg.threads)
} catch {
    FileHandle.standardError.write("[skw] 初始化失败: \(error)\n".data(using: .utf8)!)
    exit(1)
}
FileHandle.standardError.write(
    "[skw] 模型 \(cfg.model)\n[skw] 滑窗 k=\(cfg.windows.map(String.init).joined(separator: ","))\n[skw] vocab \(tokenizer.vocabSize) | 停用词 \(stops.count)（\(cfg.stopwords)）| dylib \(cfg.ortLib)\n"
        .data(using: .utf8)!)

if cfg.selfTest { exit(runSelfTest(cfg, embedder, tokenizer, stops)) }

var globalAllSims: [Double] = []
var globalUnkRates: [Double] = []

/// 处理一批句子；demo 模式带主题真值。
func process(_ items: [(String, String)], quiet: Bool) -> ([SentenceResult], [Double], TimeInterval) {
    var results: [SentenceResult] = []
    var top3: [Double] = []
    var allSims: [Double] = []
    var unkRates: [Double] = []
    let t0 = Date()
    for (idx, item) in items.enumerated() {
        let (sentence, topic) = item
        do {
            let (ranked, n) = try KeywordExtractor.extract(
                sentence: sentence, embedder: embedder, tokenizer: tokenizer, stops: stops,
                maxLen: cfg.maxLen, batchSize: cfg.batchSize, windows: cfg.windows,
                rank: cfg.rank, mmrLambda: cfg.mmrLambda,
                candidateLimit: cfg.candidateLimit)
            let hits = Array(ranked.prefix(cfg.top))
            allSims.append(contentsOf: ranked.map { $0.similarity })
            let exact = hits.contains { $0.word == topic }
            let partial = hits.contains {
                $0.word.contains(topic) || (topic.contains($0.word) && $0.word.count >= 2)
            }
            let enc = tokenizer.encode(sentence, maxLen: cfg.maxLen)
            let valid = enc.1.reduce(0) { $0 + Int($1) }
            let unk = enc.0.filter { $0 == tokenizer.unkId }.count
            unkRates.append(Double(unk) / Double(max(1, valid)))
            let r = SentenceResult(sentence: sentence, topic: topic, candidateCount: n,
                                   keywords: hits, exactHit: exact, partialHit: partial)
            results.append(r)
            if let last = hits.last { top3.append(last.similarity) }
            if !quiet {
                let kw = hits.map { String(format: "%@(%.3f)", $0.word, $0.similarity) }
                    .joined(separator: " ")
                print(String(format: "%3d  [%@] 备选 %3d → %@", idx + 1, topic, n, kw))
            }
        } catch {
            FileHandle.standardError.write("[skw] 第 \(idx + 1) 句失败: \(error)\n".data(using: .utf8)!)
        }
        if quiet && (idx + 1) % 10 == 0 {
            FileHandle.standardError.write("[skw] 进度 \(idx + 1)/\(items.count)\n".data(using: .utf8)!)
        }
    }
    globalAllSims = allSims
    globalUnkRates = unkRates
    return (results, top3, Date().timeIntervalSince(t0))
}

if cfg.demo > 0 {
    let items = makeSentences(cfg.demo)
    print("生成 \(items.count) 句测试文本（\(topics.count) 个主题 × \(templates.count) 种句式）\n")
    let (results, top3Means, elapsed) = process(items, quiet: false)
    let sims = globalAllSims
    func mean(_ xs: [Double]) -> Double { xs.isEmpty ? 0 : xs.reduce(0, +) / Double(xs.count) }
    let allMean = mean(sims)
    let allStd = sims.isEmpty ? 0 : sqrt(mean(sims.map { ($0 - allMean) * ($0 - allMean) }))
    let sorted = top3Means.sorted()
    let stats = DemoStats(
        sentences: results.count,
        avgCandidates: mean(results.map { Double($0.candidateCount) }),
        top1Mean: mean(results.compactMap { $0.keywords.first?.similarity }),
        top2Mean: mean(results.compactMap { $0.keywords.count >= 2 ? $0.keywords[1].similarity : nil }),
        top3Mean: mean(results.compactMap { $0.keywords.last?.similarity }),
        top3Median: sorted.isEmpty ? 0 : sorted[sorted.count / 2],
        top3Min: sorted.first ?? 0, top3Max: sorted.last ?? 0,
        exactHitRate: mean(results.map { $0.exactHit ? 1.0 : 0.0 }),
        partialHitRate: mean(results.map { $0.partialHit ? 1.0 : 0.0 }),
        allCandidatesMean: allMean,
        allCandidatesStd: allStd,
        top1MinusMean: mean(results.compactMap { $0.keywords.first?.similarity }) - allMean,
        avgUnkRate: mean(globalUnkRates),
        elapsedSeconds: elapsed)
    print("\n=== 统计（\(results.count) 句）===")
    print(String(format: "平均备选词 %.1f 个/句 | 耗时 %.1fs", stats.avgCandidates, elapsed))
    print(String(format: "核心词与整句余弦相似度: top1 均值 %.4f | top2 %.4f | top3 %.4f",
                 stats.top1Mean, stats.top2Mean, stats.top3Mean))
    print(String(format: "top3 相似度: 中位 %.4f | 最低 %.4f | 最高 %.4f",
                 stats.top3Median, stats.top3Min, stats.top3Max))
    print(String(format: "全体 %d 个候选相似度: 均值 %.4f | 标准差 %.4f | top1 高出均值 %.4f",
                 sims.count, allMean, allStd, stats.top1MinusMean))
    print(String(format: "整句 token 的平均 [UNK] 比例: %.1f%%", stats.avgUnkRate * 100))
    print(String(format: "主题词命中率: 精确(top3含完整主题词) %.1f%% | 部分命中 %.1f%%",
                 stats.exactHitRate * 100, stats.partialHitRate * 100))
    writeReport(DemoReport(model: cfg.model, windows: cfg.windows.map(String.init).joined(separator: ","),
                           rank: cfg.rank, mmrLambda: cfg.mmrLambda,
                           maxLen: cfg.maxLen, top: cfg.top,
                           stats: stats, results: results), cfg.report)
    exit(0)
}

var sentences: [(String, String)] = []
if !cfg.inputFile.isEmpty {
    if let text = try? String(contentsOfFile: cfg.inputFile, encoding: .utf8) {
        sentences = text.split(separator: "\n").map { (String($0), "") }
    }
} else if !cfg.sentence.isEmpty {
    sentences = [(cfg.sentence, "")]
}
guard !sentences.isEmpty else { usage(); exit(2) }
let (results, _, _) = process(sentences, quiet: false)
writeReport(DemoReport(model: cfg.model, windows: cfg.windows.map(String.init).joined(separator: ","),
                       rank: cfg.rank, mmrLambda: cfg.mmrLambda,
                       maxLen: cfg.maxLen, top: cfg.top,
                       stats: DemoStats(sentences: results.count, avgCandidates: 0,
                                        top1Mean: 0, top2Mean: 0, top3Mean: 0,
                                        top3Median: 0, top3Min: 0, top3Max: 0,
                                        exactHitRate: 0, partialHitRate: 0,
                                        allCandidatesMean: 0, allCandidatesStd: 0,
                                        top1MinusMean: 0, avgUnkRate: 0,
                                        elapsedSeconds: 0),
                       results: results), cfg.report)
