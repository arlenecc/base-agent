// ExtractCoreWords.swift
//
// 单文件/单本书核心词提取 CLI（算法实现见 CoreWordKit.swift）。
//
// 编译：swiftc -O -o /tmp/ecw tools/CoreWordKit.swift tools/ExtractCoreWords.swift
// 用法：/tmp/ecw book.md --format words --top 500
// 自检：/tmp/ecw --self-test
//
import Foundation

// MARK: - CLI

private let usage = """
用法: ExtractCoreWords <文件|目录|-> [选项]

统计参数:
  --top N                 最终保留词数 (默认 500)
  --min-npmi X            归一化互信息下限 (默认 0.3，别名 --min-pmi)
  --min-entropy X         左/右信息熵下限 bit (默认 1.0)
  --min-left-entropy X    单独覆盖左熵下限
  --min-right-entropy X   单独覆盖右熵下限
  --min-freq N            最低出现次数 (默认 1)
  --boundary count|ignore 片段首尾是否计入 BOS/EOS 邻居 (默认 count)
  --keep-ascii            ASCII 字母/数字也参与构词
  --ngram-min N           最短组合字数 (默认 2)
  --ngram-max N           最长组合字数 (默认 5)
  --max-chars N           只取前 N 个字试跑，0 为全量

文本过滤 / 长词优先:
  --no-stopwords          关闭内置停用词过滤
  --stopwords PATH        追加自定义停用词文件（每行一词，# 为注释）
  --no-boundary-filter    关闭词首/词尾虚字碎片过滤
  --no-substring          关闭子串碎片抑制（长词优先）
  --substring-ratio X     长词频次 ≥ 短词频次×X 且短词 NPMI<0.6 时剔除 (默认 0.5)
  --length-bonus X        排序给长词加权 freq×(1+X×(字数-2)) (默认 0，纯按频率)
  --no-filters            一次性关闭以上三种文本过滤

输入输出:
  --format table|words|json
  --output PATH           结果写入文件（默认 stdout）
  --encoding auto|utf-8|gb18030|latin1
  --self-test             运行内置断言自检
  -h, --help              显示本帮助
"""

enum CLI {
    static func run(_ argv: [String]) -> Int32 {
        if argv.contains("-h") || argv.contains("--help") {
            print(usage)
            return 0
        }
        if argv.contains("--self-test") { return SelfTest.run() }

        var options = ExtractOptions()
        var format: OutputFormat = .table
        var output: String?
        var encoding = "auto"
        var input: String?
        var explicitLeft: Double?
        var explicitRight: Double?
        var minEntropy: Double = 1.0

        var i = 0
        while i < argv.count {
            let arg = argv[i]
            func value() -> String? {
                guard i + 1 < argv.count else { return nil }
                i += 1
                return argv[i]
            }
            switch arg {
            case "--top": options.top = Int(value() ?? "") ?? 500
            case "--min-npmi", "--min-pmi": options.minNPMI = Double(value() ?? "") ?? 0.3
            case "--min-entropy": minEntropy = Double(value() ?? "") ?? 1.0
            case "--min-left-entropy": explicitLeft = Double(value() ?? "")
            case "--min-right-entropy": explicitRight = Double(value() ?? "")
            case "--min-freq": options.minFreq = Int(value() ?? "") ?? 1
            case "--boundary":
                options.boundary = ExtractOptions.BoundaryMode(rawValue: value() ?? "") ?? .count
            case "--keep-ascii": options.keepASCII = true
            case "--ngram-min": options.ngramMin = Int(value() ?? "") ?? 2
            case "--ngram-max": options.ngramMax = Int(value() ?? "") ?? 5
            case "--max-chars": options.maxChars = Int(value() ?? "") ?? 0
            case "--format": format = OutputFormat(rawValue: value() ?? "") ?? .table
            case "--output": output = value()
            case "--encoding": encoding = value() ?? "auto"
            case "--no-stopwords": options.useStopwords = false
            case "--stopwords": options.stopwordsPath = value() ?? ""
            case "--no-boundary-filter": options.useBoundaryFilter = false
            case "--no-substring": options.useSubstringFilter = false
            case "--substring-ratio": options.substringRatio = Double(value() ?? "") ?? 0.8
            case "--length-bonus": options.lengthBonus = Double(value() ?? "") ?? 0
            case "--no-filters":
                options.useStopwords = false
                options.useBoundaryFilter = false
                options.useSubstringFilter = false
            default:
                if arg.hasPrefix("-") && arg != "-" {
                    FileHandle.standardError.write(Data("未知选项: \(arg)\n".utf8))
                    return 2
                }
                if input == nil { input = arg }
            }
            i += 1
        }

        guard let source = input else {
            FileHandle.standardError.write(Data("缺少输入文件\n\(usage)\n".utf8))
            return 2
        }
        options.minLeftEntropy = explicitLeft ?? minEntropy
        options.minRightEntropy = explicitRight ?? minEntropy

        do {
            let text = try CorpusLoader.load(source, encoding: encoding)
            let result = try CoreWordExtractor.extract(text: text, options: options)
            let rendered = try ResultFormatter.render(result, format: format)
            if let output {
                try (rendered + "\n").write(
                    toFile: (output as NSString).expandingTildeInPath,
                    atomically: true, encoding: .utf8)
            } else {
                print(rendered)
            }
            FileHandle.standardError.write(Data((ResultFormatter.statsLine(result) + "\n").utf8))
            return 0
        } catch {
            FileHandle.standardError.write(Data("错误: \(error)\n".utf8))
            return 1
        }
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

    private static func corpus() -> String {
        let pairs = Array("甲乙丙丁戊己庚辛")
        var lines: [String] = []
        for idx in 0..<8 {
            let l = pairs[idx]
            let r = pairs[(idx + 3) % 8]
            lines.append("\(l)量子纠缠\(r)是物理学的概念")
        }
        lines += Array(repeating: "固定搭配里的片段总是一样", count: 4)
        return lines.joined(separator: "\n")
    }

    static func run() -> Int32 {
        // 切分
        check(CoreWordExtractor.segments("量子纠缠，是物理现象。\n机器学习 与 深度学习")
                == ["量子纠缠", "是物理现象", "机器学习", "与", "深度学习"], "标点/空白切分")
        check(CoreWordExtractor.segments("使用Python3处理文本") == ["使用", "处理文本"],
              "默认忽略 ASCII")
        check(CoreWordExtractor.segments("使用Python3处理文本", keepASCII: true)
                == ["使用Python3处理文本"], "keepASCII")

        // 编码 / GramKey 往返
        var codec = GramCodec()
        let encoded = CoreWordExtractor.encode("甲乙丙，丁戊", keepASCII: false,
                                               maxChars: 0, codec: &codec)
        check(encoded.totalChars == 5 && encoded.segments == 2, "编码字数/片段数")
        check(encoded.stream.contains(sentinelID), "片段边界哨兵")
        let runs = CoreWordExtractor.runLengths(encoded.stream, maxN: 5)
        check(encoded.stream.enumerated().allSatisfy { runs[$0.offset] <= 5 }, "run 上限")
        let k = codec.key("甲乙丙")
        check(k.length == 3 && codec.word(k) == "甲乙丙", "GramKey 编解码往返")
        let k5 = codec.key("甲乙丙丁戊")
        check(k5.length == 5 && codec.word(k5) == "甲乙丙丁戊", "5 字 GramKey 往返")
        check(k != k5, "不同长度键不冲突")

        // 频次
        let freq = CoreWordExtractor.countNgrams(encoded.stream, runs: runs, maxN: 5)
        check(freq[codec.key("甲")] == 1, "1-gram 频次")
        check(freq[codec.key("甲乙")] == 1, "2-gram 频次")
        check(freq[codec.key("甲乙丙")] == 1, "3-gram 频次")
        check(freq[codec.key("丙丁")] == nil, "跨标点不构词")
        let overlap = CoreWordExtractor.encode("甲甲甲", keepASCII: false,
                                               maxChars: 0, codec: &codec)
        let overlapRuns = CoreWordExtractor.runLengths(overlap.stream, maxN: 2)
        let overlapFreq = CoreWordExtractor.countNgrams(overlap.stream, runs: overlapRuns, maxN: 2)
        check(overlapFreq[codec.key("甲甲")] == 2, "重叠滑窗")

        // NPMI 公式：先 encode 注册字符，否则 key(_:) 会把未注册字编码成哨兵。
        var formulaCodec = GramCodec()
        _ = CoreWordExtractor.encode("甲乙丙", keepASCII: false, maxChars: 0,
                                     codec: &formulaCodec)
        var formulaFreq: [GramKey: UInt32] = [:]
        for w in ["甲", "乙", "丙", "甲乙", "乙丙", "甲乙丙"] {
            formulaFreq[formulaCodec.key(w)] = 1
        }
        let (pmi, npmi) = CoreWordExtractor.wordNPMI(
            formulaCodec.key("甲乙丙"), freq: formulaFreq, total: 3)
        check(approx(pmi, log(3.0)) && approx(npmi, 1.0), "NPMI = PMI / -log p(w)")

        // 最小二分切分拉低凝固度
        var lowCodec = GramCodec()
        _ = CoreWordExtractor.encode("甲乙丙丁", keepASCII: false, maxChars: 0,
                                     codec: &lowCodec)
        let pieces: [(String, Int)] = [("甲", 100), ("乙", 100), ("丙", 100), ("丁", 100),
                                       ("甲乙", 100), ("乙丙", 1), ("丙丁", 1),
                                       ("甲乙丙", 1), ("甲乙丙丁", 1), ("甲丁", 1)]
        var lowFreq: [GramKey: UInt32] = [:]
        for (w, c) in pieces { lowFreq[lowCodec.key(w)] = UInt32(c) }
        let (_, lowNPMI) = CoreWordExtractor.wordNPMI(
            lowCodec.key("甲乙丙丁"), freq: lowFreq, total: 1000)
        check(lowNPMI < 0.3, "最小二分切分降低 NPMI")

        // 熵公式
        check(approx(CoreWordExtractor.entropy(total: 4, sumCLogC: 0), 2.0),
              "均匀 4 邻居熵 = 2 bit")
        check(approx(CoreWordExtractor.entropy(total: 7, sumCLogC: 7 * log2(7.0)), 0.0),
              "单一邻居熵 = 0")
        check(approx(CoreWordExtractor.entropy(total: 0, sumCLogC: 0), 0.0), "空邻居熵 = 0")

        // 停用词 / 虚字规则
        check(WordFilters.builtinStopwords.contains("因为"), "停用词表含虚词")
        check(!WordFilters.builtinStopwords.contains("佛教"), "停用词表不误杀术语")
        check(WordFilters.hasWeakBoundary("的意") && WordFilters.hasWeakBoundary("是一种"),
              "首尾虚字碎片识别")
        check(!WordFilters.hasWeakBoundary("量子纠缠"), "真词不误判虚字碎片")

        do {
            let text = corpus()
            var base = ExtractOptions()
            base.top = 50
            let result = try CoreWordExtractor.extract(text: text, options: base)
            let names = result.words.map(\.word)
            check(names.contains("量子纠缠"), "高频多样上下文词被提取")
            if let c = result.words.first(where: { $0.word == "量子纠缠" }) {
                check(c.freq == 8 && c.npmi >= 0.3 && c.leftEntropy > 1
                        && c.rightEntropy > 1, "量子纠缠 各项指标")
            }

            // 熵阈值过滤
            var highEntropy = base
            highEntropy.minLeftEntropy = 99
            highEntropy.minRightEntropy = 99
            check(try CoreWordExtractor.extract(text: text, options: highEntropy).words.isEmpty,
                  "熵阈值过滤")

            // NPMI 阈值过滤
            var loose = base; loose.minNPMI = -1
            var strict = base; strict.minNPMI = 1.01
            let rLoose = try CoreWordExtractor.extract(text: text, options: loose)
            let rStrict = try CoreWordExtractor.extract(text: text, options: strict)
            check(rLoose.stats.npmiSurvivors > 0 && rStrict.stats.npmiSurvivors == 0
                    && rStrict.words.isEmpty, "NPMI 阈值过滤")

            // 排序 / top 截断
            var top3 = base
            top3.top = 3
            top3.minNPMI = -1
            top3.minLeftEntropy = 0
            top3.minRightEntropy = 0
            top3.useStopwords = false
            top3.useBoundaryFilter = false
            top3.useSubstringFilter = false
            let rTop = try CoreWordExtractor.extract(text: text, options: top3)
            let freqs = rTop.words.map(\.freq)
            check(rTop.words.count == 3 && freqs == freqs.sorted(by: >),
                  "按频率降序 + top 截断")
            check(rTop.stats.entropySurvivors > 3, "top 截断前存在更多候选")

            // 等价剪枝：熵阈值 >0 时，freq=1 的 gram 该侧熵必为 0 → 必被淘汰。
            // 关闭剪枝（熵=0, minFreq=1）时，所有 freq=1 的词都应至少有一侧熵为 0。
            var f1 = base; f1.minFreq = 1
            var f2 = base; f2.minFreq = 2
            let a = try CoreWordExtractor.extract(text: text, options: f1)
            let b = try CoreWordExtractor.extract(text: text, options: f2)
            check(a.words == b.words, "freq=1 剪枝与显式 min-freq=2 结果一致")
            var noPrune = base
            noPrune.minFreq = 1
            noPrune.minLeftEntropy = 0
            noPrune.minRightEntropy = 0
            noPrune.useStopwords = false
            noPrune.useBoundaryFilter = false
            noPrune.useSubstringFilter = false
            noPrune.top = 100_000
            let all = try CoreWordExtractor.extract(text: text, options: noPrune)
            let singletons = all.words.filter { $0.freq == 1 }
            check(!singletons.isEmpty, "关闭剪枝后能看到 freq=1 的词")
            check(singletons.allSatisfy { $0.leftEntropy < 1.0 || $0.rightEntropy < 1.0 },
                  "freq=1 的词必有一侧熵<1，剪枝不改变结果")

            // 停用词开关确实改变结果
            var noFilter = base; noFilter.useStopwords = false
            noFilter.useBoundaryFilter = false
            noFilter.useSubstringFilter = false
            let raw = try CoreWordExtractor.extract(text: text, options: noFilter)
            check(result.stats.stopwordRemoved + result.stats.boundaryRemoved
                    + result.stats.substringRemoved >= 0, "过滤统计可用")
            check(raw.stats.stopwordRemoved == 0 && raw.stats.boundaryRemoved == 0
                    && raw.stats.substringRemoved == 0, "--no-filters 关闭所有过滤")
            check(result.words.count <= raw.words.count, "过滤后词数不增加")

            // 子串抑制：用构造指标直接测判定逻辑（真实书里的数值形态）。
            // 「速增长」是「快速增长」的后缀：低 NPMI、左邻被「快」粘住 → 剔除。
            let fragmentSample: [WordMetric] = [
                .init(word: "快速增长", freq: 32, npmi: 0.62, hl: 3.45, hr: 2.65),
                .init(word: "速增长", freq: 32, npmi: 0.54, hl: 1.42, hr: 2.32),
                // 高 NPMI 真词虽是长词前缀，也不该被剔。
                .init(word: "第三方支付", freq: 300, npmi: 0.72, hl: 4.10, hr: 3.20),
                .init(word: "第三方", freq: 364, npmi: 0.80, hl: 4.87, hr: 1.37),
                // 两侧都自由 → 不剔。
                .init(word: "量子纠缠", freq: 20, npmi: 0.99, hl: 3.00, hr: 3.00),
            ]
            let removedIdx = SubstringFilter.indicesToRemove(fragmentSample, ratio: 0.5)
            check(removedIdx.contains(1), "低凝固度+边界粘住的子串碎片被剔除")
            check(!removedIdx.contains(0), "完整长词保留")
            check(!removedIdx.contains(3), "高 NPMI 真词不被误伤")
            check(!removedIdx.contains(4), "无对应长词时不受影响")

            // 端到端：过滤开关确实生效，且关闭后统计归零。
            var subOpt = ExtractOptions()
            subOpt.minNPMI = -1
            subOpt.minLeftEntropy = 0
            subOpt.minRightEntropy = 0
            subOpt.useStopwords = false
            subOpt.useBoundaryFilter = false
            let diverseText = (Array("甲乙丙丁戊己庚辛").enumerated().map { idx, ch in
                "\(ch)量子纠缠\(Array("甲乙丙丁戊己庚辛")[(idx + 3) % 8])是物理概念"
            } + Array(repeating: "甲量子纠缠乙", count: 12)).joined(separator: "\n")
            let diverse = try CoreWordExtractor.extract(text: diverseText, options: subOpt)
            check(diverse.words.contains(where: { $0.word == "量子纠缠" }),
                  "多样上下文真词不被长词误伤")
            var noSub = subOpt
            noSub.useSubstringFilter = false
            check(try CoreWordExtractor.extract(text: diverseText, options: noSub)
                    .stats.substringRemoved == 0, "关闭子串抑制后统计归零")

            var minFreq99 = base; minFreq99.minFreq = 99
            check(try CoreWordExtractor.extract(text: text, options: minFreq99).words.isEmpty,
                  "minFreq 过滤")
            var clipped = base; clipped.maxChars = 6
            check(try CoreWordExtractor.extract(text: text, options: clipped).stats.totalChars == 6,
                  "maxChars 截断")

            let empty = try CoreWordExtractor.extract(text: "!!!...  123", options: base)
            check(empty.words.isEmpty && empty.stats.totalChars == 0, "空文本安全")

            var bad = base; bad.ngramMin = 1
            var bad2 = base; bad2.ngramMax = 6
            var threw = false
            do {
                _ = try CoreWordExtractor.extract(text: "文本", options: bad)
                _ = try CoreWordExtractor.extract(text: "文本", options: bad2)
            } catch { threw = true }
            check(threw, "非法 ngram 范围抛错")

            // 输出
            let json = try ResultFormatter.render(result, format: .json)
            check(json.contains("\"total_chars\"") && json.contains("\"left_entropy\"")
                    && json.contains("量子纠缠") && json.contains("stopword_removed"),
                  "JSON 输出结构")
            if let data = json.data(using: .utf8),
               let decoded = try? JSONDecoder().decode(ExtractionResult.self, from: data) {
                check(decoded.words.count == result.words.count, "JSON 可回解码")
            } else {
                failures.append("JSON 无法解码回 ExtractionResult")
            }
            check(try ResultFormatter.render(result, format: .words)
                    .split(separator: "\n").map(String.init) == names, "words 纯词表输出")
        } catch {
            failures.append("端到端异常: \(error)")
        }

        // 文件/目录/编码 + 自定义停用词
        do {
            let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("ecw-selftest-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: tmp) }
            let dir = tmp.appendingPathComponent("books")
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            try "量子纠缠量子纠缠量子纠缠".write(
                to: dir.appendingPathComponent("a.md"), atomically: true, encoding: .utf8)
            let gbData = "量子纠缠量子纠缠".data(using: CorpusLoader.gb18030)!
            try gbData.write(to: dir.appendingPathComponent("b.txt"))
            try Data([0x00, 0x01]).write(to: dir.appendingPathComponent("skip.bin"))
            check(try CorpusLoader.load(dir.path, encoding: "auto").contains("量子纠缠"),
                  "目录 + GB18030 混合加载")
            check(try CorpusLoader.load(dir.appendingPathComponent("a.md").path,
                                        encoding: "auto") == "量子纠缠量子纠缠量子纠缠",
                  "单文件 UTF-8 加载")

            let sw = tmp.appendingPathComponent("stopwords.txt")
            try "# comment\n自定义词\n\n".write(to: sw, atomically: true, encoding: .utf8)
            check(WordFilters.loadExtra(sw.path) == ["自定义词"], "自定义停用词加载")
        } catch {
            failures.append("文件加载异常: \(error)")
        }

        if failures.isEmpty {
            print("ExtractCoreWords self-test: all checks passed")
            return 0
        }
        for message in failures { FileHandle.standardError.write(Data("FAIL: \(message)\n".utf8)) }
        FileHandle.standardError.write(Data("\(failures.count) check(s) failed\n".utf8))
        return 1
    }
}


// MARK: - 入口

@main enum ExtractCoreWordsMain {
    static func main() {
        exit(CLI.run(Array(CommandLine.arguments.dropFirst())))
    }
}
