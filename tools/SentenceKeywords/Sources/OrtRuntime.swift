// OrtRuntime.swift — 纯 Swift 调用 ONNX Runtime C API（运行时 dlopen，不链接 dylib、不用 Python）
// 针对 BGE-micro-v2 导出图：输入 input_ids / attention_mask / token_type_ids (int64)，
// 输出 last_hidden_state [batch, seq, H]，H 从张量维度动态读取（micro=384 / small-zh=512）。
// 按 attention_mask 做 mean pooling，再 L2 归一化，
// 使点积即为余弦相似度。
import COrt
import Foundation
import Darwin

enum OrtRuntimeError: Error, CustomStringConvertible {
    case library(String)
    case api(String)
    var description: String {
        switch self { case .library(let s), .api(let s): return s }
    }
}

/// 自动发现 libonnxruntime*.dylib（可用 --ort-lib 覆盖）。
func findOrtDylib() -> String? {
    let fm = FileManager.default
    var roots = ["/opt/homebrew/lib", "/usr/local/lib", "/usr/lib"]
    let home = NSHomeDirectory()
    roots.append(home + "/.pyenv/versions")
    roots.append(home + "/Library/Python")
    for root in roots {
        guard let en = fm.enumerator(atPath: root) else { continue }
        for case let sub as String in en {
            guard sub.hasSuffix(".dylib"),
                  (sub as NSString).lastPathComponent.hasPrefix("libonnxruntime") else { continue }
            let full = (root as NSString).appendingPathComponent(sub)
            if full.contains("capi") || root == "/opt/homebrew/lib" || root == "/usr/local/lib" {
                return full
            }
        }
    }
    return nil
}

final class BGEEmbedder {
    private let api: UnsafePointer<OrtApi>
    private var env: OpaquePointer?
    private var session: OpaquePointer?
    private var memoryInfo: OpaquePointer?
    private var hidden = 384
    private let inputNames = ["input_ids", "attention_mask", "token_type_ids"]
    private let outputName = "last_hidden_state"

    init(modelPath: String, dylibPath: String, threads: Int = 4) throws {
        guard let handle = dlopen(dylibPath, RTLD_NOW) else {
            throw OrtRuntimeError.library("dlopen 失败: \(dylibPath) — \(String(cString: dlerror()))")
        }
        guard let sym = dlsym(handle, "OrtGetApiBase") else {
            throw OrtRuntimeError.library("找不到符号 OrtGetApiBase")
        }
        typealias GetBaseFn = @convention(c) () -> UnsafePointer<OrtApiBase>?
        let getBase = unsafeBitCast(sym, to: GetBaseFn.self)
        guard let base = getBase(),
              let apiPtr = base.pointee.GetApi(UInt32(ORT_API_VERSION)) else {
            throw OrtRuntimeError.library("GetApi(ORT_API_VERSION) 失败")
        }
        api = apiPtr

        var envTmp: OpaquePointer?
        try ck { api.pointee.CreateEnv(ORT_LOGGING_LEVEL_WARNING, "sentencekw", &envTmp) }
        env = envTmp

        var opts: OpaquePointer?
        try ck { api.pointee.CreateSessionOptions(&opts) }
        try ck { api.pointee.SetIntraOpNumThreads(opts, Int32(threads)) }
        try ck { api.pointee.SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL) }

        var sess: OpaquePointer?
        try modelPath.withCString { cstr in
            try ck { api.pointee.CreateSession(env, cstr, opts, &sess) }
        }
        session = sess
        if let o = opts { api.pointee.ReleaseSessionOptions(o) }

        var mi: OpaquePointer?
        try ck { api.pointee.CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &mi) }
        memoryInfo = mi
    }

    deinit {
        if let session { api.pointee.ReleaseSession(session) }
        if let env { api.pointee.ReleaseEnv(env) }
        if let memoryInfo { api.pointee.ReleaseMemoryInfo(memoryInfo) }
    }

    /// 批量把已分词序列转成 L2 归一化的句向量（维度由模型输出动态决定）。
    func embed(batches: [([Int64], [Int64], [Int64])], seqLen: Int,
               batchSize: Int) throws -> [[Float]] {
        var result: [[Float]] = []
        for start in stride(from: 0, to: batches.count, by: batchSize) {
            let group = Array(batches[start..<min(start + batchSize, batches.count)])
            let b = group.count
            var ids = [Int64](), mask = [Int64](), types = [Int64]()
            ids.reserveCapacity(b * seqLen); mask.reserveCapacity(b * seqLen)
            types.reserveCapacity(b * seqLen)
            for (a, m, t) in group { ids.append(contentsOf: a); mask.append(contentsOf: m); types.append(contentsOf: t) }
            let shape: [Int64] = [Int64(b), Int64(seqLen)]
            let idVal = try makeTensor(&ids, shape)
            let maskVal = try makeTensor(&mask, shape)
            let typeVal = try makeTensor(&types, shape)
            let (raw, dims) = try run(idVal, maskVal, typeVal)
            if dims.count >= 3, dims[2] > 0 { hidden = Int(dims[2]) }
            api.pointee.ReleaseValue(idVal)
            api.pointee.ReleaseValue(maskVal)
            api.pointee.ReleaseValue(typeVal)
            for i in 0..<b {
                let m = Array(mask[i * seqLen..<(i + 1) * seqLen])
                result.append(pool(raw, batch: i, seqLen: seqLen, mask: m))
            }
        }
        return result
    }

    private func run(_ a: OpaquePointer, _ m: OpaquePointer,
                     _ t: OpaquePointer) throws -> ([Float], [Int64]) {
        var output: OpaquePointer?
        try withCStringArray(inputNames) { names in
            try withCStringArray([outputName]) { outNames in
                let values: [OpaquePointer?] = [a, m, t]
                var outs: [OpaquePointer?] = [nil]
                let outputCount = outs.count
                try values.withUnsafeBufferPointer { vb in
                    try outs.withUnsafeMutableBufferPointer { ob in
                        try ck {
                            api.pointee.Run(session, nil, names, vb.baseAddress!, values.count,
                                            outNames, outputCount, ob.baseAddress!)
                        }
                    }
                }
                output = outs[0]
            }
        }
        guard let value = output else { throw OrtRuntimeError.api("Run 无输出") }
        defer { api.pointee.ReleaseValue(value) }
        return try readFloats(value)
    }

    /// mean pooling over attention_mask=1 的 token + L2 归一化。
    private func pool(_ data: [Float], batch: Int, seqLen: Int, mask: [Int64]) -> [Float] {
        var acc = [Float](repeating: 0, count: hidden)
        let base = batch * seqLen * hidden
        var n: Float = 0
        for t in 0..<seqLen where mask[t] == 1 {
            for d in 0..<hidden { acc[d] += data[base + t * hidden + d] }
            n += 1
        }
        if n > 0 { for d in 0..<hidden { acc[d] /= n } }
        let norm = sqrt(acc.reduce(Float(0)) { $0 + $1 * $1 })
        if norm > 0 { for d in 0..<hidden { acc[d] /= norm } }
        return acc
    }

    private func makeTensor(_ data: inout [Int64], _ shape: [Int64]) throws -> OpaquePointer {
        var value: OpaquePointer?
        try data.withUnsafeMutableBytes { raw in
            try ck {
                api.pointee.CreateTensorWithDataAsOrtValue(
                    memoryInfo, raw.baseAddress!, raw.count,
                    shape, shape.count, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &value)
            }
        }
        guard let v = value else { throw OrtRuntimeError.api("CreateTensor 失败") }
        return v
    }

    private func readFloats(_ value: OpaquePointer) throws -> ([Float], [Int64]) {
        var typeInfo: OpaquePointer?
        try ck { api.pointee.GetTypeInfo(value, &typeInfo) }
        defer { api.pointee.ReleaseTypeInfo(typeInfo) }
        var tensorInfo: OpaquePointer?
        try ck { api.pointee.CastTypeInfoToTensorInfo(typeInfo, &tensorInfo) }
        var count: Int = 0
        try ck { api.pointee.GetDimensionsCount(tensorInfo, &count) }
        var dims = [Int64](repeating: 0, count: count)
        try ck { api.pointee.GetDimensions(tensorInfo, &dims, count) }
        let total = dims.reduce(Int64(1), *)
        var ptr: UnsafeMutableRawPointer?
        try ck { api.pointee.GetTensorMutableData(value, &ptr) }
        guard let p = ptr else { throw OrtRuntimeError.api("GetTensorMutableData 失败") }
        return (Array(UnsafeBufferPointer(start: p.assumingMemoryBound(to: Float.self),
                                          count: Int(total))), dims)
    }

    private func ck(_ body: () -> OpaquePointer?) throws {
        if let status = body() {
            defer { api.pointee.ReleaseStatus(status) }
            let msg = api.pointee.GetErrorMessage(status).map { String(cString: $0) } ?? "unknown"
            throw OrtRuntimeError.api(msg)
        }
    }
}

private func withCStringArray(_ strings: [String],
                              _ body: (UnsafePointer<UnsafePointer<CChar>?>) throws -> Void) rethrows {
    let ptrs = strings.map { strdup($0)! }
    defer { for p in ptrs { free(p) } }
    let nullable: [UnsafePointer<CChar>?] = ptrs.map { UnsafePointer<CChar>($0) }
    try nullable.withUnsafeBufferPointer { buf in
        try body(UnsafePointer(buf.baseAddress!))
    }
}
