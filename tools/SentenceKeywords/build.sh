#!/bin/bash
# 编译 SentenceKeywords（纯 Swift + ONNX Runtime C API，运行时 dlopen 动态库）
# 用绝对路径传源文件，使 #filePath 能定位到随工具附带的停用词表。
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-/tmp/sentencekw}"
swiftc -O -I "$ROOT/cort" \
    "$ROOT/../CoreWordKit.swift" \
    "$ROOT/Sources/OrtRuntime.swift" \
    "$ROOT/Sources/WordPieceTokenizer.swift" \
    "$ROOT/Sources/KeywordExtractor.swift" \
    "$ROOT/Sources/main.swift" \
    -o "$OUT"
echo "已生成 $OUT"
echo "运行: $OUT --self-test   |   $OUT --demo 100 --report /tmp/skw_report.json"
