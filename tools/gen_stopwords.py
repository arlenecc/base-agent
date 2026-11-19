#!/usr/bin/env python3
"""从 tools/stopwords_full.txt 重新生成 CoreWordKit.swift 中的编译期停用词数据段。

编辑停用词请改数据文件，然后运行：python3 tools/gen_stopwords.py
生成后需重新编译各工具（swiftc -O ... tools/CoreWordKit.swift ...）。
"""
import io
import re
import sys
from pathlib import Path

base = Path(__file__).resolve().parent
data_file = base / "stopwords_full.txt"
kit_file = base / "CoreWordKit.swift"

words = [line.strip() for line in
         io.open(data_file, encoding="utf-8").read().splitlines() if line.strip()]
body = "\n".join(words)
if '"""' in body or "\\" in body or "\r" in body:
    sys.exit("词表含 Swift 多行字符串无法安全嵌入的字符（三连引号/反斜杠/CR）")

text = io.open(kit_file, encoding="utf-8").read()
pattern = re.compile(r'(static let stopwordsFullRaw: String = """\n).*?(\n""")', re.S)
if not pattern.search(text):
    sys.exit("未在 CoreWordKit.swift 找到 stopwordsFullRaw 数据段")

updated = pattern.sub(lambda m: m.group(1) + body + m.group(2), text, count=1)
if updated != text:
    with io.open(kit_file, "w", encoding="utf-8") as f:
        _ = f.write(updated)
    print("已同步 %d 条停用词到 CoreWordKit.swift" % len(words))
else:
    print("已同步 %d 条停用词（内容无变化）" % len(words))
