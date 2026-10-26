"""切片内容保真测试：确保分片不丢失原文（无真实内容间隔、token 不超 bge 窗口）。

背景：修复 bge 512 窗口截断时加了 token 预算收口 + 500 字硬上限，并用
``_split_by_chars`` 兜底。本测试确保这些「硬切」操作只是把原文切成子串，绝不
跳过任何非空白字符；并在所有路径（含语义/混合分片在退化 EF 下漏切）下，经
``chunk_documents`` 的覆盖性兜底回退到确定性递归分片后，输出必然覆盖全文。

注意：必须用「非重复文本 + 逼真 EF」测试——高度重复文本会让 ``str.find`` 定位歧义，
退化（全零/前 8 字）EF 会让语义分片器产出垃圾切片，二者都是测量伪影而非真实丢内容。
"""
import hashlib

import pytest

from agent.rag.chunker import chunk_documents, _covers_source, estimate_tokens


class _HashEF:
    """全文本 n-gram 哈希向量：不同文本向量不同、相似文本相近，逼近平真 EF，
    避免全零/截断 EF 让语义分片器退化产生伪影。"""

    def embed_documents(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 64
            for i in range(0, len(t), 3):
                h = int.from_bytes(
                    hashlib.md5(t[i:i + 3].encode()).digest()[:4], "little"
                ) % 64
                v[h] += 1.0
            out.append(v)
        return out

    def __call__(self, texts):
        return self.embed_documents(texts)

    def name(self):
        return "hash"


def _nonrep_texts():
    en = "".join(
        f"Sentence {i}: Natural language processing enables computers to understand "
        f"human language. Retrieval augmented generation combines search with "
        f"generation to ground answers in private documents. "
        for i in range(60)
    )
    zh = "".join(
        f"第{i}章：人工智能是计算机科学的一个分支，研究如何让机器理解并处理人类的自然"
        f"语言与知识，进而支撑检索增强生成的工程实践。"
        for i in range(400)
    )
    code = "".join(
        f"def func_{i}(x):\n    return x * {i}  # 第{i}个工具函数\n" for i in range(60)
    )
    url = "".join(
        f"详见 https://example.com/docs?id={i}&page=2 相关内容说明与索引说明条目。 "
        for i in range(100)
    )
    mixed = zh[:5000] + en[-800:]
    return {
        "中文": zh,
        "英文": en,
        "代码": code,
        "URL": url,
        "混合": mixed,
    }


@pytest.mark.parametrize("name", ["中文", "英文", "代码", "URL", "混合"])
def test_chunking_preserves_content(name):
    text = _nonrep_texts()[name]
    chunks = chunk_documents(
        [{"source": "t.txt", "text": text}],
        chunk_size=500, chunk_overlap=50, min_chunk_size=100, overlap_percent=0.10,
        embedding_function=_HashEF(),
    )
    assert chunks, name
    # 覆盖性兜底保证：输出必覆盖原文所有非空白字符（无真实内容间隔）。
    assert _covers_source(chunks, text), f"{name}: 切片未覆盖全文，疑似丢内容"
    # token 不超窗：每片估算 ≤ chunk_size（bge=510）⇒ 真实 ≤ 512，嵌入不截断。
    for c in chunks:
        assert estimate_tokens(c["text"]) <= 500, f"{name}: 切片超 token 预算"


def test_covers_source_detects_real_gap():
    src = "abcdefghijklmnopqrstuvwxyz0123456789"
    # 丢失中间 "mnop" → 应判未覆盖
    chunks = [
        {"text": "abcdefghijkl"},
        {"text": "qrstuvwxyz0123456789"},
    ]
    assert not _covers_source(chunks, src)


def test_covers_source_allows_whitespace_only_gap():
    # 边界空白被 .strip() 掉属正常，不计为丢失
    src = "hello world foo bar"
    chunks = [{"text": "hello"}, {"text": "world foo bar"}]
    assert _covers_source(chunks, src)


def test_degenerate_ef_falls_back_to_lossless():
    # 全零 EF 会让语义分片器退化漏切；覆盖性兜底应回退递归分片，仍不丢内容。
    class _ZeroEF:
        def embed_documents(self, texts):
            return [[0.0] * 8 for _ in texts]

        def __call__(self, texts):
            return self.embed_documents(texts)

        def name(self):
            return "zero"

    zh = "".join(
        f"第{i}章：人工智能是计算机科学的一个分支，研究如何让机器理解并处理人类的自然"
        f"语言与知识，进而支撑检索增强生成的工程实践。"
        for i in range(200)
    )
    chunks = chunk_documents(
        [{"source": "t.txt", "text": zh}],
        chunk_size=500, chunk_overlap=50, min_chunk_size=100, overlap_percent=0.10,
        embedding_function=_ZeroEF(),
    )
    assert _covers_source(chunks, zh), "退化 EF 下兜底应保证不丢内容"
