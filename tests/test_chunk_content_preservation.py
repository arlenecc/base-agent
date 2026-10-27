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


def _assert_strictly_lossless(chunks, source):
    """比 _covers_source 更严：每片必须是原文子串，且顺序定位后覆盖所有非空白字符。

    ``_covers_source`` 对重复文本有容错（区间文本出现在任一分片里即算覆盖），
    这里不放宽——专测「重复代码块导致定位错位、中间内容被吞」这类真实丢失。
    分片带重叠，定位起点允许落在上一片之内（回看 ≤ len(chunk)）。
    """
    covered = bytearray(len(source))
    cursor = 0
    for c in chunks:
        t = c["text"]
        assert source.find(t) >= 0, f"切片不是原文子串: {t[:40]!r}"
        p = source.find(t, max(0, cursor - len(t)))
        assert p >= 0 and p <= cursor, f"切片顺序与原文不一致: {t[:40]!r}"
        covered[p:p + len(t)] = b"\x01" * len(t)
        cursor = max(cursor, p + len(t))
    gap = [i for i, ch in enumerate(source) if not covered[i] and not ch.isspace()]
    assert not gap, f"内容丢失 @{gap[0]}"


def test_duplicate_blocks_do_not_swallow_content():
    """书里重复的代码块/样板行不应让中间的正文从所有切片里消失。

    背景：合并/重叠用 ``str.find`` 定位时命中更早的同名实例，会把不相邻的两段
    拼成一个 chunk——既不是原文子串，中间那段真实内容也再没被任何切片覆盖
    （真实 EPUB 实测丢 0.5% 正文）。修复后按原文跨度重建，结果必须严格无损。
    """
    block = "function demo(){\n  return 42;//示例代码\n}\n\n"
    body = "".join(
        f"第{i}节：JavaScript 的作用域链与闭包，函数会记住它定义时的作用域环境。"
        for i in range(60)
    )
    doc = block + body + block + body + block

    chunks = chunk_documents(
        [{"source": "dup.txt", "text": doc}],
        chunk_size=500, chunk_overlap=50, min_chunk_size=100, overlap_percent=0.10,
        embedding_function=_HashEF(),
    )
    assert chunks
    for c in chunks:
        assert estimate_tokens(c["text"]) <= 500, "切片超 token 预算"
    _assert_strictly_lossless(chunks, doc)


def test_oversized_single_sentence_is_split_not_passed_through():
    """chonkie 不切「单句超预算」，语义路径必须补一刀（EPUB 里有 7000+ 字符长行）。"""
    pytest.importorskip("chonkie")
    from agent.rag.semantic_chunker import _split_oversized

    giant = "x" * 5000  # 无标点、无换行 → chonkie 会当成一个句子
    pieces = _split_oversized([giant], 500)
    assert len(pieces) > 1
    assert "".join(pieces) == giant, "拆句不得丢字符"
    assert all(estimate_tokens(p) <= 500 for p in pieces)


def test_sentence_snap_respects_budget():
    """句末收口不能把下一片撑过预算——否则交给兜底硬切，等于白收口。"""
    from agent.rag.chunker import _snap_to_sentence_end

    cur = "句子到此结束。" + "残句" * 30 + "，"
    nxt = "下一段正文" + "内容" * 100
    src = cur + "\n" + nxt

    def fresh():
        return [{"source": "s", "text": cur}, {"source": "s", "text": nxt}]

    tight = _snap_to_sentence_end(fresh(), source=src,
                                  max_tokens=estimate_tokens(nxt) + 5)
    assert tight[1]["text"] == nxt, "并入后超预算时应放弃收口"

    loose = _snap_to_sentence_end(fresh(), source=src, max_tokens=10_000)
    assert loose[1]["text"] != nxt, "预算足够时应把残句并入下一片"
    _assert_strictly_lossless(loose, src)


def test_pipeline_holds_budget_without_final_clamp(monkeypatch):
    """关掉最后一道硬切，上游各环节也必须自己守住 token 预算。

    以前 ``_clamp_chunk_budget`` 是「日常接活」的：语义分片放过超长单句、
    句末收口无预算检查、重叠直接叠加，都要靠它兜底。现在各环节自己收口，
    关掉 clamp 也不该出现超预算切片。
    """
    import agent.rag.chunker as ch

    monkeypatch.setattr(ch, "_clamp_chunk_budget", lambda chunks, size: chunks)

    doc = "".join(
        f"第{i}段开头。这里是一段很长的描述，用逗号连接、没有句号的句子一直延续下去，"
        * 4
        for i in range(120)
    )
    for use_semantic in (False,):
        chunks = chunk_documents(
            [{"source": "t.txt", "text": doc}],
            chunk_size=500, chunk_overlap=50, min_chunk_size=100,
            overlap_percent=0.0, use_semantic=use_semantic,
        )
        assert chunks
        assert all(estimate_tokens(c["text"]) <= 500 for c in chunks), \
            "无最终硬切时仍有切片超预算 → 上游某环节没守预算"
        _assert_strictly_lossless(chunks, doc)


def test_tokenizer_backed_estimator_counts_real_tokens(tmp_path):
    """绑定模型自带分词器后，预算按真实 token 计（不再是字符类别近似）。"""
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers

    from agent.rag.chunker import configure_tokenizer, chunk_text, tokenizer_path

    tok = Tokenizer(models.WordLevel(
        {"[UNK]": 0, "hello": 1, "world": 2, "你好": 3}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    path = tmp_path / "tokenizer.json"
    tok.save(str(path))

    try:
        assert configure_tokenizer(str(path)), "本地 tokenizer.json 应绑定成功"
        assert tokenizer_path() == str(path)
        assert estimate_tokens("hello world") == 2
        # chunk_text 的 chunk_size 是正文预算（重叠由 _merge_overlap 追加），
        # 这里用零重叠验证「每片 ≤ 预算」本身。
        chunks = chunk_text("hello world 你好 " * 400, 64, 0)
        assert chunks
        assert all(estimate_tokens(c) <= 64 for c in chunks), "切片必须守住 token 预算"
    finally:
        configure_tokenizer(None)

    assert tokenizer_path() == "", "未绑定时应退回启发式估算"
    empty = tmp_path / "no_tokenizer"
    empty.mkdir()
    assert not configure_tokenizer(str(empty)), "目录无 tokenizer.json 时不应绑定"
