"""estimate_tokens 字符分类的测试（bge-small-zh WordPiece 校准）。

背景：数字与标点在 BERT 词表里是每字符独立 token，旧实现把「非 CJK」统一按
3 字符/token 估算——标点/代码/URL 密集文档的切片真实 token 数超出估算 30%+
（实测 450 → 627），嵌入时被 512 窗口静默截断。新实现按字符类别计权，
保证估算 ≥ 真实值（误差仅 [CLS]/[SEP] 两个特殊 token），切片绝不超窗。
"""
from __future__ import annotations

from agent.rag.chunker import chunk_documents, estimate_tokens


def test_cjk_one_token_per_char():
    assert estimate_tokens("人工智能") == 4


def test_fullwidth_punct_one_token_per_char():
    # 旧实现：4//3=1；真实 WordPiece：4 → 每字符 1 token
    assert estimate_tokens("！！？？") == 4
    assert estimate_tokens("：，「」") == 4


def test_ascii_punct_one_token_per_char():
    assert estimate_tokens("=====") == 5
    assert estimate_tokens("(){}[]") == 6


def test_digits_one_token_per_char():
    assert estimate_tokens("1234567890") == 10


def test_ascii_letters_three_chars_per_token():
    assert estimate_tokens("abcdefghij") == 10 // 3
    assert estimate_tokens("abcdefghij" * 9) == 90 // 3


def test_short_acronyms_floored_by_run_count():
    # 两字母缩写按 chars//3 会算出 0；按字母段数兜底 = 每词 1 token
    assert estimate_tokens("UI") == 1
    assert estimate_tokens("AI API") == 2  # 两个缩写词各 1 token


def test_whitespace_and_control_zero_token():
    assert estimate_tokens("\n\n  \t") == 0
    assert estimate_tokens("\x00\x01\x07") == 0


def test_empty():
    assert estimate_tokens("") == 0


def test_mixed_sentence():
    # 版本(2 wide) ：(wide) 1.2.3(5 other) 空格(0) 发布(2 wide) ？(wide)
    assert estimate_tokens("版本：1.2.3 发布？") == 2 + 1 + 5 + 0 + 2 + 1


# ---------------------------------------------------------------------------
# 分片预算性质：标点/代码/URL 密集文档的每个切片估算 ≤ chunk_size
# （估算 ≥ 真实值 − 2 ⇒ 真实 token ≤ 512，嵌入不截断）
# ---------------------------------------------------------------------------

class _FakeEF:
    def embed_documents(self, texts):
        return [[0.0] * 8 for _ in texts]

    def __call__(self, input):
        return self.embed_documents(input)

    def name(self):
        return "fake"


def _assert_budget(name: str, text: str) -> None:
    chunks = chunk_documents(
        [{"source": "t.txt", "text": text}],
        chunk_size=500, chunk_overlap=50, min_chunk_size=100, overlap_percent=0.10,
        embedding_function=_FakeEF(),
    )
    assert chunks, name
    for c in chunks:
        # 关键不变量：每片估算 ≤ chunk_size。estimate_tokens 保证 ≥ 真实 token − 2，
        # 故真实 token ≤ 512（bge 窗口），嵌入阶段不截断。字符数不能作为判据——
        # 字母密集切片（代码/英文）字符数可远超 token 数（英文 ≈3.4 字符/token）。
        est = estimate_tokens(c["text"])
        assert est <= 500, f"{name}: 切片估算 {est} > 500（真实 token 将超 bge 512 窗口）"


def test_budget_punct_heavy():
    _assert_budget("标点", "结论：第一，性能提升「显著」；第二，成本下降约30%！" * 80)


def test_budget_code_heavy():
    _assert_budget("代码", "def f(x):\n    return x*2  # 注释\n" + "pip install lancedb==0.37.1\n" * 60)


def test_budget_url_heavy():
    _assert_budget("URL", "详见 https://example.com/docs?id=12345&page=2&lang=zh " * 100)


def test_budget_mixed_chinese_english():
    _assert_budget("中英", "第1章：RAG 系统的召回率提升了 12.5%，详见附录 A。" * 200)


def test_clamp_splits_oversized_chunk_from_degenerate_semantic_path():
    # 退化情形：无句末标点的超长文本 + 退化 EF（所有向量相同）时，chonkie 可能
    # 整段返回；_clamp_chunk_budget 必须兜底把每片切到 ≤ chunk_size，否则真实
    # token 会超 bge 的 512 窗口被静默截断。
    text = "人工智能是计算机科学的一个分支" * 400  # 6000 字符、无句末标点
    chunks = chunk_documents(
        [{"source": "t.txt", "text": text}],
        chunk_size=500, chunk_overlap=50, min_chunk_size=100, overlap_percent=0.10,
        embedding_function=_FakeEF(),
    )
    assert len(chunks) > 1, "超长文本必须被切分为多片"
    for c in chunks:
        assert estimate_tokens(c["text"]) <= 500, estimate_tokens(c["text"])


def test_clamp_enforces_500_char_cap_for_small_window_model():
    # bge 类小窗口模型（chunk_size 被夹到 510 ≤ 512，1 字≈1 token）：无论内容
    # 语种，最终每片「含重叠」字符数必须 ≤ 500，且真实 token ≤ 512（不截断）。
    docs = {
        # 英文长文：510 token 英文 ≈ 1700 字符，须被字数上限再切到 ≤ 500 字。
        "英文": ("Natural language processing enables computers to understand human "
                 "language. Retrieval augmented generation combines search with "
                 "generation to ground answers in private documents. ") * 60,
        "URL": "详见 https://example.com/docs?id=12345&page=2&lang=zh " * 100,
        "代码": ("def f(x):\n    return x * 2  # comment\n" + "pip install lancedb==0.37.1\n") * 60,
        "中文": "人工智能是计算机科学的一个分支" * 400,
    }
    for name, text in docs.items():
        chunks = chunk_documents(
            [{"source": "t.txt", "text": text}],
            chunk_size=500, chunk_overlap=50, min_chunk_size=100, overlap_percent=0.10,
            embedding_function=_FakeEF(),
        )
        assert chunks, name
        for c in chunks:
            assert len(c["text"]) <= 500, f"{name}: 切片 {len(c['text'])} 字超 500 字上限"
            assert estimate_tokens(c["text"]) <= 510


def test_no_char_cap_for_large_window_model():
    # nomic 类大窗口模型（chunk_size=1000 > 512）：不卡 500 字，token 预算即足够；
    # 英文长片可达上千字符但不被无谓碎裂（仍在 1000 token 窗口内）。
    text = ("Retrieval augmented generation grounds answers in private documents. "
            "Natural language processing enables computers to understand text. ") * 60
    chunks = chunk_documents(
        [{"source": "t.txt", "text": text}],
        chunk_size=1000, chunk_overlap=50, min_chunk_size=100, overlap_percent=0.10,
        embedding_function=_FakeEF(),
    )
    assert chunks
    # 存在字符数 > 500 的片（说明未被字数上限碎裂）
    assert any(len(c["text"]) > 500 for c in chunks)
    for c in chunks:
        assert estimate_tokens(c["text"]) <= 1000
