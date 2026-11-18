"""长文档切片测试：标题、句子边界、代码/表格原子块、语义低谷、Token 剪枝。"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.rag.doc_chunker import (  # noqa: E402
    chunk_document, clean_query_tokens, prune_tokens,
)


def _long_para(n=40, sentence="This is a complete sentence. "):
    return sentence * n


def test_heading_sections_skip_empty_parent_and_merge_small():
    text = (
        "# Book Title\n\n"
        "## Chapter One\n\n"
        "Short body one.\n\n"
        "## Chapter Two\n\n"
        "Short body two.\n"
    )
    chunks = chunk_document(text, "Book", encoder=None,
                            max_chars=500, min_chars=300)
    assert chunks
    assert all(c.content.strip() != "# Book Title" for c in chunks)
    assert any("Chapter One" in c.content for c in chunks)
    assert any("Chapter Two" in c.content for c in chunks)
    # 两段都短，应尽量合并而不是留两个碎片。
    assert len(chunks) <= 2
    assert "Chapter One" in chunks[0].title


def test_code_and_table_kept_atomic_and_sentence_boundaries():
    text = (
        "## Section\n\n" + _long_para(15) +
        "\n```python\nprint('hello')\nprint('world')\n```\n\n"
        "| a | b |\n|---|---|\n| 1 | 2 |\n\n" + _long_para(10)
    )
    chunks = chunk_document(text, "Doc", encoder=None,
                            max_chars=420, min_chars=250)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.content.count("```") % 2 == 0, "代码围栏不能被拆开"
        if "| a | b |" in chunk.content:
            assert "| 1 | 2 |" in chunk.content, "表格头和数据必须在同一片"
        # 非代码/表格结尾的切片必须停在句末标点。
        tail = chunk.content.rstrip()
        if not tail.endswith(("```", "| 1 | 2 |")):
            assert tail.endswith((".", "。", "！", "？", "!", "?", ";", "；"))


def test_overlap_starts_at_sentence_boundary():
    text = "## S\n\n" + _long_para(30)
    chunks = chunk_document(text, "Doc", encoder=None,
                            max_chars=300, min_chars=200,
                            overlap_ratio=0.01)
    assert len(chunks) >= 2
    for chunk in chunks[1:]:
        first = chunk.content.lstrip()
        assert first.startswith(("This is a complete sentence.",
                                 "## S", "```", "|")), first[:40]


class _ValleyEncoder:
    """前 4 句同一语义向量，后 4 句另一向量；边界相似度为 0。"""

    def encode_documents(self, texts, batch_size=16):
        out = []
        for text in texts:
            if "AAA" in text:
                out.append(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
            elif "BBB" in text:
                out.append(np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32))
            else:
                out.append(np.array([[0.7, 0.7, 0.0, 0.0]], dtype=np.float32))
        return out


class _SentenceEncoder:
    """model2vec 风格：一句一个 1D 向量，而不是词向量矩阵。"""

    def encode_documents(self, texts, batch_size=16):
        out = []
        for text in texts:
            if "AAA" in text:
                out.append(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
            elif "BBB" in text:
                out.append(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
            else:
                out.append(np.array([0.7, 0.7, 0.0, 0.0], dtype=np.float32))
        return out


def test_semantic_valley_accepts_sentence_vectors():
    """model2vec 返回 1D 句向量时也要能识别语义低谷。"""
    text = "## S\n\n" + "".join(
        f"AAA sentence number {i}. " for i in range(14)
    ) + "".join(f"BBB sentence number {i}. " for i in range(14))
    chunks = chunk_document(text, "Doc", encoder=_SentenceEncoder(),
                            max_chars=500, min_chars=300,
                            overlap_ratio=0.0)
    assert len(chunks) >= 2
    assert "BBB" not in chunks[0].content
    assert any("BBB" in c.content for c in chunks[1:])


def test_semantic_valley_cut_and_adaptive_threshold():
    text = "## S\n\n" + "".join(
        f"AAA sentence number {i}. " for i in range(14)
    ) + "".join(f"BBB sentence number {i}. " for i in range(14))
    chunks = chunk_document(text, "Doc", encoder=_ValleyEncoder(),
                            max_chars=500, min_chars=300,
                            overlap_ratio=0.0)
    assert len(chunks) >= 2
    # 第一片只能含 AAA；BBB 必须从后续片开始（允许 overlap 带一句 AAA）。
    assert "BBB" not in chunks[0].content
    assert any("BBB" in c.content for c in chunks[1:])
    # 切点落在语义低谷，而不是在 AAA/BBB 中间。
    assert chunks[0].content.rstrip().endswith(".")


def test_recursive_fallback_for_pathological_long_sentence():
    text = "## S\n\n" + ("A" * 1400)
    chunks = chunk_document(text, "Doc", encoder=None,
                            max_chars=500, min_chars=300)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk.content) <= 520, "递归兜底也不能无限超长"


def test_prune_tokens_removes_stopwords_and_punctuation():
    text = "的 人工智能，是 一门 学科；机器学习 是 其 分支。"
    pruned = prune_tokens(text)
    assert "人工智能" in pruned
    assert "机器" in pruned and "学习" in pruned
    assert "，" not in pruned and "。" not in pruned
    assert " 的 " not in f" {pruned} "
    code = "def foo(x):\n    return x + 1"
    pruned_code = prune_tokens(code, is_code=True)
    assert "foo" in pruned_code and "return" in pruned_code
    assert "(" not in pruned_code


def test_clean_query_tokens_keeps_question_words_and_adjectives():
    """查询侧轻量清理：只去标点/停用词，形容词与疑问词必须保留。"""
    q = clean_query_tokens("死亡为什么可怕？")
    assert "为什么" in q and "可怕" in q
    assert "？" not in q
    assert clean_query_tokens("人类与神是一体的") == "人类神一体"
    en = clean_query_tokens("What does ColBERT do?")
    assert "What" in en and "ColBERT" in en
    assert "does" not in en.split() and "do" not in en.split()
    # 清理后为空时返回空串，由检索侧回退原查询。
    assert clean_query_tokens("的 是，。") == ""


def test_chunk_document_prefers_chapter_and_keeps_subsections():
    """章节优先：小节整体归属所属章节，不单独成片。"""
    small = "这是小节里的正文内容，讲述基础概念与细节。" * 10
    doc = (
        "## 第一章\n\n### 1.1 背景\n\n" + small
        + "\n\n### 1.2 方法\n\n" + small
        + "\n\n## 第二章\n\n" + ("第二章整体内容，篇幅较短同样成片。" * 8)
    )
    chunks = chunk_document(doc, "Book", max_chars=3000, min_chars=2500)
    c1 = next(c for c in chunks if "第一章" in c.title)
    # 两个小节都留在第一章整片里，小节标题仍是内容的一部分
    assert "### 1.1 背景" in c1.content and "### 1.2 方法" in c1.content
    assert len(c1.content) <= 3000


def test_oversized_chapter_packs_whole_subsections():
    """超长章节按小节整体装箱：小节正文绝不拦腰切断。"""
    body = "小节正文内容，完整保留不切分。" * 30          # ~450 字/小节
    doc = "## 大章\n\n" + "".join(
        f"### 小节{i}\n\n{body}\n\n" for i in range(10))
    chunks = chunk_document(doc, "Doc", max_chars=3000, min_chars=2500)
    assert chunks
    joined = "\n".join(c.content for c in chunks)
    for c in chunks:
        assert len(c.content) <= 3000
    # 10 个小节全部以完整块存在（若小节被拦腰切开会少于 10）
    assert joined.count(body) == 10
    # 每片都以某个小节标题开头（装箱从结构边界起片）
    assert all("### 小节" in c.content for c in chunks)
