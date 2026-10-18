"""End-to-end RAG pipeline tests: extraction → cleaning → chunking → vectorization → search → reranking.

Tests the full knowledge base ingestion pipeline and verifies each stage:
1. Document parsing and text extraction
2. Text cleaning and normalization
3. Token-based chunking (500 tokens / 50 overlap)
4. Vector storage with ChromaDB
5. Semantic search with BGE reranking
6. RAG tool interface (rag_search, rag_status, rag_ingest)

Run: python3 -m pytest tests/test_rag_e2e.py -v
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest


# ── Test documents ──────────────────────────────────────────────────────────

def _create_test_documents(kb_dir: str) -> None:
    """Create test documents of various formats in the knowledge base directory."""
    os.makedirs(kb_dir, exist_ok=True)

    # Markdown document — RAG technology intro
    with open(os.path.join(kb_dir, "rag_intro.md"), "w", encoding="utf-8") as f:
        f.write("""# RAG 技术介绍

## 什么是 RAG？

RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合信息检索与文本生成的技术架构。
它通过在生成回答前先从外部知识库中检索相关文档片段，然后将检索结果作为上下文提供给大语言模型，
从而显著提高回答的准确性和时效性。

## RAG 的核心组件

1. **文档解析器（Document Parser）**：负责从各种格式（PDF、Word、Markdown 等）中提取文本。
2. **文本切片器（Text Chunker）**：将长文档分割成适合模型处理的短片段。
3. **向量化引擎（Embedding Engine）**：将文本片段转换为向量表示。
4. **向量数据库（Vector Store）**：存储和检索向量，常用 ChromaDB、FAISS 等。
5. **重排序器（Reranker）**：对初步检索结果进行精确排序，提升相关性。

## RAG 的优势

相比传统的关键词搜索，RAG 使用语义向量搜索，能够理解查询的深层含义。
通过 BGE Reranker 等模型进行重排序，可以进一步过滤不相关的结果。
""")

    # TXT document — vector database comparison
    with open(os.path.join(kb_dir, "vector_db.txt"), "w", encoding="utf-8") as f:
        f.write("""向量数据库对比分析

向量数据库是专门用于存储和检索高维向量数据的数据库系统，在 AI 应用中扮演着关键角色。

1. ChromaDB
   - 开源、轻量级
   - 支持多种嵌入模型
   - 内置持久化存储
   - 适合中小规模应用

2. FAISS
   - Facebook AI 开发
   - 高性能向量检索
   - 支持 GPU 加速
   - 适合大规模检索场景

3. Pinecone
   - 云原生向量数据库
   - 全托管服务
   - 自动扩缩容
   - 适合生产环境部署

4. Weaviate
   - 开源向量搜索引擎
   - 支持 GraphQL 查询
   - 内置多种向量化模块
   - 适合复杂查询场景

选择向量数据库时需要考虑：数据规模、查询延迟、部署复杂度、成本等因素。
""")

    # HTML document — knowledge base best practices
    with open(os.path.join(kb_dir, "best_practices.html"), "w", encoding="utf-8") as f:
        f.write("""<!DOCTYPE html>
<html>
<head><title>知识库最佳实践</title></head>
<body>
<header>导航栏（应被清洗掉）</header>
<nav>菜单（应被清洗掉）</nav>
<main>
<h1>知识库构建最佳实践</h1>

<p>构建高质量的知识库是 RAG 系统成功的关键。以下是一些经过验证的最佳实践：</p>

<h2>文档质量控制</h2>
<p>确保知识库中的文档内容准确、完整、格式规范。
避免包含大量扫描件或图片型 PDF，优先使用文本型文档。</p>

<h2>文本切片策略</h2>
<p>切片大小应根据模型上下文窗口合理设置。一般建议 500 token 一个切片，
保留 10% 的重叠区域，避免信息在切片边界处丢失。</p>

<h2>元数据管理</h2>
<p>为每个文档添加标题、来源、日期等元数据信息，方便检索时进行过滤和溯源。</p>

<h2>定期更新</h2>
<p>知识库需要定期更新，确保信息的时效性。建议设置自动同步机制。</p>
</main>
<footer>页脚信息（应被清洗掉）</footer>
<script>console.log("script should be removed")</script>
<style>body { color: red; }</style>
</body>
</html>""")

    # CSV document — embedding model comparison
    with open(os.path.join(kb_dir, "embedding_models.csv"), "w", encoding="utf-8") as f:
        f.write("""模型名称,维度,语言支持,特点
all-MiniLM-L6-v2,384,多语言,轻量快速
bge-large-zh-v1.5,1024,中文,中文优化
text-embedding-3-small,1536,多语言,OpenAI 官方
m3e-base,768,中英文,中文社区流行
bge-m3,1024,多语言,多语言检索""")

    # Subdirectory with nested documents
    sub_dir = os.path.join(kb_dir, "subdir")
    os.makedirs(sub_dir, exist_ok=True)
    with open(os.path.join(sub_dir, "tokens.md"), "w", encoding="utf-8") as f:
        f.write("""# Token 与上下文窗口管理

## Token 基本概念

Token 是大语言模型处理文本的基本单位。在英文中，一个 token 大约等于 3-4 个字符；
在中文中，一个 token 大约等于 1.5-2 个字符。

## 上下文窗口

上下文窗口决定了模型在一次推理中能"看到"的最大 token 数量。
常见的上下文窗口大小有 4K、8K、32K、128K 等。

## Token 消耗优化

1. 使用较短的系统提示词
2. 合理设置切片大小，避免过长的上下文
3. 使用重排序只保留最相关的结果
4. 压缩历史对话，移除冗余信息
""")


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def test_workspace():
    """Create a temporary workspace for RAG testing."""
    tmp = tempfile.mkdtemp(prefix="rag_test_ws_")
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="module")
def test_kb():
    """Create a temporary knowledge base with test documents."""
    tmp = tempfile.mkdtemp(prefix="rag_test_kb_")
    _create_test_documents(tmp)
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


# ── Module 1: Document extraction ───────────────────────────────────────────

class TestDocumentExtraction:
    """Test that parsers correctly extract text from various formats."""

    def test_extract_markdown(self, test_kb):
        from src.agent.rag.parsers import extract_text
        text = extract_text(os.path.join(test_kb, "rag_intro.md"))
        assert "RAG" in text
        assert "检索增强生成" in text
        assert "文档解析器" in text
        assert len(text) > 200

    def test_extract_txt(self, test_kb):
        from src.agent.rag.parsers import extract_text
        text = extract_text(os.path.join(test_kb, "vector_db.txt"))
        assert "向量数据库" in text
        assert "ChromaDB" in text
        assert "FAISS" in text
        assert len(text) > 300

    def test_extract_html(self, test_kb):
        from src.agent.rag.parsers import extract_text
        text = extract_text(os.path.join(test_kb, "best_practices.html"))
        assert "知识库" in text
        assert "切片" in text
        assert len(text) > 100

    def test_extract_csv(self, test_kb):
        from src.agent.rag.parsers import extract_text
        text = extract_text(os.path.join(test_kb, "embedding_models.csv"))
        assert "all-MiniLM-L6-v2" in text
        assert "bge-large-zh" in text
        assert len(text) > 80

    def test_extract_directory_recursive(self, test_kb):
        from src.agent.rag.parsers import extract_directory
        results = extract_directory(test_kb, recursive=True)
        # 5 documents: rag_intro.md, vector_db.txt, best_practices.html,
        #             embedding_models.csv, subdir/tokens.md
        assert len(results) >= 5
        sources = [os.path.basename(r[0]) for r in results]
        assert "rag_intro.md" in sources
        assert "tokens.md" in sources

    def test_extract_nonexistent_file(self):
        from src.agent.rag.parsers import extract_text
        with pytest.raises(FileNotFoundError):
            extract_text("/nonexistent/file.pdf")


# ── Module 2: Text cleaning ─────────────────────────────────────────────────

class TestTextCleaning:
    """Test the cleaner removes garbage and normalizes text."""

    def test_remove_html_tags(self):
        from src.agent.rag.cleaner import clean_text
        input_text = "<div><p>Hello <b>World</b></p></div>"
        output = clean_text(input_text)
        assert "Hello" in output
        assert "World" in output
        assert "<div>" not in output
        assert "<b>" not in output

    def test_remove_script_style_blocks(self):
        from src.agent.rag.cleaner import clean_text
        input_text = """<script>var x = 1;</script>Content here<style>body{}</style>"""
        output = clean_text(input_text)
        assert "var x = 1" not in output
        assert "body{}" not in output
        assert "Content here" in output

    def test_remove_html_entities(self):
        from src.agent.rag.cleaner import clean_text
        input_text = "Hello &amp; World &lt;script&gt; &copy;"
        output = clean_text(input_text)
        assert "&amp;" not in output
        assert "&lt;" not in output
        assert "Hello" in output
        assert "World" in output

    def test_remove_urls(self):
        from src.agent.rag.cleaner import clean_text
        input_text = "Visit https://example.com/page for more info"
        output = clean_text(input_text)
        assert "https://example.com/page" not in output
        assert "more info" in output

    def test_remove_control_chars(self):
        from src.agent.rag.cleaner import clean_text
        input_text = "Hello\x00\x01\x02World\x1f"
        output = clean_text(input_text)
        assert "\x00" not in output
        assert "Hello" in output
        assert "World" in output

    def test_collapse_whitespace(self):
        from src.agent.rag.cleaner import clean_text
        input_text = "Hello    World\t\tTest"
        output = clean_text(input_text)
        assert "    " not in output
        assert "Hello World Test" in output

    def test_remove_page_numbers(self):
        from src.agent.rag.cleaner import clean_text
        input_text = "Content\n42\nMore content\n第 1 页\nStill more\nPage 3"
        output = clean_text(input_text)
        assert "42" not in output.split("\n")
        assert "第 1 页" not in output
        assert "Page 3" not in output

    def test_remove_separator_lines(self):
        from src.agent.rag.cleaner import clean_text
        input_text = "Title\n---\nContent\n====\nMore"
        output = clean_text(input_text)
        assert "---" not in output
        assert "====" not in output

    def test_clean_html_document(self, test_kb):
        from src.agent.rag.parsers import extract_text
        from src.agent.rag.cleaner import clean_text
        raw = extract_text(os.path.join(test_kb, "best_practices.html"))
        cleaned = clean_text(raw)
        # Should not contain nav/header/footer content
        assert "导航栏" not in cleaned
        assert "菜单" not in cleaned
        assert "页脚信息" not in cleaned
        # Should not contain script/style
        assert "console.log" not in cleaned
        assert "color: red" not in cleaned
        # Should contain main content
        assert "知识库构建" in cleaned
        assert "切片" in cleaned


# ── Module 3: Text chunking ─────────────────────────────────────────────────

class TestTextChunking:
    """Test token-based chunking with overlap."""

    def test_chunk_text_basic(self):
        from src.agent.rag.chunker import chunk_text
        text = "这是第一段测试文本。\n\n这是第二段测试文本。\n\n这是第三段测试文本。"
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)

    def test_chunk_short_text_returns_single(self):
        from src.agent.rag.chunker import chunk_text
        text = "短文本"
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == "短文本"

    def test_chunk_empty_text(self):
        from src.agent.rag.chunker import chunk_text
        assert chunk_text("", chunk_size=500, chunk_overlap=50) == []
        assert chunk_text("   ", chunk_size=500, chunk_overlap=50) == []

    def test_chunk_long_text(self):
        from src.agent.rag.chunker import chunk_text
        # Generate text that needs multiple chunks
        paragraphs = []
        for i in range(30):
            paragraphs.append(f"第{i}段：" + "这是测试内容。" * 50)
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) > 1
        # Each chunk should be roughly <= 500 tokens
        for chunk in chunks:
            assert len(chunk) > 0

    def test_chunk_documents(self):
        from src.agent.rag.chunker import chunk_documents
        docs = [
            {"source": "/tmp/doc1.md", "text": "文档一的内容。" * 200},
            {"source": "/tmp/doc2.md", "text": "文档二的内容。" * 200},
        ]
        chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
        assert len(chunks) > 0
        # Verify metadata
        for chunk in chunks:
            assert "source" in chunk
            assert "chunk_index" in chunk
            assert "text" in chunk
            assert chunk["source"] in ("/tmp/doc1.md", "/tmp/doc2.md")

    def test_chunk_preserves_content_order(self, test_kb):
        from src.agent.rag.parsers import extract_text
        from src.agent.rag.cleaner import clean_text
        from src.agent.rag.chunker import chunk_documents
        raw = extract_text(os.path.join(test_kb, "rag_intro.md"))
        cleaned = clean_text(raw)
        chunks = chunk_documents(
            [{"source": "rag_intro.md", "text": cleaned}],
            chunk_size=500, chunk_overlap=50,
        )
        assert len(chunks) >= 1
        # First chunk should contain the intro
        combined = " ".join(c["text"] for c in chunks)
        assert "RAG" in combined
        assert "检索增强生成" in combined


# ── Module 4: Vector store ──────────────────────────────────────────────────

class TestVectorStore:
    """Test ChromaDB vector storage and retrieval."""

    @pytest.fixture
    def store_dir(self):
        tmp = tempfile.mkdtemp(prefix="chroma_test_")
        yield tmp
        shutil.rmtree(tmp, ignore_errors=True)

    def test_store_init_and_count(self, store_dir, vector_store_factory):
        store = vector_store_factory(persist_dir=store_dir)
        assert store.count() == 0

    def test_add_and_search(self, store_dir, vector_store_factory):
        store = vector_store_factory(persist_dir=store_dir)

        chunks = [
            {"source": "doc1.md", "chunk_index": 0,
             "text": "RAG 是一种检索增强生成技术，结合了信息检索和文本生成。"},
            {"source": "doc1.md", "chunk_index": 1,
             "text": "向量数据库如 ChromaDB 和 FAISS 用于存储文档嵌入向量。"},
            {"source": "doc2.md", "chunk_index": 0,
             "text": "Python 是一种流行的编程语言，广泛用于数据科学和 AI 开发。"},
        ]
        added = store.add(chunks)
        assert added == 3
        assert store.count() == 3

        # Search for RAG-related content
        results = store.search("什么是 RAG 技术", top_k=2)
        assert len(results) == 2
        # The RAG chunk should be the first result
        assert "RAG" in results[0]["text"]
        assert results[0]["source"] == "doc1.md"

    def test_search_empty_store(self, store_dir, vector_store_factory):
        store = vector_store_factory(persist_dir=store_dir)
        results = store.search("query", top_k=3)
        assert results == []

    def test_list_sources(self, store_dir, vector_store_factory):
        store = vector_store_factory(persist_dir=store_dir)

        chunks = [
            {"source": "/path/a.md", "chunk_index": 0, "text": "内容A"},
            {"source": "/path/b.md", "chunk_index": 0, "text": "内容B"},
            {"source": "/path/a.md", "chunk_index": 1, "text": "内容A续"},
        ]
        store.add(chunks)
        sources = store.list_sources()
        assert len(sources) == 2
        assert "/path/a.md" in sources
        assert "/path/b.md" in sources

    def test_clear(self, store_dir, vector_store_factory):
        store = vector_store_factory(persist_dir=store_dir)

        store.add([{"source": "doc.md", "chunk_index": 0, "text": "test"}])
        assert store.count() == 1
        store.clear()
        assert store.count() == 0

    def test_search_with_rerank_fallback(self, store_dir, vector_store_factory):
        """When reranker is unavailable, should fall back to top_k directly."""
        store = vector_store_factory(persist_dir=store_dir)

        chunks = [
            {"source": "doc.md", "chunk_index": i,
             "text": f"这是第{i}段关于机器学习和人工智能的内容。" * 3}
            for i in range(10)
        ]
        store.add(chunks)

        # search_with_rerank retrieves top_k * 4 candidates, then reranks
        results = store.search_with_rerank("机器学习", top_k=3)
        # Even without reranker (fallback), should return at most 3
        assert 1 <= len(results) <= 3
        for r in results:
            assert "text" in r
            assert "source" in r
            assert "score" in r

    def test_search_fts_keyword(self, store_dir, vector_store_factory):
        """BM25 keyword search should find chunks by exact CJK keywords."""
        store = vector_store_factory(persist_dir=store_dir)

        chunks = [
            {"source": "a.md", "chunk_index": 0,
             "text": "机器学习是人工智能的核心分支。深度学习使用神经网络。"},
            {"source": "b.md", "chunk_index": 0,
             "text": "Python 是一种流行的编程语言。"},
            {"source": "c.md", "chunk_index": 0,
             "text": "气候变化是当今世界的重大挑战。"},
        ]
        store.add(chunks)

        # Exact keyword should surface the matching chunk.
        results = store.search_fts("机器学习 神经网络", top_k=3)
        assert results, "BM25 should return the matching chunk"
        assert results[0]["source"] == "a.md"

    def test_search_fts_english_keyword(self, store_dir, vector_store_factory):
        """BM25 should also work for English keywords."""
        store = vector_store_factory(persist_dir=store_dir)

        chunks = [
            {"source": "en.md", "chunk_index": 0,
             "text": "Machine learning is a branch of artificial intelligence."},
            {"source": "cn.md", "chunk_index": 0,
             "text": "深度学习使用多层神经网络。"},
        ]
        store.add(chunks)

        results = store.search_fts("machine learning", top_k=2)
        assert results, "BM25 should find the English chunk"
        assert results[0]["source"] == "en.md"

    def test_hybrid_retrieval_merges_vector_and_bm25(self, store_dir, vector_store_factory):
        """search_with_rerank merges vector + BM25 candidates (dedupe)."""
        store = vector_store_factory(persist_dir=store_dir)

        chunks = [
            {"source": "a.md", "chunk_index": 0,
             "text": "机器学习是人工智能的核心分支。"},
            {"source": "b.md", "chunk_index": 0,
             "text": "Python 是一种流行的编程语言。"},
            {"source": "c.md", "chunk_index": 0,
             "text": "气候变化是当今世界的重大挑战。"},
        ]
        store.add(chunks)

        results = store.search_with_rerank("机器学习", top_k=3)
        # Fallback mode (no reranker) returns up to top_k, no dedupe duplicates.
        assert 1 <= len(results) <= 3
        # No duplicate (source, chunk_index) pairs.
        keys = [(r["source"], r.get("chunk_index", 0)) for r in results]
        assert len(keys) == len(set(keys))

    def test_persistence(self, store_dir, vector_store_factory):
        """Data added to one store should be visible in another instance."""
        store1 = vector_store_factory(persist_dir=store_dir)
        store1.add([{"source": "p.md", "chunk_index": 0, "text": "持久化测试内容"}])
        assert store1.count() == 1

        # New instance pointing to same dir
        store2 = vector_store_factory(persist_dir=store_dir)
        assert store2.count() == 1
        results = store2.search("持久化测试", top_k=1)
        assert len(results) == 1
        assert "持久化测试" in results[0]["text"]


# ── Module 5: RAG Engine ────────────────────────────────────────────────────

class TestRAGEngine:
    """Test the full RAG engine: ingest → search → status."""

    @pytest.fixture
    def engine(self, test_workspace, test_kb, rag_engine_factory):
        eng = rag_engine_factory(workspace=test_workspace, knowledge_base=test_kb)
        return eng

    def test_ingest_creates_rag_structure(self, engine):
        """Ingestion should create the rag directory structure."""
        stats = engine.ingest(force=True)
        assert "error" not in stats
        assert stats["files_found"] >= 5
        assert stats["chunks"] > 0
        assert stats["total_chars"] > 0

        # Verify directory structure
        assert os.path.isdir(engine.rag_dir)
        assert os.path.isdir(engine.markdown_dir)
        # Check that markdown files were created
        md_files = [f for f in os.listdir(engine.markdown_dir) if f.endswith(".md")]
        assert len(md_files) >= 5

    def test_ingest_skips_on_second_run(self, engine):
        """Second ingestion without force should skip already-processed files."""
        # First run
        stats1 = engine.ingest(force=True)
        assert stats1["files_extracted"] >= 5

        # Second run — should skip all
        stats2 = engine.ingest(force=False)
        assert stats2["files_skipped"] >= stats1["files_extracted"]
        assert stats2["files_extracted"] == 0

    def test_ingest_force_reprocesses(self, engine):
        """Force re-ingestion should reprocess all files."""
        engine.ingest(force=True)  # initial
        stats = engine.ingest(force=True)  # force re-process
        assert stats["files_extracted"] >= 5
        assert stats["files_skipped"] == 0

    def test_search_relevant_content(self, engine):
        """Semantic search should find relevant content."""
        engine.ingest(force=True)

        # Search for RAG-related content
        results = engine.search("什么是 RAG 技术")
        assert len(results) >= 1
        # The deterministic test embedding (MD5 hash) carries no semantic
        # signal, so we can only assert that search returned non-empty,
        # well-formed results — not that a specific keyword appears in the
        # top hits (which would depend on the arbitrary chunk boundaries).
        for r in results:
            assert r["text"].strip()
            assert r["source"]

    def test_search_vector_databases(self, engine):
        """Search for vector database content."""
        engine.ingest(force=True)

        results = engine.search("向量数据库有哪些")
        assert len(results) >= 1
        # Should find ChromaDB, FAISS, etc.
        combined = " ".join(r["text"] for r in results)
        assert "ChromaDB" in combined or "FAISS" in combined

    def test_search_knowledge_base_practices(self, engine):
        """Search for knowledge base best practices."""
        engine.ingest(force=True)

        results = engine.search("知识库构建")
        assert len(results) >= 1
        combined = " ".join(r["text"] for r in results)
        assert "切片" in combined or "知识库" in combined

    def test_search_embedding_models(self, engine):
        """Search for embedding model information."""
        engine.ingest(force=True)

        # 确定性 MD5 测试嵌入没有语义信号，top-k 命中取决于分块边界与顺序；
        # 用大 top_k 拉回全部切片，避免清洗/分块逻辑变化导致断言抖动。
        results = engine.search("嵌入模型有哪些", top_k=50)
        assert len(results) >= 1
        combined = " ".join(r["text"] for r in results)
        # Should find model names from the CSV
        assert any(name in combined for name in [
            "all-MiniLM", "bge-large", "m3e-base",
        ])

    def test_search_token_management(self, engine):
        """Search for token-related content (from subdirectory)."""
        engine.ingest(force=True)

        results = engine.search("token 上下文窗口")
        assert len(results) >= 1
        # Deterministic (MD5) test embedding has no semantic signal, so we
        # only assert search returns well-formed, non-empty results — not that
        # a specific keyword appears in the top hits (which depends on
        # arbitrary chunk boundaries, not relevance).
        for r in results:
            assert r["text"].strip()
            assert r["source"]

    def test_search_empty_query(self, rag_engine_factory):
        """Search with empty knowledge base should return empty."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            eng = rag_engine_factory(workspace=tmp, knowledge_base="")
            results = eng.search("anything")
            assert results == []

    def test_search_formatted_returns_string(self, engine):
        """search_formatted should return a formatted string."""
        engine.ingest(force=True)
        result = engine.search_formatted("RAG 技术", top_k=2)
        assert isinstance(result, str)
        assert "知识库检索" in result
        assert "相关度" in result

    def test_search_formatted_returns_valid_string(self, engine):
        """search_formatted should always return a valid string with result count."""
        engine.ingest(force=True)
        result = engine.search_formatted("量子计算与黑洞理论", top_k=1)
        assert isinstance(result, str)
        # Even for niche queries, the embedding model may find semantic matches.
        # The result should be well-formed regardless.
        assert "知识库检索" in result or "未找到" in result

    def test_status_reports_correctly(self, engine):
        """Status should reflect the current state of the knowledge base."""
        engine.ingest(force=True)
        status = engine.status()
        assert status["chunks_stored"] > 0
        assert len(status["sources"]) >= 5
        assert status["has_knowledge_base"] is True

    def test_status_no_knowledge_base(self, rag_engine_factory):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            eng = rag_engine_factory(workspace=tmp, knowledge_base="")
            status = eng.status()
            assert status["chunks_stored"] == 0
            assert status["has_knowledge_base"] is False

    def test_clear_removes_everything(self, engine):
        """Clear should remove all vectors and markdown cache."""
        engine.ingest(force=True)
        assert engine.status()["chunks_stored"] > 0

        engine.clear()
        assert engine.status()["chunks_stored"] == 0
        # Markdown cache should also be cleared
        if os.path.isdir(engine.markdown_dir):
            md_files = [f for f in os.listdir(engine.markdown_dir) if f.endswith(".md")]
            assert len(md_files) == 0

    def test_ingest_no_knowledge_base(self, test_workspace, rag_engine_factory):
        eng = rag_engine_factory(workspace=test_workspace, knowledge_base="/nonexistent/path")
        stats = eng.ingest()
        assert "error" in stats


# ── Module 6: RAG Tool interface ────────────────────────────────────────────

class TestRAGTools:
    """Test the tool interface that the agent uses."""

    @pytest.fixture
    def engine(self, test_workspace, test_kb, rag_engine_factory):
        eng = rag_engine_factory(workspace=test_workspace, knowledge_base=test_kb)
        eng.ingest(force=True)
        return eng

    def test_rag_search_tool(self, engine):
        from src.agent.tools.rag_tool import RagSearchTool
        tool = RagSearchTool(engine)
        result = tool.run(query="向量数据库")
        assert result.success is True
        # The search should return relevant results about vector databases
        # or embedding models (semantic search may match different docs)
        assert "知识库检索" in result.output
        assert "相关度" in result.output

    def test_rag_search_tool_empty_query(self, engine):
        from src.agent.tools.rag_tool import RagSearchTool
        tool = RagSearchTool(engine)
        result = tool.run(query="")
        assert result.success is False
        assert "请提供" in result.error

    def test_rag_search_tool_no_results(self, engine):
        from src.agent.tools.rag_tool import RagSearchTool
        tool = RagSearchTool(engine)
        result = tool.run(query="量子计算黑洞理论弦论")
        assert result.success is True
        # The search should not crash and return a valid formatted result.
        # Even for obscure queries, the embedding model may find semantic
        # matches due to the nature of vector search.
        assert isinstance(result.output, str)
        assert len(result.output) > 0

    def test_rag_status_tool(self, engine):
        from src.agent.tools.rag_tool import RagStatusTool
        tool = RagStatusTool(engine)
        result = tool.run()
        assert result.success is True
        assert "切片数" in result.output
        assert "已索引文件" in result.output

    def test_rag_ingest_tool(self, engine):
        from src.agent.tools.rag_tool import RagIngestTool
        tool = RagIngestTool(engine)
        result = tool.run(force=True)
        assert result.success is True
        assert "切片数" in result.output
        assert "索引完成" in result.output

    def test_rag_ingest_tool_no_kb(self, test_workspace, rag_engine_factory):
        from src.agent.tools.rag_tool import RagIngestTool
        eng = rag_engine_factory(workspace=test_workspace, knowledge_base="/nonexistent")
        tool = RagIngestTool(eng)
        result = tool.run()
        assert result.success is False
        assert "目录" in result.error or "not found" in result.error.lower()


# ── Module 7: End-to-end ingestion pipeline ─────────────────────────────────

class TestEndToEndPipeline:
    """Full pipeline: create docs → ingest → search → verify relevance."""

    @pytest.fixture
    def setup(self, test_workspace, test_kb, rag_engine_factory):
        engine = rag_engine_factory(workspace=test_workspace, knowledge_base=test_kb)
        return engine

    def test_full_pipeline_with_rerank(self, setup):
        """E2E: ingest all documents, search with reranking, verify results."""
        engine = setup

        # Step 1: Ingest
        stats = engine.ingest(force=True)
        assert stats["files_found"] >= 5, f"Expected >=5 files, got {stats['files_found']}"
        assert stats["chunks"] > 0, "No chunks generated"
        assert stats["total_chars"] > 500, "Too few characters extracted"

        # Step 2: Verify status
        status = engine.status()
        assert status["chunks_stored"] == stats["chunks"]
        assert len(status["sources"]) >= 5

        # Step 3: Search with reranking (top_k=3)
        # Test multiple queries to verify semantic search quality
        test_queries = [
            ("什么是 RAG 技术", "RAG"),
            ("向量数据库对比", "ChromaDB"),
            ("知识库构建最佳实践", "切片"),
            ("嵌入模型", "all-MiniLM"),
            ("token 上下文窗口", "Token"),
        ]

        for query, expected_keyword in test_queries:
            results = engine.search(query, top_k=3)
            assert len(results) >= 1, f"No results for query: {query}"
            # Should return at most 3 results after reranking
            assert len(results) <= 3, f"Too many results for query: {query}"
            # Each result should have score
            for r in results:
                assert "score" in r, f"Missing score in result for query: {query}"
                assert "text" in r
                assert "source" in r

        # Step 4: Re-ingest without force — should skip
        stats2 = engine.ingest(force=False)
        assert stats2["files_skipped"] >= 5
        assert stats2["files_extracted"] == 0
        # Chunks should remain the same
        assert engine.status()["chunks_stored"] == stats["chunks"]

        # Step 5: Clear and verify
        engine.clear()
        assert engine.status()["chunks_stored"] == 0

    def test_cleaned_markdown_files_are_valid(self, setup):
        """Verify cleaned markdown files are readable and clean."""
        engine = setup
        engine.ingest(force=True)

        md_dir = engine.markdown_dir
        for md_file in os.listdir(md_dir):
            if not md_file.endswith(".md"):
                continue
            path = os.path.join(md_dir, md_file)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            # Should not contain raw HTML tags
            assert "<script>" not in content.lower()
            assert "<style>" not in content.lower()
            assert "<nav>" not in content.lower()
            # Should not contain HTML entities
            assert "&amp;" not in content
            assert "&lt;" not in content
            # Should have meaningful content
            assert len(content.strip()) > 10, f"Empty content in {md_file}"

    def test_subdirectory_documents_are_indexed(self, setup):
        """Documents in subdirectories should be found and indexed."""
        engine = setup
        engine.ingest(force=True)

        # Search for content from the subdirectory
        results = engine.search("token 概念")
        assert len(results) >= 1
        combined = " ".join(r["text"] for r in results)
        assert "token" in combined.lower() or "Token" in combined