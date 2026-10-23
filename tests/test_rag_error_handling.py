"""测试 RAG 同步过程中的异常处理机制。

构造包含各种异常文件的测试数据集，模拟真实同步场景，验证：
1. 单个文件异常不会导致整个同步崩溃
2. 异常信息正确记录到 stats["errors"]
3. 正常文件仍然被正确处理（部分成功）
4. manifest 只记录成功处理的文件
5. 向量库只包含正常文件的数据

异常文件类型：
- 损坏的 PDF（无效 PDF 头）
- 损坏的 docx（无效 zip 结构）
- 损坏的 xlsx（无效 zip 结构）
- 权限不足的文件（无读权限）
- 空文件（0 字节）
- 只有空白字符的文件
- 二进制内容伪装成 .txt
- 不支持的扩展名（被 SUPPORTED_EXTENSIONS 过滤，不进入处理）
- 超大文件名（边界测试）
- Unicode 文件名
"""
from __future__ import annotations

import os
import sys
import tempfile
import stat
from pathlib import Path

import pytest

# 让 tests/ 可导入
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from conftest import FakeEmbeddingFunction  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture: 构造异常文件数据集
# ---------------------------------------------------------------------------

@pytest.fixture
def error_dataset(tmp_path):
    """构造包含各种异常文件的知识库目录。

    返回 dict，包含：
      - kb_dir: 知识库目录 Path
      - good_files: 正常文件列表（应被成功处理）
      - bad_files: 异常文件列表（应被跳过并记录到 errors）
      - skipped_files: 不支持的扩展名文件（应被 SUPPORTED_EXTENSIONS 过滤）
    """
    kb = tmp_path / "kb"
    kb.mkdir()

    good_files = []
    bad_files = []
    skipped_files = []

    # ---- 正常文件 ----
    (kb / "good_text.txt").write_text(
        "Python is a high-level programming language with dynamic typing."
    )
    good_files.append("good_text.txt")

    (kb / "good_markdown.md").write_text(
        "# RAG Overview\n\n"
        "Retrieval-Augmented Generation combines search with LLMs.\n\n"
        "## Components\n\n"
        "- Vector store\n"
        "- Embedding model\n"
        "- Reranker\n"
    )
    good_files.append("good_markdown.md")

    (kb / "good_csv.csv").write_text("name,value\nalpha,1\nbeta,2\ngamma,3\n")
    good_files.append("good_csv.csv")

    (kb / "good_html.html").write_text(
        "<html><body><h1>Title</h1><p>HTML content here.</p></body></html>"
    )
    good_files.append("good_html.html")

    # 子目录中的正常文件
    sub = kb / "subdir"
    sub.mkdir()
    (sub / "nested.txt").write_text("This file is in a subdirectory.")
    good_files.append("subdir/nested.txt")

    # ---- 损坏的 PDF（无效 PDF 头，PyMuPDF 会抛异常）----
    (kb / "corrupt.pdf").write_bytes(b"Not a real PDF file, just plain text with .pdf extension")
    bad_files.append("corrupt.pdf")

    # ---- 损坏的 docx（无效 zip 结构，python-docx 会抛异常）----
    (kb / "corrupt.docx").write_bytes(b"This is not a valid ZIP/OOXML file")
    bad_files.append("corrupt.docx")

    # ---- 损坏的 xlsx（无效 zip 结构，openpyxl 会抛异常）----
    (kb / "corrupt.xlsx").write_bytes(b"Invalid xlsx content not a zip file")
    bad_files.append("corrupt.xlsx")

    # ---- 损坏的 pptx ----
    (kb / "corrupt.pptx").write_bytes(b"Invalid pptx content not a zip file")
    bad_files.append("corrupt.pptx")

    # ---- 空文件（0 字节）----
    (kb / "empty.txt").write_bytes(b"")
    bad_files.append("empty.txt")

    # ---- 只有空白字符的文件 ----
    (kb / "whitespace.txt").write_text("   \n\n\t\t  \n   \n")
    bad_files.append("whitespace.txt")

    # ---- 二进制内容伪装成 .txt（能读取但内容无意义）----
    (kb / "binary.txt").write_bytes(bytes(range(256)) * 4)
    # 二进制文件能被 _extract_text 读取（errors="replace"），但 chunk 后可能无有效内容
    # 实际上它会被"成功"处理，因为 _extract_text 用 errors="replace"
    # 所以不放 bad_files，而是作为边界测试

    # ---- 权限不足的文件（无读权限）----
    # 注意：root 用户可绕过权限检查，所以这个测试在 root 下可能不生效
    no_perm = kb / "nopermission.txt"
    no_perm.write_text("This file has no read permission.")
    try:
        os.chmod(str(no_perm), 0o000)
        bad_files.append("nopermission.txt")
    except (OSError, PermissionError):
        # 某些系统/环境下无法修改权限，跳过此场景
        bad_files = [f for f in bad_files if f != "nopermission.txt"]

    # ---- Unicode 文件名（边界测试，应正常处理）----
    (kb / "中文文件.txt").write_text("这是一个包含中文内容的文件。")
    good_files.append("中文文件.txt")

    (kb / "emoji_🎉.txt").write_text("File with emoji in name and content 🚀")
    good_files.append("emoji_🎉.txt")

    # ---- 不支持的扩展名（应被 SUPPORTED_EXTENSIONS 过滤）----
    (kb / "readme.rst").write_text("RST format")  # rst 是支持的
    good_files.append("readme.rst")
    (kb / "image.jpg").write_bytes(b"fake jpg")  # jpg 不支持
    skipped_files.append("image.jpg")
    (kb / "data.json").write_text("{}")  # json 不支持
    skipped_files.append("data.json")
    (kb / "script.py").write_text("print('hello')")  # py 不支持
    skipped_files.append("script.py")
    (kb / ".hidden.txt").write_text("hidden file")  # 隐藏文件被过滤
    skipped_files.append(".hidden.txt")

    return {
        "kb_dir": str(kb),
        "good_files": good_files,
        "bad_files": bad_files,
        "skipped_files": skipped_files,
    }


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

def test_sync_does_not_crash_with_corrupt_files(error_dataset, tmp_path):
    """同步包含损坏文件的目录不应崩溃，异常应记录到 stats["errors"]。"""
    from agent.rag.engine import RAGEngine

    engine = RAGEngine(
        workspace=str(tmp_path),
        knowledge_base=error_dataset["kb_dir"],
        embedding_function=FakeEmbeddingFunction(),
    )
    try:
        stats = engine.ingest(force=False)

        # 不应崩溃
        assert "error" not in stats, f"ingest returned error: {stats.get('error')}"

        # 应该有错误记录
        errors = stats.get("errors", [])
        assert len(errors) > 0, "应该有错误记录，但 errors 为空"

        # 检查具体错误类型
        error_text = "\n".join(errors)
        # 损坏的 PDF 应被记录
        assert "corrupt.pdf" in error_text, f"corrupt.pdf 错误未记录: {errors}"

        # 不应出现任何 PDF/docx/xlsx 成功提取的记录
        # （它们都是损坏的，应该都失败）
    finally:
        engine.close()


def test_good_files_processed_despite_bad_ones(error_dataset, tmp_path):
    """混合目录中，正常文件应被成功处理，即使有损坏文件存在。"""
    from agent.rag.engine import RAGEngine

    engine = RAGEngine(
        workspace=str(tmp_path),
        knowledge_base=error_dataset["kb_dir"],
        embedding_function=FakeEmbeddingFunction(),
    )
    try:
        stats = engine.ingest(force=False)

        good_files = error_dataset["good_files"]
        bad_files = error_dataset["bad_files"]

        # files_found 应包含所有支持扩展名的文件（good + bad，不含 skipped）
        # 注意：.rst 是支持的，.jpg/.json/.py/.hidden.txt 不支持
        assert stats["files_found"] >= len(good_files) + len(bad_files)

        # 应该有成功提取的文件
        assert stats["files_extracted"] > 0, "应该有文件被成功提取"

        # 应该有错误
        assert len(stats["errors"]) > 0

        # 验证正常文件在向量库中可搜到
        results = engine.search("Python programming language", top_k=5)
        assert len(results) > 0, "应该能搜到正常文件内容"
        found_texts = " ".join(r["text"] for r in results).lower()
        assert "python" in found_texts, f"搜索结果应包含 python: {results}"

        # 验证中文文件可搜到
        results_cn = engine.search("中文内容", top_k=5)
        assert len(results_cn) > 0, "应该能搜到中文文件"

        # 验证 sources 列表包含正常文件
        sources = engine.status()["sources"]
        # good_files 中的文件应出现在 sources（或其子路径）
        # 注意：source 是完整路径
        good_basenames = [os.path.basename(f) for f in good_files]
        source_basenames = [os.path.basename(s) for s in sources]
        for good in good_basenames:
            if good in ["binary.txt"]:  # binary 可能被当作空内容跳过
                continue
            assert good in source_basenames, f"正常文件 {good} 未在 sources 中: {source_basenames}"

    finally:
        engine.close()


def test_manifest_only_contains_successfully_processed_files(error_dataset, tmp_path):
    """manifest 应只记录成功处理的文件，损坏文件不应在 manifest 中。"""
    import json
    from agent.rag.engine import RAGEngine, _manifest_path

    engine = RAGEngine(
        workspace=str(tmp_path),
        knowledge_base=error_dataset["kb_dir"],
        embedding_function=FakeEmbeddingFunction(),
    )
    try:
        stats = engine.ingest(force=False)

        manifest_path = _manifest_path(engine._rag_dir)
        assert os.path.exists(manifest_path), "manifest.json 应存在"
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        bad_basenames = [os.path.basename(f) for f in error_dataset["bad_files"]]
        # 损坏文件不应在 manifest 中
        for path in manifest:
            basename = os.path.basename(path)
            assert basename not in bad_basenames, \
                f"损坏文件 {basename} 不应在 manifest 中: {manifest.keys()}"

    finally:
        engine.close()


def test_incremental_sync_retries_failed_files(error_dataset, tmp_path):
    """增量同步：之前失败的文件下次同步仍会重试（manifest 未记录）。"""
    from agent.rag.engine import RAGEngine

    engine1 = RAGEngine(
        workspace=str(tmp_path),
        knowledge_base=error_dataset["kb_dir"],
        embedding_function=FakeEmbeddingFunction(),
    )
    try:
        stats1 = engine1.ingest(force=False)
        errors1 = len(stats1["errors"])
        extracted1 = stats1["files_extracted"]

        # 第二次同步（增量）：损坏文件仍然存在，应再次失败
        # 但因为它们没被记录到 manifest，会被再次尝试处理
        engine2 = RAGEngine(
            workspace=str(tmp_path),
            knowledge_base=error_dataset["kb_dir"],
            embedding_function=FakeEmbeddingFunction(),
        )
        try:
            stats2 = engine2.ingest(force=False)
            # 损坏文件应再次产生错误（因为没记录到 manifest，会重新尝试）
            assert len(stats2["errors"]) > 0, "损坏文件应再次失败"
            # 正常文件应被跳过（已记录在 manifest）
            assert stats2["files_skipped"] > 0, "正常文件应被跳过"
        finally:
            engine2.close()
    finally:
        engine1.close()


def test_force_sync_reprocesses_all_files(error_dataset, tmp_path):
    """force=True 时所有文件重新处理，包括之前成功的。"""
    from agent.rag.engine import RAGEngine

    engine = RAGEngine(
        workspace=str(tmp_path),
        knowledge_base=error_dataset["kb_dir"],
        embedding_function=FakeEmbeddingFunction(),
    )
    try:
        # 第一次同步
        stats1 = engine.ingest(force=False)
        extracted1 = stats1["files_extracted"]

        # 强制重新同步
        stats2 = engine.ingest(force=True)
        # force 模式下所有支持的文件都会被重新处理
        assert stats2["files_extracted"] > 0, "force 同步应重新处理文件"
        # 损坏文件仍会产生错误
        assert len(stats2["errors"]) > 0
    finally:
        engine.close()


def test_worker_exception_does_not_kill_other_workers(error_dataset, tmp_path):
    """一个 worker 处理损坏文件时异常，其他 worker 应继续正常工作。

    通过 monkeypatch 注入异常到一个文件的 chunk_documents 调用，
    验证其他文件仍被处理。
    """
    import agent.rag.engine as eng_mod
    original_chunk = eng_mod.chunk_documents
    call_count = [0]
    injected_file = "good_text.txt"

    def flaky_chunk(docs, **kw):
        call_count[0] += 1
        # 对第一个文件注入异常
        if any(injected_file in (d.get("source", "") if isinstance(d, dict) else "") for d in docs):
            raise RuntimeError("Injected worker failure")
        return original_chunk(docs, **kw)

    eng_mod.chunk_documents = flaky_chunk
    try:
        from agent.rag.engine import RAGEngine
        engine = RAGEngine(
            workspace=str(tmp_path),
            knowledge_base=error_dataset["kb_dir"],
            embedding_function=FakeEmbeddingFunction(),
        )
        try:
            stats = engine.ingest(force=False)
            # 不应崩溃
            assert "error" not in stats
            # 注入的异常应被记录
            errors_text = "\n".join(stats["errors"])
            assert "Injected worker failure" in errors_text, \
                f"注入的异常未记录: {stats['errors']}"
            # 其他正常文件应被处理
            assert stats["files_extracted"] > 0, "其他文件应被成功处理"
            # 搜索应返回结果（来自其他正常文件）
            results = engine.search("RAG", top_k=5)
            assert len(results) > 0, "应能搜到其他正常文件的内容"
        finally:
            engine.close()
    finally:
        eng_mod.chunk_documents = original_chunk


def test_progress_callback_exception_does_not_crash(error_dataset, tmp_path):
    """progress_callback 抛异常不应中断 ingest。"""
    from agent.rag.engine import RAGEngine

    engine = RAGEngine(
        workspace=str(tmp_path),
        knowledge_base=error_dataset["kb_dir"],
        embedding_function=FakeEmbeddingFunction(),
    )
    try:
        call_count = [0]

        def buggy_callback(done, total, current):
            call_count[0] += 1
            if call_count[0] == 3:
                raise RuntimeError("Bug in UI callback")

        # 不应抛异常
        stats = engine.ingest(force=False, progress_callback=buggy_callback)
        assert "error" not in stats
        # callback 应被调用过
        assert call_count[0] > 0
    finally:
        engine.close()


def test_search_on_empty_kb_returns_empty(tmp_path):
    """空知识库目录的搜索应返回空列表，不崩溃。"""
    from agent.rag.engine import RAGEngine

    kb = tmp_path / "empty_kb"
    kb.mkdir()
    engine = RAGEngine(
        workspace=str(tmp_path),
        knowledge_base=str(kb),
        embedding_function=FakeEmbeddingFunction(),
    )
    try:
        stats = engine.ingest(force=False)
        assert stats["files_found"] == 0
        assert stats["chunks"] == 0

        results = engine.search("anything", top_k=3)
        assert results == []
    finally:
        engine.close()


def test_sync_with_nonexistent_kb_returns_error(tmp_path):
    """不存在的知识库目录应返回 error，不崩溃。"""
    from agent.rag.engine import RAGEngine

    engine = RAGEngine(
        workspace=str(tmp_path),
        knowledge_base=str(tmp_path / "nonexistent"),
        embedding_function=FakeEmbeddingFunction(),
    )
    try:
        stats = engine.ingest(force=False)
        assert "error" in stats
        assert "not found" in stats["error"].lower() or "not a directory" in stats["error"].lower()
    finally:
        engine.close()


def test_cancel_during_sync_does_not_crash(error_dataset, tmp_path):
    """同步过程中取消不应导致崩溃或数据损坏。"""
    from agent.rag.engine import RAGEngine

    engine = RAGEngine(
        workspace=str(tmp_path),
        knowledge_base=error_dataset["kb_dir"],
        embedding_function=FakeEmbeddingFunction(),
    )
    try:
        # 立即取消（在 ingest 开始前）
        engine.cancel()
        stats = engine.ingest(force=False)
        # 即使取消了，stats 应该是有效的 dict
        assert isinstance(stats, dict)
        # cancelled 标志应为 True（如果 worker 检查到了取消信号）
        # 注意：如果取消太快可能在 worker 启动前就完成了，所以不强制检查 cancelled
    finally:
        engine.close()


def test_cancel_mid_sync_completed_files_not_redone(tmp_path):
    """取消同步：已完整入库的文件记入 manifest，下次同步跳过不再重做。

    2026-10-23 用户确认点。语义（engine.ingest 取消分支）：
      * 只有「全部切片已写入向量库」的文件（committed_manifest）才记入
        manifest——正在处理/只写入一半的文件不记录；
      * 下次同步：已记录文件走增量跳过，未记录文件重新处理（add_streaming
        按 source 删旧写新，幂等、无重复）；
      * 最终全库内容完整（30 个来源齐全）且 id 唯一。
    """
    from agent.rag.engine import RAGEngine, _load_manifest

    kb = tmp_path / "kb"
    kb.mkdir()
    # 每个文件 ~17.5k 字符 → chunk_size 500/overlap 50 下约 39 个切片：
    # 写入批大小 200 → 首批恰好装下 ~5 个文件，第 8 个文件的进度回调必然
    # 落在首批已写入之后的第二批拉取期间（首批 pull 中的取消会整批丢弃，
    # committed=0，测不到续传）。
    n_files = 12
    for i in range(n_files):
        body = " ".join(
            f"文件{i}第{s}段：" + "这是用于分块与续传测试的正文内容。" * 20
            for s in range(50)
        )
        (kb / f"book_{i:02d}.txt").write_text(body)

    engine = RAGEngine(
        workspace=str(tmp_path),
        knowledge_base=str(kb),
        embedding_function=FakeEmbeddingFunction(),
    )
    try:
        # 确定性钩子：_on_batch 的进度回调带「已写入 N 切片」文案——首次出现
        # 即首批已写完并提交。此刻同步做两件事：
        #   1) 快照磁盘 manifest → 验证「批次提交即增量落盘」（用户直接杀
        #      程序的场景：此时离收尾保存还很远，manifest 却已在盘上）；
        #   2) 触发取消 → 第二批拉取的缓冲被整批丢弃（不 committed）。
        # 注意 done 计数在 worker（生产者）侧递增，跑在消费端前面——用它
        # 触发取消有竞态，必须用批次回调。
        mid_manifest = [None]
        batch_seen = [False]

        def cancel_on_first_batch(done, total, current):
            if batch_seen[0] or not str(current).startswith("已写入"):
                return
            batch_seen[0] = True
            from agent.rag.engine import _load_manifest as _lm
            mid_manifest[0] = _lm(engine.rag_dir)
            engine.cancel()

        stats1 = engine.ingest(force=False, progress_callback=cancel_on_first_batch)
        assert stats1.get("cancelled") is True
        assert batch_seen[0]  # 首批回调确实发生过
        # 增量落盘生效：首批提交的文件在 ingest 结束前就已在磁盘 manifest 里
        assert mid_manifest[0], "首批写完但 manifest 未增量落盘"
        assert all(e.get("chunk_count") for e in mid_manifest[0].values())

        manifest = _load_manifest(engine.rag_dir)
        committed = {fp for fp, e in manifest.items() if e.get("chunk_count")}
        assert 0 < len(committed) < n_files          # 有续传进度可复用

        # 第二次同步：已入库文件跳过，其余重新处理
        stats2 = engine.ingest(force=False)
        assert not stats2.get("cancelled")
        assert stats2["files_skipped"] == len(committed)
        assert stats2["files_extracted"] == n_files - len(committed)

        # 最终内容完整：全部来源在库；id 唯一（半截文件被整体替换，无重复）
        tbl = engine._get_store()._table
        df = tbl.to_arrow().to_pandas()
        assert df["id"].is_unique
        assert df["source"].nunique() == n_files
    finally:
        engine.close()


def test_cleanup_after_sync(error_dataset, tmp_path):
    """同步后 close() 应释放资源，不影响后续操作。"""
    from agent.rag.engine import RAGEngine

    engine = RAGEngine(
        workspace=str(tmp_path),
        knowledge_base=error_dataset["kb_dir"],
        embedding_function=FakeEmbeddingFunction(),
    )
    engine.ingest(force=False)
    engine.close()

    # 创建新引擎访问同一数据，应能正常工作
    engine2 = RAGEngine(
        workspace=str(tmp_path),
        knowledge_base=error_dataset["kb_dir"],
        embedding_function=FakeEmbeddingFunction(),
    )
    try:
        results = engine2.search("Python", top_k=3)
        assert len(results) > 0, "close 后新引擎应能访问数据"
    finally:
        engine2.close()


# ---------------------------------------------------------------------------
# 清理 fixture：恢复权限
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _restore_permissions():
    """测试后恢复文件权限，避免 tmp_path 清理失败。"""
    yield
    # tmp_path 由 pytest 自动清理，但无权限文件可能导致清理失败
    # 这里不做特殊处理，因为 tmp_path 是临时目录，pytest 会处理
