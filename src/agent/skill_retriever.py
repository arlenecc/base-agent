"""Skill retriever: hybrid (vector + BM25) retrieval over SKILL.md files.

Scans the ``skills`` directory (recursively) for ``SKILL.md`` files, parses
their YAML frontmatter (``name`` / ``description`` / ``tags``) and body
sections (``When to Use This Skill``, ``When NOT to Use``, examples), then
builds a dedicated LanceDB table for retrieval.

Retrieval combines:
  * vector search over the ``description`` embedding and the ``examples``
    embedding, fused 0.6 : 0.4;
  * BM25 full-text search over a combined ``name + tags + description +
    examples`` text column.

The union of vector candidates (3) and BM25 candidates (3) is de-duplicated,
filtered by a minimum similarity of 0.5, then the remaining candidates'
``name + tags + description`` are handed to the LLM for a final pick.

This module is independent of the RAG ``VectorStore`` (which stores document
chunks); it uses its own LanceDB table and the shared FastEmbed embedding
function (nomic-embed-text-v1.5).
"""
from __future__ import annotations

import logging
import math
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SKILL_FILENAME = "SKILL.md"
_SKILL_TABLE_NAME = "skills"

# Similarity fusion weights for description vs examples embeddings.
_DESC_WEIGHT = 0.6
_EXAMPLES_WEIGHT = 0.4
# Minimum similarity for a candidate to survive filtering.
_MIN_SIMILARITY = 0.5
# Number of candidates from each channel (vector / BM25).
_VECTOR_TOP_K = 3
_BM25_TOP_K = 3
# Max chars of description/examples kept for embedding and combined text.
# SKILL.md body sections (especially "Examples") can be tens of KB; embedding
# them whole is wasteful (the model truncates at ~2k tokens anyway) and makes
# jieba tokenization + LanceDB storage balloon. Cap to keep memory O(1) per skill.
_MAX_DESC_CHARS = 2000
_MAX_EXAMPLES_CHARS = 3000
_MAX_COMBINED_CHARS = 6000

# 扫描时单个 SKILL.md 的读取上限。技能目录里可能混进几百 MB 的语料/模型文件
# （用户误放，或符号链接指错），无上限的 f.read() 会把它整个读进内存，
# 之后才被 _truncate 砍掉 —— 峰值内存已经爆了。与其他工具保持同一量级。
_MAX_SKILL_FILE_CHARS = 200_000

# Directories to skip while scanning (tests, caches, etc.).
_SKIP_DIR_NAMES = {"tests", "test", "__pycache__", ".git", ".DS_Store", "node_modules"}

# Body section headings that describe "when to use / when NOT to use".
_WHEN_TO_USE_HEADINGS = (
    "when to use this skill",
    "when to use",
    "use this skill when",
    "when you should use",
)
_WHEN_NOT_TO_USE_HEADINGS = (
    "when not to use",
    "when not to use this skill",
    "do not use",
    "don't use",
)
_EXAMPLES_HEADINGS = (
    "examples",
    "example",
    "usage examples",
)


# ---------------------------------------------------------------------------
# SKILL.md parsing
# ---------------------------------------------------------------------------

_BLOCK_SCALAR_RE = re.compile(r"^([>|])([+-]?)$")


def _parse_frontmatter(text: str) -> Dict[str, Any]:
    """Parse the YAML frontmatter (``---`` delimited block) at the top.

    The frontmatter is simple key/value YAML; we parse top-level scalar fields
    (``name``, ``description``, ``license``, ``compatibility``,
    ``allowed-tools``) plus a ``tags`` list if present.  Returns a flat dict.

    支持块标量（``description: >-`` / ``|``）：社区与官方 SKILL.md 大量用
    折叠块写多行 description，只按单行解析会把值取成 ">-"，技能照样进索引
    但描述是垃圾 —— 语义检索永远召回不到它。
    """
    result: Dict[str, Any] = {}
    # BOM / 前导空白会让 startswith("---") 判false：整个 frontmatter 不可见，
    # name 取空 → 技能被静默丢弃（Windows 编辑器保存的 SKILL.md 必带 BOM）。
    if not text.lstrip("\ufeff \t\r\n").startswith("---"):
        return result
    text = text.lstrip("\ufeff \t\r\n")

    end = text.find("\n---", 3)
    if end == -1:
        return result
    fm_text = text[3:end]

    tags: List[str] = []
    lines = fm_text.splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"^(\s*)(\w[\w-]*)\s*:\s*(.*)$", lines[i].rstrip())
        if not m:
            i += 1
            continue
        indent, key, val = len(m.group(1)), m.group(2), m.group(3).strip()
        if indent > 0:
            # 嵌套块（metadata: 之类）内的行，不作为顶层字段解析。
            i += 1
            continue

        block = _BLOCK_SCALAR_RE.match(val)
        if block:
            kind = block.group(1)
            i += 1
            collected: List[str] = []
            while i < len(lines):
                ln = lines[i]
                if not ln.strip():
                    collected.append("")
                    i += 1
                    continue
                if not ln[:1].isspace():
                    break  # 缩进行结束，回到顶层
                collected.append(ln.strip())
                i += 1
            # '>' 折叠：换行变空格；'|' 字面：保留换行。
            val = " ".join(x for x in collected if x) if kind == ">" else "\n".join(collected).strip()
        else:
            i += 1

        # strip surrounding quotes
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key == "tags":
            tags.extend(_parse_list_field(val))
        elif key in ("name", "description", "license", "compatibility", "allowed-tools"):
            result[key] = val
    if tags:
        result["tags"] = tags
    return result


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to ``max_chars`` (no-op if shorter)."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _sql_quote(value: str) -> str:
    """Escape a value for use inside a single-quoted SQL/LanceDB literal.

    Skill directory names come from the filesystem (and, indirectly, from LLM
    output), so they may contain a single quote.  Interpolating them raw makes
    the filter expression malformed — and injectable.
    """
    return str(value).replace("'", "''")


def _normalize_vec(vec: List[float]) -> List[float]:
    """Normalize a vector to unit length (for cosine similarity via L2 distance)."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return [float(x) for x in vec]
    return [float(x) / norm for x in vec]


def _parse_list_field(val: str) -> List[str]:
    """Parse a YAML list field value into a list of strings.

    Handles both flow style (``[a, b, c]``) and bare comma-separated strings
    (``a, b``), stripping brackets, quotes, and empty entries.
    """
    s = val.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    items = re.split(r"[,，]", s)
    out = []
    for it in items:
        it = it.strip().strip("\"'")
        if it:
            out.append(it)
    return out


def _extract_section(text: str, headings: Tuple[str, ...]) -> str:
    """Extract the body text under the first matching ``##`` heading.

    Returns the section text (until the next ``##`` heading), or "" if the
    heading is absent.
    """
    lines = text.splitlines()
    capture = False
    out: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            title = stripped[3:].strip().lower()
            if capture:
                break  # reached the next section
            if title in headings:
                capture = True
                continue
        if capture:
            out.append(line)
    return "\n".join(out).strip()


class SkillMetadata:
    """Parsed metadata for a single SKILL.md."""

    __slots__ = ("name", "description", "tags", "dir", "when_to_use",
                 "when_not_to_use", "examples", "combined_text", "fingerprint")

    def __init__(self, name: str, description: str, tags: List[str], dir: str,
                 when_to_use: str, when_not_to_use: str, examples: str,
                 fingerprint: str = ""):
        self.name = name
        self.description = description
        self.tags = tags
        self.dir = dir
        self.when_to_use = when_to_use
        self.when_not_to_use = when_not_to_use
        self.examples = examples
        self.fingerprint = fingerprint
        self.combined_text = self._build_combined()

    def _build_combined(self) -> str:
        parts = [self.name]
        if self.tags:
            parts.append(" ".join(self.tags))
        if self.description:
            parts.append(_truncate(self.description, _MAX_DESC_CHARS))
        if self.when_to_use:
            parts.append(_truncate(self.when_to_use, _MAX_EXAMPLES_CHARS))
        if self.examples:
            parts.append(_truncate(self.examples, _MAX_EXAMPLES_CHARS))
        return _truncate("\n".join(parts), _MAX_COMBINED_CHARS)

    def summary_text(self) -> str:
        """Short text (name + tags + description) handed to the LLM for final pick."""
        tag_str = ", ".join(self.tags) if self.tags else ""
        return f"name: {self.name}\ntags: {tag_str}\ndescription: {self.description}".strip()


def parse_skill_md(text: str, dir: str) -> Optional[SkillMetadata]:
    """Parse a SKILL.md file's text into a :class:`SkillMetadata`.

    Returns None if the file has no ``name`` (not a valid skill).
    """
    fm = _parse_frontmatter(text)
    name = (fm.get("name") or "").strip()
    if not name:
        return None
    description = (fm.get("description") or "").strip()
    tags = fm.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in re.split(r"[,，]", tags) if t.strip()]

    # description may already contain "when to use / when not to use" clauses;
    # keep it whole. Body sections augment it.
    when_to_use = _extract_section(text, _WHEN_TO_USE_HEADINGS)
    when_not_to_use = _extract_section(text, _WHEN_NOT_TO_USE_HEADINGS)
    examples = _extract_section(text, _EXAMPLES_HEADINGS)

    # Content fingerprint for incremental sync (dir unchanged but content edited).
    import hashlib
    fingerprint = hashlib.md5(text.encode("utf-8")).hexdigest()

    return SkillMetadata(
        name=name,
        description=description,
        tags=tags,
        dir=dir,
        when_to_use=when_to_use,
        when_not_to_use=when_not_to_use,
        examples=examples,
        fingerprint=fingerprint,
    )


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

def _walk_skill_dirs(root: str):
    """Yield ``(dirpath, dirnames, filenames)`` like ``os.walk``, but following
    directory symlinks.

    ``os.walk`` 默认 ``followlinks=False``：符号链接的技能目录会被列进
    ``dirs`` 但不会被递归进去，里面的 SKILL.md 永远扫不到。而把外部技能
    ``ln -s`` 挂进 skills/ 是最常见的用法（例如链接 ~/.claude/skills/*），
    旧行为等于这些技能不存在。

    跟随链接必须自己防环：自引用/互引用的链接会让递归无限下去，这里用
    realpath 集合去重（同一个目录不会被第二次展开）。
    """
    visited = set()
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            real = os.path.realpath(current)
        except OSError:
            continue
        if real in visited:
            continue
        visited.add(real)
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        dirs: List[str] = []
        files: List[str] = []
        for e in entries:
            try:
                if e.is_dir(follow_symlinks=True):
                    dirs.append(e.name)
                elif e.is_file():
                    files.append(e.name)
            except OSError:
                continue
        # 先 yield 再压栈：调用方（scan/_dir_mtime）会就地裁剪 dirs 做剪枝，
        # 裁剪结果必须能被下面的入栈逻辑看到。
        yield current, dirs, files
        for name in reversed(sorted(dirs)):
            stack.append(os.path.join(current, name))


class SkillScanner:
    """Recursively scan a skills directory for SKILL.md files.

    Thread-safe; debounced like SkillIndex — a rescan is triggered when the
    directory mtime changes or the interval elapses.
    """

    def __init__(self, skills_dir: str, scan_interval: float = 300.0):
        self._dir = skills_dir
        self._scan_interval = scan_interval
        self._lock = threading.Lock()
        self._skills: List[SkillMetadata] = []
        self._last_scan = 0.0
        self._last_mtime = 0.0
        # _dir_mtime 的短 TTL 缓存：它是 O(目录数) 的全树 walk（只 stat 目录
        # 与 SKILL.md），5 分钟扫描间隔内**每次检索**都会因 _should_scan 调它
        # 一次、scan() 后再调一次 —— 技能树上千时每次查询都是几十上百个
        # syscall。外部写入有 mark_stale() 显式标脏兜底，60s 的探测缓存不会
        # 漏掉真实变更（最坏延迟 60s 被检索发现，可接受）。
        self._mtime_cache: float = -1.0
        self._mtime_cache_at: float = 0.0
        self._mtime_cache_ttl: float = 60.0

    def _dir_mtime(self, use_cache: bool = True) -> float:
        """Most recent mtime of any SKILL.md (plus its parent dirs).

        We deliberately stat only directories and ``SKILL.md`` files: walking
        and stat-ing *every* file (the old behaviour) is O(files) syscalls.
        ``use_cache=True``（默认）在 TTL 内复用上次结果 —— 每次检索都会经过
        _should_scan，无缓存时它就是一次全树 walk。
        """
        now = time.time()
        if use_cache and self._mtime_cache >= 0 and (
            now - self._mtime_cache_at < self._mtime_cache_ttl
        ):
            return self._mtime_cache
        max_mtime = 0.0
        if not os.path.isdir(self._dir):
            self._mtime_cache, self._mtime_cache_at = 0.0, now
            return 0.0
        try:
            for root, dirs, files in _walk_skill_dirs(self._dir):
                dirs[:] = [d for d in dirs if d not in _SKIP_DIR_NAMES and not d.startswith(".")]
                try:
                    mtime = os.stat(root).st_mtime
                    if mtime > max_mtime:
                        max_mtime = mtime
                except OSError:
                    pass
                if _SKILL_FILENAME in files:
                    try:
                        mtime = os.stat(os.path.join(root, _SKILL_FILENAME)).st_mtime
                        if mtime > max_mtime:
                            max_mtime = mtime
                    except OSError:
                        pass
        except OSError:
            pass
        self._mtime_cache, self._mtime_cache_at = max_mtime, now
        return max_mtime

    def _should_scan(self) -> bool:
        if self._last_scan == 0.0:
            return True
        if time.time() - self._last_scan > self._scan_interval:
            return True
        try:
            if self._dir_mtime() != self._last_mtime:
                return True
        except OSError:
            pass
        return False

    def scan(self, force: bool = False) -> List[SkillMetadata]:
        """Scan the skills directory and return the current skill list."""
        with self._lock:
            if not force and not self._should_scan():
                return list(self._skills)
            skills: List[SkillMetadata] = []
            if os.path.isdir(self._dir):
                for root, dirs, files in _walk_skill_dirs(self._dir):
                    dirs[:] = [d for d in dirs if d not in _SKIP_DIR_NAMES and not d.startswith(".")]
                    if _SKILL_FILENAME not in files:
                        continue
                    path = os.path.join(root, _SKILL_FILENAME)
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            # 只读到上限+1：技能目录里可能混进几百 MB 的语料/
                            # 模型文件（用户误放、或符号链接指错），整本读进来会
                            # 直接吃满内存，之后才被 _truncate 砍掉。
                            content = f.read(_MAX_SKILL_FILE_CHARS + 1)
                    except (OSError, UnicodeDecodeError) as e:
                        logger.warning("skill_retriever: failed to read %s: %s", path, e)
                        continue
                    if len(content) > _MAX_SKILL_FILE_CHARS:
                        content = content[:_MAX_SKILL_FILE_CHARS]
                    # dir is relative to the skills root, or absolute if not under it.
                    rel = os.path.relpath(root, self._dir) if root.startswith(self._dir) else root
                    meta = parse_skill_md(content, rel)
                    if meta:
                        skills.append(meta)
            self._skills = skills
            self._last_scan = time.time()
            # 刚扫完整棵树，mtime 探测缓存必须刷新（扫描本身会改不了目录
            # mtime，但缓存值可能已过期 → 下一次检索立刻再 walk 一遍全树）。
            self._last_mtime = self._dir_mtime(use_cache=False)
            logger.info("skill_retriever: scanned %d skills from %s", len(skills), self._dir)
            return list(self._skills)

    def list_skills(self) -> List[SkillMetadata]:
        return self.scan()


# ---------------------------------------------------------------------------
# Vector store (LanceDB) for skills
# ---------------------------------------------------------------------------

class SkillVectorStore:
    """LanceDB-backed store for skill embeddings + BM25 text.

    Columns:
      * ``dir`` (str)            — SKILL.md's directory (relative)
      * ``name`` (str)           — skill name
      * ``tags`` (str)           — comma-joined tags
      * ``description`` (str)    — frontmatter description
      * ``desc_vec`` (vector)    — description embedding
      * ``examples_vec`` (vector)— examples embedding
      * ``combined`` (str)       — name+tags+description+examples combined text
      * ``combined_fts`` (str)   — jieba-tokenized combined (for BM25)
    """

    def __init__(self, db_path: str, embedding_function=None):
        self._db_path = db_path
        self._ef = embedding_function
        self._db = None
        self._table = None
        # RLock：build()/sync() 持锁期间会调用 _get_db()/_get_table()，
        # 普通 Lock 会让这些路径自锁死（与 VectorStore._documents_lock 同类）。
        self._lock = threading.RLock()

    def _get_ef(self):
        # 双检锁：并发首次调用（两个 skill_semantic_search 线程）会各自
        # get_or_create_embedding_function() —— 该函数内部有锁，模型不会
        # 加载两份，但这里的赋值竞态会让一个实例拿到后被另一个覆盖。
        if self._ef is None:
            with self._lock:
                if self._ef is None:
                    from .rag.vector_store import get_or_create_embedding_function
                    self._ef = get_or_create_embedding_function()
        return self._ef

    def _get_db(self):
        # 双检锁：两个线程并发首次调用会各建一个 lancedb.connect()，
        # 后写者覆盖 _db，先建的那个连接（含文件句柄）永不关闭。
        if self._db is None:
            with self._lock:
                if self._db is None:
                    import lancedb
                    os.makedirs(self._db_path, exist_ok=True)
                    self._db = lancedb.connect(self._db_path)
        return self._db

    def _get_table(self):
        # 双检锁：check-then-act 竞态会让两个线程各自 open_table，
        # 其中一个句柄被丢弃（LanceDB 表句柄持有版本快照与缓存）。
        if self._table is None:
            with self._lock:
                if self._table is None:
                    db = self._get_db()
                    if _SKILL_TABLE_NAME in db.table_names():
                        self._table = db.open_table(_SKILL_TABLE_NAME)
                    else:
                        self._table = None
        return self._table

    def _embed(self, texts: List[str]) -> List[List[float]]:
        ef = self._get_ef()
        return ef.embed_documents(texts)

    def build(self, skills: List[SkillMetadata], batch_size: int = 8) -> int:
        """Build/rebuild the skills table from the given skill list.

        Drops and recreates the table, then processes skills in **small
        batches** (default 8): embed ``batch_size`` texts per forward pass and
        write each batch to LanceDB immediately.  This keeps peak memory at
        O(batch_size) instead of O(total) — critical for thousands of skills
        (the previous whole-list approach ballooned memory to tens of GB and
        froze the OS).
        """
        from .rag.vector_store import _tokenize_for_fts

        if not skills:
            return 0

        db = self._get_db()

        # Probe the embedding dimension BEFORE dropping anything.
        # 旧实现先 drop 表再探测维度：一旦 embedding 失败（模型未下载、
        # 网络故障），已有的可用索引已经被删掉了，功能直接归零。
        try:
            probe = self._embed([skills[0].description or skills[0].name])
        except Exception as e:
            logger.warning("skill_retriever: embedding probe failed, keeping existing index: %s", e)
            return 0
        dim = len(probe[0]) if probe else 0
        if dim <= 0:
            return 0

        with self._lock:
            import pyarrow as pa
            vec_type = pa.list_(pa.float32(), dim)
            schema = self._table_schema(vec_type)

            # 先建**临时表**再替换：旧实现一上来就 drop 旧表，若第一批 embed 之后
            # 失败（模型下载中断 / 磁盘错误），旧表已经没了、新表只写了一部分，
            # 且 self._table 未赋值 —— 下次 _get_table() 打开这张半张表，
            # 技能检索静默只剩一部分结果且不会有任何报错。
            staging = f"{_SKILL_TABLE_NAME}__staging"
            if staging in db.table_names():
                db.drop_table(staging)
            table = db.create_table(staging, [], schema=schema)
            try:
                total = 0
                for i in range(0, len(skills), batch_size):
                    batch = skills[i:i + batch_size]
                    rows = self._embed_batch(batch)
                    table.add(rows)
                    total += len(rows)
                    logger.info(
                        "skill_retriever: built %d/%d skills",
                        total, len(skills),
                    )
                # 全部写完才切换：drop 旧表 → 用 staging 的 arrow 数据建新表。
                # LanceDB OSS 不支持 rename_table，所以走「复制数据」这条路；
                # 技能表只有几十到几百行，一次 to_arrow() 的代价可以接受。
                payload = table.to_arrow()
                if _SKILL_TABLE_NAME in db.table_names():
                    db.drop_table(_SKILL_TABLE_NAME)
                db.create_table(_SKILL_TABLE_NAME, data=payload, schema=schema)
                del payload
                db.drop_table(staging)
            except Exception:
                # 失败必须清理半张 staging 表：否则下次 build 又要先 drop 它，
                # 且它会一直占着磁盘空间。旧表在切换前始终完好，检索不受影响。
                try:
                    db.drop_table(staging)
                except Exception:
                    pass
                raise
            self._table = db.open_table(_SKILL_TABLE_NAME)
            self._ensure_fts_index(self._table)
            logger.info("skill_retriever: built skills table with %d rows", total)
            return total

    @staticmethod
    def _table_schema(vec_type):
        import pyarrow as pa
        return pa.schema([
            pa.field("dir", pa.string()),
            pa.field("name", pa.string()),
            pa.field("tags", pa.string()),
            pa.field("description", pa.string()),
            pa.field("desc_vec", vec_type),
            pa.field("examples_vec", vec_type),
            pa.field("combined", pa.string()),
            pa.field("combined_fts", pa.string()),
            pa.field("fingerprint", pa.string()),
        ])

    def _embed_batch(self, batch: List[SkillMetadata]) -> List[Dict[str, Any]]:
        """Embed one batch of skills and build their table rows."""
        from .rag.vector_store import _tokenize_for_fts

        # 截断后再 embedding，避免超长 SKILL.md 导致内存/耗时爆炸。
        descriptions = [_truncate(s.description or s.name, _MAX_DESC_CHARS) for s in batch]
        examples = [
            _truncate(s.examples or s.when_to_use or s.description or s.name, _MAX_EXAMPLES_CHARS)
            for s in batch
        ]
        desc_vecs = self._embed(descriptions)
        examples_vecs = self._embed(examples)

        rows = []
        for s, dv, ev in zip(batch, desc_vecs, examples_vecs):
            rows.append({
                "dir": s.dir,
                "name": s.name,
                "tags": ", ".join(s.tags),
                "description": s.description,
                "desc_vec": _normalize_vec(dv),
                "examples_vec": _normalize_vec(ev),
                "combined": s.combined_text,
                "combined_fts": _tokenize_for_fts(s.combined_text),
                "fingerprint": s.fingerprint,
            })
        return rows

    def sync(self, skills: List[SkillMetadata], batch_size: int = 8) -> Dict[str, int]:
        """Incrementally sync the table against the scanned skill list.

        Only touches changed items — NO full rebuild:
          * ``added``    — dirs not yet in the table → embed + add.
          * ``updated``  — dirs present but fingerprint changed → delete + re-add.
          * ``removed``  — dirs in the table but absent from scan → delete.

        Returns ``{"added": n, "updated": n, "removed": n}``.
        """
        table = self._get_table()
        if table is None:
            # No table yet — fall back to a full build.
            self.build(skills, batch_size=batch_size)
            return {"added": len(skills), "updated": 0, "removed": 0}

        # Current table contents: dir -> fingerprint.
        # 必须在 LanceDB 层做列裁剪：表内还有两条 768 维向量列，旧实现
        # ``to_arrow().select([...])`` 是先物化**全部列**再 select——数千技能
        # 仍是数百 MB 的 Arrow 拷贝，注释声称「只取两列」并未兑现。
        # search().limit(N).select(cols) 会把列裁剪下推到存储层。
        try:
            n_rows = table.count_rows()
            existing_rows = (
                table.search()
                .limit(max(n_rows, 1))
                .select(["dir", "fingerprint"])
                .to_arrow()
                .to_pylist()
            )
        except Exception as e:
            logger.warning("skill_retriever: sync failed to read table: %s", e)
            self.build(skills, batch_size=batch_size)
            return {"added": len(skills), "updated": 0, "removed": 0}

        existing: Dict[str, str] = {}
        for r in existing_rows:
            d = r.get("dir", "")
            if d:
                existing[d] = r.get("fingerprint", "")

        scanned: Dict[str, SkillMetadata] = {s.dir: s for s in skills}

        added_dirs = [d for d in scanned if d not in existing]
        updated_dirs = [d for d in scanned if d in existing and existing[d] != scanned[d].fingerprint]
        removed_dirs = [d for d in existing if d not in scanned]

        with self._lock:
            # 锁内重新取表：上面的读取在锁外，并发的 build()/close() 可能
            # 已经替换或丢弃了句柄（拿着旧句柄写会打到已关闭的连接上）。
            table = self._get_table()
            if table is None:
                # 表在锁内被清掉了 → 直接锁内重建，不再走外面的 build。
                self.build(skills, batch_size=batch_size)
                return {"added": len(skills), "updated": 0, "removed": 0}

            # Remove deleted skills.
            # 目录名来自文件系统/LLM，可能含单引号——直接拼进 SQL 字面量会
            # 产生语法错误甚至注入。转义 '' 后再拼接。
            for d in removed_dirs:
                try:
                    table.delete(f"dir = '{_sql_quote(d)}'")
                except Exception as e:
                    logger.warning("skill_retriever: delete %s failed: %s", d, e)

            # Embed + add new and updated skills in small batches.
            # 更新项**先 add 再按旧 fingerprint 删**：旧实现先 delete 再 add，
            # 中途 embedding/add 失败时该技能从索引里永久消失且无回滚。
            # 先 add 失败 → 旧行还在（功能不丢）；成功后删旧行失败 → 最多
            # 短暂重复行（检索按 dir 字典合并，重复行无害）。
            # 不能按 dir 删旧行——那会把刚 add 的新行一起删掉。
            to_add = [scanned[d] for d in (added_dirs + updated_dirs)]
            for i in range(0, len(to_add), batch_size):
                batch = to_add[i:i + batch_size]
                rows = self._embed_batch(batch)
                table.add(rows)

            # 新行落盘后再清理更新项的旧行（按旧 fingerprint 精确匹配）。
            for d in updated_dirs:
                old_fp = existing.get(d, "")
                try:
                    table.delete(
                        f"dir = '{_sql_quote(d)}' AND fingerprint = '{_sql_quote(old_fp)}'"
                    )
                except Exception as e:
                    logger.debug("skill_retriever: stale-row cleanup %s failed: %s", d, e)

            # 数据增删后重建 FTS（BM25）索引，确保新增/删除项参与全文检索。
            if added_dirs or updated_dirs or removed_dirs:
                self._ensure_fts_index(table)

        logger.info(
            "skill_retriever: synced +%d ~%d -%d (total %d)",
            len(added_dirs), len(updated_dirs), len(removed_dirs), len(scanned),
        )
        return {"added": len(added_dirs), "updated": len(updated_dirs), "removed": len(removed_dirs)}

    def _ensure_fts_index(self, table) -> None:
        """Create/refresh the BM25 full-text index on the ``combined_fts`` column.

        必须 ``replace=True``：索引已存在时重复创建会抛
        ``Index name 'combined_fts_idx' already exists``，于是每次 sync（新增/
        修改/删除技能）都重建失败，索引停在首次建表时的快照上。
        参数与 RAG 侧 ``VectorStore._ensure_fts_index`` 保持一致（jieba 预分词
        列 + simple tokenizer，不 stem、不做 ascii folding，否则中文词形被破坏）。
        """
        fts_kwargs = dict(
            base_tokenizer="simple",
            lower_case=True,
            stem=False,
            remove_stop_words=False,
            ascii_folding=False,
            with_position=True,
        )
        try:
            from lancedb.index import FTS
            table.create_index("combined_fts", config=FTS(**fts_kwargs), replace=True)
        except TypeError:
            # lancedb 0.25.x 的 create_index 无 config 参数。
            try:
                table.create_fts_index(
                    "combined_fts", use_tantivy=False, replace=True, **fts_kwargs
                )
            except Exception as e:
                logger.warning("skill_retriever: BM25 index failed (fallback to vector only): %s", e)
        except Exception as e:
            logger.warning("skill_retriever: BM25 index failed (fallback to vector only): %s", e)

    def search_vector(self, query: str, top_k: int = _VECTOR_TOP_K) -> List[Dict[str, Any]]:
        """Vector search. Returns [{dir, name, tags, description, score}]."""
        table = self._get_table()
        if table is None:
            return []
        ef = self._get_ef()
        qv = ef.embed_query(query)
        # 归一化查询向量，与建表时的归一化向量保持一致（L2 距离转余弦相似度）。
        norm = math.sqrt(sum(x * x for x in qv))
        if norm > 0:
            qv = [x / norm for x in qv]
        try:
            desc_res = table.search(qv, vector_column_name="desc_vec").limit(top_k).to_list()
        except Exception as e:
            logger.warning("skill_retriever: desc vector search failed: %s", e)
            desc_res = []
        try:
            ex_res = table.search(qv, vector_column_name="examples_vec").limit(top_k).to_list()
        except Exception as e:
            logger.warning("skill_retriever: examples vector search failed: %s", e)
            ex_res = []

        # Fuse the two vector results by skill dir, weighting 0.6:0.4.
        fused: Dict[str, Dict[str, Any]] = {}
        for r in desc_res:
            d = r.get("dir", "")
            if not d:
                continue
            fused[d] = self._row_to_dict(r)
        for r in ex_res:
            d = r.get("dir", "")
            if not d:
                continue
            ex_score = self._distance_to_similarity(r.get("_distance"))
            if d in fused:
                # Both channels hit the same skill: weighted fusion.
                desc_score = fused[d]["score"]
                fused[d]["score"] = _DESC_WEIGHT * desc_score + _EXAMPLES_WEIGHT * ex_score
            else:
                entry = self._row_to_dict(r)
                entry["score"] = ex_score
                fused[d] = entry

        results = list(fused.values())
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    @staticmethod
    def _distance_to_similarity(distance) -> float:
        """Convert a normalized L2 distance to a cosine similarity in [0,1].

        归一化后 ||a-b||² = 2 - 2·cos(θ)，故 cos(θ) = 1 - d²/2。

        ``distance`` 为 None（行里没有 ``_distance`` 字段）时返回 **0.0**：
        旧实现按 0.0 距离算成满分 1.0，任何缺字段的行都会拿到完美相似度、
        必然通过 _MIN_SIMILARITY 阈值并排到最前，排序随之失效。
        """
        if distance is None:
            return 0.0
        try:
            d = float(distance)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, 1.0 - (d * d) / 2.0)

    def _row_to_dict(self, r: Dict[str, Any]) -> Dict[str, Any]:
        score = self._distance_to_similarity(r.get("_distance"))
        return {
            "dir": r.get("dir", ""),
            "name": r.get("name", ""),
            "tags": r.get("tags", ""),
            "description": r.get("description", ""),
            "score": score,
        }

    def search_bm25(self, query: str, top_k: int = _BM25_TOP_K) -> List[Dict[str, Any]]:
        """BM25 full-text search over the combined text. Returns dict rows."""
        table = self._get_table()
        if table is None:
            return []
        from .rag.vector_store import _tokenize_for_fts
        tokenized = _tokenize_for_fts(query)
        try:
            results = (
                table.search(tokenized, query_type="fts", fts_columns=["combined_fts"])
                .limit(top_k)
                .to_list()
            )
        except Exception as e:
            logger.warning("skill_retriever: BM25 search failed: %s", e)
            return []
        out = []
        for r in results:
            out.append({
                "dir": r.get("dir", ""),
                "name": r.get("name", ""),
                "tags": r.get("tags", ""),
                "description": r.get("description", ""),
                "bm25_score": float(r.get("_score", 0.0)),
            })
        return out

    def close(self) -> None:
        """Drop the table handle and close the LanceDB connection.

        旧实现只把引用置 None，从不调用 ``db.close()``。LanceDB 的连接对象
        持有底层文件句柄与对象存储缓存，反复「应用配置」重建 retriever 会让
        这些资源一直堆积到进程退出。
        """
        self._table = None
        db = self._db
        self._db = None
        if db is not None:
            try:
                close_fn = getattr(db, "close", None)
                if callable(close_fn):
                    close_fn()
            except Exception as e:
                logger.debug("skill_retriever: db close failed: %s", e)


# ---------------------------------------------------------------------------
# Retriever (hybrid search + LLM final pick)
# ---------------------------------------------------------------------------

class SkillRetriever:
    """High-level hybrid skill retriever.

    Orchestrates scanning, indexing, hybrid retrieval, threshold filtering,
    and (optionally) an LLM final pick among candidates.
    """

    def __init__(self, skills_dir: str, db_path: str, embedding_function=None):
        self._scanner = SkillScanner(skills_dir)
        self._store = SkillVectorStore(db_path, embedding_function=embedding_function)
        self._skills_dir = skills_dir        # 索引同步必须在锁内完成：skill_semantic_search 在并发白名单里，
        # 两个线程同时进 sync() 会读到同一份 existing、各写一份 → 重复行；
        # 同时进 build() 则第二个会把第一个刚建好的表 drop 掉重算。
        self._index_lock = threading.RLock()
        # 外部（skill_save / UI）直接写了 SKILL.md：此时目录 mtime 可能还没
        # 变到 _should_scan 能察觉，必须显式标脏，否则新技能永远进不了索引。
        self._dirty = False

    @property
    def skills_dir(self) -> str:
        """本检索器扫描的技能根目录（``<workspace>/skills``）。

        调用方（如 skill_load 标注「技能目录是符号链接」）必须用**这个**
        目录做基线——用别的目录（如 .agent/skills）会把每个正常技能都误判
        成外部链接。
        """
        return self._skills_dir

    def mark_stale(self) -> None:
        """外部写入技能目录后调用：强制下一次检索前重新同步索引。"""
        self._dirty = True

    def ensure_indexed(self) -> None:
        """Sync the vector table against the skills directory (incremental).

        Idempotent + incremental:
          * table missing (first run) → full build;
          * table exists + directory unchanged → no-op (open table only);
          * directory changed or ``mark_stale()`` → ``sync()``: only add new
            skills, update edited skills (fingerprint differs), and remove
            deleted skills — never a full rebuild.
        """
        with self._index_lock:
            table = self._store._get_table()
            if table is None:
                # First run: table missing → build.
                skills = self._scanner.scan(force=True)
                self._store.build(skills)
                self._dirty = False
                return
            # 表已存在：目录变了、或被显式标脏时才同步。
            #
            # 注意不能只靠 _should_scan()：skill_save 写完 SKILL.md 后调用
            # scan(force=True) 会把 _last_scan/_last_mtime 刷新掉，之后
            # _should_scan() 恒为 False → 刚保存的技能永远不会被嵌入。
            if self._dirty or self._scanner._should_scan():
                skills = self._scanner.scan(force=True)
                self._store.sync(skills)
                self._dirty = False

    def retrieve(self, query: str, llm=None) -> Dict[str, Any]:
        """Hybrid retrieve for a query.

        Returns:
            {"candidates": [...], "picked": <dir|None>, "reason": str}
        ``picked`` is set when ``llm`` is provided and it selects a skill;
        otherwise ``picked`` is None and callers use ``candidates`` directly.

        Fusion semantics:
          * Vector results carry a cosine-similarity ``score`` in [0,1] (the
            description/examples channels already fused 0.6:0.4).
          * BM25 results carry a raw ``bm25_score``; for a unified, comparable
            ranking we also attach a cosine similarity as ``score`` (reused from
            the vector channel when the same skill matched there, otherwise
            looked up on demand).
          * De-duplicated by ``dir``; candidates with ``score < 0.5`` dropped.
        """
        self.ensure_indexed()

        vec_results = self._store.search_vector(query, top_k=_VECTOR_TOP_K)
        bm25_results = self._store.search_bm25(query, top_k=_BM25_TOP_K)

        # De-duplicate by dir. Vector channel is the authoritative score.
        merged: Dict[str, Dict[str, Any]] = {}
        for r in vec_results:
            d = r.get("dir", "")
            if not d:
                continue
            merged[d] = dict(r)

        # BM25 results: keep raw bm25 score; attach a vector similarity as the
        # unified ``score``. If the skill already matched in the vector channel,
        # reuse that score; otherwise look it up via a vector search.
        bm25_by_dir = {r.get("dir", ""): r for r in bm25_results if r.get("dir")}
        # 只为「仅 BM25 命中」的候选准备查询向量并复用。embedding 失败时
        # 短路：不再对每个候选重试注定失败的 ONNX 调用，BM25 命中本身已足以
        # 保留候选，score 记 0 让其自然排到后面。
        query_vec = None
        embed_dead = False
        for d, bm in bm25_by_dir.items():
            if d in merged:
                merged[d]["bm25_score"] = bm["bm25_score"]
                continue
            if embed_dead:
                sim = 0.0
            elif query_vec is None:
                query_vec = self._embed_query_vector(query)
                if query_vec is None:
                    embed_dead = True
                    sim = 0.0
                else:
                    sim = self._lookup_similarity(query, d, qv=query_vec)
            else:
                sim = self._lookup_similarity(query, d, qv=query_vec)
            merged[d] = dict(bm)
            merged[d]["bm25_score"] = bm["bm25_score"]
            merged[d]["score"] = sim

        candidates = list(merged.values())
        # Filter by minimum similarity (vector similarity in [0,1]).
        # 但 BM25 命中本身就是关键词证据，不能因为余弦没过阈值就丢掉：缩写 /
        # 专有名词 / 跨语言查询下余弦天然偏低（实测查询 "CI" 对中文描述的技能
        # BM25 命中 0.88、余弦 0.478 —— 旧逻辑会把它整条丢弃，模型只看到
        # 「找不到匹配的技能」）。这类候选保留但分数仍是余弦，排序自然靠后。
        candidates = [
            c for c in candidates
            if c["score"] >= _MIN_SIMILARITY or c.get("bm25_score", 0) > 0
        ]
        candidates.sort(key=lambda x: x["score"], reverse=True)

        if not candidates:
            return {"candidates": [], "picked": None, "reason": "no candidates"}

        picked = None
        reason = "candidates returned"
        if llm is not None:
            picked = self._llm_pick(llm, query, candidates)
            reason = "llm selected" if picked else "llm found no match"

        return {"candidates": candidates, "picked": picked, "reason": reason}

    def _embed_query_vector(self, query: str):
        """Embed + L2-normalize a query, or None if embedding is unavailable."""
        try:
            ef = self._store._get_ef()
            qv = ef.embed_query(query)
        except Exception as e:
            logger.debug("skill_retriever: query embedding failed: %s", e)
            return None
        norm = math.sqrt(sum(x * x for x in qv))
        if norm > 0:
            qv = [x / norm for x in qv]
        return qv

    def _lookup_similarity(self, query: str, dir: str, qv=None) -> float:
        """Return the cosine similarity of ``query`` against skill ``dir``.

        Used to give BM25-only matches a comparable vector score.  Falls back
        to 0.0 if the skill can't be located or embedding fails.

        ``qv`` is an optional pre-normalized query vector.  The caller scores
        every BM25-only candidate through this method, so passing the vector in
        avoids re-running the ONNX query embedding once per candidate.
        """
        table = self._store._get_table()
        if table is None:
            return 0.0
        try:
            if qv is None:
                ef = self._store._get_ef()
                qv = ef.embed_query(query)
                norm = math.sqrt(sum(x * x for x in qv))
                if norm > 0:
                    qv = [x / norm for x in qv]
            res = (
                table.search(qv, vector_column_name="desc_vec")
                .where(f"dir = '{_sql_quote(dir)}'")
                .limit(1)
                .to_list()
            )
            if res:
                return self._store._distance_to_similarity(res[0].get("_distance"))
        except Exception as e:
            logger.debug("skill_retriever: lookup_similarity failed for %s: %s", dir, e)
        return 0.0

    def _llm_pick(self, llm, query: str, candidates: List[Dict[str, Any]]) -> Optional[str]:
        """Ask the LLM to pick the best skill from candidates.

        Returns the chosen skill's ``dir`` (or None if none match).
        """
        options = "\n\n".join(
            f"[{i}] dir={c['dir']}\n{c['description'][:300]}"
            for i, c in enumerate(candidates)
        )
        prompt = (
            "Given the task and candidate skills, pick the single most relevant "
            "skill, or reply 'none' if none fit.\n\n"
            f"Task: {query}\n\nCandidates:\n{options}\n\n"
            "Reply with only the candidate index number (e.g. '0'), or 'none'."
        )
        try:
            events = list(llm.chat_stream([{"role": "user", "content": prompt}],
                                          kv_group="aux"))
            answer = ""
            for ev in events:
                # StreamEvent 是 dataclass，不是 dict —— ev.get() 必然抛
                # AttributeError，被下面的 except 吞掉，于是 _llm_pick 永远
                # 返回 None。按属性访问，同时兼容 dict 形态的事件。
                etype = getattr(ev, "type", None)
                if etype is None and isinstance(ev, dict):
                    etype = ev.get("type")
                if etype != "content":
                    continue
                answer += getattr(ev, "content", None) or (
                    ev.get("content", "") if isinstance(ev, dict) else "")
            answer = answer.strip()
        except Exception as e:
            logger.warning("skill_retriever: LLM pick failed: %s", e)
            return None

        m = re.search(r"\d+", answer)
        if m and "none" not in answer.lower():
            idx = int(m.group())
            if 0 <= idx < len(candidates):
                return candidates[idx]["dir"]
        return None

    def read_skill_dir(self, dir: str) -> Optional[str]:
        """Return the absolute path to a skill's directory (for reading support files).

        ``dir`` 由模型提供 → 拒绝绝对路径与 ``..`` 穿越（这两类不是用户建立
        的链接，而是调用方的恶意输入）。

        技能目录本身是符号链接时**有条件放行**：外部技能常以 ``ln -s`` 挂载，
        realpath 必然落在 skills 目录之外，旧实现的 ``startswith(base)`` 校验
        会把它们一律判为非法。但无条件放行等于把 skill_load 变成任意文件读取
        —— agent 可以先 ``ln -s /etc <ws>/skills/x`` 再 ``skill_load("x",
        "passwd")`` 读到任意文件。因此这里要求链接目标必须长得像一个技能
        目录（顶层就有 SKILL.md / skill.json / prompt.md），``/etc``、``/usr``
        这类系统目录会被直接拒绝。

        调用方可用 :meth:`is_external_skill_dir` 判断返回的目录是否越出了
        skills 根，以便在结果里向用户标注。
        """
        dir = (dir or "").strip()
        if not dir or os.path.isabs(dir) or ".." in dir.replace("\\", "/").split("/"):
            return None
        base = os.path.realpath(self._skills_dir)
        # 到这里为止路径在词法上仍位于 base 之内；唯一能走出 base 的途径就是
        # 链接，只放行「长得像技能目录」的目标。
        resolved = os.path.realpath(os.path.join(base, dir))
        if os.path.isdir(resolved):
            if not (resolved == base or resolved.startswith(base + os.sep)):
                if not self._looks_like_skill_dir(resolved):
                    logger.warning(
                        "skill_retriever: refusing external skill link %r -> %s "
                        "(target is not a skill directory)", dir, resolved)
                    return None
            return resolved
        return None

    @staticmethod
    def _looks_like_skill_dir(path: str) -> bool:
        """外部技能链接目标的形态校验：顶层必须含任一技能入口文件。"""
        for marker in ("SKILL.md", "skill.json", "prompt.md"):
            if os.path.isfile(os.path.join(path, marker)):
                return True
        return False

    @staticmethod
    def is_external_skill_dir(resolved: str, skills_dir: str) -> bool:
        """``resolved`` 是否越出了 skills 根（即本身是符号链接）。"""
        if not resolved:
            return False
        base = os.path.realpath(skills_dir)
        return not (resolved == base or resolved.startswith(base + os.sep))

    def close(self) -> None:
        self._store.close()
