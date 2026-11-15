"""Skill index: scan the .agent/skills/ directory and maintain a JSON
index table for fast skill-search lookup.

Each skill lives in its own subdirectory under .agent/skills/:

    .agent/skills/
    ├── _index.json          ← auto-generated index table
    ├── pdf-extraction/
    │   ├── skill.json        ← metadata (name, description, keywords, tags)
    │   ├── prompt.md         ← the skill's prompt template
    │   └── ... (other support files)
    └── web-scraping/
        ├── skill.json
        └── prompt.md

The index is rebuilt on a timer and on-demand.  It's a flat list of
{name, description, keywords, tags, path, entry} records, kept in a
single JSON file for O(1) loading.

This module is the *index* layer — it reads skill directories and
produces the index.  The SkillSearchTool (in tools/) uses this index
to serve search queries from the LLM.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

from .atomic_io import atomic_write_json, backup_corrupt
from .skills import _keywords_of

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INDEX_FILENAME = "_index.json"
_SKILL_META_FILENAME = "skill.json"
_DEFAULT_ENTRY = "prompt.md"
_SCAN_INTERVAL = 300  # seconds between automatic scans (5 min)
# 单文件读取上限（与 file_ops / shell_run 的 200K 级保持一致）
_MAX_READ_CHARS = 200_000

# Fields we read from skill.json
_REQUIRED_FIELDS = {"name"}


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------------------
# Skill record (plain dict for easy JSON round-trip)
# ---------------------------------------------------------------------------
#
#   {
#     "name":        "PDF Extraction",
#     "description": "Extract structured data from PDF files...",
#     "keywords":    ["pdf", "extract", "ocr"],
#     "tags":        ["document", "data"],
#     "path":        "pdf-extraction",          # relative to skills/ dir
#     "entry":       "prompt.md",                # file to read for the prompt
#   }
#
# ---------------------------------------------------------------------------


class SkillIndex:
    """Scans the skills/ directory and maintains _index.json.

    Thread-safe.  The scan is debounced — if called again within
    _SCAN_INTERVAL seconds and no files have changed, it's a no-op.
    """

    def __init__(self, skills_dir: str):
        self._dir = skills_dir
        self._index_path = os.path.join(skills_dir, _INDEX_FILENAME)
        self._lock = threading.Lock()
        self._index: List[Dict[str, Any]] = []
        self._last_scan = 0.0
        self._last_dir_mtime = 0.0
        self._load()

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load the index from disk (if it exists and is fresh)."""
        if not os.path.exists(self._index_path):
            self._index = []
            return
        try:
            with open(self._index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("index root is not an object")
            skills = data.get("skills", [])
            # 逐条校验：手写坏/旧版本的 _index.json 可能含非 dict 记录，
            # 原样留下会让 search() 的 skill.get(...) / SkillSearchTool 的
            # s["name"] 在后续某次查询时突然炸掉。
            self._index = [r for r in skills if isinstance(r, dict)] \
                if isinstance(skills, list) else []
            ts = data.get("updated_at_ts", 0)
            self._last_scan = ts if isinstance(ts, (int, float)) else 0
            # Sync _last_dir_mtime so _should_scan() doesn't immediately
            # trigger a rescan right after loading a fresh index.
            self._last_dir_mtime = self._dir_mtime()
        except (OSError, json.JSONDecodeError, ValueError, TypeError, AttributeError) as e:
            # 手工写坏 / 版本不兼容的 _index.json 不能让技能检索整体失效：
            # 记日志后按空索引处理，下次 scan() 会重建。
            logger.warning("skill_index: failed to load %s: %s", self._index_path, e)
            backup_corrupt(self._index_path)
            self._index = []
            # 关键：同时把 _last_scan 前移。否则它停留在 0，_should_scan() 恒
            # 为真——每次 search()/list_all() 都全量扫描目录并重写一个下次
            # 仍会加载失败的索引文件（磁盘写放大 + 查询延迟）。
            self._last_scan = time.time()

    def _save(self) -> None:
        """Atomic write of the index."""
        try:
            payload = {
                "version": 1,
                "updated_at": _now_iso(),
                "updated_at_ts": time.time(),
                "skills": self._index,
            }
            atomic_write_json(self._index_path, payload)
        except OSError as e:
            logger.warning("skill_index: failed to save %s: %s", self._index_path, e)

    # ------------------------------------------------------------------
    # scanning
    # ------------------------------------------------------------------

    def _should_scan(self) -> bool:
        """Check if a rescan is needed (directory mtime changed or timeout)."""
        # Always scan if never scanned
        if self._last_scan == 0:
            return True
        # Scan if interval elapsed
        if time.time() - self._last_scan > _SCAN_INTERVAL:
            return True
        # Scan if directory mtime changed
        try:
            current_mtime = self._dir_mtime()
            if current_mtime != self._last_dir_mtime:
                return True
        except OSError:
            pass
        return False

    def _dir_mtime(self) -> float:
        """Get the most recent mtime among skill subdirectories *and* their
        ``skill.json`` metadata files.

        Stat-ing only the subdirectories (the old behaviour) misses the common
        edit case: rewriting ``skill.json``/``prompt.md`` does not change the
        parent directory's mtime, so the index stayed stale for up to
        ``_SCAN_INTERVAL`` seconds after a skill was edited.
        """
        max_mtime = 0.0
        if not os.path.isdir(self._dir):
            return 0.0
        try:
            mtime = os.stat(self._dir).st_mtime
            if mtime > max_mtime:
                max_mtime = mtime
        except OSError:
            pass
        # scandir 本身也要包 try：目录权限变化/被删时 PermissionError 会
        # 一路抛出 _load()（__init__）与 scan()，整个技能检索不可用。
        try:
            entries = list(os.scandir(self._dir))
        except OSError as e:
            logger.debug("skill_index: scandir %s failed: %s", self._dir, e)
            return max_mtime
        for entry in entries:
            if not entry.is_dir() or entry.name.startswith("_"):
                continue
            try:
                mtime = entry.stat(follow_symlinks=False).st_mtime
                if mtime > max_mtime:
                    max_mtime = mtime
                meta = os.path.join(entry.path, _SKILL_META_FILENAME)
                mtime = os.stat(meta).st_mtime
                if mtime > max_mtime:
                    max_mtime = mtime
            except OSError:
                pass
        return max_mtime

    def scan(self, force: bool = False) -> int:
        """Scan skill subdirectories and rebuild the index.

        Returns the number of skills indexed.
        """
        with self._lock:
            if not force and not self._should_scan():
                return len(self._index)

            if not os.path.isdir(self._dir):
                # 目录不存在时同样要更新 _last_scan/_last_dir_mtime，否则
                # _should_scan() 永远为真：每次 search() 都会 makedirs +
                # 重写一遍 _index.json（无谓的磁盘写入，且目录被外部创建后
                # 也感知不到）。
                self._index = []
                self._last_scan = time.time()
                self._last_dir_mtime = 0.0
                return 0

            new_index: List[Dict[str, Any]] = []
            try:
                entries = list(os.scandir(self._dir))
            except OSError as e:
                # 目录不可读：按「没有技能」处理并刷新时间戳，避免每次
                # search() 都重试扫描 + 重写索引文件。
                logger.warning("skill_index: cannot scan %s: %s", self._dir, e)
                self._index = []
                self._last_scan = time.time()
                self._last_dir_mtime = 0.0
                return 0
            for entry in entries:
                if not entry.is_dir():
                    continue
                if entry.name.startswith("_") or entry.name.startswith("."):
                    continue
                skill_dir = entry.path
                meta_path = os.path.join(skill_dir, _SKILL_META_FILENAME)
                if not os.path.isfile(meta_path):
                    # Not a skill directory (no skill.json)
                    continue
                record = self._read_skill_meta(entry.name, skill_dir, meta_path)
                if record:
                    new_index.append(record)

            self._index = new_index
            self._last_scan = time.time()
            self._last_dir_mtime = self._dir_mtime()
            self._save()
            logger.info("skill_index: indexed %d skills from %s", len(new_index), self._dir)
            return len(new_index)

    def _read_skill_meta(
        self, dirname: str, skill_dir: str, meta_path: str
    ) -> Optional[Dict[str, Any]]:
        """Read skill.json from a skill directory."""
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if not isinstance(meta, dict):
                return None
            # 字段逐个校验类型：手写坏的 skill.json（name 是数字、
            # keywords 是字符串…）会让 .strip()/列表推导抛 AttributeError/
            # TypeError——旧实现只捕 OSError/JSONDecodeError，一个坏文件
            # 就能让整次 scan/search 失败。
            raw_name = meta.get("name", "")
            if not isinstance(raw_name, str):
                return None
            name = raw_name.strip()
            if not name:
                return None
            raw_desc = meta.get("description", "")
            description = raw_desc if isinstance(raw_desc, str) else ""
            raw_kw = meta.get("keywords", [])
            if isinstance(raw_kw, list):
                keywords = [str(k).lower() for k in raw_kw if isinstance(k, str)]
            else:
                keywords = []
            raw_tags = meta.get("tags", [])
            if isinstance(raw_tags, list):
                tags = [str(t) for t in raw_tags if isinstance(t, str)]
            else:
                tags = []
            entry = meta.get("entry", _DEFAULT_ENTRY)
            if not isinstance(entry, str) or not entry.strip():
                entry = _DEFAULT_ENTRY
            # Verify the entry file exists
            entry_path = os.path.join(skill_dir, entry)
            if not os.path.isfile(entry_path):
                # Try fallback
                entry = _DEFAULT_ENTRY
                if not os.path.isfile(os.path.join(skill_dir, entry)):
                    logger.warning(
                        "skill_index: skill '%s' has no entry file (%s)",
                        dirname, entry,
                    )
                    return None
            return {
                "name": name,
                "description": description,
                "keywords": keywords,
                "tags": tags,
                "path": dirname,
                "entry": entry,
            }
        except (OSError, json.JSONDecodeError, TypeError, AttributeError) as e:
            logger.warning("skill_index: failed to read %s: %s", meta_path, e)
            return None

    # ------------------------------------------------------------------
    # query
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Keyword-based search over the skill index.

        Returns a list of skill records (including path and entry),
        sorted by relevance score.
        """
        # Ensure index is fresh
        self.scan()

        # 在锁内取快照：scan() 结束后另一个线程可能立刻再次扫描并替换
        # self._index，直接遍历会在迭代过程中被改写。
        with self._lock:
            index = list(self._index)

        if not index or not query.strip():
            return []

        query_tokens = set(_tokenize(query.lower()))
        if not query_tokens:
            return []

        scored = []
        for skill in index:
            score = 0
            # Match against name（记录可能来自旧版/手改的 _index.json，
            # 逐个字段兜底类型，一个坏记录不能让整次查询炸掉）
            name = skill.get("name", "")
            name_tokens = set(_tokenize(name.lower())) if isinstance(name, str) else set()
            score += len(query_tokens & name_tokens) * 3
            # Match against keywords
            raw_kw = skill.get("keywords", [])
            kw_tokens = set()
            if isinstance(raw_kw, list):
                for k in raw_kw:
                    if isinstance(k, str):
                        # 关键词也要分词：中文关键词「知识库」与查询
                        # 「知识库检索」在 bigram 层匹配，不能整串比较。
                        kw_tokens.update(_tokenize(k))
            score += len(query_tokens & kw_tokens) * 2
            # Match against tags
            raw_tags = skill.get("tags", [])
            tag_tokens = set()
            if isinstance(raw_tags, list):
                for t in raw_tags:
                    if isinstance(t, str):
                        tag_tokens.update(_tokenize(t))
            score += len(query_tokens & tag_tokens) * 2
            # Match against description (substring)
            desc = skill.get("description", "")
            desc = desc.lower() if isinstance(desc, str) else ""
            for token in query_tokens:
                if token in desc:
                    score += 1
            if score > 0:
                scored.append((score, skill))

        # 次级排序键（name）：同分时仅按 score 排序的结果取决于 os.scandir
        # 的任意顺序，两次查询可能返回不同顺序，模型侧表现为「结果不稳定」。
        scored.sort(key=lambda x: (-x[0], str(x[1].get("name", ""))))
        # 带 score 返回：调用方（Agent 的每轮技能提示）需要按阈值过滤，
        # 避免一个弱 bigram 命中就把不相关技能塞进上下文。
        return [{**s, "score": sc} for sc, s in scored[:top_k]]

    def _resolve_within(self, *parts: str) -> Optional[str]:
        """Resolve parts under the skills dir, rejecting path traversal.

        Returns the absolute path if it stays inside self._dir, else None.
        The LLM supplies `path`/`entry` strings, so we must defend against
        e.g. path="../../../../etc" or entry="../../../secret".
        """
        base = os.path.realpath(self._dir)
        resolved = os.path.realpath(os.path.join(base, *parts))
        if resolved == base or not resolved.startswith(base + os.sep):
            return None
        return resolved

    def get_skill_path(self, skill_path: str) -> Optional[str]:
        """Return the absolute path to a skill directory by its relative path."""
        abs_path = self._resolve_within(skill_path)
        if abs_path and os.path.isdir(abs_path):
            return abs_path
        return None

    def read_skill_content(self, skill_path: str, entry: str = "prompt.md") -> Optional[str]:
        """Read the content of a skill's entry file."""
        abs_path = self._resolve_within(skill_path, entry)
        if abs_path is None:
            logger.warning("skill_index: rejected unsafe path: %s/%s", skill_path, entry)
            return None
        try:
            # 上限保护：技能目录里可能混进大语料/模型文件，无上限 f.read()
            # 会把它整个读进内存再塞进模型上下文。
            with open(abs_path, "r", encoding="utf-8") as f:
                text = f.read(_MAX_READ_CHARS + 1)
            if len(text) > _MAX_READ_CHARS:
                return text[:_MAX_READ_CHARS] + \
                    f"\n… [文件过大，已截断至 {_MAX_READ_CHARS} 字符]"
            return text
        except OSError as e:
            logger.warning("skill_index: failed to read %s: %s", abs_path, e)
            return None

    def list_all(self) -> List[Dict[str, Any]]:
        """Return the full index (triggers a scan if needed)."""
        self.scan()
        with self._lock:
            # 深拷贝记录：浅拷贝会让调用方与索引共享同一个 step/record
            # dict，调用方改字段会污染索引（UI 面板/工具各持一份快照）。
            return [dict(r) for r in self._index]

    def reload(self) -> None:
        """Force reload from disk."""
        with self._lock:
            self._load()


# ---------------------------------------------------------------------------
# Tokenizer (shared with skills.py)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """CJK-aware tokenizer.

    复用 ``skills._keywords_of``：``\\w+`` 对连续中文只能得到一个整串 token，
    「知识库检索」无法命中关键词「知识库」/「检索」；该实现额外产出 CJK
    bigram，中文技能检索才真正可用。
    """
    return _keywords_of(text)
