"""轻量技能提示索引（无向量、无 LLM）。

用途：每轮对话前判断 ``<workspace>/skills`` 里是否有与用户输入明显相关的
技能，给模型一个明确的 ``skill_tools(op="load")`` 入口。只读 SKILL.md 的
frontmatter（最多 8KB），不加载 embedding，也不写入任何索引文件。

语义检索（SkillRetriever）仍然是主通道；本索引只负责「自动提示」这条
兜底路径，避免模型不主动调用 semantic_search 时完全错过技能。
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List

from .skill_retriever import (
    _SKIP_DIR_NAMES, _parse_frontmatter, _walk_skill_dirs,
)
from .skills import _keywords_of

_SCAN_INTERVAL = 60.0
_MAX_HEAD_CHARS = 8_000


class SkillHintIndex:
    """按 name/tags/description 做关键词匹配的内存索引。"""

    def __init__(self, skills_dir: str):
        self._dir = skills_dir
        self._lock = threading.RLock()
        self._records: List[Dict[str, Any]] = []
        self._last_scan = 0.0
        self._last_mtime = 0.0
        self._mtime_cache = 0.0
        self._mtime_cache_at = 0.0

    def _dir_mtime(self, use_cache: bool = False) -> float:
        now = time.time()
        if use_cache and (now - self._mtime_cache_at) < 5.0:
            return self._mtime_cache
        max_mtime = 0.0
        if not os.path.isdir(self._dir):
            return 0.0
        for root, dirs, files in _walk_skill_dirs(self._dir):
            dirs[:] = [d for d in dirs
                       if d not in _SKIP_DIR_NAMES and not d.startswith(".")]
            try:
                max_mtime = max(max_mtime, os.stat(root).st_mtime)
            except OSError:
                pass
            if "SKILL.md" in files:
                try:
                    max_mtime = max(
                        max_mtime, os.stat(os.path.join(root, "SKILL.md")).st_mtime
                    )
                except OSError:
                    pass
        self._mtime_cache = max_mtime
        self._mtime_cache_at = now
        return max_mtime

    def _should_scan(self) -> bool:
        if self._last_scan == 0.0:
            return True
        if time.time() - self._last_scan > _SCAN_INTERVAL:
            return True
        try:
            return self._dir_mtime(use_cache=True) != self._last_mtime
        except OSError:
            return False

    def scan(self, force: bool = False) -> List[Dict[str, Any]]:
        with self._lock:
            if not force and not self._should_scan():
                return list(self._records)
            records: List[Dict[str, Any]] = []
            if os.path.isdir(self._dir):
                for root, dirs, files in _walk_skill_dirs(self._dir):
                    dirs[:] = [d for d in dirs
                               if d not in _SKIP_DIR_NAMES and not d.startswith(".")]
                    if "SKILL.md" not in files:
                        continue
                    try:
                        with open(os.path.join(root, "SKILL.md"),
                                  "r", encoding="utf-8") as f:
                            head = f.read(_MAX_HEAD_CHARS)
                    except (OSError, UnicodeDecodeError):
                        continue
                    fm = _parse_frontmatter(head)
                    name = str(fm.get("name") or "").strip()
                    if not name:
                        continue
                    tags = fm.get("tags") or []
                    if isinstance(tags, str):
                        tags = [t.strip() for t in tags.replace("，", ",").split(",")
                                if t.strip()]
                    try:
                        rel = os.path.relpath(root, self._dir)
                    except ValueError:
                        rel = root
                    records.append({
                        "name": name,
                        "path": rel,
                        "description": str(fm.get("description") or ""),
                        "tags": [str(t) for t in tags][:8],
                    })
            self._records = records
            self._last_scan = time.time()
            self._last_mtime = self._dir_mtime()
            return list(records)

    def search(self, query: str, top_k: int = 1) -> List[Dict[str, Any]]:
        self.scan()
        tokens = set(_keywords_of(query or ""))
        if not tokens:
            return []
        with self._lock:
            records = list(self._records)
        scored = []
        for r in records:
            name_tokens = set(_keywords_of(str(r.get("name") or "")))
            tag_tokens = set()
            for t in r.get("tags") or []:
                tag_tokens.update(_keywords_of(str(t)))
            desc_tokens = set(_keywords_of(str(r.get("description") or "")))
            score = (3 * len(tokens & name_tokens)
                     + 2 * len(tokens & tag_tokens)
                     + len(tokens & desc_tokens))
            if score > 0:
                scored.append((score, r))
        scored.sort(key=lambda x: (-x[0], str(x[1].get("name", ""))))
        return [{**r, "score": sc} for sc, r in scored[:top_k]]
