"""Skill manager: 固化 frequently-run tasks into reusable skills.

A skill records a name, trigger keywords, and a prompt template. When the same
intent is requested 2+ times, the manager suggests saving it as a skill.

Also supports directory-based skills under .agent/skills/. Each skill
directory has a skill.json metadata file and a prompt.md entry file.
The SkillIndex (skill_index.py) scans these directories and maintains
an _index.json table for fast lookup via the skill_search tool.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from .atomic_io import atomic_write_text
from .memory import _JsonStore

_TOKEN_RE = re.compile(r"\w+")
# 中文没有空格，\w+ 会把整段连续中文当成一个 token（"提取pdf表格" →
# {"提取pdf表格"}），导致 "pdf"/"提取" 这类关键词永远匹配不上，中文技能
# 检索彻底失效。因此额外切出 CJK 子串及其二元组（bigram）用于匹配。
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")

# 停用词表提到模块级：_request_key() 每轮对话都会调用，旧实现在函数体内重建
# 这个集合，白白重复分配。
_STOPWORDS = frozenset({
    "a", "an", "the", "to", "of", "in", "on", "at", "for",
    "and", "or", "is", "are", "was", "were", "be", "been",
    "do", "does", "did", "with", "from", "by", "as", "it",
    "this", "that", "these", "those", "i", "we", "you",
    "please", "me", "my", "our",
})

# 「请求计数」表的条目上限（见 SkillManager.record_request）。
_MAX_TRACKED_REQUESTS = 500


def _keywords_of(text: str) -> List[str]:
    """Tokenize for keyword matching, with CJK support.

    Latin words are taken whole; CJK runs are additionally emitted as the full
    run plus its character bigrams, so a keyword like "提取" matches inside
    "提取pdf表格" (without this, ``\\w+`` yields one giant token and Chinese
    skill matching silently never hits).
    """
    toks = [w.lower() for w in _TOKEN_RE.findall(text)]
    for run in _CJK_RUN_RE.findall(text):
        toks.append(run)
        for i in range(len(run) - 1):
            toks.append(run[i:i + 2])
    return toks


@dataclass
class Skill:
    name: str
    keywords: List[str]
    prompt: str
    count: int = 0

    def score(self, text: str) -> int:
        toks = set(_keywords_of(text))
        return self.score_tokens(toks)

    def score_tokens(self, toks: set) -> int:
        """用**已分词**的 token 集合打分（match() 的对多技能快速路径）。

        ``score(text)`` 内部每次都对同一 text 重新分词（中文还要展开二元组），
        ``match()`` 对每个技能调一次 → 同一 query 被分词 N 次。技能量大时
        每轮对话都是明显 CPU 浪费。保留 ``score(text)`` 作兼容入口。
        """
        return sum(1 for k in self.keywords if k.lower() in toks)


class SkillManager:
    SUGGEST_THRESHOLD = 2

    def __init__(self, path: str, skills_dir: str = ""):
        self._store = _JsonStore(path)
        self._skills_dir = skills_dir  # .agent/skills/ directory
        # Defensive: a corrupted store may have null values. Directly modify
        # _data to avoid triggering a debounced _save() which would set
        # _last_save and delay subsequent real writes.
        # 修复必须**落盘**：只改内存的话，下次 reload()/重启又会从磁盘读回
        # 坏值，防御代码每轮都白修一遍（且 reload 前的 set() 才会顺带写出去，
        # 纯查询型用法永远修不掉）。
        with self._store._lock:
            repaired = False
            if not isinstance(self._store._data.get("skills"), list):
                self._store._data["skills"] = []
                repaired = True
            if not isinstance(self._store._data.get("requests"), dict):
                self._store._data["requests"] = {}
                repaired = True
            if repaired:
                self._store._dirty = True
        if repaired:
            # _JsonStore._lock 不是可重入锁，必须在 with 之外 flush。
            self._store.flush(force=True)
            # flush(force=True) 会把 _last_save 刷成当前时刻——紧接着的
            # 第一次 set() 就会被 0.5s 防抖吞掉（磁盘上只有空骨架），
            # 表现为「刚创建的技能换实例就读不到」。复位时间戳，让后续
            # 真实写入保持「首次立即落盘」的旧语义。
            with self._store._lock:
                self._store._last_save = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def reload(self) -> None:
        """Reload the store from disk.

        Called after an external write (e.g. the UI thread saved a new skill
        via create_skill while the worker thread's SkillManager instance still
        holds the old in-memory cache). Without this, match() in the worker
        would keep returning None for the just-saved skill and re-trigger
        suggestions every turn.
        """
        with self._lock:
            self._store.reload()
            if not isinstance(self._store.get("skills"), list):
                self._store.set("skills", [])
            if not isinstance(self._store.get("requests"), dict):
                self._store.set("requests", {})

    def _load_skills(self) -> List[Skill]:
        """Read persisted skill records, tolerating corrupted entries.

        A single malformed record (hand-edited skills.json, an aborted write,
        a record from an older schema) used to raise TypeError/KeyError inside
        ``Skill(**s)`` — which propagated out of ``match()`` and broke every
        turn's skill lookup, not just the bad record.  Skip bad entries instead.
        """
        raw = self._store.get("skills") or []
        if not isinstance(raw, list):
            return []
        out: List[Skill] = []
        for s in raw:
            if not isinstance(s, dict):
                continue
            try:
                out.append(Skill(
                    name=str(s.get("name", "")),
                    keywords=list(s.get("keywords") or []),
                    prompt=str(s.get("prompt", "")),
                    count=int(s.get("count", 0) or 0),
                ))
            except (TypeError, ValueError):
                continue
        return [s for s in out if s.name]

    def list(self) -> List[Skill]:
        with self._lock:
            return self._load_skills()

    def create_skill(self, name: str, keywords: List[str], prompt: str) -> Skill:
        skill = Skill(name=name, keywords=[k.lower() for k in (keywords or [])], prompt=prompt)
        with self._lock:
            skills = [s for s in self._load_skills() if s.name != name]
            skills.append(skill)
            self._store.set("skills", [asdict(s) for s in skills])
        return skill

    def delete(self, name: str) -> None:
        with self._lock:
            skills = [s for s in self._load_skills() if s.name != name]
            self._store.set("skills", [asdict(s) for s in skills])

    def create_dir_skill(self, name: str, keywords: List[str], prompt: str,
                         description: str = "", tags: List[str] = None) -> str:
        """Create a directory-based skill under .agent/skills/.

        This is the new unified format: each skill gets its own directory
        with skill.json + prompt.md.  Returns the skill directory path.

        The old flat-skill create_skill() still works for backwards compat,
        but new skills should use this method.
        """
        if not self._skills_dir:
            raise ValueError("skills_dir not configured")

        keywords = list(keywords or [])
        tags = list(tags or [])

        # Sanitize name → directory name
        dir_name = re.sub(r"[^\w\-.]", "-", name.lower().strip()).strip("-")
        if not dir_name:
            dir_name = "unnamed-skill"
        # Ensure uniqueness.
        # 旧实现 while os.path.exists(...) 是 TOCTOU：两个线程同时建同名
        # 技能会选到同一目录并互相覆盖。改用 makedirs 的 FileExistsError
        # 重试——「建目录」本身就是原子占用，不存在判断窗口。
        skill_dir = os.path.join(self._skills_dir, dir_name)
        counter = 1
        for _ in range(100):
            try:
                os.makedirs(skill_dir, exist_ok=False)
                break
            except FileExistsError:
                skill_dir = os.path.join(self._skills_dir, f"{dir_name}-{counter}")
                counter += 1
        else:
            raise ValueError(f"cannot create a unique skill directory for {name!r}")

        # Write skill.json + prompt.md —— 原子写：崩溃/并发不会留下半截文件
        # （旧实现裸 open()+write()，半截 skill.json 会让 SkillIndex 静默
        # 跳过该技能）。
        meta = {
            "name": name,
            "description": description or f"Skill: {name}",
            "keywords": [k.lower() for k in keywords],
            "tags": tags,
            "version": "1.0",
            "entry": "prompt.md",
        }
        atomic_write_text(
            os.path.join(skill_dir, "skill.json"),
            json.dumps(meta, ensure_ascii=False, indent=2),
        )
        atomic_write_text(os.path.join(skill_dir, "prompt.md"), prompt)

        # Also register a flat record so _match_skill() can hit this skill.
        # The flat record's prompt tells the agent to load the full
        # instructions via skill_load instead of embedding them in the
        # system prompt (keeps prompts lean).
        dir_name = os.path.basename(skill_dir)
        with self._lock:
            skills = self._load_skills()
            if not any(s.name == name for s in skills):
                skills.append(Skill(
                    name=name,
                    keywords=list(keywords),
                    prompt=(
                        f"A saved skill '{name}' matches this task. "
                        f"Call skill_load with path=\"{dir_name}\" to read its "
                        f"instructions, then follow them."
                    ),
                ))
                self._store.set("skills", [asdict(s) for s in skills])

        return skill_dir

    def match(self, text: str) -> Optional[Skill]:
        best: Optional[Skill] = None
        best_score = 0
        # 分词一次，全部技能共用（旧实现每技能内部各分词一遍）
        toks = set(_keywords_of(text))
        for s in self.list():
            sc = s.score_tokens(toks)
            if sc > best_score:
                best_score = sc
                best = s
        return best if best_score > 0 else None

    # ------------------------------------------------------------------
    # request tracking -> suggest固化 after threshold
    # ------------------------------------------------------------------
    @staticmethod
    def _request_key(text: str) -> str:
        """Normalize a request to a stable key for counting similar requests.

        Uses the sorted set of unique keywords (minus stopwords) so paraphrased
        requests like "deploy to staging" and "deploy staging" map to the same
        key. The old implementation joined ALL keywords in original order, so
        dropping or adding a single word produced a different key and the
        counter never reached the suggestion threshold.

        超过 80 字符的长 key 不能直接截断：两个只有尾部不同的长请求会碰撞成
        同一个计数键，产生虚假的「重复请求 → 建议固化」。截断后补一段全文
        哈希，保证不同 key 必然不同（已有的短 key 不受影响，格式兼容）。
        """
        kws = sorted({w for w in _keywords_of(text) if w not in _STOPWORDS})
        full = " ".join(kws)
        if len(full) <= 80:
            return full
        digest = hashlib.sha1(full.encode("utf-8")).hexdigest()[:8]
        return f"{full[:72]}#{digest}"

    def record_request(self, text: str) -> Optional[Skill]:
        """Track a user request. Returns a *suggested* (unsaved) Skill once the
        same intent has been seen SUGGEST_THRESHOLD times. The caller decides
        whether to persist it via create_skill().
        """
        key = self._request_key(text)
        if not key:
            return None
        with self._lock:
            raw = self._store.get("requests")
            requests = dict(raw) if isinstance(raw, dict) else {}
            requests[key] = requests.get(key, 0) + 1
            # 请求计数表只增不删：长期使用的 skills.json 会无界膨胀（既占内存
            # 也让每次 set() 的 JSON 序列化越来越大）。超过上限时淘汰最旧的
            # 一半条目（dict 保持插入顺序），但必须跳过当前 key——否则刚 +1
            # 的计数会被立刻删掉，返回值与持久化状态不一致（下次从 1 重新计）。
            if len(requests) > _MAX_TRACKED_REQUESTS:
                evict = _MAX_TRACKED_REQUESTS // 2
                for k in list(requests):
                    if evict <= 0:
                        break
                    if k == key:
                        continue
                    requests.pop(k, None)
                    evict -= 1
            self._store.set("requests", requests)
            count = requests[key]

        if count >= self.SUGGEST_THRESHOLD:
            kws = sorted(set(_keywords_of(text)))[:8]
            # Don't auto-persist; just suggest. Avoid suggesting repeatedly:
            # if a real skill with overlapping keywords exists, return None.
            existing = self.match(text)
            if existing is not None:
                return None
            return Skill(name="_suggested", keywords=kws, prompt=text, count=count)
        return None
