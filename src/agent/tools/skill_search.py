"""Skill search tool: lets the agent discover and load skill prompts.

The agent can call skill_search to find skills relevant to the current
task, then skill_load to read the full prompt and inject it into the
conversation.

Skills live in .agent/skills/, each in its own subdirectory with
a skill.json metadata file and a prompt.md entry file.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from typing import Optional

from ..atomic_io import atomic_write_text
from ..config import AgentConfig
from ..skill_index import SkillIndex
from .base import Tool, ToolRegistry, ToolResult, _FanoutTool


logger = logging.getLogger(__name__)

_skill_index_lock = threading.Lock()

# 单文件读取上限：技能目录里可能混进几百 MB 的语料/模型文件，
# 无上限地 f.read() 会把它整个读进内存再塞进模型上下文。
# 与其他工具保持一致（file_ops / shell_run / mcp 都是 200K 级）。
_MAX_SKILL_FILE_CHARS = 200_000


def _clamp_int(value, default: int, lo: int, hi: int) -> int:
    """把模型传来的 top_k 之类归一化到 [lo, hi]。

    ``int("abc")`` / ``int(None)`` 会抛异常，被外层兜底成一句没有上下文的
    裸错误；这里统一吃掉，退化到默认值。
    """
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(n, hi))


def _read_text_capped(path: str, cap: int = _MAX_SKILL_FILE_CHARS) -> str:
    """读取文本文件，超过 cap 时截断并标注（防止巨型技能文件打爆上下文）。"""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read(cap + 1)
    if len(text) > cap:
        return text[:cap] + f"\n… [文件过大，已截断至 {cap} 字符]"
    return text


def _is_external_skill_dir(abs_dir: str, skills_root: str) -> bool:
    """``abs_dir`` 是否位于 ``skills_root`` 之外（即本身是符号链接）。

    ``skills_root`` 必须是**产出 ``abs_dir`` 的那个检索器的根目录**：
    语义检索器扫 ``<workspace>/skills``，旧体系扫
    ``<workspace>/.agent/skills``。拿其中一个当基线去判另一个的结果，
    会把每一个正常技能都误标成「来自工作区之外」。
    """
    if not abs_dir:
        return False
    base = os.path.realpath(skills_root)
    return not (abs_dir == base or abs_dir.startswith(base + os.sep))


def _get_skill_index(config: AgentConfig, registry: ToolRegistry) -> SkillIndex:
    """Get or create the singleton SkillIndex, cached on the registry.

    The cache key includes the skills directory: 「应用配置」 can switch the
    workspace mid-session, and returning the previous workspace's index would
    silently show (and load) skills from the wrong place.

    构造 SkillIndex 会全目录扫描并可能落盘 —— 无锁时并发首次调用会产生两个
    实例（后写者覆盖缓存，前者泄漏一份扫描结果）。
    """
    skills_dir = os.path.join(config.workspace, ".agent", "skills")
    with _skill_index_lock:
        cache = getattr(registry, "_skill_index_cache", None)
        if isinstance(cache, dict):
            idx = cache.get(skills_dir)
            if idx is not None:
                return idx
        else:
            cache = {}
            setattr(registry, "_skill_index_cache", cache)
        idx = SkillIndex(skills_dir)
        cache[skills_dir] = idx
        # Keep the legacy attribute in sync so shutdown()/other callers still see it.
        setattr(registry, "_skill_index", idx)
        return idx


def _resolve_within(base_dir: str, *parts: str) -> Optional[str]:
    """Join ``parts`` under ``base_dir`` and verify the result stays inside it.

    Returns ``None`` for traversal attempts (``..``), absolute paths, and
    anything else that escapes ``base_dir``.
    """
    base = os.path.realpath(base_dir)
    resolved = os.path.realpath(os.path.join(base, *parts))
    if resolved == base or not resolved.startswith(base + os.sep):
        return None
    return resolved


class SkillSearchTool(Tool):
    """Search for skills relevant to the current task.

    The agent should call this when it encounters a task that might
    benefit from a pre-defined skill (e.g. "extract PDF tables", "scrape
    a website").  Returns matching skills with name, description, and
    path.
    """
    name = "skill_search"
    description = (
        "Keyword search over the skill directory (.agent/skills). "
        "Returns matching records (name, description, path, entry). "
        "op: 'search' (query) or 'list' (all). "
        "Then use skill_load to read the full skill."
    )
    parameters = {
        "type": "object",
        "properties": {
            "op": {"type": "string", "enum": ["search", "list"], "default": "search"},
            "query": {"type": "string", "description": "Task description to find matching skills"},
            "top_k": {"type": "integer", "default": 5},
        },
    }

    config: AgentConfig
    registry: ToolRegistry

    def bind(self, config: AgentConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry

    def run(self, op: str = "search", query: str = "", top_k: int = 5) -> ToolResult:
        idx = _get_skill_index(self.config, self.registry)
        # 模型可能传 null / 非数字 / 超限值：全部归一化后再用
        # （int("abc") 会抛 ValueError，被外层兜底成难以理解的裸异常）。
        query = (query or "").strip()
        top_k = _clamp_int(top_k, 5, 1, 20)
        if op == "list":
            results = idx.list_all()
            # 必须设上限：技能目录可能积累数百个，全量塞进上下文会挤掉真正
            # 要处理的内容（memory_graph list 限 200、rag_status 限 50，
            # 此处保持一致并明确告知总数）。
            _MAX_LIST = 50
            compact = [
                {
                    "name": s["name"],
                    "path": s["path"],
                    "description": s.get("description", "")[:100],
                }
                for s in results[:_MAX_LIST]
            ]
            out = json.dumps(compact, ensure_ascii=False)
            if len(results) > _MAX_LIST:
                out += (f"\n… (共 {len(results)} 个技能，仅显示前 {_MAX_LIST} 个；"
                        f"用 op='search' + query 精确查找)")
            return ToolResult(True, output=out)
        # Default: search
        if not query:
            return ToolResult(False, error="op 'search' requires 'query'")
        results = idx.search(query, top_k=top_k)
        return ToolResult(True, output=json.dumps(results, ensure_ascii=False))


class SkillSemanticSearchTool(Tool):
    """Hybrid (vector + BM25) skill search over SKILL.md files.

    Scans ``<workspace>/skills`` recursively for ``SKILL.md`` files, indexes
    their descriptions + examples (vector) and combined text (BM25), then
    retrieves candidate skills for a task description.  Optionally hands the
    candidates to the LLM for a final pick.

    This is the semantic counterpart to ``skill_search`` (keyword-based).
    """
    name = "skill_semantic_search"
    description = (
        "语义检索最合适的 skill（扫描 workspace/skills 下的 SKILL.md，向量 + BM25）。"
        "若要找的是 .agent/skills 下的技能请用 skill_search（两者目录不同）。"
        "返回候选的 name + tags + description；再用 skill_load 读完整指令。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "任务描述（用用户原始措辞，勿翻译/音译）"},
            "top_k": {"type": "integer", "description": "返回候选数，默认 5"},
        },
        "required": ["query"],
    }

    config: AgentConfig
    registry: ToolRegistry

    def bind(self, config: AgentConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry

    def run(self, query: str, top_k: int = 5) -> ToolResult:
        if not (query or "").strip():
            return ToolResult(False, error="requires 'query'")
        query = query.strip()
        top_k = _clamp_int(top_k, 5, 1, 20)
        retriever = self.registry.get_skill_retriever()
        # 只做混合检索 + 阈值过滤；最终的「选哪个」由 agent（大模型）根据
        # 返回的 name+tags+description 自行判断，避免工具层再调一次 LLM。
        result = retriever.retrieve(query, llm=None)

        candidates = result.get("candidates", [])
        if not candidates:
            return ToolResult(True, output="找不到匹配的技能。")

        # Compact output: dir + name + tags + description (truncated).
        items = []
        for c in candidates[:top_k]:
            item = {
                "dir": c["dir"],
                "name": c["name"],
                "tags": c.get("tags", ""),
                "description": (c.get("description", "") or "")[:200],
                "score": round(c["score"], 4),
            }
            items.append(item)
        return ToolResult(True, output=json.dumps(items, ensure_ascii=False, indent=2))


class SkillLoadTool(Tool):
    """Load a skill's full prompt content by its path.

    After skill_search finds a relevant skill, use skill_load to read
    its prompt/instructions file and inject them into the conversation.
    """
    name = "skill_load"
    description = (
        "读取 skill 的完整指令内容。提供 'path'（来自 skill_search / "
        "skill_semantic_search 结果的 'dir' 字段），entry 默认 SKILL.md。"
        "也可读取 skill 目录下的其它支持文件（如 references/xxx.md）。"
        "注意：skill 目录内的符号链接会被跟随（用户自建技能的既有行为），"
        "因此可能读到 skills 目录之外的文件。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Skill 目录（检索结果的 'dir' 字段）"},
            "entry": {"type": "string", "description": "文件名，默认 SKILL.md（旧体系默认 prompt.md）"},
        },
        "required": ["path"],
    }

    config: AgentConfig
    registry: ToolRegistry

    def bind(self, config: AgentConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry

    def run(self, path: str, entry: str = "SKILL.md") -> ToolResult:
        # 入口名由模型提供，先做显式校验：绝对路径会被 os.path.join 直接当作
        # 结果（丢掉技能目录），".." 又能逃出目录 —— 两者都会让 skill_load
        # 变成任意文件读取。校验失败必须直接报错，不能「回退到别的文件」
        # 从而把非法请求伪装成一次成功读取。
        entry = (entry or "SKILL.md").strip()
        if os.path.isabs(entry) or ".." in entry.replace("\\", "/").split("/"):
            return ToolResult(
                False,
                error=f"refusing unsafe entry outside the skill directory: {entry!r}",
            )

        # 1) Try the semantic retriever first (SKILL.md directory).
        # 必须走 get_skill_retriever()（按目录缓存、跟随当前 workspace）。
        # 旧实现读 registry._skill_retriever 这个 legacy 属性：它只指向第一次
        # 创建时的 workspace，切换工作区后 skill_semantic_search 用新目录而
        # skill_load 还在旧目录上做路径校验——新技能读不出、旧技能照读。
        retriever = self.registry.get_skill_retriever()
        if retriever is not None:
            abs_dir = retriever.read_skill_dir(path)
            if abs_dir:
                # `entry` 由模型提供，必须校验：os.path.join 在 entry 为绝对
                # 路径时会直接丢弃 abs_dir，而 "../" 又能逃出技能目录——
                # 两者合起来等于允许读取磁盘上的任意文件。
                target = _resolve_within(abs_dir, entry)
                if target is None:
                    return ToolResult(
                        False,
                        error=f"refusing entry outside the skill directory: {entry!r}",
                    )
                # 旧体系（SkillIndex）的技能入口是 prompt.md，SKILL.md 缺失时
                # 回退一次，否则这些技能永远读不出来。
                if not os.path.isfile(target) and entry == "SKILL.md":
                    legacy = _resolve_within(abs_dir, "prompt.md")
                    if legacy is not None and os.path.isfile(legacy):
                        target = legacy
                if os.path.isfile(target):
                    try:
                        text = _read_text_capped(target)
                    except (OSError, UnicodeDecodeError) as e:
                        return ToolResult(False, error=f"read failed: {e}")
                    # 技能目录是符号链接时，读到的内容来自工作区之外——把真实
                    # 来源标注出来，否则用户（和模型）都看不出自己读的是哪。
                    # 基线必须用**语义检索器自己的** skills 根目录：abs_dir 来自
                    # 它（根是 <workspace>/skills），拿 .agent/skills 当基线会把
                    # 每一个正常技能都误标成外部链接。
                    try:
                        if _is_external_skill_dir(abs_dir, retriever.skills_dir):
                            text = (
                                f"[提示] 该技能目录是符号链接，实际内容来自工作区之外：\n"
                                f"{abs_dir}\n\n{text}"
                            )
                    except Exception:  # pragma: no cover - 标注失败不影响读取
                        pass
                    return ToolResult(True, output=text)

        # 2) Fall back to the legacy SkillIndex (skill.json + prompt.md).
        idx = _get_skill_index(self.config, self.registry)
        content = idx.read_skill_content(path, entry)
        if content is None:
            # 目录技能在 skill.json 里声明自己的入口文件（老技能是
            # prompt.md）。工具默认 entry 是 SKILL.md，老技能因此永远读不到
            # —— 回退到索引里记录的 entry，再回退到 prompt.md。
            declared = self._declared_entry(idx, path)
            for fallback in (declared, "prompt.md"):
                if not fallback or fallback == entry:
                    continue
                content = idx.read_skill_content(path, fallback)
                if content is not None:
                    return ToolResult(True, output=content)
        if content is None:
            return ToolResult(False, error=f"skill not found or entry file missing: {path}/{entry}")
        return ToolResult(True, output=content)

    @staticmethod
    def _declared_entry(idx: SkillIndex, path: str) -> Optional[str]:
        """Return the entry file declared by the skill's own skill.json."""
        try:
            for record in idx.list_all():
                if record.get("path") == path:
                    return record.get("entry") or None
        except Exception:
            return None
        return None


# ---------------------------------------------------------------------------
# 能力进化：把成功经验固化成外置技能
# ---------------------------------------------------------------------------

_SAFE_SKILL_NAME_RE = re.compile(r"[^a-z0-9\-]+")
_MAX_SKILL_BODY = 8_000

_SKILL_TEMPLATE = """---
name: {name}
description: {description}
tags: [{tags}]
---

# {name}

{description}

## When to use

{when_to_use}

## Steps

{steps}

## Pitfalls

{pitfalls}
"""


class SkillSaveTool(Tool):
    """Freeze a procedure that just succeeded into a reusable SKILL.md.

    这是「能力进化」的写入端：任务成功后把流程写成 ``<workspace>/skills/<name>/SKILL.md``，
    ``SkillRetriever`` 下次扫描时会自动增量索引（按 fingerprint），于是后续同类任务
    可以通过 ``skill_semantic_search`` / ``skill_search`` 直接召回并复用。
    """

    name = "skill_save"
    description = (
        "Freeze a procedure that succeeded in THIS session into a reusable skill "
        "(writes <workspace>/skills/<name>/SKILL.md). It becomes discoverable by "
        "skill_semantic_search/skill_search next time. "
        "Only freeze verified, reusable procedures — not one-off tasks."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "kebab-case id, e.g. 'extract-pdf-tables'"},
            "description": {"type": "string", "description": "One line: what it does (<=200 chars)"},
            "when_to_use": {"type": "string", "description": "Trigger conditions (<=400 chars)"},
            "steps": {"type": "string", "description": "Numbered steps that were verified to work"},
            "pitfalls": {"type": "string", "description": "Gotchas / things that failed (optional)"},
            "tags": {"type": "string", "description": "Comma-separated tags (optional)"},
        },
        "required": ["name", "description", "steps"],
    }

    config: AgentConfig
    registry: ToolRegistry

    def bind(self, config: AgentConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry

    def run(self, name: str, description: str, steps: str,
            when_to_use: str = "", pitfalls: str = "", tags: str = "") -> ToolResult:
        slug = _SAFE_SKILL_NAME_RE.sub("-", (name or "").strip().lower()).strip("-")[:64]
        if not slug:
            return ToolResult(False, error="invalid skill name (need [a-z0-9-])")
        if not (steps or "").strip():
            return ToolResult(False, error="steps is required")

        root = os.path.join(self.config.workspace, "skills")
        # 双重防护：slug 已过滤掉非 [a-z0-9-]，再校验一次 realpath 前缀
        target_dir = _resolve_within(root, slug)
        if target_dir is None:
            return ToolResult(False, error=f"refusing path outside skills dir: {slug!r}")

        tag_list = [t.strip() for t in re.split(r"[,，]", tags or "") if t.strip()][:8]
        body = _SKILL_TEMPLATE.format(
            name=slug,
            description=(description or "").strip().replace("\n", " ")[:200],
            tags=", ".join(tag_list),
            when_to_use=(when_to_use or "（未指定）").strip()[:400],
            steps=(steps or "").strip()[:_MAX_SKILL_BODY],
            pitfalls=(pitfalls or "（暂无）").strip()[:2000],
        )
        try:
            os.makedirs(target_dir, exist_ok=True)
            path = os.path.join(target_dir, "SKILL.md")
            # 原子写：唯一临时文件 + fsync + replace。旧实现用
            # `.SKILL.md.{pid}.tmp`：同一进程内 agent 线程的 skill_save 与后台
            # 蒸馏线程的 skill_save 会写到**同一路径**，内容交错产出损坏的
            # SKILL.md；写失败时 tmp 还不清理。
            atomic_write_text(path, body)
        except OSError as e:
            return ToolResult(False, error=f"write failed: {e}")

        # 让检索器下次调用前重新同步向量索引。
        # 不能只靠目录 mtime：scan(force=True) 会把扫描时间戳刷新掉，
        # 之后 _should_scan() 恒为 False，新技能永远不会被嵌入。
        try:
            retriever = self.registry.get_skill_retriever()
            if retriever is not None:
                retriever.mark_stale()
        except Exception:
            pass

        # 关键词检索（skill_search / SkillIndex）扫的是 .agent/skills，与语义
        # 检索扫的 workspace/skills 不是同一个目录。只写一处会让另一条检索路径
        # 永远召不回刚固化的技能（回执却声称"都能召回"）。这里镜像一份过去，
        # 且只在目标不存在时才写——不覆盖用户自己在 .agent/skills 里的版本。
        #
        # 镜像必须同时写 skill.json：SkillIndex._read_skill_meta 只收录
        # 「目录里有 skill.json」的技能，只写 SKILL.md 的镜像永远不会被
        # 关键词索引收录（回执里「已镜像以便关键词检索命中」是空话）。
        mirrored = ""
        try:
            legacy_root = os.path.join(self.config.workspace, ".agent", "skills")
            legacy_dir = _resolve_within(legacy_root, slug)
            if legacy_dir is not None and not os.path.exists(
                    os.path.join(legacy_dir, "SKILL.md")):
                os.makedirs(legacy_dir, exist_ok=True)
                atomic_write_text(os.path.join(legacy_dir, "SKILL.md"), body)
                meta = {
                    "name": slug,
                    "description": (description or "").strip()[:200],
                    "keywords": tag_list,
                    "tags": tag_list,
                    "version": "1.0",
                    "entry": "SKILL.md",
                }
                atomic_write_text(
                    os.path.join(legacy_dir, "skill.json"),
                    json.dumps(meta, ensure_ascii=False, indent=2),
                )
                mirrored = str(legacy_dir)
                idx = _get_skill_index(self.config, self.registry)
                idx.scan(force=True)
        except OSError as e:
            logger.warning("skill_save: mirror to .agent/skills failed: %s", e)

        note = ("下次可用 skill_tools(op=\"semantic_search\") 或 "
                "skill_tools(op=\"search\") 召回（索引会自动增量同步）")
        if mirrored:
            note += f"；已镜像到 {mirrored} 以便关键词检索命中"
        return ToolResult(True, output=f"skill saved: {path}\n{note}。")


class SkillToolsTool(_FanoutTool):
    """技能统一门面：op 分发到 search / load / semantic_search / save。"""

    name = "skill_tools"
    description = (
        "Unified skill operations. op one of:\n"
        "- search(query, top_k?): keyword search over .agent/skills; "
        "omit query to list all skills\n"
        "- semantic_search(query, top_k?): hybrid (vector+BM25) search over "
        "workspace/skills (SKILL.md)\n"
        "- load(path, entry?): read a skill's full instructions (path = 'dir' "
        "from search results)\n"
        "- save(name, description, steps, when_to_use?, pitfalls?, tags?): "
        "freeze a verified procedure into a reusable skill"
    )
    parameters = {
        "type": "object",
        "properties": {
            "op": {"type": "string", "enum": ["search", "load", "semantic_search", "save"]},
            "query": {"type": "string", "description": "Task/keywords to find matching skills (search/semantic_search)"},
            "top_k": {"type": "integer", "default": 5},
            "path": {"type": "string", "description": "Skill dir to load (search result 'dir' field)"},
            "entry": {"type": "string", "description": "Entry file name, default SKILL.md"},
            "name": {"type": "string", "description": "kebab-case id (save)"},
            "description": {"type": "string", "description": "One-line summary (save)"},
            "when_to_use": {"type": "string", "description": "Trigger conditions (save)"},
            "steps": {"type": "string", "description": "Numbered verified steps (save)"},
            "pitfalls": {"type": "string", "description": "Gotchas (save)"},
            "tags": {"type": "string", "description": "Comma-separated tags (save)"},
        },
        "required": ["op"],
    }

    # op 级参数隔离：search/semantic_search 用 query/top_k，load 用 path/entry，
    # save 用 name/description/steps/…；避免扁平 schema 下互相串参。
    _op_params = {
        "search": {"query", "top_k"},
        "semantic_search": {"query", "top_k"},
        "load": {"path", "entry"},
        "save": {"name", "description", "steps", "when_to_use", "pitfalls", "tags"},
    }
    # 合并前 skill_search 本身有 op="list"（列出全部），现在等价形态是
    # skill_tools(op="search") 不带 query。
    _legacy_op_hints = {
        "list": "这是合并前 skill_search 的 op 值；现在请用 "
                "skill_tools(op='search') 且不带 query（即列出全部技能）。",
    }

    def __init__(self):
        self._sub_tools = {
            "search": SkillSearchTool(),
            "semantic_search": SkillSemanticSearchTool(),
            "load": SkillLoadTool(),
            "save": SkillSaveTool(),
        }

    def run(self, op: str = "", **kwargs) -> ToolResult:
        op = (op or "").strip().lower()
        sub = self._sub_tools.get(op)
        if sub is None:
            return self._unknown_op_error(op)
        err = self._validate_params(op, kwargs)
        if err is not None:
            return err
        query = str(kwargs.get("query", "") or "")
        top_k = kwargs.get("top_k", 5)
        if op == "search":
            # SkillSearchTool 自身也有 op（search/list）：keyword search 在
            # query 为空时退化为「列出全部技能」，保留其 list 能力。
            sub_op = "list" if not query.strip() else "search"
            return self._sub_tools["search"].run(sub_op, query=query, top_k=top_k)
        if op == "load":
            return self._sub_tools["load"].run(path=kwargs.get("path", ""),
                                              entry=kwargs.get("entry", "SKILL.md"))
        if op == "semantic_search":
            return self._sub_tools["semantic_search"].run(query=query, top_k=top_k)
        # op == "save"
        return self._sub_tools["save"].run(
            name=kwargs.get("name", ""), description=kwargs.get("description", ""),
            steps=kwargs.get("steps", ""), when_to_use=kwargs.get("when_to_use", ""),
            pitfalls=kwargs.get("pitfalls", ""), tags=kwargs.get("tags", ""),
        )
