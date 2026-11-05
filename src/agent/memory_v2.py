"""L0~L4 分层记忆：会话落盘、索引注入、归档压缩、辅助引擎维护。

层级（工作目录 workspace 下）：
    L0  META-SOP            —— 提炼版常驻 system prompt；全文物化到
                               memory/memory_management_sop.md 按需读取
    L1  memory/global_mem_insight.txt  —— ≤30 行极简索引，紧跟 L0 进 system prompt
    L2  memory/global_mem.txt          —— 事实库
    L3  memory/*.md / *.py             —— 任务级 SOP / 脚本
    L4  memory/raw_sessions/           —— 当天全文 + 当天摘要 + 骨架汇总 + 往日 zip

设计要点：
  * 全文/摘要由 ``Session.finish_turn`` 的统一回调落盘，覆盖所有收尾分支；
  * L0 恒定、L1 只在 L2/L3 变更时重写 —— 两次变更之间 system prompt 前缀不变，
    本地 MLX 的 ArraysCache 前缀仍能命中（见 agent._system_prompt 注释）；
  * 归档/挖掘/整理都在后台线程由辅助引擎驱动，不占主引擎推理流。
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import zipfile
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional, Union

from .memory_sop_texts import (
    DEFAULT_L1_TEMPLATE,
    HISTORY_MINING_SOP,
    MEMORY_CLEANUP_SOP,
    MEMORY_MANAGEMENT_SOP,
    MEMORY_MANAGEMENT_SOP_FULL,
    MEMORY_PURIFY_PROMPT,
)

logger = logging.getLogger(__name__)

# 改进 6：短任务不许碰长期记忆
LONG_TERM_MIN_TURNS = 5
# 改进 4：L1 硬约束
_L1_MAX_LINES = 30
_L1_MAX_L3_ITEMS = 12
# 改进 5/6 内核检测时排除的「系统维护产物」，它们不是 L3 SOP
_SYSTEM_FILES = {"history_insight.md", "cleanup_report.md", "global_mem_insight.txt"}
# L0 全文文件：不是 L3 任务 SOP，不进 L3 索引行；但 file_read 命中它时仍应
# 触发「提炼进 key_info」联动（与读普通 L3 SOP 同机制）。
_L0_FILE = "memory_management_sop.md"

_IO_LOCK = threading.RLock()

_DAY_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.md$")
_TURN_RE = re.compile(r"^## \[T(\d+)\] (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*$", re.M)
_SECTION_RE = re.compile(r"^##\s*\[?([^\]\n]+?)\]?\s*$", re.M)

# 分段标记（改进 7 按这些标记剥离）
_H_SYS = "### SYSTEM_PROMPT"
_H_USER = "### USER"
_H_AGENT = "### AGENT"
_H_RESP = "### ASSISTANT_RESPONSE"
_H_TOOLS = "### TOOLS"
_KEEP_HEADS = (_H_USER, _H_AGENT)


# ---------------------------------------------------------------------------
# 目录布局
# ---------------------------------------------------------------------------
class MemoryLayout:
    """工作目录下 memory/ 的全部路径；构造时不建目录，ensure() 才建。"""

    def __init__(self, workspace: str):
        self.workspace = workspace or ""
        self.memory_dir = os.path.join(self.workspace, "memory")
        # 改进 1/2/7 用户文字里 raw_session / raw_sessions 混用，统一用复数
        # （改进 7 与历史挖掘 SOP 均以 raw_sessions 为准）。
        self.raw_dir = os.path.join(self.memory_dir, "raw_sessions")
        self.l0_path = os.path.join(self.memory_dir, "memory_management_sop.md")
        self.l1_path = os.path.join(self.memory_dir, "global_mem_insight.txt")
        self.l2_path = os.path.join(self.memory_dir, "global_mem.txt")
        self.all_histories_path = os.path.join(self.raw_dir, "all_histories.txt")
        self.history_insight_path = os.path.join(self.memory_dir, "history_insight.md")
        self.cleanup_report_path = os.path.join(self.memory_dir, "cleanup_report.md")
        self.archive_state_path = os.path.join(self.raw_dir, ".archive_state.json")
        self.maintenance_state_path = os.path.join(self.memory_dir, ".maintenance_state.json")

    def ensure(self) -> None:
        os.makedirs(self.raw_dir, exist_ok=True)

    def full_path(self, day: str) -> str:
        return os.path.join(self.raw_dir, f"{day}.md")

    def digest_path(self, day: str) -> str:
        return os.path.join(self.raw_dir, f"{day}-digest.md")

    def zip_path(self, day: str) -> str:
        return os.path.join(self.raw_dir, f"{day}.zip")


# ---------------------------------------------------------------------------
# 低层 IO（全部加锁：主线程落盘 + 后台线程归档可能同时发生）
# ---------------------------------------------------------------------------
def _append(path: str, text: str) -> None:
    with _IO_LOCK:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)


def _read(path: str) -> str:
    with _IO_LOCK:
        if not os.path.isfile(path):
            return ""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except (OSError, UnicodeDecodeError):
            return ""


def _write(path: str, text: str) -> None:
    from .atomic_io import atomic_write_text

    with _IO_LOCK:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        atomic_write_text(path, text)


def _json_state(path: str) -> Dict[str, Any]:
    raw = _read(path)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except (ValueError, TypeError):
        return {}


def _normalize_rel(path: str) -> str:
    p = str(path or "").replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p.lstrip("/")


# ---------------------------------------------------------------------------
# 改进 1 + 2：每轮全文与摘要按当天日期追加
# ---------------------------------------------------------------------------
def record_turn(layout: MemoryLayout, rec: Any,
                system_prompt: Union[str, Callable[[], str]] = "",
                now: Optional[datetime] = None) -> None:
    """把一轮对话全文与摘要分别追加到当天两份文件。

    挂在 ``Session.finish_turn`` 的 ``on_turn_finished`` 回调上：所有收尾分支
    （文字收尾 / 工具轮 / 反思门）都会经过，不会漏轮；被 rollback 的轮不会
    走到 finish_turn，天然不计入有效对话。
    """
    if not layout.workspace:
        return
    now = now or datetime.now()
    day = now.strftime("%Y-%m-%d")
    stamp = now.strftime("%Y-%m-%d %H:%M:%S")
    idx = getattr(rec, "index", 0)
    user_text = str(getattr(rec, "prompt", "") or "")
    assistant_text = str(getattr(rec, "assistant_text", "") or "")
    raw_content = str(getattr(rec, "raw_content", "") or "") or assistant_text
    summary = str(getattr(rec, "summary", "") or "")
    calls = list(getattr(rec, "tool_calls", []) or [])
    results = list(getattr(rec, "tool_results", []) or [])
    call_names = [getattr(c, "name", "") for c in calls]

    full_path = layout.full_path(day)
    chunks: List[str] = []
    # SYSTEM_PROMPT 段只在当天文件首次创建时写一份：它每轮恒定，重复写会让
    # 日志体积按轮次线性膨胀；改进 7 归档时按该标记整段剥离。
    if not os.path.isfile(full_path):
        # 传 callable 可延迟求值：只有当天首轮才需要完整的 system prompt。
        sp = system_prompt() if callable(system_prompt) else system_prompt
        chunks.append(
            f"# 会话全文 {day}\n\n{_H_SYS}\n{sp or '(未注入)'}\n\n"
        )
    chunks.append(f"## [T{idx}] {stamp}\n\n")
    chunks.append(f"{_H_USER}\n{user_text}\n\n")
    # AGENT 骨架：摘要 + 精简正文 + 工具名（改进 7 保留的部分）
    skeleton = summary or "(无摘要)"
    if assistant_text:
        skeleton += f"\n{assistant_text}"
    if call_names:
        skeleton += f"\ntools: {', '.join(call_names)}"
    chunks.append(f"{_H_AGENT}\n{skeleton}\n\n")
    # 完整原文与工具结果：改进 7 归档时剥离
    chunks.append(f"{_H_RESP}\n{raw_content}\n\n")
    if calls or results:
        tool_lines = []
        for c in calls:
            tool_lines.append(f"- call {getattr(c, 'name', '')}: {getattr(c, 'args', {})}")
        for r in results:
            tool_lines.append(f"- result {r.get('name', '')}: {str(r.get('content', ''))}")
        chunks.append(f"{_H_TOOLS}\n" + "\n".join(tool_lines) + "\n\n")
    try:
        _append(full_path, "".join(chunks))
        digest = (
            f"[T{idx}] {now.strftime('%H:%M:%S')}\n"
            f"U: {user_text[:200].replace(chr(10), ' ')}\n"
            f"A: {summary or '(无摘要)'}"
            + (f" | tools: {', '.join(call_names)}" if call_names else "")
            + "\n\n"
        )
        _append(layout.digest_path(day), digest)
    except OSError as e:  # 落盘失败绝不影响主流程
        logger.warning("[memory] record_turn failed: %s", e)


# ---------------------------------------------------------------------------
# 改进 3 + 4：L0 恒定 + L1 实时索引注入 system prompt
# ---------------------------------------------------------------------------
def ensure_l0(layout: MemoryLayout) -> str:
    """把 L0 META-SOP 全文物化到 memory/memory_management_sop.md。

    提炼版只常驻核心公理与指针，模型按指针 ``file_read`` 时必须有真文件可读，
    否则「写记忆前先读全文」会变成死链。只在文件缺失/为空时写，绝不覆盖用户
    手改；顺带把旧版 L1 里无后缀的 L0 指针 patch 成明确路径（只动这两处）。
    """
    if not layout.workspace:
        return ""
    layout.ensure()
    if not os.path.isfile(layout.l0_path) or not _read(layout.l0_path).strip():
        try:
            _write(layout.l0_path, MEMORY_MANAGEMENT_SOP_FULL)
        except OSError as e:
            logger.warning("[memory] ensure_l0 write failed: %s", e)
    l1 = _read(layout.l1_path)
    if l1 and "memory/memory_management_sop.md" not in l1:
        patched = (
            l1.replace("META-SOP(L0):memory_management_sop",
                       "META-SOP(L0):memory/memory_management_sop.md")
              .replace("L0(META-SOP): memory_management_sop",
                       "L0(META-SOP): memory/memory_management_sop.md")
        )
        if patched != l1:
            try:
                _write(layout.l1_path, patched)
            except OSError:
                pass
    return _read(layout.l0_path)


def read_l1(layout: MemoryLayout) -> str:
    """读 L1；不存在则以默认模板初始化（改进 4：文件真实存在且可维护）。"""
    text = _read(layout.l1_path)
    if not text.strip():
        text = DEFAULT_L1_TEMPLATE
        try:
            _write(layout.l1_path, text)
        except OSError:
            pass
    return text


def memory_system_block(layout: MemoryLayout) -> str:
    """system prompt 中紧跟基础提示之后注入的恒定 L0 提炼版 + 实时 L1。"""
    if not layout.workspace:
        return MEMORY_MANAGEMENT_SOP
    # 构建提示词时顺手保证全文文件存在：L0 指针必须可读。
    ensure_l0(layout)
    return MEMORY_MANAGEMENT_SOP + "\n\n---\n" + read_l1(layout).strip()


def _l2_sections(layout: MemoryLayout) -> List[str]:
    raw = _read(layout.l2_path)
    return [m.strip() for m in _SECTION_RE.findall(raw) if m.strip()]


def _l3_names(layout: MemoryLayout) -> List[str]:
    try:
        names = sorted(
            n for n in os.listdir(layout.memory_dir)
            if n.endswith((".md", ".py")) and n not in _SYSTEM_FILES
            and n != _L0_FILE and not n.startswith(".")
        )
    except OSError:
        return []
    return names


def sync_l1(layout: MemoryLayout) -> bool:
    """L2/L3 变更后同步 L1 索引行（改进 4/6）。

    只 patch ``L2:`` / ``L3:`` / ``L4:`` 三行，其余手写 RULES 与高频映射原样
    保留 —— 符合 L0「只能少量 patch、禁 overwrite」红线。返回是否发生变更。
    """
    if not layout.workspace:
        return False
    text = read_l1(layout)
    sections = _l2_sections(layout)
    names = _l3_names(layout)
    l2_line = "L2: " + (" | ".join(sections) if sections else "现空")
    shown = names[:_L1_MAX_L3_ITEMS]
    l3_line = "L3: " + (" | ".join(shown) if shown else "现空")
    if len(names) > len(shown):
        l3_line += f" | (…共 {len(names)} 项, ls memory 查全量)"
    l4_line = "L4: raw_sessions/ 历史会话"

    lines = text.splitlines()
    wanted = {"L2": l2_line, "L3": l3_line, "L4": l4_line}
    seen = set()
    changed = False
    for i, line in enumerate(lines):
        for key, new in wanted.items():
            if key in seen:
                continue
            if re.match(rf"^{key}:\s", line):
                if line != new:
                    lines[i] = new
                    changed = True
                seen.add(key)
                break
    for key, new in wanted.items():
        if key not in seen:
            # 插到 [RULES] 之前，保证索引区连续
            pos = next((i for i, l in enumerate(lines) if l.strip() == "[RULES]"), len(lines))
            lines.insert(pos, new)
            changed = True
    # 硬约束 ≤30 行：只裁索引行本身，绝不删 RULES
    while len(lines) > _L1_MAX_LINES:
        idx = next((i for i, l in enumerate(lines) if l.startswith("L3: ")), None)
        if idx is None:
            break
        items = lines[idx][4:].split(" | ")
        if len(items) <= 1:
            break
        lines[idx] = "L3: " + " | ".join(items[:-1]) + " | (…更多, ls memory)"
        changed = True
    if len(lines) > _L1_MAX_LINES:
        # L3 索引行已裁无可裁仍超限 → 说明是用户手写的映射/RULES 过多。
        # 按 L0「神圣不可删改」，系统绝不擅自删这些行，只如实告警让人整理。
        logger.warning("[memory] L1 仍 %d 行（硬上限 %d）：手写内容过多，"
                       "请人工精简映射/RULES", len(lines), _L1_MAX_LINES)
    if changed:
        try:
            _write(layout.l1_path, "\n".join(lines) + "\n")
        except OSError as e:
            logger.warning("[memory] sync_l1 failed: %s", e)
            return False
    return changed


# ---------------------------------------------------------------------------
# 改进 5 + 6：内核侧工具识别
# ---------------------------------------------------------------------------
def is_l3_sop_read(call: Any) -> bool:
    """file_read 是否读了 memory/ 下的 L3 SOP（改进 5 的触发条件）。"""
    if getattr(call, "name", "") != "file_read":
        return False
    p = _normalize_rel(getattr(call, "args", {}).get("path", ""))
    if not p.startswith("memory/"):
        return False
    name = os.path.basename(p)
    return name.endswith((".md", ".py")) and name not in _SYSTEM_FILES


def is_l23_write(call: Any) -> bool:
    """是否是 L2/L3 的正式写入（改进 6：写完后同步 L1）。"""
    if getattr(call, "name", "") not in ("file_append", "file_write", "file_modify"):
        return False
    p = _normalize_rel(getattr(call, "args", {}).get("path", ""))
    if not p.startswith("memory/"):
        return False
    name = os.path.basename(p)
    if name == _L0_FILE:
        return False  # L0 全文是系统维护文件，不是 L2/L3 写入目标
    if name == "global_mem.txt" or name == "global_mem_insight.txt":
        return True
    return name.endswith((".md", ".py")) and name not in _SYSTEM_FILES


def long_term_update_payload(transcript: str, focus: str, layout: MemoryLayout) -> Dict[str, Any]:
    """改进 6：start_long_term_update 的回执（prompt 门控，不直接写记忆）。"""
    return {
        "status": "MEMORY_PURIFY_REQUESTED",
        "focus": focus or "",
        "prompt": MEMORY_PURIFY_PROMPT,
        "transcript": (transcript or "")[-4000:],
        "decision_tree": (
            "写前先 file_read('memory/memory_management_sop.md') 读 META-SOP 全文与决策树，"
            "读完提炼进 key_info。速记：环境特异性事实 → file_append 到 "
            "memory/global_mem.txt 的 ## [SECTION]；特定任务技术 → 新建或追加 "
            "memory/xxx_sop.md；通用规律 → L1 [RULES] 一句；通用常识/易变状态 → 丢弃"
        ),
        "write_rules": [
            "只写经工具验证成功的信息；No Execution, No Memory",
            "先 file_read 再 patch；memory 下文件只能追加/少量 patch，禁整文件 overwrite 或 code_run",
            "L2/L3 写完后内核会自动同步 L1 索引并重新注入 system prompt",
            "禁止：临时状态、推理过程、未验证信息、通用常识、易重建细节、模型推测",
        ],
        "paths": {
            "L1": "memory/global_mem_insight.txt",
            "L2": "memory/global_mem.txt",
            "L3": "memory/*.md | *.py",
            "L4": "memory/raw_sessions/",
        },
        "current_l2_sections": _l2_sections(layout),
        "current_l3_files": _l3_names(layout),
    }


# ---------------------------------------------------------------------------
# 改进 7：往日会话归档（剥离 → 骨架汇总 → zip → 删原文件）
# ---------------------------------------------------------------------------
def _session_id(text: str, day: str) -> str:
    """从当天全文日志首/末轮时间戳生成与 zip 内文件名一致的 session 标识。"""
    stamps = _TURN_RE.findall(text)
    if stamps:
        first = datetime.strptime(stamps[0][1], "%Y-%m-%d %H:%M:%S")
        last = datetime.strptime(stamps[-1][1], "%Y-%m-%d %H:%M:%S")
        return f"{first.strftime('%m%d_%H%M')}-{last.strftime('%m%d_%H%M')}"
    d = day.replace("-", "")
    return f"{d[4:]}_0000-{d[4:]}_0000"


def _skeleton(text: str) -> str:
    """剥离 SYSTEM_PROMPT / ASSISTANT_RESPONSE / TOOLS 段，仅留 USER/Agent 骨架。"""
    kept: List[str] = []
    # 按 `### ` 分段，段头决定是否保留；`## [T…]` 轮次行单独保留作定位锚。
    parts = re.split(r"(?=^### |^## \[T\d+\] )", text, flags=re.M)
    for part in parts:
        if not part.strip():
            continue
        head = part.splitlines()[0].strip()
        if head.startswith("## [T"):
            kept.append(head)
        elif head in _KEEP_HEADS:
            kept.append(part.rstrip())
    return "\n".join(kept)


def archive_past_sessions(layout: MemoryLayout, today: Optional[str] = None) -> int:
    """把今天之前的对话记录预处理、打包压缩、删除原文件。返回处理天数。"""
    if not layout.workspace:
        return 0
    today = today or date.today().isoformat()
    layout.ensure()
    state = _json_state(layout.archive_state_path)
    done = set(state.get("archived_days") or [])
    processed = 0
    try:
        files = sorted(os.listdir(layout.raw_dir))
    except OSError:
        return 0
    for fname in files:
        m = _DAY_RE.match(fname)
        if not m:
            continue
        day = m.group(1)
        if day >= today or day in done:
            continue
        full_path = layout.full_path(day)
        text = _read(full_path)
        digest_text = _read(layout.digest_path(day))
        sid = _session_id(text, day)
        # 1) 骨架追加到 all_histories.txt（增量：state 保证同一天只追加一次）
        bone = _skeleton(text)
        if bone.strip():
            _append(layout.all_histories_path,
                    f"\n===== session {sid} =====\n{bone}\n")
        # 2) 打包压缩：zip 内文件名与 session 标识一致，便于 grep 回原日志
        try:
            with zipfile.ZipFile(layout.zip_path(day), "w",
                                 compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(f"{sid}.md", text)
                if digest_text:
                    zf.writestr(f"{sid}-digest.md", digest_text)
        except OSError as e:
            logger.warning("[memory] archive zip failed for %s: %s", day, e)
            continue
        # 3) 删除原文件（zip 已落盘成功才删）
        for p in (full_path, layout.digest_path(day)):
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except OSError:
                pass
        done.add(day)
        processed += 1
    if processed:
        state["archived_days"] = sorted(done)
        _write(layout.archive_state_path, json.dumps(state, ensure_ascii=False, indent=2))
    return processed


# ---------------------------------------------------------------------------
# 改进 7：辅助模型闲时语义提炼 → history_insight.md
# ---------------------------------------------------------------------------
_H_ONGOING = "## 活动知识层 / 持续活动"
_H_GONE = "## 活动知识层 / 已消失事项"
_H_EMOTION = "## 情绪事件列表"
_H_CURSOR = "## 增量标记"
_CURSOR_RE = re.compile(r"LAST_SESSION:\s*(\S+)")
# 增量用「已处理 session 集合」而不是游标字符串比较：session 标识是 MMDD_HHMM，
# 跨年时 1231… > 0101…，纯字典序会把新一年的 session 误判成已处理。
_PROCESSED_RE = re.compile(r"PROCESSED:\s*([^\n]*)")


def _insight_template() -> str:
    return (f"{_H_ONGOING}\n（暂无）\n\n{_H_GONE}\n（暂无）\n\n"
            f"{_H_EMOTION}\n（暂无）\n\n{_H_CURSOR}\nLAST_SESSION: -\nPROCESSED:\n")


def _split_insight(text: str) -> Dict[str, str]:
    heads = [_H_ONGOING, _H_GONE, _H_EMOTION, _H_CURSOR]
    out = {h: "" for h in heads}
    positions = [(text.find(h), h) for h in heads if text.find(h) >= 0]
    positions.sort()
    for i, (pos, h) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        out[h] = text[pos + len(h):end].strip()
    return out


def _json_from_llm(raw: str) -> Dict[str, Any]:
    """LLM 输出可能是带代码围栏的 JSON；解析失败返回空 dict（不丢原文）。"""
    s = (raw or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", s)
    start, end = s.find("{"), s.rfind("}")
    if start >= 0 and end > start:
        s = s[start:end + 1]
    try:
        data = json.loads(s)
        return data if isinstance(data, dict) else {}
    except (ValueError, TypeError):
        return {}


def _activity_lines(text: str) -> List[str]:
    return [l.strip().lstrip("- ").strip()
            for l in (text or "").splitlines()
            if l.strip() and l.strip() != "（暂无）"]


def _entry_key(line: str) -> str:
    """活动条目的互斥判据：取第一段（标签+名称），忽略 session/频次元信息。"""
    return line.split("|")[0].strip()[:40]


def _extract_lines(items: Any) -> List[str]:
    if isinstance(items, str) and items.strip():
        return [items.strip()]
    if isinstance(items, list):
        return [str(x).strip() for x in items if str(x).strip()]
    return []


def _merge_activity(old_text: str, incoming: List[str],
                    opponent_incoming: List[str]) -> str:
    """活动知识层是持久状态：旧条目必须保留，新发现合并进来。

    以本批新发现为最终裁决：同名旧条目被新条目替换；若同名条目被对方
    （持续 vs 已消失）本批裁决，则从本侧移走，保证两个区域互斥。
    """
    incoming_keys = {_entry_key(x) for x in incoming}
    opponent_keys = {_entry_key(x) for x in opponent_incoming}
    old = [l for l in _activity_lines(old_text)
           if _entry_key(l) not in incoming_keys
           and _entry_key(l) not in opponent_keys]
    merged = old + [x for x in incoming if x not in old]
    return "\n".join(f"- {x}" for x in merged) or "（暂无）"


def mine_history_insight(layout: MemoryLayout,
                         llm_call: Callable[[str, str], str]) -> str:
    """增量挖掘：只扫「已处理集合」里没有的新 session，写回三个持久状态。"""
    if not layout.workspace:
        return ""
    layout.ensure()
    histories = _read(layout.all_histories_path)
    if not histories.strip():
        return ""
    blocks = re.split(r"(?=^===== session (\S+) =====$)", histories, flags=re.M)
    sessions: List[tuple] = []
    for i in range(1, len(blocks) - 1, 2):
        sessions.append((blocks[i], blocks[i + 1]))
    if not sessions:
        return ""
    insight_text = _read(layout.history_insight_path) or _insight_template()
    parts = _split_insight(insight_text)
    cursor_area = parts.get(_H_CURSOR, "")
    pm = _PROCESSED_RE.search(cursor_area)
    if pm:
        processed = {s for s in re.split(r"[,\s]+", pm.group(1)) if s}
    else:
        # 兼容只有 LAST_SESSION 的旧产物：按游标一次性迁移成集合。
        cm = _CURSOR_RE.search(cursor_area)
        cursor = cm.group(1) if cm else "-"
        processed = {sid for sid, _ in sessions
                     if cursor != "-" and sid <= cursor}
    # 集合判定而非字典序：跨年后 0101… < 1231… 也不会被误判为已处理。
    fresh = [(sid, body) for sid, body in sessions if sid not in processed]
    if not fresh:
        return ""
    source = "\n\n".join(
        f"===== session {sid} ====={body}" for sid, body in fresh
    )[:12000]
    known = (f"现有活动知识层（每次扫描要读取、更新、写回）:\n"
             f"持续:\n{parts.get(_H_ONGOING, '')}\n\n已消失:\n{parts.get(_H_GONE, '')}")
    prompt = (f"{HISTORY_MINING_SOP}\n\n---\n{known}\n\n---\n"
              f"本次需要扫描的新 session 原文（只处理这些，不得拆分同一 session）：\n{source}\n\n"
              "输出 JSON：{\"ongoing\": [...], \"gone\": [...], \"emotions\": [...]}; "
              "每个元素为字符串，必须含 session 标识、关键原文片段、发现类型标签。"
              "ongoing 与 gone 必须互斥；emotions 只增不改。")
    try:
        raw = llm_call(HISTORY_MINING_SOP, prompt)
    except Exception as e:  # 辅助引擎不可用 → 不动增量标记，下次重试
        logger.warning("[memory] history mining failed: %s", e)
        return ""
    data = _json_from_llm(raw)
    if data:
        new_ongoing = _extract_lines(data.get("ongoing"))
        new_gone = _extract_lines(data.get("gone"))
        new_emotions = _extract_lines(data.get("emotions"))
    else:
        # 非 JSON 输出不能丢：作为「未解析发现」保存，增量也照常推进，
        # 否则每次扫描都会卡在同一批 session 上反复调辅助模型。
        new_ongoing = [f"[未解析发现] {raw.strip()[:2000]}"]
        new_gone, new_emotions = [], []

    # 活动知识层是持久状态：旧条目保留 + 新发现合并 + 两侧互斥裁决。
    ongoing = _merge_activity(parts.get(_H_ONGOING, ""), new_ongoing, new_gone)
    gone = _merge_activity(parts.get(_H_GONE, ""), new_gone, new_ongoing)
    # 情绪事件：追加式、只增不改，整行去重
    emotion_lines = _activity_lines(parts.get(_H_EMOTION, ""))
    for line in new_emotions:
        item = line.lstrip("- ").strip()
        if item not in emotion_lines:
            emotion_lines.append(item)
    emotions = "\n".join(f"- {x}" for x in emotion_lines) or "（暂无）"

    processed |= {sid for sid, _ in fresh}
    new_cursor = fresh[-1][0]
    output = (
        f"{_H_ONGOING}\n{ongoing}\n\n"
        f"{_H_GONE}\n{gone}\n\n"
        f"{_H_EMOTION}\n{emotions}\n\n"
        f"{_H_CURSOR}\nLAST_SESSION: {new_cursor}\n"
        f"PROCESSED: {','.join(sorted(processed))}\n"
    )
    _write(layout.history_insight_path, output)
    return new_cursor


# ---------------------------------------------------------------------------
# 改进 8：辅助引擎驱动的记忆整理（L1 只做词级 patch）
# ---------------------------------------------------------------------------
def cleanup_memory(layout: MemoryLayout,
                   llm_call: Callable[[str, str], str]) -> int:
    """按记忆整理 SOP 审计 L1/L2/L3；只自动应用 L1 词级 patch，禁 overwrite。"""
    if not layout.workspace:
        return 0
    l1 = read_l1(layout)
    sections = _l2_sections(layout)
    names = _l3_names(layout)
    prompt = (
        "按 SOP 整理下面的记忆索引。只输出 JSON："
        '{"l1_patches":[{"old":"...","new":"..."}], "report":"..."}。'
        "l1_patches 最多 8 条、每条是词级别替换（old 必须逐字出现在 L1 中）；"
        "禁止删除 RULES 行、禁止新增细节、禁止整文件重写。\n\n"
        f"--- L1（共 {len(l1.splitlines())} 行，硬上限 30）---\n{l1}\n\n"
        f"--- L2 sections ---\n{', '.join(sections) or '(空)'}\n\n"
        f"--- L3 文件名（真实存在）---\n{', '.join(names) or '(空)'}\n"
    )
    try:
        raw = llm_call(MEMORY_CLEANUP_SOP, prompt)
    except Exception as e:
        logger.warning("[memory] cleanup LLM failed: %s", e)
        return 0
    data = _json_from_llm(raw)
    patches = data.get("l1_patches") or []
    if not isinstance(patches, list):
        patches = []
    text = l1
    applied = 0
    for p in patches[:8]:
        if not isinstance(p, dict):
            continue
        old, new = str(p.get("old", "")), str(p.get("new", ""))
        if not old or not new.strip():
            continue  # 禁止整行删除（神圣不可删改）
        if len(old) > 200 or len(new) > 200:
            continue
        pos = text.find(old)
        if pos < 0:
            continue
        # [RULES] 是人工维护的红线区：自动整理只动索引行，绝不碰 RULES 内容。
        if "[RULES]" in text[:pos]:
            continue
        candidate = text.replace(old, new, 1)
        if len(candidate.splitlines()) > _L1_MAX_LINES:
            continue
        text = candidate
        applied += 1
    if applied:
        _write(layout.l1_path, text)
    report = str(data.get("report", "") or raw)[:2000]
    if report.strip():
        _append(layout.cleanup_report_path,
                f"\n## {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                f"(applied {applied} L1 patches)\n{report.strip()}\n")
    return applied


# ---------------------------------------------------------------------------
# 后台维护入口（Agent 在任务终止边调用；每天至多一次完整流程）
# ---------------------------------------------------------------------------
def run_memory_maintenance(layout: MemoryLayout,
                           llm_call: Optional[Callable[[str, str], str]] = None,
                           force: bool = False) -> Dict[str, Any]:
    """归档（同步、轻量）+ 挖掘/整理（辅助引擎，调用方保证空闲）。"""
    result: Dict[str, Any] = {"archived": 0, "mined": "", "cleaned": 0}
    if not layout.workspace:
        return result
    today = date.today().isoformat()
    state = _json_state(layout.maintenance_state_path)
    if not force and state.get("last_day") == today:
        # 归档每天都可能有新文件，天天跑；挖掘/整理每天一次
        result["archived"] = archive_past_sessions(layout, today)
        return result
    result["archived"] = archive_past_sessions(layout, today)
    if llm_call is not None:
        result["mined"] = mine_history_insight(layout, llm_call)
        result["cleaned"] = cleanup_memory(layout, llm_call)
    state["last_day"] = today
    _write(layout.maintenance_state_path, json.dumps(state, ensure_ascii=False, indent=2))
    return result
