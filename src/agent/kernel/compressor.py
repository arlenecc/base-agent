"""五级信息压缩（L0 → L4）。

    L0 原始捕获   工具返回瞬间：硬上限，head+tail 保头保尾
    L1 工具级裁剪 注入本轮前：按预算分配 maxlen，结构化降噪
    L2 回合摘要   每轮结束：强制 <summary> 30 字物理快照，替换整轮原文
    L3 会话折叠   预算告罄/每 N 轮：旧轮 → 一行摘要 + 同类项合并
    L4 记忆蒸馏   任务完成/检查点：沉淀进工作记忆 → 图谱 → SKILL.md

L2 + L3 是「历史恒定 O(1)」的关键：base-agent 旧实现把整条历史每轮重发，
上下文随轮次线性膨胀，只能靠超阈值后的被动摘要兜底。
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .budget import estimate_tokens

# ---------------------------------------------------------------------------
# 协议标签（与 GenericAgent 一致）
# ---------------------------------------------------------------------------

_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL | re.IGNORECASE)
# 代码块：L2 摘要前先剥掉，避免把代码当摘要
_CODE_RE = re.compile(r"```[\s\S]*?```")
_TAG_RE = re.compile(r"<(thinking|think|tool_use|tool_result)>[\s\S]*?</\1>", re.I)

# 噪声标签：L1 降噪时整块剔除（保留内部文本）
_NOISE_HTML = re.compile(
    r"<(script|style|noscript|svg|head)\b[\s\S]*?</\1>", re.I
)
# HTML 标签白名单：只剥**真正的 HTML 标签**。旧正则 ``<[^>]{1,200}>`` 命中
# 任何尖括号对——``std::vector<int>`` → ``std::vector ``、``Map<String,Object>``
# → ``Map ``、``if a < b && c > d`` → ``if a b``：file_read / shell_run /
# code_run 返回的源码、配置、日志被静默改写后注入上下文，模型读到的是错误
# 内容（且不会报错）。这里限定标签名 + 允许属性，非白名单尖括号原样保留。
_HTML_TAG = re.compile(
    r"</?(?:div|span|p|a|br|img|table|tr|td|th|script|style|noscript|svg|head"
    r"|body|html|ul|ol|li|h[1-6]|pre|code|blockquote|em|strong|b|i|u|font"
    r"|form|input|button|label|section|article|header|footer|nav|main"
    r"|dl|dt|dd|caption|col|colgroup|thead|tbody|tfoot)\b[^>]{0,200}>?",
    re.I,
)
_MULTI_BLANK = re.compile(r"(\r?\n){3,}")


def _head_tail(text: str, max_chars: int) -> str:
    """保头保尾的截断。错误/总结常在结尾，因此尾部必须留。"""
    if len(text) <= max_chars:
        return text
    marker = f"\n...[已裁剪 {len(text)} → {max_chars} 字符；需完整内容请重新调用]\n"
    head = max_chars // 2
    tail = max_chars - head - len(marker)
    if tail <= 0:
        return text[:max_chars]
    return text[:head] + marker + text[-tail:]


# ---------------------------------------------------------------------------
# L0：原始捕获
# ---------------------------------------------------------------------------

def l0_capture(raw: Any, hard_cap: int = 20_000) -> str:
    """工具返回的那一刻就上硬上限。

    工具内部通常已有自己的上限（file_read 200K、shell_run 200KB…），
    这里是内核侧的统一兜底，保证任何工具都不可能把 MB 级内容带进上下文。
    """
    if raw is None:
        return ""
    if isinstance(raw, (dict, list)):
        try:
            text = json.dumps(raw, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(raw)
    else:
        text = str(raw)
    return _head_tail(text, hard_cap)


# ---------------------------------------------------------------------------
# L1：工具级裁剪
# ---------------------------------------------------------------------------

def l1_trim(text: str, max_chars: int, denoise: bool = True) -> str:
    """注入本轮前的裁剪 + 结构化降噪。

    ``max_chars`` 由 ``ContextBudget.remaining_for_tool()`` 动态算出 —— 预算
    紧张时自动收紧，不用改任何工具代码。
    """
    if not text:
        return ""
    if denoise:
        text = _NOISE_HTML.sub(" ", text)
        if "<" in text and ">" in text:
            text = _HTML_TAG.sub(" ", text)
        text = _MULTI_BLANK.sub("\n\n", text)
        text = text.strip()
    return _head_tail(text, max_chars)


def l1_trim_result(name: str, content: str, max_chars: int) -> str:
    """带工具名前缀的 L1 裁剪，便于模型区分多工具结果。"""
    body = l1_trim(content, max_chars)
    return f"<tool_result name={name}>\n{body}\n</tool_result>"


# ---------------------------------------------------------------------------
# L2：回合摘要
# ---------------------------------------------------------------------------

def l2_summary(content: str, tool_names: Optional[List[str]] = None,
               max_chars: int = 60) -> str:
    """抽取本轮的物理快照（≤30 字）。

    优先用模型按协议输出的 ``<summary>``；缺失时回退到「剥掉代码/标签后的
    首句」，再不行就用工具名拼一句。**永不返回空串** —— 空摘要会让 L3 折叠
    区出现无法追溯的空洞。
    """
    if not content:
        if tool_names:
            return "调用 " + ",".join(tool_names[:3])
        return "（无文本输出）"

    m = _SUMMARY_RE.search(content)
    if m:
        s = m.group(1).strip().replace("\n", " ")
        if s:
            return s[:max_chars]

    stripped = _CODE_RE.sub(" ", content)
    stripped = _TAG_RE.sub(" ", stripped)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    if stripped:
        # 取第一句：中文句末标点直接切；英文句点需跟空白，避免把 "1.5" 切断。
        first = re.split(r"(?<=[。！？；!?])|(?<=[.])\s", stripped)[0]
        return (first or stripped)[:max_chars]
    if tool_names:
        return "调用 " + ",".join(tool_names[:3])
    return "（空回复）"


def strip_summary_tags(content: str) -> str:
    """把 ``<summary>`` 从回复正文里剥掉（UI 展示 / 入库前用）。"""
    return _SUMMARY_RE.sub("", content).strip()


def has_summary_tag(content: str) -> bool:
    """模型是否**按协议**输出了 ``<summary>``。

    内核的「回复必须带 <summary>」纪律检查要用这个，不能用 ``l2_summary``
    的返回值判空：后者对任何非空 content 都保证返回非空字符串（最差也给
    「（空回复）」），拿它判断会让纪律提醒变成永不触发的死代码。
    """
    return bool(content) and _SUMMARY_RE.search(content) is not None


# ---------------------------------------------------------------------------
# L3：会话折叠
# ---------------------------------------------------------------------------

def l3_fold(lines: List[str], max_lines: int = 40,
            dedupe_window: int = 40) -> str:
    """把摘要行折叠成 ``<earlier_context>`` 块。

    连续相同的摘要合并为 ``（N turns）`` —— 这是长任务里最有效的一招：
    模型反复重试同一动作时，几十轮会塌缩成一行。

    「相同」按**剥掉 ``[T{n}]`` 索引前缀后的正文**比较：每一行都来自
    ``TurnRecord.digest_line()``，恒带唯一轮次前缀，直接比整行永远不相等
    （旧实现如此，``cnt`` 永不递增，合并是死代码）。输出仍保留原行，
    可追溯性不受影响。
    """
    if not lines:
        return ""

    def _key(line: str) -> str:
        return line.split("] ", 1)[-1] if line.startswith("[T") else line

    lines = lines[-dedupe_window:]
    folded: List[str] = []
    cnt = 0
    last = ""
    last_key = ""
    for line in lines:
        k = _key(line)
        if k == last_key and k:
            cnt += 1
            continue
        if last:
            folded.append(f"{last}（{cnt + 1} turns）" if cnt else last)
        last = line
        last_key = k
        cnt = 0
    if last:
        folded.append(f"{last}（{cnt + 1} turns）" if cnt else last)
    return "\n".join(folded[-max_lines:])


# ---------------------------------------------------------------------------
# L4：记忆蒸馏
# ---------------------------------------------------------------------------

DISTILL_SYSTEM = (
    "你是记忆提纯器。只保留【经工具调用验证成功、且长期有效】的信息：\n"
    "- 环境事实（路径、端口、版本、配置值）\n"
    "- 用户偏好与红线\n"
    "- 踩坑点与避坑方法\n"
    "禁止写入：临时状态、推理过程、未验证信息、通用常识、易重建细节、"
    "模型推测（不得记作用户要求）。\n"
    "以更少文字保留更高复用价值；无提纯价值则输出空列表。"
)

DISTILL_PROMPT = """把下面这一段任务记录提纯为结构化条目，输出 JSON，不要任何前言：

{{"facts": ["..."], "lessons": ["..."], "skill": {{"name": "", "summary": "", "when_to_use": "", "steps": ""}}}}

- facts：验证过的环境事实/用户偏好（路径、端口、版本、配置值），每条 ≤ 30 字，最多 5 条
- lessons：踩坑与避坑方法，每条 ≤ 40 字，最多 3 条
- skill：若该流程**明确可复用**（同类任务下次能照做），给出
  - name：英文小写连字符 id，如 extract-pdf-tables
  - summary：一句话说明做什么（≤ 50 字）
  - when_to_use：触发场景（≤ 40 字）
  - steps：编号步骤（≤ 600 字，只写验证过的步骤）
  否则 skill.name 留空字符串
- 没有内容就输出空列表/空字符串，不要编造

任务记录：
{transcript}
"""


def l4_extract_json(text: str) -> Dict[str, Any]:
    """从 LLM 提纯输出里稳健地取出 JSON（容忍 ```json 围栏与尾随文字）。"""
    if not text:
        return {}
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s).strip()
    try:
        data = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        start, end = s.find("{"), s.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            data = json.loads(s[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            return {}
    return data if isinstance(data, dict) else {}


def l4_normalize(data: Dict[str, Any]) -> Dict[str, Any]:
    """归一化提纯结果，所有字段都有确定类型，避免下游 KeyError/TypeError。"""
    def _strlist(v: Any, limit: int, cap: int) -> List[str]:
        if not isinstance(v, list):
            return []
        out = [str(x).strip() for x in v if isinstance(x, (str, int, float))]
        return [x for x in out if x][:limit][:cap]

    skill = data.get("skill")
    name = summary = when_to_use = steps = ""
    if isinstance(skill, dict):
        name = str(skill.get("name", "") or "").strip()
        summary = str(skill.get("summary", "") or "").strip()
        when_to_use = str(skill.get("when_to_use", "") or "").strip()
        steps = str(skill.get("steps", "") or "").strip()
    # skill 名只允许 [a-z0-9-]，防止写穿目录
    name = re.sub(r"[^a-z0-9\-]", "-", name.lower()).strip("-")[:64]
    return {
        "facts": _strlist(data.get("facts"), 5, 5),
        "lessons": _strlist(data.get("lessons"), 3, 3),
        "skill": {
            "name": name,
            "summary": summary[:200],
            "when_to_use": when_to_use[:400],
            "steps": steps[:6000],
        },
    }


# ---------------------------------------------------------------------------
# 助手：把任意内容压到指定 token 数以内
# ---------------------------------------------------------------------------

def fit_tokens(text: str, max_tokens: int) -> str:
    """把文本压到 ``max_tokens`` 以内（先降噪，再 head+tail）。"""
    if not text:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    return _head_tail(l1_trim(text, len(text)), max(200, int(max_tokens * 3)))


def compact_json(obj: Any, max_chars: int = 400) -> str:
    """把 dict/list 压成紧凑字符串（工作记忆注入系统提示时用）。"""
    try:
        s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        s = str(obj)
    return s if len(s) <= max_chars else s[:max_chars] + "…"
