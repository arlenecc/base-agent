"""RAG 后处理：用辅助模型把检索到的多个文本分片归纳精简，减轻主模型上下文压力。

流程位置（见需求）：RAG 重排结果不直送主模型，而是先把全部文本分片交给辅助
模型做归纳整理（保留核心观点与重要信息），浓缩到 ``max_chars`` 字以内，再把
精简结果作为工具消息注入主模型上下文。主模型用完该内容后的五级信息压缩
（L2 回合摘要 / L3 会话折叠 / L4 记忆蒸馏）由既有内核管线自动接续，无需在此接线。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MAX_CHARS = 2000
# 送进辅助模型的总输入上限：辅助模型（如 2B 级别）上下文有限，检索分片可能很长，
# 超过此上限按相关性顺序截断尾部（重排已按相关度排序，保留头部最关键内容）。
_MAX_AUX_INPUT_CHARS = 12_000


_CONDENSE_SYSTEM = (
    "你是一个严谨的信息压缩助手。下面是从知识库检索到的若干文本片段"
    "（已按相关性从高到低排序）。请通读全部内容，提炼核心观点与重要信息，"
    "删除冗余、重复以及与查询无关的细枝末节，将全部内容浓缩到 {max_chars} 字以内"
    "（中文按字符计数）。\n"
    "要求：\n"
    "1. 保留关键事实、数据、定义、结论与来源线索，不得臆造或篡改原文信息；\n"
    "2. 保持信息准确、连贯、可直接被下游模型引用；\n"
    "3. 输出纯文本摘要，可用必要分段，但不要添加前言、客套话或超出原文的评论；\n"
    "4. 若检索内容不足以回答查询，应如实说明已知与未知。"
)


def _stream_collect(llm: Any, system: str, prompt: str, temperature: float = 0.2) -> str:
    """用流式接口向 llm 提问并收集全部 content，返回纯文本（done 即止）。"""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    out: List[str] = []
    for ev in llm.chat_stream(messages, tools=None, temperature=temperature):
        if ev.type == "content":
            out.append(ev.content)
        elif ev.type == "done":
            break
    return "".join(out)


def condense_retrieved_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    llm: Any,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """把检索到的多个分片交给辅助模型归纳精简到 ``max_chars`` 字以内。

    返回精简后的摘要文本；辅助模型不可用或调用失败时返回空串（调用方应回退到
    原始格式化结果，不影响 RAG 功能）。
    """
    usable = [r for r in (chunks or []) if (r.get("text") or "").strip()]
    if not usable:
        return ""

    parts: List[str] = []
    for i, r in enumerate(usable, 1):
        src = r.get("source") or ""
        parts.append(f"[片段 {i}] 来源: {src}\n{r.get('text') or ''}")
    doc = "\n\n".join(parts)

    # 送进辅助模型的输入做上限保护，避免超出其上下文窗口。
    if len(doc) > _MAX_AUX_INPUT_CHARS:
        doc = doc[:_MAX_AUX_INPUT_CHARS].rstrip() + "\n…（输入过长，已截断尾部低相关片段）"

    prompt = (
        f"检索查询：{query}\n\n"
        f"检索到的文本片段（共 {len(usable)} 条）：\n{doc}\n\n"
        f"请按系统要求将上述内容浓缩到 {max_chars} 字以内。"
    )
    system = _CONDENSE_SYSTEM.format(max_chars=max_chars)

    try:
        out = _stream_collect(llm, system, prompt, temperature=0.2).strip()
    except Exception as e:  # 辅助模型调用失败不应让 RAG 整体失败
        logger.warning("辅助模型精简 RAG 结果失败，回退原始结果: %s", e)
        return ""

    if not out:
        return ""

    # 兜底：若辅助模型未遵守字数上限，硬截断（保留头部最关键内容）。
    if len(out) > max_chars:
        out = out[:max_chars].rstrip() + "…（已截断）"
    return out


def build_condensed_output(
    query: str,
    chunks: List[Dict[str, Any]],
    condensed: str,
) -> str:
    """把精简摘要包装成工具结果文本，并附带来源文件线索供主模型引用。"""
    sources = sorted({_basename(r.get("source") or "") for r in (chunks or []) if r.get("source")})
    src_line = f"（来源文件：{', '.join(sources)}）" if sources else ""
    return (
        f"从知识库检索到 {len(chunks)} 条相关内容（已重排序），"
        f"经辅助模型归纳精简如下{src_line}：\n\n{condensed}"
    )


def _basename(path: str) -> str:
    import os
    return os.path.basename(path)
