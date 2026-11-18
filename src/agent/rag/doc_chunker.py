"""长文档语义切片：标题优先、语义低谷、句子边界、代码/表格原子块。

约束（与产品需求一致）：
* 优先按 Markdown / 中文章节标题分块，每个「标题 + 正文」是一个候选分片；
* 单段过长时才做语义切分，切点落在相邻句子相似度的低谷；
  句向量由 model2vec 静态嵌入计算（不是 ColBERT）；
* 切片 7500~8000 字（末尾/短文档除外），1% overlap；
* 断点必须落在句子结束处；代码块、表格（不超长时）整体保留；
* 章节结构失效时退回递归切分，禁止把一句话切成两半。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_MAX_CHARS = 8000
DEFAULT_MIN_CHARS = 7500
DEFAULT_OVERLAP_RATIO = 0.01
DEFAULT_SEMANTIC_THRESHOLD = 0.55
MIN_SEMANTIC_THRESHOLD = 0.15


@dataclass
class TextChunk:
    title: str
    content: str
    start: int = 0
    end: int = 0


@dataclass
class _Section:
    title: str
    heading: str
    level: int
    text: str
    start: int
    end: int


@dataclass
class _Unit:
    text: str
    start: int
    end: int
    atomic: bool = False
    code: bool = False


# ---------------------------------------------------------------------------
# 标题识别
# ---------------------------------------------------------------------------

_ATX_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_CN_CHAPTER_RE = re.compile(
    r"^第([零一二三四五六七八九十百千两0-9]+)([章节部分篇])\s*(.*)$")
_CN_NUM_RE = re.compile(r"^([一二三四五六七八九十百]+)[、.．]\s*(.+)$")
_ARABIC_HEAD_RE = re.compile(r"^(\d+(?:\.\d+){0,3})[、.．]?\s+(\S.*)$")
_BOLD_HEAD_RE = re.compile(r"^\*\*([^*]{2,80})\*\*\s*$")
_SETEXT_RE = re.compile(r"^(=+|-+)\s*$")
_SENT_END = "。！？；.!?;"


def _looks_like_heading(line: str) -> Optional[Tuple[int, str]]:
    text = (line or "").strip()
    if not text or len(text) > 120:
        return None
    m = _ATX_RE.match(text)
    if m:
        return len(m.group(1)), m.group(2).strip()
    m = _CN_CHAPTER_RE.match(text)
    if m:
        level = 1 if m.group(2) in ("章", "篇", "部") else 2
        return level, text
    m = _CN_NUM_RE.match(text)
    if m:
        return 2, text
    m = _BOLD_HEAD_RE.match(text)
    if m:
        return 3, m.group(1).strip()
    m = _ARABIC_HEAD_RE.match(text)
    if m and not text.endswith(tuple(_SENT_END)):
        level = m.group(1).count(".") + 1
        return min(level, 6), text
    return None


def _split_sections(text: str, doc_title: str) -> List[_Section]:
    """按标题切段；代码围栏内的 # 不当作标题。"""
    lines = text.splitlines(keepends=True)
    offsets: List[int] = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line)
    total = len(text)

    heading_rows: List[Tuple[int, int, str]] = []  # (line_idx, level, title)
    in_fence = False
    fence_mark = ""
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if stripped.startswith(("```", "~~~")):
            mark = stripped[:3]
            if not in_fence:
                in_fence, fence_mark = True, mark
            elif mark == fence_mark:
                in_fence, fence_mark = False, ""
            i += 1
            continue
        if not in_fence:
            head = _looks_like_heading(raw)
            # Setext 标题：下一行是 === 或 ---
            if head is None and i + 1 < len(lines) and stripped:
                under = _SETEXT_RE.match(lines[i + 1].strip())
                if under and len(stripped) <= 80:
                    level = 1 if under.group(1).startswith("=") else 2
                    heading_rows.append((i, level, stripped))
                    i += 2
                    continue
            if head is not None:
                heading_rows.append((i, head[0], head[1]))
        i += 1

    sections: List[_Section] = []
    if not heading_rows:
        return [_Section(doc_title or "Document", "", 0, text, 0, total)]

    first_line = heading_rows[0][0]
    if first_line > 0:
        preface = text[:offsets[first_line]]
        if preface.strip():
            sections.append(_Section(
                doc_title or "Document", "", 0, preface, 0, len(preface)))

    stack: List[Tuple[int, str]] = []
    for idx, (line_idx, level, heading) in enumerate(heading_rows):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading))
        path = " > ".join(h for _, h in stack)
        start = offsets[line_idx]
        end = (offsets[heading_rows[idx + 1][0]]
               if idx + 1 < len(heading_rows) else total)
        section_text = text[start:end]
        # 只有标题、没有正文的父级标题（如文档大标题）不单独成片；
        # 它的层级仍保留在子章节的 title path 里。
        body_lines = section_text.splitlines()[1:]
        if not "\n".join(body_lines).strip() and idx + 1 < len(heading_rows):
            continue
        sections.append(_Section(path, heading, level, section_text,
                                 start, end))
    return sections


# ---------------------------------------------------------------------------
# 句子 / 代码 / 表格原子块
# ---------------------------------------------------------------------------

_SENT_RE = re.compile(
    r"[。！？；!?;]+|[.](?=\s|$)|[\n]{2,}")


def _split_sentences(text: str, base: int = 0) -> List[_Unit]:
    """按句末标点切句；保留标点，绝不产生半句。"""
    units: List[_Unit] = []
    start = 0
    for m in _SENT_RE.finditer(text):
        end = m.end()
        piece = text[start:end]
        if piece.strip():
            units.append(_Unit(piece, base + start, base + end))
        start = end
    if start < len(text) and text[start:].strip():
        units.append(_Unit(text[start:], base + start, base + len(text)))
    return units


def _split_units(section: _Section) -> List[_Unit]:
    """把 section 内容拆成句子；代码围栏/表格作为原子单元。"""
    text = section.text
    units: List[_Unit] = []
    lines = text.splitlines(keepends=True)
    offsets: List[int] = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line)

    i = 0
    para_start = None
    para_buf: List[str] = []

    def flush_para(end_pos: int):
        nonlocal para_start, para_buf
        if para_buf:
            para = "".join(para_buf)
            units.extend(_split_sentences(para, section.start + (para_start or 0)))
        para_start, para_buf = None, []

    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if stripped.startswith(("```", "~~~")):
            flush_para(offsets[i])
            mark = stripped[:3]
            start = offsets[i]
            j = i + 1
            while j < len(lines):
                if lines[j].strip().startswith(mark):
                    j += 1
                    break
                j += 1
            end = offsets[j] if j < len(lines) else len(text)
            units.append(_Unit(text[start:end], section.start + start,
                               section.start + end, atomic=True, code=True))
            i = j
            continue

        # Markdown 表格：连续包含 | 的行（含分隔行）作为一个原子块。
        if "|" in raw and _is_table_line(raw):
            flush_para(offsets[i])
            start = offsets[i]
            j = i + 1
            while j < len(lines) and "|" in lines[j] and _is_table_line(lines[j]):
                j += 1
            end = offsets[j] if j < len(lines) else len(text)
            block = text[start:end]
            if block.count("\n") >= 2 or _has_table_separator(block):
                units.append(_Unit(block, section.start + start,
                                   section.start + end, atomic=True))
                i = j
                continue

        if not stripped:
            flush_para(offsets[i])
            i += 1
            continue
        if para_start is None:
            para_start = offsets[i]
        para_buf.append(raw)
        i += 1
    flush_para(len(text))
    return units


def _is_table_line(line: str) -> bool:
    return line.count("|") >= 2


def _has_table_separator(block: str) -> bool:
    for line in block.splitlines():
        if re.match(r"^\s*\|?[\s:|-]{3,}\|?\s*$", line) and "-" in line:
            return True
    return False


# ---------------------------------------------------------------------------
# Token 剪枝（ColBERT 编码前）
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "for",
    "of", "to", "in", "on", "at", "by", "with", "from", "as", "is", "are",
    "was", "were", "be", "been", "being", "this", "that", "these", "those",
    "it", "its", "we", "you", "they", "he", "she", "i", "not", "no",
    "do", "does", "did", "can", "could", "should", "would", "will",
    "has", "have", "had", "there", "their", "our", "your", "about",
    "的", "了", "和", "与", "及", "或", "在", "是", "有", "为", "对",
    "就", "都", "而", "也", "这", "那", "一个", "一种", "一些", "我们",
    "你们", "他们", "可以", "进行", "通过", "以及", "但是", "因为",
    "所以", "如果", "然后", "并且", "其中", "对于", "关于", "这个",
    "那个", "这些", "那些", "其", "之", "于", "把", "被", "让",
}
# 保留名词/动词/专名/英文/**数词**（5000、24 等是检索高信息词）/**形容词**
# （重要、危险、可怕等 a/ad/an 常是问句的语义核心；旧实现把它们从文档向量里
# 删掉，导致「为什么可怕」这类查询在索引里根本不存在对应词）；量词只保留
# 时间单位（天/小时/年…），避免「个/次」这类低信息词。
_KEEP_POS_PREFIX = ("n", "v", "eng", "j", "l", "i", "m", "a")
_KEEP_POS_EXACT = {"nz", "nr", "ns", "nt", "vn", "vd", "an"}
_KEEP_Q = {"天", "日", "周", "月", "年", "小时", "分钟", "秒"}


def _join_pruned(tokens: Sequence[str]) -> str:
    """中文词直接拼接（去掉空格），英文/数字两侧保留一个空格。

    用户要求「剔除所有空格，只留纯文字」；`" ".join` 会把中文词全部拆开，
    让 ColBERT 的 tokenizer 产生大量碎片、MaxSim 明显下降（实测长切片
    7.1 → 8.7）。英文/数字之间仍需空格，否则会粘成一个不存在的词。
    """
    out = ""
    prev_latin = False
    for token in tokens:
        is_latin = bool(re.search(r"[A-Za-z0-9]", token))
        if out and (is_latin or prev_latin):
            out += " "
        out += token
        prev_latin = is_latin
    return out


def prune_tokens(text: str, is_code: bool = False) -> str:
    """清理标点/空白/停用词/低信息量词，只留核心词。

    jieba.posseg 可用时按词性保留名词/动词/专名/英文/数词；不可用时退回
    「CJK 连续片段 + 英文/数字」正则。代码块不按词性过滤，保留标识符。
    """
    body = (text or "").strip()
    if not body:
        return ""
    if is_code:
        tokens = re.findall(r"[A-Za-z_][\w./:-]*|\d+(?:\.\d+)*|[\u4e00-\u9fff]{2,}",
                            body)
        return " ".join(t for t in tokens
                        if t.lower() not in _STOPWORDS)

    try:
        import jieba.posseg as pseg
        kept: List[str] = []
        for word, flag in pseg.cut(body):
            w = word.strip()
            if not w or w.lower() in _STOPWORDS:
                continue
            keep = (flag in _KEEP_POS_EXACT
                    or any(flag.startswith(p) for p in _KEEP_POS_PREFIX)
                    or (flag == "q" and w in _KEEP_Q))
            if keep:
                if len(w) == 1 and flag.startswith(("u", "p", "c", "d", "r")):
                    continue
                kept.append(w)
        if kept:
            return _join_pruned(kept)
    except Exception:
        pass

    tokens = re.findall(
        r"[A-Za-z][A-Za-z0-9_.-]+|\d+(?:\.\d+)*|[\u4e00-\u9fff]{2,}", body)
    return _join_pruned([t for t in tokens if t.lower() not in _STOPWORDS])


_HAS_CONTENT_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")


def clean_query_tokens(text: str) -> str:
    """查询侧轻量清理：只去标点/空白/停用词，保留形容词与疑问词。

    与文档侧的 ``prune_tokens``（按词性过滤）不同：查询不能做词性过滤，
    否则「为什么/可怕/重要」等语义核心会被一并删掉（实测会显著掉点）。
    这里仅剔除无法参与语义匹配的标点/空白和低信息停用词；清理后为空时
    返回空串，由调用方回退到原查询。
    """
    body = (text or "").strip()
    if not body:
        return ""
    try:
        import jieba
        tokens = [w.strip() for w in jieba.cut(body)]
    except Exception:
        tokens = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", body)
    kept: List[str] = []
    for word in tokens:
        w = word.strip()
        if not w or not _HAS_CONTENT_RE.search(w):
            continue          # 纯标点/空白 token
        if w.lower() in _STOPWORDS:
            continue          # 低信息停用词；疑问词（什么/为什么/怎么…）不在表内
        kept.append(w)
    return _join_pruned(kept)


# ---------------------------------------------------------------------------
# 语义低谷
# ---------------------------------------------------------------------------

def _unit_vectors(units: Sequence[_Unit], encoder: Any) -> Optional[np.ndarray]:
    if encoder is None or not units:
        return None
    texts = [prune_tokens(u.text, u.code) or u.text for u in units]
    try:
        emb_list = encoder.encode_documents(texts, batch_size=16)
    except Exception:
        return None
    vecs: List[np.ndarray] = []
    for raw in emb_list:
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 1 and arr.size:
            # model2vec 等静态句向量：一句一个向量，直接使用。
            v = arr
        elif arr.ndim == 2 and arr.size:
            # ColBERT 等词向量矩阵：对 token 向量做均值池化。
            v = arr.mean(axis=0)
        else:
            return None
        n = float(np.linalg.norm(v))
        vecs.append(v / max(n, 1e-12))
    return np.vstack(vecs)


def _lexical_sim(a: _Unit, b: _Unit) -> float:
    ta = set(re.findall(r"[\w\u4e00-\u9fff]+", a.text.lower()))
    tb = set(re.findall(r"[\w\u4e00-\u9fff]+", b.text.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _cut_scores(units: Sequence[_Unit], encoder: Any,
                semantic_threshold: float) -> Tuple[List[float], float]:
    """返回相邻单元相似度（长度 n-1）与自适应阈值。"""
    vecs = _unit_vectors(units, encoder)
    if vecs is not None and len(vecs) == len(units):
        sims = [float(np.dot(vecs[i], vecs[i + 1]))
                for i in range(len(vecs) - 1)]
    else:
        sims = [_lexical_sim(units[i], units[i + 1])
                for i in range(len(units) - 1)]
    threshold = semantic_threshold
    if len(sims) >= 5:
        adaptive = float(np.percentile(sims, 25))
        threshold = max(MIN_SEMANTIC_THRESHOLD,
                        min(semantic_threshold, adaptive))
    return sims, threshold


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def _oversized_atomic(unit: _Unit, max_chars: int) -> List[_Unit]:
    """超长代码/表格：按行切成不超过 max_chars 的块（否则无法容纳）。"""
    if len(unit.text) <= max_chars:
        return [unit]
    out: List[_Unit] = []
    buf: List[str] = []
    size = 0
    offset = unit.start
    for line in unit.text.splitlines(keepends=True):
        if buf and size + len(line) > max_chars:
            text = "".join(buf)
            out.append(_Unit(text, offset, offset + len(text), True, unit.code))
            offset += len(text)
            buf, size = [], 0
        buf.append(line)
        size += len(line)
    if buf:
        text = "".join(buf)
        out.append(_Unit(text, offset, offset + len(text), True, unit.code))
    return out


def _recursive_split_plain(unit: _Unit, max_chars: int) -> List[_Unit]:
    """结构/语义都失效时的递归兜底：优先换行、空格，最后才硬切。"""
    if len(unit.text) <= max_chars:
        return [unit]
    text = unit.text
    cut = text.rfind("\n", 0, max_chars)
    if cut < max_chars // 2:
        cut = text.rfind(" ", 0, max_chars)
    if cut < max_chars // 2:
        cut = text.rfind("。", 0, max_chars)
    if cut < max_chars // 2:
        cut = max_chars
    cut = max(1, cut)
    head = _Unit(text[:cut], unit.start, unit.start + cut)
    tail = _Unit(text[cut:], unit.start + cut, unit.end)
    return [head] + _recursive_split_plain(tail, max_chars)


def _split_long_section(section: _Section, encoder: Any, max_chars: int,
                        min_chars: int, overlap_ratio: float,
                        semantic_threshold: float) -> List[TextChunk]:
    units: List[_Unit] = []
    for u in _split_units(section):
        if u.atomic:
            units.extend(_oversized_atomic(u, max_chars))
        elif len(u.text) > max_chars:
            units.extend(_recursive_split_plain(u, max_chars))
        else:
            units.append(u)
    if not units:
        return []
    if sum(len(u.text) for u in units) <= max_chars:
        return [TextChunk(section.title, section.text.strip(),
                          section.start, section.end)]

    sims, threshold = _cut_scores(units, encoder, semantic_threshold)
    # 代码/表格边界永远安全；语义低谷作为优先切点。
    safe = [True] * max(1, len(units) - 1)
    semantic = [False] * max(1, len(units) - 1)
    for i, sim in enumerate(sims):
        if units[i].atomic or units[i + 1].atomic:
            semantic[i] = True
        elif sim < threshold:
            semantic[i] = True

    lens = [len(u.text) for u in units]
    cum = [0]
    for n in lens:
        cum.append(cum[-1] + n)
    overlap_chars = max(1, int(max_chars * overlap_ratio))
    chunks: List[TextChunk] = []
    start = 0
    n = len(units)
    while start < n:
        remaining = cum[n] - cum[start]
        if remaining <= max_chars:
            end = n
        else:
            end = None
            last_any = None
            for i in range(start + 1, n + 1):
                length = cum[i] - cum[start]
                if length > max_chars:
                    break
                last_any = i
                if length >= min_chars and semantic[i - 1]:
                    end = i
            if end is None:
                end = last_any or min(start + 1, n)
        if end <= start:
            end = min(start + 1, n)
        text = "".join(u.text for u in units[start:end]).strip()
        if text:
            chunks.append(TextChunk(
                section.title, text,
                units[start].start, units[end - 1].end))
        if end >= n:
            break
        # 1% overlap：从切点向前回退整句，避免从半句开始。
        next_start = end
        overlap = 0
        while next_start > start + 1 and overlap < overlap_chars:
            next_start -= 1
            overlap += lens[next_start]
        start = max(next_start, start + 1)

    # 段内小切片合并（末尾除外）。
    merged: List[TextChunk] = []
    for chunk in chunks:
        if (merged and (len(chunk.content) < min_chars
                        or len(merged[-1].content) < min_chars)
                and len(merged[-1].content) + len(chunk.content) <= max_chars):
            merged[-1].content = (merged[-1].content + "\n" + chunk.content).strip()
            merged[-1].end = chunk.end
        else:
            merged.append(chunk)
    return merged


def _dedupe_overlap(prev: str, nxt: str) -> str:
    """合并相邻切片时去掉 1% overlap 造成的重复前缀。"""
    limit = min(len(prev), len(nxt), 300)
    for size in range(limit, 20, -1):
        if prev.endswith(nxt[:size]):
            return nxt[size:]
    return nxt


def chunk_document(text: str, title: str = "", encoder: Any = None,
                   max_chars: int = DEFAULT_MAX_CHARS,
                   min_chars: int = DEFAULT_MIN_CHARS,
                   overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
                   semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD
                   ) -> List[TextChunk]:
    """把长文档切成 7500~8000 字、句子边界完整的分片。"""
    if not (text or "").strip():
        return []
    max_chars = max(500, int(max_chars))
    min_chars = max(100, min(int(min_chars), max_chars))
    sections = _split_sections(text, title)
    chunks: List[TextChunk] = []
    for section in sections:
        if len(section.text) <= max_chars:
            chunks.append(TextChunk(section.title, section.text.strip(),
                                    section.start, section.end))
        else:
            chunks.extend(_split_long_section(
                section, encoder, max_chars, min_chars, overlap_ratio,
                semantic_threshold))

    # 跨章节合并碎片；保留第一个标题，内容去重 overlap。
    merged: List[TextChunk] = []
    for chunk in chunks:
        if not chunk.content:
            continue
        if merged and (len(chunk.content) < min_chars
                       or len(merged[-1].content) < min_chars):
            prev = merged[-1]
            addition = _dedupe_overlap(prev.content, chunk.content)
            if len(prev.content) + len(addition) <= max_chars:
                prev.content = (prev.content + "\n" + addition).strip()
                prev.end = chunk.end
                if prev.title != chunk.title:
                    prev.title = f"{prev.title} / {chunk.title}"
                continue
        merged.append(chunk)
    return merged
