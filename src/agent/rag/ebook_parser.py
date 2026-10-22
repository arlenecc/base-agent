"""电子书/办公文档处理：pandoc 统一转 Markdown + ebooklib 目录树 + 按章节切块。

分工：
  * ``.epub`` → pandoc 直接转 Markdown；ebooklib 读目录树（目录缩略版）；
    按 spine（阅读顺序，通常即章节）切块，跳过通用切块器
  * ``.mobi/.azw3/.azw`` → 先用本地 calibre 的 ``ebook-convert`` 转 epub，
    再走 epub 流程
  * ``.html/.htm/.docx/.tex/.latex`` → pandoc 转 Markdown（替代 docling）

清洗（与 :mod:`agent.rag.cleaner` 共享）：
  * ``<br>`` 变体 → 空格
  * 不以句末标点（。！?等）结尾的换行符合并为空格（解决断行）
  * 冗余 HTML 实体（&nbsp;/&mdash;/&#8212; 等）→ 空格

外部依赖（均为可选，缺失时相关格式回退原有解析路径）：
  * ``pandoc``（brew install pandoc）
  * ``ebook-convert``（calibre.app，macOS 常见路径自动探测）
  * ``ebooklib``（pip，epub 目录树/章节读取）
"""
from __future__ import annotations

import html as _html
import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 电子书格式（走「calibre→epub→pandoc→按章节切块」完整流程）
EBOOK_EXTENSIONS = {".epub", ".mobi", ".azw3", ".azw"}
# pandoc 直接转 Markdown 的文档格式（替代 docling 的统一解析）
PANDOC_EXTENSIONS = {".html", ".htm", ".docx", ".tex", ".latex", ".ltx"}

_PANDOC_FROM = {
    ".html": "html", ".htm": "html", ".docx": "docx",
    ".tex": "latex", ".latex": "latex", ".ltx": "latex", ".epub": "epub",
}

_pandoc_bin: Optional[str] = None
_pandoc_checked = False
_calibre_bin: Optional[str] = None
_calibre_checked = False

# calibre 常见安装位置（macOS 的 ebook-convert 通常不在 PATH）
_CALIBRE_CANDIDATES = (
    "/Applications/calibre.app/Contents/MacOS/ebook-convert",
    "/Applications/calibre.app/Contents/MacOS/calibre-convert",
)


def is_pandoc_available() -> bool:
    global _pandoc_bin, _pandoc_checked
    if _pandoc_checked:
        return _pandoc_bin is not None
    _pandoc_checked = True
    _pandoc_bin = shutil.which("pandoc") or shutil.which("/opt/homebrew/bin/pandoc")
    if _pandoc_bin is None:
        logger.info("pandoc not found; epub/html/docx/latex will use fallback parsers")
    return _pandoc_bin is not None


def is_calibre_available() -> bool:
    global _calibre_bin, _calibre_checked
    if _calibre_checked:
        return _calibre_bin is not None
    _calibre_checked = True
    _calibre_bin = shutil.which("ebook-convert")
    if _calibre_bin is None:
        for cand in _CALIBRE_CANDIDATES:
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                _calibre_bin = cand
                break
    if _calibre_bin is None:
        logger.info("calibre ebook-convert not found; mobi/azw3/azw will use fallback parsers")
    return _calibre_bin is not None


def _run(cmd: List[str], stdin_bytes: Optional[bytes] = None,
         timeout: int = 300) -> Optional[str]:
    """Run an external converter, returning stdout text or None on failure."""
    try:
        proc = subprocess.run(
            cmd, input=stdin_bytes, capture_output=True, timeout=timeout,
        )
        if proc.returncode != 0:
            logger.warning("%s failed (rc=%d): %s",
                           cmd[0], proc.returncode, proc.stderr[-300:] if proc.stderr else "")
            return None
        return proc.stdout.decode("utf-8", errors="replace")
    except FileNotFoundError:
        logger.warning("%s not found", cmd[0])
        return None
    except subprocess.TimeoutExpired:
        logger.warning("%s timed out after %ss", cmd[0], timeout)
        return None
    except Exception as e:
        logger.warning("%s error: %s", cmd[0], e)
        return None


def pandoc_to_markdown(path: str, from_format: Optional[str] = None) -> Optional[str]:
    """pandoc 把文档转成 Markdown。

    ``--wrap=none`` 关闭 pandoc 默认的 72 列折行（否则转换器自己制造硬回车）；
    ``--strip-comments`` 去注释。输出 pandoc-flavored markdown，保留标题/列表/
    表格结构，供后续清洗与按章节切块使用。
    """
    if not is_pandoc_available():
        return None
    fmt = from_format or _PANDOC_FROM.get(os.path.splitext(path)[1].lower())
    if fmt is None:
        return None
    out = _run([_pandoc_bin, "-f", fmt, "-t", "gfm", "--wrap=none",
                "--strip-comments", path])
    if out is None or not out.strip():
        return None
    return out


def convert_to_epub(path: str) -> Optional[str]:
    """用 calibre 把 mobi/azw3/azw 转成 epub，返回临时 epub 路径（调用方负责清理）。"""
    if not is_calibre_available():
        return None
    tmpdir = tempfile.mkdtemp(prefix="rag_ebook_")
    out_path = os.path.join(tmpdir, "book.epub")
    out = _run([_calibre_bin, path, out_path], timeout=600)
    if out is None or not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
        logger.warning("ebook-convert failed for %s", os.path.basename(path))
        _safe_rmtree(tmpdir)
        return None
    return out_path


def _safe_rmtree(path: str) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 清洗（复用 cleaner 的共享逻辑）
# ---------------------------------------------------------------------------

def clean_ebook_markdown(md: str) -> str:
    """电子书 Markdown 清洗：实体→空格、<br>→空格、硬回车合并。

    pandoc ``--wrap=none`` 已避免折行，这里再兜底处理源文档自带的断行与实体
    （用户规则：非句末标点结尾的换行符与 <br> 替换为空格；&nbsp;/&mdash; 等
    冗余实体替换为空格）。
    """
    if not md:
        return ""
    from .cleaner import (html_tables_to_markdown, join_hard_wrapped_lines,
                          strip_html_entities)

    md = strip_html_entities(md)
    md = _html.unescape(md)  # 残留的功能性实体（&amp; 等）还原为真实字符
    # pandoc gfm 会把无法用 Markdown 表达的元素原样保留为 raw HTML
    # （ruby/rt/sup/sub/u/small/table/colgroup/svg/math… 白名单永远列不全
    # ——这正是入库文本带大量 HTML 标签的根因）——统一剥除所有 HTML 标签
    # 只留内容。代码块/行内代码与 autolink 先占位保护避免误伤。
    stash: List[str] = []
    _ph = chr(0) + "CODE"                     # 占位前缀（NUL 不会出现在正文）
    _ph_re = re.compile(re.escape(chr(0)) + r"CODE(\d+)" + re.escape(chr(0)))

    def _stash_code(m: "re.Match[str]") -> str:
        stash.append(m.group(0))
        return _ph + str(len(stash) - 1) + chr(0)

    _fence = "`" * 3
    md = re.sub(_fence + r"[\s\S]*?" + _fence + "|" + "`" + r"[^`\n]*" + "`",
                _stash_code, md)
    # pandoc 对 pre/code 常输出 **4 空格缩进代码块**（非围栏）——同样保护，
    # 否则代码里的 <div>/<T> 等会被通用剥标签吃掉。
    md = re.sub(r"(?m)^(?: {4}|\t).*(?:\n(?: {4}|\t).*)*", _stash_code, md)
    # 转义括号 \[ \] \( \) \> 还原——必须在 autolink 保护和剥标签**之前**：
    # pandoc 会把字面尖括号转义成 \<…\>，不先还原的话 autolink 保护不到，
    # 会被下面的通用剥标签当成标签剥掉。
    md = re.sub(r'\\([\[\]()#*_<>{}])', r'\1', md)
    # gfm autolink <https://…> / <mailto:…> 还原成裸链接（否则被当标签剥掉）
    md = re.sub(r"<(https?://[^>\s]+|mailto:[^>\s]+)>", r"\1", md)
    # HTML 表格先转 Markdown 管道表（代码块已被上方 stash 保护）——否则下面
    # 的通用剥标签会把 <td> 单元格文本粘连成一行，表格结构信息全丢。
    md = html_tables_to_markdown(md)
    md = re.sub(r"<!--[\s\S]*?-->", " ", md)      # 注释兜底（pandoc 已剥一轮）
    md = re.sub(r"</?[a-zA-Z][^>]*>", " ", md)    # 所有 HTML/XML 标签（属性可跨行）
    md = re.sub(r"<[!?][^>]*>", " ", md)          # DOCTYPE / 处理指令残留
    md = _ph_re.sub(lambda m: stash[int(m.group(1))], md)
    # pandoc gfm 把源文档的 <br> 转成行尾反斜杠硬换行转义——同样按断行处理
    md = re.sub(r'\\\n', ' ', md)
    md = re.sub(r'\\$', ' ', md, flags=re.MULTILINE)
    # 图片对 RAG 文本无意义（epub 内嵌图/数据 URI），整段去掉——必须先于
    # 下面的链接解包，否则 ![](x.png) 会被解包成孤立的 "!" 残留。
    md = re.sub(r'!\[[^\]]*\]\([^)]*\)', ' ', md)
    # 内联链接只保留链接文字（epub 内部 filepos/#anchor 跳转对检索无价值）
    md = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', md)
    md = join_hard_wrapped_lines(md, preserve_markdown=True)
    md = re.sub(r'[ \t]+', ' ', md)           # 合并产生的连续空格
    md = re.sub(r'\n{3,}', '\n\n', md)        # 收敛连续空行
    md = re.sub(r'(^|\n)\s*((?:!\[[^\]]*\]\([^)]*\)\s*)+)\s*(?=\n|$)',
                r'\1', md)                    # 连续裸图片占位行折叠
    return md.strip()


# ---------------------------------------------------------------------------
# epub 目录树与章节
# ---------------------------------------------------------------------------

def _flatten_toc(items, level: int = 1,
                 out: Optional[List[Tuple[int, str]]] = None) -> List[Tuple[int, str]]:
    """把 ebooklib 的嵌套目录树拍平成 [(level, title), ...]。"""
    if out is None:
        out = []
    from ebooklib import epub as _epub_mod
    for it in items:
        if isinstance(it, tuple):
            # (Section, [children])：Section 是目录节点
            sec, children = it
            title = getattr(sec, "title", "") or ""
            if title:
                out.append((level, title))
            _flatten_toc(children, level + (1 if title else 0), out)
        elif isinstance(it, _epub_mod.Link):
            out.append((level, it.title or ""))
        elif isinstance(it, _epub_mod.Section):
            if it.title:
                out.append((level, it.title))
    return out


def build_toc_digest(book_title: str, flat_toc: List[Tuple[int, str]]) -> str:
    """目录缩略版：书名 + 缩进目录树（入 RAG 作为全书结构锚点）。"""
    lines = [f"# {book_title or '目录'}", "", "# 目录", ""]
    for level, title in flat_toc:
        if not title.strip():
            continue
        lines.append(f"{'  ' * (level - 1)}- {title.strip()}")
    return "\n".join(lines).strip()


def _read_epub(epub_path: str):
    from ebooklib import epub
    return epub.read_epub(epub_path, options={"ignore_ncx": False})


def _manifest_href_map(epub_path: str) -> Dict[str, str]:
    """解析 OPF manifest 得到 {item_id: href}。

    ebooklib 对部分 epub（尤其 calibre 产出）读出的 EpubHtml 会缺失 href
    属性，导致 spine→目录无法关联——以 OPF 原始数据为准自建映射。
    """
    import zipfile
    import xml.etree.ElementTree as ET
    from urllib.parse import unquote

    out: Dict[str, str] = {}
    try:
        with zipfile.ZipFile(epub_path) as z:
            container = ET.fromstring(z.read("META-INF/container.xml"))
            rootfile = container.find(".//{*}rootfile")
            if rootfile is None:
                return out
            opf = ET.fromstring(z.read(rootfile.get("full-path")))
            for it in opf.findall(".//{*}manifest/{*}item"):
                iid, href = it.get("id"), it.get("href")
                if iid and href:
                    out[iid] = unquote(href)
    except Exception as e:
        logger.debug("manifest parse failed for %s: %s", epub_path, e)
    return out


def _spine_items(book, href_map: Dict[str, str]) -> List[Tuple[str, str]]:
    """按 spine（阅读顺序）取 (item_id, href) 列表，仅文档项。

    href 优先取 OPF manifest 映射（item.href 属性可能缺失）。
    """
    pairs: List[Tuple[str, str]] = []
    for idref, _linear in getattr(book, "spine", []):
        item = book.get_item_with_id(idref)
        if item is None or getattr(item, "media_type", "") != "application/xhtml+xml":
            continue
        href = href_map.get(idref) or getattr(item, "href", "") or ""
        pairs.append((idref, href))
    return pairs


def _item_content(book, item_id: str) -> Optional[bytes]:
    item = book.get_item_with_id(item_id)
    if item is None:
        return None
    return getattr(item, "content", None)


# ---------------------------------------------------------------------------
# 扫描版 EPUB：整页图片 + 极少文本 → 逐页图片 OCR（ocrmac 优先 → RapidOCR）
# ---------------------------------------------------------------------------

# 扫描版判定阈值。判据按「文档分布」而不是「平均文本量」：扫描版 EPUB
# 常带目录页/版权页/书商标签等文本大户（合计可达上万字），平均值会被拉高
# 造成漏判；但含图的正文页自身几乎无文本——这个占比是稳健信号。
_SCANNED_MIN_DOCS = 3           # spine 文档过少样本不可靠，不判扫描版
_SCANNED_MIN_IMG_RATIO = 0.5    # 含图片的文档占比下限
_SCANNED_TEXTLESS = 100         # 文档自身文本 < 100 字 = 「无实质文本」
_SCANNED_TEXTLESS_RATIO = 0.6   # 含图文档中「无实质文本」的占比下限


def _epub_scanned_stats(book, href_map: Dict[str, str]) -> Tuple[int, int, int, int]:
    """遍历 spine：返回 (文档数, 含图文档数, 含图且无实质文本的文档数, 总文本字符)。

    纯文本大部头（遍历 20 个文档仍一张图都没有）提前退出。
    """
    from bs4 import BeautifulSoup
    docs = img_docs = textless_img_docs = text_chars = 0
    for item_id, _href in _spine_items(book, href_map):
        content = _item_content(book, item_id)
        if not content:
            continue
        docs += 1
        try:
            soup = BeautifulSoup(content.decode("utf-8", errors="replace"),
                                 "html.parser")
        except Exception:
            continue
        text = soup.get_text(strip=True)
        text_chars += len(text)
        if soup.find_all("img"):
            img_docs += 1
            if len(text) < _SCANNED_TEXTLESS:
                textless_img_docs += 1
        if docs >= 20 and img_docs == 0:
            break
    return docs, img_docs, textless_img_docs, text_chars


def _is_scanned_epub(book, href_map: Dict[str, str]) -> bool:
    docs, img_docs, textless_img_docs, _chars = _epub_scanned_stats(book, href_map)
    if docs < _SCANNED_MIN_DOCS:
        return False
    if img_docs < docs * _SCANNED_MIN_IMG_RATIO:
        return False
    return textless_img_docs / max(img_docs, 1) >= _SCANNED_TEXTLESS_RATIO


def extract_scanned_epub(path: str, book=None) -> Optional[str]:
    """扫描版 EPUB → 按 spine 顺序逐页图片 → 统一归一化 → OCR。

    OCR 走 :func:`parsers._ocr_image_cached` 的统一入口（Apple Vision 优先、
    RapidOCR 回退），图片先经 :func:`ocr_engine.preprocess_for_ocr` 归一化
    （长边 ≤1500、Lanczos、灰度、二值化）。
    非扫描版返回 ``None``（调用方走常规提取路径）。
    """
    import posixpath
    from urllib.parse import unquote
    from bs4 import BeautifulSoup
    from .ocr_engine import decode_image, preprocess_for_ocr
    from .parsers import _get_ocr_engine, _ocr_image_cached

    href_map = _manifest_href_map(path)
    if book is None:
        book = _read_epub(path)
    if not _is_scanned_epub(book, href_map):
        return None
    logger.info("EPUB '%s' 判定为扫描版（整页图片 + 极少文本），走逐页 OCR",
                os.path.basename(path))

    # engine 可能为 None（RapidOCR 未安装）：ocrmac 可用时仍能完整识别，
    # 只有 ocrmac 也不可用时 _ocr_image_cached 内部才会失败并逐页返回 ""。
    engine = _get_ocr_engine()
    # 图片 item 收集：ebooklib 的 get_items_of_type(ITEM_IMAGE) 依赖 item 的
    # type 属性，部分制作工具产出的图片 item 类型不全（实测 4 张图只回收 1
    # 张）。以 media_type 为主 + type 兜底全量扫一遍并去重。
    from ebooklib import epub as _epub_mod
    item_image = getattr(_epub_mod, "ITEM_IMAGE", 4)
    image_candidates = []
    seen = set()
    for it in book.get_items():
        mt = (getattr(it, "media_type", "") or "")
        if mt.startswith("image/") or getattr(it, "type", None) == item_image:
            key = id(it)
            if key not in seen:
                seen.add(key)
                image_candidates.append(it)
    # href 的来源必须是 OPF manifest（_manifest_href_map 的 id→href），不能信
    # item.href——实测 ebooklib read_epub 后 item.href 恒为 None。
    img_items = {}
    for it in image_candidates:
        iid = getattr(it, "id", None)
        href_key = unquote(href_map.get(iid, "")
                           or getattr(it, "href", "") or "")
        if not href_key:
            continue
        img_items[href_key] = it
        img_items[os.path.basename(href_key)] = it

    fname = os.path.basename(path)
    parts: List[str] = []
    page_no = 0
    for item_id, href in _spine_items(book, href_map):
        content = _item_content(book, item_id)
        if not content:
            continue
        try:
            soup = BeautifulSoup(content.decode("utf-8", errors="replace"),
                                 "html.parser")
        except Exception:
            continue
        base_dir = posixpath.dirname(href)
        for img in soup.find_all("img"):
            src = img.get("src") or img.get("xlink:href") or ""
            if not src:
                continue
            rel = unquote(src.split("#")[0])
            name = posixpath.normpath(posixpath.join(base_dir, rel))
            it = img_items.get(name) or img_items.get(os.path.basename(rel))
            if it is None:
                continue
            arr = decode_image(it.get_content())
            if arr is None:
                continue
            page_no += 1
            norm = preprocess_for_ocr(arr)
            txt = _ocr_image_cached(engine, norm, page_no, fname)
            if txt and txt.strip():
                parts.append(txt)
            if page_no % 25 == 0:
                logger.info("扫描版 EPUB OCR 进度: %s 已识别 %d 页", fname, page_no)
    if not parts:
        return None
    logger.info("扫描版 EPUB OCR 完成: %s 共 %d 页图片", fname, page_no)
    return "\n\n".join(parts)


def _href_matches(href: str, toc_hrefs: Dict[str, str]) -> str:
    """spine href → 目录标题（忽略锚点片段；目录 href 可能带 ../ 前缀）。"""
    base = os.path.basename(href or "")
    base_no_frag = base.split("#")[0]
    return toc_hrefs.get(base_no_frag, "") or toc_hrefs.get(base, "")


# 章节切分哨兵：pandoc 转换前注入 HTML、转换后按它 split 出章节边界。
# 纯 ASCII 且带包围符，与正文撞车概率可忽略。
_CHAP_SENTINEL = "XCHAPBREAKX"


def _collect_toc_anchors(toc_items) -> Dict[str, List[Tuple[str, str]]]:
    """TOC 树 → {basename: [(fragment, title), ...]}（保持目录顺序）。

    epub 目录直接给出准确的章节边界（Link 的 href = 文件 + 锚点片段），
    优先用它做章节切分；仅当目录缺失/无有效锚点时才回退 spine 整文件切块。
    """
    from urllib.parse import unquote
    from ebooklib import epub as _epub_mod

    out: Dict[str, List[Tuple[str, str]]] = {}

    def walk(items) -> None:
        for it in items or []:
            if isinstance(it, tuple):
                walk(it[1])
            elif isinstance(it, _epub_mod.Link):
                base, _, frag = (it.href or "").partition("#")
                base = os.path.basename(unquote(base))
                title = (it.title or "").strip()
                if base and title:
                    out.setdefault(base, []).append((frag, title))

    walk(toc_items)
    return out


def _find_anchor_offset(html_str: str, frag: str) -> int:
    """在 item HTML 里定位锚点元素（id= 或 name=）的字节偏移，找不到返回 -1。"""
    m = (re.search(r'id=["\']%s["\']' % re.escape(frag), html_str)
         or re.search(r'name=["\']%s["\']' % re.escape(frag), html_str))
    if not m:
        return -1
    tag_start = html_str.rfind("<", 0, m.start())
    return tag_start if tag_start >= 0 else -1


def _convert_item_chapters(content: bytes, anchors: List[Tuple[str, str]]) -> Optional[List[str]]:
    """按 TOC 锚点把单个 spine item 的 HTML 精确切分成多个章节 Markdown。

    做法：在每个章节锚点元素前注入哨兵段落 → 整 item 一次 pandoc → 按哨兵
    split 出章节。返回 None 表示无法按目录切分（锚点全部定位失败），调用方
    回退整 item 一章。
    """
    html_str = content.decode("utf-8", errors="replace")
    offsets: List[Tuple[int, str]] = []
    for frag, title in anchors:
        if not title:
            continue
        off = _find_anchor_offset(html_str, frag) if frag else 0
        if off < 0:
            continue
        offsets.append((off, title))
    if not offsets:
        return None
    offsets.sort()
    # 同偏移去重（保第一个），空标题的边界丢弃
    deduped: List[Tuple[int, str]] = []
    for off, title in offsets:
        if not title or (deduped and off == deduped[-1][0]):
            continue
        deduped.append((off, title))
    if not deduped:
        return None

    merged_parts: List[str] = []
    for i, (off, title) in enumerate(deduped):
        end = deduped[i + 1][0] if i + 1 < len(deduped) else len(html_str)
        merged_parts.append(f"<p>{_CHAP_SENTINEL}{title}</p>" + html_str[off:end])
    md = _run([_pandoc_bin, "-f", "html", "-t", "gfm", "--wrap=none",
               "--strip-comments"],
              stdin_bytes="\n".join(merged_parts).encode("utf-8"), timeout=120)
    if not md:
        return None

    pieces = md.split(_CHAP_SENTINEL)
    if len(pieces) < len(deduped) + 1:
        # 哨兵在转换中丢失（理论上不会），退回整 item 一章
        return None
    chapters: List[str] = []
    for i, (_off, title) in enumerate(deduped):
        body = pieces[i + 1].strip()
        # 哨兵后紧跟的目录标题文本与 `## 标题` 重复，去掉首行
        if body.startswith(title):
            body = body[len(title):].strip()
        body = clean_ebook_markdown(body)
        if not body:
            continue
        chapters.append(f"## {title}\n\n{body}")
    return chapters or None


def extract_ebook(path: str) -> Optional[Dict[str, Any]]:
    """电子书完整处理：转换→pandoc→清洗→目录树→章节切块素材。

    Returns dict::

        {
          "full_markdown": str,   # 全书清洗后 Markdown（md 缓存/摘要用）
          "toc_digest": str,      # ebooklib 目录缩略版
          "toc_chapters": [...],  # [{"level": int, "title": str}] 供 digest 表
          "chapters": [str],      # 每章清洗后 Markdown（含章节标题头），按阅读序
        }
    失败返回 None（调用方回退通用解析路径）。
    """
    ext = os.path.splitext(path)[1].lower()
    tmp_epub: Optional[str] = None
    epub_path = path
    try:
        if ext != ".epub":
            tmp_epub = convert_to_epub(path)
            if tmp_epub is None:
                return None
            epub_path = tmp_epub

        if not is_pandoc_available():
            return None
        try:
            book = _read_epub(epub_path)
        except Exception as e:
            logger.warning("epub read failed for %s: %s", os.path.basename(path), e)
            return None

        # 扫描版 EPUB：书页几乎无文本、由整页图片构成——pandoc/ebooklib 只会
        # 产出空壳。逐页图片走统一 OCR（归一化 → ocrmac 优先 → RapidOCR 回退），
        # 整本作为一个"章节"返回，交给通用切块器二次切分。
        if ext == ".epub" or tmp_epub:
            try:
                scanned = extract_scanned_epub(epub_path, book)
            except Exception as e:
                logger.warning("扫描版 EPUB OCR 异常，回退常规路径: %s: %s",
                               os.path.basename(path), e)
                scanned = None
            if scanned and scanned.strip():
                return {
                    "full_markdown": scanned,
                    "toc_digest": "",
                    "toc_chapters": [],
                    "chapters": [scanned],
                }

        book_title = ""
        try:
            book_title = book.get_metadata("DC", "title")[0][0]
        except Exception:
            pass

        flat_toc = _flatten_toc(getattr(book, "toc", []) or [])
        # href→标题映射（目录 Link 的 href 可能带路径前缀/锚点，统一比 basename）
        toc_hrefs: Dict[str, str] = {}
        from ebooklib import epub as _epub_mod
        def _collect_links(items):
            for it in items:
                if isinstance(it, tuple):
                    _collect_links(it[1])
                elif isinstance(it, _epub_mod.Link):
                    base = os.path.basename(it.href or "").split("#")[0]
                    if base and it.title and base not in toc_hrefs:
                        toc_hrefs[base] = it.title
        _collect_links(getattr(book, "toc", []) or [])

        toc_digest = build_toc_digest(book_title, flat_toc)
        toc_chapters = [{"level": lv, "title": t} for lv, t in flat_toc if t.strip()]

        # 准确章节优先：epub 目录 Link 的锚点直接给出章节边界；仅当某 item
        # 无法按目录定位任何锚点时，才回退整 item 一章（启发式）。
        toc_anchors = _collect_toc_anchors(getattr(book, "toc", []) or [])
        href_map = _manifest_href_map(epub_path)

        chapters: List[str] = []
        full_parts: List[str] = [toc_digest, ""]
        for idx, (item_id, href) in enumerate(_spine_items(book, href_map)):
            content = _item_content(book, item_id)
            if not content:
                continue
            base = os.path.basename(href or "").split("#")[0]
            chaps: Optional[List[str]] = None
            if toc_anchors.get(base):
                try:
                    chaps = _convert_item_chapters(content, toc_anchors[base])
                except Exception as e:
                    logger.debug("TOC anchor chapter split failed for %s: %s", base, e)
            if chaps is None:
                # 回退：整 spine item 一章（原实现）
                md = _run([_pandoc_bin, "-f", "html", "-t", "gfm", "--wrap=none",
                           "--strip-comments"],
                          stdin_bytes=content, timeout=120)
                if not md or not md.strip():
                    continue
                md = clean_ebook_markdown(md)
                if not md:
                    continue
                title = _href_matches(href, toc_hrefs) or f"第 {idx + 1} 节"
                chaps = [f"## {title}\n\n{md}"]
            chapters.extend(chaps)
            full_parts.extend(chaps)
            full_parts.append("")

        if not chapters:
            logger.warning("epub produced no chapters: %s", os.path.basename(path))
            return None

        full_md = "\n".join(full_parts).strip()
        return {
            "full_markdown": full_md,
            "toc_digest": toc_digest,
            "toc_chapters": toc_chapters,
            "chapters": chapters,
        }
    finally:
        if tmp_epub:
            _safe_rmtree(os.path.dirname(tmp_epub))
