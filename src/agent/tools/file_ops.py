"""File operation tools: read, write, modify, list, delete, move, copy, search."""
from __future__ import annotations

import fnmatch
import os
import shutil
import tempfile

from ..config import AgentConfig
from .base import Tool, ToolRegistry, ToolResult, _FanoutTool

# file_read 单次返回的最大字符数。模型把一次 file_read 的输出原样放进上下文，
# 没有上限的话一个几 GB 的日志/数据文件就能撑爆内存和模型窗口。与 shell_run
# (200K)、web (8MB) 的上限策略保持一致。
_MAX_READ_CHARS = 200_000

# file_modify 全文读入内存做替换，超过这个大小直接拒绝——既避免把整个大文件
# 拽进内存（读 + 替换副本是 2~3 倍文件体积），也因为对超大文件的「替换第一处」
# 本来就不可靠，模型应改用 file_write 重写或 shell 处理。
_MAX_MODIFY_BYTES = 10 * 1024 * 1024

# file_write 单次写入的最大字符数。模型幻觉出的「完整文件」动辄几十 MB，
# 一次性进内存 + 全量落盘会拖慢 agent 并可能写满磁盘。取读取上限的 5 倍
# （1M 字符）——足够写正常源码/配置，又不会放任病态体积。
_MAX_WRITE_CHARS = 5 * _MAX_READ_CHARS

# file_list 单次列出的最大条目数 / 递归树最大深度：超大目录（几万个文件）
# 一次性塞进上下文会撑爆模型窗口，深度无上限则符号链接循环会导致无限递归。
_MAX_LIST_ENTRIES = 500
_MAX_LIST_DEPTH = 4

# file_search 单次返回的最大命中数 / 单文件参与搜索的最大字节数。前者防止
# 一个高频词（如 `def`）在几百个文件里命中几万行撑爆上下文；后者防止把
# 几百 MB 的日志/数据文件整份读进内存做逐行匹配。
_MAX_SEARCH_RESULTS = 50
_MAX_SEARCH_FILE_BYTES = 1024 * 1024

# 进程 umask 在模块导入时读一次。旧实现每次新建文件都
# ``mask = os.umask(0); os.umask(mask)``——这是进程级全局操作且非线程安全，
# 窗口内其它线程建文件会得到 0666。导入期尚无并发，读一次缓存即可。
try:
    _PROC_UMASK = os.umask(0)
    os.umask(_PROC_UMASK)
except OSError:  # pragma: no cover - 极端平台
    _PROC_UMASK = 0o022


def _atomic_write(full: str, content: str, encoding: str = "utf-8") -> None:
    """Temp + os.replace 原子写。

    临时文件名必须唯一：固定 ``full + ".tmp"`` 在两个线程并发写同一目标时会
    互相覆盖、落盘混合内容；写失败时固定名还会把 .tmp 残留在 workspace 里被
    后续读取当成数据。用 mkstemp 生成唯一名，finally 里清理。

    mkstemp 出于安全考虑以 0600 建文件——直接 replace 会让目标文件的权限从
    常规的 0644 变成「仅属主可读」。这里在 replace 前把权限恢复为「目标文件
    原有权限，不存在则 0644 & ~umask」，避免每次 file_write 都悄悄改变文件
    可见性（例如另一个用户/服务读不了 agent 写的文件）。
    """
    dir_name = os.path.dirname(full) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=dir_name)
    try:
        try:
            # 目标已存在 → 继承其权限；否则用导入期缓存的 umask 推导
            # （不再现场 os.umask(0)：进程级全局操作，非线程安全）。
            try:
                mode = os.stat(full).st_mode & 0o7777
            except OSError:
                mode = 0o666 & ~_PROC_UMASK
            os.fchmod(fd, mode)
            with os.fdopen(fd, "w", encoding=encoding) as f:
                f.write(content)
                # 与 atomic_io.atomic_write_text 对齐：replace 前 fsync。
                # 否则断电/崩溃时 os.replace 后的目标文件可能是空/半写内容
                # （page cache 未落盘），file_write 的持久性承诺是假的。
                f.flush()
                os.fsync(f.fileno())
            fd = -1  # fdopen 已接管， finally 不再重复 close
            os.replace(tmp, full)
        finally:
            if fd >= 0:  # fchmod/fdopen 抛异常时避免 fd 泄漏
                try:
                    os.close(fd)
                except OSError:
                    pass
    finally:
        # os.replace 成功后 tmp 已不存在；失败时清理掉半成品。
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def _decode_text(raw: bytes):
    """按常见编码顺序解码字节串，返回 (text, encoding)；非文本返回 None。

    旧实现只试 utf-8，失败即当二进制 base64——一个 GBK/GB18030 编码的中文
    文件（Windows 上仍很常见）会被返回成一串不可读的 base64 而不是文本。
    这里按「严格→宽松」顺序尝试：utf-8（主流）→ gb18030（gbk 超集，覆盖
    简体/繁体/生僻字）。都失败说明大概率是真二进制，返回 None 走 base64。
    注意顺序不能反：gb18030 对很多字节序列都能「成功」解码但产生乱码，
    必须让 utf-8 先严格尝试。
    """
    for enc in ("utf-8", "gb18030"):
        try:
            return raw.decode(enc), enc
        except (UnicodeDecodeError, LookupError):
            continue
    return None, None


class _FileTool(Tool):
    config: AgentConfig
    registry: ToolRegistry

    def bind(self, config: AgentConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry

    def _resolve(self, path: str) -> str:
        # Use realpath (not abspath) so symlinks pointing outside the workspace
        # are resolved and rejected — abspath leaves symlinks unresolved, which
        # means a symlink inside the workspace to /etc would pass the prefix
        # check and allow escaping the workspace.
        ws = os.path.realpath(self.config.workspace)
        full = path if os.path.isabs(path) else os.path.join(ws, path)
        full = os.path.realpath(full)
        if not (full == ws or full.startswith(ws + os.sep)):
            raise PermissionError(f"Path '{path}' is outside the workspace")
        return full


class FileReadTool(_FileTool):
    name = "file_read"
    description = (
        "Read text from a workspace file. Use tail=N to read the last N chars "
        "(e.g. logs); offset/limit to read a middle slice. Auto-detects UTF-8 "
        "and GBK/GB18030; non-text files come back as base64."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative path within the workspace (or absolute inside it)."},
            "offset": {"type": "integer", "description": "Approx. character offset to start from (default 0)."},
            "limit": {"type": "integer", "description": "Max chars to return (default: up to the built-in cap)."},
            "tail": {"type": "integer", "description": "Return the last N chars of the file (overrides offset/limit)."},
        },
        "required": ["path"],
    }

    def run(self, path: str, offset: int = 0, limit: int = 0, tail: int = 0) -> ToolResult:
        try:
            full = self._resolve(path)
        except PermissionError as e:
            return ToolResult(False, error=str(e))
        if not os.path.isfile(full):
            return ToolResult(False, error=f"File does not exist: {path}")
        try:
            offset = max(0, int(offset or 0))
            limit = max(0, int(limit or 0))
            tail = max(0, int(tail or 0))
        except (TypeError, ValueError):
            return ToolResult(False, error="offset/limit/tail 必须是整数")
        try:
            size = os.path.getsize(full)
            with open(full, "rb") as f:
                if tail:
                    # 读文件末尾：seek 到末尾往前 tail*4 字节（UTF-8 最坏 4B/
                    # 字符），解码后取最后 tail 字符。日志「看最后 N 行」场景
                    # 最常见，且对超大文件也 O(tail) 不读全文。
                    f.seek(max(0, size - tail * 4))
                    raw = f.read()
                else:
                    # 从头（或 offset 附近）读。offset 按最坏 4B/字符估算字节
                    # 窗口：ASCII 会多跳、多字节编码会少跳，属「近似字符偏移」，
                    # 但足够定位到目标区域；真正精确用 tail 或 limit 收窄。
                    f.seek(min(size, offset * 4))
                    want = (limit or _MAX_READ_CHARS)
                    raw = f.read(want * 4 + 1)  # 多读 1 字节判定是否仍需截断
        except OSError as e:
            return ToolResult(False, error=f"Read failed: {e}")
        text, _enc = _decode_text(raw)
        if text is None:
            import base64

            # 二进制同样限制体积：base64 会再放大 4/3，不截断的话一个大
            # 二进制文件就能把内存和模型上下文一起撑爆。
            return ToolResult(
                True,
                output="(binary) " + base64.b64encode(raw[:_MAX_READ_CHARS // 2]).decode(),
            )
        if tail:
            return ToolResult(True, output=text[-tail:])
        if limit:
            text = text[:limit]
            if len(raw) > limit * 4:  # 还有更多内容没读进来
                text += f"\n… [已截断，文件共 {size} 字节]"
            return ToolResult(True, output=text)
        if len(text) > _MAX_READ_CHARS:
            text = text[:_MAX_READ_CHARS]
            text += f"\n… [文件共 {size} 字节，仅显示前 {_MAX_READ_CHARS} 字符]"
        return ToolResult(True, output=text)


class FileWriteTool(_FileTool):
    name = "file_write"
    description = "Write text content to a file inside the workspace, creating parent directories."
    # Confined to the workspace by _resolve(); cannot touch system/other files.
    destructive = False
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }

    def run(self, path: str, content: str) -> ToolResult:
        try:
            full = self._resolve(path)
        except PermissionError as e:
            return ToolResult(False, error=str(e))
        # 写入上限：模型偶尔会吐出几十 MB 的字符串（幻觉出的「完整文件」），
        # 一次性进内存并全量落盘既拖慢 agent 也可能把磁盘写满。与
        # file_read 的 _MAX_READ_CHARS 策略对齐，超限时明确提示改用 shell_run
        # 分块/流式写，而不是让它在超时后变成一个难懂的失败。
        if not isinstance(content, str):
            content = str(content)
        if len(content) > _MAX_WRITE_CHARS:
            return ToolResult(
                False,
                error=(
                    f"content too large ({len(content)} chars > {_MAX_WRITE_CHARS} cap). "
                    "Split the write into several smaller file_tools(op='write') "
                    "calls, or use shell_run with a heredoc / streaming write for "
                    "large files."
                ),
            )
        parent = os.path.dirname(full) or "."
        try:
            # makedirs 也放进 try：磁盘满/只读目录/权限不足都表现为 OSError，
            # 应作为工具失败返回而不是让 registry 兜底成裸异常。
            os.makedirs(parent, exist_ok=True)
            _atomic_write(full, content)
        except OSError as e:
            # makedirs 成功但写入失败时，别把新建的空目录留在磁盘上
            # （下次 list 会看到幽灵目录，模型也会被误导）。
            try:
                if (parent != "." and os.path.isdir(parent)
                        and not os.listdir(parent)):
                    os.rmdir(parent)
            except OSError:
                pass
            return ToolResult(False, error=f"Write failed: {e}")
        return ToolResult(True, output=f"Wrote {len(content)} chars to {path}")


class FileAppendTool(_FileTool):
    """追加写入（L0 红线：memory 下已验证数据只能追加/少量 patch，禁 overwrite）。

    改进 6：L2/L3 的唯一正式写入通道。``file_write`` 会整文件覆盖，模型一旦
    把旧事实库当成新内容重写就会造成「神圣不可删改性」的持久伤害；追加工具
    从工具层消除这类误操作。
    """

    name = "file_append"
    description = (
        "Append text to a workspace file, creating it and parent dirs if absent. "
        "Use this (never file_tools(op='write')/code_run) to add verified facts "
        "to memory/ L2 (global_mem.txt) or L3 (*_sop.md). Does not overwrite "
        "existing data."
    )
    destructive = False
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }

    def run(self, path: str, content: str) -> ToolResult:
        try:
            full = self._resolve(path)
        except PermissionError as e:
            return ToolResult(False, error=str(e))
        if not isinstance(content, str):
            content = str(content)
        if len(content) > _MAX_WRITE_CHARS:
            return ToolResult(
                False,
                error=(f"content too large ({len(content)} chars > {_MAX_WRITE_CHARS} cap). "
                       "Split into several file_tools(op='append') calls."),
            )
        try:
            os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
            with open(full, "a", encoding="utf-8") as f:
                f.write(content)
        except OSError as e:
            return ToolResult(False, error=f"Append failed: {e}")
        return ToolResult(True, output=f"Appended {len(content)} chars to {path}")


class FileModifyTool(_FileTool):
    name = "file_modify"
    description = "Replace the first occurrence of `old` with `new` in a workspace file."
    # Confined to the workspace by _resolve(); cannot touch system/other files.
    destructive = False
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old": {"type": "string"},
            "new": {"type": "string"},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence (default false)."},
        },
        "required": ["path", "old", "new"],
    }

    def run(self, path: str, old: str, new: str, replace_all: bool = False) -> ToolResult:
        # 与 rag_ingest(force) / checkpoint(persist) 一致：字符串 "false" 是
        # 真值，旧实现会把它当 True 全量替换整份文件。
        if isinstance(replace_all, str):
            replace_all = replace_all.strip().lower() in ("1", "true", "yes", "on")
        else:
            replace_all = bool(replace_all)
        old = "" if old is None else str(old)
        new = "" if new is None else str(new)
        # 空 old 会让 str.replace("", new) 在每个字符之间插入内容，瞬间毁掉
        # 整个文件；必须在碰盘前拦下。
        if not old:
            return ToolResult(False, error="'old' 不能为空字符串（会破坏整份文件）")
        try:
            full = self._resolve(path)
        except PermissionError as e:
            return ToolResult(False, error=str(e))
        if not os.path.isfile(full):
            return ToolResult(False, error=f"File does not exist: {path}")
        # 全文读入内存做替换，必须设上限：读 + 替换副本 ≈ 2~3 倍文件体积，
        # 一个几百 MB 的日志就能把内存打满。超限时给出明确指引而不是 OOM。
        size = os.path.getsize(full)
        if size > _MAX_MODIFY_BYTES:
            return ToolResult(
                False,
                error=f"File too large for file_modify ({size} bytes > "
                      f"{_MAX_MODIFY_BYTES}); use file_tools(op='write') to "
                      f"rewrite it or shell_run (sed/python) instead",
            )
        try:
            with open(full, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            # 二进制文件替换文本没有意义，且旧实现在这里抛未捕获的
            # UnicodeDecodeError，被 registry 兜底成难懂的裸异常。
            return ToolResult(False, error=f"File is not UTF-8 text: {path}")
        except OSError as e:
            return ToolResult(False, error=f"Read failed: {e}")
        if old not in text:
            return ToolResult(False, error=f"'old' string not found in {path}")
        if replace_all:
            text = text.replace(old, new)
        else:
            text = text.replace(old, new, 1)
        # Atomic write: write to temp then os.replace.
        try:
            _atomic_write(full, text)
        except OSError as e:
            return ToolResult(False, error=f"Modify failed: {e}")
        return ToolResult(True, output=f"Modified {path}")


# 目录/批量操作默认跳过的路径（版本控制、缓存、依赖目录）——file_list/
# file_search 递归遍历时这些目录会贡献成百上千个无信息量条目，且可能触发
# 大体积扫描，默认过滤掉，模型需要时可用 path 直接指向它们。
_IGNORED_DIRS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache",
    "node_modules", ".venv", "venv", "env", ".idea", ".vscode", "dist",
    "build", ".agent", ".codebuddy",
})


def _fmt_size(n: int) -> str:
    for unit in ("B", "K", "M", "G"):
        if n < 1024 or unit == "G":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}G"  # pragma: no cover


class FileListTool(_FileTool):
    name = "file_list"
    description = (
        "List files and subdirectories under a workspace directory. "
        "Use recursive=true for a tree view. This is the way to discover what "
        "already exists before reading/editing."
    )
    destructive = False
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to list (default: workspace root)."},
            "recursive": {"type": "boolean", "description": "Recurse into subdirectories (default false)."},
        },
        "required": [],
    }

    def run(self, path: str = "", recursive: bool = False) -> ToolResult:
        if isinstance(recursive, str):
            recursive = recursive.strip().lower() in ("1", "true", "yes", "on")
        try:
            full = self._resolve(path or ".")
        except PermissionError as e:
            return ToolResult(False, error=str(e))
        if not os.path.isdir(full):
            return ToolResult(False, error=f"Not a directory: {path or '.'}")
        lines: list = []
        count = 0

        def _emit(rel: str, is_dir: bool, size: int = 0) -> None:
            nonlocal count
            if count >= _MAX_LIST_ENTRIES:
                return
            if is_dir:
                lines.append(f"{rel}/")
            else:
                lines.append(f"{rel}  ({_fmt_size(size)})")
            count += 1

        if not recursive:
            try:
                entries = sorted(os.scandir(full), key=lambda e: e.name.lower())
            except OSError as e:
                return ToolResult(False, error=f"List failed: {e}")
            dirs = [e for e in entries if e.is_dir()]
            files = [e for e in entries if not e.is_dir()]
            for e in dirs:
                _emit(e.name, True)
            for e in files:
                try:
                    _emit(e.name, False, e.stat().st_size)
                except OSError:
                    _emit(e.name, False, 0)
        else:
            # 树形递归：按「目录优先、字母序」输出；深度与条目数双上限防
            # 超大目录 / 符号链接循环撑爆上下文。
            def _walk(dir_path: str, prefix: str, depth: int) -> None:
                if depth > _MAX_LIST_DEPTH or count >= _MAX_LIST_ENTRIES:
                    return
                try:
                    entries = sorted(os.scandir(dir_path), key=lambda e: e.name.lower())
                except OSError:
                    return
                dirs = [e for e in entries if e.is_dir()
                        and e.name not in _IGNORED_DIRS and not e.is_symlink()]
                files = [e for e in entries if not e.is_dir()]
                for i, e in enumerate(dirs + files):
                    if count >= _MAX_LIST_ENTRIES:
                        return
                    is_last = i == len(dirs + files) - 1
                    branch = "└─ " if is_last else "├─ "
                    child_prefix = prefix + ("   " if is_last else "│  ")
                    rel = e.name
                    if e.is_dir():
                        _emit(prefix + branch + rel, True)
                        _walk(e.path, child_prefix, depth + 1)
                    else:
                        try:
                            _emit(prefix + branch + rel, False, e.stat().st_size)
                        except OSError:
                            _emit(prefix + branch + rel, False, 0)

            _walk(full, "", 0)

        if count >= _MAX_LIST_ENTRIES:
            lines.append(f"… [已截断，超过 {_MAX_LIST_ENTRIES} 个条目]")
        header = os.path.relpath(full, os.path.realpath(self.config.workspace)) or "."
        out = f"[{header}] 共 {count} 个条目\n" + "\n".join(lines)
        return ToolResult(True, output=out)


class FileDeleteTool(_FileTool):
    name = "file_delete"
    description = (
        "Delete a file or directory (recursively) inside the workspace. "
        "Irreversible — double-check the path before calling."
    )
    destructive = False  # workspace 内锁定，与 file_write 一致（不影响系统）
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File or directory to delete."},
        },
        "required": ["path"],
    }

    def run(self, path: str) -> ToolResult:
        try:
            full = self._resolve(path)
        except PermissionError as e:
            return ToolResult(False, error=str(e))
        ws = os.path.realpath(self.config.workspace)
        if full == ws:
            return ToolResult(False, error="拒绝删除 workspace 根目录本身")
        if not os.path.lexists(full):
            return ToolResult(False, error=f"Does not exist: {path}")
        try:
            if os.path.isdir(full) and not os.path.islink(full):
                shutil.rmtree(full)
                return ToolResult(True, output=f"Deleted directory {path}")
            os.remove(full)
            return ToolResult(True, output=f"Deleted {path}")
        except OSError as e:
            return ToolResult(False, error=f"Delete failed: {e}")


class FileMoveTool(_FileTool):
    name = "file_move"
    description = (
        "Move or rename a file/directory inside the workspace. "
        "destination may be a target directory or a new path/name."
    )
    destructive = False
    parameters = {
        "type": "object",
        "properties": {
            "source": {"type": "string", "description": "Existing path to move."},
            "destination": {"type": "string", "description": "Target path/name (inside workspace)."},
        },
        "required": ["source", "destination"],
    }

    def run(self, source: str, destination: str) -> ToolResult:
        try:
            src = self._resolve(source)
            dst = self._resolve(destination)
        except PermissionError as e:
            return ToolResult(False, error=str(e))
        ws = os.path.realpath(self.config.workspace)
        if src == ws:
            return ToolResult(False, error="拒绝移动 workspace 根目录本身")
        if not os.path.lexists(src):
            return ToolResult(False, error=f"Source does not exist: {source}")
        try:
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
            shutil.move(src, dst)
            return ToolResult(True, output=f"Moved {source} → {destination}")
        except OSError as e:
            return ToolResult(False, error=f"Move failed: {e}")


class FileCopyTool(_FileTool):
    name = "file_copy"
    description = (
        "Copy a file or directory (recursively) inside the workspace. "
        "destination may be a target directory or a new path/name."
    )
    destructive = False
    parameters = {
        "type": "object",
        "properties": {
            "source": {"type": "string", "description": "Existing path to copy."},
            "destination": {"type": "string", "description": "Target path/name (inside workspace)."},
        },
        "required": ["source", "destination"],
    }

    def run(self, source: str, destination: str) -> ToolResult:
        try:
            src = self._resolve(source)
            dst = self._resolve(destination)
        except PermissionError as e:
            return ToolResult(False, error=str(e))
        if not os.path.lexists(src):
            return ToolResult(False, error=f"Source does not exist: {source}")
        try:
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
            if os.path.isdir(src) and not os.path.islink(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
            return ToolResult(True, output=f"Copied {source} → {destination}")
        except OSError as e:
            return ToolResult(False, error=f"Copy failed: {e}")


class FileSearchTool(_FileTool):
    name = "file_search"
    description = (
        "Search file contents under a workspace directory (grep). Returns "
        "path:line_number: matching line. Use file_pattern to filter by name "
        "(e.g. '*.py')."
    )
    destructive = False
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Literal text to search for."},
            "path": {"type": "string", "description": "Directory or file to search (default: workspace root)."},
            "file_pattern": {"type": "string", "description": "Optional glob filter, e.g. '*.py'."},
            "case_sensitive": {"type": "boolean", "description": "Case-sensitive match (default false)."},
            "max_results": {"type": "integer", "description": "Max matches to return (default 50)."},
        },
        "required": ["query"],
    }

    def run(self, query: str, path: str = "", file_pattern: str = "",
            case_sensitive: bool = False, max_results: int = 50) -> ToolResult:
        if not isinstance(query, str) or not query:
            return ToolResult(False, error="query 不能为空")
        if isinstance(case_sensitive, str):
            case_sensitive = case_sensitive.strip().lower() in ("1", "true", "yes", "on")
        try:
            max_results = max(1, min(int(max_results or _MAX_SEARCH_RESULTS),
                                    _MAX_SEARCH_RESULTS))
        except (TypeError, ValueError):
            max_results = _MAX_SEARCH_RESULTS
        try:
            full = self._resolve(path or ".")
        except PermissionError as e:
            return ToolResult(False, error=str(e))
        if not os.path.exists(full):
            return ToolResult(False, error=f"Does not exist: {path or '.'}")

        needle = query if case_sensitive else query.lower()
        matches: list = []

        def _search_file(file_path: str, rel: str) -> None:
            if len(matches) >= max_results:
                return
            try:
                if os.path.getsize(file_path) > _MAX_SEARCH_FILE_BYTES:
                    return
                with open(file_path, "rb") as f:
                    raw = f.read(_MAX_SEARCH_FILE_BYTES + 1)
                text, _enc = _decode_text(raw)
                if text is None:
                    return
            except OSError:
                return
            for lineno, line in enumerate(text.splitlines(), 1):
                if needle in (line if case_sensitive else line.lower()):
                    matches.append(f"{rel}:{lineno}: {line.strip()[:200]}")
                    if len(matches) >= max_results:
                        return

        if os.path.isfile(full):
            rel = os.path.relpath(full, os.path.realpath(self.config.workspace))
            if not file_pattern or fnmatch.fnmatch(os.path.basename(full), file_pattern):
                _search_file(full, rel)
        else:
            for root, dirs, files in os.walk(full):
                dirs[:] = [d for d in dirs
                           if d not in _IGNORED_DIRS and not os.path.islink(os.path.join(root, d))]
                for name in files:
                    if len(matches) >= max_results:
                        break
                    if file_pattern and not fnmatch.fnmatch(name, file_pattern):
                        continue
                    fp = os.path.join(root, name)
                    rel = os.path.relpath(fp, os.path.realpath(self.config.workspace))
                    _search_file(fp, rel)

        if not matches:
            return ToolResult(True, output=f"未找到匹配「{query}」的内容")
        out = "\n".join(matches)
        if len(matches) >= max_results:
            out += f"\n… [已达 {max_results} 条上限，可缩小查询或指定 path/file_pattern]"
        return ToolResult(True, output=out)


class FileToolsTool(_FanoutTool):
    """文件操作统一门面：op 分发到 read/write/append/modify/patch/list/
    delete/move/copy/search 十个子操作。"""

    # 每个 op 允许的参数（与子工具 run 签名严格对应）。合并后是同一张扁平
    # schema，必须逐 op 隔离，避免 `source`(move/copy) 之类被误传给 read。
    _op_params = {
        "read": {"path", "offset", "limit", "tail"},
        "write": {"path", "content"},
        "append": {"path", "content"},
        "modify": {"path", "old", "new", "replace_all"},
        "patch": {"path", "old", "new", "replace_all"},
        "list": {"path", "recursive"},
        "delete": {"path"},
        "move": {"source", "destination"},
        "copy": {"source", "destination"},
        "search": {"query", "path", "file_pattern", "case_sensitive", "max_results"},
    }

    name = "file_tools"
    description = (
        "Unified file operations (inside the workspace). op one of:\n"
        "- read(path, offset?, limit?, tail?): read text; tail=N = last N chars\n"
        "- write(path, content): create/overwrite a file (atomic, makes dirs)\n"
        "- append(path, content): append text (use this for memory/SOP writes)\n"
        "- modify(path, old, new, replace_all?): replace first (or all) match\n"
        "- patch(path, old, new): alias of modify (exact-match text patch)\n"
        "- list(path?, recursive?): list dir contents or a tree\n"
        "- delete(path): delete a file or directory (irreversible)\n"
        "- move(source, destination): move/rename\n"
        "- copy(source, destination): copy file or directory\n"
        "- search(query, path?, file_pattern?, case_sensitive?, max_results?): grep contents"
    )
    destructive = False
    parameters = {
        "type": "object",
        "properties": {
            "op": {
                "type": "string",
                "enum": ["read", "write", "append", "modify", "patch",
                         "list", "delete", "move", "copy", "search"],
            },
            "path": {"type": "string", "description": "File or directory path (most ops)."},
            "content": {"type": "string", "description": "Text content (write/append)."},
            "old": {"type": "string", "description": "Exact old text to replace (modify/patch)."},
            "new": {"type": "string", "description": "Replacement text (modify/patch)."},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence (modify, default false)."},
            "offset": {"type": "integer", "description": "Approx. char offset to start from (read)."},
            "limit": {"type": "integer", "description": "Max chars to return (read)."},
            "tail": {"type": "integer", "description": "Return last N chars (read)."},
            "recursive": {"type": "boolean", "description": "Recurse into subdirs (list)."},
            "source": {"type": "string", "description": "Existing path to move/copy."},
            "destination": {"type": "string", "description": "Target path (move/copy)."},
            "query": {"type": "string", "description": "Literal text to find (search)."},
            "file_pattern": {"type": "string", "description": "Glob filter like '*.py' (search)."},
            "case_sensitive": {"type": "boolean", "description": "Case-sensitive (search, default false)."},
            "max_results": {"type": "integer", "description": "Max matches (search, default 50)."},
        },
        "required": ["op"],
    }

    def __init__(self):
        self._sub_tools = {
            "read": FileReadTool(),
            "write": FileWriteTool(),
            "append": FileAppendTool(),
            "modify": FileModifyTool(),
            "patch": FileModifyTool(),  # patch = modify 的别名
            "list": FileListTool(),
            "delete": FileDeleteTool(),
            "move": FileMoveTool(),
            "copy": FileCopyTool(),
            "search": FileSearchTool(),
        }
