"""File operation tools: read, write, modify."""
from __future__ import annotations

import os
import tempfile

from ..config import AgentConfig
from .base import Tool, ToolRegistry, ToolResult

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
    description = "Read the textual contents of a file inside the workspace."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative path within the workspace (or absolute inside it)."},
        },
        "required": ["path"],
    }

    def run(self, path: str) -> ToolResult:
        try:
            full = self._resolve(path)
        except PermissionError as e:
            return ToolResult(False, error=str(e))
        if not os.path.isfile(full):
            return ToolResult(False, error=f"File does not exist: {path}")
        try:
            size = os.path.getsize(full)
            with open(full, "r", encoding="utf-8") as f:
                # 按**字符**读：多字节 UTF-8（中日韩）下字节数超上限不代表字符
                # 数也超（旧实现误判截断：300KB 中文其实只有约 10 万字符）。
                # 多读一个字符用于判定是否真的需要截断。
                data = f.read(_MAX_READ_CHARS + 1)
            if len(data) > _MAX_READ_CHARS:
                data = data[:_MAX_READ_CHARS]
                data += f"\n… [文件共 {size} 字节，仅显示前 {_MAX_READ_CHARS} 字符]"
            return ToolResult(True, output=data)
        except UnicodeDecodeError:
            import base64

            with open(full, "rb") as f:
                # 二进制同样限制体积：base64 会再放大 4/3，不截断的话一个
                # 大二进制文件就能把内存和模型上下文一起撑爆。
                raw = f.read(_MAX_READ_CHARS // 2)
            return ToolResult(True, output="(binary) " + base64.b64encode(raw).decode())
        except FileNotFoundError:
            # isfile 与 open 之间存在竞态（文件被删/符号链接失效）。
            return ToolResult(False, error=f"File does not exist: {path}")
        except OSError as e:
            return ToolResult(False, error=f"Read failed: {e}")


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
                    "Split the write into several smaller file_write calls, or use "
                    "shell_run with a heredoc / streaming write for large files."
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
                      f"{_MAX_MODIFY_BYTES}); use file_write to rewrite it "
                      f"or shell_run (sed/python) instead",
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
