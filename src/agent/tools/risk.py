"""code_run / shell_run 的风险分级判定。

旧策略：两个工具 destructive=True，每次调用都弹人工确认——而 code_run 是
能力核心（几乎每轮都用），shell_run 是它对系统命令的延伸，高频弹窗把
工作流切成碎片。

新策略（按「破坏性 / 影响系统与安全」分级）：
  - 默认直接执行：只读操作、工作区内的读写、工作区内子命令等；
  - 弹确认：sudo / 系统与电源控制 / 进程杀伤 / 磁盘操作 / 管道下载执行 /
    git push --force / 写工作区之外的路径。

判定是静态启发式（读参数文本，不执行）：宁可漏报不可误杀——漏掉的破坏性
操作有 code_run 的写护栏与工具超时兜底，而误报的确认会让用户养成无脑点
「Yes」的习惯，确认机制就失去意义。
"""
from __future__ import annotations

import os
import re

# 明确安全的特殊路径（常见重定向惯用法）
_SAFE_ABS_PATHS = frozenset({
    "/dev/null", "/dev/stdout", "/dev/stderr", "/dev/stdin",
    "/dev/zero", "/dev/urandom", "/dev/tty",
})

# ~ 开头的 HOME 展开、/ 开头的绝对路径、../ 逃逸。先剥掉 URL（http://x/y
# 会被当成 //x/y 的绝对路径误报）。
_PATH_TOKEN_RE = re.compile(r"(?:[~/]|\.\./)[^\s'\"`|;&<>()\[\]{}]+")

# Python 侧的「破坏性调用」记号：与危险路径同现才判风险。
# 2026-10 修补：旧表只有 open 之外的 remove/rmtree/chmod/subprocess，漏掉了
# os.symlink/os.rename/os.link/os.replace/shutil.move 这些同样能改文件系统
# 拓扑的调用，导致 `os.rename("/etc/hosts", ...)` 这类改写系统文件的写法
# 既零确认也不被任何护栏覆盖。
# 2026-10-21 再修：pathlib / io / os 的文件 API 全部不在表内——
# `Path("/etc/hosts").write_text(...)`、`io.open("/etc/hosts","w")`、
# `os.open("/etc/hosts", os.O_WRONLY)` 三条路同时绕过确认门与写护栏
# （code_run 的 open 护栏只替换了 builtins.open）。补齐这些记号。
_PY_CALL_RE = re.compile(
    r"\b(os\.system|os\.popen|os\.remove|os\.unlink|os\.rmdir|os\.removedirs"
    r"|os\.chmod|os\.chown|os\.symlink|os\.link|os\.rename|os\.replace"
    r"|os\.truncate|os\.makedirs|os\.mkdir|os\.open|os\.fdopen"
    r"|io\.open|pathlib\.Path|Path\.open|write_text|write_bytes|touch"
    r"|shutil\.rmtree|shutil\.chown|shutil\.move|shutil\.copy|shutil\.copy2"
    r"|shutil\.copytree|subprocess)\b"
)

# shell 侧的破坏性动词（读文件不算：cat/ls/grep/head/tail 都不在表里）。
# 同时收录 Python 侧调用名：`python3 -c "shutil.rmtree('/Users/x')"` 这类写法
# 不走任何 shell 动词，只靠解释器名是拦不住的（与危险路径同现才判风险，
# 因此 `python3 script.py` 这种正常开发用法不会误报）。
# 2026-10-21：补 pathlib / io / os 文件 API——`python3 -c
# "from pathlib import Path; Path('/etc/hosts').write_text('x')"` 旧表完全不命中。
_SHELL_VERB_RE = re.compile(
    r"\b(rm|rmdir|mv|cp|chmod|chown|ln|truncate|shred|tee|install|mkfile"
    r"|hdiutil|defaults"
    # 磁盘级操作
    r"|dd|mkfs|fdisk|parted|wipefs|badblocks|mkswap|swapon|swapoff"
    # Python 侧等价的破坏性调用
    r"|rmtree|removedirs|unlink|os\.system|os\.popen|os\.remove|os\.unlink"
    r"|os\.rmdir|os\.removedirs|os\.chmod|os\.chown|os\.rename|os\.replace"
    r"|os\.symlink|os\.link|os\.truncate|os\.open|os\.fdopen"
    r"|io\.open|pathlib\.Path|write_text|write_bytes"
    r"|shutil\.rmtree|shutil\.move"
    r"|shutil\.copy|shutil\.copy2|shutil\.copytree"
    r"|subprocess\.(run|call|check_call|check_output|Popen))\b"
)
# 重定向到工作区之外的路径（> /etc/hosts、> ~/.bashrc、>> ../outside）。
# 旧正则只匹配 ``>/``：``~`` 展开与 ``../`` 逃逸两种工作区外写入零确认。
# 注意 ``./`` 是工作区内相对路径，不能算风险（旧行为就不算）。
_SHELL_REDIRECT_RE = re.compile(r">>?\s*(?:~|\.\./|/)")
# sudo / 系统-电源-进程-磁盘控制
_SHELL_CTRL_RE = re.compile(
    r"\b(sudo|doas|su\s+-|shutdown|reboot|halt|poweroff|pmset|csrutil"
    r"|launchctl|nvram|diskutil|mkfs|kill|killall|pkill|spctl"
    r"|systemctl|service|ufw|iptables|pfctl|crontab)\b"
)
# 远程脚本直接执行（curl … | sh）。2026-10 修补：旧目标只有 (ba|z|da|fi)?sh，
# `curl … | python3` / `| node` / `| perl` 完全漏网。
_PIPE_EXEC_RE = re.compile(
    r"(curl|wget)\b[^|\n]*\|\s*(sudo\s+)?"
    r"((ba|z|da|fi)?sh|python[\d.]*|node|perl|ruby|php)\b"
)
# 改写远端历史
_GIT_FORCE_RE = re.compile(r"\bgit\s+push\b[^\n]*(--force\b|--force-with-lease\b|-f\b)")


def mentions_outside_path(text: str, workspace: str) -> bool:
    """text 中出现**工作区之外**的路径字面量 → True。

    相对路径默认落在工作区内（两个工具都以 workspace 为 cwd/写护栏基线），
    视为安全；绝对路径与 ~ 展开做 realpath 前缀比对。
    """
    if not text or not workspace:
        return False
    cleaned = re.sub(r"\b\w+://\S+", " ", text)          # 剥 URL
    try:
        ws = os.path.realpath(workspace)
    except OSError:
        ws = os.path.realpath(workspace or "/")
    for tok in _PATH_TOKEN_RE.findall(cleaned):
        # 尾部标点全部剥掉：旧字符集漏了 ``!``/``?``/``]``/``}``，``~/foo!``
        # 会被当成不存在的绝对路径字面量而漏判。
        tok = tok.rstrip(".,;:)\"'!?]}>")
        expanded = os.path.expanduser(tok)
        if tok.startswith("../") or tok == "..":
            # ../ 逃逸：shell 子进程不受写护栏约束，必须按 workspace 解析
            resolved = os.path.normpath(os.path.join(ws, tok))
            if resolved != ws and not resolved.startswith(ws + os.sep):
                return True
            continue
        if not os.path.isabs(expanded):
            continue
        norm = os.path.normpath(expanded).replace("\\", "/")
        if norm in _SAFE_ABS_PATHS:
            continue
        real = os.path.realpath(norm)
        if real != ws and not real.startswith(ws + os.sep):
            return True
    return False


def is_risky_command(command: str, workspace: str) -> bool:
    """shell_run 的风险判定。"""
    cmd = command or ""
    low = cmd.lower()
    if _SHELL_CTRL_RE.search(low):
        return True
    if _PIPE_EXEC_RE.search(low):
        return True
    if _GIT_FORCE_RE.search(low):
        return True
    # 破坏性动词（或向绝对路径重定向）+ 工作区之外的路径 → 风险
    if (_SHELL_VERB_RE.search(low) or _SHELL_REDIRECT_RE.search(low)) \
            and mentions_outside_path(cmd, workspace):
        return True
    return False


def is_risky_code(code: str, workspace: str) -> bool:
    """code_run 的风险判定。"""
    src = code or ""
    if re.search(r"\bsudo\b", src):
        return True
    if _PIPE_EXEC_RE.search(src):
        return True
    if _GIT_FORCE_RE.search(src):
        return True
    # 破坏性调用 + 工作区之外的路径字面量同现才判风险：
    # shutil.rmtree("build") 在工作区内是安全的；
    # subprocess.run(["pip", "install", "x"]) 无外部路径也安全。
    if _PY_CALL_RE.search(src) and mentions_outside_path(src, workspace):
        return True
    return False
