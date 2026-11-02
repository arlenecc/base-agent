# -*- coding: utf-8 -*-
"""默认本地模型自动下载（HF 镜像站）。

首次启动时，若「默认配置模型」（本地 embedding / 本地 reranker，provider=local）
在模型目录缺失，则从 HF 镜像站（默认 ``https://hf-mirror.com``）按
``<namespace>/<模型名>``（默认 ``arlenecc/<模型名>``）拉取**必需文件**：

  1. REST API 树接口（``GET /api/models/{repo}/tree/main?recursive=true&expand=true``）
     获取仓库文件清单（路径 + 字节数，LFS 文件取真实大小）；
  2. 白名单过滤，只拉运行所需的文件（onnx / safetensors / gguf / 配置与分词器），
     排除 PyTorch 冗余权重与仓库元文件；
  3. 逐文件下载，``<file>.part`` 半成品 + ``Range: bytes=N-`` **断点续传**；
     服务端不支持 Range（返回 200）时自动整文件重下；完成后按字节数校验；
  4. 落盘到 ``<项目根>/models/<模型名>/``（与 registry 解析顺序一致，
     兼容 ``~/models/<模型名>`` 已有放置）。

进度经回调贯穿（已下载字节 / 总字节 / 当前文件），UI 层据此显示进度条。
下载完成前应用不可用（阻塞由 UI 层负责）；本模块为纯 Python + requests，
不依赖 huggingface_hub（PyInstaller 冻结环境同样可用）。
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests

from .registry import _project_root, config_dir, config_models_dir

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://hf-mirror.com"
DEFAULT_NAMESPACE = "arlenecc"

# 「只拉需要的文件」白名单：模型权重 + 运行必需的配置/分词器/自定义代码。
# .py：trust_remote_code 模型（jina-v5 MLX 自定义架构、xlm-roberta 配置类）
# 加载时需要随目录提供的自定义代码文件——镜像仓库里它们必须一起落盘。
_INCLUDE_EXTS = (".onnx", ".safetensors", ".gguf", ".json", ".txt",
                 ".model", ".tiktoken", ".py")
# fp32 ONNX 的外挂权重数据没有常规扩展名（model.onnx_data），按后缀特判。
_INCLUDE_SUFFIXES = (".onnx_data",)
# 排除：PyTorch 冗余权重（与 onnx/safetensors 二选一）+ 文档/仓库元文件/图片。
_EXCLUDE_EXTS = (".bin", ".pt", ".pth", ".h5", ".msgpack", ".ckpt",
                 ".md", ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".pyc")
_EXCLUDE_NAMES = (".gitattributes",)
# ONNX 权重文件的选择优先级：量化优先（体积小、CPU 上有 int8 加速），
# 找不到再退 fp32。同一仓库常并存 fp16/int8/quantized/uint8 四个变体
# （~1.1GB），只下载一个即可运行。
_ONNX_PREFERENCE = ("model_quantized.onnx", "model_int8.onnx",
                    "model_uint8.onnx", "model.onnx")

_DOWNLOAD_RETRIES = 3          # 单文件网络错误重试次数（续传语义）
_CHUNK = 1024 * 1024           # 1MB 流式块
_HTTP_TIMEOUT = (10, 60)       # (连接, 读) 超时


class ModelDownloadError(Exception):
    """模型下载失败（网络 / HTTP 状态 / 大小校验不一致）。"""


class ModelDownloadCancelled(Exception):
    """用户取消下载（cancel_event 置位）。"""


# ---------------------------------------------------------------------------
# 模型任务定义与缺失检测
# ---------------------------------------------------------------------------

@dataclass
class ModelTask:
    """一个待确保存在的本地模型。"""
    key: str                # "embedding" / "reranker" / ...
    display_name: str       # UI 展示名
    repo_name: str          # HF 仓库名（实际仓库 = namespace/repo_name）
    target_dirs: List[str] = field(default_factory=list)  # 候选目录，[0] 为下载落点
    checker: str = "mlx"    # 完整性判定方式："mlx"（safetensors+config）| "onnx"
    blocking: bool = True   # 是否属于「下载完成前不可用」的必需模型

    @property
    def dest_dir(self) -> str:
        return self.target_dirs[0] if self.target_dirs else ""


def _registry_candidates(value: str, root: str) -> List[str]:
    """与 ``registry.resolve_model_path`` 完全一致的候选目录顺序（去重保序）。

    缺失检测必须验证「应用实际会加载的那条路径」：用户在配置里指定了自己的
    模型目录（绝对路径 / 相对子目录）时，不能只扫默认 ``models/``——否则
    自定义模型的用户会被永远判定缺失。
    """
    out: List[str] = []
    if os.path.isabs(value):
        # 绝对路径：用户显式指定的就是唯一位置（registry 步骤 1 存在即用，
        # 不存在也无其它回退语义）。
        out.append(value)
    else:
        # registry 步骤 1：按 CWD 相对路径存在即用
        out.append(value)
        # registry 步骤 2：<config_dir>/models/<value>（自动下载默认落点）
        out.append(os.path.join(config_models_dir(), value))
        # registry 步骤 3：<config_dir>/<value>
        out.append(os.path.join(config_dir(), value))
        # registry 步骤 4：<root>/models/<value>（仓库自带权重）
        out.append(os.path.join(root, "models", value))
        # registry 步骤 5：<root>/<value>
        out.append(os.path.join(root, value))
        # registry 步骤 6：~/models/<value>
        out.append(os.path.join(os.path.expanduser("~"), "models", value))
    seen, uniq = set(), []
    for d in out:
        if d and d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq


def _download_dest(value: str, root: str) -> str:
    """自动下载的落点：用户指定的目录优先，未指定时落配置目录下的 models/。

    - 绝对路径 → 原位补全用户指定的目录（下载后应用直接可用）；
    - 相对路径 → 已存在（哪怕不完整）的候选目录原位补全；全新则落
      **配置文件所在目录的 models 子目录**（默认 ``~/.base-agent/models``，
      冻结环境项目根只读）：
      ``models/`` 前缀值落 ``<config_dir>/<value>``（即 ``<config>/models/<名>``），
      其余（裸名/相对子路径）落 ``<config_dir>/models/<value>``——
      均能被 ``resolve_model_path`` 解析加载。
    """
    if os.path.isabs(value):
        return value
    for d in _registry_candidates(value, root):
        if os.path.isdir(d):
            return d  # 已存在（可能不完整）→ 原地补全
    if value.startswith("models/"):
        return os.path.join(config_dir(), value)
    return os.path.join(config_models_dir(), value)


def _dir_complete(directory: str, checker: str) -> bool:
    """目录级完整性：按后端类型判断权重是否「看起来就绪」。

    - mlx：至少一个 *.safetensors 且含 config.json（mlx_lm/AutoTokenizer 加载前提）；
    - onnx：含任一 *.onnx（LocalReranker 找 model.onnx / model_quantized.onnx）。
    """
    if not os.path.isdir(directory):
        return False
    names = os.listdir(directory)
    if checker == "onnx":
        if any(n.lower().endswith(".onnx") for n in names):
            return True
        # HF 仓库常见布局：权重在 onnx/ 子目录（下载保持镜像原始路径）。
        onnx_dir = os.path.join(directory, "onnx")
        if os.path.isdir(onnx_dir):
            return any(n.lower().endswith(".onnx")
                       for n in os.listdir(onnx_dir))
        return False
    return (any(n.lower().endswith(".safetensors") for n in names)
            and "config.json" in names)


def task_present(task: ModelTask) -> bool:
    return any(_dir_complete(d, task.checker) for d in task.target_dirs)


def get_default_model_tasks(config: Any,
                            project_root: Optional[str] = None) -> List[ModelTask]:
    """按当前配置收集「默认本地模型」任务（provider=local 且给了模型名才需要）。

    ``target_dirs[0]`` 是下载落点（对齐用户配置的路径），其余为 registry
    同序的既有候选（用户在 ``~/models`` 等处已有完整副本时不再重复下载）。
    """
    root = project_root or _project_root()
    tasks: List[ModelTask] = []
    if getattr(config, "embedding_provider", "local") == "local":
        val = (getattr(config, "local_embedding_model", "") or "").strip()
        if val:
            checker = ("onnx" if getattr(config, "local_embedding_backend",
                                         "mlx") == "onnx" else "mlx")
            tasks.append(ModelTask(
                key="embedding", display_name="本地 Embedding",
                repo_name=os.path.basename(val.rstrip("/\\")),
                target_dirs=_task_dirs(val, root), checker=checker))
    if getattr(config, "reranker_provider", "local") == "local":
        val = (getattr(config, "local_reranker_model", "") or "").strip()
        if val:
            tasks.append(ModelTask(
                key="reranker", display_name="本地 Reranker",
                repo_name=os.path.basename(val.rstrip("/\\")),
                target_dirs=_task_dirs(val, root), checker="onnx"))
    return tasks


def _task_dirs(value: str, root: str) -> List[str]:
    """任务目录表：[下载落点] + 其余 registry 候选（去重保序）。"""
    dest = _download_dest(value, root)
    return [dest] + [d for d in _registry_candidates(value, root) if d != dest]


def check_missing_models(config: Any,
                         project_root: Optional[str] = None) -> List[ModelTask]:
    """返回缺失的默认本地模型（目录级完整性不满足）。"""
    return [t for t in get_default_model_tasks(config, project_root)
            if not task_present(t)]


# ---------------------------------------------------------------------------
# HF REST API：文件清单 + 白名单过滤
# ---------------------------------------------------------------------------

def list_repo_files(base_url: str, repo: str,
                    timeout: Tuple[float, float] = _HTTP_TIMEOUT
                    ) -> List[Tuple[str, int]]:
    """``GET /api/models/{repo}/tree/main`` 递归列出仓库文件 (path, size)。

    LFS 文件的 ``size`` 是指针文件大小，真实大小在 ``lfs.size``——优先取之。
    """
    url = f"{base_url.rstrip('/')}/api/models/{repo}/tree/main"
    resp = requests.get(url, params={"recursive": "true", "expand": "true"},
                        timeout=timeout)
    if resp.status_code == 404:
        raise ModelDownloadError(
            f"镜像站上不存在仓库 {repo}（404）。请确认模型名，或先在"
            f" https://hf-mirror.com/{repo} 手动确认该仓库已上传。")
    resp.raise_for_status()
    entries = resp.json()
    out: List[Tuple[str, int]] = []
    for e in entries:
        if not isinstance(e, dict) or e.get("type") != "file":
            continue
        lfs = e.get("lfs") or {}
        size = int(lfs.get("size") or e.get("size") or 0)
        out.append((str(e.get("path") or ""), size))
    return out


def filter_needed_files(entries: List[Tuple[str, int]],
                        kind: str = "generic") -> List[Tuple[str, int]]:
    """白名单过滤：只保留运行所需文件（排除 PyTorch 冗余权重/文档/元文件）。

    kind 按模型后端细分（与 ``ModelTask.checker`` 对齐）：
      - ``"mlx"``：保留 safetensors 等权重；**不拉** onnx/gguf（互为备选实现）；
      - ``"onnx"``：只拉**一个** onnx 权重（按 _ONNX_PREFERENCE 量化优先）及其
        外挂数据；不拉 safetensors（ONNX Runtime 用不到，动辄 500MB+）；
      - ``"generic"``：全部白名单文件（旧行为）。
    """
    needed: List[Tuple[str, int]] = []
    for path, size in entries:
        parts = path.replace("\\\\", "/").split("/")
        if any(seg == "__pycache__" for seg in parts[:-1]):
            continue  # 字节码缓存，永不下载
        name = parts[-1]
        low = name.lower()
        if low in _EXCLUDE_NAMES or low.startswith(("readme", "license")):
            continue
        if low.endswith(_INCLUDE_SUFFIXES):
            needed.append((path, size))
            continue
        ext = os.path.splitext(low)[1]
        if ext in _EXCLUDE_EXTS:
            continue
        if ext in _INCLUDE_EXTS:
            needed.append((path, size))

    if kind == "mlx":
        return [(p, s) for p, s in needed
                if not p.lower().endswith((".onnx", ".onnx_data", ".gguf"))]
    if kind == "onnx":
        onnxs = [(p, s) for p, s in needed if p.lower().endswith(".onnx")]
        if onnxs:
            by_name = {p.rsplit("/", 1)[-1].lower(): (p, s) for p, s in onnxs}
            picked = next((by_name[n] for n in _ONNX_PREFERENCE if n in by_name),
                          min(onnxs, key=lambda ps: ps[1]))
            # 只保留所选权重的伴随数据（model.onnx_data）；其余 onnx 变体、
            # 外挂数据与 safetensors/npz/mlx/gguf 权重（ONNX Runtime 用不到）
            # 全部剔除。
            data_suffix = picked[0] + "_data"
            extras = [(p, s) for p, s in needed if p == data_suffix]
            kept = [(p, s) for p, s in needed
                    if not p.lower().endswith(
                        (".onnx", ".onnx_data", ".safetensors",
                         ".npz", ".mlx", ".gguf"))]
            return kept + [picked] + extras
    return needed


# ---------------------------------------------------------------------------
# 下载（断点续传）
# ---------------------------------------------------------------------------

def _download_one(session: "requests.Session", url: str, dest: str,
                  expected_size: int,
                  progress_cb: Callable[[int], None],
                  cancel_event: Optional[threading.Event]) -> None:
    """下载单个文件到 dest，``.part`` + Range 断点续传，完成后按字节数校验。"""
    part = dest + ".part"
    done = os.path.getsize(part) if os.path.exists(part) else 0
    # 半成品比预期还大 → 内容已错乱，只能整文件重下
    if expected_size and done > expected_size:
        try:
            os.remove(part)
        except OSError:
            pass
        done = 0

    last_err: Optional[Exception] = None
    for attempt in range(_DOWNLOAD_RETRIES):
        if cancel_event is not None and cancel_event.is_set():
            raise ModelDownloadCancelled()
        headers: Dict[str, str] = {}
        mode = "wb"
        if done > 0:
            headers["Range"] = f"bytes={done}-"
            mode = "ab"
        try:
            with session.get(url, headers=headers, stream=True,
                             timeout=_HTTP_TIMEOUT) as resp:
                if resp.status_code == 416:
                    # Range 越界：半成品与服务器状态脱节，清掉重下
                    done = 0
                    try:
                        os.remove(part)
                    except OSError:
                        pass
                    continue
                if resp.status_code == 200 and done > 0:
                    # 服务端不支持 Range（忽略请求头返回全量）→ 重下
                    done = 0
                    mode = "wb"
                elif resp.status_code not in (200, 206):
                    raise ModelDownloadError(
                        f"HTTP {resp.status_code}: {url}")
                os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
                with open(part, mode) as f:
                    for chunk in resp.iter_content(chunk_size=_CHUNK):
                        if cancel_event is not None and cancel_event.is_set():
                            raise ModelDownloadCancelled()
                        if not chunk:
                            continue
                        f.write(chunk)
                        progress_cb(len(chunk))
            size = os.path.getsize(part)
            if expected_size and size != expected_size:
                if size > expected_size:
                    # 错乱 → 清掉，下一轮整文件重下
                    try:
                        os.remove(part)
                    except OSError:
                        pass
                    done = 0
                else:
                    done = size  # 不完整 → 保留半成品续传
                raise ModelDownloadError(
                    f"文件大小校验失败: {os.path.basename(dest)} "
                    f"({size} != {expected_size})")
            os.replace(part, dest)  # 原子落盘（同目录 rename）
            return
        except ModelDownloadCancelled:
            raise
        except (requests.RequestException, ModelDownloadError) as e:
            last_err = e
            done = os.path.getsize(part) if os.path.exists(part) else 0
            logger.warning("下载 %s 失败（第 %d 次，已续传 %d 字节）: %s",
                           url, attempt + 1, done, e)
            time.sleep(min(2 ** attempt, 4))
    raise ModelDownloadError(f"下载失败（已重试 {_DOWNLOAD_RETRIES} 次）: "
                             f"{url}: {last_err}")


def download_model(task: ModelTask, base_url: str, namespace: str,
                   progress_cb: Callable[[int, int, str], None],
                   cancel_event: Optional[threading.Event] = None) -> Dict[str, Any]:
    """下载一个模型到 ``task.dest_dir``（对齐用户配置的路径），返回下载统计。

    progress_cb(done_bytes, total_bytes, current_label) —— done/total 为累计值，
    可直接喂给进度条；current_label 为当前文件（或阶段）描述。
    """
    repo = f"{namespace}/{task.repo_name}"
    entries = list_repo_files(base_url, repo)
    needed = filter_needed_files(entries, kind=task.checker)
    if not needed:
        raise ModelDownloadError(f"仓库 {repo} 中没有可识别的必需文件")
    total = sum(s for _, s in needed)
    progress_cb(0, total, f"{task.display_name}: 获取到 {len(needed)} 个文件，"
                          f"共 {total / 1e9:.2f} GB")
    dest_dir = task.dest_dir
    session = requests.Session()
    done_total = 0

    for path, size in needed:
        dest = os.path.join(dest_dir, *path.split("/"))
        if os.path.exists(dest) and (not size or os.path.getsize(dest) == size):
            done_total += size  # 已完整落盘 → 跳过（续传跨进程也生效）
            progress_cb(done_total, total, f"{task.display_name} · {path}（已存在）")
            continue

        def file_progress(n: int, _path: str = path) -> None:
            nonlocal done_total
            done_total += n
            progress_cb(done_total, total, f"{task.display_name} · {_path}")

        url = (f"{base_url.rstrip('/')}/{repo}/resolve/main/"
               f"{quote(path)}")
        _download_one(session, url, dest, size, file_progress, cancel_event)
    return {"repo": repo, "files": len(needed), "bytes": total,
            "dest": dest_dir}


def download_missing_models(config: Any,
                            progress_cb: Callable[[int, int, str], None],
                            cancel_event: Optional[threading.Event] = None,
                            project_root: Optional[str] = None) -> List[Dict[str, Any]]:
    """依次下载全部缺失的默认本地模型，返回每个模型的下载统计。

    落点由各任务自带（``task.dest_dir``，对齐用户配置的模型路径）。
    """
    base_url = (getattr(config, "hf_mirror_base_url", "") or DEFAULT_BASE_URL)
    namespace = (getattr(config, "hf_model_namespace", "") or DEFAULT_NAMESPACE)
    missing = check_missing_models(config, project_root)
    results: List[Dict[str, Any]] = []
    for task in missing:
        results.append(download_model(
            task, base_url, namespace, progress_cb, cancel_event))
    return results
