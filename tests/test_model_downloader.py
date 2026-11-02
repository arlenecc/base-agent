# -*- coding: utf-8 -*-
"""model_downloader 测试：文件清单解析、白名单过滤、断点续传、端到端下载。

用 ThreadingHTTPServer 起一个迷你 HF 镜像（树接口 JSON + 支持 Range 的文件
服务），覆盖：
  - filter_needed_files：只留运行所需文件，排除 PyTorch 冗余权重/文档/元文件
  - list_repo_files：LFS 文件取 lfs.size 真实大小、跳过目录项
  - download_model：只下载白名单文件、.part 清理、进度累计
  - 断点续传：半成品 + Range 续传；超大错乱半成品整文件重下
  - check_missing_models：目录级完整性判定 + provider 切换
  - download_missing_models：多模型端到端，二次运行零下载

路径约定与真实 HF 一致：树接口 entry.path 是**相对仓库根**的路径；
下载 URL = {base}/{repo}/resolve/main/{相对路径}（故文件表按 f"{repo}/{相对}" 键存）。
"""
import json
import os
import pathlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import unquote

import pytest

from agent.local import model_downloader as md


@pytest.fixture(autouse=True)
def _isolated_config_dir(tmp_path, monkeypatch):
    """把配置目录重定向到 tmp：默认落点/解析不碰真实 ~/.base-agent。"""
    from agent.local import registry
    monkeypatch.setattr(registry, "CONFIG_DIR_OVERRIDE", str(tmp_path / "cfg"))


# ---------------------------------------------------------------------------
# 迷你 HF 镜像服务器
# ---------------------------------------------------------------------------

class _FakeMirror:
    """本地假镜像：/api/models/{repo}/tree/main 返回 JSON，resolve/main 支持 Range。"""

    def __init__(self, trees, files):
        self.trees = trees          # {repo: [entry, ...]}
        self.files = files          # {f"{repo}/{path}": bytes}
        self.range_starts = []      # 记录每次 Range 请求 (path, start)
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # 静默访问日志
                pass

            def do_GET(self):
                if self.path.startswith("/api/models/"):
                    # /api/models/{repo}/tree/main；未知 repo → 404（镜像语义）
                    rest = self.path[len("/api/models/"):]
                    repo = rest.split("/tree/main", 1)[0]
                    if repo not in outer.trees:
                        self.send_response(404)
                        self.end_headers()
                        return
                    body = json.dumps(outer.trees[repo]).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                marker = "/resolve/main/"
                if marker in self.path:
                    # URL 里 {repo}/resolve/main/{相对路径} → 取全路径查文件表
                    rel = unquote(self.path.split(marker, 1)[1])
                    repo_prefix = self.path[1:].split("/resolve/main/", 1)[0]
                    key = f"{repo_prefix}/{rel}"
                    data = outer.files.get(key)
                    if data is None:
                        self.send_response(404)
                        self.end_headers()
                        return
                    rng = self.headers.get("Range")
                    if rng and rng.startswith("bytes="):
                        start = int(rng[6:].split("-")[0] or 0)
                        outer.range_starts.append((key, start))
                        chunk = data[start:]
                        self.send_response(206)
                        self.send_header(
                            "Content-Range",
                            f"bytes {start}-{len(data) - 1}/{len(data)}")
                    else:
                        chunk = data
                        self.send_response(200)
                    self.send_header("Content-Length", str(len(chunk)))
                    self.end_headers()
                    self.wfile.write(chunk)
                    return
                self.send_response(404)
                self.end_headers()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self._thread = threading.Thread(
            target=self.server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def mirror_factory():
    made = []

    def _make(trees, files):
        m = _FakeMirror(trees, files)
        made.append(m)
        return m

    yield _make
    for m in made:
        m.stop()


def _tree(rel_files, lfs_exts=(".safetensors", ".onnx", ".gguf")):
    """由 {相对路径: bytes} 构造树接口 JSON；大文件标记 LFS（真实大小）。"""
    out = []
    for p, b in rel_files.items():
        entry = {"type": "file", "path": p, "size": len(b)}
        if p.endswith(lfs_exts):
            entry["lfs"] = {"size": len(b)}
        out.append(entry)
    return out


def _http_files(repo, rel_files):
    """把 {相对路径: bytes} 映射成服务器文件表 {f"{repo}/{相对}": bytes}。"""
    return {f"{repo}/{p}": b for p, b in rel_files.items()}


# ---------------------------------------------------------------------------
# 白名单过滤 / 文件清单解析
# ---------------------------------------------------------------------------

def test_filter_needed_files():
    entries = [
        ("config.json", 2), ("tokenizer.json", 3),
        ("model.safetensors", 5000), ("model.onnx", 7),
        ("model.onnx_data", 123), ("sub/dir/vocab.txt", 4),
        ("weights.gguf", 9), ("tokenizer.model", 2),
        # —— 应排除 ——
        ("pytorch_model.bin", 9), ("weights.pt", 5), ("weights.pth", 5),
        ("model.h5", 5), ("model.ckpt", 5), ("README.md", 1),
        ("LICENSE", 1), (".gitattributes", 1), ("card.png", 1),
    ]
    got = {p for p, _ in md.filter_needed_files(entries)}
    assert got == {"config.json", "tokenizer.json", "model.safetensors",
                   "model.onnx", "model.onnx_data", "sub/dir/vocab.txt",
                   "weights.gguf", "tokenizer.model"}


def test_filter_needed_files_by_backend_kind():
    entries = [
        ("config.json", 2), ("tokenizer.json", 3),
        ("onnx/model.onnx", 1100), ("onnx/model_fp16.onnx", 557),
        ("onnx/model_int8.onnx", 279), ("onnx/model_quantized.onnx", 279),
        ("onnx/model_uint8.onnx", 279), ("onnx/model.onnx_data", 5),
        ("model.safetensors", 556), ("modeling_x.py", 43),
        ("__pycache__/model.cpython-313.pyc", 1),
        ("pytorch_model.bin", 561),
    ]
    # onnx 任务：只拉一个 onnx（量化优先）+ 配置/分词/自定义代码；
    # 不拉 safetensors（ONNX Runtime 用不到）、不拉其余变体及其外挂数据
    got = {p for p, _ in md.filter_needed_files(entries, kind="onnx")}
    assert got == {"config.json", "tokenizer.json", "onnx/model_quantized.onnx",
                   "modeling_x.py"}
    # mlx 任务：保留 safetensors，不拉任何 onnx
    got = {p for p, _ in md.filter_needed_files(entries, kind="mlx")}
    assert got == {"config.json", "tokenizer.json", "model.safetensors",
                   "modeling_x.py"}


def test_filter_needed_files_onnx_data_companion_follows_pick():
    entries = [("onnx/model.onnx", 1000), ("onnx/model.onnx_data", 900)]
    # 仓库无量化变体 → 按优先级退回 model.onnx，其外挂数据跟随下载
    got = {p for p, _ in md.filter_needed_files(entries, kind="onnx")}
    assert got == {"onnx/model.onnx", "onnx/model.onnx_data"}


def test_list_repo_files_reads_lfs_size_and_skips_dirs(mirror_factory):
    trees = {"arlenecc/x": [
        {"type": "file", "path": "a.safetensors", "size": 32,
         "lfs": {"size": 999}},
        {"type": "file", "path": "config.json", "size": 3},
        {"type": "directory", "path": "sub"},
    ]}
    m = mirror_factory(trees, {})
    got = md.list_repo_files(m.url, "arlenecc/x")
    assert ("a.safetensors", 999) in got   # LFS 真实大小
    assert ("config.json", 3) in got
    assert all(p != "sub" for p, _ in got)  # 目录项跳过


def test_list_repo_files_404_gives_friendly_error(mirror_factory):
    m = mirror_factory({}, {})
    with pytest.raises(md.ModelDownloadError, match="404|不存在"):
        md.list_repo_files(m.url, "arlenecc/missing")


# ---------------------------------------------------------------------------
# 下载：只拉需要的文件
# ---------------------------------------------------------------------------

def test_download_model_only_needed_files(mirror_factory, tmp_path):
    repo = "arlenecc/fake"
    rel = {
        "config.json": b"{}",
        "tokenizer.json": b"tok",
        "model.safetensors": b"W" * 5000,
        "pytorch_model.bin": b"X" * 5000,   # 冗余，不下载
        "README.md": b"doc",
    }
    m = mirror_factory({repo: _tree(rel)}, _http_files(repo, rel))
    dest_root = str(tmp_path / "models")
    task = md.ModelTask("embedding", "Embedding", "fake",
                        [dest_root + "/fake"], "mlx")
    progress = []
    stats = md.download_model(task, m.url, "arlenecc",
                              lambda d, t, s: progress.append((d, t, s)))
    dest = tmp_path / "models" / "fake"
    assert (dest / "config.json").read_bytes() == b"{}"
    assert (dest / "model.safetensors").read_bytes() == b"W" * 5000
    assert not (dest / "pytorch_model.bin").exists()
    assert not (dest / "README.md").exists()
    assert not (dest / "arlenecc").exists()    # 相对路径不得嵌套出 repo 目录
    assert not list(dest.glob("*.part"))       # 半成品已清理
    assert stats["bytes"] == 2 + 3 + 5000
    assert progress[-1][0] == stats["bytes"]   # 进度累计到总量


# ---------------------------------------------------------------------------
# 断点续传
# ---------------------------------------------------------------------------

def test_download_resumes_from_part(mirror_factory, tmp_path):
    repo = "arlenecc/fake"
    data = b"R" * 10000
    rel = {"model.safetensors": data}
    m = mirror_factory({repo: _tree(rel)}, _http_files(repo, rel))
    dest_dir = tmp_path / "models" / "fake"
    dest_dir.mkdir(parents=True)
    part = dest_dir / "model.safetensors.part"
    part.write_bytes(data[:1234])              # 模拟上次中断留下的半成品

    task = md.ModelTask("embedding", "Embedding", "fake", [str(dest_dir)], "mlx")
    md.download_model(task, m.url, "arlenecc", lambda d, t, s: None)

    assert (dest_dir / "model.safetensors").read_bytes() == data
    assert not part.exists()
    # 服务器收到了从 1234 字节续传的 Range 请求
    assert (f"{repo}/model.safetensors", 1234) in m.range_starts


def test_download_restarts_oversized_corrupt_part(mirror_factory, tmp_path):
    repo = "arlenecc/fake"
    data = b"C" * 4000
    rel = {"model.safetensors": data}
    m = mirror_factory({repo: _tree(rel)}, _http_files(repo, rel))
    dest_dir = tmp_path / "models" / "fake"
    dest_dir.mkdir(parents=True)
    part = dest_dir / "model.safetensors.part"
    part.write_bytes(data + b"junk-junk")      # 比预期大 → 内容错乱

    task = md.ModelTask("embedding", "Embedding", "fake", [str(dest_dir)], "mlx")
    md.download_model(task, m.url, "arlenecc", lambda d, t, s: None)

    assert (dest_dir / "model.safetensors").read_bytes() == data
    # 错乱半成品被清掉 → 首次请求不带 Range（整文件重下）
    assert m.range_starts == []


def test_existing_complete_file_is_skipped(mirror_factory, tmp_path):
    repo = "arlenecc/fake"
    data = b"S" * 1000
    rel = {"model.safetensors": data}
    m = mirror_factory({repo: _tree(rel)}, _http_files(repo, rel))
    dest_dir = tmp_path / "models" / "fake"
    dest_dir.mkdir(parents=True)
    (dest_dir / "model.safetensors").write_bytes(data)  # 已完整

    task = md.ModelTask("embedding", "Embedding", "fake", [str(dest_dir)], "mlx")
    md.download_model(task, m.url, "arlenecc", lambda d, t, s: None)
    assert m.range_starts == [] and not list(dest_dir.glob("*.part"))


# ---------------------------------------------------------------------------
# 缺失检测
# ---------------------------------------------------------------------------

def _config(tmp_path, emb="models/emb-a", rer="rerank-b",
            eprov="local", rprov="local", **extra):
    return SimpleNamespace(
        embedding_provider=eprov, local_embedding_model=emb,
        reranker_provider=rprov, local_reranker_model=rer, **extra)


def test_check_missing_models_by_directory_state(tmp_path):
    config = _config(tmp_path)
    # 全缺失
    missing = md.check_missing_models(config, project_root=str(tmp_path))
    assert {t.key for t in missing} == {"embedding", "reranker"}
    # embedding 就绪（safetensors + config.json）
    d = tmp_path / "models" / "emb-a"
    d.mkdir(parents=True)
    (d / "model.safetensors").write_bytes(b"x")
    (d / "config.json").write_text("{}")
    missing = md.check_missing_models(config, project_root=str(tmp_path))
    assert {t.key for t in missing} == {"reranker"}
    # reranker 就绪（model_quantized.onnx）
    d2 = tmp_path / "models" / "rerank-b"
    d2.mkdir(parents=True)
    (d2 / "model_quantized.onnx").write_bytes(b"x")
    assert md.check_missing_models(config, project_root=str(tmp_path)) == []


def test_check_missing_models_respects_provider(tmp_path):
    # provider 切到 api（fastembed）→ 无本地目录要求，无任务
    config = _config(tmp_path, eprov="api", rprov="api")
    assert md.check_missing_models(config, project_root=str(tmp_path)) == []


def test_check_missing_models_ignores_incomplete_dirs(tmp_path):
    config = _config(tmp_path)
    # 目录存在但只有 config.json（没有权重）→ 仍视为缺失
    d = tmp_path / "models" / "emb-a"
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    missing = md.check_missing_models(config, project_root=str(tmp_path))
    assert {t.key for t in missing} == {"embedding", "reranker"}


def test_default_download_lands_in_config_dir_models(tmp_path):
    """用户没指定路径时：默认落配置文件所在目录的 models/ 子目录。"""
    config = _config(tmp_path)  # emb="models/emb-a", rer="rerank-b"
    missing = md.check_missing_models(config, project_root=str(tmp_path))
    by_key = {t.key: t for t in missing}
    # "models/" 前缀值 → <config_dir>/<value>（即 <config>/models/<名>）
    assert by_key["embedding"].dest_dir == str(
        tmp_path / "cfg" / "models" / "emb-a")
    # 裸名 → <config_dir>/models/<名>
    assert by_key["reranker"].dest_dir == str(
        tmp_path / "cfg" / "models" / "rerank-b")


# ---------------------------------------------------------------------------
# 用户自定义路径：缺失检测必须验证「应用实际会加载的那条路径」
# ---------------------------------------------------------------------------

def test_check_missing_models_respects_custom_absolute_path(tmp_path):
    # 绝对路径：检测该路径本身，下载落点也是它（原位补全）
    custom = tmp_path / "my-models" / "custom-emb"
    config = _config(tmp_path, emb=str(custom))
    missing = md.check_missing_models(config, project_root=str(tmp_path))
    assert {t.key for t in missing} == {"embedding", "reranker"}
    emb_task = next(t for t in missing if t.key == "embedding")
    assert emb_task.dest_dir == str(custom), "下载应直接补全用户指定的目录"
    # 用户把模型放到自己指定的目录 → 立即判定就绪（不再永远缺失）
    custom.mkdir(parents=True)
    (custom / "model.safetensors").write_bytes(b"x")
    (custom / "config.json").write_text("{}")
    missing = md.check_missing_models(config, project_root=str(tmp_path))
    assert {t.key for t in missing} == {"reranker"}


def test_check_missing_models_respects_custom_relative_path(tmp_path):
    # 相对子目录：默认 models/<名> 扫不到 → 落点按 registry 顺序可解析
    config = _config(tmp_path, emb="custom-models/emb-x")
    missing = md.check_missing_models(config, project_root=str(tmp_path))
    emb_task = next(t for t in missing if t.key == "embedding")
    assert emb_task.dest_dir == str(
        tmp_path / "cfg" / "models" / "custom-models" / "emb-x")
    d = pathlib.Path(emb_task.dest_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.safetensors").write_bytes(b"x")
    (d / "config.json").write_text("{}")
    assert all(t.key != "embedding"
               for t in md.check_missing_models(config,
                                                project_root=str(tmp_path)))


def test_custom_absolute_path_download_fills_user_dir(mirror_factory, tmp_path):
    # 端到端：用户指定绝对路径 → 自动下载直接补全到该目录
    rel_emb = {"config.json": b"{}", "model.safetensors": b"E" * 100}
    rel_rer = {"config.json": b"{}", "model.onnx": b"O" * 50}
    m = mirror_factory(
        {"arlenecc/custom-emb": _tree(rel_emb),
         "arlenecc/rerank-b": _tree(rel_rer)},
        {**_http_files("arlenecc/custom-emb", rel_emb),
         **_http_files("arlenecc/rerank-b", rel_rer)})
    custom = tmp_path / "my-models" / "custom-emb"
    config = _config(tmp_path, emb=str(custom), rer="rerank-b",
                     hf_mirror_base_url=m.url, hf_model_namespace="arlenecc")
    results = md.download_missing_models(config, lambda d, t, s: None,
                                         project_root=str(tmp_path))
    assert (custom / "model.safetensors").read_bytes() == b"E" * 100
    assert results[0]["dest"] == str(custom)
    assert md.check_missing_models(config, project_root=str(tmp_path)) == []


def test_embedding_checker_respects_onnx_backend(tmp_path):
    # 用户把 embedding 指到 ONNX 版并声明 backend=onnx → 完整性按 onnx 判定
    d = tmp_path / "models" / "emb-onnx"
    d.mkdir(parents=True)
    (d / "onnx").mkdir()
    (d / "onnx" / "model_quantized.onnx").write_bytes(b"x")
    (d / "config.json").write_text("{}")
    config = _config(tmp_path, emb="models/emb-onnx")
    # mlx 判定（默认）会误判缺失；声明 backend=onnx 后即就绪
    config.local_embedding_backend = "onnx"
    assert all(t.key != "embedding"
               for t in md.check_missing_models(config,
                                                project_root=str(tmp_path)))


# ---------------------------------------------------------------------------
# 端到端：多模型下载 + 二次运行零下载
# ---------------------------------------------------------------------------

def test_download_missing_models_end_to_end(mirror_factory, tmp_path):
    emb = {"config.json": b"{}", "model.safetensors": b"E" * 3000,
           "pytorch_model.bin": b"Z" * 3000}
    rer = {"config.json": b"{}", "tokenizer.json": b"t",
           "model.onnx": b"O" * 2000}
    trees = {"arlenecc/emb-a": _tree(emb), "arlenecc/rerank-b": _tree(rer)}
    files = {**_http_files("arlenecc/emb-a", emb),
             **_http_files("arlenecc/rerank-b", rer)}
    m = mirror_factory(trees, files)

    config = _config(tmp_path, hf_mirror_base_url=m.url,
                     hf_model_namespace="arlenecc")
    progress = []
    results = md.download_missing_models(
        config, lambda d, t, s: progress.append((d, t, s)),
        project_root=str(tmp_path))

    assert [r["repo"] for r in results] == ["arlenecc/emb-a", "arlenecc/rerank-b"]
    # 默认落点：配置文件所在目录的 models/ 子目录（隔离夹具重定向到 tmp/cfg）
    assert (tmp_path / "cfg" / "models" / "emb-a" / "model.safetensors").read_bytes() == b"E" * 3000
    assert not (tmp_path / "cfg" / "models" / "emb-a" / "pytorch_model.bin").exists()
    assert (tmp_path / "cfg" / "models" / "rerank-b" / "model.onnx").read_bytes() == b"O" * 2000
    # 下载后必须能被 resolve_model_path 解析到（应用可加载）
    from agent.local.registry import resolve_model_path
    assert resolve_model_path("embedding", "models/emb-a",
                              project_root=str(tmp_path)) == \
        str(tmp_path / "cfg" / "models" / "emb-a")
    assert resolve_model_path("reranker", "rerank-b",
                              project_root=str(tmp_path)) == \
        str(tmp_path / "cfg" / "models" / "rerank-b")
    # 进度按模型各自累计：每个模型的进度都走到自己的总量
    totals = {r["bytes"] for r in results}
    reached = {d for d, _t, _s in progress}
    assert totals <= reached

    # 二次运行：全部就绪 → 零下载、零请求
    assert md.check_missing_models(config, project_root=str(tmp_path)) == []
    m.range_starts.clear()
    assert md.download_missing_models(config, lambda *a: None,
                                      project_root=str(tmp_path)) == []
    assert m.range_starts == []
