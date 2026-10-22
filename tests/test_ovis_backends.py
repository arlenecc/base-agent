"""OvisOCR2 多后端（mlx-vlm / gguf / onnx）的单元测试。

不加载真实权重：模型目录用临时目录 + 哨兵文件，模型对象用假类，依赖探测
用 monkeypatch。覆盖：
- resolve_model_dir 按后端判据（safetensors / .gguf / genai_config.json）
- is_ovis_available 的依赖 + 目录双重门控
- set_ovis_model_config 配置变化时释放已加载单例
- gguf 推理的消息形态（text + image_url）与结果清洗
- _vlm_ocr 的后端分发（gguf / mlx / onnx 缺依赖时回退 None）

注意：_candidate_model_dirs 会兜底到仓库真实 models/ 目录，测试一律用
monkeypatch 把候选目录钉死到 tmp_path，保证与宿主机环境无关。
"""
import importlib.util
import json
import sys
import types

import pytest

from agent.rag import ovis_ocr_parser as ovis


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _mk(tmp_path, name, files):
    d = tmp_path / name
    d.mkdir()
    for f in files:
        (d / f).write_bytes(b"x")
    return d


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """把候选模型目录钉死到 tmp_path（屏蔽真实 models/ 兜底）。"""
    dirs = []

    def _add(name, files):
        d = _mk(tmp_path, name, files)
        dirs.append(d)
        return d

    monkeypatch.setattr(ovis, "_candidate_model_dirs", lambda: dirs)
    return _add


# ---------------------------------------------------------------------------
# resolve_model_dir：按后端判据
# ---------------------------------------------------------------------------

def test_resolve_mlx_requires_safetensors(isolated, monkeypatch):
    d = isolated("m", ["config.json"])
    monkeypatch.setattr(ovis, "_engine", "mlx-vlm")
    assert ovis.resolve_model_dir("mlx-vlm") is None
    (d / "model.safetensors").write_bytes(b"x")
    assert ovis.resolve_model_dir("mlx-vlm") == d


def test_resolve_mlx_accepts_index_json(isolated, monkeypatch):
    isolated("m2", ["model.safetensors.index.json"])
    monkeypatch.setattr(ovis, "_engine", "mlx-vlm")
    assert ovis.resolve_model_dir("mlx-vlm") is not None


def test_resolve_gguf_requires_gguf_file(isolated, monkeypatch):
    d = isolated("g", ["tokenizer.json"])
    monkeypatch.setattr(ovis, "_engine", "gguf")
    assert ovis.resolve_model_dir("gguf") is None
    (d / "ocr-model-q4.gguf").write_bytes(b"x")
    assert ovis.resolve_model_dir("gguf") == d


def test_resolve_onnx_requires_genai_config(isolated, monkeypatch):
    d = isolated("o", ["model.onnx"])
    monkeypatch.setattr(ovis, "_engine", "onnx")
    # 只有裸 model.onnx 不够——必须带 genai_config.json（ORT-GenAI 导出标志）
    assert ovis.resolve_model_dir("onnx") is None
    (d / "genai_config.json").write_text("{}")
    assert ovis.resolve_model_dir("onnx") == d


def test_gguf_files_separates_mmproj_from_main(isolated):
    d = isolated("g2", ["mmproj-model-f16.gguf", "ocr-q4.gguf", "notes.txt"])
    main, proj = ovis._gguf_files(d)
    assert main.endswith("ocr-q4.gguf")
    assert proj.endswith("mmproj-model-f16.gguf")


def test_genai_config_judgement_matches_real_export(isolated, monkeypatch):
    # 真实 ORT-GenAI 导出的 genai_config.json 是合法 JSON；判据只看文件存在
    d = isolated("o3", ["genai_config.json"])
    (d / "genai_config.json").write_text(json.dumps(
        {"model_type": "ovis", "vision": {"filename": "vision.onnx"}}))
    monkeypatch.setattr(ovis, "_engine", "onnx")
    assert ovis.resolve_model_dir("onnx") == d


# ---------------------------------------------------------------------------
# is_ovis_available：依赖 + 目录双重门控
# ---------------------------------------------------------------------------

def test_available_onnx_needs_og_package(isolated, monkeypatch):
    isolated("o2", ["genai_config.json"])
    monkeypatch.setattr(ovis, "_engine", "onnx")
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name: None if name == "onnxruntime_genai" else real(name))
    assert ovis.is_ovis_available() is False


def test_available_gguf_needs_dir(isolated, monkeypatch):
    monkeypatch.setattr(ovis, "_engine", "gguf")
    # llama_cpp 已装，但候选目录为空 → False
    assert ovis.is_ovis_available() is False
    isolated("g3", ["a.gguf", "mmproj-b.gguf"])
    assert ovis.is_ovis_available() is True


def test_available_unknown_engine_false(monkeypatch):
    monkeypatch.setattr(ovis, "_engine", "not-a-backend")
    assert ovis.is_ovis_available() is False


# ---------------------------------------------------------------------------
# set_ovis_model_config：变化即释放单例
# ---------------------------------------------------------------------------

def test_config_change_releases_model(monkeypatch):
    monkeypatch.setattr(ovis, "_model", object(), raising=False)
    monkeypatch.setattr(ovis, "_processor", object(), raising=False)
    monkeypatch.setattr(ovis, "_loaded_engine", "mlx-vlm", raising=False)
    monkeypatch.setattr(ovis, "_model_load_error", None, raising=False)
    ovis.set_ovis_model_config("gguf", "Ovis-gguf", "")
    assert ovis._model is None
    assert ovis._processor is None
    assert ovis._loaded_engine is None
    assert ovis._engine == "gguf"
    assert ovis._model_name == "Ovis-gguf"


def test_config_unchanged_keeps_model(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(ovis, "_engine", "mlx-vlm", raising=False)
    monkeypatch.setattr(ovis, "_model_name", "OvisOCR2-0.8B-MLX-4bit",
                        raising=False)
    monkeypatch.setattr(ovis, "_model_path", "", raising=False)
    monkeypatch.setattr(ovis, "_model", sentinel, raising=False)
    monkeypatch.setattr(ovis, "_model_load_error", None, raising=False)
    monkeypatch.setattr(ovis, "_loaded_engine", "mlx-vlm", raising=False)
    ovis.set_ovis_model_config("mlx-vlm", "OvisOCR2-0.8B-MLX-4bit", "")
    assert ovis._model is sentinel


# ---------------------------------------------------------------------------
# 推理分发
# ---------------------------------------------------------------------------

class _FakeGgufModel:
    def __init__(self):
        self.messages = None

    def create_chat_completion(self, messages, max_tokens, temperature,
                               stream=False):
        self.messages = messages
        assert stream is False
        return {"choices": [{"message": {"content": "<table><tr><td>A</td>"
                                          "<td>B</td></tr></table>"}}]}


def test_vlm_ocr_gguf_dispatch_and_clean(tmp_path, monkeypatch):
    fake = _FakeGgufModel()
    monkeypatch.setattr(ovis, "_engine", "gguf")
    monkeypatch.setattr(ovis, "_get_model", lambda: (fake, None))
    img = _mk(tmp_path, "img", ["page.png"]) / "page.png"
    out = ovis._vlm_ocr(str(img))
    # HTML 表格被 _clean_vlm_output 转成 Markdown 管道表格
    assert "| A | B |" in out
    msg = fake.messages[0]
    assert [c["type"] for c in msg["content"]] == ["text", "image_url"]
    assert msg["content"][1]["image_url"]["url"] == str(img)


def test_vlm_ocr_mlx_dispatch(monkeypatch):
    captured = {}

    def fake_generate(model, processor, prompt, image=None, max_tokens=0,
                      verbose=False):
        captured.update(prompt=prompt, image=image, max_tokens=max_tokens)

        class R:
            text = "hello"

        return R()

    monkeypatch.setitem(sys.modules, "mlx_vlm",
                        types.SimpleNamespace(generate=fake_generate))
    monkeypatch.setattr(ovis, "_engine", "mlx-vlm")
    monkeypatch.setattr(ovis, "_get_model", lambda: (object(), object()))
    assert ovis._vlm_ocr("/tmp/x.png") == "hello"
    assert captured["image"] == "/tmp/x.png"
    assert captured["max_tokens"] == 1024
    assert "<|vision_start|>" in captured["prompt"]


def test_vlm_ocr_onnx_missing_package_returns_none(monkeypatch):
    # onnxruntime_genai 缺失（sys.modules 置 None 强制 import 失败）→
    # _onnx_ocr 抛 RuntimeError → _vlm_ocr 捕获返回 None（回退链继续）。
    monkeypatch.setitem(sys.modules, "onnxruntime_genai", None)
    monkeypatch.setattr(ovis, "_engine", "onnx")
    monkeypatch.setattr(ovis, "_get_model",
                        lambda: (object(), (object(), object())))
    assert ovis._vlm_ocr("/tmp/x.png") is None


def test_gguf_load_missing_mmproj_gives_clear_error(isolated):
    isolated("g4", ["main.gguf"])
    d = ovis.resolve_model_dir("gguf")
    with pytest.raises(FileNotFoundError, match="mmproj"):
        ovis._load_gguf_model(d)
