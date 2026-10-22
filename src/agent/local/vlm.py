"""本地 VLM（视觉语言模型）后端：MLX / GGUF。

与 ``LocalLLMClient`` 同源思路，但接口面向「图像 + 文本」输入：

    client.chat(prompt: str, images: List[str] = []) -> str

``images`` 是本地图片路径列表（或 base64 data-uri）；MLX 走 ``mlx_vlm``，
GGUF 走 llama-cpp-python 的 multimodal chat completion。应用目前没有的原生
图文工具，因此这里只提供能力封装 + 工厂入口，由上层在需要时调用。
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LocalVLMClient:
    """本地多模态模型客户端（MLX / GGUF）。"""

    def __init__(self, path: str, backend: str = "mlx",
                 max_tokens: int = 2048, temperature: float = 0.2,
                 dtype: str = "float16"):
        self._path = path
        self._backend = (backend or "mlx").strip().lower()
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)
        self._dtype = dtype
        self._model = None
        self._processor = None
        self._load_lock = threading.Lock()
        self._loaded = False

    def _ensure_model(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            if self._backend == "mlx":
                self._load_mlx()
            elif self._backend == "gguf":
                self._load_gguf()
            else:
                raise ValueError(f"未知 VLM 后端: {self._backend!r}")
            self._loaded = True

    def _load_mlx(self) -> None:
        if not os.path.isdir(self._path):
            raise FileNotFoundError(f"本地 VLM 模型目录不存在: {self._path}")
        try:
            from mlx_vlm import load
        except Exception as e:
            raise RuntimeError(
                "MLX VLM 后端不可用：未安装 mlx-vlm。\n"
                "安装：pip install mlx-vlm\n"
                f"原始错误: {type(e).__name__}: {e}"
            ) from e
        self._model, self._processor = load(self._path, trust_remote_code=True)

    def _load_gguf(self) -> None:
        gguf = self._path
        mmproj = None
        if os.path.isdir(gguf):
            for f in sorted(os.listdir(gguf)):
                low = f.lower()
                if not low.endswith(".gguf"):
                    continue
                if low.startswith("mmproj"):
                    mmproj = mmproj or os.path.join(gguf, f)
                else:
                    gguf = os.path.join(gguf, f)
                    break
        try:
            from llama_cpp import Llama
        except Exception as e:
            raise RuntimeError(
                "GGUF VLM 后端不可用：未安装 llama-cpp-python。\n"
                f"原始错误: {type(e).__name__}: {e}"
            ) from e
        # llama-cpp-python 没有 multimodal 构造参数（传入会 TypeError）——
        # 多模态必须挂 chat handler：≥0.3.30 用 MTMDChatHandler(mmproj)，
        # 更旧版本回退 Llava15ChatHandler（同一构造签名）。mmproj 缺失时
        # 退纯文本 Llama（调用方传图会在生成期报错，错误信息可定位）。
        kwargs: Dict[str, Any] = dict(model_path=gguf, n_ctx=4096,
                                      n_gpu_layers=-1, verbose=False)
        if mmproj:
            handler = None
            try:
                from llama_cpp.llama_chat_format import MTMDChatHandler
                handler = MTMDChatHandler(clip_model_path=mmproj, verbose=False)
            except Exception:
                try:
                    from llama_cpp.llama_chat_format import Llava15ChatHandler
                    handler = Llava15ChatHandler(clip_model_path=mmproj,
                                                 verbose=False)
                except Exception as e:
                    logger.warning("GGUF VLM chat handler 构造失败: %s", e)
            if handler is not None:
                kwargs["chat_handler"] = handler
        self._model = Llama(**kwargs)

    # ------------------------------------------------------------------
    def chat(self, prompt: str, images: Optional[List[str]] = None) -> str:
        """返回模型对（图像+文本）的回复字符串。"""
        self._ensure_model()
        images = images or []
        if self._backend == "mlx":
            return self._chat_mlx(prompt, images)
        return self._chat_gguf(prompt, images)

    def _chat_mlx(self, prompt: str, images: List[str]) -> str:
        from mlx_vlm import generate
        # mlx_vlm 的 generate 接受图像路径或 data-uri；data-uri 转临时文件交给库。
        out = generate(
            self._model, self._processor, prompt=prompt,
            image=images if images else None,
            max_tokens=self._max_tokens, temperature=self._temperature,
        )
        if isinstance(out, (list, tuple)):
            return "".join(str(x) for x in out)
        return str(out)

    def _chat_gguf(self, prompt: str, images: List[str]) -> str:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        # chat handler（MTMD/Llava）只识别 image_url 形态；url 可为
        # data-uri 或本地路径（load_image 两者都支持）。
        for img in images:
            content.append({"type": "image_url", "image_url": {"url": img}})
        messages = [{"role": "user", "content": content}]
        resp = self._model.create_chat_completion(
            messages=messages, temperature=self._temperature,
            max_tokens=self._max_tokens, stream=False,
        )
        return resp["choices"][0]["message"]["content"]

    def name(self) -> str:
        base = os.path.basename(self._path.rstrip("/")) or self._path
        return f"local-vlm:{self._backend}:{base}"

    def close(self) -> None:
        self._model = None
        self._loaded = False
