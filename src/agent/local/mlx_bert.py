"""MLX 实现：BERT encoder（bge-small-zh-v1.5 等标准 BERT 架构 embedding 模型）。

为什么需要自己写：仓库里的 MLX 版 BERT 权重是 **mlx-lm 格式**
（``encoder.layers.N.attention.query_proj`` / ``linear1`` / ``ln1`` / ``pooler``
这种键名），而 ``mlx-embeddings`` 期望的是 HF 风格键名（``self_attn.q_proj``
等），直接 ``mlx_embeddings.load`` 会报 ``Received N parameters not in model``。
因此这里按 transformers 的 ``BertModel`` 参考实现移植一份 MLX 版。

与 ``mlx_nomic``（RoPE + SwiGLU）不同，标准 BERT 的特点：
  - 绝对位置编码（``embeddings.position_embeddings``，无 RoPE）；
  - MLP 是 GELU：``linear2(gelu(linear1(x)))``（非 SwiGLU）；
  - 双向注意力（bidirectional），无 causal mask；
  - embedding 出口有一个 ``embeddings.LayerNorm``。

权重键名（mlx-lm 格式）：
    embeddings.word_embeddings.weight
    embeddings.position_embeddings.weight
    embeddings.token_type_embeddings.weight
    embeddings.norm.{weight,bias}
    encoder.layers.N.attention.{query,key,value,out}_proj.{weight,bias}
    encoder.layers.N.linear1.{weight,bias}
    encoder.layers.N.linear2.{weight,bias}
    encoder.layers.N.ln1.{weight,bias}
    encoder.layers.N.ln2.{weight,bias}
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# _require_mlx() 首次调用时缓存 mlx.nn 句柄，供前向（__call__）里用 gelu 等。
_NN = None


def _require_mlx():
    """惰性 import mlx，缺库时给出可操作的错误信息。"""
    global _NN
    try:
        import mlx.core as mx  # noqa: F401
        import mlx.nn as nn  # noqa: F401
        _NN = nn
        return mx, nn
    except Exception as e:  # pragma: no cover - 取决于运行环境
        raise RuntimeError(
            "MLX 后端不可用：未安装 mlx（Apple Silicon 专属）。\n"
            "安装：pip install mlx\n"
            f"原始错误: {type(e).__name__}: {e}"
        ) from e


@dataclass
class BertConfig:
    vocab_size: int = 21128
    hidden_size: int = 512
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0

    @classmethod
    def from_json(cls, path: str) -> "BertConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)


class _Attention:
    """标准多头注意力（带 bias）。"""

    def __init__(self, config: BertConfig):
        mx, nn = _require_mlx()
        h = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = h // config.num_attention_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(h, h, bias=True)
        self.k = nn.Linear(h, h, bias=True)
        self.v = nn.Linear(h, h, bias=True)
        self.o = nn.Linear(h, h, bias=True)

    def __call__(self, x, mask=None):
        mx, _ = _require_mlx()
        b, t, _ = x.shape
        q = self.q(x).reshape(b, t, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(b, t, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(b, t, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:  # [B, T] → 加性 mask
            scores = scores + (1.0 - mask[:, None, None, :]) * -1e9
        probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        ctx = (probs @ v).transpose(0, 2, 1, 3).reshape(b, t, self.q.weight.shape[0])
        return self.o(ctx)


class _Layer:
    def __init__(self, config: BertConfig):
        mx, nn = _require_mlx()
        h = config.hidden_size
        self.attn = _Attention(config)
        self.mlp_fc1 = nn.Linear(h, config.intermediate_size, bias=True)
        self.mlp_fc2 = nn.Linear(config.intermediate_size, h, bias=True)
        self.ln1 = nn.LayerNorm(h, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(h, eps=config.layer_norm_eps)

    def __call__(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp_fc2(_NN.gelu(self.mlp_fc1(self.ln2(x))))
        return x


class BertModel:
    """MLX 版 BERT encoder。前向返回最后一层 hidden states [B, T, H]。"""

    def __init__(self, config: BertConfig):
        mx, nn = _require_mlx()
        h = config.hidden_size
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, h)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, h)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, h)
        self.embed_norm = nn.LayerNorm(h, eps=config.layer_norm_eps)
        self.layers: List[_Layer] = [_Layer(config) for _ in range(config.num_hidden_layers)]

    def load_weights(self, weights: Dict[str, Any]) -> None:
        cfg = self.config

        # 仿 mlx_nomic：对子模块直接设 .weight / .bias（MLX 的 nn.Module 用属性
        # 持有 mx.array 参数，赋值即可注册；不要用带点的 setattr 名——setattr 不
        # 分层遍历）。
        def _set(submodule, key: str, *, bias_key: str = None) -> None:
            arr = weights.get(key)
            if arr is None:
                logger.warning("bert-mlx: 权重缺失 %s", key)
                return
            submodule.weight = arr
            if bias_key is not None and bias_key in weights:
                submodule.bias = weights[bias_key]

        _set(self.word_embeddings, "embeddings.word_embeddings.weight")
        _set(self.token_type_embeddings, "embeddings.token_type_embeddings.weight")
        _set(self.position_embeddings, "embeddings.position_embeddings.weight")
        _set(self.embed_norm, "embeddings.norm.weight", bias_key="embeddings.norm.bias")

        for i, layer in enumerate(self.layers):
            b = f"encoder.layers.{i}."
            _set(layer.attn.q, b + "attention.query_proj.weight",
                 bias_key=b + "attention.query_proj.bias")
            _set(layer.attn.k, b + "attention.key_proj.weight",
                 bias_key=b + "attention.key_proj.bias")
            _set(layer.attn.v, b + "attention.value_proj.weight",
                 bias_key=b + "attention.value_proj.bias")
            _set(layer.attn.o, b + "attention.out_proj.weight",
                 bias_key=b + "attention.out_proj.bias")
            _set(layer.mlp_fc1, b + "linear1.weight", bias_key=b + "linear1.bias")
            _set(layer.mlp_fc2, b + "linear2.weight", bias_key=b + "linear2.bias")
            _set(layer.ln1, b + "ln1.weight", bias_key=b + "ln1.bias")
            _set(layer.ln2, b + "ln2.weight", bias_key=b + "ln2.bias")

    def __call__(self, input_ids, attention_mask=None):
        mx, _ = _require_mlx()
        t = input_ids.shape[1]
        pos = mx.arange(t, dtype=mx.int32)
        h = (self.word_embeddings(input_ids)
             + self.token_type_embeddings(mx.zeros_like(input_ids))
             + self.position_embeddings(pos))
        h = self.embed_norm(h)
        if attention_mask is None:
            mask = None
        else:
            mask = attention_mask.astype(mx.float32)
        for layer in self.layers:
            h = layer(h, mask)
        return h


def load_bert(path: str, dtype: str = "float32"):
    """从 MLX 模型目录加载 BERT encoder。"""
    mx, _ = _require_mlx()
    cfg_path = os.path.join(path, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"模型目录缺少 config.json: {path}")
    config = BertConfig.from_json(cfg_path)

    st_files = [f for f in os.listdir(path) if f.endswith(".safetensors")]
    if not st_files:
        raise FileNotFoundError(f"模型目录缺少 *.safetensors: {path}")
    weights: Dict[str, Any] = {}
    for f in sorted(st_files):
        weights.update(mx.load(os.path.join(path, f)))
    # 标准 BERT 是稠密 fp32（非 MLX 量化），无需反量化；直接统一 dtype。
    target = {"float32": mx.float32, "float16": mx.float16,
              "bfloat16": mx.bfloat16}.get(dtype, mx.float32)
    weights = {k: v.astype(target) for k, v in weights.items()}

    model = BertModel(config)
    model.load_weights(weights)
    mx.eval(model)
    logger.info("bert-mlx: 装载 %d 层 (hidden=%d heads=%d)",
                config.num_hidden_layers, config.hidden_size, config.num_attention_heads)
    return model, config


def mean_pool(hidden, attention_mask):
    """按 attention_mask 的均值池化（sentence-transformers 语义）。"""
    mx, _ = _require_mlx()
    m = attention_mask[:, :, None].astype(hidden.dtype)
    summed = (hidden * m).sum(axis=1)
    counts = mx.clip(m.sum(axis=1), a_min=1e-9, a_max=None)
    return summed / counts


def l2_normalize(x):
    mx, _ = _require_mlx()
    norm = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True))
    return x / mx.clip(norm, a_min=1e-12, a_max=None)
