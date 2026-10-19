"""MLX 实现：NomicBert encoder（nomic-embed-text-v1.5 系列）。

为什么需要自己写：``mlx-embeddings`` 的模型注册表不认 ``nomic_bert`` 这个
``model_type``（0.1.0 报 "Model type nomic_bert not supported."），而仓库里
的 MLX 版权重又**不带** modeling 代码（config 的 ``auto_map`` 指向 HF 远端仓库，
离线环境不可用）。因此这里按 transformers 的参考实现
（``transformers.models.nomic_bert.modeling_nomic_bert``）移植一份 MLX 版。

权重命名沿用 nomic 原生的紧凑布局（与 HF 不同，需映射）：

    embeddings.word_embeddings.weight        HF: embeddings.word_embeddings.weight
    embeddings.token_type_embeddings.weight  同
    embeddings.norm.{weight,bias}            HF: embeddings.LayerNorm.{weight,bias}
    encoder.layers.N.attn.Wqkv.weight        HF: self_attn.{q,k,v}_proj.weight（融合）
    encoder.layers.N.attn.out_proj.weight    HF: self_attn.o_proj.weight
    encoder.layers.N.mlp.fc11.weight         HF: mlp.gate_proj.weight
    encoder.layers.N.mlp.fc12.weight         HF: mlp.up_proj.weight
    encoder.layers.N.mlp.fc2.weight          HF: mlp.down_proj.weight
    encoder.layers.N.norm1.*                 HF: post_attention_layernorm.*
    encoder.layers.N.norm2.*                 HF: post_mlp_layernorm.*

架构要点（与参考实现逐条对应）：
  - 无 position embedding，用 RoPE（base=rope_theta，dim=head_dim）
  - 注意力 q/k/v/o 全无 bias；scale = head_dim ** -0.5
  - MLP 是 SwiGLU：down(silu(gate(x)) * up(x))
  - 层结构：x = norm1(x + attn(x)); x = norm2(x + mlp(x))
  - 双向注意力（非 causal），mask 用加性 -10000
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class NomicBertConfig:
    vocab_size: int = 30528
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    layer_norm_eps: float = 1e-12
    type_vocab_size: int = 2
    max_position_embeddings: int = 2048
    pad_token_id: int = 0
    head_dim: int = 64
    rope_theta: float = 1000.0

    @classmethod
    def from_json(cls, path: str) -> "NomicBertConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rope = data.get("rope_parameters") or {}
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in known}
        if "rope_theta" in rope and "rope_theta" not in kwargs:
            kwargs["rope_theta"] = float(rope["rope_theta"])
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# MLX 模块
# ---------------------------------------------------------------------------


def _require_mlx():
    """惰性 import mlx，缺库时给出可操作的错误信息。"""
    try:
        import mlx.core as mx  # noqa: F401
        import mlx.nn as nn  # noqa: F401
        return mx, nn
    except Exception as e:  # pragma: no cover - 取决于运行环境
        raise RuntimeError(
            "MLX 后端不可用：未安装 mlx（Apple Silicon 专属）。\n"
            "安装：pip install mlx\n"
            "或在配置里把 embedding/reranker/llm 的后端切成 onnx / gguf。\n"
            f"原始错误: {type(e).__name__}: {e}"
        ) from e


class _RotaryEmbedding:
    """RoPE：inv_freq = 1 / base**(arange(0, dim, 2)/dim)，与参考实现一致。"""

    def __init__(self, dim: int, base: float):
        self.dim = int(dim)
        self.base = float(base)
        inv_freq = 1.0 / (base ** (mx_arange(0, dim, 2).astype(mx_float32()) / dim))
        self.inv_freq = inv_freq

    def __call__(self, seq_len: int):
        mx, _ = _require_mlx()
        pos = mx.arange(seq_len, dtype=mx.float32)
        freqs = pos[:, None] * self.inv_freq[None, :]
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb), mx.sin(emb)


def mx_arange(*args, **kwargs):
    mx, _ = _require_mlx()
    return mx.arange(*args, **kwargs)


def mx_float32():
    mx, _ = _require_mlx()
    return mx.float32


class _Attention:
    """q/k/v 由融合的 Wqkv 一次算出，再按输出维切三段（顺序 q,k,v）。"""

    def __init__(self, config: NomicBertConfig):
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.scaling = self.head_dim ** -0.5
        self.wqkv = None      # nn.Linear(hidden, 3*hidden, bias=False)
        self.out_proj = None  # nn.Linear(hidden, hidden, bias=False)

    def __call__(self, x, cos, sin, mask=None):
        mx, nn = _require_mlx()
        bsz, seq_len, _ = x.shape
        qkv = self.wqkv(x)  # [B, T, 3*H]
        q, k, v = mx.split(qkv, 3, axis=-1)

        def _shape(t):
            return t.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = _shape(q)
        k = _shape(k)
        v = _shape(v)

        # RoPE：cos/sin 形状 [T, dim] → 广播到 [B, H, T, dim]
        cos_b = cos[None, None, :, :].astype(q.dtype)
        sin_b = sin[None, None, :, :].astype(q.dtype)

        def _rotate_half(t):
            half = t.shape[-1] // 2
            return mx.concatenate([-t[..., half:], t[..., :half]], axis=-1)

        q = q * cos_b + _rotate_half(q) * sin_b
        k = k * cos_b + _rotate_half(k) * sin_b

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scaling
        if mask is not None:
            scores = scores + mask
        probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        ctx = probs @ v
        ctx = ctx.transpose(0, 2, 1, 3).reshape(bsz, seq_len, self.hidden_size)
        return self.out_proj(ctx)


class _MLP:
    """SwiGLU：``down(fc11(x) * silu(fc12(x)))``。

    注意 gate/up 的归属与常见实现**相反**：激活作用在 ``fc12`` 上，``fc11``
    直通。这不是笔误——fastembed 的 nomic ONNX 图里两个上投影共用同一个
    中间张量名（``MatMul_output_0``），``Sigmoid`` 消费的是后写入的那个
    （fc12），逐层对拍确认 ``silu(fc12) * fc11`` 与 ONNX 余弦 0.996，而
    ``silu(fc11) * fc12`` 只有 0.22。按常见顺序写会让 12 层误差累积成
    「不同文本嵌入完全退化」（ pairwise 余弦 1.0）。
    """

    def __init__(self):
        self.fc11 = None  # up（直通）
        self.fc12 = None  # gate（过 silu）
        self.fc2 = None   # down

    def __call__(self, x):
        mx, nn = _require_mlx()
        return self.fc2(self.fc11(x) * nn.silu(self.fc12(x)))


class _Layer:
    def __init__(self, config: NomicBertConfig):
        self.attn = _Attention(config)
        self.mlp = _MLP()
        self.norm1 = None  # post-attention LayerNorm
        self.norm2 = None  # post-MLP LayerNorm

    def __call__(self, x, cos, sin, mask=None):
        mx, _ = _require_mlx()
        h = self.norm1(x + self.attn(x, cos, sin, mask))
        h = self.norm2(h + self.mlp(h))
        return h


class NomicBertModel:
    """MLX 版 NomicBert encoder。前向返回最后一层 hidden states [B, T, H]。

    不继承 ``nn.Module``：参数由 ``load_nomic_bert`` 用 MXU 兼容的
    ``nn.Linear``/``nn.LayerNorm`` 实例装配后整体 eval，避免手写参数树时
    漏掉 quantized 权重的反量化路径（MXFP8/affine 量化权重由 MLX 自动处理）。
    """

    def __init__(self, config: NomicBertConfig):
        mx, nn = _require_mlx()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.rotary = _RotaryEmbedding(config.head_dim, config.rope_theta)
        self.layers: List[_Layer] = []
        for _ in range(config.num_hidden_layers):
            layer = _Layer(config)
            layer.attn.wqkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
            layer.attn.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            layer.mlp.fc11 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            layer.mlp.fc12 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            layer.mlp.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
            layer.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            layer.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.layers.append(layer)

    # ---- 参数装载 -----------------------------------------------------
    # MLX 的 nn.Module 用属性访问参数，这里按「参考实现的 HF 名」写回，再用
    # mx.eval 触发真正的量化权重反量化（lazy load 的 safetensors 是 mmap）。
    def load_weights(self, weights: Dict[str, Any]) -> None:
        mx, _ = _require_mlx()
        cfg = self.config

        def _set(obj, attr: str, key: str, optional: bool = False) -> bool:
            arr = weights.get(key)
            if arr is None:
                if not optional:
                    logger.warning("nomic-mlx: 权重缺失 %s", key)
                return False
            target = getattr(obj, attr)
            # nn.Linear / nn.Embedding / nn.LayerNorm 的 weight 属性可直接赋值
            target.weight = arr if hasattr(target, "weight") else arr
            if hasattr(target, "bias") and f"{key[:-len('.weight')]}.bias" in weights:
                target.bias = weights[f"{key[:-len('.weight')]}.bias"]
            return True

        _set(self, "word_embeddings", "embeddings.word_embeddings.weight")
        _set(self, "token_type_embeddings", "embeddings.token_type_embeddings.weight")
        _set(self, "norm", "embeddings.norm.weight")
        if "embeddings.norm.bias" in weights:
            self.norm.bias = weights["embeddings.norm.bias"]

        loaded = 0
        for i, layer in enumerate(self.layers):
            p = f"encoder.layers.{i}."
            layer.attn.wqkv.weight = weights[p + "attn.Wqkv.weight"]
            layer.attn.out_proj.weight = weights[p + "attn.out_proj.weight"]
            layer.mlp.fc11.weight = weights[p + "mlp.fc11.weight"]
            layer.mlp.fc12.weight = weights[p + "mlp.fc12.weight"]
            layer.mlp.fc2.weight = weights[p + "mlp.fc2.weight"]
            layer.norm1.weight = weights[p + "norm1.weight"]
            layer.norm1.bias = weights[p + "norm1.bias"]
            layer.norm2.weight = weights[p + "norm2.weight"]
            layer.norm2.bias = weights[p + "norm2.bias"]
            loaded += 1
        logger.info("nomic-mlx: 装载 %d 层 (hidden=%d heads=%d)",
                    loaded, cfg.hidden_size, cfg.num_attention_heads)

    def __call__(self, input_ids, attention_mask=None):
        mx, _ = _require_mlx()
        seq_len = input_ids.shape[1]
        h = self.word_embeddings(input_ids)
        if attention_mask is None:
            tok_types = mx.zeros(input_ids.shape, dtype=mx.int32)
        else:
            tok_types = mx.zeros(input_ids.shape, dtype=mx.int32)
        h = h + self.token_type_embeddings(tok_types)
        h = self.norm(h)

        # 加性 mask：padding 位置 -10000（参考实现的 create_bidirectional_mask）
        if attention_mask is None:
            mask = None
        else:
            m = attention_mask[:, None, None, :].astype(mx.float32)
            mask = (1.0 - m) * -10000.0

        cos, sin = self.rotary(seq_len)
        for layer in self.layers:
            h = layer(h, cos, sin, mask)
        return h


def load_nomic_bert(path: str, dtype: str = "float32"):
    """从 MLX 模型目录加载 NomicBert。

    ``dtype`` 默认 float32：mean-pooling 后再做 L2 归一化，fp16 累加误差会
    让余弦相似度漂移（量化 embedding 模型对精度敏感）。需要省内存可传
    "float16"/"bfloat16"。
    """
    mx, _ = _require_mlx()
    cfg_path = os.path.join(path, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"模型目录缺少 config.json: {path}")
    config = NomicBertConfig.from_json(cfg_path)

    st_files = [f for f in os.listdir(path) if f.endswith(".safetensors")]
    if not st_files:
        raise FileNotFoundError(f"模型目录缺少 *.safetensors: {path}")
    weights: Dict[str, Any] = {}
    for f in sorted(st_files):
        weights.update(mx.load(os.path.join(path, f)))

    target = {"float32": mx.float32, "float16": mx.float16,
              "bfloat16": mx.bfloat16}.get(dtype, mx.float32)
    weights = {k: v.astype(target) for k, v in weights.items()}

    model = NomicBertModel(config)
    model.load_weights(weights)
    # 触发 lazy 权重的真正物化（mmap 的 safetensors 直到首次计算才读盘）
    mx.eval(model)
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
