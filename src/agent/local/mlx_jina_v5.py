"""jina-embeddings-v5-text-nano（EuroBERT-210M）本地 MLX 加载器。

与 ``mlx_jina.py``（jina-v3）同模式的 vendored 实现：模型目录自带 MLX 权重，
但 ``mlx-embeddings`` / ``mlx-lm`` 都不认 ``eurobert`` 架构，这里内置模型代码。

与 v3 的关键差异：
  * 双向编码器（EuroBERT）：RoPE + SwiGLU + RMSNorm，**无因果掩码**；
  * **last-token 池化**（v3 是 mean-pool）+ L2 归一化；
  * 任务适配器为 4 个独立 LoRA（retrieval/text-matching/clustering/
    classification），官方 MLX 发行版通过原位矩阵加法切换——**量化权重
    （packed uint32）无法再做原位加法**，因此 int4/int8 发行版已把
    retrieval LoRA 预合并进权重（query/passage 共用同一 LoRA，仅文本前缀
    ``Query: `` / ``Document: `` 区分，前缀由 embedding.py 的前缀族表叠加）；
  * fp16 发行版（带 adapters/）加载时会自动合并 retrieval LoRA，行为对齐。

权重加载：config.json 带 ``quantization``（bits/group_size）时先构建量化
结构再加载 packed 权重；词嵌入表随 Linear 一并量化（mlx nn.quantize 默认）。
输出统一转 float32（基座权重实为 bfloat16，numpy 无法直接消费 bf16 缓冲）。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


@dataclass
class ModelArgs:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    max_position_embeddings: int
    head_dim: int
    tie_word_embeddings: bool
    rope_theta: float = 1000000.0
    hidden_act: str = "silu"
    attention_bias: bool = False


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=args.attention_bias)
        self.rotary_emb = nn.RoPE(head_dim, traditional=False, base=args.rope_theta)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, L, _ = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        queries = self.rotary_emb(queries)
        keys = self.rotary_emb(keys)
        if self.n_kv_heads != self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            keys = mx.repeat(keys, n_rep, axis=1)
            values = mx.repeat(values, n_rep, axis=1)
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size,
                       bias=args.attention_bias)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class EuroBertModel(nn.Module):
    """双向编码器主干（无因果掩码，仅 padding 掩码）。"""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args=args)
                       for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs: mx.array, mask: Optional[mx.array] = None):
        h = self.embed_tokens(inputs)
        for layer in self.layers:
            h = layer(h, mask)
        return self.norm(h)


class JinaV5Model(nn.Module):
    """jina-embeddings-v5-text-nano：last-token 池化 + L2 归一化。

    ``encode(input_ids, attention_mask, truncate_dim=None, task=None)`` 与
    ``mlx_jina.JinaV3Model.encode`` 同接口（``task`` 仅为兼容占位——量化版
    已预合并 retrieval LoRA，此处不做运行时任务切换）。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        keys = {f.name for f in dataclass_fields(ModelArgs)}
        args = ModelArgs(**{k: v for k, v in config.items() if k in keys})
        self.model = EuroBertModel(args)
        self.config = config

    def __call__(self, input_ids: mx.array,
                 attention_mask: Optional[mx.array] = None) -> mx.array:
        if attention_mask is not None:
            padding_mask = mx.where(attention_mask == 0, -1e9, 0.0)
            mask = mx.expand_dims(mx.expand_dims(padding_mask, 1), 1)
            h_dtype = self.model.embed_tokens.weight.dtype
            if mask.dtype != h_dtype:
                mask = mask.astype(h_dtype)
        else:
            mask = None
        hidden = self.model(input_ids, mask)
        if attention_mask is not None:
            seq_len = mx.sum(attention_mask, axis=1) - 1
            batch_idx = mx.arange(hidden.shape[0])
            emb = hidden[batch_idx, seq_len]
        else:
            emb = hidden[:, -1, :]
        return emb / mx.linalg.norm(emb, axis=1, keepdims=True)

    def encode(self, input_ids: mx.array, attention_mask: Optional[mx.array] = None,
               truncate_dim: Optional[int] = None,
               task: Optional[str] = None) -> mx.array:
        emb = self(input_ids, attention_mask).astype(mx.float32)
        if truncate_dim is not None:      # MRL：截维后重新 L2 归一化
            emb = emb[:, :truncate_dim]
            emb = emb / mx.linalg.norm(emb, axis=1, keepdims=True)
        return emb


def dataclass_fields(cls):
    from dataclasses import fields
    return fields(cls)


def _merge_retrieval_lora(model: JinaV5Model, model_dir: str) -> None:
    """fp16 多任务发行版：把 retrieval LoRA 预合并进权重（一次性）。"""
    adapter_dir = os.path.join(model_dir, "adapters", "retrieval")
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    w_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not (os.path.isfile(cfg_path) and os.path.isfile(w_path)):
        return
    with open(cfg_path) as f:
        acfg = json.load(f)
    scale = acfg["lora_alpha"] / acfg["r"]
    w = mx.load(w_path)
    w = {k: v.astype(mx.float16) if v.dtype == mx.float32 else v
         for k, v in w.items()}
    params = dict(tree_flatten(model.parameters()))
    n = 0
    for key, A in w.items():
        if "lora_A" not in key:
            continue
        base_key = ("model." + key.replace("base_model.model.", "")
                    .replace(".lora_A.weight", ".weight"))
        if base_key not in params:
            continue
        B = w[key.replace("lora_A", "lora_B")]
        params[base_key] = params[base_key] + scale * (B @ A)
        n += 1
    model.load_weights(list(params.items()), strict=False)
    mx.eval(model.parameters())


def load_model(path: str) -> Tuple[JinaV5Model, Dict[str, Any]]:
    """加载 v5-nano（fp16 或 int4/int8 量化），返回 (model, config)。"""
    with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model = JinaV5Model(cfg)
    if (q := cfg.get("quantization")) is not None:
        nn.quantize(model, group_size=q["group_size"], bits=q["bits"])
    weights = mx.load(os.path.join(path, "model.safetensors"))
    model.load_weights(list(weights.items()))
    del weights
    if cfg.get("quantization") is None:
        _merge_retrieval_lora(model, path)
    mx.eval(model.parameters())
    return model, cfg
