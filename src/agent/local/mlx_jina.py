"""MLX implementation of jinaai/jina-embeddings-v3 (XLM-RoBERTa + RoPE + task LoRA).

Vendored from the model directory's ``example/jina_v3_mlx.py`` so the app can load
the in-repo ``models/jina-embeddings-v3-mlx-int4`` weights without depending on a
volatile absolute path. Architecture faithfully mirrors the official
flash-attention implementation:

    word_emb + token_type_emb[0] -> emb_ln
    -> 24 x postnorm block: norm1(x + MHA(x)), norm2(x + MLP(x))
    -> mean pooling over non-pad tokens -> L2 normalise

Task LoRA (rank 4, alpha 1) is **merged into the base weights at conversion
time** for the non-unified export, or selected at runtime for the unified export
via the ``task=`` argument (``retrieval.query`` / ``retrieval.passage`` / ...).

See also ``example/example_mlx.py`` in the model directory for the reference
usage.
"""

import json
import math
import os
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

TASKS = ["retrieval.query", "retrieval.passage", "separation", "classification", "text-matching"]


@dataclass
class JinaV3Args:
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    layer_norm_eps: float = 1e-5
    vocab_size: int = 250002
    type_vocab_size: int = 1
    rotary_emb_base: float = 20000.0
    lora_rank: int = 4
    lora_alpha: int = 1
    model_type: str = "jina-v3-mlx"

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @property
    def lora_scaling(self):
        return self.lora_alpha / self.lora_rank


def gelu_exact(x):
    """Exact GELU (erf), matching torch F.gelu(approximate='none')."""
    return x * 0.5 * (1.0 + mx.erf(x * (1.0 / math.sqrt(2.0))))


def _loraparam(lora, name, kind, shape):
    """Return a LoRA adapter tensor, or a zero placeholder when none is supplied."""
    if lora is not None and name in lora and kind in lora[name]:
        return lora[name][kind]
    return mx.zeros(shape)


class RotaryEmbedding(nn.Module):
    """GPT-NeoX style (non-interleaved, half-rotation) RoPE applied to Q and K only."""

    def __init__(self, dim: int, base: float = 20000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self._inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))

    def __call__(self, qkv):
        # qkv: (B, S, 3, H, D)
        seq_len = qkv.shape[1]
        half = self.dim // 2
        t = mx.arange(seq_len, dtype=mx.float32)
        freqs = t[:, None] * self._inv_freq[None, :]  # (S, D/2)
        # The official implementation builds cos/sin of shape (S, D) by concatenating the
        # half-dim vectors twice, so each half is identical: (S, D/2) broadcasts identically.
        cos = mx.cos(freqs)[None, :, None, :]  # (1, S, 1, D/2)
        sin = mx.sin(freqs)[None, :, None, :]

        def rotate(x):
            x1 = x[..., :half]
            x2 = x[..., half:]
            return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

        q = rotate(qkv[:, :, 0, :, :])
        k = rotate(qkv[:, :, 1, :, :])
        v = qkv[:, :, 2, :, :]
        return mx.stack([q, k, v], axis=2)


class MHA(nn.Module):
    def __init__(self, args: JinaV3Args, unified: bool = False, lora: dict = None):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.scaling = args.lora_scaling
        self.Wqkv = nn.Linear(args.hidden_size, 3 * args.hidden_size, bias=True)
        self.out_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=True)
        self.rotary = RotaryEmbedding(self.head_dim, args.rotary_emb_base)
        if unified:
            r, h, three = args.lora_rank, args.hidden_size, 3 * args.hidden_size
            self.lora_Wqkv_A = _loraparam(lora, "Wqkv", "lora_A", (5, r, h)) if lora else mx.zeros((5, r, h))
            self.lora_Wqkv_B = _loraparam(lora, "Wqkv", "lora_B", (5, three, r)) if lora else mx.zeros((5, three, r))
            self.lora_out_A = _loraparam(lora, "out_proj", "lora_A", (5, r, h)) if lora else mx.zeros((5, r, h))
            self.lora_out_B = _loraparam(lora, "out_proj", "lora_B", (5, h, r)) if lora else mx.zeros((5, h, r))

    def __call__(self, x, task_id=None, add_mask=None):
        B, S, _ = x.shape
        H, D = self.num_heads, self.head_dim
        qkv = self.Wqkv(x)
        if task_id is not None:
            a, b = self.lora_Wqkv_A[task_id], self.lora_Wqkv_B[task_id]
            qkv = qkv + ((x @ a.T) @ b.T) * self.scaling
        qkv = qkv.reshape(B, S, 3, H, D)
        qkv = self.rotary(qkv)
        q = qkv[:, :, 0].transpose(0, 2, 1, 3)  # (B, H, S, D)
        k = qkv[:, :, 1].transpose(0, 2, 1, 3)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)
        scores = (q * self.scale) @ k.transpose(0, 1, 3, 2)  # (B, H, S, S)
        if add_mask is not None:
            scores = scores + add_mask
        attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        attn_out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, S, H * D)
        out = self.out_proj(attn_out)
        if task_id is not None:
            a, b = self.lora_out_A[task_id], self.lora_out_B[task_id]
            out = out + ((attn_out @ a.T) @ b.T) * self.scaling
        return out


class MLP(nn.Module):
    def __init__(self, args: JinaV3Args, unified: bool = False, lora: dict = None):
        super().__init__()
        self.scaling = args.lora_scaling
        self.fc1 = nn.Linear(args.hidden_size, args.intermediate_size, bias=True)
        self.fc2 = nn.Linear(args.intermediate_size, args.hidden_size, bias=True)
        if unified:
            r, h, inter = args.lora_rank, args.hidden_size, args.intermediate_size
            self.lora_fc1_A = _loraparam(lora, "fc1", "lora_A", (5, r, h)) if lora else mx.zeros((5, r, h))
            self.lora_fc1_B = _loraparam(lora, "fc1", "lora_B", (5, inter, r)) if lora else mx.zeros((5, inter, r))
            self.lora_fc2_A = _loraparam(lora, "fc2", "lora_A", (5, r, inter)) if lora else mx.zeros((5, r, inter))
            self.lora_fc2_B = _loraparam(lora, "fc2", "lora_B", (5, h, r)) if lora else mx.zeros((5, h, r))

    def __call__(self, x, task_id=None):
        h = self.fc1(x)
        if task_id is not None:
            a, b = self.lora_fc1_A[task_id], self.lora_fc1_B[task_id]
            h = h + ((x @ a.T) @ b.T) * self.scaling
        h = gelu_exact(h)
        out = self.fc2(h)
        if task_id is not None:
            a, b = self.lora_fc2_A[task_id], self.lora_fc2_B[task_id]
            out = out + ((h @ a.T) @ b.T) * self.scaling
        return out


class Block(nn.Module):
    def __init__(self, args: JinaV3Args, unified: bool = False, lora: dict = None):
        super().__init__()
        self.mixer = MHA(args, unified, lora["mixer"] if lora else None)
        self.mlp = MLP(args, unified, lora["mlp"] if lora else None)
        self.norm1 = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)
        self.norm2 = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)

    def __call__(self, x, task_id=None, add_mask=None):
        h = self.norm1(self.mixer(x, task_id, add_mask) + x)
        return self.norm2(self.mlp(h, task_id) + h)


class JinaV3Model(nn.Module):
    def __init__(self, args: JinaV3Args, unified: bool = False, lora: dict = None):
        super().__init__()
        self.args = args
        self.unified = unified
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.token_type_embeddings = nn.Embedding(args.type_vocab_size, args.hidden_size)
        self.emb_ln = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)
        self.layers = [
            Block(args, unified, lora[f"layers.{i}"] if lora else None)
            for i in range(args.num_hidden_layers)
        ]
        if unified:
            r, h, tv = args.lora_rank, args.hidden_size, args.type_vocab_size
            self.lora_emb_A = _loraparam(lora, "word_embeddings", "lora_A", (5, args.vocab_size, r)) if lora else mx.zeros((5, args.vocab_size, r))
            self.lora_emb_B = _loraparam(lora, "word_embeddings", "lora_B", (5, r, h)) if lora else mx.zeros((5, r, h))
            self.lora_tt_A = _loraparam(lora, "token_type_embeddings", "lora_A", (5, tv, r)) if lora else mx.zeros((5, tv, r))
            self.lora_tt_B = _loraparam(lora, "token_type_embeddings", "lora_B", (5, r, h)) if lora else mx.zeros((5, r, h))

    @staticmethod
    def make_additive_mask(attention_mask):
        """(B, S) with 1=keep -> (B, 1, 1, S) additive mask, matching the official -10000 bias."""
        if attention_mask is None:
            return None
        additive = (1 - attention_mask) * -10000.0
        return additive[:, None, None, :]

    def __call__(self, input_ids, attention_mask=None, task_id=None):
        h = self.word_embeddings(input_ids)
        if task_id is not None:
            a, b = self.lora_emb_A[task_id], self.lora_emb_B[task_id]
            e = a[input_ids.reshape(-1).astype(mx.int32)].reshape(*input_ids.shape, a.shape[-1])
            h = h + (e @ b) * self.args.lora_scaling
        # Go through the module call (not `.weight[0]`) so quantised embeddings dequantize.
        token_type = self.token_type_embeddings(mx.array([0], dtype=mx.int32))
        if task_id is not None:
            a, b = self.lora_tt_A[task_id], self.lora_tt_B[task_id]
            tt = a[mx.array([0], dtype=mx.int32)].reshape(1, 1, a.shape[-1])
            token_type = token_type + (tt @ b) * self.args.lora_scaling
        h = h + token_type.reshape(1, 1, -1)
        h = self.emb_ln(h)
        add_mask = self.make_additive_mask(attention_mask)
        for layer in self.layers:
            h = layer(h, task_id, add_mask)
        return h

    def encode(self, input_ids, attention_mask=None, truncate_dim=None, task=None):
        """Mean-pool over non-pad tokens, L2 normalise (identical to sentence-transformers).
        `task` selects the LoRA adapter at runtime (unified models only)."""
        task_id = TASKS.index(task) if (task and self.unified) else None
        h = self(input_ids, attention_mask, task_id)
        h = h.astype(mx.float32)
        mask = attention_mask.astype(mx.float32)[:, :, None]
        pooled = (h * mask).sum(axis=1) / mx.maximum(mask.sum(axis=1), 1e-9)
        if truncate_dim is not None:
            pooled = pooled[:, :truncate_dim]
        return pooled / mx.sqrt((pooled * pooled).sum(axis=-1, keepdims=True))


# ---------------------------------------------------------------------------
# Weight conversion
# ---------------------------------------------------------------------------

HF_KEY_MAP = {
    "roberta.embeddings.word_embeddings": "word_embeddings",
    "roberta.embeddings.token_type_embeddings": "token_type_embeddings",
    "roberta.emb_ln": "emb_ln",
}


def _layer_key(hf_key: str) -> str:
    """roberta.encoder.layers.3.mixer.Wqkv -> layers.3.mixer.Wqkv (top-level keys via map)."""
    if hf_key in HF_KEY_MAP:
        return HF_KEY_MAP[hf_key]
    if hf_key.startswith("roberta.encoder.layers."):
        return "layers." + hf_key[len("roberta.encoder.layers.") :]
    if hf_key.startswith("roberta."):
        return hf_key[len("roberta.") :]
    return hf_key


def load_hf_weights(model_dir: str):
    """Return (base_weights, lora_weights) as float32 mlx arrays keyed by mlx names."""
    from safetensors import safe_open

    path = os.path.join(model_dir, "model.safetensors")
    base, lora = {}, {}
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        for k in keys:
            if ".lora_A" in k or ".lora_B" in k:
                stem = k.replace(".parametrizations.weight.0.lora_A", "").replace(
                    ".parametrizations.weight.0.lora_B", ""
                )
                kind = "lora_A" if ".lora_A" in k else "lora_B"
                lora.setdefault(_layer_key(stem), {})[kind] = mx.array(f.get_tensor(k).float().numpy())
            elif k.endswith(".parametrizations.weight.original"):
                stem = k[: -len(".parametrizations.weight.original")]
                base[_layer_key(stem) + ".weight"] = mx.array(f.get_tensor(k).float().numpy())
            elif k.endswith(".bias") or k.endswith(".weight"):
                base[_layer_key(k)] = mx.array(f.get_tensor(k).float().numpy())
    return base, lora


def merge_lora(base: dict, lora: dict, task: str, args: JinaV3Args):
    """Merge the LoRA delta of `task` into the base weights (float32 math)."""
    if task is None:
        return base
    task_id = TASKS.index(task)
    scaling = args.lora_alpha / args.lora_rank
    merged = dict(base)

    for stem, ab in lora.items():
        key = stem + ".weight"
        if key not in merged:
            continue  # e.g. the pooler, which we never use
        W = merged[key]
        is_embedding = stem.endswith("word_embeddings") or stem.endswith("token_type_embeddings")
        A = ab["lora_A"][task_id]  # (rank, fan_in) for Linear, (fan_in, rank) for Embedding
        B = ab["lora_B"][task_id]  # (fan_out, rank) for Linear, (rank, fan_out) for Embedding
        delta = (A @ B) if is_embedding else (B @ A)
        merged[key] = W + scaling * delta
    return merged


def _nest_lora(lora, args):
    """Group the flat LoRA dict into the nesting JinaV3Model expects (per-layer + embeddings)."""
    out = {
        "word_embeddings": lora["word_embeddings"],
        "token_type_embeddings": lora["token_type_embeddings"],
    }
    for i in range(args.num_hidden_layers):
        pre = f"layers.{i}."
        out[f"layers.{i}"] = {
            "mixer": {
                "Wqkv": lora[pre + "mixer.Wqkv"],
                "out_proj": lora[pre + "mixer.out_proj"],
            },
            "mlp": {
                "fc1": lora[pre + "mlp.fc1"],
                "fc2": lora[pre + "mlp.fc2"],
            },
        }
    return out


def build_model(args: JinaV3Args, weights: dict, lora=None, dtype=mx.float16, unified=False):
    if unified and lora is not None:
        lora = _nest_lora(lora, args)
    model = JinaV3Model(args, unified=unified, lora=lora)
    model.load_weights(
        [(k, v.astype(dtype)) for k, v in weights.items() if k in _expected_keys(args, unified)]
    )
    return model


def load_model(path: str):
    """Load a converted MLX model (quantisation is restored from config.json)."""
    with open(os.path.join(path, "config.json")) as f:
        cfg = json.load(f)
    fields = set(JinaV3Args.__dataclass_fields__)
    args = JinaV3Args(**{k: v for k, v in cfg.items() if k in fields})
    model = JinaV3Model(args, unified=cfg.get("unified", False))
    weights = mx.load(os.path.join(path, "weights.safetensors"))

    if (q := cfg.get("quantization")) is not None:
        gs, bits = q["group_size"], q["bits"]
        mode = q.get("mode", "affine")
        mixed = q.get("mixed")
        patterns = mixed["patterns"] if mixed else []

        def is_linear(p, m):
            return (
                hasattr(m, "to_quantized")
                and "layers." in p
                and not p.endswith("norm1")
                and not p.endswith("norm2")
            )

        if mixed:
            nn.quantize(
                model,
                group_size=gs,
                bits=mixed["bits"],
                mode=mode,
                class_predicate=lambda p, m: is_linear(p, m) and any(x in p for x in patterns),
            )
        nn.quantize(
            model,
            group_size=gs,
            bits=bits,
            mode=mode,
            class_predicate=lambda p, m: is_linear(p, m) and not any(x in p for x in patterns),
        )
        nn.quantize(
            model,
            group_size=gs,
            bits=bits,
            mode=mode,
            class_predicate=lambda p, m: p == "word_embeddings",
        )
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    model.eval()
    return model, cfg


def _expected_keys(args: JinaV3Args, unified: bool = False):
    keys = ["word_embeddings.weight", "token_type_embeddings.weight", "emb_ln.weight", "emb_ln.bias"]
    for i in range(args.num_hidden_layers):
        p = f"layers.{i}."
        for sub in ["mixer.Wqkv", "mixer.out_proj", "mlp.fc1", "mlp.fc2", "norm1", "norm2"]:
            keys.append(p + sub + ".weight")
            keys.append(p + sub + ".bias")
    if unified:
        keys += ["lora_emb_A", "lora_emb_B", "lora_tt_A", "lora_tt_B"]
        for i in range(args.num_hidden_layers):
            p = f"layers.{i}."
            for sub in [
                "mixer.lora_Wqkv_A", "mixer.lora_Wqkv_B", "mixer.lora_out_A", "mixer.lora_out_B",
                "mlp.lora_fc1_A", "mlp.lora_fc1_B", "mlp.lora_fc2_A", "mlp.lora_fc2_B",
            ]:
                keys.append(p + sub)
    return set(keys)
