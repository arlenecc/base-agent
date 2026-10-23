"""本地后端共用的 tokenizer 兼容层。

问题：``getattr(tok, "tokenizer", tok)`` 这类「取内层 tokenizer」的写法在不同
来源下含义完全不同：

* mlx_lm 加载出来的是 ``TokenizerWrapper``，它**不可调用**、也没有
  ``apply_chat_template`` / ``decode``——必须取内层的 HF tokenizer；
* transformers 的 ``AutoTokenizer.from_pretrained`` 直接返回可用的 fast
  tokenizer（可调用、有上述方法），而它的 ``.tokenizer`` 是底层 Rust
  ``tokenizers.Tokenizer``：**不可调用，也没有 apply_chat_template/decode**。

ONNX 后端用的是后者，旧实现照搬 MLX 的取值方式 → 实测直接崩：
``TypeError: 'tokenizers.Tokenizer' object is not callable``（reranker）
以及 ``AttributeError: apply_chat_template``（LLM）。

:func:`hf_tokenizer` 按「能力」而不是「来源」取值：优先返回本身具备所需方法的
对象，否则退回内层。
"""
from __future__ import annotations

from typing import Any, Optional


def _usable(obj: Any) -> bool:
    return (obj is not None and callable(obj)
            and hasattr(obj, "decode") and hasattr(obj, "apply_chat_template"))


def hf_tokenizer(tok: Any) -> Optional[Any]:
    """返回真正可用的 HF tokenizer（可调用 + decode + apply_chat_template）。

    取不到就原样返回（由调用方的报错暴露真实原因，不静默吞掉）。
    """
    if tok is None:
        return None
    if _usable(tok):
        return tok
    for attr in ("_tokenizer", "tokenizer"):
        inner = getattr(tok, attr, None)
        if _usable(inner):
            return inner
    return tok
