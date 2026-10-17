"""Configuration for the agent."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Runtime configuration. Persisted to ~/.base-agent/config.json by the UI."""

    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 32768
    top_p: float = 0.95
    min_p: float = 0.05
    top_k: int = 20
    repetition_penalty: float = 1.0
    workspace: str = field(default_factory=lambda: os.path.expanduser("~/base-agent-workspace"))
    # Default to <workspace>/knowledge_base so RAG tools are available out of
    # the box.  Users can override via the UI or config.json.
    knowledge_base: str = ""
    request_timeout: float = 120.0
    max_iterations: int = 15
    max_history: int = 50  # max messages retained in conversation history
    # Context window budget (tokens). When the estimated prompt size reaches
    # 90% of this value, the agent proactively summarizes older messages and
    # persists the summary to memory.md. Also used to recover from
    # context_length_exceeded errors returned by the LLM API.
    # Default matches max_tokens (32768) so the shrink threshold tracks the
    # model's actual context window.
    max_context_tokens: int = 32768
    # Shrink ratio: trigger proactive shrink when estimated context size
    # exceeds this fraction of max_context_tokens.
    context_shrink_ratio: float = 0.9

    # ---- 内核（三态自动机 + 五级压缩）----
    # 硬上限：整个 prompt（系统提示 + 历史 + 工具 schema）的目标预算。
    # 五级压缩的目标就是让它恒定落在这个数以内。
    kernel_max_context_tokens: int = 30_000
    # 保留「原文」（未折叠）的最近轮数。更早的轮次折叠成一行摘要（L3）。
    # 这是历史从 O(轮次) 变 O(1) 的旋钮：调大 → 上下文更准但更贵。
    kernel_raw_turns: int = 2
    # 每多少轮进入一次 CONSOLIDATE（不消耗 LLM 决策的压缩态）。
    kernel_consolidate_every: int = 4
    # 单个工具结果注入上下文的字符上限（L1 裁剪后的目标值）。
    kernel_max_tool_chars: int = 6_000
    # L3 折叠摘要区保留的最大摘要行数。
    kernel_max_digest_lines: int = 40

    # RAG settings
    rag_chunk_size: int = 500       # tokens per chunk
    rag_chunk_overlap: int = 50     # 10% of chunk_size
    rag_embedding_model: str = "nomic-ai/nomic-embed-text-v1.5-Q"
    # 重排序模型：**INT8 量化版**（~570MB）。fp32 全量是 2.27GB，4 倍体积
    # 4 倍内存，而 cross-encoder 只输出排序分数，INT8 的排序质量损失可忽略
    # （ONNX int8 加速在 CPU EP 上还更快）。与 deps.RERANK_MODEL_ID /
    # vector_store.DEFAULT_RERANK_MODEL 必须是同一个值——三处曾不一致：
    # deps 预下载并探测 int8，运行时却按 fp32 名加载，导致首次检索额外触发
    # 一次 2.27GB 下载（预下载的 int8 白下）。_migrate() 会把旧配置里的
    # fp32 名迁到 int8。
    rag_rerank_model: str = "BAAI/bge-reranker-v2-m3-int8"
    rag_rerank_enabled: bool = True
    rag_auto_ingest: bool = True
    # OCR（图片型 PDF）：检测到 Apple M 系列且 onnxruntime >= 1.31.0 时，
    # 让 RapidOCR 走 CoreML EP（CPU+GPU+ANE）。关掉则回退 onnxruntime 默认 EP。
    rag_ocr_coreml: bool = True
    # OCR 前的图像归一化与增强（尺寸归一化 + 灰度 CLAHE + 锐化）。
    # 关掉则把渲染出的原始位图直接送进引擎（旧行为）。
    rag_ocr_preprocess: bool = True

    # Browser endpoint for webexec_js (Playwright-like bridge). Optional.
    browser_endpoint: str = ""

    # code_run / shell_run 的确认策略。默认 False = 风险分级（tools/risk.py）：
    # 只读与工作区内操作直接执行，sudo/系统控制/写工作区外路径等才弹确认。
    # True = 恢复旧的逐次确认行为。
    always_confirm_code: bool = False

    # MCP servers – list of {name, command, args?, env?} dicts.
    # Example:
    #   [{"name": "filesystem", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]}]
    mcp_servers: list = field(default_factory=list)

    # 数值字段的类型，用于把 config.json 里的字符串/浮点值强制转换回正确类型。
    # 手工编辑过的 config.json 常出现 "32768"（字符串）或 32768.0（浮点），
    # 直接喂给 dataclass 会让后续算术/比较在运行时炸掉。
    _INT_FIELDS = ("max_tokens", "top_k", "max_iterations", "max_history",
                   "max_context_tokens", "rag_chunk_size", "rag_chunk_overlap",
                   "kernel_max_context_tokens", "kernel_raw_turns",
                   "kernel_consolidate_every", "kernel_max_tool_chars",
                   "kernel_max_digest_lines")
    _FLOAT_FIELDS = ("temperature", "top_p", "min_p", "repetition_penalty",
                     "request_timeout", "context_shrink_ratio")
    _STR_FIELDS = ("base_url", "api_key", "model", "workspace", "knowledge_base",
                   "rag_embedding_model", "rag_rerank_model", "browser_endpoint")
    _BOOL_FIELDS = ("rag_rerank_enabled", "rag_auto_ingest", "always_confirm_code",
                    "rag_ocr_coreml", "rag_ocr_preprocess")

    @classmethod
    def _coerce(cls, name: str, value):
        """Force a raw JSON value into the declared field type.

        Returns ``None`` when the value cannot be coerced, so the caller can
        fall back to the dataclass default instead of storing a wrong type.
        """
        # null 一律视为「字段缺失」：str(None) 会产生字符串 "None"，随后
        # base_url + "/models" 之类的拼接会把它当真实值用；workspace = "None"
        # 更是会在当前目录创建一个名为 None 的目录。
        if value is None:
            return None
        try:
            if name in cls._INT_FIELDS:
                return int(value)
            if name in cls._FLOAT_FIELDS:
                return float(value)
            if name in cls._STR_FIELDS:
                # 只接受真正的字符串：手写 config.json 把 model 写成列表/字典
                # 时，str() 会存成 "['x']" 这类垃圾值而不是回退默认。
                return value if isinstance(value, str) else None
            if name in cls._BOOL_FIELDS:
                # bool("false") 是 True——手工编辑配置时常见的 "false"/"no"/"0"
                # 字符串会把用户显式的关闭悄悄变成打开。只接受明确的映射。
                if isinstance(value, bool):
                    return value
                if isinstance(value, (int, float)):
                    return bool(value)
                s = str(value).strip().lower()
                if s in ("true", "1", "yes", "on"):
                    return True
                if s in ("false", "0", "no", "off", ""):
                    return False
                return None
        except (TypeError, ValueError):
            return None
        return value

    @classmethod
    def load(cls, path: str) -> "AgentConfig":
        import json

        if not os.path.exists(path):
            return cls()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("config root is not an object")
        except (OSError, json.JSONDecodeError, ValueError) as e:
            # 配置文件损坏（写一半崩溃 / 手工编辑出错）时不能让程序起不来：
            # 备份坏文件后用默认配置启动，用户重新「应用配置」即可恢复。
            logger.warning("config: %s 无法解析 (%s)，备份后使用默认配置", path, e)
            try:
                os.replace(path, path + ".corrupt")
            except OSError:
                pass
            return cls()

        known = {k for k in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {}
        for k, v in data.items():
            if k not in known:
                # 手写 config.json 打错字段名（如 rag_chunk_size 写成
                # rag_chunksize）时不该静默用默认值 —— 那会让「明明改了配置
                # 却没生效」变成一道没有线索的题。
                logger.warning(
                    "config: ignoring unknown key %r (check for typos)", k
                )
                continue
            coerced = cls._coerce(k, v)
            if coerced is not None:
                kwargs[k] = coerced
        cfg = cls(**kwargs)
        cfg._migrate()
        return cfg

    def _migrate(self) -> None:
        """Normalize obsolete config values from older versions.

        Runs after loading from disk so that stale values saved before a
        migration don't break the current code. Silently rewrites the
        in-memory config; the next ``save()`` persists the fix.
        """
        # Pre-LanceDB era used sentence-transformers model names or
        # ChromaDB defaults that FastEmbed doesn't recognize. Map any
        # unsupported embedding model to the current default.
        _DEPRECATED_EMBEDDING_MODELS = {
            "all-MiniLM-L6-v2",
            "sentence-transformers/all-MiniLM-L6-v2",
            "BAAI/bge-small-zh-v1.5",
            "BAAI/bge-large-zh-v1.5",
            "",
        }
        if self.rag_embedding_model in _DEPRECATED_EMBEDDING_MODELS:
            self.rag_embedding_model = "nomic-ai/nomic-embed-text-v1.5-Q"
        # Non-quantized nomic-embed-text-v1.5 (~548MB) → quantized -Q (~137MB).
        # Same 768-dim embeddings, 4x smaller download, same quality.
        elif self.rag_embedding_model == "nomic-ai/nomic-embed-text-v1.5":
            self.rag_embedding_model = "nomic-ai/nomic-embed-text-v1.5-Q"
        # fp32 全量 reranker（2.27GB）→ 官方 INT8 量化版（~570MB）。
        # cross-encoder 只产出排序分数，INT8 的排序质量损失可忽略，体积与
        # 常驻内存降到 1/4。旧配置里持久化的 fp32 名必须迁走，否则用户会
        # 继续每次加载 2.27GB。
        elif self.rag_rerank_model == "BAAI/bge-reranker-v2-m3":
            self.rag_rerank_model = "BAAI/bge-reranker-v2-m3-int8"
        # Old LLMClient builds defaulted to max_tokens=4096. Only rewrite the
        # value when it is still that legacy default — an explicitly chosen
        # small budget (e.g. 4096 for a local 4K model) must survive; silently
        # bumping every sub-8192 value would fight the user on every start.
        if self.max_tokens == 4096:
            self.max_tokens = 32768
        # max_context_tokens was introduced at 32000; older configs that
        # saved a stale small value should also be bumped. Track max_tokens
        # so the shrink threshold follows the model's actual window.
        if self.max_context_tokens < 8192:
            self.max_context_tokens = self.max_tokens

    def save(self, path: str) -> None:
        import json
        import tempfile

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # 原子写：先写临时文件再 os.replace。旧实现直接 open(path, "w")，
        # 写一半崩溃（断电/强杀）会留下截断的 JSON，下次启动 load() 直接抛
        # JSONDecodeError 导致程序无法启动。
        # tmp 名唯一：固定 `path + ".tmp"` 在两个线程同时保存时互相覆盖。
        fd, tmp = tempfile.mkstemp(prefix=".config.", suffix=".tmp",
                                   dir=os.path.dirname(path) or ".")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.__dict__, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def ensure_workspace(self) -> str:
        os.makedirs(self.workspace, exist_ok=True)
        return self.workspace
