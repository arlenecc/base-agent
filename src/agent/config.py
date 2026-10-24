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
    rag_chunk_size: int = 1000      # tokens per chunk
    rag_chunk_overlap: int = 50     # 5% of chunk_size
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
    # PDF 统一解析引擎切换（仅影响「有文本层的常规 PDF/office 文档」首选谁，
    # 扫描件/图片重 PDF 始终走 parsers 的 RapidOCR 路径）：
    #   - "docling"（默认，现状）：docling 统一解析，不可用回退按格式解析器
    #   - "marker"：marker-pdf 优先（需手动 pip install marker-pdf，另依赖
    #     llama-server 二进制；未装时静默回退 docling → 按格式解析器）。
    # 实测（#基准）：marker 输出结构化 Markdown 且 CJK 文本层无乱码，但扫描
    # 件比 RapidOCR 慢 8 倍、依赖重（torch + 1.4GB VLM + datalab S3 模型）；
    # docling 对无 ToUnicode CMap 的中文文本 PDF 会输出乱码。两者取舍交给用户。
    rag_pdf_engine: str = "docling"
    # 统一 PDF/office 解析引擎切换（#133 三选一）：
    #   - "docling"（默认）：docling 统一管线（布局/表格模型走 MPS，OCR 走
    #     Apple Vision/ANE），输出版面良好的 Markdown
    #   - "marker"：marker-pdf 优先（需手动 pip install marker-pdf；未装时
    #     静默回退 docling）
    #   - "ovis"：OvisOCR2 VLM（PyMuPDF 取页/裁表 → OvisOCR2 转 Markdown），
    #     作为 docling/marker 之外的第三选项与最后兜底（替换原 PyMuPDF+
    #     RapidOCR 方案，后者只能出无版面纯文本）
    # 回退链：主引擎 → 其余统一引擎 → ovis → PyMuPDF 纯文本兜底。
    # OvisOCR2 VLM 模型配置（ovis 引擎用）：
    #   rag_ocr_engine    实现后端：mlx-vlm（safetensors 目录）| gguf（主模型
    #                     .gguf + mmproj*.gguf，llama-cpp 多模态）| onnx
    #                     （genai_config.json 目录，onnxruntime-genai 推理）
    #   rag_ocr_model_name 模型目录名（默认 OvisOCR2-0.8B-MLX-4bit）
    #   rag_ocr_model_path 显式本地路径（优先于 models/<name>，留空自动）
    rag_ocr_engine: str = "mlx-vlm"
    rag_ocr_model_name: str = "OvisOCR2-0.8B-MLX-4bit"
    rag_ocr_model_path: str = ""
    # docling OCR 渲染 scale（#132）：
    #   rag_ocr_scale_base    固定 scale 值（默认 2.0；原 3.0，位图内存 ∝ scale²）
    #   rag_ocr_scale_dynamic 动态策略开关（默认 False=固定用 base 值）
    #   rag_ocr_scale_table   动态策略下表格页的 scale（默认 2.5）
    # 动态策略开启时：convert 前预分析 PDF，表格页用 table 值、其余用 base 值。
    rag_ocr_scale_base: float = 2.0
    rag_ocr_scale_dynamic: bool = False
    rag_ocr_scale_table: float = 2.5
    # OCR（图片型 PDF）：Apple M 系列上让 RapidOCR 走 CoreML EP（CPU+GPU+ANE），
    # 关掉则回退 onnxruntime 默认 EP。
    #
    # 默认 False（2026-10 实测修订）：CoreML EP 的原生缓存按**输入形状**保留，
    # 真实扫描书实测 ~67MB/页——182 页扫描件单文件就把峰值推到 4.99GB，叠加
    # 常驻的本地 LLM（MLX 统一内存不可换出）在 16GB 机器上触发系统级内存压力
    # （整机卡死/崩溃）。而同一本书走 CPU EP：峰值 3.10GB、耗时 172.6s vs
    # 282.0s（快 39%），输出字符数 58246 完全一致。即 CoreML 在本负载下既更慢
    # 又更耗内存且无质量收益，故默认关闭。
    # 仍保留开关：EP 实现改进、或小图/固定形状的大批量场景可能反转结论。
    # 另见 parsers._OCR_RSS_BUDGET_GB——即使本项为 True，内存护栏也会在原生
    # 缓存超预算时自动冲刷并永久降级到 CPU EP。
    rag_ocr_coreml: bool = False
    # 跨轮 KV / 前缀复用开关（本地主模型，2026-10-24 新增）。
    # 系统提示与历史在多轮间保持恒定，前缀不变的部分可直接复用上一轮算好的
    # KV，首字延迟大幅下降（GGUF 实测首轮 2.3s → 续接轮只 eval 增量）。
    # 关闭后每轮整体重建（全量预填充，最慢但最保守）——用于排查前缀错位导致
    # 的罕见乱码，或在上下文/显存极紧张时以时间换空间。
    local_llm_kv_reuse: bool = True
    # KV cache INT8 量化（省显存，质量近乎无损）。2026-10-24 需求：对 MLX 与
    # GGUF、主模型与辅助模型都启用。MLX 在模型默认缓存是标准 KV 缓存时换成
    # QuantizedKVCache；Qwen3.5 等线性注意力模型用 ArraysCache（无标准 KV），
    # 量化不适用，自动退回默认缓存（功能不降级）。GGUF 走 llama.cpp 的
    # type_k/type_v="q8_0"。None 同样按默认 True（bool(None) 会被当成 False）。
    local_llm_kv_quantize: bool = True
    # RapidOCR 引擎池大小（2026-10-23 用户定版：2 个实例、纯 CPU EP 并发）。
    # 背景：ocrmac 识别不了的页会整页透传到 RapidOCR，透传请求占比不低，
    # 单实例串行是扫描书解析的吞吐瓶颈。两个实例 ≈ 2 × 0.5GB 常驻。
    # 固定纯 CPU EP——CoreML 单实例基线 1.7GB 且按输入形状 67MB/页爬升，
    # 并发会翻倍；CPU EP 的 arena 增量实测为 0 且更快（见 parsers 模块头）。
    # 设 1 退化为单实例串行（旧行为）。
    rag_ocr_pool_size: int = 2
    # OCR 前的图像归一化与增强（尺寸归一化 + 灰度 CLAHE + 锐化）。
    # 关掉则把渲染出的原始位图直接送进引擎（旧行为）。
    rag_ocr_preprocess: bool = True
    # OCR 前的二值化：灰度经锐化后用自适应阈值压成黑白。
    # 默认开——实测白底黑字的扫描件上字符数完全一致且更快（单页 4.15s →
    # 2.50s）。但二值化会丢掉灰度层次，对**有阴影/渐变/彩色插图**的页面可能
    # 少识别字（全书实测 -0.45%）。这类文档可关掉本项退回灰度路径。
    rag_ocr_binarize: bool = True

    # ---- 本地模型 provider 开关（应用内直接加载，不走网络）----
    # 每个模型用途一个开关：``api``（默认走网络实现）/ ``local``（加载本地权重）。
    # embedding / reranker / 主 LLM / 辅助 LLM 默认 ``local``（仓库自带 MLX 权重）。
    # 具体模型路径为空时走 local.registry 的默认值（models/ 下的自带权重）。
    #
    # 主 LLM：默认本地模式，默认加载 models/qwen3.5-4b-agentic-coder-v4-4bit-mlx。
    llm_provider: str = "local"
    local_llm_model: str = ""
    local_llm_backend: str = "mlx"
    # 本地主模型是否启用思考模式（Qwen3 系模板的 enable_thinking）。False 时
    # 模板注入空 <think></think> 块，模型直接作答——首字延迟显著降低，但
    # 复杂任务的推理质量会下降。
    local_llm_enable_thinking: bool = True

    # 辅助 LLM（本地专用，用于事实抽取等轻量子任务）：默认加载
    # models/Qwen3.5-2B-MLX-4bit。不带 OpenAI 模式（辅助任务始终本地）。
    aux_llm_provider: str = "local"
    aux_local_llm_model: str = ""
    aux_local_llm_backend: str = "mlx"
    aux_temperature: float = 0.7
    aux_max_tokens: int = 32768
    aux_top_p: float = 0.95
    aux_min_p: float = 0.05
    aux_top_k: int = 20
    aux_repetition_penalty: float = 1.0
    aux_request_timeout: float = 120.0
    # 辅助模型思考模式（同 local_llm_enable_thinking，Qwen3 系模板专用）。
    # 默认 **关闭**：辅助模型只干事实抽取、要点提炼这类轻量子任务——输入短、
    # 输出短、要的是结构化结果而不是推理链。开思考会让它先吐一长串推理再给
    # 结论，首字延迟成倍上升（主模型实测 3.49s → 0.38s 的差额就是这段），而
    # 抽取质量并没有可观测收益。主模型默认仍开着（复杂任务需要推理）。
    aux_local_llm_enable_thinking: bool = False
    # 投机解码：主模型生成时用小草稿模型并行猜 token、一次验证多个，解码吞吐
    # 通常 1.5-2.5x。本仓库固定复用「已加载的辅助模型」当草稿（不另起权重），
    # 故无额外显存；辅助模型未加载时自动退化为普通自回归生成。0 = 关闭。
    local_llm_num_draft_tokens: int = 6
    # prefill 批大小（mlx-lm 的 prefill_step_size）：长 prompt（系统提示约 5900
    # token）预填充的分块大小，调大加快缓存失效轮（压缩）的重填；0 = 用默认值。
    local_llm_prefill_step_size: int = 4096

    embedding_provider: str = "local"
    local_embedding_model: str = ""
    local_embedding_backend: str = "mlx"

    reranker_provider: str = "local"
    local_reranker_model: str = ""
    local_reranker_backend: str = "mlx"

    vlm_provider: str = "api"
    local_vlm_model: str = ""
    local_vlm_backend: str = "mlx"

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
                   "kernel_max_digest_lines",
                   "aux_max_tokens", "aux_top_k",
                   "local_llm_num_draft_tokens", "local_llm_prefill_step_size")
    _FLOAT_FIELDS = ("temperature", "top_p", "min_p", "repetition_penalty",
                     "request_timeout", "context_shrink_ratio",
                     "aux_temperature", "aux_top_p", "aux_min_p",
                     "aux_repetition_penalty", "aux_request_timeout",
                     # 这些是 float/bool 却被错分进 _STR_FIELDS —— config.json
                     # 里手写成 "2.0" 或 2 后，运行期 set_ocr_scale_config 会
                     # 把 str 当 truthy（bool）或对 float 比较报错。
                     "rag_ocr_scale_base", "rag_ocr_scale_table")
    _STR_FIELDS = ("base_url", "api_key", "model", "workspace", "knowledge_base",
                   "rag_embedding_model", "rag_rerank_model", "rag_pdf_engine",
                   "rag_ocr_engine", "rag_ocr_model_name", "rag_ocr_model_path",
                   "browser_endpoint", "llm_provider", "local_llm_model",
                   "local_llm_backend", "embedding_provider", "local_embedding_model",
                   "local_embedding_backend", "reranker_provider", "local_reranker_model",
                   "local_reranker_backend", "vlm_provider", "local_vlm_model",
                   "local_vlm_backend",
                   "aux_llm_provider", "aux_local_llm_model", "aux_local_llm_backend")
    _BOOL_FIELDS = ("rag_rerank_enabled", "rag_auto_ingest", "always_confirm_code",
                    "rag_ocr_coreml", "rag_ocr_preprocess", "rag_ocr_binarize",
                    "rag_ocr_scale_dynamic",
                    "local_llm_enable_thinking", "aux_local_llm_enable_thinking")

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
        # 后端标识归一化：配置被手工编辑 / 旧版本 / 外部工具写入时，*_backend
        # 可能是 UI 显示名（"mlx-lm"）或 llama.cpp 旧称（"llama_cpp"）。不归一化
        # 会让 LocalLLMClient 抛「未知 LLM 后端」→ 主模型加载失败、每句都报错
        # （用户侧表现就是「莫名抽风」）。统一走 local.backends 的别名表。
        from .local.backends import normalize_backend
        for fname in ("local_llm_backend", "aux_local_llm_backend"):
            raw = getattr(self, fname, None)
            if raw is None:
                continue
            norm = normalize_backend(raw)
            if norm and norm != raw:
                logger.warning(
                    "config: %s=%r 不是合法后端标识，已归一化为 %r",
                    fname, raw, norm)
                setattr(self, fname, norm)
            elif norm is None:
                logger.warning(
                    "config: %s=%r 无法识别，已回退为 mlx", fname, raw)
                setattr(self, fname, "mlx")
        # local_llm_kv_reuse 被写成 null 时 bool(None)=False，会把跨轮 KV 复用
        # 静默关掉（每句全量预填充、明显变慢）。None → 默认值 True。
        if getattr(self, "local_llm_kv_reuse", None) is None:
            logger.warning(
                "config: local_llm_kv_reuse=null 已回退为默认值 True"
                "（null 会被当成 False，导致跨轮 KV 复用被关掉）")
            self.local_llm_kv_reuse = True
        # 同上：local_llm_kv_quantize=null → 默认 True（否则 INT8 KV 量化被静默关掉）。
        if getattr(self, "local_llm_kv_quantize", None) is None:
            self.local_llm_kv_quantize = True
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
