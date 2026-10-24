# Changelog

本文件记录每次提交的可见行为变更。格式参考 Conventional Commits，按时间倒序排列。

---

## 2026-10-24 · 功能：KV cache INT8 量化（MLX + GGUF，主模型与辅助模型都启用）

> 用户需求：对 KV cache 做 INT8 量化节省显存，对 mlx 与 gguf 格式、主模型和辅助模型都采用。

### 实现
- **MLX**：`_make_prompt_cache()` 在模型默认缓存是**标准 KV 缓存**（`KVCache`/`RotatingKVCache`）时，改用 `mlx_lm.models.cache.QuantizedKVCache(group_size=64, bits=8)` 构建缓存列表传给 `stream_generate` 的 `prompt_cache`；跨轮前缀复用（form①）与裁剪复用（form②③，`QuantizedKVCache.is_trimmable()` 返回 True）均不受影响。
- **GGUF（llama.cpp）**：`Llama(..., type_k=GGML_TYPE_Q8_0, type_v=GGML_TYPE_Q8_0)`（`type_k`/`type_v` 是**整数枚举**，不是字符串 `"q8_0"`——此坑已在实测中踩中并修正；硬编码值取自 `llama_cpp.GGML_TYPE_Q8_0`，带兜底 8）。
- **配置 / UI**：新增 `local_llm_kv_quantize`（默认 `True`，`null` 当 `True` 处理）；`factory.build_llm_client` 与 `build_aux_llm_client` 都透传 `kv_quant=True`（主、辅助都启用）；配置区新增「KV INT8 量化」勾选框，与「跨轮复用」并列。

### 优雅降级（关键）
- **MLX 线性注意力模型**（如 Qwen3.5，默认缓存是 `ArraysCache`，无标准 KV）量化无意义 → 自动退回默认缓存并 info 级日志说明，**功能不降级**。
- **部分 llama.cpp 构建/模型不支持 `type_k`/`type_v` 覆盖**（实测任何取值都触发 `Failed to create llama_context`）→ 捕获后**退回默认 f16 KV**，仅 warning，绝不因量化把模型加载搞挂（与上面 MLX 跳过逻辑一致）。

### 验证
- 单测 6 例（`tests/test_kv_quant.py`）：标准 KV → `QuantizedKVCache` 列表；`RotatingKVCache` 同样量化；`ArraysCache` → 原样返回（不量化）；`kv_quant=False` → 默认；非 mlx 后端 → 防御性透传；GGUF `type_k`/`type_v` 参数存在性。
- 真实 GGUF 0.8B 模型：q8_0 尝试失败 → 降级默认 KV → 成功加载并 `eval=17` 生成，证明降级路径不阻断生成。

### 对你当前环境的重要说明（诚实告知）
- 你的**主模型 Qwen3.5 走 MLX 线性注意力**（`ArraysCache`），MLX KV 量化对本机该模型**自动跳过**（无标准 KV 可量化）——这是模型架构决定的，不是缺陷；换用标准注意力的 MLX 模型（如 Llama/Qwen 标准版）即生效。
- 你的 **llama.cpp 构建拒绝 `type_k`/`type_v` 覆盖**（任何值都 context 创建失败），GGUF KV 量化在本机构建下**同样降级为默认 f16 KV**。需换用支持 KV 量化的 llama.cpp 构建（或该构建开启对应编译选项）才能真正拿到 GGUF 侧显存收益。

### 测试
新增 `tests/test_kv_quant.py`（6 例）。

---

## 2026-10-24 · 修复：技能检索链路上的三处断层（量化权重 / 模型路径 / 技能放错工作区）

> 用户要求：检查技能检索是否真的能按意图检索到需要的 skill 并读取应用。

### 排查结论（端到端逐通道隔离）
意图检索**此前完全失效**，三层原因叠加：

1. **嵌入权重是 MLX 量化格式，加载器不识别**：用户今天把 nomic embedding 转成 8-bit/group64 量化版（`.weight` 打包 + `.scales`/`.biases`），`mlx_nomic.load_nomic_bert` 直接 `astype(float32)`——打包整数被当浮点、列数停在打包宽度 192（而非 768）→ 前向在第一层 LayerNorm 就崩（`weight must have the same size as the last dimension of x but has 768 elements`）→ 嵌入不可用 → **向量通道全灭**，技能表因探测失败也建不起来，BM25 随之无表可查。
2. **裸模型名解析不到 `~/models`**：权重已从 `<project>/models/`（该目录已不存在）搬到 `~/models/`，`resolve_model_path` 只搜项目内 → 嵌入/重排模型「目录不存在」。
3. **技能建错了工作区**：`fetch-ate-job-board` 写在了 `base-agent-workspace/skills`，而当前配置的 workspace 是 `app-space`（检索器扫 `<workspace>/skills`）→ 扫不到。另发现重排器配置值拼写错误（`-mx-fp8` vs 实际 `-mxfp8`）。

### 修复
- `mlx_nomic._dequantize_quantized()`：自动识别量化三件套并按「形状 + config 候选维度」反推 `group_size`/`bits` 调 `mx.dequantize` 还原稠密权重（不写死 8/64，2/3/4/6/8-bit 与常见 group 都覆盖）；无三件套的稠密模型原样返回。实测反量化后 768 维、L2 归一化正确、同句余弦 0.84。
- `registry.resolve_model_path` 新增 **`~/models/<name>` 兜底**（搜索顺序：绝对/相对 → `<root>/models` → `<root>` → `~/models`）。
- 技能复制到当前工作区 `/Volumes/ebook/app-space/skills/`；配置里的重排器路径改为正确的绝对路径。
- 本机 `config.json` 同步修正（重排器两个键）。

### 验证（端到端）
`skill_semantic_search("…获取今天所有的招聘信息，并整理成列表")` → **`fetch-ate-job-board` 排名第一**（向量 0.808 + BM25 9.80）→ `skill_load` 读到完整 1935 字技能内容。嵌入/重排器真实权重加载与推理均恢复正常。

### 测试
新增 3 用例：量化权重自动推断反量化（合成张量，含近似误差与附属键清理）/ 稠密权重原样返回 / `~/models` 兜底解析。

---

## 2026-10-24 · 修复：rollback 把原始任务从上下文里抹掉（模型自问「你没有给出具体任务」的真凶）

> 用户反馈（第四次）：消息里 URL 和任务都写得很清楚，模型却反复说「用户没有给出具体任务内容」，甚至幻觉出用户说过「直接开始」。

### 根因（探针逐轮打印 messages 实证）
规划前置门/反思门在拦截时会 `_rollback_turn(session)` —— 它把**含用户消息**的轮次整个 pop 掉；随后 `pending` 又被纠正提示**整体覆盖**（而非追加）。而 `anchor_prompt()` 只回注 `pending`、不会重新注入原始用户消息。于是：

| 轮次 | messages |
|---|---|
| 1 | `[system, user(带 URL 的任务)]` |
| 2 | `[system, user("[System] 你只是在回复文本里写出了计划…")]` ← **任务没了** |

任务两轮内从上下文永久消失，模型后来说「没有具体任务内容」是**字面事实**，随即进入复读直到空响应熔断。（此前加的四条禁语规则没用——模型确实看不到任务。）

### 修复
- 新增 `kernel/loop.py::_carry_task_text(task, hint)`：**rollback 路径的纠正提示必须自带原始任务原文**（`任务 + --- + 提示`；提示已含任务时幂等不重复）。4 处 rollback+覆盖点（反思门计划提示 / 空响应重试 / 大段代码未调工具 / 规划前置门）全部改走它。
- 回归测试 `test_kernel_rollback_keeps_user_task_in_context`：脚本化 LLM 先发“只在文本里写计划”触发 rollback，断言**此后每一轮**的 user 消息都仍含原始 URL。基线（无修复）如预期失败。
- 真机复测：同一条消息现在 7 轮跑完，正确建计划（4 步）→ 调 web_scan → 输出 25 条招聘列表。

---

## 2026-10-24 · 修复：本地模型拿到明确任务却反复自问「你没有给出具体任务」

> 用户反馈（第三次同类）：任务描述（目标+步骤）就摆在消息里，本地小模型却进入复读循环——「用户没有给出具体任务内容…需要先询问用户」。

### 根因
小模型把**元描述式**的任务消息（「我需要先规划步骤：1)…2)…3)…」）误解成“用户在给我讲规则/系统提示”，从而认为没有任务；一旦进入该分支就用复读代替推进。诱因还包括：那条消息里确实没带 URL（上下文丢失），模型遂以“缺任务”泛化提问。

### 修复（双管）
1. **SYSTEM_PROMPT 新增硬规则**：用户消息里写了具体目标就是任务，**直接开做**（哪怕用了「先规划步骤」这类元描述）；**明确禁语**「你没有给出具体任务 / 请告诉我你想做什么」；只缺某个参数（网址/路径）时用 `ask_user` 问**那一个参数**；**禁止复读**——同一句话不许输出第二遍。测试同步加断言锁死这四条。
2. **固化技能 `fetch-ate-job-board`**（写入 `<workspace>/skills/`）：把「applytoeducation 当日新发布职位」这条反复要用的流程做成确定性配方——实测可用的 JSON 端点（`GET /api/SearchIndex/GetSearch?SearchText=job`，页面是 JS 空壳不能直抓）、字段表（`position_Subject`/`employer_Name`/`posted_On`/`closing_Date`/`longLink`）、排序与拼链规则、输出格式。已用真实 SkillRetriever 验证该技能对「获取今天的招聘信息并整理成列表」检索**排名第一**（BM25 9.66）。

---

## 2026-10-24 · 清理启动警告：tokenizer 正则修复 + MLX 缓存 API 升级

> 用户贴出启动日志里的残留警告。其中 tokenizer 那条 transformers 明确提示 "will lead to incorrect tokenization"——不是噪声，是真实的分词偏差。

### 修复
- **`fix_mistral_regex=True`**：Qwen3.5 系 tokenizer 的 pre-tokenizer 正则用 Mistral 风格 `\p{...}` 类，transformers 后端不认 → 启动刷警告且分词结果确实偏差（中英混合/代码场景尤明显）。`mlx_lm.load` 的 `tokenizer_config` 会透传给 `AutoTokenizer.from_pretrained`，`_load_mlx` 显式传该参数；老版本 transformers 不认识时 `TypeError` 回退默认加载（功能不降级）。实测：警告消除、中英混排往返解码一致。
- **`mx.set_cache_limit` 替代 `mx.metal.set_cache_limit`**：后者已标记 deprecated 并将在未来版本移除。探测式调用（有新 API 用它，没有则回退旧路径）。
- 启动日志现已 0 条配置死键警告 / 0 条 tokenizer 警告 / 0 条 deprecation。

---

## 2026-10-24 · 修复：手写 config.json 的后端标识导致主模型加载失败（“莫名抽风”）

> 用户反馈：又开始莫名抽风。

### 根因
`~/.base-agent/config.json` 是**手工编辑**的（含 23 个本版本不认识的死键，如 `llm_kv_reuse`/`llama_gpu_layers`/`rag_fts_overfetch`），其中 `local_llm_backend` 与 `aux_local_llm_backend` 被写成了配置窗口下拉的**显示名** `mlx-lm`——而代码只认 `mlx`/`gguf`/`onnx`。`LocalLLMClient._ensure_model()` 直接抛 `未知 LLM 后端: 'mlx-lm'`，主/辅模型都加载失败，每一句对话都报错。另外 `local_llm_kv_reuse: null` 在直接构造客户端时会被 `bool(None)=False` 当成“关闭跨轮复用”。

### 修复
- 新增 `local/backends.py`：后端标识的**单一来源**（`normalize_backend` + `VALID_BACKENDS`），识别 UI 显示名（`mlx-lm`/`MLX-LM`/`mlx_vlm`）、llama.cpp 旧称（`llama_cpp`/`llama.cpp`）、onnxruntime 等写法，统一归一化为 `mlx`/`gguf`/`onnx`；无法识别返回 None 由调用方报错（错误信息带合法值列表）。
- `LocalLLMClient.__init__` 构造时即归一化，后续 `_ensure_model` 分支一律认规范值；`kv_reuse=None` 按默认 `True` 处理（不再被静默关掉）。
- `AgentConfig._migrate()` 加载时归一化 `*_backend` 并把 `local_llm_kv_reuse: null` 修回 `True`，两条都打 WARNING 说明改了什么。
- `factory.build_llm_client` 对 None 的 kv_reuse 同样按 True 处理。
- 已用代码路径重写本机 `~/.base-agent/config.json`（`mlx-lm` 改回 `mlx`、清掉 23 个死键；原文件备份 `/tmp/config.json.bak`）。

### 测试
- `test_local_models.py` 新增 4 用例：别名归一化表（显示名/旧称/大小写/空/无法识别）/ 构造时归一化 + 非法值报错带合法值列表 / `kv_reuse=None` 按默认开 / `AgentConfig.load` 修好手写配置。

---

## 2026-10-24 · web_scan 增强：JSON API 直采 / 动态页端点发现 / iframe 嵌套

> 用户需求：增强 web_scan 使其能处理 API 类数据源、动态页面与 iframe 嵌套页面，避免抓不到信息返回空；尽量不新增工具。

### 能力（单工具三层兜底，不引浏览器）
1. **JSON API 直采**：响应体是 JSON 时输出结构化摘要（顶层键、主列表字段、条目数/总数、逐条关键字段，`Document` 之类内层实体自动展开），不再被当 HTML 剥成乱码。
2. **JS 渲染空壳 → 端点发现**：可见文本过薄或含加载提示（"Loading jobs please wait…"）时，从**内联脚本 + 同源 .js**（配置类脚本优先，最多 6 个）里挖站点自己的数据端点并逐个试探，命中即返回其数据：
   - 端点路径与查询串常**分开放在不同脚本**——额外挖掘 `?Param=` 片段与语义键字面量（如 settings 的 `module:"job"`）做**跨脚本拼接**；
   - 候选按「数据类命名」优先、页面型（.aspx/.html…）排最后；查询片段按参数名语义排序（SearchText/q/query 优先，ip/utm/token 靠后）；
   - 预算：子请求 ≤10、端点变体 ≤8、总时限 25s。
3. **iframe 嵌套**：提取 `<iframe src>` 抓取子框架内容并拼回（≤3 个）。

### 安全与兼容
- 所有子请求复用既有 `_read_capped`（逐跳 SSRF 校验 + 8MB 流式上限）；新增**同源豁免**：同源子请求不做私网判定（透明代理/VPN 环境所有域名都解析到 fake-ip 198.18.x.x，严格判定会把一切误判为内网、导致全部候选被跳过——本机实测），跨源目标仍严格拦截私网/回环/链路本地。
- `raw=true`、协议白名单、ReDoS 安全的线性降噪、截断标注等既有行为不变；未发现端点时返回**可操作提示 + 候选端点列表**，不再空手而归。

### 实测
applytoeducation 招聘页（Azure Search 驱动的 JS 渲染站）：自动拼出 `?SearchText=job` 并返回 **25 条当日发布职位（库内共 3505 条）**，全程 ~6s；直调其 JSON API 亦返回结构化摘要。

### 测试
- `tests/test_web_scan_enhanced.py`（新文件，10 用例）：JSON 对象/数组摘要、动态页跨脚本拼接发现端点（含 `Document` 展开）、无候选时给提示不返回空、iframe 内容拼回、既有纯 HTML/raw/协议白名单不回退、端点正则回归（`/` 与 `api` 间无字符也必须匹配）、同源豁免。`test_tools.py`/`test_web_scan_enhanced.py` 等全过（kernel 2 个失败为存量，基线对照确认）。

---

## 2026-10-24 · 同步日志输出每文件分阶段耗时 + 全局阶段汇总

> 用户需求：同步过程中每个文件处理结束时输出日志，包含解析抽取、切片、向量化、入库的耗时，方便定位卡点。

### 新增
- **每文件一行耗时日志**（切片全部入库时输出）：
  `⏱ book_1.txt: 35 切片 | 解析 0.01s | 清洗 0.01s | 切片 0.33s | 向量化 0.13s | 入库 0.00s | 排队 0.63s | 合计 0.48s`
- **全局阶段汇总**（整轮写入结束）：
  `阶段耗时汇总: 解析 0.2s | 清洗 0.0s | 切片 0.7s | 向量化 0.4s | 入库 0.0s (合计 1.2s)`——一眼看出整体卡在哪一段。
- `VectorStore.add_streaming` 的 `on_batch` 回调支持**新第三参数** `timing={"embed": …, "store": …}`（向量化与入库分开计时）；两参数旧回调完全兼容。

### 实现要点
向量化/入库是**按 200 切片批量跨文件**执行的，单文件没有直接计时点：按「批次耗时 ×（该文件在批内切片数 / 批大小）」摊回各文件（切片按文件顺序产出，摊回映射确定）。解析/清洗/切片由 worker 线程按文件直接计时（清洗计时收在 Markdown 落盘处，不含缩略版生成）。另记录「排队」耗时（切片完成 → 首片写入）。

### 测试
- `tests/test_rag_timing_log.py`（新文件，3 用例）：每文件耗时行含全部阶段名 / 阶段汇总行存在 / `on_batch` 回调 2 参与 3 参两种签名都兼容。

---

## 2026-10-24 · 修复：ONNX 后端多处必崩缺陷（LLM 路径从未可用过）

> 用户要求：核查 ONNX 支持后端的实现，确认 ONNX 模型能正常加载使用。

### 实测结论
ONNX **LLM** 路径此前**从未可用过**（本地也没有 ONNX LLM 权重，只能靠构造型 ORT 会话 + 真实 HF tokenizer 端到端验证）。逐一修掉 5 个必崩点：

1. **tokenizer 取值错误**（连带 ONNX reranker 实测直接崩）：`hf = getattr(tok, "tokenizer", tok)` 是 MLX 的写法，在 ONNX（transformers AutoTokenizer）上取到的是底层 Rust `tokenizers.Tokenizer`——**不可调用、也没有 `apply_chat_template`**，表现为 `TypeError: 'tokenizers.Tokenizer' object is not callable`（reranker 实测）。新增 `local/tokenizer_compat.hf_tokenizer()`：**按能力取值**（可调用 + decode + apply_chat_template），MLX wrapper 与 ONNX tokenizer 都正确。
2. **缺 `attention_mask` / `position_ids`**：旧实现只喂 `input_ids` + past，带这些输入的导出直接 `Missing Input`。改为按会话声明喂，且 `attention_mask` 覆盖 **past+new 全长**（只喂增量时按增量长度拼会算在错位位置）。
3. **首轮缺空 past**：很多导出把 `past_key_values.*` 列为**必需**输入，没有历史时不喂 → `Required inputs missing`。新增 `_onnx_zero_past()`（按声明形状、序列维=0）。
4. **对 numpy 数组用 `or` 取回退值**：`outs.get("logits") or next(...)` 在数组/空数组上抛 `truth value ... is ambiguous`（两处）。改为显式 `is None` 判断。
5. **`_sample` 的 top_k 越界**：`np.sort(logits)[-top_k]` 在词表 < top_k（小模型/裁剪 logits）时 IndexError。改为按实际词表裁剪。

### 顺带修的 KV 续算错位
`_onnx_prompt_ids` 只记 prompt 段、**不含已生成 token** → 第二轮起把 assistant 那段重复喂一遍（正是该文件注释说要修掉的历史重复计入问题）。改为 `prompt + generated`，并让 ONNX 也输出 `raw_content`、参与双视图历史重建，跨轮 past 复用才真正触发（此前 raw_content 被剥 → 永不复用）。

### 其它
- 自定义 tokenizer 代码的模型目录（如 jina ONNX）会弹 **trust_remote_code 交互式确认**，GUI 无 stdin → EOFError/卡死。ONNX 的 tokenizer 加载统一 `trust_remote_code=True`（与 MLX 路径同款取舍）。
- 实测：`bge-reranker-base-ONNX`、`jina-reranker-v2-base-multilingual-onnx` 均可正常加载打分（相关文档高分、无关文档低分）；MLX reranker 未受影响。

### 测试
- `test_local_models.py` 新增 5 用例：`hf_tokenizer` 按能力取值（fast/wrapper/None）/ top_k 越界不崩 / **真实 ORT 会话端到端跑 `_stream_onnx`**（喂空 past、attention_mask 全长、`_onnx_prompt_ids` 含生成、第二轮复用 past、失效清空）/ 真实 ONNX reranker 权重加载打分。58 个全过。

---

## 2026-10-24 · GGUF 后端支持跨轮 KV / 前缀复用（对齐 MLX）+ 复用开关

> 用户需求：在 GGUF 后端增加跨轮 KV/前缀复用开关，对齐 MLX 行为，用 KV cache 缓解系统提示不变部分，加快响应。

### 背景（实测）
llama-cpp-python 0.3.36 的 `generate()` 虽有 prefix-match 逻辑，但本 build 的 **`kv_cache_seq_rm` 不支持**（日志：`prefix-match found but partial kv removal not supported`），随即 reset 全量重算——GGUF 此前**每轮都付全量预填充**（1500 token ≈ 2.3s）。

### 实现
- **append-only 复用**：自己渲染 prompt（复用 llama.cpp 内部同款 `Jinja2ChatFormatter`，template/eos/bos 取自 GGUF metadata）→ 自己 `eval()`。新 prompt 是已 eval 序列的**严格扩展**时只 eval 增量；分叉/回退/超 `n_ctx` 才 reset 全量（无法裁剪，与 MLX 的 ArraysCache 退化路径同款）。
- **`preload_system_kv` 扩展到 GGUF**：启动空闲期把系统提示 token eval 进上下文（渲染**不带** generation prompt，保证真实首句 prompt 是它的严格扩展）。
- **`raw_content` 对齐 MLX**：必须是 `gen_prefix + 原始生成文本`（实测不带 gen_prefix 只能对齐 1512/1576，带上后 1576/1576 全对齐），否则下轮在 assistant 边界分叉、复用失效。
- **开关 `local_llm_kv_reuse`（默认开）**：MLX 与 GGUF 都遵守；关闭后每轮整体重建（最保守，用于排查前缀错位乱码）。辅助模型恒为 False（prompt 每轮都变）。配置窗口新增「跨轮复用」勾选框。
- 健壮性：生成期逼近 `n_ctx` 自动停手；复用路径异常→reset+清状态后退回旧路径（功能不降级）；`invalidate_kv_cache` 对 GGUF 显式 `reset()`。

### 实测效果（Qwen3.5-2B-Agent-Q4_K_M）
| 场景 | 旧 | 新 |
|---|---|---|
| 首轮（已预填系统提示） | 2.28s | **0.85s** |
| 第 2/3 轮续接 | 2.3~2.4s | **0.27~0.34s**（只 eval 16~17 token） |
| 开关关闭 / 缓存失效 | 2.3s | 2.1~2.4s（全量，符合预期） |

### 测试
- `test_local_models.py` 新增 5 用例：默认开 + config 字段 / GGUF 关闭开关走 legacy / 开启时复用失败自动回落（reset+清状态）/ `invalidate_kv_cache` 显式 reset / formatter 缺失时渲染返回 None / factory 传参（aux 恒 False）。

---

## 2026-10-23 · 修复：模型路径选择只能选目录（GGUF/ONNX 选不到权重文件）+ 新填配置不落盘

> 用户反馈：设置界面只能选目录，选不到 .gguf 文件；手工输入文件路径后加载不了。

### 修复
- **选择按钮按「模型类型/实现」下拉分流**（`_pick_model_path`，替代只弹目录框的 `_pick_model_dir`）：GGUF → `*.gguf` 文件框、ONNX → `*.onnx` 文件框、MLX/fastembed → 目录框。覆盖主模型、辅助模型、OCR、Embedding、Reranker 五处（`gguf`/`onnx` 子串匹配，OCR 的 `mlx-vlm` 走目录）。GGUF 目录形式（目录内有 .gguf）仍可手工输入，加载层支持。
- **加载/应用配置改为「先 gather 再 save」**（`_on_load_main_model` / `_on_load_aux_model` / `_on_apply_config`）：旧实现先 `config.save()` 再 rebuild（rebuild 内部才 gather 控件）——新填的路径/后端只进内存、**不写进 config.json，重启应用即丢失**，且磁盘存的是旧值、与内存不一致。
- 配合上一条 GGUF 加载修复（`~` 展开 + 明确报错），手工输入路径现在也可用。

### 测试
- `test_ui_flow.py` 新增 3 用例：GGUF 弹文件框（过滤 `*.gguf`）/ MLX 弹目录框 / ONNX 弹文件框（过滤 `*.onnx`）。

---

## 2026-10-23 · 修复：GGUF 加载失败时错误不可读 + 不支持 `~` 路径

> 用户反馈：GGUF 模型加载不了，实现可能有问题。

### 排查结论（实测）
- 标准 GGUF 加载/流式/工具调用**全部正常**：Qwen3.5-2B-Agent-Q4_K_M.gguf 5.5s 加载、`create_chat_completion` 流式与 tools 解析均 OK；IQ3_XXS 亦可加载。失败的是 `Ternary-Bonsai-2-27B-PTQ1_0.gguf`——PTQ1_0 是 llama.cpp 不支持的特殊量化（需定制构建），属模型文件问题。
- 真正的实现缺陷：**路径无效时报错不可读**——路径不存在 / 目录里没有 `.gguf` / 把 MLX 目录当成 GGUF 路径时，旧代码把无效路径直接喂给 llama.cpp，只报一句 `Failed to load model from file: xxx`，用户完全看不出是路径问题（MLX/ONNX 路径都有明确 `FileNotFoundError`，唯独 GGUF 没有）；且**不支持 `~` 路径**（填 `~/Downloads/x.gguf` 直接失败）。

### 修复
- `llm._load_gguf` / `vlm._load_gguf` 加载前做路径校验：不存在 / 目录无 `.gguf` → 明确的 `FileNotFoundError`（含目录提示与填法指引）；支持 `~` 展开。
- llama.cpp 构造失败时包装为带常见原因的 `RuntimeError`：文件损坏 / 非 GGUF 格式 / 量化类型不被当前构建支持（PTQ/IQ 等需定制构建）/ `n_gpu_layers=-1` 显存不足。
- 显式置 `self._tokenizer = None`（GGUF 的对话模板由 llama.cpp 内建处理，预填/校验路径对非 mlx 后端均提前返回）。

### 测试
- `test_local_models.py` 新增 4 用例：不存在路径报 `FileNotFoundError` / 目录无 `.gguf` 报明确错误 / 目录取字典序首个 `.gguf` 且 `n_ctx=max_tokens+1024` / `~` 路径展开。

---

## 2026-10-23 · manifest 逐批增量落盘（杀进程也不丢同步进度）

> 用户反馈：manifest 只在收尾一次性写盘——用户不是点「停止同步」而是直接杀掉程序时，本轮已完整入库的文件下次同步仍然全部重做。

### 修复
- `_on_batch` 每个写入批次提交后，立即把新 committed 条目（全部切片已入库的文件）**增量合并写盘**（`_save_manifest` 读改写合并 + 跨进程锁，只合并本次新条目，历史/并发实例条目不受影响）。
- 一致性方向安全：写盘发生在 `table.add` 之后——manifest 里记了的一定已入库；没来得及记的（正在处理/写入一半）下次幂等重做，`add_streaming` 按 source 删旧写新，无重复无缺失。
- 收尾 Step 6 的最终保存保留不变（全量归并 + 清理已删除条目），与增量落盘幂等兼容。

### 测试
- `test_cancel_mid_sync_completed_files_not_redone` 强化：取消触发改用 `_on_batch` 的「已写入 N 切片」回调（`done` 计数在生产者侧递增，跑在消费端前面，用它触发有竞态），并在首批回调内快照磁盘 manifest——断言「首批写完但尚未收尾时 manifest 已在盘上」。旧实现（收尾才写盘）在该断言处失败，已验证。

---

## 2026-10-23 · 纯文本 → Markdown：启发式章节还原（txt / 非结构化纯文本）

> 用户需求：对 txt 或转换产出的非结构化纯文本，用现有「纯文本启发式还原章节目录」的方法生成章节结构，再参照该结构把纯文本转成 Markdown。

### 新增
- `document_outline.plain_text_to_markdown()`：用 `extract_chapters_plain` 同款启发式（中文章节 `第X章/节/卷/篇`、编号 `1.2.3`、全大写行）在正文定位独立标题行，按层级加 `#`~`######` 生成 Markdown——目录 digest 从此走标准标题路径、分块获得章节边界、按章检索可用。
  - **目录块只作权威级别映射，不转标题**：目录里的短标题（自序/后记等强模式覆盖不到的）在正文原样重现时按目录级别补标；目录条目本身变标题会把封面目录页炸成一堆假章节。
  - **目录区间终止改进**：目录里已见过的标题再次出现 → 目录块结束（否则紧邻目录的第一章标题会被吞掉漏标）。
  - **同名标题只标首现**（后续重现多为页眉噪声）；标记数 < 2 视为误判、整体不改写。
- `has_markdown_headings()`：已含标题的文本（.md / docling 输出）保守跳过转换。

### 接入（engine ingest）
- 在非 docling 分支、**`clean_text` 之前**调用：`clean_text` 第 11.5 步的硬回车合并会把「第三章 …」这类独立标题行吞进段落，必须先标记（`#` 行是结构行，`join_hard_wrapped_lines` 不合并）——这同时修好了纯文本文档的目录 digest（此前启发式跑在已合并文本上，标题早被吞了）。
- ⚠ 历史缓存：此前同步过的 txt 文档已有无标题的 `.md` 缓存，需**强制重新同步**才会按新逻辑重建。

### 测试
- `test_document_outline.py` 新增 7 用例：标题层级标记 / 目录块跳过+级别补标+正文紧邻不吞 / 无结构不改写 / 同名去重 / **标记后过 `clean_text` 标题存活**（锁住时序约束）。

---

## 2026-10-23 · 修复：问什么都去查知识库 +「知识库里有什么」查不出结果

> 用户反馈：问「你能做什么」去查知识库；问「知识库里有什么」也去查，关键还没结果。

### 根因
- **过度泛化**：上一轮把「知识库里都有什么资料」写成了「简单问答该调工具」规则的**示例**，模型把一切问题都往知识库查询上套——连「你能做什么」这种答案就在系统提示里的问题也去查库。
- **查不出结果**：`rag_search("知识库里有什么资料")` 这种**元问题当检索词**只会捞回噪声（实测 43 个文档的库只返回 1 条无关切片）；而正确的姿势是 `rag_outline` 不带参数列出全部文档（实测同款引擎路径返回 43 个文档——库本身完好）。

### 修复
- **SYSTEM_PROMPT 规则拆成三条**：纯寒暄（禁工具）；**自我认知/能力类（「你能做什么」「你是谁」）→ 直接回答，不查知识库不查记忆**（知识库存的是用户的资料，不是能力说明）；依赖用户资料/文件的问题才查证，并给出正确姿势——「有哪些资料」→ `rag_outline` 不带参数；检索词 ≠ 用户问句原文。
- **工具描述同步**：`rag_search` 明确「仅当问题依赖用户资料时使用，元问题不适用、query 不要传问句原文」；`rag_outline` 明确「问知识库里有什么 → 不带 book_name 列出全部文档」。
- **空结果指引**：`search_formatted` 空结果文案从「未找到相关内容」改为给出下一步指引（用 rag_outline 列文档 / 换措辞重试），模型不会再把空结果当成「知识库坏了」。

### 验证
- 探针实测：`list_documents()` 经 GUI 同款引擎路径返回 43 个文档（库完好）；元问题作检索词仅 1 条噪声——印证根因在调用姿势而非数据。
- 回归：提示词测试更新为新契约（自我认知直接答 / rag_outline 无参列文档 / 问句≠检索词），`test_agent.py` / `test_tools.py` / RAG 相关 68 个全过。

---

## 2026-10-23 · 修复：启动预热形同虚设（首句 18s 才出字）

> 用户反馈：应用启动后自动加载的主模型理应已预热，但第一句话等了 18s 才开始返回——预热根本没生效。

### 根因（探针实测复现）
- `preload_system_kv` 的前向 `self._model(input, cache)` 是 **MLX 惰性求值**：没有 `mx.eval()`，计算图根本没执行，KV 全是懒数组。实测 preload 耗时 **0.0s**（假完成），首句 `stream_generate` 第一次求值时才把 1400+ token 的 prefill 整体补算——预热完全白做，首句 TTFT 4.5s 里全是这笔迟到账。
- 叠加因素：预热（preload + warm_up）**串行占住 worker 线程**后才进消息循环。用户手快在预热窗口内发消息，就得在队列里干等 warm_up 完成（应用内表现为 18s+）。且内核首句生成前还有技能检索等 aux LLM 调用（各自 prefill）。

### 修复
- `preload_system_kv` 在前向后 `mx.eval(logits)` 强制求值——logits 级联依赖所有层的 KV，预热成本在启动空闲期一次性付清。实测 preload 0.0s→4.2s，**首句 TTFT 4.5s→0.9s**（`reuse=1410/1426`，只喂 16 个新 token）。
- `AgentWorker.run` 在做空闲期预热前先查队列：**已有用户消息排队就跳过预热**，直接处理真实消息（首句自己会把编译/预填付掉——反正躲不掉；让它干等 warm_up 是纯浪费）。
- 主对话组（kv_group="main"）的 KV 命中/复用日志从 debug 升为 info：`MLX KV cache: cached=… reuse=… suffix=…`，UI 日志里直接可查预热是否生效（重建另有「KV 缓存整体重建」info 行说明原因）。

### 验证
- 探针复刻应用时序（加载→preload→warm_up→首句→第二句）：preload 0.0s→4.2s，首句 TTFT 4.5s→0.9s。同步结束后重载路径走同一 `AgentWorker.run`，一并生效。

---

## 2026-10-23 · 修复：同步前卸载对话模型并未真正回收 MLX 内存

> 用户反馈：点「同步知识」理论上会先卸载主/辅本地模型释放显存，但实测同步时内存不降。

### 根因
- 卸载代码路径**没失效**：同步只能在 agent 空闲时启动（`_on_sync_knowledge` 在 `_busy` 时直接拒绝），且同步期间不置 `_busy`，所以 `_start_sync` 里的 `_on_unload_main_model()` / `_on_unload_aux_model()` 确实会被调用。
- 但 `LocalLLMClient.close()`（llm.py）只把 `self._model = None`，**没有 `gc.collect()` + `mx.clear_cache()`**。MLX 有独立的 Metal 缓冲池，权重 `del` 后只是回到池里、不立刻还给系统；而 worker 线程的 `release_mlx_pool()`（gc+clear_cache）运行时那批刚摘引用的模型阵列往往还没 freed，抓不到 → 权重滞留统一内存，同步峰值照旧冲高。

### 修复
- `LocalLLMClient.close()` 对 MLX 后端在摘引用后立即 `gc.collect() + mx.clear_cache()`，使「同步前卸载模型」**同步、确定性**地回收 ~3GB 统一内存；已加载的其它 MLX 模型（嵌入/重排）仍被引用，clear_cache 只回收无引用缓冲，不会误伤。非 MLX 后端（ONNX）走原来的 `session=None` + GC，不触发 clear_cache。
- 新增回归测试 `test_llm_close_reclaims_mlx_memory` / `test_llm_close_no_clear_cache_for_non_mlx`，锁住「MLX 后端必须调 clear_cache、非 MLX 后端不调」。

---


## 2026-10-23 · RapidOCR 引擎池：2 实例纯 CPU 并发

> 用户定版：ocrmac 识别不了的页会整页透传到 RapidOCR，透传请求占比不低，单实例串行是扫描书解析的吞吐瓶颈。启用 2 个实例、纯 CPU EP 并发。

### 新增
- `parsers` 的 OCR 引擎从「单实例 + Semaphore(1) 全程串行」改为**引擎池**：默认 2 个实例、**纯 CPU EP**，逐页借用/归还（PDF 兜底路径与扫描版 EPUB 路径共用同一池）。`rag_ocr_pool_size` 可调（设 1 退化串行）。
- 为什么纯 CPU：CoreML EP 单实例基线 ~1.7GB 且按输入形状 67MB/页爬升（182 页推到 5GB 峰值），并发会翻倍；CPU EP arena 增量实测 0、同书更快（172.6s vs 282.0s）、输出一字不差。
- 冲刷机制改用「代数」：`release_ocr_engine()` 丢空闲实例并 +1 代，借出中的实例归还时旧代即淘汰——周期重建/内存护栏对池化后依然生效（docling 池同款教训：整池归零计数会让池大小永久缩水）。
- 构建在锁外进行（~1-2s），不阻塞他人归还/借用；加载失败 60s 负缓存、失败不占池位。

### 测试
- `tests/test_parsers.py` 重写为池语义 + 新增 8 用例：借出不同实例 / 池满超时返回 None / 归还唤醒等待者 / release 只丢空闲且不缩水池 / 缺依赖不占池位 / 固定 CPU EP（配置请求 CoreML 也不注入）/ 调小池大小丢空闲 / `set_ocr_config` 注入池大小。

---

## 2026-10-23 · EPUB 扫描版串行闸门（2 路并发不变，扫描版之间串行）

> 用户定版：整体 EPUB 解析仍 2 路并发（扫描版 + 文字版可同时跑），但同一时刻只解析一本扫描版。

### ✨ 新增
- `ebook_parser._SCANNED_EPUB_GATE`（Semaphore(1)）：只闸「扫描版 EPUB 的 OCR」段——两份 OCR 同时打共享引擎单例既有线程安全风险，也让内存/CPU 尖峰翻倍。
- **探测在锁外**（`_detect_scanned_epub`，毫秒级）：文字版 EPUB 完全不碰闸门，永不为排队中的扫描版让路；排队时才打「排队中」日志（无阻塞试拿，不刷屏）。
- 逐页 OCR 改为按页粒度与图片型 PDF 兜底路径共用 `_OCR_SEMAPHORE`（300s 超时跳页），保护共享引擎单例。

### 🧪 测试
- 新增 `tests/test_ebook_scanned_gate.py` 4 用例：两个扫描版峰值并发 == 1 / 文字版不被扫描版闸门堵住 / 两个文字版仍真并行 / OCR 抛异常后闸门必然释放。

---

## 2026-10-23 · 回退 docling 多实例池（修复同步内存暴涨导致系统崩溃）

> 用户实测：同步时内存 20+GB，系统卡死崩溃；旧版可长期稳定运行。

### ⏪ 回退
- **docling 转换器改回「单实例 + 全程持锁串行」**。多实例池方案（小 PDF 并行转换）实测不可用：单实例构建即 +4.6GB（import torch + 布局 + TableFormer + 首次推理激活），两实例常驻 ≈9GB，再叠加两路转换各自工作集（~2.6GB/路）+ 嵌入/重排，峰值越过 16GB 物理内存 → swap 死亡 → 系统冻结崩溃。
- 连带移除：`rag_parse_parallel` / `rag_parse_large_mb` 配置字段、`set_docling_concurrency()` 注入、实例池测试（改为「转换串行化」契约测试）。PDF 解析吞吐的瓶颈在嵌入侧（主线程单路），转换并发只堆内存不提速。
- 保留的改进：空闲 90s 自动释放、同步结束显式释放、`release_parsers()` 全路径释放、心跳日志、MPS 缓存护栏。

### 🧪 测试
- `test_rag_audit_fixes.py`：三线程并发 `extract_markdown` 断言 convert 峰值并发 == 1；断言并发配置/池 API 已移除。

---

## 2026-10-23 · 全链路审计：修复泄漏 / 数据丢失 / 性能，文档同步

> 主题：审计全部流程与实现逻辑，修复确认的资源泄漏、状态不一致与性能问题，更新 README 与软件设计文档。

### 🐛 修复（内存泄漏）
- **`rag_ingest` 工具路径泄漏（最严重）**：抽取 `RAGEngine.release_parsers()`，在 `ingest()` 的 `finally` 无条件调用——UI 同步路径会 `close()`，但 agent 侧引擎被 `ToolRegistry` 长期持有，一次 ingest 后 docling 池（~4.6GB）+ OCR（~500MB）+ ovis（~0.9GB）会常驻到会话结束。
- **docling 实例池空闲释放失效**：`release_all()` 把实例计数整池归零，若此刻有实例在租，归还后计数不再增加 → 90s 空闲释放永久短路。改为只释放空闲实例、保留在租计数。
- **关窗口不关辅助模型**：`closeEvent` 补 `_aux_llm.close()`。
- **ONNX 后端 KV 只增不清且多轮错位**：`_past` 只在 prompt 是旧 prompt 延长时续用（且只喂新增段），否则清空；`invalidate_kv_cache` 一并清 `_onnx_prompt_ids`。

### 🐛 修复（数据丢失 / 状态不一致）
- **去重后 owner 删除 → 副本内容永久检索不到**：提升判据从 `content_hash` 改为 `chunk_count`（向量是否还在）。
- **无向量的副本被当 owner → 同内容新文件永不入库**：预播种排除 `duplicate_of` / `chunk_count == 0` 条目。
- **删除失败的文件仍被移出 manifest**（"下次重试"承诺失效）：`_cleanup_deleted_files` 只返回真正删成功的集合。
- **去重清副本时留孤儿 digest**：同步删除 `.md` 缓存与 documents 表行。

### 🐛 修复（UI）
- **同步期间点发送会在主线程重建 Agent**（与 sync worker 并发读写同一张 LanceDB 表）→ 明确拒绝并说明。
- **依赖检查 3 个提前 return 漏复位 `_sync_active`** → 同步按钮此后永久脱离 busy 管理。

### 🔧 优化（性能）
- 切块回退路径 `_gap_between` O(N²)（每个 chunk 从头 find 全文档）→ `_GapLocator` 游标，约 O(N)。
- 采样指纹重复读盘（预扫描 + 消费端各算一次）→ 预扫描结果复用。
- 「向量去重」入库时间每行一次 `os.stat` → 逐 source 缓存。

### 📝 文档
- README：更新 RAG 流水线（分片 1000/5%、并发 2、增量指纹）、新增内容去重 / 向量去重 / 解析并发分流 / 清洗与密度压缩 / 同步期卸载对话模型 / 取消续传；修正过时描述。
- 软件设计文档：更新 §4.2 摄取流水线（7 步）、§6 资源生命周期表与 `release_parsers` 说明、新增 §3.6.8 去重与 §3.6.9 清洗、§8.1 配置补 RAG 解析/并发字段。

### 🧪 测试
- 新增 `tests/test_rag_audit_fixes.py`（10 用例）锁定上述修复；`test_rag_dedup.py` 用例按新判据同步。

---

## 2026-10-22 · 31c1915 — UI 体验与本地推理优化

> 主题：输入体验、模型预热时序、辅助模型状态、启动加载、技能提示收敛、本地推理参数化。

### ✨ 新增
- **启动自动加载模型**
  - 本地模式（`llm_provider == "local"`）下，应用启动即主动加载主模型。
  - 辅助模型恒为本地（无 OpenAI 模式），**无论主模型什么模式，启动即加载**，供事实抽取 / RAG 归纳等子任务，避免首次用到才冷启动。
  - 加载走后台线程，不阻塞 UI；加载完触发常驻 worker 在空闲期预热。
- **本地推理可调参数**（见 `config.py`）
  - `local_llm_num_draft_tokens`（默认 6）：投机解码草稿 token 数，0 = 关闭。
  - `local_llm_prefill_step_size`（默认 4096）：prefill 批大小，加快长 prompt 预填。

### 🐛 修复
- **输入框回车发送**：Enter 发送，Shift+Enter 换行（此前只有按钮可发送）。
- **重载/加载主模型不再误伤辅助模型**：`_rebuild_agent` 新增 `rebuild_aux` 参数，仅重载主模型时保留已加载的辅助模型（含权重）与按钮态。此前任意「加载/重新加载主模型」都会把辅助模型按钮错置为「未加载」。
- **预热时序修正**：模型加载完成后**立即**启动常驻 worker（不再等 `warm_up` 编译），worker 内先预填系统提示 KV、再串行跑 `warm_up` 编译。消除「加载完头 30s 内发首句 → worker 未起、首句内联预热偏慢」的边角。

### 🔧 优化
- **技能提示收敛**：寒暄 / 问询 / 查询类简单输入（短且无多步信号，对应「不涉及计划、非多步」）不再触发「是否保存为技能」弹窗；真正的多步 / 执行类任务仍正常累积、达阈值后建议固化。
- **本地推理（mlx-lm）**
  - 投机解码已完整接线（复用辅助模型当草稿、零额外显存），但 **Qwen3.5 线性注意力层用 `ArraysCache`（不可裁剪），与 mlx-lm 投机解码要求的可裁剪缓存互斥，且 mlx-lm 已是最新 0.32.2 无版本可升**——故在此模型 / 库上**运行时自动安全降级**为普通生成（探测到缓存不可裁剪即跳过，不崩溃）；`prefill_step_size` 优化仍生效。
  - 接线全部保留，未来升级支持混合模型投机解码的 mlx-lm 版本、或换非线性注意力模型时会自动启用。

### 📝 备注
- 之前"感觉预热是在用户说了第一句话后才开始"，根因是**默认不加载模型**——用户直接打字会触发懒加载 + 内联预热。启动自动加载后该假象已消除。
- 默认不加载的行为保留在 API 模式（主模型为网络调用，无需本地权重），但辅助模型仍会启动即加载。

---

## 2026-10-22 · d439a35 — 系统提示 KV 缓存预热 + 首句命中

- 模型加载完成后用完整 system prompt（含工具描述，约 5900 token）预填 KV 缓存，用户第一条消息直接命中缓存、跳过数千 token 全量预填充。
- 修正在 `ArraysCache`（不可裁剪）下 `trim_prompt_cache` 分叉修复此前从未真正生效的问题。
