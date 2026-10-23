# Base Agent

本地自主 Agent，基于 PyQt6 图形界面，支持 OpenAI 兼容的大语言模型，内置可插拔工具系统和本地 RAG 知识库。

## 内核架构（三态自动机 + 五级信息压缩）

底层控制流参照 GenericAgent 项目的内核（`agent_loop.py` / `agentmain.py` /
`llmcore.py` / `ga.py`）重构：**内核只控制流**，
历史交给 `Session`，能力交给 `code_run`/`shell_run`，经验交给 `memory`/`skills`。
详见 [docs/KERNEL_REFACTOR.md](docs/KERNEL_REFACTOR.md)。

```
                    ┌─────────────────────────────────────────┐
                    │                                         │
   ┌────────────┐   │ tool_calls         ┌────────────┐       │
   │   THINK    │───┼───────────────────►│    ACT     │       │
   │  (决策态)   │   │                    │  (执行态)   │       │
   └─────┬──────┘   │                    └──────┬─────┘       │
         │          │                           │             │
         │ 无 tool  │                           │ next_prompt │
         │          │                           ├─────────────┘
         ▼          │                           │ 预算告罄 / 每 N 轮
   ┌────────────┐   │                    ┌──────▼──────────┐
   │   EXIT     │◄──┴────────────────────│   CONSOLIDATE   │
   │  (终止态)   │   压缩完成 → 回 THINK    │   (沉淀态)      │
   └────────────┘                        └─────────────────┘
```

**五级信息压缩**，把上下文硬性控制在 30k 以内：

| 级 | 名称 | 时机 | 手段 | 上限 |
|---|---|---|---|---|
| L0 | 原始捕获 | 工具返回瞬间 | 硬上限 head+tail | 单条 ≤ 20K 字符 |
| L1 | 工具级裁剪 | 注入本轮前 | 按 RAW 区余额 ÷ 并发数分配 `maxlen` | 单条 ≤ 6K 字符 |
| L2 | 回合摘要 | 每轮结束 | 强制 `<summary>` ≤30 字物理快照 | 每轮 ≤ 30 字 |
| L3 | 会话折叠 | 预算告罄 / 每 N 轮 | 旧轮 → 一行摘要（同类项合并） | 折叠区 ≤ 2.7k token |
| L4 | 记忆蒸馏 | 任务完成 / 检查点 | 沉淀进工作记忆 → 知识图谱 → SKILL.md | 系统提示 ≤ 3.9k token |

**30k 预算分配**：`SYS 3.9k / EARLIER 2.7k / RAW 12k / TOOLS 5.1k / RESERVE 6.3k`。
关键不变量是 **L2+L3 让历史恒定 O(1)** —— 旧实现把整条历史每轮重发，
上下文随轮次线性膨胀，只能靠超阈值后的被动摘要兜底。

## 功能特性

- **多模型兼容** — 支持所有 OpenAI 兼容 API（OpenAI / DeepSeek / Qwen / 本地 vLLM 等）
- **流式对话** — 实时显示推理过程（thinking trace）和生成内容，支持 token 速度实时统计（仅计输出 token，不含 prompt）；结构化 read 超时（流式 chunk 间隔上限 60s），本地模型生成中途卡住时自动报错而非永久卡死
- **工具调用** — 内置 21 个工具（原子 8 + 能力 12 + 1 个条件注册；另有 2 个原子命名别名，`webexec_js` / `web_execute_js` 仅在配置浏览器端点时注册），支持模型自主决策和工具链式调用
- **内核控制流** — 三态自动机（THINK / ACT / CONSOLIDATE）+ 五级信息压缩，上下文硬性控制在 30k 以内
- **知识图谱记忆** — 长期记忆采用知识图谱（实体 + 关系 + 观察），对话结束后自动用 LLM 抽取事实入图谱（异步 daemon 线程，不阻塞用户回复）；观察通过 FastEmbed + LanceDB 语义检索，精简注入 prompt 避免全量灌入拖慢响应
- **本地 RAG 知识库** — 自动解析文档、清洗、切片、向量化，BGE 重排序精准召回
- **增量同步** — 通过 manifest.json 记录文件签名（mtime + size + content_hash），未修改文件完全跳过；右键「同步知识」可强制全量重处理
- **依赖自动管理** — 同步知识库前自动扫描文件类型、检查并安装缺失的解析依赖；chonkie（语义分块）、FastEmbed Reranker（bge-reranker-v2-m3-int8 重排序，INT8 量化 ~570MB）、opencv-python-headless（OCR 预处理）等可选增强也会一并检测与自动安装
- **统一解析引擎三选一** — PDF/office 文档走统一引擎回退链：`docling`（默认）→ `marker` → `ovis`（OvisOCR2 VLM）→ PyMuPDF 纯文本兜底。docling 全量管线：布局/表格模型走 **MPS**（torch Metal，比 CPU 快 ~35%），OCR 走 **Apple Vision（ocrmac，ANE 推理）**；单本图片书实测 ~6-10 分钟（MPS + scale 2.0 调优后，比早期 CPU 配置快约 10 倍）。OvisOCR2（`OvisOCR2-0.8B-MLX-4bit`，mlx-vlm）作为第三配置项：PyMuPDF 取页/裁表 → VLM 转 Markdown（文本页的表格区域自动裁剪回填），配置窗口「OCR模型」行可选模型与路径。三引擎实测（M4，10 页中文扫描书）：docling 18-25s / ovis 50-51s / marker 215s
- **OCR scale 可配置** — docling 的 OCR 渲染 scale 默认固定 **2.0**（原 3.0；位图内存 ∝ scale²，实测转换快 34%、质量不变）；`rag_ocr_scale_dynamic` 开启后按页动态决策（表格页 2.5、低 DPI/文本页 2.0，预分析带缓存）
- **OCR 兜底路径与内存护栏** — 引擎链末端是 PyMuPDF+RapidOCR 轻量路径（仅出纯文本）：引擎全局单例 + 信号量串行，每 5 页检查原生缓存增量（超 1.2GB 自动冲刷并降级 CPU EP），CoreML 模式下每 50 页周期重建；识别前做尺寸归一化 + 灰度 CLAHE + 锐化预处理（均可配置关闭）。以上行为可在配置 `rag_ocr_coreml` / `rag_ocr_preprocess` / `rag_ocr_binarize` 中调整
- **全链路日志** — 知识同步的每个关键节点（文件解析 → 清洗 → Markdown 转换 → 切片 → 向量化 → 入库 → 清理 → Manifest 保存）都通过 QTimer 轮询 + 线程安全日志缓冲区实时显示在右侧日志面板中；知识检索时展示向量检索候选数、FastEmbed Reranker 精排过程和最终结果排名
- **失败自动恢复** — 计划步骤失败时**不停车**：内核依据失败信息自动换方案重试（反思门失败感知：失败后提醒预算 2→8 次、引用最近错误**和已试过且失败的方案**、明确禁止「停下来问用户是否继续」）；**首次失败**即要求换方法（禁止原样重发、禁止把错误念给用户）；原样重试同一失败调用会被点名禁止；连续失败会要求先诊断再换方法；失败尝试不足 3 次就把步骤标 `blocked` 会被守卫拦回。失败上下文跨自动续跑段传递，续跑段仍知道上一段卡在哪
- **协作式取消** — 同步过程中可随时停止，worker 在文件/批处理边界安全退出；已入库数据不丢失，Manifest 保持一致性；「终止对话」会立即中断并回收 `code_run`/`shell_run` 子进程（旧实现子进程要跑到自身超时，最长 600s）
- **内存管理** — Embedding 模型进程级单例共享（RAG + 知识图谱共用同一 FastEmbed 实例，~137MB 而非 ~274MB）；LongTermMemory 单例缓存避免重复加载；`ToolRegistry.shutdown()` 统一释放 RAG 引擎 + 知识图谱 LanceDB 连接 + ONNX 模型 + FastEmbed reranker；ONNX Runtime InferenceSession 共享时不 release（避免破坏其他调用方）；OCR 引擎与 docling 转换器全局单例且同步结束后显式释放；`_JsonStore` 用单一全局 atexit 回调 + WeakSet 统一 flush（避免每实例注册一条永不注销的回调）；窗口关闭时统一清理 QThread worker 和 logger handler。**MLX 缓存上限显式压到 1GB**（默认可达统一内存的 75%，16GB 机器上实测把同步进程 footprint 顶到 19GB 的元凶）；docling converter 空闲 90 秒自动释放 + 每次转换后 `torch.mps.empty_cache()` 归还 MPS 缓冲；HuggingFace 全程离线（`HF_HUB_OFFLINE=1`，根治 convert 期间联网 connect 挂死）；单文档转换 1 小时硬超时兜底；日志/思考/计划面板均有行数上限防无限增长
- **健壮性** — 子进程统一 `start_new_session` 独立进程组，超时/停止时用 `killpg` 回收整棵树（`shell_run` 不留孤儿、MCP 不留僵尸）；确认弹窗默认拒绝（超时不会自动放行高风险操作）；`skill_load` 拒绝绝对路径/`..` 入口；`config.json` 原子写入 + 损坏自动回退默认配置；`web_scan` 限制 http/https 与响应体大小
- **Prompt 优化** — SYSTEM_PROMPT 精简至 ~200 tok（去掉模型已知的协议说明与冗余反例，补充「语义检索 query 用原始语言、勿翻译/音译」与「工具报错后改参数重试、两次失败即换方法」两条高价值约束）；Knowledge base 状态条件注入（无 KB 时省 tok）；`webexec_js` 条件注册（无浏览器时省 146 tok/轮）；Tool descriptions 精简（`rag_search`/`rag_outline` 去除冗长重复解释）；Work memory 注入用 compact JSON；长期记忆不注入 prompt 而靠 `memory_search` 按需检索；记忆抽取 prompt 精简并加条数/语言约束
- **MCP 协议** — 对接通用 MCP Server，自动注册远程工具
- **确认机制（风险分级）** — 工作区内操作自动执行；`code_run` / `shell_run` 默认**直接执行**，仅破坏性/影响系统与安全的操作才弹确认（`tools/risk.py` 静态分级：sudo、系统电源控制、进程杀伤、磁盘操作、管道下载执行、`git push --force`、写工作区之外的路径；读系统文件、工作区内读写、`pip install` 等一律不拦），`config.always_confirm_code=True` 可恢复逐次确认；MCP 工具仍逐次确认；确认弹窗只显示**用途说明**（启发式优先：注释/docstring/命令名映射，零延迟；无注释时才用 LLM 兜底生成 ≤20 字用途），不贴代码全文，避免超长文本把按钮顶出屏幕
- **技能系统** — 自动识别用户意图匹配技能；支持目录化技能（`.agent/skills/` 下每个技能一个目录，含 `skill.json` 元数据 + `prompt.md` 指令），`SkillIndex` 维护 `_index.json` 索引，`skill_search` / `skill_load` 按需发现与加载，技能指令不进系统提示词，节省 token

## 界面布局

```
┌──────────────────────────────────────────────────────────────────┐
│ [Base URL] [API Key] [模型] [获取模型]                    Row 0  │
│ [上下文长度] [温度] [top_p] [min_p] [top_k] [重复惩罚] [超时] [应用配置] │
│ [工作目录] [知识库] [同步知识] [停止同步]                   Row 2  │
│ [OCR模型] [mlx-vlm▾] [模型名] [本地路径]                   Row 1.5 │
│ [Embedding] [impl▾] [模型名] [本地路径]                    Row 3  │
├──────────────────────────────┬───────────────────────────────────┤
│ [对话] — [清空对话] [终止对话] │ 思考过程 / 日志                    │
│                              │                                   │
│ 对话内容...                   │ 推理过程实时显示...                 │
│                              │ 知识同步进度实时显示...              │
│                              │ 检索过程日志实时显示...              │
├──────────────────────────────┴───────────────────────────────────┤
│ 输入框                                               [发送]      │
├──────────────────────────────────────────────────────────────────┤
│ [状态栏]                                   [进度条] [token 速度]  │
└──────────────────────────────────────────────────────────────────┘
```

**按钮说明**：
- 「同步知识」：增量同步知识库（右键可强制全量重处理）
- 「停止同步」：仅在同步进行中启用，协作式取消（完成当前文件/批次后退出）
- 「终止对话」：终止当前 agent 对话（完成当前工具后退出）
- 「应用配置」：在 agent 运行期间会被阻止，需先终止当前对话

## 快速开始

### 安装

```bash
# 从源码安装
git clone https://github.com/arlenecc/Base-Aug-Agent.git
cd Base-Aug-Agent

# 安装核心依赖（GUI + LLM + MCP）
pip install -e .

# 安装 RAG 依赖（知识库功能）
pip install -e ".[rag]"

# 安装全部可选依赖（测试 + RAG + OCR）
pip install -e ".[all]"
```

**依赖分组**：

| 分组 | 安装命令 | 包含内容 |
|------|---------|---------|
| 核心 | `pip install -e .` | PyQt6, httpx, beautifulsoup4 |
| RAG | `pip install -e ".[rag]"` | lancedb, fastembed, python-docx, openpyxl, python-pptx, PyMuPDF, ebooklib, docling, chonkie, jieba |
| OCR | `pip install -e ".[ocr]"` | rapidocr 3.x（图片型 PDF 识别，未装时回退 1.x 兼容层）+ opencv-python-headless（图像预处理归一化/增强） |
| 测试 | `pip install -e ".[test]"` | pytest, pytest-qt（`qapp` fixture，UI 测试必需） |
| 全部 | `pip install -e ".[all]"` | 以上所有 |

RAG 子系统依赖也可在首次同步知识库时通过界面自动检测并安装。

### 环境要求（macOS Intel）

RAG 依赖（`fastembed` / `docling` / `chonkie` 等）按需安装即可；reranker 已随 fastembed 提供，无需额外的 torch / FlagEmbedding。

- **运行**：使用 3.12 解释器启动 `python main.py`（或用 3.12 创建 venv）。
- **打包**：`build.sh` 已自动优先使用 `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3`。

> **Reranker 的 numpy 兼容性**：reranker 现由 fastembed 的 ONNX Runtime（TextCrossEncoder）提供，与 embedding 同源，**不依赖 torch / FlagEmbedding，在 numpy 2.x 下默认可用**，无需降级 numpy。首次检索时自动加载 `BAAI/bge-reranker-v2-m3-int8`（INT8 量化，约 570MB，可经依赖预下载提前缓存）。
>
> **手工预下载注意**：fastembed 的模型名与它实际拉取的 HuggingFace 仓库不是一回事——`BAAI/bge-reranker-v2-m3-int8` 实际拉的是 `onnx-community/bge-reranker-v2-m3-ONNX`（与 fp32 全量版同一个仓库，只是文件不同：`onnx/model_int8.onnx` vs `onnx/model.onnx`），缓存目录名由**仓库名**推导（`~/.cache/fastembed/models--onnx-community--bge-reranker-v2-m3-ONNX`）。下到默认的 `~/.cache/huggingface` fastembed 看不到，报错依旧。依赖检查阶段的缓存探测是纯文件系统的，不会触发下载。

### 启动

```bash
python main.py
```

### 配置

首次启动后在界面填写以下信息：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| Base URL | API 服务地址 | `https://api.openai.com/v1` |
| API Key | 密钥 | - |
| 模型 | 模型名称 | `gpt-4o-mini` |
| 上下文长度 | `max_tokens` | `32768` |
| 温度 | `temperature` | `0.7` |
| top_p | 核采样参数 | `0.95` |
| min_p | 最小概率阈值 | `0.05` |
| top_k | Top-K 采样 | `20` |
| 重复惩罚 | 重复惩罚系数 | `1.0` |
| 超时(秒) | 请求超时 | `120` |

配置自动持久化到 `~/.base-agent/config.json`。

RAG 相关配置（写在同一个 `config.json` 里；标 🌐 的项在配置窗口「知识同步配置」区有对应控件，其余可手工编辑）：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `rag_embedding_model` | 嵌入模型（FastEmbed 模型名） | `nomic-ai/nomic-embed-text-v1.5-Q` |
| `rag_rerank_model` | 重排序模型（FastEmbed cross-encoder） | `BAAI/bge-reranker-v2-m3-int8` |
| `rag_rerank_enabled` | 是否启用重排序；关闭后不加载 reranker 权重 | `true` |
| `rag_pdf_engine` | 统一 PDF/office 解析引擎首选：`docling` / `marker` / `ovis` | `docling` |
| 🌐 `rag_ocr_engine` | OvisOCR2 后端：`mlx-vlm`（已实现）/ `onnx` / `gguf`（预留） | `mlx-vlm` |
| 🌐 `rag_ocr_model_name` | OvisOCR2 模型目录名（自动从 `models/` 加载） | `OvisOCR2-0.8B-MLX-4bit` |
| 🌐 `rag_ocr_model_path` | OvisOCR2 显式本地路径（优先于 `models/<name>`） | 空 |
| 🌐 `rag_ocr_scale_base` | docling OCR 渲染固定 scale（位图内存 ∝ scale²） | `2.0` |
| 🌐 `rag_ocr_scale_dynamic` | OCR scale 动态策略开关（表格页用 `table` 值、其余用 `base`） | `false` |
| 🌐 `rag_ocr_scale_table` | 动态策略下表格页的 scale | `2.5` |
| `rag_ocr_coreml` | RapidOCR（兜底路径）是否启用 CoreML EP | `false` |
| `rag_ocr_preprocess` | RapidOCR 图像预处理（归一化 + CLAHE + 锐化） | `true` |
| `rag_ocr_binarize` | RapidOCR 二值化 | `true` |

> 注意 fastembed 的**模型名**与它实际拉取的 HuggingFace 仓库不是一回事：
> `nomic-ai/nomic-embed-text-v1.5-Q` → 仓库 `nomic-ai/nomic-embed-text-v1.5`；
> `BAAI/bge-reranker-v2-m3-int8` → 仓库 `onnx-community/bge-reranker-v2-m3-ONNX`。
> 缓存目录名由**仓库名**推导（`~/.cache/fastembed/models--{org}--{repo}`），
> 所以手工预下载时必须落到 `~/.cache/fastembed`，下到默认的
> `~/.cache/huggingface` fastembed 看不到。

## 内置工具

工具分三层：**原子工具**（对齐 GenericAgent 的 9 个）→ **能力工具**（base-agent 原有）→ **外部协议**（MCP）。

### 第一层：原子工具

| 工具 | 功能 | 需确认 |
|------|------|--------|
| `code_run` | 执行 Python 代码 —— **能力核心**，可编程覆盖任意操作 | ✅ |
| `file_read` | 读取文件 | ❌ |
| `file_write` | 写入/追加文件 | ❌ |
| `file_patch` | 精确替换唯一文本块（`file_modify` 的原子命名别名） | ❌ |
| `web_scan` | 抓取网页内容 | ❌ |
| `web_execute_js` | 浏览器执行 JS（`webexec_js` 的原子命名别名，需配置端点） | ❌ |
| `update_working_checkpoint` | 短期工作记忆（每轮自动回注） | ❌ |
| `ask_user` | 向用户提问（中断自动机） | ❌ |
| `start_long_term_update` | 触发 L4 经验蒸馏 | ❌ |

### 第二层：能力工具

| 工具 | 功能 | 需确认 |
|------|------|--------|
| `shell_run` | 执行 Shell 命令 —— `code_run` 的系统侧延伸 | ✅ |
| `work_memory` | 短期草稿（键值对 scratchpad） | ❌ |
| `memory_graph` | 知识图谱 CRUD（实体/关系/观察） | ❌ |
| `memory_search` | 长期记忆语义检索 | ❌ |
| `skill_search` | 关键词搜索技能目录 | ❌ |
| `skill_semantic_search` | 混合检索技能（SKILL.md 向量 + BM25） | ❌ |
| `skill_load` | 加载技能完整指令 | ❌ |
| `skill_save` | **能力进化**：把成功流程固化为 SKILL.md（自动进索引） | ❌ |
| `rag_search` | 搜索本地知识库（支持按文档名定向检索） | ❌ |
| `rag_outline` | 获取文档缩略版本（目录结构 + 章节标题） | ❌ |
| `rag_status` | 查看知识库状态 | ❌ |
| `rag_ingest` | 增量索引知识库 | ❌ |

### 能力进化闭环

```
任务成功
  ├─ start_long_term_update → 蒸馏事实/偏好/坑 → memory_graph（经验）
  ├─ skill_save             → 固化可复用流程 → workspace/skills/<name>/SKILL.md（能力）
  └─ rag_ingest             → 外部语料灌入本地知识库（素材）
下次同类任务 → skill_semantic_search 命中 → skill_load → 直接复用
```

## 本地知识库 (RAG)

### 工作流程

```
文档目录 → 增量比对（manifest：mtime+size+字节采样指纹）
→ 统一解析 (docling → Markdown，单实例串行) → 清洗（注释/残留标签剥离 + 密度压缩）
→ Hybrid 智能分片 (结构+语义, 1000 tokens, 5% overlap)
→ 内容去重 / 向量去重 → 向量化 (nomic-embed-text-v1.5 / FastEmbed ONNX)
→ LanceDB 存储（向量 + 完整原文 + BM25 全文索引）
检索时: 向量相似度 top_k×4 + BM25 关键词 top_k×4 → 合并去重 → FastEmbed Reranker 精排
       → 过滤相似度 < 0.7 → 返回最相关 ≤ 3 条
```

### 同步全链路日志

同步过程中，右侧日志面板会实时显示每个阶段的详细进度：

```
══════════════════════════════════════════
▶ 知识库同步开始: 目录=/data/kb 模式=增量同步 并发数=2
══════════════════════════════════════════
━━━ Step 1/6: 扫描知识库目录 ━━━
  扫描完成: 共发现 12 个文件 [.pdf(5), .docx(3), .md(4)]
━━━ Step 2/6: 增量对比（Manifest） ━━━
  ✓ 跳过(未修改): report.pdf
  → 待处理(新增): new_doc.docx
  对比结果: 总计12 文件, 3 个待处理, 9 个跳过
━━━ Step 3/6: 解析文档 + 清洗 + 切片 ━━━
  [Worker-123] 启动，负责 1 个文件
  [1/3] 开始处理: new_doc.docx
    ├─ 解析文件: new_doc.docx
    ├─ 解析完成: new_doc.docx (提取 12500 字符, 耗时 0.15s)
    ├─ 清洗文本: new_doc.docx (原始 12500 字符)
    ├─ 清洗完成: new_doc.docx → 11800 字符 (耗时 0.02s)
    ├─ 转换 Markdown: new_doc.docx
    ├─ Markdown 完成: new_doc.docx (11800 字符)
    ├─ 文档切片: new_doc.docx → 24 个切片 (每片约 1000 tokens, 重叠 50 tokens, 耗时 0.01s)
    └─ ✅ 文件处理完成: new_doc.docx (24 切片, 总耗时 0.21s)
━━━ Step 4/6: 向量化 + 存入向量库 ━━━
  ├─ 向量化: 首批 24 个切片正在嵌入...
  ├─ 向量化完成: 首批 24 个切片
  ├─ 向量库写入: 首批 24 条记录
  └─ 向量化写入完成: 共 48 个切片 (2 批次)
━━━ Step 5/6: 清理已删除文件 ━━━
  🗑 已清理: old_draft.pdf
━━━ Step 6/6: 保存同步记录 ━━━
  同步记录已保存: 12 个文件条目
══════════════════════════════════════════
✅ 知识库同步完成 (耗时 3.52s)
   扫描文件: 12 | 新增/更新: 3 | 跳过(未修改): 9 | 清理删除: 1
   总字符数: 42000 | 向量切片: 48 | 错误: 0
══════════════════════════════════════════
```

### 检索日志

检索知识库时，日志面板也会展示详细的检索过程：

```
🔍 知识库检索开始: 查询="如何配置 RAG" 目标结果数=3 向量库总量=48
  ├─ 混合初检: 向量相似度 + BM25 关键词, 各取 top 12 候选
  ├─ 向量相似度搜索: 在 48 条向量中检索 top 12...
  ├─ BM25 检索: 12 条候选
  ├─ 混合候选 20 条 (向量 12 + BM25 12, 去重后 20)
  ├─ FastEmbed Reranker 精排: 对 20 个候选进行交叉编码打分...
  ├─ BGE 重排序完成: 20 → 3 条结果
  │   [1] rag_setup.md (重排序分: 0.8523)
  │   [2] config_guide.docx (重排序分: 0.7641)
  │   [3] faq.pdf (重排序分: 0.6218)
  ├─ 相似度过滤: 丢弃 1 条 < 0.70 的结果, 保留 2 条
  └─ ✅ 检索完成: 返回 2 条结果 (耗时 0.48s)
```

### 关键设计

- **嵌入模型**：`nomic-ai/nomic-embed-text-v1.5-Q`（Q4 量化版，~137MB），通过 FastEmbed (ONNX Runtime) 本地推理，**显式强制 CPU-only**（`providers=["CPUExecutionProvider"]`），无需 GPU、无需 HuggingFace 网络、无需 PyTorch。768 维向量，量化后 4x 更小（fp32 约 548MB），质量无损。首次下载后离线可用。CPU-only 是显式传递而非依赖默认回退——不传时 FastEmbed 的 `cuda=Device.AUTO` 会在环境存在 CUDAExecutionProvider（装了 onnxruntime-gpu）时静默切到 CUDA，与 OCR/解析全程 CPU 的设计不符，也让同一向量库在不同机器上的数值行为漂移。
- **向量数据库**：LanceDB（本地 Lance 列式格式），轻量、零配置、支持高效向量检索。向量归一化为单位长度后用 L2 距离模拟余弦相似度。每切片同时存入 embedding 向量、完整原文、来源文件名、切片编号，并开启 BM25 全文索引（原生 FTS + jieba 预分词）。
- **混合检索**：向量相似度 + BM25 关键词并行检索（各取 top_k×4 候选），按 (source, chunk_index) 合并去重，兼顾语义召回与精确关键词命中（中文经 jieba 分词）。
- **重排序**：FastEmbed Reranker (`BAAI/bge-reranker-v2-m3-int8`，INT8 量化版 ~570MB，fp32 全量 2.27GB)，对混合候选精排，提升召回精度。无重排序依赖时自动回退到距离排序。**CPU-only 显式强制**（`providers=["CPUExecutionProvider"]`），不依赖 FastEmbed 的默认回退。
  - 可在配置里关掉重排序（`rag_rerank_enabled=false`）或换成别的模型（`rag_rerank_model`）；关闭后检索直接走混合排序，**不会**加载 reranker 权重。
  - INT8 量化对 cross-encoder 的排序质量损失可忽略（只输出排序分数），体积与常驻内存降到 1/4，且 ONNX int8 在 CPU EP 上更快。
  - reranker 的原始输出是 logits，代码在首次打分时探测一次输出范围并**按模型固定**决定是否做 sigmoid 归一化（不再按「本批候选」自适应），保证 `min_similarity=0.7` 这个绝对阈值在不同查询间语义一致。
- **相似度过滤**：rerank 后过滤语义相似度 < 0.7 的结果（不相关的切片不发回模型），最终返回 ≤ 3 条最相关切片（含完整原文与来源信息）。

### Meta-context + Targeted RAG（动态按需检索）

针对"用户提到具体书名/文档"的场景，实现**先掌握全局结构、再按需提取细节**的动态 RAG：

```
文档解析 (docling → Markdown)
  └─ 提取目录结构（章节树 + 标题层级）→ 形成「缩略版本」
  └─ 存入 LanceDB documents 表（文档名 / 缩略版本 / 章节列表 / md 缓存路径）

用户提问「某本书里 XXX 是怎么说的？」
  ├─ ① rag_outline("书名") → 返回全书目录结构（掌握全局）
  ├─ ② 大模型规划需要哪些章节的细节
  ├─ ③ rag_search(query, source="书名") → 定向检索该书内的原文片段
  ├─ ④ 细节追加到 context，滚动累加
  └─ ⑤ 综合全局结构 + 按需细节 → 最终答案
```

- **缩略版本**：`rag_outline` 返回完整目录结构（标题层级），让大模型对全书有全局视野。**不生成逐章摘要**——逐章 LLM 摘要慢、费 token、且对本地模型易产生空结果，标题层级已足以支撑定向检索。
- **纯文本目录提取**：除标准 Markdown 标题（`#`~`######`）外，还支持从纯文本（无标题语法的 PDF/EPUB 解析结果）启发式提取目录：识别「目录/Contents」标记块、中文章节标题（`第X章`/`第X节`/`X卷`）、编号标题（`1.1`）、EPUB 链接（`[TITLE](#anchor)`）、全大写短行。
- **定向检索**：`rag_search` 支持 `source` 参数（文档名关键词），限定在某本书内检索，避免混入其它文档。
- **增量重刷**：同步时无论文件是否修改，都会扫描 Markdown 缓存，对缺记录或旧格式（含已废弃的「章节摘要」段/空目录）的文档重刷为纯目录结构。

- **增量同步**：`manifest.json` 记录每个文件的签名（mtime + size + **字节采样指纹** `source_hash`，小文件全量哈希 / 大文件头中尾各 64KB），未修改文件完全跳过（不解析、不切片、不嵌入、不写入），已删除文件自动清理对应向量和 manifest 条目。`force=True` 可强制全量重处理。取消同步后仍会保存 Manifest 以确保下次同步的增量准确性。
  - 指纹只在「本来要跳过」的路径上读盘（192KB/文件），正常路径零额外开销；它专门兜住 `cp -p` / `rsync -t` / 解压保留 mtime / 云盘回滚等「内容换了但 mtime+size 不变」的永久漏同步场景。
- **内容去重（跨目录同一本书只入一次）**：待处理文件先算采样指纹，命中「已入库文件」或本批更早出现的副本即记 `duplicate_of` 跳过——不解析、不切片、不嵌入；同步收尾再按 `content_hash` 分组，删掉副本的向量与目录元数据行并标记 `duplicate_of`（清理去重功能上线前各自入库留下的双份向量）。owner 被删除时副本自动转正（向量还在则直接转正，已被清理则下轮重新完整入库，不会静默丢失）。
- **向量去重（手动按钮）**：设置 → 知识同步区「向量去重」按钮，扫描向量库按**归一化文本哈希**分组，每组只保留**最早入库**的一份（入库时间 = Markdown 缓存 mtime 代理），其余按 id 分批删除，用于降低同段内容双倍召回的检索噪声。
- **流式 ingest**：worker 池（默认 2 线程）按 LPT 负载均衡分片（文件字节降序 + 贪心最轻负载）并行解析+切片，主线程通过有界队列（容量 = 2×workers）消费并增量写入向量库，背压机制避免内存堆积。写入批量 200 个切片。
- **解析串行（单实例）**：`DocumentConverter` 非线程安全，docling 转换器是**单实例 + 全程持锁串行**——曾试过「多实例池 + 小文件并行转换」，实测两实例常驻 ~9GB（单实例构建即 +4.6GB）叠加两路转换工作集（~2.6GB/路）+ 嵌入/重排，把 16GB 机器打到 20+GB 触发系统卡死崩溃，2026-10-23 回退。PDF 解析吞吐的瓶颈本就在嵌入（主线程单路），加转换并发只堆内存不提速。marker 同理保持单实例（surya ~3GB/实例）。
- **Markdown 缓存**：解析结果缓存为 Markdown，基于源文件 mtime 判断是否需要重新解析，避免重复处理。
- **批量嵌入**：嵌入按批次进行（默认 200 条/批），避免大知识库一次性嵌入导致 OOM。每批次内**按文本长度降序分桶**（批内长度相近使 padding 归零——混杂批的 padding 可浪费 2-4 倍算力）再拆为 1-5 条/子批，子批大小由自适应调优器按实际吞吐在边界上收敛；子批间主动释放 GIL 让 UI 保持响应，并按激活形状预算（条数 × 批内最长序列²）钳制超长文本的子批大小、以 8GB RSS 软墙降档避让。每批次嵌入前检查取消标志，及时响应停止请求。
- **协作式取消**：取消后 worker 在文件/批处理边界退出，已写入的向量数据保持完整；`_chunk_iter` 使用 0.5s 短超时确保取消后快速响应；`add_streaming` 在每批处理前检查取消标志避免无效计算。
- **资源释放**：同步结束（正常/取消/异常任一路径，由 `ingest()` 的 finally 保证）都会调 `RAGEngine.release_parsers()` 释放 docling 转换器池（~4.6GB）、OCR 引擎（~500MB）、marker 与 OvisOCR2（~0.9GB）——agent 侧的 `rag_ingest` 工具不会走 `close()`，只有这条路径能兜住它。`close()` 额外释放 LanceDB 连接、FastEmbed ONNX 模型与 reranker；`VectorStore.close()` 显式调用 ONNX `InferenceSession.release()` 立即回收 C++ 堆内存；窗口关闭时移除 logger handler 并关闭主/辅模型客户端防止悬空引用；`_JsonStore` 采用 0.5s 节流写盘避免高频 I/O。
- **同步期间释放对话模型**：点「同步知识」会先卸载主/辅本地模型（释放数 GB 显存给解析/嵌入），同步结束再原样重载并对主模型做预热（预填系统提示 KV + `warm_up` 编译）；同步期间禁止发送消息（避免在主线程重建 Agent 与 sync worker 并发读写同一张 LanceDB 表）。
- **取消即断点续传**：中途点停止时，**已完整写入向量库**的文件记入 manifest（下次跳过，不重做解析/清洗/切片/嵌入），进行中或只入库一半的文件不记录、下次从断点重来（幂等 upsert，不留「记录完成却只入库一半」的永久缺失）。
- **Markdown / 纯文本清洗**：统一引擎（docling/marker/OvisOCR2）与 pandoc 输出都过一遍轻清洗——剥除 `<!-- image -->` 等注释与占位块（docling 给每张图输出）、`<script>/<style>/<svg>` 整块丢弃、HTML 表格转 Markdown 管道表（**合并单元格/嵌套/caption 等管道表表达不了的结构保留 HTML 块**，不强行摊平）、残标剥除、实体还原；再做**密度压缩**（行内连续空格折叠、行尾空白剥除、纯空白行归为段间空行、连续空行收敛为一个），标题/列表/表格/引用/代码栅栏/行内代码/缩进代码块全部原样保留。
- **文档元数据缓存**：`VectorStore` 对 documents 表内容做内存缓存，`get_document_digest` / `list_documents` 不再每次全表 `to_pylist()`；写入（upsert/delete/clear）时失效缓存。documents 表写入用 `_documents_lock` 串行化，避免多 worker 并发 ingest 时 delete+add 交错。
- **不存全文 Markdown**：documents 表只存「缩略版本 + 章节列表 + Markdown 缓存路径」，**不存全文**（全文已有 `rag/documents/*.md` 缓存，表内再存是第三份冗余，且会被缓存全量载入内存；该字段实际零消费者）。缓存按列读取并显式剔除 `markdown`，兼容旧数据。
- **删除即清理**：源文件从知识库移除后，同步时一并清理向量、documents 元数据行与 manifest 条目。
- **BM25 索引一致性**：写入（含单批写完的提前出口）与删除都会失效 FTS 标记并按需重建，`reload()` 也会重置——避免「混合检索静默退化成纯向量」或「已删文档仍被召回」。
- **Reranker 加载去重**：FastEmbed reranker 加载用状态机（`idle/loading/done/failed`）标记，并发调用不会各自起加载线程；超时后后台线程完成时仍写入结果，后续调用直接复用。close 之后的实例直接拒绝加载；后台加载完成时若实例已 close，状态置 `failed` 而非卡在 `loading`（后者会让重排静默失效）。加载的是 INT8 量化版（`BAAI/bge-reranker-v2-m3-int8`，~570MB，fp32 全量 2.27GB）且**显式 CPU-only**。

### 支持的文档格式

**统一解析引擎（docling）**：Word、Excel、PowerPoint、PDF、EPUB 通过 [docling](https://docling-project.github.io/) 统一解析，直接输出结构化 Markdown（含布局、表格、标题、图片 OCR）。docling 未安装时自动回退到按格式的轻量解析器。

| 格式 | 扩展名 | 解析方式 |
|------|--------|------|
| Word | `.docx`, `.doc` | docling（回退 `python-docx`） |
| Excel | `.xlsx`, `.xls` | docling（回退 `openpyxl`） |
| PowerPoint | `.pptx`, `.ppt` | docling（回退 `python-pptx`） |
| PDF | `.pdf` | docling（OCR 优先 Apple Vision/ocrmac（ANE，中英文 `zh-Hans`+`en-US`），不可用时回退 RapidOCR；整页位图 scale 默认 2.0，内存 ∝ scale²） |
| EPUB | `.epub` | docling（回退 `ebooklib`） |
| MOBI/AZW | `.mobi`, `.azw3` | calibre（可选） |
| 纯文本 | `.txt`, `.md`, `.csv` | 无额外依赖 |
| HTML | `.html`, `.htm` | `beautifulsoup4` |

**OCR 优化（图片型 PDF）**：扫描/图片型 PDF 优先经 rapidocr 3.x 识别（未安装时回退 rapidocr-onnxruntime 1.x）。在 **Apple M 系列 Mac** 上，3.x 通过 `EngineConfig.onnxruntime.use_coreml` 启用 **CoreML EP**（`MLComputeUnits=ALL`，调度 CPU+GPU+ANE 神经网络引擎），相比纯 CPU 推理显著加速；CoreML 初始化失败自动回退 CPU EP，Intel/非 Apple 平台直接用 CPU EP（避免必然失败的 CoreML 初始化）。识别前对每页做**图像预处理与归一化**：按页面物理尺寸自适应 DPI 渲染、短边放大到目标像素、灰度 + **暗色主题自动反色** + CLAHE 对比度增强 + 锐化，提升小字、低对比度与暗色主题扫描件的识别率；渲染后为纯色/空白的页直接跳过。引擎每 50 页周期重建，冲刷 CoreML/ORT 按输入形状保留的原生缓存（真实扫描页约 35MB/页，400 页可滞留 ~14GB）。以上行为均可在配置 `rag_ocr_coreml` / `rag_ocr_preprocess` 中关闭。

### Hybrid 智能分片

分片采用**结构感知 + 语义分块**的混合（Hybrid）策略：

```
Markdown 文本
  ├─ 按标题结构（#~######）+ 空行段落边界切分
  ├─ 超长章节（>1000 tokens）→ chonkie SemanticChunker 语义细分
  │    （复用 nomic-embed-text 嵌入模型，按语义相似度边界切分）
  ├─ 合并过小块至 ≥100 tokens
  └─ 相邻块 5% overlap（50 tokens，句子边界对齐，保证跨块上下文连续）
```

- **大小**：语义块最小 100 tokens、最大 1000 tokens（`rag_chunk_size`）、重叠 5%（`rag_chunk_overlap` = 50）
  - 1000/5% 是「切片够大以承载完整语义单元、重叠够小以免冗余向量」的折中；旧值 800/10% 会让重叠 token 占比偏高、向量库膨胀。
- **嵌入模型**：`nomic-embed-text-v1.5-Q`（与向量存储共享同一份 ONNX 模型，不重复加载）
- **三级回退**：hybrid → 语义分块（chonkie）→ 递归分块（无模型依赖），任何环境都能工作

### 依赖自动管理

点击「同步知识」时会自动：

1. 扫描知识库目录，识别所有文件扩展名
2. 对照依赖表检查所需 Python 包是否已安装
3. 缺失的依赖自动 `pip install`（含版本约束）
4. 核心依赖（`lancedb`、`fastembed`）也会一并检查
5. 可选增强依赖（chonkie 语义分块、FastEmbed Reranker 重排序、opencv-python-headless OCR 预处理）纳入检查与自动安装，缺失不阻塞同步（均有回退路径）
6. Apple M 系列 Mac 上若 `onnxruntime` 低于 1.31（CoreML EP 加速 OCR 所需），给出升级建议命令（不强制升级，避免破坏现有环境）
7. 安装完成后自动开始同步

若自动安装失败（如网络问题），会弹出提示列出需手动安装的包。

### 使用方式

1. 在界面第三行「知识库」中输入文档目录路径（或点击 `…` 选择）
2. 点击「同步知识」执行增量同步（只处理新增/修改的文件）；右键点击可强制全量重处理
3. 需要清理存量重复内容时，点击「向量去重」（扫描向量库，清理文本完全相同的重复向量，每组只留最早入库的一份）；也可直接删掉 `workspace/rag/documents/*.md` 缓存——但注意**缓存缺失不会触发重新解析**（跳过判断只看 manifest 指纹），要让新清洗规则应用到已入库文档必须选「强制全量重新处理」
4. 右侧日志面板实时显示同步进度
5. 同步完成后，对话中模型会自动调用 `rag_search` 检索相关知识
6. 同步进行中禁止发送消息（此时主/辅模型已卸载，且 Agent 侧引擎与同步 worker 会并发读写同一张表）

### Token 控制

每次 RAG 检索最多返回 3 条结果，单次消耗 ≤ 1500 tokens（在 32768 上下文窗口中占比 < 5%），确保响应速度不受影响。

## 知识图谱记忆

长期记忆采用**知识图谱**架构，模仿 Memory MCP Server 的设计模式，用本地 JSON 文件 + LanceDB 语义索引实现，无需外部数据库。

### 数据模型

```
Entity（实体）    = {name, type, observations: [str], created_at}
Relation（关系）  = {source: name, target: name, label: str}
Observation（观察）= 挂在实体上的自由文本事实
```

图以实体 **name**（大小写不敏感）为主键，创建同名实体会合并观察。

### 文件布局

```
workspace/.agent/
├── work_memory.json           # 短期工作记忆（键值对 scratchpad）
├── long_memory.graph.json     # 知识图谱 JSON（实体 + 关系）
├── long_memory.graph.json.vectors/  # LanceDB 观察向量索引
├── skills.json                # SkillManager 数据（扁平技能记录 + 请求计数）
└── skills/                    # 目录化技能（每个技能一个子目录）
    ├── _index.json            # SkillIndex 自动生成的索引表
    └── <skill-name>/
        ├── skill.json         # 元数据（name/description/keywords/tags/entry）
        └── prompt.md          # 技能指令全文
```

### 自动事实抽取（一次抽取，双写）

每次 agent 对话结束（无 tool_calls 的最终回复）后，**异步**调用 LLM 从最近几轮对话中抽取实体、关系和观察（daemon 线程，不阻塞用户回复）。**一次抽取，同时写入两份记忆**，避免重复抽取：

```
对话结束 → _maybe_extract_facts_async()
  ├─ daemon 线程启动，on_finished() 立即回调
  ├─ 收集最近 6 轮对话窗口（解析代词/项目指代）
  ├─ LLM 低温度(0.1)抽取 JSON: {entities: [...], relations: [...]}
  ├─ 写入长期记忆 GraphMemoryStore（去重 + 向量索引）
  └─ 同步写入短期记忆当天桶 __auto_facts__（只保留当天，跨天重置）
```

- **长期记忆**：全部事实持久化到知识图谱，靠 `memory_search` 按需语义检索。
- **短期记忆**：当天抽取的事实保留在 `work_memory` 的 `__auto_facts__` 键（`{"date": "YYYY-MM-DD", "facts": [...]}`），对话时直接注入上下文；跨天后自动清空（历史已在长期记忆）。
- **独立抽取客户端**：抽取线程使用独立的 `LLMClient` 实例（`_get_extract_llm()` 惰性创建，独立的 `httpx.Client` 连接池），避免与对话线程并发争用同一个非线程安全的 `httpx.Client`；`_fact_extract_lock` 非阻塞锁进一步防止多个抽取堆积。`Agent.close()` 显式释放该独立客户端。

### 语义检索

观察通过 `nomic-embed-text-v1.5-Q` 嵌入后存入 LanceDB，支持语义检索：

```
memory_search(query) → GraphMemoryStore.search()
  ├─ LanceDB 向量检索 top_k 条观察
  ├─ 返回 {entity, text, score}
  └─ 向量索引不可用时回退到关键词子串匹配
```

索引侧的关键设计：

- **批量嵌入**：`add_observations` / `create_entity` 收集新观察后调用 `embed_documents()` 一次性批量嵌入（单次 ONNX 前向），替代逐条 `embed_query()`（N 条观察从 N×10-50ms 降为一次批量计算）
- **非对称前缀**：观察作为检索目标用 document 前缀嵌入，查询用 query 前缀嵌入（nomic 模型要求）
- **注入防护**：删除向量时对实体名单引号做 SQL 转义，防止 filter 表达式被畸形/注入

### Embedding 模型共享

RAG 知识库和知识图谱记忆共用同一 FastEmbed ONNX 模型实例（进程级单例 `get_or_create_embedding_function()`），避免加载两份 ~137MB 模型。ONNX Runtime `session.run()` 线程安全，可并发调用。

### 精简 Prompt 注入

**旧方案**：全量注入所有工作记忆 + 所有长期记忆事实 → 记忆多时 prompt 膨胀，拖慢响应。

**新方案**：
- **SYSTEM_PROMPT**：精简至 ~200 tok，操作原则/确认策略/工具约定/记忆/技能/RAG 引导各一条、无冗余
- **Knowledge base 状态**：仅当 RAG 可用时注入（无 KB 时不注入），仅含文档名列表（不含路径），最多 8 个
- **工作记忆**：手动 scratchpad（compact JSON，value 截断 200 字符）+ 当天自动抽取事实（列表形式）
- **长期记忆**：不注入 prompt，靠 `memory_search` 工具按需检索（避免长期记忆膨胀拖慢每轮响应）
- **Tool descriptions**：精简（`rag_search` 的 query 描述从 ~150 字压到一句话，`rag_outline` 去除重复的「定向检索」解释）
- **记忆抽取 prompt**：`_EXTRACT_SYSTEM` 规则压缩去重

```
# Work memory
{key:"value[:200]…"}

# Long-term memory (snapshot)
- PostgreSQL (tool): 用 16.3 版本; 连接池 50
- Alice (person): 负责 API 层
- Alice --[works_on]--> API Gateway
```

### 工具

| 工具 | 操作 |
|------|------|
| `work_memory` | `set` / `get` / `list` / `clear`（短期 scratchpad） |
| `memory_graph` | `create_entity` / `delete_entity` / `add_observations` / `create_relation` / `list_entities` / `get_entity` / `snapshot` / `clear` |
| `memory_search` | 语义检索观察，返回 `{entity, text, score}` |

### 兼容性

旧的 `long_memory.json`（扁平事实列表）在首次加载时自动迁移到知识图谱（作为 "General" 实体的观察），迁移后旧文件清空。

## 技能系统

技能用于把高频任务固化为可复用的指令模板。系统包含三层：

- **SkillManager**（`skills.json`）：按关键词匹配用户意图，命中后把技能 prompt 注入系统提示；同一意图出现 ≥ 2 次时建议固化为技能
- **SkillIndex**（`.agent/skills/` + `_index.json`）：目录化技能注册表。每个技能一个目录，含 `skill.json`（元数据：name / description / keywords / tags / entry）和 `prompt.md`（指令全文）
- **SkillRetriever**（`skills/` + LanceDB）：对 **SKILL.md** 的混合语义检索。递归扫描 `workspace/skills` 下的所有 `SKILL.md`，解析 frontmatter（name/tags/description）与正文段落（When to Use / Examples），用 nomic-embed-text 分别对 description 和 examples 计算向量存入独立 LanceDB 表，并对「name+tags+description+examples 拼接文本」建 BM25 全文索引

工作方式：

```
用户请求
  ├─ SkillManager.match() 命中扁平技能 → prompt 注入系统提示
  │   （目录技能注册的扁平记录引导 agent 调用 skill_load 加载全文）
  └─ 复杂任务 → agent 主动调用 skill_semantic_search(query) 混合检索
                │  向量检索（description 0.6 + examples 0.4 融合，取 3 候选）
                │  + BM25 全文检索（取 3 候选）→ 去重 → 剔除相似度 < 0.5
                │  返回候选的 name+tags+description，由 agent 判断选用
                └─ skill_load(dir) 读取 SKILL.md 全文（或 references/ 支持文件）并遵循
```

关键设计：

- **按需加载**：技能全文不进系统提示词，仅在需要时通过 `skill_load` 读入，避免技能增多导致 prompt 膨胀
- **索引防抖**：扫描器在目录 mtime 变化或 300s 间隔后重建，LanceDB 表懒构建
- **混合检索**：向量相似度（description/examples 0.6:0.4 加权融合）+ BM25 关键词双通道，统一用余弦相似度排序，阈值 0.5 过滤
- **小批次索引（防内存爆炸）**：`build()` 以 `batch_size=8` 分批 embed + 写入，每批处理 8 个 skill 后立即释放；description/examples/combined 文本分别截断至 2000/3000/6000 字符。实测 2757 个 skill 全量索引：峰值 RSS 4.19 GB（此前一次性处理全部文本时达 65 GB 导致系统卡死），耗时 12.4 分钟
- **路径安全**：`skill_load` 对 `path` / `entry` 做 realpath 前缀检查，拒绝 `..` 等路径穿越
- **创建技能**：`SkillManager.create_dir_skill(name, keywords, prompt, ...)` 自动生成目录结构并注册扁平记录

## 上下文治理（30k 硬上限）

Agent 不再「超了才压」，而是**先分配后消费**：30k 预算按
`SYS 3.9k / EARLIER 2.7k / RAW 12k / TOOLS 5.1k / RESERVE 6.3k` 切分，
每一级压缩都在自己的配额内花钱。

- **L1 工具级裁剪**：单条工具结果按「RAW 区余额 ÷ 并发调用数」动态算出 `maxlen`，
  并发 4 个工具时每个自动只拿 1/4，不用改任何工具代码
- **L2 回合摘要**：每轮强制 `<summary>` ≤30 字物理快照，替换掉整轮原文
- **L3 会话折叠**：旧轮整轮折叠成一行摘要（连续相同动作用 `（N turns）` 塌缩）；
  折叠以「整轮」为单位，因此**结构上不可能**产生悬挂 `tool` 消息
- **L4 记忆蒸馏**：任务完成/检查点时异步沉淀进知识图谱与 SKILL.md
- **兜底收缩**：单个巨型工具输出（如整本 PDF）仍可能一次性击穿预算 —— 此时才启用
  LLM 摘要式收缩（旧的 `_shrink_context` 等价物），摘要持久化到 `workspace/memory.md`
- **注入上限**：系统提示的动态注入段（当天事实、命中技能 prompt）**不在压缩范围内**，
  因此各自设上限（事实 ≤20 条/2000 字符，技能 prompt ≤1500 字符）
- **被动收缩**：遇到 LLM API 返回 `context_length_exceeded` 类错误时进入兜底收缩；
  不重试同一个超长 prompt（重试只会再失败一次）

相关配置（`config.json`，均有默认值）：

| 配置 | 默认 | 说明 |
|---|---|---|
| `kernel_max_context_tokens` | 30000 | 整个 prompt 的硬上限 |
| `kernel_raw_turns` | 2 | 保留原文的最近轮数（调大更准、更贵） |
| `kernel_consolidate_every` | 4 | 每多少轮进入一次 CONSOLIDATE |
| `kernel_max_tool_chars` | 6000 | 单条工具结果的字符上限 |
| `kernel_max_digest_lines` | 40 | 折叠摘要区保留的最大行数 |

## MCP 协议支持

在 `~/.base-agent/config.json` 中配置 MCP 服务器：

```json
{
  "mcp_servers": [
    {
      "name": "filesystem",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    }
  ]
}
```

启动后自动连接 MCP Server，注册其提供的工具。

## 项目结构

```
src/agent/
├── kernel/                  # 内核：只控制流
│   ├── outcome.py           # 三态枚举 / StepOutcome / TurnRecord / 转移表
│   ├── budget.py            # 30k 预算配额与核算
│   ├── compressor.py        # L0–L4 五级压缩 + 蒸馏 prompt
│   ├── session.py           # 历史托管：滚动折叠 + 工作记忆 + 锚点提示
│   └── loop.py              # KernelLoop：三态自动机主循环
├── agent.py              # 门面：保留 callbacks/skills/配置，把控制流委托给内核
├── config.py             # 配置管理
├── llm_client.py         # LLM 流式客户端
├── memory.py             # 工作记忆 / 长期记忆（知识图谱，节流写盘）
├── graph_memory.py       # 知识图谱存储引擎（Entity/Relation/Observation + 批量嵌入 + LanceDB 语义检索）
├── skills.py             # 技能管理（意图匹配 / 请求固化建议 / 目录技能创建）
├── skill_index.py        # 技能索引（扫描 .agent/skills/、维护 _index.json、加权检索）
├── skill_retriever.py    # SKILL.md 混合检索（扫描 + 解析 + LanceDB 向量/BM25 检索）
├── rag/
│   ├── engine.py            # RAG 引擎（编排：扫描→对比→解析→清洗→切片→向量化→清理→保存）
│   ├── docling_parser.py    # docling 统一解析器（Word/Excel/PPT/PDF/EPUB → Markdown）
│   ├── parsers.py           # 按格式的轻量解析器（docling 回退方案 + MOBI/HTML/文本）
│   ├── cleaner.py           # 文本清洗（去标签/URL/控制字符/零宽字符/空白归一化）
│   ├── chunker.py           # Token 级别递归切片 + 分片回退链编排
│   ├── hybrid_chunker.py    # Hybrid 分片（文档结构 + 语义分块）
│   ├── semantic_chunker.py  # chonkie SemanticChunker 语义分片（复用 nomic-embed-text）
│   ├── vector_store.py      # LanceDB 向量存储 + FastEmbed 嵌入 + BGE 重排序
│   └── deps.py              # 依赖检查与自动安装
├── tools/
│   ├── base.py           # 工具注册中心 + AliasTool（MCP + RAG 生命周期管理）
│   ├── checkpoint.py     # update_working_checkpoint / start_long_term_update
│   ├── file_ops.py       # 文件读写工具
│   ├── shell_run.py      # Shell 执行
│   ├── code_run.py       # Python 代码执行（协作式取消 + 泄漏线程追踪）
│   ├── web.py            # 网页抓取 / JS 执行
│   ├── interact.py       # 用户交互
│   ├── memory.py         # 记忆工具（work_memory / memory_graph / memory_search）
│   ├── skill_search.py   # 技能工具（skill_search / skill_load）
│   ├── rag_tool.py       # RAG 检索工具
│   ├── mcp_client.py     # MCP JSON-RPC 客户端
│   └── mcp_tool.py       # MCP 工具适配
└── ui/
    └── main_window.py    # PyQt6 主窗口（QTimer 日志轮询、信号槽、QThread worker）
```

## 运行测试

```bash
pip install -e ".[test]"

# 全部测试（含 RAG 端到端，需下载模型，较慢）
pytest tests/ -v

# 快速测试（排除需下载模型的 e2e 测试）
pytest tests/ -k "not rag_e2e and not rag_full_pipeline" -v
```

测试覆盖：Agent 推理循环、工具调用、LLM 客户端、RAG 全流程（端到端 + 集成 + manifest 增量同步 + 异常文件处理）、RAG 工具自主发现与调用、依赖检查、技能系统（意图匹配 + SkillIndex 索引/检索/路径安全 + SKILL.md 混合检索）、UI Bridge、内存存储、文档目录提取（Markdown 标题 + 纯文本启发式）等。

`tests/test_kernel.py` 覆盖内核专项：三态转移表、L0–L4 五级压缩原语、Session 折叠与
「无悬挂 tool 消息」不变量、30k 预算配额、**长任务下 prompt 峰值不越过 30k**、
9 个原子工具与能力工具的注册、`AliasTool` 委派、`update_working_checkpoint` /
`start_long_term_update` / `skill_save` 的行为与路径穿越防护。

近期补充的回归测试：首次 ingest 不再死锁（全新知识库）、`shell_run` 超时不留孤儿进程、`web_scan` 协议白名单、损坏 config 回退、`skill_load` 路径穿越防护、配置迁移不再改写用户显式设置、**「终止对话」在流式与工具链中途立即生效**、**向量化失败不再泄漏 worker 线程**、**`_JsonStore.reload` 不再丢弃防抖窗口内的写入**、**manifest 并发合并与有意删除**、**LPT 分片不产生空片**、**OCR 路由的 CoreML 门控与暗色页反色**。

内核专项回归：`max_turns` 是「每次 run()」而非会话累计、reask 不掉最终轮、周期性 CONSOLIDATE 真的折叠、旧轮折叠后无悬挂 `tool` 消息、长任务 prompt 峰值不越过 30k。工具/RAG 专项：`skill_save` 后索引被标脏并能召回、`code_run` 超时关闭缓冲区、删除的知识库文件不再被 Step 7 复活、mtime 被保留的内容变更能被重新同步。

失败自动恢复专项（`tests/test_failure_recovery.py`，14 例）：步骤失败后想收尾被失败感知的反思门拦回、失败放宽提醒预算（8 vs 2）、首次失败即要求换方案、原样重试被点名禁止、连续失败提示诊断换方案、防过早 blocked 守卫及其上限、成功后计数清零、失败上下文跨自动续跑段保留（续跑段锚点仍带上一错误）。复查修复专项（`tests/test_review_fixes.py`，19 例）：MCP 写队列、图谱确认/类型校验/clear 清队列、WorkMemory 淘汰强制落盘、plan patch 目标语义、抽取结果归一化、config 类型收敛与原子写、取消立即中断 code_run/shell_run、CJK 按字符截断、checkpoint 显式失败等。第十一轮专项（`tests/test_full_review_fixes.py`，5 例）：ingest errors 保持列表、WorkMemory LRU 并发安全、code_run 护栏非路径参数、超时后泄漏线程自行退出且全局流恢复。第十二轮专项（`tests/test_round12_fixes.py`，24 例）：OCR 两种 EP 注入路径与 rapidocr 1.2.x 降级、graph_memory 畸形 JSON 归一化、skills 原子写/唯一目录/损坏修复落盘且不吞后续写入、skill_index 畸形 skill.json 不致命、BOM frontmatter、空 ATX 标题不吞行、`> ~/...` 重定向判风险、MCP 重启清理残留等。

当前共 682 个用例全部通过（`test_rag_full_pipeline` 的 13 例依赖 `python-pptx`；未安装这些可选依赖时报环境错误，不影响其余测试）。

> UI 相关测试（`test_bridge.py` / `test_integration.py` / `test_ui_flow.py`）依赖 `pytest-qt` 提供的 `qapp` fixture；未安装时这三个文件的用例会在 setup 阶段报错，其余测试不受影响。

## 近期修复（全量代码复查）

按影响面排序，详见 [软件设计文档 §11](docs/软件设计文档.md)：

| # | 问题 | 影响 | 修复 |
|---|------|------|------|
| 1 | `upsert_document` 持 `_documents_lock` 后再取同一把非可重入锁 | **全新知识库首次同步永久卡死** | 锁改 `RLock` + 拆分 `_ensure_documents_table_locked()` |
| 2 | `InstallDepsWorker.finished` 遮盖 `QThread.finished` | `deleteLater` 在 `run()` 内销毁运行中的线程 | 改名 `deps_finished`，`deleteLater` 挂内置信号 |
| 3 | 确认弹窗 `wait()` 返回值被忽略 + 初值 `True` | 超时后 `shell_run`/`code_run` 被自动放行 | 初值 `False` + 检查 `wait()` 结果 + `reset()` |
| 4 | `skill_load` 未校验 `entry` | 绝对路径 / `..` 可读磁盘任意文件 | 显式拒绝 + `realpath` 前缀校验 |
| 5 | 语义切片无视注入的 embedding function | 离线环境卡在模型下载 | 全链路透传 `embedding_function` |
| 6 | 切片拼接凭空造字符 | chunk 文本不是原文子串，模型看到不存在的内容 | 取源码真实间隔 / 句末边界直接切片 |
| 7 | docling 锁惰性创建竞态 + `convert()` 无锁 | 并发加载两份模型、状态污染 | 模块级锁 + 串行化转换 + `release_converter()` |
| 8 | `shell_run` 只杀顶层 `sh` + 输出无上限 | 孤儿进程、大输出打满内存 | 独立进程组 `killpg` + 200KB 截断 |
| 9 | MCP `stop()` 以 `_running` 为条件提前 return | 僵尸进程永不回收 | 以 `_process` 为条件 + `killpg` + 关闭管道 |
| 10 | 配置损坏即无法启动、迁移改写用户设置、非原子写 | 启动崩溃 / 设置被反复改回 | 备份回退 + 仅迁移旧默认值 + 原子写 |
| 11 | 技能索引目录缺失时每次查询写盘、mtime 失效 | 无谓磁盘 I/O、索引最长 300s 陈旧 | 补更新时间戳 + 同时 stat `skill.json` |
| 12 | `skill_retriever.build()` 先删表后探测维度 | embedding 失败即丢失已有索引 | 先探测再重建 |
| 13 | `skills.json` 损坏记录拖垮 `match()`、请求表无界 | 整轮技能检索失败、文件膨胀 | 逐条校验跳过 + 淘汰上限 |
| 14 | `_JsonStore` 每实例注册 atexit 回调且从不注销 | 缓慢内存泄漏 | 单一全局回调 + WeakSet |
| 15 | `_describe_code_purpose` 用 `ev.get()` 访问 `StreamEvent` | LLM 兜底是死代码 | 按属性访问 |
| 16 | PDF/OCR 异常分支漏 `doc.close()`、加密 PDF 整份失败 | 句柄与解码缓存泄漏 | `try/finally` + 空密码解锁 |
| 17 | `VectorStore.close()` 未重置 `_fts_ready` | 重连后 BM25 静默失效 | 一并重置 |
| 18 | `web_scan` 无协议白名单、响应体全量入内存 | `file://` 读本地文件、OOM | http/https 白名单 + 8MB 截断 |
| 19 | 技能缓存未按 workspace 区分 | 切换工作区后读到旧技能 | 按目录路径缓存 |
| 20 | `chat_view` 无 `setMaximumBlockCount` | 长会话文档无限增长 | 限 5000 段 |
| 21 | 取消标志只在 confirm/ask_user 被检查 | **「终止对话」对流式/工具链不生效** | `is_cancelled()` 在迭代/流式/工具边界轮询，跳过的工具补合成结果保持历史一致 |
| 22 | `add_streaming` 异常后 worker 阻塞在 `q.put` | 每次同步失败泄漏 4 线程 + 文档内存 | 消费端失败置位 `_abort`，worker 立即退出 |
| 23 | manifest 在写库成功前提交 | 失败/取消的文件下次被跳过，内容**永久缺失** | `pending_manifest` 延迟到整条链路成功才并入 |
| 24 | `code_run` 劫持进程级 cwd/stdout | 超时泄漏线程永久破坏全局状态 | 主线程快照 + join 后恢复 |
| 25 | `code_run` 沙箱 `abspath` 不解析 symlink | workspace 内 symlink 写穿到系统文件 | `realpath` 前缀校验 |
| 26 | `file_read`/`code_run` 输出无上限 | 大文件 → OOM / 撑爆上下文 | 200K 上限（流式有界收集、按需读取） |
| 27 | MCP `tools/call` 复用 30s 超时 | 长任务必然超时且结果被丢弃 | 独立 300s 下限 |
| 28 | `_get_reranker` 无锁 check-then-act | 并发加载两份 ~500MB 模型；close 后复活 | 状态机加锁 + `_closed` 标志 |
| 29 | `_escape_like` 不转义单引号/通配符 | 畸形 SQL；含 `%`/`_` 的文件名匹配所有文档 | 引号翻倍 + `ESCAPE '\\'`（实测验证） |
| 30 | `_ensure_initialized` / 懒单例无锁 | 并发首调重复建连接/模型并泄漏 | 双重检查锁 |
| 31 | 依赖检查异常仍启动同步 | 缺依赖环境跑 RAG 引擎，掩盖真实错误 | 异常与取消分开，失败不启动 |
| 32 | `file_ops` 原子写固定 `.tmp` 名 | 并发互踩混合内容、失败残留 | `mkstemp` 唯一名 + finally 清理 |
| 33 | `graph_memory` 批量嵌入 `zip` 截断 | 图谱与向量库永久不一致 | 数量校验，整批失败 |
| 34 | `deps` 把 `config.json` 当重排序缓存命中 | 中断下载后跳过预下载 | 只认权重文件 |
| 35 | `skill_load` 读过期 retriever 属性 | 切 workspace 后校验发生在旧目录 | 走按目录缓存的 `get_skill_retriever()` |
| 36 | `file_modify` 无界读 + 二进制裸异常 | 大文件 OOM；二进制报难懂错误 | 10MB 上限 + 显式捕获，指引改用 file_write |
| 37 | `_atomic_write` 保留 mkstemp 的 0600 | 每次写文件悄悄变成「仅属主可读」 | 恢复原权限 / 按 umask 取 0644 |
| 38 | `mcp` 工具输出无上限 | 一次调用撑爆模型上下文 | 200K 截断 + 标注原始长度 |
| 39 | `mcp.stop()` 最终 `wait()` 无超时 | D 状态进程卡死 shutdown | 5s 超时放弃等待 |
| 40 | `memory_search` 无 top_k 钳制 / `list_entities` 全量 | 全部观察/实体灌进上下文 | top_k ≤ 20；列表截断到 200 |
| 41 | `rag_status` 文件列表全量 | 数百文件挤占上下文 | 截断到 50 并标注总数 |
| 42 | `execute()` 把工具内部 TypeError 误报为参数错误 | 工具出 bug 时模型反复重试而不是上报 | `inspect.signature` 预检区分两类错误 |
| 43 | `code_run` 泄漏线程可吞掉后续执行的 stdout | 并发执行时输出丢失 | 恢复权全部移交主线程，worker 不再碰全局状态 |
| 44 | `_JsonStore.reload()` 丢弃防抖窗口内的未落盘写入 | **0.5s 内 set 后 reload，刚固化的技能静默丢失** | reload 前先把脏数据强制写回 |
| 45 | `add_streaming` 单批写完提前 return，跳过 FTS 重建 | **小知识库同步后 BM25 静默失效**（混合检索退化成纯向量，无报错） | 提前出口同样重建索引 |
| 46 | `delete_by_source` 删向量后不失效 FTS | 已删除文档的内容仍被关键词检索召回并返回给 LLM | 删除后置 `_fts_ready = False` |
| 47 | `reload()` 只刷向量表句柄 | 同步后 `rag_outline` 读到旧 digest、BM25 看不到新切片 | 一并重置 documents 表/缓存/FTS 标志 |
| 48 | documents 元数据表存每篇文档**完整 markdown** 且全量进缓存 | 语料在磁盘存 3 份；进程常驻内存 ≈ 整个知识库；该字段**零消费者** | 不入库，改存 `md_path`（缓存按列读取，绝不加载 markdown） |
| 49 | 删除源文件只清向量与 manifest，不删文档元数据 | `rag_outline` 列出已无向量的文档 | `_cleanup_deleted_files` 同步 `delete_document` |
| 50 | `_fetch_worker` 成功后从不置 None | **获取模型后关闭窗口必崩**（`isRunning()` 抛 RuntimeError → PyQt abort） | 两个槽都清空引用；`closeEvent` 全部改用 `_is_worker_running()` |
| 51 | `_on_stop_sync` / `closeEvent` 直接调 `isRunning()` | 同上，`deleteLater` 后点击「停止同步」崩溃 | 统一走带 RuntimeError 保护的守卫 |
| 52 | `usage` 字段为 `null` 时 `None > 0` | 部分网关返回 null → **整轮 run() 崩溃，回复丢失** | `usage.get(x) or 0` 归一 |
| 53 | 事实抽取 prompt 无条数/语言约束 + 当天事实桶无上限 | 长会话系统提示单调膨胀（且系统提示**不在上下文收缩范围内**） | 抽取限 5 实体 × 3 观察；桶限 200 条；注入限 20 条/2000 字符 |
| 54 | 命中技能 prompt、当天事实注入零截断 | 同上，膨胀无自愈路径 | skill prompt ≤1500 字符并提示 `skill_load` |
| 55 | `_estimate_history_tokens` 不计 `tools` schema | 工具多时系统性低估 → 主动收缩几乎不触发，只剩被动收缩 | 计入 tools schema（带缓存） |
| 56 | `_get_extract_llm()` 阻塞等 `_fact_extract_lock` | **确认弹窗被抽取线程阻塞最长一个 LLM 请求（~120s）**，与「不延迟弹窗」意图相反 | 非阻塞取锁，取不到直接复用 `self.llm` |
| 57 | `memory_search` 关键词兜底路径完全忽略 `top_k` | 未装 rag 依赖时（默认路径）一次查询灌入全部观察 | `_keyword_search` 接收并生效 top_k |
| 58 | 技能分词用 `\w+` 对中文失效 | `"提取pdf表格"` 成单 token，中文关键词**永远匹配不上** | 补 CJK 整串 + bigram |
| 59 | `mcp.stop()` 先关管道再发 SIGTERM | 读线程阻塞在 `readline()` 时 `close()` 等到子进程退出（实测挂起 60s 的 server 卡 59s），且信号根本发不出去 | 先 `killpg` + 带超时 `wait`，进程退出后才关管道；仍存活则放弃关管道 |
| 60 | `call_tool` 只看 JSON-RPC `error`，忽略 `result.isError` | 工具失败（"permission denied"）被当成成功结果喂给模型 | `isError` 为真时抛 `MCPError`，错误正文一并带出 |
| 61 | MCP 工具不校验 `required`（`run(**kwargs)` 使签名预检失效） | 缺参原样发到远端，server 不校验时静默产生错误结果 | 按 `inputSchema.required` 本地校验，缺参不发请求 |
| 62 | skill 扫描 `os.walk` 不跟随链接 + `read_skill_dir` 用 realpath 判包含 | **`ln -s` 挂载的外部技能完全搜不到、也读不出来**（最常用挂载方式） | 自写 walker 跟随链接 + realpath 集合防环；`read_skill_dir` 只拦绝对路径与 `..` |
| 63 | SKILL.md frontmatter 只按单行解析 | `description: >-` 折叠块被解析成 `"-"`，技能进了索引但语义检索**永远召回不到** | 支持 `>` / `\|` 块标量；缩进行不再污染顶层字段 |
| 64 | skill BM25 索引缺 `replace=True` | 每次 sync 重建失败（`index already exists`），索引停在首建快照 | 与 RAG 侧一致：`replace=True` + simple tokenizer 参数 |
| 65 | `retrieve()` 用向量阈值统一过滤 BM25 候选 | 关键词精确命中（实测查询 "CI"：BM25 0.88、余弦 0.478）被整体丢弃 | BM25 命中不再受余弦阈值影响，排序仍用余弦 |
| 66 | `list_sources` 用 `to_arrow(columns=[...])`（lancedb 0.25 不支持该参数） | 两次调用都抛 `TypeError` → 永远回退全表**全列**物化（含 768 维向量），而它每轮对话都被 `_system_prompt()` 调一次：2 万切片 **7.5s**、内存 GB 级 | 只投影 `source` 列（2 万切片 7.47s → 35ms，结果一致） |
| 67 | LLM 调用失败即整轮终止 | 首字节前的网络抖动 / 429 / 502 让用户白等一场，只能手动重发 | 仅在「未收到任何 chunk」时退避重试 2 次；已吐内容、上下文过长、4xx 不重试 |
| 68 | 多个独立工具串行执行 | 多工具回合 wall time = Σ（每个 0.4s 的两个调用要 0.8s） | 只读白名单工具并发执行（UI 回调仍在主线程发出，历史保持原序）；写操作永不并发 |
| 69 | 工具执行期间无法响应取消 | `shell_run` 最长 600s、MCP `tools/call` 300s → 点「终止对话」要等它跑完 | 工具放到线程执行，主线程 0.1s 轮询取消，命中立即返回合成结果 |
| 70 | 历史里旧 tool 输出从不裁剪（单条上限 200K） | 上下文随轮次线性膨胀，且每轮整体重发 | 较早的 tool 结果裁剪为 head+tail 并标注（末尾 6 条不动）；需要完整内容可重新调用 |
| 71 | tools schema 每轮、每次迭代都重建 | 工具集固定后的无谓开销 | 缓存 schema 列表与 token 估算 |
| 72 | 迭代上限耗尽只写日志 | 回复戛然而止，模型以为任务已完成 | 追加一条 user 角色提示：先总结进展，再询问是否继续 |
| 73 | 「流未收到 done 事件」日志连写两行 | 日志噪音 | 去重 |
| 74 | `GraphMemoryStore.close()` 只置 None 从不 close | 每次「应用配置」泄漏一个 LanceDB 连接 | 显式 close + 重置 `_vec_tried` |
| 75 | 工具层 RAG 引擎初始化在 try 之外 | 知识库不可用时**所有工具（文件/shell/web）一起失效** | 整段包 try，失败仅跳过 RAG 工具 |
| 76 | `shell_run` 的 `timeout` 无上下限钳制 | 传 99999 → 阻塞 27h 且无法取消；传 0 → 立刻 kill | 钳制到 [1s, 600s] |
| 77 | `web_scan` 流式失败 `except Exception` 静默重试 | 真实超时后重复请求（最坏 2×30s）、已读内容丢弃 | 仅 `AttributeError`（无 stream 实现）才回退 |
| 78 | `_reranker_state` 打分失败后停在 `"done"` 而模型为 None | 状态机失真 | 同步置 `"failed"` |
| 79 | 每 chunk 重复计算 `_hash_source` | 写入侧纯浪费（同批同源） | 批次内缓存 source→hash |
| 80 | SYSTEM_PROMPT 含模型已知的协议说明与冗余反例 | 每轮浪费 ~100 tok | 精简；补充「语义检索 query 语言」与「工具报错重试」两条关键约束 |

### 第七轮：内核重构（三态自动机）

把 `agent.py` 的单体 400 行循环拆成「内核只控制流」的分层结构，详见
[docs/KERNEL_REFACTOR.md](docs/KERNEL_REFACTOR.md)。`Agent` 退化为门面，
历史交给 `Session`、能力交给 `code_run`/`shell_run`、经验交给 `memory`/`skills`。
旧行为全部保留，`run()` 签名与 `AgentCallbacks` 不变。

### 第八轮：内核实现复查

| # | 问题 | 影响 | 修复 |
|---|------|------|------|
| 81 | `max_turns` 用**会话累计**轮次判定 | 累计 15 轮后 Agent **彻底不再调 LLM** | 改为「本次 run()」的轮次计数 |
| 82 | reask 分支先回滚再写回 | 最终答复轮丢失，L4 蒸馏缺结论 | 不在回滚分支里 finish_turn |
| 83 | 周期性 CONSOLIDATE 传 `force_fold(0)` | **完全空操作**，turns 单调增长 | 传 `consolidate_every` |
| 84 | 预算漏算 `tool_calls` 参数 | 并发多工具静默突破 30k | 计入参数体 |
| 85 | `system`/`tools` 超支不从 RAW 扣 | 30k 只是记账，不真正约束 | 超支转嫁 RAW |
| 86 | `estimate_tokens` 密度过乐观 | 同上 | 收紧 3.0/1.5 字符/token |
| 87 | 上下文超长被当致命错误 | 兜底收缩永远走不到 | 单独分类并触发兜底 |
| 88 | 蒸馏线程无并发上限、退回共享 httpx | 并发蒸馏 + 非线程安全客户端 | 非阻塞锁 + 独立抽取客户端 |
| 89 | `skill_save` 用 `scan(force=True)` | 保存的技能**永远进不了索引** | 改用 `mark_stale()` |
| 90 | `code_run` 超时后不关缓冲区 | 泄漏线程往**真实** stdout 写 | 超时即 close，其下次写入自行退出 |
| 91 | 删除的知识库文件被 Step 7 复活 | 日志打「已清理」但仍在索引里 | 清理一并删 Markdown 缓存 |
| 92 | `content_hash` 只写不读 | `cp -p` / 解压导致**永久漏同步** | 判为「未修改」时再核一次源指纹 |
| 93 | `_on_fetch_models` 覆盖运行中 QThread | Qt6 `qFatal` **进程 abort** | 加并发守卫 |
| 94 | `closeEvent` 先 `terminate()` 再 `shutdown()` | **关窗口挂死** | 标记后跳过取锁的释放 |
| 95 | 「终止对话」无兜底 | 阻塞在长工具时所有按钮永久禁用 | 30s 看门狗强制收尾 |
| 96 | `_rebuild_agent` 失败留下半死 agent | MCP 全挂、httpx 已关 | 失败即清空引用 |
| 97 | 流式每 chunk 两次全文档光标操作 | 100 tok/s 时 UI 明显卡顿 | 40ms 合并刷新（~25fps） |
| 98 | 日志 handler 用模块级全局且不置空 | 多窗口误删、handler 常驻 | 用实例引用 + 置空 |
| 99 | docx/xlsx/pptx 句柄不关闭 | 4 worker × 数千文件累积句柄 | `try/finally` 显式关闭 |
| 100 | `skill_retriever._llm_pick` 用 `ev.get()` | LLM 择优是死代码 | 按 dataclass 属性访问 |

### 第九轮：失败自动恢复 + 全量复查（bug / 泄漏 / 性能 / 体验）

**失败自动恢复**（用户需求：计划中某项任务失败时，自动依据失败信息找方案解决并继续后续任务，而不是停下等用户输入「继续」）：

- 失败感知反思门：本次 run 发生过失败时，提醒预算 2→8 次，且提醒引用最近一次失败的**具体错误**与**已试过且失败的方案清单**，明确要求「读完整错误 → 换没试过的方法 → 至少 3 种方法都失败才标 blocked」，禁止停下来问用户是否继续
- 「只规划不执行」门：建完计划但本段一次实际工具调用都没有就想收尾时，预算走 6 次（默认档仍是 2 次），且提醒点名「你还一次工具都没调用」+ 直指 Now 那一步。`update_plan(op="create")` 的结果里也会直接下达「▶ 现在立即开始执行第 1 步」。修复前这类场景第 3 次文字就被放行，正是「规划完不执行、必须手动说继续」的成因
- 首次失败即干预：第一次失败就要求判定错误类别（参数/路径/权限/依赖/语法/网络）并换方法，禁止原样重发、禁止把错误念给用户听（旧实现要等连续失败 2 次才提示）
- 原样重试检测：同名工具 + 相同参数再次失败 → 点名「禁止重复」，要求换参数/换工具/用 `code_run` 写诊断脚本
- 连续失败提示：连续失败 ≥2 次 → 要求先诊断再换方案，同一思路只试一次
- 防过早 blocked 守卫：失败尝试 <2 次就把步骤标 `blocked` → 提醒恢复 `doing` 再试（守卫有上限，模型坚持放弃时最终尊重）
- 系统提示 rule 8 与 `update_plan` 描述同步强化；回归测试 `tests/test_failure_recovery.py`

**自动续跑与任务边界（消灭「必须人工输入继续」）**：

- 自动续跑：两类停车不再甩回给用户——① MAX_TURNS 截断（长任务跑满轮次上限）；
  ② 模型提前收尾但计划仍有未完成步骤（反思门提醒额度用尽）。每段续跑都是
  完整的内核 run（取消/反思门/预算全部生效），最多续 4 段（默认每段 15 轮 →
  合计最多 75 轮），额度用尽才交回用户（`agent._auto_continue` 可关闭）
- 失败上下文跨段传递：续跑段通过 `kernel.seed_failures(snapshot)` 接住上一段的
  `_failed_total / _last_fail / _fail_sigs`，因此续跑段仍享有失败感知的反思门预算，
  且续跑首轮的锚点会带上「上一段最后一次失败：工具 → 错误」。修复前续跑段记账被
  重置，模型不知道上一段卡在哪，只能瞎试后再次停住
- 反思门提醒补全出口：明确告诉模型「若该步骤的工作**实际已完成**，用
  `update_plan(op=patch, status=done)` 收尾即可，不要为响应提醒而重复已完成的工作」
  ——修复「实际都做完了但面板一直显示有未完成项，反复要求打继续」
- 任务边界 fork 提示：新请求到来而旧计划仍有 open 步骤时，run() 注入一次性提示，
  强制模型显式判断「新任务 → create 替换计划 / 继续旧任务 → 接着做」——修复
  「新任务的计划与输入无关（沿用了上一个任务的旧计划）」
- `update_plan` patch 用错 step_id 时，报错附上当前计划渲染，模型一次自纠

**复查修复**：

| # | 问题 | 影响 | 修复 |
|---|------|------|------|
| 101 | MCP `stdin.write()` 无超时且持锁 | server 不读 stdin 时 agent 线程**无限期挂死** | 专用写线程 + 有界队列（满/超时抛 MCPError），写失败立刻唤醒全部等待者 |
| 102 | RAG ingest 互斥锁是实例级 | UI 同步与 agent 的 `rag_ingest` 是两个 engine 实例，锁完全无效 | 模块级按 `rag_dir` 键控的跨实例锁 |
| 103 | `graph.clear()` 不清待重试队列 | 用户删掉的事实被向量检索**复活** | `clear()` 一并清空 `_pending_index`（带锁） |
| 104 | 事实抽取结果不校验内部结构 | 一个坏实体让**整轮抽取全部丢弃** | 逐条归一化：坏条目跳过、好条目保留 |
| 105 | `memory_graph` 的 clear/delete 不弹确认 | 模型一次误调用清空长期记忆 | `should_confirm` 对破坏性 op 返回 True |
| 106 | `observations` 传字符串被按字符迭代 | 生成几十条逐字符 observation 污染图谱 | 入口类型校验，明确报错 |
| 107 | WorkMemory LRU 淘汰不传 `force` | 防抖窗口内的写入被**静默丢弃** | 淘汰路径 `flush(force=True)` |
| 108 | `update_plan` patch 无 step_id 时污染 note | 第一步的结论被覆盖、目标步骤漏改 | 先筛目标集合再统一写入；只给 note 必须带 step_id |
| 109 | patch 原地改 `get_plan()` 活字典 | 校验失败时计划半改 | 深拷贝后改，一次性 `set_plan` |
| 110 | append 的 id 回退撞号 / create 强升 doing | patch 一次改中两步 / 显式状态被覆盖 | id 跳过已占用；仅全 pending 时提升 |
| 111 | `skill_save` 临时文件名只含 pid | agent 线程与蒸馏线程并发写同一 tmp → 产物损坏 | `mkstemp` 唯一名 + finally 清理 |
| 112 | `skill_search` list 无上限 | 技能多时撑爆上下文 | 截断 50 条 + 总数提示 |
| 113 | checkpoint 无 session 时静默成功 | 模型以为已写入，实际什么都没存 | 显式返回失败 |
| 114 | shell_run 取消后子进程继续跑 | 用户点终止后命令最长再跑 600s | 0.2s 粒度轮询取消标志，命中即杀进程组 |
| 115 | shell_run reader join 后无条件 close 管道 | 孙进程持管道时 **close 永久挂死**工具线程 | 仅 reader 已结束时才关管道 |
| 116 | shell_run 无条件对 pgid 发 SIGKILL | pgid 被 OS 回收复用时**误杀无关进程组** | 组主已退出时先信号 0 探测组是否仍存在 |
| 117 | code_run 取消后线程跑满 timeout | 占 CPU 且 sys.stdout 被持续替换 | 轮询取消立即收尾 + PermissionError/SystemExit 统一带 traceback |
| 118 | `ToolRegistry.shutdown()` 裸清单例 | 与懒初始化竞态 → 每次应用配置泄漏 ~137MB | 单例清空持 `_singleton_lock` |
| 119 | skill embedding 失败回退 `None` | SkillVectorStore 悄悄自建第二个默认模型单例 | 显式回退默认模型单例 |
| 120 | FastEmbed 懒加载无锁 | 首次并发各加载一份 ~137MB ONNX | 进程级加载锁 + 双重检查 |
| 121 | FTS 建索引跨实例并发 | 写入侧与检索侧对同一张表并发 replace | 模块级建索引锁 + 「已存在」容错 |
| 122 | reranker 后台加载/close 裸写共享状态 | 已 close 的实例「复活」持有 ~500MB 权重 | 状态写回/close 全部持 `_reranker_lock` 并复查 `_closed` |
| 123 | documents 表 doc_id 用 basename | 不同子目录同名文件互相覆盖/误删 | doc_id 改用 `md_path`（唯一），兼容清理旧影子行 |
| 124 | 语义切片负缓存 TTL 从未生效 | 离线时每个超长 section 各重试一次模型加载 | embed 真实失败记时间戳，同一 TTL 节流 |
| 125 | MOBI latin-1 回退被 `del data` 废掉 | 非 utf-8 MOBI 静默返回空 | 不再提前释放字节，回退可用 |
| 126 | 取消后仍执行 Step7 全量 digest 重刷 | 同步已取消还多跑几秒 | cancelled 时跳过 |
| 127 | `delete_by_source` 恒返回 1 | 与 docstring 语义不符 | 前后 count 差值尽力统计 |
| 128 | `rag_ingest` force 字符串 `"false"` 为真值 | 意外触发全量重建 | 严格布尔解析 |
| 129 | file_write 现场翻转进程 umask | 非线程安全，窗口内他线程建文件权限 0666 | 导入期缓存 umask |
| 130 | file_read 按字节数判截断 | 多字节 UTF-8（中文）被误判截断 | 按字符增量读取判定 |
| 131 | 图谱回填/去重全表物化（含向量列） | 记忆量大时内存与耗时线性恶化 | 列投影下推到存储层 |
| 132 | 图谱待重试队列无锁无上限 | 并发丢条目；嵌入持续失败时内存只增 | 专用锁 + 500 条上限 |
| 133 | skill 目录 mtime 探测每次检索两次全树 walk | 千级技能树每次查询 O(目录数) syscall | 60s TTL 缓存（`mark_stale` 显式标脏兜底） |
| 134 | config 字符串字段 `str()` 收敛垃圾值 | model 写成列表 → 存成 `"['x']"` | 非 str 回退默认 |
| 135 | config.save 固定 tmp 名 | 并发保存互踩 | mkstemp 唯一名 + 失败清理 |
| 136 | 强杀后流式状态残留 | **下一回合整条回复被吞**（用户看到空回复） | 回合开始统一复位 stripper/流式状态；强杀先 flush |
| 137 | 强杀丢弃运行中 QThread 引用 | wait 超时时 sip 析构 → **qFatal 进程 abort** | 超时挂孤儿列表，closeEvent 再统一 wait |
| 138 | `_worker_terminated` 永不复位 | 之后每次退出都跳过资源释放（泄漏 MCP/连接池） | `_rebuild_agent` 成功后复位 |
| 139 | 连点「终止」重置看门狗倒计时 | 30s 强杀可被无限推迟 | 仅在未激活时 start |
| 140 | 推理流逐 chunk 全视口重绘 | 推理模型 UI 明显卡顿 | 复用 40ms 合并刷新（与正文同节奏） |
| 141 | 工具入参不截断 | 一次大 file_write 把 MB 级文本常驻 QTextDocument | 与 tool-end 一致截断 400 字符 |
| 142 | 日志/流式输出无条件吸底 | 用户选中文本复制被反复打断、无法上翻 | 仅吸底时跟随；插入用临时光标不碰用户光标 |
| 143 | `_on_finished` 忽略 wait 结果 | 与 137 同构的析构 abort 风险 | 超时挂孤儿列表 |
| 144 | 计划面板跨线程读 Session | 渲染出半新半旧的中间态 | `render_plan_dict` 纯函数 + 信号快照渲染 |
| 145 | busy 时按 Enter 静默丢弃 | 像输入框坏了 | 状态栏提示「正在运行中」 |
| 146 | 清空对话无 busy 守卫、不清计数器 | 运行中清空破坏会话；token 计数残留 | 加守卫 + 复位计数 |
| 147 | Ctrl+Enter 也发送、按键事件未 accept | 误发送；事件向父控件意外传播 | 仅裸 Enter 发送 + `e.accept()` |

### 第十一轮：全量代码重审（正确性 / 泄漏 / 进程稳定性）

对全部 45 个源文件逐文件重审，详见 [软件设计文档 §13](docs/软件设计文档.md)：

| # | 问题 | 影响 | 修复 |
|---|------|------|------|
| 148 | `_BoundedIO.write` 截断快速路径绕过 closed 检查 | close 后 write 仍返回 0 → code_run 超时/取消的泄漏线程**永不退出**，往真实 stdout 无限刷屏，解释器退出时持 stdout 锁 → `Fatal Python error` 进程 abort（pytest exit=134） | write 在任何快速路径前显式检查 `self.closed` |
| 149 | code_run 超时/取消路径先恢复全局 stdout 再 close 缓冲 | 泄漏线程的 print 动态查找到真实流，永远碰不到已关闭缓冲（与 148 叠加导致自杀机制整体失效） | 超时/取消路径 close 后延迟 50ms 恢复全局流（泄漏线程撞 ValueError 退出）；正常完成路径仍立即恢复 |
| 150 | 50ms 恢复窗口内新一场 code_run 把上一场已关闭缓冲当外部流恢复 | 全局 stdout 被永久钉死在死流上 | 模块级真实流缓存 + 快照识别 stale；延迟恢复仅在全局流仍指向本场缓冲时动手 |
| 151 | `engine._chunk_iter` 把 `stats["errors"]` 列表覆盖成 int（且不持锁） | 文件在同步中被删 → 收尾 `len(errors)` TypeError，**整轮已成功的同步以异常告终** | 持 `stats_lock` append 错误记录，与其他错误路径同构 |
| 152 | `_work_memory` 的 `move_to_end` 在锁外调用 | 与锁内 `popitem` 并发修改 OrderedDict → RuntimeError / 条目错乱 | 命中/新建/淘汰三者同锁 |
| 153 | code_run 护栏对非 str/PathLike 的 `open()` 参数走 `realpath("")`（=cwd） | 绕过前缀检查后炸出与根因无关的底层 TypeError，模型无法自纠 | 明确报 `open() expects a path (str/PathLike)` |
| 154 | UI 不校验 `request_timeout ≤ 0` / `temperature < 0` | timeout=0 → httpx 立即超时，所有请求必失败且无提示 | 钳到合法下限 + 复位输入框 + 状态栏点名 |

### 第十三轮：RAG 主线同步（向量库 / OCR / 切片 / 调度）

对照 Base-Aug-Agent 主线对 RAG 子系统做了一次完整同步，详见[软件设计文档
§15](docs/软件设计文档.md)。两侧自 #88 起分叉，主线又积累了约 130 个 RAG 提交；
本轮把其中仍然成立、本工程尚未吸收的部分全部移植过来（主线自身已修掉的
一次性诊断脚本、线程数自适应的实时阶段等已过时项不收）。

| # | 领域 | 移植内容 |
|---|------|---------|
| 197 | 向量库 | `@_store_op` 在途操作计数：`clear()`/`close()`/`reload()` 先排空在途 add/search 再动表（旧实现与在途写并发会打到已 drop 的表）；`close()` 不再调用 `.cpu()`（ONNX 无该方法）；`_fts_failed` 让建索引失败后不再每轮检索重试 |
| 198 | 向量库 | `search_fts` 空查询短路；BM25 原始分 `s/(1+s)` 归一化进 `score`（旧实现把 BM25=15 当余弦 1.0 塞进 `1-s²/2`，纯关键词候选集体排末尾）；`search` 的 `where` 加 `prefilter=True`（定向检索不再被先截断再筛） |
| 199 | 向量库 | reranker 状态机：close 后拒绝加载、加载完成撞 close 置 `failed`（旧实现状态卡在 `loading` 让重排静默失效）；`_get_reranker` 锁内重查 + `timeout=5s` 等加载 |
| 200 | 向量库 | `_embed_chunks` 改为**按文本长度降序分桶**（padding 归零）+ 自适应子批 1-5 + 激活形状预算钳制 + 8GB RSS 软墙 + 子批级进度日志。主线上 32 条混杂批 309s ≈ 40 tok/s，同模型「长度排序 + 8-12 条/批」可达 ~2000 tok/s |
| 201 | OCR | rapidocr **3.x 优先**（模型目录与 docling 共享、支持 CoreML EP、懒加载），1.x 经 `ocr_engine` 注入 CoreML 回退；Apple Silicon 上用 `params={"EngineConfig.onnxruntime.use_coreml": True}`，Intel/非 Apple 直接用 CPU EP |
| 202 | OCR | 暗色主题页自动反色（黑底白字/灰字在原模型上几乎检不出文本框——大面积漏检的根因）；`_pixmap_is_blank` 跳过纯色/空白扫描页（每页省 2-5s）；`_ocr_image_cached` 兼容 3.x 的 `RapidOCROutput.txts` 与 1.x 的 `(result, elapse)` |
| 203 | OCR | 引擎**周期回收**（每 50 页）+ **每页从单例取用**：CoreML/ORT 按形状保留的原生缓存在真实扫描页上实测 ~35MB/页，400 页滞留 ~14GB；页末释放局部引用让任意 worker 的回收都全局生效。单页 OCR 后同步 `store_shrink(100)` + `gc.collect()` |
| 204 | 调度 | `_shard_files_balanced` LPT 负载均衡分片（旧连续切片在 9 文件 4 worker 时第 4 片为空，慢书扎堆拉长尾延迟）；写入批量 20 → 200（LanceDB fragment 碎片）；`stats["chunks"]` 改记本次写入量 + 新增 `chunks_total` |
| 205 | 同步 | `_ManifestLock`（flock 跨进程 + 线程锁 + TLS 重入计数）+ `_save_manifest` **读改写合并**：UI 与 agent 两个实例并发同步不再互相抹掉对方的 manifest 条目；`clear_knowledge_base` 删 manifest 也持锁 |
| 206 | 同步 | Markdown 缓存 sidecar（`*.md.meta.json`）记录「缓存文件名 → 原始文档名」，`ensure_digests` 按 sidecar → md_path → 文件名反推的优先级解析；`build_and_store_digest` 支持传入预计算的 digest/chapters（批量重刷不再算两遍）并存完整 doc_name |
| 207 | 清理 | `_cleanup_deleted_files` 逐文件 try/except 隔离 + 失败保留 manifest 条目（旧实现无条件 pop，「下次重试」永不成立，失败文件的向量永久残留）；删除范围补齐 rel_path / source 全键 + sidecar |
| 208 | 解析 | 旧版二进制格式（`.doc`/`.xls`/`.ppt`）移出 `SUPPORTED_EXTENSIONS` 与 `EXTENSION_DEPS`（解析必然失败，保留只制造「已支持」的假象并白装依赖），直接调用时给出「另存为」的可行动提示 |
| 209 | 依赖 | OCR 从 `EXTENSION_DEPS` 的静态项改为 `check_dependencies` 检测到图片版 PDF 时**动态升级**为必需（「未装」与「装了但加载失败」在依赖层无法区分，统一 optional 会让两者都只产出无差别提示）；pyproject 的 ocr extra 改 `rapidocr>=3.10`、rag extra 加 `model2vec>=0.3` |

回归测试：`tests/test_rag_sync_round13.py`（18 例）覆盖 manifest 锁的合并/removed/
重入/异常释放、LPT 分片不产生空片且大文件摊开、OCR 路由的 CoreML 门控与 3.x/1.x
回退、暗色页反色与正常页不误触发、回收边界与批量常量。

### 第十二轮：全量代码重审（OCR 失效 / 竞态 / 加载校验 / 性能）

对全部源码再次逐文件重审（两个子代理并行审查技能/记忆/RAG 子系统 + 主代理审查
内核/UI/工具），详见 [软件设计文档 §14](docs/软件设计文档.md)。本轮最严重的发现是
**OCR 功能在两个层面上从未真正工作**：CoreML EP 注入的模块清单不匹配
rapidocr 1.2.x 的真实布局，且其 kwargs 体系变更会让引擎构造直接抛
`KeyError`——图片型 PDF 的识别在装了 rapidocr 的环境里也 100% 失败。

| # | 问题 | 影响 | 修复 |
|---|------|------|------|
| 197 | OCR CoreML EP 注入的候选模块清单不含 `rapidocr_onnxruntime.utils`（1.2.x 真实布局），且强制要求 `_get_ep_list`（1.2.x 没有） | Apple Silicon 上 CoreML 加速**从未生效**，日志只打「找不到 OrtInferSession」后静默回退 CPU | 候选清单补 1.2.x 布局；按「模块里有 OrtInferSession」判定；新增 factory 注入路径（包装模块级 `InferenceSession`，在 providers 里注入 CoreML），`unpatch` 两条路径都能恢复 |
| 198 | rapidocr 1.2.x 的 `UpdateParameters` 收到任何 `det_*` kwargs 都无条件读 `det_dict['model_path']` | 只传 `det_limit_side_len` 就抛 `KeyError: 'model_path'` → **OCR 引擎永远建不起来**（提示却是「未安装」） | `create_ocr_engine` 增加降级链：完整 kwargs → 补 `det_model_path=""` → 裸构造；OCR 可用性优先于分辨率调优 |
| 199 | `skill_search.py` 镜像写入的 except 分支引用未定义的 `logger` | 主写成功后的镜像失败抛 `NameError`，`skill_save` 整体报失败且原始错误丢失 | 补 `import logging` + `logger` |
| 200 | `skill_load` 用 `.agent/skills` 当基线判断「技能目录是符号链接」，而 `abs_dir` 来自语义检索器（根是 `skills/`） | 每一个正常技能都被打上「内容来自工作区之外」的错误横幅 | 基线改用检索器自己的 `skills_dir`（新增公开 property） |
| 201 | `skill_save` 镜像到 `.agent/skills` 只写 `SKILL.md`，而 `SkillIndex` 要求同目录有 `skill.json` | 镜像目录永远进不了关键词索引，回执里「已镜像以便关键词检索命中」是空话 | 镜像同时写 `skill.json`（name/description/keywords/entry） |
| 202 | `SkillVectorStore._get_ef/_get_db/_get_table` 完全无锁 | 并发首次调用各建一个 LanceDB 连接（后写者覆盖，先建的永不 close）；`_table` 赋值竞态 | 双检锁；`_lock` 改 RLock（build/sync 持锁期间会调这些 getter） |
| 203 | `skill_retriever.sync()` 在取锁**前**拿表，且更新项先 delete 再 add | 与并发 build/close 交错时打到已 drop 的表；add 失败时技能从索引永久消失且无回滚 | 锁内重新取表；更新项改「先 add、再按旧 fingerprint 精确删」（不能按 dir 删——会连新行一起删） |
| 204 | `SkillVectorStore._distance_to_similarity(None)` 按距离 0 算成满分 | 缺 `_distance` 的行必过 0.5 阈值并排最前，排序失效 | None/非数字返回 0.0 |
| 205 | `_parse_frontmatter` 要求 `text.startswith("---")` | 带 UTF-8 BOM（Windows 编辑器常见）的 SKILL.md frontmatter 不可见 → 技能被静默丢弃 | 先 `lstrip("\ufeff \t\r\n")` |
| 206 | `skill_index._read_skill_meta` 只捕 `OSError/JSONDecodeError`；`_dir_mtime`/`scan` 的 `os.scandir` 未包 try；`_load` 不校验记录类型 | 一个手写坏的 skill.json（name 是数字/keywords 是字符串）或目录权限变化就让整次 scan/search 失败 | 逐字段 isinstance 校验 + 异常面补 `TypeError/AttributeError`；scandir 包 try；非 dict 记录过滤；`list_all` 返回拷贝；同分排序加 name 次级键 |
| 207 | `skills.py` 的 `skill.json`/`prompt.md` 用裸 `open()+write()`；目录唯一性靠 `while os.path.exists`（TOCTOU）；`keywords=None` 直接 TypeError | 崩溃留下半截 skill.json（SkillIndex 静默跳过）；两个线程同名技能互相覆盖 | 统一走 `atomic_write_text`；`makedirs(exist_ok=False)` + `FileExistsError` 重试；入口处 `keywords or []` |
| 208 | `SkillManager.__init__` 的损坏修复只改内存不落盘 | 下次 reload/重启又从磁盘读回坏值，防御每轮白修 | 修复后 `flush(force=True)` 并复位 `_last_save`（防止把随后真正的 set() 吞进防抖窗口——后者是本轮修复自身引入并被回归测试抓住的回归） |
| 209 | `_request_key` 把长 key 截断到 80 字符 | 两个只有尾部不同的长请求碰撞成同一计数键，产生虚假「建议固化」 | 超长 key 补全文 sha1 后缀；短 key 格式不变（兼容已有计数） |
| 210 | `graph_memory._load` 对 relations 完全不校验、entities 值不校验、根节点不校验 | 手改坏/旧版本的 JSON（只要能解析）就让 `_prune_locked`/`snapshot`/`list_relations` 抛异常，**整个记忆系统不可用** | 归一化：非 dict 根按空处理；实体值必须 dict 且 observations 必须 list；关系必须 source/target/label 齐全 |
| 211 | `document_outline._HEADING_RE` 用 `\s+`（匹配换行）且作用于整篇文本 | `#\n标题` 这种空 ATX 标题行把下一行正文吞并成章节标题 | 改 `[ \t]+` |
| 212 | `cleaner` 逐字符 `isalnum()/isspace()` 统计 | 数 MB 文档的清洗阶段是纯解释器级 O(n) 迭代 | 预编译正则 C 层计数（语义等价，纯分隔行仍被分隔符规则收掉） |
| 213 | `risk._SHELL_REDIRECT_RE` 只匹配 `>/` | `> ~/.bashrc`、`>> ../outside` 两种工作区外写入零确认执行 | 放宽为 `>>?\s*(?:~|\.\./|/)`（`./` 仍不算风险）；路径 token 的尾部标点剥离补 `!?]>` |
| 214 | `MCPClient.start()` 不清理上一次连接的残留 | `_fail_pending` 置 `_running=False` 后重启：旧写队列里的 `None` 哨兵被新写线程立刻消费并退出 → 之后所有 `_call` 干等 30~300s 超时；旧 writer 还会把新连接再次判死 | `start()` 先排空写队列 + 清 `_pending/_results`；`stop()` 换全新写队列隔离卡死的旧 writer |
| 215 | `MCPTool` 把模型多传的未声明参数原样转发 | 严格校验的 MCP server 直接失败，错误还指向远端 | 按 schema properties 过滤未知参数（properties 为空时不过滤） |
| 216 | MCP 重名工具被静默吞掉且无日志 | 用户以为两个 server 的工具都挂了，实际只注册了先连上的 | 记 warning 说明冲突 |
| 217 | `parsers._extract_pdf` 信号量内的「双重检查」是死代码（局部变量其它 worker 改不到） | 无（误导性注释） | 删除死分支 |
| 218 | ebook-convert 返回码 0 即采用，即使输出为空 | 静默跳过承诺的 raw 回退路径，文档变零内容 | 仅非空才采用，否则继续回退 |
| 219 | `_get_ocr_engine` 对非 ImportError 的初始化失败无负缓存 | 模型下载失败时每个图片型 PDF 都重试一遍完整加载（可能数分钟） | 60s 冷却期；`release_ocr_engine` 同时清冷却 |

回归测试：`tests/test_round12_fixes.py`（24 例）覆盖以上各项；
`tests/test_ocr_engine.py` 重写为同时覆盖 `_get_ep_list` 与 factory 两种注入路径
（含 rapidocr 1.2.x 的 `KeyError` 降级与裸构造回退），并用「记录器子类」在真实
rapidocr 模块上验证 CoreML providers 真的排到最前、CPU 保留作回退、unpatch 恢复原厂。

### 第十轮：全量复查（安全 / 泄漏 / 正确性 / 性能 / 体验）

三个模块并行深审（RAG、agent 编排层、工具层与 UI）后按严重度修复。核心是三类：
**风险分级的两类绕过**、**三处持久化的并发写损坏**、**自动续跑把防死循环预算放大多倍**。

| # | 问题 | 影响 | 修复 |
|---|------|------|------|
| 148 | 风险分级可被一句话绕过：`_SHELL_VERB_RE` 只有 `rm/mv/cp…`，`python3 -c "shutil.rmtree('/Users/x')"` / `dd if=… of=/dev/rdisk3` / `curl … \| python3` 全部**零确认直接执行** | 破坏性命令绕过人工确认 | 动词表补磁盘级操作与 Python 侧等价调用名；`_PIPE_EXEC_RE` 目标扩展到 `python/node/perl/ruby/php` |
| 149 | `code_run` 写护栏只包 `open`，`os.symlink/os.rename/os.link/shutil.move` 既拦不住也不弹确认 | 改写系统文件拓扑无感知 | 风险分级记号补齐这些调用（与工作区外路径同现即弹确认）；文档说明「护栏是提示性的、第一道防线是风险分级」 |
| 150 | `skill_load` 可经符号链接升级成任意文件读取（`ln -s /etc <ws>/skills/x` 后读 `passwd`） | 任意文件读取 | 外部技能链接目标必须「长得像技能目录」（顶层含 SKILL.md/skill.json/prompt.md）；读取结果标注真实来源 |
| 151 | `memory` / `graph_memory` / `skill_index` 三处持久化都用固定 `.tmp` 名 | 并发写互相 truncate → 落盘**混合 JSON**，`_load` 失败后按 `{}` 初始化 = 数据静默丢失 | 新增 `agent/atomic_io.py`（mkstemp 唯一名 + 权限继承 + fsync + finally 清理），三处统一收敛 |
| 152 | `_load` 解析失败后静默按空数据继续，下次 `_save` 把坏文件覆盖 | 手工编辑写错/版本不兼容 → 用户数据无从排查地丢失 | 失败时先把坏文件改名备份为 `.corrupt-<时间戳>` 再空初始化 |
| 153 | 知识图谱实体/observation 只增不减，且每次写入全量序列化 | 长跑 workspace 内存与 JSON 体积单调增长，写放大 O(N) | 每实体 200 条 / 总 5000 实体 / 20000 关系上限（按「最近优先」淘汰，同步清理向量行）；写入加 0.5s 防抖 + `flush()` + atexit 落盘 |
| 154 | 自动续跑把防死循环预算放大多倍：`run()` 每段清零全部 nudge 计数，最多续 4 段 | 单次用户输入最坏 4×15=60+ 轮 LLM 调用、无墙钟上限，推理模型下几十分钟不返回 | nudge 计数与 `_did_work` 并入 `failure_snapshot()` 跨段传递 |
| 155 | `hooks.finished` 每个续跑段都触发一次 | 单次输入最多 8 次冗余后台 LLM 调用 + 向图谱重复写入近似实体 | 内核 `run(emit_finished=False)`，由 `Agent.run` 在整个用户轮次结束后只触发一次 |
| 156 | 一次性提示在 rollback 时被永久吞掉（`anchor_prompt` 里 pop，该轮随即被 rollback） | 自动续跑注入的「上一条被判提前收尾」模型**永远看不到**，只能收到泛化提示 | 改为 peek 不消费，`finish_turn` 真正落地后才清除 |
| 157 | `simple` 豁免让反思门对短多步任务完全失效（「帮我重构 foo.py」不含多步关键词且 <60 字） | 任务在无计划、无监督状态下跑完 | 豁免条件补多步动词表（重构/批量/部署/迁移/搭建…） |
| 158 | resume 段 `_did_work=False` | 反思门文案变成「你还一次工具都没调用」（对已干过活的段是错的，会误导模型），失败感知预算从 6 掉回 2 | `_did_work` 跨段传递；提醒文案按 `_resume` 分支 |
| 159 | `close()` 与自己的文档相反：拿不到锁仍然裸关 `httpx.Client` | 后台抽取/蒸馏线程正在流式使用的连接被关，半路炸掉且连接未归还 | 只有拿到锁才关闭 |
| 160 | UI 回调未包 try/except | 一个 UI 异常把整轮变成一句「LLM error」，看不出是界面问题 | `_emit_ui` 统一包装并降级为日志 |
| 161 | 连续空回复 → 返回空字符串 | 用户看到一个完全空白的回复 | 耗尽重试后返回可见提示 |
| 162 | `has_summary` 恒为 True | 「必须输出 `<summary>`」提醒是死代码，L2 摘要纪律实际失效 | 改为 `has_summary_tag()` 直接问模型有没有按协议写标签 |
| 163 | `max_iterations` 同步读的是 `__init__` 快照 | UI 改 `config.max_iterations` 后不生效（「改了最大轮次却没反应」） | 改成 property：读实时取 config，写也落 config |
| 164 | `_append_summary_to_memory` 用固定 `.tmp` 名 | 并发收缩上下文时写坏 memory.md，失败留僵尸 tmp | 统一走 `atomic_write_text` |
| 165 | `on_plan_updated` 传 Session 活字典给 UI 线程 | 正是 `session.py` 里列为已修 bug 的跨线程共享可变状态模式 | 传深拷贝 |
| 166 | `deps` 的模型缓存探测靠 `TextEmbedding(lazy_load=True)` | fastembed 部分版本的下载逻辑在基类 `__init__`，`lazy_load` 只跳 ONNX Session → **依赖检查可能触发 2.3GB 下载** | 纯文件系统探测（支持 hub 缓存 `models--{org}--{repo}/trees/*.json` 与旧布局），不实例化模型 |
| 167 | 手动下载指引指向 `~/.cache/huggingface` 且仓库名少了 `-Q` 语义 | 用户照做后 fastembed 仍看不到模型，报错依旧 | 指引改为落到 `~/.cache/fastembed` 并说明模型名→仓库名映射 |
| 168 | 探测只看 `onnx/model.onnx` 存在 | 一次中断的下载只留下 656KB 外部图，被判成「已缓存」→ 跳过预下载 | 按清单校验每个文件的 blob 完整性（排除 `.incomplete`） |
| 169 | 装完包后重跑完整 `check_dependencies()` | 内含整树 rglob 扫描 + 逐本打开 PDF 抽前 10 页，最贵的一步做两遍 | 只复查刚装的那些包的导入情况 |
| 170 | `install_packages` 用 `sys.executable -m pip` | PyInstaller 冻结环境下必然失败（app 二进制没有 pip） | 优先 `sys._base_executable`；拿不到时明确报错 |
| 171 | `rag_rerank_enabled` / `rag_rerank_model` 两个配置项从未被读取 | 用户关重排序或换模型完全无效 | 透传到 `RAGEngine`，关闭时短路到混合排序且不加载 reranker |
| 172 | `close()` 未复位 `_documents_table` | 实例复活后持有旧连接的表句柄，upsert 对新连接不可见甚至失败 | 一并置空 |
| 173 | reranker 分数按「本批候选」自适应 sigmoid + `math.exp(-s)` 大负数溢出 | 同一 chunk 在不同查询下可能走/不走归一化，`0.7` 绝对阈值语义漂移；溢出被吞后静默回退 | 按模型固定归一化（首次探测一次并缓存）+ 数值稳定 sigmoid |
| 174 | `**sig_now` 在文件于同步过程中被删时抛 `TypeError` | 一次抖动让剩余文件向量全部丢弃、整个 manifest 作废 | `sig_now is None` 时跳过该文件并计入 errors |
| 175 | `delete_document` 只按 doc_id 删 | `upsert` 在 md_path 为空时按 basename 建键，那些行永远删不掉 | 同时按 `doc_id` 与 `doc_name` 两种键删除 |
| 176 | `search` 循环变量 `source` 覆盖函数形参 `source` | 定向检索过滤条件被覆盖成最后一条结果的文件名 | 改名 `src_name`；回退结果缺键改用 `.get()` |
| 177 | `_save_manifest` 固定 `.tmp` 名 | 并发写坏 manifest | 统一走 `atomic_write_json` |
| 178 | `parsers` EPUB 不关 book | ebooklib zip 句柄在 4 worker × 数千文件下累积（Windows 锁文件） | try/finally 关闭 |
| 179 | OCR 信号量无超时 acquire | 单页 OCR 卡死让所有 worker 连同 daemon 线程永久挂起，进度不再动且无报错 | 300s 超时，拿不到就跳过该页并告警 |
| 180 | `_ChonkieEmbeddingAdapter.dimension` 触发一次真实 embed | chonkie 构造期访问它 → 在 ingest worker 里强制加载 ONNX 模型（全局锁串行化，拖慢整体） | 先用已知维度占位，首批 embed 后用实测长度刷新 |
| 181 | `skill_retriever` 扫描时整文件读入 | 误放入的几百 MB 文件直接进内存后才截断 | 读到上限+1 再截断 |
| 182 | `skill_retriever.build()` 先 drop 旧表再分批 embed | 第一批之后失败 → 留下半张表且 `_table` 未赋值，下次打开半张表，检索静默残缺 | 先建 staging 表、全部成功后复制数据再切换，失败只清 staging（旧表始终完好） |
| 183 | `graph_memory._deindex_observations` 全表物化只为收集拼写集合 | 删除延迟随图谱线性增长 | 保留列投影；新增按 observation 粒度的 `_deindex_observation`（淘汰时用） |
| 184 | `skill_save` 只写 `workspace/skills`，而关键词检索扫 `.agent/skills` | 刚固化的技能 `skill_search`（关键词）永远召不回，回执却声称都能召回 | 镜像一份到 `.agent/skills`（不覆盖用户已有版本），回执如实说明 |
| 185 | `cleaner` 用 `&[a-zA-Z]+;` 一刀切 + 短数字行宽条件 | "AT&T;"、"R&D;" 与 "AT&T; 财报" 这类正文被误删 | 只替换真实存在的 HTML 实体名；页码行仅删独立成行的纯数字 |
| 186 | `file_write` 对 content 体积无上限 | 模型传几十 MB 字符串一次性进内存并全量落盘 | 1M 字符上限，超限提示改用 `shell_run` 分块/流式写 |
| 187 | `MCPTool` 不兜底畸形 inputSchema | server 返回 `"properties": null` 时 schema 发给 LLM API 可能被整体拒绝（那轮所有工具不可用） | 校验 `properties`/`required` 类型并回退空值 |
| 188 | `ask_user` 裸访问 `registry.callbacks` | headless/测试下 AttributeError 被兜底成难懂错误 | `getattr` 链 + 明确错误文案 |
| 189 | `web_scan` 只校验初始 URL 协议，`follow_redirects=True` | 公网 URL 可被 302 引到内网/其它主机（SSRF 面） | `max_redirects=5` 收紧 |
| 190 | `update_plan` patch done 时把「第一个 pending」提为 doing | 被标 done 的步骤前面还有 pending（被跳过）时，`Now:` 指回更早步骤，模型反复做同一件事 | 只提升目标步骤**之后**的第一个 pending |
| 191 | append 超限提示说反了 | `normalize_plan` 保留 `raw_steps[:20]`，丢掉的是**最新**追加步，文案写「最早 N 步被丢弃」 | 文案改为「最新追加的 N 步被丢弃」并建议重建更聚焦的计划 |
| 192 | 中文输入法组字时按回车直接发送 | 中文用户每确认一个候选词就误发半条消息 | 放行 IME 预编辑态（`text()` 为空）的 Return |
| 193 | 首个流式 chunk 后又重置 `_SummaryStripper` | 首块以 `<summ` 结尾时 carry 被丢弃，下一 chunk 的 `ary>` 被当正文显示 | 删除多余重置（每回合的重置已在 `_on_send`） |
| 194 | 重试守卫不在成功轮后重置 | 长任务后期偶发空回复被过早判成 "done" 收尾 | 一轮被真正提交（`finish_turn`）后才清零，重试轮不清 |
| 195 | LLM 首字节前失败重试 `time.sleep(delay)` | 最长 3s 完全不响应「终止对话」 | 0.1s 粒度分段休眠并轮询取消 |
| 196 | 工具 schema 获取失败静默降级成空 | 模型在无法调用任何工具的状态下空跑，用户只看到反复文字回复 | 记日志 + UI 错误提示 |

回归测试：`tests/test_kernel.py` 新增「一次性提示必须活过 rollback」用例；
`tests/test_confirm_policy.py` 覆盖新增的绕过向量；`tests/test_rag_manifest.py`
覆盖 `delete_document` 双键删除。

### 第十四轮：全量代码重审（正确性 / 内存泄漏 / 性能 / 体验）

对全部源码再次逐文件重审（内核 / 工具 / RAG / 本地模型 / UI 分层审查），按
严重度修复。核心是四类：**控制流与上下文治理的静默失效**、**资源泄漏**、
**并发竞态**、**性能与体验**。详见 [软件设计文档 §16](docs/软件设计文档.md)。

| # | 问题 | 影响 | 修复 |
|---|------|------|------|
| 220 | `Agent.run()` 无 try/finally | 内核任一未覆盖异常路径让 `on_finished` 不触发，UI 永久停在「思考中」 | 整段包 try/finally，正常/取消/异常/空回复四类出口都触发 finished + 按原因给可见文案 |
| 221 | 周期性 CONSOLIDATE 恒折 1 轮 | `turns` 净增长 +3 轮/4 轮，与「L2+L3 历史 O(1)」矛盾，长跑线性膨胀 | 折叠量匹配 `raw_turns+1` 窗口，每轮沉淀把 turns 压回窗口 |
| 222 | `_windows` 收缩阈值只跟 `quota.raw` 比 | system/earlier/tools 超支时原文窗不收缩 → 实际发送量越过 30k | 阈值改 `quota.raw - overspent()`，与越界判定同一口径 |
| 223 | `l3_fold` 同类项合并是死代码 | 每行带唯一 `[T{n}]` 前缀，整行比较永不相等，`cnt` 永不递增 | 按「剥掉前缀后的正文」比较，输出保留原行可追溯 |
| 224 | `l1_trim` HTML 降噪正则啃坏代码 | `std::vector<int>`→`std::vector `、`if a<b && c>d` 被改写，file_read/shell 返回的源码静默变形 | 改为 HTML 标签白名单正则，非白名单尖括号原样保留 |
| 225 | `plan._notify` 传活字典给 UI 线程 | 面板渲染半新半旧中间态，遍历中增删 step 抛 RuntimeError | 传 deepcopy（与 session/agent 同类 hook 对齐） |
| 226 | 空回复按原因给错文案 | 用户点「终止对话」却被告知「模型返回空响应」；LLM 报错时看不到真实错误 | 按 ExitReason 分流：CANCELLED/ERROR/其它各给对应文案 |
| 227 | `_estimate_tokens` 与 `budget.estimate_tokens` 两套口径 | tok/s 显示与预算记账对同段文本算出两个数 | 统一复用 `budget.estimate_tokens` |
| 228 | 并行工具执行双重起线程 + 忙等 | 池 worker 再调 `_execute_tool` 各起一根 10Hz 轮询线程，8 并发 = 16 线程其中 8 纯忙等 | 新增 `_exec_tool_blocking` 供池内直接调用，取消语义由 dispatcher 兜底 |
| 229 | 工作记忆注入无总量上限 | key 由模型自由创建，系统提示不在压缩范围，随会话挤爆 30k | 加 `_MAX_WM_KEYS=20` / `_MAX_WM_CHARS=2000` 双重钳制 |
| 230 | 上下文收缩重试时规划门提示重复附加 | 同一段 `_PLAN_GATE_HINT` 进 prompt 最多 3 份 | 重试 `continue` 前重置 `pending=user_input` |
| 231 | `_ask` 重试时 `raw_buf` 未随重置 | 当前不出错但属脆弱点，重试路径提前 break 会污染双视图历史 | 每轮重置 `raw_buf=""` |
| 232 | `web_scan` `follow_redirects=True` 只禁跨协议 | 公网 URL 可被 302 引到云元数据/loopback/内网（SSRF） | 手动逐跳跟随 + 目标网段判定（`_is_blocked_host`） |
| 233 | bs4 缺失时 `web_scan` 回退正则 ReDoS | `<script.*?</script>` 配 re.S 在超长未闭合 HTML 上大量回溯 | 改用 `str.find` 线性扫描（`_strip_blocks_linear`），任何输入 O(n) |
| 234 | `web_scan` 20000 截断无标记 | 静默硬切让模型把半截输出当全文 | 统一 `_truncate_marked` 加「已截断」标注 |
| 235 | `webexec_js` 非合法 JSON body 抛 JSONDecodeError 逃逸 | 被外层 except 误报成无上下文的裸异常 | content-type 声称 JSON 但解析失败时回退原文 + note |
| 236 | MCP reader 循环每轮读 `self._process` | stop() 置 None 后已通过检查的 readline 抛 ValueError，正常停止被误报成「连接丢失」并触发误重连 | 局部持有 pipe 引用 |
| 237 | MCP server 注册 0 工具仍留在 `_mcp_clients` | 全重名/空列表的 server 子进程 + 3 线程纯白占 | 注册数为 0 时立即 stop 并移出列表 |
| 238 | `ToolRegistry._signature_cache` 普通dict只增不减 | 测试/cli 动态生成的工具子类回收后仍留一条 Signature | 改 `WeakKeyDictionary` |
| 239 | `shell_run` reader join 超时静默丢迟到输出 | 模型以为看到的是完整输出 | 超时给 collector 注入「子进程仍在写入，输出可能不完整」标记 |
| 240 | `_BoundedIO.write` 截断返回 0 | 违反 write 契约，print/第三方代码误判写入失败 | 截断后返回 `len(s)` |
| 241 | `memory_graph`/`memory_search` 输出无体积上限 | 单实体累积数十 KB observation 全塞上下文 | `get_entity` 观察按 50 条/4000 字符截断；统一 `_cap_output` 200K 兜底 |
| 242 | `tools.memory._long_memory` 每轮一条 deprecation warning | agent 每轮调一次，日志被洪水刷掉 | 只警告一次（`_long_memory_warned` 标志） |
| 243 | `rag_ingest` 同步管线不可取消 | 用户点「终止对话」要等整库跑完 | 用 `progress_callback` 轮询终止标志，命中即 `engine.cancel()`（引擎签名不接受该参数时自动跳过） |
| 244 | `rag_search`/`rag_outline` 回退输出无截断 | 一次检索撑爆上下文 | 统一 `_cap_output` 200K + 文档列表截到 100 |
| 245 | `file_write` makedirs 成功但写入失败留空目录 | 下次 list 看到幽灵目录，模型被误导 | 失败时清理新建的空目录 |
| 246 | `llm_client` `chat_stream` 网络中断抛 httpx.* 非 LLMError | 调用方只捕 LLMError 时漏判真实断网 | `except httpx.HTTPError` 归一为 `LLMError` |
| 247 | 并行 tool_calls 无 index 时全归 slot 0 | 多个无 index 调用的 arguments 拼进同一 slot，产出非法 JSON | 按 id/name 判定新调用分配递增 index，纯参数片段归并上一 slot |
| 248 | MLX/ONNX 后端 `usage` 恒空 | tok/s 显示与计量全失效 | MLX/ONNX done 事件填 prompt/completion/total tokens |
| 249 | GGUF `n_ctx` 硬编码 4096 与 `max_tokens` 不匹配 | 长上下文生成早停/报错，max_tokens 形同虚设 | `n_ctx` 由 max_tokens 推导（+1k 余量，下限 8192） |
| 250 | `Skill.match` 对每技能重新分词 | 同一 query 被分词 N 次，技能多时每轮浪费 CPU | 分词一次共用（`score_tokens`） |
| 251 | `graph_memory.create_relation` 全表线性判重 | 关系密集时写入退化 O(N²)，持全局锁 | 维护 `_relation_keys` set，O(1) 判重 + 各变更点重建 |
| 252 | `_ensure_vectors` 异常后 `_db` 悬挂 | EF 获取/open_table 失败时连接不关 | except 分支回滚 `_db/_table/_ef` |
| 253 | `config` 把 `rag_ocr_scale_*` 错分进 `_STR_FIELDS` | config.json 手写成 "2.0"/2 后运行期 `set_ocr_scale_config` 把 str 当 truthy | 移到 `_FLOAT_FIELDS`/`_BOOL_FIELDS` |
| 254 | `_on_unload_main_model` 只关 client 不收尾 agent | MCP 子进程/httpx/ONNX/LanceDB 全泄漏，常驻 worker 仍引用旧 agent | 一并 `agent.close()` + `tools.shutdown()` |
| 255 | `_rebuild_agent` `wait(2000)` 无成功检查 | 未结束的 worker 引用归零触发 sip 析构 → qFatal | 超时挂 `_orphan_workers`，closeEvent 统一 wait |
| 256 | `_preload_client` daemon 线程无跟踪 | 连点「加载模型」起 N 个加载线程抢 GPU 内存 | 同 key 前一个未结束不再起新；引用归 `_preload_threads` |
| 257 | 知识同步与 agent 运行无互斥 | ingest 与检索抢同一向量表，「table modified during query」 | `_on_sync_knowledge` 在 `_busy` 时拒绝并提示先终止对话 |
| 258 | `_set_controls_busy` 重置同步控件 | agent 一跑同步按钮变回可点，与 agent 检索抢表 | `_sync_active` 期间跳过对同步控件/进度条的重置 |
| 259 | `add_streaming` 后续批次「先删后嵌」拉长空窗 | embed 耗时数秒内该 source 在库中被删空，并发检索返回空 | 改为「先嵌后删+写」（与首批同序），空窗缩到一次 add |
| 260 | `reload()` 无脑清 FTS 标志 | 同步后首次检索全库重建 BM25（秒级到分钟级），索引其实在磁盘 | 探测磁盘索引存在则直接 `_fts_ready=True`，不存在才回落重建 |

## 部署

详细部署文档见 [docs/deploy.md](docs/deploy.md)。

## License

MIT
