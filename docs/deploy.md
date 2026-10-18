# Base Agent 生产环境部署指南

## 1. 环境要求

| 组件 | 最低版本 | 说明 |
|------|----------|------|
| Python | 3.9+ | 推荐 3.10+ |
| 操作系统 | macOS / Linux | Windows 需额外适配 Qt 依赖 |
| 内存 | 4 GB+ | 加载嵌入模型（~130MB）和重排序模型（~500MB）需要额外内存 |
| 磁盘 | 2 GB+ | 模型缓存 + 向量数据库持久化存储 |

## 2. 安装

### 2.1 安装核心依赖

```bash
# 克隆项目
git clone <repo-url> base-agent
cd base-agent

# 安装核心包（GUI + LLM + MCP）
pip install -e .
```

### 2.2 安装 RAG 依赖（按需）

```bash
# ===== 一键安装全部 RAG 依赖 =====
pip install -e ".[rag]"

# ===== 或按需安装单个组件 =====

# 向量存储 + 嵌入模型
pip install lancedb fastembed

# 重排序模型（可选，提升检索精度；由 fastembed 的 TextCrossEncoder 提供，
# 无需单独安装 —— `pip install 'fastembed>=0.9.0'` 已包含）

# 文档解析依赖（按需）
pip install python-docx      # Word 文档
pip install openpyxl          # Excel 表格
pip install python-pptx       # PowerPoint 演示文稿
pip install PyMuPDF           # PDF 文档
pip install ebooklib          # EPUB 电子书

# OCR 依赖（可选，图片型 PDF 识别，~500MB）
pip install -e ".[ocr]"
# 或: pip install rapidocr-onnxruntime
```

### 2.3 一键安装全部依赖

```bash
pip install -e ".[all]"
```

## 3. 模型下载（离线环境）

首次运行时，FastEmbed 会自动从 HuggingFace 下载所需的嵌入模型。如果生产环境无法访问 HuggingFace，需要预先下载模型文件。

### 3.1 所需模型

| 模型 | 用途 | 大小 |
|------|------|------|
| `nomic-ai/nomic-embed-text-v1.5-Q` | 文本嵌入向量化（INT8 量化版） | ~130 MB |
| `BAAI/bge-reranker-v2-m3-int8` | 检索结果重排序（可选，fastembed ONNX，INT8 量化版） | ~570 MB |

### 3.2 离线部署方案

```bash
# 在有网络的机器上预下载模型
pip install huggingface_hub
# 嵌入模型：HF 仓库名是 nomic-ai/nomic-embed-text-v1.5（与 fastembed 的 -Q
# 量化版映射到同一个仓库），量化权重是仓库里的 onnx/model_quantized.onnx
# （~137MB；fp32 的 onnx/model.onnx 约 548MB，不需要）
huggingface-cli download nomic-ai/nomic-embed-text-v1.5 \
  --include "onnx/model_quantized.onnx" "tokenizer.json" "config.json" \
  --local-dir ./models/nomic-embed-text-v1.5
# 重排序模型由 fastembed 在首次检索时自动下载并缓存到 ~/.cache/fastembed，
# 无需手动下载（如需离线预置，见 README「手工预下载注意」：
# fastembed 模型名 BAAI/bge-reranker-v2-m3-int8 实际拉取 HF 仓库
# onnx-community/bge-reranker-v2-m3-ONNX，缓存目录按仓库名推导）

# 将模型目录复制到生产服务器
scp -r ./models/ user@server:/path/to/models/

# 生产环境设置环境变量指向本地模型
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

模型缓存默认路径：`~/.cache/huggingface/hub/`，将模型文件放到该路径即可。

## 4. 配置

配置文件位于 `~/.base-agent/config.json`，首次启动应用后自动生成。也可手动创建：

```json
{
  "base_url": "https://api.openai.com/v1",
  "api_key": "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "model": "gpt-4o-mini",
  "temperature": 0.7,
  "max_tokens": 32768,
  "top_p": 0.95,
  "min_p": 0.05,
  "top_k": 20,
  "repetition_penalty": 1.0,
  "workspace": "~/base-agent-workspace",
  "knowledge_base": "/data/knowledge-base",
  "request_timeout": 120.0,
  "max_iterations": 15,
  "rag_chunk_size": 500,
  "rag_chunk_overlap": 50,
  "rag_embedding_model": "nomic-ai/nomic-embed-text-v1.5-Q",
  "rag_rerank_model": "BAAI/bge-reranker-v2-m3-int8",
  "rag_rerank_enabled": true,
  "rag_auto_ingest": true,
  "browser_endpoint": "",
  "mcp_servers": []
}
```

### 关键配置项说明

| 配置项 | 说明 | 推荐值 |
|--------|------|--------|
| `knowledge_base` | 知识库文档目录路径 | 指向存放原始文档的目录 |
| `rag_chunk_size` | 每个切片的 token 数 | 500 |
| `rag_chunk_overlap` | 切片间重叠 token 数（10%） | 50 |
| `rag_embedding_model` | 嵌入向量模型（INT8 量化版） | `nomic-ai/nomic-embed-text-v1.5-Q` |
| `rag_rerank_model` | 重排序模型（INT8 量化版） | `BAAI/bge-reranker-v2-m3-int8` |
| `rag_rerank_enabled` | 是否启用重排序 | `true` |
| `rag_auto_ingest` | 是否启动时自动同步知识库 | `true` |

## 5. 知识库目录结构

将需要索引的文档放入 `knowledge_base` 指定的目录，支持子目录递归扫描：

```
/data/knowledge-base/
├── 技术文档/
│   ├── 架构设计.docx
│   ├── API文档.md
│   └── 部署手册.pdf
├── 产品资料/
│   ├── 需求文档.xlsx
│   ├── 用户手册.epub
│   └── 竞品分析.html
├── 培训材料/
│   └── 新人指南.pptx
└── 参考资料/
    ├── paper.pdf
    └── book.mobi
```

### 支持的文档格式

| 格式 | 扩展名 | 依赖 |
|------|--------|------|
| Word | `.docx`, `.doc` | `python-docx` |
| Excel | `.xlsx`, `.xls` | `openpyxl` |
| PowerPoint | `.pptx`, `.ppt` | `python-pptx` |
| PDF | `.pdf` | `PyMuPDF`（图片型 PDF 需 `rapidocr-onnxruntime`） |
| EPUB | `.epub` | `ebooklib` + `beautifulsoup4` |
| MOBI/AZW | `.mobi`, `.azw3`, `.azw` | calibre `ebook-convert`（可选） |
| 纯文本 | `.txt`, `.md`, `.rst`, `.csv`, `.tsv` | 无额外依赖 |
| HTML | `.html`, `.htm` | `beautifulsoup4` |

## 6. 启动服务

### 6.1 交互式 GUI 模式

```bash
cd base-agent
python main.py
```

启动后：
1. 在界面顶部的「知识库」输入框中填写知识库路径，或点击「…」按钮选择目录
2. 点击「同步知识」按钮，右侧日志面板实时显示处理进度
3. 同步流程：文档扫描 → 增量对比 → 文本提取 → 清洗 → Markdown 转换 → 切片 → 向量化 → LanceDB 存储 → 清理已删除文件 → 保存 Manifest
4. 同步完成后，在对话框中输入问题时，Agent 会自动调用 `rag_search` 工具检索知识库

### 6.2 无头模式（编程调用）

```python
from src.agent.rag.engine import RAGEngine

# 初始化引擎
engine = RAGEngine(
    workspace="/data/agent-workspace",
    knowledge_base="/data/knowledge-base",
)

# 同步知识库（首次运行或文档更新后）
stats = engine.ingest(force=True)
print(f"索引完成: {stats['files_found']} 文件 → {stats['chunks']} 切片")

# 检索知识库
results = engine.search("如何配置 RAG 服务", top_k=3)
for r in results:
    print(f"[{r['score']:.4f}] {r['source']}")
    print(r["text"][:200])
    print("---")

# 查看状态
status = engine.status()
print(f"已索引: {status['chunks_stored']} 切片, {len(status['sources'])} 文件")
```

### 6.3 定时同步脚本（cron）

```bash
# 创建同步脚本 /usr/local/bin/sync-knowledge.sh
cat > /usr/local/bin/sync-knowledge.sh << 'EOF'
#!/bin/bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /opt/base-agent
python3 -c "
from src.agent.rag.engine import RAGEngine
import json

engine = RAGEngine(
    workspace='/data/agent-workspace',
    knowledge_base='/data/knowledge-base',
)
stats = engine.ingest(force=True)
print(json.dumps(stats, ensure_ascii=False, indent=2))
"
EOF

chmod +x /usr/local/bin/sync-knowledge.sh

# 添加定时任务：每天凌晨 2 点同步
crontab -e << 'EOF'
0 2 * * * /usr/local/bin/sync-knowledge.sh >> /var/log/rag-sync.log 2>&1
EOF
```

## 7. 数据目录说明

启动后自动创建的目录结构：

```
~/base-agent-workspace/
└── rag/
    ├── documents/          # 清洗后的 Markdown 文件（缓存）
    │   ├── rag_intro.md
    │   ├── vector_db.txt.md
    │   └── ...
    ├── vectors/            # LanceDB 向量数据库（持久化，Lance 列式格式）
    │   └── chunks.lance/
    └── manifest.json       # 增量同步签名记录
```

- `rag/documents/` — 清洗后的 Markdown 缓存，二次同步时跳过未修改的文件
- `rag/vectors/` — LanceDB 持久化存储，包含所有向量索引
- `rag/manifest.json` — 记录每个文件的 mtime + size + content_hash，用于增量同步判断

## 8. 生产环境 Checklist

### 部署前

- [ ] Python 3.9+ 已安装
- [ ] 核心依赖已安装：`pip install -e .`
- [ ] RAG 依赖已安装：`pip install -e ".[rag]"`
- [ ] 嵌入模型已下载到本地缓存（见第 3 节）
- [ ] 重排序模型已下载（如需启用重排序）
- [ ] 配置文件 `~/.base-agent/config.json` 已就绪
- [ ] `knowledge_base` 目录存在且包含文档
- [ ] `workspace` 目录有写入权限

### 首次启动

- [ ] 点击「同步知识」或执行 `ingest(force=True)`
- [ ] 右侧日志面板确认同步完成，查看到切片数和文件数统计
- [ ] 发送一条测试查询，确认 `rag_search` 工具能返回相关知识库内容

### 运维

- [ ] 文档更新后，重新执行「同步知识」（增量，无需 force）
- [ ] 设置 cron 定时同步（见 6.3 节）
- [ ] 监控 `rag/vectors/` 目录磁盘使用
- [ ] 定期清理过期的 Markdown 缓存（`rag/documents/`）

## 9. 故障排查

### 问题：同步知识卡住不动

**原因**：HuggingFace 模型下载超时或网络不可达。

**解决**：设置离线模式环境变量，确保模型已预下载到本地。

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### 问题：同步完成后按钮仍显示"同步中"

**原因**：同步结束信号未被 UI 正确处理（罕见）。

**解决**：QThread.finished 信号会自动兜底恢复按钮状态。如果问题持续，重启应用即可。

### 问题：搜索返回空结果

**检查清单**：
1. `knowledge_base` 路径是否正确，目录下是否有受支持的文档
2. 是否已执行「同步知识」（查看 `rag/vectors/` 目录下是否有 `chunks.lance/`）
3. 嵌入模型与重排序模型是否已下载（`~/.cache/fastembed/`）
4. 查看右侧日志面板的检索日志，确认候选切片数和重排序结果

### 问题：重排序不生效

**原因**：`fastembed` 未安装（或版本过低，需 >=0.9.0）或 `BAAI/bge-reranker-v2-m3-int8` 模型未下载。

**解决**：
```bash
pip install 'fastembed>=0.9.0'
# 重排序模型由 fastembed 首次检索时自动下载并缓存到 ~/.cache/fastembed
```

如果不需要重排序，可在配置中设置 `"rag_rerank_enabled": false`，系统会自动回退到向量距离排序。

### 问题：PDF 文档提取为空

**原因**：PDF 为纯图片格式，且未安装 OCR 依赖。

**解决**：
```bash
pip install rapidocr-onnxruntime
```

### 问题：LanceDB 文件锁冲突

**原因**：多个进程同时访问同一个向量数据库。

**解决**：确保同一时间只有一个进程在写入。LanceDB 使用 Lance 列式格式，不支持多进程并发写入。

### 问题：日志面板不显示同步/检索日志

**原因**：RAG 模块的 logger 未被正确桥接到 UI。

**解决**：正常情况下，应用启动时会自动设置日志桥接（将 RAG 模块的 logger handler 挂到 root logger 并通过 Filter 过滤）。如果日志仍不显示，检查终端是否有 `Log bridge:` 日志输出，以及 `_RAG_LOG_PREFIXES` 配置是否覆盖了所有 RAG 模块。
