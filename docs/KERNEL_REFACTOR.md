# base-agent 内核重构：三态自动机 + 五级信息压缩

参照 GenericAgent 内核（`agent_loop.py` / `agentmain.py` / `llmcore.py` / `ga.py`）对
base-agent 的底层实现逻辑做重构。**上层能力（工具、RAG、技能、记忆、UI、配置）全部保留**，
只替换「控制流本身」。

---

## 一、现状差距

| 维度 | base-agent（现状） | GenericAgent（参照） | 重构后 |
|---|---|---|---|
| 控制流 | `while iteration < 15` 单循环，上下文管理/取消/重试/工具分发/记忆抽取全塞在 `run()` 里（≈420 行） | `agent_runner_loop` 132 行，只做「问 LLM → 分发 → 取 next_prompt」 | 三态自动机，内核只做状态转移 |
| 历史 | `_history` 全量重发；靠「条数裁剪 + 旧 tool 输出裁剪 + LLM 摘要收缩」三道防线兜底 | 历史存 `Session`，每轮只发**新消息**；旧历史折叠成 `history_info` 摘要行 | 历史交 `Session`，恒定 O(1) |
| 上下文 | 32k 阈值触发，**超了才压** | 每轮 `<summary>` 强制产出 30 字快照，天然恒定 | 五级压缩，预算分配制，硬上限 30k |
| 能力 | 13 个扁平工具，无主次 | 9 个原子工具，`code_run` 为能力核心 | 原子工具 + 能力工具分层 |
| 经验 | 图谱记忆 + 技能目录，但无强制沉淀时机 | `start_long_term_update` / `update_working_checkpoint` 是**一等公民工具** | 沉淀进状态机（CONSOLIDATE 态） |

---

## 二、三态自动机

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
         ▼          │                           │ 预算告罄 / turn%N
   ┌────────────┐   │                    ┌──────▼──────────┐
   │   EXIT     │◄──┴────────────────────│   CONSOLIDATE   │
   │  (终止态)   │   压缩完成 → 回 THINK    │   (沉淀态)      │
   └────────────┘                        └─────────────────┘
```

**状态语义**

| 状态 | 职责 | 输入 | 输出 |
|---|---|---|---|
| `THINK` | 把「折叠历史 + 工作记忆 + 工具集」交给 LLM，取得决策 | `Session.build_messages()` | `response`（文本 + tool_calls） |
| `ACT` | 分发工具；**能力**全部在这里（code_run / shell_run / rag / skill / mcp） | `tool_calls` | `StepOutcome(data, next_prompt, should_exit)` |
| `CONSOLIDATE` | 不调 LLM 决策，只做信息减熵：L3 折叠 + L4 蒸馏 + checkpoint 落盘 | 预算/轮次事件 | 压缩后的 session |

**转移表**

| 当前 | 事件 | 下一 | 副作用 |
|---|---|---|---|
| THINK | 有 `tool_calls` | ACT | — |
| THINK | 无 `tool_calls` 且文本非空 | EXIT(`DONE`) | L4 蒸馏 |
| THINK | 空响应 / 流中断 | THINK（重试 ≤3） | 注入纠错提示 |
| THINK | LLM 异常 | EXIT(`ERROR`) | — |
| ACT | `outcome.should_exit` | EXIT(`INTERRUPT`) | ask_user |
| ACT | `outcome.next_prompt` 为空 | EXIT(`DONE`) | — |
| ACT | 预算超阈值 或 `turn % consolidate_every == 0` | CONSOLIDATE | L3 折叠 |
| ACT | 其余 | THINK | next_prompt 注入 |
| CONSOLIDATE | 压缩完成 | THINK | — |
| ANY | `max_turns` / 用户取消 | EXIT(`MAX_TURNS` / `CANCELLED`) | — |

**内核不拥有**：工具实现、记忆存储、RAG、技能检索、UI 回调。它们全部是**注入的能力**。
内核只有三个动作：`ask_llm()`、`dispatch()`、`consolidate()`。

---

## 三、职责分离（内核只控制流）

```
                       ┌──────────────────────────┐
                       │   KernelLoop (三态自动机)   │  ← 只控制流
                       └───┬──────────┬────────┬───┘
             历史 ─────────┘          │        └───────── 经验
                  ▼                   ▼                    ▼
        ┌────────────────┐  ┌──────────────────┐  ┌──────────────────┐
        │    Session     │  │   ToolRegistry   │  │  Memory / Skills │
        │  L2/L3 压缩     │  │  code_run(能力核) │  │  L4 蒸馏产物      │
        │  滚动折叠       │  │  shell_run(辅助)  │  │  memory_graph    │
        │  checkpoint    │  │  rag_/skill_/mcp_ │  │  skills/**.md    │
        └────────────────┘  └──────────────────┘  └──────────────────┘
```

- **历史交给 session**：内核不再持有 `_history`。`Session` 负责保留最近 K 轮原文 + 更早轮折叠摘要。
- **能力交给 code_run**：`code_run` 是唯一「可编程」出口，`shell_run` 作为它对系统命令的延伸；
  二者组合覆盖任意操作。其余工具是「高频操作的快捷方式」。
- **经验交给 memory**：`update_working_checkpoint`（短期）→ `memory_graph`（长期图谱）→
  `skills/**/SKILL.md`（可执行经验）→ RAG 知识库（外部语料）。

---

## 四、五级信息压缩（硬上限 30k）

| 级 | 名称 | 触发时机 | 手段 | 单位上限 |
|---|---|---|---|---|
| **L0** | 原始捕获 | 工具返回瞬间 | 工具内硬上限（字节/行数/head+tail） | 单条 ≤ 20K |
| **L1** | 工具级裁剪 | 结果注入本轮前 | 按剩余预算/并发数分配 `maxlen`；HTML→文本、JSON→摘字段 | 单条 ≤ 6K |
| **L2** | 回合摘要 | 每轮结束 | 强制 `<summary>` 30 字物理快照，替换整轮 assistant+tool 原文 | 每轮 ≤ 30 字 |
| **L3** | 会话折叠 | 预算告罄 / 每 N 轮 | 旧轮 → 摘要行；同类项合并；`earlier_context` 只留尾部窗口 | 折叠区 ≤ 2.5K |
| **L4** | 记忆蒸馏 | 任务完成 / 检查点 | 沉淀进工作记忆 → 长期图谱 → SKILL.md → 知识库 | 系统提示 ≤ 4K 恒定 |

**关键不变量**：`L2 + L3` 使历史**恒定 O(1)**。
base-agent 现状是 O(轮次 × 8K)，每轮重发全部 —— 这是最本质的差距。

**预算分配（总计 30 000 tokens）**

```
SYS     5 000  系统提示 + 记忆索引 + 工作记忆 checkpoint
EARLIER 2 500  L3 折叠摘要区
RAW    12 000  L1 裁剪后的最近 K 轮原文（含当前轮工具输出）
TOOLS   5 000  工具 schema
RESERVE 5 500  模型输出 + 估算误差
```

`ContextBudget` 按上述配额核算；`Session.build_messages()` 在超配额时**强制折叠**最老的原文轮，
直到落回预算内；`L1` 的 `maxlen` 由 `budget.remaining_for_tool(n)` 动态算出。

---

## 五、工具分层

### 第一层：9 个原子工具（对齐 GenericAgent）

| # | 工具 | base-agent 现状 | 动作 |
|---|---|---|---|
| 1 | `code_run` | ✅ 已有 | 标记为**能力核心** |
| 2 | `file_read` | ✅ 已有 | — |
| 3 | `file_patch` | `file_modify` | 加 `file_patch` 别名 |
| 4 | `file_write` | ✅ 已有 | — |
| 5 | `web_scan` | ✅ 已有 | — |
| 6 | `web_execute_js` | `webexec_js`（条件注册） | 加 `web_execute_js` 别名 |
| 7 | `update_working_checkpoint` | ❌ | **新增** |
| 8 | `ask_user` | ✅ 已有 | — |
| 9 | `start_long_term_update` | ❌ | **新增** |

### 第二层：能力工具（base-agent 现有能力，全部保留）

- **系统控制**：`shell_run`（`code_run` 的系统侧延伸）
- **本地知识库**：`rag_search` / `rag_outline` / `rag_status` / `rag_ingest`
- **外置经验**：`skill_search` / `skill_semantic_search` / `skill_load` / **`skill_save`（新增，能力固化）**
- **记忆**：`work_memory` / `memory_graph` / `memory_search`
- **外部协议**：MCP 工具（动态注册）

### 第三层：能力进化闭环

```
任务成功
  ├─ start_long_term_update  → 蒸馏「事实/偏好/坑」→ memory_graph（经验）
  ├─ skill_save              → 固化「可复用流程」→ workspace/skills/<name>/SKILL.md（能力）
  └─ rag_ingest              → 把外部语料/历史会话灌入本地知识库（素材）
下次遇到同类任务 → skill_semantic_search 命中 → skill_load → 直接复用
```

`skill_save` 写出的 SKILL.md 会被 `SkillRetriever` 自动增量索引（fingerprint 同步），
因此「能力进化」无需额外索引代码。

---

## 六、目录结构

```
src/agent/
├── kernel/                  ← 新增：内核（只控制流）
│   ├── outcome.py           StepOutcome / KernelState / ExitReason
│   ├── budget.py            ContextBudget：30k 配额核算
│   ├── compressor.py        L0–L4 五级压缩
│   ├── session.py           Session：历史托管 + 滚动折叠 + checkpoint
│   └── loop.py              KernelLoop：三态自动机
├── agent.py                 ← 改为门面：保留 callbacks / skills / 配置，委托 KernelLoop
├── tools/
│   ├── checkpoint.py        ← 新增：update_working_checkpoint / start_long_term_update
│   ├── skill_search.py      ← 追加 skill_save
│   └── ...
└── ...                      RAG / memory / skills / ui 全部不动
```

---

## 七、兼容性

- `Agent.run(user_input) -> str` 签名与返回不变，UI（`ui/main_window.py`）零改动。
- `AgentCallbacks` 全部信号保留。
- 旧的上下文收缩逻辑保留为**兜底**（`Agent._shrink_session`）：仅当 L3 折叠用尽
  （`fold-exhausted`）仍未落回预算时才触发，摘要同样持久化到 `workspace/memory.md`。
- 配置项向后兼容，新增项均有默认值。

## 八、实现清单（已完成）

| 模块 | 文件 | 状态 |
|---|---|---|
| 状态 / 转移表 / `StepOutcome` | `src/agent/kernel/outcome.py` | ✅ |
| 30k 预算配额与核算 | `src/agent/kernel/budget.py` | ✅ |
| L0–L4 五级压缩 + 蒸馏 prompt | `src/agent/kernel/compressor.py` | ✅ |
| Session（折叠 / 工作记忆 / 锚点） | `src/agent/kernel/session.py` | ✅ |
| 三态自动机主循环 | `src/agent/kernel/loop.py` | ✅ |
| 原子工具：checkpoint / 蒸馏 | `src/agent/tools/checkpoint.py` | ✅ |
| 能力进化：skill_save | `src/agent/tools/skill_search.py` | ✅ |
| 原子命名别名 `AliasTool` | `src/agent/tools/base.py` | ✅ |
| Agent 退化为门面 | `src/agent/agent.py` | ✅ |
| 内核配置 | `src/agent/config.py` | ✅ |
| 测试 | `tests/test_kernel.py`（23 例）+ `tests/test_agent.py`（已迁移） | ✅ |

**分层落点**

| 关注点 | 归属 | 内核是否持有 |
|---|---|---|
| 历史 | `Session` | ❌ 注入 |
| 能力 | `_ToolDispatcher` → `ToolRegistry`（`code_run` 为核心，`shell_run` 辅助） | ❌ 注入 |
| 经验 | `_AgentKernelHooks` → 知识图谱 / `skills/**/SKILL.md` / RAG | ❌ 注入 |
| 控制流 | `KernelLoop`（THINK / ACT / CONSOLIDATE） | ✅ 唯一职责 |

**实测**（`tests/test_kernel.py::test_prompt_stays_within_30k_over_long_run`）：
连续 10 轮读取 40K 字符文件的长任务，每轮 prompt 峰值均 ≤ 30 000 token。

## 九、落地后复查（第八轮）

内核上线后做了一轮实现级复查，修掉三个 P0 —— 都是「设计对了、代码写错了」：

| # | 缺陷 | 后果 | 修复 |
|---|---|---|---|
| 1 | 循环条件用 `session.current_turn`（**会话累计**） | 累计 15 轮后 Agent **彻底不再调 LLM** | 引入 `_turns_used`，每次 `run()` 归零 |
| 2 | reask 分支先 `_rollback_turn` 再 `finish_turn` | 写的是游离对象 → 最终答复轮丢失，L4 缺结论 | 回滚分支不再 finish_turn；加 `_reask_guard` |
| 3 | 周期性 CONSOLIDATE 传 `force_fold(0)` | **完全空操作**，turns 单调增长 | 传 `consolidate_every` |

另修：预算漏算 `tool_calls` 参数体；`system`/`tools` 超支不从 `raw` 扣除（30k 沦为记账）；
`estimate_tokens` 密度过乐观；上下文超长被当致命错误（兜底收缩永远走不到）；
蒸馏线程无并发上限且退回共享 `httpx.Client`。

**教训**：三条 P0 都不会抛异常、也不会让测试变红（#1 前 15 轮一切正常，#3 只是没生效）。
「设计正确」与「实现正确」之间必须有针对性的不变量测试：
`test_kernel.py` 为此补了轮次上限 per-run、周期性 CONSOLIDATE 真的折叠、
折叠后无悬挂 `tool` 消息、长任务 prompt 峰值不越过 30k 四项。

详见[软件设计文档 §11.21](软件设计文档.md)。
