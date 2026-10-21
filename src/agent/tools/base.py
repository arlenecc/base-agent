"""Tool base class, registry, and exports."""
from __future__ import annotations

import inspect
import json
import logging
import os
import threading
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..config import AgentConfig

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    success: bool
    output: str = ""
    error: str = ""

    def to_message(self) -> str:
        if self.success:
            return self.output
        return f"ERROR: {self.error}" if self.error else "ERROR: unknown"


class Tool:
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {}
    # destructive marks tools that *can* affect the system or files outside the
    # workspace. Workspace-locked file ops set this False (they cannot escape).
    destructive: bool = False

    def run(self, **kwargs) -> ToolResult:  # pragma: no cover - abstract
        raise NotImplementedError

    def should_confirm(self, args: Dict[str, Any]) -> bool:
        """Return True if this invocation needs user confirmation before running.

        Default policy: confirm iff `self.destructive` is set. Override for
        finer-grained control (e.g. confirm only when args target paths outside
        the workspace). Operations confined to the workspace never need
        confirmation, since they cannot harm the system or other files.
        """
        return self.destructive

    def schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class AliasTool(Tool):
    """同一实现的另一个名字。

    用于把 base-agent 的原生工具名对齐 GenericAgent 的原子工具命名
    （``file_modify`` → ``file_patch``、``webexec_js`` → ``web_execute_js``），
    不复制任何实现。schema 与 target 完全一致，因此不会产生行为分叉。
    """

    def __init__(self, target_name: str, alias: str, description: str = ""):
        self._target_name = target_name
        self.name = alias
        self.description = description or ""
        self.destructive = False

    def bind(self, config: "AgentConfig", registry: "ToolRegistry") -> None:
        self.config = config
        self.registry = registry

    @property
    def parameters(self) -> Dict[str, Any]:
        t = self.registry.get(self._target_name) if self.registry else None
        return getattr(t, "parameters", {}) or {"type": "object", "properties": {}}

    def should_confirm(self, args: Dict[str, Any]) -> bool:
        t = self.registry.get(self._target_name) if self.registry else None
        return bool(t.should_confirm(args)) if t else False

    def run(self, **kwargs) -> ToolResult:
        if not self.registry:
            return ToolResult(False, error=f"alias {self.name} not bound")
        return self.registry.execute(self._target_name, kwargs)


class ToolRegistry:
    # run 的 Signature 缓存（按工具类）：工具集注册后固定，没必要每次
    # execute() 都重新构造 Signature 对象。
    # WeakKeyDictionary：类键弱引用——测试套件/cli 动态生成的工具子类被
    # 回收时缓存条目随之消失（旧实现是普通 dict，只增不减，长跑进程里
    # 每个临时子类泄露一条 Signature）。
    _signature_cache: "weakref.WeakKeyDictionary[type, inspect.Signature]" = (
        weakref.WeakKeyDictionary()
    )

    def __init__(self, config: AgentConfig, callbacks: "Any", aux_llm_getter: "Any" = None):
        self.config = config
        self.callbacks = callbacks
        # 辅助模型客户端惰性获取器：由 Agent 注入（aux 在运行时才加载/可卸载），
        # RAG 工具用它把检索分片交给辅助模型先归纳精简。为 None 时跳过精简。
        self._aux_llm_getter = aux_llm_getter
        self.config.ensure_workspace()
        self._tools: Dict[str, Tool] = {}
        self._mcp_clients: list = []  # MCPClient instances, for cleanup
        self._rag_engine = None  # lazy-initialized RAGEngine
        # 保护懒初始化的重资源单例（SkillRetriever / LongTermMemory）：
        # agent 线程与 UI 线程可能并发首次调用。
        self._singleton_lock = threading.RLock()
        self._retriever_lock = self._singleton_lock
        self._register_defaults()

    # ------------------------------------------------------------------
    def _register_defaults(self) -> None:
        from .file_ops import FileReadTool, FileWriteTool, FileModifyTool
        from .code_run import CodeRunTool
        from .shell_run import ShellRunTool
        from .web import WebScanTool, WebExecJsTool
        from .interact import AskUserTool
        from .memory import WorkMemoryTool, MemoryGraphTool, MemorySearchTool
        from .skill_search import SkillSearchTool, SkillLoadTool, SkillSemanticSearchTool, SkillSaveTool
        from .checkpoint import UpdateWorkingCheckpointTool, StartLongTermUpdateTool
        from .plan import UpdatePlanTool

        tools_to_register = [
            FileReadTool(), FileWriteTool(), FileModifyTool(),
            CodeRunTool(),
            ShellRunTool(),
            WebScanTool(),
            AskUserTool(),
            WorkMemoryTool(),
            MemoryGraphTool(),
            MemorySearchTool(),
            SkillSearchTool(),
            SkillLoadTool(),
            SkillSemanticSearchTool(),
            SkillSaveTool(),
            # 沉淀态原子工具（对齐 GenericAgent）
            UpdateWorkingCheckpointTool(),
            StartLongTermUpdateTool(),
            # 规划态原子工具：目标 + 路径的一等公民表示
            UpdatePlanTool(),
        ]

        # webexec_js: only register when browser_endpoint is configured.
        # Saves ~146 tokens of schema when no browser bridge is available
        # (the common case).  Without an endpoint the tool just returns
        # guidance text — no functional value.
        if getattr(self.config, "browser_endpoint", ""):
            tools_to_register.append(WebExecJsTool())

        for t in tools_to_register:
            t.bind(self.config, self)
            self._tools[t.name] = t

        # 原子工具命名对齐 GenericAgent（同一实现，只是多一个入口名）
        for target, alias, desc in (
            ("file_modify", "file_patch",
             "Replace a unique block of text (old) with new. old must match "
             "exactly (whitespace/indentation). Use file_read first to get "
             "current content."),
            ("webexec_js", "web_execute_js",
             "Execute JavaScript in the attached browser for full page control."),
        ):
            if target in self._tools and alias not in self._tools:
                a = AliasTool(target, alias, desc)
                a.bind(self.config, self)
                self._tools[alias] = a

        # Register RAG tools
        self._register_rag_tools()

        # Register MCP tools from configured servers
        self._register_mcp_tools()

    def _register_mcp_tools(self) -> None:
        """Connect to configured MCP servers and register their tools."""
        mcp_servers = getattr(self.config, "mcp_servers", None) or []
        if not mcp_servers:
            return

        from .mcp_client import MCPClient, MCPError
        from .mcp_tool import MCPTool

        for server_cfg in mcp_servers:
            if not isinstance(server_cfg, dict):
                continue
            name = server_cfg.get("name", "")
            command = server_cfg.get("command", "")
            if not name or not command:
                logger.warning("MCP server config missing name or command: %s", server_cfg)
                continue

            try:
                client = MCPClient(
                    name=name,
                    command=command,
                    args=server_cfg.get("args", []),
                    env=server_cfg.get("env"),
                )
                client.start()
                self._mcp_clients.append(client)

                registered = 0
                for tool_def in client.list_tools():
                    t = MCPTool(client, tool_def)
                    if t.name and t.name not in self._tools:
                        t.bind(self.config, self)
                        self._tools[t.name] = t
                        registered += 1
                        logger.info("MCP tool registered: %s (server: %s)", t.name, name)
                    elif t.name:
                        # 重名工具静默吞掉第二个：用户以为两个 server 的工具
                        # 都挂了，实际只注册了先连上的那个，且无任何线索。
                        logger.warning(
                            "MCP tool %s from server '%s' skipped: name already "
                            "registered (another server or builtin tool wins)",
                            t.name, name,
                        )
                if registered == 0:
                    # 一个工具都没注册上（全重名 / server 暴露空列表）→ 这个
                    # server 的子进程 + 3 个线程纯属白占。立刻停掉并不留在
                    # _mcp_clients 里，否则要挂到 shutdown 才回收。
                    logger.warning(
                        "MCP server '%s' 未注册任何工具（重名或空列表），停止该 server",
                        name,
                    )
                    self._mcp_clients.remove(client)
                    try:
                        client.stop()
                    except Exception:
                        pass
            except MCPError as e:
                logger.warning("MCP server '%s' init failed: %s", name, e)
            except Exception as e:
                logger.warning("MCP server '%s' unexpected error: %s", name, e)

    def _register_rag_tools(self) -> None:
        """Initialize RAG engine and register RAG tools."""
        kb = getattr(self.config, "knowledge_base", "") or ""
        # If knowledge_base is not explicitly set, default to
        # <workspace>/knowledge_base so that RAG tools are available out of
        # the box when users place documents there (or have previously
        # synced knowledge).  Without this, the agent never registers RAG
        # tools and cannot discover the local knowledge base.
        if not kb:
            kb = os.path.join(self.config.workspace, "knowledge_base")
            logger.info(
                "RAG: knowledge_base not configured, defaulting to %s", kb
            )

        # RAG 是可选能力：整段初始化（建目录 + 构造引擎 + 注册工具）都必须
        # 包在 try 里。否则知识库路径不可创建/引擎构造失败时，异常会冒泡到
        # ToolRegistry.__init__ → Agent 构造失败 → 文件读写、shell、web 等
        # **所有**工具一起失效。
        try:
            # Create the directory if it doesn't exist so that RAGEngine
            # initialization doesn't fail on a missing path.
            os.makedirs(kb, exist_ok=True)

            from ..rag.engine import RAGEngine
            from ..rag.vector_store import DEFAULT_RERANK_MODEL
            from .rag_tool import RagSearchTool, RagStatusTool, RagIngestTool, RagOutlineTool

            logger.info(
                "RAG: initializing engine (workspace=%s kb=%s embedding=%s)",
                self.config.workspace, kb,
                getattr(self.config, "rag_embedding_model", "nomic-ai/nomic-embed-text-v1.5-Q"),
            )
            engine = RAGEngine(
                workspace=self.config.workspace,
                knowledge_base=kb,
                embedding_model=getattr(self.config, "rag_embedding_model", "nomic-ai/nomic-embed-text-v1.5-Q"),
                rerank_model=getattr(self.config, "rag_rerank_model",
                                     DEFAULT_RERANK_MODEL),
                rerank_enabled=bool(getattr(self.config, "rag_rerank_enabled", True)),
            )
            self._rag_engine = engine

            registered = []
            for t in [RagSearchTool(engine, aux_llm_getter=self._aux_llm_getter),
                      RagStatusTool(engine), RagIngestTool(engine),
                      RagOutlineTool(engine)]:
                if t.name not in self._tools:
                    t.bind(self.config, self)
                    self._tools[t.name] = t
                    registered.append(t.name)
            logger.info("RAG: tools registered: %s", ", ".join(registered))
        except Exception as e:
            logger.warning(
                "RAG 工具不可用（其余工具不受影响）: %s: %s", type(e).__name__, e
            )
            return

        # NOTE: auto-ingest is intentionally NOT performed here.
        # _register_rag_tools() runs in ToolRegistry.__init__, which is called
        # from the UI main thread (via _rebuild_agent). Synchronous ingest would
        # block the UI for minutes on first run (model download + parsing +
        # embedding of all files). The UI provides SyncKnowledgeWorker (async,
        # on a background thread) for this purpose. For non-UI usage, call
        # engine.ingest() explicitly after construction.

    def get_rag_engine(self):
        """Return the RAG engine instance, or None if not configured."""
        return self._rag_engine

    def get_skill_retriever(self):
        """Return the shared SkillRetriever (hybrid skill search), lazily built.

        The retriever scans ``<workspace>/skills`` for SKILL.md files and builds
        a LanceDB table (embeddings + BM25).  Building is lazy: the first call
        triggers scanning + indexing (can be slow for thousands of skills), so
        it is cached on the registry and rebuilt on subsequent calls only if
        the skills directory changed.
        """
        # 缓存按 skills_dir 区分：「应用配置」切换 workspace 后，复用旧目录的
        # retriever 会检索到错误工作区的技能。
        skills_dir = os.path.join(self.config.workspace, "skills")
        cache = getattr(self, "_skill_retriever_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(self, "_skill_retriever_cache", cache)
        retriever = cache.get(skills_dir)
        if retriever is not None:
            return retriever
        # 首次构建是重活（扫描 + LanceDB 建表 + embedding），且 agent 线程与
        # UI 线程可能并发首次调用：不加锁会各建一个 SkillRetriever，后写者
        # 覆盖属性，先建的 LanceDB 连接泄漏（shutdown 只回收缓存里的那个）。
        with self._retriever_lock:
            retriever = cache.get(skills_dir)
            if retriever is not None:
                return retriever
            db_path = os.path.join(self.config.workspace, ".agent", "skill_index.lancedb")
            from ..skill_retriever import SkillRetriever
            from ..rag.vector_store import get_or_create_embedding_function
            # 必须复用 RAG 配置里的 embedding 模型：SkillVectorStore 默认用
            # DEFAULT_EMBEDDING_MODEL，用户改了模型就会额外驻留第二个
            # FastEmbed 单例（各 ~137MB C++ 堆），且两侧索引不一致。
            # 失败时显式回退到默认模型单例（与 SkillVectorStore 内部 _get_ef
            # 的回退是同一个进程级单例），而不是传 None 让它悄悄自建。
            model_name = (getattr(self.config, "rag_embedding_model", "") or "").strip()
            try:
                ef = get_or_create_embedding_function(model_name or None)
            except Exception as e:
                logger.warning(
                    "skill embedding function init failed (%s); "
                    "falling back to default model", e,
                )
                try:
                    ef = get_or_create_embedding_function()
                except Exception:
                    ef = None
            retriever = SkillRetriever(
                skills_dir=skills_dir, db_path=db_path,
                embedding_function=ef,
            )
            cache[skills_dir] = retriever
            setattr(self, "_skill_retriever", retriever)
            return retriever

    def reload_rag(self) -> None:
        """Refresh the RAG engine's table handle so new sync data is visible.

        Call this after SyncKnowledgeWorker finishes — the agent's RAGEngine
        holds a stale LanceDB table snapshot that won't reflect newly written
        rows until the table is re-opened.
        """
        if self._rag_engine is not None:
            try:
                self._rag_engine.reload()
            except Exception as e:
                logger.warning("ToolRegistry.reload_rag error: %s", e)

    def shutdown(self) -> None:
        """Stop all MCP server subprocesses and release RAG engine resources."""
        # 先回收在跑的 shell_run 子进程：start_new_session=True 让它们脱离
        # 主进程组，终端信号与 atexit 之外的路径都感知不到。旧实现只靠
        # atexit——「应用配置」重建 agent 时旧 registry 被 shutdown 丢弃，
        # 正在执行的 shell_run（默认 120s、上限 600s）子进程变成无人看管的
        # 孤儿，继续占 CPU/磁盘/端口直到自身超时。
        try:
            from .shell_run import kill_all_running_procs
            killed = kill_all_running_procs()
            if killed:
                logger.info("ToolRegistry.shutdown: 回收 %d 个在跑的 shell_run 子进程", killed)
        except Exception as e:
            logger.warning("ToolRegistry.shutdown: 回收 shell_run 子进程失败: %s", e)
        for client in self._mcp_clients:
            try:
                client.stop()
            except Exception:
                pass
        self._mcp_clients.clear()
        # Close shared httpx clients in web tools to release connection pools.
        for tool in self._tools.values():
            http_client = getattr(tool, "_http", None)
            if http_client is not None:
                try:
                    http_client.close()
                except Exception:
                    pass
        # Release the RAG engine's VectorStore (LanceDB connection + FastEmbed
        # ONNX model + BGE reranker). Without this, each _rebuild_agent() call
        # leaks ~600MB of C++ heap memory.
        if self._rag_engine is not None:
            try:
                logger.info("🔌 正在释放 RAG 引擎资源 (向量库连接 + 嵌入模型 + Reranker)...")
                self._rag_engine.close()
                logger.info("✅ RAG 引擎资源已释放")
            except Exception as e:
                logger.warning("⚠ RAG 引擎资源释放异常: %s", e)
            self._rag_engine = None
        # Release the singleton LongTermMemory's GraphMemoryStore (LanceDB +
        # FastEmbed ONNX model ~137MB).  Without this, each _rebuild_agent()
        # call leaks the model loaded for the knowledge graph.
        #
        # 必须持 _singleton_lock（RLock）：_get_long_memory / get_skill_retriever
        # 的懒初始化也走这把锁。旧实现裸清属性，与 agent 线程的懒初始化竞态：
        # shutdown 清空后紧跟着一次新建 → 新实例不在任何缓存里，永不被 close
        # （每次「应用配置」泄漏 ~137MB ONNX + LanceDB 连接）。
        from .memory import _LONG_MEMORY_SINGLETON_ATTR
        with self._singleton_lock:
            lm = getattr(self, _LONG_MEMORY_SINGLETON_ATTR, None)
            if lm is not None:
                try:
                    lm.close()
                except Exception as e:
                    logger.warning("⚠ 长期记忆资源释放异常: %s", e)
                setattr(self, _LONG_MEMORY_SINGLETON_ATTR, None)
            # Release the SkillIndex singleton(s) (if any)
            setattr(self, "_skill_index", None)
            setattr(self, "_skill_index_cache", None)
            # Release the SkillRetriever (LanceDB connection). Note: it shares
            # the process-wide embedding function singleton, which we do NOT
            # close here.
            cache = getattr(self, "_skill_retriever_cache", None)
            retrievers = list(cache.values()) if isinstance(cache, dict) else []
            legacy = getattr(self, "_skill_retriever", None)
            if legacy is not None and legacy not in retrievers:
                retrievers.append(legacy)
            for skill_retriever in retrievers:
                try:
                    skill_retriever.close()
                except Exception as e:
                    logger.warning("⚠ skill retriever 资源释放异常: %s", e)
            setattr(self, "_skill_retriever", None)
            setattr(self, "_skill_retriever_cache", None)
        # 清空工具表：RAG 引擎与 skill 检索器都已关闭，RagSearchTool /
        # SkillSemanticSearchTool 等却仍持有对它们的引用。留着的话，shutdown
        # 之后任何工具调用都会打到已关闭的 LanceDB/ONNX 会话上，报的却是
        # 「client is closed」这种与根因无关的错误。清空后 get() 返回 None，
        # execute() 给出明确的 "Unknown tool"。
        self._tools.clear()

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def schemas(self) -> List[Dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    # ------------------------------------------------------------------
    def execute(self, name: str, args: Dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"Unknown tool: {name}")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return ToolResult(
                success=False,
                error=f"Invalid arguments for {name}: expected a JSON object, "
                      f"got {type(args).__name__}",
            )

        # 签名级预检：把「模型传错参数」和「工具内部抛 TypeError」区分开。
        # 旧实现靠捕获 run() 抛出的 TypeError 一律报 "Invalid arguments"——
        # 工具内部代码出 bug（比如对 None 调 .strip()）时模型会收到"参数
        # 无效"，于是反复重试同样的调用而不是上报问题。
        try:
            self._signature_of(tool).bind(**args)
        except TypeError as e:
            # 参数错误最常见的后果是「模型原样重试 N 次」——它只看到
            # "missing a required argument: 'op'"，并不知道该怎么填。工具可
            # 选提供 usage_hint（一个能直接照抄的调用示例），在这里附上，
            # 让小模型有机会一次就自纠。
            msg = f"Invalid arguments for {name}: {e}"
            hint = getattr(tool, "usage_hint", "") or ""
            if hint:
                msg += f"\n正确用法（照抄这个形状）：{hint}"
            return ToolResult(success=False, error=msg)

        try:
            result = tool.run(**args)
        except TypeError as e:
            # 签名已通过 → 这里的 TypeError 是工具内部缺陷，如实报告。
            logger.exception("tool %s raised TypeError internally (args=%r)", name, args)
            return ToolResult(
                success=False,
                error=f"{name} failed internally (not an argument problem): "
                      f"{type(e).__name__}: {e}",
            )
        except Exception as e:  # pragma: no cover - defensive
            return ToolResult(success=False, error=f"{type(e).__name__}: {e}")

        if not isinstance(result, ToolResult):
            # 自定义/第三方工具忘了返回 ToolResult（返回 None 或 str）时，
            # 上层 `result.success` 会以 AttributeError 的形式炸开，且看不出
            # 是哪个工具的锅。这里归一成明确的错误。
            logger.error("tool %s returned %s instead of ToolResult",
                         name, type(result).__name__)
            return ToolResult(
                success=False,
                error=f"{name} returned {type(result).__name__} instead of ToolResult",
            )
        return result

    @staticmethod
    def _signature_of(tool: "Tool") -> "inspect.Signature":
        """按工具类缓存 ``run`` 的 Signature。

        ``inspect.signature()`` 每次调用都重新构造 Signature 对象（10-30µs，
        对 AliasTool.parameters 这类 property 还有连带开销）。工具集在注册后
        固定，按 ``type(tool)`` 缓存即可（实例级 run 未被逐实例改写）。
        """
        key = type(tool)
        sig = ToolRegistry._signature_cache.get(key)
        if sig is None:
            sig = inspect.signature(tool.run)
            ToolRegistry._signature_cache[key] = sig
        return sig


def parse_args(raw: str) -> Dict[str, Any]:
    """Parse a tool-call arguments JSON string.

    Returns ``{}`` for empty input (a model may legitimately call a no-arg
    tool with ``""``). For non-empty but malformed JSON, raises ``ValueError``
    with a clear message so the agent can report it back to the model —
    silently returning ``{}`` here would turn a JSON typo into a confusing
    "missing required argument" error one layer up, which the model cannot
    easily diagnose.
    """
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in tool arguments ({e.msg} at pos {e.pos}): {raw[:200]!r}") from e
