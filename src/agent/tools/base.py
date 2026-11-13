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

# 工具参数由模型提供，可能携带密钥/token/MCP env；日志与异常回执里不能
# 原样落盘（UI 的 kernel debug log 也走同一条 logger）。
_SENSITIVE_KEY = ("key", "token", "secret", "password", "auth",
                  "credential", "env", "cookie", "_raw_arguments")


def _redact_args(args: Any) -> Dict[str, Any]:
    """给日志用的参数摘要：敏感键打码、其余截断，避免整份参数落日志。"""
    if not isinstance(args, dict):
        return {"_": f"<{type(args).__name__}>"}
    out: Dict[str, Any] = {}
    for k, v in args.items():
        key = str(k)
        if any(s in key.lower() for s in _SENSITIVE_KEY):
            out[key] = f"<redacted:{type(v).__name__}>"
        else:
            s = str(v)
            out[key] = s[:120] + ("…" if len(s) > 120 else "")
    return out


@dataclass
class ToolResult:
    success: bool
    output: str = ""
    error: str = ""
    # 仅 UI 使用的旁路数据（如检索命中的文档切片），**绝不进模型上下文**：
    # to_message() 只返回 output/error，内核回灌与历史落盘都走它。
    attachments: List[Dict[str, Any]] = field(default_factory=list)

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


class _FanoutTool(Tool):
    """把多个原子工具合并成一个「门面」工具：``op`` 参数分发到子工具。

    用于按用户要求把一组同域工具收敛成单个工具（file_* → file_tools、
    memory_* → long_memory、skill_* → skill_tools、rag_* → rag_tools）。
    子类只需声明合并后的 ``name`` / ``description`` / ``parameters``，并
    在 ``__init__`` 里填 ``self._sub_tools = {op: 子工具实例}``。

    关键语义：
    - ``bind`` 会同时给所有子工具 bind（它们各自持有 config/registry 引用）；
    - ``should_confirm`` 按 op 分发到子工具（保留 rag_ingest 的破坏性确认、
      memory clear/delete_entity 的确认等细粒度策略）；
    - ``run`` 按 op 分发，子工具抛 TypeError（参数错误）时转成可读的错误，
      而不是被 execute() 误报成「工具内部缺陷」。

    参数隔离（合并工具的核心风险）：所有子工具的参数被平铺进**同一张**
    schema 表。若不校验，模型把 A op 的参数误传给 B op 时，带 ``**kwargs``
    的子工具会静默吞掉（模型以为生效），不带 ``**kwargs`` 的会抛 TypeError，
    行为不一致。因此子类声明 ``_op_params``：每个 op 允许的参数名集合，
    分发前统一拒绝越界参数并给出该 op 的正确参数表。
    """

    _op_key = "op"
    _sub_tools: Dict[str, Tool] = {}
    # 子类声明：{op: 该 op 允许的参数名集合}（不含 op 本身）
    _op_params: Dict[str, Any] = {}
    # 子类可选：合并前原子工具自身 op 参数的旧值域 → 提示如何改写成新形态。
    # 例：memory_graph(op="clear") → long_memory(op="graph", action="clear")
    _legacy_op_hints: Dict[str, str] = {}

    def bind(self, config: "AgentConfig", registry: "ToolRegistry") -> None:
        self.config = config
        self.registry = registry
        for sub in self._sub_tools.values():
            sub.bind(config, registry)

    def should_confirm(self, args: Dict[str, Any]) -> bool:
        op = str((args or {}).get(self._op_key, "") or "").strip().lower()
        sub = self._sub_tools.get(op)
        if sub is not None:
            return sub.should_confirm(args)
        return bool(self.destructive)

    def _validate_params(self, op: str, supplied: Dict[str, Any]):
        """拒绝不属于该 op 的参数；返回 ToolResult 错误或 None。"""
        allowed = self._op_params.get(op)
        if allowed is None:
            return None  # 子类未声明 → 不做白名单（保持宽松）
        extra = set(supplied) - set(allowed) - {self._op_key}
        if extra:
            return ToolResult(
                False,
                error=(f"Invalid arguments for {self._op_key}={op!r}: "
                       f"unexpected parameter(s) {', '.join(sorted(extra))}. "
                       f"For op={op!r} use only: {', '.join(sorted(allowed))}. "
                       f"(其它 op 的参数不能混用)"),
            )
        return None

    def _unknown_op_error(self, op: str) -> ToolResult:
        hint = self._legacy_op_hints.get(op)
        detail = (f"unknown {self._op_key} {op!r}; valid ops: "
                  f"{', '.join(sorted(self._sub_tools))}")
        if hint:
            # 合并前该值曾是某个原子工具自己的 op 值（如 clear / list），
            # 直接告诉模型新形态，避免它反复用旧值重试。
            detail += f"。{hint}"
        return ToolResult(False, error=detail)

    def run(self, op: str = "", **kwargs) -> ToolResult:
        op = str(op or "").strip().lower()
        sub = self._sub_tools.get(op)
        if sub is None:
            return self._unknown_op_error(op)
        err = self._validate_params(op, kwargs)
        if err is not None:
            return err
        try:
            return sub.run(**kwargs)
        except TypeError as e:
            # 子工具因缺参抛 TypeError：转成与 execute 签名预检一致的
            # 「Invalid arguments」措辞，避免被上层误读成「工具内部缺陷」。
            return ToolResult(
                False,
                error=f"Invalid arguments for {self._op_key}={op!r}: {e}",
            )


# ---------------------------------------------------------------------------
# 向后兼容别名：把合并前的旧工具名路由到新的门面工具 + 固定 op。
#
# 设计要点（模型可见面 vs 执行面分离）：
# - 这些别名**不进 schemas()/names()**：LLM 只看到 12 个合并工具，不会因
#   新旧两套名字混淆，也不重复占用 schema token。
# - 但已落盘的会话历史、L3 SOP / 技能里记录的旧调用，以及既有测试脚本，
#   仍可通过 get()/execute() 命中，避免「工具重命名后历史数据全部失效」。
# - should_confirm 委托给门面工具（按 op 分发到原子工具），因此 rag_ingest /
#   graph clear 等破坏性操作的确认策略不因合并而丢失。
# - webexec_js 按本次需求**彻底删除**（其渲染能力已并入 web_scan 的
#   render=true），刻意不设别名。
# ---------------------------------------------------------------------------
# (旧名, 门面工具名, 固定 op, 旧参数中需要改名的键, 需要从旧参数里丢弃的键)
_LEGACY_ALIASES = [
    ("file_read", "file_tools", "read", {}, ()),
    ("file_write", "file_tools", "write", {}, ()),
    ("file_append", "file_tools", "append", {}, ()),
    ("file_modify", "file_tools", "modify", {}, ()),
    ("file_patch", "file_tools", "patch", {}, ()),
    ("file_list", "file_tools", "list", {}, ()),
    ("file_delete", "file_tools", "delete", {}, ()),
    ("file_move", "file_tools", "move", {}, ()),
    ("file_copy", "file_tools", "copy", {}, ()),
    ("file_search", "file_tools", "search", {}, ()),
    ("memory_graph", "long_memory", "graph", {"op": "action"}, ()),
    ("memory_search", "long_memory", "search", {}, ()),
    ("skill_search", "skill_tools", "search", {}, ("op",)),
    ("skill_load", "skill_tools", "load", {}, ()),
    ("skill_semantic_search", "skill_tools", "semantic_search", {}, ()),
    ("skill_save", "skill_tools", "save", {}, ()),
    ("rag_search", "rag_tools", "search", {}, ()),
    ("rag_outline", "rag_tools", "outline", {}, ()),
    ("rag_status", "rag_tools", "status", {}, ()),
    ("rag_ingest", "rag_tools", "ingest", {}, ()),
]


class _LegacyAliasTool(Tool):
    """旧工具名的兼容入口：把调用规范化后交给门面工具执行。

    本身不出现在 schemas/names 里（``_legacy=True``），只服务于 get/execute。
    """

    _legacy = True

    def __init__(self, old_name: str, facade: str, op: str,
                 renames: Dict[str, str], drop: tuple):
        self.name = old_name
        self._facade = facade
        self._op = op
        self._renames = dict(renames)
        self._drop = tuple(drop)
        self.destructive = False
        self.description = f"(legacy alias) use {facade}(op={op!r}) instead."
        self.parameters = {"type": "object", "properties": {}}

    def bind(self, config: "AgentConfig", registry: "ToolRegistry") -> None:
        self.config = config
        self.registry = registry

    def _normalize(self, args: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in (args or {}).items():
            if k in self._drop:
                continue
            if k == "op":
                # 旧调用的 op 是「原子工具自己的 op」（如 memory_graph 的
                # create_entity）：有改名规则就保留为 action，否则丢弃
                # （门面工具的 op 由 self._op 覆盖）。
                if "op" in self._renames:
                    out[self._renames["op"]] = v
                continue
            out[self._renames.get(k, k)] = v
        out["op"] = self._op
        return out

    def should_confirm(self, args: Dict[str, Any]) -> bool:
        facade = self.registry.get(self._facade)
        return facade.should_confirm(self._normalize(args)) if facade else False

    def run(self, **kwargs) -> ToolResult:
        return self.registry.execute(self._facade, self._normalize(kwargs))


class _AuxLLMProxy:
    """辅助模型懒解析代理（KG 查询期关键词抽取用）。

    每次调用都通过 getter 取当前辅助模型实例（UI 可随时加载/卸载）；
    未加载时 ``available`` 为 False、``batch_chat`` 返回空输出——KG 抽取
    降级（回退原始 query 向量）而不让检索崩溃。
    """

    def __init__(self, getter):
        self._getter = getter

    def _client(self):
        try:
            client = self._getter() if callable(self._getter) else None
            return client
        except Exception:
            return None

    @property
    def available(self) -> bool:
        return self._client() is not None

    def batch_chat(self, *args, **kwargs):
        client = self._client()
        if client is None:
            return [""]
        return client.batch_chat(*args, **kwargs)

    def __bool__(self) -> bool:
        return self._client() is not None

    def __repr__(self) -> str:
        return f"_AuxLLMProxy(loaded={self.available})"


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
        from .file_ops import FileToolsTool
        from .code_run import CodeRunTool
        from .shell_run import ShellRunTool
        from .web import WebScanTool
        from .interact import AskUserTool
        from .memory import WorkMemoryTool, LongMemoryTool
        from .skill_search import SkillToolsTool
        from .checkpoint import UpdateWorkingCheckpointTool, StartLongTermUpdateTool
        from .plan import UpdatePlanTool

        tools_to_register = [
            # 合并后的门面工具（op 参数分发）
            FileToolsTool(),          # read/write/append/modify/patch/list/delete/move/copy/search
            LongMemoryTool(),         # graph/search
            SkillToolsTool(),         # search/load/semantic_search/save
            CodeRunTool(),
            ShellRunTool(),
            WebScanTool(),            # 唯一 web 工具（含 render=true 浏览器渲染）
            AskUserTool(),
            WorkMemoryTool(),
            # 沉淀态原子工具（对齐 GenericAgent）
            UpdateWorkingCheckpointTool(),
            StartLongTermUpdateTool(),
            # 规划态原子工具：目标 + 路径的一等公民表示
            UpdatePlanTool(),
        ]

        for t in tools_to_register:
            t.bind(self.config, self)
            self._tools[t.name] = t

        # Register RAG tools（rag_tools 在这里注册，别名要等它就绪）
        self._register_rag_tools()

        # 向后兼容别名：注册进 _tools（get/execute 能命中），但带 _legacy
        # 标记，不进 schemas()/names()——模型只看到合并后的工具集。
        for old, facade, op, renames, drop in _LEGACY_ALIASES:
            if facade in self._tools and old not in self._tools:
                a = _LegacyAliasTool(old, facade, op, renames, drop)
                a.bind(self.config, self)
                self._tools[old] = a

        # Register MCP tools from configured servers
        self._register_mcp_tools()

    def _register_mcp_tools(self) -> None:
        """Connect to configured MCP servers and register their tools.

        连接移到 daemon 后台线程：旧实现在构造函数里同步 `client.start()`
        （握手 timeout，默认 30s）+ `list_tools()`（再等一个 timeout），且
        `ToolRegistry.__init__` 跑在 **UI 主线程**（_rebuild_agent）——配置
        N 个 server 最坏 N×60s 冻结整个界面。改为后台连接：注册表先以内置
        工具可用，MCP 就绪后一次性并入（dict.update 原子），并通过
        ``on_mcp_tools_changed`` 回调让内核失效 schema 缓存（下一轮生效）。
        """
        mcp_servers = getattr(self.config, "mcp_servers", None) or []
        if not mcp_servers:
            return

        import threading
        threading.Thread(target=self._connect_mcp_servers, args=(mcp_servers,),
                         daemon=True, name="mcp-connect").start()

    def _connect_mcp_servers(self, mcp_servers: list) -> None:
        from .mcp_client import MCPClient, MCPError
        from .mcp_tool import MCPTool

        for server_cfg in mcp_servers:
            if not isinstance(server_cfg, dict):
                continue
            name = server_cfg.get("name", "")
            command = server_cfg.get("command", "")
            url = server_cfg.get("url", "")
            if not name or (not command and not url):
                # server_cfg 可能含 mcp_servers[].env / headers（整份 API
                # key/token），日志里只打印键名，绝不打印值。
                logger.warning(
                    "MCP server config missing name or command/url (keys=%s)",
                    list(server_cfg.keys()))
                continue

            try:
                # command → stdio 子进程（旧配置，继续兼容）；url → fastmcp
                # 的 streamable-HTTP（默认）或 SSE 远程 server。transport /
                # headers / cwd / timeout 均为可选。
                client = MCPClient(
                    name=name,
                    command=command,
                    args=server_cfg.get("args", []),
                    env=server_cfg.get("env"),
                    timeout=float(server_cfg.get("timeout") or 30.0),
                    url=url,
                    transport=server_cfg.get("transport"),
                    headers=server_cfg.get("headers"),
                    cwd=server_cfg.get("cwd"),
                )
                client.start()
                self._mcp_clients.append(client)

                # 先并进 staging dict，最后一次性 update 进注册表：后台线程
                # 逐键写 `self._tools` 会与 worker 线程的 dispatch/schemas 读
                # 交错（半注册状态）。dict.update 单调用原子，读侧只会看到
                # 「全有或全无」。
                staged = {}
                registered = 0
                for tool_def in client.list_tools():
                    t = MCPTool(client, tool_def)
                    if t.name and t.name not in self._tools and t.name not in staged:
                        t.bind(self.config, self)
                        staged[t.name] = t
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
                    # 一个工具都没注册上（全重名 / server 暴露空列表）→ 这条
                    # 连接（stdio 子进程 / HTTP 会话 + 后台 event loop 线程）
                    # 纯属白占。立刻停掉并不留在 _mcp_clients 里，否则要挂到
                    # shutdown 才回收。
                    logger.warning(
                        "MCP server '%s' 未注册任何工具（重名或空列表），停止该 server",
                        name,
                    )
                    self._mcp_clients.remove(client)
                    try:
                        client.stop()
                    except Exception:
                        pass
                    continue
                if staged:
                    self._tools.update(staged)
                    cb = getattr(self, "on_mcp_tools_changed", None)
                    if callable(cb):
                        try:
                            cb()
                        except Exception as e:
                            logger.warning("MCP tools-changed callback error: %s", e)
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

            from ..rag.engine import RAGEngine, effective_embedding_model
            from ..rag.vector_store import DEFAULT_RERANK_MODEL
            from .rag_tool import RagToolsTool

            # 打印 RAG 管线「实际采用」的各模型：local 模式优先 *_local_* 字段，
            # 与 engine 的 effective_embedding_model 同口径；api 模式才用
            # fastembed 网络模型名。否则会出现「日志显示 nomic 实际跑 jina」这类误导。
            _cfg = self.config

            def _prov(field: str, default: str = "local") -> str:
                return str(getattr(_cfg, field, default) or default).strip().lower()

            # embedding：local → local_embedding_model；api → rag_embedding_model
            if _prov("embedding_provider") == "local":
                _emb_model = getattr(_cfg, "local_embedding_model", "") or ""
                _emb_prov = "local"
            else:
                _emb_model = getattr(_cfg, "rag_embedding_model", "nomic-ai/nomic-embed-text-v1.5-Q")
                _emb_prov = "api"
            if not _emb_model:
                _emb_model = "nomic-ai/nomic-embed-text-v1.5-Q"

            # reranker：local → local_reranker_model；api → rag_rerank_model
            if _prov("reranker_provider") == "local":
                _rr_model = getattr(_cfg, "local_reranker_model", "") or ""
                _rr_prov = "local"
            else:
                _rr_model = getattr(_cfg, "rag_rerank_model", "BAAI/bge-reranker-v2-m3-int8")
                _rr_prov = "api"
            _rr_enabled = bool(getattr(_cfg, "rag_rerank_enabled", True))

            # pdf 解析引擎 / OCR 后端+模型 / VLM
            _pdf = getattr(_cfg, "rag_pdf_engine", "docling") or "docling"
            _ocr_engine = getattr(_cfg, "rag_ocr_engine", "mlx-vlm") or "mlx-vlm"
            _ocr_model = getattr(_cfg, "rag_ocr_model_name", "") or ""
            _ocr = f"{_ocr_engine} / {_ocr_model}" if _ocr_model else _ocr_engine
            if _prov("vlm_provider") == "local":
                _vlm = getattr(_cfg, "local_vlm_model", "") or "(本地默认)"
            else:
                _vlm = "api (云端)"

            logger.info(
                "RAG: initializing engine\n"
                "  workspace : %s\n"
                "  kb        : %s\n"
                "  embedding : %s  [provider=%s]\n"
                "  reranker  : %s  [provider=%s, %s]\n"
                "  pdf       : %s\n"
                "  ocr       : %s\n"
                "  vlm       : %s",
                self.config.workspace, kb,
                _emb_model, _emb_prov,
                _rr_model, _rr_prov, "enabled" if _rr_enabled else "disabled",
                _pdf, _ocr, _vlm,
            )
            engine = RAGEngine(
                workspace=self.config.workspace,
                knowledge_base=kb,
                embedding_model=_emb_model,
                rerank_model=getattr(self.config, "rag_rerank_model",
                                     DEFAULT_RERANK_MODEL),
                rerank_enabled=bool(getattr(self.config, "rag_rerank_enabled", True)),
                # 切片参数此前从未从 config 接线（引擎一直用自身默认 1000/50/5%），
                # rag_chunk_size / rag_chunk_overlap / rag_chunk_overlap_percent
                # 是死配置。补上传参，让切片大小/重叠真正可配置。
                # 单位 = token（估算器已按 bge-small-zh 校准：中文 1 字/token）。
                chunk_size=int(getattr(self.config, "rag_chunk_size", 800)),
                chunk_overlap=int(getattr(self.config, "rag_chunk_overlap", 50)),
                overlap_percent=float(getattr(self.config, "rag_chunk_overlap_percent", 0.10)),
                use_semantic=bool(getattr(self.config, "rag_chunk_semantic", True)),
                # 窗口夹紧用实际生效的本地模型（local 模式取 local_embedding_model），
                # 避免 rag_embedding_model 的遗留值把 800 夹成 bge 的 510。
                clamp_model=effective_embedding_model(self.config),
                # 阈值此前只经 UI 的 configure_local_models 注入模块级全局，
                # 脚本/嵌入式/直接构造 Agent 的路径永远拿到硬编码 0.5。
                # 由引擎自己持有并显式下传（非法值在引擎构造里安全回退）。
                min_similarity=getattr(self.config, "rag_rerank_min_similarity", 0.5),
                # 知识图谱检索策略（naive/local/global/hybrid/mix，spec 第八步）。
                kg_strategy=getattr(self.config, "rag_kg_strategy", "naive"),
                # 三元组抽取方案（llm_raw/llm_pydantic/gliner，构建+检索共用）。
                kg_extractor=getattr(self.config, "rag_kg_extractor", "llm_raw"),
                # 单片抽取上限（prompt 约束 + 入库兜底）。
                kg_max_entities=getattr(self.config, "rag_kg_max_entities", 20),
                kg_max_relations=getattr(self.config, "rag_kg_max_relations", 50),
                kg_max_description_chars=getattr(
                    self.config, "rag_kg_max_description_chars", 100),
                kg_glean=bool(getattr(self.config, "rag_kg_glean_enabled", True)),
                kg_raw_json=bool(getattr(self.config, "rag_kg_llm_raw_json", False)),
            )
            self._rag_engine = engine
            # KG 查询期关键词抽取用辅助模型：工具注册时已持有 aux_llm_getter，
            # 每次调用取最新实例（辅助模型可能被 UI 加载/卸载）。
            if getattr(self, "_aux_llm_getter", None) is not None:
                engine.set_aux_llm(_AuxLLMProxy(self._aux_llm_getter))

            registered = []
            for t in [RagToolsTool(engine, aux_llm_getter=self._aux_llm_getter)]:
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
            # 工具对象仍持有刚 close 的引擎；若不摘掉，shutdown 后一次
            # execute() 会经 _get_store() 静默重建 LanceDB 连接并重新加载
            # embedding 权重（~600MB C++ 堆），且拿到半死引擎的状态。
            # 覆盖两种形态：合并后的 rag_tools 门面（引擎在 _sub_tools 里）
            # 与直接注册进 _tools 的 RAG 原子工具（含测试/兼容别名场景）。
            _seen: set = set()
            for _t in list(self._tools.values()):
                _candidates = [_t]
                _candidates.extend(getattr(_t, "_sub_tools", {}).values())
                for _cand in _candidates:
                    if id(_cand) in _seen:
                        continue
                    _seen.add(id(_cand))
                    if hasattr(_cand, "_engine"):
                        _cand._engine = None
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
        # 兼容别名不对外暴露：模型只看到合并后的工具名。
        return [n for n, t in self._tools.items()
                if not getattr(t, "_legacy", False)]

    def schemas(self) -> List[Dict[str, Any]]:
        # 同上：别名不进下发给 LLM 的 schema（省 token、避免新旧名混淆）。
        return [t.schema() for t in self._tools.values()
                if not getattr(t, "_legacy", False)]

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
            logger.exception("tool %s raised TypeError internally (args=%s)",
                             name, _redact_args(args))
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
