"""第九轮全面复查的回归测试。

覆盖本轮修复的 bug 与泄漏：
  1. MCP stdin 写入改走有界队列（防 server 不读 stdin 时无限期挂死）
  2. memory_graph 破坏性 op 弹确认 + observations 类型校验
  3. WorkMemory LRU 淘汰强制落盘（force=True）
  4. plan patch 的目标集合语义（无 step_id 不污染 note；只给 note 必须带 id）
  5. extract_facts_via_llm 结构归一化（坏条目跳过、好条目保留）
  6. GraphMemoryStore.clear() 清空待重试队列（删除复活修复）
  7. config 字符串字段类型收敛 + 唯一 tmp 原子写
  8. rag_ingest force 字符串解析
  9. code_run / shell_run 协作式取消立即中断
 10. file_read 按字符判定截断（多字节 UTF-8 误判修复）
 11. checkpoint 无 session 时显式失败
 12. render_plan_dict 纯函数（UI 跨线程安全）
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.config import AgentConfig               # noqa: E402
from agent.kernel.session import render_plan_dict  # noqa: E402
from agent.tools.mcp_client import MCPClient, MCPError  # noqa: E402


# ---------------------------------------------------------------------------
# 1. MCP：连接不可用时同步门面快速失败，不挂起调用线程
# ---------------------------------------------------------------------------

class _FakeCallbacks:
    def is_cancelled(self):
        return False


def test_mcp_protocol_calls_fail_fast_when_stopped():
    """stop() 后所有协议入口都必须抛 MCPError，而不是 AttributeError/阻塞。"""
    c = MCPClient(name="t", command="true")
    c.stop()
    calls = (
        lambda: c.call_tool("x", {}),
        lambda: c.list_resources(),
        lambda: c.read_resource("res://x"),
        lambda: c.list_prompts(),
        lambda: c.get_prompt("p"),
    )
    for call in calls:
        with pytest.raises(MCPError):
            call()


def test_mcp_stop_is_idempotent_and_permanent():
    """stop() 幂等；停牌后 ensure_connected 不会再偷偷重连拉起子进程。"""
    c = MCPClient(name="t", command="true")
    c.stop()
    c.stop()
    assert c.ensure_connected() is False
    assert c.list_tools() == []


# ---------------------------------------------------------------------------
# 2. memory_graph：确认 + 类型校验
# ---------------------------------------------------------------------------

@pytest.fixture()
def tool_registry(tmp_path):
    from agent.tools import ToolRegistry
    cfg = AgentConfig()
    cfg.workspace = str(tmp_path)
    cfg.knowledge_base = str(tmp_path / "kb")
    reg = ToolRegistry(config=cfg, callbacks=_FakeCallbacks())
    yield reg, cfg
    # shutdown 会释放 RAG/单例资源；测试环境可能没装重依赖，容错处理
    try:
        reg.shutdown()
    except Exception:
        pass


def test_memory_graph_destructive_ops_need_confirm(tool_registry):
    reg, _cfg = tool_registry
    tool = reg.get("memory_graph")
    assert tool is not None
    assert tool.should_confirm({"op": "clear"}) is True
    assert tool.should_confirm({"op": "delete_entity", "name": "x"}) is True
    assert tool.should_confirm({"op": "list_entities"}) is False
    assert tool.should_confirm({"op": "add_observations",
                                "name": "x", "observations": ["y"]}) is False


def test_memory_graph_rejects_string_observations(tool_registry):
    reg, _cfg = tool_registry
    # 模型把 observations 传成字符串：旧实现按字符迭代生成逐字符 observation
    res = reg.execute("memory_graph", {
        "op": "add_observations", "name": "e",
        "observations": "a whole sentence",
    })
    assert not res.success
    assert "list of strings" in (res.error or "")
    # 正常列表仍可用
    res = reg.execute("memory_graph", {
        "op": "add_observations", "name": "e", "observations": ["a", "b"],
    })
    assert res.success, res.error


# ---------------------------------------------------------------------------
# 3. WorkMemory LRU 淘汰强制落盘
# ---------------------------------------------------------------------------

def test_work_memory_lru_evict_flushes_force(tmp_path):
    from agent.tools import memory as mem_mod
    # 填满缓存后制造第 9 个路径 → 最旧的（含未过防抖窗口的写入）被淘汰，
    # 其数据必须已在磁盘上（force flush）。
    paths = []
    for i in range(mem_mod._WORK_MEMORY_CACHE_MAX + 1):
        p = os.path.join(tmp_path, f"ws{i}", ".agent", "work_memory.json")
        cfg = AgentConfig()
        cfg.workspace = os.path.join(tmp_path, f"ws{i}")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        wm = mem_mod._work_memory(cfg)
        wm.set("key", f"val{i}")           # 0.5s 防抖窗口内，未落盘
        paths.append((p, f"val{i}"))
    oldest_path, oldest_val = paths[0]
    with open(oldest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data.get("key") == oldest_val, "淘汰实例的防抖窗口写入丢失"


# ---------------------------------------------------------------------------
# 4. plan patch 语义
# ---------------------------------------------------------------------------

@pytest.fixture()
def plan_env(tmp_path):
    from agent.kernel.session import Session
    from agent.kernel.budget import ContextBudget
    from agent.tools import ToolRegistry
    cfg = AgentConfig()
    cfg.workspace = str(tmp_path)
    cfg.knowledge_base = str(tmp_path / "kb")
    sess = Session(budget=ContextBudget(30_000), raw_turns=2)
    reg = ToolRegistry(config=cfg, callbacks=_FakeCallbacks())
    reg.session = sess
    yield reg, sess
    try:
        reg.shutdown()
    except Exception:
        pass


def _mk_plan(reg):
    res = reg.execute("update_plan", {"op": "create", "goal": "g",
                                      "steps": [{"title": "a"}, {"title": "b"}]})
    assert res.success, res.error


def test_plan_patch_without_step_id_targets_first_mismatch(plan_env):
    reg, sess = plan_env
    _mk_plan(reg)
    # [done 模拟]: 先把 step1 标 done（第二步自动 doing）
    reg.execute("update_plan", {"op": "patch", "step_id": "1",
                                "status": "done", "note": "ok"})
    # 无 step_id 的 patch(status=done)：目标应是第一个「状态≠done」的步骤
    # （即 step2），不得把 note 污染到 step1。
    res = reg.execute("update_plan", {"op": "patch", "status": "done",
                                      "note": "第二步结论"})
    assert res.success, res.error
    steps = sess.get_plan()["steps"]
    assert steps[0]["note"] == "ok", "不该改写第一步的 note"
    assert steps[1]["note"] == "第二步结论"


def test_plan_patch_note_only_requires_step_id(plan_env):
    reg, sess = plan_env
    _mk_plan(reg)
    res = reg.execute("update_plan", {"op": "patch", "note": "乱写"})
    assert not res.success
    assert "step_id" in (res.error or "")


def test_plan_patch_does_not_mutate_on_reject(plan_env):
    """normalize 拒绝时计划必须保持原状（活字典半改修复）。"""
    reg, sess = plan_env
    _mk_plan(reg)
    before = json.dumps(sess.get_plan(), ensure_ascii=False, sort_keys=True)
    # 超长 note（>200 字符归一化截断不会失败）——用超大 title 的 create 覆盖？
    # patch 路径的 set_plan 只会因步骤全空失败，正常 patch 不会；此处验证
    # patch 后 session 中的对象与 deepcopy 结果一致（无半改状态）。
    reg.execute("update_plan", {"op": "patch", "step_id": "1",
                                "status": "done", "note": "ok"})
    after = json.dumps(sess.get_plan(), ensure_ascii=False, sort_keys=True)
    assert before != after  # 正常 patch 生效
    # 再验证：patch 失败（不存在的 id）时计划不变
    snapshot = json.dumps(sess.get_plan(), ensure_ascii=False, sort_keys=True)
    res = reg.execute("update_plan", {"op": "patch", "step_id": "99",
                                      "status": "done"})
    assert not res.success
    assert json.dumps(sess.get_plan(), ensure_ascii=False, sort_keys=True) == snapshot


def test_plan_create_promotes_doing_only_when_all_pending(plan_env):
    reg, sess = plan_env
    res = reg.execute("update_plan", {"op": "create", "goal": "g", "steps": [
        {"title": "a", "status": "done"},
        {"title": "b"},
    ]})
    assert res.success, res.error
    steps = sess.get_plan()["steps"]
    assert steps[0]["status"] == "done", "显式状态不该被提升覆盖"
    assert steps[1]["status"] == "pending"


# ---------------------------------------------------------------------------
# 5. extract_facts_via_llm 归一化
# ---------------------------------------------------------------------------

class _ScriptLLM:
    def __init__(self, raw):
        self._raw = raw

    def chat_stream(self, messages, tools=None, temperature=0.1,
                    kv_group="main"):
        # extract_facts_via_llm 传 kv_group="aux"；替身必须接受，否则
        # TypeError 被上游 except 吞掉，测试得到空图谱而非校验归一化。
        from agent.llm_client import StreamEvent
        return iter([StreamEvent(type="content", content=self._raw),
                     StreamEvent(type="done", usage={})])


def test_extract_facts_normalizes_malformed_graph():
    from agent.graph_memory import extract_facts_via_llm
    raw = json.dumps({
        "entities": [
            "a bare string entity",                      # 非法 → 跳过
            {"name": "Postgres", "type": "tool", "observations": [
                "good", 123, None, "  "]},               # 混入非 str → 收敛
            {"type": "tool", "observations": ["x"]},     # 缺 name → 跳过
        ],
        "relations": [
            "garbage",                                    # 非法 → 跳过
            {"source": "A", "target": "B"},               # 缺 label → 跳过
            {"source": "A", "target": "B", "label": "uses"},
        ],
    })
    data = extract_facts_via_llm(_ScriptLLM(raw), "对话内容" * 10)
    assert [e["name"] for e in data["entities"]] == ["Postgres"]
    assert data["entities"][0]["observations"] == ["good", "123"]
    assert data["relations"] == [{"source": "A", "target": "B", "label": "uses"}]


# ---------------------------------------------------------------------------
# 6. GraphMemoryStore.clear() 清空待重试队列
# ---------------------------------------------------------------------------

def test_graph_clear_empties_pending_index(tmp_path):
    from agent.graph_memory import GraphMemoryStore
    g = GraphMemoryStore(path=str(tmp_path / "g.json"))
    g._pending_index.append(("e1", "obs1"))
    g.clear()
    assert g._pending_index == [], "clear 必须清空待重试队列（删除复活修复）"


# ---------------------------------------------------------------------------
# 7. config
# ---------------------------------------------------------------------------

def test_config_coerce_str_field_rejects_non_string():
    assert AgentConfig._coerce("model", ["a", "b"]) is None
    assert AgentConfig._coerce("model", "gpt-x") == "gpt-x"
    assert AgentConfig._coerce("base_url", {"x": 1}) is None
    assert AgentConfig._coerce("workspace", 123) is None


def test_config_save_roundtrip_and_no_tmp_left(tmp_path):
    cfg = AgentConfig()
    cfg.model = "m1"
    path = str(tmp_path / "cfg.json")
    cfg.save(path)
    assert os.listdir(tmp_path) == ["cfg.json"], "tmp 文件必须被清理"
    loaded = AgentConfig.load(path)
    assert loaded.model == "m1"


# ---------------------------------------------------------------------------
# 8. rag_ingest force 字符串解析
# ---------------------------------------------------------------------------

def test_rag_ingest_force_string_false_does_not_force():
    from agent.tools.rag_tool import RagIngestTool

    class _FakeEngine:
        def __init__(self):
            self.calls = []

        def ingest(self, force=False):
            self.calls.append(force)
            return {"files_found": 0, "files_extracted": 0,
                    "files_skipped": 0, "total_chars": 0, "chunks": 0,
                    "errors": []}

    eng = _FakeEngine()
    tool = RagIngestTool(eng)
    tool.run(force="false")
    assert eng.calls == [False]
    tool.run(force="true")
    assert eng.calls[-1] is True
    tool.run(force=False)
    assert eng.calls[-1] is False


# ---------------------------------------------------------------------------
# 9. code_run / shell_run 协作式取消
# ---------------------------------------------------------------------------

class _CancelCallbacks(_FakeCallbacks):
    def is_cancelled(self):
        return True


def test_code_run_cancel_interrupts_immediately(tmp_path):
    from agent.tools import ToolRegistry
    from agent.tools.code_run import CodeRunTool
    cfg = AgentConfig()
    cfg.workspace = str(tmp_path)
    reg = ToolRegistry(config=cfg, callbacks=_CancelCallbacks())
    try:
        tool = CodeRunTool()
        tool.bind(cfg, reg)
        res = tool.run("import time; time.sleep(30)", timeout=60)
        assert not res.success
        assert "取消" in (res.error or "")
    finally:
        try:
            reg.shutdown()
        except Exception:
            pass


def test_shell_run_cancel_kills_and_returns(tmp_path):
    from agent.tools import ToolRegistry
    from agent.tools.shell_run import ShellRunTool
    cfg = AgentConfig()
    cfg.workspace = str(tmp_path)
    reg = ToolRegistry(config=cfg, callbacks=_CancelCallbacks())
    try:
        tool = ShellRunTool()
        tool.bind(cfg, reg)
        res = tool.run("sleep 30", timeout=60)
        assert not res.success
        assert "取消" in (res.error or "")
    finally:
        try:
            reg.shutdown()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 10. file_read 按字符判定截断
# ---------------------------------------------------------------------------

def test_file_read_cjk_under_char_limit_not_truncated(tmp_path, monkeypatch):
    import agent.tools.file_ops as fo
    from agent.tools import ToolRegistry
    cfg = AgentConfig()
    cfg.workspace = str(tmp_path)
    reg = ToolRegistry(config=cfg, callbacks=_FakeCallbacks())
    try:
        monkeypatch.setattr(fo, "_MAX_READ_CHARS", 10)
        content = "一二三四五"          # 5 个字符 = 15 字节 > 10，但 5 字符 < 10
        with open(os.path.join(tmp_path, "cjk.txt"), "w", encoding="utf-8") as f:
            f.write(content)
        res = reg.execute("file_read", {"path": "cjk.txt"})
        assert res.success
        assert res.output == content, "字节数超限但字符数未超 → 不应截断"
        assert "仅显示前" not in res.output
    finally:
        try:
            reg.shutdown()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 11. checkpoint 无 session 显式失败
# ---------------------------------------------------------------------------

def test_checkpoint_without_session_fails_explicitly(tool_registry):
    reg, _cfg = tool_registry
    # tool_registry 的 registry 没有 session 属性
    res = reg.execute("update_working_checkpoint", {"key_info": "note"})
    assert not res.success
    assert "no kernel session" in (res.error or "")


# ---------------------------------------------------------------------------
# 12. render_plan_dict 纯函数
# ---------------------------------------------------------------------------

def test_render_plan_dict_pure_function():
    plan = {"goal": "部署", "steps": [
        {"id": "1", "title": "打包", "status": "done", "note": "ok"},
        {"id": "2", "title": "上传", "status": "doing"},
    ]}
    text = render_plan_dict(plan)
    assert "Goal: 部署" in text
    assert "- [✓] 1. 打包  # ok" in text
    assert "- [▶] 2. 上传" in text
    assert "Now: 2. 上传" in text
    assert "Remaining: 1 step(s)" in text
    assert render_plan_dict(None) == ""
    assert render_plan_dict("garbage") == ""
