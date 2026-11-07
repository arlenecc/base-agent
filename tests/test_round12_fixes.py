"""第十二轮全量复查的回归测试。

覆盖本轮修复：
* graph_memory 加载时的畸形 entities/relations 归一化
* skills：create_dir_skill 唯一目录 + keywords=None + 长请求键防碰撞
* skill_index：畸形 skill.json / _index.json 不让 scan/search 崩
* skill_retriever：BOM frontmatter、_distance_to_similarity(None)
* document_outline：空 ATX 标题行不吞下一行
* cleaner：字母比例计数与旧逐字符实现等价
* risk：``> ~/...`` / ``>> ../...`` 重定向判风险
* skill_search：external-dir 基线用检索器自己的 skills 根
* mcp_client：stop() 后显式 start() 重建干净 session
"""
import json
import os
import sys
import threading

import pytest

sys.path.insert(0, "src")

from agent.graph_memory import GraphMemoryStore  # noqa: E402
from agent.skills import SkillManager  # noqa: E402
from agent.skill_index import SkillIndex  # noqa: E402
from agent.skill_retriever import (  # noqa: E402
    SkillScanner,
    SkillVectorStore,
    _parse_frontmatter,
)
from agent.rag.document_outline import build_digest  # noqa: E402
from agent.rag.cleaner import clean_text  # noqa: E402
from agent.tools.risk import is_risky_command  # noqa: E402
from agent.tools.skill_search import _is_external_skill_dir  # noqa: E402


# ---------------------------------------------------------------------------
# graph_memory：畸形 JSON 不得打炸记忆系统
# ---------------------------------------------------------------------------


class TestGraphMemoryLoadValidation:
    def _store(self, tmp_path):
        return GraphMemoryStore(path=str(tmp_path / "g.graph.json"))

    def test_malformed_relations_are_dropped(self, tmp_path):
        path = tmp_path / "g.graph.json"
        path.write_text(json.dumps({
            "entities": {"A": {"name": "A", "type": "t", "observations": ["x"]}},
            "relations": [
                None,
                "not-a-dict",
                {"source": "A"},                      # 缺 target/label
                {"source": "A", "target": "B", "label": "knows"},
            ],
        }), encoding="utf-8")
        store = self._store(tmp_path)
        rels = store.list_relations()
        assert len(rels) == 1
        assert rels[0]["label"] == "knows"
        # snapshot / 写入路径都不得抛
        store.snapshot()
        store.add_observations("A", ["y"])

    def test_malformed_entities_are_skipped(self, tmp_path):
        path = tmp_path / "g.graph.json"
        path.write_text(json.dumps({
            "entities": {
                "A": "not-a-dict",
                "B": {"name": "B", "observations": "not-a-list"},
                "C": {"name": "C", "observations": ["ok"]},
            },
            "relations": [],
        }), encoding="utf-8")
        store = self._store(tmp_path)
        names = {e["name"] for e in store.list_entities()}
        assert names == {"B", "C"}
        assert store.get_entity("B")["observations"] == []

    def test_non_dict_root_falls_back_to_empty(self, tmp_path):
        path = tmp_path / "g.graph.json"
        path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
        store = self._store(tmp_path)
        assert store.list_entities() == []
        assert store.list_relations() == []


# ---------------------------------------------------------------------------
# skills
# ---------------------------------------------------------------------------


class TestSkillManagerFixes:
    def test_create_dir_skill_writes_valid_files(self, tmp_path):
        skills_dir = tmp_path / ".agent" / "skills"
        sm = SkillManager(path=str(tmp_path / "skills.json"),
                          skills_dir=str(skills_dir))
        d = sm.create_dir_skill("My Skill!", keywords=["PDF"], prompt="do it")
        meta = json.loads(open(os.path.join(d, "skill.json"), encoding="utf-8").read())
        assert meta["name"] == "My Skill!"
        assert meta["keywords"] == ["pdf"]
        assert meta["entry"] == "prompt.md"
        assert open(os.path.join(d, "prompt.md"), encoding="utf-8").read() == "do it"

    def test_create_dir_skill_unique_dirs(self, tmp_path):
        skills_dir = tmp_path / ".agent" / "skills"
        sm = SkillManager(path=str(tmp_path / "skills.json"),
                          skills_dir=str(skills_dir))
        d1 = sm.create_dir_skill("dup", keywords=[], prompt="a")
        d2 = sm.create_dir_skill("dup", keywords=[], prompt="b")
        assert d1 != d2
        assert os.path.isdir(d1) and os.path.isdir(d2)
        # 第二个目录的内容没被第一个覆盖
        assert open(os.path.join(d2, "prompt.md"), encoding="utf-8").read() == "b"

    def test_keywords_none_tolerated(self, tmp_path):
        skills_dir = tmp_path / ".agent" / "skills"
        sm = SkillManager(path=str(tmp_path / "skills.json"),
                          skills_dir=str(skills_dir))
        # keywords=None 不能抛 TypeError
        d = sm.create_dir_skill("nk", keywords=None, prompt="p")
        assert os.path.isdir(d)

    def test_corrupt_store_repair_persists(self, tmp_path):
        p = tmp_path / "skills.json"
        p.write_text(json.dumps({"skills": None, "requests": None}), encoding="utf-8")
        SkillManager(path=str(p))  # 构造即修复
        on_disk = json.loads(p.read_text(encoding="utf-8"))
        assert on_disk["skills"] == []
        assert on_disk["requests"] == {}

    def test_repair_does_not_swallow_next_write(self, tmp_path):
        """修复落盘不得把随后真正的 set() 吞进防抖窗口。"""
        p = tmp_path / "skills.json"
        p.write_text(json.dumps({"skills": None}), encoding="utf-8")
        sm = SkillManager(path=str(p))
        sm.create_skill(name="greet", keywords=["hello"], prompt="say hi")
        sm2 = SkillManager(path=str(p))
        assert any(s.name == "greet" for s in sm2.list())

    def test_long_request_keys_do_not_collide(self):
        base = " ".join(f"tok{i:03d}" for i in range(40))  # > 80 chars
        k1 = SkillManager._request_key(base + " alpha")
        k2 = SkillManager._request_key(base + " beta")
        assert k1 != k2
        assert len(k1) > 80 and "#" in k1  # 截断 + 哈希后缀
        # 短 key 保持原样（兼容已有持久化计数）
        assert SkillManager._request_key("read data.txt") == "data read txt"


# ---------------------------------------------------------------------------
# skill_index：畸形输入
# ---------------------------------------------------------------------------


class TestSkillIndexRobustness:
    def test_malformed_skill_json_skipped_not_fatal(self, tmp_path):
        d = tmp_path / "skills"
        (d / "bad").mkdir(parents=True)
        (d / "bad" / "skill.json").write_text(
            json.dumps({"name": 123, "keywords": "not-a-list"}), encoding="utf-8")
        (d / "bad" / "prompt.md").write_text("x", encoding="utf-8")
        (d / "good").mkdir(parents=True)
        (d / "good" / "skill.json").write_text(
            json.dumps({"name": "Good", "keywords": ["a"]}), encoding="utf-8")
        (d / "good" / "prompt.md").write_text("x", encoding="utf-8")
        idx = SkillIndex(str(d))
        n = idx.scan(force=True)
        assert n == 1
        results = idx.search("Good")
        assert [r["name"] for r in results] == ["Good"]

    def test_index_json_non_dict_records_filtered(self, tmp_path):
        d = tmp_path / "skills"
        d.mkdir()
        (d / "_index.json").write_text(json.dumps({
            "skills": ["not-a-dict", {"name": "Kept", "keywords": ["k"]}],
            "updated_at_ts": 0,
        }), encoding="utf-8")
        idx = SkillIndex(str(d))
        # 非 dict 记录被过滤，search 不炸
        idx.search("Kept")
        assert all(isinstance(r, dict) for r in idx.list_all())

    def test_list_all_returns_copies(self, tmp_path):
        d = tmp_path / "skills"
        (d / "s").mkdir(parents=True)
        (d / "s" / "skill.json").write_text(
            json.dumps({"name": "S"}), encoding="utf-8")
        (d / "s" / "prompt.md").write_text("x", encoding="utf-8")
        idx = SkillIndex(str(d))
        rows = idx.list_all()
        rows[0]["name"] = "MUTATED"
        assert idx.list_all()[0]["name"] == "S"


# ---------------------------------------------------------------------------
# skill_retriever
# ---------------------------------------------------------------------------


class TestSkillRetrieverFixes:
    def test_frontmatter_with_bom(self):
        fm = _parse_frontmatter("\ufeff---\nname: x\ndescription: d\n---\nbody")
        assert fm.get("name") == "x"
        assert fm.get("description") == "d"

    def test_distance_none_is_zero(self):
        assert SkillVectorStore._distance_to_similarity(None) == 0.0
        assert SkillVectorStore._distance_to_similarity(0.0) == 1.0
        assert SkillVectorStore._distance_to_similarity("bad") == 0.0

    def test_scanner_tolerates_missing_dir(self, tmp_path):
        # 目录不存在时 scan 不得抛
        scanner = SkillScanner(str(tmp_path / "nope"))
        assert scanner.scan() == []


# ---------------------------------------------------------------------------
# document_outline / cleaner / risk
# ---------------------------------------------------------------------------


class TestOutlineAndCleaner:
    def test_empty_atx_heading_does_not_swallow_next_line(self):
        md = "#\n真实标题行\n\n正文内容"
        _digest, chapters = build_digest(md)
        # 旧正则 ``\s+`` 跨行：单独的 ``#`` 行会把下一行正文吞成标题
        assert all(c.title.strip() != "真实标题行" for c in chapters)

    def test_heading_still_extracted(self):
        md = "# 第一章\n\n内容\n\n## 1.1 小节\n\n更多内容"
        _digest, chapters = build_digest(md)
        titles = [c.title for c in chapters]
        assert any("第一章" in t for t in titles)

    def test_cleaner_keeps_mixed_short_lines(self):
        # 与旧逐字符实现的判定保持一致：字母数字占比高的短行必须保留
        assert "AT&T; 财报" in clean_text("AT&T; 财报\n")
        assert clean_text("!!!???###\n") == ""


class TestRiskRedirect:
    def test_home_redirect_is_risky(self, tmp_path):
        ws = str(tmp_path)
        assert is_risky_command("echo x > ~/.bashrc", ws) is True
        assert is_risky_command("echo x >> ~/.zshrc", ws) is True

    def test_parent_redirect_is_risky(self, tmp_path):
        ws = str(tmp_path / "ws")
        assert is_risky_command("echo x >> ../outside.txt", ws) is True

    def test_workspace_redirect_not_risky(self, tmp_path):
        ws = str(tmp_path / "ws")
        assert is_risky_command("echo x > ./inside.txt", ws) is False
        assert is_risky_command("cat a.txt > b.txt", ws) is False


# ---------------------------------------------------------------------------
# skill_search external-dir 基线
# ---------------------------------------------------------------------------


class TestExternalSkillDir:
    def test_normal_skill_not_flagged(self, tmp_path):
        root = tmp_path / "skills"
        inside = root / "pdf"
        assert _is_external_skill_dir(str(inside), str(root)) is False

    def test_symlinked_skill_flagged(self, tmp_path):
        root = tmp_path / "skills"
        root.mkdir()
        outside = tmp_path / "outside-skill"
        outside.mkdir()
        link = root / "ext"
        try:
            os.symlink(str(outside), str(link))
        except (OSError, NotImplementedError):  # pragma: no cover
            pytest.skip("symlink not available")
        resolved = os.path.realpath(str(link))
        assert _is_external_skill_dir(resolved, str(root)) is True


# ---------------------------------------------------------------------------
# mcp_client：start() 清残留写队列
# ---------------------------------------------------------------------------


class TestMCPClientRestart:
    def test_restart_rebuilds_clean_session(self):
        """stop() 之后显式 start() 能复活，且不残留上一次 session 的状态。"""
        from fastmcp import FastMCP
        from agent.tools.mcp_client import MCPClient

        mcp = FastMCP("mcp-restart")

        @mcp.tool()
        def echo(msg: str) -> str:
            return msg

        c = MCPClient(name="t", server=mcp)
        c.start()
        try:
            assert c.call_tool("echo", {"msg": "one"}) == "one"
            c.stop()
            assert c.ensure_connected() is False
            # 显式 start 是用户意图：清掉停牌并重建全新 session
            c.start()
            assert c.ensure_connected() is True
            assert c.call_tool("echo", {"msg": "two"}) == "two"
            assert [t["name"] for t in c.list_tools()] == ["echo"]
        finally:
            c.stop()
