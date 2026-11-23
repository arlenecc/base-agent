"""Outlines 约束解码集成的轻量测试（不加载真实模型）。"""
import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from agent.config import AgentConfig
from agent.rag.kg.extractor import create_extractor, PydanticLLMExtractor


def test_config_switch_defaults():
    """约束解码默认开启；llm_raw JSON 子模式开、分隔符子模式关。"""
    cfg = AgentConfig()
    assert cfg.rag_kg_constrained_decode is True
    assert cfg.rag_kg_raw_constrained_json is True
    assert cfg.rag_kg_raw_constrained_delim is False
    for field in ("rag_kg_constrained_decode", "rag_kg_raw_constrained_json",
                  "rag_kg_raw_constrained_delim"):
        assert field in AgentConfig._BOOL_FIELDS


def test_create_extractor_passes_constrained():
    """工厂把 constrained 开关传入 PydanticLLMExtractor。"""
    ext = create_extractor("llm_pydantic", None, constrained=True)
    assert isinstance(ext, PydanticLLMExtractor)
    assert ext.constrained is True
    ext2 = create_extractor("llm_pydantic", None)
    assert ext2.constrained is False


def test_extract_one_falls_back_without_chat_constrained():
    """client 不支持 chat_constrained 时回退自由生成（_chat）。"""
    calls = []

    class _FakeChatClient:
        def batch_chat(self, messages_list, **kwargs):
            calls.append((messages_list, kwargs))
            return ['{"entities":[],"relations":[]}']

    ext = create_extractor("llm_pydantic", _FakeChatClient(), constrained=True)
    out = ext._extract_one("sys", "user")
    assert out == '{"entities":[],"relations":[]}'
    assert calls and calls[0][1].get("kv_group") == "kg"


def test_local_client_has_chat_constrained():
    """LocalLLMClient 暴露 chat_constrained 接口（MLX 约束解码入口）。"""
    from agent.local.llm import LocalLLMClient
    assert callable(getattr(LocalLLMClient, "chat_constrained", None))


def test_raw_extractor_constrained_flags():
    """llm_raw 两种子模式的约束开关独立生效（总开关关闭时全部失效）。"""
    from agent.rag.kg.extractor import KGExtractor

    # 分隔符模式：默认不受约束
    raw = create_extractor("llm_raw", None, constrained=True,
                           raw_json=False)
    assert isinstance(raw, KGExtractor)
    assert raw.constrained_delim is False
    assert raw.constrained_json is True   # 子默认值仍带，只是本模式不用

    # 分隔符模式显式开约束
    raw_d = create_extractor("llm_raw", None, constrained=True,
                             raw_json=False, raw_constrained_delim=True)
    assert raw_d.constrained_delim is True

    # 总开关关闭 → 两个子开关都失效
    raw_off = create_extractor("llm_raw", None, constrained=False,
                               raw_json=True, raw_constrained_json=True)
    assert raw_off.constrained_json is False
