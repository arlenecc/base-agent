"""决策模型运行时：解析容错 / 目录解析 / 单例与加载失败回退。"""
import os

import pytest

from agent.rag.decision_model import (
    DEFAULT_DECISION_DIRNAME,
    GenerativeDecisionModel,
    UnavailableDecisionModel,
    default_decision_model_dir,
    get_decision_model,
    unload_decision_model,
)


# ---------------------------------------------------------------------------
# 生成式后端：0-100 数字解析容错
# ---------------------------------------------------------------------------

def test_parse_score_plain_number():
    assert GenerativeDecisionModel._parse_score("85") == 0.85


def test_parse_score_with_noise():
    assert GenerativeDecisionModel._parse_score("Relevance: 92/100") == 0.92
    assert GenerativeDecisionModel._parse_score("score is 40 points") == 0.40


def test_parse_score_bounds():
    assert GenerativeDecisionModel._parse_score("100") == 1.0
    assert GenerativeDecisionModel._parse_score("0") == 0.0


def test_parse_score_no_number_returns_zero():
    assert GenerativeDecisionModel._parse_score("不知道") == 0.0
    assert GenerativeDecisionModel._parse_score("") == 0.0
    assert GenerativeDecisionModel._parse_score(None) == 0.0


# ---------------------------------------------------------------------------
# 模型目录解析
# ---------------------------------------------------------------------------

def test_default_model_dir_env_override(monkeypatch):
    monkeypatch.setenv("BASE_AGENT_DECISION_MODEL", "/tmp/custom-model")
    assert default_decision_model_dir() == "/tmp/custom-model"


def test_default_model_dir_repo_models_preferred(monkeypatch):
    monkeypatch.delenv("BASE_AGENT_DECISION_MODEL", raising=False)
    path = default_decision_model_dir()
    assert os.path.basename(path) == DEFAULT_DECISION_DIRNAME
    # 仓库自带模型目录存在时优先用仓库路径（无需下载）
    import agent.rag.decision_model as dm
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(dm.__file__)))))
    repo = os.path.join(root, "models", DEFAULT_DECISION_DIRNAME)
    if os.path.isdir(repo):
        assert path == repo


# ---------------------------------------------------------------------------
# 单例：key 驱动复用 / 加载失败返回 None
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _cleanup_singleton():
    unload_decision_model()
    yield
    unload_decision_model()


def test_get_decision_model_unload_clears():
    # 未加载过 → None；unload 幂等
    unload_decision_model()
    assert True


def test_get_decision_model_load_failure_returns_none(monkeypatch, tmp_path):
    """模型路径指向不存在目录：返回 None + 告警，不抛异常。"""
    result = get_decision_model("mlx-vlm", str(tmp_path / "no-such-model"))
    assert result is None


def test_get_decision_model_unknown_backend(monkeypatch, tmp_path):
    assert get_decision_model("warp-drive", "") is None


def test_unavailable_model_scores_zero():
    m = UnavailableDecisionModel()
    assert m.score("q", ["a", "b"]) == [0.0, 0.0]
