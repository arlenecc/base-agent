"""Tests for AgentConfig: persistence, defaults, and migration of stale values."""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from agent.config import AgentConfig


def test_load_returns_defaults_when_missing(tmp_path):
    p = tmp_path / "missing.json"
    cfg = AgentConfig.load(str(p))
    assert cfg.rag_embedding_model == "nomic-ai/nomic-embed-text-v1.5-Q"
    assert cfg.base_url == "https://api.openai.com/v1"


def test_load_preserves_known_keys(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({
        "base_url": "https://api.deepseek.com/v1",
        "api_key": "sk-abc",
        "model": "deepseek-chat",
        "rag_embedding_model": "nomic-ai/nomic-embed-text-v1.5-Q",
        "rag_chunk_size": 800,
    }))
    cfg = AgentConfig.load(str(p))
    assert cfg.base_url == "https://api.deepseek.com/v1"
    assert cfg.api_key == "sk-abc"
    assert cfg.model == "deepseek-chat"
    assert cfg.rag_chunk_size == 800


def test_load_ignores_unknown_keys(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({
        "base_url": "http://x",
        "obsolete_field": "should_be_ignored",
        "another_unknown": 123,
    }))
    # Should not raise; unknown keys silently dropped.
    cfg = AgentConfig.load(str(p))
    assert cfg.base_url == "http://x"
    assert not hasattr(cfg, "obsolete_field")


def test_save_round_trip(tmp_path):
    p = tmp_path / "c.json"
    cfg = AgentConfig(base_url="http://api.example.com/v1", api_key="k", model="m")
    cfg.rag_chunk_size = 800
    cfg.save(str(p))
    loaded = AgentConfig.load(str(p))
    assert loaded.base_url == "http://api.example.com/v1"
    assert loaded.api_key == "k"
    assert loaded.rag_chunk_size == 800


# ---------------------------------------------------------------------------
# Migration: stale embedding model names from pre-LanceDB era
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stale_name", [
    "all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-zh-v1.5",
    "BAAI/bge-large-zh-v1.5",
    "",
])
def test_migrate_deprecated_embedding_models(stale_name, tmp_path):
    """Stale embedding model names saved before the LanceDB migration must be
    silently rewritten to the current default so old config files don't break
    RAG sync."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps({
        "base_url": "http://x",
        "api_key": "",
        "model": "m",
        "rag_embedding_model": stale_name,
    }))
    cfg = AgentConfig.load(str(p))
    assert cfg.rag_embedding_model == "nomic-ai/nomic-embed-text-v1.5-Q"


def test_migrate_preserves_valid_embedding_model(tmp_path):
    """A currently-supported embedding model name must not be touched."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps({
        "base_url": "http://x",
        "api_key": "",
        "model": "m",
        "rag_embedding_model": "nomic-ai/nomic-embed-text-v1.5-Q",
    }))
    cfg = AgentConfig.load(str(p))
    assert cfg.rag_embedding_model == "nomic-ai/nomic-embed-text-v1.5-Q"


def test_migrate_nonquantized_to_quantized(tmp_path):
    """Non-quantized nomic-embed-text-v1.5 (~548MB) must be migrated to the
    quantized -Q variant (~137MB) on load: same 768-dim, 4x smaller."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps({
        "base_url": "http://x",
        "api_key": "",
        "model": "m",
        "rag_embedding_model": "nomic-ai/nomic-embed-text-v1.5",
    }))
    cfg = AgentConfig.load(str(p))
    assert cfg.rag_embedding_model == "nomic-ai/nomic-embed-text-v1.5-Q"


def test_migrate_persisted_after_save(tmp_path):
    """After migration, saving the config should persist the corrected value
    so the migration only runs once."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps({
        "base_url": "http://x",
        "api_key": "",
        "model": "m",
        "rag_embedding_model": "all-MiniLM-L6-v2",
    }))
    cfg = AgentConfig.load(str(p))
    cfg.save(str(p))  # persist the migrated value
    # Reload — should now read the corrected value from disk directly.
    cfg2 = AgentConfig.load(str(p))
    assert cfg2.rag_embedding_model == "nomic-ai/nomic-embed-text-v1.5-Q"


def test_migrate_bumps_small_max_tokens(tmp_path):
    """Older configs saved max_tokens=4096 (the old LLMClient default). Migration
    must bump any sub-8192 value to the current default (32768) so users
    upgrading don't keep hitting the tiny old budget."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"max_tokens": 4096, "max_context_tokens": 4096}))
    cfg = AgentConfig.load(str(p))
    assert cfg.max_tokens == 32768, f"expected 32768, got {cfg.max_tokens}"
    assert cfg.max_context_tokens == 32768, f"expected 32768, got {cfg.max_context_tokens}"


def test_migrate_preserves_large_max_tokens(tmp_path):
    """If the user explicitly set a large max_tokens, migration must not touch it."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"max_tokens": 16384, "max_context_tokens": 16384}))
    cfg = AgentConfig.load(str(p))
    assert cfg.max_tokens == 16384
    assert cfg.max_context_tokens == 16384


def test_default_max_tokens_is_32768():
    """The default max_tokens must be 32768, not the old 4096."""
    cfg = AgentConfig()
    assert cfg.max_tokens == 32768
    assert cfg.max_context_tokens == 32768


def test_default_aux_model_params_tuned():
    """Aux model defaults tuned for speed/faithfulness (2026-10-31): smaller
    output cap, low temp, mild repetition penalty; thinking stays off."""
    cfg = AgentConfig()
    assert cfg.aux_max_tokens == 8192
    assert cfg.aux_temperature == 0.3
    assert cfg.aux_repetition_penalty == 1.1
    assert cfg.aux_local_llm_enable_thinking is False


def test_migrate_aux_defaults_to_tuned(tmp_path):
    """Older configs at the legacy aux defaults (32768 / 0.7 / 1.0) must be
    migrated to the tuned defaults (8192 / 0.3 / 1.1)."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps({
        "aux_max_tokens": 32768,
        "aux_temperature": 0.7,
        "aux_repetition_penalty": 1.0,
    }))
    cfg = AgentConfig.load(str(p))
    assert cfg.aux_max_tokens == 8192, f"expected 8192, got {cfg.aux_max_tokens}"
    assert cfg.aux_temperature == 0.3, f"expected 0.3, got {cfg.aux_temperature}"
    assert cfg.aux_repetition_penalty == 1.1, f"expected 1.1, got {cfg.aux_repetition_penalty}"


def test_migrate_preserves_explicit_aux_params(tmp_path):
    """Explicitly chosen aux params must survive migration."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps({
        "aux_max_tokens": 16384,
        "aux_temperature": 0.9,
        "aux_repetition_penalty": 1.2,
    }))
    cfg = AgentConfig.load(str(p))
    assert cfg.aux_max_tokens == 16384
    assert cfg.aux_temperature == 0.9
    assert cfg.aux_repetition_penalty == 1.2


def test_ensure_workspace_creates_dir(tmp_path):
    cfg = AgentConfig(workspace=str(tmp_path / "ws"))
    cfg.ensure_workspace()
    assert os.path.isdir(str(tmp_path / "ws"))


def test_coerce_string_bools(tmp_path):
    """`"rag_auto_ingest": "false"` was `bool("false")` == True — the user's
    explicit opt-out silently became opt-in on every load."""
    for raw, expected in [("false", False), ("no", False), ("0", False),
                          ("true", True), ("1", True)]:
        p = tmp_path / "c.json"
        p.write_text(json.dumps({"rag_auto_ingest": raw}))
        assert AgentConfig.load(str(p)).rag_auto_ingest is expected, raw


def test_coerce_null_string_fields_fall_back_to_default(tmp_path):
    """`"api_key": null` used to become the literal string "None" — then
    `base_url + "/models"` and friends used it as a real value, and
    `workspace = "None"` created a directory literally named None."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"api_key": None, "workspace": None}))
    cfg = AgentConfig.load(str(p))
    assert cfg.api_key == AgentConfig().api_key
    assert cfg.workspace == AgentConfig().workspace
    assert cfg.workspace != "None"


# ---------------------------------------------------------------------------
# Robustness: corrupted config files and user-chosen small budgets
# ---------------------------------------------------------------------------

def test_load_recovers_from_corrupted_config(tmp_path):
    """A config.json truncated by a crash (or hand-edited badly) must not make
    the app unreadable: back it up and start from defaults."""
    p = tmp_path / "c.json"
    p.write_text('{"base_url": "http://x", "api_key":')
    cfg = AgentConfig.load(str(p))
    assert cfg.base_url == AgentConfig().base_url
    # The bad file is kept for inspection instead of being silently lost.
    assert os.path.exists(str(p) + ".corrupt")


def test_load_rejects_non_object_root(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(["not", "an", "object"]))
    cfg = AgentConfig.load(str(p))
    assert cfg.base_url == AgentConfig().base_url


def test_load_coerces_string_typed_numbers(tmp_path):
    """Hand-edited configs often quote numbers. They must be coerced back to
    int/float so later arithmetic and comparisons don't blow up."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"max_tokens": "8192", "temperature": "0.3"}))
    cfg = AgentConfig.load(str(p))
    assert cfg.max_tokens == 8192
    assert isinstance(cfg.max_tokens, int)
    assert cfg.temperature == 0.3
    assert isinstance(cfg.temperature, float)


def test_save_is_atomic(tmp_path):
    """save() must write via temp + rename so a crash can never leave a
    half-written config.json behind."""
    p = tmp_path / "c.json"
    cfg = AgentConfig(base_url="http://x")
    cfg.save(str(p))
    assert os.path.exists(str(p))
    assert not os.path.exists(str(p) + ".tmp")


def test_migrate_keeps_explicitly_small_budget(tmp_path):
    """A deliberately small context budget (e.g. a local 4K model) must survive
    a restart — migration should only rewrite the legacy 4096 default."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"max_tokens": 2048, "max_context_tokens": 2048}))
    cfg = AgentConfig.load(str(p))
    assert cfg.max_tokens == 2048
    assert cfg.max_context_tokens == 2048
