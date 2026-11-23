"""知识图谱抽取：辅助模型优先 + 主模型 isolated 兜底测试。"""
import os
import sys

from PyQt6.QtWidgets import QApplication

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from agent.config import AgentConfig
from agent.ui.main_window import BuildKGWorker


def test_extraction_client_aux_disabled_falls_back_to_main():
    """辅助模型未启用 → 返回主模型 client（共享，owns=False）。"""
    app = QApplication.instance() or QApplication([])
    cfg = AgentConfig()
    cfg.aux_llm_enabled = False
    main_client = object()
    worker = BuildKGWorker("ws", "kb", config=cfg, fallback_main_client=main_client)
    client, label, owns = worker._extraction_client()
    assert client is main_client
    assert "主模型" in label
    assert owns is False


def test_extraction_client_no_fallback_raises():
    """辅助模型未启用且无主模型兜底 → 明确报错。"""
    app = QApplication.instance() or QApplication([])
    cfg = AgentConfig()
    cfg.aux_llm_enabled = False
    worker = BuildKGWorker("ws", "kb", config=cfg)
    try:
        worker._extraction_client()
    except RuntimeError:
        pass
    else:
        raise AssertionError("应抛出 RuntimeError")


def test_worker_has_fallback_notice_signal():
    """worker 具备回退提醒信号（辅助模型构建失败时通知主线程弹框）。"""
    worker = BuildKGWorker("ws", "kb", config=AgentConfig())
    assert hasattr(worker, "kg_fallback_notice")
