"""base-agent: a streamlined local agent with a PyQt6 UI."""
from __future__ import annotations

import os as _os

# HuggingFace 离线模式必须在**任何** agent 子模块（fastembed/transformers/
# docling/marker/chonkie）触发 huggingface_hub 导入之前设置——
# huggingface_hub.constants 在 import 时一次性读取 HF_HUB_OFFLINE，晚了就
# 固化为 False（#131：docling convert 期间向 huggingface.co 检查模型更新，
# 网络不通时 connect() 挂死整个同步 20+ 分钟）。所有模型均已本地缓存；
# 未缓存的组件会立即报错而非挂死。setdefault：用户显式设置的值优先。
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from .agent import Agent, AgentCallbacks, SYSTEM_PROMPT
from .config import AgentConfig
from .llm_client import LLMClient, StreamEvent, LLMError
from .memory import LongTermMemory, WorkMemory
from .skills import Skill, SkillManager
from .tools import Tool, ToolRegistry, ToolResult

__all__ = [
    "Agent", "AgentCallbacks", "SYSTEM_PROMPT",
    "AgentConfig",
    "LLMClient", "StreamEvent", "LLMError",
    "LongTermMemory", "WorkMemory",
    "Skill", "SkillManager",
    "Tool", "ToolRegistry", "ToolResult",
]
