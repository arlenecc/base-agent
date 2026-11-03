"""ask_user tool: pause and ask the human a question via callbacks."""
from __future__ import annotations

from ..config import AgentConfig
from .base import Tool, ToolRegistry, ToolResult


class AskUserTool(Tool):
    name = "ask_user"
    description = (
        "Ask the user ONE specific blocking question via a popup dialog — a "
        "real interruption for the user, so use it sparingly. Only while a "
        "task is actually in progress and a piece of information you cannot "
        "obtain yourself is missing (a path/param, a choice, or a "
        "confirmation). NEVER for greetings or small talk (just reply "
        "naturally), never to ask what the user wants to do, never for "
        "anything inferable from context or prior tool results, and never to "
        "request permission to continue."
    )
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Question to ask the user."},
        },
        "required": ["prompt"],
    }

    config: AgentConfig
    registry: ToolRegistry

    def bind(self, config: AgentConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry

    def run(self, prompt: str) -> ToolResult:
        # callbacks 可能缺失（headless / 测试 / 未 bind 的 registry），裸访问会
        # AttributeError 被 registry 兜底成一句没有上下文的错误。
        cb = getattr(self.registry, "callbacks", None)
        ask = getattr(cb, "ask_user", None)
        if not callable(ask):
            return ToolResult(
                False,
                error="ask_user is unavailable in this context (no UI callbacks bound).",
            )
        answer = ask(prompt)
        if answer is None:
            return ToolResult(False, error="User dismissed the question.")
        return ToolResult(True, output=str(answer))
