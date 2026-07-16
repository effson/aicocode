"""会话层：进程内单会话多轮历史。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ToolUseBlock:
    tool_use_id: str
    tool_name: str
    arguments: dict[str, Any]


@dataclass
class ToolResultBlock:
    tool_use_id: str
    content: str
    is_error: bool = False

@dataclass
class ThinkingBlock:
    thinking: str
    signature: str

@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str
    tool_uses: list[ToolUseBlock] = field(default_factory=list)
    tool_results: list[ToolResultBlock] = field(default_factory=list)
    thinking_blocks: list[ThinkingBlock] = field(default_factory=list)

@dataclass
class Conversation:
    """
        维护完整对话历史（user / assistant 交替）。
    """
    messages: list[Message] = field(default_factory=list)
    env_injected: bool = field(default=False, init=False)

    def add_user_message(self, content: str) -> None:
        """追加用户消息。"""
        self.messages.append(Message(role="user", content=content))

    def add_assistant_message(
            self,
            content: str,
            tool_uses: list[ToolUseBlock] | None = None,
            thinking_blocks: list[ThinkingBlock] | None = None,
    ) -> None:
        """追加助手消息。"""
        self.messages.append(
            Message(
                role="assistant",
                content=content,
                tool_uses=tool_uses or [],
                thinking_blocks=thinking_blocks or [],
            )
        )

    def add_system_reminder(self, content: str) -> None:
        self.messages.append(
            Message(
                role="user",
                content=f"<system-reminder>\n{content}\n</system-reminder>",
            )
        )

    def add_tool_results_message(self, tool_results: list[ToolResultBlock]) -> None:
        self.messages.append(
            Message(
                role="user",
                content="",
                tool_results=tool_results
            )
        )

    def inject_environment_context(self, context: str) -> None:
        if not self.env_injected:
            self.messages.insert(0, Message(role="user", content=context))
            self.env_injected = True

    def fetch_messages(self) -> list[Message]:
        return list(self.messages)

    def __len__(self) -> int:
        return len(self.messages)
