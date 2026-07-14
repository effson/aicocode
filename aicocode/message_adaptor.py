"""
    把 provider 无关的内部消息转化成各家 API 的请求格式。
    对话层 Conversation只管理消息、不负责线上请求格式。
"""

from __future__ import annotations

import json
from typing import Any

from aicocode.conversation import Message

def construct_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in messages:
        if message.tool_uses or message.thinking_blocks:
            content: list[dict[str, Any]] = []
            for tb in message.thinking_blocks:
                content.append({
                    "type": "thinking",
                    "thinking": tb.thinking,
                    "signature": tb.signature,
                })
            if message.content:
                content.append({"type": "text", "text": message.content})
            for tu in message.tool_uses:
                content.append({
                    "type": "tool_use",
                    "id": tu.tool_use_id,
                    "name": tu.tool_name,
                    "input": tu.arguments,
                })
            if not content:
                content.append({"type": "text", "text": ""})
            result.append({"role": "assistant", "content": content})
        elif message.tool_results:
            content = []
            for tr in message.tool_results:
                content.append({
                    "type": "tool_result",
                    "tool_use_id": tr.tool_use_id,
                    "content": tr.content,
                    "is_error": tr.is_error,
                })
            result.append({"role": "user", "content": content})
        else:
            # 合并连续的 user 纯文本消息（system-reminder 或普通 user 文本）。
            if (
                message.role == "user"
                and result
                and result[-1]["role"] == "user"
                and isinstance(result[-1]["content"], str)
            ):
                result[-1]["content"] = result[-1]["content"] + "\n" + message.content
            else:
                result.append({"role": message.role, "content": message.content})
    return result


def construct_openai_input(messages: list[Message]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in messages:
        if message.tool_uses:
            # Responses API: thinking blocks 作为 reasoning item 回传。
            for tb in (message.thinking_blocks or []):
                result.append({
                    "type": "reasoning",
                    "id": tb.signature,
                    "summary": [{"type": "summary_text", "text": tb.thinking}],
                })
            if message.content:
                result.append({"role": "assistant", "content": message.content})
            for tu in message.tool_uses:
                result.append({
                    "type": "function_call",
                    "name": tu.tool_name,
                    "call_id": tu.tool_use_id,
                    "arguments": json.dumps(tu.arguments),
                })
        elif message.tool_results:
            for tr in message.tool_results:
                result.append({
                    "type": "function_call_output",
                    "call_id": tr.tool_use_id,
                    "output": tr.content,
                })
        else:
            # 非 tool 的 assistant 消息也回传 reasoning。
            for tb in (message.thinking_blocks or []):
                result.append({
                    "type": "reasoning",
                    "id": tb.signature,
                    "summary": [{"type": "summary_text", "text": tb.thinking}],
                })
            result.append({"role": message.role, "content": message.content})
    return result


def construct_chat_completion_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """
        OpenAI Chat Completions 格式。
        - 用户消息：{"role": "user", "content": "..."}
        - 助手文本+工具调用：{"role": "assistant", "content": "...", "tool_calls": [...]}
        - 工具结果：{"role": "tool", "tool_call_id": "...", "content": "..."}
        - thinking 块作为 reasoning_content 回传（DeepSeek/小米等 provider 要求）。
    """
    result: list[dict[str, Any]] = []
    for message in messages:
        reasoning = "".join(tb.thinking for tb in message.thinking_blocks) if message.thinking_blocks else ""

        if message.tool_uses:
            tool_calls = []
            for tu in message.tool_uses:
                tool_calls.append({
                    "id": tu.tool_use_id,
                    "type": "function",
                    "function": {
                        "name": tu.tool_name,
                        "arguments": json.dumps(tu.arguments),
                    },
                })
            msg: dict[str, Any] = {
                "role": "assistant",
                "content": message.content or None,
                "tool_calls": tool_calls,
            }
            if reasoning:
                msg["reasoning_content"] = reasoning
            result.append(msg)
        elif message.tool_results:
            for tr in message.tool_results:
                result.append({
                    "role": "tool",
                    "tool_call_id": tr.tool_use_id,
                    "content": tr.content,
                })
        else:
            msg = {"role": message.role, "content": message.content}
            if reasoning:
                msg["reasoning_content"] = reasoning
            result.append(msg)
    return result


def construct_messages(messages: list[Message], protocol: str = "anthropic") -> list[dict[str, Any]]:
    if protocol == "openai":
        return construct_openai_input(messages)
    if protocol == "openai-compat":
        return construct_chat_completion_messages(messages)
    return construct_anthropic_messages(messages)