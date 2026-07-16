from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from pydantic import ValidationError
from aicocode.llm_client import LLMClient

from aicocode.agent_event import (
    AgentEvent,
    StreamText,
    ThinkingText,
    RetryEvent,
    ToolUseEvent,
    ToolResultEvent,
    TurnComplete,
    LoopComplete,
    UsageEvent,
    ErrorEvent,
    StreamCollector,
    ThinkingBlock,
    StreamingExecutor,
    _ToolExecResult,
    LLMResponse,
)

from aicocode.prompt import build_environment_context, build_system_prompt

from aicocode.base import (
    TextDelta,
    StreamEvent,
    ToolCallStart,
    ThinkingDelta,
    ToolCallDelta,
    ThinkingComplete,
    ToolCallComplete,
    StreamEnd,
)

from aicocode.tools.tool_base import (
    MAX_OUTPUT_CHARS,
    ToolResult,
)

from aicocode.conversation import Conversation, ToolUseBlock, ToolResultBlock

from aicocode.tools import ToolRegistry

from .config_validator import Protocols

log = logging.getLogger(__name__)


"""
    AgentLoop
"""
class Agent:
    def __init__(
        self,
        client: LLMClient,
        registry: ToolRegistry,
        protocol: Protocols,
        work_dir: str = ".",
        max_iterations: int = 0,
        context_window: int = 200_000,
        instructions_content: str = "",
    ) -> None:
        self.client: LLMClient = client
        self.registry = registry
        self.protocol = protocol
        self.work_dir = work_dir
        self.max_iterations = max_iterations
        self.context_window = context_window
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.agent_id: str = uuid.uuid4().hex[:12]
        self.parent_id: str | None = None

    async def _execute_single_tool_direct(
        self, tc: ToolCallComplete
    ) -> _ToolExecResult:
        tool = self.registry.get_tool(tc.tool_name)
        start = time.monotonic()

        if tool is None:
            return _ToolExecResult(
                tool_id=tc.tool_id,
                tool_name=tc.tool_name,
                result=ToolResult(output=f"Error: unknown tool '{tc.tool_name}'", is_error=True),
                elapsed=time.monotonic() - start,
                is_unknown=True,
            )

        if not self.registry.tool_is_enabled(tc.tool_name):
            return _ToolExecResult(
                tool_id=tc.tool_id,
                tool_name=tc.tool_name,
                result=ToolResult(output=f"Error: tool '{tc.tool_name}' is disabled", is_error=True),
                elapsed=time.monotonic() - start,
                is_unknown=False,
            )

        try:
            params = tool.params_model.model_validate(tc.arguments)
            result = await tool.execute(params)
        except ValidationError as e:
            result = ToolResult(output=f"Parameter validation error: {e}", is_error=True)
        except Exception as e:
            result = ToolResult(output=f"Tool execution error: {e}", is_error=True)

        return _ToolExecResult(
            tool_id=tc.tool_id,
            tool_name=tc.tool_name,
            result=result,
            elapsed=time.monotonic() - start,
            is_unknown=False,
        )

    async def run(self, conversation: Conversation) -> AsyncIterator[AgentEvent]:
        self._current_conversation = conversation
        env_context = build_environment_context(self.work_dir)
        conversation.inject_environment_context(env_context)

        iteration = 0
        consecutive_unknown = 0
        max_tokens_escalated = False
        output_recoveries = 0

        while True:
            iteration += 1
            
            system = build_system_prompt()
            
            deferred_names = self.registry.get_deferred_tool_names()
            if deferred_names:
                conversation.add_system_reminder(
                    "The following deferred tools are available via ToolSearch. "
                    "Their schemas are NOT loaded - use ToolSearch with "
                    'query "select:<name>[,<name>...]" to load tool schemas before calling them:\n'
                    + "\n".join(deferred_names)
                )

            tools = self.registry.get_all_schemas(self.protocol)

            collector = StreamCollector()
            executor = StreamingExecutor()
            llm_stream = self.client.stream(conversation, system=system, tools=tools)
            async for event in collector.consume(llm_stream):
                if isinstance(event, ToolUseEvent):
                    tc = collector.response.tool_calls[-1]
                    # 需要交互式权限确认的工具延迟到流结束后顺序执行
                    tool = self.registry.get_tool(tc.tool_name)
                    executor.submit(self._execute_single_tool_direct(tc))
                yield event

            response = collector.response
            self.total_input_tokens += response.input_tokens
            self.total_output_tokens += response.output_tokens
            yield UsageEvent(
                input_tokens=self.total_input_tokens,
                output_tokens=self.total_output_tokens,
            )

            conv_thinking = [
                ThinkingBlock(thinking=tb.thinking, signature=tb.signature)
                for tb in response.thinking_blocks
            ]

            if not response.tool_calls:
                conversation.add_assistant_message(
                    response.text, thinking_blocks=conv_thinking
                )

                yield LoopComplete(total_turns=iteration)
                break

            tool_uses = [
                ToolUseBlock(
                    tool_use_id=tc.tool_id,
                    tool_name=tc.tool_name,
                    arguments=tc.arguments,
                )
                for tc in response.tool_calls
            ]

            conversation.add_assistant_message(
                response.text, tool_uses, thinking_blocks=conv_thinking
            )

            # 收集流式执行器中已提交的工具结果（工具在 LLM 流式输出期间已开始执行）
            tool_results: list[ToolResultBlock] = []
            streaming_results: list[_ToolExecResult] = await executor.collect_results()

            for tool_exec_res in streaming_results:
                if tool_exec_res.is_unknown:
                    consecutive_unknown += 1
                else:
                    consecutive_unknown = 0

                tool_results.append(
                    ToolResultBlock(
                        tool_use_id=tool_exec_res.tool_id,
                        content=tool_exec_res.result.output,
                        is_error=tool_exec_res.result.is_error,
                    )
                )
                yield ToolResultEvent(
                    tool_id=tool_exec_res.tool_id,
                    tool_name=tool_exec_res.tool_name,
                    output=tool_exec_res.result.output,
                    is_error=tool_exec_res.result.is_error,
                    elapsed=tool_exec_res.elapsed,
                )

            if consecutive_unknown >= 3:
                yield ErrorEvent(
                    message="Agent terminated: too many consecutive unknown tool calls"
                )
                break

            conversation.add_tool_results_message(tool_results)
            yield TurnComplete(turn=iteration)