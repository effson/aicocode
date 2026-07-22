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
from aicocode.Permissions import (
    PermissionValidator,
    PermissionRes,
    PermissionMode
)

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
    PermissionResponse,
    PermissionRequest,
    AskUserRequest,
    CompactNotification
)

from aicocode.prompt import build_environment_context, build_system_prompt, build_plan_mode_reminder

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
from aicocode.context import (
    apply_tool_result_budget,
    ensure_session_dir,
    ToolResReplacementState,
    create_replacement_state,
    append_replacement_records,
    CompactBreaker,
    auto_compact,
    RecoveryState,
    CompactEvent,
)


from aicocode.tools import ToolRegistry
from aicocode.tools.ask_user import AskUserTool

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
        permission_validator: PermissionValidator | None = None,
        context_window: int = 200_000,
        instructions_content: str = "",
    ) -> None:
        self.client: LLMClient = client
        self.registry = registry
        self.protocol = protocol
        self.work_dir = work_dir
        self.max_iterations = max_iterations
        self._loop_count = 0
        self.permission_validator = permission_validator
        self.permission_mode: PermissionMode = (
            permission_validator.permission_mode if permission_validator else PermissionMode.DEFAULT
        )
        self.context_window = context_window
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.agent_id: str = uuid.uuid4().hex[:12]
        self.parent_id: str | None = None
        self.session_dir = ensure_session_dir(work_dir)
        self.replacement_state: ToolResReplacementState = create_replacement_state()
        self.compact_breaker = CompactBreaker()
        self.recovery_state: RecoveryState = RecoveryState()
        self.session_id: str = ""

    @property
    def _transcript_path(self) -> str:
        if self.session_id:
            return str(Path(self.work_dir) / ".aicocode" / "sessions" / f"{self.session_id}.jsonl")
        return ""

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

        if self.permission_validator:
            permission_res: PermissionRes = self.permission_validator.check(tool, tc.arguments)
            if permission_res.permission == "deny":
                return _ToolExecResult(
                    tool_id=tc.tool_id,
                    tool_name=tc.tool_name,
                    result=ToolResult(output=f"Permission denied: {permission_res.reason}", is_error=True),
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

        self._snapshot_for_recovery(tc, result)

        return _ToolExecResult(
            tool_id=tc.tool_id,
            tool_name=tc.tool_name,
            result=result,
            elapsed=time.monotonic() - start,
            is_unknown=False,
        )
    
    def _build_permission_request_description(self, tc: ToolCallComplete) -> str:
        """为 HITL 权限确认生成人类可读的操作描述。"""
        return PermissionValidator.describe_tool_action(tc.tool_name, tc.arguments)
    
    async def _execute_tool(
        self, tc: ToolCallComplete
    ) -> AsyncIterator[tuple[ToolResult, float, bool]]:
        tool = self.registry.get_tool(tc.tool_name)
        start = time.monotonic()
        is_unknown = False

        if tool is None:
            result = ToolResult(
                output=f"Error: unknown tool '{tc.tool_name}'", is_error=True
            )
            is_unknown = True
            elapsed = time.monotonic() - start
            yield result, elapsed, is_unknown
            return

        if not self.registry.tool_is_enabled(tc.tool_name):
            result = ToolResult(
                output=f"Error: tool '{tc.tool_name}' is disabled in current mode",
                is_error=True,
            )
            elapsed = time.monotonic() - start
            yield result, elapsed, is_unknown
            return

        if self.permission_validator:
            permission_res: PermissionRes = self.permission_validator.check(tool, tc.arguments)
            if permission_res.permission == "deny":
                result = ToolResult(
                    output=f"Permission denied: {permission_res.reason}",
                    is_error=True,
                )
                elapsed = time.monotonic() - start
                yield result, elapsed, is_unknown
                return

            if permission_res.permission == "query":
                loop = asyncio.get_running_loop()
                future: asyncio.Future[PermissionResponse] = loop.create_future()
                permission_req_desc = self._build_permission_request_description(tc)
                yield PermissionRequest(
                    tool_name=tc.tool_name,
                    description=permission_req_desc,
                    future=future,
                )

                response = await future

                if response == PermissionResponse.DENY:
                    result = ToolResult(
                        output="Permission denied: 用户拒绝了此操作",
                        is_error=True,
                    )
                    elapsed = time.monotonic() - start
                    yield result, elapsed, is_unknown
                    return

                if response == PermissionResponse.ALLOW_ALWAYS:
                    from aicocode.Permissions.rules import Rule, extract_content
                    content = extract_content(tc.tool_name, tc.arguments)
                    pattern = f"{content[:60]}*" if len(content) > 60 else f"{content}*"
                    # 持久化规则写入本地文件
                    rule = Rule(tool_name=tc.tool_name, pattern=pattern, permission="allow")
                    self.permission_validator.rule_engine.append_local_rule(rule)
                    # 同时加入会话级放行集合，本轮立即生效无需磁盘读取
                    self.permission_validator.add_session_allow(tc.tool_name, content)

        # AskUserQuestion：交互由 Agent 事件流接管。yield AskUserRequest 让 UI
        # 弹窗，await future 拿到回答后用 format_result 汇总，不走 tool.execute。
        if isinstance(tool, AskUserTool):
            try:
                au_params = tool.params_model.model_validate(tc.arguments)
            except ValidationError as e:
                result = ToolResult(
                    output=f"Parameter validation error: {e}", is_error=True
                )
                elapsed = time.monotonic() - start
                yield result, elapsed, is_unknown
                return
            loop = asyncio.get_running_loop()
            future: asyncio.Future[dict[str, str]] = loop.create_future()
            yield AskUserRequest(
                questions=[q.model_dump() for q in au_params.questions],
                future=future,
            )
            try:
                answers = await asyncio.wait_for(future, timeout=300)
            except asyncio.TimeoutError:
                result = ToolResult(
                    output="User did not respond within 5 minutes", is_error=True
                )
            else:
                result = AskUserTool.format_result(au_params, answers)
            elapsed = time.monotonic() - start
            yield result, elapsed, is_unknown
            return

        try:
            params = tool.params_model.model_validate(tc.arguments)
            result = await tool.execute(params)
        except ValidationError as e:
            result = ToolResult(
                output=f"Parameter validation error: {e}", is_error=True
            )
        except Exception as e:
            result = ToolResult(
                output=f"Tool execution error: {e}", is_error=True
            )

        self._snapshot_for_recovery(tc, result)

        elapsed = time.monotonic() - start
        yield result, elapsed, is_unknown
    
    def _snapshot_for_recovery(
        self, tc: ToolCallComplete, result: ToolResult
    ) -> None:
        """捕获 ReadFile 刚交给模型的内容，以便 Layer 2 压缩对话后
        auto_compact 能重新附加这些数据。每次 ReadFile 多一次磁盘读取，
        比从 tool 输出中反向解析行号要划算。
        """
        if result.is_error or tc.tool_name != "ReadFile":
            return
        path = tc.arguments.get("file_path") if isinstance(tc.arguments, dict) else None
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        except OSError:
            return
        logging.info(f"Snapshot for: {path}")
        self.recovery_state.record_file_read(path, content)
    
    def set_permission_mode(self, permission_mode: PermissionMode) -> None:
        self.permission_mode = permission_mode
        if self.permission_validator:
            self.permission_validator.permission_mode = permission_mode

    @property
    def in_plan_mode(self) -> bool:
        return self.permission_mode == PermissionMode.PLAN

    _plan_path_cache: Path | None = None

    def _get_plan_path(self) -> Path:
        if self._plan_path_cache is not None:
            return self._plan_path_cache
        import random
        import datetime
        _ADJECTIVES = ["bold", "bright", "calm", "cool", "deep", "fair", "fast", "fine",
                       "glad", "keen", "kind", "lean", "mild", "neat", "pure", "safe",
                       "slim", "soft", "tall", "warm", "wise", "grand", "swift", "vivid"]
        _NOUNS = ["sketch", "draft", "spark", "bloom", "trail", "ridge", "creek", "grove",
                  "cliff", "cloud", "field", "forge", "frost", "haven", "pearl", "stone",
                  "storm", "river", "tower", "delta", "flame", "orbit", "pulse", "shore"]
        plans_dir = Path(self.work_dir) / ".aicocode" / "plans"
        plans_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%m%d-%H%M")
        slug = f"{random.choice(_ADJECTIVES)}-{random.choice(_NOUNS)}-{ts}"
        self._plan_path_cache = plans_dir / f"{slug}.md"
        return self._plan_path_cache

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

            if self.in_plan_mode:
                plan_file_path = str(self._get_plan_path())
                if self.permission_validator:
                    self.permission_validator.plan_file_path = plan_file_path
                plan_file_exists = self._get_plan_path().exists()
                plan_reminder = build_plan_mode_reminder(
                    plan_file_path, plan_file_exists, iteration
                )
                conversation.add_system_reminder(plan_reminder)

            deferred_names = self.registry.get_deferred_tool_names()
            if deferred_names:
                conversation.add_system_reminder(
                    "The following deferred tools are available via ToolSearch. "
                    "Their schemas are NOT loaded - use ToolSearch with "
                    'query "select:<name>[,<name>...]" to load tool schemas before calling them:\n'
                    + "\n".join(deferred_names)
                )

            tools = self.registry.get_all_schemas(self.protocol)

            new_records = apply_tool_result_budget(
                conversation, self.session_dir, self.replacement_state
            )
            if new_records:
                append_replacement_records(self.session_dir, new_records)

            compact_result = await auto_compact(
                conversation=conversation,
                client=self.client,
                context_window=self.context_window,
                session_dir=self.session_dir,
                protocol=self.protocol,
                breaker=self.compact_breaker,
                recovery=self.recovery_state,
                tool_schemas=self.registry.get_all_schemas(self.protocol),
                transcript_path=self._transcript_path,
            )

            if isinstance(compact_result, CompactEvent):
                yield CompactNotification(
                    before_tokens=compact_result.before_tokens,
                    message=f"上下文已压缩（压缩前 [{compact_result.before_tokens}] tokens）",
                    boundary=compact_result.boundary,
                )
                conversation.inject_environment_context(env_context)
                apply_tool_result_budget(
                    conversation, self.session_dir, self.replacement_state
                )

            elif isinstance(compact_result, str):
                yield ErrorEvent(message=compact_result)

            collector = StreamCollector()
            executor = StreamingExecutor()
            deferred_tool_calls: list[ToolCallComplete] = []
            llm_stream = self.client.stream(conversation, system=system, tools=tools)
            async for event in collector.consume(llm_stream):
                if isinstance(event, ToolUseEvent):
                    tc = collector.response.tool_calls[-1]
                    # 需要交互式权限确认的工具延迟到流结束后顺序执行
                    tool = self.registry.get_tool(tc.tool_name)
                    needs_query = False
                    if tool and self.permission_validator:
                        permission_res: PermissionRes = self.permission_validator.check(tool, tc.arguments)
                        needs_query = permission_res.permission == "query"
                    # 需要交互式确认（权限 query）或需要 UI 弹窗（AskUserQuestion）
                    # 的工具，延迟到流结束后顺序执行，以便 yield 交互事件给 UI。
                    if needs_query or isinstance(tool, AskUserTool):
                        deferred_tool_calls.append(tc)
                    else:
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

                self._loop_count += 1

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

            conversation.record_usage_anchor(
                response.input_tokens,
                response.output_tokens,
                response.cache_read,
                response.cache_creation,
            )

            # 收集流式执行器中已提交的工具结果（工具在 LLM 流式输出期间已开始执行）
            tool_results: list[ToolResultBlock] = []
            streaming_results: list[_ToolExecResult] = await executor.collect_results()
            args_by_id = {t.tool_id: t.arguments for t in response.tool_calls}

            for tool_exec_res in streaming_results:
                if tool_exec_res.is_unknown:
                    consecutive_unknown += 1
                else:
                    consecutive_unknown = 0

                content = self._maybe_persist_or_truncate(
                    tool_exec_res.tool_id,
                    tool_exec_res.result.output,
                    tool_exec_res.tool_name,
                    args_by_id.get(tool_exec_res.tool_id, {}),
                )

                tool_results.append(
                    ToolResultBlock(
                        tool_use_id=tool_exec_res.tool_id,
                        content=content,
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

            # 需要交互式权限确认的工具，在流结束后顺序执行
            for tc in deferred_tool_calls:
                result: ToolResult | None = None
                elapsed = 0.0
                is_unknown = False

                async for item in self._execute_tool(tc):
                    if isinstance(item, (PermissionRequest, AskUserRequest)):
                        yield item
                    else:
                        result, elapsed, is_unknown = item

                if result is None:
                    result = ToolResult(output="Error: no result from tool", is_error=True)

                if is_unknown:
                    consecutive_unknown += 1
                else:
                    consecutive_unknown = 0

                content = self._maybe_persist_or_truncate(
                    tc.tool_id, result.output, tc.tool_name, tc.arguments
                )

                tool_results.append(
                    ToolResultBlock(
                        tool_use_id=tc.tool_id,
                        content=content,
                        is_error=result.is_error,
                    )
                )
                yield ToolResultEvent(
                    tool_id=tc.tool_id,
                    tool_name=tc.tool_name,
                    output=result.output,
                    is_error=result.is_error,
                    elapsed=elapsed,
                )

            if consecutive_unknown >= 3:
                yield ErrorEvent(
                    message="Agent terminated: too many consecutive unknown tool calls"
                )
                break

            exit_plan_called = any(
                tc.tool_name == "ExitPlanMode" for tc in response.tool_calls
            )

            conversation.add_tool_results_message(tool_results)

            if exit_plan_called:
                yield TurnComplete(turn=iteration)
                yield LoopComplete(total_turns=iteration)
                break

            yield TurnComplete(turn=iteration)

    def _maybe_persist_or_truncate(
        self,
        tool_use_id: str,
        text: str,
        tool_name: str = "",
        arguments: dict | None = None,
    ) -> str:
        from aicocode.context.manager import (
            SINGLE_RESULT_CHAR_LIMIT,
            is_spill_readback_call,
            make_persisted_tool_result_preview,
            persist_tool_result,
        )

        # 回读豁免：模型用 ReadFile 读取 session_dir 下的落盘文件时，结果本身
        # 就是已落盘内容的回读。再次落盘/截断会陷入「落盘→回读→再落盘」循环，
        # 且让模型刚读到的完整内容瞬间又被截掉。这里原样返回，把这条结果交给
        # Layer 2（auto_compact）按总量清理。
        if is_spill_readback_call(tool_name, arguments, self.session_dir):
            return text

        if len(text) > SINGLE_RESULT_CHAR_LIMIT:
            fp = persist_tool_result(tool_use_id, text, self.session_dir)
            return make_persisted_tool_result_preview(text, fp)
        # if len(text) > MAX_OUTPUT_CHARS:
        #     return text[:MAX_OUTPUT_CHARS] + "\n… (output truncated)"
        return text