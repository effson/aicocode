from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from aicocode.tools.tool_base import Tool, ToolResult

if TYPE_CHECKING:
    from aicocode.agent import Agent
    from aicocode.agents.loader import AgentLoader
    from aicocode.agents.task_manager import TaskManager
    from aicocode.agents.trace import TraceManager
    from aicocode.llm_client import LLMClient


log = logging.getLogger(__name__)

class AgentToolParams(BaseModel):
    prompt: str
    description: str
    subagent_type: str | None = None
    model: str | None = None
    run_in_background: bool = False
    name: str | None = None
    isolation: str | None = None

PERMISSION_MODE_MAP = {
    "default": "DEFAULT",
    "acceptEdits": "ACCEPT_EDITS",
    "bypassPermissions": "BYPASS",
}

GENERAL_PURPOSE_AGENT_TYPE = "general-purpose"
FORK_QUERY_SOURCE = "agent:builtin:fork"

class AgentTool(Tool):
    name = "Agent"
    description = (
        "Launch a sub-agent to handle a task in an isolated context. "
        "Use subagent_type to select a predefined agent type (e.g. Explore, Plan, general-purpose), "
        "or leave it empty to fork the current conversation. "
        # "Use team_name to spawn a teammate in an existing team."
    )
    params_model = AgentToolParams
    category = "command"
    is_concurrency_safe = False

    def __init__(
        self,
        agent_loader: AgentLoader,
        task_manager: TaskManager,
        trace_manager: TraceManager,
        parent_agent: Agent,
        enable_fork: bool = False,
        provider_config: Any = None,
        worktree_manager: Any = None,
    ) -> None:
        self._agent_loader = agent_loader
        self._task_manager = task_manager
        self._trace_manager = trace_manager
        self._parent_agent = parent_agent
        self._enable_fork = enable_fork
        self._provider_config = provider_config
        self.query_source: str = ""
        self._worktree_manager = worktree_manager

    async def execute(self, params: BaseModel) -> ToolResult:
        p: AgentToolParams = params
        
        isolation = ""
        if p.subagent_type:
            defn = self._agent_loader.get(p.subagent_type)
            if defn and defn.isolation:
                isolation = defn.isolation

        if isolation == "worktree":
            return await self._execute_with_worktree(p)

        from aicocode.agents.fork import ForkError, build_forked_messages
        from aicocode.agents.parser import AgentDef
        from aicocode.agents.tool_filter import clone_registry_for_fork, resolve_agent_tools
        from aicocode.agent import Agent as AgentClass
        from aicocode.conversation import Conversation
        from aicocode.Permissions import (
            DangerousCommandDetector,
            PathSandbox,
            PermissionValidator,
            PermissionMode,
            RuleEngine,
        )

        agent_def: AgentDef | None = None
        conversation: Conversation

        subagent_type = p.subagent_type
        if not subagent_type and not self._enable_fork:
            subagent_type = GENERAL_PURPOSE_AGENT_TYPE

        if subagent_type:
            agent_def = self._agent_loader.get(subagent_type)
            if agent_def is None:
                return ToolResult(
                    output=f"Unknown agent type: '{subagent_type}'. "
                    f"Available types: {', '.join(t for t, _ in self._agent_loader.list_agents())}",
                    is_error=True,
                )
            conversation = Conversation()
        else:
            if not self._enable_fork:
                return ToolResult(
                    output="Fork mode is not enabled. "
                    "Set 'enable_fork: true' in config.yaml to use fork, "
                    "or specify a subagent_type parameter.",
                    is_error=True,
                )

            # fork 子 Agent 不允许再次 fork，防止无限嵌套
            if self.query_source == FORK_QUERY_SOURCE:
                return ToolResult(
                    output="Error: cannot fork from a forked agent. "
                            "Use subagent_type to spawn a definition-based agent instead.",
                    is_error=True,
                )

            try:
                parent_conv = getattr(self._parent_agent, '_current_conversation', None)
                if parent_conv is None:
                    return ToolResult(
                        output="Cannot fork: no active conversation in parent agent.",
                        is_error=True,
                    )
                conversation = build_forked_messages(parent_conv, p.prompt)
            except ForkError as e:
                return ToolResult(output=str(e), is_error=True)

            agent_def = AgentDef(
                agent_type="fork",
                when_to_use="Forked from parent agent",
                system_prompt="",
                disallowed_tools=[],
                model="inherit",
                max_turns=self._parent_agent.max_iterations,
                permission_mode="bypassPermissions",
                source="builtin",
            )

        llm_client = self._select_llm(p, agent_def)

        is_fork = not subagent_type
        is_background = p.run_in_background or agent_def.background
        if is_fork:
            is_background = True

        # 构建子 agent 工具注册表
        _base_registry = getattr(self._parent_agent, '_full_registry', None) or self._parent_agent.registry

        if is_fork:
            filtered_registry = clone_registry_for_fork(_base_registry)
        else:
            filtered_registry = resolve_agent_tools(
                _base_registry, agent_def, is_background
            )

        permission = agent_def.permission_mode
        permission_enum = getattr(
            PermissionMode,
            PERMISSION_MODE_MAP.get(permission, "DEFAULT"),
            PermissionMode.DEFAULT,
        )

        permission_validator = PermissionValidator(
            danger_command_detector=DangerousCommandDetector(),
            path_sandbox=PathSandbox(self._parent_agent.work_dir),
            rule_engine=RuleEngine(),
            permission_mode=permission_enum,
        )

        sub_agent = AgentClass(
            client=llm_client,
            registry=filtered_registry,
            protocol=self._parent_agent.protocol,
            work_dir=self._parent_agent.work_dir,
            max_iterations=agent_def.max_turns,
            permission_validator=permission_validator,
            context_window=self._parent_agent.context_window,
            instructions_content=agent_def.system_prompt,
            hook_engine=self._parent_agent.hook_engine,
        )
        sub_agent.parent_id = self._parent_agent.agent_id
        sub_agent.trace_id = self._parent_agent.trace_id or self._parent_agent.agent_id

        if p.subagent_type is None:
            from aicocode.context import clone_replacement_state
            sub_agent.replacement_state = clone_replacement_state(
                self._parent_agent.replacement_state
            )

        trace_node = self._trace_manager.create(
            agent_type=agent_def.agent_type,
            parent_id=self._parent_agent.agent_id,
            trace_id=sub_agent.trace_id,
        )
        sub_agent.agent_id = trace_node.agent_id

        agent_name = p.name or p.subagent_type or f"agent-{trace_node.agent_id}"

        if is_background:
            if is_fork:
                sub_agent._fork_conversation = conversation
            task_id = self._task_manager.launch(
                agent=sub_agent,
                task="" if is_fork else p.prompt,
                name=agent_name,
                fork_conversation=conversation if is_fork else None,
            )
            return ToolResult(
                output=f"Sub-agent launched in background.\n"
                f"Task ID: {task_id}\n"
                f"Agent: {agent_name}\n"
                f"Type: {agent_def.agent_type}\n"
                f"The system will notify automatically when it completes.\n"
                f"Do NOT wait, sleep, or poll. Report the task ID to the user and move on.",
            )

        try:
            if is_fork:
                result_text = await sub_agent.run_to_completion("", conversation)
            else:
                result_text = await sub_agent.run_to_completion(p.prompt)
        except Exception as e:
            self._trace_manager.complete(trace_node.agent_id, "failed")
            return ToolResult(
                output=f"Sub-agent failed: {e}", is_error=True
            )

        self._trace_manager.update(
            trace_node.agent_id,
            input_tokens=sub_agent.total_input_tokens,
            output_tokens=sub_agent.total_output_tokens,
        )
        self._trace_manager.complete(trace_node.agent_id, "completed")

        return ToolResult(output=result_text or "(sub-agent returned no output)")


    async def _execute_with_worktree(self, p: AgentToolParams) -> ToolResult:
        if self._worktree_manager is None:
            return ToolResult(
                output="Worktree isolation is not available: WorktreeManager not configured.",
                is_error=True,
            )

        from aicocode.agents.fork import ForkError, build_forked_messages
        from aicocode.agents.parser import AgentDef
        from aicocode.agents.tool_filter import clone_registry_for_fork, resolve_agent_tools
        from aicocode.agent import Agent as AgentClass
        from aicocode.conversation import Conversation
        from aicocode.Permissions import (
            DangerousCommandDetector,
            PathSandbox,
            PermissionValidator,
            PermissionMode,
            RuleEngine,
        )

        from aicocode.worktree.intergration import (
            build_worktree_notice,
            generate_worktree_name,
        )

        agent_def: AgentDef | None = None
        subagent_type = p.subagent_type
        if subagent_type:
            agent_def = self._agent_loader.get(subagent_type)
            if agent_def is None:
                return ToolResult(
                    output=f"Unknown agent type: '{subagent_type}'. "
                           f"Available types: {', '.join(t for t, _ in self._agent_loader.list_agents())}",
                    is_error=True,
                )
        else:
            agent_def = AgentDef(
                agent_type="worktree-agent",
                when_to_use="Isolated worktree agent",
                system_prompt="",
                disallowed_tools=[],
                model="inherit",
                max_turns=self._parent_agent.max_iterations,
                permission_mode="bypassPermissions",
                source="builtin",
            )

        wt_name = generate_worktree_name()
        try:
            wt = await self._worktree_manager.create(wt_name, "HEAD")
        except Exception as e:
            return ToolResult(
                output=f"Failed to create worktree: {e}",
                is_error=True,
            )
        notice = build_worktree_notice(self._parent_agent.work_dir, wt.path)
        task = notice + "\n\n" + p.prompt

        llm_client = self._select_llm(p, agent_def)

        # 构建子 agent 工具注册表
        _base_registry = getattr(self._parent_agent, '_full_registry', None) or self._parent_agent.registry

        filtered_registry = resolve_agent_tools(
            _base_registry, agent_def, False
        )

        permission = agent_def.permission_mode
        permission_enum = getattr(
            PermissionMode,
            PERMISSION_MODE_MAP.get(permission, "DEFAULT"),
            PermissionMode.DEFAULT,
        )

        permission_validator = PermissionValidator(
            danger_command_detector=DangerousCommandDetector(),
            path_sandbox=PathSandbox(self._parent_agent.work_dir),
            rule_engine=RuleEngine(),
            permission_mode=permission_enum,
        )

        sub_agent = AgentClass(
            client=llm_client,
            registry=filtered_registry,
            protocol=self._parent_agent.protocol,
            work_dir=self._parent_agent.work_dir,
            max_iterations=agent_def.max_turns,
            permission_validator=permission_validator,
            context_window=self._parent_agent.context_window,
            instructions_content=agent_def.system_prompt,
            hook_engine=self._parent_agent.hook_engine,
        )
        sub_agent.parent_id = self._parent_agent.agent_id
        sub_agent.trace_id = self._parent_agent.trace_id or self._parent_agent.agent_id

        trace_node = self._trace_manager.create(
            agent_type=agent_def.agent_type,
            parent_id=self._parent_agent.agent_id,
            trace_id=sub_agent.trace_id,
        )
        sub_agent.agent_id = trace_node.agent_id

        try:
            result_text = await sub_agent.run_to_completion("", task)

        except Exception as e:
            self._trace_manager.complete(trace_node.agent_id, "failed")
            return ToolResult(
                output=f"Sub-agent in worktree failed: {e}", is_error=True
            )

        self._trace_manager.update(
            trace_node.agent_id,
            input_tokens=sub_agent.total_input_tokens,
            output_tokens=sub_agent.total_output_tokens,
        )
        self._trace_manager.complete(trace_node.agent_id, "completed")

        cleanup = await self._worktree_manager.auto_cleanup(wt_name, wt.head_commit)
        if cleanup.kept:
            result_text = (result_text or "") + (
                f"\n[Worktree preserved at {cleanup.path}, branch {cleanup.branch}]"
            )
            
        return ToolResult(output=result_text or "(sub-agent returned no output)")

    def _select_llm(
        self,
        params: AgentToolParams,
        agent_def: AgentDef,
    ) -> LLMClient:
        from aicocode.agents.parser import AgentDef

        model_override = params.model or (
            agent_def.model if agent_def.model != "inherit" else None
        )

        if model_override and model_override != "inherit":
            client = self._create_client_for_model(model_override)
            if client is not None:
                return client

        return self._parent_agent.client
    
    def _create_client_for_model(self, model_alias: str) -> LLMClient | None:
        if self._provider_config is None:
            return None

        from aicocode.llm_client import create_client
        from aicocode.config import ProviderConfig

        model_map = {
            "haiku": "claude-haiku-4-5-20251001",
            "sonnet": "claude-sonnet-4-6-20250514",
            "opus": "claude-opus-4-6-20250514",
            "glm": "glm-4.6",
        }
        model_id = model_map.get(model_alias, model_alias)

        config = ProviderConfig(
            name=f"sub-{model_alias}",
            protocol=self._provider_config.protocol,
            base_url=self._provider_config.base_url,
            model=model_id,
            api_key=self._provider_config.api_key,
            context_window=self._provider_config.context_window,
        )
        try:
            return create_client(config)
        except Exception:
            return None