"""`python -m aicocode` 入口。"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

from aicocode.config import ConfigError, load_config
from aicocode.Permissions import PermissionMode
from aicocode.hooks import HookConfigError, HookEngine, load_hooks

def _parse_teammate_flags(args: list[str]) -> tuple[str, str] | None:
    """
    从 CLI 参数里解析队友 worker 模式。仅当首个参数是 --teammate 时返回 (team_name, agent_name)，表示 worker 模式；
    否则返回 None，调用方应启动正常 TUI。格式对齐 build_teammate_cli 的产出：--teammate --team-name <t> --agent-name <n>
    """
    if not args or args[0] != "--teammate":
        return None
    team_name = ""
    agent_name = ""
    i = 1
    while i < len(args):
        if args[i] == "--team-name" and i + 1 < len(args):
            team_name = args[i + 1]
            i += 2
            continue
        if args[i] == "--agent-name" and i + 1 < len(args):
            agent_name = args[i + 1]
            i += 2
            continue
        i += 1
    return team_name, agent_name


async def _build_teammate_registry(
    work_dir: str,
    protocol: str,
    team_manager: "TeamManager",
    team_name: str,
    agent_name: str,
    mcp_servers: list,
):
    """
    组装队友工具集。
    文件与命令工具、工具检索、Worktree 切换、Skill、MCP 扩展
    团队协作工具（按自己的名字发消息，以及读写团队共享任务板）。任务板按团队名解析到同一份
    tasks.json，所以队友之间看到的是同一张表。
    """
    from aicocode.config import WorkTreeConfig
    from aicocode.mcp import MCPManager
    from aicocode.tools import create_default_registry
    from aicocode.tools.enter_worktree import EnterWorktreeTool
    from aicocode.tools.exit_worktree import ExitWorktreeTool
    from aicocode.tools.impl.tool_search import ToolSearchTool
    from aicocode.tools.install_skill import InstallSkillTool
    from aicocode.tools.load_skill import LoadSkill
    from aicocode.tools.send_message import SendMessageTool
    from aicocode.tools.synthetic_output import SyntheticOutputTool
    from aicocode.tools.task_create import TaskCreateTool
    from aicocode.tools.task_get import TaskGetTool
    from aicocode.tools.task_list import TaskListTool
    from aicocode.tools.task_update import TaskUpdateTool
    from aicocode.worktree import WorktreeManager

    registry = create_default_registry()
    registry.register_tool(ToolSearchTool(registry, protocol=protocol))
    registry.register_tool(SyntheticOutputTool())

    wt_manager = WorktreeManager(
        repo_root=work_dir,
        symlink_directories=WorkTreeConfig().symlink_directories,
    )
    registry.register_tool(EnterWorktreeTool(worktree_manager=wt_manager))
    registry.register_tool(ExitWorktreeTool(worktree_manager=wt_manager))

    # 未注入执行器，声明 fork 模式的 skill 会退回 inline 执行
    registry.register_tool(LoadSkill())
    registry.register_tool(InstallSkillTool())

    registry.register(SendMessageTool(
        team_manager=team_manager,
        team_name=team_name,
        from_agent_id=agent_name,
        from_agent_name=agent_name,
    ))
    registry.register_tool(TaskCreateTool(team_manager, team_name, agent_name))
    registry.register_tool(TaskGetTool(team_manager, team_name))
    registry.register_tool(TaskListTool(team_manager, team_name))
    registry.register_tool(TaskUpdateTool(team_manager, team_name))

    if mcp_servers:
        try:
            manager = MCPManager()
            manager.load_configs(mcp_servers)
            result = await manager.register_all_tools(registry)
            for err in result.errors:
                print(f"MCP warning: {err}", file=sys.stderr)
        except Exception as e:  # MCP 连不上不应该拖垮队友进程
            print(f"MCP setup failed: {e}", file=sys.stderr)

    return registry


async def _run_teammate(team_name: str, agent_name: str) -> None:
    """把本进程作为已有团队的队友 worker 启动。

    流程：加载 config → 建 LLM client → 建工具集（含 SendMessage）→ 定位团队邮箱
    （lead 已在磁盘上创建）→ 建子 agent → 注册成员名字 → 跑队友主循环，
    首个任务由 lead 在 spawn 前写进邮箱、worker 首次空闲轮询取出。
    """
    from aicocode.agent import Agent
    from aicocode.llm_client import create_client, resolve_context_window
    from aicocode.memory.instructions import load_instructions
    from aicocode.Permissions import (
        DangerousCommandDetector,
        PathSandbox,
        PermissionValidator,
        PermissionMode,
        RuleEngine,
    )
    from aicocode.teams.manager import TeamManager
    from aicocode.teams.registry import AgentNameRegistry
    from aicocode.teams.spawn_inprocess import LEAD_NAME, spawn_inprocess_teammate

    # worker 无 TUI，日志走 stderr，供 tmux/iTerm2 窗格直接显示
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, force=True)

    if not team_name or not agent_name:
        print("--teammate requires --team-name and --agent-name", file=sys.stderr)
        sys.exit(1)

    try:
        config = load_config()
    except ConfigError as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    if not config.providers:
        print("No providers configured", file=sys.stderr)
        sys.exit(1)

    provider = config.providers[0]
    client = create_client(provider)
    await resolve_context_window(provider)

    work_dir = os.getcwd()

    # 团队目录由 lead 在磁盘上建好，worker 按团队名加载团队与邮箱
    team_manager = TeamManager()
    team = team_manager.get_team(team_name)
    if team is None:
        print(f"Team '{team_name}' not found", file=sys.stderr)
        sys.exit(1)
    mailbox = team_manager.get_mailbox(team_name)
    if mailbox is None:
        print(f"Mailbox for team '{team_name}' not found", file=sys.stderr)
        sys.exit(1)

    # 名字解析表：登记自己和 lead，便于 SendMessage 按名字投递
    name_registry = AgentNameRegistry.instance()
    name_registry.register(agent_name, agent_name)
    name_registry.register(LEAD_NAME, team.lead_agent_id)

    registry = await _build_teammate_registry(
        work_dir=work_dir,
        protocol=provider.protocol,
        team_manager=team_manager,
        team_name=team_name,
        agent_name=agent_name,
        mcp_servers=config.mcp_servers,
    )

    checker = PermissionValidator(
        danger_command_detector=DangerousCommandDetector(),
        path_sandbox=PathSandbox(work_dir),
        rule_engine=RuleEngine(
            user_rules_path=Path.home() / ".aicocode" / "permissions.yaml",
            project_rules_path=Path(work_dir) / ".aicocode" / "permissions.yaml",
            local_rules_path=Path(work_dir) / ".aicocode" / "permissions.local.yaml",
        ),
        mode=PermissionMode.BYPASS,
    )

    agent = Agent(
        client=client,
        registry=registry,
        protocol=provider.protocol,
        work_dir=work_dir,
        permission_validator=checker,
        context_window=provider.get_context_window(),
        instructions_content=load_instructions(work_dir),
    )

    # 不传初始 prompt：lead 已把首个任务写进邮箱，主循环首次轮询即可取到，
    print(f"[teammate {team_name}/{agent_name}] booted, awaiting tasks", file=sys.stderr)
    handle = spawn_inprocess_teammate(
        agent=agent,
        prompt="",
        name=agent_name,
        team_name=team_name,
        mailbox=mailbox,
        # 外部 worker 把 idle 通知写到 lead 实际读取的键，保证回传对得上
        lead_key=team.lead_agent_id,
    )
    try:
        await handle.task
    except (KeyboardInterrupt, asyncio.CancelledError):
        handle.cancel()


async def _run_prompt(config, permission_mode, hook_engine, prompt: str, output_format: str = "text") -> None:
    from aicocode.agent import Agent
    from aicocode.agent_event import (
        CompactNotification,
        ErrorEvent,
        ErrorEvent,
        LoopComplete,
        PermissionRequest,
        PermissionResponse,
        RetryEvent,
        StreamText,
        ThinkingText,
        ToolResultEvent,
        ToolUseEvent,
        TurnComplete,
        UsageEvent,
        AskUserRequest,
        
    )

    from aicocode.llm_client import create_client, resolve_context_window
    from aicocode.conversation import Conversation
    from aicocode.memory.instructions import load_instructions
    from aicocode.Permissions import (
        DangerousCommandDetector,
        PathSandbox,
        PermissionValidator,
        RuleEngine,
    )
    from aicocode.tools import create_default_registry
    from aicocode.agents.loader import AgentLoader
    from aicocode.agents.task_manager import TaskManager
    from aicocode.agents.trace import TraceManager
    from aicocode.tools.agent_tool import AgentTool
    from aicocode.tools.impl.tool_search import ToolSearchTool
    from aicocode.teams.manager import TeamManager
    from aicocode.teams.models import BackendType
    from aicocode.tools.team_create import TeamCreateTool
    from aicocode.tools.team_delete import TeamDeleteTool
    from aicocode.worktree import WorktreeManager
    from aicocode.config import WorkTreeConfig

    is_json = output_format == "stream-json"

    def emit_json(obj: dict) -> None:
        """输出一行 NDJSON 到 stdout"""
        print(json.dumps(obj, ensure_ascii=False), flush=True)

    provider = config.providers[0]
    client = create_client(provider)

    await resolve_context_window(provider)
    work_dir = os.getcwd()
    home = Path.home()

    checker = PermissionValidator(
        danger_command_detector=DangerousCommandDetector(),
        path_sandbox=PathSandbox(work_dir),
        rule_engine=RuleEngine(
            user_rules_path=home / ".aicocode" / "permissions.yaml",
            project_rules_path=Path(work_dir) / ".aicocode" / "permissions.yaml",
            local_rules_path=Path(work_dir) / ".aicocode" / "permissions.local.yaml",
        ),
        permission_mode=permission_mode,
    )

    instructions = load_instructions(work_dir)
    registry = create_default_registry()
    registry.register_tool(ToolSearchTool(registry, protocol=provider.protocol))

    agent = Agent(
        client=client,
        registry=registry,
        protocol=provider.protocol,
        work_dir=work_dir,
        permission_validator=checker,
        context_window=provider.get_context_window(),
        instructions_content=instructions,
        hook_engine=hook_engine,
    )

    wt_cfg = config.worktree or WorkTreeConfig()
    wt_manager = WorktreeManager(
        repo_root=work_dir,
        symlink_directories=wt_cfg.symlink_directories,
    )
    trace_manager = TraceManager()
    task_manager = TaskManager()
    agent_loader = AgentLoader(work_dir, enable_verification=config.enable_verification_agent)
    agent_loader.load_all()
    team_manager = TeamManager(worktree_manager=wt_manager, trace_manager=trace_manager)

    agent_tool = AgentTool(
        agent_loader=agent_loader,
        task_manager=task_manager,
        trace_manager=trace_manager,
        parent_agent=agent,
        enable_fork=config.enable_fork,
        provider_config=provider,
        worktree_manager=wt_manager,
        team_manager=team_manager,
    )
    registry.register_tool(agent_tool)
    registry.register_tool(TeamCreateTool(
        team_manager=team_manager,
        parent_agent=agent,
        teammate_mode="in-process",
        is_interactive=False,
        enable_coordinator_mode=config.enable_coordinator_mode,
    ))
    registry.register_tool(TeamDeleteTool(team_manager=team_manager, parent_agent=agent))

    from aicocode.tools.synthetic_output import SyntheticOutputTool
    from aicocode.tools.task_stop import TaskStopTool

    registry.register_tool(SyntheticOutputTool())
    registry.register_tool(TaskStopTool(team_manager=team_manager))

    # coordinator 模式由配置决定，开了就从第一轮起收窄工具集
    if config.enable_coordinator_mode:
        from aicocode.agents.tool_filter import apply_coordinator_filter

        agent.enable_coordinator_mode = True
        agent.registry = apply_coordinator_filter(agent.registry)

    def drain_notifications() -> list[str]:
        notes: list[str] = []
        for t in task_manager.poll_completed():
            notes.append(
                f"<task-notification>\n<task_id>{t.id}</task_id>\n"
                f"<status>{t.status}</status>\n<result>{t.result}</result>\n"
                f"</task-notification>"
            )
        notes.extend(team_manager.drain_lead_mailbox())
        return notes

    def drain_mailbox_only() -> list[str]:
        return team_manager.drain_lead_mailbox()

    agent.notification_fn = drain_mailbox_only

    # 使用事件驱动的 agent.run()，支持 text 和 stream-json 两种输出格式
    conv = Conversation()
    conv.add_user_message(prompt)

    start = time.monotonic()
    text_buf = ""
    total_input = 0
    total_output = 0
    tool_calls: list[dict] = []

    async for event in agent.run(conv):
        if isinstance(event, StreamText):
            text_buf += event.text
            if is_json:
                emit_json({"type": "assistant", "text": event.text})

        elif isinstance(event, ThinkingText):
            if is_json:
                emit_json({"type": "thinking", "text": event.text})

        elif isinstance(event, ToolUseEvent):
            tool_calls.append({"name": event.tool_name, "is_error": False})
            if is_json:
                emit_json({
                    "type": "tool_use",
                    "tool_name": event.tool_name,
                    "tool_id": event.tool_id,
                    "args": event.arguments,
                })

        elif isinstance(event, ToolResultEvent):
            # 回填最后一个同名 tool_call 的 is_error
            if tool_calls:
                tool_calls[-1]["is_error"] = event.is_error
            if is_json:
                emit_json({
                    "type": "tool_result",
                    "tool_name": event.tool_name,
                    "tool_id": event.tool_id,
                    "output": event.output,
                    "is_error": event.is_error,
                    "elapsed": round(event.elapsed, 3),
                })

        elif isinstance(event, UsageEvent):
            total_input = event.input_tokens
            total_output = event.output_tokens
            if is_json:
                emit_json({
                    "type": "usage",
                    "input_tokens": event.input_tokens,
                    "output_tokens": event.output_tokens,
                })

        elif isinstance(event, TurnComplete):
            if is_json:
                emit_json({"type": "turn_complete", "turn": event.turn})

        elif isinstance(event, LoopComplete):
            # 最终结果：stream-json 输出 result 行，text 模式直接打印文本
            elapsed_ms = int((time.monotonic() - start) * 1000)
            if is_json:
                emit_json({
                    "type": "result",
                    "result": text_buf,
                    "duration_ms": elapsed_ms,
                    "num_turns": event.total_turns,
                    "tool_calls": tool_calls,
                    "usage": {
                        "input_tokens": total_input,
                        "output_tokens": total_output,
                    },
                    "stop_reason": "end_turn",
                })
            else:
                print(text_buf, end="", flush=True)
            break

        elif isinstance(event, ErrorEvent):
            if is_json:
                emit_json({"type": "error", "message": event.message})
            else:
                print(f"Error: {event.message}", file=sys.stderr, flush=True)

        elif isinstance(event, CompactNotification):
            if is_json:
                emit_json({"type": "compact", "message": event.message})

        elif isinstance(event, RetryEvent):
            if is_json:
                emit_json({"type": "retry", "reason": event.reason})

        elif isinstance(event, PermissionRequest):
            # -p 非交互模式：自动批准所有权限请求
            event.future.set_result(PermissionResponse.ALLOW)

    if not team_manager._teams:
        return

    for i in range(90):
        await asyncio.sleep(2)
        running = {k: not t.done() for k, t in task_manager._async_tasks.items()}
        completed_ids = [t.id for t in task_manager._tasks.values() if t.status != "running"]
        print(f"[poll {i}] running={running} completed={completed_ids} teams={list(team_manager._teams.keys())} queue_size={task_manager._notify_queue.qsize()}", file=sys.stderr, flush=True)
        notes = drain_notifications()
        if not notes:
            has_running = any(v for v in running.values())
            if not has_running:
                print(f"[poll {i}] no running tasks, breaking", file=sys.stderr, flush=True)
                break
            continue
        for note in notes:
            conv.add_system_reminder(note)
        # 后续 team 轮询仍用 run_to_completion，避免重复事件循环
        last_result = await agent.run_to_completion(
            "Teammate notifications received. Process them and continue.", conv
        )
        if is_json:
            emit_json({"type": "assistant", "text": last_result})
        else:
            print(last_result, flush=True)


def main() -> None:
    teammate = _parse_teammate_flags(sys.argv[1:])

    if teammate is not None:
        asyncio.run(_run_teammate(*teammate))
        return

    Path(".aicocode").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        filename=".aicocode/debug.log",
        filemode="w",
    )

    parser = argparse.ArgumentParser(prog="aicocode", description="AicoCode AI coding assistant")
    parser.add_argument(
        "--mode",
        choices=[m.value for m in PermissionMode],
        default=None,
        help="Permission mode (overrides config.yaml)",
    )

    parser.add_argument(
        "-p",
        metavar="PROMPT",
        default=None,
        help="Run non-interactively: execute the prompt and print the result to stdout",
    )
    parser.add_argument(
        "--output-format",
        choices=["text", "stream-json"],
        default="text",
        help="Output format for -p mode: 'text' (default) prints final text, 'stream-json' emits NDJSON events",
    )

    args = parser.parse_args()

    try:
        config = load_config()
    except ConfigError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    mode = args.mode if args.mode else config.permission_mode
    permission_mode = PermissionMode(mode)

    try:
        hooks = load_hooks(config.raw_hooks)
    except HookConfigError as e:
        print(f"Hook config error: {e}", file=sys.stderr)
        sys.exit(1)

    hook_engine = HookEngine(hooks) if hooks else None

    if args.p is not None:
        output_format = getattr(args, "output_format", "text")
        asyncio.run(_run_prompt(config, permission_mode, hook_engine, args.p, output_format))
        return

    from aicocode.app import CodeApp
    from aicocode.driver import NoAltScreenDriver

    app = CodeApp(
        providers=config.providers,
        permission_mode=permission_mode,
        driver_class=NoAltScreenDriver,
        sandbox_config=config.sandbox,
        mcp_servers=config.mcp_servers,
        hook_engine=hook_engine,
        enable_fork=config.enable_fork,
        enable_verification_agent=config.enable_verification_agent,
        worktree_config=config.worktree,
        enable_coordinator_mode=config.enable_coordinator_mode,
        teammate_mode=config.teammate_mode,
    )
    app.run()

if __name__ == "__main__":
    main()