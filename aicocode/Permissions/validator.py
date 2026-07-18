from __future__ import annotations

import os
from dataclasses import dataclass

from typing import Any

from aicocode.tools.tool_base import Tool
from aicocode.Permissions.danger import DangerousCommandDetector, is_safe_command
from aicocode.Permissions.permission_mode import PermissionLevel, PermissionMode, PermissionValidateRes, mode_permission_decision
from aicocode.Permissions.rules import RuleEngine, extract_content
from aicocode.Permissions.path_sandbox import PathSandbox

_PLAN_MODE_ALLOWED_TOOLS = frozenset({"Agent", "ToolSearch", "AskUserQuestion", "ExitPlanMode"})

@dataclass
class PermissionRes:
    permission: PermissionLevel
    reason: str

class PermissionValidator:
    def __init__(
        self,
        danger_command_detector: DangerousCommandDetector,
        path_sandbox: PathSandbox,
        rule_engine: RuleEngine,
        permission_mode: PermissionMode = PermissionMode.DEFAULT,
        os_sandbox_enabled: bool = False,
    ) -> None:
        self.danger_command_detector = danger_command_detector  # layer 1:危险命令拦截
        self.path_sandbox = path_sandbox      # layer 2: 拦截文件类工具，把读写限制在项目目录内、并禁写敏感配置
        self.rule_engine = rule_engine
        self.permission_mode = permission_mode
        self.plan_file_path: str = ""
        # OS 级沙箱是否启用（开启后命令类工具可自动放行，因为内核会兜底）
        self.os_sandbox_enabled = os_sandbox_enabled
        # Layer 4b: 会话级 allow-always 集合（内存中，不持久化）
        # 存放格式为 "ToolName:pattern"，用户选择 "don't ask again" 时记录
        self._session_allowed: set[str] = set()

    def add_session_allow(self, tool_name: str, content: str) -> None:
        """
            将工具+内容模式加入会话级放行集合
            比持久化规则引擎优先级更高，但不写入磁盘——会话结束即消失。
        """
        key = f"{tool_name}:{content}"
        self._session_allowed.add(key)

    def _check_session_allowed(self, tool_name: str, content: str) -> bool:
        """检查是否匹配会话级放行记录。"""
        if not self._session_allowed:
            return False
        key = f"{tool_name}:{content}"
        if key in self._session_allowed:
            return True
        # 前缀匹配：已记录的 pattern 可能带通配尾缀 *
        for allowed in self._session_allowed:
            if allowed.endswith("*") and key.startswith(allowed[:-1]):
                return True
        return False

    @staticmethod
    def describe_tool_action(tool_name: str, arguments: dict[str, Any]) -> str:
        """HITL 生成人类可读的操作描述"""
        content = extract_content(tool_name, arguments)
        if content:
            return content
        # 无法从标准字段提取时，拼接参数摘要
        parts = []
        for k, v in arguments.items():
            sv = str(v)
            if len(sv) > 80:
                sv = sv[:77] + "..."
            parts.append(f"{k}={sv}")
        return ", ".join(parts) if parts else tool_name

    def check(self, tool: Tool, arguments: dict[str, Any]) -> PermissionRes:
        content = extract_content(tool.name, arguments)

        if self.permission_mode == PermissionMode.PLAN:
            if tool.name in _PLAN_MODE_ALLOWED_TOOLS:
                return PermissionRes(permission="allow", reason="Plan mode: allowed tool")
            if tool.name in ("WriteFile", "EditFile") and content:
                if self._is_plan_file(content):
                    return PermissionRes(permission="allow", reason="Plan mode: plan file write")

        # Layer 1: 安全的只读命令（自动放行）
        if tool.category == "command" and is_safe_command(content or ""):
            return PermissionRes(permission="allow", reason="Safe read-only command")

        # Layer 1b: 危险命令黑名单（仅 Bash）
        if tool.category == "command":
            hit, reason = self.danger_command_detector.detect(content)
            if hit:
                return PermissionRes(permission="deny", reason=f"危险命令拦截: {reason}")

        # Layer 1c: OS 沙箱自动放行
        # 沙箱开启时，命令类工具通过了危险命令检查后直接放行——
        # 内核级隔离会阻止越权写入，无需再弹确认。
        # 拆分复合命令逐条检查，防止通过命令拼接绕过权限检查，
        # deny 规则和 query 规则不受沙箱影响。
        if self.os_sandbox_enabled and tool.category == "command":
            import re
            subcommands = [s.strip() for s in re.split(r'\s*(?:&&|\|\||[;|])\s*', content) if s.strip()]
            if not subcommands:
                subcommands = [content]
            has_query = False
            for sub in subcommands:
                rule_result = self.rule_engine.evaluate(tool.name, sub)
                if rule_result == "deny":
                    return PermissionRes(permission="deny", reason="权限规则拒绝")
                if rule_result == "query":
                    has_query = True
            if has_query:
                return PermissionRes(permission="query", reason="权限规则要求确认")
            return PermissionRes(permission="allow", reason="OS 沙箱自动放行")

        # Layer 2: 路径沙箱（仅文件类工具）
        if tool.category in ("read", "write") and content:
            ok, reason = self.path_sandbox.check(content)
            if not ok and self.permission_mode != PermissionMode.BYPASS:
                return PermissionRes(permission="query", reason=f"路径沙箱拦截: {reason}")

        # Layer 3: 规则引擎匹配
        rule_result = self.rule_engine.evaluate(tool.name, content)
        if rule_result == "allow":
            return PermissionRes(permission="allow", reason="权限规则放行")
        if rule_result == "deny":
            return PermissionRes(permission="deny", reason="权限规则拒绝")

        # Layer 4b: 会话级放行（内存中，优先于模式兜底）
        if self._check_session_allowed(tool.name, content or ""):
            return PermissionRes(permission="allow", reason="会话级放行（session allow-always）")

        # Layer 4: 权限模式兜底判定
        permission = mode_permission_decision(self.permission_mode, tool.category)
        if permission == "allow":
            return PermissionRes(permission="allow", reason=f"权限模式 {self.permission_mode.value} 放行")
        if permission == "deny":
            return PermissionRes(permission="deny", reason=f"权限模式 {self.permission_mode.value} 拒绝")

        # Layer 5: 触发人工确认（HITL）
        return PermissionRes(permission="query", reason="需要用户确认")

    def _is_plan_file(self, target_path: str) -> bool:
        if not self.plan_file_path or not target_path:
            return ".aicocode/plans/" in target_path
        try:
            abs_target = os.path.abspath(target_path)
            abs_plan = os.path.abspath(self.plan_file_path)
            if abs_target == abs_plan:
                return True
        except Exception:
            pass
        if os.path.basename(target_path) == os.path.basename(self.plan_file_path):
            return True
        return ".aicocode/plans/" in target_path