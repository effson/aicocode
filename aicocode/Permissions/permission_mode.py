from __future__ import annotations

from enum import Enum
from typing import Literal

from aicocode.tools.tool_base import ToolCategory

PermissionLevel = Literal["allow", "deny", "query"]
PermissionValidateRes = Literal["allow", "deny", "query"]


class PermissionMode(str, Enum):
    DEFAULT = "default"
    ACCEPT_EDITS = "acceptEdits"
    PLAN = "plan"
    BYPASS = "bypassPermissions"


_MODE_MATRIX: dict[PermissionMode, dict[ToolCategory, PermissionLevel]] = {
    PermissionMode.DEFAULT: {"read": "allow", "write": "query", "command": "query"},
    PermissionMode.ACCEPT_EDITS: {"read": "allow", "write": "allow", "command": "query"},
    PermissionMode.PLAN: {"read": "allow", "write": "query", "command": "query"},
    PermissionMode.BYPASS: {"read": "allow", "write": "allow", "command": "allow"},
}


def mode_permission_decision(mode: PermissionMode, category: ToolCategory) -> PermissionValidateRes:
    return _MODE_MATRIX[mode][category]