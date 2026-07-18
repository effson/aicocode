from __future__ import annotations

import re
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Literal

import yaml

"""
YAML文件：
- rule: "Bash(npm install:*)"                                                                                                                                                          
  permission: allow                                                                                                                                                                        
- rule: "Bash(rm -rf:*)"                                                                                                                                                               
  permission: deny
  ...
"""

PermissionLevel = Literal["allow", "deny", "query"]
_RULE_RE = re.compile(r"^(\w+)\((.+)\)$")

_CONTENT_FIELDS = {
    "Bash": "command", "ReadFile": "file_path", "WriteFile": "file_path",
    "EditFile": "file_path", "Glob": "pattern", "Grep": "pattern",
}

@dataclass
class Rule:
    tool_name: str
    pattern: str
    permission: PermissionLevel

    def matches(self, tool_name: str, content: str) -> bool:
        if self.tool_name != tool_name:
            return False
        return fnmatch(content, self.pattern)

"""
 Bash(git push:*) -> Rule(tool_name="Bash", pattern="git push:*", ...)
"""
def parse_rule(raw: str, permission: PermissionLevel) -> Rule:
    m = _RULE_RE.match(raw.strip())
    if not m:
        raise ValueError(f"无效的规则语法: {raw}")
    return Rule(tool_name=m.group(1), pattern=m.group(2), permission=permission)

"""
    提取工具参数里的「关键内容」，用_CONTENT_FIELDS 映射                                                                                                                                                                                     
    extract_content("Bash", {"command": "git push"}) → "git push"。其他工具返回空串。
"""
def extract_content(tool_name: str, arguments: dict[str, Any]) -> str:
    field = _CONTENT_FIELDS.get(tool_name)
    if field is None:
        return ""
    return str(arguments.get(field, ""))


def _load_rules_file(path: Path) -> list[Rule]:
    if not path.is_file():
        return []
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError):
        return []
    if not isinstance(raw, list):
        return []
    rules: list[Rule] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        rule_str = entry.get("rule", "")
        permission = entry.get("permission", "")
        if permission not in ("allow", "deny", "query"):
            continue
        try:
            rules.append(parse_rule(rule_str, permission))
        except ValueError:
            continue
    return rules

class RuleEngine:
    def __init__(
        self,
        user_rules_path: Path | None = None,
        project_rules_path: Path | None = None,
        local_rules_path: Path | None = None,
    ) -> None:
        self._user_path = user_rules_path
        self._project_path = project_rules_path
        self._local_path = local_rules_path

    def _load_tiers(self) -> list[list[Rule]]:
        tiers: list[list[Rule]] = []
        for p in (self._user_path, self._project_path, self._local_path):
            tiers.append(_load_rules_file(p) if p else [])
        return tiers


    def evaluate(self, tool_name: str, content: str) -> PermissionLevel | None:
        for rules in self._load_tiers():
            for rule in reversed(rules):
                if rule.matches(tool_name, content):
                    return rule.permission
        return None


    def append_local_rule(self, rule: Rule) -> None:
        if self._local_path is None:
            return
        self._local_path.parent.mkdir(parents=True, exist_ok=True)
        existing = _load_rules_file(self._local_path)
        existing.append(rule)
        entries = [{"rule": f"{r.tool_name}({r.pattern})", "permission": r.permission} for r in existing]
        self._local_path.write_text(yaml.dump(entries, allow_unicode=True), encoding="utf-8")