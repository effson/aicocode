from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aicocode.tools.file_state_cache import FileStateCache
from aicocode.tools.tool_base import Tool
if TYPE_CHECKING:
    from aicocode.file_cache import FileCache

class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._disabled: set[str] = set()
        self._discovered: set[str] = set()

    def register_tool(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get_tool(self, tool_name: str) -> Tool | None:
        return self._tools.get(tool_name)

    def tool_is_enabled(self, tool_name: str) -> bool:
        return tool_name in self._tools and tool_name not in self._disabled

    def enable_tool(self, tool_name: str) -> None:
        self._disabled.discard(tool_name)

    def disable_tool(self, name: str) -> None:
        if name in self._tools:
            self._disabled.add(name)

    def enable_all_tools(self) -> None:
        self._disabled.clear()

    def mark_discovered_tools(self, name: str) -> None:
        self._discovered.add(name)

    def tool_is_discovered(self, name: str) -> bool:
        return name in self._discovered

    def get_deferred_tool_names(self) -> list[str]:
        return [
            name
            for name, tool in self._tools.items()
            if getattr(tool, "should_defer", False)
            and name not in self._discovered
            and name not in self._disabled
        ]

    def search_deferred_tools(
        self, query: str, max_results: int, protocol: str = "anthropic"
    ) -> list[dict[str, Any]]:
        query_lower = query.lower()
        scored: list[tuple[int, str, Tool]] = []
        for name, tool in self._tools.items():
            if not getattr(tool, "should_defer", False):
                continue
            if name in self._disabled:
                continue
            score = 0
            name_lower = name.lower()
            desc_lower = (tool.description or "").lower()
            if query_lower in name_lower:
                score += 10
            if query_lower in desc_lower:
                score += 5
            for word in query_lower.split():
                if word in name_lower:
                    score += 3
                if word in desc_lower:
                    score += 1
            if score > 0:
                scored.append((score, name, tool))
        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[dict[str, Any]] = []
        for _, _name, tool in scored[:max_results]:
            base = tool.get_schema()
            if protocol in ("openai", "openai-compatible"):
                results.append({
                    "type": "function",
                    "name": base["name"],
                    "description": base["description"],
                    "parameters": base["input_schema"],
                })
            else:
                results.append(base)
        return results

    def find_deferred_by_names(
        self, names: list[str], protocol: str = "anthropic"
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for name in names:
            tool = self._tools.get(name)
            if tool is None:
                continue
            if not getattr(tool, "should_defer", False):
                continue
            base = tool.get_schema()
            if protocol in ("openai", "openai-compatible"):
                results.append({
                    "type": "function",
                    "name": base["name"],
                    "description": base["description"],
                    "parameters": base["input_schema"],
                })
            else:
                results.append(base)
        return results

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())


    def get_all_schemas(self, protocol: str = "anthropic") -> list[dict[str, Any]]:
        schemas: list[dict[str, Any]] = []
        for name, tool in self._tools.items():
            if name in self._disabled:
                continue
            if getattr(tool, "should_defer", False) and name not in self._discovered:
                continue
            base = tool.get_schema()
            if protocol in ("openai", "openai-compatible"):
                """
                {
                  "tools": [
                    {
                      "type": "function",
                      "function": {
                        "name": "get_weather",
                        "description": "获取指定城市的天气",
                        "parameters": {
                          "type": "object",
                          "properties": {
                            "location": {
                              "type": "string",
                              "description": "城市名称，例如：北京"
                            }
                          },
                          "required": ["location"]
                        }
                      }
                    }
                  ]
                }
                """
                schemas.append({
                    "type": "function",
                    "name": base["name"],
                    "description": base["description"],
                    "parameters": base["input_schema"],
                })
            else:
                """
                {
                  "tools": [
                    {
                      "name": "get_weather",
                      "description": "获取指定城市的天气",
                      "input_schema": {
                        "type": "object",
                        "properties": {
                          "location": {
                            "type": "string",
                            "description": "城市名称，例如：北京"
                          }
                        },
                        "required": ["location"]
                      }
                    }
                  ]
                }
                """
                schemas.append(base)
        return schemas

    def create_default_registry(file_cache: FileCache | None = None, file_history: Any = None) -> ToolRegistry:

        from aicocode.tools.read_file import ReadFile
        from aicocode.tools.edit_file import EditFile
        from aicocode.tools.write_file import WriteFile
        from aicocode.tools.bash import Bash

        file_state_cache = FileStateCache()
        registry = ToolRegistry()

        registry.register_tool(ReadFile(file_cache=file_cache, file_state_cache=file_state_cache))
        registry.register(EditFile(file_cache=file_cache, file_history=file_history, file_state_cache=file_state_cache))
        registry.register(WriteFile(file_cache=file_cache, file_history=file_history, file_state_cache=file_state_cache))
        registry.register(Bash())
        
        return registry