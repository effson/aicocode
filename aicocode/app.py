from __future__ import annotations

import asyncio
import os
import random
import time as _time
from pathlib import Path
from typing import Any, AsyncIterator

from pydantic import ValidationError
from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message as TMessage
from textual.widgets import Markdown, OptionList, Static, TextArea
from textual.widgets.option_list import Option
from rich.text import Text as RichText
from textual.theme import Theme

from aicocode.conversation import (
    ToolUseBlock,
    ToolResultBlock,
)

from aicocode.file_cache import FileCache
from aicocode.tools import ToolRegistry, create_default_registry
from aicocode.tools.impl.tool_search import ToolSearchTool
from aicocode.tools.tool_base import ToolResult

import aicocode.prompt
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
)

from aicocode.llm_client import (
    AuthenticationError,
    LLMClient,
    LLMError,
    create_client,
    resolve_context_window,
)

from aicocode.config import ProviderConfig
from aicocode.conversation import Conversation, Message
from aicocode.commands.popup_completion import CompletionPopup
from aicocode.prompt import build_environment_context
from aicocode.agent import Agent

import re

MAX_TRUNCATED_LINES = 20
MAX_AT_REF_DOC_BYTES = 10240
_AT_REF_DOC_RE = re.compile(r"@([\w./_\-]+(?:\.[\w]+)*)")
_SKIPPED_DOC_DIRS = {".git", "node_modules", ".venv", "__pycache__", ".mewcode", "build", ".gradle"}

"""
    扫描工作目录,为输入框输入的 @文件名 引用提供自动补全候选(跳过 .git/node_modules 等)。
"""
def search_docs_for_at_doc(prefix: str, work_dir: str, limit: int = 10) -> list[str]:
    matches: list[str] = []
    base = os.path.join(work_dir, os.path.dirname(prefix)) if "/" in prefix else work_dir
    name_prefix = os.path.basename(prefix).lower()
    if not os.path.isdir(base):
        return matches
    try:
        for entry in sorted(os.listdir(base)):
            if entry in _SKIPPED_DOC_DIRS or entry.startswith("."):
                continue
            if entry.lower().startswith(name_prefix):
                rel = os.path.join(os.path.dirname(prefix), entry) if "/" in prefix else entry
                if os.path.isdir(os.path.join(base, entry)):
                    rel += "/"
                matches.append(rel)
                if len(matches) >= limit:
                    break
    except OSError:
        pass
    return matches

"""
    把用户输入里的 @path/to/file 展开成 
    `[File: ...]\n
    ```
    内容
    ````,
    塞进 prompt(最多 10KB)
"""
def expand_at_refs_doc(text: str, work_dir: str) -> str:
    def _replace(m: re.Match) -> str:
        rel_path = m.group(1)
        full_path = os.path.join(work_dir, rel_path)
        if not os.path.isfile(full_path):
            return m.group(0)
        try:
            content = open(full_path, encoding="utf-8", errors="replace").read(MAX_AT_REF_DOC_BYTES)
            return f"[File: {rel_path}]\n```\n{content}\n```"
        except Exception:
            return m.group(0)
    return _AT_REF_DOC_RE.sub(_replace, text)

"""
  带键位绑定的多行输入框:                                                                                                                                                            
  - Enter 提交、Shift+Enter/Ctrl+J 换行、Tab 补全、Esc 关弹窗、↑↓ 翻历史。                                                                                                               
  - 历史记录持久化到 .aicocode/history(load_history / _persist_entry_history)。                                                                                                                   
  - 实时检测 / 触发斜杠命令菜单、@ 触发文件补全,通过 SlashMenuUpdate / AtFileRequest / TabComplete / Submitted 四种消息通知 App。
"""
class ChatInput(TextArea):
    BINDINGS = [
        Binding("enter", "submit", "Submit", priority=True),
        Binding("shift+enter", "newline", "Newline", priority=True),
        Binding("ctrl+j", "newline", "Newline", priority=True),
        Binding("tab", "complete", "Complete", priority=True),
        Binding("escape", "dismiss_popup", "Dismiss", priority=True),
        Binding("up", "nav_up", "Navigate up", priority=True),
        Binding("down", "nav_down", "Navigate down", priority=True),
    ]

    class Submitted(TMessage):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    class TabComplete(TMessage):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.cursor_blink = False
        self._history: list[str] = []
        self._history_index: int = -1
        self._history_draft: str = ""
        self._history_file: Path | None = None

    def load_history(self, work_dir: str) -> None:
        self._history_file = Path(work_dir) / ".aicocode" / "history"
        if self._history_file.exists():
            try:
                lines = self._history_file.read_text(encoding="utf-8").splitlines()
                self._history = [l for l in lines if l.strip()]
            except Exception:
                pass

    def _persist_entry_history(self, text: str) -> None:
        if self._history_file is None:
            return
        try:
            self._history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._history_file, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except Exception:
            pass

    def _popup(self) -> CompletionPopup | None:
        try:
            return self.app.query_one(CompletionPopup)
        except Exception:
            return None

    def action_submit(self) -> None:
        popup = self._popup()
        if popup is not None and popup.is_visible:
            selected = popup.get_selected()
            popup.hide()
            if selected:
                self._history.append(selected)
                self._persist_entry_history(selected)
                self._history_index = -1
                self._history_draft = ""
                self.post_message(self.Submitted(selected))
                self.clear()
                return
        text = self.text.strip()
        if text:
            self._history.append(text)
            self._persist_entry_history(text)
            self._history_index = -1
            self._history_draft = ""
            self.post_message(self.Submitted(text))
            self.clear()

    def action_newline(self) -> None:
        self.insert("\n")

    def action_complete(self) -> None:
        popup = self._popup()
        if popup is not None and popup.is_visible:
            selected = popup.get_selected()
            if selected:
                popup.hide()
                self.clear()
                self.insert(selected + " ")
            return
        text = self.text.strip()
        if text.startswith("/"):
            self.post_message(self.TabComplete(text))
        else:
            self.insert("\t")

    def action_dismiss_popup(self) -> None:
        popup = self._popup()
        if popup is not None:
            popup.hide()

    def action_nav_up(self) -> None:
        popup = self._popup()
        if popup is not None and popup.is_visible:
            popup.move_up()
            return
        if not self._history:
            return
        if self._history_index == -1:
            self._history_draft = self.text
            self._history_index = len(self._history) - 1
        elif self._history_index > 0:
            self._history_index -= 1
        else:
            return
        self.clear()
        self.insert(self._history[self._history_index])

    def action_nav_down(self) -> None:
        popup = self._popup()
        if popup is not None and popup.is_visible:
            popup.move_down()
            return
        if self._history_index == -1:
            return
        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            self.clear()
            self.insert(self._history[self._history_index])
        else:
            self._history_index = -1
            self.clear()
            self.insert(self._history_draft)

    class AtFileRequest(TMessage):
        def __init__(self, prefix: str) -> None:
            super().__init__()
            self.prefix = prefix

    class SlashMenuUpdate(TMessage):
        def __init__(self, prefix: str | None) -> None:
            super().__init__()
            self.prefix = prefix
    """
        用 text.rfind("@") 找最后一个 @,取它后面的 after;若 after 非空且不含空格/换行,post AtFileRequest(after)请求消息
        用def on_chat_input_at_file_request(self, event: ChatInput.AtFileRequest) -> None处理请求
    """
    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        text = self.text
        if text.startswith("/") and self._history_index < 0:
            prefix = text[1:]
            if " " not in prefix and "\n" not in prefix:
                self.post_message(self.SlashMenuUpdate(prefix))
            else:
                self.post_message(self.SlashMenuUpdate(None))
        else:
            self.post_message(self.SlashMenuUpdate(None))

        at_idx = text.rfind("@")
        if at_idx < 0:
            return
        after = text[at_idx + 1:]
        if " " in after or "\n" in after:
            return
        if after:
            self.post_message(self.AtFileRequest(after))

_AICOCODE_THEME = Theme(
    name="aicocode",
    primary="#875FFF",
    background="#1a1a1a",
    surface="#1a1a1a",
    panel="#1a1a1a",
    dark=True,
)

THINKING_VERBS = [
    "Accomplishing", "Architecting", "Baking", "Beboppin'", "Befuddling",
    "Bloviating", "Boogieing", "Boondoggling", "Bootstrapping", "Brewing",
    "Calculating", "Canoodling", "Caramelizing", "Cascading", "Cerebrating",
    "Choreographing", "Churning", "Coalescing", "Cogitating", "Combobulating",
    "Composing", "Computing", "Concocting", "Considering", "Contemplating",
    "Cooking", "Crafting", "Creating", "Crunching", "Crystallizing",
    "Cultivating", "Deciphering", "Deliberating", "Dilly-dallying",
    "Discombobulating", "Doodling", "Elucidating", "Enchanting", "Envisioning",
    "Fermenting", "Finagling", "Flambéing", "Flibbertigibbeting", "Flummoxing",
    "Forging", "Frolicking", "Gallivanting", "Garnishing", "Generating",
    "Germinating", "Grooving", "Harmonizing", "Hatching", "Honking",
    "Hullaballooing", "Ideating", "Imagining", "Improvising", "Incubating",
    "Inferring", "Infusing", "Kneading", "Lollygagging", "Manifesting",
    "Marinating", "Meandering", "Metamorphosing", "Mewing", "Moonwalking",
    "Moseying", "Mulling", "Musing", "Noodling", "Orbiting",
    "Orchestrating", "Percolating", "Philosophising", "Pondering",
    "Pontificating", "Pouncing", "Purring", "Puzzling", "Razzle-dazzling",
    "Ruminating", "Scampering", "Simmering", "Sketching", "Spelunking",
    "Spinning", "Sprouting", "Synthesizing", "Thinking", "Tinkering",
    "Transfiguring", "Transmuting", "Undulating", "Unfurling", "Unravelling",
    "Vibing", "Wandering", "Whisking", "Working", "Wrangling", "Zigzagging",
]

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

def _to_past_tense(verb: str) -> str:
    """把现在进行时动词转换为过去式。"""
    if verb.endswith("ing"):
        stem = verb[:-3]
        if stem.endswith("e"):
            return stem + "d"
        if stem and stem[-1] in "atutitet":
            return stem + "ed"
        return stem + "ed"
    return verb + "ed"

COLLAPSIBLE_TOOLS = {"ReadFile", "Glob", "Grep", "ToolSearch"}

class ToolGroupSummary(Static, can_focus=True):


    def __init__(self, count: int, total_elapsed: float, **kwargs: Any) -> None:
        label = f"● Done ({count} tool uses · {total_elapsed:.1f}s)  (ctrl+o to expand)"
        super().__init__(label, **kwargs)
        self._count = count
        self._total = total_elapsed
        self._expanded = False

    def _refresh_display(self) -> None:
        if self._expanded:
            self.update(f"▼ Done ({self._count} tool uses · {self._total:.1f}s)")
        else:
            self.update(
                f"● Done ({self._count} tool uses · {self._total:.1f}s)"
                "  (ctrl+o to expand)"
            )

    def toggle(self) -> None:
        self._expanded = not self._expanded
        self._refresh_display()


    def on_click(self) -> None:
        self.toggle()

def _tool_title(tool_name: str, arguments: dict[str, Any]) -> str:
    if tool_name == "ReadFile":
        path = os.path.basename(arguments.get("file_path", ""))
        return f"Read {path}" if path else "Read"
    if tool_name == "WriteFile":
        path = os.path.basename(arguments.get("file_path", ""))
        content = arguments.get("content", "")
        lines = content.count("\n") + 1 if content else 0
        return f"Write {path} ({lines} lines)" if path else "Write"
    if tool_name == "EditFile":
        path = os.path.basename(arguments.get("file_path", ""))
        return f"Edit {path}" if path else "Edit"
    if tool_name == "Bash":
        cmd = arguments.get("command", "")
        short = cmd[:50] + "…" if len(cmd) > 50 else cmd
        return f"Bash: {short}" if short else "Bash"
    if tool_name == "Glob":
        return f"Glob: {arguments.get('pattern', '')}"
    if tool_name == "Grep":
        return f"Grep: {arguments.get('pattern', '')}"
    return tool_name

def _format_detail(tool_name: str, arguments: dict[str, Any], output: str) -> str:
    parts: list[str] = []

    if tool_name == "Bash":
        parts.append(f"  IN   {arguments.get('command', '')}")
        parts.append("")
        for line in output.splitlines():
            parts.append(f"  OUT  {line}")
    elif tool_name == "EditFile":
        # EditFile 的 output 是 build_diff() 生成的带行号 diff 文本：
        # "+ " 开头绿色、"- " 开头红色，其余（上下文行/摘要行）走 dim。
        # 转义 Rich markup 特殊字符，避免代码里的方括号被当成标签解析。
        for line in output.splitlines()[:MAX_TRUNCATED_LINES]:
            escaped = escape(line)
            if line.startswith("+ "):
                parts.append(f"  [green]{escaped}[/]")
            elif line.startswith("- "):
                parts.append(f"  [red]{escaped}[/]")
            else:
                parts.append(f"  [dim]{escaped}[/]")
        total = output.count("\n") + 1
        if total > MAX_TRUNCATED_LINES:
            parts.append(f"  [dim]… ({total - MAX_TRUNCATED_LINES} more lines)[/]")
    elif tool_name in ("ReadFile", "WriteFile"):
        parts.append(f"  {arguments.get('file_path', '')}")
        parts.append("")
        for line in output.splitlines()[:MAX_TRUNCATED_LINES]:
            parts.append(f"  {line}")
        total = output.count("\n") + 1
        if total > MAX_TRUNCATED_LINES:
            parts.append(f"  … ({total - MAX_TRUNCATED_LINES} more lines)")
    else:
        for line in output.splitlines()[:MAX_TRUNCATED_LINES]:
            parts.append(f"  {line}")
        total = output.count("\n") + 1
        if total > MAX_TRUNCATED_LINES:
            parts.append(f"  … ({total - MAX_TRUNCATED_LINES} more lines)")

    return "\n".join(parts)

class ToolCallBlock(Static, can_focus=True):

    def __init__(self, tool_name: str, arguments: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tool_name = tool_name
        self._arguments = arguments
        self._title = _tool_title(tool_name, arguments)
        self._full_output = ""
        self._is_error = False
        self._elapsed = 0.0
        self._collapsed = True
        self._loading = True
        self._render_loading()

    def _render_loading(self) -> None:
        self.update(f"  ● {self._title} …")
        self.add_class("tool-block-loading")

    def set_result(self, output: str, is_error: bool, elapsed: float) -> None:
        self._full_output = output
        self._is_error = is_error
        self._elapsed = elapsed
        self._loading = False
        self.remove_class("tool-block-loading")
        if is_error:
            self.add_class("tool-block-error")
        # EditFile 的 diff 是最高频需要的信息，默认直接展开，不用等用户点
        # 或按 ctrl+o；其余工具仍然默认折叠，避免刷屏。
        if self.tool_name == "EditFile" and not is_error:
            self._collapsed = False
            self._render_expanded()
        else:
            self._collapsed = True
            self._render_collapsed()

    def _render_collapsed(self) -> None:
        if self._is_error:
            self.update(f"  ✗ {self._title} ({self._elapsed:.1f}s)")
        else:
            self.update(f"  ✓ {self._title} ({self._elapsed:.1f}s)")

    def _render_expanded(self) -> None:
        if self._is_error:
            header = f"  ✗ {self._title} ({self._elapsed:.1f}s)"
        else:
            header = f"  ✓ {self._title} ({self._elapsed:.1f}s)"
        detail = _format_detail(self.tool_name, self._arguments, self._full_output)
        self.update(f"{header}\n{detail}")

    def on_click(self) -> None:
        if self._loading:
            return
        self._collapsed = not self._collapsed
        if self._collapsed:
            self._render_collapsed()
        else:
            self._render_expanded()

class CodeApp(App):
    CSS_PATH = "styles.tcss"
    TITLE = "AicoCode"
    INLINE_PADDING = 0
    theme = "aicocode"
    BINDINGS = [
        Binding("ctrl+c", "handle_ctrl_c", "Quit", priority=True),
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("shift+tab", "cycle_mode", "Cycle mode", priority=True),
        Binding("ctrl+o", "toggle_tool_blocks", "Toggle tools", priority=True),
    ]

    def __init__(
        self,
        providers: list[ProviderConfig],
        driver_class: type | None = None,
    ) -> None:
        super().__init__(driver_class=driver_class)
        self.providers = providers
        self.client: LLMClient | None = None
        self.agent: Agent | None = None
        self.conversation = Conversation()
        self._selected_provider: ProviderConfig | None = None
        self._agent_task: asyncio.Task[None] | None = None
        self._streaming = False
        self._thinking_start: float = 0.0
        self._thinking_verb: str = ""
        self._spinner_idx: int = 0
        self._spinner_timer = None
        self._spinner_label: Static | None = None
        self.file_cache = FileCache()
        self.registry: ToolRegistry = create_default_registry(file_cache=self.file_cache)
        self.work_dir: str = ""
        self._instructions_content: str = ""

    @staticmethod
    def _make_banner(model: str = "", work_dir: str = "") -> RichText:
        t = RichText()
        t.append(" _\\_/_   ", style="bold color(109)")
        t.append(" AicoCode v0.1.0\n", style="color(242)")
        t.append("[ o.o ]   ", style="bold color(109)")
        t.append(f"{model}\n" if model else "\n", style="color(242)")
        t.append("/[_¥_]\\  ", style="bold color(109)")
        t.append(" " + work_dir, style="color(242)")
        return t

    def compose(self) -> ComposeResult:
        yield Static(self._make_banner(), id="title-bar")
        if len(self.providers) > 1:
            with Vertical(id="provider-select"):
                yield Static("Select a Provider", id="select-label")
                yield OptionList(
                    *[
                        Option(f"{p.name}  [{p.model}]", id=p.name)
                        for p in self.providers
                    ],
                    id="provider-list",
                )
        yield VerticalScroll(id="chat-area")
        with Vertical(id="input-area"):
            yield ChatInput(id="chat-input")
            with Horizontal(id="status-bar"):
                yield Static("  default", id="mode-label")
                # yield Static("", id="teammates-label")
                yield Static("", id="model-label")
            yield CompletionPopup()

    def on_mount(self) -> None:
        self.register_theme(_AICOCODE_THEME)
        self.theme = "aicocode"
        if len(self.providers) == 1:
            self._select_provider(self.providers[0])
        else:
            self.query_one("#chat-area").display = False
            self.query_one("#input-area").display = False

    def _select_provider(self, provider: ProviderConfig) -> None:
        self._selected_provider = provider
        try:
            self.client = create_client(provider)
        except AuthenticationError as e:
            self._show_error(str(e))
            return

        work_dir = os.getcwd()
        home = Path.home()

        self.registry.register_tool(
            ToolSearchTool(self.registry, protocol=provider.protocol)
        )

        self.agent = Agent(
            client=self.client,
            registry=self.registry,
            protocol=provider.protocol,
            work_dir=work_dir,
            context_window=provider.get_context_window(),
            instructions_content=self._instructions_content,
        )

        self.query_one("#model-label", Static).update(provider.model)
        work_dir = os.getcwd()
        self.query_one("#title-bar", Static).update(
            self._make_banner(provider.model, work_dir)
        )

        select = self.query("#provider-select")
        if select:
            select.first().display = False
        self.query_one("#chat-area").display = True
        self.query_one("#input-area").display = True
        chat_input = self.query_one("#chat-input", ChatInput)
        chat_input.placeholder = "Send a message..."
        chat_input.load_history(work_dir)
        chat_input.focus()

    def _show_error(self, text: str) -> None:
        chat = self.query_one("#chat-area", VerticalScroll)
        error_widget = Static(f"✖ {text}", classes="message error-message")
        chat.mount(error_widget)
        self.call_after_refresh(chat.scroll_end, animate=False)

    # 选择好provider后会进入_select_provider，和只有一个provider一样的处理流程
    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "provider-list":
            provider = self.providers[event.option_index]
            self._select_provider(provider)

    """
      ┌────────────────────────┬───────────────────────────┐                                                                                                                                
      │ Ctrl+C(流式处理中)       │ 中断当前响应,不退出          │                                                                                                                                 
      ├────────────────────────┼───────────────────────────┤                                                                                                                                 
      │ Ctrl+C(空闲)            │ 清理后退出                  │                                                                                                                                 
      ├────────────────────────┼───────────────────────────┤                                                                                                                                 
      │ Ctrl+C(清理卡住时再按)    │ 立即强制退出                │                                                                                                                                 
      └────────────────────────┴───────────────────────────┘
    """
    async def action_handle_ctrl_c(self) -> None:
        if self._streaming:
            if self._agent_task and not self._agent_task.done():
                self._agent_task.cancel()
            self._show_system_message("(response interrupted)")
            self._finish_streaming()
            try:
                inp = self.query_one("#chat-input", ChatInput)
                inp.disabled = False
                inp.focus()
            except Exception:
                pass
            return

        if getattr(self, "_exit_requested", False):
            self.exit()
            return
        self._exit_requested = True

        async def _cleanup() -> None:
            tasks: list[asyncio.Task] = []

            if tasks:
                await asyncio.wait(tasks, timeout=3.0)
                for t in tasks:
                    if not t.done():
                        t.cancel()

        try:
            await _cleanup()
        except Exception:
            pass
        self.exit()

    """
        处理 @命令 弹窗 + 选择文件， 收到消息触发
    """
    def on_chat_input_at_file_request(self, event: ChatInput.AtFileRequest) -> None:
        work_dir =  os.getcwd()
        matches = search_docs_for_at_doc(event.prefix, work_dir)
        if matches:
            popup = self.query_one(CompletionPopup)
            popup.show([f"@{m}" for m in matches])

    """
        找到输入框里最后一个 @, 把它到末尾的内容替换成选中的 @ path + 空格
    """
    def on_completion_popup_selected(self, event: CompletionPopup.Selected) -> None:
        input_widget = self.query_one("#chat-input", ChatInput)
        selected = event.value
        text = input_widget.text
        if selected.startswith("@"):
            at_idx = text.rfind("@")
            if at_idx >= 0:
                input_widget.clear()
                input_widget.insert(text[:at_idx] + selected + " ")
                input_widget.focus()
                return
        input_widget.clear()
        input_widget.insert(selected + " ")
        input_widget.focus()

    """
        用户按 Enter 输入后的处理逻辑，包括终端agent正在处理任务时，用户输入内容的处理逻辑
    """
    async def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        text = event.text.strip()
        if self._streaming and not text.startswith("/"):
            if self._agent_task and not self._agent_task.done():
                self._agent_task.cancel()
                try:
                    await self._agent_task
                except (asyncio.CancelledError, Exception):
                    pass
            self._finish_streaming()
            self._show_system_message("(response interrupted)")
        await self._dispatch_command_or_input(text)

    async def _dispatch_command_or_input(self, text: str) -> None:
        if self._streaming:
            return
        self._agent_task = asyncio.create_task(self._send_message(text))
        return

    async def _send_message(self, text: str, is_notification: bool = False) -> None:
        self._streaming = True
        chat = self.query_one("#chat-area", VerticalScroll)
        input_widget = self.query_one("#chat-input", ChatInput)

        if text and "@" in text:
            work_dir = os.getcwd()
            text = expand_at_refs_doc(text, work_dir)

        # Start memory recall prefetch before UI work.
        # prefetch_task = asyncio.create_task(
        #     self._prefetch_relevant_memories(text)
        # ) if text else None

        if text:
            user_row = Vertical(classes="user-row")
            await chat.mount(user_row)
            from rich.text import Text as RichText
            user_rich = RichText()
            user_rich.append("❯ ", style="bold color(80)")
            user_rich.append(text, style="bold color(255)")
            user_bubble = Static(user_rich, classes="message user-message")
            await user_row.mount(user_bubble)
            self.call_after_refresh(chat.scroll_end, animate=False)

            self.conversation.add_user_message(text)

        history_cursor = len(self.conversation.messages)

        # 准备 AI 回复区域
        ai_row = Vertical(classes="ai-row")
        await chat.mount(ai_row)
        streaming_label = Static("", classes="message ai-message")
        await ai_row.mount(streaming_label)

        accumulated_text = ""
        tool_blocks: dict[str, ToolCallBlock] = {}

        # 在聊天区底部启动持续旋转的加载动画
        self._thinking_start = _time.monotonic()
        self._thinking_verb = random.choice(THINKING_VERBS)
        self._spinner_idx = 0
        self._spinner_label = Static(
            f"  {SPINNER_FRAMES[0]} {self._thinking_verb}…",
            id="spinner-live",
        )
        await chat.mount(self._spinner_label)

        self.call_after_refresh(chat.scroll_end, animate=False)
        self._start_spinner()

        await asyncio.sleep(0)

        try:
            async for event in self.agent.run(self.conversation):
                if isinstance(event, ThinkingText):
                    self.call_after_refresh(chat.scroll_end, animate=False)

                elif isinstance(event, StreamText):
                    if streaming_label is not None and not accumulated_text:
                        await streaming_label.remove()
                        streaming_label = Static("", classes="message ai-message")
                        await ai_row.mount(streaming_label)
                    accumulated_text += event.text
                    from rich.text import Text as RichText
                    t = RichText()
                    t.append("● ", style="bold color(99)")
                    t.append(accumulated_text)
                    streaming_label.update(t)
                    self.call_after_refresh(chat.scroll_end, animate=False)

                elif isinstance(event, ToolUseEvent):
                    if accumulated_text:
                        if streaming_label is not None:
                            await streaming_label.remove()
                        from rich.text import Text as RichText
                        prefix = Static(RichText("●  ", style="bold color(99)"), classes="message")
                        await ai_row.mount(prefix)
                        md = Markdown(accumulated_text, classes="message ai-message")
                        await ai_row.mount(md)
                        streaming_label = None
                        accumulated_text = ""
                    elif streaming_label is not None:
                        await streaming_label.remove()
                        streaming_label = None

                    block = ToolCallBlock(
                        event.tool_name, event.arguments, classes="tool-block"
                    )

                    await ai_row.mount(block)
                    tool_blocks[event.tool_id] = block
                    self.call_after_refresh(chat.scroll_end, animate=False)
                elif isinstance(event, ToolResultEvent):
                    block = tool_blocks.get(event.tool_id)
                    if block:
                        block.set_result(event.output, event.is_error, event.elapsed)
                    self.call_after_refresh(chat.scroll_end, animate=False)

                elif isinstance(event, TurnComplete):
                    collapsible = [
                        (tid, blk) for tid, blk in tool_blocks.items()
                        if isinstance(blk, ToolCallBlock)
                           and blk.tool_name in COLLAPSIBLE_TOOLS
                           and not blk._loading
                    ]

                    if len(collapsible) >= 2:
                        total_elapsed = sum(b._elapsed for _, b in collapsible)
                        summary = ToolGroupSummary(
                            len(collapsible), total_elapsed,
                            classes="tool-block tool-group-summary",
                        )
                        for _, blk in collapsible:
                            blk.display = False
                        await ai_row.mount(summary)

                    tool_blocks.clear()
                    ai_row = Vertical(classes="ai-row")
                    await chat.mount(ai_row)
                    streaming_label = Static("", classes="message ai-message")
                    await ai_row.mount(streaming_label)
                    accumulated_text = ""
                    self.call_after_refresh(chat.scroll_end, animate=False)

                elif isinstance(event, UsageEvent):
                    pass  # token 展示已移除

                elif isinstance(event, ErrorEvent):
                    # 保留错误前已输出的流式文本
                    if accumulated_text and streaming_label is not None:
                        await streaming_label.remove()
                        md = Markdown(accumulated_text, classes="message ai-message")
                        await ai_row.mount(md)
                        streaming_label = None
                        accumulated_text = ""
                    self._show_error(event.message)

                elif isinstance(event, LoopComplete):
                    total_time = _time.monotonic() - self._thinking_start
                    done_label = Static(
                        f"✻ {_to_past_tense(self._thinking_verb)} for {total_time:.1f}s",
                        classes="message thinking-done",
                    )
                    await ai_row.mount(done_label)

            # 收尾：渲染剩余的累积文本
            if accumulated_text and streaming_label is not None:
                await streaming_label.remove()
                md = Markdown(accumulated_text, classes="message ai-message")
                await ai_row.mount(md)
            elif streaming_label is not None:
                await streaming_label.remove()

            self.call_after_refresh(chat.scroll_end, animate=False)

        except asyncio.CancelledError:
            if accumulated_text:
                if streaming_label is not None:
                    await streaming_label.remove()
                md = Markdown(
                    accumulated_text + "\n\n*[cancelled]*",
                    classes="message ai-message",
                )
                await ai_row.mount(md)
            self._show_system_message("Operation cancelled")
        except LLMError as e:
            self._show_error(str(e))
        finally:
            self._finish_streaming()
            input_widget.focus()

    def _show_system_message(self, text: str) -> None:
        chat = self.query_one("#chat-area", VerticalScroll)
        msg = Static(f"  {text}", classes="message system-message")
        chat.mount(msg)
        self.call_after_refresh(chat.scroll_end, animate=False)
    """
        清理所有 streaming 状态（取消或完成时调用）。
    """
    def _finish_streaming(self) -> None:
        self._streaming = False
        self._stop_spinner()
        self._agent_task = None
        if self._spinner_label is not None:
            self._spinner_label.remove()
            self._spinner_label = None

    def _start_spinner(self) -> None:
        """启动 braille spinner 动画（每帧 80ms）。"""
        if self._spinner_timer is not None:
            return
        self._spinner_timer = self.set_interval(0.08, self._tick_spinner)

    def _stop_spinner(self) -> None:
        """停止 spinner 动画。"""
        if self._spinner_timer is not None:
            self._spinner_timer.stop()
            self._spinner_timer = None

    def _tick_spinner(self) -> None:
        """推进持久 spinner 标签上的动画帧。"""
        self._spinner_idx += 1
        frame = SPINNER_FRAMES[self._spinner_idx % len(SPINNER_FRAMES)]
        elapsed = _time.monotonic() - self._thinking_start
        if self._spinner_label is not None:
            self._spinner_label.update(
                f"  {frame} {self._thinking_verb}…  ({elapsed:.0f}s)"
            )
            if self._spinner_idx % 5 == 0:
                try:
                    self.query_one("#chat-area", VerticalScroll).scroll_end(animate=False)
                except Exception:
                    pass