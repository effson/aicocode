from __future__ import annotations

import json
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from aicocode.conversation import (
    Conversation,
    Message,
    ToolResultBlock,
    estimate_tokens,
)

from aicocode.message_adaptor import construct_messages

SINGLE_RESULT_CHAR_LIMIT = 50_000 # 单个工具结果存盘阈值
AGGREGATE_CHAR_LIMIT = 200_000  # 聚合限制，超过会将最大的结果存盘，直至 < 200K
PREVIEW_CHARS = 2_000 # 工具结果落盘时，保留多少字节的预览内容

SUMMARY_OUTPUT_RESERVE = 20_000 # 为摘要预留窗口
AUTO_COMPACT_SAFETY_MARGIN = 13_000 # 到阈值触发自动压缩的安全边界
MANUAL_COMPACT_SAFETY_MARGIN = 3_000 # 手动/compact 命令压缩的安全边界

""""保留近期原文配置"""
KEEP_RECENT_TOKENS = 10_000 # 尾部保留原样的最大 token 数
MIN_KEEP_MESSAGES = 5 # 尾部保留的最大消息数
KEEP_MAX_TOKENS = 40_000 # 尾部保留的最大token数

MIN_SUMMARIZE_PREFIX_TOKENS = 2_000 # 可以做摘要的最少tokens数，少于这个则压缩不划算
PERSISTED_TAG = "<persisted-output>"


# ----------------------------Session 目录管理----------------------------

SESSION_SUBDIR = ".aicocode/session/tool-results"

def ensure_session_dir(work_dir: str) -> Path:
    session_dir = Path(work_dir) / SESSION_SUBDIR
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def cleanup_tool_results(session_dir: Path) -> None:
    if session_dir.exists():
        shutil.rmtree(session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class CompactBoundary:
    """
    `summary` 是大模型对被摘要前缀生成的摘要；
    `keep` 是 auto_compact 原样保留、未做改动的近期尾部消息。
    """
    summary: str
    keep: list[Message]

@dataclass
class CompactEvent:
    before_tokens: int
    # 摘要成功时填充，调用方可据此持久化 compact_boundary 记录。
    boundary: CompactBoundary | None = None # 未产出摘要时为 None。


# ----------------------------内容替换----------------------------

@dataclass
class ToolResReplacementState:
    seen_ids: set[str] = field(default_factory=set)
    replacements: dict[str, str] = field(default_factory=dict)

@dataclass
class ToolResReplacementRecord:
    tool_use_id: str
    replacement: str
    kind: str = "tool-result"

def create_replacement_state() -> ToolResReplacementState:
    return ToolResReplacementState()

REPLACEMENT_RECORDS_FILENAME = "replacement_records.jsonl"

def append_replacement_records(
    session_dir: Path, records: list[ToolResReplacementRecord]
) -> None:
    if not records:
        return
    path = session_dir / REPLACEMENT_RECORDS_FILENAME
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "kind": r.kind,
                "tool_use_id": r.tool_use_id,
                "replacement": r.replacement,
            }, ensure_ascii=False) + "\n")

# ----------------------------策略 1.大的工具结果落盘----------------------------

def persist_tool_result(tool_use_id: str, content: str, session_dir: Path) -> Path:
    file_path = session_dir / f"{tool_use_id}.txt"
    try:
        fd = os.open(str(file_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
    except FileExistsError:
        pass
    return file_path

def make_persisted_tool_result_preview(content: str, file_path: Path) -> str:
    size_kb = len(content.encode("utf-8")) // 1024
    preview = content[:PREVIEW_CHARS]
    return (
        f"{PERSISTED_TAG}\n"
        f"输出结果太大（{size_kb}KB），完整内容已保存到：\n"
        f"{file_path}\n"
        f"预览（前 2KB）：\n"
        f"{preview}\n"
        f"</persisted-output>"
    )

def _is_spill_readback(tool_use_id: str, tool_use_index: dict, abs_spill_dir: str) -> bool:
    tu = tool_use_index.get(tool_use_id)
    if tu is None or tu.tool_name != "ReadFile":
        return False
    raw = tu.arguments.get("file_path", "")
    if not raw:
        return False
    abs_path = os.path.abspath(raw)
    return abs_path.startswith(abs_spill_dir)

def is_spill_readback_call(
    tool_name: str,
    arguments: Mapping[str, Any] | None,
    session_dir: Path,
) -> bool:
    """
    判断一次工具调用是否是「回读落盘文件」的 ReadFile。

    供 agent 层在工具结果产生当下使用（budget 层用 _is_spill_readback 反查
    tool_use_index）。模型用 ReadFile 读取 session_dir 下的落盘文件时，结果本身
    就是已落盘内容的回读；若再次落盘/截断会陷入「落盘→回读→再落盘」循环，且让
    模型刚读到的完整内容瞬间又被截掉。命中时调用方应原样返回，把这条结果交给
    Layer 2（auto_compact）按总量清理。两者共享同一条短路语义。
    """
    if tool_name != "ReadFile":
        return False
    raw = (arguments or {}).get("file_path", "")
    if not raw:
        return False
    abs_spill_dir = os.path.abspath(str(session_dir))
    return os.path.abspath(raw).startswith(abs_spill_dir)

def apply_tool_result_budget(
    conversation: Conversation,
    session_dir: Path,
    state: ToolResReplacementState,
) -> list[ToolResReplacementRecord]:
    """
    Design A: 就地修改原始对话历史，避免拷贝开销。
    直接修改 conversation.messages中ToolResultBlock.content，超限 tool result->落盘预览文本。
    state 会被 mutate：本轮新决定的 id 进入 seen_ids，新决定替换的 id 进入 replacements。
    返回本轮新产生的替换记录列表（List[ToolResReplacementRecord]）。
    """
    new_records: list[ToolResReplacementRecord] = []

    abs_spill_dir = os.path.abspath(str(session_dir))
    tool_use_index: dict = {}
    for msg in conversation.messages:
        for tu in msg.tool_uses:  # list[ToolUseBlock]
            tool_use_index[tu.tool_use_id] = tu

    for msg in conversation.messages:
        if not msg.tool_results:
            continue

        fresh: list[ToolResultBlock] = []

        for tr in msg.tool_results:
            if tr.tool_use_id in state.replacements:
                # 已有历史决策：就地应用替换
                tr.content = state.replacements[tr.tool_use_id]
            elif tr.tool_use_id in state.seen_ids:
                # 见过但未替换：保持原内容不动
                pass
            elif tr.content.startswith(PERSISTED_TAG):
                # 已被外部（如某些工具本身）打上 persisted-output 标签 —— 视为已知决策
                state.seen_ids.add(tr.tool_use_id)
                state.replacements[tr.tool_use_id] = tr.content
                new_records.append(ToolResReplacementRecord(
                    tool_use_id=tr.tool_use_id, replacement=tr.content,
                ))
            else:
                fresh.append(tr)

        # Pass 1：单条超限
        persisted_p1: set[str] = set()
        for tr in fresh:
            if len(tr.content) > SINGLE_RESULT_CHAR_LIMIT:
                if _is_spill_readback(tr.tool_use_id, tool_use_index, abs_spill_dir):
                    persisted_p1.add(tr.tool_use_id)
                    continue
                fp = persist_tool_result(tr.tool_use_id, tr.content, session_dir)
                preview = make_persisted_tool_result_preview(tr.content, fp)
                state.replacements[tr.tool_use_id] = preview
                state.seen_ids.add(tr.tool_use_id)
                new_records.append(ToolResReplacementRecord(
                    tool_use_id=tr.tool_use_id, replacement=preview,
                ))
                # 就地替换消息内容
                tr.content = preview
                persisted_p1.add(tr.tool_use_id)

        # Pass 2：聚合超限
        remaining = [tr for tr in fresh if tr.tool_use_id not in persisted_p1]
        total = sum(
            len(state.replacements[tr.tool_use_id]) if tr.tool_use_id in state.replacements
            else len(tr.content)
            for tr in msg.tool_results
            if tr.tool_use_id not in [r.tool_use_id for r in fresh
                                       if r.tool_use_id not in persisted_p1
                                       and r.tool_use_id not in state.replacements]
        ) + sum(len(tr.content) for tr in remaining)
        # 重新简单计算：所有 tool_results 的当前内容长度之和
        total = sum(len(tr.content) for tr in msg.tool_results)
        if total > AGGREGATE_CHAR_LIMIT:
            ranked = sorted(remaining, key=lambda tr: len(tr.content), reverse=True)
            for tr in ranked:
                if sum(len(t.content) for t in msg.tool_results) <= AGGREGATE_CHAR_LIMIT:
                    break
                if _is_spill_readback(tr.tool_use_id, tool_use_index, abs_spill_dir):
                    continue
                fp = persist_tool_result(tr.tool_use_id, tr.content, session_dir)
                preview = make_persisted_tool_result_preview(tr.content, fp)
                state.replacements[tr.tool_use_id] = preview
                state.seen_ids.add(tr.tool_use_id)
                new_records.append(ToolResReplacementRecord(
                    tool_use_id=tr.tool_use_id, replacement=preview,
                ))
                # 就地替换消息内容
                tr.content = preview

        # 剩余未替换的 fresh 标记为"已见但未替换"
        for tr in fresh:
            if tr.tool_use_id not in state.replacements:
                state.seen_ids.add(tr.tool_use_id)

    return new_records

# ----------------------------策略 2.全对话摘要（Auto-Compact）----------------------------

"""
    计算触发 上下文压缩的阈值
"""
def compute_compact_threshold(context_window: int, manual: bool = False) -> int:
    effective_window = context_window - SUMMARY_OUTPUT_RESERVE
    margin = MANUAL_COMPACT_SAFETY_MARGIN if manual else AUTO_COMPACT_SAFETY_MARGIN
    return effective_window - margin

"""
    判断是否需要压缩
"""
def should_auto_compact(last_input_tokens: int, context_window: int) -> bool:
    return last_input_tokens >= compute_compact_threshold(context_window)

SUMMARY_PROMPT = """\
CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.

- Do NOT use ReadFile, Bash, Grep, Glob, EditFile, WriteFile, or ANY other tool.
- You already have all the context you need in the conversation above.
- Tool calls will be REJECTED and will waste your only turn — you will fail the task.
- Your entire response must be plain text: an <analysis> block followed by a <summary> block.

Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.
This summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing development work without losing context.

Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and ensure you've covered all necessary points. In your analysis process:

1. Chronologically analyze each message and section of the conversation. For each section thoroughly identify:
   - The user's explicit requests and intents
   - Your approach to addressing the user's requests
   - Key decisions, technical concepts and code patterns
   - Specific details like:
     - file names
     - full code snippets
     - function signatures
     - file edits
   - Errors that you ran into and how you fixed them
   - Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
2. Double-check for technical accuracy and completeness, addressing each required element thoroughly.

After your analysis, output your final summary wrapped in <summary> tags. Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay special attention to the most recent messages and include full code snippets where applicable and include a summary of why this file read or edit is important.
4. Errors and fixes: List all errors that you ran into, and how you fixed them. Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
6. All user messages: List ALL user messages that are not tool results. These are critical for understanding the users' feedback and changing intent.
7. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
8. Current Work: Describe in detail precisely what was being worked on immediately before this summary request, paying special attention to the most recent messages from both user and assistant. Include file names and code snippets where applicable.
9. Optional Next Step: List the next step that you will take that is related to the most recent work you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's most recent explicit requests, and the task you were working on immediately before this summary request. If your last task was concluded, then only list next steps if they are explicitly in line with the users request. Do not start on tangential requests or really old requests that were already completed without confirming with the user first.
   If there is a next step, include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off. This should be verbatim to ensure there's no drift in task interpretation.

Output structure:

<analysis>
[Your thought process, ensuring all points are covered thoroughly and accurately]
</analysis>

<summary>
1. Primary Request and Intent:
   [Detailed description]

2. Key Technical Concepts:
   - [Concept 1]
   - [Concept 2]
   - [...]

3. Files and Code Sections:
   - [File Name 1]
      - [Summary of why this file is important]
      - [Summary of the changes made to this file, if any]
      - [Important Code Snippet]
   - [File Name 2]
      - [Important Code Snippet]
   - [...]

4. Errors and fixes:
    - [Detailed description of error 1]:
      - [How you fixed the error]
      - [User feedback on the error if any]
    - [...]

5. Problem Solving:
   [Description of solved problems and ongoing troubleshooting]

6. All user messages:
    - [Detailed non tool use user message]
    - [...]

7. Pending Tasks:
   - [Task 1]
   - [Task 2]
   - [...]

8. Current Work:
   [Precise description of current work]

9. Optional Next Step:
   [Optional Next step to take]

</summary>

Please provide your summary based on the conversation so far, following this structure and ensuring precision and thoroughness in your response.

REMINDER: Do NOT call any tools. Respond with plain text only — an <analysis> block followed by a <summary> block. Tool calls will be rejected and you will fail the task."""

def extract_summary(llm_output: str) -> str:
    start = llm_output.find("<summary>")
    end = llm_output.find("</summary>")
    if start == -1 or end == -1:
        return llm_output
    return llm_output[start + len("<summary>"):end].strip()

def build_compact_messages(
    summary: str,
    attachment: str = "",
    has_keep_tail: bool = False,
    transcript_path: str = "",
) -> list[Message]:
    content = "本次会话延续自之前的对话，因上下文空间不足进行了压缩。以下是早期对话的摘要：\n\n" + summary
    if has_keep_tail:
        content += "\n\n近期消息已原样保留。"
    if transcript_path:
        content += f"\n\n如果你需要压缩前的具体细节（代码片段、报错信息等），请用 ReadFile 读取完整会话记录：{transcript_path}"
    if attachment:
        content += "\n\n---\n\n" + attachment
    return [
        Message(role="user", content=content),
    ]

"""
    压缩后恢复状态
"""

RECOVERY_FILE_LIMIT = 5
RECOVERY_TOKENS_PER_FILE = 5_000

@dataclass
class FileReadRecord:
    path: str
    content: str
    timestamp: float

class RecoveryState:
    """
    记录 ReadFile 返回的字节内容，这些记录被重新附加到摘要的 user 消息上
    模型仍然保有可用的工作上下文。
    """
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._files: dict[str, FileReadRecord] = {}

    def record_file_read(self, path: str, content: str) -> None:
        if not path:
            return
        with self._lock:
            self._files[path] = FileReadRecord(
                path=path, content=content, timestamp=time.time()
            )

    def snapshot_files(self, limit: int) -> list[FileReadRecord]:
        with self._lock:
            records = list(self._files.values())
        records.sort(key=lambda r: r.timestamp, reverse=True)
        if limit > 0:
            records = records[:limit]
        return records

def _approx_tokens(s: str) -> int:
    if not s:
        return 0
    return int(len(s) / _RECOVERY_CHARS_PER_TOKEN)

def _truncate_by_tokens(s: str, token_budget: int) -> str:
    if token_budget <= 0 or not s:
        return s
    if _approx_tokens(s) <= token_budget:
        return s
    max_chars = int(token_budget * _RECOVERY_CHARS_PER_TOKEN)
    if max_chars <= 0 or max_chars >= len(s):
        return s
    return s[:max_chars] + "\n… (内容已截断)"

def _first_line(s: str) -> str:
    for line in s.split("\n"):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""

def build_recovery_attachment(
    state: RecoveryState | None,
    tool_schemas: list[Mapping[str, Any]] | None,
) -> str:
    """渲染压缩后附件的四个小节。
    `tool_schemas` 应当是 agent 在下一次请求中将要发送的 schema —— 这里用其中的
    名称和描述来提醒模型当前都接入了哪些工具。
    """
    sections: list[str] = []

    if state is not None:
        files = state.snapshot_files(RECOVERY_FILE_LIMIT)
        if files:
            buf = ["## 最近读过的文件\n",
                   "以下快照是文件读取工具上次返回的内容。如需当前字节请重新读取。\n"]
            for rec in files:
                content = _truncate_by_tokens(rec.content, RECOVERY_TOKENS_PER_FILE)
                ts = time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(rec.timestamp)
                )
                buf.append(f"### {rec.path}  (read {ts})\n")
                buf.append("```\n")
                buf.append(content)
                if not content.endswith("\n"):
                    buf.append("\n")
                buf.append("```\n")
            sections.append("".join(buf))

    if tool_schemas:
        buf = ["## 可用工具\n",
               "你仍然可以调用以下工具，需要时直接发起调用即可：\n"]
        for t in tool_schemas:
            name = t.get("name") if isinstance(t, Mapping) else None
            if not name:
                continue
            desc = t.get("description", "") if isinstance(t, Mapping) else ""
            desc = _first_line(desc or "")
            if desc:
                buf.append(f"- {name} — {desc}\n")
            else:
                buf.append(f"- {name}\n")
        sections.append("".join(buf))

    if not sections:
        return ""

    sections.append(
        "## 提示\n\n以上恢复的上下文是重建的。若需要原文代码、错误信息或用户原话，"
        "请用文件读取工具重新读取，不要根据摘要猜测细节。\n"
    )
    return "\n".join(sections)

def _group_messages_by_turn(messages: list[Message]) -> list[list[Message]]:
    groups: list[list[Message]] = []
    current: list[Message] = []
    for msg in messages:
        current.append(msg)
        if msg.role == "assistant" and not msg.tool_uses:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups

def _message_tokens(msg: Message) -> int:
    """估算单条消息的 token 数，复用共享的字符数启发式算法。"""
    return estimate_tokens([msg])


def _compute_keep_start_index(messages: list[Message]) -> int:
    """
        决定压缩时尾部要原样保留多少条消息。从尾部向头部遍历 `messages`，逐条累加 token 估算值。
        确保被保留的 tool_result 不会和它对应的 tool_use 被拆散—— `_align_keep_start_to_tool_pair`。
    """
    n = len(messages)
    if n == 0:
        return 0

    kept_tokens = 0
    kept_count = 0
    keep_start = n  # 尚未保留任何消息

    for i in range(n - 1, -1, -1):
        tok = _message_tokens(messages[i])

        # 在已经保留了至少一条消息的前提下，如果纳入当前消息会突破硬上限则停止
        # （但绝不拒绝保留最后一条消息，即使它单独就超限）。
        if kept_count > 0 and kept_tokens + tok > KEEP_MAX_TOKENS:
            break

        kept_tokens += tok
        kept_count += 1
        keep_start = i

        # 保底条件已满足（token 下限或消息条数下限达到其一）：
        # 近期原文保留足够了，停止回溯。
        if kept_tokens >= KEEP_RECENT_TOKENS or kept_count >= MIN_KEEP_MESSAGES:
            break

    return _align_keep_start_to_tool_pair(messages, keep_start)

def _align_keep_start_to_tool_pair(messages: list[Message], keep_start: int) -> int:
    """
        确保携带 tool_results 的 user 消息和前面那条发起对应 tool_uses 的 assistant
        消息配成一对。如果 keep_start 正好落在这样一条 user 消息上，就把它往前回退到
        （至少）配对的那条 assistant 消息，让 tool_use 与 tool_result 的配对关系保持完整。
    """
    while 0 < keep_start < len(messages):
        msg = messages[keep_start]
        if msg.role == "user" and msg.tool_results:
            prev = messages[keep_start - 1]
            if prev.role == "assistant" and prev.tool_uses:
                keep_start -= 1
                continue
        break
    return keep_start

def _prefix_too_small_to_compact(prefix: list[Message]) -> bool:
    """当摘要 `prefix` 能回收的空间太少、不值得做时返回 True。"""
    if not prefix:
        return True
    return estimate_tokens(prefix) < MIN_SUMMARIZE_PREFIX_TOKENS

# ----------------------------熔断器----------------------------

@dataclass
class CompactBreaker:
    max_failures: int = 3
    consecutive_failures: int = field(default=0, init=False)

    def record_failure(self) -> None:
        self.consecutive_failures += 1

    def record_success(self) -> None:
        self.consecutive_failures = 0

    def is_open(self) -> bool:
        return self.consecutive_failures >= self.max_failures


# -------------------------UsageAnchor—真实 API 用量锚点-------------------------

@dataclass
class UsageAnchor:
    """
    记录上一次真实 API 用量和当时的对话长度 :
    1. baseline_tokens 是 input + cache_read + cache_creation + output 的合计值
    2. anchor_count: 记录该数值时的 conversation.messages 长度
    3. has_usage 为 False 时表示尚未收到任何 API用量报告（冷启动），退化为对整个 messages 做字符估算
    """

    baseline_tokens: int = 0
    anchor_count: int = 0
    has_usage: bool = False

    @staticmethod
    def from_api_usage(
        input_tokens: int,
        output_tokens: int = 0,
        cache_read: int = 0,
        cache_creation: int = 0,
        msg_count: int = 0,
    ) -> UsageAnchor:
        """根据一次 API 响应构造token用量锚点。"""
        return UsageAnchor(
            baseline_tokens=input_tokens + cache_read + cache_creation + output_tokens,
            anchor_count=msg_count,
            has_usage=True,
        )

# --------------------------------Auto-compact--------------------------------

async def auto_compact(
    conversation: Conversation,
    client: Any,
    context_window: int,
    session_dir: Path,
    protocol: str = "anthropic",
    manual: bool = False,
    breaker: CompactBreaker | None = None,
    recovery: RecoveryState | None = None,
    tool_schemas: list[Mapping[str, Any]] | None = None,
    transcript_path: str = "",
    budget_messages: list[Message] | None = None,
) -> CompactEvent | str | None:
    # （input + cache_read + cache_creation + output）加上锚点之后新增消息的字符估算
    # 冷启动或刚压缩清空锚点时退化为对整个 history 做字符估算
    current = conversation.current_tokens()

    if manual:
        pass # 手动压缩（/compact）不检查阈值
    else:
        # 双阈值判断：
        # 1) 软触发线（auto margin 13K）：低于此线不需要压缩
        soft_threshold = compute_compact_threshold(context_window, manual=False)
        if current < soft_threshold:
            return None

        # 2) 硬触发线（manual margin 3K）：超过此线强制压缩，绕过熔断器，
        #    因为上下文已经过于接近窗口上限，不能冒跳过的风险
        hard_threshold = compute_compact_threshold(context_window, manual=True)
        if current >= hard_threshold:
            # 强制压缩路径：不检查熔断器
            pass
        else:
            # 处于软硬阈值之间：走正常的熔断器保护逻辑
            if breaker is not None and breaker.is_open():
                return "自动压缩已连续失败 3 次，请手动处理或使用 /compact"

    before_tokens = current

    effective_history = budget_messages if budget_messages else conversation.messages

    # 决定保留多少尾部消息原文。只有前缀 messages[:keep_start] 会被摘要；
    # messages[keep_start:] 原样保留，让模型看到近期原文而非靠有损摘要复述。
    keep_start = _compute_keep_start_index(effective_history)
    to_summarize = effective_history[:keep_start]
    keep_tail = effective_history[keep_start:]

    # 待摘要的前缀太小时不压缩——要么全部消息都落在保留窗口内
    # （keep_start <= 0），要么摘要回收的 token 还不够摘要本身的开销。
    if keep_start <= 0 or _prefix_too_small_to_compact(to_summarize):
        return None

    # messages_for_summary = construct_messages(list(to_summarize), protocol)
    #
    # summary_messages: list[dict[str, Any]] = [
    #     {"role": "user", "content": SUMMARY_PROMPT},
    # ]
    # summary_messages.extend(messages_for_summary)
    # summary_messages.append(
    #     {"role": "user", "content": "Please provide your summary of the conversation above now. REMINDER: Do NOT call any tools — respond with plain text only."}
    # )

    summary_conv = Conversation()
    summary_conv.messages = [
        Message(role="user", content=SUMMARY_PROMPT),
    ]
    # 只摘要前缀；保留的尾部在下面重建时原样拼回。
    for msg in to_summarize:
        summary_conv.messages.append(msg)
    summary_conv.messages.append(
        Message(role="user", content="Please provide your summary of the conversation above now. REMINDER: Do NOT call any tools — respond with plain text only.")
    )

    max_retries = 3
    llm_output: str | None = None

    for attempt in range(max_retries):
        try:
            from aicocode.base import StreamEnd, StreamEvent, TextDelta

            collected_text = ""
            async for event in client.stream(summary_conv, system=SUMMARY_PROMPT, tools=tool_schemas):
                if isinstance(event, TextDelta):
                    collected_text += event.text
                elif isinstance(event, StreamEnd):
                    pass
            llm_output = collected_text
            break

        except Exception as e:
            err_msg = str(e).lower()
            if "prompt" in err_msg and "long" in err_msg or "too many" in err_msg:
                groups = _group_messages_by_turn(summary_conv.messages[1:-1])
                drop_count = max(1, len(groups) // 5)
                remaining = groups[drop_count:]
                summary_conv.messages = (
                    [summary_conv.messages[0]]
                    + [m for g in remaining for m in g]
                    + [summary_conv.messages[-1]]
                )
                continue
            if breaker is not None:
                breaker.record_failure()
            return f"摘要生成失败: {e}"

    if llm_output is None:
        if breaker is not None:
            breaker.record_failure()
        return "摘要生成失败：多次重试后仍超出上下文限制"

    summary = extract_summary(llm_output)
    attachment = build_recovery_attachment(recovery, tool_schemas)
    # 重建 = 摘要(user) + 尾部原文。
    new_messages = build_compact_messages(
        summary,
        attachment=attachment,
        has_keep_tail=bool(keep_tail),
        transcript_path=transcript_path,
    )
    new_messages = new_messages + list(keep_tail)

    conversation.replace_messages(new_messages)
    cleanup_tool_results(session_dir)

    if breaker is not None:
        breaker.record_success()

    return CompactEvent(
        before_tokens=before_tokens,
        boundary=CompactBoundary(summary=summary, keep=list(keep_tail)),
    )
