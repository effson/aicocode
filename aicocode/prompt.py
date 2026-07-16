"""内置 system prompt 与启动 ASCII banner。"""

from __future__ import annotations
from datetime import datetime
import platform

SYSTEM_PROMPT = """\
You are AicoCode, a concise command-line AI assistant running in the user's terminal.
You help with software engineering tasks including writing code, debugging, refactoring, explaining code, and running commands.
IMPORTANT: Be careful not to introduce security vulnerabilities such as command injection, XSS, SQL injection, and other common vulnerabilities.
Prioritize writing safe, secure, and correct code.
IMPORTANT: You must NEVER generate or guess URLs unless you are confident they help the user with programming. You may use URLs provided by the user.\
"""


def render_banner(version: str, cwd: str) -> str:
    """拼出启动 banner：ASCII 机器人 + 应用名与版本 + 工作目录 + 就绪提示行。"""
    lines = [
        ASCII_BANNER.strip("\n"),
        f"AicoCode v{version}",
        f"cwd: {cwd}",
        "Ready. Type a message and press Enter to send (Alt+Enter for a new line, /exit to quit).",
    ]
    return "\n".join(lines)

def build_environment_context(
    work_dir: str,
) -> str:
    parts = [
        f"Current working directory: {work_dir}",
        f"Operating system: {platform.system()} {platform.release()}",
        f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]

    # active_skills 不再注入 env context —— Skill 内容作为普通消息注入一次到对话历史，
    # 随对话自然推远，auto-compact 时会被摘要。

    return "\n".join(parts)