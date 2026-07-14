"""内置 system prompt 与启动 ASCII banner。"""

from __future__ import annotations

ASCII_BANNER = r"""
   _\_/_
  [ o.o ]
  |__v__|
"""

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
