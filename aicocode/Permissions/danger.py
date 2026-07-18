from __future__ import annotations

import re

_DANGEROUS_CMD_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"rm\s+-[a-z]*r[a-z]*f[a-z]*\s+/\s*$"), "递归强制删除根目录"),
    (re.compile(r"rm\s+-[a-z]*r[a-z]*f[a-z]*\s+(\/|~)"), "递归强制删除根目录或用户主目录"),
    (re.compile(r"mkfs\."), "格式化磁盘"),
    (re.compile(r"dd\s+if=.*of=/dev/"), "直接写磁盘设备"),
    (re.compile(r"chmod\s+-R\s+777\s+/"), "递归修改根目录权限"),
    (re.compile(r"(chmod|chown)\s+-R\s+\S+\s+\/"), "递归修改根目录权限或所有者"),
    (re.compile(r":\(\)\{\s*:\|:&\s*\};:"), "fork bomb"),
    (re.compile(r"(:\(\)\{\s*:\|:&\s*\};:|python.*-c.*while\s+True:.*os\.fork|perl\s+-e.*fork\(\))"), "Fork Bomb (Bash/Python/Perl等)"),
    (re.compile(r"curl\s+.*\|\s*(ba)?sh"), "管道执行远程脚本"),
    (re.compile(r"wget\s+.*\|\s*(ba)?sh"), "管道执行远程脚本"),
    (re.compile(r"(kill|killall)\s+-9\s+(-1|1|\w+)"), "强制终止所有或关键系统进程"),
    (re.compile(r"(shutdown|poweroff|reboot|halt|init\s+0)\b"), "系统关机或重启"),
    (re.compile(r"umount\s+(-a|/)\s*$"), "卸载根目录或所有挂载点"),
    (re.compile(r">\s*(/dev/(sd|nvme|hd|vd|mem|kmem)|/etc/(passwd|shadow|fstab|sudoers|group)|/boot/(vmlinuz|initrd))"), "覆盖关键系统设备或核心配置文件"),
]

# 1. 移除了 xargs, find, awk, sed, tee, npx, env, dir 等本身可执行代码或修改文件的命令
_SAFE_COMMANDS = frozenset({
    "ls", "pwd", "echo", "cat", "head", "tail", "wc",
    "which", "whereis", "whoami", "hostname", "uname",
    "date", "cal", "uptime", "df", "du", "free", "printenv",
    "file", "stat", "readlink", "realpath", "basename", "dirname",
    "sort", "uniq", "tr", "cut", "grep", "egrep", "fgrep",
    "diff", "comm", "true", "false", "test",
    "go version", "go env",
    "node -v", "npm -v", "python --version", "python3 --version", "pip list", "pip3 list",
    "cargo --version", "rustc --version", "java -version", "java --version",
})
# 2. 提取安全的 git 子命令，并单独处理
_SAFE_GIT_SUBCMDS = frozenset({
    "status", "log", "diff", "show", "branch", "tag", "remote",
    "rev-parse", "ls-files", "blame", "stash"
})
# 3. 拦截所有可能的 Shell 元字符、控制结构、变量替换和换行符
_DANGEROUS_PATTERNS = re.compile(
    r"[|;&><`]|\$\(|\$\{|\$IFS|\$\w+|\\|\n|\r|\b(if|for|while|case|do|done|then|fi|esac|function)\b"
)


def is_safe_command(command: str) -> bool:
    trimmed = command.strip()
    if not trimmed:
        return False
    # 4. 拦截所有危险的特殊字符和 Shell 控制结构
    if _DANGEROUS_PATTERNS.search(trimmed):
        return False
    parts = trimmed.split()
    if not parts:
        return False
    cmd = parts[0]
    # 5. 严格校验 git 命令，拦截破坏性操作
    if cmd == "git":
        if len(parts) < 2 or parts[1] not in _SAFE_GIT_SUBCMDS:
            return False
        # 处理 git stash list 特例
        if parts[1] == "stash":
            return len(parts) == 3 and parts[2] == "list"
        # 拦截删除分支、删除标签、移动等破坏性操作
        if parts[1] in ("branch", "tag"):
            for arg in parts[2:]:
                if arg in ("-d", "-D", "--delete", "-m", "-M"):
                    return False
        return True
    # 6. 严格校验两词命令（如版本号查询），拒绝追加任何额外参数以防注入
    if len(parts) >= 2 and f"{parts[0]} {parts[1]}" in _SAFE_COMMANDS:
        # 如果是版本查询或列表命令，强制参数长度为 2
        if "--version" in parts[1] or "-v" in parts[1] or parts[1] == "list":
            return len(parts) == 2
        return True
    # 7. 校验单词命令，并对特定命令做参数限制
    if cmd in _SAFE_COMMANDS:
        # 拦截 sort 命令的 -o (覆盖写文件) 参数
        if cmd == "sort" and any(arg in ("-o", "--output") for arg in parts[1:]):
            return False
        return True
    return False

class DangerousCommandDetector:
    def __init__(self, extra_patterns: list[tuple[str, str]] | None = None) -> None:
        self._patterns = list(_DANGEROUS_CMD_PATTERNS)
        if extra_patterns:
            for regex_str, reason in extra_patterns:
                self._patterns.append((re.compile(regex_str), reason))


    def detect(self, command: str) -> tuple[bool, str]:
        for pattern, reason in self._patterns:
            if pattern.search(command):
                return True, reason
        return False, ""