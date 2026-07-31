from __future__ import annotations

import platform
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SandboxConfig:
    """沙箱配置：控制可写路径、禁写路径和网络访问。"""
    # 允许写入的路径列表（白名单）
    allow_write: list[str] = field(default_factory=list)
    # 强制只读的路径列表（优先级高于 allow_write）
    deny_write: list[str] = field(default_factory=list)
    # 是否允许网络访问
    network_enabled: bool = False


class Sandbox(ABC):
    """沙箱抽象基类，各平台实现 wrap() 和 available()。"""
    @abstractmethod
    def wrap(self, command: str, config: SandboxConfig) -> str:
        """将原始命令包装为沙箱内执行的命令字符串。"""
        ...

    @abstractmethod
    def available(self) -> bool:
        """检测当前环境是否支持该沙箱。"""
        ...


def create_sandbox() -> Sandbox | None:
    """根据操作系统自动选择沙箱实现。

    macOS -> SeatbeltSandbox（基于 sandbox-exec）
    Linux -> BwrapSandbox（基于 bubblewrap）
    其他系统 -> None（不支持沙箱）
    """
    system = platform.system()
    if system == "Darwin":
        from .seatbelt import SeatbeltSandbox
        return SeatbeltSandbox()
    elif system == "Linux":
        from .bwrap import BwrapSandbox
        return BwrapSandbox()
    return None