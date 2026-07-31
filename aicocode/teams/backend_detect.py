from __future__ import annotations

import os
import sys

from aicocode.teams.models import BackendType


class BackendDetectionError(Exception):
    pass


def detect_backend_from_env() -> BackendType:
    """
    只按环境变量判断后端，抽出来便于单测（不受运行平台影响）
    """
    if os.environ.get("TMUX"):
        return BackendType.TMUX
    if os.environ.get("ITERM_SESSION_ID"):
        return BackendType.ITERM2
    return BackendType.IN_PROCESS


def detect_backend(
    teammate_mode: str = "",
    is_interactive: bool = True,
) -> BackendType:
    """
    选择 teammate 后端。
    """
    if teammate_mode == "in-process" or not is_interactive:
        return BackendType.IN_PROCESS
    if sys.platform == "win32":
        return BackendType.IN_PROCESS
    return detect_backend_from_env()


def detect_pane_backend(
    teammate_mode: str = "",
    is_interactive: bool = True,
) -> BackendType:
    """
    检测窗格后端
    """
    if teammate_mode == "in-process" or not is_interactive:
        return BackendType.IN_PROCESS
    if sys.platform == "win32":
        return BackendType.IN_PROCESS
    return detect_backend_from_env()