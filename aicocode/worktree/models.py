from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Worktree:
    name: str   # Worktree 的名称
    path: str   # Worktree 在本地磁盘上的具体文件路径
    branch: str # 当前 Worktree 绑定的 Git 分支名
    based_on: str   # 该 Worktree 创建时所基于的基础分支或基准 Commit（如 main 或 v1.0.0）
    head_commit: str    # 当前 HEAD 所指向的具体 Commit SHA-1 散列值
    created: datetime = field(default_factory=datetime.now)


@dataclass
class WorktreeSession:
    original_cwd: str
    worktree_path: str
    worktree_name: str
    original_branch: str
    original_head_commit: str
    session_id: str = ""
    hook_based: bool = False