# AicoCode

> 终端 AI 编程助手 — A terminal AI coding assistant powered by LLMs.

AicoCode 是一个运行在终端中的 AI 编程助手，基于 [Textual](https://textual.textualize.io/) TUI 框架构建。支持 Anthropic、OpenAI 及OpenAI兼容协议的 LLM 提供商，提供 Agent Team 协作、MCP 协议集成、权限沙箱等特性。

---

## 快速开始

### 环境要求

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)（推荐）或 pip

### 安装

```bash
git clone https://github.com/effson/aicocode.git
cd aicocode

# uv
uv sync

# pip
pip install -e .
```

### 配置

```bash
cp config.example.yaml .aicocode/config.yaml
# 编辑 .aicocode/config.yaml，填入 API Key
```

配置文件按优先级合并加载（后者覆盖前者）：

| 优先级 | 路径 | 用途 |
|--------|------|------|
| 1 | `~/.aicocode/config.yaml` | 用户级全局配置 |
| 2 | `./.aicocode/config.yaml` | 项目级配置 |
| 3 | `./.aicocode/config.local.yaml` | 本地覆盖（gitignore） |

**配置示例：**

```yaml
providers:
  # Anthropic Claude（官方，x-api-key 鉴权）
  - name: claude
    protocol: anthropic
    model: claude-sonnet-5
    api_key: ${ANTHROPIC_API_KEY}

  # OpenAI（官方或兼容端点）
  - name: openai
    protocol: openai
    model: gpt-4o
    api_key: ${OPENAI_API_KEY}

  # 智谱 GLM（OpenAI兼容）
  - name: glm
    protocol: openai-compatible
    model: glm-5.2
    api_key: your-zhipu-api-key
    base_url: https://open.bigmodel.cn/api/paas/v4
  
permission_mode: default
  
mcp_servers:
  - name: context7
    command: npx
    args: ["-y", "@upstash/context7-mcp"]
  - name: sequential-thinking
    command: npx
    args: ["-y", "@modelcontextprotocol/server-sequential-thinking"]
    
hooks:
  - event: post_tool_use	# 工具执行之后事件
  	if: tool == "WriteFile" # 在写文件时触发
  	once: false
  	action:
  	  type: command
  	  command: "you command"

enable_fork: false # 是否允许 fork 出子agent，该子agent具有父的上下文
enable_coordinator_mode: false   # 设为 true 后创建 Team 时 Lead 只能调度不能写代码
```

**配置说明**

- `protocol` 可选：`anthropic` | `openai` | `openai-compatible`

​	API Key 支持 `${ENV_VAR}` 语法从环境变量读取。`auth_token: true` 将鉴权头从 `x-api-key` 切换为 `Authorization: Bearer`，	用于 Anthropic 兼容端点。

- `permission_mode`可选：`default` | `acceptEdits` | `plan`| `bypassPermissions`

- `hooks`. `event`可选: `session_start` | `session_end` | `turn_start`| `turn_end`等，详见`aicocode/hooks/event.py`

  `hooks`.`once`可选：`false`|`true`,  true表示只执行一次

  `hooks`.`action`.`type`可选：`command`|`prompt`|`http`|`agent`,  type为  `command`必有`hooks`.`action`.`command`, type为  `prompt`必有`hooks`.`action`.`message`, type为  `http`必有`hooks`.`action`.`url`,  type为  `agent`必有`hooks`.`action`.`prompt`

- `enable_fork`可选：`false`|`true`

### 启动

```bash
# 交互式 TUI 模式
aicocode

# 非交互 Prompt 模式（直接输出结果）
aicocode -p "用 Python 写一个快速排序函数"

# NDJSON 流式输出（管道 / 脚本集成）
aicocode -p "列出所有 TODO" --output-format stream-json

# 覆盖权限模式,默认default
aicocode --mode bypass
aicocode --mode plan
```

---

## 架构总览

```
┌─────────────────────────────────────────────────────────┐
│                    __main__.py (CLI)                     │
│          TUI 模式 │ -p Prompt 模式 │ --teammate Worker   │
└────────┬──────────────────┬──────────────────┬──────────┘
         ▼                  ▼                  ▼
┌────────────────┐  ┌──────────────┐  ┌──────────────────┐
│    app.py      │  │ _run_prompt  │  │ _run_teammate    │
│  (Textual App) │  │ (事件循环)    │  │ (Worker 进程)     │
└───────┬────────┘  └──────┬───────┘  └────────┬─────────┘
        │                  │                    │
        └──────────────────┼────────────────────┘
                           ▼
                ┌─────────────────────┐
                │      Agent          │  ← agent.py
                │  (核心 Agent 循环)   │
                └────────┬────────────┘
                         │
      ┌──────────────────┼──────────────────┐
      ▼                  ▼                  ▼
 ┌─────────┐  ┌──────────────────┐  ┌─────────────┐
 │ LLM     │  │   ToolRegistry   │  │ Permission  │
 │ Client  │  │   (builtin tools │  │ Validator   │
 │         │  │   & mcp tools)   │  │             │
 └─────────┘  └──────────────────┘  └─────────────┘
```

**数据流**：用户输入 → `Conversation`（消息管理）→ `Agent.run()` 流式循环 → `LLMClient.stream()` → 解析 `StreamEvent` → 执行工具 / 输出文本 → 下一轮直到 `LoopComplete`

---

## Agent 核心循环

`Agent` 类是整个系统的中枢，管理完整的 Agent 生命周期：

- **流式对话循环** (`run`)：发送对话 → 接收 SSE 事件流 → 文本/思考输出 → 工具调用 → 执行工具 → 结果注入 → 下一轮
- **工具执行** (`_execute_single_tool_direct`)：查找工具 → 权限校验 → 调用工具 → 收集结果
- **自动压缩**：接近 context window 上限时，触发 `auto_compact` 将早期轮次总结为摘要
- **记忆提取**：每 N 轮自动提取用户偏好和项目上下文到 `.aicocode/memory/`
- **Plan 模式**：注入 Plan Mode 提示词（5 阶段工作流），限制为只读工具
- **Coordinator 模式**：收窄工具集为纯委托工具
- **Hook 集成**：在 `turn_start`、`turn_end`、`pre_tool_use`、`post_tool_use` 等事件点触发 Hook
- **Skill 目录**：向 System Prompt 注入可用 Skill 列表
- **Agent 目录**：向 System Prompt 注入可用 Agent 列表

关键配置：`context_window`（token 预算）、`max_iterations`（最大轮次）、`permission_validator`（权限校验器）

###  LLM 客户端

根据配置的`provider`.`protocal`生成对应客户端，发送对应api 格式的消息:

| 客户端 | 协议 | 关键特性 |
|--------|------|----------|
| `AnthropicClient` | Anthropic Messages API | 扩展思考（extended thinking）、自适应 thinking 检测、prompt cache |
| `OpenAIClient` | OpenAI Responses API | 推理摘要（reasoning summary）、function call、缓存 token 扣减 |
| `OpenAICompatibleClient` | OpenAI Chat Completions API | 广泛兼容 vLLM / Ollama / Together / Azure 等，DeepSeek 式 reasoning_content |

**上下文窗口自动检测**（`resolve_context_window`）：从 `/v1/models` 拉取真实 context window → 缓存到 `ProviderConfig._fetched_context_window`，配合四层 fallback：显式配置 > API 拉取 > 内置映射表 > 保守默认值。

**多协议消息适配**：将上层 `Conversation` 转换为各 LLM 协议的消息格式。

### System Prompt 构建

`PromptConstructer` 构建器模式，组合以下 `PromptPart`（按 priority 排序）：

| Priority | Part | 职责 |
|----------|------|------|
| 0 | ROLE_DEFINITION | 角色定位 + 安全红线 |
| 10 | SYSTEM | 系统规则（工具执行、权限、summarization） |
| 20 | DOING_TASKS | 任务执行策略（理解需求、验证结果、避免过度设计） |
| 30 | EXECUTING_ACTIONS | 操作风险评估（可逆/不可逆/对外可见） |
| 40 | USING_TOOLS | 工具使用规范（专用工具优先、并行调用、Agent 委托） |
| 50 | TONE_STYLE | 输出风格（简洁、无 emoji、文件引用格式） |
| 60 | TEXT_OUTPUT | 文本输出规范（阶段性更新、不做流水账） |
| 70 | ENVIRONMENT | 运行时环境上下文（OS、Shell、Git 分支、工作目录、日期） |
| 80 | CUSTOM | 用户自定义指令（AICOCODE.md / AGENTS.md） |

Plan 模式提示词（`build_plan_mode_reminder`）按 5 阶段工作流注入：理解 → 设计 → 审查 → 最终方案 → 退出 Plan 模式。

---

## 工具系统

### ToolRegistry

工具注册表（`tools/__init__.py`），管理工具的生命周期：

- **注册与发现**：`register_tool()` / `get_tool()` / `list_tools()`
- **启用/禁用**：`enable_tool()` / `disable_tool()` / `tool_is_enabled()`
- **延迟加载**（核心优化）：标记 `should_defer=True` 的工具不会在初始 tool list 中暴露 schema，减少 context window 占用。模型通过 `ToolSearch` 按关键词搜索并按需发现
- **协议适配**：`get_all_schemas()` 根据 `protocol` 参数输出 Anthropic 或 OpenAI 格式的 tool schema
- **搜索**：`search_deferred_tools()` 按名称/描述的相关性评分排序

`create_default_registry()` 注册 6 个基础工具：ReadFile、EditFile、WriteFile、Bash、Glob、Grep

### 工具清单

#### 文件操作
| 工具 | 文件 | 说明 |
|------|------|------|
| `ReadFile` | `read_file.py` | 读取文件内容，支持分页和行范围 |
| `WriteFile` | `write_file.py` | 创建或覆写文件 |
| `EditFile` | `edit_file.py` | 精确字符串替换编辑，依赖 `edit_diff.py` 生成 diff |
| `FileStateCache` | `file_state_cache.py` | 跟踪文件修改状态，为 Edit 提供上下文 |

#### 代码搜索
| 工具 | 文件 | 说明 |
|------|------|------|
| `Grep` | `grep.py` | 正则内容搜索（基于 ripgrep），支持 glob 过滤、上下文行 |
| `Glob` | `glob.py` | 文件模式匹配，按修改时间排序 |

#### 命令执行
| 工具 | 文件 | 说明 |
|------|------|------|
| `Bash` | `bash.py` | Shell 命令执行，支持后台运行、超时控制、沙箱包装 |

#### Agent 与任务
| 工具 | 文件 | 说明 |
|------|------|------|
| `AgentTool` | `agent_tool.py` | 派生子 Agent，支持 fork（继承对话）、worktree 隔离、Team 成员创建 |
| `TaskCreate` | `task_create.py` | 创建任务 |
| `TaskGet` | `task_get.py` | 查询任务详情 |
| `TaskList` | `task_list.py` | 列出所有任务 |
| `TaskUpdate` | `task_update.py` | 更新任务状态/字段 |
| `TaskStop` | `task_stop.py` | 停止运行中的后台任务 |

#### 团队协作
| 工具 | 文件 | 说明 |
|------|------|------|
| `SendMessage` | `send_message.py` | Agent 间消息传递（通过 Mailbox） |
| `TeamCreate` | `team_create.py` | 创建 Agent Team，生成 team 目录和共享任务板 |
| `TeamDelete` | `team_delete.py` | 删除 Team 并清理资源 |

#### Worktree 隔离
| 工具 | 文件 | 说明 |
|------|------|------|
| `EnterWorktree` | `enter_worktree.py` | 进入独立的 git worktree 工作区 |
| `ExitWorktree` | `exit_worktree.py` | 退出并清理 worktree |

#### Skill 系统
| 工具 | 文件 | 说明 |
|------|------|------|
| `LoadSkill` | `load_skill.py` | 加载 Skill 的完整指令到上下文 |
| `InstallSkill` | `install_skill.py` | 安装第三方 Skill |

#### MCP 扩展
| 工具 | 文件 | 说明 |
|------|------|------|
| `ToolSearch` | `impl/tool_search.py` | 搜索延迟加载的工具（MCP 工具 + deferred tools） |

#### UI 交互
| 工具 | 文件 | 说明 |
|------|------|------|
| `AskUser` | `ask_user.py` | 向用户发起询问 |
| `ExitPlanMode` | `exit_plan_mode.py` | 退出 Plan 模式，提交计划供用户审批 |
| `SyntheticOutput` | `synthetic_output.py` | 生成结构化输出（供 Workflow 脚本使用） |

---

## Agent 系统 (`agents/`)

### Agent 定义格式

在 `.aicocode/agents/` 或 `~/.aicocode/agents/` 下创建 `.md` 文件，YAML frontmatter 配置 + Markdown 正文作为 System Prompt：

```markdown
---
name: my-agent
description: 自定义 Agent 的用途
disallowedTools: [EditFile, WriteFile]
model: claude-xxx
maxTurns: 30
background: true
---

你是专注于 XXX 的专家。你的工作流程：...
```

加载优先级：项目级 (`.aicocode/agents/`) > 用户级 (`~/.aicocode/agents/`) > 内置 Agent

### 内置 Agent

| Agent | 文件 | 工具限制 | 模型 | 说明 |
|-------|------|----------|------|------|
| **general-purpose** | `general-purpose.md` | 无限制 | 继承父 Agent | 全能力子 Agent，适合需要完整工具集的任务 |
| **Explore** | `explore.md` | 禁用 Write/Edit | `glm-4.6` | 只读搜索专家，不限轮次，快速遍历代码库 |
| **Plan** | `plan.md` | 禁用 Write/Edit/Agent | 继承父 Agent | 只读架构师，最多 15 轮，产出实现方案 |
| **Verification** | `verification.md` | 禁用 Write/Edit/Agent | 继承父 Agent | 后台验证专家，运行测试/lint 验证实现 |

---

## Agent Team 系统

### 架构

```
Lead Agent ──TeamCreate──→ Team (目录 + config.json)
    │                         │
    │                         ├── Mailbox (消息队列)
    │                         ├── SharedTaskBoard (tasks.json)
    │                         │
    │   ──Agent(team_name)──→ ├── Teammate 1 (in-process / tmux / iTerm2)
    │                         ├── Teammate 2
    │                         └── ...
    │
    └── SendMessage ←── Teammates ──→ Shared Tasks
```

### Backend 对比

| Backend | 隔离级别 | 适用场景 |
|---------|----------|----------|
| `in-process` | asyncio 并发 | 默认，开发/测试 |
| `tmux` | 独立进程 + 窗格 | 生产，需独立终端 |
| `iterm2` | 独立进程 + 标签页 | macOS 用户 |

---

## 权限系统 
AicoCode设置了多道防御

### Layer 1: 危险命令拦截
- 采用黑名单模式
- 识别高风险 Shell 模式（`rm -rf /`、`git push --force` 等），直接拦截

### Layer 2: 路径沙箱（仅文件工具: ）
- 仅文件工具:`WriteFile`、`EditFile`、`ReadFile`
- 限制文件访问在项目根目录和系统临时目录内
- 超出项目目录需询问，防止读到敏感重要信息
- 解析符号链接，防止通过创建链接读取到危险敏感信息

### Layer 3: 权限规则匹配
可通过yaml文件设定规则：
- `~/.aicocode/permission.yaml`，全局配置，对所有项目生效
- 项目根目录`.aicocode/permission.yaml`，项目维护者设置
- `.aicocode/permission.loacl.yaml`,用户个人配置

```yaml
- permission: allow
  rule: TeamCreate(*)
- permission: allow
  rule: Agent(*)
- permission: allow
  rule: Bash(git *)   # 允许所有git命令
- permission: deny
  rule: Bash(git push --force)   # 禁止 force push
```
### Layer 4: 权限模式

| 模式 | Read 工具 | Write 工具 | Command 工具 | 切换方式 |
|------|-----------|------------|-------------|----------|
| `DEFAULT` | allow | query | query | 默认 |
| `ACCEPT_EDITS` | allow | allow | query | 文件编辑自动通过 |
| `PLAN` | allow | query | query | 计划模式，约束写入 |
| `BYPASS` | allow | allow | allow | 全部跳过 |

`Shift+Tab` 循环切换。

### Layer 5: Human In The Loop(HITL)
**前4层都无法决策，交给用户决定**

### Final Layer: OS Sandbox
- 让操作系统执行限制，macOS使用`seatbelt`，linux使用`bubblewrap`+`seccomp`
- 沙箱默认断网
- 敏感路径禁止改动，防止改重要信息

---

## MCP 集成

`MCPManager` 管理多个 MCP 服务器的生命周期：

- **连接管理**：加载配置 → 并行连接所有服务器
- **工具注册**：将 MCP 工具包装为 `MCPTool`（继承 `Tool`），注册到 `ToolRegistry`
- **延迟加载**：MCP 工具默认标记 `should_defer=True`，模型通过 `ToolSearch` 按需发现
- **指令注入**：从 `InitializeResult` 提取 `instructions`，注入 System Prompt
- **传输支持**：stdio（子进程）和 WebSocket 两种 MCP transport

`MCPClient` 封装单服务器连接，管理 MCP session 的生命周期。AicoCode管理多个MCP client。

---

## 上下文管理

### 工具结果压缩
**对每条新ToolResult：**
- 对每条新的 ToolResult，超出阈值（`50K`字符）的工具结果写入磁盘文件（`.aicocode/session/tool-results/{tool_use_id}.txt`）
- 工具结果替换为预览文本（前 2,000 字符 + 文件路径）

**多个结果超限：**
- 每个ToolResult都小于 `50K`，但有多个结果，总和大于`200K`字符
- 按 content 长度从大到小排序，从最长的开始逐个落盘
- 直到长度降到 `200K` 以下

### 全对话摘要 — Auto-Compact
- 触发时机：每轮 LLM 响应完成后，Agent 循环检查调用，超过一定阈值会执行压缩
- 阈值计算：
  -  `有效窗口` = context_window - 20,000（为摘要输出预留）
  - `软阈值` = `有效窗口` - 13,000，超过这个阈值触发压缩，多次(设为3次)失败后停止压缩
  - `硬阈值`= `有效窗口` - 3,000，超过这个阈值必须压缩，直到压缩成功
- 尾部保留
  - 保留 ≥ 5 条消息 AND 保留 token ≥ 10,000
  - 保留的消息有 40,000 tokens 硬上限
  - **配对保护**：确保留下的消息中，`tool_use` ↔ `tool_result` 的配对完整性。
- 生成摘要
  - 将早期轮次发送给 LLM 生成摘要
  - new_messages =  摘要消息（user role，含摘要 + 恢复附件 + 会话记录路径）+ 尾部原样保留的消息（keep_tail）

恢复附件包括：

| 文件 | 职责 |
|------|------|
| 最近读过的文件快照 | 最多 5 个文件，每个截断到 5,000 tokens  |
| 已激活的 Skill | 总预算 25,000 tokens，每个截断到 5,000 tokens |
| 可用工具列表  | 名称 + 首行描述  |

---

## 记忆系统

### 工作记忆
- 也就是上面的上下文，当前正在处理的上下文信息，容量有限，需要压缩管理

### 长期记忆
- 会话持久化：用户和agent对话都保留下来，可以恢复
- 项目指令文件：预先写好的项目知识和编码规范
- 自动记忆：Agent在对话中自动积累经验，用户的编码偏好，项目的技术实现细节，每轮Loop结束后自动整理、更新、删除记忆

### 记忆治理
- 距离上次整理超过24H
- 累积的会话是否大道5个
满足条件就触发记忆整理，删除重复的记忆，修正矛盾的


---

## Hook 系统

`HookEngine` 在 Agent 生命周期的关键事件点执行用户配置的操作：

**事件类型**：`startup` | `shutdown` | `session_start` | `session_end` | `turn_start` | `turn_end` | `pre_send` | `post_receive` | `pre_tool_use` | `post_tool_use`

**Action 类型**：
- `command` — 执行 Shell 命令
- `prompt` — 注入 LLM 提示词
- `http` — 发送 HTTP 请求
- `agent` — 触发 Agent 操作

---

## Skill 系统 (`skills/`)

Skill 是封装好的领域工作流（Markdown 文件），通过 slash command 调用或由模型自动检测激活。

- **加载链**：`SkillLoader` 扫描 project (`.aicocode/skills/`) > user (`~/.aicocode/skills/`) > builtins
- **安装**：`install.py` 从 GitHub 下载 Skill（支持 `skills.sh`、`github.com/tree/`、`raw` 三种 URL 格式），原子化安装到 `~/.aicocode/skills/`
- **执行**：支持 `fork` 模式和 `inline` 模式

---

## 命令系统 

### 命令列表

| 命令 | Handler |  说明 |
|------|---------|------|
| `/help` | `help.py` | 显示所有命令及说明 |
| `/compact` | `compact.py`  | 手动触发上下文压缩 |
| `/clear` | `clear.py`  | 清除对话历史 |
| `/plan` | `plan.py` | 进入/退出 Plan 模式 |
| `/session` | `session.py` | 保存/恢复/列出会话 |
| `/mcp` | `mcp.py`  | 查看 MCP 服务器连接状态 |
| `/memory` | `memory.py`  | 查看/管理记忆 |
| `/permission` | `permission.py`  | 管理权限规则 |
| `/rewind` | `rewind.py`  | 回退到之前的对话轮次 |
| `/status` | `status.py` | 查看 token 用量、模型等状态 |
| `/skill` | `skill.py`  | 调用指定 Skill |
| `/sandbox` | `sandbox.py`  | 管理 OS 沙箱设置 |
| `/tasks` | `tasks.py`  | 查看后台 Agent 任务 |
| `/trace` | `trace.py`  | 查看 Agent 执行轨迹 |
| `/worktree` | `worktree.py`  | 管理 git worktree |

---

## Worktree 隔离 (`worktree/`)

支持 git worktree 的创建、切换和清理， 并行开发提高效率。

---

## 项目文件树

```
aicocode/
├── __init__.py              # 版本信息 (__version__ = "0.1.0")
├── __main__.py              # CLI 入口: TUI / -p Prompt / --teammate Worker
├── app.py                   # Textual App (CodeApp) — 交互式 TUI
├── agent.py                 # Agent 核心循环 (流式、工具执行、compact、记忆)
├── base.py                  # LLM 流式事件基类 (TextDelta, ToolCallStart, ...)
├── agent_event.py           # Agent 层高级事件 (StreamText, LoopComplete, ...)
├── conversation.py          # 对话状态 (Conversation, Message, ToolUseBlock)
├── prompt.py                # System Prompt 构建器 (PromptConstructer, 8 个 Part)
├── llm_client.py            # LLM 客户端 (AnthropicClient, OpenAIClient)
├── message_adaptor.py       # 多协议消息格式转换
├── driver.py                # NoAltScreenDriver — scrollback 支持
├── config.py                # 配置模型 (ProviderConfig, AppConfig, ...)
├── config_validator.py      # YAML 校验 + 模型 context window 映射表
├── file_cache.py            # 文件内容缓存
├── permission_dialog.py     # TUI 权限确认对话框
├── askuser_dialog.py        # TUI AskUser 选择题对话框
│
├── agents/                  # Agent 定义与子 Agent 管理
│   ├── parser.py            #   AgentDef 解析器 (YAML frontmatter)
│   ├── loader.py            #   AgentLoader（三层扫描加载）
│   ├── fork.py              #   Fork 模式（继承对话历史）
│   ├── notification.py      #   通知模型
│   ├── trace.py             #   TraceManager（执行轨迹）
│   └── builtins/            #   内置 Agent 定义 (.md)
│       ├── general-purpose.md
│       ├── explore.md
│       ├── plan.md
│       └── verification.md
│
├── commands/                # 斜杠命令系统
│   ├── registry.py          #   CommandRegistry（注册/别名/分发）
│   ├── parser.py            #   命令参数解析器
│   ├── loader.py            #   动态命令发现
│   └── handlers/            #   命令处理器
│       ├── help.py          #   /help
│       ├── compact.py       #   /compact
│       ├── clear.py         #   /clear
│       ├── plan.py          #   /plan
│       ├── session.py       #   /session
│       ├── mcp.py           #   /mcp
│       ├── memory.py        #   /memory
│       ├── permission.py    #   /permission
│       ├── rewind.py        #   /rewind
│       ├── status.py        #   /status
│       ├── skill.py         #   /skill
│       ├── sandbox.py       #   /sandbox
│       ├── tasks.py         #   /tasks
│       ├── trace.py         #   /trace
│       └── worktree.py      #   /worktree
│
├── context/                 # 上下文管理
│   └── manager.py           #   auto_compact, 工具结果存盘, CompactBreaker
│
├── file_history/            # 文件修改历史
│   └── file_history.py      #   按会话追踪快照
│
├── hooks/                   # Hook 生命周期系统
│   ├── engine.py            #   HookEngine（匹配 + 执行）
│   ├── models.py            #   Hook, Action, HookContext
│   ├── conditions.py        #   ConditionGroup 条件引擎
│   ├── events.py            #   事件常量
│   ├── executors.py         #   Action 执行器分发
│   └── loader.py            #   YAML 配置加载
│
├── mcp/                     # Model Context Protocol 集成
│   ├── manager.py           #   MCPManager（多服务器管理）
│   ├── client.py            #   MCPClient（单服务器连接）
│   ├── tool_wrapper.py      #   MCPTool（工具适配包装）
│   └── __init__.py
│
├── memory/                  # 记忆与会话持久化
│   ├── auto_memory.py       #   MemoryManager（自动记忆提取）
│   ├── recall.py            #   记忆回召与注入
│   ├── auto_dream.py        #   后台记忆整合
│   ├── session.py           #   SessionManager（会话持久化）
│   └── instructions.py      #   指令文件加载 (AICOCODE.md, @include)
│
├── Permissions/             # 权限与安全
│   ├── permission_mode.py   #   PermissionMode 枚举 + 模式矩阵
│   ├── validator.py         #   PermissionValidator（多层串联）
│   ├── danger.py            #   危险命令检测
│   ├── path_sandbox.py      #   路径沙箱
│   └── rules.py             #   YAML 规则引擎
│
├── sandbox/                 # OS 级沙箱
│   ├── __init__.py          #   Sandbox 抽象 + create_sandbox()
│   ├── bwrap.py             #   Linux Bubblewrap 实现
│   └── seatbelt.py          #   macOS Seatbelt 实现
│
├── skills/                  # Skill 工作流系统
│   ├── parser.py            #   SkillDef 解析器
│   ├── loader.py            #   SkillLoader（三层扫描）
│   ├── executor.py          #   SkillExecutor（fork / inline）
│   └── install.py           #   Skill 安装器（GitHub 下载 + 原子化安装）
│
├── teams/                   # Agent Team 协作
│   ├── manager.py           #   TeamManager（创建/删除/加载）
│   ├── models.py            #   Team, TeammateInfo, BackendType
│   ├── mailbox.py           #   消息队列（Agent 间通信）
│   ├── shared_task.py       #   共享任务板 (tasks.json)
│   ├── registry.py          #   名字 → ID 映射
│   ├── spawn.py             #   统一 spawn 入口
│   ├── spawn_inprocess.py   #   in-process 后端
│   ├── spawn_tmux.py        #   tmux 后端
│   ├── spawn_iterm2.py      #   iTerm2 后端
│   ├── backend_detect.py    #   后端自动检测
│   ├── progress.py          #   实时进度 TUI
│   ├── coordinator.py       #   Coordinator 模式过滤器
│   └── protocol.py          #   通信协议
│
├── tools/                   # 工具系统
│   ├── __init__.py          #   ToolRegistry + create_default_registry()
│   ├── tool_base.py         #   Tool 基类, ToolResult, ToolCategory
│   ├── file_state_cache.py  #   文件状态缓存
│   ├── edit_diff.py         #   Diff 生成
│   ├── read_file.py         #   ReadFile
│   ├── write_file.py        #   WriteFile
│   ├── edit_file.py         #   EditFile
│   ├── bash.py              #   Bash
│   ├── glob.py              #   Glob
│   ├── grep.py              #   Grep
│   ├── ask_user.py          #   AskUser
│   ├── exit_plan_mode.py    #   ExitPlanMode
│   ├── agent_tool.py        #   Agent（子 Agent 派生）
│   ├── send_message.py      #   SendMessage（Agent 间通信）
│   ├── task_create.py       #   TaskCreate
│   ├── task_get.py          #   TaskGet
│   ├── task_list.py         #   TaskList
│   ├── task_update.py       #   TaskUpdate
│   ├── task_stop.py         #   TaskStop
│   ├── team_create.py       #   TeamCreate
│   ├── team_delete.py       #   TeamDelete
│   ├── load_skill.py        #   LoadSkill
│   ├── install_skill.py     #   InstallSkill
│   ├── enter_worktree.py    #   EnterWorktree
│   ├── exit_worktree.py     #   ExitWorktree
│   ├── synthetic_output.py  #   SyntheticOutput
│   └── impl/
│       └── tool_search.py   #   ToolSearch
│
└── worktree/                # Git Worktree 隔离
    ├── models.py            #   WorktreeInfo
    ├── slug.py              #   名称生成
    ├── setup.py             #   创建 + 符号链接
    ├── changes.py           #   变更检测
    ├── cleanup.py           #   清理
    ├── session.py           #   会话持久化
    └── intergration.py      #   系统集成
```

---

## 开发

```bash
# 安装开发依赖
uv sync --group dev

# 运行测试
uv run pytest

# 代码检查 (ruff: E, F, W, I, UP, B)
uv run ruff check .
```

---

## 技术栈

| 依赖 | 版本 | 用途 |
|------|------|------|
| [Textual](https://textual.textualize.io/) | >= 2.1.0 | 终端 UI 框架 |
| anthropic | >= 0.42.0 | Anthropic Claude API SDK |
| openai | >= 1.60.0 | OpenAI API SDK |
| mcp | >= 1.12.0 | Model Context Protocol |
| pydantic | >= 2.0 | 数据校验 / Tool Schema |
| pyyaml | >= 6.0 | YAML 配置解析 |
| httpx | >= 0.27.0 | HTTP 客户端 |
| websockets | >= 14.0 | WebSocket (MCP transport) |
| npx | >= 0.1.8 | NPX 执行 (MCP 服务器) |
| hatchling | — | 构建系统 |
| pytest | >= 9.0.3 | 测试框架 (dev) |
| ruff | >= 0.8.0 | Linter (dev) |
