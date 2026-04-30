# Mini Agent — 从零构建 AI Agent

通过逐阶段构建一个 AI Agent 来学习 **LangChain** 和 **LangGraph**。

## 分支管理

每个 Phase 对应一个分支:

| Phase | 分支 | 主题 | 核心技术 |
|-------|------|------|---------|
| 1 | `phase-1` | **基础对话** | `ChatModel`, `SystemMessage/HumanMessage/AIMessage`, `invoke()` vs `stream()` |
| 2 | `phase-2` | **Agent 核心** | `StateGraph`, `ToolNode`, `tools_condition`, ReAct 模式 |
| 3 | `phase-3` | **沙箱工具** | 文件系统 + bash 执行，路径隔离，`InjectedState` |
| 4 | `phase-4` | **上下文管理** | Token 计数，对话摘要，`RemoveMessage` 压缩历史 |
| 5 | `phase-5` | **MCP 协议** | MCP stdio/sse 客户端，外部工具动态加载 |
| 6 | `phase-6` | **Skills** | 知识模块发现，YAML frontmatter 解析，渐进式加载 |
| 7 | `phase-7` | **子 Agent** | 后台委派，三级线程池，协作式取消，并发限制 |
| 8 | `phase-8` | **Todo List** | 任务追踪工具，上下文丢失检测，提前退出防护 |

## Phase 1: 基础对话

**学习重点**

- **ChatModel 统一抽象**: 一个 `ChatOpenAI` 类通过切换 `base_url` 对接所有 OpenAI 兼容提供商
- **消息类型**: `SystemMessage` / `HumanMessage` / `AIMessage` 组成对话历史
- **invoke() vs stream()**: `invoke()` 一次性返回，`stream()` 逐 token 流式输出

**实现**

- [chat.py](src/mini_agent/chat.py) — `ChatAgent` + `create_model`
- [config.py](src/mini_agent/config.py) — `.env` 配置加载

## Phase 2: Agent 核心

**学习重点**

- **StateGraph**: 节点(nodes) + 边(edges) 定义 Agent 控制流
- **State + Reducer**: `TypedDict` + `add_messages` 在节点间传递数据
- **ToolNode**: 预置节点，自动执行 `AIMessage.tool_calls`
- **tools_condition**: 条件边，有 tool_calls → tools，没有 → END
- **ReAct 模式**: `agent → tools → agent → ...` 循环
- **Checkpointer**: `MemorySaver` 按线程隔离保存状态

**实现**

- [agent.py](src/mini_agent/agent.py) — `LangGraphAgent` 图构建和流式执行
- [tools.py](src/mini_agent/tools.py) — `calculator` + `get_current_time`

## Phase 3: 沙箱工具

**学习重点**

- **路径隔离**: 虚拟路径 `/mnt/workspace/` → 真实路径 `~/.mini-agent/workspaces/{thread_id}/`
- **虚拟挂载点**: `_resolve()` 虚拟→真实，`_display_path()` 真实→虚拟，路径穿越被拒绝
- **InjectedState**: 工具参数声明 `state: Annotated[dict, InjectedState]`，框架自动注入 `AgentState`
- **工具分组**: `BUILTIN_TOOLS` + `SANDBOX_TOOLS` → `ALL_TOOLS`
- **bash 沙箱**: `_translate_command()` 路径翻译 + `subprocess.run` 超时控制

**实现**

- [sandbox.py](src/mini_agent/sandbox.py) — `Sandbox` 类，路径隔离 + bash 执行
- [tools.py](src/mini_agent/tools.py) — 7 个沙箱工具 + `InjectedState` + 工具分组

## Phase 4: 上下文管理

**学习重点**

- **Token 计数**: `tiktoken`（cl100k_base 编码）本地估算消息 token 数
- **对话摘要**: 超过阈值时用 LLM 压缩旧消息为摘要，保留最近 N 条不动
- **RemoveMessage**: 放入 state 更新后删除对应 ID 的消息，实现"压缩"
- **孤立 ToolMessage 清理**: 删除 AIMessage 后，清理无对应 tool_call 的 ToolMessage
- **check_context 节点**: 图变为 `agent → tools → agent → check_context → END`

**实现**

- [context.py](src/mini_agent/context.py) — `count_message_tokens()` + `summarize_messages()`

## Phase 5: MCP 协议

**学习重点**

- **MCP 协议**: 开放协议，让 LLM 连接外部工具和数据源。MCP 服务器提供特定能力，客户端连接获取工具
- **三种传输方式**: `stdio`（子进程 stdin/stdout）、`sse`（HTTP SSE）、`streamable_http`
- **配置驱动**: `mcp_config.json` 声明式定义服务器，支持 `env` 环境变量替换
- **MultiServerMCPClient**: 一个客户端同时连接多个服务器，`get_tools()` 返回所有工具
- **tool_name_prefix**: 工具名加服务器前缀防冲突（如 `filesystem__read_file`）
- **异步转同步**: `ThreadPoolExecutor` + `asyncio.run()` 包装异步 MCP 工具

**实现**

- [mcp_client.py](src/mini_agent/mcp_client.py) — `load_mcp_tools()` + `_wrap_async_tools()`
- [mcp_config.json](mcp_config.json) — MCP 服务器配置

## Phase 6: Skills

**学习重点**

- **Skill 的本质**: 预定义的知识模块，SKILL.md 文件 = YAML frontmatter（名称+描述）+ Markdown 正文（工作流程）。Agent 是"通才员工"，Skill 是"操作手册"
- **SKILL.md 格式**: `---` 之间的 YAML frontmatter 用 `yaml.safe_load()` 解析，包含 `name` 和 `description` 字段
- **Skill 发现**: `load_skills()` 扫描 skills/ 目录，找到所有包含 SKILL.md 的子目录，解析为 `Skill` dataclass
- **渐进式加载**: 系统提示词只放技能摘要（名称+描述+路径），不放入完整内容。Agent 遇到匹配任务时用 `read_file` 按需读取 SKILL.md。好处：提示词不会太长，Agent 只加载需要的技能，添加新技能只需放文件
- **XML 提示词注入**: `get_skills_prompt_section()` 生成 `<skill_system><available_skills>...` 格式的 XML 块，注入到系统提示词末尾
- **虚拟挂载点扩展**: Sandbox 新增 `/mnt/skills/` 挂载点，Agent 可以用 `read_file /mnt/skills/code-review/SKILL.md` 读取技能内容

**实现**

- [skills.py](src/mini_agent/skills.py) — `Skill` dataclass + `load_skills()` + `get_skills_prompt_section()`
- [skills/code-review/SKILL.md](skills/code-review/SKILL.md) — 代码审查技能
- [skills/deep-research/SKILL.md](skills/deep-research/SKILL.md) — 深度研究技能

## Phase 7: 子 Agent

**Phase 7 新增文件**

```
src/mini_agent/
├── subagents/             # (Phase 7 新增) 子 Agent 系统
│   ├── __init__.py        # 包导出
│   ├── config.py          # SubagentConfig 数据类
│   ├── registry.py        # 注册表: 查询、注册、注销子 Agent
│   ├── executor.py        # 执行引擎: 三级线程池 + 后台任务管理
│   └── builtins/          # 内置子 Agent
│       ├── __init__.py    # BUILTIN_SUBAGENTS 字典
│       ├── general_purpose.py  # 通用子 Agent (全部工具)
│       └── bash_agent.py       # Bash 子 Agent (仅沙箱工具)
├── task_tool.py           # (Phase 7 新增) task 委派工具 + 并发限制
├── agent.py               # (Phase 7 扩展) subagent_limit 节点
├── chat.py                # (Phase 7 扩展) build_subagent_section()
└── main.py                # (Phase 7 扩展) /agents 命令 + task_tool 集成
```

**学习重点**

- **任务委派模式**: Lead Agent 通过 `task` 工具将子任务委派给专门的子 Agent，每个子 Agent 是独立的 LangGraph Agent 实例，拥有独立的 system prompt、工具列表和 checkpointer 线程
- **SubagentConfig**: 子 Agent 配置数据类，包含 name、description、system_prompt、tools（允许列表）、disallowed_tools（禁止列表）、model、max_turns、timeout_seconds。默认禁止 `task` 工具防止递归委派
- **三级线程池**: `_scheduler_pool`（接收任务提交）→ `_execution_pool`（实际执行子 Agent）→ `_isolated_loop_pool`（在已有 event loop 时创建隔离的异步执行环境）。三级设计解决 Python 线程中嵌套 asyncio event loop 的限制
- **协作式取消**: 通过 `threading.Event` 实现。取消请求设置 event，子 Agent 在 `astream()` 每次迭代边界检查 event，发现已设置则提前退出。非抢占式，依赖 Agent 主动检查
- **并发限制**: `SubagentLimitMiddleware` 检查 LLM 单次输出中的 `task` 调用数量，超过 `max_concurrent`（默认 3）的截断丢弃。返回 `[RemoveMessage, AIMessage]` 替换原始消息
- **全局任务存储**: `_background_tasks` 字典 + `_background_tasks_lock` 线程锁，`task_tool` 通过轮询（每 5 秒）读取任务状态，终端状态时自动清理防止内存泄漏
- **工具过滤**: 子 Agent 的工具列表由 `allowlist` ∩ `denylist` 计算得出，实现最小权限原则。例如 bash 子 Agent 只能使用沙箱工具
- **实时状态推送**: `status_callback` 回调函数在子 Agent 产生新 AI 消息时将预览推送到 CLI，用户可以实时看到子 Agent 的进度

**实现**

```
任务委派流程:
  LLM 输出 task(description, prompt, subagent_type)
  → task_tool 解析 SubagentConfig
  → SubagentExecutor.execute_async() 提交到调度池
  → 调度池 → 执行池 (带超时)
  → 创建独立 LangGraphAgent (独立 thread_id)
  → astream() 流式执行 (检查 cancel_event)
  → 结果写入 _background_tasks
  → task_tool 轮询获取结果
  → 返回给 lead agent
```

- [subagents/config.py](src/mini_agent/subagents/config.py) — `SubagentConfig` 数据类
- [subagents/registry.py](src/mini_agent/subagents/registry.py) — `get_subagent_config()` + `register_subagent()`
- [subagents/executor.py](src/mini_agent/subagents/executor.py) — `SubagentExecutor` + 三级线程池 + 全局任务管理
- [task_tool.py](src/mini_agent/task_tool.py) — `create_task_tool()` + `truncate_task_calls()`

### 试玩

```bash
git checkout phase-7
cp .env.example .env  # 填入 LLM_API_KEY
uv run mini-agent
```

尝试输入:
- `/agents` 查看可用的子 Agent 列表
- "帮我写一个 Python 排序算法，保存到 /mnt/workspace/sort.py" — Agent 会委派给 general-purpose 子 Agent
- "运行一下 /mnt/workspace/sort.py 看看效果" — Agent 会委派给 bash 子 Agent

## Phase 8: Todo List

**Phase 8 新增文件**

```
src/mini_agent/
├── todo.py             # (Phase 8 新增) Todo 数据模型 + 工具 + 图节点 + CLI 渲染
├── agent.py            # (Phase 8 扩展) todos 字段 + todo_check + todo_guard 节点
├── chat.py             # (Phase 8 扩展) build_todo_section() 系统提示词
└── main.py             # (Phase 8 扩展) /todos 命令 + todo 面板渲染
```

**学习重点**

- **整体替换模式**: `write_todos` 工具每次调用都整体替换整个 todo 列表，不做增量更新。好处是实现简单、无状态一致性问题，LLM 只需维护一个完整列表
- **Todo 数据模型**: `Todo` TypedDict 包含 `content`（任务描述）和 `status`（`pending`/`in_progress`/`completed`）。三个状态的转换完全由 LLM 自主决定，没有后端强制约束
- **State 持久化**: `todos` 作为 `AgentState` 的字段，通过 LangGraph checkpointer 持久化。`total=False` + 无自定义 reducer 意味着每次赋值都替换整个列表
- **上下文丢失检测**: `todo_check` 节点在每轮结束时检查：如果 `state.todos` 存在但消息历史中找不到 `write_todos` 的 `ToolMessage`（被摘要压缩掉了），则注入一条 `todo_reminder` 消息让 LLM 重新感知未完成的任务
- **提前退出防护**: `todo_guard` 节点检测 LLM 是否试图在 todo 未完成时结束（最后一条 AIMessage 无 `tool_calls`）。如果是，注入 `todo_completion_reminder` 并路由回 `agent` 节点强制继续。上限 `_MAX_COMPLETION_REMINDERS = 2` 防止死循环
- **图结构变化**: `check_context → todo_check → todo_guard → END`。`todo_check` 和 `todo_guard` 都是条件边，有提醒时回到 `agent`，没有则继续向下
- **CLI 面板渲染**: `render_todo_panel()` 将 todos 列表渲染为 Unicode 边框面板，`completed` 显示 ✓，`in_progress` 显示蓝色 →，`pending` 显示 ○

**实现**

```
Todo 生命周期:
  用户请求 → LLM 判断需要规划 → write_todos([pending, pending, pending])
  → 开始第一项 → write_todos([in_progress, pending, pending])
  → 完成第一项 → write_todos([completed, in_progress, pending])
  → ... → 全部 completed → 给出最终回复

上下文丢失检测:
  摘要压缩删除了 write_todos 的 ToolMessage
  → todo_check 检测到 todos 非空但历史无 write_todos
  → 注入 todo_reminder 消息
  → 路由回 agent → LLM 重新感知任务列表

提前退出防护:
  LLM 输出最终回复 (无 tool_calls) 但 todos 未全部完成
  → todo_guard 注入 todo_completion_reminder
  → 路由回 agent → LLM 继续工作
  → 最多提醒 2 次 → 允许退出
```

- [todo.py](src/mini_agent/todo.py) — `Todo` 模型 + `write_todos` 工具 + `check_todo_context()` + `guard_todo_completion()` + `render_todo_panel()`

### 试玩

```bash
git checkout phase-8
cp .env.example .env  # 填入 LLM_API_KEY
uv run mini-agent
```

尝试输入:
- `/todos` 查看当前任务列表
- "帮我一步一步实现一个快速排序算法，先写代码，再写测试，最后运行测试" — Agent 会先用 write_todos 创建计划
- "分析一下 /mnt/workspace/sort.py 的代码质量，给出改进建议" — 多步骤任务会触发 todo 追踪
