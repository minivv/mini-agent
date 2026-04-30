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
| 7 | — | 子 Agent | 后台委派，三级线程池，协作式取消，并发限制 |
| 8 | — | Todo List | 任务追踪工具，结构化状态管理 |

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

**Phase 6 新增文件**

```
src/mini_agent/
├── skills.py        # (Phase 6 新增) Skill 发现 + 解析 + 提示词生成
├── skills/          # (Phase 6 新增) 技能目录
│   ├── code-review/SKILL.md
│   └── deep-research/SKILL.md
├── sandbox.py       # (Phase 6 扩展) 新增 /mnt/skills/ 虚拟挂载点
├── tools.py         # (不变)
├── agent.py         # (Phase 6 扩展) AgentState 新增 skills_dir
└── main.py         # (Phase 6 扩展) /skills 命令 + 启动时加载技能
```

**学习重点**

- **Skill 的本质**: 预定义的知识模块，SKILL.md 文件 = YAML frontmatter（名称+描述）+ Markdown 正文（工作流程）。Agent 是"通才员工"，Skill 是"操作手册"
- **SKILL.md 格式**: `---` 之间的 YAML frontmatter 用 `yaml.safe_load()` 解析，包含 `name` 和 `description` 字段
- **Skill 发现**: `load_skills()` 扫描 skills/ 目录，找到所有包含 SKILL.md 的子目录，解析为 `Skill` dataclass
- **渐进式加载**: 系统提示词只放技能摘要（名称+描述+路径），不放入完整内容。Agent 遇到匹配任务时用 `read_file` 按需读取 SKILL.md。好处：提示词不会太长，Agent 只加载需要的技能，添加新技能只需放文件
- **XML 提示词注入**: `get_skills_prompt_section()` 生成 `<skill_system><available_skills>...` 格式的 XML 块，注入到系统提示词末尾
- **虚拟挂载点扩展**: Sandbox 新增 `/mnt/skills/` 挂载点，Agent 可以用 `read_file /mnt/skills/code-review/SKILL.md` 读取技能内容
- **AgentState 扩展**: 新增 `skills_dir` 字段，`stream()` 注入沙箱时传递 skills 目录路径

**实现**

```
Skill 加载流程:
  skills/ 目录扫描 → parse_skill_file(YAML frontmatter)
  → get_skills_prompt_section() → 注入系统提示词
  → Agent 匹配任务 → read_file(/mnt/skills/xxx/SKILL.md)
  → 按技能工作流程执行
```

- [skills.py](src/mini_agent/skills.py) — `Skill` dataclass + `load_skills()` + `get_skills_prompt_section()`
- [skills/code-review/SKILL.md](skills/code-review/SKILL.md) — 代码审查技能
- [skills/deep-research/SKILL.md](skills/deep-research/SKILL.md) — 深度研究技能

### 试玩

```bash
git checkout phase-6
cp .env.example .env  # 填入 LLM_API_KEY
uv run mini-agent
```

尝试输入:
- `/skills` 查看已加载的技能列表
- "帮我审查一下代码质量" — Agent 会先读取 code-review 的 SKILL.md
- "深入研究一下 RAG 技术" — Agent 会先读取 deep-research 的 SKILL.md
