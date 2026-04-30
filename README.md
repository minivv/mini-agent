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
| 6 | — | Skills | 知识模块发现，YAML frontmatter 解析，渐进式加载 |
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

**Phase 5 新增文件**

```
src/mini_agent/
├── mcp_client.py    # (Phase 5 新增) MCP 客户端，加载外部工具
├── mcp_config.json  # (Phase 5 新增) MCP 服务器配置
├── tools.py         # (不变)
├── agent.py         # (不变)
└── main.py         # (Phase 5 扩展) 启动时加载 MCP 工具
```

**学习重点**

- **MCP 协议**: Anthropic 提出的开放协议，让 LLM 连接外部工具和数据源。MCP 服务器就像"手脚"——每个服务器提供一组特定能力（文件系统、GitHub、数据库等）
- **三种传输方式**: `stdio`（启动子进程，stdin/stdout 通信，最常用）、`sse`（HTTP Server-Sent Events，远程服务器）、`streamable_http`（Streamable HTTP）
- **配置驱动**: `mcp_config.json` 声明式定义服务器列表，支持 `enabled` 开关、`env` 环境变量替换（`$VAR_NAME` → `os.environ`）
- **MultiServerMCPClient**: `langchain-mcp-adapters` 库，一个客户端同时连接多个 MCP 服务器，`get_tools()` 返回所有服务器的工具
- **tool_name_prefix**: 给 MCP 工具名加服务器名前缀（如 `filesystem__read_file`），避免与内置工具名冲突
- **异步转同步**: MCP 工具底层是异步的（async protocol），但 `ToolNode` 默认同步调用。用 `ThreadPoolExecutor` + `asyncio.run()` 包装异步工具为同步接口
- **工具动态加载**: MCP 工具在启动时加载，追加到 `ALL_TOOLS` 列表，和内置工具一起绑定给 Agent

**实现**

```
MCP 加载流程:
  mcp_config.json → load_mcp_config() → build_connections()
      → MultiServerMCPClient(connections).get_tools()
      → _wrap_async_tools() (异步转同步)
      → 返回 list[BaseTool]
```

- [mcp_client.py](src/mini_agent/mcp_client.py) — `load_mcp_tools()` 主入口 + `_wrap_async_tools()` 异步包装
- [mcp_config.json](mcp_config.json) — MCP 服务器声明式配置

### 试玩

```bash
git checkout phase-5
cp .env.example .env  # 填入 LLM_API_KEY
uv run mini-agent
```

尝试输入:
- 查看 `/tools` 命令，确认 MCP 工具已加载
- 使用 MCP 服务器提供的工具（取决于 mcp_config.json 的配置）
