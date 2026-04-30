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
| 5 | — | MCP 协议 | MCP stdio/sse 客户端，外部工具动态加载 |
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
- **AgentState 扩展**: `TypedDict(total=False)` 灵活新增字段

**实现**

- [sandbox.py](src/mini_agent/sandbox.py) — `Sandbox` 类，路径隔离 + bash 执行
- [tools.py](src/mini_agent/tools.py) — 7 个沙箱工具 + `InjectedState` + 工具分组

## Phase 4: 上下文管理

**Phase 4 新增文件**

```
src/mini_agent/
├── context.py       # (Phase 4 新增) Token 计数 + 对话摘要
├── agent.py         # (Phase 4 扩展) 新增 check_context 节点
└── config.py        # (Phase 4 扩展) 新增 max_context_tokens / keep_recent_messages
```

**学习重点**

- **Token 计数**: `tiktoken`（cl100k_base 编码）在本地估算消息 token 数，不需要 API 调用。每条消息有 ~4 token 固定开销，加上内容的 token 数
- **对话摘要（Summarization）**: 当 token 数超过阈值时，用 LLM 把旧消息压缩成一段摘要。类似"前情提要"——保留关键信息，丢弃细节
- **RemoveMessage**: LangGraph 的特殊消息——放入 state 更新列表后，会从消息历史中删除对应 ID 的消息。这是实现"压缩"的核心机制
- **摘要策略**: 保留最近 N 条消息不动，只摘要更早的消息。跳过 `SystemMessage`（系统提示词保留）、`ToolMessage`（工具结果价值低）、含 `tool_calls` 的 `AIMessage`（工具调用链的一部分）
- **check_context 节点**: 在图中作为"收尾节点"，每次 agent 回复后（或工具执行后）检查是否需要摘要。图变为 `agent → tools → agent → ... → check_context → END`
- **孤立 ToolMessage 清理**: 删除 AIMessage 后，对应的 ToolMessage 变成孤立消息。摘要逻辑额外检查并清理这些孤立的 ToolMessage

**实现**

```
摘要流程:
  1. count_message_tokens(messages) → 超过阈值?
  2. 分割: [旧消息] + [最近 N 条]
  3. _format_for_summary() → 文本格式
  4. LLM.invoke(摘要 prompt) → 生成摘要
  5. 返回: [RemoveMessage(旧), SystemMessage(摘要)]
```

- [context.py](src/mini_agent/context.py) — `count_message_tokens()` + `summarize_messages()`
- [agent.py](src/mini_agent/agent.py) — `check_context` 节点 + 孤立 ToolMessage 清理
- [config.py](src/mini_agent/config.py) — `MAX_CONTEXT_TOKENS` / `KEEP_RECENT_MESSAGES`

### 试玩

```bash
git checkout phase-4
cp .env.example .env  # 填入 LLM_API_KEY
uv run mini-agent
```

尝试输入:
- 连续对话 10+ 轮，观察 `/tokens` 命令的 token 变化
- 当 token 超过阈值（默认 30000）时，自动触发摘要压缩
- 用 `/state` 查看消息数在摘要前后从 N 条变为 keep_recent + 1 条
