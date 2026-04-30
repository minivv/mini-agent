# Mini Agent — 从零构建 AI Agent

通过逐阶段构建一个 AI Agent 来学习 **LangChain** 和 **LangGraph**。

## 分支管理

每个 Phase 对应一个分支:


| Phase | 分支      | 主题           | 核心技术                                                                      |
| ----- | --------- | -------------- | ----------------------------------------------------------------------------- |
| 1     | `phase-1` | **基础对话**   | `ChatModel`, `SystemMessage/HumanMessage/AIMessage`, `invoke()` vs `stream()` |
| 2     | `phase-2` | **Agent 核心** | `StateGraph`, `ToolNode`, `tools_condition`, ReAct 模式                       |
| 3     | —        | 沙箱工具       | 文件系统 + bash 执行，路径隔离，`InjectedState`                               |
| 4     | —        | 上下文管理     | Token 计数，对话摘要，`RemoveMessage` 压缩历史                                |
| 5     | —        | MCP 协议       | MCP stdio/sse 客户端，外部工具动态加载                                        |
| 6     | —        | Skills         | 知识模块发现，YAML frontmatter 解析，渐进式加载                               |
| 7     | —        | 子 Agent       | 后台委派，三级线程池，协作式取消，并发限制                                    |
| 8     | —        | Todo List      | 任务追踪工具，结构化状态管理                                                  |

## Phase 1: 基础对话

**学习重点**

- **ChatModel 统一抽象**: 一个 `ChatOpenAI` 类通过切换 `base_url` 对接所有 OpenAI 兼容提供商（OpenAI / DeepSeek / 豆包 / Ollama 等）
- **消息类型**: `SystemMessage`（行为设定）、`HumanMessage`（用户输入）、`AIMessage`（AI 回复）——三者组成 `list` 就是完整对话历史，模型每次看到整个 list，这就是多轮对话的基础
- **invoke() vs stream()**: `invoke()` 一次性返回完整结果，适合内部调用；`stream()` 逐 token 返回，用于打字机效果。两者共享同一个底层的 Chat Completion API

**实现**

```
ChatAgent
├── __init__()     → 创建 model + 初始化 SystemMessage
├── chat()         → invoke() 同步对话（内部使用）
├── chat_stream()  → stream() 流式对话（generator，yield token）
└── reset()        → 清空历史，保留 system prompt
```

- [chat.py](src/mini_agent/chat.py) — `ChatModel` 创建 + `ChatAgent` 实现
- [config.py](src/mini_agent/config.py) — `.env` 配置加载
- [main.py](src/mini_agent/main.py) — CLI 入口，`/reset` `/history` `/quit` 命令

## Phase 2: Agent 核心

**Phase 2 新增文件**

```
src/mini_agent/
├── config.py       # (Phase 1, 不变)
├── chat.py         # (Phase 1, 保留作为对比) ChatModel 封装
├── tools.py        # (Phase 2 新增) @tool 装饰器定义工具
├── agent.py        # (Phase 2 新增) LangGraph ReAct Agent
└── main.py         # (Phase 2 改写) 支持 tool-calling 的流式 CLI
```

**学习重点**

- **StateGraph**: Agent 控制流的蓝图，由节点(nodes)和边(edges)组成。比 Phase 1 的 while 循环更强大——支持条件分支、状态持久化、流式事件、人工干预
- **State + Reducer**: 用 `TypedDict` 定义在节点间流动的数据。`add_messages` 是 LangGraph 内置的 reducer——不是替换 messages 列表，而是追加，并自动合并流式碎片
- **ToolNode**: LangGraph 预置的工具执行节点，自动读取 `AIMessage.tool_calls` 并逐一执行，返回 `ToolMessage` 列表
- **tools_condition**: 预置的条件边函数——最后一条消息有 `tool_calls` → 去 tools 节点，没有 → 去 END
- **ReAct 模式**: `START → agent → [有 tool_calls?] → tools → agent → ... → END`。LLM 自主决定何时调用工具、何时给出最终回答
- **Checkpointer**: `MemorySaver` 按线程隔离保存状态，`thread_id` 切换即可重置对话
- **bind_tools()**: 把工具列表注入 model，LLM 通过函数的 docstring 和 type hints 理解何时调用哪个工具
- **stream_mode=["messages"]**: 逐 token 流式输出，通过 `metadata["langgraph_node"]` 区分 LLM 文本和工具事件

**实现**

```
LangGraphAgent
├── __init__()      → model + checkpointer + graph
├── _build_graph()  → StateGraph 构建
│   ├── agent 节点   → call_model (bind_tools → invoke)
│   ├── tools 节点   → ToolNode (自动执行 tool_calls)
│   ├── START → agent
│   ├── agent → tools_condition → tools | END
│   └── tools → agent (循环)
├── stream()        → 流式执行，yield (token_text, msg_chunk, metadata)
└── invoke()        → 非流式执行，返回完整消息列表
```

- [agent.py](src/mini_agent/agent.py) — `LangGraphAgent`，核心的图构建和流式执行
- [tools.py](src/mini_agent/tools.py) — `@tool` 装饰器定义 `calculator` 和 `get_current_time`
- [main.py](src/mini_agent/main.py) — CLI 入口，区分 LLM 文本 / tool_call / tool_result 事件
- [chat.py](src/mini_agent/chat.py) — 保留 Phase 1 的 `ChatAgent` 和 `create_model` 工具函数

### 试玩

```bash
git checkout phase-2
cp .env.example .env  # 填入 LLM_API_KEY
uv run mini-agent
```

尝试输入:

- "帮我算一下 (123 + 456) * 789"
- "现在几点了？"
- "先算 100/3，再乘以 6，然后告诉我现在几点"
- "/reset" 重置对话后试试多轮上下文是否保持
