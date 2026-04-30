# Mini Agent — 从零构建 AI Agent

通过逐阶段构建一个 AI Agent 来学习 **LangChain** 和 **LangGraph**。

## 分支管理

每个 Phase 对应一个分支:

| Phase | 分支 | 主题 | 核心技术 |
|-------|------|------|---------|
| 1 | `phase-1` | **基础对话** | `ChatModel`, `SystemMessage/HumanMessage/AIMessage`, `invoke()` vs `stream()` |
| 2 | `phase-2` | **Agent 核心** | `StateGraph`, `ToolNode`, `tools_condition`, ReAct 模式 |
| 3 | — | 沙箱工具 | 文件系统 + bash 执行，路径隔离 |
| 4 | — | 上下文管理 | 对话历史、summarization、token 管理 |
| 5 | — | 长期记忆 | 记忆提取、存储、注入 |
| 6 | — | MCP 协议 | MCP client, 外部工具集成 |
| 7 | — | 子 Agent | 后台任务委派，multi-agent |
| 8 | — | REST API | FastAPI gateway, SSE streaming |
| 9 | — | Web UI | Next.js 聊天界面 |

## Phase 2 新增内容

```
src/mini_agent/
├── config.py       # (Phase 1, 不变)
├── chat.py         # (Phase 1, 保留作为对比) ChatModel 封装
├── tools.py        # (Phase 2 新增) @tool 装饰器定义工具
├── agent.py        # (Phase 2 新增) LangGraph ReAct Agent
└── main.py         # (Phase 2 改写) 支持 tool-calling 的流式 CLI
```

### 核心概念

- **StateGraph**: agent 控制流的蓝图，由节点和边组成
- **State**: 在节点间流动的数据（TypedDict + reducer）
- **Nodes**: agent 节点（调用 LLM）+ tools 节点（执行工具）
- **Conditional Edges**: 用 `tools_condition` 判断 LLM 是否想调用工具
- **Checkpointer**: `MemorySaver` 自动管理多轮对话状态
- **Tool Calling**: `bind_tools()` 让 LLM 知道可用工具，LLM 自主决定何时调用
- **ReAct 模式**: Reasoning + Acting 循环

### 试玩

```bash
# 切换到 Phase 2 分支
git checkout phase-2

# 配置 API key (如果还没做)
cp .env.example .env
# 编辑 .env，填入 LLM_API_KEY

# 运行
uv run mini-agent
```

尝试输入:
- "帮我算一下 (123 + 456) * 789"
- "现在几点了？"
- "先算 100/3，再乘以 6，然后告诉我现在几点"
