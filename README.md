# Mini Agent — 从零构建 AI Agent

通过逐阶段构建一个 AI Agent 来学习 **LangChain** 和 **LangGraph**。

## 路线图


| Phase | 主题           | 核心技术                                                                      | 状态 |
| ----- | -------------- | ----------------------------------------------------------------------------- | ---- |
| 1     | **基础对话**   | `ChatModel`, `SystemMessage/HumanMessage/AIMessage`, `invoke()` vs `stream()` | done |
| 2     | **Agent 核心** | `StateGraph`, `ToolNode`, tool-calling, ReAct 模式                            | done |
| 3     | **沙箱工具**   | 文件系统 + bash 执行，路径隔离，`InjectedState`                               | done |
| 4     | **上下文管理** | Token 计数，对话摘要，`RemoveMessage` 压缩历史                                | done |
| 5     | **MCP 协议**   | MCP stdio/sse 客户端，外部工具动态加载                                        | done |
| 6     | **Skills**     | 知识模块发现，YAML frontmatter 解析，渐进式加载                               | done |
| 7     | **子 Agent**   | 后台委派，三级线程池，协作式取消，并发限制                                    | done |
| 8     | **Todo List**  | 任务追踪工具，结构化状态管理                                                  | todo |

## 开始使用

```bash
# 1. 配置 API Key
cp .env.example .env
# 编辑 .env，填入你的 LLM_API_KEY（支持 OpenAI 兼容接口）

# 2. 运行
uv run mini-agent
```

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
