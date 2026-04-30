# CLAUDE.md — Mini Agent 项目上下文

## 项目宗旨

从零开始、逐阶段构建一个 AI Agent。每个 Phase 都是**可运行、可理解**的最小里程碑。

## 分支策略

```
phase-1  →  phase-2  →  phase-3  →  ...  →  main (最终合并)
```

- 每个 Phase 在独立分支开发，基于上一阶段分支创建
- 当前分支提交即完成该阶段，不 push
- `.env` 文件纳入版本管理（本地项目，不推送远端）

## 阶段计划


| # | 分支      | 主题       | 新增/引入                                          |
| - | --------- | ---------- | -------------------------------------------------- |
| 1 | `phase-1` | 基础对话   | `ChatModel`, Messages, `invoke/stream`             |
| 2 | `phase-2` | Agent 核心 | `StateGraph`, `ToolNode`, `tools_condition`, ReAct |
| 3 | —        | 沙箱工具   | 文件读写 + bash 执行, 路径隔离                     |
| 4 | —        | 上下文管理 | 对话历史截断, token 计数, summarization            |
| 5 | —        | 长期记忆   | 记忆提取/存储/注入                                 |
| 6 | —        | MCP 协议   | MCP client, 外部工具发现与调用                     |
| 7 | —        | 子 Agent   | 后台委派, multi-agent                              |
| 8 | —        | REST API   | FastAPI, SSE streaming, thread/run 生命周期        |
| 9 | —        | Web UI     | Next.js 聊天界面                                   |

## 关键约束

- **从简开始**: 不提前引入后续 Phase 才需要的依赖或抽象
- **可运行**: 每个 Phase 结束时代码必须能跑
- **中文注释**: 所有注释用中文，解释核心概念和设计意图

## 当前技术栈

```
uv (包管理) → langchain-core, langchain-openai, langgraph, python-dotenv
Python ≥ 3.12
```

## 开发命令

```bash
uv run python -c "..."          # 临时验证
uv run mini-agent               # 运行 CLI
cd src/mini_agent && uv run python -m mini_agent.main  # 等价方式
```

## 文件结构约定

```
src/mini_agent/
├── config.py   # 配置（.env 加载 → dataclass）
├── chat.py     # 基础 ChatModel 封装（Phase 1，后续可能被替代）
├── tools.py    # 工具定义（@tool 装饰器）
├── agent.py    # Agent 核心（LangGraph 图构建）
└── main.py     # CLI 入口
```

新增能力优先考虑新增独立文件，而非往已有文件塞逻辑。
