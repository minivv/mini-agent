# Mini Agent — 从零构建 AI Agent

通过逐阶段构建一个 AI Agent 来学习 **LangChain** 和 **LangGraph**。

每个 Phase 对应一个独立的 Git 分支，切换分支即可查看该阶段的完整代码。

## 分支总览

| Phase | 分支 | 主题 | 你将学到 |
|-------|------|------|---------|
| 1 | `phase-1` | 基础对话 | ChatModel 统一抽象、消息类型、invoke() vs stream() |
| 2 | `phase-2` | Agent 核心 | StateGraph、ToolNode、tools_condition、ReAct 模式、Checkpointer |
| 3 | `phase-3` | 沙箱工具 | 路径隔离、虚拟挂载点、InjectedState、bash 沙箱执行 |
| 4 | `phase-4` | 上下文管理 | tiktoken Token 计数、对话摘要、RemoveMessage 压缩历史 |
| 5 | `phase-5` | MCP 协议 | stdio/sse 客户端、配置驱动、异步转同步、MultiServerMCPClient |
| 6 | `phase-6` | Skills | YAML frontmatter 解析、渐进式加载、XML 提示词注入 |
| 7 | `phase-7` | 子 Agent | 三级线程池、协作式取消、并发限制、工具过滤 |
| 8 | `phase-8` | Todo List | 结构化任务追踪、上下文丢失检测、提前退出防护 |

## 如何使用

```bash
# 克隆仓库
git clone https://github.com/minivv/mini-agent.git
cd mini-agent

# 切换到任意 Phase 分支
git checkout phase-1    # 从最简单的开始

# 安装依赖
uv sync

# 配置 API Key
cp .env.example .env
# 编辑 .env，填入你的 LLM_API_KEY

# 运行
uv run mini-agent
```

## 学习路线

建议按 Phase 1 → 8 的顺序逐步学习。每个分支的 README 包含该阶段的详细学习重点和实现说明。

每个 Phase 都是在前一个的基础上增量构建的——`phase-8` 包含了全部 8 个阶段的完整代码，`phase-1` 只有最基础的对话功能。
