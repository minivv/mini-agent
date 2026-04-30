# Mini Agent — 从零构建 AI Agent

通过逐阶段构建一个 AI Agent 来学习 **LangChain** 和 **LangGraph**。

## 分支管理

每个 Phase 对应一个分支:


| Phase | 分支      | 主题           | 核心技术                                                                      |
| ----- | --------- | -------------- | ----------------------------------------------------------------------------- |
| 1     | `phase-1` | **基础对话**   | `ChatModel`, `SystemMessage/HumanMessage/AIMessage`, `invoke()` vs `stream()` |
| 2     | `phase-2` | **Agent 核心** | `StateGraph`, `ToolNode`, `tools_condition`, ReAct 模式                       |
| 3     | `phase-3` | **沙箱工具**   | 文件系统 + bash 执行，路径隔离，`InjectedState`                               |
| 4     | —        | 上下文管理     | Token 计数，对话摘要，`RemoveMessage` 压缩历史                                |
| 5     | —        | MCP 协议       | MCP stdio/sse 客户端，外部工具动态加载                                        |
| 6     | —        | Skills         | 知识模块发现，YAML frontmatter 解析，渐进式加载                               |
| 7     | —        | 子 Agent       | 后台委派，三级线程池，协作式取消，并发限制                                    |
| 8     | —        | Todo List      | 任务追踪工具，结构化状态管理                                                  |

## Phase 1: 基础对话

**学习重点**

- **ChatModel 统一抽象**: 一个 `ChatOpenAI` 类通过切换 `base_url` 对接所有 OpenAI 兼容提供商
- **消息类型**: `SystemMessage` / `HumanMessage` / `AIMessage` 组成对话历史，模型每次看到整个 list
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

**Phase 3 新增文件**

```
src/mini_agent/
├── sandbox.py      # (Phase 3 新增) 文件系统 + bash 沙箱
├── tools.py        # (Phase 3 扩展) 新增 7 个沙箱工具 + InjectedState
├── agent.py        # (Phase 3 扩展) AgentState 新增 workspace_dir
└── main.py         # (Phase 3 扩展) /tools /skills 命令
```

**学习重点**

- **路径隔离**: Agent 看到虚拟路径 `/mnt/workspace/foo.txt`，实际映射到 `~/.mini-agent/workspaces/{thread_id}/workspace/foo.txt`。路径穿越 (`../../etc/passwd`) 被拒绝
- **虚拟挂载点**: `_resolve()` 将虚拟路径翻译为真实路径，`_display_path()` 反向翻译给 Agent 展示。
- **InjectedState**: LangGraph 机制——工具函数声明 `state: Annotated[dict, InjectedState]` 参数，框架自动注入当前 `AgentState`。这样沙箱工具能在每次调用时知道"工作目录在哪"
- **工具分组**: `BUILTIN_TOOLS`（无状态）+ `SANDBOX_TOOLS`（需要 state）→ `ALL_TOOLS`。
- **bash 沙箱**: `_translate_command()` 在执行前把虚拟路径替换为真实路径，`cwd` 设为沙箱目录，`subprocess.run` 带超时
- **AgentState 扩展**: Phase 2 只有 `messages`，Phase 3 新增 `workspace_dir`——State 是 `TypedDict(total=False)`，可以灵活扩展

**实现**

```
Sandbox
├── __init__(thread_id)  → 创建隔离目录 ~/.mini-agent/workspaces/{id}/
├── _resolve()           → 虚拟路径 → 真实路径 (含路径穿越检查)
├── _display_path()      → 真实路径 → 虚拟路径
├── read_file()          → 带行号的文件读取
├── write_file()         → 创建/覆盖文件 (自动 mkdir)
├── str_replace()        → 字符串替换 (单次)
├── ls()                 → 树形目录列表
├── glob()               → 文件模式匹配
├── grep()               → 文本搜索
└── bash()               → 沙箱内命令执行 (路径翻译 + 超时)
```

- [sandbox.py](src/mini_agent/sandbox.py) — `Sandbox` 类，路径隔离 + bash 执行
- [tools.py](src/mini_agent/tools.py) — 7 个沙箱工具 + `InjectedState` + 工具分组
- [agent.py](src/mini_agent/agent.py) — `AgentState` 新增 `workspace_dir`，`stream()` 注入沙箱路径

### 试玩

```bash
git checkout phase-3
cp .env.example .env  # 填入 LLM_API_KEY
uv run mini-agent
```

尝试输入:

- "帮我创建一个 hello.py，内容是 print('Hello World')"
- "用 bash 运行这个文件"
- "查看一下 /mnt/workspace 目录下有什么"
- "用 Python 算一下 2 的 10 次方，把结果写入 result.txt"
