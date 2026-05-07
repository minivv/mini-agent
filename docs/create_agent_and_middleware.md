# create_agent + Middleware 模式

## 1. create_agent 是什么

LangGraph 提供的**快捷工厂函数**，一行代码完成 ReAct Agent 的图构建。

### 旧方式（手动搭建）

```python
builder = StateGraph(AgentState)
builder.add_node("agent", call_model)       # 手动添加 LLM 节点
builder.add_node("tools", ToolNode(tools))   # 手动添加工具节点
builder.add_edge(START, "agent")            # 手动连线
builder.add_conditional_edges("agent", tools_condition)  # 手动条件边
builder.add_edge("tools", "agent")          # 手动连线
graph = builder.compile(checkpointer=...)
```

### 新方式（create_agent）

```python
from langchain.agents import create_agent

graph = create_agent(
    model=model,
    tools=tools,
    system_prompt=prompt,
    middleware=[],              # 扩展点: 通过 middleware 注入功能
    checkpointer=checkpointer,
)
```

`create_agent` 内部自动完成:

1. 创建 StateGraph + AgentState（messages, jump_to, structured_response）
2. 添加 **model** 节点（调用 LLM，支持 tool_calling）
3. 添加 **tools** 节点（执行工具）
4. 添加条件边（有 tool_calls → tools，没有 → END）
5. 编译图
6. 返回 CompiledStateGraph（支持 stream / invoke / get_state）

> **注意**: `create_agent` 内部的 LLM 节点名是 `"model"`（不是 `"agent"`），流式事件中 metadata 对应修改。

---

## 2. Middleware 是什么

Middleware 是 `AgentMiddleware` 的子类，提供**钩子函数**，在 agent 执行的不同阶段插入自定义逻辑。

```python
from langchain.agents.middleware import AgentMiddleware

class MyMiddleware(AgentMiddleware):
    def before_agent(self, state, config):
        # 图执行前
        return state

    def after_model(self, state, config):
        # LLM 返回后
        return {}
```

### 钩子执行顺序

```
用户消息
  │
  ▼
before_agent    ← 图执行前（注入初始状态）
  │
  ▼
before_model    ← LLM 调用前
  │
  ▼
wrap_model_call ← 包裹 LLM 调用（可替换模型）
  │
  ▼
after_model     ← LLM 返回后（检查/修改输出）
  │
  ▼
wrap_tool_call  ← 包裹工具调用（可拦截/修改）
  │
  ▼
after_agent     ← 图执行后（最终处理）
  │
  ▼
  输出
```

---

## 3. 实际例子: SandboxMiddleware

### 旧方式（散落在 stream/invoke 中）

```python
class LangGraphAgent:
    def stream(self, user_input, thread_id):
        sandbox = Sandbox(thread_id=thread_id)         # 每次手动创建
        graph_input = {
            "messages": input_messages,
            "workspace_dir": str(sandbox.workspace),    # 手动注入
        }
        for event in self.graph.stream(graph_input, ...):
            ...

    def invoke(self, user_input, thread_id):
        sandbox = Sandbox(thread_id=thread_id)         # 重复代码
        result = self.graph.invoke(
            {"messages": ..., "workspace_dir": str(sandbox.workspace)},
            ...
        )
```

### 新方式（middleware 集中处理）

```python
class SandboxMiddleware(AgentMiddleware):
    def before_agent(self, state, config):
        thread_id = config["configurable"]["thread_id"]
        sandbox = Sandbox(thread_id=thread_id)
        state["workspace_dir"] = str(sandbox.workspace)
        return state

# 注册一次，stream()/invoke() 自动生效
graph = create_agent(
    model=model, tools=tools,
    middleware=[SandboxMiddleware()],
)
```

---

## 4. Middleware 链的组合

多个 middleware 按列表顺序组合，像洋葱一样层层包裹:

```python
middleware = [
    SandboxMiddleware(),           # 第 1 层: 沙箱初始化
    SummarizationMiddleware(),     # 第 2 层: 上下文压缩
    SubagentLimitMiddleware(),     # 第 3 层: 子 Agent 限制
    TodoMiddleware(),              # 第 4 层: Todo 追踪
]
```

- `before_agent` 从外到内执行（先 Sandbox，再 Summarization，...）
- `after_model` 从内到外执行（先 Todo，再 SubagentLimit，...）

类似 Koa/Express 的中间件模式。

---

## 5. 项目中的 Middleware 清单

| Middleware | 钩子 | 功能 |
|-----------|------|------|
| `SandboxMiddleware` | `before_agent` | 注入 workspace_dir 和 skills_dir |
| `SummarizationMiddleware` | `after_model` | token 超阈值时用 LLM 摘要压缩 |
| `SubagentLimitMiddleware` | `after_model` | 截断多余的 task 调用 |
| `TodoMiddleware` | `after_model` | 处理 write_todos + 上下文丢失检测 + 完成防护 |

---

## 6. 为什么用 Middleware

| 对比 | 手动图节点 | Middleware |
|------|-----------|-----------|
| 添加功能 | 改 `_build_graph()`，加节点+边 | 写一个类，加到列表 |
| 复用 | 复制粘贴 | `middleware=[X()]` |
| 可测试 | 需要跑整个图 | 单独测试钩子函数 |
| 关注点分离 | 散落在各处 | 每个 middleware 一个职责 |

**本质**: 把图的构建逻辑和扩展逻辑解耦。`create_agent` 处理标准的 ReAct 循环，middleware 处理各种"旁路"逻辑（注入状态、压缩上下文、限制调用等）。
