# LangGraph 四个核心概念

用「点外卖」类比来理解。

---

## 一个完整的例子走一遍

用户输入："算一下 100/3 乘以 6"

```

第 1 轮:
  START → agent 节点
    LLM 收到消息，想了一下：我需要先算 100/3
    LLM 输出 AIMessage(tool_calls=[{name:"calculator", args:{expression:"100/3"}}])

  tools_condition 判断: 有 tool_calls → 走 tools 节点

  tools 节点:
    执行 calculator("100/3") → 返回 "100/3 = 33.333..."
    结果作为 ToolMessage 加到对话里

  箭头指向 agent

第 2 轮:
  agent 节点:
    LLM 看到工具结果 "100/3 = 33.333..."
    继续想：现在乘以 6
    LLM 输出 AIMessage(tool_calls=[{name:"calculator", args:{expression:"33.333...*6"}}])

  tools_condition: 有 tool_calls → 走 tools 节点

  tools 节点:
    执行 calculator("33.333...*6") → 返回 "= 200.0"

第 3 轮:
  agent 节点:
    LLM 看到 "= 200.0"
    想：已经算完了，给用户最终回答
    LLM 输出 AIMessage(content="答案是 200")  ← 没有 tool_calls！

  tools_condition: 没有 tool_calls → END（结束）

```

---

## 翻译成 LangGraph 的四个概念

上面的流程用 LangGraph 来表达，就是四个概念：

### 1. StateGraph — 一张"流程图"

```python
# Phase 2 的图只有 2 个节点、3 条线：

builder = StateGraph(AgentState)   # 拿一张白纸，准备画流程图

#   ┌─────────┐      ┌──────────┐
#   │  agent  │ ←→   │  tools   │
#   │ (大脑)  │      │ (双手)   │
#   └────┬────┘      └──────────┘
#        ↓ 没有工具调用时
#      END
```

### 2. Nodes（节点）— 流程图上的"步骤"

```python
# 节点 1: agent — 你的大脑，负责"想"
def call_model(state):
    response = model.invoke(state["messages"])  # 把对话发给 LLM
    return {"messages": [response]}              # LLM 的回复加到对话里

builder.add_node("agent", call_model)  # 给这个步骤起名叫 "agent"

# 节点 2: tools — 你的双手，负责"做"
builder.add_node("tools", ToolNode(tools))  # 执行计算器、查时间等
```

### 3. Edges（边）— 步骤之间的"箭头"

```python
# 普通箭头：总是从 A 到 B
builder.add_edge(START, "agent")   # 开始 → 大脑思考
builder.add_edge("tools", "agent") # 工具执行完 → 大脑继续想

# 条件箭头：大脑想完之后，走哪条路？
builder.add_conditional_edges("agent", tools_condition)
#                              ↑            ↑
#                         从哪出发     用哪个规则判断
```

### 4. tools_condition — 一个"判断规则"

`tools_condition` 做的事情极其简单：

```python
# 伪代码，这就是 tools_condition 的完整逻辑：
def tools_condition(state):
    最后一条消息 = state["messages"][-1]

    if 最后一条消息里有 tool_calls:    # LLM 说 "我要用计算器"
        return "tools"                  # → 走 tools 节点
    else:                               # LLM 说 "答案是 200"
        return END                      # → 结束
```

---

## 对比 Phase 1：为什么需要图？

Phase 1 的代码是一个简单的 `for` 循环：

```python
# Phase 1: 只能聊天，不能做事
while True:
    用户输入 → LLM 回复 → 显示回复
```

Phase 2 用图来实现了 **"想 → 做 → 想 → 做 → ..."** 的循环：

```python
# Phase 2: LLM 可以自己决定什么时候用工具
agent(想) → 有工具要调用? → tools(做) → agent(想) → 没有? → 结束
```

关键是：**LLM 自主决定**什么时候调用工具、调用哪个、传什么参数。不是你写 if/else，而是 LLM 根据对话内容自己判断。

---

## 对应到本项目代码

```python
# agent.py 第 145-170 行

builder = StateGraph(AgentState)              # ① 拿张白纸

def call_model(state):                        # ② 定义"想"这个步骤
    model_with_tools = self.model.bind_tools(self.tools)
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

builder.add_node("agent", call_model)         # ③ 把步骤画到纸上，取名"agent"
builder.add_node("tools", ToolNode(self.tools))  # ④ 把"做"也画上去

builder.add_edge(START, "agent")              # ⑤ 画箭头：开始→大脑
builder.add_conditional_edges("agent", tools_condition)  # ⑥ 画判断箭头
builder.add_edge("tools", "agent")            # ⑦ 画箭头：手做完→大脑继续

self.graph = builder.compile(checkpointer=MemorySaver())  # ⑧ 图完成
```

---

## 用 LangSmith 对照看

运行一次对话后，打开 [smith.langchain.com](https://smith.langchain.com) 看 trace：

```

时间 →
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. agent 节点（LLM 大脑想了一下）
   │
   输入: "用户说：算一下 100/3"
   输出: "我要调用 calculator 工具，参数是 100/3"  ← 这是一个 tool_call
   │
2. tools 节点（执行 calculator）
   │
   输入: calculator(expression="100/3")
   输出: "100/3 = 33.33..."
   │
3. agent 节点（LLM 看到结果，组织回答）
   │
   输入: "工具返回了 33.33..."
   输出: "100除以3等于33.33..."

```

| 概念 | 一句话 | 在 trace 里长什么样 |
|------|--------|-------------------|
| **StateGraph** | 整条时间线的**图纸** | 你看不到图纸本身，但看到的就是图纸跑出来的结果 |
| **agent 节点** | LLM 的**一次思考** | trace 里标着 `agent` 的那行 |
| **tools 节点** | 工具的**一次执行** | trace 里标着 `tools` 的那行 |
| **tools_condition** | **判断规则**：LLM 想用工具就去 tools，否则结束 | trace 里步骤 1 之后为什么去了步骤 2，就是它在起作用 |
| **ReAct** | **循环**：想→做→想→做→... | trace 里 1→2→3 这个流程本身就是 ReAct |
