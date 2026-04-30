"""
Phase 2: LangGraph ReAct Agent

这是从 Phase 1 到 Phase 2 的核心变化: 从简单的 ChatModel 调用演进为
基于 LangGraph 的 ReAct Agent。

════════════════════════════════════════════════════════════════════════════
LangGraph 核心概念
════════════════════════════════════════════════════════════════════════════

1. StateGraph — Agent 的"蓝图"
   一个 StateGraph 由节点(nodes)和边(edges)组成，定义了 agent 的控制流。
   这比 Phase 1 的 while 循环更强大，因为它支持:
     - 条件分支 (conditional edges)
     - 状态持久化 (checkpointer)
     - 流式事件 (streaming events)
     - 人工干预 (interrupts)

2. State — 在节点间流动的数据
   用 TypedDict 定义，每个字段可以有自己的 reducer。
   比如 messages 用 add_messages (追加而非替换)。

3. Nodes — 执行逻辑的"步骤"
   每个节点是一个函数: 接收 State → 返回 State 的部分更新。
   Phase 2 有两个节点:
     - agent  节点: 调用 LLM (bind_tools 后)，LLM 决定是回复还是调用工具
     - tools  节点: 执行 LLM 请求的工具调用

4. Edges — 节点间的"箭头"
     - Normal edge:  总是从 A 到 B
     - Conditional edge: 根据 State 决定下一个节点
       比如: LLM 输出里有 tool_calls → 去 tools 节点
            没有 tool_calls → 结束

5. Checkpointer — 状态持久化
   MemorySaver 把每次节点的输入/输出存到内存里。
   通过 thread_id 区分不同对话。这就是"多轮对话"在 LangGraph 中的实现方式。

════════════════════════════════════════════════════════════════════════════
ReAct 模式
════════════════════════════════════════════════════════════════════════════

ReAct = Reasoning + Acting
  1. LLM 收到用户消息 → 思考(Reasoning)
  2. 如果需要工具 → 输出 tool_call (Acting)
  3. 工具执行 → 返回结果给 LLM
  4. LLM 基于结果继续思考 → 直到给出最终回答

在 LangGraph 中的实现:
  START → agent → [有 tool_calls?] → tools → agent → [有 tool_calls?] → ... → END
                    ↓ 没有                                      ↓ 没有
"""

from typing import Annotated, TypedDict
from typing import Any

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import (
    BaseMessage, HumanMessage, SystemMessage,
    AIMessage, ToolMessage, RemoveMessage,
)

from mini_agent.config import Config
from mini_agent.chat import create_model, build_system_prompt
from mini_agent.context import count_message_tokens, summarize_messages


# ---------------------------------------------------------------------------
# 辅助: 从 AIMessageChunk 提取可显示文本
# ---------------------------------------------------------------------------
# 不同模型/提供商把 token 文本放在不同字段:
#   - 大多数模型:  chunk.content 直接就是文本
#   - DeepSeek thinking 模型: 文本在 additional_kwargs["reasoning_content"]
#     且 content 通常为空字符串

def _extract_display_text(chunk) -> str:
    """从 AIMessageChunk 中提取可显示的文本 token。"""
    if chunk is None:
        return ""

    # 优先级 1: chunk.content（大多数模型的文本在这里）
    if chunk.content:
        if isinstance(chunk.content, str):
            return chunk.content
        if isinstance(chunk.content, list):
            return "".join(str(c) for c in chunk.content)

    # 优先级 2: additional_kwargs 中的 reasoning_content (DeepSeek thinking)
    reasoning = chunk.additional_kwargs.get("reasoning_content", "")
    if reasoning:
        return reasoning

    return ""


# ---------------------------------------------------------------------------
# State 定义: 什么数据在节点之间流动
# ---------------------------------------------------------------------------
# add_messages 是 LangGraph 内置的 reducer:
#   不是替换 messages 列表，而是把新消息追加到列表末尾。
#   并且它自动合并 AIMessageChunk（把流式碎片拼成完整消息）。

class AgentState(TypedDict, total=False):
    """Agent 的状态定义。每个字段在不同节点之间流动。

    total=False 表示字段可以不存在，首次注入时不会因为缺少字段报错。
    """
    messages: Annotated[list[BaseMessage], add_messages]
    # Phase 3: 沙箱工作目录 (绝对路径)，每个线程有独立的沙箱
    workspace_dir: str
    # Phase 6: Skills 目录路径
    skills_dir: str


# ---------------------------------------------------------------------------
# LangGraphAgent: 基于图的 Agent
# ---------------------------------------------------------------------------

class LangGraphAgent:
    """
    Phase 2 的 Agent: 用 LangGraph 构建的 ReAct Agent。

    用法:
        agent = LangGraphAgent(config, tools)
        for event in agent.stream("帮我算一下 123 * 456"):
            print(event)
    """

    def __init__(self, config: Config, tools: list, skills_section: str = "", skills_dir: str | None = None):
        self.config = config
        self.tools = tools
        self.skills_section = skills_section
        self.skills_dir = skills_dir
        self.model = create_model(config, streaming=True)
        self.model_sync = create_model(config, streaming=False)
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # LangGraph 核心概念 #1: 构建图 (build graph)
    # ------------------------------------------------------------------
    # StateGraph 是图的结构定义。
    # 我们用 .add_node() 添加节点, .add_edge() 和 .add_conditional_edges() 添加边。
    # .compile() 把图编译成一个可执行的 Runnable。

    def _build_graph(self):
        """构建 ReAct Agent 图。"""
        builder = StateGraph(AgentState)

        # --- 节点 1: agent ---
        # bind_tools() 把工具列表注入到 model 中，让 LLM 知道:
        #   "你可以调用这些工具，这是它们的签名。"
        # 当 LLM 想用工具时，AIMessage 里会包含 tool_calls 字段。
        def call_model(state: AgentState):
            model_with_tools = self.model.bind_tools(self.tools)
            response = model_with_tools.invoke(state["messages"])
            return {"messages": [response]}

        builder.add_node("agent", call_model)

        # --- 节点 2: tools ---
        # ToolNode 是 LangGraph 预置的工具执行节点。
        # 它读取上一条 AIMessage 的 tool_calls，逐一执行，返回 ToolMessage 列表。
        builder.add_node("tools", ToolNode(self.tools))

        # --- 节点 3: check_context (Phase 4) ---
        # 在 agent 回复后（或工具执行后），检查对话是否过长。
        # 如果 token 数超过阈值，用 LLM 总结旧消息来压缩上下文。
        # 这样对话可以无限进行下去，不会超出 LLM 的上下文窗口。
        def check_context(state: AgentState):
            messages = state["messages"]
            total_tokens = count_message_tokens(messages)
            if total_tokens > self.config.max_context_tokens:
                updates = summarize_messages(
                    messages, self.model_sync,
                    keep_recent=self.config.keep_recent_messages,
                )
                if updates:
                    # 清理孤立的 ToolMessage:
                    # 摘要删除了 AIMessage（含 tool_calls），但对应的 ToolMessage
                    # 可能还在消息列表里。找到被删除的 AIMessage 的 tool_call ids，
                    # 把引用这些 ids 的 ToolMessage 也删掉。
                    removed_tool_call_ids: set[str] = set()
                    for u in updates:
                        if isinstance(u, RemoveMessage):
                            for msg in messages:
                                if msg.id == u.id and isinstance(msg, AIMessage):
                                    for tc in (msg.tool_calls or []):
                                        if "id" in tc:
                                            removed_tool_call_ids.add(tc["id"])
                    for msg in messages:
                        if isinstance(msg, ToolMessage):
                            tc_id = getattr(msg, "tool_call_id", None)
                            if tc_id and tc_id in removed_tool_call_ids:
                                updates.append(RemoveMessage(id=msg.id))
                return {"messages": updates}
            return {}  # 不需要摘要，返回空更新

        builder.add_node("check_context", check_context)

        # --- 边 ---
        # Entry point: 从 START 开始
        builder.add_edge(START, "agent")

        # 条件边: 检查最后一条消息
        #   如果有 tool_calls → 去 tools 节点
        #   如果没有         → 去 check_context（检查是否需要摘要）
        def route_after_agent(state: AgentState):
            last_msg = state["messages"][-1]
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                return "tools"
            return "check_context"

        builder.add_conditional_edges("agent", route_after_agent)

        # 普通边: tools 执行完后回到 agent（让 LLM 看到工具结果后继续思考）
        builder.add_edge("tools", "agent")

        # 普通边: 上下文检查完后总是结束
        # （check_context 内部判断是否需要摘要，不需要时直接返回空更新）
        builder.add_edge("check_context", END)

        # --- 编译 ---
        # checkpointer 让图在执行后自动保存状态。
        # thread_id 是隔离的 key —— 不同的对话用不同的 thread_id。
        return builder.compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------------------
    # LangGraph 核心概念 #2: stream()
    # ------------------------------------------------------------------
    # graph.stream() 返回一个迭代器，每完成一个"步骤"就产出事件。
    # stream_mode=["messages"] 返回 (message_chunk, metadata)，其中:
    #   - message_chunk: AIMessageChunk (文本或 tool_call 片段)
    #   - metadata:      包含 langgraph_node 标识来源节点
    #
    # LangGraph 的 stream_mode 选项:
    #   "values"   — 每个 super-step 后完整的 State 快照
    #   "messages" — 每个 LLM token 产出 (流式打字机效果)
    #   "updates"  — 每个节点产出的 State 更新 (不包含原始输入)
    #   "custom"   — 通过 StreamWriter 发送的自定义事件

    def stream(self, user_input: str, thread_id: str = "default", debug: bool = False):
        """
        流式执行 agent。

        产出元组: (message_chunk, metadata)
        调用方通过 metadata["langgraph_node"] 来区分别是 LLM token 还是工具结果。
        """
        config = {"configurable": {"thread_id": thread_id}}

        # LangGraph 核心概念 #3: 对话延续
        # -----------------------------------------------------------------
        # 通过 thread_id 和 checkpointer，LangGraph 自动管理对话历史。
        # 第一次调用: state 为空，我们注入 SystemMessage
        # 后续调用: state 已包含之前的消息，只需追加新的 HumanMessage
        try:
            state_snapshot = self.graph.get_state(config)
            has_history = state_snapshot and state_snapshot.values
        except Exception:
            has_history = False

        if not has_history:
            input_messages = [
                SystemMessage(content=build_system_prompt(self.skills_section)),
                HumanMessage(content=user_input),
            ]
        else:
            input_messages = [HumanMessage(content=user_input)]

        # 为当前线程创建沙箱，注入 workspace_dir 和 skills_dir
        from mini_agent.sandbox import Sandbox
        sandbox = Sandbox(thread_id=thread_id, skills_dir=self.skills_dir)
        graph_input = {
            "messages": input_messages,
            "workspace_dir": str(sandbox.workspace),
            "skills_dir": str(sandbox.skills_dir) if sandbox.skills_dir else "",
        }

        # 流式执行
        # stream_mode=["messages"] 产出 LLM token 流和工具结果。
        # LangGraph 1.x 格式 (subgraphs=False):
        #   ("messages", (AIMessageChunk, metadata_dict))
        #     ↑ channel名   ↑ 实际数据
        for event in self.graph.stream(
            graph_input,
            config=config,
            stream_mode=["messages"],
            subgraphs=False,
        ):
            if debug:
                print(f"\n[DEBUG event: len={len(event)}, types={tuple(type(e).__name__ for e in event)}]")

            # 解包: ("messages", (msg_chunk, metadata))
            channel_name = event[0]
            msg_chunk = event[1][0] if isinstance(event[1], tuple) else event[1]
            metadata = event[1][1] if isinstance(event[1], tuple) else {}

            # 从 msg_chunk 提取可显示文本
            # 注意: DeepSeek thinking 模型把文本放在 additional_kwargs["reasoning_content"]
            token_text = _extract_display_text(msg_chunk)

            yield token_text, msg_chunk, metadata

    def invoke(self, user_input: str, thread_id: str = "default") -> list[BaseMessage]:
        """非流式执行，返回最终的消息列表(用于内部调用)。"""
        config = {"configurable": {"thread_id": thread_id}}
        try:
            state_snapshot = self.graph.get_state(config)
            has_history = state_snapshot and state_snapshot.values
        except Exception:
            has_history = False

        if not has_history:
            input_messages = [
                SystemMessage(content=build_system_prompt(self.skills_section)),
                HumanMessage(content=user_input),
            ]
        else:
            input_messages = [HumanMessage(content=user_input)]

        # 注入 workspace_dir 和 skills_dir
        from mini_agent.sandbox import Sandbox
        sandbox = Sandbox(thread_id=thread_id, skills_dir=self.skills_dir)
        result = self.graph.invoke(
            {"messages": input_messages, "workspace_dir": str(sandbox.workspace),
             "skills_dir": str(sandbox.skills_dir) if sandbox.skills_dir else ""},
            config=config,
        )
        return result["messages"]

    def reset(self, thread_id: str = "default") -> None:
        """重置对话: 用新的 thread_id 就相当于新对话。"""
        # MemorySaver 没有直接删除某个 thread 的 API，
        # 最简单的方式是切换 thread_id。调用方可以用 UUID 生成新的。
        pass

    def get_state(self, thread_id: str = "default") -> Any | None:
        """获取当前对话状态（用于调试）。"""
        config = {"configurable": {"thread_id": thread_id}}
        try:
            state = self.graph.get_state(config)
            return state
        except Exception:
            return None
