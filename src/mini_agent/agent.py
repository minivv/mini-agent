"""
Phase 2-mw: LangGraph ReAct Agent — create_agent + Middleware 版

════════════════════════════════════════════════════════════════════════════
旧 vs 新
════════════════════════════════════════════════════════════════════════════

旧方式 (Phase 2 原版):
  builder = StateGraph(AgentState)
  builder.add_node("agent", call_model)       # 手动定义节点
  builder.add_node("tools", ToolNode(tools))   # 手动定义节点
  builder.add_edge(START, "agent")            # 手动连线
  builder.add_conditional_edges("agent", tools_condition)  # 手动条件边
  builder.add_edge("tools", "agent")
  graph = builder.compile(checkpointer=...)

新方式 (Phase 2-mw):
  from langchain.agents import create_agent
  graph = create_agent(
      model=model,
      tools=tools,
      system_prompt=prompt,
      middleware=[],        # 扩展点: 后续 Phase 通过 middleware 注入功能
      checkpointer=checkpointer,
  )

create_agent 内部完成了:
  1. 创建 StateGraph + AgentState (messages, jump_to, structured_response)
  2. 添加 model 节点 (调用 LLM，支持 tool_calling)
  3. 添加 tools 节点 (执行工具)
  4. 添加条件边 (有 tool_calls → tools，没有 → END)
  5. 编译图
  6. 返回 CompiledStateGraph (支持 stream/invoke/get_state)

后续 Phase 通过 middleware 参数注入功能:
  Phase 3: SandboxMiddleware — 沙箱工作目录初始化
  Phase 4: SummarizationMiddleware — 上下文压缩
  Phase 7: SubagentLimitMiddleware — 并发限制
  Phase 8: TodoMiddleware — 任务追踪
"""

from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from mini_agent.config import Config
from mini_agent.chat import create_model, build_system_prompt


def _extract_display_text(chunk) -> str:
    if chunk is None:
        return ""
    if chunk.content:
        if isinstance(chunk.content, str):
            return chunk.content
        if isinstance(chunk.content, list):
            return "".join(str(c) for c in chunk.content)
    reasoning = chunk.additional_kwargs.get("reasoning_content", "")
    if reasoning:
        return reasoning
    return ""


class LangGraphAgent:
    """用 create_agent 构建的 Agent，支持 middleware 扩展。"""

    def __init__(self, config: Config, tools: list, middleware: list | None = None):
        self.config = config
        self.tools = tools
        self.middleware = middleware or []
        self.model = create_model(config, streaming=True)
        self.checkpointer = MemorySaver()

        self.graph = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=build_system_prompt(),
            middleware=self.middleware,
            checkpointer=self.checkpointer,
        )

    def stream(self, user_input: str, thread_id: str = "default", debug: bool = False):
        """流式执行。产出 (token_text, msg_chunk, metadata)。"""
        config = {"configurable": {"thread_id": thread_id}}

        try:
            state_snapshot = self.graph.get_state(config)
            has_history = state_snapshot and state_snapshot.values
        except Exception:
            has_history = False

        if not has_history:
            input_messages = [HumanMessage(content=user_input)]
        else:
            input_messages = [HumanMessage(content=user_input)]

        for event in self.graph.stream(
            {"messages": input_messages},
            config=config,
            stream_mode=["messages"],
            subgraphs=False,
        ):
            if debug:
                print(f"\n[DEBUG event: len={len(event)}, types={tuple(type(e).__name__ for e in event)}]")

            msg_chunk = event[1][0] if isinstance(event[1], tuple) else event[1]
            metadata = event[1][1] if isinstance(event[1], tuple) else {}
            token_text = _extract_display_text(msg_chunk)

            yield token_text, msg_chunk, metadata

    def invoke(self, user_input: str, thread_id: str = "default") -> list[BaseMessage]:
        """非流式执行。"""
        config = {"configurable": {"thread_id": thread_id}}
        try:
            state_snapshot = self.graph.get_state(config)
            has_history = state_snapshot and state_snapshot.values
        except Exception:
            has_history = False

        if not has_history:
            input_messages = [HumanMessage(content=user_input)]
        else:
            input_messages = [HumanMessage(content=user_input)]

        result = self.graph.invoke({"messages": input_messages}, config=config)
        return result["messages"]

    def reset(self, thread_id: str = "default") -> None:
        pass

    def get_state(self, thread_id: str = "default") -> Any | None:
        config = {"configurable": {"thread_id": thread_id}}
        try:
            return self.graph.get_state(config)
        except Exception:
            return None
