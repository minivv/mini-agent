"""
Phase 3-mw: LangGraph ReAct Agent — create_agent + SandboxMiddleware 版

旧方式 (Phase 3 原版):
  手动 StateGraph + AgentState + _build_graph
  stream()/invoke() 中手动创建 Sandbox 注入 workspace_dir

新方式 (Phase 3-mw):
  from langchain.agents import create_agent
  SandboxMiddleware 在 before_agent 中注入 workspace_dir
"""

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
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


class SandboxMiddleware(AgentMiddleware):
    """在每次 agent 执行前初始化沙箱工作目录。"""

    def before_agent(self, state: dict, config: dict) -> dict:
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        from mini_agent.sandbox import Sandbox
        sandbox = Sandbox(thread_id=thread_id)
        state["workspace_dir"] = str(sandbox.workspace)
        return state


class LangGraphAgent:
    """用 create_agent + SandboxMiddleware 构建的 Agent。"""

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
            middleware=[SandboxMiddleware()] + self.middleware,
            checkpointer=self.checkpointer,
        )

    def stream(self, user_input: str, thread_id: str = "default", debug: bool = False):
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
