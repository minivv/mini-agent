"""
Phase 8-mw: LangGraph ReAct Agent — create_agent + SandboxMiddleware + SummarizationMiddleware + SubagentLimitMiddleware + TodoMiddleware

旧方式 (Phase 8 原版):
  手动 StateGraph + AgentState (含 todos)
  process_todos, todo_check, todo_guard 图节点

新方式 (Phase 8-mw):
  SandboxMiddleware: before_agent 注入 workspace_dir 和 skills_dir
  SummarizationMiddleware: after_model 检查 token 数，超阈值时摘要压缩
  SubagentLimitMiddleware: after_model 截断多余的 task 调用
  TodoMiddleware: after_model 处理 write_todos 状态更新 + 上下文丢失检测 + 完成防护
"""

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, ToolMessage, RemoveMessage,
)
from langgraph.checkpoint.memory import MemorySaver

from mini_agent.config import Config
from mini_agent.chat import create_model, build_system_prompt, build_subagent_section, build_todo_section
from mini_agent.context import count_message_tokens, summarize_messages


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
    """在每次 agent 执行前初始化沙箱工作目录和 skills 目录。"""

    def __init__(self, skills_dir: str | None = None):
        super().__init__()
        self.skills_dir = skills_dir

    def before_agent(self, state: dict, config: dict) -> dict:
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        from mini_agent.sandbox import Sandbox
        sandbox = Sandbox(thread_id=thread_id, skills_dir=self.skills_dir)
        state["workspace_dir"] = str(sandbox.workspace)
        if sandbox.skills_dir:
            state["skills_dir"] = str(sandbox.skills_dir)
        return state


class SummarizationMiddleware(AgentMiddleware):
    """在 LLM 回复后检查对话长度，超阈值时用 LLM 摘要压缩。"""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model_sync = create_model(config, streaming=False)

    def after_model(self, state: dict, config: dict) -> dict:
        messages = state.get("messages", [])
        total_tokens = count_message_tokens(messages)
        if total_tokens > self.config.max_context_tokens:
            updates = summarize_messages(
                messages, self.model_sync,
                keep_recent=self.config.keep_recent_messages,
            )
            if updates:
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
        return {}


class SubagentLimitMiddleware(AgentMiddleware):
    """截断 LLM 单次输出中多余的 task 调用。"""

    def __init__(self, limit: int = 3):
        super().__init__()
        self.limit = limit

    def after_model(self, state: dict, config: dict) -> dict:
        from mini_agent.task_tool import truncate_task_calls
        messages = state.get("messages", [])
        updates = truncate_task_calls(messages, self.limit)
        return {"messages": updates} if updates else {}


class TodoMiddleware(AgentMiddleware):
    """处理 write_todos 状态更新 + 上下文丢失检测 + 完成防护。"""

    def after_model(self, state: dict, config: dict) -> dict:
        from mini_agent.todo import process_todos, check_todo_context, guard_todo_completion
        # 1. 从 write_todos 提取 todos 更新 state
        result = process_todos(state)
        # 2. 检查上下文丢失
        todo_check = check_todo_context(state)
        if todo_check:
            result.update(todo_check)
        # 3. 防止提前退出
        todo_guard = guard_todo_completion(state)
        if todo_guard:
            result.update(todo_guard)
        return result


class LangGraphAgent:
    """用 create_agent + Middleware 链构建的 Agent。"""

    def __init__(
        self,
        config: Config,
        tools: list,
        skills_section: str = "",
        skills_dir: str | None = None,
        subagent_enabled: bool = False,
        subagent_limit: int = 3,
        middleware: list | None = None,
    ):
        self.config = config
        self.tools = tools
        self.skills_section = skills_section
        self.skills_dir = skills_dir
        self.subagent_enabled = subagent_enabled
        self.subagent_limit = subagent_limit
        self.middleware = middleware or []
        self.model = create_model(config, streaming=True)
        self.checkpointer = MemorySaver()

        subagent_section = ""
        if subagent_enabled:
            subagent_section = build_subagent_section(subagent_limit)

        system_prompt = build_system_prompt(
            skills_section, subagent_section, build_todo_section(),
        )

        mw_list = [
            SandboxMiddleware(skills_dir=skills_dir),
            SummarizationMiddleware(config),
        ]
        if subagent_enabled:
            mw_list.append(SubagentLimitMiddleware(limit=subagent_limit))
        mw_list.append(TodoMiddleware())
        mw_list.extend(self.middleware)

        self.graph = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=system_prompt,
            middleware=mw_list,
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
