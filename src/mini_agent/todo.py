"""
Phase 8: Todo List — 结构化任务追踪

════════════════════════════════════════════════════════════════════════════
什么是 Todo List?
════════════════════════════════════════════════════════════════════════════

Todo List 是一个让 LLM 自己管理任务进度的机制。

核心思想:
  - 给 LLM 一个 write_todos 工具，它可以在工作中创建和更新任务列表
  - 每次 write_todos 调用都**整体替换**整个列表（不是增删改）
  - todos 存储在 LangGraph state 中，通过 checkpointer 持久化
  - CLI 在每次 state 更新后重新渲染 todo 面板

三个状态:
  - pending:      待处理
  - in_progress:  进行中（同一时间通常只有一个）
  - completed:    已完成

════════════════════════════════════════════════════════════════════════════
智能提醒机制
════════════════════════════════════════════════════════════════════════════

问题 1: 上下文被摘要压缩后，LLM 忘记了还有未完成的 todo
  解决: todo_check 节点检测 write_todos 是否已从历史中丢失，
        如果丢失，注入一条提醒消息

问题 2: LLM 所有 todo 还没做完就想结束（给出最终回复）
  解决: todo_guard 节点检测是否有未完成项，如果有则阻止结束，
        强制 LLM 继续工作（最多提醒 2 次防止死循环）
"""

import logging
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

class Todo(TypedDict):
    """一个 todo 项。"""
    content: str
    status: Literal["pending", "in_progress", "completed"]


# 提醒次数上限，防止 LLM 永远完不成 todo 导致死循环
_MAX_COMPLETION_REMINDERS = 2


# ---------------------------------------------------------------------------
# 工具定义
# ---------------------------------------------------------------------------

@tool("write_todos")
def write_todos(
    todos: list[dict],
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState] = None,
) -> str:
    """Create and manage a structured task list for the current work session.

    Use this tool to plan and track your progress on complex tasks.
    Call with the COMPLETE list every time (it replaces the entire list).
    Each item is a dict with "content" (str) and "status" ("pending"/"in_progress"/"completed").
    Only ONE item should be "in_progress" at a time.
    """
    # 注意: @tool 装饰的函数只能返回字符串（变成 ToolMessage），
    # 无法直接修改 LangGraph state。
    # 实际的 state["todos"] 更新由 process_todos 图节点完成，
    # 它从 AIMessage.tool_calls 中提取 write_todos 的参数写入 state。
    return f"Updated todo list with {len(todos)} items"


# ---------------------------------------------------------------------------
# Graph 节点: 从 tool_call 参数提取 todos 并写入 state
# ---------------------------------------------------------------------------

def process_todos(state: dict) -> dict:
    """从最近的 write_todos tool_call 中提取 todos，写入 state。

    为什么需要这个节点?
      @tool 装饰的函数只能返回字符串（ToolMessage），无法直接更新 state。
      所以 write_todos 工具本身只返回确认文本，
      由这个节点在工具执行后从 AIMessage.tool_calls 中提取参数，
      然后通过 state 更新写入 todos 字段。

    流程:
      LLM 输出 AIMessage(tool_calls=[{name: "write_todos", args: {todos: [...]}}])
      → ToolNode 执行 write_todos，返回 ToolMessage("Updated todo list with 3 items")
      → process_todos 节点检查最后的 tool_calls，提取 todos 参数
      → 返回 {"todos": [...]} 更新 state
    """
    messages = state.get("messages", [])
    if not messages:
        return {}

    # 找到最近的 AIMessage，检查是否包含 write_todos 的 tool_call
    last_ai = None
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "ai":
            last_ai = msg
            break

    if last_ai is None or not getattr(last_ai, "tool_calls", None):
        return {}

    # 在 tool_calls 中找 write_todos
    for tc in last_ai.tool_calls:
        if tc.get("name") == "write_todos":
            todos = tc.get("args", {}).get("todos")
            if todos is not None:
                logger.info("Updated state todos: %d items", len(todos))
                return {"todos": todos}

    return {}


# ---------------------------------------------------------------------------
# Graph 节点: 上下文丢失检测
# ---------------------------------------------------------------------------

def check_todo_context(state: dict) -> dict:
    """检测 write_todos 是否已从消息历史中丢失（被摘要压缩掉了）。

    如果 todos 存在于 state 但消息中没有最近的 write_todos 调用，
    注入一条提醒消息让 LLM 记得还有未完成的任务。
    """
    todos = state.get("todos")
    messages = state.get("messages", [])

    # 没有 todos 或者全部完成，不需要提醒
    if not todos:
        return {}
    has_incomplete = any(t.get("status") != "completed" for t in todos)
    if not has_incomplete:
        return {}

    # 检查最近 N 条消息中是否有 write_todos 的 ToolMessage
    recent = messages[-20:] if len(messages) > 20 else messages
    has_write_todos = any(
        isinstance(m, ToolMessage) and m.name == "write_todos"
        for m in recent
    )

    if not has_write_todos:
        logger.info("Todo context lost detected, injecting reminder")
        todos_summary = "\n".join(
            f"  [{'✓' if t.get('status') == 'completed' else '○'}] {t.get('content', '')}"
            for t in todos
        )
        reminder = HumanMessage(
            content=f"[System Reminder] You have an active todo list that was truncated from context. "
                    f"Here are the current items:\n{todos_summary}\n\n"
                    f"Please continue working through these items. "
                    f"Call write_todos to update their status as you progress.",
            name="todo_reminder",
        )
        return {"messages": [reminder]}

    return {}


# ---------------------------------------------------------------------------
# Graph 节点: 提前退出防护
# ---------------------------------------------------------------------------

def guard_todo_completion(state: dict) -> dict:
    """检查 LLM 是否在 todo 未完成时试图结束。

    如果有未完成的 todo 且 LLM 刚输出了不含 tool_calls 的消息，
    注入提醒并强制 LLM 回到 agent 节点继续工作。
    """
    todos = state.get("todos")
    messages = state.get("messages", [])

    if not todos:
        return {}

    has_incomplete = any(t.get("status") != "completed" for t in todos)
    if not has_incomplete:
        return {}

    # 检查是否已经有太多提醒
    reminder_count = sum(
        1 for m in messages
        if isinstance(m, HumanMessage) and getattr(m, "name", None) == "todo_completion_reminder"
    )
    if reminder_count >= _MAX_COMPLETION_REMINDERS:
        logger.info("Max completion reminders reached, allowing exit")
        return {}

    # 检查最后一条 AI 消息是否没有 tool_calls（试图结束）
    last_ai = None
    for m in reversed(messages):
        if hasattr(m, "type") and m.type == "ai":
            last_ai = m
            break

    if last_ai is None:
        return {}

    if not getattr(last_ai, "tool_calls", None):
        logger.info("Premature exit detected, forcing LLM back to work")
        todos_summary = "\n".join(
            f"  [{'✓' if t.get('status') == 'completed' else '○'}] {t.get('content', '')}"
            for t in todos
        )
        reminder = HumanMessage(
            content=f"[System Reminder] You have {sum(1 for t in todos if t.get('status') != 'completed')} "
                    f"incomplete todo item(s). Please continue working:\n{todos_summary}\n\n"
                    f"Call write_todos to update status, then continue with the remaining items.",
            name="todo_completion_reminder",
        )
        return {"messages": [reminder]}

    return {}


# ---------------------------------------------------------------------------
# CLI 渲染
# ---------------------------------------------------------------------------

def render_todo_panel(todos: list[dict] | None) -> str:
    """渲染 todo 面板为 CLI 文本。"""
    if not todos:
        return ""

    symbols = {
        "completed": "✓",
        "in_progress": "→",
        "pending": "○",
    }

    lines = ["┌─── Todo List ───────────────────"]
    completed_count = 0
    for i, todo in enumerate(todos, 1):
        status = todo.get("status", "pending")
        content = todo.get("content", "")
        symbol = symbols.get(status, "?")

        if status == "completed":
            completed_count += 1
            lines.append(f"│ {i}. {symbol} {content}")
        elif status == "in_progress":
            lines.append(f"│ {i}. \033[96m{symbol} {content}\033[0m")
        else:
            lines.append(f"│ {i}. {symbol} {content}")

    total = len(todos)
    lines.append(f"└─── {completed_count}/{total} completed ──")
    return "\n".join(lines)
