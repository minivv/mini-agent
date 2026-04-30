"""
Phase 7: Task 委派工具 + 并发限制

task 工具是 lead agent 委派子任务的核心入口。

流程:
  1. LLM 输出 tool_call: task(description, prompt, subagent_type)
  2. task_tool 解析子 Agent 配置
  3. 创建 SubagentExecutor，启动后台执行
  4. 轮询等待结果 (每 5 秒)
  5. 通过 status_callback 实时推送状态给 CLI
  6. 返回最终结果给 lead agent

truncate_task_calls:
  截断 LLM 单次输出中多余的 task 调用 (超过 MAX_CONCURRENT 的丢弃)。
"""

import logging
import time
from typing import Annotated

from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState

from mini_agent.config import Config
from mini_agent.subagents.executor import (
    SubagentExecutor,
    SubagentStatus,
    cleanup_background_task,
    get_background_task_result,
)
from mini_agent.subagents.registry import get_subagent_config, get_subagent_names

logger = logging.getLogger(__name__)


def create_task_tool(
    parent_config: Config,
    all_tools: list,
    status_callback=None,
    workspace_dir: str = "",
    skills_dir: str = "",
    skills_section: str = "",
):
    """创建 task 工具的工厂函数。

    Args:
        parent_config: 父 Agent 的 Config
        all_tools: 所有可用工具列表 (会被子 Agent 过滤使用)
        status_callback: 状态回调函数 (str) -> None，用于 CLI 实时输出
        workspace_dir: 工作目录
        skills_dir: Skills 目录
        skills_section: Skills 提示词段
    """

    @tool("task", parse_docstring=True)
    def task(
        description: str,
        prompt: str,
        subagent_type: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict, InjectedState] = None,
        max_turns: int | None = None,
    ) -> str:
        """Delegate a task to a specialized subagent that runs in its own context.

        Subagents help you:
        - Preserve context by keeping exploration and implementation separate
        - Handle complex multi-step tasks autonomously
        - Execute commands or operations in isolated contexts

        Built-in subagent types:
        - **general-purpose**: A capable agent for complex, multi-step tasks.
          Use when the task requires complex reasoning or multiple dependent steps.
        - **bash**: Command execution specialist. Use for series of bash commands,
          build/test/deploy operations, or verbose terminal output.

        When to use:
        - Complex tasks requiring multiple steps or tools
        - Tasks that produce verbose output
        - Parallel research or exploration tasks

        When NOT to use:
        - Simple, single-step operations (use tools directly)

        Args:
            description: Short (3-5 word) description for logging.
            prompt: The task description. Be specific and clear.
            subagent_type: Type of subagent to use.
            max_turns: Optional max turns. Defaults to subagent's configured max.
        """
        config = get_subagent_config(subagent_type)
        if config is None:
            available = ", ".join(get_subagent_names())
            return f"Error: Unknown subagent type '{subagent_type}'. Available: {available}"

        if max_turns is not None:
            from dataclasses import replace
            config = replace(config, max_turns=max_turns)

        thread_id = "default"
        actual_workspace = workspace_dir
        if state:
            thread_id = state.get("thread_id", "default")
            actual_workspace = state.get("workspace_dir", workspace_dir)

        executor = SubagentExecutor(
            config=config,
            tools=all_tools,
            parent_config=parent_config,
            workspace_dir=actual_workspace,
            skills_dir=skills_dir,
            thread_id=thread_id,
            skills_section=skills_section,
        )

        task_id = executor.execute_async(prompt, task_id=tool_call_id)

        if status_callback:
            status_callback(f"  🚀 子任务启动: {description} ({subagent_type}) [task_id={task_id}]")

        poll_count = 0
        last_status = None
        last_message_count = 0
        max_poll_count = (config.timeout_seconds + 60) // 5

        logger.info("Started task %s (subagent=%s, timeout=%ds)", task_id, subagent_type, config.timeout_seconds)

        try:
            while True:
                result = get_background_task_result(task_id)

                if result is None:
                    if status_callback:
                        status_callback(f"  ❌ 子任务 {task_id} 消失")
                    cleanup_background_task(task_id)
                    return f"Error: Task {task_id} disappeared"

                if result.status != last_status:
                    logger.info("Task %s status: %s", task_id, result.status.value)
                    last_status = result.status

                current_msg_count = len(result.ai_messages)
                if current_msg_count > last_message_count and status_callback:
                    for i in range(last_message_count, current_msg_count):
                        msg = result.ai_messages[i]
                        content = msg.get("content", "")
                        if isinstance(content, str) and content:
                            preview = content[:100] + "..." if len(content) > 100 else content
                            status_callback(f"  📝 [{subagent_type}] 消息 #{i+1}: {preview}")
                    last_message_count = current_msg_count

                if result.status == SubagentStatus.COMPLETED:
                    if status_callback:
                        status_callback(f"  ✅ 子任务完成: {description}")
                    cleanup_background_task(task_id)
                    return f"Task Succeeded. Result: {result.result}"

                elif result.status == SubagentStatus.FAILED:
                    if status_callback:
                        status_callback(f"  ❌ 子任务失败: {result.error}")
                    cleanup_background_task(task_id)
                    return f"Task failed. Error: {result.error}"

                elif result.status == SubagentStatus.CANCELLED:
                    if status_callback:
                        status_callback(f"  🛑 子任务已取消")
                    cleanup_background_task(task_id)
                    return "Task cancelled."

                elif result.status == SubagentStatus.TIMED_OUT:
                    if status_callback:
                        status_callback(f"  ⏰ 子任务超时: {result.error}")
                    cleanup_background_task(task_id)
                    return f"Task timed out. Error: {result.error}"

                time.sleep(5)
                poll_count += 1

                if poll_count > max_poll_count:
                    if status_callback:
                        status_callback(f"  ⏰ 轮询超时")
                    return f"Task polling timed out. Status: {result.status.value}"

        except Exception as e:
            logger.exception("Task %s polling error", task_id)
            cleanup_background_task(task_id)
            return f"Task error: {e}"

    return task


# ---------------------------------------------------------------------------
# SubagentLimitMiddleware (作为普通函数，在 agent.py 图节点中调用)
# ---------------------------------------------------------------------------

MIN_SUBAGENT_LIMIT = 2
MAX_SUBAGENT_LIMIT = 4


def _clamp_subagent_limit(value: int) -> int:
    return max(MIN_SUBAGENT_LIMIT, min(MAX_SUBAGENT_LIMIT, value))


def truncate_task_calls(
    messages: list,
    max_concurrent: int = 3,
) -> list | None:
    """截断多余的 task 调用。

    检查最后一条 AIMessage 的 tool_calls，
    如果 task 类型的调用超过 max_concurrent，只保留前 max_concurrent 个。

    Returns:
        更新后的消息列表 (如果需要截断)，否则 None。
    """
    if not messages:
        return None

    last_msg = messages[-1]
    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        return None

    tool_calls = last_msg.tool_calls
    task_indices = [i for i, tc in enumerate(tool_calls) if tc.get("name") == "task"]

    if len(task_indices) <= max_concurrent:
        return None

    indices_to_drop = set(task_indices[max_concurrent:])
    truncated = [tc for i, tc in enumerate(tool_calls) if i not in indices_to_drop]

    dropped = len(indices_to_drop)
    logger.warning("Truncated %d excess task call(s) (limit: %d)", dropped, max_concurrent)

    from langchain_core.messages import AIMessage, RemoveMessage
    updated_msg = AIMessage(
        content=last_msg.content,
        tool_calls=truncated,
        id=last_msg.id,
        additional_kwargs=last_msg.additional_kwargs,
    )

    return [RemoveMessage(id=last_msg.id), updated_msg]
