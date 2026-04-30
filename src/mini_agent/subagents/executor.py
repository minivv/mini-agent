"""
Phase 7: 子 Agent 后台执行引擎

执行引擎的核心设计:

1. 三级线程池
   _scheduler_pool  → 提交调度任务 (3 workers)
   _execution_pool  → 实际执行子 Agent (3 workers)
   _isolated_loop_pool → 在已有 event loop 时隔离执行 (3 workers)

2. 全局任务存储
   _background_tasks: dict[str, SubagentResult]
   用 thread-safe lock 保护，task_tool 通过轮询读取状态

3. 协作式取消
   通过 threading.Event 在 astream 迭代边界检查是否需要取消

4. 超时机制
   scheduler 提交到 execution_pool 时设 Future.timeout，
   超时后设置 cancel_event 通知子 Agent 停止
"""

import asyncio
import logging
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from mini_agent.config import Config
from mini_agent.subagents.config import SubagentConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 状态和结果
# ---------------------------------------------------------------------------

class SubagentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


@dataclass
class SubagentResult:
    """子 Agent 执行结果。"""
    task_id: str
    trace_id: str
    status: SubagentStatus
    result: str | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    ai_messages: list[dict[str, Any]] = field(default_factory=list)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)


# ---------------------------------------------------------------------------
# 全局任务存储
# ---------------------------------------------------------------------------

_background_tasks: dict[str, SubagentResult] = {}
_background_tasks_lock = threading.Lock()


# ---------------------------------------------------------------------------
# 线程池 (三级)
# ---------------------------------------------------------------------------

# 调度池: 接收 execute_async 提交的 run_task 闭包
_scheduler_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="subagent-scheduler-")

# 执行池: 实际运行子 Agent
_execution_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="subagent-exec-")

# 隔离池: 当调用方已有 event loop 时，在新线程中创建隔离的 event loop
_isolated_loop_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="subagent-isolated-")

# 并发上限
MAX_CONCURRENT_SUBAGENTS = 3


# ---------------------------------------------------------------------------
# 工具过滤
# ---------------------------------------------------------------------------

def _filter_tools(
    all_tools: list[BaseTool],
    allowed: list[str] | None,
    disallowed: list[str] | None,
) -> list[BaseTool]:
    """按 allowlist/denylist 过滤工具列表。"""
    filtered = all_tools
    if allowed is not None:
        allowed_set = set(allowed)
        filtered = [t for t in filtered if t.name in allowed_set]
    if disallowed is not None:
        disallowed_set = set(disallowed)
        filtered = [t for t in filtered if t.name not in disallowed_set]
    return filtered


# ---------------------------------------------------------------------------
# 执行器
# ---------------------------------------------------------------------------

class SubagentExecutor:
    """子 Agent 执行器。

    每个 task 调用创建一个 executor 实例，负责:
      1. 创建独立的 LangGraph Agent
      2. 在后台线程中运行
      3. 管理超时和取消
      4. 实时更新 SubagentResult
    """

    def __init__(
        self,
        config: SubagentConfig,
        tools: list[BaseTool],
        parent_config: Config,
        parent_model: str | None = None,
        workspace_dir: str = "",
        skills_dir: str = "",
        thread_id: str = "default",
        trace_id: str | None = None,
        skills_section: str = "",
    ):
        self.config = config
        self.parent_config = parent_config
        self.parent_model = parent_model or parent_config.model
        self.workspace_dir = workspace_dir
        self.skills_dir = skills_dir
        self.thread_id = thread_id
        self.trace_id = trace_id or str(uuid.uuid4())[:8]
        self.skills_section = skills_section

        self.tools = _filter_tools(tools, config.tools, config.disallowed_tools)

        logger.info(
            "[trace=%s] SubagentExecutor: %s with %d tools",
            self.trace_id, config.name, len(self.tools),
        )

    def _create_agent(self):
        """创建子 Agent 实例。

        子 Agent 是一个独立的 LangGraphAgent，拥有:
        - 自己的 model (可能和父 Agent 不同)
        - 过滤后的工具列表 (不含 task 工具)
        - 独立的 system prompt
        - 共享的 workspace 和 skills 目录
        """
        from mini_agent.chat import build_system_prompt
        from mini_agent.agent import LangGraphAgent

        # 子 Agent 使用自己的 system prompt，但保留 skills 信息
        subagent_system = self.config.system_prompt
        if self.skills_section:
            subagent_system += "\n" + self.skills_section

        # 创建子 Agent 的 Config
        if self.config.model == "inherit":
            sub_config = Config(
                api_key=self.parent_config.api_key,
                base_url=self.parent_config.base_url,
                model=self.parent_config.model,
                temperature=self.parent_config.temperature,
                extra_body=self.parent_config.extra_body,
                max_context_tokens=self.parent_config.max_context_tokens,
                keep_recent_messages=self.parent_config.keep_recent_messages,
            )
        else:
            sub_config = Config(
                api_key=self.parent_config.api_key,
                base_url=self.parent_config.base_url,
                model=self.config.model,
                temperature=self.parent_config.temperature,
                extra_body=self.parent_config.extra_body,
                max_context_tokens=self.parent_config.max_context_tokens,
                keep_recent_messages=self.parent_config.keep_recent_messages,
            )

        # 子 Agent 用独立的 checkpointer thread (避免污染父 Agent 状态)
        sub_thread_id = f"{self.thread_id}_sub_{self.trace_id}"

        return LangGraphAgent(
            sub_config,
            self.tools,
            skills_section=subagent_system,
            skills_dir=self.skills_dir if self.skills_dir else None,
            # 使用共享的 workspace
            _sub_thread_id=sub_thread_id,
        )

    def _extract_text_content(self, content) -> str:
        """从消息 content 中提取纯文本。"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and "text" in block:
                    parts.append(block["text"])
            return "\n".join(parts)
        return str(content) if content else ""

    async def _aexecute(self, task: str, result_holder: SubagentResult) -> SubagentResult:
        """异步执行子 Agent。"""
        result = result_holder

        try:
            agent = self._create_agent()

            # 构建子 Agent 的输入
            from mini_agent.sandbox import Sandbox
            sandbox = Sandbox(thread_id=self.thread_id, skills_dir=self.skills_dir or None)

            input_messages = [
                HumanMessage(content=task),
            ]

            graph_input = {
                "messages": input_messages,
                "workspace_dir": str(sandbox.workspace),
                "skills_dir": str(sandbox.skills_dir) if sandbox.skills_dir else "",
            }

            run_config = {
                "configurable": {"thread_id": f"{self.thread_id}_sub_{self.trace_id}"},
                "recursion_limit": self.config.max_turns,
            }

            logger.info(
                "[trace=%s] Subagent %s starting, max_turns=%d",
                self.trace_id, self.config.name, self.config.max_turns,
            )

            # 检查取消
            if result.cancel_event.is_set():
                with _background_tasks_lock:
                    result.status = SubagentStatus.CANCELLED
                    result.error = "Cancelled before start"
                    result.completed_at = datetime.now()
                return result

            # 流式执行，收集 AI 消息
            final_state = None
            async for chunk in agent.graph.astream(
                graph_input, config=run_config, stream_mode="values",
            ):
                if result.cancel_event.is_set():
                    with _background_tasks_lock:
                        result.status = SubagentStatus.CANCELLED
                        result.error = "Cancelled by user"
                        result.completed_at = datetime.now()
                    return result

                final_state = chunk

                # 提取 AI 消息用于实时状态更新
                messages = chunk.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if isinstance(last_msg, AIMessage):
                        msg_dict = last_msg.model_dump()
                        msg_id = msg_dict.get("id")
                        is_dup = False
                        if msg_id:
                            is_dup = any(m.get("id") == msg_id for m in result.ai_messages)
                        else:
                            is_dup = msg_dict in result.ai_messages
                        if not is_dup:
                            result.ai_messages.append(msg_dict)

            # 提取最终结果
            if final_state is None:
                result.result = "No response generated"
            else:
                messages = final_state.get("messages", [])
                last_ai = None
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage):
                        last_ai = msg
                        break

                if last_ai is not None:
                    result.result = self._extract_text_content(last_ai.content)
                elif messages:
                    result.result = self._extract_text_content(
                        getattr(messages[-1], "content", str(messages[-1]))
                    )
                else:
                    result.result = "No response generated"

            with _background_tasks_lock:
                result.status = SubagentStatus.COMPLETED
                result.completed_at = datetime.now()

        except Exception as e:
            logger.exception("[trace=%s] Subagent %s failed", self.trace_id, self.config.name)
            with _background_tasks_lock:
                result.status = SubagentStatus.FAILED
                result.error = str(e)
                result.completed_at = datetime.now()

        return result

    def _execute_in_isolated_loop(self, task: str, result_holder: SubagentResult) -> SubagentResult:
        """在隔离的 event loop 中执行 (用于已有 event loop 的场景)。"""
        try:
            previous_loop = asyncio.get_event_loop()
        except RuntimeError:
            previous_loop = None

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self._aexecute(task, result_holder))
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                if pending:
                    for t in pending:
                        t.cancel()
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            finally:
                try:
                    loop.close()
                finally:
                    asyncio.set_event_loop(previous_loop)

    def execute(self, task: str, result_holder: SubagentResult) -> SubagentResult:
        """同步执行 (包装异步执行)。"""
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and loop.is_running():
                future = _isolated_loop_pool.submit(
                    self._execute_in_isolated_loop, task, result_holder,
                )
                return future.result()

            return asyncio.run(self._aexecute(task, result_holder))
        except Exception as e:
            logger.exception("[trace=%s] Subagent %s execution failed", self.trace_id, self.config.name)
            with _background_tasks_lock:
                result_holder.status = SubagentStatus.FAILED
                result_holder.error = str(e)
                result_holder.completed_at = datetime.now()
            return result_holder

    def execute_async(self, task: str, task_id: str) -> str:
        """在后台启动任务执行。

        流程:
          1. 创建 PENDING 状态的 SubagentResult
          2. 提交到 _scheduler_pool
          3. scheduler 将状态改为 RUNNING，提交到 _execution_pool
          4. execution_pool 带超时执行，超时则设 TIMED_OUT
        """
        if task_id is None:
            task_id = str(uuid.uuid4())[:8]

        result = SubagentResult(
            task_id=task_id,
            trace_id=self.trace_id,
            status=SubagentStatus.PENDING,
        )

        with _background_tasks_lock:
            _background_tasks[task_id] = result

        logger.info(
            "[trace=%s] Subagent %s async start, task_id=%s, timeout=%ds",
            self.trace_id, self.config.name, task_id, self.config.timeout_seconds,
        )

        def run_task():
            with _background_tasks_lock:
                _background_tasks[task_id].status = SubagentStatus.RUNNING
                _background_tasks[task_id].started_at = datetime.now()
                result_holder = _background_tasks[task_id]

            try:
                future: Future = _execution_pool.submit(self.execute, task, result_holder)
                try:
                    exec_result = future.result(timeout=self.config.timeout_seconds)
                    with _background_tasks_lock:
                        _background_tasks[task_id].status = exec_result.status
                        _background_tasks[task_id].result = exec_result.result
                        _background_tasks[task_id].error = exec_result.error
                        _background_tasks[task_id].completed_at = datetime.now()
                        _background_tasks[task_id].ai_messages = exec_result.ai_messages
                except FuturesTimeoutError:
                    logger.error(
                        "[trace=%s] Subagent %s timed out after %ds",
                        self.trace_id, self.config.name, self.config.timeout_seconds,
                    )
                    with _background_tasks_lock:
                        if _background_tasks[task_id].status == SubagentStatus.RUNNING:
                            _background_tasks[task_id].status = SubagentStatus.TIMED_OUT
                            _background_tasks[task_id].error = (
                                f"Timed out after {self.config.timeout_seconds}s"
                            )
                            _background_tasks[task_id].completed_at = datetime.now()
                    result_holder.cancel_event.set()
                    future.cancel()
            except Exception as e:
                logger.exception("[trace=%s] Subagent %s async failed", self.trace_id, self.config.name)
                with _background_tasks_lock:
                    _background_tasks[task_id].status = SubagentStatus.FAILED
                    _background_tasks[task_id].error = str(e)
                    _background_tasks[task_id].completed_at = datetime.now()

        _scheduler_pool.submit(run_task)
        return task_id


# ---------------------------------------------------------------------------
# 全局任务管理 API
# ---------------------------------------------------------------------------

def get_background_task_result(task_id: str) -> SubagentResult | None:
    """获取后台任务结果。"""
    with _background_tasks_lock:
        return _background_tasks.get(task_id)


def request_cancel_background_task(task_id: str) -> None:
    """请求取消后台任务 (协作式取消)。"""
    with _background_tasks_lock:
        result = _background_tasks.get(task_id)
        if result is not None:
            result.cancel_event.set()
            logger.info("Requested cancellation for task %s", task_id)


def cleanup_background_task(task_id: str) -> None:
    """清理已完成的任务，防止内存泄漏。"""
    with _background_tasks_lock:
        result = _background_tasks.get(task_id)
        if result is None:
            return

        is_terminal = result.status in {
            SubagentStatus.COMPLETED,
            SubagentStatus.FAILED,
            SubagentStatus.CANCELLED,
            SubagentStatus.TIMED_OUT,
        }
        if is_terminal or result.completed_at is not None:
            del _background_tasks[task_id]
            logger.debug("Cleaned up background task: %s", task_id)
