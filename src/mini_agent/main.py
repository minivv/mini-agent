"""
Phase 8-mw: CLI 入口 — create_agent + TodoMiddleware + SubagentLimitMiddleware + SandboxMiddleware + SummarizationMiddleware

显示逻辑:
  - 文本 chunk → 逐字打印（打字机效果）
  - tool_call 开始 → 打印工具调用信息
  - ToolMessage → 打印工具返回结果
  - 子 Agent 状态 → 实时打印启动/执行/完成状态
  - Todo List → 每轮对话结束后渲染 todo 面板
"""

import sys
from langchain_core.messages import AIMessageChunk, ToolMessage

from mini_agent.config import Config
from mini_agent.agent import LangGraphAgent
from mini_agent.context import count_message_tokens
from mini_agent.mcp_client import load_mcp_tools
from mini_agent.skills import load_skills, get_skills_prompt_section
from mini_agent.tools import ALL_TOOLS, BUILTIN_TOOLS, SANDBOX_TOOLS
from mini_agent.chat import build_subagent_section
from mini_agent.todo import write_todos, render_todo_panel
import json as _json


def _extract_todos_from_chunks(
    tool_names: dict[int, str],
    tool_args: dict[int, str],
) -> list[dict] | None:
    """从流式累积的 tool_call_chunks 中提取 write_todos 的 todos 参数。

    流式输出中，LLM 的 tool_call 被拆成多个 chunk 分片到达:
      chunk 1: {index: 0, name: "write_todos"}
      chunk 2: {index: 0, args: '{"todos'}
      chunk 3: {index: 0, args: '": [{"content": "...", "status": "..."}]}'}

    这个函数按 index 拼接 args 片段，找到 write_todos 的调用，
    解析 JSON 提取 todos 列表。
    """
    for idx, name in tool_names.items():
        if name == "write_todos" and idx in tool_args:
            try:
                args = _json.loads(tool_args[idx])
                todos = args.get("todos")
                if isinstance(todos, list) and todos:
                    return todos
            except (_json.JSONDecodeError, TypeError):
                pass
    return None


def print_banner() -> None:
    """Phase 8 欢迎信息。"""
    print()
    print("╔══════════════════════════════════════════╗")
    print("║        Mini Agent - Phase 8              ║")
    print("║  Todo List: 结构化任务追踪                ║")
    print("╠══════════════════════════════════════════╣")
    print("║  命令:                                   ║")
    print("║    /reset   — 新的对话 (新 thread_id)    ║")
    print("║    /tools   — 查看可用工具               ║")
    print("║    /skills  — 查看可用技能               ║")
    print("║    /agents  — 查看可用子 Agent            ║")
    print("║    /todos   — 查看当前任务列表            ║")
    print("║    /state   — 查看当前状态               ║")
    print("║    /tokens  — 查看 token 使用情况        ║")
    print("║    /quit    — 退出                       ║")
    print("║  试试:                                   ║")
    print("║    帮我实现一个快排算法并测试             ║")
    print("║    一步一步分析这段代码的问题             ║")
    print("╚══════════════════════════════════════════╝")
    print()


def handle_command(cmd: str, agent: LangGraphAgent, thread_id: list, mcp_tools: list = None, skills: list = None, subagent_configs: list = None) -> bool:
    """处理斜杠命令。"""
    if mcp_tools is None:
        mcp_tools = []
    if skills is None:
        skills = []
    cmd = cmd.strip().lower()

    if cmd in ("/quit", "/exit", "/q"):
        print("再见！")
        return False

    if cmd == "/reset":
        import uuid
        thread_id[0] = uuid.uuid4().hex[:8]
        print(f"[对话已重置，新 thread_id: {thread_id[0]}]")
        return True

    if cmd == "/tools":
        print("━━━ 内置工具 ━━━")
        for t in BUILTIN_TOOLS:
            desc = (t.description or "").split("\n")[0]
            print(f"  • {t.name}: {desc}")
        print("━━━ 沙箱工具 ━━━")
        for t in SANDBOX_TOOLS:
            desc = (t.description or "").split("\n")[0]
            print(f"  • {t.name}: {desc}")
        if mcp_tools:
            print("━━━ MCP 工具 ━━━")
            for t in mcp_tools:
                desc = (t.description or "").split("\n")[0]
                print(f"  • {t.name}: {desc}")
        return True

    if cmd == "/skills":
        if not skills:
            print("未发现技能")
        else:
            print("━━━ 可用技能 ━━━")
            for s in skills:
                print(f"  • {s.name}: {s.description}")
        return True

    if cmd == "/agents":
        if not subagent_configs:
            print("未启用子 Agent 系统")
        else:
            print("━━━ 可用子 Agent ━━━")
            for c in subagent_configs:
                desc = c.description.replace("\n", " ").strip()[:80]
                print(f"  • {c.name}: {desc}")
                print(f"    工具: {c.tools or '全部'}")
                print(f"    最大轮次: {c.max_turns}, 超时: {c.timeout_seconds}s")
        return True

    if cmd == "/todos":
        state = agent.get_state(thread_id[0])
        if state and state.values:
            todos = state.values.get("todos")
            if todos:
                print(render_todo_panel(todos))
            else:
                print("当前没有任务列表")
        else:
            print("当前对话无历史状态")
        return True

    if cmd == "/state":
        state = agent.get_state(thread_id[0])
        if state and state.values:
            msg_count = len(state.values.get("messages", []))
            print(f"thread_id: {thread_id[0]}")
            print(f"消息数: {msg_count}")
        else:
            print("当前对话无历史状态")
        return True

    if cmd == "/tokens":
        state = agent.get_state(thread_id[0])
        if state and state.values:
            messages = state.values.get("messages", [])
            total = count_message_tokens(messages)
            print(f"消息数: {len(messages)}")
            print(f"Token 数: {total}")
            print(f"摘要阈值: {agent.config.max_context_tokens}")
            print(f"保留最近: {agent.config.keep_recent_messages} 条")
        else:
            print("当前对话无历史状态")
        return True

    print(f"未知命令: {cmd}")
    return True


def main() -> None:
    """Phase 7 主循环。"""
    # 1. 加载配置
    try:
        config = Config()
        config.validate()
    except ValueError as e:
        print(f"配置错误: {e}")
        sys.exit(1)

    # 2. 加载 Skills
    from pathlib import Path
    skills_dir = Path(__file__).parent.parent.parent / "skills"
    skills = load_skills(skills_dir)
    skills_section = get_skills_prompt_section(skills, skills_dir)
    if skills:
        print(f"已加载 {len(skills)} 个技能: {', '.join(s.name for s in skills)}")
    else:
        print("未发现技能（在 skills/ 目录下添加 SKILL.md）")

    # 3. 加载 MCP 工具（从 mcp_config.json）
    print("正在连接 MCP 服务器...")
    mcp_tools = load_mcp_tools()
    if mcp_tools:
        print(f"已加载 {len(mcp_tools)} 个 MCP 工具")
    else:
        print("未配置 MCP 服务器（编辑 mcp_config.json 添加）")

    # 4. 加载子 Agent 配置
    from mini_agent.subagents.registry import list_subagents as list_subagent_configs
    subagent_configs = list_subagent_configs()
    subagent_enabled = True
    subagent_section = build_subagent_section(max_concurrent=3)
    print(f"已启用子 Agent 系统 ({len(subagent_configs)} 个类型): {', '.join(c.name for c in subagent_configs)}")

    # 5. 创建 task 工具
    from mini_agent.task_tool import create_task_tool

    # 状态回调: 子 Agent 执行时实时打印到 CLI
    def status_callback(message: str) -> None:
        print(f"\n{message}")
        print("AI: ", end="", flush=True)

    # all_tools 不含 task (task 单独创建并管理)
    all_tools_no_task = ALL_TOOLS + mcp_tools

    task_tool = create_task_tool(
        parent_config=config,
        all_tools=all_tools_no_task,
        status_callback=status_callback,
        skills_dir=str(skills_dir),
        skills_section=skills_section,
    )

    # 6. 创建 LangGraph Agent (绑定全部工具: 内置 + 沙箱 + MCP + task + write_todos)
    all_tools = all_tools_no_task + [task_tool, write_todos]
    agent = LangGraphAgent(
        config, all_tools,
        skills_section=skills_section,
        skills_dir=str(skills_dir),
        subagent_enabled=subagent_enabled,
        subagent_limit=3,
    )

    # 7. 对话状态 (用可变列表来在循环中切换 thread_id)
    import uuid
    thread_id = [uuid.uuid4().hex[:8]]

    print_banner()
    print(f"模型: {config.model}")
    tool_names = [t.name for t in all_tools]
    print(f"工具: {', '.join(tool_names)}")
    print()

    # 8. 对话循环
    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        # 命令处理
        if user_input.startswith("/"):
            if not handle_command(user_input, agent, thread_id, mcp_tools, skills, subagent_configs):
                break
            print()
            continue

        # ──────────────────────────────────────────────────────────
        # Phase 8: 流式执行，区分 LLM 文本、Tool 事件、Todo 面板
        # ──────────────────────────────────────────────────────────
        print()
        print("AI: ", end="", flush=True)

        is_first_chunk = True
        in_tool_call = False

        # 累积 tool_call 的参数片段，用于实时渲染 write_todos 面板
        # 流式输出中 tool_call_chunks 的 args 是分片到达的:
        #   chunk 1: {"todos"
        #   chunk 2: ": [{"content"
        #   chunk 3": ": "写代码", "status": "pending"}]}
        # 需要按 index 拼接完整 JSON 后才能解析
        pending_tool_args: dict[int, str] = {}  # index → 累积的 args JSON 字符串
        pending_tool_names: dict[int, str] = {}  # index → 工具名

        try:
            for token_text, msg_chunk, metadata in agent.stream(user_input, thread_id[0]):
                node = metadata.get("langgraph_node", "")

                # --- LLM 产出的 chunk ---
                if node == "model":
                    if token_text and isinstance(token_text, str):
                        print(token_text, end="", flush=True)
                        in_tool_call = False

                    tc_chunks = getattr(msg_chunk, "tool_call_chunks", None)
                    if tc_chunks:
                        for tc in tc_chunks:
                            if not isinstance(tc, dict):
                                continue
                            idx = tc.get("index", 0)

                            # 累积工具名
                            if tc.get("name"):
                                pending_tool_names[idx] = tc["name"]
                                if tc["name"] != "write_todos":
                                    if not is_first_chunk:
                                        print()
                                    print(f"  🔧 调用工具: {tc['name']}")
                                in_tool_call = True

                            # 累积参数 JSON 片段
                            if tc.get("args"):
                                pending_tool_args[idx] = pending_tool_args.get(idx, "") + tc["args"]

                    is_first_chunk = False

                # --- 工具执行结果 ---
                elif node == "tools":
                    if isinstance(msg_chunk, ToolMessage):
                        tool_name = getattr(msg_chunk, "name", "")

                        # write_todos: 从累积的参数中解析 todos 并渲染面板
                        if tool_name == "write_todos":
                            todos = _extract_todos_from_chunks(
                                pending_tool_names, pending_tool_args,
                            )
                            if todos:
                                panel = render_todo_panel(todos)
                                if panel:
                                    print(f"\n{panel}")
                                print("AI: ", end="", flush=True)
                            else:
                                content = str(msg_chunk.content)
                                print(f"  📋 {content}")
                        else:
                            content = str(msg_chunk.content)
                            if len(content) > 200:
                                content = content[:200] + "..."
                            print(f"  📋 工具结果: {content}")

                        # 每个工具结果后清除已累积的数据
                        pending_tool_args.clear()
                        pending_tool_names.clear()

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[错误: {e}]")

        print("\n")


if __name__ == "__main__":
    main()
