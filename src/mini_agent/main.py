"""
Phase 7-mw: CLI 入口 — create_agent + SubagentLimitMiddleware + SandboxMiddleware + SummarizationMiddleware

显示逻辑:
  - 文本 chunk → 逐字打印（打字机效果）
  - tool_call 开始 → 打印工具调用信息
  - ToolMessage → 打印工具返回结果
  - 子 Agent 状态 → 实时打印启动/执行/完成状态
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


def print_banner() -> None:
    """Phase 7 欢迎信息。"""
    print()
    print("╔══════════════════════════════════════════╗")
    print("║        Mini Agent - Phase 7              ║")
    print("║  Subagents: 多 Agent 委派系统              ║")
    print("╠══════════════════════════════════════════╣")
    print("║  命令:                                   ║")
    print("║    /reset   — 新的对话 (新 thread_id)    ║")
    print("║    /tools   — 查看可用工具               ║")
    print("║    /skills  — 查看可用技能               ║")
    print("║    /agents  — 查看可用子 Agent            ║")
    print("║    /state   — 查看当前状态               ║")
    print("║    /tokens  — 查看 token 使用情况        ║")
    print("║    /quit    — 退出                       ║")
    print("║  试试:                                   ║")
    print("║    帮我同时研究 Python 和 Rust 的优缺点   ║")
    print("║    审查一下代码质量并运行测试             ║")
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

    # 6. 创建 LangGraph Agent (绑定全部工具: 内置 + 沙箱 + MCP + task)
    all_tools = all_tools_no_task + [task_tool]
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
        # Phase 7 核心: 流式执行，区分 LLM 文本、Tool 事件、子 Agent 状态
        # ──────────────────────────────────────────────────────────
        print()
        print("AI: ", end="", flush=True)

        is_first_chunk = True
        in_tool_call = False

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
                            if isinstance(tc, dict) and tc.get("name"):
                                if not is_first_chunk:
                                    print()
                                print(f"  🔧 调用工具: {tc['name']}")
                                in_tool_call = True

                    is_first_chunk = False

                # --- 工具执行结果 ---
                elif node == "tools":
                    if isinstance(msg_chunk, ToolMessage):
                        content = str(msg_chunk.content)
                        if len(content) > 200:
                            content = content[:200] + "..."
                        print(f"  📋 工具结果: {content}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[错误: {e}]")

        print("\n")


if __name__ == "__main__":
    main()
