"""
Phase 5: CLI 入口 — 支持 Tool Calling + MCP 的流式对话

显示逻辑:
  - 文本 chunk → 逐字打印（打字机效果）
  - tool_call 开始 → 打印工具调用信息
  - ToolMessage → 打印工具返回结果
  - 工具执行后继续的文本 → 继续逐字打印
"""

import sys
from langchain_core.messages import AIMessageChunk, ToolMessage

from mini_agent.config import Config
from mini_agent.agent import LangGraphAgent
from mini_agent.context import count_message_tokens
from mini_agent.mcp_client import load_mcp_tools
from mini_agent.tools import ALL_TOOLS, BUILTIN_TOOLS, SANDBOX_TOOLS


def print_banner() -> None:
    """Phase 5 欢迎信息。"""
    print()
    print("╔══════════════════════════════════════════╗")
    print("║        Mini Agent - Phase 5              ║")
    print("║  MCP 协议: 连接外部工具服务器             ║")
    print("╠══════════════════════════════════════════╣")
    print("║  命令:                                   ║")
    print("║    /reset   — 新的对话 (新 thread_id)    ║")
    print("║    /tools   — 查看可用工具               ║")
    print("║    /state   — 查看当前状态               ║")
    print("║    /tokens  — 查看 token 使用情况        ║")
    print("║    /quit    — 退出                       ║")
    print("║  试试:                                   ║")
    print("║    用 MCP 工具读取 /tmp 下的文件         ║")
    print("║    帮我列出 /tmp 目录的内容              ║")
    print("╚══════════════════════════════════════════╝")
    print()


def handle_command(cmd: str, agent: LangGraphAgent, thread_id: list, mcp_tools: list = None) -> bool:
    """处理斜杠命令。"""
    if mcp_tools is None:
        mcp_tools = []
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
    """Phase 5 主循环。"""
    # 1. 加载配置
    try:
        config = Config()
        config.validate()
    except ValueError as e:
        print(f"配置错误: {e}")
        sys.exit(1)

    # 2. 加载 MCP 工具（从 mcp_config.json）
    print("正在连接 MCP 服务器...")
    mcp_tools = load_mcp_tools()
    if mcp_tools:
        print(f"已加载 {len(mcp_tools)} 个 MCP 工具")
    else:
        print("未配置 MCP 服务器（编辑 mcp_config.json 添加）")

    # 3. 创建 LangGraph Agent (绑定全部工具: 内置 + 沙箱 + MCP)
    all_tools = ALL_TOOLS + mcp_tools
    agent = LangGraphAgent(config, all_tools)

    # 4. 对话状态 (用可变列表来在循环中切换 thread_id)
    import uuid
    thread_id = [uuid.uuid4().hex[:8]]  # 短 ID，方便展示

    print_banner()
    print(f"模型: {config.model}")
    tool_names = [t.name for t in ALL_TOOLS] + [t.name for t in mcp_tools]
    print(f"工具: {', '.join(tool_names)}")
    print()

    # 4. 对话循环
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
            if not handle_command(user_input, agent, thread_id, mcp_tools):
                break
            print()
            continue

        # ──────────────────────────────────────────────────────────
        # Phase 2 核心: 流式执行，区分 LLM 文本 和 Tool 事件
        # ──────────────────────────────────────────────────────────
        print()
        print("AI: ", end="", flush=True)

        is_first_chunk = True   # 第一个文本 chunk 前不加空行
        in_tool_call = False    # 当前是否正在处理 tool_call

        try:
            # agent.stream() → (token_text, AIMessageChunk, metadata)
            for token_text, msg_chunk, metadata in agent.stream(user_input, thread_id[0]):
                node = metadata.get("langgraph_node", "")

                # --- LLM 产出的 chunk ---
                if node == "agent":
                    # 逐 token 打字机效果
                    if token_text and isinstance(token_text, str):
                        print(token_text, end="", flush=True)
                        in_tool_call = False

                    # tool_call 检测 (从 AIMessageChunk 中)
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
