"""
Phase 2: CLI 入口 — 支持 Tool Calling 的流式对话

与 Phase 1 的区别:
  - 使用 LangGraphAgent 替代 ChatAgent
  - stream 产出的不只是文本，还有 tool_call 和 tool_result 事件
  - 用户可以看到 agent 何时调用工具、工具返回了什么

LangGraph stream_mode=["messages"] 的产物:
  1. (AIMessageChunk, {"langgraph_node": "agent"})
     - LLM 产出的 token。可能有 tool_call_chunks 在里面。
  2. (ToolMessage, {"langgraph_node": "tools"})
     - 工具执行完成后产出的结果消息。

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
from mini_agent.tools import BUILTIN_TOOLS


def print_banner() -> None:
    """Phase 2 欢迎信息。"""
    print()
    print("╔══════════════════════════════════════════╗")
    print("║       Mini Agent - Phase 2-mw             ║")
    print("║   create_agent + Middleware 模式          ║")
    print("╠══════════════════════════════════════════╣")
    print("║  命令:                                   ║")
    print("║    /reset   — 新的对话 (新 thread_id)    ║")
    print("║    /tools   — 查看可用工具               ║")
    print("║    /state   — 查看当前状态               ║")
    print("║    /quit    — 退出                       ║")
    print("║  试试:                                   ║")
    print("║    帮我算一下 (123 + 456) * 789          ║")
    print("║    现在几点了？                           ║")
    print("║    先算 100/3，再乘以 6，然后告诉我现在几点 ║")
    print("╚══════════════════════════════════════════╝")
    print()


def handle_command(cmd: str, agent: LangGraphAgent, thread_id: list) -> bool:
    """处理斜杠命令。"""
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
        print("可用工具:")
        for t in BUILTIN_TOOLS:
            # 取 docstring 第一行
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

    print(f"未知命令: {cmd}")
    return True


def main() -> None:
    """Phase 2 主循环。"""
    # 1. 加载配置
    try:
        config = Config()
        config.validate()
    except ValueError as e:
        print(f"配置错误: {e}")
        sys.exit(1)

    # 2. 创建 LangGraph Agent (绑定 BUILTIN_TOOLS)
    agent = LangGraphAgent(config, BUILTIN_TOOLS)

    # 3. 对话状态 (用可变列表来在循环中切换 thread_id)
    import uuid
    thread_id = [uuid.uuid4().hex[:8]]  # 短 ID，方便展示

    print_banner()
    print(f"模型: {config.model}")
    print(f"工具: {', '.join(t.name for t in BUILTIN_TOOLS)}")
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
            if not handle_command(user_input, agent, thread_id):
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
                if node == "model":  # create_agent 的节点名是 "model"
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
