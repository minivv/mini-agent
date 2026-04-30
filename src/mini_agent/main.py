"""
Phase 1: CLI 入口

启动一个命令行对话界面，与 AI 进行多轮对话。

使用方式:
    uv run python -m mini_agent.main

或在项目根目录:
    uv run mini-agent
"""

import sys
from mini_agent.config import Config
from mini_agent.chat import ChatAgent


def print_banner() -> None:
    """打印欢迎信息。"""
    print()
    print("╔══════════════════════════════════════╗")
    print("║         Mini Agent - Phase 1         ║")
    print("║         基础 LLM 对话                 ║")
    print("╠══════════════════════════════════════╣")
    print("║  命令:                               ║")
    print("║    /reset   — 重置对话               ║")
    print("║    /history — 查看对话历史            ║")
    print("║    /quit    — 退出                   ║")
    print("╚══════════════════════════════════════╝")
    print()


def handle_command(cmd: str, agent: ChatAgent) -> bool:
    """处理斜杠命令。返回 True 表示继续，False 表示退出。"""
    cmd = cmd.strip().lower()

    if cmd in ("/quit", "/exit", "/q"):
        print("再见！")
        return False

    if cmd == "/reset":
        agent.reset()
        print("[对话已重置]")
        return True

    if cmd == "/history":
        h = agent.history
        if not h:
            print("[对话历史为空]")
            return True
        for i, msg in enumerate(h):
            role = "用户" if msg.type == "human" else "AI"
            content = str(msg.content)[:80]
            print(f"  {i+1}. [{role}] {content}...")
        return True

    print(f"未知命令: {cmd}")
    return True


def main() -> None:
    """主循环。"""
    # 1. 加载配置
    try:
        config = Config()
        config.validate()
    except ValueError as e:
        print(f"配置错误: {e}")
        sys.exit(1)

    # 2. 创建 Agent
    agent = ChatAgent(config)

    print_banner()
    print(f"模型: {config.model}")
    print(f"接口: {config.base_url}")
    print()

    # 3. 对话循环
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
            if not handle_command(user_input, agent):
                break
            print()
            continue

        # 流式输出 AI 回复
        print()
        print("AI: ", end="", flush=True)
        try:
            for chunk in agent.chat_stream(user_input):
                print(chunk, end="", flush=True)
        except Exception as e:
            print(f"\n[错误: {e}]")
        print("\n")


if __name__ == "__main__":
    main()
