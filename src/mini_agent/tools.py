"""
Phase 3: 工具定义 (Phase 2 工具 + 沙箱工具)

LangChain 的 @tool 装饰器是定义 Tool 的最简方式。
每个 Tool 有:
  - name:      函数名 (自动推断)
  - description: 函数的 docstring (LLM 用它来判断何时调用)
  - args_schema: 从函数的 type hints 自动生成 (告诉 LLM 参数的格式)

当 LLM 决定调用一个工具时，它输出一个 tool_call，包含:
  {name: "calculator", args: {expression: "2+3"}}

════════════════════════════════════════════════════════════════════════════
InjectedState: 让工具能读到当前 AgentState
════════════════════════════════════════════════════════════════════════════
  沙箱工具需要在每次调用时知道"工作目录在哪"。
  我们用 InjectedState 把 state["workspace_dir"] 注入到工具参数中。
  这样每个线程的工具操作都在自己的沙箱内执行。
"""

from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from mini_agent.sandbox import Sandbox


# ---------------------------------------------------------------------------
# Phase 2 工具 (保持不变)
# ---------------------------------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """计算数学表达式的结果。支持 + - * / ** () 等 Python 运算符。
    例子: 2+3*4 → 14
    例子: (10+5)*2 → 30
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


@tool
def get_current_time() -> str:
    """获取当前的日期、时间和星期。不需要任何参数。"""
    from datetime import datetime

    now = datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return now.strftime(f"%Y年%m月%d日 %H:%M:%S {weekdays[now.weekday()]}")


# ---------------------------------------------------------------------------
# Phase 3: 沙箱工具
# ---------------------------------------------------------------------------
# 这些工具通过 InjectedState 读取 state["workspace_dir"] 来确定沙箱目录。
# LLM 看到的是虚拟路径 (如 /mnt/workspace/foo.txt)，
# Sandbox 负责把它映射到实际的磁盘路径。

def _get_sandbox(state: dict) -> Sandbox:
    """从 state 中获取沙箱实例。"""
    workspace_dir = state.get("workspace_dir", "")
    return Sandbox(workspace_dir=workspace_dir) if workspace_dir else Sandbox()


@tool
def read_file(
    path: str,
    offset: int = 0,
    limit: int = 2000,
    state: Annotated[dict, InjectedState] = None,
) -> str:
    """读取文件内容。
    path: 文件路径，如 /mnt/workspace/main.py
    offset: 起始行号 (默认 0)
    limit: 最大行数 (默认 2000)
    返回带行号的文件内容。
    """
    return _get_sandbox(state).read_file(path, offset=offset, limit=limit)


@tool
def write_file(
    path: str,
    content: str,
    state: Annotated[dict, InjectedState] = None,
) -> str:
    """创建或覆盖一个文件。
    path: 文件路径，如 /mnt/workspace/hello.py
    content: 要写入的完整文件内容
    会自动创建父目录。
    """
    return _get_sandbox(state).write_file(path, content)


@tool
def str_replace(
    path: str,
    old_str: str,
    new_str: str,
    state: Annotated[dict, InjectedState] = None,
) -> str:
    """替换文件中第一次出现的指定字符串。
    path: 文件路径
    old_str: 要被替换的字符串 (必须精确匹配)
    new_str: 替换后的新字符串
    """
    return _get_sandbox(state).str_replace(path, old_str, new_str)


@tool
def ls(
    path: str = "/mnt/workspace",
    state: Annotated[dict, InjectedState] = None,
) -> str:
    """列出目录内容，树形格式显示。
    path: 目录路径，默认 /mnt/workspace
    """
    return _get_sandbox(state).ls(path)


@tool
def glob(
    pattern: str,
    path: str = "/mnt/workspace",
    state: Annotated[dict, InjectedState] = None,
) -> str:
    """按模式匹配文件，如 *.py 或 **/*.py。
    pattern: glob 模式
    path: 搜索起始目录 (默认 /mnt/workspace)
    """
    return _get_sandbox(state).glob(pattern, path)


@tool
def grep(
    pattern: str,
    path: str = "/mnt/workspace",
    glob_pattern: str = "*",
    state: Annotated[dict, InjectedState] = None,
) -> str:
    """在文件中搜索文本，返回匹配的行号和内容。
    pattern: 要搜索的文本 (普通字符串，不是正则)
    path: 搜索目录 (默认 /mnt/workspace)
    glob_pattern: 文件名过滤，如 *.py (默认 *)
    """
    return _get_sandbox(state).grep(pattern, path, glob_pattern)


@tool
def bash(
    command: str,
    timeout: int = 60,
    state: Annotated[dict, InjectedState] = None,
) -> str:
    """在沙箱工作目录中执行 bash 命令。
    command: 要执行的 shell 命令。可以组合多个命令，如: ls -la && cat foo.txt
    常用工具: ls, cat, find, grep, python, pip, git, curl, node, npm
    """
    return _get_sandbox(state).bash(command, timeout=timeout)


# ---------------------------------------------------------------------------
# 工具分组
# ---------------------------------------------------------------------------

# Phase 2 工具 (不需要沙箱)
BUILTIN_TOOLS: list = [calculator, get_current_time]

# Phase 3 沙箱工具 (需要在 AgentState 中有 workspace_dir)
SANDBOX_TOOLS: list = [read_file, write_file, str_replace, ls, glob, grep, bash]

# 完整工具列表
ALL_TOOLS: list = BUILTIN_TOOLS + SANDBOX_TOOLS
