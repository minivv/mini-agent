"""
Phase 2: 内置工具

LangChain 的 @tool 装饰器是定义 Tool 的最简方式。
每个 Tool 有:
  - name:      函数名 (自动推断)
  - description: 函数的 docstring (LLM 用它来判断何时调用)
  - args_schema: 从函数的 type hints 自动生成 (告诉 LLM 参数的格式)

当 LLM 决定调用一个工具时，它输出一个 tool_call，包含:
  {name: "calculator", args: {expression: "2+3"}}
"""

from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# @tool 装饰器: 把一个普通函数变成 LangChain Tool
# ---------------------------------------------------------------------------
# LLM 通过函数的 docstring 和参数类型来理解工具的作用。
# docstring 写得越清楚，LLM 就越能在正确的时机调用它。

@tool
def calculator(expression: str) -> str:
    """计算数学表达式的结果。支持 + - * / ** () 等 Python 运算符。
    例子: 2+3*4 → 14
    例子: (10+5)*2 → 30
    """
    try:
        # 只允许安全的数学运算
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
# 工具列表: 后续 Phase 可以在这里追加新工具
# ---------------------------------------------------------------------------

BUILTIN_TOOLS: list = [calculator, get_current_time]
