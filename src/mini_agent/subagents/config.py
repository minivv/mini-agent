"""
Phase 7: 子 Agent 配置

SubagentConfig 定义子 Agent 的行为:
  - name:           唯一标识 (如 "general-purpose", "bash")
  - description:    何时使用该子 Agent (LLM 看到这个来决定委派)
  - system_prompt:  子 Agent 的系统提示词
  - tools:          允许使用的工具列表 (None = 继承全部)
  - disallowed_tools: 禁止使用的工具 (默认 ["task"] 防止递归)
  - skills:         允许加载的技能 (None = 全部, [] = 无)
  - model:          使用的模型 ("inherit" = 继承父 Agent)
  - max_turns:      最大轮次 (防止无限循环)
  - timeout_seconds: 超时时间 (默认 900 秒 = 15 分钟)
"""

from dataclasses import dataclass, field


@dataclass
class SubagentConfig:
    """子 Agent 配置。"""

    name: str
    description: str
    system_prompt: str
    tools: list[str] | None = None
    disallowed_tools: list[str] = field(default_factory=lambda: ["task"])
    skills: list[str] | None = None
    model: str = "inherit"
    max_turns: int = 50
    timeout_seconds: int = 900
