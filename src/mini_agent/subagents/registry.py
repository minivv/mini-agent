"""
Phase 7: 子 Agent 注册表
"""

from mini_agent.subagents.builtins import BUILTIN_SUBAGENTS
from mini_agent.subagents.config import SubagentConfig

# 自定义子 Agent 配置 (可通过代码注册)
_custom_subagents: dict[str, SubagentConfig] = {}


def register_subagent(config: SubagentConfig) -> None:
    """注册一个自定义子 Agent。"""
    _custom_subagents[config.name] = config


def unregister_subagent(name: str) -> None:
    """注销一个自定义子 Agent。"""
    _custom_subagents.pop(name, None)


def get_subagent_config(name: str) -> SubagentConfig | None:
    """按名称获取子 Agent 配置。

    解析顺序:
      1. 内置子 Agent (general-purpose, bash)
      2. 自定义子 Agent
    """
    if name in BUILTIN_SUBAGENTS:
        return BUILTIN_SUBAGENTS[name]
    return _custom_subagents.get(name)


def list_subagents() -> list[SubagentConfig]:
    """列出所有可用的子 Agent 配置。"""
    configs = list(BUILTIN_SUBAGENTS.values())
    configs.extend(_custom_subagents.values())
    return configs


def get_subagent_names() -> list[str]:
    """获取所有子 Agent 名称。"""
    names = list(BUILTIN_SUBAGENTS.keys())
    for name in _custom_subagents:
        if name not in names:
            names.append(name)
    return names
