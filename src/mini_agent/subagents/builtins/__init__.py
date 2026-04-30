from mini_agent.subagents.builtins.general_purpose import GENERAL_PURPOSE_CONFIG
from mini_agent.subagents.builtins.bash_agent import BASH_AGENT_CONFIG

BUILTIN_SUBAGENTS: dict[str, "SubagentConfig"] = {
    "general-purpose": GENERAL_PURPOSE_CONFIG,
    "bash": BASH_AGENT_CONFIG,
}

__all__ = ["BUILTIN_SUBAGENTS", "GENERAL_PURPOSE_CONFIG", "BASH_AGENT_CONFIG"]
