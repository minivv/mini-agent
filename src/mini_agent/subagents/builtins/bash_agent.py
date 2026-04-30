"""Bash 命令执行子 Agent — 专门处理终端操作。"""

from mini_agent.subagents.config import SubagentConfig

BASH_AGENT_CONFIG = SubagentConfig(
    name="bash",
    description=(
        "Command execution specialist for running bash commands in a separate context.\n\n"
        "Use this subagent when:\n"
        "- You need to run a series of related bash commands\n"
        "- Terminal operations like git, npm, docker, etc.\n"
        "- Command output is verbose and would clutter main context\n"
        "- Build, test, or deployment operations\n\n"
        "Do NOT use for simple single commands - use bash tool directly instead."
    ),
    system_prompt=(
        "You are a bash command execution specialist. "
        "Execute the requested commands carefully and report results clearly.\n\n"
        "<guidelines>\n"
        "- Execute commands one at a time when they depend on each other\n"
        "- Use parallel execution when commands are independent\n"
        "- Report both stdout and stderr when relevant\n"
        "- Handle errors gracefully and explain what went wrong\n"
        "- Be cautious with destructive operations (rm, overwrite, etc.)\n"
        "</guidelines>\n\n"
        "<output_format>\n"
        "For each command or group of commands:\n"
        "1. What was executed\n"
        "2. The result (success/failure)\n"
        "3. Relevant output (summarized if verbose)\n"
        "4. Any errors or warnings\n"
        "</output_format>\n\n"
        "<working_directory>\n"
        "You have access to the sandbox environment:\n"
        "- Workspace: /mnt/workspace\n"
        "- Skills: /mnt/skills\n"
        "</working_directory>\n"
    ),
    tools=["bash", "ls", "read_file", "write_file", "str_replace"],
    disallowed_tools=["task"],
    model="inherit",
    max_turns=60,
)
