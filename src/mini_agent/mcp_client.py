"""
Phase 5: MCP (Model Context Protocol) 客户端

════════════════════════════════════════════════════════════════════════════
什么是 MCP?
════════════════════════════════════════════════════════════════════════════

MCP 是 Anthropic 提出的一个开放协议，让 LLM 能连接外部工具和数据源。

打个比方:
  - LLM 就像一个大脑，很聪明但没有手脚
  - MCP 服务器就像"手脚"——每个服务器提供一组特定能力
  - 比如: 文件系统服务器能读写文件，GitHub 服务器能操作仓库
  - MCP 客户端（我们）连接这些服务器，获取工具，给 LLM 使用

协议支持三种传输方式:
  - stdio: 启动子进程，通过 stdin/stdout 通信（最常用）
  - sse:   通过 HTTP Server-Sent Events 通信（远程服务器）
  - http:  通过 Streamable HTTP 通信（远程服务器）
"""

import json
import os
from pathlib import Path

from langchain_core.tools import BaseTool


# 默认配置文件路径
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "mcp_config.json"


def load_mcp_config(config_path: str | Path | None = None) -> dict:
    """
    从 JSON 文件加载 MCP 服务器配置。

    配置文件格式 (mcp_config.json):
    {
      "mcpServers": {
        "filesystem": {
          "transport": "stdio",
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
          "description": "文件系统操作"
        },
        "remote-tools": {
          "transport": "sse",
          "url": "http://localhost:8080/sse",
          "description": "远程工具服务器"
        }
      }
    }

    环境变量替换:
      以 "$" 开头的值会被替换为对应的环境变量。
      例如: "$GITHUB_TOKEN" → os.environ["GITHUB_TOKEN"]
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}

    with open(path) as f:
        config = json.load(f)

    # 环境变量替换
    servers = config.get("mcpServers", {})
    for name, server in servers.items():
        if "env" in server:
            for key, value in server["env"].items():
                if isinstance(value, str) and value.startswith("$"):
                    server["env"][key] = os.environ.get(value[1:], "")

    return servers


def build_connections(servers_config: dict) -> dict:
    """
    将配置文件的服务器定义转换为 MultiServerMCPClient 需要的 connections 格式。

    配置文件格式:
      {
        "transport": "stdio" | "sse" | "streamable_http",
        "command": "...",      (stdio)
        "args": ["..."],       (stdio)
        "env": {"...": "..."}, (stdio, 可选)
        "url": "...",          (sse/http)
        "headers": {...}       (sse/http, 可选)
      }

    转换为 langchain-mcp-adapters 的连接格式:
      {
        "transport": "stdio",
        "command": "...",
        "args": ["..."],
        ...
      }
    """
    connections = {}
    for name, server in servers_config.items():
        if not server.get("enabled", True):
            continue

        transport = server.get("transport", "stdio")

        if transport == "stdio":
            if "command" not in server:
                print(f"[MCP] 跳过服务器 '{name}': 缺少 command 字段")
                continue
            conn = {
                "transport": "stdio",
                "command": server["command"],
                "args": server.get("args", []),
            }
            if "env" in server:
                conn["env"] = server["env"]
            if "cwd" in server:
                conn["cwd"] = server["cwd"]

        elif transport in ("sse", "streamable_http"):
            if "url" not in server:
                print(f"[MCP] 跳过服务器 '{name}': 缺少 url 字段")
                continue
            conn = {
                "transport": transport,
                "url": server["url"],
            }
            if "headers" in server:
                conn["headers"] = server["headers"]

        else:
            print(f"[MCP] 跳过服务器 '{name}': 不支持的传输方式 '{transport}'")
            continue

        connections[name] = conn

    return connections


def load_mcp_tools(
    config_path: str | Path | None = None,
) -> list[BaseTool]:
    """
    加载所有 MCP 服务器提供的工具。

    这是对外的主入口函数。流程:
      1. 从配置文件读取服务器定义
      2. 构建连接参数
      3. 创建 MultiServerMCPClient，获取所有工具
      4. 返回 LangChain BaseTool 列表（可直接绑定到 agent）

    返回:
      工具列表。如果配置文件不存在或没有服务器，返回空列表。
    """
    servers_config = load_mcp_config(config_path)
    if not servers_config:
        return []

    connections = build_connections(servers_config)
    if not connections:
        return []

    # 使用 langchain-mcp-adapters 的 MultiServerMCPClient
    # 注意: get_tools() 是异步的，需要用 asyncio.run() 在同步上下文中调用
    from langchain_mcp_adapters.client import MultiServerMCPClient

    import asyncio

    async def _fetch_tools() -> list[BaseTool]:
        # tool_name_prefix=True: 给工具名加服务器名前缀，避免与内置工具冲突
        # 例如: filesystem 服务器的 read_file → filesystem__read_file
        client = MultiServerMCPClient(connections, tool_name_prefix=True)
        return await client.get_tools()

    try:
        tools = asyncio.run(_fetch_tools())
    except Exception as e:
        print(f"[MCP] 加载工具失败: {e}")
        return []

    # MCP 工具是异步的（通过 async 协议与 MCP 服务器通信），
    # 但 LangGraph 的 ToolNode 默认用同步方式调用工具。
    # 需要把异步工具包装成同步工具，否则会报
    # "StructuredTool does not support sync invocation" 错误。
    tools = _wrap_async_tools(tools)

    # 打印加载的工具
    for tool in tools:
        desc = (tool.description or "").split("\n")[0]
        print(f"  • {tool.name}: {desc}")

    return tools


def _wrap_async_tools(tools: list[BaseTool]) -> list[BaseTool]:
    """
    把异步 MCP 工具包装成同步工具。

    MCP 工具的底层是异步的（async protocol），但 LangGraph 的 ToolNode
    默认用同步方式调用。这个函数用 ThreadPoolExecutor 把异步调用转为同步。
    """
    from concurrent.futures import ThreadPoolExecutor
    from langchain_core.tools import StructuredTool

    executor = ThreadPoolExecutor(max_workers=4)
    wrapped = []

    for tool in tools:
        # 如果工具已经有同步调用方式（func），不需要包装
        if hasattr(tool, 'func') and tool.func is not None:
            wrapped.append(tool)
            continue

        # 如果工具只有 coroutine（异步），创建同步包装
        coro = getattr(tool, 'coroutine', None)
        if coro is None:
            wrapped.append(tool)
            continue

        def make_sync(coro_func):
            def sync_wrapper(**kwargs):
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    # 嵌套事件循环: 用线程池执行
                    future = executor.submit(asyncio.run, coro_func(**kwargs))
                    return future.result()
                else:
                    return asyncio.run(coro_func(**kwargs))
            return sync_wrapper

        sync_func = make_sync(coro)
        new_tool = StructuredTool(
            name=tool.name,
            description=tool.description,
            args_schema=tool.args_schema,
            func=sync_func,
            coroutine=coro,
        )
        wrapped.append(new_tool)

    return wrapped
