"""
Phase 1: 对话 Agent

这是最小化的 Agent 实现 —— 本质上就是：
  - 一个 ChatModel（LLM）
  - 一个消息列表（对话历史）
  - 一个循环来管理多轮对话

LangChain 核心概念：
  - ChatModel    — LLM 的统一抽象，invoke()=一次性返回，stream()=逐字流式返回
  - SystemMessage  — 设定 AI 的行为和角色（"你是一个...助手"）
  - HumanMessage  — 用户输入
  - AIMessage     — AI 回复

多轮对话的关键：把整个消息列表（messages）送给模型，模型能看到所有历史。
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

from mini_agent.config import Config


# ---------------------------------------------------------------------------
# LangChain 核心概念 #1: ChatModel
# ---------------------------------------------------------------------------
# ChatOpenAI 是 LangChain 对 OpenAI Chat Completion API 的封装。
# 因为大多数 LLM 提供商都提供 OpenAI 兼容接口，所以这一个类就能对接:
# OpenAI / DeepSeek / 豆包 / Kimi / 通义千问 / Ollama / vLLM ...
#
# 关键参数:
#   model       — 具体的模型名（由提供商定义）
#   api_key     — API 密钥
#   base_url    — API 端点地址（换了它就换了提供商）
#   temperature — 生成温度（0=确定, 1=随机, >1=更随机）
#   streaming   — 是否支持流式输出
def create_model(config: Config, streaming: bool = True) -> ChatOpenAI:
    """根据配置创建一个 ChatModel 实例。"""
    return ChatOpenAI(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=config.temperature,
        streaming=streaming,
    )


# ---------------------------------------------------------------------------
# LangChain 核心概念 #2: Messages（消息类型）
# ---------------------------------------------------------------------------
# LangChain 定义了多种消息类型来表达对话的结构:
#
#   SystemMessage  — 系统指令：设定 AI 的行为边界和身份
#   HumanMessage   — 用户消息
#   AIMessage      — AI 的回复（可以包含 tool_calls，Phase 2 会用到）
#   ToolMessage    — 工具执行结果（Phase 2 会用到）
#
# 这些消息组成一个 list，就是对话历史。模型每次接收完整的 list，
# 所以它能看到之前说了什么 —— 这就是"多轮对话"的基础。


def build_system_prompt() -> str:
    """构建系统提示词。这部分决定了 AI 的行为方式。"""
    return """你是一个有帮助的 AI 助手，名叫 Mini。

你的特点:
- 回答简洁明了，不啰嗦
- 如果不知道就说不知道，不编造
- 可以用中文和英文交流"""


# ---------------------------------------------------------------------------
# Phase 1 核心：对话循环
# ---------------------------------------------------------------------------
class ChatAgent:
    """
    最小化的对话 Agent。

    它的工作流程:
      1. 维护一个消息列表（初始只有 SystemMessage）
      2. 用户输入 → 包装为 HumanMessage → 追加到列表
      3. 整个列表发给 LLM → 获取 AIMessage → 追加到列表
      4. 返回 AI 的回复
      5. 重复步骤 2-4

    这就是最基本的 "Agent 循环"。
    """

    def __init__(self, config: Config):
        self.config = config
        # 创建两个 model: 一个流式（用于打字机效果），一个非流式（用于内部调用）
        self.model = create_model(config, streaming=True)
        self.model_sync = create_model(config, streaming=False)

        # 对话历史的起点：系统指令
        self.messages: list[BaseMessage] = [
            SystemMessage(content=build_system_prompt())
        ]

    # ------------------------------------------------------------------
    # LangChain 核心概念 #3: invoke() vs stream()
    # ------------------------------------------------------------------
    # invoke()  — 发送请求，等待完整结果返回 → 得到一个完整的 AIMessage
    # stream()  — 发送请求，逐 token 返回 → 每个 chunk 是一个 AIMessageChunk
    #
    # stream() 的好处：用户看到一个字一个字地输出，体验好、等待感弱。
    # invoke() 的好处：简单、适合内部调用（比如做总结）。

    def chat(self, user_input: str) -> str:
        """非流式对话：发送整条消息，返回完整回复。内部使用。"""
        self.messages.append(HumanMessage(content=user_input))
        response = self.model_sync.invoke(self.messages)
        self.messages.append(response)
        return response.content  # type: ignore[return-value]

    def chat_stream(self, user_input: str):
        """
        流式对话：逐 token 产出，用于打字机效果。

        这是一个 generator 函数，每次 yield 一个文本片段。
        调用方用 for chunk in agent.chat_stream(...) 来逐字打印。
        """
        self.messages.append(HumanMessage(content=user_input))

        full_content: str = ""
        for chunk in self.model.stream(self.messages):
            if chunk.content:
                full_content += chunk.content  # type: ignore[operator]
                yield chunk.content

        # 流式结束后，手动构建一个完整的 AIMessage 追加到历史
        self.messages.append(AIMessage(content=full_content))

    def reset(self) -> None:
        """重置对话，清空历史（保留 system prompt）。"""
        self.messages = [SystemMessage(content=build_system_prompt())]

    @property
    def history(self) -> list[BaseMessage]:
        """返回当前对话历史（不含 system prompt）。"""
        return self.messages[1:] if len(self.messages) > 1 else []
