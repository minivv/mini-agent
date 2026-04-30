"""
Phase 1: 配置加载

从 .env 文件和系统环境变量中读取 LLM 配置。
Config 是一个简单的 dataclass，它封装了创建 ChatModel 所需的全部参数。
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

# 在模块加载时自动读取 .env 文件
load_dotenv()


@dataclass
class Config:
    """
    LLM 配置。

    LangChain 的 ChatModel 遵循 OpenAI 兼容接口规范，
    这意味着同一个 ChatOpenAI 类可以对接几乎所有主流 LLM 提供商，
    只需修改 base_url 即可。
    """

    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.7")))

    def validate(self) -> None:
        """检查必要配置是否已填写。"""
        if not self.api_key or "your-api-key" in self.api_key:
            raise ValueError(
                "请先配置 API Key:\n"
                "  1. cp .env.example .env\n"
                "  2. 编辑 .env，填入你的 LLM_API_KEY"
            )
