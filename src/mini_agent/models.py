"""
Phase 2 补丁: DeepSeek 模型支持

DeepSeek thinking 模型 (deepseek-reasoner, deepseek-v4-flash 等) 会在回复中
返回 reasoning_content。API 要求后续请求必须把这个字段原样传回，
否则报错:

  'The `reasoning_content` in the thinking mode must be passed back to the API.'

ChatDeepSeek (langchain-deepseek) 把 reasoning_content 存入了
AIMessage.additional_kwargs，但在生成下一次请求时没有把它注入到 payload 中。

PatchedChatDeepSeek 修复了这个问题。
"""

from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek


class PatchedChatDeepSeek(ChatDeepSeek):
    """
    修复了 reasoning_content 在多轮对话中丢失的问题。

    原理:
      1. DeepSeek API 返回 reasoning_content → ChatDeepSeek 存入
         AIMessage.additional_kwargs["reasoning_content"]
      2. 下一轮对话时，_get_request_payload 把消息转成 API 格式
      3. 原生实现没有把 additional_kwargs 里的 reasoning_content 带过去
      4. 我们覆写 _get_request_payload，手动把 reasoning_content 注入 payload
    """

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        # 保存原始消息（在转换为 payload format 之前）
        original_messages = self._convert_input(input_).to_messages()

        # 调用父类获取基础 payload
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        payload_messages = payload.get("messages", [])

        # 逐条匹配: 如果原始 AIMessage 有 reasoning_content，注入对应 payload msg
        if len(payload_messages) == len(original_messages):
            for payload_msg, orig_msg in zip(payload_messages, original_messages):
                if payload_msg.get("role") == "assistant" and isinstance(orig_msg, AIMessage):
                    reasoning = orig_msg.additional_kwargs.get("reasoning_content")
                    if reasoning is not None:
                        payload_msg["reasoning_content"] = reasoning
        else:
            # fallback: 按位置匹配 assistant 消息
            ai_messages = [m for m in original_messages if isinstance(m, AIMessage)]
            assistant_indices = [
                i for i, m in enumerate(payload_messages) if m.get("role") == "assistant"
            ]
            for idx, ai_msg in zip(assistant_indices, ai_messages):
                reasoning = ai_msg.additional_kwargs.get("reasoning_content")
                if reasoning is not None:
                    payload_messages[idx]["reasoning_content"] = reasoning

        return payload
