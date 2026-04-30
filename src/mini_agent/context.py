"""
Phase 4: 上下文管理 — Token 计数 + 对话摘要

════════════════════════════════════════════════════════════════════════════
为什么需要上下文管理?
════════════════════════════════════════════════════════════════════════════

LLM 有一个固定的"上下文窗口"（context window），比如 8K、32K、128K tokens。
对话越长，消息越多，占用的 tokens 就越多。

当对话接近上下文窗口上限时，会出现两种问题:
  1. 超出限制 → API 报错
  2. 即使没超，太长的对话也会让 LLM 变慢、变贵、注意力分散

解决方案: 对话摘要（Summarization）
  - 当消息太多时，用 LLM 把旧消息总结成一段摘要
  - 用摘要替换旧消息，保留最近的 N 条消息不动
  - 这样对话历史被"压缩"了，但关键信息还在

打个比方:
  你在看一本很长的书，看到第 200 页时，前面的细节已经记不清了。
  于是你写了个"前情提要"（摘要），然后只保留最近几页和这个提要。
  这样你既不会忘掉主线剧情，又不用带着整本书。
"""

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    RemoveMessage,
)


# ---------------------------------------------------------------------------
# Token 计数
# ---------------------------------------------------------------------------
# tiktoken 是 OpenAI 开源的 tokenizer，GPT 系列模型用它来把文本切成 tokens。
#
# 什么是 token?
#   - 英文: 大约 1 个单词 ≈ 1-2 个 tokens
#   - 中文: 大约 1 个汉字 ≈ 1-2 个 tokens
#   - "Hello world" → ["Hello", " world"] → 2 tokens
#   - 你好世界 → ["你好", "世界"] → 2-4 tokens (取决于编码)
#
# 为什么用 tiktoken 而不是直接调 API?
#   - tiktoken 在本地计算，不需要网络请求，速度快、无费用
#   - 精度: 和 OpenAI API 的计数一致（误差 <1%）
#   - 对于非 OpenAI 模型（如 DeepSeek），用 cl100k_base 编码做近似估算

def _get_encoder():
    """获取 tiktoken 编码器（cl100k_base 是 GPT-4/GPT-3.5 使用的编码方案）。"""
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


def count_message_tokens(messages: list[BaseMessage]) -> int:
    """
    估算消息列表的 token 数量。

    每条消息有固定开销（role 标记等），加上内容的 token 数。
    这是一个近似值，但足够用来判断是否需要摘要。
    """
    encoder = _get_encoder()
    total = 0
    for msg in messages:
        # 每条消息的固定开销（role、格式标记等）
        total += 4
        # 内容的 token 数
        content = msg.content
        if isinstance(content, str):
            total += len(encoder.encode(content))
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    total += len(encoder.encode(item))
                elif isinstance(item, dict) and "text" in item:
                    total += len(encoder.encode(item["text"]))
    return total


# ---------------------------------------------------------------------------
# 摘要生成
# ---------------------------------------------------------------------------

SUMMARY_PROMPT = """请将以下对话总结成简洁的摘要。
保留关键信息：用户问了什么、AI做了什么（包括调用了哪些工具、得到了什么结果）。
只输出摘要本身，不要添加额外说明或格式。

对话:
{history}"""


def _format_for_summary(messages: list[BaseMessage]) -> str:
    """把消息列表格式化为文本，供摘要 LLM 阅读。"""
    parts = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            parts.append(f"用户: {msg.content}")
        elif isinstance(msg, AIMessage):
            content = (msg.content or "").strip()
            if msg.tool_calls:
                tools = ", ".join(tc["name"] for tc in msg.tool_calls)
                # 只记录工具调用，跳过 LLM 的内部思考过程
                parts.append(f"AI: [调用工具: {tools}]")
            elif content:
                parts.append(f"AI: {content}")
        # ToolMessage 和 SystemMessage 跳过，它们对摘要价值不大
    return "\n".join(parts)


def summarize_messages(
    messages: list[BaseMessage],
    model,
    keep_recent: int = 6,
    max_tokens_to_summarize: int = 4000,
) -> list:
    """
    用 LLM 总结旧消息，返回 LangGraph state 更新。

    流程:
      1. 把消息分成两部分: [旧消息] + [最近消息]
      2. 用 LLM 把旧消息总结成一段摘要
      3. 返回 state 更新:
         - RemoveMessage 删除所有旧消息
         - 新的 SystemMessage 包含摘要
         - 最近消息保持不变（LangGraph 的 add_messages 会自动处理）

    参数:
      messages: 当前所有消息（不含 SystemMessage）
      model: 用于生成摘要的 LLM 实例
      keep_recent: 保留最近几条消息不被摘要
      max_tokens_to_summarize: 送给摘要 LLM 的最大 token 数
    """
    if len(messages) <= keep_recent:
        return []  # 消息不多，不需要摘要

    # 分割: 旧消息（要摘要的）+ 最近消息（保留的）
    to_summarize = messages[:-keep_recent]
    recent = messages[-keep_recent:]

    # 跳过 SystemMessage（保留原始系统提示词）
    # 跳过 ToolMessage（工具结果，摘要价值低，且删除对应的 AIMessage 后会变成孤立消息）
    # 跳过有 tool_calls 的 AIMessage（它们是工具调用链的一部分，不应单独摘要）
    to_summarize = [
        m for m in to_summarize
        if not isinstance(m, (SystemMessage, ToolMessage))
        and not (isinstance(m, AIMessage) and m.tool_calls)
    ]

    # 截断旧消息，避免摘要 LLM 本身也超出上下文
    summary_text = _format_for_summary(to_summarize)
    encoder = _get_encoder()
    tokens = encoder.encode(summary_text)
    if len(tokens) > max_tokens_to_summarize:
        summary_text = encoder.decode(tokens[:max_tokens_to_summarize])

    # 调用 LLM 生成摘要
    prompt = SUMMARY_PROMPT.format(history=summary_text)
    try:
        response = model.invoke([HumanMessage(content=prompt)])
        # DeepSeek thinking 模式: 文本在 reasoning_content 而非 content
        summary_content = response.content
        if not summary_content:
            reasoning = response.additional_kwargs.get("reasoning_content", "")
            if reasoning:
                summary_content = reasoning
    except Exception as e:
        # 摘要失败不影响正常对话，只是跳过压缩
        print(f"\n[摘要生成失败: {e}]")
        return []

    # 构建 state 更新
    # 1. 删除所有旧消息
    updates = [RemoveMessage(id=msg.id) for msg in to_summarize if msg.id]
    # 2. 在最近消息前插入摘要（作为 SystemMessage）
    summary_msg = SystemMessage(
        content=f"[对话摘要] {summary_content}"
    )
    updates.append(summary_msg)

    return updates
