"""Prompt Engineering 层：在 LLMClient 之上封装不同提示风格。

本模块主要目标：
- 提供统一的接口来调用不同「思维模式」：
  - baseline_answer: 普通直接回答
  - cot_answer: Chain-of-Thought（思维链）回答
  - react_like_reasoning: 简化版 ReAct 风格推理结构（不真正调用工具）
- 为后续接 LangChain Agent、真正的工具调用做「思维框架」铺垫。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from llm_client import LLMClient, load_default_client


ReasoningStyle = Literal["baseline", "cot", "react"]


@dataclass
class PromptContext:
    """一次提问的上下文信息。

    目前先保留最基本字段，后面在做 RAG / Agent 时，可以在这里增加：
    - user_profile, history, knowledge_snippets 等等。
    """

    question: str
    # 将来可以扩展：language, domain, constraints 等


def build_baseline_prompt(ctx: PromptContext) -> str:
    """构造最简单的「直接回答」Prompt。"""

    return (
        "你是一个专业、简洁的 AI 助手。\n"
        "请直接用清晰的方式回答用户的问题，如果需要可以给出适当的解释。\n\n"
        f"用户问题：{ctx.question}\n"
    )


def build_cot_prompt(ctx: PromptContext) -> str:
    """构造 Chain-of-Thought（思维链）风格 Prompt。

    核心思想：明确要求模型「先分析再给结论」。
    """

    return (
        "你是一个擅长系统化思考的 AI 助手。\n"
        "请严格按照以下步骤回答问题：\n"
        "1. 先用条理清晰的步骤分析问题（标注为【思考过程】）。\n"
        "2. 再给出简洁明确的结论（标注为【最终结论】）。\n"
        "注意：思考过程要尽量具体，但避免废话。\n\n"
        f"用户问题：{ctx.question}\n"
    )


def build_react_prompt(ctx: PromptContext) -> str:
    """构造简化版 ReAct 风格 Prompt。

    这里还没有真正调用外部工具，只是训练模型按照：
    Thought -> Action -> Observation -> ... -> Final Answer 的格式来推理。
    """

    return (
        "你是一个善于思考并采取行动的 AI 助手。\n"
        "请按照 ReAct 风格来推理和回答：\n"
        "- 使用【Thought】描述你当前的思考；\n"
        "- 使用【Action】描述你打算执行的操作（此阶段只需要用自然语言描述，不必真的执行）；\n"
        "- 使用【Observation】描述（假设的）执行结果；\n"
        "- 可以多轮 Thought -> Action -> Observation；\n"
        "- 最后使用【Final Answer】给出总结性的回答。\n"
        "注意：本阶段不需要真正调用工具，只需要以这种结构化方式展示思考过程。\n\n"
        f"用户问题：{ctx.question}\n"
    )


class PromptingEngine:
    """面向上层应用的 Prompt 调用封装。

    上层只需要关心：
    - 想要哪种 reasoning_style（baseline / cot / react）
    - 用户的问题是什么
    """

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def answer(
        self,
        ctx: PromptContext,
        style: ReasoningStyle = "baseline",
    ) -> str:
        """根据指定风格生成回答。"""

        if style == "baseline":
            user_prompt = build_baseline_prompt(ctx)
            system_prompt = "你是一个专业、简洁的中文 AI 助手。"
        elif style == "cot":
            user_prompt = build_cot_prompt(ctx)
            system_prompt = "你是一个擅长逐步推理、逻辑严谨的 AI 助手。"
        elif style == "react":
            user_prompt = build_react_prompt(ctx)
            system_prompt = "你是一个会先思考再行动的 AI 助手。"
        else:
            raise ValueError(f"未知的 reasoning_style: {style}")

        # 这里复用底层 LLMClient，而不用关心 HTTP 细节
        return self._client.chat(prompt=user_prompt, system_prompt=system_prompt)


def _demo() -> None:
    """自测脚本：演示三种不同 Prompt 风格的差异。"""

    client = load_default_client()
    engine = PromptingEngine(client)

    question = "小明有 3 个苹果，又买了 5 个，送给朋友 2 个，还剩下多少个？请解释你的计算过程。"
    ctx = PromptContext(question=question)

    print("=== Baseline 普通回答 ===")
    print(engine.answer(ctx, style="baseline"))
    print("\n=== CoT 思维链回答 ===")
    print(engine.answer(ctx, style="cot"))
    print("\n=== ReAct 风格结构化思考 ===")
    print(engine.answer(ctx, style="react"))


if __name__ == "__main__":
    _demo()