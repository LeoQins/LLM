"""阶段4：简化版 ReAct Agent Demo（命令行）。

功能：
- 定义一组简单工具（当前时间 / 项目说明 / RAG 问答 / 计算器）；
- 使用大模型决定何时调用哪个工具；
- 执行工具并将结果回传给大模型，得到最终答案。

注意：
- 这是一个教学用、单文件版 Agent Demo，后续会在 FastAPI 阶段升级为服务接口。
"""

from __future__ import annotations

import datetime
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from src.llm_client import LLMClient, LLMConfig, load_default_client
from src.rag_pipeline import load_default_rag_pipeline


# ---------- 工具定义层 ----------


@dataclass
class Tool:
    """Agent 可调用的工具。

    name: 工具内部名称（模型会在 Action 里使用）
    description: 用自然语言描述这个工具能做什么、需要什么参数
    func: 实际执行逻辑（输入字符串，返回字符串）
    """

    name: str
    description: str
    func: Callable[[str], str]


def tool_get_current_time(_: str) -> str:
    """返回当前时间（本地时区）。"""

    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def tool_read_project_overview(_: str) -> str:
    """返回项目的高层说明文字（直接从内置字符串，后续可改为读取 README）。"""

    return (
        "项目名称：ai-agent-rag-lab\n"
        "项目目标：构建一个可用于求职简历的 AI Agent + RAG + 自动化综合项目，"
        "演示从 LLM 调用、Prompt Engineering、RAG、Agent、自动化到 FastAPI 服务化的完整链路。\n"
        "核心技术栈：Python、FastAPI、LangChain/LlamaIndex、RAG、本地/云端大模型 API。\n"
    )


def tool_simple_calculator(expression: str) -> str:
    """一个非常简化的计算器。

    只支持安全的四则运算和 sqrt 等基础函数。
    """

    allowed_names = {
        "sqrt": math.sqrt,
        "abs": abs,
        "round": round,
        "ceil": math.ceil,
        "floor": math.floor,
        "pi": math.pi,
        "e": math.e,
    }

    try:
        # 使用受限的 eval 环境，避免执行任意代码
        result = eval(expression, {"__builtins__": {}}, allowed_names)  # type: ignore[arg-type]
    except Exception as exc:
        return f"计算表达式失败: {expression!r}，错误：{exc}"

    return f"表达式 {expression!r} 的结果是：{result}"


def tool_rag_qa(question: str) -> str:
    """调用现有 RAGPipeline，基于知识库回答问题。"""

    pipeline = load_default_rag_pipeline()
    pipeline.build_index()
    answer = pipeline.ask(question)
    return f"基于知识库的回答：{answer}"


def build_toolbox() -> Dict[str, Tool]:
    """构建工具字典，供 Agent 使用。"""

    tools = [
        Tool(
            name="get_current_time",
            description="获取当前本地时间，输入可以忽略或留空。",
            func=tool_get_current_time,
        ),
        Tool(
            name="read_project_overview",
            description="获取 ai-agent-rag-lab 项目的高层概要介绍。",
            func=tool_read_project_overview,
        ),
        Tool(
            name="simple_calculator",
            description=(
                "执行简单数学计算，例如 '1 + 2 * 3' 或 'sqrt(2)', 'round(3.14159, 2)'。\n"
                "输入为一个数学表达式字符串。"
            ),
            func=tool_simple_calculator,
        ),
        Tool(
            name="rag_qa",
            description="基于项目 DOC/README 构建的 RAG 知识库回答问题，输入为自然语言问题。",
            func=tool_rag_qa,
        ),
    ]
    return {t.name: t for t in tools}


# ---------- Agent 核心逻辑 ----------


AGENT_SYSTEM_PROMPT = """
你是一个可以调用工具的智能 AI Agent，使用 ReAct 风格进行推理。

你可以使用的工具列表会在上下文中给出，每次思考/行动/观察请严格使用以下格式：

Thought: 你当前的思考（用中文）
Action: 工具名
ActionInput: JSON 字符串或简单文本，作为传给工具的输入
Observation: 工具返回的结果
...（可以多轮 Thought/Action/Observation）...
FinalAnswer: 给出最终面向用户的回答（用中文）

重要约束：
- 如果可以直接根据已有信息回答，就不必调用工具。
- 如果需要时间、项目说明、数学计算或基于知识库的回答，请选择合适的工具并调用。
- 工具名必须严格来自提供的工具列表。
- 每轮最多执行一次 Action，拿到 Observation 后再进行新的 Thought。
""".strip()


def build_agent_prompt(user_question: str, tools: Dict[str, Tool], history: str = "") -> str:
    """构建给大模型的 Prompt，包含工具列表和历史思考。"""

    tools_description = "\n".join(
        f"- {name}: {tool.description}" for name, tool in tools.items()
    )

    base = (
        f"可用工具列表：\n{tools_description}\n\n"
        f"对话历史（如果有）：\n{history}\n\n"
        "现在用户的问题是：\n"
        f"{user_question}\n\n"
        "请按照 ReAct 格式继续推理，如果需要调用工具，请给出 Action 和 ActionInput；"
        "如果已经可以给出最终答案，请直接给出 FinalAnswer。"
    )
    return base


def parse_agent_action(model_output: str) -> Dict[str, Optional[str]]:
    """从模型输出中解析出 Action / ActionInput / FinalAnswer。

    解析策略较为宽松，只要包含对应前缀即可。
    """

    action = None
    action_input = None
    final_answer = None

    for line in model_output.splitlines():
        line = line.strip()
        if line.lower().startswith("action:"):
            action = line.split(":", 1)[1].strip()
        elif line.lower().startswith("actioninput:"):
            action_input = line.split(":", 1)[1].strip()
        elif line.lower().startswith("finalanswer:"):
            final_answer = line.split(":", 1)[1].strip()

    return {"action": action or None, "action_input": action_input or None, "final_answer": final_answer}


def run_agent_once(client: LLMClient, question: str, max_steps: int = 3) -> str:
    """运行一个简化版 ReAct Agent 循环，最多执行 max_steps 轮工具调用。"""

    tools = build_toolbox()
    history = ""
    for step in range(1, max_steps + 1):
        prompt = build_agent_prompt(question, tools, history=history)
        model_output = client.chat(prompt, system_prompt=AGENT_SYSTEM_PROMPT)

        parsed = parse_agent_action(model_output)
        action = parsed["action"]
        action_input = parsed["action_input"]
        final_answer = parsed["final_answer"]

        print(f"\n=== Step {step} 模型输出 ===")
        print(model_output)

        if final_answer:
            # 模型已经给出最终答案，结束循环
            return final_answer

        if not action:
            # 没有指定 Action，则认为模型选择直接回答或无法继续
            return model_output

        tool = tools.get(action)
        if not tool:
            # 未知工具名，反馈给模型
            observation = f"工具 {action!r} 不存在，请检查工具名。"
        else:
            # 执行工具
            tool_input = action_input or ""
            observation = tool.func(tool_input)

        # 将本轮 Action 与 Observation 追加到历史，供下一轮模型参考
        history += (
            f"\nThought/Action/Observation 记录（第 {step} 轮）：\n"
            f"Action: {action}\n"
            f"ActionInput: {action_input}\n"
            f"Observation: {observation}\n"
        )

    # 超出最大步数，要求模型给出总结
    final_prompt = (
        f"{history}\n\n"
        "你已经多次调用工具，请根据以上 Observation 给出最终总结性回答，"
        "使用格式：FinalAnswer: <你的回答>"
    )
    final_output = client.chat(final_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
    parsed = parse_agent_action(final_output)
    return parsed["final_answer"] or final_output


def _demo() -> None:
    """命令行 Demo：允许用户多次提问，体验 Agent 行为。"""

    client = load_default_client()
    print("简化版 ReAct Agent 已启动，输入问题并回车（输入 'exit' 退出）。\n")

    while True:
        question = input("你：").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Agent 已退出。")
            break

        answer = run_agent_once(client, question)
        print("\nAgent：", answer)
        print("-" * 60)


if __name__ == "__main__":
    _demo()