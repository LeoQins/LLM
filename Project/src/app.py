"""FastAPI 应用入口。

阶段7：将现有能力（LLM 调用 / RAG / Agent / 网页总结）逐步服务化。

本文件先实现最小可用版本：
- GET /health: 健康检查
- POST /chat: 直接大模型对话

后续会在此基础上增加：
- POST /rag: 基于知识库的 RAG 问答
- POST /agent: 带工具调用的 Agent
- POST /web-summarize: 网页抓取 + 总结
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.llm_client import LLMClient, load_default_client
from src.rag_pipeline import load_default_rag_pipeline
from src.agent_demo import run_agent_once
from src.web_automation_demo import fetch_page_text, summarize_page


app = FastAPI(title="ai-agent-rag-lab API", version="0.1.0")


# ---------- 请求/响应模型 ----------


class ChatRequest(BaseModel):
    """/chat 接口的请求体。"""

    prompt: str
    system_prompt: Optional[str] = None


class ChatResponse(BaseModel):
    """/chat 接口的响应体。"""

    answer: str


# ---------- 依赖注入 / 客户端管理 ----------


@dataclass
class AppState:
    """保存应用级别的共享状态。"""

    llm_client: LLMClient


state = AppState(llm_client=load_default_client())


# ---------- 路由定义 ----------


@app.get("/health")
async def health() -> dict:
    """健康检查：用于确认服务已启动。"""

    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """直接调用大模型进行对话。

    这是对 `LLMClient.chat` 的简单 HTTP 封装，便于前端或其他服务调用。
    """

    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt 不能为空")

    try:
        answer = state.llm_client.chat(prompt=prompt, system_prompt=req.system_prompt)
    except Exception as exc:  # 简单错误处理，后续可细化
        raise HTTPException(status_code=500, detail=f"LLM 调用失败: {exc}") from exc

    return ChatResponse(answer=answer)


# ---------- RAG 接口 ----------


class RAGRequest(BaseModel):
    """/rag 接口的请求体。"""

    question: str
    backend: Optional[str] = None  # 可选：bm25 | local_embedding


class RAGResponse(BaseModel):
    """/rag 接口的响应体。"""

    answer: str
    backend: str


@app.post("/rag", response_model=RAGResponse)
async def rag_qa(req: RAGRequest) -> RAGResponse:
    """基于项目 DOC/README 知识库的 RAG 问答接口。"""

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question 不能为空")

    # 如果显式指定 backend，就通过环境变量覆盖一次
    if req.backend:
        import os

        os.environ["RAG_BACKEND"] = req.backend

    try:
        pipeline = load_default_rag_pipeline()
        pipeline.build_index()
        answer = pipeline.ask(question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG 问答失败: {exc}") from exc

    backend = (req.backend or "bm25").lower()
    return RAGResponse(answer=answer, backend=backend)


# ---------- Agent 接口 ----------


class AgentRequest(BaseModel):
    """/agent 接口的请求体。"""

    question: str
    max_steps: int = 3


class AgentTraceStep(BaseModel):
    """可选：用于返回 Agent 的中间思考/动作轨迹（当前先占位）。"""

    step: int
    raw_output: str


class AgentResponse(BaseModel):
    """/agent 接口的响应体。"""

    answer: str
    # trace 字段目前占位，后续如需可返回更详细的 Thought/Action/Observation
    trace: Optional[List[AgentTraceStep]] = None


@app.post("/agent", response_model=AgentResponse)
async def agent_answer(req: AgentRequest) -> AgentResponse:
    """对外暴露的 Agent 问答接口。

    当前实现直接调用现有的 run_agent_once，暂不返回详细 trace，
    但预留了 AgentTraceStep 结构，后续可以在 agent_demo 中改造以支持。"""

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question 不能为空")

    max_steps = max(1, min(req.max_steps, 5))  # 做个简单的上限保护

    try:
        answer = run_agent_once(state.llm_client, question, max_steps=max_steps)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent 调用失败: {exc}") from exc

    return AgentResponse(answer=answer, trace=None)


# ---------- 网页总结接口 ----------


class WebSummarizeRequest(BaseModel):
    """/web-summarize 接口的请求体。"""

    url: str


class WebSummarizeResponse(BaseModel):
    """/web-summarize 接口的响应体。"""

    url: str
    title: str
    snippet: str
    summary: str


@app.post("/web-summarize", response_model=WebSummarizeResponse)
async def web_summarize(req: WebSummarizeRequest) -> WebSummarizeResponse:
    """给定 URL，抓取网页并用大模型生成中文总结。"""

    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="url 不能为空")

    try:
        page = fetch_page_text(url)
        # 取一小段作为 snippet，避免响应体过大
        snippet = page.text[:500]
        summary = summarize_page(state.llm_client, page)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"网页抓取或总结失败: {exc}") from exc

    return WebSummarizeResponse(
        url=page.url,
        title=page.title,
        snippet=snippet,
        summary=summary,
    )
