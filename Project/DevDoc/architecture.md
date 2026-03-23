# 项目架构与模块说明

本项目 `ai-agent-rag-lab` 的目标是：

> 从零实现一个 AI 应用项目，完整覆盖：LLM 基础封装、Prompt Engineering、RAG、Agent、自动化抓取与 FastAPI 服务化。

本篇文档从架构视角整理项目的核心模块与依赖关系，便于进行系统性讲解。

## 1. 顶层结构

项目根目录（简化）：

- `pyproject.toml`：项目元数据与依赖管理
- `README.md`：项目总览与使用说明
- `DOC/`：RAG 知识库文档（如 `rag_notes.md`）
- `DevDoc/`：开发文档（安装、架构、原理说明等）
- `src/`：核心源码目录
  - `__init__.py`：将 `src` 变为 Python 包，方便 `src.xxx` 导入
  - `llm_client.py`：LLM API 封装
  - `prompting.py`：Prompt Engineering 模块
  - `rag_pipeline.py`：RAG 管线（BM25 + 本地向量后端，可插拔）
  - `agent_demo.py`：简化版 ReAct Agent + 工具系统
  - `web_automation_demo.py`：HTTP 网页抓取 + 总结
  - `app.py`：FastAPI 服务入口（统一对外 API）

## 2. LLMClient：统一的大模型访问层

文件：`src/llm_client.py`

职责：
- 封装 OpenAI-compatible `/v1/chat/completions` 接口；
- 从 `.env` 中加载 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL` 等配置；
- 暴露简单的 `chat(prompt, system_prompt)` 方法，供上层模块复用。

关键设计：
- 使用 `LLMConfig` dataclass 管理配置；
- 使用 `httpx.Client` 做 HTTP 请求，便于超时与重试扩展；
- 将 API 调用细节与 Prompt 设计、RAG、Agent 完全解耦。

依赖关系：
- 被 `prompting.py`、`rag_pipeline.py`、`agent_demo.py`、`web_automation_demo.py` 与 `app.py` 统一调用。

## 3. Prompting：Baseline / CoT / 简化 ReAct

文件：`src/prompting.py`

职责：
- 定义多种 Prompt 策略，帮助对比不同思维链路对答案质量的影响；
- 对外暴露 `PromptingEngine.answer(style=...)` 接口。

主要内容：
- `PromptContext`：封装问题、上下文等信息；
- `build_baseline_prompt`：简单直接答题；
- `build_cot_prompt`：鼓励“先思考后回答”；
- `build_react_prompt`：鼓励 Thought / Action / Observation 结构；
- `_demo()`：对比同一问题在不同 Prompt 风格下的输出。

## 4. RAGPipeline：可插拔的检索增强生成

文件：`src/rag_pipeline.py`

职责：
- 从 `DOC/` 目录加载文本（`.md` / `.txt`）；
- 切分为若干 `TextChunk` 片段；
- 使用可插拔检索后端进行召回（BM25 / 本地向量）；
- 将检索结果拼接成上下文，调用 LLM 给出答案。

关键结构：
- `RAGConfig`：保存 `api_key`、`base_url`、`chat_model` 等配置；
- `TextChunk`：保存文本片段内容与来源文件路径；
- `RetrieverBackend` Protocol：定义后端接口 `build_index` / `search`；
- `BM25Backend`：纯 Python 实现的 BM25 类检索；
- `LocalEmbeddingBackend`：`sentence-transformers` + `faiss-cpu` 的向量检索；
- `RAGPipeline`：聚合上述组件，对外暴露 `build_index()` 与 `ask(question)`。

后端选择：
- 通过环境变量 `RAG_BACKEND` 控制：
  - `bm25`：不依赖向量库，适用于无法使用 embeddings 的环境；
  - `local_embedding`：依赖 `sentence-transformers` 与 `faiss-cpu`，在本地进行语义检索；
- 如果本地向量后端初始化失败（依赖未装、模型下载失败等），自动回退 BM25。

## 5. AgentDemo：手写 ReAct Agent + 工具系统

文件：`src/agent_demo.py`

职责：
- 定义一组可被大模型调用的工具（工具函数 + 描述）；
- 设计 ReAct 风格的系统 Prompt；
- 实现一个最小可用的 ReAct 循环：
  - Thought → Action → ActionInput → Observation → FinalAnswer。

关键结构：
- `Tool` dataclass：包含 `name`、`description`、`func`；
- 工具函数集合：
  - `tool_get_current_time`：返回本地当前时间；
  - `tool_read_project_overview`：返回项目概要；
  - `tool_simple_calculator`：安全沙箱下 eval 数学表达式；
  - `tool_rag_qa`：调用 `RAGPipeline` 进行知识库问答；
- `build_toolbox()`：汇总工具为字典；
- `AGENT_SYSTEM_PROMPT`：约束模型输出 ReAct 结构；
- `build_agent_prompt()`：拼接工具列表、历史记录与用户问题；
- `parse_agent_action()`：从模型输出中解析 Action / ActionInput / FinalAnswer；
- `run_agent_once()`：循环执行 ReAct 流程，最多若干步工具调用后给出总结。

Agent 的位置：
- 逻辑层：站在 LLMClient 与各类工具（RAG、时间、计算器、自动化）之间；
- 能力层：能够根据自然语言问题自动选择是否调用工具、调用哪个工具。

## 6. WebAutomationDemo：HTTP 网页抓取 + 总结

文件：`src/web_automation_demo.py`

职责：
- 给定一个 URL，通过 HTTP 抓取网页 HTML；
- 做一次简易 HTML → 文本的清洗；
- 调用 LLM 对页面内容进行中文总结。

关键结构：
- `PageContent(url, title, text)`：结构化表示网页内容；
- `_html_to_text(html)`：
  - 去除 `<script>` / `<style>`；
  - 移除所有 HTML 标签；
  - 合并多余空白；
  - 提取 `<title>`；
- `fetch_page_text(url)`：使用 `httpx` GET 网页，返回 `PageContent`；
- `summarize_page(client, page)`：调用 LLM 输出结构化摘要。

该模块既可以单独作为 Demo，也可以将其能力暴露为 Agent 工具或 FastAPI 路由。

## 7. FastAPI 应用：统一对外 API

文件：`src/app.py`

职责：
- 提供统一的 HTTP API 入口，封装已有能力；
- 方便前端、第三方服务或 Postman 进行集成与调试。

目前路由：
- `GET /health`：健康检查；
- `POST /chat`：直接调用 LLM；
- `POST /rag`：知识库问答；
- `POST /agent`：Agent 工具调用问答；
- `POST /web-summarize`：网页抓取 + 总结。

FastAPI 与核心逻辑的关系：
- FastAPI 层只负责：参数校验 + 调用已有 Python 函数 + 错误转换为 HTTP 状态码；
- 业务逻辑（RAG、Agent、自动化）全部在独立模块中实现，便于单元测试与复用。

## 8. 依赖与技术选型

主要依赖：

- **httpx**：
  - 用于调用 LLM 网关（OpenAI-compatible 接口）；
  - 用于抓取网页 HTML（自动化模块）。

- **python-dotenv**：
  - 从 `.env` 文件加载 `LLM_API_KEY`、`LLM_BASE_URL` 等配置；
  - 避免将敏感信息写死在代码中。

- **fastapi + uvicorn**：
  - 快速搭建高性能 HTTP API；
  - 自带 `/docs` Swagger UI，便于演示与调试。

- **faiss-cpu + sentence-transformers**（可选）：
  - 用于本地向量检索（`LocalEmbeddingBackend`）；
  - 在无法访问云端 embeddings 接口时，仍可实现语义检索。

## 9. 架构亮点（简历可用）

- **清晰的分层设计**：
  - LLM 封装层（LLMClient）
  - Prompt 策略层（Prompting）
  - 检索层（RAGPipeline + 可插拔后端）
  - 决策层（ReAct Agent + 工具系统）
  - 接入层（FastAPI、CLI、自动化脚本）

- **可插拔 RAG 后端**：
  - 同一套 RAGPipeline 对接 BM25 与本地向量检索两种后端；
  - 通过环境变量平滑切换，并在失败时自动回退保底方案。

- **从 CLI 到 API 的渐进式演进**：
  - 所有能力先通过命令行脚本自测；
  - 再统一封装到 FastAPI 路由中对外暴露，方便组成完整应用。

- **无重型框架的 ReAct Agent 实现**：
  - 手写工具列表、Prompt 与循环逻辑，清楚展示 Agent 的工作原理；
  - 便于在面试中讲解“背后发生了什么”。

更多关于各模块的内部工作机制，可参考 `DevDoc/principles.md`（程序调用原理文档）。
