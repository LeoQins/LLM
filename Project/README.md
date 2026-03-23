# ai-agent-rag-lab

AI Agent + RAG + 自动化 + FastAPI 的综合实战项目，面向

- AI 应用工程师 / AI 开发工程师 校招 / 转岗
- 希望系统掌握「从调用大模型 → Prompt 工程 → RAG → Agent → 自动化 → 服务化」完整链路的同学

项目从零搭建，不依赖重型 Agent 框架，全部逻辑可读性强、适合讲解原理。

---

## 1. 功能总览

本项目已经实现的能力：

- ✅ 大模型封装：
	- `LLMClient` 封装 OpenAI-compatible `/v1/chat/completions` 接口
	- 基于 `.env` 管理 `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL`

- ✅ Prompt Engineering：
	- Baseline / Chain-of-Thought / 简化 ReAct 三种 Prompt 策略
	- 演示不同 Prompt 对结果质量和可解释性的影响

- ✅ RAG 管线（检索增强生成）：
	- 文档存储在 `DOC/` 目录下（`.md` / `.txt`）
	- 可插拔检索后端：
		- `bm25`：纯 Python BM25 关键词检索（无向量依赖）
		- `local_embedding`：`sentence-transformers` + `faiss-cpu` 本地向量检索
	- 使用 LLM 在检索结果基础上生成最终回答

- ✅ ReAct Agent + 工具调用：
	- 手写 ReAct 循环（Thought / Action / ActionInput / Observation / FinalAnswer）
	- 工具集合：当前时间 / 项目概览 / 安全计算器 / RAG 问答

- ✅ HTTP 网页抓取 + 总结：
	- 使用 `httpx` 抓取 HTML，正则清洗 HTML → 文本
	- 调用 LLM 对网页进行结构化中文总结

- ✅ FastAPI 服务化：
	- `GET /health`：健康检查
	- `POST /chat`：直接大模型对话
	- `POST /rag`：知识库 RAG 问答
	- `POST /agent`：带工具的 Agent 问答
	- `POST /web-summarize`：网页抓取 + 总结

所有核心能力都可以通过 **命令行脚本** 和 **FastAPI** 两种方式调用，目前你已经验证了命令行脚本链路全部跑通。

---

## 2. 项目结构

```text
ai-agent-rag-lab/
├─ pyproject.toml        # 项目依赖和元数据
├─ README.md             # 当前文件，项目总览
├─ .env / .env.example   # 环境变量配置（API Key 等）
├─ DOC/                  # RAG 知识库文档（markdown/txt）
│   └─ rag_notes.md      # 示例项目说明文档
├─ DevDoc/               # 开发/设计文档
│   ├─ installation.md   # 安装与运行指南
│   ├─ architecture.md   # 架构与模块说明
│   └─ principles.md     # 调用原理与关键流程
└─ src/
		├─ __init__.py           # 使 src 成为 Python 包
		├─ llm_client.py         # LLMClient 封装
		├─ prompting.py          # Prompt Engineering 模块
		├─ rag_pipeline.py       # RAG 管线（BM25 + 向量后端）
		├─ agent_demo.py         # ReAct Agent + 工具调用
		├─ web_automation_demo.py# HTTP 网页抓取 + 总结
		└─ app.py                # FastAPI 应用入口
```

详细的架构说明与调用流程请参考：

- `DevDoc/architecture.md`
- `DevDoc/principles.md`

---

## 3. 环境准备与依赖安装（速查）

完整版本请看：`DevDoc/installation.md`。

### 3.1 Python 与虚拟环境

- Python 版本：**3.9+**（建议与本机一致）
- 在项目根目录创建并激活虚拟环境（Windows / PowerShell）：

```powershell
cd C:\Users\<你的用户名>\Desktop\LLM\Project
python -m venv .venv
 .\.venv\Scripts\Activate.ps1
```

### 3.2 安装依赖

推荐使用 `pyproject.toml` 统一安装：

```powershell
pip install -e .
```

或直接指定主要依赖（简化版）：

```powershell
pip install httpx python-dotenv fastapi uvicorn faiss-cpu sentence-transformers
```

### 3.3 配置 .env

复制 `.env.example` 为 `.env`，并根据你的网关修改：

```env
LLM_API_KEY="your-api-key-here"
LLM_BASE_URL="https://your-openai-compatible-endpoint"
LLM_MODEL="your-chat-model-name"

RAG_BACKEND="bm25"                 # 或 local_embedding
RAG_EMBEDDING_MODEL="moka-ai/m3e-small"  # 可选，本地向量模型名
```

---

## 4. 命令行使用示例

### 4.1 测试 LLMClient

```powershell
(.venv) python -m src.llm_client
```

预期：打印一段来自大模型的回答，说明网络和配置都正常。

### 4.2 对比 Prompt 策略

```powershell
(.venv) python -m src.prompting
```

预期：对同一个问题，分别输出 Baseline / CoT / ReAct 三种风格的回答，适合在面试中展示 Prompt 工程的效果。

### 4.3 运行 RAG Demo

```powershell
(.venv) python -m src.rag_pipeline
```

预期：

- 从 `DOC/` 加载文档；
- 基于 BM25 或本地向量检索相关内容；
- 调用 LLM 生成基于知识库的回答。

### 4.4 运行 Agent Demo

```powershell
(.venv) python -m src.agent_demo
```

交互示例：

- 输入：`现在几点？`
- Agent 可能会：
	- 先调用 `get_current_time` 工具；
	- 然后根据 Observation 输出 `FinalAnswer`。

你也可以问：

- `这个项目是干什么的？`
- `用 RAG 帮我介绍一下项目的目标和技术栈？`

### 4.5 网页抓取 + 总结 Demo

```powershell
(.venv) python -m src.web_automation_demo
```

输入一个 URL（例如你之前成功访问过的技术问答页面），等待输出结构化的中文总结。

---

## 5. FastAPI 接口（当前可按需使用）

如果你需要对外提供 HTTP API 或与前端联调，可以启用 FastAPI：

```powershell
(.venv) uvicorn src.app:app --reload --port 8000
```

启动后可访问：

- 健康检查：`http://127.0.0.1:8000/health`
- API 文档（Swagger UI）：`http://127.0.0.1:8000/docs`

主要路由：

- `POST /chat`：
	- 请求体：`{"prompt": "...", "system_prompt": "可选"}`
	- 响应：`{"answer": "..."}`

- `POST /rag`：
	- 请求体：`{"question": "...", "backend": "bm25" | "local_embedding"}`
	- 响应：`{"answer": "...", "backend": "..."}`

- `POST /agent`：
	- 请求体：`{"question": "...", "max_steps": 3}`
	- 响应：`{"answer": "...", "trace": null}`（目前 trace 预留）

- `POST /web-summarize`：
	- 请求体：`{"url": "https://..."}`
	- 响应：`{"url": "...", "title": "...", "snippet": "...", "summary": "..."}`

> 说明：你已经验证了代码层面的调用全部跑通；如需 HTTP 级联调，可以优先通过 Swagger UI 或 Postman 进行，不受 PowerShell 编码影响。

---

## 6. 如何介绍这个项目

可以从以下几个角度总结（也可直接参考 `DevDoc/architecture.md` 与 `DevDoc/principles.md`）：

1. **项目背景**：
	 - 个人主导的 AI 应用工程项目，用于系统学习与展示从 LLM 调用到 Agent/RAG/自动化/服务化的完整链路。

2. **技术栈**：
	 - Python 3.9+、FastAPI、httpx、python-dotenv、faiss-cpu、sentence-transformers 等。

3. **核心能力**：
	 - 自研 LLM 封装 + Prompt 工程；
	 - RAG（BM25 + 本地向量）可插拔架构；
	 - 手写 ReAct Agent 与工具系统；
	 - HTTP 网页抓取 + 总结；
	 - 统一 API 网关（FastAPI）。

4. **架构亮点**：
	 - 分层清晰、解耦良好；
	 - 对不确定外部能力（embeddings、浏览器）提供回退方案；
	 - 所有能力先在 CLI 验证，再服务化暴露，便于开发和测试。

5. **个人收获**：
	 - 深度理解 LLM 应用工程常见模式（RAG / Agent / 自动化）；
	 - 熟悉从本地脚本到 HTTP 服务化的完整落地过程；
	 - 在真实第三方网关环境下处理网络错误、限流等工程问题。

---

## 7. 后续可扩展方向

- 增加更复杂的 Agent 工具（如代码执行、文件读写、任务编排等）。
- 引入简单的前端界面，调用 FastAPI 接口构建「AI 工作台」。
- 对 RAG 管线增加多文档类型支持（PDF、网页抓取结果持久化等）。
- 在 FastAPI 层增加限流、鉴权、中间件日志等工程能力。

当前版本已经足以作为一份「完整度较高」的项目，可以根据需要继续迭代。
