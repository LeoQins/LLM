# 安装与运行指南

本项目基于 Python 3.9+，建议使用虚拟环境（venv）进行依赖隔离。

## 1. 克隆与进入项目目录

```powershell
# 克隆仓库（示例）
# git clone <your-repo-url>
cd C:\Users\LeoQin\Desktop\LLM\Project
```

## 2. 创建并激活虚拟环境（Windows / PowerShell）

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

激活后，命令行前面会出现 `(.venv)` 前缀。

## 3. 安装项目依赖

本项目使用 `pyproject.toml` 管理依赖，可以直接使用 `pip` 安装：

```powershell
pip install -e .
```

如不使用 `-e` 开发模式，也可以改为：

```powershell
pip install .
```

若你不想从源码安装，也可以手动安装主要依赖（不推荐长期使用，仅供调试）：

```powershell
pip install httpx python-dotenv fastapi uvicorn faiss-cpu sentence-transformers
```

> 提示：`sentence-transformers` 首次运行时会下载预训练向量模型，需要联网，可能花费数分钟。

## 4. 配置环境变量（.env）

项目使用 `.env` 文件管理敏感配置（API Key 等），示例文件在根目录的 `.env.example` 中：

```env
LLM_API_KEY="your-api-key-here"
LLM_BASE_URL="https://your-openai-compatible-endpoint"
LLM_MODEL="your-chat-model-name"

# 可选：RAG 检索后端
# RAG_BACKEND 可选值：bm25 | local_embedding（默认 bm25）
RAG_BACKEND="bm25"

# 可选：本地向量检索使用的 embedding 模型（sentence-transformers 模型名）
RAG_EMBEDDING_MODEL="moka-ai/m3e-small"
```

在 `.env` 中填入你自己的网关地址、模型名和 API Key。

## 5. 基础功能自测

### 5.1 LLMClient 自测

```powershell
python -m src.llm_client
```

终端应打印出模型返回的内容，证明：
- `.env` 配置正确；
- 网络连通，API Key 有效。

### 5.2 Prompting 自测

```powershell
python -m src.prompting
```

可以看到同一个问题分别用 Baseline / CoT / ReAct 三种 Prompt 策略的输出差异。

### 5.3 RAG 管线自测

```powershell
python -m src.rag_pipeline
```

如果 `DOC/` 目录下有示例文档（如 `rag_notes.md`），会输出基于知识库的回答。

### 5.4 Agent Demo 自测

```powershell
python -m src.agent_demo
``;

按提示输入自然语言问题，例如：

- "现在几点？"
- "这个项目是干什么的？"

观察 Agent 的多轮思考与工具调用行为。

### 5.5 网页抓取 + 总结 Demo

```powershell
python -m src.web_automation_demo
```

输入一个可访问的 URL（例如你之前成功抓取过的页面），等待抓取和总结结果。

## 6. FastAPI 服务（可选，当前阶段可暂时忽略）

如果希望以 HTTP API 形式提供统一服务，可运行：

```powershell
uvicorn src.app:app --reload --port 8000
```

然后在浏览器打开：

- 健康检查：`http://127.0.0.1:8000/health`
- Swagger UI：`http://127.0.0.1:8000/docs`

接口包括：
- `POST /chat`：直接模型对话
- `POST /rag`：知识库 RAG 问答
- `POST /agent`：带工具的 Agent 问答
- `POST /web-summarize`：网页抓取 + 总结

> 注：如在 PowerShell 中看到返回 JSON 的乱码，多半是控制台编码问题，与服务端无关，可使用浏览器或 Postman 调试。
