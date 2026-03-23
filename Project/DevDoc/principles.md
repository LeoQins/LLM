# 程序调用原理与关键流程

本篇文档聚焦「背后的调用链路」，帮助清晰讲解：

- 每条功能链路从哪里开始、经过哪些模块、怎么结束；
- 模块之间的数据是如何流动的；
- 出错时会发生什么。

## 1. LLM 基础调用链路

以最简单的对话为例（`src/llm_client.py` 自测）：

1. 脚本入口调用 `load_default_client()`：
   - `python-dotenv` 读取 `.env` 中的 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL`；
   - 组装为 `LLMConfig`；
   - 实例化 `LLMClient(config)`。

2. 调用 `LLMClient.chat(prompt, system_prompt=None)`：
   - 组装 OpenAI-compatible 请求体：`{"model": ..., "messages": [...]} `；
   - 使用 `httpx.Client` 发送 `POST {base_url}/v1/chat/completions`；
   - 解析返回 JSON，提取 `choices[0].message.content` 作为最终回答。

3. 出错处理：
   - 网络错误 / HTTP 状态码异常 → 抛出异常，由上层脚本捕获并打印；
   - 这部分在 FastAPI 层会转成 500 错误返回给客户端。

## 2. Prompt Engineering 调用链路

以 `src/prompting.py` 为例：

1. 用户调用 `PromptingEngine(llm_client).answer(question, style="cot")`；
2. 根据 `style` 选择不同的 Prompt 构造函数：
   - `build_baseline_prompt()`：问题几乎原样转发；
   - `build_cot_prompt()`：在 user message 中显式要求“先思考再回答”；
   - `build_react_prompt()`：约定 Thought / Action / Observation / FinalAnswer 结构；
3. 将构造好的 Prompt 交给 `llm_client.chat(...)`；
4. 最终答案由 LLM 直接产生，Prompt 工程只影响“模型怎么思考”。

## 3. RAG 调用链路（BM25 / 本地向量）

以 `RAGPipeline.ask(question)` 为例：

### 3.1 索引构建阶段

1. 上游调用 `pipeline = load_default_rag_pipeline()`：
   - 与 LLMClient 相同，从 `.env` 构建 `RAGConfig` 和内部 `LLMClient`；

2. 调用 `pipeline.build_index()`：
   - `_load_documents()`：
     - 遍历 `DOC/` 目录下的 `.md`/`.txt` 文件；
     - 简单用正则按段落/句号切分为多个 `TextChunk`；
   - `_init_backend()`：
     - 读取环境变量 `RAG_BACKEND`（默认为 `bm25`）；
     - 如果是 `local_embedding`：
       - 尝试初始化 `LocalEmbeddingBackend(chunks, model_name)`；
       - 失败则 print 提示并回退到 `BM25Backend`；
     - 如果是 `bm25` 或未知值：
       - 直接初始化 `BM25Backend(chunks)`。

### 3.2 问答阶段

调用 `pipeline.ask(question, top_k=4)` 时：

1. 调用 `_backend.search(question, k=top_k)`：
   - BM25Backend：
     - 对 query 分词；
     - 对每个文档用 BM25 公式近似打分；
     - 返回得分最高的 k 个 `TextChunk`；
   - LocalEmbeddingBackend：
     - 使用 SentenceTransformer 将 query 编码为向量；
     - 在 FAISS 索引中做最近邻搜索；
     - 返回相似度最高的 k 个 `TextChunk`。

2. 将检索结果格式化为上下文字符串 `context_text`：
   - `"[来源: path | 得分: 0.xx]\n内容..."`；

3. 构造 `system_prompt`：
   - 强调“只能基于上下文回答，不要编造”；

4. 构造 `user_prompt`：
   - 把 `context_text` 与用户问题一起发给 LLM；

5. 调用内部 `LLMClient.chat()`，得到最终回答。

## 4. Agent 调用链路（ReAct + 工具）

以 `run_agent_once(client, question, max_steps=3)` 为例：

### 4.1 初始化

1. 构建工具集合 `tools = build_toolbox()`：
   - 每个工具是 `Tool(name, description, func)`；
   - 例如：
     - `get_current_time` → `tool_get_current_time`；
     - `rag_qa` → `tool_rag_qa`（内含 RAG 调用链路）。

2. 初始化 `history = ""`，用于记录每轮 Thought/Action/Observation。

### 4.2 ReAct 循环

for step in 1..max_steps：

1. 构造 Prompt：`build_agent_prompt(question, tools, history)`：
   - 包含：工具列表、历史记录、用户问题；

2. 调用 `client.chat(prompt, system_prompt=AGENT_SYSTEM_PROMPT)`：
   - 系统 Prompt 要求模型返回：
     - Thought / Action / ActionInput / Observation / FinalAnswer；

3. 解析输出：`parse_agent_action(model_output)`：
   - 从文本中提取 `Action`、`ActionInput`、`FinalAnswer`；

4. 分支处理：
   - 若包含 `FinalAnswer`：
     - 认为模型已经给出最终结果，直接返回；
   - 否则：
     - 根据 `Action` 名称在 `tools` 中查找工具；
     - 执行对应 `func(action_input)` 得到 `observation`；
     - 将本轮 `Action/ActionInput/Observation` 追加到 `history`，供下一轮参考。

### 4.3 超出最大步数

- 如果循环结束仍没有 `FinalAnswer`：
  - 构造一个总结 Prompt，将 `history` 整体交给 LLM；
  - 要求模型给出 `FinalAnswer`；
  - 返回这次调用的结果。

## 5. 网页抓取 + 总结链路

以 `summarize_page(client, page)` 为例：

1. `fetch_page_text(url)`：
   - 自动补全 `https://` 前缀；
   - `httpx.Client` GET 请求，跟随重定向；
   - 取得 HTML 文本；
   - `_html_to_text(html)`：
     - 去除 `<script>` / `<style>` 中内容；
     - 去除所有标签，提取 `<title>`；
     - 压缩多余空格；
   - 组装为 `PageContent(url, title, text)`；

2. `summarize_page(client, page, max_chars=4000)`：
   - 截断正文，避免太长；
   - 构造系统 Prompt（要求输出结构化总结：主题 / 要点 / 技术等）；
   - 构造 user Prompt（带上 URL、标题和内容片段）；
   - 调用 `client.chat()` 得到中文总结。

## 6. FastAPI 调用链路（简要）

以 `POST /agent` 为例：

1. FastAPI 接收 JSON 请求体，反序列化为 `AgentRequest`；
2. 进行参数校验（问题不能为空，步数有上限保护等）；
3. 调用 `run_agent_once(state.llm_client, question, max_steps)`；
4. 捕获异常并转换为 HTTP 500 错误；
5. 将结果封装为 `AgentResponse` 返回给客户端。

同理：

- `/rag`：内部调用 `load_default_rag_pipeline().build_index().ask(question)`；
- `/web-summarize`：内部调用 `fetch_page_text(url)` + `summarize_page(client, page)`。

## 7. 出错模式与鲁棒性设计

1. **环境配置错误**：
   - `.env` 中缺少 `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL`：
     - 在构建 LLMClient / RAGPipeline 时抛出 RuntimeError；
   - 解决：在安装指南中提供 `.env.example` 并给出字段说明。

2. **网络 & 网关错误**：
   - DNS / 连接超时 / 429 限流：
     - `httpx` 抛出异常，上层脚本打印错误信息；
     - FastAPI 捕获后返回 500 和错误详情。

3. **RAG 后端依赖问题**：
   - 未安装 `faiss-cpu` / `sentence-transformers`：
     - 初始化 `LocalEmbeddingBackend` 抛出异常；
     - `_init_backend()` 捕获异常并回退到 `BM25Backend`。

4. **Agent 工具异常**：
   - 模型输出未知 `Action` 名称：
     - 返回 Observation："工具 'xxx' 不存在，请检查工具名"；
     - 追加到 history，由模型在下一轮修正；
   - 工具内部异常（如非法表达式）：
     - 被 try/except 捕获，转化为友好的错误文本 Observation。

## 8. 总结：一条典型“全链路”调用

以“通过 API 让 Agent 基于知识库回答问题”为例：

1. 客户端调用 `POST /agent`，问题类似：
   - "根据项目知识库介绍一下 ai-agent-rag-lab 的架构"；
2. FastAPI 将请求转为 `AgentRequest`，调用 `run_agent_once(client, question)`；
3. Agent 第一轮：
   - Thought：需要读知识库；
   - Action：`rag_qa`；
   - ActionInput：原始问题；
4. 工具 `rag_qa` 调用：
   - `load_default_rag_pipeline()`；
   - `build_index()`（BM25/向量检索）；
   - `ask(question)` → 内部调用 LLM；
5. Observation："基于知识库的回答：..." 被写入 history；
6. Agent 第二轮：
   - 看到 Observation 后，整合答案；
   - 输出 `FinalAnswer`：面向用户的结构化说明；
7. FastAPI 将 `FinalAnswer` 放入 `AgentResponse.answer`，以 JSON 返回。

通过这套链路，你可以在面试时清晰描述：
- 如何从原始 HTTP 请求一路走到 LLM 调用再回来；
- 中间每一层（RAG、Agent、工具系统）的职责划分与出错处理策略。
