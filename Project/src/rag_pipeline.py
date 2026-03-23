"""RAG 阶段：最小可用的检索增强生成（Retrieval-Augmented Generation）管线。

设计目标：
- 使用本地 DOC 目录中的 markdown/txt 文档，构建一个简单的向量检索器；
- 使用 OpenAI-compatible 的嵌入接口生成向量（与当前 LLM 调用方式兼容）；
- 用 LangChain 提供的向量存储与检索接口，完成 RAG 问答；
- 对上层暴露一个简单的 `RAGPipeline` 类和一个自测 demo。

说明：
- 为了简化依赖，这里直接使用 langchain + faiss-cpu 构建本地向量库；
- 嵌入和聊天都复用同一套 API Key 和 Base URL（前提是你的网关支持 /v1/embeddings）。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Protocol, runtime_checkable

from dotenv import load_dotenv

from src.llm_client import LLMClient, LLMConfig

try:  # 可选导入，embedding 后端依赖
    import faiss  # type: ignore
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - 缺依赖时自动回退 BM25
    faiss = None
    SentenceTransformer = None


load_dotenv()


DOC_DIR = Path(__file__).resolve().parent.parent / "DOC"


@dataclass
class RAGConfig:
    """RAG 管线相关配置（无向量、纯 BM25 版本）。"""

    api_key: str
    base_url: str
    chat_model: str


@dataclass
class TextChunk:
    """表示一个可检索的文本片段。"""

    content: str
    source: str


@runtime_checkable
class RetrieverBackend(Protocol):
    """检索后端协议。

    不同实现（BM25、本地向量检索、远程向量检索）都应实现同样的接口，
    便于在配置层进行切换。
    """

    def build_index(self) -> None:  # pragma: no cover - 协议本身不具体实现
        """基于 DOC 目录构建索引。"""

    def search(self, query: str, k: int = 4) -> List[Tuple[TextChunk, float]]:  # pragma: no cover
        """根据 query 返回得分最高的 k 个文本片段。"""


class BM25Backend:
    """非常简化版的 BM25 检索器（教学用）。

    说明：
    - 不依赖外部库，适合你当前不支持 embeddings 的环境；
    - 用分词 + 词频 + 文档频率近似实现 BM25 打分；
    - 对中文会比较粗糙，但足够支撑 demo 和原理讲解。
    """

    def __init__(self, chunks: List[TextChunk]) -> None:
        self._chunks = chunks
        # 预处理：构建倒排索引
        self._index = {}
        self._doc_lengths = []
        for doc_id, chunk in enumerate(chunks):
            tokens = self._tokenize(chunk.content)
            self._doc_lengths.append(len(tokens))
            seen = set()
            for token in tokens:
                if token not in self._index:
                    self._index[token] = {"df": 0, "postings": {}}
                postings = self._index[token]["postings"]
                if doc_id not in postings:
                    postings[doc_id] = 0
                postings[doc_id] += 1
                if token not in seen:
                    self._index[token]["df"] += 1
                    seen.add(token)

        self._avgdl = sum(self._doc_lengths) / max(len(self._doc_lengths), 1)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """非常简陋的中英文混合分词：按非字母数字和中文字符切分。"""

        # 将中文字符和英文/数字分开处理
        # 把所有非字母数字和非中文字符替换为空格
        text = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", " ", text)
        tokens = text.strip().split()
        return [t.lower() for t in tokens if t.strip()]

    def _score(self, query: str, doc_id: int) -> float:
        """对单个文档计算 BM25 近似得分。"""

        # 标准 BM25 的超参，取常用值
        k1 = 1.5
        b = 0.75

        tokens = self._tokenize(query)
        score = 0.0
        N = len(self._doc_lengths)
        dl = self._doc_lengths[doc_id]

        for token in tokens:
            if token not in self._index:
                continue
            entry = self._index[token]
            df = entry["df"]
            postings = entry["postings"]
            f = postings.get(doc_id, 0)
            if f == 0:
                continue

            # idf 部分
            idf = max(0.0, (N - df + 0.5) / (df + 0.5))
            # BM25 主体
            denom = f + k1 * (1 - b + b * dl / self._avgdl)
            score += idf * (f * (k1 + 1) / denom)

        return score

    def search(self, query: str, k: int = 4) -> List[Tuple[TextChunk, float]]:
        """根据 query 返回得分最高的 k 个文本片段。"""

        scores: List[Tuple[TextChunk, float]] = []
        for doc_id, chunk in enumerate(self._chunks):
            s = self._score(query, doc_id)
            if s > 0:
                scores.append((chunk, s))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


class LocalEmbeddingBackend:
    """基于本地 SentenceTransformer + FAISS 的向量检索后端。

    说明：
    - 使用轻量级中文向量模型，将 TextChunk 编码为向量；
    - 使用 FAISS 构建向量索引进行最近邻检索；
    - 如依赖或模型加载失败，应在上层优雅回退至 BM25 后端。
    """

    def __init__(self, chunks: List[TextChunk], model_name: str | None = None) -> None:
        if faiss is None or SentenceTransformer is None:  # 依赖缺失
            raise RuntimeError(
                "本地 embedding 后端需要 faiss-cpu 和 sentence-transformers，请先安装依赖。"
            )

        self._chunks = chunks
        # 选择一个中文效果较好的小模型，CPU 也能跑
        self._model_name = model_name or "moka-ai/m3e-small"
        self._model = SentenceTransformer(self._model_name)

        # 编码所有文本片段为向量并构建 FAISS 索引
        texts = [c.content for c in chunks]
        embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        # 为了用内积近似余弦相似度，先对向量做归一化
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

        self._index = index

    def _encode_query(self, query: str):
        q_emb = self._model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(q_emb)
        return q_emb

    def search(self, query: str, k: int = 4) -> List[Tuple[TextChunk, float]]:
        q_emb = self._encode_query(query)
        scores, indices = self._index.search(q_emb, k)
        results: List[Tuple[TextChunk, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((self._chunks[int(idx)], float(score)))
        return results


class RAGPipeline:
    """最小可用的 RAG 管线（BM25 关键词检索版）。

    使用步骤：
    1. 初始化实例：从环境变量中读取 API 信息；
    2. 调用 `build_index()` 构建关键词检索索引；
    3. 使用 `ask(question)` 发起基于检索增强的问答。
    """

    def __init__(self, config: RAGConfig) -> None:
        self._config = config
        self._llm_client = LLMClient(
            LLMConfig(
                api_key=config.api_key,
                base_url=config.base_url,
                model=config.chat_model,
            )
        )
        self._chunks: List[TextChunk] = []
        self._backend: RetrieverBackend | None = None

    # ---------- 文档加载与切分 ----------

    def _load_documents(self) -> List[TextChunk]:
        """从 DOC 目录加载所有 markdown/txt 文档，并做简单切分。

        注意：此方法与具体检索后端无关，只负责把原始文档转成 TextChunk，
        是 RAGPipeline 的通用数据准备层。
        """

        if not DOC_DIR.exists():
            raise RuntimeError(f"DOC 目录不存在: {DOC_DIR}")

        chunks: List[TextChunk] = []
        for path in DOC_DIR.rglob("*"):
            if path.suffix.lower() not in {".md", ".txt"}:
                continue
            text = path.read_text(encoding="utf-8")
            # 简单按两个换行或句号分段，避免 chunk 过长
            raw_parts = re.split(r"\n\n+|[。！？!?]", text)
            for part in raw_parts:
                content = part.strip()
                if not content:
                    continue
                if len(content) < 20:  # 太短的片段忽略
                    continue
                chunks.append(TextChunk(content=content, source=str(path)))

        if not chunks:
            raise RuntimeError(f"在 {DOC_DIR} 下没有找到任何可用文本片段")

        return chunks

    def _init_backend(self) -> None:
        """根据配置初始化检索后端。

        当前支持：
        - bm25: 关键词 BM25 检索（默认）
        - local_embedding: 本地向量检索（依赖 faiss + sentence-transformers）
        未知或初始化失败时，自动回退到 bm25，保证系统可用。
        """

        backend_name = os.getenv("RAG_BACKEND", "bm25").lower()

        self._chunks = self._load_documents()

        if backend_name == "local_embedding":
            try:
                model_name = os.getenv("RAG_EMBEDDING_MODEL", "moka-ai/m3e-small")
                self._backend = LocalEmbeddingBackend(self._chunks, model_name=model_name)
                return
            except Exception as exc:
                print(
                    f"[RAG] 初始化 local_embedding 后端失败，将回退到 BM25。原因: {exc}",
                )

        # 默认或回退方案：BM25
        self._backend = BM25Backend(self._chunks)

    def build_index(self) -> None:
        """构建检索索引（委托给具体后端）。"""

        self._init_backend()

    # ---------- RAG 问答相关 ----------

    def ask(self, question: str, top_k: int = 4) -> str:
        """基于 BM25 检索增强的问答。"""

        if self._backend is None:
            raise RuntimeError("索引尚未构建，请先调用 build_index().")

        results = self._backend.search(question, k=top_k)
        if not results:
            context_text = "(当前知识库中没有检索到相关内容)"
        else:
            context_text = "\n\n".join(
                f"[来源: {chunk.source} | 得分: {score:.2f}]\n{chunk.content}"
                for chunk, score in results
            )

        system_prompt = (
            "你是一个 RAG 助手，会根据提供的知识片段回答用户问题。\n"
            "必须优先依据给定的上下文回答，如果上下文中没有相关信息，"
            "要明确说出'在当前知识库中找不到答案'，不要编造。"
        )

        user_prompt = (
            "下面是知识库中与问题相关的内容片段：\n\n"
            f"{context_text}\n\n"
            "请基于以上内容回答用户的问题：\n"
            f"{question}\n"
        )

        return self._llm_client.chat(prompt=user_prompt, system_prompt=system_prompt)


def load_default_rag_pipeline() -> RAGPipeline:
    """从环境变量中构建一个默认的 RAGPipeline（BM25 版）。"""

    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    chat_model = os.getenv("LLM_MODEL")

    if not all([api_key, base_url, chat_model]):
        raise RuntimeError(
            "缺少 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL 环境变量，"
            "请在 .env 中进行配置。"
        )

    return RAGPipeline(
        RAGConfig(
            api_key=api_key,
            base_url=base_url,
            chat_model=chat_model,
        )
    )


def _demo() -> None:
    """自测脚本：演示一次最小 RAG 问答。

    问题会问到 README/DOC 中明确存在的知识点，方便你验证确实是 "基于知识库" 在回答。
    """

    pipeline = load_default_rag_pipeline()
    # 启动时基于 DOC 目录构建 BM25 检索索引
    pipeline.build_index()

    question = "这个 ai-agent-rag-lab 项目的目标和技术栈是什么？"
    answer = pipeline.ask(question)
    print("RAG 回答：\n", answer)


if __name__ == "__main__":
    _demo()
