"""基础阶段：最小可运行的大模型 API 调用封装。

本模块的目标：
- 提供一个干净的 LLMClient 类，负责和大模型 HTTP API 交互；
- 隐藏底层 HTTP 请求细节，后面可以无痛替换为不同的大模型厂商；
- 为后续接入 LangChain、Agent、RAG 打基础。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx
from dotenv import load_dotenv
import os


load_dotenv()


@dataclass
class LLMConfig:
    """大模型客户端配置。

    这里保持足够通用，方便你切换 OpenAI / Azure OpenAI / Moonshot / 讯飞星火等。
    实际使用时，只需要在 .env 中配置好对应的 key 和 base_url 即可。
    """

    api_key: str
    base_url: str
    model: str
    timeout: float = 30.0


class LLMClient:
    """最小可用的大模型 HTTP 客户端封装。"""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client = httpx.Client(base_url=config.base_url, timeout=config.timeout)

    def chat(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """向大模型发起最简单的一次对话请求。

        不同厂商的请求体格式不同，这里以兼容 OpenAI-compatible 接口为例：
        - POST /v1/chat/completions
        - body: {model, messages: [{role, content}, ...]}
        """

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": 0.7,
        }

        response = self._client.post("/v1/chat/completions", json=payload, headers={
            "Authorization": f"Bearer {self._config.api_key}",
        })
        response.raise_for_status()
        data = response.json()

        # 兼容 OpenAI 风格返回
        return data["choices"][0]["message"]["content"].strip()


def load_default_client() -> LLMClient:
    """从环境变量中加载默认的 LLMClient。

    你需要在项目根目录创建 .env 文件，配置：
    - LLM_API_KEY
    - LLM_BASE_URL（例如 https://api.openai.com 或者你自建的兼容网关）
    - LLM_MODEL（例如 gpt-4.1, gpt-4o, qwen-max 等）
    """

    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model = os.getenv("LLM_MODEL")

    if not api_key or not base_url or not model:
        raise RuntimeError(
            "缺少 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL 环境变量，请在项目根目录创建 .env 并配置。"
        )

    return LLMClient(
        LLMConfig(
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
    )


def _demo() -> None:
    """开发自测用的小脚本：演示一次最简单的对话调用。"""

    client = load_default_client()
    answer = client.chat("用一句话介绍你自己，尽量简洁。")
    print("模型回答：", answer)


if __name__ == "__main__":
    _demo()
