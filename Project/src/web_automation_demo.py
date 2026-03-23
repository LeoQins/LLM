"""阶段6：自动化演示 - 使用 HTTP 抓取网页并用大模型总结。

功能：
- 给定一个 URL，使用 httpx 发起 HTTP 请求抓取 HTML 内容；
- 做一次简单的 HTML -> 纯文本清洗；
- 调用现有 LLMClient，对页面内容做中文总结；
- 提供命令行交互式 Demo。

说明：
- 不依赖浏览器或 Playwright，更接近 curl 的抓取方式；
- 后续可以作为 Agent 工具或 FastAPI 接口的实现基础。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

from src.llm_client import LLMClient, load_default_client


@dataclass
class PageContent:
    """表示抓取到的网页内容。"""

    url: str
    title: str
    text: str


def _html_to_text(html: str) -> str:
    """非常简化的 HTML -> 文本转换（教学用）。

    - 去掉 <script>/<style> 等标签内容；
    - 去掉所有 HTML 标签；
    - 合并多余空白。
    """

    # 去掉 script 和 style 中的内容
    html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
    html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.IGNORECASE)

    # 提取 <title>
    title_match = re.search(r"<title>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    title = title_match.group(1).strip() if title_match else ""

    # 去掉所有其他 HTML 标签
    text = re.sub(r"<[^>]+>", " ", html)
    # 合并空白
    text = re.sub(r"\s+", " ", text).strip()

    return title, text


def fetch_page_text(url: str, timeout: float = 20.0) -> PageContent:
    """使用 httpx 抓取网页 HTML 并转为 PageContent。"""

    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        html = resp.text

    title, text = _html_to_text(html)
    return PageContent(url=url, title=title, text=text)


def summarize_page(client: LLMClient, page: PageContent, max_chars: int = 4000) -> str:
    """调用大模型对页面内容做中文总结。"""

    content = page.text[:max_chars]

    system_prompt = (
        "你是一个善于阅读和总结网页内容的中文助手。\n"
        "请用简洁的方式提炼网页的关键信息，包括但不限于：\n"
        "- 网页主要主题\n"
        "- 关键结论或要点\n"
        "- 如果是技术/产品页面，可以简单说明涉及的核心功能或技术栈。\n"
    )

    user_prompt = (
        f"这是我通过 HTTP 抓取到的网页内容片段（URL: {page.url}，标题: {page.title}）：\n\n"
        f"{content}\n\n"
        "请用中文给出一个结构化的总结，适合让我快速了解这个页面在讲什么。"
    )

    return client.chat(prompt=user_prompt, system_prompt=system_prompt)


def _demo() -> None:
    """命令行 Demo：输入 URL，自动抓取并总结页面内容。"""

    client = load_default_client()
    print("网页自动化抓取 + 总结 Demo（HTTP 版），输入 URL 并回车（输入 'exit' 退出）。\n")

    while True:
        url = input("请输入 URL：").strip()
        if not url:
            continue
        if url.lower() in {"exit", "quit"}:
            print("已退出。")
            break

        try:
            page = fetch_page_text(url)
            print(f"\n抓取成功：标题 = {page.title!r}，文本长度 = {len(page.text)} 字符。\n")

            summary = summarize_page(client, page)
            print("=== 网页内容总结 ===")
            print(summary)
            print("-" * 60)
        except Exception as exc:
            print(f"抓取或总结过程中出现错误：{exc}")
            print("请检查 URL 是否可访问，或稍后重试。\n")


if __name__ == "__main__":
    _demo()
