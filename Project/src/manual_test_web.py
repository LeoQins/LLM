from src.llm_client import load_default_client
from src.web_automation_demo import fetch_page_text, summarize_page

def main() -> None:
    client = load_default_client()
    url = "https://www.fwxgx.com/questions/2785101"
    page = fetch_page_text(url)
    print(f"抓取成功：{page.url}，标题：{page.title!r}，文本长度：{len(page.text)}")
    summary = summarize_page(client, page)
    print("网页总结：")
    print(summary)

if __name__ == "__main__":
    main()