from src.llm_client import load_default_client
from src.agent_demo import run_agent_once

def main() -> None:
    client = load_default_client()
    question = "现在几点？顺便用一两句话介绍一下这个项目。"
    answer = run_agent_once(client, question, max_steps=3)
    print("Agent 最终回答：")
    print(answer)

if __name__ == "__main__":
    main()