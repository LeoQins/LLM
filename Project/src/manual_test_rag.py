from src.rag_pipeline import load_default_rag_pipeline

def main() -> None:
    pipeline = load_default_rag_pipeline()
    pipeline.build_index()
    question = "这个 ai-agent-rag-lab 项目的目标和技术栈是什么？"
    answer = pipeline.ask(question)
    print("RAG 回答：")
    print(answer)

if __name__ == "__main__":
    main()