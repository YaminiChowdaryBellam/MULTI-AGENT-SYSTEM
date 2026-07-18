import argparse
import uuid

from dotenv import load_dotenv

# Load API keys from .env before any agent imports
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Clinical Research Assistant")
    parser.add_argument(
        "--legacy", action="store_true",
        help="Use the original hand-rolled orchestrator (agents/orchestrator.py) instead of the LangGraph core.",
    )
    args = parser.parse_args()

    if args.legacy:
        from agents.orchestrator import run_orchestrator as ask
        engine_label = "hand-rolled orchestrator (--legacy)"
    else:
        from graph.build import run_graph
        thread_id = str(uuid.uuid4())

        def ask(query):
            return run_graph(query, thread_id=thread_id)

        engine_label = "LangGraph agentic core"

    print("=" * 60)
    print("  Multi-Agent Clinical Research Assistant")
    print(f"  Engine: {engine_label}")
    print("  Powered by Groq + PubMed + openFDA + ClinicalTrials.gov + RAG + Tavily")
    print("=" * 60)

    # Accept queries in a loop until the user types 'exit'
    while True:
        query = input("\nAsk a question (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("Goodbye!")
            break
        if not query:
            continue

        answer = ask(query)

        print("\n" + "=" * 60)
        print("FINAL ANSWER")
        print("=" * 60)
        print(answer)
        print("=" * 60)


if __name__ == "__main__":
    main()
