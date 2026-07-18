import json
from graph.llm import call_groq
from tools.tavily_tool import search_web


def run_tavily_agent(query: str) -> str:
    results = search_web(query)
    if not results:
        return "No web results found for this query."

    raw = json.dumps(results, indent=2)
    prompt = (
        f"You are a web research assistant. A user asked: '{query}'\n\n"
        f"Here are the top web search results:\n{raw}\n\n"
        "Summarize the most relevant and up-to-date information in 3-5 sentences. "
        "Cite sources by title where helpful."
    )
    return call_groq(prompt)
