from tools.rag_tool import ask_rag


def run_rag_agent(query: str) -> str:
    """
    Unlike the other specialists, the RAG service already returns a
    synthesized, cited answer — so this agent just formats it rather than
    running another LLM pass over it.
    """
    result = ask_rag(query)
    if result is None:
        return "RAG knowledge base service is unavailable — no internal answer for this query."

    answer = result.get("answer", "")
    sources = result.get("sources", [])
    if not sources:
        return answer

    citations = "\n".join(
        f"- {s.get('title', 'Untitled')}"
        + (f" (PMID: {s['pmid']})" if s.get("pmid") else "")
        for s in sources
    )
    return f"{answer}\n\nSources:\n{citations}"
