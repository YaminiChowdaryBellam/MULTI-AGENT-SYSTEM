import json
from graph.llm import call_groq
from tools.pubmed_tool import search_pubmed


def run_pubmed_agent(query: str) -> str:
    articles = search_pubmed(query)
    if not articles:
        return "No PubMed articles found for this query."

    raw = json.dumps(articles, indent=2)
    prompt = (
        f"You are a clinical research assistant. A user asked: '{query}'\n\n"
        f"Here are the most relevant PubMed articles found:\n{raw}\n\n"
        "Summarize the key clinical findings and themes across these articles in 3-5 sentences. "
        "Cite specific articles by title and PMID where relevant."
    )
    return call_groq(prompt)
