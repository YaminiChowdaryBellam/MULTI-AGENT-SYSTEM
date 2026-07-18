import os
from tavily import TavilyClient


def search_web(query: str, max_results: int = 5) -> list[dict]:
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    response = client.search(query=query, max_results=max_results)
    results = []
    for r in response.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", "")[:500],
        })
    return results
