import os

import requests
from tavily import TavilyClient
from tavily.errors import BadRequestError, ForbiddenError, UsageLimitExceededError


def search_web(query: str, max_results: int = 5) -> list[dict]:
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    try:
        response = client.search(query=query, max_results=max_results)
    except (requests.RequestException, BadRequestError, ForbiddenError, UsageLimitExceededError) as e:
        # Tavily rejects some queries outright (e.g. an empty/malformed sub-query
        # from a weaker router model, or a plain HTTP error) rather than just
        # returning zero results — degrade gracefully instead of crashing the
        # whole graph run over one specialist's bad input.
        print(f"[tavily_tool] Tavily search failed for {query!r} — {e}. Returning empty results.")
        return []

    results = []
    for r in response.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", "")[:500],
        })
    return results
