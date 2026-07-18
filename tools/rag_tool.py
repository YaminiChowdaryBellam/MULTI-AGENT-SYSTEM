"""
HTTP client for the RAG-medical-system's `POST /ask` endpoint. Not keyless in
the sense of an API key, but requires the RAG service to be running locally
(or at RAG_SERVICE_URL) — degrades gracefully if it's unreachable.
"""

import os

import requests

BASE_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")


def ask_rag(question: str) -> dict | None:
    """
    Calls the RAG service's /ask endpoint. Returns None if the service is
    down, times out, or errors — callers should treat that as "no answer".
    """
    try:
        resp = requests.post(f"{BASE_URL}/ask", json={"question": question}, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"[rag_tool] RAG service unreachable at {BASE_URL} — {e}. Skipping.")
        return None
