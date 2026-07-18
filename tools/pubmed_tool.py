"""
PubMed search via NCBI E-utilities (esearch -> efetch). Keyless — NCBI asks
scrapers to self-identify via `tool`/`email` params rather than an API key.
"""

import os
import time
import xml.etree.ElementTree as ET

import requests

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL_NAME = "multi-agent-clinical-research-assistant"
CONTACT_EMAIL = os.getenv("NCBI_EMAIL", "")


def _ncbi_get(endpoint: str, params: dict, retries: int = 3) -> requests.Response:
    params = {**params, "tool": TOOL_NAME}
    if CONTACT_EMAIL:
        params["email"] = CONTACT_EMAIL
    for attempt in range(retries):
        try:
            resp = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=30)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            if attempt == retries - 1:
                print(f"[pubmed_tool] Failed after {retries} attempts — {e}. Returning None.")
                return None
            wait = 2 ** attempt
            print(f"[pubmed_tool] Retry {attempt + 1}/{retries} after {wait}s — {e}")
            time.sleep(wait)


def _parse_article(elem: ET.Element) -> dict:
    pmid = elem.findtext("MedlineCitation/PMID", "")

    title_elem = elem.find("MedlineCitation/Article/ArticleTitle")
    title = "".join(title_elem.itertext()) if title_elem is not None else ""

    abstract_parts = []
    for at in elem.findall("MedlineCitation/Article/Abstract/AbstractText"):
        label = at.get("Label", "")
        text = "".join(at.itertext())
        abstract_parts.append(f"{label}: {text}" if label else text)
    abstract = "\n".join(abstract_parts)[:1000]

    pub_date = ""
    pd = elem.find("MedlineCitation/Article/Journal/JournalIssue/PubDate")
    if pd is not None:
        year = pd.findtext("Year", "")
        month = pd.findtext("Month", "")
        pub_date = f"{year}-{month}".strip("-")

    return {"pmid": pmid, "title": title, "abstract": abstract, "published": pub_date}


def search_pubmed(query: str, max_results: int = 5) -> list[dict]:
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    search_resp = _ncbi_get("esearch.fcgi", search_params)
    if search_resp is None:
        return []

    pmids = search_resp.json().get("esearchresult", {}).get("idlist", [])
    if not pmids:
        return []

    fetch_params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }
    fetch_resp = _ncbi_get("efetch.fcgi", fetch_params)
    if fetch_resp is None:
        return []

    root = ET.fromstring(fetch_resp.content)
    return [
        article for elem in root.findall("PubmedArticle")
        if (article := _parse_article(elem)).get("pmid")
    ]
