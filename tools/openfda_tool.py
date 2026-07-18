"""
openFDA drug label search (api.fda.gov/drug/label.json). Keyless.
"""

import time

import requests

BASE_URL = "https://api.fda.gov/drug/label.json"

# Label sections most relevant to a clinical Q&A assistant.
FIELDS = [
    "indications_and_usage",
    "warnings",
    "warnings_and_cautions",
    "drug_interactions",
    "boxed_warning",
    "contraindications",
]


def _fda_get(params: dict, retries: int = 3) -> requests.Response | None:
    for attempt in range(retries):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)
            if resp.status_code == 404:
                # openFDA returns 404 (not an error body) when a search has zero matches.
                return None
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            if attempt == retries - 1:
                print(f"[openfda_tool] Failed after {retries} attempts — {e}. Returning empty results.")
                return None
            wait = 2 ** attempt
            print(f"[openfda_tool] Retry {attempt + 1}/{retries} after {wait}s — {e}")
            time.sleep(wait)
    return None


def search_drug_label(drug_name: str, limit: int = 1) -> list[dict]:
    params = {
        "search": f'openfda.generic_name:"{drug_name}" OR openfda.brand_name:"{drug_name}"',
        "limit": limit,
    }
    resp = _fda_get(params)
    if resp is None:
        return []

    results = []
    for entry in resp.json().get("results", []):
        openfda = entry.get("openfda", {})
        record = {
            "brand_name": openfda.get("brand_name", [""])[0] if openfda.get("brand_name") else "",
            "generic_name": openfda.get("generic_name", [""])[0] if openfda.get("generic_name") else "",
            "manufacturer": openfda.get("manufacturer_name", [""])[0] if openfda.get("manufacturer_name") else "",
        }
        for field in FIELDS:
            value = entry.get(field)
            if value:
                record[field] = value[0][:1000]
        results.append(record)
    return results
