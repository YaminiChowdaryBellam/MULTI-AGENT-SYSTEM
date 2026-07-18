"""
ClinicalTrials.gov v2 API search (clinicaltrials.gov/api/v2/studies). Keyless.
"""

import time

import requests

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"


def _ct_get(params: dict, retries: int = 3) -> requests.Response | None:
    for attempt in range(retries):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            if attempt == retries - 1:
                print(f"[clinicaltrials_tool] Failed after {retries} attempts — {e}. Returning empty results.")
                return None
            wait = 2 ** attempt
            print(f"[clinicaltrials_tool] Retry {attempt + 1}/{retries} after {wait}s — {e}")
            time.sleep(wait)
    return None


def _parse_study(study: dict) -> dict:
    protocol = study.get("protocolSection", {})
    ident = protocol.get("identificationModule", {})
    status = protocol.get("statusModule", {})
    design = protocol.get("designModule", {})
    contacts = protocol.get("contactsLocationsModule", {})

    locations = [
        loc.get("city", "") + (f", {loc.get('country')}" if loc.get("country") else "")
        for loc in contacts.get("locations", [])[:5]
    ]

    return {
        "nct_id": ident.get("nctId", ""),
        "title": ident.get("briefTitle", ""),
        "status": status.get("overallStatus", ""),
        "phases": design.get("phases", []),
        "locations": locations,
    }


def search_clinical_trials(condition: str, max_results: int = 5) -> list[dict]:
    params = {
        "query.cond": condition,
        "pageSize": max_results,
        "filter.overallStatus": "RECRUITING,ACTIVE_NOT_RECRUITING",
    }
    resp = _ct_get(params)
    if resp is None:
        return []
    return [_parse_study(s) for s in resp.json().get("studies", [])]
