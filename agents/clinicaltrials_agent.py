import json
from graph.llm import call_groq
from tools.clinicaltrials_tool import search_clinical_trials


def run_clinicaltrials_agent(query: str) -> str:
    trials = search_clinical_trials(query)
    if not trials:
        return "No active ClinicalTrials.gov trials found for this query."

    raw = json.dumps(trials, indent=2)
    prompt = (
        f"You are a clinical trials research assistant. A user asked: '{query}'\n\n"
        f"Here are active/recruiting trials found on ClinicalTrials.gov:\n{raw}\n\n"
        "Summarize the most relevant trials in 3-5 sentences, mentioning phase, status, "
        "and NCT IDs where relevant."
    )
    return call_groq(prompt)
