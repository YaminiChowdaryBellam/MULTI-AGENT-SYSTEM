import json
from graph.llm import call_groq
from tools.openfda_tool import search_drug_label


def run_openfda_agent(query: str) -> str:
    labels = search_drug_label(query)
    if not labels:
        return "No openFDA drug label found for this query."

    raw = json.dumps(labels, indent=2)
    prompt = (
        f"You are a clinical pharmacology assistant. A user asked: '{query}'\n\n"
        f"Here is the FDA-approved drug label data:\n{raw}\n\n"
        "Summarize the indications, warnings, and known drug interactions in 3-5 sentences. "
        "Flag any boxed warnings or contraindications explicitly."
    )
    return call_groq(prompt)
