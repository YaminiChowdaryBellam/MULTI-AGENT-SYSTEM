import os
import json
import re
from groq import Groq

# Import all specialist agents
from agents.pubmed_agent import run_pubmed_agent
from agents.openfda_agent import run_openfda_agent
from agents.clinicaltrials_agent import run_clinicaltrials_agent
from agents.rag_agent import run_rag_agent
from agents.tavily_agent import run_tavily_agent

_client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.1-8b-instant"

# Map agent names to their runner functions
AGENT_REGISTRY = {
    "pubmed": run_pubmed_agent,
    "openfda": run_openfda_agent,
    "clinicaltrials": run_clinicaltrials_agent,
    "rag": run_rag_agent,
    "tavily": run_tavily_agent,
}


def _is_conversational(query: str) -> tuple[bool, str]:
    """
    Ask Groq whether the query is conversational (greetings, opinions, small talk)
    or a researchable question that needs agents.
    Returns (True, direct_reply) if conversational, (False, "") if it needs agents.
    """
    prompt = (
        "You are a classifier for a clinical research assistant system.\n"
        "Decide if the user's message is:\n"
        "  A) CONVERSATIONAL — greetings, personal statements, small talk, opinions, "
        "things that don't need medical literature/data search (e.g. 'My name is Yamini', 'Hello', 'How are you?')\n"
        "  B) RESEARCH — clinical/medical questions that need searching literature, drug data, trials, or health news\n\n"
        f"User message: '{query}'\n\n"
        "If A: reply with JSON {\"type\": \"conversational\", \"reply\": \"<your friendly response>\"}\n"
        "If B: reply with JSON {\"type\": \"research\"}\n"
        "Return ONLY the JSON. No explanation."
    )
    response = _client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content.strip()

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    cleaned = match.group(0) if match else raw

    try:
        result = json.loads(cleaned)
        if result.get("type") == "conversational":
            return True, result.get("reply", "I'm here to help with research questions!")
    except json.JSONDecodeError:
        pass
    return False, ""


def _route_and_split(query: str) -> dict[str, str]:
    """
    Ask Groq two things in one call:
      1. Which agents are relevant for this query.
      2. What specific sub-query to send each agent (keywords, not the full question).
    Returns a dict like {"pubmed": "atrial fibrillation treatment", "tavily": "latest AFib guidelines"}.
    """
    prompt = (
        "You are a routing assistant for a multi-agent clinical research system.\n"
        "You have access to these agents:\n"
        "  - pubmed: searches peer-reviewed medical literature (PubMed). Send SHORT keyword search terms "
        "(e.g. 'atrial fibrillation warfarin amiodarone interaction').\n"
        "  - openfda: looks up FDA drug labels — indications, warnings, interactions, recalls. "
        "Send just the drug name (e.g. 'warfarin').\n"
        "  - clinicaltrials: searches ClinicalTrials.gov for active/recruiting trials. "
        "Send a condition or disease name (e.g. 'atrial fibrillation').\n"
        "  - rag: queries our internal clinical knowledge base (StatPearls-derived). Send a natural language "
        "clinical question (e.g. 'What are treatment options for atrial fibrillation?').\n"
        "  - tavily: searches the live web. Best for current health news, drug recalls, or new clinical "
        "guidelines. Send a natural language question (e.g. 'latest FDA warfarin recall news').\n\n"
        f"User query: '{query}'\n\n"
        "Decide which agents are needed and write a focused sub-query for each one.\n"
        "Return ONLY a JSON object. Example:\n"
        "{\"pubmed\": \"atrial fibrillation treatment\", \"openfda\": \"warfarin\", \"clinicaltrials\": \"atrial fibrillation\"}\n"
        "Only include agents that are relevant. No explanation, no extra text."
    )
    response = _client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences (```json ... ```) that Groq sometimes wraps around JSON
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    cleaned = match.group(0) if match else raw

    # Parse the routing decision; fall back to sending full query to all agents
    try:
        routing = json.loads(cleaned)
        # Filter to valid agents only
        return {k: v for k, v in routing.items() if k in AGENT_REGISTRY}
    except json.JSONDecodeError:
        print(f"[Orchestrator] Failed to parse routing response: {raw!r}")
        return {name: query for name in AGENT_REGISTRY}


def _synthesize(query: str, agent_outputs: dict[str, str]) -> str:
    """Ask Groq to merge all agent summaries into one final coherent answer."""
    sections = "\n\n".join(
        f"[{source.upper()} AGENT]\n{summary}"
        for source, summary in agent_outputs.items()
    )
    prompt = (
        f"You are a helpful clinical research assistant. A user asked: '{query}'\n\n"
        f"Multiple specialist agents gathered the following information:\n\n{sections}\n\n"
        "Synthesize all of this into a single, well-structured, and comprehensive answer. "
        "Do not repeat information. Prioritize accuracy and clarity."
    )
    response = _client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def run_orchestrator(query: str) -> str:
    """
    Main entry point for the multi-agent system.
    1. Routes the query and splits it into focused sub-queries per agent.
    2. Dispatches each agent with only its relevant sub-query.
    3. Synthesizes all responses into one final answer.
    """
    print(f"\n[Orchestrator] Received query: '{query}'")

    # Step 0: Check if query is conversational — if so, reply directly without agents
    is_conversational, direct_reply = _is_conversational(query)
    if is_conversational:
        print("[Orchestrator] Query is conversational — replying directly.")
        return direct_reply

    # Step 1: Routing + splitting — get {agent_name: sub_query} dict
    routing = _route_and_split(query)
    print(f"[Orchestrator] Routing plan: {routing}")

    # Step 2: Dispatch — each agent receives its own focused sub-query
    agent_outputs = {}
    for agent_name, sub_query in routing.items():
        print(f"[Orchestrator] Running {agent_name} agent with sub-query: '{sub_query}'")
        agent_fn = AGENT_REGISTRY[agent_name]
        agent_outputs[agent_name] = agent_fn(sub_query)

    # Step 3: Synthesis — combine all summaries into one final answer
    print("[Orchestrator] Synthesizing final answer...")
    final_answer = _synthesize(query, agent_outputs)

    return final_answer
