from langfuse import get_client, observe
from langgraph.types import Send

from agents.orchestrator import AGENT_REGISTRY
from graph.audit import audit_entry
from graph.guardrails import (
    DISCLAIMER,
    check_out_of_scope,
    detect_prompt_injection,
    enforce_citations,
    redact_phi,
)
from graph.llm import call_groq, extract_json
from graph.state import RESET, GraphState

MAX_RETRIES = 1
KNOWN_SOURCES = {name.upper() for name in AGENT_REGISTRY}

AGENT_DESCRIPTIONS = (
    "  - pubmed: peer-reviewed published research — evidence on treatment efficacy, mechanisms, "
    "risk factors, diagnostic criteria, published guidelines. Send SHORT keyword search terms "
    "(e.g. 'atrial fibrillation warfarin amiodarone interaction').\n"
    "  - openfda: the OFFICIAL FDA drug label ONLY — indications, warnings, contraindications, "
    "interactions, boxed warnings, as approved on the label. Send just the drug name (e.g. 'warfarin').\n"
    "  - clinicaltrials: ClinicalTrials.gov active/recruiting trials ONLY. Use only when the query "
    "explicitly asks about trials, studies enrolling patients, or research participation. "
    "Send a condition or disease name (e.g. 'atrial fibrillation').\n"
    "  - rag: internal knowledge base for foundational clinical explanations — pathophysiology, "
    "differential diagnosis, disease classification, staging, workup, general management overviews. "
    "Textbook-style reference material. Send a natural language clinical question "
    "(e.g. 'What is the pathophysiology of atrial fibrillation?').\n"
    "  - tavily: live web search for CURRENT events only — breaking news, this week's recall, a "
    "shortage, a just-announced guideline update — information NOT already covered by the sources "
    "above. Send a natural language question (e.g. 'latest FDA warfarin recall news').\n"
)

ROUTING_RULES = (
    "Routing rules:\n"
    "- Be selective: choose the SMALLEST set of agents that can answer the query. Do not include an "
    "agent 'just in case' — only include one if the query specifically needs what it uniquely provides.\n"
    "- A single-topic factual/explanatory question usually needs exactly ONE agent. Use rag for "
    "foundational/textbook explanations and pubmed for questions about published evidence or research "
    "findings specifically — rarely both for the same query.\n"
    "- Only add openfda when a specific drug's label information (warnings, interactions, indications) "
    "is asked about.\n"
    "- Only add clinicaltrials when trials/studies are explicitly mentioned.\n"
    "- Only add tavily when the query asks for something current/recent that a static source wouldn't have.\n"
    "- If the query is entirely unrelated to clinical/medical/pharmaceutical topics (e.g. small talk, "
    "weather, unrelated trivia), return an empty JSON object {}.\n"
)


def _filter_routing(routing: dict) -> dict[str, str]:
    return {k: v for k, v in routing.items() if k in AGENT_REGISTRY}


# ── Node: input guard (PHI redaction, injection heuristics, scope check) ────
@observe(name="input_guard", capture_input=False, capture_output=False)
def input_guard_node(state: GraphState) -> dict:
    """
    Runs first, before classify/router. Redacts PHI so nothing downstream —
    not even our own classify/router LLM calls — ever sees it; refuses
    outright on a detected prompt-injection attempt or a personal-treatment
    (out-of-scope) question.

    capture_input/output are off: `state["query"]` here is the one place raw,
    pre-redaction PHI exists in this graph, and @observe's default auto-capture
    would otherwise ship it straight to Langfuse's cloud — undermining the
    point of redacting it everywhere else. Only a redaction count is traced.
    """
    raw_query = state["query"]
    redacted_query, redactions = redact_phi(raw_query)
    injection_flags = detect_prompt_injection(redacted_query)
    out_of_scope, scope_reason = (False, "") if injection_flags else check_out_of_scope(redacted_query)
    get_client().update_current_span(
        metadata={"phi_redaction_count": len(redactions), "injection_detected": bool(injection_flags)}
    )

    entry = audit_entry(
        "input_guard",
        phi_redactions=redactions,
        injection_flags=injection_flags,
        out_of_scope=out_of_scope,
        scope_reason=scope_reason,
    )

    if injection_flags:
        return {
            "redacted_query": redacted_query,
            "phi_redactions": redactions,
            "injection_flags": injection_flags,
            "is_refused": True,
            "final_answer": (
                "I can't process that request — it looks like an attempt to override my "
                "instructions rather than a clinical research question."
            ),
            "audit": [entry],
        }

    if out_of_scope:
        return {
            "redacted_query": redacted_query,
            "phi_redactions": redactions,
            "injection_flags": injection_flags,
            "is_refused": True,
            "final_answer": (
                "I'm a clinical research assistant, not a substitute for medical care, and I "
                "can't give personalized treatment advice. Please consult a qualified clinician "
                f"about your specific situation. ({scope_reason})" if scope_reason else
                "I'm a clinical research assistant, not a substitute for medical care, and I "
                "can't give personalized treatment advice. Please consult a qualified clinician "
                "about your specific situation."
            ),
            "audit": [entry],
        }

    return {
        "redacted_query": redacted_query,
        "phi_redactions": redactions,
        "injection_flags": injection_flags,
        "is_refused": False,
        "audit": [entry],
    }


def route_after_input_guard(state: GraphState) -> str:
    return "END" if state.get("is_refused") else "classify"


# ── Node: classify ──────────────────────────────────────────────────────────
@observe(name="classify")
def classify_node(state: GraphState) -> dict:
    """Conversational messages get a direct reply and skip the rest of the graph."""
    query = state["redacted_query"]
    prompt = (
        "You are a classifier for a clinical research assistant system.\n"
        "Decide if the user's message is:\n"
        "  A) CONVERSATIONAL — greetings, personal statements, small talk, opinions, "
        "things that don't need medical literature/data search (e.g. 'My name is Yamini', 'Hello', 'How are you?')\n"
        "  B) RESEARCH — clinical/medical questions that need searching literature, drug data, trials, or health news\n\n"
        f"User message: '{query}'\n\n"
        'If A: reply with JSON {"type": "conversational", "reply": "<your friendly response>"}\n'
        'If B: reply with JSON {"type": "research"}\n'
        "Return ONLY the JSON. No explanation."
    )
    result = extract_json(call_groq(prompt)) or {}

    if result.get("type") == "conversational":
        reply = result.get("reply", "I'm here to help with clinical research questions!")
        return {
            "is_conversational": True,
            "final_answer": reply,
            "audit": [audit_entry("classify", conversational=True)],
        }
    # Explicitly set False every turn — `final_answer` from a prior turn in this
    # checkpointed session would otherwise still be truthy and never get overwritten
    # here, so branching on its mere presence would wrongly short-circuit follow-ups.
    return {"is_conversational": False, "audit": [audit_entry("classify", conversational=False)]}


def route_after_classify(state: GraphState) -> str:
    return "END" if state.get("is_conversational") else "router"


# ── Node: router (ported from agents.orchestrator._route_and_split) ────────
@observe(name="router")
def router_node(state: GraphState) -> dict:
    query = state["redacted_query"]
    history = state.get("history") or []
    context = (
        "\n\nRecent conversation, for resolving follow-ups:\n" + "\n".join(history[-2:])
        if history else ""
    )
    prompt = (
        "You are a routing assistant for a multi-agent clinical research system.\n"
        "You have access to these agents:\n"
        f"{AGENT_DESCRIPTIONS}\n"
        f"{ROUTING_RULES}\n"
        f"User query: '{query}'{context}\n\n"
        "Decide which agents are needed and write a focused sub-query for each one.\n"
        "Return ONLY a JSON object. Single-agent example:\n"
        '{"rag": "pathophysiology of atrial fibrillation"}\n'
        "Only include agents that are relevant — an empty object {} is a valid answer "
        "if none of the agents apply. No explanation, no extra text."
    )
    parsed = extract_json(call_groq(prompt))
    if parsed is None:
        # The model failed to return parseable JSON at all — fall back to every agent
        # as a safety net, rather than silently answering with no evidence at all.
        routing = {name: query for name in AGENT_REGISTRY}
    else:
        # A validly parsed {} means the model deliberately found nothing relevant —
        # respected as-is; fan_out_to_specialists routes that straight to confidence_gate.
        routing = _filter_routing(parsed)
    get_client().update_current_span(metadata={"routing": routing, "num_agents_selected": len(routing)})
    return {
        "routing": routing,
        "agent_outputs": RESET,
        "retry_count": 0,
        "audit": [audit_entry("router", routing=routing)],
    }


# ── Fan-out: one Send per selected agent, converging back on run_specialist ─
def fan_out_to_specialists(state: GraphState) -> list[Send] | str:
    routing = state["routing"]
    if not routing:
        # No agents selected (router legitimately found nothing relevant) — a Send
        # list of zero elements would leave nothing to advance the graph, so route
        # straight to confidence_gate, which already handles empty agent_outputs.
        return "confidence_gate"
    return [Send("run_specialist", {"agent_name": name, "sub_query": sub_query})
            for name, sub_query in routing.items()]


@observe(name="run_specialist")
def run_specialist_node(state: GraphState) -> dict:
    name = state["agent_name"]
    sub_query = state["sub_query"]
    # Rename per-agent so parallel Send branches show up as distinct, labeled
    # spans (e.g. "run_specialist:pubmed") instead of five identical "run_specialist"s.
    get_client().update_current_span(name=f"run_specialist:{name}", metadata={"agent": name})
    output = AGENT_REGISTRY[name](sub_query)
    return {
        "agent_outputs": {name: output},
        "audit": [audit_entry("run_specialist", agent=name, sub_query=sub_query)],
    }


# ── Node: confidence gate ───────────────────────────────────────────────────
@observe(name="confidence_gate")
def confidence_gate_node(state: GraphState) -> dict:
    outputs = state["agent_outputs"]
    if not outputs:
        return {
            "confidence": "low",
            "confidence_reason": "No specialist produced output.",
            "audit": [audit_entry("confidence_gate", confidence="low")],
        }

    joined = "\n\n".join(f"[{k.upper()}]\n{v}" for k, v in outputs.items())
    prompt = (
        "You are an evidence-quality reviewer for a clinical research assistant.\n"
        f"User question: '{state['redacted_query']}'\n\n"
        f"Specialist findings:\n{joined}\n\n"
        "Assess whether this evidence is sufficient and non-conflicting to confidently answer the "
        "question. Weak, missing ('no results found'), or contradictory evidence should be 'low'.\n"
        'Return ONLY JSON: {"confidence": "high"|"low", "reason": "<one sentence>"}'
    )
    result = extract_json(call_groq(prompt)) or {}
    confidence = result.get("confidence", "high")
    reason = result.get("reason", "")
    return {
        "confidence": confidence,
        "confidence_reason": reason,
        "audit": [audit_entry("confidence_gate", confidence=confidence, reason=reason)],
    }


def should_retry(state: GraphState) -> str:
    if state.get("confidence") == "low" and state.get("retry_count", 0) < MAX_RETRIES:
        return "reflect"
    return "synthesize"


# ── Node: reflect (rewrite sub-queries, at most MAX_RETRIES times) ─────────
@observe(name="reflect")
def reflect_node(state: GraphState) -> dict:
    query = state["redacted_query"]
    outputs = state["agent_outputs"]
    reason = state.get("confidence_reason", "")
    joined = "\n\n".join(f"[{k.upper()}]\n{v}" for k, v in outputs.items())
    prompt = (
        "You are a routing assistant refining a research pass that came back weak or conflicting.\n"
        f"Original question: '{query}'\n"
        f"Why the evidence was insufficient: {reason}\n"
        f"Previous specialist findings:\n{joined}\n\n"
        "Available agents:\n"
        f"{AGENT_DESCRIPTIONS}\n"
        "Rewrite the sub-queries (and/or pick different agents) to close the gap. "
        "Return ONLY a JSON object mapping agent name to a focused sub-query, same format as before."
    )
    routing = _filter_routing(extract_json(call_groq(prompt)) or {})
    if not routing:
        routing = state["routing"]
    new_retry_count = state.get("retry_count", 0) + 1
    get_client().update_current_span(
        metadata={"retry_count": new_retry_count, "revised_routing": routing, "reason": reason}
    )
    return {
        "routing": routing,
        "retry_count": new_retry_count,
        "audit": [audit_entry("reflect", routing=routing)],
    }


# ── Node: synthesize (citation-enforced) ────────────────────────────────────
@observe(name="synthesize")
def synthesize_node(state: GraphState) -> dict:
    query = state["redacted_query"]
    outputs = state["agent_outputs"]
    sections = "\n\n".join(f"[{source.upper()} AGENT]\n{summary}" for source, summary in outputs.items())
    prompt = (
        f"You are a helpful clinical research assistant. A user asked: '{query}'\n\n"
        f"Multiple specialist agents gathered the following information:\n\n{sections}\n\n"
        "Synthesize this into a single, well-structured, comprehensive answer.\n\n"
        "CITATION RULES (mandatory):\n"
        "- Every factual claim must end with a bracketed tag naming the specialist(s) that support it, "
        "e.g. '...reduces stroke risk [PUBMED].' or '...may cause bleeding [OPENFDA][TAVILY].'\n"
        "- Only cite specialists listed above; never invent a source.\n"
        "- If no specialist supports a claim, omit the claim rather than stating it uncited.\n"
        "- Do not repeat information. Prioritize accuracy and clarity."
    )
    answer = call_groq(prompt)
    return {
        "final_answer": answer,
        "history": [f"Q: {query}\nA: {answer}"],
        "audit": [audit_entry("synthesize", confidence=state.get("confidence"))],
    }


REVIEW_MESSAGE = (
    "I wasn't able to gather confident, non-conflicting evidence to answer this — "
    "even after retrying with refined searches. Rather than guess, I've flagged this "
    "question for a specialist's review."
)


# ── Node: human review gate (low-confidence answers aren't returned as-is) ─
@observe(name="human_review_gate")
def human_review_gate_node(state: GraphState) -> dict:
    """
    should_retry only lands here with confidence still "low" once the
    reflection retry budget (MAX_RETRIES) is exhausted — the high-confidence
    path never sets confidence back to "low" before reaching synthesize. So a
    "low" confidence at this point means the retry already happened and
    didn't help; withhold the draft rather than hand the user weak evidence.
    """
    if state.get("confidence") == "low":
        return {
            "needs_review": True,
            "drafted_answer": state["final_answer"],
            "final_answer": REVIEW_MESSAGE,
            "audit": [audit_entry("human_review_gate", needs_review=True)],
        }
    return {"needs_review": False, "audit": [audit_entry("human_review_gate", needs_review=False)]}


# ── Node: output guard (strip uncited claims, append disclaimer) ───────────
@observe(name="output_guard")
def output_guard_node(state: GraphState) -> dict:
    """
    Defense-in-depth after synthesize: the synthesis prompt already asks the
    LLM to self-cite, but this node deterministically re-checks rather than
    trusting the model's compliance, then appends the disclaimer every
    clinical answer must carry.

    A needs_review answer is already our own fixed REVIEW_MESSAGE, not LLM
    output making clinical claims — citation enforcement has nothing to check
    there (and would otherwise strip the whole message, since it's a long
    uncited line), so it's skipped.
    """
    if state.get("needs_review"):
        return {
            "final_answer": state["final_answer"] + DISCLAIMER,
            "audit": [audit_entry("output_guard", stripped_claims=0, skipped="needs_review")],
        }
    cleaned, stats = enforce_citations(state["final_answer"], KNOWN_SOURCES)
    final = cleaned.strip() + DISCLAIMER
    return {
        "final_answer": final,
        "audit": [audit_entry("output_guard", **stats)],
    }
