"""
Embedding-similarity router — an alternative to router_node's LLM-based
routing, built for the Phase 5 A/B experiment (LLM router vs
embedding-similarity router). Not wired into the production graph; it's
evaluated standalone via evals/ab_experiment.py.

Each agent is represented by a handful of canonical example queries (adapted
from graph/nodes.py's AGENT_DESCRIPTIONS). A query is routed to every agent
whose closest example exceeds SIMILARITY_THRESHOLD — zero LLM calls, so this
variant costs ~$0 and runs entirely locally via a small sentence-transformers
model, at the cost of the nuance an LLM router gets from reading full
natural-language instructions and few-shot examples.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from agents.orchestrator import AGENT_REGISTRY

MODEL_NAME = "all-MiniLM-L6-v2"
# Swept 0.35-0.60 against the gold set; 0.45 is the accuracy peak (~34%) —
# noticeably weaker than the LLM router's ~70%, which is the point of the
# comparison, but not artificially crippled by an unreasonable cutoff.
SIMILARITY_THRESHOLD = 0.45

AGENT_EXAMPLES = {
    "pubmed": [
        "What does the published research say about this treatment?",
        "What is the evidence for this therapy's efficacy?",
        "Summarize recent studies on this drug class.",
        "What are the diagnostic criteria according to recent research?",
        "What are the risk factors identified in the literature?",
    ],
    "openfda": [
        "What are the warnings for this drug?",
        "What are the contraindications for this medication?",
        "What drug interactions does this medication have?",
        "What is the FDA label for this drug?",
        "What are the boxed warnings for this drug?",
    ],
    "clinicaltrials": [
        "Are there active clinical trials for this condition?",
        "What trials are recruiting for this disease?",
        "Are there ongoing studies enrolling patients?",
        "What phase 3 trials are active for this treatment?",
    ],
    "rag": [
        "What is the pathophysiology of this condition?",
        "Explain the differential diagnosis for this symptom.",
        "Describe the management of this disease.",
        "What is the workup for this presentation?",
        "Explain the classification and staging of this condition.",
    ],
    "tavily": [
        "What is the latest news on this topic?",
        "Has there been a recent recall of this drug?",
        "What are the newest guideline updates?",
        "What is the current shortage status of this medication?",
    ],
}

_model = None
_agent_embeddings = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _get_agent_embeddings():
    global _agent_embeddings
    if _agent_embeddings is None:
        model = _get_model()
        _agent_embeddings = {
            name: model.encode(examples)
            for name, examples in AGENT_EXAMPLES.items()
            if name in AGENT_REGISTRY
        }
    return _agent_embeddings


def embedding_route(query: str, threshold: float = SIMILARITY_THRESHOLD) -> dict[str, str]:
    """
    Returns {agent_name: query} for every agent whose best-matching example
    exceeds `threshold` cosine similarity — the same {agent: sub_query} shape
    router_node returns, but computed with zero LLM calls. Every agent gets
    the full, unmodified query as its sub-query (no LLM available here to
    write a focused one).
    """
    model = _get_model()
    agent_embeddings = _get_agent_embeddings()
    query_embedding = model.encode([query])

    routing = {}
    for name, example_embeddings in agent_embeddings.items():
        similarities = cosine_similarity(query_embedding, example_embeddings)[0]
        if similarities.max() >= threshold:
            routing[name] = query
    return routing
