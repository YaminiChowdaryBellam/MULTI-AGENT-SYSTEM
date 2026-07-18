"""
Graph state schema for the LangGraph agentic core.

Channels without an `Annotated` reducer are simply overwritten by whichever
node returns them last. `agent_outputs` and `audit` need reducers because
multiple parallel `run_specialist` branches write to them in the same
superstep, and `history` accumulates across turns in a checkpointed session.
"""

import operator
from typing import Annotated, TypedDict


# A plain string (not a custom object) so it survives the checkpointer's
# msgpack serialization of pending node writes. A node returns this for
# `agent_outputs` to clear it instead of merging — without this, stale
# results from a prior turn in the same checkpointed session would leak
# into a follow-up query's synthesis.
RESET = "__RESET_AGENT_OUTPUTS__"


def merge_dicts(left: dict | None, right):
    if right == RESET:
        return {}
    return {**(left or {}), **right}


class GraphState(TypedDict, total=False):
    query: str
    redacted_query: str
    phi_redactions: list[dict]
    injection_flags: list[str]
    is_refused: bool
    is_conversational: bool
    routing: dict[str, str]

    # Per-branch fields, set only on the state passed to a single
    # `run_specialist` invocation via Send — not part of the top-level input.
    agent_name: str
    sub_query: str

    agent_outputs: Annotated[dict[str, str], merge_dicts]
    confidence: str
    confidence_reason: str
    retry_count: int
    needs_review: bool
    final_answer: str
    # Only set when needs_review is True — the synthesized draft before
    # human_review_gate swaps final_answer for the review placeholder.
    drafted_answer: str

    # Accumulates "Q: ...\nA: ..." per turn so a checkpointed session can
    # reference prior turns when a follow-up query comes in.
    history: Annotated[list[str], operator.add]

    # Replayable per-node audit trail (expanded further in Phase 3).
    audit: Annotated[list[dict], operator.add]
