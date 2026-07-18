"""
Builds and compiles the LangGraph agentic core that replaces the hand-rolled
orchestrator (agents/orchestrator.py, kept around for the "from scratch ->
framework" story).

Flow:
    input_guard -> [END (refused) | classify]
    classify -> [END (conversational) | router]
    router -> fan-out (Send) -> run_specialist (parallel) -> confidence_gate
             (or straight to confidence_gate if the router selected zero agents)
    confidence_gate -> [reflect -> fan-out -> run_specialist -> confidence_gate | synthesize]
    synthesize -> human_review_gate -> output_guard -> END

Checkpointed with an in-memory saver keyed by thread_id, so a session can ask
follow-ups and the router/synthesis prompts see recent turn history. Every
request's full state is also appended to logs/audit.jsonl for replay, and a
still-low-confidence draft (after the reflection retry is exhausted) is
appended to logs/review_queue.jsonl instead of being handed to the user.
"""

from langfuse import get_client, observe
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from graph.audit import write_audit_record
from graph.nodes import (
    classify_node,
    confidence_gate_node,
    fan_out_to_specialists,
    human_review_gate_node,
    input_guard_node,
    output_guard_node,
    reflect_node,
    route_after_classify,
    route_after_input_guard,
    router_node,
    run_specialist_node,
    should_retry,
    synthesize_node,
)
from graph.review_queue import write_review_record
from graph.state import GraphState


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("input_guard", input_guard_node)
    graph.add_node("classify", classify_node)
    graph.add_node("router", router_node)
    graph.add_node("run_specialist", run_specialist_node)
    graph.add_node("confidence_gate", confidence_gate_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("human_review_gate", human_review_gate_node)
    graph.add_node("output_guard", output_guard_node)

    graph.add_edge(START, "input_guard")
    graph.add_conditional_edges(
        "input_guard", route_after_input_guard, {"END": END, "classify": "classify"}
    )
    graph.add_conditional_edges("classify", route_after_classify, {"END": END, "router": "router"})
    graph.add_conditional_edges("router", fan_out_to_specialists, ["run_specialist", "confidence_gate"])
    graph.add_edge("run_specialist", "confidence_gate")
    graph.add_conditional_edges(
        "confidence_gate", should_retry, {"reflect": "reflect", "synthesize": "synthesize"}
    )
    graph.add_conditional_edges("reflect", fan_out_to_specialists, ["run_specialist", "confidence_gate"])
    graph.add_edge("synthesize", "human_review_gate")
    graph.add_edge("human_review_gate", "output_guard")
    graph.add_edge("output_guard", END)

    return graph.compile(checkpointer=InMemorySaver())


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


@observe(name="run_graph", capture_input=False, capture_output=False)
def run_graph(query: str, thread_id: str = "default") -> str:
    """
    Main entry point — mirrors agents.orchestrator.run_orchestrator's signature.

    capture_input/output are off here for the same reason as input_guard_node:
    `query` is raw, pre-redaction text. Node-level spans underneath this trace
    (classify, router, ...) run on the already-redacted query and are safe to
    auto-capture as normal.
    """
    get_client().update_current_span(metadata={"thread_id": thread_id})
    app = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"query": query}, config=config)
    if result.get("needs_review"):
        write_review_record(result, thread_id)
    write_audit_record(result, thread_id)
    return result["final_answer"]
