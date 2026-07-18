# Project Outline: Multi-Agent Clinical Research Assistant

**One-liner:** A LangGraph multi-agent system that answers clinical questions by orchestrating five healthcare specialists — PubMed, openFDA, ClinicalTrials.gov, your own RAG service, and live web — with PHI guardrails, LLM-as-judge evals in CI, full observability, and quantized-model routing on Azure.

---

## Phase 1 — Specialist Swap (healthcare data layer)

**Step 1.1 — Remove** `arxiv_tool/agent` and `wikipedia_tool/agent` (keep in git history).

**Step 1.2 — PubMed tool + agent.** NCBI E-utilities (esearch → efetch), adapted from RAG project's `fetch_statpearls.py`. Returns title, PMID, abstract, pub date. Keyless. Reuse the 429-handling pattern from the ArXiv fix.

**Step 1.3 — openFDA tool + agent.** Drug labels endpoint (`api.fda.gov/drug/label.json`): indications, warnings, interactions, recalls. Keyless.

**Step 1.4 — ClinicalTrials.gov tool + agent.** v2 API: condition → active trials, phases, locations. Keyless.

**Step 1.5 — RAG-as-tool agent.** HTTP client calling RAG-medical-system's `POST /ask`; returns its cited answer + sources. Graceful fallback if the service is down.

**Step 1.6 — Keep Tavily**, reframe routing description to "current health news, recalls, new guidelines."

**Milestone:** old orchestrator works with 5 healthcare specialists. Demo query: *"What are treatment options for atrial fibrillation, any interactions between warfarin and amiodarone, and active trials?"*

---

## Phase 2 — LangGraph Migration (agentic core)

**Step 2.1 — Define graph state:** query, redacted query, routing plan, agent outputs, confidence, final answer, audit metadata.

**Step 2.2 — Nodes:** router (LLM picks agents + writes sub-queries — port `_route_and_split`) → parallel specialist nodes (fan-out/fan-in) → confidence gate → synthesis (citation-enforced).

**Step 2.3 — Reflection loop:** if confidence gate finds weak/conflicting evidence, rewrite sub-queries and re-dispatch (max 1 retry) before synthesizing.

**Step 2.4 — Checkpointing:** LangGraph memory so a session can ask follow-ups.

**Milestone:** graph replaces hand-rolled orchestrator; parallel execution measurably faster. Keep the old orchestrator file for the "from scratch → framework" interview story.

---

## Phase 3 — Guardrails & Governance

**Step 3.1 — Input guard node** (runs before router): Presidio PHI/PII detection → redact; prompt-injection heuristics; out-of-scope check → refusal path ("not medical advice, consult a clinician") for personal-treatment questions.

**Step 3.2 — Output guard node** (after synthesis): every claim must carry a citation; uncited claims flagged or stripped; disclaimer appended.

**Step 3.3 — Audit trail:** JSONL log per request — raw query, redactions, routing plan, each agent's output, retries, final answer, timestamps. Replayable.

**Milestone:** demo showing a PHI-laden query getting redacted and an off-scope query getting refused, both visible in the audit log.

---

## Phase 4 — Evaluation Harness

**Step 4.1 — Gold set:** ~50 queries labeled with expected agent selection → routing accuracy metric.

**Step 4.2 — LLM-as-judge:** score synthesis faithfulness (answer supported by agent outputs?) and citation coverage on ~25 end-to-end cases.

**Step 4.3 — Human-in-the-loop:** answers below confidence threshold → flagged for review queue instead of returned.

**Step 4.4 — Regression tracking:** eval runs write scores to a results file; compare across versions.

**Milestone:** `python -m evals` prints routing accuracy + faithfulness scores; expand `tests/` with unit tests per tool (mocked APIs).

---

## Phase 5 — LLMOps & Observability

**Step 5.1 — Langfuse tracing:** per-agent latency, token cost, routing decisions, retries on a dashboard.

**Step 5.2 — FastAPI service** (`POST /query`, `GET /health`) + Dockerfile (mirror RAG project's compose pattern).

**Step 5.3 — GitHub Actions CI:** lint + unit tests + eval harness on every PR; block merge if routing accuracy or faithfulness regresses.

**Step 5.4 — A/B experiment:** LLM router vs embedding-similarity router; log accuracy, latency, cost per variant. Simple drift check on routing distribution.

**Milestone:** a PR that degrades routing fails CI — screenshot it for the README.

---

## Phase 6 — Optimization & Cloud

**Step 6.1 — Tiered models:** quantized local model (Llama-3.2-3B GGUF via Ollama) for routing + guard checks; Groq hosted model only for synthesis. Benchmark cost/query and latency before vs after.

**Step 6.2 — Azure deploy:** Container Apps (free tier), secrets in env config, public demo endpoint.

**Step 6.3 — README rewrite:** problem → architecture diagram → implementation → measured impact (routing accuracy, faithfulness, latency, cost reduction from tiering).

**Milestone:** live Azure URL + metrics table in README.

---

## Order & Effort

| Phase | Output | Est. effort |
|---|---|---|
| 1. Healthcare specialists | Working domain pivot | 1–2 days |
| 2. LangGraph | Agentic core | 2–3 days |
| 3. Guardrails | Governance layer | 1–2 days |
| 4. Evals | Measurable quality | 2 days |
| 5. LLMOps | CI + observability | 2 days |
| 6. Optimization + Azure | Deployed + benchmarked | 1–2 days |

Each phase ends in a working, demoable state — safe to stop and interview at any milestone.
