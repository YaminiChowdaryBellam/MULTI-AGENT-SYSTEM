# Multi-Agent Clinical Research Assistant — Project Notes

## Overview

A multi-agent system that answers clinical research queries by dispatching to 5 healthcare specialist agents (PubMed, openFDA, ClinicalTrials.gov, an internal RAG knowledge base, and live web via Tavily), each backed by a real API, and synthesizing results through a central orchestrator. All agents use **Groq** as the LLM provider (fast inference on open-source models like LLaMA).

> **Phase 1 pivot (healthcare):** the system originally used ArXiv, Wikipedia, and Tavily as general-purpose research specialists (see the build log and issues below — that history still applies to the orchestrator's routing/splitting/synthesis mechanics). ArXiv and Wikipedia were removed and replaced with PubMed, openFDA, ClinicalTrials.gov, and a RAG-as-tool agent to pivot the system toward clinical research. Tavily was kept, reframed for "current health news, recalls, new guidelines."

> **Phase 2 (LangGraph migration):** `agents/orchestrator.py`'s hand-rolled routing/dispatch/synthesis loop was ported into a LangGraph `StateGraph` (`graph/`) with real parallel fan-out (`Send`), a confidence-gated reflection retry, and checkpointed sessions for follow-ups. `main.py` runs the graph by default; `python main.py --legacy` still runs the original orchestrator for the "from scratch → framework" story. `agents/orchestrator.py` itself is still imported by `graph/nodes.py` for its `AGENT_REGISTRY`.

> **Phase 3 (guardrails):** an `input_guard` node now runs before everything else — PHI/PII redaction (Presidio), prompt-injection heuristics, and an LLM out-of-scope check that refuses personal-treatment questions. An `output_guard` node runs after synthesis — strips uncited claims and appends a "not medical advice" disclaimer. Every request's full state (raw query, redactions, routing, agent outputs, retries, final answer) is appended to `logs/audit.jsonl`, which is gitignored since it retains the raw pre-redaction query.

> **Phase 4 (evals):** a `human_review_gate` node sits between `synthesize` and `output_guard` — if confidence is still "low" after the reflection retry is exhausted, the drafted answer is withheld and queued to `logs/review_queue.jsonl` (also gitignored) instead of being returned to the user. `python -m evals` (package: `evals/`) runs a ~50-query gold-set routing accuracy eval (calls `router_node` directly, one Groq call each) plus an LLM-as-judge faithfulness + deterministic citation-coverage pass on the ~25 entries flagged `e2e=True` (full graph runs, real specialist/API calls). Results are written to `evals/results/*.json` (committed, unlike `logs/` — that's the point of cross-version tracking) and compared against the previous run to flag regressions.
>
> Running the eval harness for real surfaced (and fixed) several bugs Phase 1-3's lighter interactive testing never hit: the specialist agents (`pubmed_agent.py`, `openfda_agent.py`, `clinicaltrials_agent.py`, `tavily_agent.py`) each had their own Groq client with no retry/backoff, unlike `graph/llm.py`'s `call_groq` — fixed by having them import and use the shared retry-protected version. `pubmed_tool.py` never truncated abstract text, so 5 untruncated PubMed abstracts in one prompt could exceed Groq's request-size limit outright (HTTP 413, not a transient rate limit — retrying wouldn't have helped) — fixed by capping it like the sibling tools. Most significantly, the very first real eval run measured **4% routing accuracy** (94% recall but 40% precision — the router was selecting nearly all 5 agents for almost every query); tightening `AGENT_DESCRIPTIONS`/adding explicit `ROUTING_RULES` to the router prompt brought it to **70% accuracy / 84% F1**. That in turn required distinguishing "the model failed to return valid JSON" from "the model validly said `{}`" in `extract_json` (both used to collapse to the same fallback-to-all-agents behavior) and teaching `fan_out_to_specialists` to route a deliberately empty selection straight to `confidence_gate` instead of dispatching zero parallel branches.
>
> Digging into the remaining 15 routing misses afterward found two gold-set bugs rather than router bugs: `R009` ("pathophysiology of migraine") was labeled `pubmed_only` while the router prompt's own canonical `rag` example was "pathophysiology of atrial fibrillation" — nearly identical phrasing, contradicting itself. `R016` ("recalls for ranitidine") expected `openfda`, but `openfda_tool.py` queries `/drug/label.json`, which has no recall data (that's a different FDA endpoint we never call) — the router's choice of `tavily` was actually more correct than the label. Both gold-set entries were relabeled to match reality, which would put accuracy at 74% on a clean re-run.

> **Phase 5 (LLMOps & observability):** `graph/llm.py`'s `call_groq` and every graph node are wrapped with Langfuse `@observe` spans — per-node latency, per-agent naming on parallel `run_specialist` branches, routing/retry decisions as span metadata, and token usage + an approximate cost estimate on every Groq generation. It's a true no-op without `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` set (a free cloud.langfuse.com account works) — `input_guard_node` and `run_graph` explicitly disable auto-capture since the raw, pre-redaction query only exists at that point, and would otherwise ship straight to Langfuse's cloud. A FastAPI service (`api.py`: `POST /query`, `GET /health`) wraps `run_graph`, mirroring RAG-medical-system's `src/api.py` pattern (slowapi rate limiting, global exception handler) — with one deliberate deviation: secrets are passed via docker-compose's `env_file` at runtime rather than `COPY .env` baked into an image layer. GitHub Actions CI (`.github/workflows/ci.yml`) runs lint (`ruff`) + unit tests + the eval harness on every PR and fails the job if routing accuracy or faithfulness regresses against the last baseline (`evals/results.py`'s `compare()`); a second workflow (`update-eval-baseline.yml`) re-runs evals after every merge to main and commits the new baseline — `GITHUB_TOKEN`-authored pushes don't retrigger `push` workflows, so this can't loop. Step 5.4's A/B experiment (`evals/ab_experiment.py`) compares the LLM router against a from-scratch embedding-similarity router (`graph/embedding_router.py`, local `sentence-transformers` model, zero API cost) on accuracy/latency/cost plus a simple total-variation-distance drift check between the two routers' agent-selection distributions.



---

## Architecture: LangGraph Core + Specialists (Hub-and-Spoke)

```
User Query
    │
    ▼
input_guard  ── PHI redaction, prompt-injection heuristics, out-of-scope check ──► refused? END
    │
    ▼
classify  ── conversational? ──► direct reply, END
    │
    ▼
router  ── LLM picks agents + writes sub-queries ──┐
    │                                               │ (Send fan-out, parallel)
    ▼                                               │
run_specialist ×N  ◄───────────────────────────────┘
    │
    ├──► PubMed Agent         ──► pubmed_tool.py         ──► NCBI E-utilities (PubMed)
    ├──► openFDA Agent        ──► openfda_tool.py         ──► openFDA drug label API
    ├──► ClinicalTrials Agent ──► clinicaltrials_tool.py  ──► ClinicalTrials.gov v2 API
    ├──► RAG Agent            ──► rag_tool.py             ──► RAG-medical-system (POST /ask)
    └──► Tavily Agent         ──► tavily_tool.py          ──► Tavily Web Search API
    │
    ▼
confidence_gate  ── weak/conflicting evidence? ──► reflect (rewrite + re-dispatch, max 1 retry)
    │
    ▼
synthesize (citation-enforced)
    │
    ▼
human_review_gate  ── still low confidence after retry? ──► queue to logs/review_queue.jsonl, withhold draft
    │
    ▼
output_guard (strip uncited claims, add disclaimer)  ──►  END
```

Checkpointed with an in-memory saver keyed by `thread_id`, so a session can ask follow-ups.

**Why this architecture?**
- `input_guard` runs first so nothing downstream — not even the classify/router LLM calls — ever sees raw PHI
- Fan-out via `Send` runs specialists in parallel instead of the old orchestrator's sequential for-loop (~2x faster on the Phase 1 demo query)
- Each specialist agent has one job and one tool — easy to debug and extend
- The confidence gate + reflect loop gives the graph one shot at recovering from a weak first pass before answering anyway
- `human_review_gate` means a still-low-confidence answer is never silently handed to the user — it's queued for a human instead
- The RAG agent degrades gracefully if the RAG-medical-system service isn't running

---

## Tech Stack

| Component | Technology | Reason |
|---|---|---|
| LLM | Groq (LLaMA / Mixtral) | Fast, free-tier open-source inference |
| Agentic core | LangGraph (`StateGraph`, `Send`, `InMemorySaver`) | Parallel fan-out, conditional reflection loop, checkpointed sessions |
| PHI/PII redaction | Presidio (`presidio-analyzer`/`-anonymizer`), spaCy `en_core_web_sm` | Detects and redacts names, phone numbers, dates, etc. before anything downstream sees them |
| Medical literature | PubMed via NCBI E-utilities (`requests`) | Peer-reviewed clinical papers, no key needed |
| Drug data | openFDA drug label API (`requests`) | Indications, warnings, interactions, recalls, no key needed |
| Clinical trials | ClinicalTrials.gov v2 API (`requests`) | Active/recruiting trials by condition, no key needed |
| Internal knowledge base | RAG-medical-system (`requests` HTTP client) | Cited answers from our own StatPearls-derived RAG service |
| Web search | Tavily (`tavily-python`) | Current health news, recalls, new guidelines |
| Config | `python-dotenv` | Loads API keys from `.env` securely |
| Tracing/observability | Langfuse (`@observe`, OTel-based) | Per-node latency, token usage/cost, routing/retry metadata; no-ops without API keys |
| HTTP API | FastAPI + `uvicorn` + `slowapi` | `POST /query`/`GET /health`, rate-limited, mirrors RAG-medical-system's API pattern |
| CI | GitHub Actions + `ruff` | Lint + tests + eval harness on every PR; blocks merge on routing/faithfulness regression |
| Embedding baseline | `sentence-transformers` (`all-MiniLM-L6-v2`) + `scikit-learn` | Local, zero-cost router baseline for the Step 5.4 A/B comparison |

---

## Project Structure

```
Multi-Agent-System/
├── graph/                      # Phase 2/3/4/5: LangGraph agentic core (the default engine)
│   ├── state.py                 # GraphState schema + reducers (merge_dicts, RESET sentinel)
│   ├── llm.py                   # Shared Groq client + call_groq() — 429 retry/backoff, Langfuse generation span, LAST_USAGE
│   ├── guardrails.py             # PHI redaction, injection heuristics, out-of-scope check, citation enforcement/coverage
│   ├── audit.py                  # Timestamped audit entries + JSONL log read/write
│   ├── review_queue.py           # Human-in-the-loop review queue read/write
│   ├── nodes.py                  # input_guard, classify, router, run_specialist, confidence_gate, reflect, synthesize, human_review_gate, output_guard — all Langfuse-traced
│   ├── embedding_router.py       # Phase 5.4: local sentence-transformers router (A/B baseline, not wired into the production graph)
│   └── build.py                  # Builds/compiles the StateGraph, run_graph() entry point (Langfuse-traced root span)
├── agents/
│   ├── orchestrator.py         # Legacy hand-rolled orchestrator (--legacy flag); AGENT_REGISTRY reused by graph/nodes.py
│   ├── pubmed_agent.py         # Specialist: peer-reviewed literature
│   ├── openfda_agent.py        # Specialist: FDA drug labels
│   ├── clinicaltrials_agent.py # Specialist: active clinical trials
│   ├── rag_agent.py            # Specialist: internal RAG knowledge base
│   └── tavily_agent.py         # Specialist: live web search
├── tools/
│   ├── pubmed_tool.py          # Raw NCBI E-utilities calls (esearch -> efetch)
│   ├── openfda_tool.py         # Raw openFDA drug label API call
│   ├── clinicaltrials_tool.py  # Raw ClinicalTrials.gov v2 API call
│   ├── rag_tool.py             # HTTP client for RAG-medical-system's POST /ask
│   └── tavily_tool.py          # Raw Tavily API call
├── evals/                      # Phase 4/5: evaluation harness — run with `python -m evals`
│   ├── gold_set.py               # ~50 routing-labeled queries, ~25 flagged e2e=True for the judge pass
│   ├── routing_eval.py           # Step 4.1 — routing accuracy (calls router_node directly)
│   ├── judge_eval.py             # Step 4.2 — LLM-as-judge faithfulness + citation coverage (full graph, e2e)
│   ├── results.py                # Step 4.4 — writes evals/results/*.json, compares vs previous run
│   ├── ab_experiment.py          # Step 5.4 — LLM router vs embedding router: accuracy/latency/cost + distribution drift
│   └── __main__.py               # CLI entry point (exits non-zero on regression — what CI gates on)
├── logs/
│   ├── audit.jsonl             # Per-request audit trail (gitignored — contains raw pre-redaction query text)
│   └── review_queue.jsonl      # Human-in-the-loop queue for still-low-confidence answers (gitignored)
├── .github/workflows/
│   ├── ci.yml                    # Every PR: lint + tests + evals --no-write (compares against committed baseline)
│   └── update-eval-baseline.yml  # Every push to main: re-runs evals and commits the new evals/results/ baseline
├── api.py                      # FastAPI service — POST /query, GET /health
├── Dockerfile                   # Serves api.py; installs the spaCy model at build time
├── docker-compose.yml
├── main.py                     # CLI entry point — runs the graph by default, --legacy for the old orchestrator
├── requirements.txt
├── .env                        # API keys (never commit this)
└── .gitignore
```

---

## Step-by-Step Build Log

### Step 1 — Project Setup

**What we did:**
- Created the folder structure (`agents/`, `tools/`)
- Wrote `requirements.txt` with all dependencies
- Created `.env` with placeholders for `GROQ_API_KEY` and `TAVILY_API_KEY`
- Added `.gitignore` to prevent committing `.env` and cache files

**Key decision:** Use Groq instead of OpenAI/Anthropic because it offers free-tier fast inference on open-source models (LLaMA 3, Mixtral), which is great for building and demoing without cost concerns.

**Python environment:** Using a dedicated conda environment (`multi-agent`, Python 3.13) to isolate project dependencies from the system Python (`/usr/bin/python3`, 3.9.6) and conda base.

```bash
conda create -n multi-agent python=3.13 -y
conda activate multi-agent
pip install -r requirements.txt
```

---

### Step 2 — Tool Wrappers

**What we did:**
- Wrote 3 pure Python functions — one per external API
- No agent logic here, just clean API calls that return structured dicts

**`tools/arxiv_tool.py` — `search_arxiv(query, max_results=5)`**
- Uses the `arxiv` Python package to search the ArXiv preprint server
- Returns: list of papers with `title`, `authors`, `summary` (truncated to 500 chars), `url`, `published` date
- Sorted by relevance
- No API key needed

**`tools/wikipedia_tool.py` — `search_wikipedia(query, sentences=5)`**
- Uses `wikipedia-api` to fetch a Wikipedia page by title/query
- Returns: `title`, `summary` (first N sentences), `url`
- Returns `found: False` gracefully if the page doesn't exist
- No API key needed

**`tools/tavily_tool.py` — `search_web(query, max_results=5)`**
- Uses `TavilyClient` from `tavily-python`
- Reads `TAVILY_API_KEY` from environment
- Returns: list of web results with `title`, `url`, `content` (truncated to 500 chars)
- Best for current events or queries not covered by ArXiv/Wikipedia

**Why separate tools from agents?**
- Tools are stateless functions — easy to unit test independently
- Agents add the LLM reasoning layer on top; keeping them separate avoids coupling
- If an API changes, you only update the tool file, not the agent

---

### Step 3 — Specialist Agents

**What we did:**
- Wrote 3 agent files — one per data source
- Each agent follows the same pattern:
  1. Call its tool function to fetch raw API data
  2. Format the raw data into a prompt
  3. Send the prompt to Groq LLM
  4. Return a clean 3–5 sentence natural-language summary

**`agents/arxiv_agent.py` — `run_arxiv_agent(query) -> str`**
- Calls `search_arxiv(query)` → gets list of papers
- Prompts Groq: "Summarize key findings and themes across these papers"
- Returns a paragraph highlighting research trends and specific paper titles

**`agents/wikipedia_agent.py` — `run_wikipedia_agent(query) -> str`**
- Calls `search_wikipedia(query)` → gets article title, summary, URL
- Prompts Groq: "Provide a clear and concise explanation based on this"
- Handles gracefully if no article is found (`found: False`)

**`agents/tavily_agent.py` — `run_tavily_agent(query) -> str`**
- Calls `search_web(query)` → gets top web results with snippets
- Prompts Groq: "Summarize the most relevant and up-to-date information"
- Best for current events, recent news, or anything not in academic/encyclopedia sources

**Groq setup pattern (shared across all agents):**
```python
from groq import Groq
_client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama3-8b-8192"

response = _client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
)
return response.choices[0].message.content
```

**Why does each agent call Groq separately?**
- Each agent interprets domain-specific data differently (papers vs articles vs web results)
- The orchestrator receives clean, domain-aware summaries — not raw JSON noise
- Easier to swap or tune prompts per agent independently

---

### Step 4 — Orchestrator Agent

**What we did:**
- Wrote `agents/orchestrator.py` with a single public function: `run_orchestrator(query) -> str`
- The orchestrator has 3 internal phases:

**Phase 1 — Routing (`_decide_agents`)**
- Sends the user query to Groq with descriptions of all 3 agents
- Groq returns a JSON list of agent names to use, e.g. `["arxiv", "wikipedia"]`
- Parses the JSON and filters to only valid agent names
- Falls back to all 3 agents if parsing fails

```python
# Example routing decision by Groq:
# "What is a transformer in ML?" → ["arxiv", "wikipedia"]
# "Latest AI news today?"        → ["tavily"]
# "Explain quantum computing"    → ["arxiv", "wikipedia", "tavily"]
```

**Phase 2 — Dispatch**
- Loops over the selected agent names
- Looks up the function in `AGENT_REGISTRY` (a dict mapping name → function)
- Calls each agent with the same query and stores its summary

```python
AGENT_REGISTRY = {
    "arxiv": run_arxiv_agent,
    "wikipedia": run_wikipedia_agent,
    "tavily": run_tavily_agent,
}
```

**Phase 3 — Synthesis (`_synthesize`)**
- Combines all agent summaries into one prompt
- Sends to Groq: "Synthesize all of this into one well-structured answer"
- Returns the final merged answer to the user

**Why use Groq twice in the orchestrator?**
- First call = routing decision (cheap, short output)
- Second call = synthesis (the most important call — produces the final answer)
- Separating them keeps each prompt focused and easier to debug

**`print` statements in orchestrator:**
- `[Orchestrator] Received query` — confirms entry
- `[Orchestrator] Routing to agents` — shows which agents were selected
- `[Orchestrator] Running X agent` — tracks dispatch progress
- `[Orchestrator] Synthesizing` — marks the final phase

---

### Step 5 — Main Entry Point

**What we did:**
- Wrote `main.py` as the CLI entry point — the only file the user needs to run

**How it works:**
1. `load_dotenv()` runs first — reads `GROQ_API_KEY` and `TAVILY_API_KEY` from `.env` into `os.environ` before any agent code is imported
2. Enters an interactive loop — accepts one query at a time
3. Passes the query to `run_orchestrator(query)` and prints the final answer
4. Typing `exit` quits cleanly

**Why `load_dotenv()` must come before imports?**
- The agent files read `os.environ["GROQ_API_KEY"]` at module load time (when `_client = Groq(...)` runs)
- If `load_dotenv()` runs after the import, the keys won't be in the environment yet and the app will crash

**To run the system:**
```bash
source venv/bin/activate
python main.py
```

---

## Full Data Flow (End to End)

```
User types a question
        │
        ▼
main.py → run_orchestrator(query)
        │
        ├─ _route_and_split(query)
        │       └─ Groq LLM: selects agents + writes sub-queries per agent
        │           → {"arxiv": "weather forecasting", "tavily": "weather today"}
        │
        ├─ run_arxiv_agent("weather forecasting")
        │       ├─ search_arxiv(...)       → raw paper list from ArXiv API
        │       └─ Groq LLM: summarize     → clean paragraph
        │
        ├─ run_tavily_agent("weather today")
        │       ├─ search_web(...)         → raw results from Tavily API
        │       └─ Groq LLM: summarize     → clean paragraph
        │
        └─ _synthesize(query, outputs)
                └─ Groq LLM: merge both   → final answer printed to user
```

---

## Issues Encountered & Resolutions

---

### Issue 1 — Decommissioned Groq Model

**When it happened:** First time running `python main.py` with query "Explain LLM in detail?"

**What went wrong:**
```
groq.BadRequestError: Error code: 400 — The model `llama3-8b-8192` has been
decommissioned and is no longer supported.
```
The model `llama3-8b-8192` was hardcoded in all 4 agent files. Groq had retired it after we wrote the code.

**Root cause:** We used a model ID that was valid at the time of writing but had since been removed from Groq's platform.

**Fix:** Replaced `llama3-8b-8192` with `llama-3.1-8b-instant` in all 4 files:
- `agents/arxiv_agent.py`
- `agents/wikipedia_agent.py`
- `agents/tavily_agent.py`
- `agents/orchestrator.py`

**Lesson:** Always check the provider's current model list before hardcoding a model ID. Prefer reading it from `.env` or a config so it's easy to update without touching code.

---

### Issue 2 — Missing `python-dotenv` Module

**When it happened:** After switching to a fresh `venv` environment.

**What went wrong:**
```
ModuleNotFoundError: No module named 'dotenv'
```

**Root cause:** `pip install -r requirements.txt` had not been run in the new `venv`. The packages were installed in the conda environment (`multi-agent`) but not in `venv`.

**Fix:**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

**Lesson:** Every new virtual environment starts empty. Always install dependencies after creating or switching environments.

---

### Issue 3 — ArXiv HTTP 429 (Rate Limit) Crash

**When it happened:** Query: "What is current weather and state two top research papers on weather?"

**What went wrong:**
```
arxiv.HTTPError: Page request resulted in HTTP 429
```
The app crashed completely instead of degrading gracefully.

**Root causes (two combined):**
1. The orchestrator routed to `arxiv` **twice** — `['tavily', 'arxiv', 'arxiv']` — causing two rapid API calls, which triggered ArXiv's rate limit.
2. There was no error handling around the ArXiv API call.

**Fix 1 — Deduplicate agent list in orchestrator:**
```python
seen = set()
return [a for a in agents if a in AGENT_REGISTRY and (a not in seen) and not seen.add(a)]
```

**Fix 2 — Catch `HTTPError` in `arxiv_tool.py`:**
```python
try:
    for paper in client.results(search):
        ...
except arxiv.HTTPError as e:
    print(f"[arxiv_tool] Rate limited by ArXiv (HTTP {e.status}). Returning empty results.")
return results  # returns [] instead of crashing
```

**Lesson:** Always handle rate-limit errors from free public APIs gracefully. Return empty results and let the agent decide how to respond rather than crashing the whole system.

---

### Issue 4 — Orchestrator Sending Full Question to Every Agent (No Splitting)

**When it happened:** Query: "What is the weather today and give me 3 research papers on weather?"

**What went wrong:**
```
[Orchestrator] Routing plan: {
  'arxiv':     'What is the weather today and give me 3 research papers on weather',
  'wikipedia': 'What is the weather today and give me 3 research papers on weather',
  'tavily':    'What is the weather today and give me 3 research papers on weather'
}
```
All 3 agents received the full conversational question instead of focused sub-queries. ArXiv doesn't understand natural language questions — it expects short keyword search terms.

**Root cause:** The original `_decide_agents()` only returned a list of agent names. Every agent then received the raw full query. There was no mechanism to tailor the query per agent.

**Fix — Replaced `_decide_agents()` with `_route_and_split()`:**

New function asks Groq in one call to both select agents AND write a focused sub-query for each:

```python
# Prompt instructs Groq to return:
{
  "arxiv":  "weather forecasting deep learning",   ← keywords only
  "tavily": "current weather conditions today"     ← natural language for web search
}
```

Each agent now receives only the context relevant to what it can search.

**Sub-issue — JSON parsing failed silently:**
Groq wrapped its JSON response in markdown code fences (` ```json ... ``` `), causing `json.loads()` to fail silently and fall back to sending the full query to all agents. This is why the fix appeared to not work at first.

**Fix — Strip markdown before parsing:**
```python
import re
match = re.search(r"\{.*\}", raw, re.DOTALL)
cleaned = match.group(0) if match else raw
routing = json.loads(cleaned)
```

**Lesson:** LLMs often wrap JSON output in markdown. Never call `json.loads()` directly on raw LLM output — always extract the JSON object/array first using a regex or string search.

---

## Full Data Flow (After All Fixes)

```
User query: "What is the weather today and give me 3 research papers on weather?"
        │
        ▼
_route_and_split()  →  Groq decides + splits:
        │               {
        │                 "arxiv":  "weather forecasting research",
        │                 "tavily": "current weather today"
        │               }
        ├─ run_arxiv_agent("weather forecasting research")
        │       ├─ search_arxiv(...)   → paper list (or [] if rate-limited)
        │       └─ Groq: summarize     → "Here are key papers on weather forecasting..."
        │
        ├─ run_tavily_agent("current weather today")
        │       ├─ search_web(...)     → live web results
        │       └─ Groq: summarize     → "Today's weather shows..."
        │
        └─ _synthesize()
                └─ Groq merges both   → Final answer to user
```

---

## Key Concepts to Know for Interview

**What is a multi-agent system?**
Multiple AI agents, each with a specific role, collaborating to solve a task that is too complex for a single agent. Agents can be specialized, run in parallel, and each can have access to different tools.

**What is an orchestrator?**
The "brain" of the system — it receives the user query, decides which specialist agents to invoke, coordinates their execution, and synthesizes the final answer.

**What is a tool in an agent context?**
A callable function that gives an agent access to an external system (API, database, search engine). The agent decides when to use it; the tool just executes the action.

**Why Groq?**
Groq runs open-source LLMs (LLaMA, Mixtral) with very low latency using custom hardware (LPU). It has a free tier, making it ideal for development and demos.

**What is hub-and-spoke vs sequential pipeline?**
- Hub-and-spoke: Central orchestrator routes to specialists — more flexible, easier to scale
- Sequential pipeline: Agents run in order, each enriching the context — simpler but no parallelism

---

## APIs & Keys

| Service | Key name | Where to get it |
|---|---|---|
| Groq | `GROQ_API_KEY` | console.groq.com |
| Tavily | `TAVILY_API_KEY` | app.tavily.com |
| PubMed (NCBI E-utilities) | *(none needed; optional `NCBI_EMAIL`)* | Free public API |
| openFDA | *(none needed)* | Free public API |
| ClinicalTrials.gov | *(none needed)* | Free public API |
| RAG-medical-system | *(none needed; optional `RAG_SERVICE_URL`, defaults to `http://localhost:8000`)* | Runs locally — see that project's repo |
