# Enervera Pulmonology Assistant — Handoff & Production Roadmap

This document is the single source of truth for **what the system is, how to run
it, what is production-ready, and exactly what to change in the next iterations.**
Read §0 first, then §9 for the prioritized action list.

> Scope: the **GraphRAG runtime** (query → answer) and its memory layers. The
> offline **chunking** pipeline (PDF → chunks) is documented separately in the
> root `README.md`.

---

## 0. TL;DR — current status

**What works (verified by offline tests):**
- Full hybrid pipeline wired and behaviorally correct: gatekeeper → routing →
  vector retrieval + rerank → entity extraction → **graph traversal** → answer → memory update.
- Pulmonology scope gate (≥75% relevance), domain-driven retarget layer,
  clinical triage upgrades (differential discipline, symptom-weighted risk,
  red-flag emergency escalation with a reasoned response), and the episodic
  memory layer are all connected.

**What is NOT yet done / required before you can ship:** see the **P0 checklist**
below and the full roadmap in §9.

### P0 checklist (do these before any real deployment)
1. **Populate the data stores** (§5). Without them, retrieval + graph return
   empty and answers fall back to model-only knowledge:
   - Pinecone main index, namespace **`pulmonology_v1`**.
   - Neo4j graph (`:Entity`/`:Chunk` nodes + relations).
   - Episodic Pinecone index (`episodicmemory`) — only if using `--user-id`.
2. **Run a real end-to-end smoke test** on a machine with network access
   (this was never executed live — the dev sandbox blocked TLS). Confirm the
   `🔗 Graph entities (...)` log line and a streamed structured answer.
3. **Decide the delivery surface.** Today it is **CLI-only** (`run_graphrag.py`).
   To serve it you need the HTTP API in §9-P1.

---

## 1. What this system is

A **pulmonology-specialised medical assistant** built on Hybrid GraphRAG:

- **Vector retrieval** (Pinecone) over a chunked clinical corpus.
- **Knowledge-graph traversal** (Neo4j) over entities + clinical relations.
- **Session memory** (Redis / RAM fallback) for multi-turn triage continuity.
- **Episodic memory** (Pinecone) for long-term, per-user recall.
- **A domain layer** that makes the whole thing retargetable to another
  specialty by editing two folders (§7).

It is gated to respiratory medicine, applies triage-grade safety logic
(emergency escalation, symptom weighting), and answers in a calm, structured,
patient-facing style.

---

## 2. Architecture & request flow

`graphrag/pipeline/graphrag_pipeline.py :: GraphRAGPipeline.run()` orchestrates:

```
user query
 │
 ├─ STAGE -2  Session memory load           (Memory_Layer via SessionMemoryAdapter)
 ├─ Deterministic red-flag detection        (clinical_policy.detect_red_flags)  → sets emergency
 ├─ STAGE -1  Gatekeeper / analyzer (LLM)    (query_understanding.analyzer)
 │              → intent, risk, pulmonology_relevance, medical_entities, follow-ups
 │              → refuse (non-medical) | scope-gate (<75% pulmonology) | emergency
 ├─ STAGE  0  Routing                        (query_understanding.routing/query_config)
 │              → NO_RETRIEVAL | MEMORY_FIRST | HYBRID_RAG ; emergencies forced HYBRID + critical
 ├─ STAGE  1  Vector retrieval + rerank      (retrievers.pinecone_retriever, namespace=pulmonology_v1)
 ├─ STAGE  2  Entity extraction              (processors.entity_processor)
 ├─ STAGE  3  Graph traversal                (retrievers.neo4j_retriever)
 │              entities = chunk + query + memory  (hybrid, normalized, deduped)
 ├─ STAGE 3.5 Episodic retrieval (opt)       (episodic.* — only with --user-id)
 ├─ STAGE  4  Answer generation (LLM stream) (llm.gemini_llm + domain.answer_prompt)
 │              risk_level drives urgency; critical → 5-part emergency structure
 ├─ STAGE  5  Episodic ingest (opt)          (episodic.*)
 └─ memory update (state extract → summarize → save)
```

**Key design rule:** all specialty/clinical knowledge lives in **domain
packages**; the orchestration + retrievers are domain-agnostic.

---

## 3. Repository layout

```
run_graphrag.py            # ⭐ CLI entrypoint for the assistant (REPL + --query)
run_pulmonology.py         # chunking entrypoint (PDF → chunks)
ingest_pinecone.py         # chunks → Pinecone (use --namespace pulmonology_v1)
ingest_neo4j.py            # chunks → Neo4j graph
test_pipeline.py           # chunking smoke tests (NOT graphrag — see §11)

graphrag/
├── domain/                # ⭐ EDIT-FOR-SPECIALTY (retrieval/answer side)
│   ├── prompts.py         #   gatekeeper system prompt (incl. red flags, relevance rubric)
│   ├── answer_prompt.py   #   answer system prompt + SPECIALTY knob + risk layers
│   ├── clinical_policy.py #   red-flag regex, high-signal symptoms, triage policy text
│   ├── query_taxonomy.py  #   query types + per-type retrieval tuning + intent map
│   ├── vocabulary.py      #   PINECONE_NAMESPACE, threshold, graph node label
│   ├── entity_rules.py    #   drug-pair heuristics
│   └── messages.py        #   refusal / emergency / out-of-scope copy
├── query_understanding/   # analyzer (gatekeeper LLM), routing, query_config
├── retrievers/            # pinecone_retriever, neo4j_retriever
├── processors/            # entity_processor (chunk-entity extraction)
├── llm/                   # gemini_client (shared), gemini_llm (answer streaming)
├── memory/                # session_adapter (sync facade over Memory_Layer)
├── pipeline/              # graphrag_pipeline (the orchestrator)
├── config/                # settings (pydantic-settings, reads .env)
└── utils/                 # logger, rate_limit

Memory_Layer/session_memory/
├── domain/                # ⭐ EDIT-FOR-SPECIALTY (memory side)
│   ├── extraction_patterns.py  # symptom/condition/drug/trigger regex
│   ├── risk_rules.py           # CRITICAL_SYMPTOMS, HIGH_SIGNAL_SYMPTOMS, RISK_ORDER
│   └── render_fields.py        # state field → label mapping
├── models.py              # StructuredState, SessionMemory, Message
├── state_extractor.py     # regex extraction + symptom-weighted risk
├── summarizer.py          # rolling summary (deterministic)
├── retriever.py           # WorkingMemory view
├── context_builder.py     # token-budgeted prompt assembly
└── session_manager.py     # Redis + RAM fallback

episodic/                  # long-term per-user memory (Pinecone), trimmed to essentials
├── api/dependencies.py    # build_container() — the embedded entrypoint
├── pipelines/             # context (retrieve) + ingest pipelines
├── services/              # storage, retriever, ranker, extractor, contradiction, clarifier, compression, decay
├── schemas/, prompts/, utils/, config.py
```

---

## 4. Configuration

All config is environment variables read by `graphrag/config/settings.py`. Copy
`.env.example` → `.env` and fill it in. **Required:** `GEMINI_API_KEY`,
`PINECONE_API_KEY`, `NEO4J_PASSWORD` (the pipeline fails fast otherwise).

**Non-env "domain knobs"** (deliberately in code so a specialty swap is one
edit, not an env change):
| Knob | File | Purpose |
|---|---|---|
| `PINECONE_NAMESPACE = "pulmonology_v1"` | `graphrag/domain/vocabulary.py` | retrieval is restricted to this namespace |
| `PULMONOLOGY_RELEVANCE_THRESHOLD = 75` | `graphrag/domain/vocabulary.py` | scope-gate cutoff |
| `GRAPH_NODE_LABEL = "Entity"` | `graphrag/domain/vocabulary.py` | must match ingest_neo4j |
| `SPECIALTY*` | `graphrag/domain/answer_prompt.py` | assistant persona/focus |
| `MAX_FOLLOWUP_QUESTIONS = 3` | `graphrag/domain/clinical_policy.py` | triage question budget |

⚠️ **Config gotcha:** `.env` may contain `EPISODIC_INDEX_NAME`, but the code
reads `PINECONE_EPISODIC_INDEX_NAME` (default `episodicmemory`). They happen to
match today, but renaming `EPISODIC_INDEX_NAME` will have **no effect**. See §9-P2.

---

## 5. Prerequisites & data population

| Store | Needed for | Populate with | Notes |
|---|---|---|---|
| Pinecone main index | vector retrieval | `python ingest_pinecone.py --namespace pulmonology_v1` | MUST use that namespace — retrieval is locked to it |
| Neo4j | graph traversal | `python ingest_neo4j.py --version v1` | local or Aura; entity names are canonical/lowercase → match graph nodes |
| Pinecone episodic index | per-user memory | created/populated by the episodic layer at runtime (or your seed) | only with `--user-id` |
| Redis | shared session memory | run a Redis instance, set `REDIS_URL` | optional — RAM fallback otherwise |

Without the first two, the pipeline still runs but answers are model-only
(no retrieved context, `⏭️ Graph skipped`).

---

## 6. Running it

```powershell
python -m pip install -r requirements.txt   # into your active venv (use `python -m pip`)
Copy-Item .env.example .env                 # then fill in keys

python run_graphrag.py                          # interactive chat (REPL)
python run_graphrag.py --query "cough and breathlessness for a week"   # one-shot
python run_graphrag.py --session-id alice       # named memory session
python run_graphrag.py --user-id alice          # enable episodic memory
python run_graphrag.py --quiet                  # answer only, hide stage logs
```

Health signs in the logs: `namespace: pulmonology_v1` (Stage 1),
`🔗 Graph entities (...)` (Stage 3). Out-of-scope queries return the 🫁 message;
red-flag queries escalate to a structured emergency answer.

---

## 7. Retargeting to a new specialty

> **Full step-by-step playbook: [`docs/RETARGETING.md`](RETARGETING.md).** Summary below.

Edit **only** these, nothing else:
1. `graphrag/domain/` — gatekeeper prompt, query taxonomy/tuning, relevance
   rubric, vocabulary (namespace + threshold + node label), entity rules, messages.
2. `Memory_Layer/session_memory/domain/` — extraction patterns, risk rules, render labels.
3. `graphrag/domain/answer_prompt.py` — the `SPECIALTY` knob + clinical focus block.
4. `graphrag/domain/clinical_policy.py` — red flags + high-signal symptoms for the new specialty.

Then re-ingest data into the new namespace and point `PINECONE_NAMESPACE` at it.
`StructuredState` keeps generic field names (symptoms/drugs/…), so for a
*non-medical* vertical you'd additionally generalize those (larger change).

---

## 8. Is it shippable?

**Shippable as a CLI demo / internal tool:** yes, once data is populated and a
live smoke test passes.

**Shippable as a production service:** not yet — it needs an HTTP API, real
Redis, observability, security, and a live verification pass. See §9.

---

## 9. NEXT ITERATIONS — what to change (prioritized)

Each item: **problem → where → what to do.**

### P0 — blockers for a real deployment
- **[Done in this pass] Accurate `.env.example`** so deployers know every var. ✅
- **Populate data stores** (§5). Until then graph/vector are empty. *Action:* run both ingest scripts; verify Neo4j has Entities and Pinecone `pulmonology_v1` has vectors.
- **Live end-to-end smoke test.** *Where:* `run_graphrag.py`. *Action:* run a real query on a networked machine; confirm retrieval, `🔗 Graph entities`, streamed structured answer, and (with `--user-id`) episodic activation.

### P1 — needed to run as a service
- **HTTP API — [SCAFFOLDED ✅].** `app/main.py` (FastAPI) exposes `GET /health` and
  `POST /chat {message, session_id?, user_id?}` over a single startup-built
  `GraphRAGPipeline`. Enforces `X-API-Key` when `API_KEY` is set; honors
  `CORS_ORIGINS`; injects truststore. Deploy via `render.yaml`. Run locally:
  `uvicorn app.main:app --host 0.0.0.0 --port 8000`.
  - *Async approach used:* `/chat` is a sync `def` so FastAPI threadpools it,
    off the event loop, which is what `SessionMemoryAdapter`'s `asyncio.run()`
    needs. Calls are **serialized with a lock** (the shared pipeline isn't
    thread-safe). *Remaining for scale:* run multiple uvicorn **worker
    processes** (each gets its own pipeline), or do an async-native rewrite of
    the memory path to drop the lock. *Remaining for UX:* add SSE/streaming
    (`pipeline.run()` currently returns the full string; it also prints the
    stream to stdout — fine for logs, not for HTTP streaming).
- **Real Redis for sessions.** *Problem:* without `REDIS_URL` reachable, `session_manager` silently uses a process-local RAM store → sessions lost on restart, not shared across workers. *Action:* provision Redis, set `REDIS_URL`, and add a startup check that logs which backend is active.
- **[Done in this pass] truststore at startup** in `run_graphrag.py` (Windows TLS). ✅ Apply the same to any new API entrypoint.
- **Episodic identity.** *Problem:* episodic memory keys on `--user-id`; production needs a real user identity. *Action:* derive `user_id` from auth and pass it through the API → pipeline.

### P2 — quality, correctness, ops
- **Honor `config.graph_enabled`.** *Problem:* `graphrag_pipeline.run()` gates the graph on `graph_hops` only and ignores `graph_enabled`, so e.g. `diagnosis` (configured `graph_enabled=False`) still traverses. *Where:* Stage 3. *Action:* `if graph_hops > 0 and config.graph_enabled and graph_entities:`.
- **Wire `EPISODIC_INDEX_NAME`.** *Problem:* `.env` var isn't a `Settings` field. *Where:* `graphrag/config/settings.py`. *Action:* add `EPISODIC_INDEX_NAME` (or alias it to `PINECONE_EPISODIC_INDEX_NAME`) so the `.env` value drives it.
- **Entity ↔ graph normalization.** *Problem:* query/memory entities (e.g. `chest_pain`, free-text) only loosely match `:Entity.name`. *Where:* `pipeline._normalize_entity` + ontology. *Action:* add a synonym/alias map (the graph already stores `aliases`) and match against it for better graph recall.
- **Automated tests.** *Problem:* only chunking has tests; graphrag/memory/episodic are covered by ad-hoc scripts. *Action:* add `pytest` suites: scope-gate, red-flag escalation, risk weighting, entity hybrid-merge, prompt-composition, memory continuity. (Stub Pinecone/Neo4j/LLM like the verification scripts used in development.)
- **Observability.** *Action:* structured per-stage timing + token/cost logging; a request id threaded through stages; surface gatekeeper/answer model + latency.
- **Provider resilience.** *Problem:* `call_with_retries` wraps Pinecone; verify Gemini calls also back off on 429/5xx. *Action:* wrap `gemini_client` calls in the same retry policy.
- **Cost/latency of the gatekeeper.** *Problem:* an LLM call every turn. *Action:* short-circuit obvious cases (already done for trivial inputs); consider caching identical analyses; consider a cheaper model for the gatekeeper.
- **Stale settings cleanup.** `EPISODIC_EVAL_*` (eval module removed) and `DATABASE_URL`/`MEMORY_CACHE_TTL_SEC` (the `memory/` longitudinal subsystem isn't in this repo) are unused. *Action:* remove or implement.
- **Packaging.** *Action:* `Dockerfile` + compose (app + Redis + Neo4j), and a CI job running the test suite + `python -m compileall`.
- **Red-flag tuning.** `clinical_policy.RED_FLAG_PATTERNS` is intentionally conservative. *Action:* expand/relax per clinical review; widening increases false-positive ER redirects.

### Compliance / safety (treat as P0/P1 for a real medical product)
- **PII & data retention:** episodic memory stores patient utterances in Pinecone keyed by `user_id`. Define retention, deletion, and consent flows; the layer has a `decay` service — wire a purge/forget path.
- **Logging hygiene:** stage logs print full queries/answers. Scrub or gate PHI in production logs.
- **Medical disclaimer & escalation:** confirmed in prompts, but have it reviewed by a clinician; the emergency backstop and scope gate are guardrails, not a substitute for clinical sign-off.

---

## 10. Operational concerns (quick reference)
- **Cost:** Gemini calls = gatekeeper (every non-trivial turn) + answer (stream) + episodic (extract/contradiction/clarify on ingest). Output tokens dominate.
- **Latency:** gatekeeper → retrieval → rerank → graph → answer are sequential; emergencies now run full retrieval (slightly slower but safer).
- **Failure modes:** Pinecone/Neo4j/Gemini down → retrieval/graph degrade to empty (pipeline continues); episodic is best-effort (warns, never breaks); Redis down → RAM fallback.
- **Scaling:** stateless except Redis/Pinecone/Neo4j connections — horizontally scalable once Redis is shared and the API exists.

---

## 11. Testing & verification
- Chunking: `python test_pipeline.py` (offline) / `--live`.
- GraphRAG/memory/episodic: currently verified via ad-hoc offline scripts that
  stub Pinecone/Neo4j/LLM and assert behavior (scope gate, red-flag escalation,
  risk weighting, hybrid entity merge, prompt content, emergency structure).
  **P2:** promote these into a committed `pytest` suite.
- Quick import/compile sanity: `python -m compileall graphrag Memory_Layer episodic`.

---

## 12. Known limitations / gotchas
- **CLI-only today** (no HTTP API).
- **No live run has been executed** end-to-end (dev sandbox blocked TLS); first
  real run is your verification gate.
- **Graph recall depends on entity-name matching** to `:Entity.name`; chunk-derived
  names match reliably, query/memory-derived ones are best-effort.
- **`graph_enabled` is ignored** (see §9-P2).
- **Gatekeeper relevance is an LLM estimate**, not deterministic; the rubric steers it strongly but edge phrasings can vary (threshold is one lever).
- **Episodic index name** is driven by `PINECONE_EPISODIC_INDEX_NAME`, not the `.env` `EPISODIC_INDEX_NAME` (see §9-P2).
```
