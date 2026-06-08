# Retargeting to a New Specialty — The Playbook

**This is THE document to follow when building the next specialty** (e.g.
cardiology, nephrology). The system is designed so a specialty change touches
only **two domain folders + your data** — never the orchestration, retrievers,
LLM clients, or memory machinery.

> Rule of thumb: if you find yourself editing anything outside
> `graphrag/domain/`, `Memory_Layer/session_memory/domain/`, or the ingestion
> namespace, stop — you're touching domain-agnostic plumbing that should stay put.

---

## The 4 steps

### Step 1 — Edit `graphrag/domain/` (retrieval + answer behavior)

| File | What to change | Example (pulmonology → cardiology) |
|---|---|---|
| `answer_prompt.py` | `SPECIALTY`, `SPECIALTY_DISPLAY`, `SPECIALTY_FOCUS` (the persona + clinical lens) | `SPECIALTY = "cardiology"`; focus on heart/vessels, chest pain, palpitations, ECG/troponin |
| `vocabulary.py` | `PINECONE_NAMESPACE` (point at the new data slice); tune `PULMONOLOGY_RELEVANCE_THRESHOLD` value | `PINECONE_NAMESPACE = "cardiology_v1"` |
| `prompts.py` | `GATEKEEPER_SYSTEM_PROMPT`: the **relevance scoring rubric** (what counts as in-specialty), the **emergency red flags**, and **symptom weighting** for the new specialty | Score chest pain/syncope/palpitations high; cardiac red flags (crushing chest pain + radiation, syncope on exertion) |
| `clinical_policy.py` | `RED_FLAG_PATTERNS` (+ `detect_red_flags`), `HIGH_SIGNAL_SYMPTOMS_TEXT`, `RED_FLAGS_TEXT`; optionally the differential/uncertainty/safeguard text | Cardiac red-flag regexes; high-signal = chest pain, syncope, etc. |
| `query_taxonomy.py` | `QUERY_TUNING` → `priority_entity_types` per query type (the entity types the graph/vector layers surface first) | `["disease","arrhythmia","drug","procedure"]` |
| `entity_rules.py` | drug-pair / salient-pair heuristics if relevant | keep or adjust |
| `messages.py` | `OUT_OF_SCOPE_MESSAGE` / `EMERGENCY_MESSAGE` wording (these mention "pulmonology"/"lung") | "I focus on cardiology…" |

### Step 2 — Edit `Memory_Layer/session_memory/domain/`

| File | What to change |
|---|---|
| `extraction_patterns.py` | `SYMPTOM_PATTERNS`, `CONDITION_PATTERNS`, `CHRONIC_PATTERNS`, `ALLERGY_PATTERNS`, `DRUG_PATTERNS`, `SEVERITY_PATTERNS`, `TRIGGER_PATTERNS` — the regex vocabulary the session memory recognises for the new specialty |
| `risk_rules.py` | `CRITICAL_SYMPTOMS` and `HIGH_SIGNAL_SYMPTOMS` (canonical symptom keys that escalate session risk) |
| `render_fields.py` | usually leave as-is (labels are generic: Symptoms, Medications…). Adjust only if you add new state fields |

### Step 3 — Re-ingest the data into the new namespace

```powershell
# Chunk the new specialty's source PDFs (offline pipeline — see root README)
python run_pulmonology.py --version v1            # produces chunks/v1/...

# Vector store: ingest into the SAME namespace you set in vocabulary.py
python ingest_pinecone.py --namespace cardiology_v1

# Knowledge graph
python ingest_neo4j.py --version v1
```

The retriever reads `PINECONE_NAMESPACE` from `vocabulary.py`, so the namespace
you ingest into **must** match it exactly.

### Step 4 — Verify

```powershell
python -m compileall graphrag Memory_Layer            # nothing broke
python run_graphrag.py --query "<a clearly in-specialty question>"
python run_graphrag.py --query "<a clearly OUT-of-specialty question>"   # should hit the scope gate
```
Confirm: Stage 1 logs `namespace: cardiology_v1`; an in-specialty query gets a
grounded answer with `🔗 Graph entities`; an out-of-specialty query is restricted.

---

## What you do NOT touch (domain-agnostic plumbing)

`graphrag/pipeline/`, `graphrag/query_understanding/` (analyzer/routing/query_config
*logic* — only the domain *data* it reads), `graphrag/retrievers/`,
`graphrag/processors/`, `graphrag/llm/`, `graphrag/memory/`, `graphrag/config/`,
`graphrag/utils/`, and all of `Memory_Layer/session_memory/*.py` outside `domain/`.
These read **from** the domain packages and never need editing for a specialty swap.

---

## Important naming note (keep the retarget "domain-only")

A few symbols carry the word "pulmonology" but are just the **active value/label**,
not pulmonology-specific logic. To preserve the guarantee that a retarget edits
**only the two domain folders**, keep these NAMES and change only their VALUES /
text:

- `vocabulary.PULMONOLOGY_RELEVANCE_THRESHOLD` — change the number; keep the name
  (renaming it would force an edit in the non-domain `graphrag_pipeline.py`).
- The gatekeeper JSON field **`pulmonology_relevance`** (in `prompts.py` and read
  by `graphrag_pipeline.py`) — keep this key name; treat it as a generic
  "specialty relevance" score. Update only the *rubric text* around it.
- `clinical_policy.detect_red_flags()` / `RED_FLAG_PATTERNS` — keep the function/
  constant names; change the regex contents.

> If you prefer fully specialty-neutral names (e.g. `SPECIALTY_RELEVANCE_THRESHOLD`,
> `specialty_relevance`), that's a one-time refactor touching `vocabulary.py`,
> `prompts.py`, and `graphrag_pipeline.py` together — see HANDOFF §9-P2. Until
> then, follow the "change values, keep names" rule above.

---

## Complete map: every specialty-coupled symbol

`graphrag/domain/`
- `answer_prompt.py` → `SPECIALTY`, `SPECIALTY_DISPLAY`, `SPECIALTY_FOCUS`, `BASE_ROLE` (persona name "Enervera"), `INTENT_LAYERS`, `RISK_LAYERS`
- `prompts.py` → `GATEKEEPER_SYSTEM_PROMPT` (intents, emergency red flags, relevance rubric, symptom-weighting section)
- `clinical_policy.py` → `RED_FLAG_PATTERNS`, `HIGH_SIGNAL_SYMPTOMS_TEXT`, `RED_FLAGS_TEXT`, `DIFFERENTIAL_POLICY`, `UNCERTAINTY_POLICY`, `QUESTIONING_POLICY`, `SAFEGUARDS`, `MAX_FOLLOWUP_QUESTIONS`
- `query_taxonomy.py` → `QUERY_TYPES`, `QUERY_TUNING` (esp. `priority_entity_types`), `INTENT_TO_QUERYTYPE`
- `vocabulary.py` → `PINECONE_NAMESPACE`, `PULMONOLOGY_RELEVANCE_THRESHOLD`, `GRAPH_NODE_LABEL`, `CLINICAL_STATE_KEYS`, `DEFAULT_ANSWER_GOAL`
- `entity_rules.py` → `DRUG_NAME_PATTERN`, `DRUG_NAME_STOPWORDS`
- `messages.py` → `REFUSAL_MESSAGE`, `EMERGENCY_MESSAGE`, `OUT_OF_SCOPE_MESSAGE`

`Memory_Layer/session_memory/domain/`
- `extraction_patterns.py` → all `*_PATTERNS` dicts + `TRIGGER_PATTERNS`
- `risk_rules.py` → `CRITICAL_SYMPTOMS`, `HIGH_SIGNAL_SYMPTOMS`, `RISK_ORDER`
- `render_fields.py` → `STATE_RENDER_FIELDS`, `SUMMARY_RENDER_FIELDS`, `ROLE_LABELS`

Data: a Pinecone namespace + a Neo4j graph populated from the new corpus.

---

## Multi-specialty (running several at once)

The current design is **single active specialty** (one namespace, one threshold,
one persona — chosen at code level). To serve multiple specialties from one
deployment you'd promote these domain constants to a per-specialty config object
selected at request time (by an arg/route). That's a larger change — see
HANDOFF §9. For now: one deployment per specialty, each pointed at its own
namespace.
