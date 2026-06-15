# Enervera: Orthopaedics Knowledge Assistant

An Orthopaedic GraphRAG Assistant that combines Vector Search, Knowledge Graph Retrieval, Clinical Memory, and Gemini reasoning to provide evidence-grounded orthopaedic information.

The system is designed with a domain-driven architecture that separates the core orchestration, retrieval layers, and client connections from the clinical rules, vocabulary, and taxonomies.

---

## 1. What This System Is

This assistant is a clinical-grade reference and decision support tool for **orthopaedics, musculoskeletal medicine, and sports medicine**. It operates across a multi-stage reasoning pipeline:

- **Vector Retrieval**: Querying a Pinecone index containing high-density semantic chunks of a curated clinical reference library.
- **Knowledge-Graph Traversal**: Navigating a Neo4j database to extract structured entity relations, ensuring connected pathologies, procedures, and complications are surfaced.
- **Session Memory**: Accumulating patient statements (symptoms, severity, duration) dynamically to maintain conversational continuity and apply triage-grade safety logic.
- **Episodic Memory**: Storing long-term, per-user consult history (across separate sessions) to recall persistent conditions and chronic history.

---

## 2. Architecture & Request Flow

The pipeline is orchestrated by `graphrag.pipeline.graphrag_pipeline.GraphRAGPipeline.run()` and executes the following sequential stages:

```
                  User Query
                      │
                      ▼
        [STAGE -2: Session Memory Load]  <── Retrieves active consult state
                      │
                      ├─► [Deterministic Red-Flag Check]  ──► Escalate if emergency
                      ▼
     [STAGE -1: Medical Gatekeeper / Analyzer]    ◄── Evaluates scope & intent
                      │
                      ├───► [Out-of-Scope Gate] ────► Refusal message (if non-orthopaedic)
                      ▼
              [STAGE 0: Routing]                 ────► Gated (NO_RETRIEVAL | MEMORY_FIRST | HYBRID_RAG)
                      │
                      ▼
     [STAGE 1: Vector Retrieval & Reranking]     ◄── Queries Pinecone index
                      │
                      ▼
          [STAGE 2: Entity Extraction]           ◄── Normalizes entities from vector context
                      │
                      ▼
         [STAGE 3: Graph Traversal]              ◄── Traverses Neo4j relationships
                      │
                      ▼
      [STAGE 3.5: Episodic Retrieval (Opt)]      ◄── Queries user's history index
                      │
                      ▼
         [STAGE 4: LLM Response Gen]             ◄── Gemini constructs clinical response
                      │
                      ▼
       [STAGE 5: Session Update & Save]          ──► Persists conversation & structured state
```

### Retrieval & Knowledge Graph Schema

Our knowledge graph maps clinical concepts into a strict schema:

- **Entities**:
  - `Condition` (e.g., *osteoarthritis*, *rotator cuff tear*, *ACL tear*)
  - `Symptom` (e.g., *knee pain*, *stiffness*, *instability*, *numbness*)
  - `Anatomical_Structure` (e.g., *knee*, *shoulder*, *lumbar spine*, *femur*)
  - `Treatment` (e.g., *immobilization*, *bracing*, *conservative management*)
  - `Surgical_Procedure` (e.g., *ACL reconstruction*, *total hip arthroplasty*, *ORIF*)
  - `Rehabilitation` (e.g., *physiotherapy*, *gait training*, *exercise therapy*)
  - `Medication` (e.g., *NSAID*, *analgesic*, *diclofenac*)
  - `Diagnostic_Test` (e.g., *x-ray*, *MRI*, *CT scan*)
  - `Complication` (e.g., *nonunion*, *compartment syndrome*, *implant failure*)
  - `Outcome` (e.g., *recovery*, *fracture union*, *functional improvement*)

- **Graph Relationships**:
  - `fracture` ── `AFFECTS` ──► `bone`
  - `ACL tear` ── `AFFECTS` ──► `knee`
  - `osteoarthritis` ── `PRESENTS_WITH` ──► `joint pain`
  - `meniscus tear` ── `PRESENTS_WITH` ──► `locking`
  - `supracondylar fracture` ── `COMPLICATED_BY` ──► `brachial artery injury`

---

## 3. Repository Layout

```
├── chunker.py                # Ingestion: segment PDFs into semantic blocks
├── ingest_pinecone.py        # Ingestion: upsert semantic vectors into Pinecone
├── ingest_neo4j.py           # Ingestion: import clinical entity graphs into Neo4j
├── run_graphrag.py           # CLI: Interactive REPL / chat application
├── api.py                    # Service: FastAPI HTTP entrypoint
├── live_check.py             # Diagnostic: verification of live Pinecone/Neo4j stores
├── smoke_test.py             # Testing: offline regression/integration suite (stubbed APIs)
├── test_pipeline.py          # Testing: chunker pipeline validation
├── render.yaml               # Deployment: Render blueprint configuration
├── pyproject.toml            # Dependencies: project requirements & metadata
├── graphrag/                 # Domain-agnostic orchestration & retrieval
│   ├── domain/               # ⭐ Domain Configuration (Edit to retarget assistant)
│   │   ├── prompts.py        #   Gatekeeper rubric & system instructions
│   │   ├── answer_prompt.py  #   Answer persona, style, & patient communication focus
│   │   ├── clinical_policy.py#   Red-flag regex, question budgets, & safeties
│   │   ├── query_taxonomy.py #   Retrieval settings & mapping per query intent
│   │   ├── vocabulary.py     #   Pinecone namespace, thresholds, & labels
│   │   └── entity_rules.py   #   Medication name extraction regex
│   ├── pipeline/             #   Retrieval flow coordinator (GraphRAGPipeline)
│   ├── query_understanding/  #   Gatekeeper routing & analyzer models
│   ├── retrievers/           #   Pinecone & Neo4j database interfaces
│   ├── processors/           #   Entity extraction and dedup logic
│   └── llm/                  #   Gemini client & streaming wrappers
├── Memory_Layer/             # Dialogue session & risk state persistence
│   └── session_memory/
│       ├── domain/           # ⭐ Domain Heuristics (Edit to retarget memory)
│       │   ├── extraction_patterns.py # Symptom, drug, and condition regex vocabularies
│       │   └── risk_rules.py          # Escalation conditions & critical symptoms
└── episodic/                 # Longitudinal memory (per-user history)
```

---

## 4. Setup & Installation

### Prerequisites
- Python 3.10+
- Pinecone Account & Index
- Neo4j Instance (Local Desktop or Neo4j Aura)
- Google Gemini API Key

### Install Dependencies
Using `uv` (recommended):
```powershell
uv sync
```
Or using plain `pip`:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Environment Configuration
Copy the template and fill in your API keys and credentials:
```powershell
Copy-Item .env.example .env
```
Ensure you provide `GEMINI_API_KEY`, `PINECONE_API_KEY`, and `NEO4J_PASSWORD`.

---

## 5. Ingestion & Data Population

Before running retrieval, you must populate your data stores from your PDF documents (placed in `dataset/`):

1. **Segment Reference Chunks**:
   ```powershell
   python chunker.py --version v1
   ```
2. **Ingest Vectors into Pinecone**:
   Upsert chunks into the configured namespace (must match the namespace target in `vocabulary.py`):
   ```powershell
   python ingest_pinecone.py --namespace orthopaedics
   ```
3. **Ingest Entity Graphs into Neo4j**:
   ```powershell
   python ingest_neo4j.py --version v1
   ```

---

## 6. Running the Assistant

### CLI Interactive Chat (REPL)
To start a conversation from the console:
```powershell
python run_graphrag.py
```
Options:
- `--query "<text>"`: Run a single query directly and print the answer (e.g. `python run_graphrag.py --query "ACL injury after football"`).
- `--session-id <id>`: Restore / track a specific session state.
- `--user-id <id>`: Engage episodic memory to load long-term user context.
- `--quiet`: Print only the assistant's answer and hide pipeline execution logs.

### HTTP API Server
Run the FastAPI application locally:
```powershell
uvicorn api:app --host 0.0.0.0 --port 8000
```
API endpoints:
- `GET /health`: Health status of databases and services.
- `POST /chat`: Post message to the pipeline (`{"message": "knee pain", "session_id": "session_1"}`).
- `POST /session/end`: Terminate a session and write a summary digest to the user's episodic memory.

---

## 7. Diagnostics & Testing

Verify that your databases are populated and retrieval works:
```powershell
python live_check.py
```
Run offline unit and integration tests (API endpoints and LLMs are mocked out to avoid cost and network dependencies):
```powershell
python test_pipeline.py
python smoke_test.py
```

---

## 8. Clinical Safeties & Triage

### Session Memory Extraction
The memory layer dynamically parses patient messages to extract key features:
- **Symptoms**: `pain`, `swelling`, `stiffness`, `instability`, `numbness`, `tingling`, `weakness`, `deformity`.
- **Conditions**: `fracture`, `dislocation`, `sprain`, `strain`, `osteoarthritis`, `ACL tear`, `meniscus tear`.

### Emergency Escalation (Triage)
If the gatekeeper or the deterministic parser detects symptoms representing a potential emergency, the pipeline escalates to critical risk. The LLM then structures the response prioritizing patient safety, explaining possible serious causes and instructing immediate actions:
- **Critical Red Flags**: `open fracture`, `compartment syndrome`, `cauda equina syndrome`, `septic arthritis`, `neurovascular compromise`, `severe trauma`, `absent distal pulses`.
- **Out of Scope Gate**: Any query that scores below `ORTHOPAEDICS_RELEVANCE_THRESHOLD` (75) on relevance checks is restricted to protect the boundaries of the assistant.
