# Knowledge Chunker (Pulmonology template)

Converts reference PDFs into validated, graph-ready **micro-chunks** — each with
extracted entities, directional relations, a summary, and a significance note —
for a RAG / knowledge-graph pipeline. Ships configured for pulmonology /
respiratory medicine, but **retargets to any domain by editing one file**.

## ⭐ Retarget to a new use case: edit `chunking/domain.py`

`chunking/domain.py` is the single source of all domain knowledge:

- entity types + synonym map, relation types + fallback
- the extraction **system prompt**
- section / concept **segmentation patterns**
- chunk **validation thresholds** (min entities, relation ratio, max tokens)

Everything else (loaders, cleaner, LLM plumbing, storage, the `MicroChunk` output
shape) is domain-agnostic and stays the same. Swap the values in `domain.py` and
the chunker targets legal, finance, engineering, etc. — the chunk JSON shape is
unchanged.

## Layout

```
├── run_pulmonology.py        # generic entry point — processes every PDF in dataset/
├── chunk_pages_27_845.py     # example: plan/smoke/full run over a page range
├── .env.example              # copy to .env and add GEMINI_API_KEY
├── dataset/                  # ← drop your source PDFs here
└── chunking/
    ├── domain.py             # ⭐ all domain config (edit this for a new use case)
    ├── loaders/              # PDF (PyMuPDF) + CSV loaders
    ├── cleaners/             # text normalization; OCR-glyph + header/footer stripping
    ├── detectors/            # section segmentation (page-tracking)
    ├── extractors/           # semantic block splitting (~350 tokens) + page ranges
    ├── pipeline/             # orchestration: batching + parallel workers + resume + summary
    ├── llm/                  # Gemini client, retry engine, jittered backoff, prompt
    ├── schemas/              # Pydantic models + strict validation
    ├── storage/              # versioned JSON output (path-sanitized)
    └── validators/           # JSON parse + schema gate (with partial recovery)
```

`chunks/` and `logs/` are generated at the project root on first run.

## Setup

```powershell
# uv (matches pyproject.toml / .python-version)
uv sync
# or plain pip
python -m venv .venv ; .\.venv\Scripts\Activate.ps1 ; pip install -r requirements.txt

Copy-Item .env.example .env   # then edit .env and set GEMINI_API_KEY
```

## Run

1. Put your reference PDFs in `dataset/`. The filename becomes the `doc_id` / `book_type`.
2. Run:

```powershell
python run_pulmonology.py
python run_pulmonology.py --version v2                  # tag a different output version
python run_pulmonology.py --start-page 27 --end-page 845  # inclusive page range
```

Results land under `chunks/`:
- per-chunk JSON → `chunks/<version>/<book>/<topic>/<chunk_id>.json`
- aggregated     → `chunks/_aggregated/<version>/<book>/<topic>.json`

## How it works

PDF → clean text → segment sections (page-tracked) → ~350-token semantic blocks →
batched LLM extraction (5 blocks/call, 4 parallel workers, jittered backoff) →
strict validation (≥3 entities, relations ≥ entities/2, ≤650 tokens) → authoritative
provenance stamping (model, page, version, timestamp) → versioned JSON.

Re-runs are **resumable**: completed blocks are marked in `logs/processed_blocks/`
and skipped; failures are written to a per-run manifest under `logs/`.

## Cost notes

Output tokens dominate cost. The pipeline keeps output lean (entities emit only
`name`+`type`; `normalized_name`/`properties` are derived locally) and sends a
compact schema per call. The biggest lever is the model — see `.env.example`.
```
