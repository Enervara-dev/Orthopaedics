"""
Upsert the generated chunks into Pinecone for vector retrieval.

Embedding model: PINECONE_EMBEDDING_MODEL (default llama-text-embed-v2), a Pinecone-
HOSTED model. Two paths, auto-detected:
  • INTEGRATED index (created for a model)  → Pinecone embeds for us (upsert_records).
  • REGULAR index (just a dimension)        → we embed via the Inference API, then upsert.

Design choices (so it's production-grade):
  • Vector id = chunk_id  → the SAME key as the Neo4j :Chunk node, and idempotent
    (re-running overwrites, never duplicates).
  • ONE namespace + `specialties` stored as metadata  → a cross-specialty chunk lives
    once and every specialty agent finds it via filter={"specialties": {"$in": [...]}}.
  • Embedded text = topic + text + summary + clinical_significance (full semantic payload).
  • Rich metadata (book/chapter/topic/page/publication_year/specialties/entities) for
    filtering, recency ranking, citation, and graph join.

Usage:
    python ingest_pinecone.py                      # ingest chunks/v1 into the configured index
    python ingest_pinecone.py --version v1
    python ingest_pinecone.py --dry-run            # build records, no network calls
    python ingest_pinecone.py --limit 20           # ingest only the first 20 (smoke)
    python ingest_pinecone.py --namespace orthopaedics
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Trust the OS certificate store so the Pinecone SDK works behind a TLS-intercepting
# proxy/VPN (otherwise: CERTIFICATE_VERIFY_FAILED). Best-effort.
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_pinecone")

from chunking.config.settings import settings  # noqa: E402
from chunking.storage.versioned_storage import _long  # extended-length path on Windows  # noqa: E402

# Pinecone limits: integrated upsert_records and inference.embed cap at ~96/call.
BATCH = 96
# The record field the integrated index embeds. MUST match the index's field_map —
# the existing 'enervara-specialists' maps {"text": "text"}, so it's "text".
EMBED_FIELD = "text"
DEFAULT_CLOUD, DEFAULT_REGION = "aws", "us-east-1"


# ── chunk → record ────────────────────────────────────────────────────────────
def iter_chunk_files(version: str):
    root = Path(settings.storage_base_path) / version
    if not root.exists():
        logger.error("No chunks at %s — run the pipeline first (or pass --version).", root)
        sys.exit(1)
    return sorted(root.rglob("*.json"))


def embed_text(c: dict) -> str:
    parts = [c.get("source", {}).get("topic", ""), c.get("text", ""),
             c.get("summary", ""), c.get("clinical_significance", "")]
    return "\n".join(p for p in parts if p).strip()


def metadata(c: dict) -> dict:
    """Filterable / displayable fields. Pinecone allows str, number, bool, list[str];
    never None — so drop empties."""
    src = c.get("source", {})
    ents = c.get("entities", [])
    # NOTE: no "text" key here — that name is reserved for the embedded field
    # (EMBED_FIELD), which carries the composed semantic payload.
    md = {
        "chunk_id": c.get("chunk_id", ""),
        "summary": c.get("summary", ""),
        "clinical_significance": c.get("clinical_significance", ""),
        "book": src.get("book", ""),
        "chapter": src.get("chapter", ""),
        "topic": src.get("topic", ""),
        "page": str(src.get("page", "")),
        "publication_year": src.get("publication_year"),
        "specialties": c.get("specialties", []),
        "entities": [e.get("name", "") for e in ents][:64],
        "entity_ids": [e.get("id", "") for e in ents][:64],
        "doc_id": c.get("chunk_id", "").rsplit("-", 1)[0],
        "version": c.get("metadata", {}).get("version", ""),
    }
    # Drop None / empty so Pinecone accepts it.
    return {k: v for k, v in md.items() if v not in (None, "", [])}


def load_records(version: str, limit: int):
    files = iter_chunk_files(version)
    if limit:
        files = files[:limit]
    records = []
    for f in files:
        try:
            with open(_long(f), encoding="utf-8") as fh:   # _long: survive >260-char paths
                c = json.load(fh)
        except Exception as e:
            logger.warning("Skipping unreadable %s: %s", f.name, e)
            continue
        cid = c.get("chunk_id")
        txt = embed_text(c)
        if not cid or not txt:
            continue
        records.append({"id": cid, "embed_text": txt, "metadata": metadata(c)})
    return records


def _batches(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# ── Pinecone ──────────────────────────────────────────────────────────────────
def ensure_index(pc, name: str, model: str):
    """Return (index_handle, is_integrated). Create an integrated index if missing."""
    if not pc.has_index(name):
        logger.info("Index '%s' not found — creating integrated index for %s (%s/%s)...",
                    name, model, DEFAULT_CLOUD, DEFAULT_REGION)
        pc.create_index_for_model(
            name=name, cloud=DEFAULT_CLOUD, region=DEFAULT_REGION,
            embed={"model": model, "field_map": {"text": EMBED_FIELD}},
        )
    desc = pc.describe_index(name)
    is_integrated = getattr(desc, "embed", None) is not None
    logger.info("Index '%s' ready | integrated-embedding=%s | host=%s",
                name, is_integrated, getattr(desc, "host", "?"))
    return pc.Index(name), is_integrated


def upsert_integrated(index, namespace, records):
    ns = namespace or "__default__"
    done = 0
    for batch in _batches(records, BATCH):
        # SDK requires keyword args (upsert_records is decorated keyword-only).
        index.upsert_records(
            namespace=ns,
            records=[{"_id": r["id"], EMBED_FIELD: r["embed_text"], **r["metadata"]}
                     for r in batch],
        )
        done += len(batch)
        logger.info("upserted %d/%d (integrated)", done, len(records))
    return done


def upsert_manual(pc, index, namespace, model, records):
    done = 0
    for batch in _batches(records, BATCH):
        emb = pc.inference.embed(
            model=model, inputs=[r["embed_text"] for r in batch],
            parameters={"input_type": "passage", "truncate": "END"},
        )
        vectors = [{"id": r["id"], "values": list(e["values"]),
                    "metadata": {**r["metadata"], "text": r["embed_text"]}}
                   for r, e in zip(batch, emb)]
        index.upsert(vectors=vectors, namespace=namespace)
        done += len(batch)
        logger.info("embedded + upserted %d/%d (manual)", done, len(records))
    return done


def main():
    ap = argparse.ArgumentParser(description="Upsert chunks into Pinecone")
    ap.add_argument("--version", default="v1", help="Chunk version under chunks/ (default: v1)")
    ap.add_argument("--namespace", default="orthopaedics", help="Pinecone namespace (default: orthopaedics)")
    ap.add_argument("--limit", type=int, default=0, help="Only ingest the first N chunks (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="Build records but make NO Pinecone calls")
    args = ap.parse_args()

    if not settings.pinecone_api_key or not settings.pinecone_index_name:
        logger.error("Set PINECONE_API_KEY and PINECONE_INDEX_NAME in .env first.")
        sys.exit(2)

    records = load_records(args.version, args.limit)
    logger.info("Built %d records from chunks/%s/ (model=%s, index=%s)",
                len(records), args.version, settings.pinecone_embedding_model,
                settings.pinecone_index_name)
    if not records:
        logger.error("No records to upsert.")
        sys.exit(1)

    if args.dry_run:
        sample = records[0]
        logger.info("DRY RUN — sample record id=%s", sample["id"])
        logger.info("  embed_text[:160]: %s", sample["embed_text"][:160].replace("\n", " "))
        logger.info("  metadata keys: %s", list(sample["metadata"].keys()))
        logger.info("  specialties: %s", sample["metadata"].get("specialties"))
        sys.exit(0)

    from pinecone import Pinecone
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index, is_integrated = ensure_index(pc, settings.pinecone_index_name,
                                        settings.pinecone_embedding_model)

    if is_integrated:
        n = upsert_integrated(index, args.namespace, records)
    else:
        n = upsert_manual(pc, index, args.namespace, settings.pinecone_embedding_model, records)

    try:
        stats = index.describe_index_stats()
        total = getattr(stats, "total_vector_count", None)
        logger.info("DONE: upserted %d chunks. Index now holds %s vectors.", n, total)
    except Exception:
        logger.info("DONE: upserted %d chunks.", n)


if __name__ == "__main__":
    main()
