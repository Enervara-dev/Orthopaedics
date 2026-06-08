"""
Push the generated chunks into a Neo4j knowledge graph.

Graph model (shares chunk_id with the Pinecone vectors, so vector hits join straight
into the graph):

    (:Chunk {chunk_id, text, summary, clinical_significance, book, chapter, topic,
             page, publication_year, version, model})
    (:Entity {id, name, type, aliases})          # MERGE on canonical id → one node
    (:Specialty {name})                          # across the whole corpus
    (:Chunk)-[:MENTIONS]->(:Entity)
    (:Chunk)-[:RELEVANT_TO]->(:Specialty)        # cross-specialty visibility
    (:Entity)-[:CAUSES|TREATS|MANIFESTS_AS|... {onset, severity, chunk_id}]->(:Entity)

Typed relationships use the controlled relation vocabulary as the actual Neo4j
relationship type (e.g. [:CAUSES]); qualifiers (onset/severity/...) ride as edge
properties so the clinical axis survives. Idempotent — re-running MERGEs, never dupes.

Usage:
    python ingest_neo4j.py                 # load chunks/v1 into Neo4j
    python ingest_neo4j.py --version v1
    python ingest_neo4j.py --dry-run       # build rows, no DB connection
    python ingest_neo4j.py --limit 20      # smoke
"""

import sys
import os
import re
import json
import argparse
import logging
from collections import defaultdict
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_neo4j")

from chunking.config.settings import settings  # noqa: E402
from chunking.storage.versioned_storage import _long  # >260-char path reads  # noqa: E402
from chunking.domain import RELATION_TYPE_SET, SPECIALTY_SYNONYMS  # noqa: E402
from chunking.schemas.models import _snake  # noqa: E402

BATCH = 100
_ALLOWED_RELTYPES = {t.upper() for t in RELATION_TYPE_SET}

CONSTRAINTS = [
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT specialty_name IF NOT EXISTS FOR (s:Specialty) REQUIRE s.name IS UNIQUE",
]


def reltype(t: str) -> str:
    """Controlled relation type → safe Neo4j relationship type (validated, no injection)."""
    ut = re.sub(r"[^A-Z0-9_]", "", str(t).upper())
    return ut if ut in _ALLOWED_RELTYPES else "ASSOCIATED_WITH"


def iter_chunk_files(version: str):
    root = Path(settings.storage_base_path) / version
    if not root.exists():
        logger.error("No chunks at %s — run the pipeline first.", root)
        sys.exit(1)
    return sorted(root.rglob("*.json"))


def norm_specialty(val: str) -> str:
    s = _snake(val) if val else ""
    return SPECIALTY_SYNONYMS.get(s, s)


def load_rows(version: str, limit: int, source_specialty_default: str = ""):
    """Flatten chunks into row lists for efficient UNWIND batching."""
    files = iter_chunk_files(version)
    if limit:
        files = files[:limit]
    default_sp = norm_specialty(source_specialty_default)
    batches = []
    cur_chunks, cur_ents, cur_specs, cur_rels = [], [], [], defaultdict(list)

    def flush():
        if cur_chunks:
            batches.append((cur_chunks[:], cur_ents[:], cur_specs[:],
                            {k: v[:] for k, v in cur_rels.items()}))

    for f in files:
        try:
            with open(_long(f), encoding="utf-8") as fh:
                c = json.load(fh)
        except Exception as e:
            logger.warning("Skipping %s: %s", f.name, e)
            continue
        cid = c.get("chunk_id")
        if not cid:
            continue
        src, meta = c.get("source", {}), c.get("metadata", {})
        cur_chunks.append({
            "chunk_id": cid, "text": c.get("text", ""), "summary": c.get("summary", ""),
            "clinical_significance": c.get("clinical_significance", ""),
            "book": src.get("book", ""), "chapter": src.get("chapter", ""),
            "topic": src.get("topic", ""), "page": str(src.get("page", "")),
            "publication_year": src.get("publication_year"),
            "source_specialty": norm_specialty(src.get("source_specialty")) or default_sp,
            "version": meta.get("version", ""), "model": meta.get("model", ""),
        })
        ids = {e.get("id") for e in c.get("entities", []) if e.get("id")}
        for e in c.get("entities", []):
            if e.get("id"):
                cur_ents.append({"chunk_id": cid, "id": e["id"], "name": e.get("name", ""),
                                 "type": e.get("type", ""), "aliases": e.get("aliases", [])})
        for s in c.get("specialties", []):
            cur_specs.append({"chunk_id": cid, "name": s})
        for r in c.get("relations", []):
            s, t = r.get("source"), r.get("target")
            if s in ids and t in ids:
                cur_rels[reltype(r.get("type"))].append(
                    {"source": s, "target": t, "chunk_id": cid,
                     "qualifiers": r.get("qualifiers", {}) or {}})

        if len(cur_chunks) >= BATCH:
            flush()
            cur_chunks, cur_ents, cur_specs, cur_rels = [], [], [], defaultdict(list)
    flush()
    return batches


def write_batch(tx, chunks, ents, specs, rels_by_type):
    tx.run("""
        UNWIND $rows AS c
        MERGE (ch:Chunk {chunk_id: c.chunk_id})
        SET ch.text=c.text, ch.summary=c.summary,
            ch.clinical_significance=c.clinical_significance, ch.book=c.book,
            ch.chapter=c.chapter, ch.topic=c.topic, ch.page=c.page,
            ch.publication_year=c.publication_year, ch.version=c.version, ch.model=c.model
    """, rows=chunks)
    # Parent node: the book's home specialty owns its chunks.
    # (:Specialty {name})-[:HAS_CHUNK]->(:Chunk)
    tx.run("""
        UNWIND $rows AS c
        WITH c WHERE c.source_specialty IS NOT NULL AND c.source_specialty <> ''
        MERGE (sp:Specialty {name: c.source_specialty})
        MERGE (ch:Chunk {chunk_id: c.chunk_id})
        MERGE (sp)-[:HAS_CHUNK]->(ch)
    """, rows=chunks)
    if ents:
        tx.run("""
            UNWIND $rows AS e
            MERGE (en:Entity {id: e.id})
            SET en.name=e.name, en.type=e.type,
                en.aliases = coalesce(en.aliases, []) +
                             [a IN e.aliases WHERE NOT a IN coalesce(en.aliases, [])]
            WITH en, e
            MATCH (ch:Chunk {chunk_id: e.chunk_id})
            MERGE (ch)-[:MENTIONS]->(en)
        """, rows=ents)
    if specs:
        tx.run("""
            UNWIND $rows AS s
            MERGE (sp:Specialty {name: s.name})
            WITH sp, s
            MATCH (ch:Chunk {chunk_id: s.chunk_id})
            MERGE (ch)-[:RELEVANT_TO]->(sp)
        """, rows=specs)
    for rt, rels in rels_by_type.items():
        # Two separate MATCHes (not a comma pattern) so Neo4j doesn't flag a
        # cartesian product — both are pinned by the unique-id index.
        tx.run(f"""
            UNWIND $rows AS r
            MATCH (s:Entity {{id: r.source}})
            MATCH (t:Entity {{id: r.target}})
            MERGE (s)-[rel:`{rt}`]->(t)
            SET rel.chunk_id = r.chunk_id, rel += r.qualifiers
        """, rows=rels)


def main():
    ap = argparse.ArgumentParser(description="Load chunks into Neo4j")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--source-specialty", default="pulmonology",
                    help="Parent specialty for chunks lacking source.source_specialty "
                         "(the existing v1 chunks). Default: pulmonology.")
    ap.add_argument("--dry-run", action="store_true", help="Build rows, no DB connection")
    args = ap.parse_args()

    batches = load_rows(args.version, args.limit, args.source_specialty)
    n_chunks = sum(len(b[0]) for b in batches)
    n_ents = sum(len(b[1]) for b in batches)
    n_rels = sum(len(v) for b in batches for v in b[3].values())
    logger.info("Built %d chunks, %d entity-mentions, %d relations from chunks/%s/",
                n_chunks, n_ents, n_rels, args.version)
    if not n_chunks:
        sys.exit(1)

    if args.dry_run:
        logger.info("DRY RUN — no DB connection made.")
        sys.exit(0)

    if not settings.neo4j_uri or not settings.neo4j_password:
        logger.error("Set NEO4J_URI and NEO4J_PASSWORD in .env first.")
        sys.exit(2)

    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_username, settings.neo4j_password))
    try:
        driver.verify_connectivity()
        logger.info("Connected to Neo4j at %s (db=%s)", settings.neo4j_uri, settings.neo4j_database)
        with driver.session(database=settings.neo4j_database) as s:
            for c in CONSTRAINTS:
                s.run(c)
            done = 0
            for chunks, ents, specs, rels in batches:
                s.execute_write(write_batch, chunks, ents, specs, rels)
                done += len(chunks)
                logger.info("loaded %d/%d chunks", done, n_chunks)
            # Summary counts
            counts = s.run("""
                CALL { MATCH (c:Chunk) RETURN count(c) AS chunks }
                CALL { MATCH (e:Entity) RETURN count(e) AS entities }
                CALL { MATCH (sp:Specialty) RETURN count(sp) AS specialties }
                CALL { MATCH ()-[r]->() RETURN count(r) AS rels }
                RETURN chunks, entities, specialties, rels
            """).single()
            logger.info("DONE. Graph now holds: %s Chunks, %s Entities, %s Specialties, %s relationships.",
                        counts["chunks"], counts["entities"], counts["specialties"], counts["rels"])
    finally:
        driver.close()


if __name__ == "__main__":
    main()
