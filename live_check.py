"""
LIVE data-store check for the GraphRAG pulmonology assistant.

Run this ON YOUR machine (with .env filled in and Neo4j running) to confirm the
ingest landed where retrieval expects:

  - Pinecone main index reachable, and the `pulmonology_v1` NAMESPACE has vectors
    (retrieval is locked to that namespace — data in another namespace = no hits).
  - Neo4j has :Entity / :Chunk nodes and relationships.
  - Episodic index reachable (info only).

These checks are FREE (no Gemini / no answer generation). Add `--query "..."`
to also run ONE real end-to-end pipeline turn (this DOES call Gemini = paid).

Usage:
    python live_check.py
    python live_check.py --query "cough and breathlessness for a week"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Windows TLS bootstrap (same fix the app uses) so Pinecone/Neo4j verify behind
# a corporate proxy/AV/VPN.
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except Exception:
        pass

_pass = _fail = 0


def ok(name: str, cond: bool, detail: str = "") -> bool:
    global _pass, _fail
    cond = bool(cond)
    _pass += cond
    _fail += (not cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  → {detail}" if detail else ""))
    return cond


def _ns_count(stats, namespace: str) -> int:
    namespaces = getattr(stats, "namespaces", None) or (stats.get("namespaces") if isinstance(stats, dict) else {}) or {}
    ns = namespaces.get(namespace)
    if ns is None:
        return 0
    return int(getattr(ns, "vector_count", None) or (ns.get("vector_count") if isinstance(ns, dict) else 0) or 0)


def check_pinecone() -> None:
    print("\n=== Pinecone ===")
    from pinecone import Pinecone
    from graphrag.config.settings import settings
    from graphrag.domain.vocabulary import PINECONE_NAMESPACE

    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(settings.PINECONE_INDEX_NAME)
    stats = index.describe_index_stats()

    namespaces = getattr(stats, "namespaces", None) or (stats.get("namespaces") if isinstance(stats, dict) else {}) or {}
    print(f"  index='{settings.PINECONE_INDEX_NAME}'  namespaces seen: "
          + (", ".join(f"{k}:{_ns_count(stats, k)}" for k in namespaces) or "(none)"))

    n = _ns_count(stats, PINECONE_NAMESPACE)
    if not ok(f"namespace '{PINECONE_NAMESPACE}' has vectors", n > 0, f"{n} vectors"):
        print(f"     ↳ Retrieval is locked to '{PINECONE_NAMESPACE}'. If your data is under a "
              f"different namespace above, re-ingest:\n"
              f"       python ingest_pinecone.py --namespace {PINECONE_NAMESPACE}")


def check_neo4j() -> None:
    print("\n=== Neo4j ===")
    from neo4j import GraphDatabase
    from graphrag.config.settings import Config

    driver = GraphDatabase.driver(Config.NEO4J_URI, auth=(Config.NEO4J_USER, Config.NEO4J_PWD))
    try:
        driver.verify_connectivity()
        ok("connected", True, Config.NEO4J_URI)
        with driver.session() as s:
            rec = s.run(
                "CALL { MATCH (e:Entity) RETURN count(e) AS entities } "
                "CALL { MATCH (c:Chunk) RETURN count(c) AS chunks } "
                "CALL { MATCH (sp:Specialty) RETURN count(sp) AS specialties } "
                "CALL { MATCH ()-[r]->() RETURN count(r) AS rels } "
                "RETURN entities, chunks, specialties, rels"
            ).single()
        print(f"  Entities={rec['entities']}  Chunks={rec['chunks']}  "
              f"Specialties={rec['specialties']}  Relationships={rec['rels']}")
        ok("graph has Entity nodes", rec["entities"] > 0, f"{rec['entities']}")
        ok("graph has relationships", rec["rels"] > 0, f"{rec['rels']}")
    finally:
        driver.close()


def check_episodic() -> None:
    print("\n=== Episodic index (info) ===")
    from pinecone import Pinecone
    from graphrag.config.settings import settings
    try:
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        if not pc.has_index(settings.PINECONE_EPISODIC_INDEX_NAME):
            print(f"  '{settings.PINECONE_EPISODIC_INDEX_NAME}' not found (fine if episodic is disabled).")
            return
        stats = pc.Index(settings.PINECONE_EPISODIC_INDEX_NAME).describe_index_stats()
        total = getattr(stats, "total_vector_count", None)
        print(f"  '{settings.PINECONE_EPISODIC_INDEX_NAME}' reachable, total vectors: {total}")
    except Exception as e:
        print(f"  episodic check skipped: {e}")


def run_query(query: str) -> None:
    print("\n=== Live pipeline query (calls Gemini — paid) ===")
    from graphrag.pipeline.graphrag_pipeline import GraphRAGPipeline
    pipe = GraphRAGPipeline()
    try:
        answer = pipe.run(query_text=query, session_id="live-check")
        ok("pipeline returned an answer", bool(answer), f"{len(answer or '')} chars")
    finally:
        pipe.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Live data-store check")
    ap.add_argument("--query", default=None, help="Also run one real pipeline turn (calls Gemini).")
    args = ap.parse_args()

    from graphrag.config.settings import ConfigError, settings
    try:
        settings.validate_required("cli")
    except ConfigError as e:
        print(f"[config] {e}")
        sys.exit(2)

    print("=" * 60)
    print("  LIVE DATA-STORE CHECK")
    print("=" * 60)
    try:
        check_pinecone()
    except Exception as e:
        ok("Pinecone reachable", False, repr(e))
    try:
        check_neo4j()
    except Exception as e:
        ok("Neo4j reachable", False, repr(e))
    check_episodic()

    if args.query:
        try:
            run_query(args.query)
        except Exception as e:
            ok("live query", False, repr(e))

    print("\n" + "=" * 60)
    print(f"  RESULT: {_pass} passed, {_fail} failed")
    print("=" * 60)
    sys.exit(1 if _fail else 0)


if __name__ == "__main__":
    main()
