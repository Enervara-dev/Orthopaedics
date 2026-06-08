"""
graphrag.domain.vocabulary
────────────────────────────
Small domain vocabulary constants shared across the retrieval layers.

⭐ EDIT FOR A NEW SPECIALTY/USE CASE.
"""

# Keys the gatekeeper emits under `medical_entities` (analyzer JSON output).
# routing._has_extracted_entities checks these to decide whether the query is
# grounded enough for full retrieval.
CLINICAL_STATE_KEYS: tuple[str, ...] = ("symptoms", "drugs", "conditions")

# Neo4j node label the graph retriever traverses. Must match what the ingestion
# pipeline writes (ingest_neo4j.py). Labels cannot be parameterised in Cypher,
# so this is interpolated into the query string.
GRAPH_NODE_LABEL: str = "Entity"

# Pinecone namespace the retriever queries. Vector retrieval is RESTRICTED to
# this namespace — the data must be ingested into it
# (e.g. `python ingest_pinecone.py --namespace pulmonology_v1`). Change this when
# retargeting to another specialty's index slice.
PINECONE_NAMESPACE: str = "pulmonology_v1"

# Minimum pulmonology-relevance score (0–100, produced by the gatekeeper) a
# non-greeting / non-follow-up query must reach to be answered. Below this the
# pipeline restricts the query as out-of-specialty.
PULMONOLOGY_RELEVANCE_THRESHOLD: int = 75

# Default human-readable answer goal when no per-type goal is supplied.
DEFAULT_ANSWER_GOAL: str = "provide a medical answer"
