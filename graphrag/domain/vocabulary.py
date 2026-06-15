"""
graphrag.domain.vocabulary
────────────────────────────
Small domain vocabulary constants shared across the retrieval layers.

⭐ EDIT FOR A NEW SPECIALTY/USE CASE.
"""

# Keys the gatekeeper emits under `medical_entities` (analyzer JSON output).
CLINICAL_STATE_KEYS: tuple[str, ...] = ("symptoms", "drugs", "conditions")

# Neo4j node label the graph retriever traverses.
GRAPH_NODE_LABEL: str = "Entity"

# Pinecone namespace the retriever queries.
# Must match the namespace used during ingestion.
PINECONE_NAMESPACE: str = "orthopaedics"

# Minimum orthopaedics-relevance score (0–100, produced by the gatekeeper)
# a non-greeting / non-follow-up query must reach to be answered.
ORTHOPAEDICS_RELEVANCE_THRESHOLD: int = 75

# Default human-readable answer goal when no per-type goal is supplied.
DEFAULT_ANSWER_GOAL: str = "provide an orthopaedic answer"