"""
graphrag.domain.query_taxonomy
───────────────────────────────
The query taxonomy + per-type retrieval tuning + gatekeeper-intent routing map.

⭐ EDIT FOR A NEW SPECIALTY/USE CASE.
  - QUERY_TYPES        : the canonical set of query categories. `QueryType`
                         (graphrag/query_understanding/query_types.py) is built
                         from this list, so members stay in sync automatically.
  - QUERY_TUNING       : per-type retrieval behaviour (how many candidates to
                         pull, graph hops, which entity types to prioritise, the
                         human-readable goal). `query_config.QUERY_CONFIGS` is
                         assembled from this. `priority_entity_types` is the most
                         specialty-sensitive knob — it names the entity types the
                         graph/vector layers should surface first.
  - INTENT_TO_QUERYTYPE: maps the gatekeeper's intent strings (see
                         prompts.GATEKEEPER_SYSTEM_PROMPT) to query-type values.

This module imports nothing from graphrag — it is a leaf the rest of the
query-understanding layer reads from.
"""

# ── Canonical query taxonomy: (ENUM_MEMBER_NAME, value) ───────────────────────
# QueryType is generated from this, so adding/removing a category here is the
# single edit needed to extend the taxonomy.
QUERY_TYPES: list[tuple[str, str]] = [
    ("SYMPTOM_QUERY",      "symptom_query"),
    ("DRUG_INTERACTION",   "drug_interaction"),
    ("DIAGNOSIS",          "diagnosis"),
    ("GUIDELINE",          "guideline"),
    ("LAB_INTERPRETATION", "lab_interpretation"),
    ("PROGNOSIS",          "prognosis"),
    ("OUT_OF_CONTEXT",     "out_of_context"),
    ("UNKNOWN",            "unknown"),
]

# Fallback query-type value when a classified type has no config.
DEFAULT_QUERY_TYPE: str = "unknown"


# ── Per-type retrieval tuning, keyed by query-type value ──────────────────────
# Field names match QueryConfig (minus `query_type`, which is filled in by the
# registry builder). Edit `priority_entity_types`, `graph_hops`, and `goal`
# per specialty.
QUERY_TUNING: dict[str, dict] = {
    "symptom_query": dict(
        vector_top_k          = 15,      # oversample before reranker
        reranker_top_k        = 5,
        graph_hops            = 1,
        graph_enabled         = True,
        priority_entity_types = ["disease", "symptom", "syndrome"],
        goal                  = "cause identification",
    ),
    "drug_interaction": dict(
        vector_top_k          = 15,      # wider net for drug combos
        reranker_top_k        = 5,
        graph_hops            = 2,       # 2-hop to find indirect interactions
        graph_enabled         = True,
        priority_entity_types = ["drug", "drug_class", "mechanism", "side_effect"],
        goal                  = "interaction and risk",
        boost_drug_pairs      = True,
    ),
    "diagnosis": dict(
        vector_top_k          = 15,
        reranker_top_k        = 5,
        graph_hops            = 1,
        graph_enabled         = False,   # summary-driven, graph less important
        priority_entity_types = ["disease", "syndrome", "condition", "disorder"],
        goal                  = "clear explanation / definition",
    ),
    "guideline": dict(
        vector_top_k          = 20,      # deepest retrieval
        reranker_top_k        = 7,       # keep more for structured protocols
        graph_hops            = 1,
        graph_enabled         = True,
        priority_entity_types = ["procedure", "drug", "treatment", "protocol", "therapy"],
        goal                  = "structured clinical protocol",
    ),
    "lab_interpretation": dict(
        vector_top_k          = 15,
        reranker_top_k        = 5,
        graph_hops            = 1,
        graph_enabled         = True,
        priority_entity_types = ["test", "lab_value", "biomarker", "threshold"],
        goal                  = "lab result interpretation",
    ),
    "prognosis": dict(
        vector_top_k          = 15,
        reranker_top_k        = 5,
        graph_hops            = 1,
        graph_enabled         = True,
        priority_entity_types = ["outcome", "risk_factor", "survival", "mortality", "disease"],
        goal                  = "future risk / survival estimate",
    ),
    "out_of_context": dict(
        vector_top_k          = 0,
        reranker_top_k        = 0,
        graph_hops            = 0,
        graph_enabled         = False,
        priority_entity_types = [],
        goal                  = "reject out of context / gibberish queries",
    ),
    "unknown": dict(
        vector_top_k          = 15,
        reranker_top_k        = 5,
        graph_hops            = 1,
        graph_enabled         = True,
        priority_entity_types = [],
        goal                  = "general medical answer",
    ),
}


# ── Gatekeeper intent → query-type value ──────────────────────────────────────
# Intents intentionally omitted (greeting, followup_query, emergency, unknown)
# are short-circuited or routed to NO_RETRIEVAL/MEMORY_FIRST before a QueryType
# is consulted, so the absence of a mapping is the correct signal.
INTENT_TO_QUERYTYPE: dict[str, str] = {
    "symptom_query":    "symptom_query",
    "diagnosis_query":  "diagnosis",
    "medication_query": "drug_interaction",
    "treatment_query":  "guideline",
}
