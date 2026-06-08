from dataclasses import dataclass
from typing import List

from graphrag.domain.query_taxonomy import DEFAULT_QUERY_TYPE, QUERY_TUNING
from graphrag.query_understanding.query_types import QueryType


@dataclass
class QueryConfig:
    """
    Drives ALL pipeline behaviour for a given query type.
    Every downstream component reads from this — no hardcoded logic elsewhere.

    The per-type VALUES (top_k, graph hops, priority entity types, goal) are
    domain-specific and live in graphrag/domain/query_taxonomy.py::QUERY_TUNING.
    This dataclass is the domain-agnostic shape they are loaded into.
    """
    query_type:             QueryType
    vector_top_k:           int         # how many candidates to pull from Pinecone
    reranker_top_k:         int         # how many to keep after reranking
    graph_hops:             int         # 1 or 2-hop Neo4j traversal
    graph_enabled:          bool        # whether to query Neo4j at all
    priority_entity_types:  List[str]   # entity types to surface / boost
    goal:                   str         # human-readable description (logged)
    boost_drug_pairs:       bool = False  # special flag for drug_interaction only


# ---------------------------------------------------------------------------
# Registry — assembled from the domain tuning table, one config per query type
# ---------------------------------------------------------------------------
QUERY_CONFIGS: dict[QueryType, QueryConfig] = {
    QueryType(qt_value): QueryConfig(query_type=QueryType(qt_value), **tuning)
    for qt_value, tuning in QUERY_TUNING.items()
}

_DEFAULT_CONFIG = QUERY_CONFIGS[QueryType(DEFAULT_QUERY_TYPE)]


def get_config(query_type: QueryType) -> QueryConfig:
    """Retrieve the pipeline config for a classified query type."""
    return QUERY_CONFIGS.get(query_type, _DEFAULT_CONFIG)
