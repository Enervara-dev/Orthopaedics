from enum import Enum

from graphrag.domain.query_taxonomy import QUERY_TYPES


# QueryType is generated from the domain taxonomy (graphrag/domain/query_taxonomy.py)
# so a new specialty edits the domain package only. Members are still accessed
# statically — e.g. QueryType.SYMPTOM_QUERY, QueryType("symptom_query") — and the
# str mixin keeps `.value` comparisons working exactly as before.
QueryType = Enum(
    "QueryType",
    {name: value for name, value in QUERY_TYPES},
    type=str,
)
