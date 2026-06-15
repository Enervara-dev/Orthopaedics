"""
Stage 2 — Relation Repair.

The LLM over-uses ASSOCIATED_WITH as a catch-all relation type. This module
replaces it with a semantically specific relation based on the TARGET entity's
type, using the mapping below.

Only ASSOCIATED_WITH relations are repaired. Relations that already use a
specific type (CAUSES, AFFECTS, LEADS_TO, etc.) are preserved untouched.

Usage:
    repair_relations(chunk)  # mutates chunk.relations in-place
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chunking.schemas.models import MicroChunk

logger = logging.getLogger(__name__)


# ── Target entity type → replacement relation type ─────────────────────────────
# Keyed by LOWERCASED entity type (matching the canonical types in domain.py).
# If the target type isn't in this map, ASSOCIATED_WITH is left as-is.

_TARGET_TYPE_TO_RELATION: dict[str, str] = {
    # Symptoms
    "symptom":              "PRESENTS_WITH",

    # Diagnostic tests
    "diagnostic_test":      "DIAGNOSED_BY",

    # Treatments (broad)
    "treatment":            "TREATED_BY",
    "surgical_procedure":   "TREATED_BY",
    "medication":           "TREATED_BY",
    "rehabilitation":       "TREATED_BY",

    # Complications
    "complication":         "COMPLICATED_BY",
}


def repair_relations(chunk: "MicroChunk") -> None:
    """Replace generic ASSOCIATED_WITH relations with specific types.

    Mutates ``chunk.relations`` in-place.  Only relations whose current type is
    ``ASSOCIATED_WITH`` (case-insensitive) are considered; all others are left
    untouched.

    The replacement is inferred from the **target** entity's type:
      - Symptom            → PRESENTS_WITH
      - Diagnostic_Test    → DIAGNOSED_BY
      - Treatment / Surgical_Procedure / Medication / Rehabilitation → TREATED_BY
      - Complication       → COMPLICATED_BY

    When the target entity id cannot be resolved to an entity in the chunk, or
    the target's type doesn't have a mapping, the relation is left as
    ASSOCIATED_WITH.
    """
    # Build a lookup: entity_id → entity_type (lowercased)
    entity_type_by_id: dict[str, str] = {
        e.id: e.type.lower() for e in chunk.entities
    }

    repaired_count = 0
    for rel in chunk.relations:
        # Only touch the generic catch-all
        if rel.type.upper() != "ASSOCIATED_WITH":
            continue

        target_type = entity_type_by_id.get(rel.target)
        if target_type is None:
            continue  # target not resolvable — leave as-is

        new_type = _TARGET_TYPE_TO_RELATION.get(target_type)
        if new_type is not None:
            rel.type = new_type
            repaired_count += 1

    if repaired_count:
        logger.debug(
            "Relation repair: %d ASSOCIATED_WITH → specific in chunk %s",
            repaired_count, getattr(chunk, "chunk_id", "?"),
        )
