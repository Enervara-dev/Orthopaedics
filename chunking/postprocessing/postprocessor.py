"""
Post-processing orchestrator.

Applies all three stages in order to a validated MicroChunk:
  1. Entity Override  — fix known entity types
  2. Relation Repair  — replace generic ASSOCIATED_WITH
  3. Canonicalization  — normalize aliases to canonical names

This is the single entry-point the pipeline calls. Each stage mutates the
chunk in-place and is idempotent (safe to re-run).
"""

import logging
from typing import TYPE_CHECKING

from chunking.postprocessing.entity_overrides import ENTITY_OVERRIDES
from chunking.postprocessing.relation_repair import repair_relations
from chunking.postprocessing.canonicalization import canonicalize_entities

if TYPE_CHECKING:
    from chunking.schemas.models import MicroChunk

logger = logging.getLogger(__name__)


def _apply_entity_overrides(chunk: "MicroChunk") -> None:
    """Stage 1: override entity types for known orthopaedic concepts.

    For each entity in the chunk, if the lowercased name exists in
    :data:`ENTITY_OVERRIDES`, replace its ``type`` with the override value.
    """
    overridden = 0
    for entity in chunk.entities:
        override_type = ENTITY_OVERRIDES.get(entity.name.lower())
        if override_type is not None and entity.type != override_type:
            logger.debug(
                "Entity override: '%s' type %s → %s",
                entity.name, entity.type, override_type,
            )
            entity.type = override_type
            overridden += 1
    if overridden:
        logger.debug(
            "Entity override: corrected %d entities in chunk %s",
            overridden, getattr(chunk, "chunk_id", "?"),
        )


def postprocess_chunk(chunk: "MicroChunk") -> "MicroChunk":
    """Apply all post-processing stages to a single chunk.

    Stages run in order — entity overrides must precede relation repair (which
    reads entity types), and canonicalization runs last (it may merge entities,
    changing ids).

    Args:
        chunk: A validated MicroChunk instance (already through Pydantic).

    Returns:
        The same chunk, mutated in-place. Returned for fluent chaining.
    """
    # Stage 1: fix mistyped entities
    _apply_entity_overrides(chunk)

    # Stage 2: replace generic ASSOCIATED_WITH with specific relation types
    repair_relations(chunk)

    # Stage 3: normalize aliases to canonical names
    canonicalize_entities(chunk)

    return chunk
