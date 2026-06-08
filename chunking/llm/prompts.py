"""Prompt assembly for the LLM providers.

The SYSTEM_PROMPT and the entity/relation vocabularies it references are defined
once in chunking/domain.py (the single file to edit for a new use case). This
module just re-exports SYSTEM_PROMPT and builds the per-call user message, so the
prompt can never drift out of sync with the schema's controlled vocabulary.
"""

from chunking.domain import SYSTEM_PROMPT  # noqa: F401  (re-exported for providers)


def build_user_content(schema_json: str, text: str) -> str:
    """Standard user message: the strict schema followed by the input text."""
    return f"""OUTPUT SCHEMA (STRICT):
{schema_json}

INPUT TEXT:
{text}
"""
