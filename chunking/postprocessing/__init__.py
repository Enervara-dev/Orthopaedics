"""
Post-processing stages applied after LLM extraction, before storage.

Pipeline order:
  1. Entity Override  — fix known entity types the LLM consistently gets wrong
  2. Relation Repair  — replace generic ASSOCIATED_WITH with specific relation types
  3. Canonicalization  — collapse aliases to a single canonical entity name
"""

from chunking.postprocessing.postprocessor import postprocess_chunk  # noqa: F401
