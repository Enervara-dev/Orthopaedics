"""
graphrag.domain.entity_rules
──────────────────────────────
Heuristics for the entity post-processing stage.

⭐ EDIT FOR A NEW SPECIALTY/USE CASE. Today this only governs the drug-pair
boost used by `drug_interaction` queries (see entity_processor.py); a new
domain may swap this for its own salient-pair logic.
"""

# Regex that pulls candidate drug-name tokens out of a query (capitalised
# single- or two-word tokens).
DRUG_NAME_PATTERN: str = r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b"

# Capitalised tokens that match DRUG_NAME_PATTERN but are NOT drug names.
DRUG_NAME_STOPWORDS: set[str] = {
    "What", "When", "Where", "How", "Why", "Can", "Does", "The",
    "This", "That", "Drug", "Medication", "Medicine", "Patient",
}
