"""
graphrag.domain.entity_rules
──────────────────────────────
Heuristics for the entity post-processing stage.

⭐ ORTHOPAEDICS VERSION
"""

# Regex that pulls candidate medication names out of a query.
MEDICATION_NAME_PATTERN: str = r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b"

# Capitalised tokens that match MEDICATION_NAME_PATTERN but are NOT medications.
MEDICATION_NAME_STOPWORDS: set[str] = {
    "What",
    "When",
    "Where",
    "How",
    "Why",
    "Can",
    "Does",
    "The",
    "This",
    "That",
    "Medicine",
    "Medication",
    "Patient",
    "Doctor",
    "Hospital",
    "Fracture",
    "Sprain",
    "Ligament",
    "Tendon",
    "Joint",
    "Bone",
    "Hip",
    "Knee",
    "Shoulder",
    "Elbow",
    "Spine",
}