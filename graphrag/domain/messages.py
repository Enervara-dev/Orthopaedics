"""
graphrag.domain.messages
──────────────────────────
User-facing canned responses emitted directly by the pipeline (no LLM call).

⭐ EDIT FOR A NEW SPECIALTY/USE CASE. These are returned when the gatekeeper
refuses a non-medical query or redirects a detected emergency.
"""

# Returned when the gatekeeper classifies the query as out-of-domain.
REFUSAL_MESSAGE: str = (
    "❌ I can only answer healthcare-related questions. "
    "Please ask a medical question."
)

# Returned when a query is medical but falls below the pulmonology relevance
# threshold (see vocabulary.PULMONOLOGY_RELEVANCE_THRESHOLD).
OUT_OF_SCOPE_MESSAGE: str = (
    "🦴 I'm focused on orthopaedics and musculoskeletal health, so I can't help "
    "with that one. Please ask about bones, joints, muscles, ligaments, tendons, "
    "fractures, injuries, arthritis, spine conditions, or rehabilitation."
)

# Returned when the gatekeeper detects an emergency red-flag.
EMERGENCY_MESSAGE: str = (
    "🚨 EMERGENCY: Your symptoms may indicate a serious orthopaedic emergency. "
    "Please seek immediate medical attention, call emergency services (112 / 911), "
    "or go to the nearest emergency department."
)
