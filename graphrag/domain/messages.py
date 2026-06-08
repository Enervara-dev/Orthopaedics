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
    "🫁 I'm focused on pulmonology and respiratory medicine, so I can't help "
    "with that one. Please ask about a lung or breathing-related concern "
    "(e.g. cough, breathlessness, asthma, COPD, chest infections)."
)

# Returned when the gatekeeper detects an emergency red-flag.
EMERGENCY_MESSAGE: str = (
    "🚨 EMERGENCY: Your symptoms sound like a serious emergency. "
    "Please call emergency services (112 / 911) immediately or go to the "
    "nearest hospital."
)
