"""
session_memory.domain
───────────────────────
╔══════════════════════════════════════════════════════════════════════════════╗
║  THE PACKAGE TO EDIT FOR A NEW SPECIALTY / USE CASE (memory layer side).       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  All domain-specific knowledge for the session memory layer lives here. The    ║
║  extractor, summarizer, retriever and context builder are domain-agnostic and  ║
║  read FROM this package. The StructuredState field NAMES stay stable           ║
║  (models.py); only the patterns, risk policy, and render labels are tunable.   ║
║                                                                                ║
║    extraction_patterns.py → symptom/condition/drug/etc. patterns + demographics║
║    risk_rules.py          → critical-symptom escalation + risk ordering        ║
║    render_fields.py       → role + state-field labels used by the renderers    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from .extraction_patterns import (
    AGE_RE,
    ALLERGY_PATTERNS,
    CHRONIC_PATTERNS,
    CONDITION_PATTERNS,
    DRUG_PATTERNS,
    DURATION_RE,
    NAME_EXPLICIT_RES,
    NAME_SOFT_RE,
    NAME_STOPWORDS,
    SEVERITY_PATTERNS,
    SEX_NORMALISE,
    SEX_RE,
    SYMPTOM_PATTERNS,
    TRIGGER_PATTERNS,
)
from .risk_rules import CRITICAL_SYMPTOMS, HIGH_SIGNAL_SYMPTOMS, RISK_ORDER
from .render_fields import (
    ROLE_LABELS,
    STATE_RENDER_FIELDS,
    SUMMARY_RENDER_FIELDS,
)

__all__ = [
    # extraction patterns
    "SYMPTOM_PATTERNS", "CHRONIC_PATTERNS", "CONDITION_PATTERNS",
    "ALLERGY_PATTERNS", "DRUG_PATTERNS", "SEVERITY_PATTERNS",
    "TRIGGER_PATTERNS",
    "DURATION_RE", "AGE_RE", "SEX_RE", "SEX_NORMALISE",
    "NAME_EXPLICIT_RES", "NAME_SOFT_RE", "NAME_STOPWORDS",
    # risk rules
    "CRITICAL_SYMPTOMS", "HIGH_SIGNAL_SYMPTOMS", "RISK_ORDER",
    # render fields
    "ROLE_LABELS", "STATE_RENDER_FIELDS", "SUMMARY_RENDER_FIELDS",
]
