"""
session_memory.domain.risk_rules
──────────────────────────────────
Domain risk policy for the session state extractor.

⭐ EDIT FOR A NEW SPECIALTY/USE CASE. CRITICAL_SYMPTOMS names the extracted
symptom terms (keys of extraction_patterns.SYMPTOM_PATTERNS) that auto-escalate
the session to CRITICAL risk when no explicit risk is supplied by the
gatekeeper. RISK_ORDER defines the severity ordering used to ensure risk only
ever escalates, never downgrades.
"""

from __future__ import annotations

# Symptom terms (canonical keys from extraction_patterns.SYMPTOM_PATTERNS) that,
# on their own, escalate risk to "critical".
CRITICAL_SYMPTOMS: frozenset[str] = frozenset({
    "inability_to_bear_weight",
    "foot_drop",
    "wrist_drop",
})

# High-signal symptoms that escalate risk to at least "high" (strong influence on
# ranking + urgency) but are not, alone, an automatic emergency.
HIGH_SIGNAL_SYMPTOMS: frozenset[str] = frozenset({
    "pain",
    "swelling",
    "deformity",
    "instability",
    "limited_range_of_motion",
    "numbness",
    "tingling",
    "weakness",
    "joint_locking",
    "fever",
})

# Ordered low → high. Used by merge_state to only escalate risk.
RISK_ORDER: list[str] = [
    "none",
    "low",
    "medium",
    "high",
    "critical",
]

