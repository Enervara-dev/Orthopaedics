"""
state_extractor.py
──────────────────
Heuristic state extractor for the Enervera memory layer.

Pulls medical context (symptoms, conditions, drugs, allergies, demographics,
risk) out of a user message. The patterns, name rules, and risk policy are
domain-specific and live in `session_memory/domain/` — this module is the
domain-agnostic extraction LOGIC that applies them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .domain import (
    AGE_RE,
    ALLERGY_PATTERNS,
    CHRONIC_PATTERNS,
    CONDITION_PATTERNS,
    CRITICAL_SYMPTOMS,
    DRUG_PATTERNS,
    DURATION_RE,
    HIGH_SIGNAL_SYMPTOMS,
    NAME_EXPLICIT_RES,
    NAME_SOFT_RE,
    NAME_STOPWORDS,
    RISK_ORDER,
    SEVERITY_PATTERNS,
    SEX_NORMALISE,
    SEX_RE,
    SYMPTOM_PATTERNS,
    TRIGGER_PATTERNS,
)
from .models import Message, RiskLevel, Role, SessionMemory, StructuredState


# ============================================================================
# Name extraction
# ============================================================================

def _extract_name(text: str) -> str | None:
    """
    Return the first sensible name found in `text`, or None.

    Prefers explicit declarations ("my name is X", "call me X") before the
    softer "I'm X" pattern. Soft matches are filtered through a stopword list
    so phrases like "I'm sick" / "I'm Diabetic" never produce a "name".
    """
    for pat in NAME_EXPLICIT_RES:
        m = pat.search(text)
        if m:
            return _format_name(m.group(1))

    m = NAME_SOFT_RE.search(text)
    if m:
        candidate = m.group(1)
        if candidate.lower() not in NAME_STOPWORDS:
            return _format_name(candidate)
    return None


def _format_name(raw: str) -> str:
    """Normalise to Title-Case, preserving internal apostrophes and hyphens."""
    return "-".join(part[:1].upper() + part[1:].lower() for part in raw.split("-"))

# ============================================================================
# Helpers
# ============================================================================

def _match_patterns(text: str, pattern_dict: dict[str, list[str]]) -> list[str]:
    found: list[str] = []
    lower = text.lower()
    for name, patterns in pattern_dict.items():
        for pat in patterns:
            if re.search(pat, lower):
                found.append(name)
                break
    return found

def _extract_demographics(text: str) -> dict[str, Any]:
    demo: dict[str, Any] = {}
    age_m = AGE_RE.search(text)
    if age_m: demo["age"] = int(age_m.group(1))
    sex_m = SEX_RE.search(text)
    if sex_m:
        val = sex_m.group(1).lower()
        demo["sex"] = SEX_NORMALISE.get(val, val)
    name = _extract_name(text)
    if name:
        demo["name"] = name
    return demo

def _deduplicate(lst: list[str]) -> list[str]:
    seen: set[str] = set()
    return [x for x in lst if not (x in seen or seen.add(x))]

def _risk_value(risk: Any) -> str:
    return risk.value if hasattr(risk, "value") else str(risk)

def _max_risk(a: RiskLevel, b: RiskLevel) -> RiskLevel:
    """Return whichever RiskLevel is higher on the RISK_ORDER scale."""
    return a if RISK_ORDER.index(_risk_value(a)) >= RISK_ORDER.index(_risk_value(b)) else b

# ============================================================================
# State Extraction Logic
# ============================================================================

@dataclass
class RawEntities:
    symptoms:           list[str] = field(default_factory=list)
    conditions:         list[str] = field(default_factory=list)
    chronic_conditions: list[str] = field(default_factory=list)
    allergies:          list[str] = field(default_factory=list)
    drugs:              list[str] = field(default_factory=list)
    severity:           list[str] = field(default_factory=list)
    duration:           list[str] = field(default_factory=list)
    triggers:           list[str] = field(default_factory=list)
    demographics:       dict[str, Any] = field(default_factory=dict)
    preferences:        dict[str, Any] = field(default_factory=dict)
    risk_level:         RiskLevel = RiskLevel.NONE

    def all_named_entities(self) -> list[str]:
        """Flat list of all recognised medical terms for discussed_entities."""
        return self.symptoms + self.conditions + self.drugs + self.chronic_conditions + self.allergies

def extract_entities(text: str, message: Message | None = None) -> RawEntities:
    symptoms   = _match_patterns(text, SYMPTOM_PATTERNS)
    conditions = _match_patterns(text, CONDITION_PATTERNS)
    chronic    = _match_patterns(text, CHRONIC_PATTERNS)
    allergies  = _match_patterns(text, ALLERGY_PATTERNS)
    drugs      = _match_patterns(text, DRUG_PATTERNS)
    severity   = _match_patterns(text, SEVERITY_PATTERNS)

    duration = [m.group(0).strip() for m in DURATION_RE.finditer(text)]
    triggers = _match_patterns(text, TRIGGER_PATTERNS)
    demo = _extract_demographics(text)

    # Symptom-weighted risk: high-signal symptoms strengthen urgency. Take the
    # higher of the gatekeeper-supplied risk and the symptom-derived risk so
    # severe features always escalate (risk only ever moves up downstream).
    symptom_set = set(symptoms)
    if symptom_set & CRITICAL_SYMPTOMS:
        symptom_risk = RiskLevel.CRITICAL
    elif symptom_set & HIGH_SIGNAL_SYMPTOMS:
        symptom_risk = RiskLevel.HIGH
    else:
        symptom_risk = RiskLevel.NONE

    msg_risk = message.risk_level if (message and message.risk_level) else RiskLevel.NONE
    risk = _max_risk(msg_risk, symptom_risk)

    return RawEntities(
        symptoms=symptoms,
        conditions=conditions,
        chronic_conditions=chronic,
        allergies=allergies,
        drugs=drugs,
        severity=severity,
        duration=duration,
        triggers=triggers,
        demographics=demo,
        risk_level=risk
    )

def update_preferences(state: StructuredState, patch: RawEntities) -> dict[str, Any]:
    merged = dict(state.preferences or {})
    for key, val in patch.preferences.items():
        merged[key] = val
    return merged

def merge_state(existing: StructuredState, patch: RawEntities) -> StructuredState:
    data = existing.model_copy(deep=True)

    data.symptoms = _deduplicate(data.symptoms + patch.symptoms)
    data.conditions = _deduplicate(data.conditions + patch.conditions)
    data.chronic_conditions = _deduplicate(data.chronic_conditions + patch.chronic_conditions)
    data.allergies = _deduplicate(data.allergies + patch.allergies)
    data.drugs = _deduplicate(data.drugs + patch.drugs)
    data.severity = _deduplicate(data.severity + patch.severity)
    data.duration = _deduplicate(data.duration + patch.duration)
    data.triggers = _deduplicate(data.triggers + patch.triggers)

    for k, v in patch.demographics.items():
        data.demographics[k] = v

    # Preferences merge
    data.preferences = update_preferences(data, patch)

    # Maintain a history of concerns for context-aware RAG
    if patch.symptoms:
        data.previous_concerns = _deduplicate(data.previous_concerns + patch.symptoms)

    # Risk only escalates — take the higher of existing and incoming.
    data.risk_level = _max_risk(data.risk_level, patch.risk_level)

    data.discussed_entities = _deduplicate(
        data.discussed_entities + patch.all_named_entities()
    )

    return data


def extract_state(session: SessionMemory, message: Message) -> StructuredState:
    if message.role != Role.USER:
        return session.state

    raw = extract_entities(message.content, message)
    updated = merge_state(session.state, raw)

    if message.query_type:
        updated.active_task = message.query_type
        updated.last_intent = message.query_type

    return updated
