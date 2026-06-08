"""
session_memory.domain.extraction_patterns
───────────────────────────────────────────
Regex/keyword patterns the heuristic state extractor uses to pull medical
context out of a user message.

⭐ EDIT FOR A NEW SPECIALTY/USE CASE. The extractor logic
(`state_extractor.py`) is domain-agnostic — it iterates EXTRACTION_PATTERNS and
applies the demographic/name patterns below. To retarget, change the patterns
here; leave the code alone.

Each pattern dict maps a canonical term → list of regex alternatives. The first
alternative that matches (case-insensitive) records the canonical term.
"""

from __future__ import annotations

import re

# ── Clinical entity patterns ──────────────────────────────────────────────────

SYMPTOM_PATTERNS: dict[str, list[str]] = {
    "fever":              [r"\bfever\b", r"\bhigh temperature\b", r"\btemperature\b"],
    "chills":             [r"\bchill(s|ing)?\b", r"\bshiver(ing)?\b"],
    "sore_throat":        [r"\bsore throat\b", r"\bthroat pain\b", r"\bthroat ache\b"],
    "cough":              [r"\bcough(ing)?\b"],
    "shortness_of_breath":[r"\bshortness of breath\b", r"\bbreathing difficult\b", r"\bcan'?t breathe\b",
                           r"\bbreathless(ness)?\b", r"\bout of breath\b", r"\bdyspn\w*\b"],
    "chest_pain":         [r"\bchest pain\b", r"\bchest tightness\b", r"\bchest heaviness\b", r"\btight chest\b"],
    "headache":           [r"\bheadache\b", r"\bhead pain\b"],
    "fatigue":            [r"\bfatigue\b", r"\btired\b", r"\bexhausted\b"],
    "nausea":             [r"\bnausea\b", r"\bfeeling sick\b"],
    "dizziness":          [r"\bdizzy\b", r"\bdizziness\b"],
    # ── Respiratory-specific (pulmonology) ──
    "wheezing":           [r"\bwheez(e|es|ing|y)\b"],
    "haemoptysis":        [r"\bcough(ing)? up blood\b", r"\bcoughing blood\b",
                           r"\bblood in (the )?(sputum|phlegm|mucus)\b",
                           r"\b(haemoptysis|hemoptysis)\b", r"\bspitting blood\b"],
    "cyanosis":           [r"\b(blue|bluish|grey|gray|purple) (lips?|fingers?|fingertips?|face|skin|nails?)\b",
                           r"\blips? (are )?(turning )?(blue|bluish)\b"],
    "tachypnea":          [r"\b(fast|rapid|quick) breathing\b", r"\bbreathing (fast|rapidly|quickly)\b",
                           r"\btachypn\w*\b"],
    "sputum":             [r"\b(sputum|phlegm|mucus|catarrh)\b"],
    "nasal_congestion":   [r"\b(blocked|stuffy|congested|runny) nose\b", r"\bnasal congestion\b",
                           r"\bnose (is )?(blocked|stuffy|running|runny)\b"],
    "sinus":              [r"\bsinus(es|itis)?\b"],
    "sneezing":           [r"\bsneez(e|es|ing)\b"],
    "hoarseness":         [r"\bhoarse(ness)?\b", r"\blost my voice\b"],
    "night_sweats":       [r"\bnight sweats?\b"],
    "confusion":          [r"\bconfus(ed|ion)\b", r"\bdisorient\w+\b"],
    "syncope":            [r"\bfaint(ed|ing)?\b", r"\bpassed out\b", r"\bblack(ed)? out\b",
                           r"\bcollaps(e|ed)\b", r"\blost consciousness\b"],
    "severe_weakness":    [r"\b(severe|extreme) weakness\b", r"\bvery weak\b", r"\bcan'?t stand\b"],
}

# Trigger / pattern recognition — when does the symptom occur or worsen? Captured
# into StructuredState.triggers for cross-turn triage continuity.
TRIGGER_PATTERNS: dict[str, list[str]] = {
    "morning":      [r"\bin the morning(s)?\b", r"\bmorning(s)?\b", r"\bwhen i wake\b"],
    "night":        [r"\bat night\b", r"\bnight ?time\b", r"\bwhen i lie down\b", r"\blying down\b"],
    "exertion":     [r"\b(exercise|exertion|exerting|physical activity)\b",
                     r"\bwhen i (walk|exert|run|climb)\b", r"\bclimbing stairs\b", r"\bwalking\b"],
    "cold_air":     [r"\bcold air\b", r"\bin the cold\b", r"\bcold weather\b"],
    "allergen":     [r"\b(dust|pollen|pet|animal|mould|mold)\b", r"\ballerg\w*\b"],
    "after_eating": [r"\bafter (eating|meals?|food)\b"],
}

# Distinguishing between acute conditions and chronic conditions
CHRONIC_PATTERNS: dict[str, list[str]] = {
    "diabetes":           [r"\bdiabetes\b", r"\bdiabetic\b"],
    "hypertension":       [r"\bhypertension\b", r"\bhigh blood pressure\b", r"\bhigh bp\b"],
    "asthma":             [r"\basthma\b", r"\basthmatic\b"],
    "heart_disease":      [r"\bheart disease\b", r"\bcardiac issue\b"],
    "thyroid":            [r"\bthyroid\b"],
    "arthritis":          [r"\barthritis\b", r"\bjoint pain\b"],
}

CONDITION_PATTERNS: dict[str, list[str]] = {
    "flu":                [r"\bflu\b", r"\binfluenza\b"],
    "strep_throat":       [r"\bstrep throat\b"],
    "covid":              [r"\bcovid\b", r"\bcoronavirus\b"],
    "infection":          [r"\binfection\b", r"\binfected\b"],
    "migraine":           [r"\bmigraine\b"],
}

ALLERGY_PATTERNS: dict[str, list[str]] = {
    "penicillin":    [r"\ballergic to penicillin\b", r"\bpenicillin allergy\b"],
    "pollen":        [r"\bpollen\b", r"\bhay fever\b"],
    "dust":          [r"\bdust allergy\b", r"\ballergic to dust\b"],
    "peanuts":       [r"\bpeanut allergy\b", r"\ballergic to peanuts\b"],
    "shellfish":     [r"\bshellfish allergy\b"],
}

DRUG_PATTERNS: dict[str, list[str]] = {
    "paracetamol":   [r"\bparacetamol\b", r"\bacetaminophen\b", r"\btylenol\b"],
    "ibuprofen":     [r"\bibuprofen\b", r"\badvil\b", r"\bnurofen\b"],
    "aspirin":       [r"\baspirin\b"],
    "insulin":       [r"\binsulin\b"],
    "amoxicillin":   [r"\bamoxicillin\b"],
}

SEVERITY_PATTERNS: dict[str, list[str]] = {
    "mild":     [r"\bmild\b", r"\bslight\b"],
    "moderate": [r"\bmoderate\b", r"\bmedium\b"],
    "severe":   [r"\bsevere\b", r"\bextreme\b", r"\bintense\b", r"\bvery bad\b"],
}

# ── Demographics ──────────────────────────────────────────────────────────────

DURATION_RE = re.compile(
    r"(?:for|since|over|past|last)\s+"
    r"(\d+\s+(?:second|minute|hour|day|week|month|year)s?|yesterday|this morning)",
    re.IGNORECASE,
)

AGE_RE  = re.compile(r"\b(\d{1,3})\s*(?:year(?:s)?\s*old|y\.?o\.?)\b", re.IGNORECASE)
SEX_RE  = re.compile(r"\b(male|female|man|woman)\b", re.IGNORECASE)

SEX_NORMALISE = {"man": "male", "woman": "female"}

# ── Name extraction ───────────────────────────────────────────────────────────
# We pull a first name when the patient explicitly introduces themselves. The
# answer prompt uses it to address the user naturally instead of as "patient".
#
# High-confidence patterns first — these are explicit declarations and almost
# never produce false positives.
NAME_EXPLICIT_RES: list[re.Pattern[str]] = [
    re.compile(r"\bmy name is ([A-Za-z][A-Za-z'\-]{1,30})\b", re.IGNORECASE),
    re.compile(r"\bthe name'?s ([A-Za-z][A-Za-z'\-]{1,30})\b", re.IGNORECASE),
    re.compile(r"\bname'?s ([A-Za-z][A-Za-z'\-]{1,30})\b", re.IGNORECASE),
    re.compile(r"\bcall me ([A-Za-z][A-Za-z'\-]{1,30})\b", re.IGNORECASE),
    re.compile(r"\bthis is ([A-Z][a-zA-Z'\-]{1,30})(?:\s+speaking|\s+here|\s*[,.])"),
    re.compile(r"\bi go by ([A-Za-z][A-Za-z'\-]{1,30})\b", re.IGNORECASE),
]

# Lower-confidence: "I am X" / "I'm X". Only trust if X looks like a name —
# capitalised in the original text AND not a common adjective / state word.
NAME_SOFT_RE = re.compile(r"\b[Ii]\s*'?\s*[am]{1,2}\s+([A-Z][a-zA-Z'\-]{1,30})\b")

# Common words that can follow "I'm" / "I am" but are NOT names. Lowercased.
NAME_STOPWORDS: frozenset[str] = frozenset({
    # states / feelings
    "sick", "tired", "fine", "ok", "okay", "good", "bad", "well", "great",
    "happy", "sad", "worried", "scared", "confused", "anxious", "depressed",
    "stressed", "exhausted", "hungry", "thirsty", "dizzy", "nauseous", "dying",
    "fasting", "bleeding", "burning", "shaking", "freezing",
    # statuses
    "married", "single", "pregnant", "diabetic", "allergic", "asthmatic",
    "hypertensive", "vegetarian", "vegan", "lost", "ready", "back", "done",
    "late", "early", "here", "there", "home", "outside", "indoors",
    # progressive verbs after "I'm"
    "having", "feeling", "going", "trying", "looking", "doing", "taking",
    "thinking", "wondering", "asking", "calling", "writing", "experiencing",
    "suffering", "noticing", "starting", "ending", "drinking", "eating",
    # other common
    "afraid", "unsure", "unable", "old", "young", "new", "sorry", "sure",
    "really", "always", "never", "still", "just", "also", "very",
})
