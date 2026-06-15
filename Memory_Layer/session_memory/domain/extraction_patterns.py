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
    "pain": [
        r"\bpain\b",
        r"\baching\b",
        r"\bache\b",
        r"\bsore\b"
    ],

    "swelling": [
        r"\bswelling\b",
        r"\bswollen\b"
    ],

    "stiffness": [
        r"\bstiff(ness)?\b"
    ],

    "deformity": [
        r"\bdeformity\b",
        r"\bdeformed\b"
    ],

    "instability": [
        r"\binstability\b",
        r"\bgiving way\b",
        r"\bbuckling\b"
    ],

    "limited_range_of_motion": [
        r"\breduced movement\b",
        r"\blimited movement\b",
        r"\blimited range of motion\b",
        r"\bcan't move\b"
    ],

    "inability_to_bear_weight": [
        r"\bcan't walk\b",
        r"\bcannot walk\b",
        r"\bunable to walk\b",
        r"\bcan't bear weight\b",
        r"\bunable to bear weight\b"
    ],

    "numbness": [
        r"\bnumb(ness)?\b"
    ],

    "tingling": [
        r"\btingling\b",
        r"\bpins and needles\b"
    ],

    "weakness": [
        r"\bweak(ness)?\b"
    ],

    "wrist_drop": [
        r"\bwrist drop\b"
    ],

    "foot_drop": [
        r"\bfoot drop\b"
    ],

    "joint_locking": [
        r"\blocking\b",
        r"\blocked knee\b"
    ],

    "joint_clicking": [
        r"\bclicking\b",
        r"\bpopping\b"
    ],

    "fever": [
        r"\bfever\b"
    ]
}

# Trigger / pattern recognition — when does the symptom occur or worsen? Captured
# into StructuredState.triggers for cross-turn triage continuity.
TRIGGER_PATTERNS: dict[str, list[str]] = {
    "walking": [
        r"\bwalking\b",
        r"\bwhile walking\b"
    ],

    "running": [
        r"\brunning\b"
    ],

    "stairs": [
        r"\bclimbing stairs\b",
        r"\bgoing upstairs\b",
        r"\bgoing downstairs\b"
    ],

    "sports": [
        r"\bfootball\b",
        r"\bcricket\b",
        r"\bbasketball\b",
        r"\bsports\b"
    ],

    "lifting": [
        r"\blifting\b",
        r"\bheavy lifting\b"
    ],

    "fall": [
        r"\bfall\b",
        r"\bfell\b",
        r"\bslipped\b"
    ],

    "twisting": [
        r"\btwist(ed|ing)?\b"
    ],

    "weight_bearing": [
        r"\bstanding\b",
        r"\bweight bearing\b"
    ]
}

# Distinguishing between acute conditions and chronic conditions
CHRONIC_PATTERNS: dict[str, list[str]] = {
    "osteoarthritis": [
        r"\bosteoarthritis\b",
        r"\boa\b"
    ],

    "rheumatoid_arthritis": [
        r"\brheumatoid arthritis\b",
        r"\bra\b"
    ],

    "osteoporosis": [
        r"\bosteoporosis\b"
    ],

    "scoliosis": [
        r"\bscoliosis\b"
    ],

    "kyphosis": [
        r"\bkyphosis\b"
    ],

    "chronic_back_pain": [
        r"\bchronic back pain\b"
    ],

    "degenerative_disc_disease": [
        r"\bdegenerative disc disease\b"
    ]
}

CONDITION_PATTERNS: dict[str, list[str]] = {
    "fracture": [
        r"\bfracture\b",
        r"\bbroken bone\b"
    ],

    "dislocation": [
        r"\bdislocation\b",
        r"\bdislocated\b"
    ],

    "sprain": [
        r"\bsprain\b"
    ],

    "strain": [
        r"\bstrain\b"
    ],

    "acl_tear": [
        r"\bacl tear\b",
        r"\banterior cruciate ligament tear\b"
    ],

    "meniscus_tear": [
        r"\bmeniscus tear\b",
        r"\bmeniscal tear\b"
    ],

    "rotator_cuff_tear": [
        r"\brotator cuff tear\b"
    ],

    "osteomyelitis": [
        r"\bosteomyelitis\b"
    ],

    "septic_arthritis": [
        r"\bseptic arthritis\b"
    ],

    "carpal_tunnel_syndrome": [
        r"\bcarpal tunnel\b"
    ]
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
