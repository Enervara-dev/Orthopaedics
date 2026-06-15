"""
graphrag.domain.clinical_policy
─────────────────────────────────
Structured clinical decision policy for the triage + answer layers — the
guideline-aligned rules that shape ranking, urgency, questioning, escalation,
and patient-facing tone.

⭐ EDIT FOR A NEW SPECIALTY/USE CASE. This is where triage behaviour is tuned:
which symptoms carry the most weight, which red flags force escalation, how many
clarifying questions are allowed, and the prose policies woven into the answer
prompt. The code (pipeline + answer_prompt) reads from here; nothing clinical is
hardcoded in the logic.
"""

from __future__ import annotations

import re

# ── Follow-up questioning budget ──────────────────────────────────────────────
# The triage layer may ask up to this many clarifying questions in one turn when
# severity or ambiguity warrants it (was effectively 1 before).
MAX_FOLLOWUP_QUESTIONS: int = 3

# ── Diagnostic loop termination ───────────────────────────────────────────────
# Terminal state for the diagnostic process. The session already carries a turn
# counter (SessionMemory.turn_count / WorkingMemory.turn_count); once the user
# has taken more than MAX_DIAGNOSTIC_TURNS turns — OR the gatekeeper stops needing
# follow-ups — the pipeline forces the intent to ASSESSMENT_READY so Stage 4
# concludes with a final assessment instead of looping on more questions.
MAX_DIAGNOSTIC_TURNS: int = 2
ASSESSMENT_READY_INTENT: str = "assessment_ready"

# Appended to the answer system prompt (Stage 4) when the diagnostic process is
# terminal — exact wording per the loop-prevention contract.
ASSESSMENT_READY_INSTRUCTION: str = (
    "CRITICAL INSTRUCTION: You have collected enough symptoms. Do NOT ask any "
    "further follow-up questions. Provide your final assessment and recommendations "
    "strictly based on the provided context."
)

# Appended when routing falls to NO_RETRIEVAL during a medical interaction — the
# model must wrap up from memory instead of defaulting to open-ended chat.
NO_RETRIEVAL_CONCLUDE_INSTRUCTION: str = (
    "CRITICAL INSTRUCTION: No new clinical information is being retrieved. "
    "Summarize the findings already gathered in this conversation, give your best "
    "assessment and clear next-step recommendations, and conclude — do NOT ask "
    "further follow-up questions or prolong the interaction."
)


def closure_directive(
    *,
    intent: str,
    needs_followup: bool,
    memory_only: bool,
    has_findings: bool,
) -> str | None:
    """
    Resolve the terminal/closure constraint to append at Stage 4, or None.

    - NO_RETRIEVAL during a medical interaction → conclude from memory.
    - assessment_ready (or the gatekeeper needing no more follow-ups) → final
      assessment, no further questions.
    Gated on `has_findings` so greetings / non-clinical turns are never forced
    to "conclude".
    """
    if not has_findings:
        return None
    if memory_only:
        return NO_RETRIEVAL_CONCLUDE_INSTRUCTION
    if intent == ASSESSMENT_READY_INTENT or not needs_followup:
        return ASSESSMENT_READY_INSTRUCTION
    return None

# ── High-signal symptoms (drive ranking + urgency) ────────────────────────────
# Prose, for prompt injection. Mirrors the canonical risk keys in the memory
# layer (session_memory/domain/risk_rules.py) but is phrased for the LLM.
HIGH_SIGNAL_SYMPTOMS_TEXT = (
    "severe pain, inability to bear weight, inability to move a limb, major deformity, "
    "neurovascular compromise, numbness, weakness, loss of sensation, open fractures, "
    "joint instability, suspected infection, and progressive neurological symptoms"
)

# ── Emergency red flags (respiratory / cardiopulmonary) ───────────────────────
# Prose list for the gatekeeper emergency section.
RED_FLAGS_TEXT = (
    "open fractures; absent pulses; cold, pale or blue limb; loss of sensation; "
    "new weakness or paralysis; severe deformity after injury; suspected compartment "
    "syndrome; inability to bear weight after trauma; suspected septic arthritis; "
    "high fever with a painful swollen joint; or progressive neurological deficits"
)

# Deterministic backstop — STRONG, present-tense red-flag phrases. The pipeline
# escalates to the emergency message when any of these match the user's message,
# even if the LLM gatekeeper missed it. Patterns are intentionally conservative
# (they require explicit severity) so ordinary complaints like "can't breathe
# properly" or a past "chest pain last week" do NOT trip them.
RED_FLAG_PATTERNS: dict[str, re.Pattern[str]] = {
    "open_fracture": re.compile(
        r"\b(open fracture|bone sticking out|bone exposed)\b",
        re.IGNORECASE,
    ),
    "neurovascular_compromise": re.compile(
        r"\b(no pulse|absent pulse|cold foot|cold hand|cold limb|blue foot|blue hand)\b",
        re.IGNORECASE,
    ),
    "loss_of_sensation": re.compile(
        r"\b(numbness|can't feel|loss of sensation|loss of feeling)\b",
        re.IGNORECASE,
    ),
    "paralysis_or_weakness": re.compile(
        r"\b(paralysis|can't move|unable to move|foot drop|wrist drop)\b",
        re.IGNORECASE,
    ),
    "compartment_syndrome": re.compile(
        r"\b(compartment syndrome|pain out of proportion)\b",
        re.IGNORECASE,
    ),
    "septic_joint": re.compile(
        r"\b(fever.*swollen joint|hot swollen joint|septic arthritis)\b",
        re.IGNORECASE,
    ),
    "major_trauma": re.compile(
        r"\b(severe deformity|major trauma|high impact accident)\b",
        re.IGNORECASE,
    ),
}

def detect_red_flags(text: str) -> list[str]:
    """Return the names of any emergency red flags present in `text`."""
    if not text:
        return []
    return [name for name, pat in RED_FLAG_PATTERNS.items() if pat.search(text)]


# ── Answer-layer policy blocks (woven into the answer system prompt) ───────────

DIFFERENTIAL_POLICY = f"""CLINICAL REASONING & DIFFERENTIAL
- Lead with the 1–3 MOST CLINICALLY LIKELY explanations for THIS patient, each with a \
one-line rationale tied to their specific features. Do not enumerate long lists of \
low-probability possibilities.
- Weight high-signal features heavily when ranking and when judging urgency: \
{HIGH_SIGNAL_SYMPTOMS_TEXT}.
- Do NOT surface rare or exotic conditions unless the symptoms strongly support them \
or the patient explicitly asks. Mention a "can't-miss" serious cause only when its \
red flags are plausibly present — and then say what would confirm or exclude it.
- Synthesise the retrieved context into coherent clinical reasoning (why these causes, \
what links the findings) — do not just summarise the source text."""

UNCERTAINTY_POLICY = """HANDLING UNCERTAINTY
- If the picture is uncertain but LOW risk: say so plainly, give sensible self-care and \
clear "see a clinician if…" criteria, and offer to narrow it down with one or two questions.
- If the picture is uncertain AND any severe/high-signal feature is present: do NOT \
reassure. Err toward caution — recommend timely or urgent assessment and state the \
specific red flags that mean "seek care now"."""

QUESTIONING_POLICY = f"""TRIAGE QUESTIONING
- Actively ask clinically important orthopaedic questions when symptoms are ambiguous.
- Focus on mechanism of injury, onset, duration, location, severity, deformity, swelling,
  weight-bearing ability, range of motion, neurological symptoms, vascular symptoms,
  prior injuries, and relevant imaging.
- Ask only what changes management. Ask at most {MAX_FOLLOWUP_QUESTIONS} questions,
  fewest possible, the most decision-relevant first. If enough information exists,
  do not ask additional questions."""

SAFEGUARDS = """GENERATION SAFEGUARDS
- No FALSE REASSURANCE: never imply something is harmless when red flags or high-signal \
features are present.
- No PANIC: stay calm and measured; avoid alarming language for low-risk situations.
- No DIAGNOSTIC DUMPING: don't overwhelm with exhaustive lists, jargon, or encyclopedic \
detail. Keep it concise and readable.
- ALWAYS end with clear, concrete next steps (self-care, what to monitor, and exactly \
when/where to seek care)."""
