"""
session_memory.domain.render_fields
──────────────────────────────────────
How the StructuredState clinical fields are LABELLED when rendered into prompt
text. Shared by the three renderers so labels live in one place:

  - context_builder._state_lines          → STATE_RENDER_FIELDS
  - retriever.format_working_memory        → STATE_RENDER_FIELDS
  - summarizer._state_context_lines        → SUMMARY_RENDER_FIELDS

⭐ EDIT FOR A NEW SPECIALTY/USE CASE. Changing a label here changes how that
field appears in every prompt. The StructuredState field NAMES (left column)
are kept stable — see models.StructuredState; only their human labels are
domain-tunable.

Note: STATE_RENDER_FIELDS and SUMMARY_RENDER_FIELDS intentionally use different
label wording (the summary block reads more prose-y, e.g. "Reported symptoms"
vs "Symptoms"). They are kept separate to preserve each renderer's output.
"""

from __future__ import annotations

# Speaker labels for dialogue rendering.
ROLE_LABELS: dict[str, str] = {
    "user": "Patient",
    "assistant": "Assistant",
    "system": "System",
}

# (state_attr, label) — clinical list fields, in display order.
# Used by the working-memory and final-context renderers.
STATE_RENDER_FIELDS: list[tuple[str, str]] = [
    ("symptoms",             "Symptoms"),
    ("conditions",           "Conditions"),
    ("chronic_conditions",   "Chronic conditions"),
    ("drugs",                "Medications"),
    ("allergies",            "Allergies"),
    ("severity",             "Severity"),
    ("duration",             "Duration"),
    ("triggers",             "Triggers / patterns"),
    ("previous_concerns",    "Previous concerns"),
    ("follow_up_references", "Follow-up references"),
]

# (state_attr, label) — clinical list fields as worded inside the rolling
# summary block. Order preserved from the original summarizer.
SUMMARY_RENDER_FIELDS: list[tuple[str, str]] = [
    ("symptoms",           "Reported symptoms"),
    ("conditions",         "Known conditions"),
    ("chronic_conditions", "Chronic conditions"),
    ("drugs",              "Medications mentioned"),
    ("allergies",          "Allergies mentioned"),
    ("severity",           "Severity descriptors"),
    ("duration",           "Duration"),
    ("triggers",           "Triggers / patterns"),
    ("previous_concerns",  "Previous concerns"),
]
