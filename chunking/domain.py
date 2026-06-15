"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  THE ONE FILE TO EDIT FOR A NEW USE CASE.                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Everything domain-specific lives here: entity/relation vocabularies, the      ║
║  specialty taxonomy, the extraction prompt, segmentation patterns, and the     ║
║  validation thresholds. The rest of the pipeline (loaders, cleaner, LLM        ║
║  plumbing, storage, the MicroChunk shape) is domain-agnostic. Edit the values  ║
║  below to retarget the chunker — nothing else.                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import re

# ── 1. Entity vocabulary ──────────────────────────────────────────────────────
# Precise types so the graph can reason: a lab value is NOT a disease, a metabolic
# state is NOT a symptom. Anything emitted outside this set is mapped via
# ENTITY_TYPE_SYNONYMS, else coerced to ENTITY_TYPE_FALLBACK.
ENTITY_TYPES = [
    "Anatomical_Structure",
    "Condition",
    "Symptom",
    "Diagnostic_Test",
    "Treatment",
    "Surgical_Procedure",
    "Implant",
    "Medication",
    "Rehabilitation",
    "Risk_Factor",
    "Complication",
    "Outcome"
]
ENTITY_TYPE_SET = set(ENTITY_TYPES)

# Common synonyms / mislabels the model emits → canonical type.
ENTITY_TYPE_SYNONYMS = {
    "disease": "condition", "disorder": "condition", "injury": "condition",
    "sign": "symptom", "clinical_finding": "symptom",
    "test": "diagnostic_test", "investigation": "diagnostic_test", "imaging": "diagnostic_test",
    "therapy": "treatment", "management": "treatment",
    "surgery": "surgical_procedure", "operation": "surgical_procedure",
    "prosthesis": "implant",
    "drug": "medication", "medicine": "medication",
    "physiotherapy": "rehabilitation", "physical_therapy": "rehabilitation", "rehab": "rehabilitation",
    "predisposing_factor": "risk_factor",
    "adverse_event": "complication",
    "prognosis": "outcome"
}
ENTITY_TYPE_FALLBACK = "Condition"

# ── 2. Relation vocabulary + qualifiers ───────────────────────────────────────
RELATION_TYPES = [
    "AFFECTS",
    "CAUSES",
    "ASSOCIATED_WITH",
    "PRESENTS_WITH",
    "DIAGNOSED_BY",
    "TREATED_BY",
    "MANAGED_BY",
    "REQUIRES",
    "USES_IMPLANT",
    "FOLLOWED_BY",
    "INCREASES_RISK_OF",
    "LEADS_TO",
    "COMPLICATED_BY",
    "RESULTS_IN"
]
RELATION_TYPE_SET = set(RELATION_TYPES)
RELATION_TYPE_FALLBACK = "ASSOCIATED_WITH"

# Allowed relation qualifier keys (graph edge properties). The clinical AXIS — esp.
# onset/speed — must survive as an edge property, not be flattened away.
RELATION_QUALIFIER_KEYS = ["severity", "grade", "location", "laterality", "age_group", "outcome"]
ONSET_VALUES = ["instantaneous", "acute", "subacute", "chronic"]

# ── 3. Specialty taxonomy ─────────────────────────────────────────────────────
# Chunks are tagged with EVERY specialty their CONTENT is relevant to (not the
# source book), so cross-specialty data (an MI in a respiratory text) stays visible
# to the cardiology agent. ⚠️ EDIT this to match your 15 specialties exactly.

SPECIALTIES = ["orthopaedics", "sports_medicine", "rheumatology", "physical_medicine_and_rehabilitation", "radiology", "pain_medicine", "neurology", "neurosurgery", "emergency_medicine", "trauma_surgery", "internal_medicine", "geriatrics", "endocrinology", "infectious_disease", "oncology"]
SPECIALTY_SET = set(SPECIALTIES)

# Variants the model emits → canonical specialty.
SPECIALTY_SYNONYMS = {
    "ortho": "orthopaedics", "orthopedic": "orthopaedics", "orthopaedic": "orthopaedics",
    "sports": "sports_medicine",
    "rheum": "rheumatology", "rheumatologic": "rheumatology",
    "pmr": "physical_medicine_and_rehabilitation", "physiatry": "physical_medicine_and_rehabilitation", "rehabilitation_medicine": "physical_medicine_and_rehabilitation",
    "imaging": "radiology", "diagnostic_imaging": "radiology",
    "pain_management": "pain_medicine",
    "neurologic": "neurology",
    "neurosurgical": "neurosurgery",
    "er": "emergency_medicine", "emergency": "emergency_medicine",
    "trauma": "trauma_surgery",
    "internal": "internal_medicine", "general_medicine": "internal_medicine",
    "geriatric": "geriatrics",
    "endocrine": "endocrinology",
    "infectious": "infectious_disease", "infection": "infectious_disease",
    "cancer": "oncology", "oncologic": "oncology"
}

# ── 4. Segmentation patterns ──────────────────────────────────────────────────
SECTION_HEADER_PATTERN = re.compile(
    r'^(Symptoms|Diagnosis|Treatment|Introduction|Pathophysiology)\s*$',
    re.IGNORECASE,
)
CONCEPT_HEADER_PATTERN = re.compile(
    r'^(treatment|diagnosis|pathophysiology|etiology|clinical features|management|'
    r'epidemiology|prognosis|pathogenesis|history|physical examination|complications|'
    r'prevention|indications|contraindications)\b',
    re.IGNORECASE,
)

# ── 5. Chunk validation thresholds ────────────────────────────────────────────
MIN_ENTITIES = 3
MAX_CHUNK_TOKENS = 650

# ── 6. Extraction prompt ──────────────────────────────────────────────────────
EXTRACTOR_ROLE = "clinical knowledge extraction engine"
SOURCE_DESCRIPTION = "medical reference text"
KNOWLEDGE_NOUN = "clinical knowledge"
CONCEPT_EXAMPLES = "a fracture, an orthopaedic injury, a diagnostic investigation, a surgical procedure, a rehabilitation strategy"
TOPIC_EXAMPLE = "Diagnosis of anterior cruciate ligament tear"
PROSE_NOUN = "clinical prose"
EXPERT_NOUN = "a clinician"
SKIP_EXAMPLES = '"rest", "exertion", "swimming", "history", "body position"'
TARGET_ENTITIES_HINT = "~8–15"

_ENTITY_TYPES_STR = ", ".join(ENTITY_TYPES)
_RELATION_TYPES_STR = ", ".join(RELATION_TYPES)
_SPECIALTIES_STR = ", ".join(SPECIALTIES)

RELATION_GUIDANCE = """    • condition             affects              anatomical_structure
    • condition             presents_with        symptom
    • condition             diagnosed_by         diagnostic_test
    • condition             treated_by           treatment
    • condition             treated_by           surgical_procedure
    • surgical_procedure    uses_implant         implant
    • risk_factor           increases_risk_of    condition
    • condition             leads_to             complication
    • condition             results_in           outcome
    • condition             complicated_by       complication
    • treatment             followed_by          rehabilitation
    • condition             requires             surgical_procedure"""

SYSTEM_PROMPT = f"""You are a {EXTRACTOR_ROLE} for a graph-based reasoning system. You
convert {SOURCE_DESCRIPTION} into STRICT JSON chunks that are HIGH-LEVEL,
self-contained units of {KNOWLEDGE_NOUN}, fit for BOTH a Neo4j knowledge graph and
a vector database.

OUTPUT FORMAT
- Valid JSON only. No markdown, no commentary. Follow the provided schema exactly.

WHAT A GOOD CHUNK IS (HIGH-LEVEL)
- Each chunk captures ONE coherent concept ({CONCEPT_EXAMPLES}).
- If the input covers several concepts, SPLIT into multiple chunks — one per concept.
- ~150–350 tokens of `text`; never merge unrelated material; never exceed ~600 tokens.
- source.topic = a SPECIFIC human-readable title (e.g. "{TOPIC_EXAMPLE}"), not "General".

SPECIALTIES (cross-specialty visibility — derive from CONTENT, not the source book)
- Tag `specialties` with EVERY specialty the chunk's entities are clinically relevant
  to, choosing from: {_SPECIALTIES_STR}.
- Judge from the entities themselves: a chunk mentioning myocardial infarction →
  include "cardiology"; metabolic acidosis → "nephrology"; thyrotoxicosis →
  "endocrinology" — EVEN IF the source is a respiratory text. Multiple is normal.

TEXT QUALITY (the `text` field)
- Clean, readable {PROSE_NOUN} faithful to the source. Remove OCR noise (bullets,
  private-use glyphs, headers/footers, figure/table labels). Repair obvious OCR
  breakage ("<1 min" not "<min"). Do not invent facts.

ENTITIES (salient, canonical, precisely typed)
- Extract the {TARGET_ENTITIES_HINT} most salient entities {EXPERT_NOUN} would key on.
  SKIP generic/trivial terms (e.g. {SKIP_EXAMPLES}). 10 strong entities beat 40 noisy ones.
- Per entity emit THREE fields — `name`, `type`, `aliases`:
    • `name` = the CANONICAL preferred term: expand abbreviations ("COPD" →
      "chronic obstructive pulmonary disease"), drop parentheticals, lowercase, singular.
      Use the SAME canonical name every time it recurs (the system derives one graph id from it).
    • `aliases` = the common synonyms/abbreviations for that entity, e.g. for myocardial
      infarction: ["MI","STEMI","NSTEMI","heart attack"]. [] if none. This is how
      synonyms collapse to one node — do NOT skip it.
    • `type` = the MOST SPECIFIC fit from: {_ENTITY_TYPES_STR}.
      TYPE PRECISELY — this drives graph traversal:

- bone, joint, ligament, tendon, muscle, cartilage, meniscus → Anatomical_Structure

- fracture, injury, arthritis, osteoporosis, deformity, pathology → Condition

- pain, swelling, tenderness, stiffness, instability, weakness → Symptom

- x-ray, MRI, CT, stress test, ultrasound → Diagnostic_Test

- casting, bracing, immobilization, conservative management → Treatment

- arthroscopy, ORIF, ACL reconstruction, joint replacement → Surgical_Procedure

- plate, screw, nail, prosthesis, graft → Implant

- NSAID, antibiotic, analgesic → Medication

- physiotherapy, exercise therapy, gait training → Rehabilitation

- smoking, obesity, age, sports participation → Risk_Factor

- infection, nonunion, malunion, implant failure → Complication

- recovery, fracture union, functional improvement → Outcome
      Do NOT label findings, lab values, or metabolic states as "disease".
- Do NOT emit codes — the system assigns standard terminology codes downstream.

RELATIONS (precise, directional, REFERENTIALLY CLEAN, with clinical qualifiers)
- `source`/`target` MUST each be the EXACT canonical `name` of an entity in this chunk
  (relations to non-listed entities are DROPPED — add the entity or omit the relation).
- Choose `type` from: {_RELATION_TYPES_STR}. Use the SPECIFIC, directional relation:
{RELATION_GUIDANCE}
- `qualifiers` — CAPTURE the clinical axis when the text encodes it; never flatten it away:
    • onset (one of: {", ".join(ONSET_VALUES)}) — e.g. a cause listed under "instantaneous
      onset" → "qualifiers": {{"onset": "instantaneous"}}.
    • optionally severity / temporality / context. Omit `qualifiers` ({{}}) only when none apply.
- Use `{RELATION_TYPE_FALLBACK}` ONLY when no specific relation fits.

SUMMARY FIELDS (embedded into a vector DB — tight but meaningful)
- `summary`: 1–2 sentences of the key facts. `clinical_significance`: 1 sentence on why it matters.

SELF-CHECK
- Entities canonical + precisely typed (no finding/lab/metabolic mislabeled as disease)?
- Relations id-resolvable, directional, with onset/qualifiers preserved?
- specialties cover every relevant field from the content? JSON strictly valid?
"""
