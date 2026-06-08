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
    "disease", "syndrome", "symptom", "clinical_finding", "lab_finding",
    "metabolic_state", "physiological_state", "biomarker", "risk_factor",
    "anatomical_entity", "drug", "drug_class", "procedure", "test", "intervention",
    "mechanism", "pathogen", "gene", "protein", "clinical_process",
]
ENTITY_TYPE_SET = set(ENTITY_TYPES)

# Common synonyms / mislabels the model emits → canonical type.
ENTITY_TYPE_SYNONYMS = {
    "medication": "drug", "medicine": "drug", "pharmaceutical": "drug",
    "surgery": "procedure", "operation": "procedure",
    "lab": "lab_finding", "laboratory_test": "test", "diagnostic_test": "test",
    "lab_value": "lab_finding", "laboratory_finding": "lab_finding",
    "sign": "clinical_finding", "finding": "clinical_finding",
    "physical_sign": "clinical_finding",
    "metabolic_disturbance": "metabolic_state", "acid_base_disorder": "metabolic_state",
    "electrolyte_abnormality": "lab_finding",
    "condition": "disease", "disorder": "disease", "illness": "disease",
    "bacteria": "pathogen", "virus": "pathogen", "organism": "pathogen",
    "anatomical_structure": "anatomical_entity", "organ": "anatomical_entity",
    "drug_category": "drug_class", "medication_class": "drug_class",
}
ENTITY_TYPE_FALLBACK = "clinical_finding"

# ── 2. Relation vocabulary + qualifiers ───────────────────────────────────────
RELATION_TYPES = [
    "causes", "leads_to", "contributes_to", "manifests_as", "mimics", "complicates",
    "increases_risk_of", "reduces_risk_of", "predisposes_to", "protects_against",
    "treats", "alleviates", "mitigates", "reduces", "improves", "worsens", "used_for",
    "indicated_for", "prevents", "contraindicated_with", "metabolized_by", "mediated_by",
    "diagnosed_by", "detected_by", "screens_for", "assesses", "evaluates", "monitors",
    "measures", "classifies", "stages", "requires", "includes", "affects",
    "correlates_with", "alternative_to", "increases_likelihood_of",
    "reduces_likelihood_of", "associated_with",
]
RELATION_TYPE_SET = set(RELATION_TYPES)
RELATION_TYPE_FALLBACK = "associated_with"

# Allowed relation qualifier keys (graph edge properties). The clinical AXIS — esp.
# onset/speed — must survive as an edge property, not be flattened away.
RELATION_QUALIFIER_KEYS = ["onset", "temporality", "severity", "certainty", "context"]
ONSET_VALUES = ["instantaneous", "acute", "subacute", "chronic"]

# ── 3. Specialty taxonomy ─────────────────────────────────────────────────────
# Chunks are tagged with EVERY specialty their CONTENT is relevant to (not the
# source book), so cross-specialty data (an MI in a respiratory text) stays visible
# to the cardiology agent. ⚠️ EDIT this to match your 15 specialties exactly.
SPECIALTIES = [
    "pulmonology", "cardiology", "nephrology", "endocrinology", "gastroenterology",
    "hepatology", "neurology", "hematology", "oncology", "rheumatology",
    "infectious_disease", "immunology", "dermatology", "emergency_medicine", "general_medicine",
]
SPECIALTY_SET = set(SPECIALTIES)

# Variants the model emits → canonical specialty.
SPECIALTY_SYNONYMS = {
    "respiratory": "pulmonology", "respiratory_medicine": "pulmonology", "pulmonary": "pulmonology",
    "renal": "nephrology", "cardiovascular": "cardiology", "cardiac": "cardiology",
    "haematology": "hematology", "gi": "gastroenterology", "liver": "hepatology",
    "infectious_diseases": "infectious_disease", "endocrine": "endocrinology",
    "neuro": "neurology", "derm": "dermatology", "emergency": "emergency_medicine",
    "er": "emergency_medicine", "ed": "emergency_medicine",
    "internal_medicine": "general_medicine", "general": "general_medicine",
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
CONCEPT_EXAMPLES = "a disease, a drug/drug class, a diagnostic approach, a mechanism, a management strategy"
TOPIC_EXAMPLE = "Diagnosis of pulmonary embolism"
PROSE_NOUN = "clinical prose"
EXPERT_NOUN = "a clinician"
SKIP_EXAMPLES = '"rest", "exertion", "swimming", "history", "body position"'
TARGET_ENTITIES_HINT = "~8–15"

_ENTITY_TYPES_STR = ", ".join(ENTITY_TYPES)
_RELATION_TYPES_STR = ", ".join(RELATION_TYPES)
_SPECIALTIES_STR = ", ".join(SPECIALTIES)

RELATION_GUIDANCE = """    • disease/syndrome  manifests_as        symptom
    • risk_factor       increases_risk_of   disease        (or predisposes_to)
    • cause             causes / leads_to    effect
    • drug/intervention treats / used_for    disease/symptom (or alleviates/indicated_for)
    • disease           diagnosed_by         test/procedure
    • test/procedure    assesses / detects   disease/biomarker
    • drug              contraindicated_with drug/condition"""

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
        - a named disease/condition → disease
        - a patient-reported complaint → symptom; an examination sign → clinical_finding
        - a lab/test result (e.g. anaemia, hyperkalaemia, raised D-dimer) → lab_finding
        - an acid–base/metabolic disturbance (e.g. metabolic acidosis) → metabolic_state
        - a physiological state (e.g. hypoxaemia, hypotension) → physiological_state
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
