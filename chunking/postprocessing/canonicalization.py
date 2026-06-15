"""
Stage 3 — Canonicalization.

Normalizes common aliases / abbreviations to a single canonical entity name so
that the downstream graph collapses them to one node. The original value is
preserved in ``entity.aliases`` and the entity ``id`` is regenerated from the
new canonical ``name``.

This runs AFTER entity overrides and relation repair because both of those
stages operate on entity names and ids that must be consistent. After
canonicalization, entity ids may change (and relations are re-pointed
accordingly).

Usage:
    canonicalize_entities(chunk)  # mutates chunk in-place
"""

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chunking.schemas.models import MicroChunk

logger = logging.getLogger(__name__)


def _snake(name: str) -> str:
    """Lowercase snake_case id from a canonical name (mirrors models._snake)."""
    s = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    return s or "unknown"


# ── Alias → canonical name ─────────────────────────────────────────────────────
# Keys are lowercased; values are the canonical preferred name (also lowercase,
# matching the convention used by the extraction prompt). Extend as needed.

CANONICAL_ENTITIES: dict[str, str] = {

    # ── Imaging ────────────────────────────────────────────────────────────────
    "mri":                              "magnetic resonance imaging",
    "mr imaging":                       "magnetic resonance imaging",
    "nmr":                              "magnetic resonance imaging",
    "ct":                               "computed tomography",
    "cat scan":                         "computed tomography",
    "ct scan":                          "computed tomography",
    "ct scanning":                      "computed tomography",
    "radiograph":                       "x-ray",
    "plain radiograph":                 "x-ray",
    "plain film":                       "x-ray",
    "roentgenogram":                    "x-ray",
    "dexa":                             "dual energy x-ray absorptiometry",
    "dexa scan":                        "dual energy x-ray absorptiometry",
    "dxa":                              "dual energy x-ray absorptiometry",
    "dxa scan":                         "dual energy x-ray absorptiometry",
    "bone densitometry":                "dual energy x-ray absorptiometry",
    "usg":                              "ultrasound",
    "ultrasonography":                  "ultrasound",
    "sonography":                       "ultrasound",
    "emg":                              "electromyography",
    "ncs":                              "nerve conduction study",
    "bone scan":                        "bone scintigraphy",

    # ── Ligament / tendon injuries ─────────────────────────────────────────────
    "acl tear":                         "anterior cruciate ligament tear",
    "acl rupture":                      "anterior cruciate ligament tear",
    "acl injury":                       "anterior cruciate ligament tear",
    "torn acl":                         "anterior cruciate ligament tear",
    "pcl tear":                         "posterior cruciate ligament tear",
    "pcl rupture":                      "posterior cruciate ligament tear",
    "pcl injury":                       "posterior cruciate ligament tear",
    "mcl tear":                         "medial collateral ligament tear",
    "mcl injury":                       "medial collateral ligament tear",
    "lcl tear":                         "lateral collateral ligament tear",
    "lcl injury":                       "lateral collateral ligament tear",

    # ── Ligaments / anatomy abbreviations ──────────────────────────────────────
    "acl":                              "anterior cruciate ligament",
    "pcl":                              "posterior cruciate ligament",
    "mcl":                              "medial collateral ligament",
    "lcl":                              "lateral collateral ligament",
    "acj":                              "acromioclavicular joint",
    "ac joint":                         "acromioclavicular joint",
    "scj":                              "sternoclavicular joint",
    "tmj":                              "temporomandibular joint",
    "sij":                              "sacroiliac joint",
    "si joint":                         "sacroiliac joint",

    # ── Procedures ─────────────────────────────────────────────────────────────
    "orif":                             "open reduction internal fixation",
    "open reduction and internal fixation": "open reduction internal fixation",
    "crif":                             "closed reduction internal fixation",
    "thr":                              "total hip replacement",
    "tha":                              "total hip arthroplasty",
    "total hip replacement":            "total hip arthroplasty",
    "tkr":                              "total knee replacement",
    "tka":                              "total knee arthroplasty",
    "total knee replacement":           "total knee arthroplasty",
    "acl reconstruction":               "anterior cruciate ligament reconstruction",
    "im nailing":                       "intramedullary nailing",
    "mua":                              "manipulation under anesthesia",
    "cpm":                              "continuous passive motion",

    # ── Implants ───────────────────────────────────────────────────────────────
    "k wire":                           "kirschner wire",
    "k-wire":                           "kirschner wire",
    "dhs":                              "dynamic hip screw",
    "dcp":                              "dynamic compression plate",
    "lcp":                              "locking compression plate",
    "pfn":                              "proximal femoral nail",
    "pmma":                             "polymethylmethacrylate",
    "bone cement":                      "polymethylmethacrylate",

    # ── Conditions ─────────────────────────────────────────────────────────────
    "oa":                               "osteoarthritis",
    "ra":                               "rheumatoid arthritis",
    "avascular necrosis":               "avascular necrosis",
    "avn":                              "avascular necrosis",
    "osteonecrosis":                    "avascular necrosis",
    "dvt":                              "deep vein thrombosis",
    "pe":                               "pulmonary embolism",
    "crps":                             "complex regional pain syndrome",
    "sudeck atrophy":                   "complex regional pain syndrome",
    "rsd":                              "complex regional pain syndrome",
    "reflex sympathetic dystrophy":     "complex regional pain syndrome",
    "ctev":                             "congenital talipes equinovarus",
    "clubfoot":                         "congenital talipes equinovarus",
    "club foot":                        "congenital talipes equinovarus",
    "ddh":                              "developmental dysplasia of hip",
    "cdh":                              "congenital dislocation of hip",
    "gct":                              "giant cell tumor",

    # ── Medications ────────────────────────────────────────────────────────────
    "nsaid":                            "non-steroidal anti-inflammatory drug",
    "nsaids":                           "non-steroidal anti-inflammatory drug",
    "dmard":                            "disease modifying anti-rheumatic drug",
    "dmards":                           "disease modifying anti-rheumatic drug",
    "prp":                              "platelet rich plasma",

    # ── Diagnostic tests / lab values ──────────────────────────────────────────
    "esr":                              "erythrocyte sedimentation rate",
    "crp":                              "c-reactive protein",
    "cbc":                              "complete blood count",
    "alp":                              "alkaline phosphatase",
    "bmd":                              "bone mineral density",

    # ── Rehabilitation ─────────────────────────────────────────────────────────
    "pt":                               "physiotherapy",
    "physical therapy":                 "physiotherapy",
    "ot":                               "occupational therapy",
    "rom exercises":                    "range of motion exercises",
    "rom":                              "range of motion",

    # ── Fracture classifications ───────────────────────────────────────────────
    "nof fracture":                     "neck of femur fracture",
    "nof #":                            "neck of femur fracture",
    "femoral neck fracture":            "neck of femur fracture",
    "colles fracture":                  "colles fracture",
    "colles' fracture":                 "colles fracture",
    "smith fracture":                   "smith fracture",
    "smith's fracture":                 "smith fracture",
    "pott's fracture":                  "pott fracture",
    "pott fracture":                    "pott fracture",
    "monteggia fracture":               "monteggia fracture dislocation",
    "galeazzi fracture":                "galeazzi fracture dislocation",
}


def canonicalize_entities(chunk: "MicroChunk") -> None:
    """Normalize entity names to canonical forms.

    Mutates ``chunk.entities`` and ``chunk.relations`` in-place.  For each
    entity whose lowercased name appears in :data:`CANONICAL_ENTITIES`:

    1. The original name is stored in ``entity.aliases`` (if not already present).
    2. ``entity.name`` is replaced with the canonical form.
    3. ``entity.id`` is regenerated from the new canonical name.
    4. All relation ``source`` / ``target`` references to the old id are updated.
    5. Entities that collapse to the same canonical id are de-duplicated (aliases
       merged).
    """
    # Phase 1: build the id-remapping table and update entities
    id_remap: dict[str, str] = {}  # old_id → new_id

    for entity in chunk.entities:
        canonical = CANONICAL_ENTITIES.get(entity.name.lower())
        if canonical is None:
            continue

        old_name = entity.name
        old_id = entity.id

        # Store original as an alias (if it's different and not already tracked)
        if old_name.lower() != canonical.lower() and old_name not in entity.aliases:
            entity.aliases.append(old_name)

        # Update name and regenerate id
        entity.name = canonical
        entity.id = _snake(canonical)
        id_remap[old_id] = entity.id

    if not id_remap:
        return  # nothing to canonicalize

    # Phase 2: re-point relation source/target ids
    for rel in chunk.relations:
        if rel.source in id_remap:
            rel.source = id_remap[rel.source]
        if rel.target in id_remap:
            rel.target = id_remap[rel.target]

    # Phase 3: de-duplicate entities that collapsed to the same canonical id
    seen: dict[str, "MicroChunk"] = {}
    for e in chunk.entities:
        if e.id in seen:
            # Merge aliases into the kept entity
            kept = seen[e.id]
            for a in e.aliases:
                if a not in kept.aliases:
                    kept.aliases.append(a)
        else:
            seen[e.id] = e
    chunk.entities = list(seen.values())

    # Phase 4: de-duplicate relations that collapsed to the same (src, tgt, type)
    rel_seen: dict[tuple, "MicroChunk"] = {}
    for r in chunk.relations:
        key = (r.source, r.target, r.type)
        if key not in rel_seen:
            rel_seen[key] = r
    chunk.relations = list(rel_seen.values())

    logger.debug(
        "Canonicalization: remapped %d entity ids in chunk %s",
        len(id_remap), getattr(chunk, "chunk_id", "?"),
    )
