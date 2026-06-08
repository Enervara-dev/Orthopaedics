from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator
import logging
import re

from chunking.domain import (
    ENTITY_TYPE_SET, ENTITY_TYPE_SYNONYMS, ENTITY_TYPE_FALLBACK,
    RELATION_TYPE_SET, RELATION_TYPE_FALLBACK,
    SPECIALTY_SET, SPECIALTY_SYNONYMS,
    RELATION_QUALIFIER_KEYS, ONSET_VALUES,
    MIN_ENTITIES, MAX_CHUNK_TOKENS,
)

logger = logging.getLogger(__name__)


def _snake(name: str) -> str:
    """lowercase snake_case of an entity name (derived, not LLM-emitted)."""
    s = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    return s or "unknown"


class DocumentMetadata(BaseModel):
    doc_id: str
    book_type: str
    version: str
    source_path: str
    publication_year: Optional[int] = None  # true source year (per document)
    edition: Optional[str] = None
    source_specialty: Optional[str] = None  # the book's home specialty (graph parent)


class ChunkSource(BaseModel):
    # All default to "" so a lean LLM payload never hard-fails validation: the
    # pipeline always stamps book + page authoritatively, and patches chapter/topic
    # from block metadata when the model leaves them blank.
    book: str = ""
    chapter: str = ""
    topic: str = ""
    page: str = ""
    # True source recency — the publication/guideline year of the SOURCE, set per
    # document (NOT the processing date in metadata.created_at). Used to rank
    # scientific recency so old textbook data doesn't override newer guidelines.
    publication_year: Optional[int] = None
    edition: Optional[str] = None
    # The book's home specialty — becomes the Neo4j parent node (:Specialty)-[:HAS_CHUNK]->.
    source_specialty: Optional[str] = None


class ChunkMetadata(BaseModel):
    tokens: Union[str, int]
    model: str
    quality_check: str
    version: Optional[str] = None
    created_at: Optional[str] = None


class ClinicalEntity(BaseModel):
    id: str = ""  # canonical MERGE key for Neo4j — derived from name (not LLM-emitted)
    name: str = Field(..., description="Canonical entity name")
    type: str
    aliases: List[str] = Field(default_factory=list)  # synonyms/abbreviations (collapse to one node)
    # Standard terminology code (SNOMED CT / ICD-10 / RxNorm / LOINC). NOT emitted by
    # the LLM (it would hallucinate) — populated by a terminology linker downstream.
    code: Optional[str] = None
    coding_system: Optional[str] = None

    @model_validator(mode='before')
    @classmethod
    def normalize_entity_fields(cls, v: Any) -> Any:
        if not isinstance(v, dict):
            return v
        # entity_name → name
        if 'entity_name' in v and 'name' not in v:
            v['name'] = v.pop('entity_name')
        if 'entity_type' in v and 'type' not in v:
            v['type'] = v.pop('entity_type')
        # Canonical id derived from the canonical name — the stable key Neo4j MERGEs
        # on so the same entity across chunks collapses to one node.
        if v.get('name'):
            v['id'] = _snake(v['name'])
        # Clean aliases: list of distinct non-empty strings, excluding the name itself.
        raw = v.get('aliases') or []
        if isinstance(raw, str):
            raw = [raw]
        nm = str(v.get('name', '')).lower().strip()
        clean = []
        for a in raw:
            a = str(a).strip()
            if a and a.lower() != nm and a not in clean:
                clean.append(a)
        v['aliases'] = clean
        return v

    @field_validator("type", mode="before")
    @classmethod
    def coerce_entity_type(cls, v: str) -> str:
        v = str(v).lower().strip()
        if v in ENTITY_TYPE_SET:
            return v
        return ENTITY_TYPE_SYNONYMS.get(v, ENTITY_TYPE_FALLBACK)


class ClinicalRelation(BaseModel):
    source: str = Field(..., description="Source entity id")
    target: str = Field(..., description="Target entity id")
    type: str
    # Graph edge properties carrying the clinical axis (esp. onset). Kept so triage
    # logic can use e.g. onset=instantaneous vs chronic, instead of a flat edge.
    qualifiers: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='before')
    @classmethod
    def normalize_relation_fields(cls, v: Any) -> Any:
        if not isinstance(v, dict):
            return v
        # Keep only allowed qualifier keys; normalize onset to the controlled set.
        q = v.get('qualifiers')
        if isinstance(q, dict):
            clean = {}
            for k, val in q.items():
                k = str(k).lower().strip()
                if k in RELATION_QUALIFIER_KEYS and val not in (None, "", []):
                    sval = str(val).lower().strip()
                    if k == "onset" and sval not in ONSET_VALUES:
                        continue
                    clean[k] = sval
            v['qualifiers'] = clean
        else:
            v['qualifiers'] = {}
        # from_entity/to_entity → source/target
        if 'from_entity' in v and 'source' not in v:
            v['source'] = v.pop('from_entity')
        if 'to_entity' in v and 'target' not in v:
            v['target'] = v.pop('to_entity')
        # subject/object → source/target
        if 'subject' in v and 'source' not in v:
            v['source'] = v.pop('subject')
        if 'object' in v and 'target' not in v:
            v['target'] = v.pop('object')
        # source_id/target_id → source/target
        if 'source_id' in v and 'source' not in v:
            v['source'] = v.pop('source_id')
        if 'target_id' in v and 'target' not in v:
            v['target'] = v.pop('target_id')
        # relation_type / relationship_type → type
        if 'relation_type' in v and 'type' not in v:
            v['type'] = v.pop('relation_type')
        if 'relationship_type' in v and 'type' not in v:
            v['type'] = v.pop('relationship_type')
        if 'relationship' in v and 'type' not in v:
            v['type'] = v.pop('relationship')
        return v

    @field_validator("type", mode="before")
    @classmethod
    def coerce_relation_type(cls, v: str) -> str:
        v = str(v).lower().strip().replace(" ", "_")
        if v in RELATION_TYPE_SET:
            return v
        return RELATION_TYPE_FALLBACK


class MicroChunk(BaseModel):
    chunk_id: str
    source: ChunkSource
    # Every specialty the CONTENT is relevant to (not the source book) — keeps
    # cross-specialty data visible to all relevant agents.
    specialties: List[str] = Field(default_factory=list)
    text: str
    entities: List[ClinicalEntity]
    relations: List[ClinicalRelation]
    summary: str
    clinical_significance: str
    metadata: ChunkMetadata

    @model_validator(mode='before')
    @classmethod
    def normalize_chunk_fields(cls, v: Any) -> Any:
        if not isinstance(v, dict):
            return v
        import uuid

        # specialties: normalize to the controlled taxonomy, drop unknowns.
        raw = v.get('specialties') or v.get('specialty') or []
        if isinstance(raw, str):
            raw = [raw]
        norm = []
        for s in raw:
            s = _snake(str(s))
            s = SPECIALTY_SYNONYMS.get(s, s)
            if s in SPECIALTY_SET and s not in norm:
                norm.append(s)
        v['specialties'] = norm

        # text field aliases
        for alias in ('chunk_text', 'content', 'passage', 'body'):
            if alias in v and 'text' not in v:
                v['text'] = v.pop(alias)
                break

        # summary aliases
        for alias in ('clinical_summary', 'chunk_summary', 'abstract'):
            if alias in v and 'summary' not in v:
                v['summary'] = v.pop(alias)
                break

        # clinical_significance aliases
        for alias in ('significance', 'clinical_sig', 'importance', 'clinical_importance'):
            if alias in v and 'clinical_significance' not in v:
                v['clinical_significance'] = v.pop(alias)
                break

        # chunk_id aliases
        for alias in ('id', 'chunk_id_str', 'identifier'):
            if alias in v and 'chunk_id' not in v:
                v['chunk_id'] = v.pop(alias)
                break

        # auto-fill missing structural fields so chunks aren't dropped
        if 'chunk_id' not in v:
            v['chunk_id'] = str(uuid.uuid4())[:8]

        if 'summary' not in v:
            text = v.get('text', '')
            v['summary'] = (text[:200] + '...') if len(text) > 200 else text

        if 'clinical_significance' not in v:
            v['clinical_significance'] = 'Clinical data extracted from medical reference text.'

        if 'source' not in v:
            v['source'] = {
                'book': 'Unknown',
                'chapter': v.pop('chapter', 'Unknown'),
                'topic': v.pop('topic', 'Unknown'),
                'page': str(v.pop('page', 'Unknown')),
            }

        if 'metadata' not in v:
            text = v.get('text', '')
            approx_tokens = int(len(text.split()) * 1.3)
            v['metadata'] = {
                'tokens': approx_tokens,
                'model': '',            # stamped authoritatively by the pipeline
                'quality_check': 'passed',
            }

        return v

    @model_validator(mode='after')
    def finalize_for_graph(self) -> 'MicroChunk':
        """Make the chunk graph-clean, then gate on quality.

        - dedupe entities by canonical id (one node per concept per chunk)
        - rewrite every relation's source/target to entity ids and DROP any edge
          whose endpoints aren't real entities (no dangling edges in Neo4j)
        - dedupe identical edges
        """
        seen = {}
        for e in self.entities:
            if e.id in seen:
                # same concept mentioned twice → one node, union the aliases
                kept = seen[e.id]
                for a in e.aliases:
                    if a not in kept.aliases:
                        kept.aliases.append(a)
            else:
                seen[e.id] = e
        self.entities = list(seen.values())
        id_set = set(seen.keys())

        resolved = {}
        for r in self.relations:
            s = r.source if r.source in id_set else _snake(r.source)
            t = r.target if r.target in id_set else _snake(r.target)
            if s in id_set and t in id_set and s != t:
                r.source, r.target = s, t
                resolved[(s, t, r.type)] = r
        self.relations = list(resolved.values())

        # Quality gates. The chunk feeds BOTH a vector DB (needs text + entities) and
        # a graph (needs clean edges), so we DON'T discard a chunk that has good nodes
        # and text just because edges didn't resolve — we keep it with whatever edges
        # survived. Only require enough entities and a sane size.
        if len(self.entities) < MIN_ENTITIES:
            raise ValueError(f"Failure: entities < {MIN_ENTITIES} (found {len(self.entities)})")
        approx_tokens = len(self.text.split()) * 1.3
        if approx_tokens > MAX_CHUNK_TOKENS:
            raise ValueError(f"Failure: tokens > {MAX_CHUNK_TOKENS} (approx {int(approx_tokens)})")

        return self


class ExtractedClinicalData(BaseModel):
    chunks: List[Any]

    @model_validator(mode='after')
    def filter_invalid_chunks(self) -> 'ExtractedClinicalData':
        valid = []
        for raw in self.chunks:
            if isinstance(raw, MicroChunk):
                valid.append(raw)
                continue
            try:
                valid.append(MicroChunk(**raw) if isinstance(raw, dict) else raw)
            except Exception as e:
                logger.warning(f"Dropping invalid chunk: {e}")
        self.chunks = valid
        return self


class SemanticBlock(BaseModel):
    block_id: str
    text: str
    section: Optional[str]
    metadata: DocumentMetadata
    page: Optional[str] = None  # 1-based PDF page or range, e.g. "27" or "27-28"
