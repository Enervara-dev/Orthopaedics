import re
from typing import List, Tuple
from graphrag.domain.entity_rules import MEDICATION_NAME_PATTERN, MEDICATION_NAME_STOPWORDS
from graphrag.utils.logger import get_logger

logger = get_logger(__name__)


class EntityProcessor:

    @staticmethod
    def process_matches(
        matches: list,
        priority_entity_types: List[str] | None = None,
        boost_drug_pairs: bool = False,
        query: str = "",
    ) -> Tuple[str, List[str], List[str]]:
        """
        Extract entities and build vector context string from reranked chunks.

        Parameters
        ----------
        matches              : reranked Pinecone match dicts
        priority_entity_types: entity types to surface first (from QueryConfig)
        boost_drug_pairs     : if True, re-rank chunks that contain BOTH drug
                               names detected in the query (drug_interaction mode)
        query                : original query text (used for drug-pair detection)
        """
        if not matches:
            logger.info("❌ No chunks to process.")
            return "No medical chunks found.", [], []

        priority_entity_types = priority_entity_types or []

        # ── Optional: drug-pair chunk boosting ──────────────────────────────
        if boost_drug_pairs and query:
            matches = EntityProcessor._boost_drug_pair_chunks(matches, query)

        # Order-preserving, de-duplicated collection so the entity set handed to
        # graph traversal is STABLE run-to-run. Clinically-relevant (priority)
        # entity types are surfaced first and never dropped by the cap, so graph
        # gating sees them consistently.
        priority_types = {t.strip().lower() for t in priority_entity_types}
        priority_entities: List[str] = []
        other_entities: List[str] = []
        seen: set[str] = set()
        chunk_summaries: List[str] = []

        for match in matches:
            md = match.get("metadata", {})
            chunk_summaries.append(md.get("summary", ""))

            for ent_str in md.get("entities", []):
                if not ent_str:
                    continue
                # Pinecone metadata stores entities as plain canonical NAMES
                # (see ingest_pinecone.metadata: [e["name"] ...]). Older/other
                # sources may use "type:name" — support both.
                if ":" in ent_str:
                    ent_type, ent_name = ent_str.split(":", 1)
                    ent_type_clean = ent_type.strip().lower()
                else:
                    ent_name = ent_str
                    ent_type_clean = ""
                ent_name_clean = ent_name.strip().lower()

                if not ent_name_clean or ent_name_clean in seen:
                    continue
                seen.add(ent_name_clean)

                if ent_type_clean and ent_type_clean in priority_types:
                    priority_entities.append(ent_name_clean)
                else:
                    other_entities.append(ent_name_clean)

        # ── Build final entity list: priority first (kept), then rest (cap 30) ─
        cap = 30
        final_entities = priority_entities[:cap] + other_entities[: max(0, cap - len(priority_entities))]

        vector_context_str = "\n".join([f"- {s}" for s in chunk_summaries if s])

        logger.info(
            f"✅ {len(matches)} chunks processed  |  "
            f"{len(final_entities)} entities  |  "
            f"{len(priority_entities)} priority ({', '.join(priority_entity_types) or 'none'})"
        )

        return vector_context_str, final_entities, chunk_summaries

    # -------------------------------------------------------------------------

    @staticmethod
    def _boost_drug_pair_chunks(matches: list, query: str) -> list:
        """
        For drug_interaction queries: re-rank chunks so that those containing
        BOTH detected drug names in their entities come first.
        """
        detected_drugs = EntityProcessor._extract_drug_names(query)
        if len(detected_drugs) < 2:
            return matches  # can't boost without a pair

        logger.info(f"💊 Drug-pair boost active — detected drugs: {detected_drugs}")

        def drug_hit_count(match):
            entities_text = " ".join(match.get("metadata", {}).get("entities", [])).lower()
            return sum(1 for d in detected_drugs if d in entities_text)

        return sorted(matches, key=drug_hit_count, reverse=True)

    @staticmethod
    def _extract_drug_names(query: str) -> List[str]:
        """
        Lightweight heuristic: extract multi-word tokens that look like drug names
        (capitalized or all-alpha strings after 'take', 'with', 'and', 'between').
        """
        candidates = re.findall(MEDICATION_NAME_PATTERN, query)
        drugs = [c for c in candidates if c not in MEDICATION_NAME_STOPWORDS]
        return [d.lower() for d in drugs]
