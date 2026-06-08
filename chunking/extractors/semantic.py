import re
from typing import List, Optional
from chunking.schemas.models import SemanticBlock, DocumentMetadata
from chunking.domain import CONCEPT_HEADER_PATTERN


def _page_str(pages: List[int]) -> Optional[str]:
    """Compact a list of page numbers into "27" or "27-29"."""
    nums = [p for p in pages if p]
    if not nums:
        return None
    lo, hi = min(nums), max(nums)
    return str(lo) if lo == hi else f"{lo}-{hi}"


class SemanticExtractor:
    def extract_blocks(self, sections: List[dict], metadata: DocumentMetadata) -> List[SemanticBlock]:
        blocks = []
        block_counter = 1

        concept_headers = CONCEPT_HEADER_PATTERN

        for p_sec in sections:
            section_name = p_sec['section']

            # Build (paragraph, page) units, preserving page provenance.
            para_units = []
            segments = p_sec.get('segments')
            if segments:
                for seg in segments:
                    for para in seg['text'].split('\n\n'):
                        para_units.append((para, seg.get('page')))
            else:
                for para in p_sec.get('text', '').split('\n\n'):
                    para_units.append((para, None))

            current_merged_text = []
            current_pages = []

            def emit():
                nonlocal block_counter
                blocks.append(SemanticBlock(
                    block_id=f"{metadata.doc_id}-blk-{block_counter}",
                    text="\n\n".join(current_merged_text),
                    section=section_name,
                    metadata=metadata,
                    page=_page_str(current_pages),
                ))
                block_counter += 1

            for para, page in para_units:
                para = para.strip()
                if not para or len(para) < 20:
                    continue

                # Split when: new concept starts, drug introduced, disease mentioned
                is_boundary = False
                if concept_headers.match(para):
                    is_boundary = True
                elif re.match(r'^([A-Z][A-Za-z0-9-]+(?:\s+[A-Za-z0-9-]+){0,3})\s*[:\.]', para):
                    is_boundary = True
                elif para.startswith('•') or para.startswith('- '):
                    is_boundary = True

                # Explicit constraint: if tokens > 350, split
                approx_para_tokens = len(para.split()) * 1.3
                approx_current_tokens = len("\n\n".join(current_merged_text).split()) * 1.3
                if approx_current_tokens + approx_para_tokens > 350:
                    is_boundary = True

                if is_boundary and current_merged_text:
                    emit()
                    current_merged_text = []
                    current_pages = []

                current_merged_text.append(para)
                if page is not None:
                    current_pages.append(page)

            if current_merged_text:
                emit()

        return blocks
