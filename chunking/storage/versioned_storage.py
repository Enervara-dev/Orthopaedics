import json
import logging
import os
import re
from pathlib import Path
from typing import List
from chunking.schemas.models import MicroChunk
from chunking.config.settings import settings

logger = logging.getLogger(__name__)

# Characters illegal in Windows file/dir names: < > : " / \ | ? * and control chars.
_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _long(path) -> str:
    """Absolute path with the Windows extended-length prefix so deep chunk paths
    (version/book/topic/long-chunk-id.json) never hit the 260-char MAX_PATH limit."""
    ap = os.path.abspath(str(path))
    if os.name == "nt" and not ap.startswith("\\\\?\\"):
        ap = "\\\\?\\" + ap
    return ap


def _safe_component(name: str, default: str = "untitled", max_len: int = 120) -> str:
    """Make a string safe to use as a single path component.

    The LLM-derived topic/book can contain ':' (e.g. "Mesothelioma: Prognosis")
    or '/' which break/mis-nest Windows directories. Replace illegal chars with a
    space, collapse whitespace, trim trailing dots/spaces (also illegal on Windows),
    and cap length so the full path stays under MAX_PATH.
    """
    name = _ILLEGAL.sub(" ", str(name))
    name = re.sub(r"\s+", " ", name).strip().rstrip(". ").strip()
    return name[:max_len].strip() or default


class VersionedStorage:
    def save_chunk(self, chunk: MicroChunk, index: int, version: str = "v1"):
        category = _safe_component(chunk.source.book.lower().replace(" ", "_"), "uncategorized")
        doc_id = _safe_component(chunk.source.topic, "untitled")
        chunk_file = _safe_component(chunk.chunk_id, "chunk")

        base_dir = Path(settings.storage_base_path) / version / category / doc_id
        os.makedirs(_long(base_dir), exist_ok=True)

        file_path = base_dir / f"{chunk_file}.json"

        with open(_long(file_path), "w", encoding="utf-8") as f:
            f.write(json.dumps(chunk.model_dump(), indent=2))
        logger.info(f"Saved chunk to {file_path}")

    def save_chunks(self, chunks: List[MicroChunk], version: str = "v1"):
        if not chunks:
            return
        category = _safe_component(chunks[0].source.book.lower().replace(" ", "_"), "uncategorized")
        base_dir = Path(settings.storage_base_path) / "_aggregated" / version / category
        os.makedirs(_long(base_dir), exist_ok=True)
        doc_id = _safe_component(chunks[0].source.topic, "untitled")
        file_path = base_dir / f"{doc_id}.json"
        with open(_long(file_path), "w", encoding="utf-8") as f:
            f.write(json.dumps([c.model_dump() for c in chunks], indent=2))
        logger.info(f"Saved {len(chunks)} chunks to {file_path}")
