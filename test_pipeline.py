"""
Smoke / sanity tests for the pulmonology chunking pipeline.

By default this runs OFFLINE (no API calls, no cost):
  - imports resolve
  - a real PDF in dataset/ loads, cleans, segments, and produces semantic blocks
  - a MicroChunk passes strict validation
  - VersionedStorage writes a chunk into chunks/

Add --live to also run ONE real LLM extraction (uses your GEMINI_API_KEY):
  python test_pipeline.py --live

Usage:
  python test_pipeline.py                 # offline tests only
  python test_pipeline.py --live          # + one live extraction
  python test_pipeline.py --start-page 60 # which PDF page the block test reads
"""

import sys
import os
import shutil
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

PASS, FAIL = "PASS", "FAIL"
_results = []


def check(name, cond, detail=""):
    status = PASS if cond else FAIL
    _results.append(cond)
    line = f"[{status}] {name}"
    if detail:
        line += f"  ->  {detail}"
    print(line)
    return cond


def skip(name, reason=""):
    """Record nothing — used when test data is absent (e.g. empty dataset/)."""
    line = f"[SKIP] {name}"
    if reason:
        line += f"  ->  {reason}"
    print(line)
    return False


# ── 1. Imports ────────────────────────────────────────────────────────────────
def test_imports():
    from chunking.pipeline.manager import DocumentProcessingPipeline  # noqa
    from chunking.loaders.pdf_loader import PDFLoader  # noqa
    from chunking.config.settings import settings
    check("imports resolve (chunking.*)", True)
    check("storage_base_path == 'chunks'", settings.storage_base_path == "chunks",
          settings.storage_base_path)


# ── 2. Offline PDF -> blocks ────────────────────────────────────────────────────
def test_pdf_to_blocks(start_page):
    from chunking.loaders.pdf_loader import PDFLoader
    from chunking.cleaners.text_cleaner import TextCleaner
    from chunking.detectors.structure import StructureDetector
    from chunking.extractors.semantic import SemanticExtractor
    from chunking.schemas.models import DocumentMetadata

    pdfs = sorted(Path("dataset").glob("*.pdf"))
    if not pdfs:
        skip("PDF -> blocks path", "no PDF in dataset/ (add one to exercise this)")
        return

    pdf = pdfs[0]
    meta = DocumentMetadata(doc_id="test_doc", book_type="Test Book",
                            version="test", source_path=str(pdf))

    pages = PDFLoader("test_doc", "Test Book", "test").load(
        str(pdf), start_page=start_page, max_pages=3)
    check("PDF loaded pages", len(pages) > 0, f"{len(pages)} pages from p.{start_page}")

    cleaner = TextCleaner()
    for p in pages:
        p["text"] = cleaner.normalize(p["text"])
    check("cleaner returns text", any(p["text"].strip() for p in pages))

    sections = StructureDetector().segment(pages)
    check("detector produced sections", len(sections) > 0, f"{len(sections)} sections")

    blocks = SemanticExtractor().extract_blocks(sections, meta)
    check("semantic blocks produced", len(blocks) > 0, f"{len(blocks)} blocks")


# ── 3. Schema validation ────────────────────────────────────────────────────────
def _make_valid_chunk(version="test", book="Test Book"):
    from chunking.schemas.models import (
        MicroChunk, ChunkSource, ChunkMetadata, ClinicalEntity, ClinicalRelation,
    )
    return MicroChunk(
        chunk_id="test-chunk-1",
        source=ChunkSource(book=book, chapter="general", topic="general", page="1"),
        text="Anterior cruciate ligament tear is an orthopaedic condition diagnosed by MRI.",
        entities=[
            ClinicalEntity(name="anterior cruciate ligament tear", type="Condition"),
            ClinicalEntity(name="anterior cruciate ligament", type="Anatomical_Structure"),
            ClinicalEntity(name="mri", type="Diagnostic_Test"),
        ],
        relations=[
            ClinicalRelation(source="anterior cruciate ligament tear", target="anterior cruciate ligament", type="AFFECTS"),
            ClinicalRelation(source="anterior cruciate ligament tear", target="mri", type="DIAGNOSED_BY"),
        ],
        summary="ACL tear overview.",
        clinical_significance="Common knee ligament injury.",
        metadata=ChunkMetadata(tokens=15, model="test", quality_check="passed"),
    )


def test_schema_validation():
    from chunking.schemas.models import MicroChunk
    try:
        _make_valid_chunk()
        check("valid MicroChunk passes validation", True)
    except Exception as e:
        check("valid MicroChunk passes validation", False, str(e))

    # too few entities must be rejected
    try:
        MicroChunk(**{**_make_valid_chunk().model_dump(),
                      "entities": _make_valid_chunk().entities[:1]})
        check("under-3-entities chunk is rejected", False, "it was accepted")
    except Exception:
        check("under-3-entities chunk is rejected", True)


# ── 4. Storage writes into chunks/ ──────────────────────────────────────────────
def test_storage_writes_to_chunks():
    from chunking.storage.versioned_storage import VersionedStorage
    from chunking.config.settings import settings

    version = "_selftest"
    chunk = _make_valid_chunk(version=version)
    category = chunk.source.book.lower().replace(" ", "_")
    expected = (Path(settings.storage_base_path) / version / category /
                chunk.source.topic / f"{chunk.chunk_id}.json")

    # clean any prior run, write, assert, then clean up
    test_root = Path(settings.storage_base_path) / version
    if test_root.exists():
        shutil.rmtree(test_root)

    VersionedStorage().save_chunk(chunk, index=1, version=version)
    ok = expected.exists()
    check("chunk written under chunks/", ok, str(expected))

    if test_root.exists():
        shutil.rmtree(test_root)  # keep the workspace clean


# ── 5. (optional) one real LLM extraction ───────────────────────────────────────
def test_live_extraction(start_page):
    from chunking.loaders.pdf_loader import PDFLoader
    from chunking.cleaners.text_cleaner import TextCleaner
    from chunking.detectors.structure import StructureDetector
    from chunking.extractors.semantic import SemanticExtractor
    from chunking.llm.retry_engine import ExtractionWithRetry
    from chunking.schemas.models import DocumentMetadata

    pdfs = sorted(Path("dataset").glob("*.pdf"))
    if not pdfs:
        skip("[live] extraction", "no PDF in dataset/")
        return
    pdf = pdfs[0]
    meta = DocumentMetadata(doc_id="test_doc", book_type="Test Book",
                            version="test", source_path=str(pdf))
    pages = PDFLoader("test_doc", "Test Book", "test").load(
        str(pdf), start_page=start_page, max_pages=3)
    cleaner = TextCleaner()
    for p in pages:
        p["text"] = cleaner.normalize(p["text"])
    sections = StructureDetector().segment(pages)
    blocks = SemanticExtractor().extract_blocks(sections, meta)
    if not check("[live] have a block to extract", bool(blocks)):
        return

    result = ExtractionWithRetry().run(blocks[0].text)
    ok = result is not None and len(getattr(result, "chunks", [])) > 0
    n = len(result.chunks) if result else 0
    check("[live] LLM produced >=1 valid chunk", ok, f"{n} chunks")


def main():
    ap = argparse.ArgumentParser(description="Orthopaedics pipeline tests")
    ap.add_argument("--live", action="store_true", help="also run one real LLM extraction")
    ap.add_argument("--start-page", type=int, default=60,
                    help="PDF page the block/live tests read from (default: 60)")
    args = ap.parse_args()

    print("=" * 60)
    print("OFFLINE TESTS (no API calls)")
    print("=" * 60)
    test_imports()
    test_pdf_to_blocks(args.start_page)
    test_schema_validation()
    test_storage_writes_to_chunks()

    if args.live:
        print("=" * 60)
        print("LIVE TEST (uses GEMINI_API_KEY)")
        print("=" * 60)
        test_live_extraction(args.start_page)

    print("=" * 60)
    passed = sum(1 for r in _results if r)
    total = len(_results)
    print(f"RESULT: {passed}/{total} checks passed")
    print("=" * 60)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
