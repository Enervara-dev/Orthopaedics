"""
Chunk the Oxford Handbook of Respiratory Medicine over pages 27–845.

Page numbers are 1-based PHYSICAL PDF pages, inclusive (same convention as
chunker.py --start-page/--end-page). The handbook front-matter ends and
clinical content begins around p.27; p.845 is the last content page before the
index, so this range is the body of the book.

THREE MODES (cost-safe by default):

  --plan   (DEFAULT)  FREE. No API calls. Loads the page range, cleans, segments
                      and builds semantic blocks, then reports how many blocks /
                      LLM batches the full run needs so you can size cost & time.

  --smoke [N]         CHEAP paid run on the first N pages of the range (default 5)
                      into a throwaway version. Validates the range yields valid
                      micro-chunks (text + >=3 entities + relations). Deletes the
                      throwaway output's resume markers so it never taints --full.

  --full              REAL paid run over pages 27–845 → chunks/<version>/.
                      Resumable: already-processed blocks are skipped on re-run.

Usage:
    python chunk_pages_27_845.py                 # plan only (free)
    python chunk_pages_27_845.py --smoke         # validate on pages 27–31 (cheap)
    python chunk_pages_27_845.py --smoke --smoke-pages 8
    python chunk_pages_27_845.py --full          # the real run → chunks/v1/
    python chunk_pages_27_845.py --full --version v2
"""

import sys
import os
import argparse
import logging
import math
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# Pipeline log lines contain non-ASCII (e.g. "→"); the default Windows console
# is cp1252 and would raise UnicodeEncodeError. Force UTF-8 for console output.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Config ───────────────────────────────────────────────────────────────────
START_PAGE = 27
END_PAGE = 845
DEFAULT_VERSION = "v1"
SMOKE_VERSION = "_smoke_27_845"
DATASET_DIR = PROJECT_ROOT / "dataset"

# ⚠️ SET THESE to the true source recency (used for scientific-recency ranking,
# distinct from the processing date). Look at your PDF's edition/copyright page.
PUBLICATION_YEAR = None      # e.g. 2020
EDITION = None               # e.g. "3rd edition"
# The book's home specialty → the Neo4j parent node (:Specialty)-[:HAS_CHUNK]->(:Chunk).
SOURCE_SPECIALTY = "pulmonology"

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "chunk_27_845.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("chunk_27_845")

from chunking.pipeline.manager import DocumentProcessingPipeline, BATCH_SIZE, MAX_WORKERS  # noqa: E402
from chunking.loaders.pdf_loader import PDFLoader  # noqa: E402
from chunking.cleaners.text_cleaner import TextCleaner  # noqa: E402
from chunking.detectors.structure import StructureDetector  # noqa: E402
from chunking.extractors.semantic import SemanticExtractor  # noqa: E402
from chunking.schemas.models import DocumentMetadata  # noqa: E402
from chunking.config.settings import settings  # noqa: E402


def _slugify(name: str) -> str:
    slug = "".join(c.lower() if c.isalnum() else "_" for c in name)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "document"


def _titleize(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").strip().title()


def _find_pdf() -> Path:
    pdfs = sorted(DATASET_DIR.glob("*.pdf"))
    if not pdfs:
        logger.error("No PDFs found in %s — drop the handbook PDF there.", DATASET_DIR)
        sys.exit(1)
    if len(pdfs) > 1:
        logger.warning("Multiple PDFs found; using the first: %s", pdfs[0].name)
    return pdfs[0]


def _build_blocks(pdf: Path, doc_id: str, book_type: str, start: int, end: int):
    """Run load → clean → segment → semantic blocks (everything before the LLM).

    This is exactly what DocumentProcessingPipeline.process_pdf does up to the
    LLM call, so block/batch counts here match the real run. No API calls.
    """
    meta = DocumentMetadata(doc_id=doc_id, book_type=book_type, version="plan",
                            source_path=str(pdf))
    pages = PDFLoader(doc_id, book_type, "plan").load(
        str(pdf), start_page=start, end_page=end)
    cleaner = TextCleaner()
    for p in pages:
        p["text"] = cleaner.normalize(p["text"])
    sections = StructureDetector().segment(pages)
    blocks = SemanticExtractor().extract_blocks(sections, meta)
    return pages, sections, blocks


def _count_on_disk(version: str) -> int:
    """Count per-chunk JSON files actually written for this version.

    Storage keys the path off each chunk's source.book/source.topic (not our
    doc_id), so we scan the whole version tree rather than guess the path.
    """
    version_dir = Path(settings.storage_base_path) / version
    return len(list(version_dir.rglob("*.json"))) if version_dir.exists() else 0


# ── Mode: plan (free) ─────────────────────────────────────────────────────────
def plan():
    pdf = _find_pdf()
    doc_id = _slugify(pdf.stem)
    book_type = _titleize(pdf.stem)

    logger.info("=" * 60)
    logger.info("PLAN (free, no API calls): %s pages %d–%d", pdf.name, START_PAGE, END_PAGE)
    logger.info("=" * 60)

    pages, sections, blocks = _build_blocks(pdf, doc_id, book_type, START_PAGE, END_PAGE)

    n_blocks = len(blocks)
    n_batches = math.ceil(n_blocks / BATCH_SIZE) if n_blocks else 0
    block_tokens = [int(len(b.text.split()) * 1.3) for b in blocks]
    total_tokens = sum(block_tokens)
    # Resume markers already on disk for the real doc_id (so a re-run is cheaper).
    done_dir = LOG_DIR / "processed_blocks"
    already_done = sum(1 for b in blocks if (done_dir / f"{b.block_id}.done").exists())

    logger.info("Pages loaded            : %d", len(pages))
    logger.info("Detector sections       : %d", len(sections))
    logger.info("Semantic blocks         : %d", n_blocks)
    logger.info("LLM batches (size %d)    : %d  (one Gemini call each)", BATCH_SIZE, n_batches)
    logger.info("Parallel workers        : %d  → ~%d sequential waves",
                MAX_WORKERS, math.ceil(n_batches / MAX_WORKERS) if n_batches else 0)
    logger.info("Block input tokens      : ~%d total, ~%d avg/block",
                total_tokens, (total_tokens // n_blocks) if n_blocks else 0)
    logger.info("Already processed       : %d/%d blocks (.done markers present)",
                already_done, n_blocks)

    logger.info("-" * 60)
    logger.info("Sample blocks:")
    for b in blocks[:3]:
        preview = " ".join(b.text.split())[:140]
        logger.info("  [%s | %s] %s…", b.block_id, b.section or "General", preview)

    logger.info("-" * 60)
    logger.info("Next: validate cheaply  → python %s --smoke", Path(__file__).name)
    logger.info("Then run for real       → python %s --full", Path(__file__).name)
    # Plan always "succeeds" as long as the range produced blocks to process.
    return n_blocks


# ── Mode: smoke (cheap paid) and full (real paid) ─────────────────────────────
def run_pipeline(full: bool, version: str, smoke_pages: int):
    pdf = _find_pdf()
    pipeline = DocumentProcessingPipeline()
    book_type = _titleize(pdf.stem)

    if full:
        start, end = START_PAGE, END_PAGE
        run_version = version
        doc_id = _slugify(pdf.stem)
        logger.info("=" * 60)
        logger.info("FULL RUN: %s pages %d–%d → chunks/%s/", pdf.name, start, end, run_version)
        logger.info("That's %d pages. This calls the LLM many times and costs money.",
                    end - start + 1)
        logger.info("=" * 60)
    else:
        start = START_PAGE
        end = min(START_PAGE + smoke_pages - 1, END_PAGE)
        run_version = SMOKE_VERSION
        # Distinct doc_id so smoke .done markers / block_ids never collide with the
        # real run (block numbering restarts at 1 for every load).
        doc_id = _slugify(pdf.stem) + "_smoke"
        # Start each smoke run clean so results are reproducible.
        base = Path(settings.storage_base_path)
        for smoke_root in (base / SMOKE_VERSION, base / "_aggregated" / SMOKE_VERSION):
            if smoke_root.exists():
                shutil.rmtree(smoke_root)
        for done in (LOG_DIR / "processed_blocks").glob(f"{doc_id}-blk-*.done"):
            done.unlink()
        logger.info("=" * 60)
        logger.info("SMOKE TEST: %s pages %d–%d → chunks/%s/ (throwaway)",
                    pdf.name, start, end, run_version)
        logger.info("=" * 60)

    chunks = pipeline.process_pdf(
        file_path=str(pdf),
        doc_id=doc_id,
        book_type=book_type,
        version=run_version,
        start_page=start,
        end_page=end,
        publication_year=PUBLICATION_YEAR,
        edition=EDITION,
        source_specialty=SOURCE_SPECIALTY,
    ) or []

    n = len(chunks)
    on_disk = _count_on_disk(run_version)
    logger.info("-" * 60)
    logger.info("RESULT: %d chunk(s) produced (%d JSON files under chunks/%s/).",
                n, on_disk, run_version)

    if full:
        return n

    # --- Smoke validation: every chunk must have text, >=3 entities, relations ---
    problems = []
    for c in chunks:
        if not (c.text or "").strip():
            problems.append(f"{c.chunk_id}: empty text")
        if len(c.entities) < 3:
            problems.append(f"{c.chunk_id}: <3 entities")
        if len(c.relations) < math.ceil(len(c.entities) / 2):
            problems.append(f"{c.chunk_id}: too few relations")

    if chunks:
        s = chunks[0]
        logger.info("Sample chunk_id=%s | book=%s | topic=%s | %d entities | %d relations",
                    s.chunk_id, s.source.book, s.source.topic, len(s.entities), len(s.relations))
        logger.info("Sample summary: %s", (s.summary or "")[:200])
        # Confirm the book-categorisation fix: chunks must NOT land under "general".
        if s.source.book.lower() == "general":
            problems.append("source.book is 'General' — book categorisation not applied")

    if n > 0 and not problems:
        logger.info("SMOKE TEST PASSED — %d valid chunks. Run: python %s --full",
                    n, Path(__file__).name)
        return n
    if problems:
        logger.error("SMOKE TEST FAILED — %d issue(s):", len(problems))
        for p in problems[:10]:
            logger.error("  - %s", p)
    else:
        logger.error("SMOKE TEST FAILED — 0 chunks produced.")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Chunk the handbook over pages 27–845 (plan / smoke / full).")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true",
                      help="FREE: analyse the range, report blocks/batches. Default.")
    mode.add_argument("--smoke", action="store_true",
                      help="CHEAP: validate on the first --smoke-pages pages (paid).")
    mode.add_argument("--full", action="store_true",
                      help=f"REAL: process the entire {START_PAGE}–{END_PAGE} range (paid).")
    ap.add_argument("--version", default=DEFAULT_VERSION,
                    help="Output version tag for the full run (default: v1).")
    ap.add_argument("--smoke-pages", type=int, default=5,
                    help="Pages from p.27 the smoke test reads (default: 5).")
    args = ap.parse_args()

    # Fail fast on bad config before any paid work (skip for the free plan).
    if args.full or args.smoke:
        try:
            settings.require_for_run()
        except RuntimeError as e:
            logger.error(str(e))
            sys.exit(2)

    if args.full:
        run_pipeline(full=True, version=args.version, smoke_pages=args.smoke_pages)
        sys.exit(0)
    if args.smoke:
        n = run_pipeline(full=False, version=args.version, smoke_pages=args.smoke_pages)
        sys.exit(0 if n > 0 else 1)

    # Default: free plan/analysis.
    n_blocks = plan()
    sys.exit(0 if n_blocks > 0 else 1)


if __name__ == "__main__":
    main()
