"""
Entry point for the chunking pipeline (PDF reference texts → micro-chunks).

Processes every *.pdf in  dataset/  into validated micro-chunks
(entities + relations + summary + clinical significance).

Usage:
    python chunker.py
    python chunker.py --version v2
    python chunker.py --dataset-dir path/to/pdfs
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Run from the project root so `import chunking.*` and relative output/ paths resolve.
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# Log lines contain non-ASCII (e.g. "→"); force UTF-8 so the cp1252 Windows
# console doesn't raise UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "chunker.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("chunker")

from chunking.pipeline.manager import DocumentProcessingPipeline  # noqa: E402

DATASET_DIR = PROJECT_ROOT / "dataset"
DEFAULT_VERSION = "v1"


def _slugify(name: str) -> str:
    """File stem -> stable doc_id, e.g. 'Oxfordhandbkrespirmed (1)' -> 'oxfordhandbkrespirmed_1'."""
    slug = "".join(c.lower() if c.isalnum() else "_" for c in name)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "document"


def _titleize(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").strip().title()


def main():
    parser = argparse.ArgumentParser(description="Orthopaedics PDF chunking pipeline")
    parser.add_argument("--version", default=DEFAULT_VERSION, help="Output version tag (default: v1)")
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR),
                        help="Directory to read PDFs from (default: ./dataset)")
    parser.add_argument("--start-page", type=int, default=1,
                        help="First PDF page to process (1-based, inclusive, default: 1)")
    parser.add_argument("--end-page", type=int, default=None,
                        help="Last PDF page to process (1-based, inclusive, default: last page).")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="Process at most this many pages (default: all). Useful for quick test runs.")
    parser.add_argument("--publication-year", type=int, default=None,
                        help="True publication/guideline year of the source (for recency ranking).")
    parser.add_argument("--edition", default=None, help="Source edition label, e.g. '3rd edition'.")
    parser.add_argument("--source-specialty", default=None,
                        help="The book's home specialty — becomes the Neo4j parent node.")
    args = parser.parse_args()

    # Fail fast on bad config before loading any PDFs or making paid calls.
    from chunking.config.settings import settings
    try:
        settings.require_for_run()
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(2)

    dataset_dir = Path(args.dataset_dir).resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(dataset_dir.glob("*.pdf"))
    if not pdfs:
        logger.error(
            "No PDFs found in %s\n"
            "-> Drop your orthopaedics reference PDFs into that folder and re-run.",
            dataset_dir,
        )
        sys.exit(1)

    logger.info("Found %d PDF(s) in %s", len(pdfs), dataset_dir)
    pipeline = DocumentProcessingPipeline()

    for i, pdf in enumerate(pdfs, start=1):
        doc_id = _slugify(pdf.stem)
        book_type = _titleize(pdf.stem)
        logger.info("=" * 60)
        logger.info("SOURCE %d/%d: %s  (doc_id=%s)", i, len(pdfs), pdf.name, doc_id)
        logger.info("=" * 60)
        try:
            pipeline.process_pdf(
                file_path=str(pdf),
                doc_id=doc_id,
                book_type=book_type,
                version=args.version,
                start_page=args.start_page,
                end_page=args.end_page,
                max_pages=args.max_pages,
                publication_year=args.publication_year,
                edition=args.edition,
                source_specialty=args.source_specialty,
            )
        except Exception:
            logger.exception("Failed to process %s — continuing with the rest.", pdf.name)

    logger.info("Done. Results in chunks/%s/.", args.version)


if __name__ == "__main__":
    main()
