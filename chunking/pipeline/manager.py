from chunking.loaders.pdf_loader import PDFLoader
from chunking.loaders.csv_loader import CSVLoader
from chunking.cleaners.text_cleaner import TextCleaner
from chunking.detectors.structure import StructureDetector
from chunking.extractors.semantic import SemanticExtractor
from chunking.llm.retry_engine import ExtractionWithRetry
from chunking.storage.versioned_storage import VersionedStorage
from chunking.postprocessing import postprocess_chunk
from chunking.schemas.models import DocumentMetadata, SemanticBlock
from chunking.config.settings import settings
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# 5 blocks per LLM call → rich context, good chunk quality
BATCH_SIZE = 5
# 4 parallel workers → slow and steady, well under Gemini rate limits
MAX_WORKERS = 4


class DocumentProcessingPipeline:
    def __init__(self):
        self.cleaner = TextCleaner()
        self.detector = StructureDetector()
        self.semantic_extractor = SemanticExtractor()
        self.extractor = ExtractionWithRetry()
        self.storage = VersionedStorage()

    def _done_path(self, block_id: str) -> Path:
        return Path("logs/processed_blocks") / f"{block_id}.done"

    def _mark_done(self, blocks):
        done_dir = Path("logs/processed_blocks")
        done_dir.mkdir(parents=True, exist_ok=True)
        for b in blocks:
            self._done_path(b.block_id).touch()

    @staticmethod
    def _batch_page_range(blocks) -> str:
        """Combine the page numbers of all blocks in a batch into "27" or "27-31".

        Chunks don't map 1:1 to blocks (the LLM re-splits a batch), so a chunk's
        page provenance is the page span of the batch it came from.
        """
        nums = []
        for b in blocks:
            if getattr(b, "page", None):
                for part in str(b.page).split("-"):
                    part = part.strip()
                    if part.isdigit():
                        nums.append(int(part))
        if not nums:
            return ""
        lo, hi = min(nums), max(nums)
        return str(lo) if lo == hi else f"{lo}-{hi}"

    def process_block_batch(self, blocks, version) -> list:
        """Send up to BATCH_SIZE blocks as one LLM call and save results."""
        pending = [b for b in blocks if not self._done_path(b.block_id).exists()]
        if not pending:
            return []

        combined = "\n\n===SECTION BREAK===\n\n".join(
            f"[SECTION: {b.section or 'General'}]\n{b.text}" for b in pending
        )

        structured_data = self.extractor.run(combined)
        chunks = []

        if structured_data:
            # Derive authoritative source info from the block metadata. The book is
            # known from the source file, so it is set unconditionally rather than
            # trusting the LLM (which tends to echo a section header like "General"
            # into source.book, scattering every chunk under chunks/<ver>/general/).
            ref_block = pending[0]
            book_title = ref_block.metadata.book_type or ref_block.metadata.doc_id.replace("_", " ").title()
            fallback_chapter = ref_block.section or "General"
            today = datetime.date.today().isoformat()
            batch_page = self._batch_page_range(pending)
            for chunk in structured_data.chunks:
                chunk.source.book = book_title
                if chunk.source.chapter in ("Unknown", "", "unknown"):
                    chunk.source.chapter = fallback_chapter
                if chunk.source.topic in ("Unknown", "", "unknown"):
                    chunk.source.topic = fallback_chapter
                # Stamp authoritative provenance — never trust the LLM for these.
                chunk.source.page = batch_page
                chunk.source.publication_year = ref_block.metadata.publication_year
                chunk.source.edition = ref_block.metadata.edition
                chunk.source.source_specialty = ref_block.metadata.source_specialty
                chunk.metadata.model = settings.model_primary
                chunk.metadata.version = version
                chunk.metadata.created_at = today
                tokens = chunk.metadata.tokens
                if not isinstance(tokens, int) and str(tokens).strip().isdigit() is False:
                    chunk.metadata.tokens = int(len(chunk.text.split()) * 1.3)
                # Deterministic, content-stable chunk_id = the join key shared by the
                # Pinecone vector and the Neo4j :Chunk node. Idempotent across re-runs.
                # Kept short (doc + content hash) — topic lives in source.topic / the
                # folder name, and long ids blow past Windows MAX_PATH.
                text_hash = hashlib.md5(chunk.text.encode("utf-8")).hexdigest()[:10]
                chunk.chunk_id = f"{ref_block.metadata.doc_id}-{text_hash}"
                # ── Post-processing: entity overrides → relation repair → canonicalization
                postprocess_chunk(chunk)
                self.storage.save_chunk(chunk, index=1, version=version)
                chunks.append(chunk)
            if chunks:
                self._mark_done(pending)
            else:
                logger.error(f"Batch produced 0 valid chunks: {[b.block_id for b in pending]}")
            logger.info(f"Batch {len(pending)} blocks → {len(chunks)} chunks")
        else:
            logger.error(f"Batch failed: {[b.block_id for b in pending]}")

        return chunks

    def _process_blocks_parallel(self, blocks, version, label="") -> list:
        """Split blocks into batches and process all in parallel."""
        batches = [blocks[i:i + BATCH_SIZE] for i in range(0, len(blocks), BATCH_SIZE)]
        total = len(batches)
        logger.info(f"{label}: {len(blocks)} blocks → {total} batches → {MAX_WORKERS} workers")

        all_chunks = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_idx = {
                executor.submit(self.process_block_batch, batch, version): i
                for i, batch in enumerate(batches)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    chunks = future.result()
                    all_chunks.extend(chunks)
                    if (idx + 1) % 10 == 0 or (idx + 1) == total:
                        logger.info(f"{label} progress: {idx+1}/{total} batches done, "
                                    f"{len(all_chunks)} chunks so far")
                except Exception as e:
                    logger.error(f"{label} batch {idx} error: {e}")

        return all_chunks

    def process_pdf(self, file_path: str, doc_id: str, book_type: str, version: str,
                    start_page: int = 1, max_pages: int = None, end_page: int = None,
                    publication_year: int = None, edition: str = None,
                    source_specialty: str = None):
        logger.info(f"Processing PDF: {file_path}")

        loader = PDFLoader(doc_id, book_type, version)
        metadata = DocumentMetadata(
            doc_id=doc_id, book_type=book_type, version=version, source_path=file_path,
            publication_year=publication_year, edition=edition,
            source_specialty=source_specialty,
        )
        raw_pages = loader.load(file_path, start_page=start_page, max_pages=max_pages,
                                end_page=end_page)
        if raw_pages:
            logger.info(f"Loaded {len(raw_pages)} pages "
                        f"(p.{raw_pages[0]['page_num']}–p.{raw_pages[-1]['page_num']}).")
        else:
            logger.warning("Loaded 0 pages — check start_page/end_page range.")

        for page in raw_pages:
            page['text'] = self.cleaner.normalize(page['text'])

        sections = self.detector.segment(raw_pages)
        semantic_blocks = self.semantic_extractor.extract_blocks(sections, metadata)

        all_chunks = self._process_blocks_parallel(semantic_blocks, version, label=doc_id)
        self.storage.save_chunks(all_chunks, version=version)
        self._report_run(semantic_blocks, all_chunks, version, doc_id)
        return all_chunks

    def _report_run(self, blocks, chunks, version, doc_id):
        """Log a clean end-of-run summary and write a manifest of failed blocks.

        A block counts as done iff it has a .done marker. Failures (LLM or save
        errors) leave none, so they're retried on the next --full run; this writes
        them out so an unattended run is auditable instead of silently lossy.
        """
        failed = [b for b in blocks if not self._done_path(b.block_id).exists()]
        ok = len(blocks) - len(failed)
        logger.info("=" * 60)
        logger.info(f"SUMMARY {doc_id}: {ok}/{len(blocks)} blocks succeeded, "
                    f"{len(failed)} failed, {len(chunks)} chunks written.")
        if failed:
            pages = sorted({b.page for b in failed if b.page})
            manifest = Path("logs") / f"failed_blocks_{version}_{doc_id}.json"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(
                json.dumps([{"block_id": b.block_id, "page": b.page, "section": b.section}
                            for b in failed], indent=2),
                encoding="utf-8",
            )
            logger.warning(f"{len(failed)} blocks failed on pages {pages}. "
                           f"Manifest: {manifest}. Re-run --full to retry them.")
        else:
            logger.info("All blocks processed cleanly — no failures.")
        logger.info("=" * 60)

    def process_csv(self, file_path: str, book_type: str, version: str):
        logger.info(f"Processing CSV: {file_path}")
        loader = CSVLoader()
        records = loader.load(file_path)
        logger.info(f"Loaded {len(records)} records.")

        # Build all semantic blocks from every CSV row first, then batch in parallel
        all_blocks = []
        for record in records:
            doc_id = record["doc_id"]
            text = record["text"]
            metadata = DocumentMetadata(
                doc_id=doc_id, book_type=book_type, version=version, source_path=file_path
            )
            cleaned = self.cleaner.normalize(text)
            fake_pages = [{"page_num": 1, "text": cleaned, "layout": []}]
            sections = self.detector.segment(fake_pages)
            blocks = self.semantic_extractor.extract_blocks(sections, metadata)
            if not blocks:
                blocks = [SemanticBlock(
                    block_id=f"{doc_id}-blk-1",
                    text=cleaned[:2000],
                    section="General",
                    metadata=metadata
                )]
            all_blocks.extend(blocks)

        logger.info(f"CSV: {len(all_blocks)} total blocks from {len(records)} records.")
        all_chunks = self._process_blocks_parallel(all_blocks, version, label="csv")
        self.storage.save_chunks(all_chunks, version=version)
        logger.info(f"CSV complete. Total chunks: {len(all_chunks)}")
