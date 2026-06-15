"""
Retroactive post-processing runner.

Applies entity overrides → relation repair → canonicalization to ALL existing
chunk JSON files under a given directory. Overwrites each file in-place.

Usage:
    python run_postprocessing.py                          # default: chunks/
    python run_postprocessing.py chunks/_smoke_27_845     # specific directory
    python run_postprocessing.py --dry-run                # preview without writing
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chunking.schemas.models import MicroChunk
from chunking.postprocessing import postprocess_chunk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("run_postprocessing")


def process_file(path: Path, dry_run: bool = False) -> dict:
    """Load a chunk JSON, apply post-processing, and overwrite.

    Returns a stats dict: {entities_changed, relations_changed, canonicalized}.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    # Snapshot before
    before_types = {e["name"]: e["type"] for e in raw.get("entities", [])}
    before_rels = {(r["source"], r["target"]): r["type"] for r in raw.get("relations", [])}

    # Parse → post-process
    chunk = MicroChunk(**raw)
    postprocess_chunk(chunk)

    # Snapshot after
    after_types = {e.name: e.type for e in chunk.entities}
    after_rels = {(r.source, r.target): r.type for r in chunk.relations}

    # Count changes
    entities_changed = sum(
        1 for name in before_types
        if name in after_types and before_types[name] != after_types[name]
    )
    relations_changed = sum(
        1 for key in before_rels
        if key in after_rels and before_rels[key] != after_rels[key]
    )
    canonicalized = len(before_types) - len(after_types)  # entities that merged

    # Write back
    if not dry_run:
        out = chunk.model_dump(mode="json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    return {
        "entities_changed": entities_changed,
        "relations_changed": relations_changed,
        "canonicalized": max(0, canonicalized),
    }


def main():
    parser = argparse.ArgumentParser(description="Apply post-processing to existing chunk JSONs")
    parser.add_argument("directory", nargs="?", default="chunks",
                        help="Root directory containing chunk JSON files (default: chunks/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without writing files")
    args = parser.parse_args()

    root = Path(args.directory)
    if not root.exists():
        logger.error(f"Directory not found: {root}")
        sys.exit(1)

    json_files = sorted(root.rglob("*.json"))
    # Skip aggregated files (they're copies)
    json_files = [f for f in json_files if "_aggregated" not in str(f)]

    if not json_files:
        logger.warning(f"No chunk JSON files found under {root}")
        sys.exit(0)

    logger.info(f"{'DRY RUN: ' if args.dry_run else ''}Processing {len(json_files)} chunk files under {root}")

    total_entities = 0
    total_relations = 0
    total_canonical = 0
    errors = 0

    for path in json_files:
        try:
            stats = process_file(path, dry_run=args.dry_run)
            total_entities += stats["entities_changed"]
            total_relations += stats["relations_changed"]
            total_canonical += stats["canonicalized"]

            if any(stats.values()):
                logger.info(f"  {path.name}: "
                            f"{stats['entities_changed']} types fixed, "
                            f"{stats['relations_changed']} relations repaired, "
                            f"{stats['canonicalized']} canonicalized")
        except Exception as e:
            errors += 1
            logger.error(f"  {path.name}: {e}")

    logger.info("=" * 60)
    logger.info(f"DONE: {len(json_files)} files processed, {errors} errors")
    logger.info(f"  Entity types fixed:    {total_entities}")
    logger.info(f"  Relations repaired:    {total_relations}")
    logger.info(f"  Entities canonicalized: {total_canonical}")
    if args.dry_run:
        logger.info("  (dry run — no files were modified)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
