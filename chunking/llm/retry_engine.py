import json
from chunking.llm.client import LLMEngine
from chunking.llm.backoff import sleep_backoff
from chunking.validators.schema_validator import OutputValidator
from chunking.schemas.models import ExtractedClinicalData
from typing import Optional

# Compact, hand-written schema sent on every call. The full MicroChunk JSON schema
# was ~640 tokens of bloat per call; this conveys the same shape in a fraction of
# that. Entities are lean (name + type only) — normalized_name/properties are derived.
COMPACT_SCHEMA = json.dumps({
    "chunks": [{
        "source": {"chapter": "broad section", "topic": "specific concept title"},
        "specialties": ["pulmonology", "cardiology"],
        "text": "clean clinical prose (~150-350 tokens)",
        "entities": [{
            "name": "canonical name (abbreviations expanded)",
            "type": "disease|symptom|clinical_finding|lab_finding|metabolic_state|physiological_state|drug|test|procedure|...",
            "aliases": ["SYNONYM", "ABBREV"],
        }],
        "relations": [{
            "source": "entity name", "target": "entity name",
            "type": "causes|treats|manifests_as|diagnosed_by|assesses|...",
            "qualifiers": {"onset": "instantaneous|acute|subacute|chronic"},
        }],
        "summary": "1-2 sentences",
        "clinical_significance": "1 sentence",
        "metadata": {"tokens": 0, "model": "", "quality_check": "passed"},
    }]
}, separators=(",", ":"))


class ExtractionWithRetry:
    def __init__(self):
        self.llm = LLMEngine()
        self.validator = OutputValidator()

    def run(self, text: str) -> Optional[ExtractedClinicalData]:
        schema_str = COMPACT_SCHEMA

        max_retries = 3
        current_error = ""
        prompt_text = text

        for attempt in range(max_retries):
            # Exponential backoff with jitter BEFORE every retry (not the first
            # attempt) so failed calls don't re-fire ~226ms apart.
            if attempt > 0:
                sleep_backoff(attempt - 1, base=2.0, cap=60.0, reason=current_error[:80])

            force_fallback = (attempt == max_retries - 1)
            raw_output, llm_err = self.llm.extract_structured_data(prompt_text, schema_str, force_fallback=force_fallback)

            if llm_err:
                current_error = llm_err
                continue

            try:
                structured_data = self.validator.validate(raw_output)
                return structured_data
            except Exception as e:
                current_error = str(e)
                # Send a terse correction, NOT the full bad output (which would
                # re-bill all those output tokens as input on the retry).
                prompt_text = f"""{text}

Your previous response failed validation: {current_error[:300]}
Return corrected JSON. Each chunk needs: text, >=3 entities (name+type), relations >= half the entities, summary, clinical_significance."""

        # If it reaches here, it failed entirely. Log it for auditing.
        import hashlib
        from pathlib import Path
        failed_dir = Path("logs/failed_blocks")
        failed_dir.mkdir(parents=True, exist_ok=True)
        # Stable content hash (built-in hash() is randomized per process).
        block_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
        error_file = failed_dir / f"failed_{block_hash}.txt"
        with open(error_file, "w", encoding="utf-8") as f:
            f.write(f"Failed extraction after {max_retries} attempts.\nError: {current_error}\n\nBad Output:\n{raw_output if 'raw_output' in locals() else 'None'}\n\nText:\n{text}")
            
        return None
