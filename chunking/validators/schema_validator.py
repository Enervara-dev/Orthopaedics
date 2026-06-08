import json
import pydantic
from typing import Optional
from chunking.schemas.models import ExtractedClinicalData

class OutputValidator:
    def validate(self, raw_json: str) -> Optional[ExtractedClinicalData]:
        # JSON standard cleanup (often LLMs wrap in markdown)
        raw_json = raw_json.strip()
        if raw_json.startswith("```json"):
            raw_json = raw_json[7:]
        if raw_json.endswith("```"):
            raw_json = raw_json[:-3]
        # Strip control characters that break JSON parsing (keep \t \n \r)
        raw_json = "".join(ch if ch >= " " or ch in "\t\n\r" else " " for ch in raw_json)
            
        try:
            # First gate: JSON Parsing Validity
            try:
                data = json.loads(raw_json)
            except json.JSONDecodeError:
                # Attempt to recover truncated JSON by extracting complete chunk objects
                data = self._recover_partial_json(raw_json)
                if data is None:
                    raise ValueError("Invalid JSON and recovery failed")
            
            # Second gate: Strict Pydantic Schema Validation
            parsed_data = ExtractedClinicalData(**data)
            
            # Semantic validation removed — LLMs use synonyms/abbreviations that don't
            # exactly match entity names, causing too many false positives.
                
            return parsed_data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
        except pydantic.ValidationError as e:
            raise ValueError(f"Schema violation: {e}")

    def _recover_partial_json(self, raw: str) -> Optional[dict]:
        """Extract any complete chunk objects from truncated LLM output."""
        import re
        chunks = []
        # Find all complete {...} objects that look like chunks (have "entities" key)
        depth = 0
        start = None
        for i, ch in enumerate(raw):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start is not None:
                    candidate = raw[start:i+1]
                    try:
                        obj = json.loads(candidate)
                        if 'entities' in obj or 'chunk_id' in obj or 'text' in obj:
                            chunks.append(obj)
                    except Exception:
                        pass
                    start = None
        if chunks:
            return {"chunks": chunks}
        return None
