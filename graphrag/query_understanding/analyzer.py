import json
import logging

from graphrag.config.settings import settings
from graphrag.domain import GATEKEEPER_SYSTEM_PROMPT
from graphrag.llm.gemini_client import (
    DEFAULT_LITE_MODEL,
    generate_text,
    generate_text_async,
)

logger = logging.getLogger(__name__)


# The gatekeeper prompt is domain-specific — it lives in graphrag/domain/prompts.py.
# Kept bound to the module-level name SYSTEM_PROMPT for backward compatibility.
SYSTEM_PROMPT = GATEKEEPER_SYSTEM_PROMPT


def _normalize_analysis(analysis: dict) -> dict:
    """
    Coerce the LLM's JSON into the strict shape the routing + terminal-state
    logic depends on (booleans must be real booleans, intent a clean string,
    followup_questions a list).
    """
    if not isinstance(analysis, dict):
        return {}

    nf = analysis.get("needs_followup")
    if isinstance(nf, str):
        analysis["needs_followup"] = nf.strip().lower() in ("true", "1", "yes")
    else:
        analysis["needs_followup"] = bool(nf)

    intent = analysis.get("intent")
    if isinstance(intent, str):
        analysis["intent"] = intent.strip().lower()

    if not isinstance(analysis.get("followup_questions"), list):
        analysis["followup_questions"] = []

    return analysis


class MedicalQueryAnalyzer:
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not set in .env")
        self.model = settings.QUERY_ANALYZER_MODEL or DEFAULT_LITE_MODEL

    def analyze(self, query_text: str) -> dict:
        if not self.api_key:
            return {"error": "API key missing"}

        try:
            content = generate_text(
                query_text,
                model=self.model,
                system_instruction=SYSTEM_PROMPT,
                temperature=0,
                json_mode=True,
            )
        except Exception as e:
            logger.error(f"Error during query analysis: {e}")
            return {}

        if not content:
            logger.error("LLM returned empty content for query analysis.")
            return {}

        try:
            return _normalize_analysis(json.loads(content))
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM: {e}\nRaw output: {content}")
            return {}

    async def aanalyze(self, query_text: str) -> dict:
        """Async sibling of analyze(). Required by the FastAPI request path."""
        if not self.api_key:
            return {"error": "API key missing"}

        try:
            content = await generate_text_async(
                query_text,
                model=self.model,
                system_instruction=SYSTEM_PROMPT,
                temperature=0,
                json_mode=True,
            )
        except Exception as e:
            logger.error(f"Error during async query analysis: {e}")
            return {}

        if not content:
            logger.error("LLM returned empty content for query analysis.")
            return {}

        try:
            return _normalize_analysis(json.loads(content))
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM: {e}\nRaw output: {content}")
            return {}
