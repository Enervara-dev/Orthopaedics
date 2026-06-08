import logging
from typing import Optional
from chunking.config.settings import settings
from chunking.llm.providers.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


class LLMEngine:
    """Coordinates extraction over a primary and fallback Gemini model.

    Retries and backoff are owned by ExtractionWithRetry; this just routes a single
    call to the primary model, or the fallback model on the final attempt.
    """

    def __init__(self):
        self.primary_provider = GeminiClient(settings.gemini_api_key, settings.model_primary)
        self.fallback_provider = GeminiClient(settings.gemini_api_key, settings.model_fallback)

    def extract_structured_data(self, text: str, schema_json: str,
                                force_fallback: bool = False) -> tuple[Optional[str], str]:
        provider = self.fallback_provider if force_fallback else self.primary_provider
        return provider.generate_json(text, schema_json)
