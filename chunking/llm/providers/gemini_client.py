import logging
import time
import re
from typing import Optional
from openai import OpenAI
from chunking.llm.providers.base import BaseLLMProvider
from chunking.llm.prompts import SYSTEM_PROMPT, build_user_content
from chunking.llm.backoff import backoff_delay

logger = logging.getLogger(__name__)

# Substrings that mark a retryable (transient) error. Anything else — bad request,
# auth, invalid argument — fails fast; retrying it just wastes time.
_TRANSIENT = ("429", "RESOURCE_EXHAUSTED", "500", "502", "503", "504",
              "UNAVAILABLE", "INTERNAL", "deadline", "timeout", "Timeout",
              "Connection", "connection")


class GeminiClient(BaseLLMProvider):
    """Google Gemini via the OpenAI-compatible endpoint (paid tier — no throttle needed)."""

    def __init__(self, api_key: str, model_name: str):
        super().__init__(api_key, model_name)
        self.client = OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=self.api_key,
            max_retries=0,
        )

    def generate_json(self, prompt: str, schema_json: str) -> tuple[Optional[str], str]:
        user_content = build_user_content(schema_json, prompt)
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                return response.choices[0].message.content, ""
            except Exception as e:
                err_str = str(e)
                is_last = attempt == max_retries - 1
                if not any(s in err_str for s in _TRANSIENT) or is_last:
                    logger.warning(f"Gemini call failed: {e}")
                    return None, err_str

                # Exponential backoff with jitter; honor the server's retryDelay as
                # a floor when it sends one (429 / RESOURCE_EXHAUSTED).
                match = re.search(r'"retryDelay"\s*:\s*"(\d+)s"', err_str)
                floor = int(match.group(1)) if match else 0.0
                delay = max(floor, backoff_delay(attempt, base=2.0, cap=60.0))
                logger.warning(
                    f"Transient Gemini error (attempt {attempt+1}/{max_retries}); "
                    f"backing off {delay:.1f}s: {err_str[:120]}"
                )
                time.sleep(delay)

        return None, "Failed after all retries"
