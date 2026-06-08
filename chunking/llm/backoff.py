"""Exponential backoff with jitter for LLM retries.

Retries were firing ~226ms apart (no delay between attempts), hammering the API.
This implements AWS-style "full jitter": wait a random duration in
[0, min(cap, base * 2**attempt)] so concurrent workers don't retry in lockstep
and transient errors / rate limits get real breathing room.
"""

import logging
import random
import time

logger = logging.getLogger(__name__)


def backoff_delay(attempt: int, base: float = 1.0, cap: float = 60.0) -> float:
    """Seconds to wait before retry `attempt` (0-based), full-jitter."""
    expo = min(cap, base * (2 ** attempt))
    return random.uniform(0, expo)


def sleep_backoff(attempt: int, base: float = 1.0, cap: float = 60.0, reason: str = "") -> float:
    """Sleep with full-jitter exponential backoff; returns the slept duration."""
    delay = backoff_delay(attempt, base, cap)
    suffix = f" — {reason}" if reason else ""
    logger.info(f"Backing off {delay:.1f}s before retry (attempt {attempt + 1}){suffix}")
    time.sleep(delay)
    return delay
