"""
Configuration facade for the episodic memory layer.

Reads from the central graphrag.config.settings.Settings (which loads .env)
so this module never reaches for os.getenv directly. Future config knobs
should be added to Settings and re-exported here.
"""

from __future__ import annotations

from graphrag.config.settings import settings


class EpisodicConfig:
    # The episodic index may live in a different Pinecone account, so prefer its
    # dedicated key (EPSIODIC_API_KEY) and fall back to the main key if unset.
    PINECONE_API_KEY = settings.EPISODIC_PINECONE_API_KEY or settings.PINECONE_API_KEY
    PINECONE_INDEX_NAME = settings.PINECONE_EPISODIC_INDEX_NAME

    EXTRACTION_MODEL = settings.EPISODIC_EXTRACTION_MODEL
    CLARIFICATION_MODEL = settings.EPISODIC_CLARIFICATION_MODEL
    CONTRADICTION_MODEL = settings.EPISODIC_CONTRADICTION_MODEL
    COMPRESSION_MODEL = settings.EPISODIC_COMPRESSION_MODEL

    DEFAULT_TOP_K = settings.EPISODIC_DEFAULT_TOP_K
    DEFAULT_RETURN_K = settings.EPISODIC_DEFAULT_RETURN_K
    DECAY_HALF_LIFE_DAYS = settings.EPISODIC_DECAY_HALF_LIFE_DAYS
    MAX_CLARIFICATIONS_PER_TURN = settings.EPISODIC_MAX_CLARIFICATIONS_PER_TURN

    # Composite ranker weights — sum to 1.0 by convention.
    RANK_W_SIMILARITY = 0.45
    RANK_W_RECENCY    = 0.20
    RANK_W_PRIORITY   = 0.15
    RANK_W_CONFIDENCE = 0.10
    RANK_W_RECURRENCE = 0.10

    PINECONE_EMBED_MODEL = "llama-text-embed-v2"
    PINECONE_DIMENSION = 1024
    PINECONE_CLOUD = "aws"
    PINECONE_REGION = "us-east-1"
