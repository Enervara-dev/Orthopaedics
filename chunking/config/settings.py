from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str = ""
    model_primary: str = "gemini-2.5-flash-lite"
    model_fallback: str = "gemini-2.5-flash-lite"
    storage_base_path: str = "chunks"

    # Pinecone (vector DB) — for ingest_pinecone.py
    pinecone_api_key: str = ""
    pinecone_index_name: str = ""
    pinecone_embedding_model: str = "llama-text-embed-v2"

    # Neo4j (knowledge graph) — for ingest_neo4j.py
    neo4j_uri: str = ""
    neo4j_username: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    def require_for_run(self) -> None:
        """Fail fast at startup if the config can't drive a real extraction run.

        Importing settings stays cheap (offline tests, tooling); the entry points
        call this before processing so a missing key surfaces immediately instead
        of after hundreds of paid blocks.
        """
        problems = []
        if not self.model_primary.strip():
            problems.append("MODEL_PRIMARY is empty")
        if not self.model_fallback.strip():
            problems.append("MODEL_FALLBACK is empty")
        # Gemini is the wired first-class provider in this build.
        uses_gemini = any("gemini" in m.lower() for m in (self.model_primary, self.model_fallback))
        if uses_gemini and not self.gemini_api_key.strip():
            problems.append("GEMINI_API_KEY is missing (required for a gemini model)")
        if problems:
            raise RuntimeError(
                "Invalid configuration — fix your .env:\n  - " + "\n  - ".join(problems)
            )


settings = Settings()
