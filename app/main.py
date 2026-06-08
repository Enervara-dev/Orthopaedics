"""
FastAPI wrapper for the GraphRAG pulmonology assistant.

Exposes a thin HTTP surface over `GraphRAGPipeline`:
    GET  /health          → liveness/readiness
    POST /chat            → { message, session_id?, user_id? } → { answer, session_id }

Run locally:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
On Render the start command + $PORT come from render.yaml.

Design notes
------------
- ONE pipeline is built at startup and reused (constructing it opens Pinecone/
  Neo4j/Gemini clients — expensive to do per request).
- `GraphRAGPipeline.run()` is SYNCHRONOUS and its memory layer calls
  `asyncio.run()`, which cannot run inside a live event loop. So `/chat` is a
  plain `def` endpoint — FastAPI runs sync endpoints in a threadpool, off the
  event loop, which is exactly what the memory layer needs.
- The pipeline (and its episodic event loop + client connections) is not proven
  thread-safe, so calls are serialized with a lock. For real concurrency run
  multiple uvicorn worker PROCESSES (each gets its own pipeline) — see HANDOFF §9.
"""

from __future__ import annotations

import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

# ── Bootstrap: project root on path + Windows TLS + UTF-8 ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Route TLS through the OS trust store BEFORE any Pinecone/Neo4j/Gemini import
# (fixes "unable to get local issuer certificate" behind a proxy/AV/VPN).
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except Exception:
        pass

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from graphrag.config.settings import ConfigError, settings


# ── Request/response models ───────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's message/question.")
    session_id: str = Field("default", description="Conversation/memory key.")
    user_id: str | None = Field(
        None, description="Enables episodic (long-term) memory when provided."
    )


class ChatResponse(BaseModel):
    answer: str
    session_id: str


# ── App lifecycle: build the pipeline once ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail fast on missing config before constructing any clients.
    try:
        settings.validate_required("api")
    except ConfigError as e:
        raise RuntimeError(str(e)) from e

    from graphrag.pipeline.graphrag_pipeline import GraphRAGPipeline

    app.state.pipeline = GraphRAGPipeline()
    app.state.lock = threading.Lock()
    try:
        yield
    finally:
        pipeline = getattr(app.state, "pipeline", None)
        if pipeline is not None:
            pipeline.close()


app = FastAPI(
    title="Enervera Pulmonology Assistant",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — from settings.CORS_ORIGINS (comma-separated, or "*").
_origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Enforce X-API-Key only when API_KEY is configured; open otherwise."""
    if settings.API_KEY and x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key.")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
def chat(req: ChatRequest, request: Request) -> ChatResponse:
    pipeline = request.app.state.pipeline
    lock = request.app.state.lock
    # Sync endpoint → runs in a threadpool (off the event loop), so the memory
    # layer's asyncio.run() works. Serialized: the shared pipeline isn't
    # thread-safe (scale with multiple worker processes instead).
    with lock:
        try:
            answer = pipeline.run(
                query_text=req.message,
                session_id=req.session_id,
                user_id=req.user_id,
            )
        except Exception as exc:  # never leak a stack trace to the caller
            raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    return ChatResponse(answer=answer or "", session_id=req.session_id)
