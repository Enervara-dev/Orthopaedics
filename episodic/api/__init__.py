"""Embedded wiring for the episodic memory layer.

The standalone FastAPI app/routes were removed — episodic memory is consumed
in-process by the GraphRAG pipeline via `build_container()`.
"""

from episodic.api.dependencies import build_container

__all__ = ["build_container"]
