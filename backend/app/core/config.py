"""Settings, loaded once from the environment.

Everything configurable lives here so you never hunt through modules for a
hardcoded model name or path. Pydantic reads the .env file, validates types,
and fails loudly at startup if something required is missing — which is far
better than failing halfway through a demo.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> parents[3] is the project root
ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- LLM (cloud-hosted, as the brief requires) --------------------- #
    groq_api_key: str = ""
    llm_model: str = "openai/gpt-oss-120b"

    # --- Qdrant -------------------------------------------------------- #
    # Three ways to run, set by QDRANT_MODE in .env:
    #
    #   "embedded" — Qdrant runs INSIDE this Python process and writes to
    #                qdrant_path. No Docker, no server, no network. Best for
    #                developing on a laptop without Docker installed.
    #   "server"   — a Qdrant you run yourself (Docker) at qdrant_url.
    #   "cloud"    — Qdrant Cloud. Set qdrant_url to your cluster URL and
    #                qdrant_api_key to the key from the dashboard.
    #
    # Note: "server" and "cloud" use identical code — cloud is just a remote
    # server with an API key. Only "embedded" takes a different path.
    qdrant_mode: str = "embedded"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_path: Path = ROOT / "qdrant_storage"
    qdrant_collection: str = "medibot_chunks"

    # --- Models -------------------------------------------------------- #
    # Dense: turns text into a 384-number vector capturing MEANING.
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Sparse: BM25-style keyword matching, stored as a sparse vector.
    sparse_model: str = "Qdrant/bm25"
    # Cross-encoder: reads query+chunk together and scores relevance.
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    dense_dim: int = 384  # must match dense_model's output size

    # --- Retrieval tuning ---------------------------------------------- #
    retrieve_top_k: int = 10  # broad candidate set from hybrid search
    rerank_top_k: int = 3     # narrowed set actually shown to the LLM

    # --- Chunking ------------------------------------------------------ #
    max_tokens_per_chunk: int = 512

    # --- Paths --------------------------------------------------------- #
    documents_dir: Path = ROOT / "data" / "documents"
    sqlite_path: Path = ROOT / "data" / "mediassist.db"

    # --- Auth ---------------------------------------------------------- #
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    token_expiry_minutes: int = 120


settings = Settings()
