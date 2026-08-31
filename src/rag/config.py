"""Shared configuration and paths for the RAG pipeline."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Repo layout -----------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"
INDEX_DIR = REPO_ROOT / "index"
EVAL_DIR = REPO_ROOT / "eval"

INTERIM_MARKDOWN = INTERIM_DIR / "1830_pp47-166.md"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"

INDEX_EMBEDDINGS = INDEX_DIR / "embeddings.npy"
INDEX_CHUNKS = INDEX_DIR / "chunks.jsonl"
INDEX_MANIFEST = INDEX_DIR / "manifest.json"

DEFAULT_PDF_PATH = Path(
    os.environ.get(
        "RAG_SOURCE_PDF",
        "/mnt/files/Downloads/Linux/1830_Technical_Description.pdf",
    )
)

# --- Scope: Chapter 1 (System concept) + Chapter 2 up through Power filters --

FIRST_PAGE = 47
LAST_PAGE = 166

# --- Embedding model ---------------------------------------------------------

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Chunking targets ---------------------------------------------------------

CHUNK_TARGET_WORDS = 220  # soft cap; MiniLM truncates at ~190 words (256 wordpieces)
CHUNK_MIN_WORDS = 60  # sections shorter than this get merged forward
CHUNK_OVERLAP_WORDS = 40

# --- Retrieval ------------------------------------------------------------

DEFAULT_TOP_K = 5
ABSTAIN_SIMILARITY_THRESHOLD = 0.25

# --- Generation -------------------------------------------------------------

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")
RAG_MODEL = os.environ.get("RAG_MODEL", "openai/gpt-oss-120b")

REFUSAL_TEXT = "Not found in the provided document."
