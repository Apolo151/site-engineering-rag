"""A small FastAPI wrapper around the RAG pipeline, plus a single static page.

Run it with one command:

    uv run rag-web            # -> http://127.0.0.1:8000

The pipeline (retrieval + prompt assembly + generation + abstain gate) lives in
``rag.pipeline``; this module only loads the persisted index once at startup,
serves ``static/index.html``, and exposes two JSON endpoints the page calls.
It works without an API key in retrieval-only mode.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from rag import config
from rag.pipeline import AnswerResult, answer_question

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"

# The fixed evaluation questions double as the page's sample prompts. Import
# them from the eval package when it is importable (source checkout / editable
# install); fall back to a copy so the GUI still works from a bare wheel.
try:  # pragma: no cover - trivial import shim
    import sys

    sys.path.insert(0, str(config.REPO_ROOT))
    from eval.questions import QUESTIONS as _QUESTIONS

    SAMPLE_QUESTIONS = [q.question for q in _QUESTIONS]
except Exception:  # pragma: no cover
    SAMPLE_QUESTIONS = [
        "How many slots does the 1830 PSS-8 shelf provide, and what is its rack-unit (RU) footprint?",
        "What rack-unit footprint does the 1830 PSS-32 shelf have, and how many slots does it provide?",
        "What are the two software load-lines supported by the 1830 PSS system?",
        "Which fan units are supported on the 1830 PSS-32 shelf?",
        "Which fan unit(s) are used on the 1830 PSS-16II shelf?",
        "Name the power filter cards supported on the 1830 PSS-8 shelf.",
        "What is the required horizontal rack aperture for mounting a 1830 PSS-8 shelf, "
        "and which common aperture size is explicitly NOT supported?",
        "What is the maximum optical reach, in kilometers, of the 1830 PSS-8 shelf without amplification?",
    ]


# --- index loading (once, at import) -----------------------------------------

_INDEX = None
_INDEX_ERROR: str | None = None
_SHELVES: list[str] = []


def _load() -> None:
    global _INDEX, _INDEX_ERROR, _SHELVES
    try:
        from rag.embed import load_index

        _INDEX = load_index()
        _SHELVES = sorted({c["shelf"] for c in _INDEX.chunks if c.get("shelf")})
    except Exception as exc:  # FileNotFoundError, RuntimeError (stale index), ...
        _INDEX_ERROR = f"{type(exc).__name__}: {exc}"


_load()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Load the sentence-transformer off the request path so the first real
    query isn't the one that pays the model-load cost."""
    if _INDEX is not None:

        def _warm() -> None:
            try:
                from rag.retrieve import _get_model

                _get_model(_INDEX.manifest["model_name"])
            except Exception:
                pass

        await run_in_threadpool(_warm)
    yield


app = FastAPI(
    title="Site-Engineering RAG Assistant",
    docs_url=None,
    redoc_url=None,
    lifespan=_lifespan,
)


# --- API --------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str
    k: int = Field(default=config.DEFAULT_TOP_K)
    shelf_filter: str | None = None
    call_llm: bool = True


def _serialize(result: AnswerResult, *, note: str | None = None) -> dict:
    return {
        "question": result.question,
        "answer": result.answer,
        "refused_by": result.refused_by,
        "top1_score": result.top1_score,
        "call_llm": bool(result.answer) and result.refused_by != "threshold",
        "note": note,
        "prompt": result.prompt,
        "retrieved": [
            {
                "score": r.score,
                "chunk_id": r.chunk["chunk_id"],
                "section_number": r.chunk["section_number"],
                "section_title": r.chunk["section_title"],
                "section_path": r.chunk["section_path"],
                "page_start": r.chunk["page_start"],
                "page_end": r.chunk["page_end"],
                "shelf": r.chunk.get("shelf"),
                "text": r.chunk["text"],
            }
            for r in result.retrieved
        ],
    }


@app.get("/api/config")
async def api_config() -> dict:
    return {
        "llm_configured": bool(config.OPENAI_API_KEY),
        "default_k": config.DEFAULT_TOP_K,
        "model": config.RAG_MODEL,
        "shelves": _SHELVES,
        "sample_questions": SAMPLE_QUESTIONS,
        "index_error": _INDEX_ERROR,
    }


@app.post("/api/ask")
async def api_ask(req: AskRequest):
    if _INDEX is None:
        return JSONResponse(
            status_code=503,
            content={"error": f"Index not loaded. {_INDEX_ERROR}"},
        )

    question = req.question.strip()
    if not question:
        return JSONResponse(status_code=422, content={"error": "Question is empty."})

    k = max(1, min(10, req.k))
    shelf = (req.shelf_filter or "").strip() or None
    call_llm = req.call_llm and bool(config.OPENAI_API_KEY)
    note = None
    if req.call_llm and not config.OPENAI_API_KEY:
        note = "No API key configured - showing retrieval only."

    from rag.llm import LLMNotConfigured

    try:
        result = await run_in_threadpool(
            answer_question, question, _INDEX, k, shelf, call_llm
        )
    except LLMNotConfigured as exc:
        note = f"{exc} Showing retrieval only."
        result = await run_in_threadpool(
            answer_question, question, _INDEX, k, shelf, False
        )
    except Exception as exc:  # upstream/network failure from the LLM call
        return JSONResponse(
            status_code=502,
            content={"error": f"Generation failed: {type(exc).__name__}: {exc}"},
        )

    return _serialize(result, note=note)


# --- static page -----------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    if INDEX_HTML.is_file():
        return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>index.html missing</h1>", status_code=500)


@app.get("/favicon.ico")
async def favicon() -> Response:
    return Response(status_code=204)


def main() -> None:
    import uvicorn

    host = os.environ.get("RAG_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("RAG_WEB_PORT", "8000"))
    uvicorn.run("rag.web:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
