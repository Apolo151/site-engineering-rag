"""Smoke tests for the web GUI wrapper. All exercise retrieval-only mode, so
no API key and no network are required."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

fastapi_testclient = pytest.importorskip("fastapi.testclient")

from rag.web import app  # noqa: E402

client = fastapi_testclient.TestClient(app)


def test_index_page_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Site-Engineering RAG Assistant" in r.text


def test_config_endpoint():
    r = client.get("/api/config")
    assert r.status_code == 200
    body = r.json()
    assert "llm_configured" in body
    assert len(body["sample_questions"]) == 8
    assert body["shelves"], "expected at least one shelf tag from the index"
    assert body["index_error"] is None


def test_ask_retrieval_only_returns_sources():
    r = client.post(
        "/api/ask",
        json={
            "question": "How many slots does the 1830 PSS-8 shelf provide?",
            "call_llm": False,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["retrieved"], "expected retrieved passages"
    first = body["retrieved"][0]
    assert "score" in first and "section_path" in first
    assert "CONTEXT:" in body["prompt"]
    assert body["answer"] == ""


def test_ask_nonsense_hits_threshold_abstain():
    r = client.post(
        "/api/ask",
        json={"question": "what is the CEO's personal phone number", "call_llm": False},
    )
    assert r.status_code == 200
    assert r.json()["refused_by"] == "threshold"


def test_ask_empty_question_rejected():
    r = client.post("/api/ask", json={"question": "   ", "call_llm": False})
    assert r.status_code == 422


def test_ask_shelf_filter_restricts_sources():
    """Part E: the /api/ask shelf_filter field, exposed by the page's shelf
    dropdown, actually restricts retrieval rather than being decorative."""
    r = client.post(
        "/api/ask",
        json={
            "question": "Which fan units are supported on the 1830 PSS-32 shelf?",
            "shelf_filter": "PSS-32",
            "call_llm": False,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["retrieved"], "expected PSS-32 chunks to be retrieved"
    assert all(r["shelf"] == "PSS-32" for r in body["retrieved"])
