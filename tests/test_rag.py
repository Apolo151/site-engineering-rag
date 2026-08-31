import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.embed import LoadedIndex  # noqa: E402
from rag.headings import parse_heading  # noqa: E402
from rag.retrieve import cosine_similarity, retrieve, search  # noqa: E402


# --- Retrieval: manual search agrees with explicit cosine similarity -------


def test_search_matches_explicit_cosine_similarity():
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(50, 16)).astype("float32")
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    query = rng.normal(size=16).astype("float32")
    query /= np.linalg.norm(query)

    results = search(query, matrix, k=5)
    assert len(results) == 5

    expected_scores = sorted(
        (cosine_similarity(query, matrix[i]) for i in range(matrix.shape[0])), reverse=True
    )[:5]
    got_scores = [score for _, score in results]
    assert got_scores == pytest.approx(expected_scores, abs=1e-5)

    # Scores must be strictly descending (top-k really is sorted).
    assert got_scores == sorted(got_scores, reverse=True)


def test_search_respects_mask():
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(10, 8)).astype("float32")
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    query = matrix[3].copy()  # best match is row 3

    mask = np.ones(10, dtype=bool)
    mask[3] = False  # exclude the best match

    results = search(query, matrix, k=3, mask=mask)
    assert all(i != 3 for i, _ in results)


# --- Part E: metadata (shelf) filtering ------------------------------------
#
# retrieve() layers shelf_filter on top of search()'s generic mask: chunks
# are excluded by an exact metadata match, not by score. The synthetic index
# below deliberately makes the excluded chunk the *best* raw match (the
# query vector is set equal to it) so a passing test proves the filter beats
# similarity, not just that it happens to agree with it.


def _synthetic_shelf_index() -> tuple[LoadedIndex, np.ndarray]:
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(4, 8)).astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    chunks = [
        {"chunk_id": "c0", "shelf": "PSS-8", "section_path": "... PSS-8 shelf ..."},
        {"chunk_id": "c1", "shelf": "PSS-32", "section_path": "... PSS-32 shelf ..."},
        {"chunk_id": "c2", "shelf": None, "section_path": "... front matter ..."},
        {"chunk_id": "c3", "shelf": "PSS-32", "section_path": "... PSS-32 fan units ..."},
    ]
    index = LoadedIndex(
        embeddings=vectors, chunks=chunks, manifest={"model_name": "all-MiniLM-L6-v2"}
    )
    return index, vectors


def test_retrieve_shelf_filter_excludes_higher_scoring_other_shelf(monkeypatch):
    index, vectors = _synthetic_shelf_index()
    # Query vector == chunk c0's vector exactly: c0 (shelf=PSS-8) is the best
    # possible match (cosine similarity 1.0), everything else scores lower.
    monkeypatch.setattr("rag.retrieve.embed_query", lambda *a, **k: vectors[0].copy())

    results = retrieve("irrelevant query text", index, k=3, shelf_filter="PSS-32")

    assert results, "expected the two PSS-32 chunks to be retrieved"
    returned_ids = {r.chunk["chunk_id"] for r in results}
    assert returned_ids == {"c1", "c3"}  # only the PSS-32 chunks
    assert all(r.chunk["shelf"] == "PSS-32" for r in results)
    # c0 has the highest raw similarity (1.0, exact match) but must never
    # appear: the filter overrides ranking, it doesn't just re-sort within it.
    assert "c0" not in returned_ids


def test_retrieve_shelf_filter_no_matching_chunks_returns_empty(monkeypatch):
    index, vectors = _synthetic_shelf_index()
    monkeypatch.setattr("rag.retrieve.embed_query", lambda *a, **k: vectors[0].copy())

    results = retrieve("irrelevant query text", index, k=3, shelf_filter="PSS-16")

    assert results == []


def test_answer_question_shelf_filter_recovers_a_retrieval_miss():
    """Regression test pinning the Part E before/after example in
    docs/part-e-metadata-filtering.md: at k=4, the PSS-32 fan-unit section
    (2.18, "FAN and FAN32H") is crowded out by near-identical fan sections
    from other shelves unless the shelf filter is applied.
    """
    from rag.embed import load_index
    from rag.pipeline import answer_question

    index = load_index()
    question = "Which fan units are supported on the 1830 PSS-32 shelf?"

    unfiltered = answer_question(question, index, k=4, call_llm=False)
    filtered = answer_question(question, index, k=4, shelf_filter="PSS-32", call_llm=False)

    def has_fan_section(result):
        return any("2.18 PSS-32 Fan Units" in r.chunk["section_path"] for r in result.retrieved)

    assert not has_fan_section(unfiltered), (
        "expected the unfiltered top-4 to miss the PSS-32 fan-unit section "
        "(this is the retrieval gap the shelf filter is meant to fix)"
    )
    assert has_fan_section(filtered), (
        "expected shelf_filter='PSS-32' to recover the PSS-32 fan-unit section into top-4"
    )
    assert all(r.chunk["shelf"] == "PSS-32" for r in filtered.retrieved)


# --- Heading detection ------------------------------------------------------


@pytest.mark.parametrize(
    "line,expected",
    [
        ("    1.1        Overview", ("1.1", "Overview")),
        (" 2.18.1        Introduction", ("2.18.1", "Introduction")),
        ("  2.18         PSS-32 Fan Units (FAN and FAN32H)", ("2.18", "PSS-32 Fan Units (FAN and FAN32H)")),
        ("       1 System concept", ("1", "System concept")),
        ("       2 Shelves and common equipment/cards", ("2", "Shelves and common equipment/cards")),
    ],
)
def test_parse_heading_accepts_genuine_headings(line, expected):
    assert parse_heading(line) == expected


@pytest.mark.parametrize(
    "line",
    [
        "1.2 System configuration                                                48",  # TOC echo
        "Figure 2-4   1830 PSS-8 shelf",  # figure caption
        "Table 5-2   Slot ranges per card/shelf type",  # table caption
        "1                                             Power filter 1 (41)",  # figure-legend callout
        "2                                             Fan handle (for insertion/removal)",
    ],
)
def test_parse_heading_rejects_non_headings(line):
    assert parse_heading(line) is None

# --- Page alignment: PDF page N == printed page N in this document --------


def test_pss8_intro_chunk_has_correct_page():
    import json

    from rag.config import CHUNKS_PATH

    chunks = [json.loads(line) for line in CHUNKS_PATH.open(encoding="utf-8") if line.strip()]
    # The fact appears twice in the guide: once in the PSS-8 shelf's own
    # detailed section (p. 82), and once in an earlier cross-shelf summary
    # (1.2.2 SWDM NE, p. 49) — both are legitimate, so check the detailed
    # section's chunk specifically.
    matches = [
        c
        for c in chunks
        if "8-slot SWDM platform in 3-RU footprint" in c["text"]
        and c["section_path"].startswith("2 Shelves and common equipment/cards > 2.2 1830 PSS-8 shelf")
    ]
    assert matches, "expected the PSS-8 shelf section to contain its own intro sentence"
    assert all(c["page_start"] == 82 for c in matches)

