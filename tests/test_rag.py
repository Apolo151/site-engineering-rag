import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.headings import parse_heading  # noqa: E402
from rag.retrieve import cosine_similarity, search  # noqa: E402


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

