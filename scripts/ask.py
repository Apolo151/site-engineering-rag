#!/usr/bin/env python3
"""Ask a single question against the persisted index.

    uv run scripts/ask.py "How many slots does the 1830 PSS-8 shelf provide?"
    uv run scripts/ask.py "..." -k 5 --shelf PSS-32
    uv run scripts/ask.py "..." --no-llm   # retrieval + assembled prompt only, no API call
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import DEFAULT_TOP_K  # noqa: E402
from rag.embed import load_index  # noqa: E402
from rag.pipeline import answer_question  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--shelf", default=None, help="Restrict retrieval to this shelf, e.g. PSS-32")
    parser.add_argument("--no-llm", action="store_true", help="Skip the generation call")
    args = parser.parse_args()

    index = load_index()
    result = answer_question(
        args.question, index, k=args.k, shelf_filter=args.shelf, call_llm=not args.no_llm
    )

    print(f"Question: {result.question}\n")
    print("Retrieved chunks:")
    for r in result.retrieved:
        c = r.chunk
        print(f"  [{r.score:.3f}] {c['chunk_id']} | {c['section_number']} {c['section_title']}"
              f" | p.{c['page_start']}-{c['page_end']}")
    print()

    if args.no_llm:
        print("--- Assembled prompt (system + user) ---")
        print(result.prompt)
        return

    print(f"Answer (refused_by={result.refused_by}):")
    print(result.answer)


if __name__ == "__main__":
    main()
