#!/usr/bin/env python3
"""End-to-end index build: PDF -> interim markdown -> chunks -> embeddings.

    uv run scripts/build_index.py [--pdf PATH]

Rebuilds everything from scratch; on a modern laptop this finishes in well
under a minute (extraction + chunking are near-instant; embedding 150-200
chunks with all-MiniLM-L6-v2 on CPU takes about 5-10 seconds).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.chunk import build_chunks, write_chunks, _print_stats  # noqa: E402
from rag.config import DEFAULT_PDF_PATH, FIRST_PAGE, LAST_PAGE  # noqa: E402
from rag.embed import build_index  # noqa: E402
from rag.extract import write_interim_markdown  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF_PATH)
    parser.add_argument("--first-page", type=int, default=FIRST_PAGE)
    parser.add_argument("--last-page", type=int, default=LAST_PAGE)
    args = parser.parse_args()

    start = time.time()

    print(f"[1/3] Extracting pages {args.first_page}-{args.last_page} from {args.pdf} ...")
    write_interim_markdown(args.pdf, first_page=args.first_page, last_page=args.last_page)

    print("[2/3] Chunking ...")
    from rag.config import CHUNKS_PATH, INTERIM_MARKDOWN

    text = INTERIM_MARKDOWN.read_text(encoding="utf-8")
    chunks = build_chunks(text)
    write_chunks(chunks, CHUNKS_PATH)
    _print_stats(chunks)

    print("[3/3] Embedding + indexing ...")
    build_index()

    print(f"\nDone in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
