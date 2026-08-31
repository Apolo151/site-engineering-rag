"""Part A.2-3 — structure-aware chunking.

Splits the extracted, page-marked text into retrieval-friendly chunks that
never cross a level-2 section boundary, carrying section metadata (heading
path, page range) with every chunk. See README "Chunking strategy" for the
rationale.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from rag.config import (
    CHUNK_MIN_WORDS,
    CHUNK_OVERLAP_WORDS,
    CHUNK_TARGET_WORDS,
    CHUNKS_PATH,
    INTERIM_MARKDOWN,
)
from rag.headings import heading_level, parse_heading

PAGE_MARKER_RE = re.compile(r"<!-- page: (\d+) -->")

# Longer/more specific names must be checked before shorter ones they
# contain (e.g. "PSS-16II" before "PSS-16").
_SHELF_NAMES = [
    "PSS-16II",
    "PSS-8x",
    "PSS-12x",
    "PSS-24x",
    "PSS-36",
    "PSS-64",
    "PSS-8",
    "PSS-16",
    "PSS-32",
    "PSI-4L",
    "PSI-8L",
]


def _detect_shelf(section_path: str) -> str | None:
    for name in _SHELF_NAMES:
        if re.search(re.escape(name) + r"\b", section_path):
            return name
    return None


@dataclass
class _LeafSection:
    section_number: str
    section_title: str
    section_path: str
    chapter: str
    # (page_number, word) tokens, in order, with paragraph-start indices
    tokens: list[tuple[int, str]] = field(default_factory=list)
    paragraph_starts: list[int] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len(self.tokens)


@dataclass
class Chunk:
    chunk_id: str
    chapter: str
    section_number: str
    section_title: str
    section_path: str
    page_start: int
    page_end: int
    shelf: str | None
    word_count: int
    text: str
    text_for_embedding: str

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "chapter": self.chapter,
            "section_number": self.section_number,
            "section_title": self.section_title,
            "section_path": self.section_path,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "shelf": self.shelf,
            "word_count": self.word_count,
            "text": self.text,
            "text_for_embedding": self.text_for_embedding,
        }


def parse_pages_from_markdown(markdown_text: str) -> list[tuple[int, str]]:
    """Splits the interim markdown (written by extract.py) back into
    (page_number, page_text) pairs using its `<!-- page: N -->` markers.
    """
    parts = PAGE_MARKER_RE.split(markdown_text)
    # re.split with a capturing group yields: [pre, page_num, body, page_num, body, ...]
    pages: list[tuple[int, str]] = []
    for i in range(1, len(parts), 2):
        page_number = int(parts[i])
        body = parts[i + 1] if i + 1 < len(parts) else ""
        pages.append((page_number, body.strip("\n")))
    return pages


def _build_leaf_sections(pages: list[tuple[int, str]]) -> list[_LeafSection]:
    stack: list[tuple[str, str]] = []  # (number, title), index 0 = level 1

    def current_context() -> tuple[str, str, str, str]:
        if not stack:
            return "0", "Front matter", "Front matter", "Front matter"
        number, title = stack[-1]
        path = " > ".join(f"{n} {t}" for n, t in stack)
        chapter = stack[0][1]
        return number, title, path, chapter

    leaves: list[_LeafSection] = []
    current: _LeafSection | None = None
    at_paragraph_start = True

    def start_new_leaf() -> None:
        nonlocal current, at_paragraph_start
        number, title, path, chapter = current_context()
        current = _LeafSection(
            section_number=number, section_title=title, section_path=path, chapter=chapter
        )
        at_paragraph_start = True

    def flush_leaf() -> None:
        if current is not None and current.tokens:
            leaves.append(current)

    start_new_leaf()

    for page_number, page_text in pages:
        for raw_line in page_text.split("\n"):
            parsed = parse_heading(raw_line) if raw_line.strip() else None
            if parsed:
                flush_leaf()
                number, title = parsed
                level = heading_level(number)
                stack = stack[: level - 1]
                stack.append((number, title))
                start_new_leaf()
                continue

            stripped = raw_line.strip()
            if not stripped:
                at_paragraph_start = True
                continue

            assert current is not None
            words = stripped.split()
            if at_paragraph_start and current.tokens:
                current.paragraph_starts.append(len(current.tokens))
            at_paragraph_start = False
            current.tokens.extend((page_number, w) for w in words)

    flush_leaf()
    return leaves


def _merge_small_sections(leaves: list[_LeafSection]) -> list[_LeafSection]:
    merged: list[_LeafSection] = []
    pending: _LeafSection | None = None

    for i, leaf in enumerate(leaves):
        is_last = i == len(leaves) - 1
        if pending is not None:
            combined = _LeafSection(
                section_number=leaf.section_number,
                section_title=leaf.section_title,
                section_path=leaf.section_path,
                chapter=leaf.chapter,
            )
            combined.tokens = pending.tokens + leaf.tokens
            combined.paragraph_starts = pending.paragraph_starts + [
                len(pending.tokens) + p for p in leaf.paragraph_starts
            ]
            if pending.tokens:
                combined.paragraph_starts = [len(pending.tokens)] + combined.paragraph_starts
            leaf = combined
            pending = None

        if leaf.word_count < CHUNK_MIN_WORDS and not is_last:
            pending = leaf
            continue

        merged.append(leaf)

    if pending is not None:
        merged.append(pending)

    return merged


def _window_tokens(
    tokens: list[tuple[int, str]], paragraph_starts: list[int]
) -> list[list[tuple[int, str]]]:
    n = len(tokens)
    if n <= CHUNK_TARGET_WORDS:
        return [tokens]

    boundaries = sorted(set(paragraph_starts))
    windows: list[list[tuple[int, str]]] = []
    start = 0
    while start < n:
        end = min(start + CHUNK_TARGET_WORDS, n)
        if end < n:
            candidates = [b for b in boundaries if start + CHUNK_MIN_WORDS <= b <= end]
            if candidates:
                end = candidates[-1]
        windows.append(tokens[start:end])
        if end >= n:
            break
        start = max(end - CHUNK_OVERLAP_WORDS, start + 1)
    return windows


def _leaf_to_chunks(leaf: _LeafSection, chunk_index: list[int]) -> list[Chunk]:
    windows = _window_tokens(leaf.tokens, leaf.paragraph_starts)
    chunks: list[Chunk] = []
    for window in windows:
        pages_in_window = [p for p, _ in window]
        text = " ".join(w for _, w in window)
        text_for_embedding = f"{leaf.section_path}\n\n{text}"
        chunk_index[0] += 1
        chunks.append(
            Chunk(
                chunk_id=f"c{chunk_index[0]:04d}",
                chapter=leaf.chapter,
                section_number=leaf.section_number,
                section_title=leaf.section_title,
                section_path=leaf.section_path,
                page_start=min(pages_in_window),
                page_end=max(pages_in_window),
                shelf=_detect_shelf(leaf.section_path),
                word_count=len(window),
                text=text,
                text_for_embedding=text_for_embedding,
            )
        )
    return chunks


def build_chunks(markdown_text: str) -> list[Chunk]:
    pages = parse_pages_from_markdown(markdown_text)
    leaves = _build_leaf_sections(pages)
    leaves = _merge_small_sections(leaves)

    chunk_index = [0]
    chunks: list[Chunk] = []
    for leaf in leaves:
        chunks.extend(_leaf_to_chunks(leaf, chunk_index))
    return chunks


def write_chunks(chunks: list[Chunk], output_path=CHUNKS_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def load_chunks(path=CHUNKS_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _print_stats(chunks: list[Chunk]) -> None:
    counts = sorted(c.word_count for c in chunks)
    n = len(counts)

    def pct(p: float) -> int:
        return counts[min(int(p * n), n - 1)]

    print(f"{n} chunks")
    print(f"word_count: min={counts[0]} p50={pct(0.5)} p90={pct(0.9)} max={counts[-1]}")
    with_shelf = sum(1 for c in chunks if c.shelf)
    print(f"{with_shelf}/{n} chunks tagged with a shelf")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", type=Path, default=INTERIM_MARKDOWN)
    parser.add_argument("--out", type=Path, default=CHUNKS_PATH)
    args = parser.parse_args()

    text = args.markdown.read_text(encoding="utf-8")
    chunks = build_chunks(text)
    write_chunks(chunks, args.out)
    _print_stats(chunks)
    print(f"-> {args.out}")
