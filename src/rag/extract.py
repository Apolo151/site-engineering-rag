"""Part A.1 — extract the relevant page range from the source PDF into a
clean, page-marked markdown/text file.

Uses the system `pdftotext` (poppler-utils) with `-layout` so that headings
keep their original indentation, which the chunker relies on to tell a
heading from body text.

De-boilerplating strategy: the guide's running headers/footers are laid out
in fixed columns (e.g. "<breadcrumb>          Nokia 1830 PSS-8/16II/16/32/"),
which `-layout` renders as runs of 2+ spaces between columns. Rather than
hand-maintaining a list of exact boilerplate strings, we split every
non-heading line on those column boundaries and drop any column whose text
recurs on >= BOILERPLATE_MIN_REPEATS pages — this catches copyright/footer
text, the doc part number, and the running "chapter/section" breadcrumb
without a fixed list, while genuine body content (which repeats far less)
survives. Heading lines are matched and passed through completely untouched
*before* this pass runs, specifically because a heading's own title is
also echoed as a running header on every later page of its section — were
we not to protect it, the one true heading occurrence would look just as
"boilerplate" as its echoes and get stripped too.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from rag.config import FIRST_PAGE, LAST_PAGE, INTERIM_MARKDOWN
from rag.headings import is_heading_line

BOILERPLATE_MIN_REPEATS = 15
BOILERPLATE_MAX_CHARS = 90

PAGE_NUMBER_LINE_RE = re.compile(r"^\s*\d{1,4}\s*$")
COLUMN_SPLIT_RE = re.compile(r"\s{2,}")

# Fixed markers that appear verbatim in the running header/footer on every
# page of this document (copyright line, doc part number, release date, the
# two-line product-family breadcrumb on the right of the header). A
# non-heading line containing any of these is unambiguously boilerplate and
# is dropped whole, regardless of what shares the line with it (e.g. the
# left-hand breadcrumb column, which otherwise varies too much per page to
# reliably catch by frequency alone). This list is a documented, fixed
# property of this PDF's page template, not a per-topic content filter.
KNOWN_BOILERPLATE_MARKERS = (
    "Nokia 1830 PSS-8/16II/16/32/",
    "PSI-4L/PSI-8L",
    "\u00a9 2023 Nokia. Nokia Confidential Information",
    "Use subject to agreed restrictions on disclosure and use.",
    "Release 23.6",
    "June 2023",
    "Issue 1",
    "3KC-71311-QAAA-HQZZA",
)


def _run_pdftotext(pdf_path: Path, first_page: int, last_page: int) -> str:
    if shutil.which("pdftotext") is None:
        raise RuntimeError(
            "pdftotext not found on PATH. Install poppler-utils "
            "(e.g. `apt install poppler-utils` or `brew install poppler`)."
        )
    result = subprocess.run(
        [
            "pdftotext",
            "-f",
            str(first_page),
            "-l",
            str(last_page),
            "-layout",
            str(pdf_path),
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _columns(line: str) -> list[str]:
    return [c.strip() for c in COLUMN_SPLIT_RE.split(line.strip()) if c.strip()]


def _find_boilerplate_columns(pages: list[str]) -> set[str]:
    """Counts column text across pages, skipping heading lines entirely so a
    heading's own title is never a boilerplate candidate.
    """
    counts: Counter[str] = Counter()
    for page in pages:
        seen_this_page: set[str] = set()
        for raw_line in page.split("\n"):
            if not raw_line.strip() or is_heading_line(raw_line):
                continue
            for col in _columns(raw_line):
                if len(col) > BOILERPLATE_MAX_CHARS or col in seen_this_page:
                    continue
                seen_this_page.add(col)
                counts[col] += 1
    frequent = {col for col, n in counts.items() if n >= BOILERPLATE_MIN_REPEATS}
    return frequent | set(KNOWN_BOILERPLATE_MARKERS)


def _clean_non_heading_line(line: str, boilerplate: set[str]) -> str | None:
    """Returns the cleaned line, or None if the whole line should be
    dropped. Assumes `line` is not a heading line.
    """
    stripped = line.strip()
    if not stripped:
        return ""
    if PAGE_NUMBER_LINE_RE.match(stripped):
        return None

    columns = _columns(line)
    dropped_any = False
    kept: list[str] = []
    for col in columns:
        if col in boilerplate:
            dropped_any = True
            continue
        kept.append(col)

    if dropped_any:
        # A composite footer line (e.g. "Issue 1   3KC-71311-QAAA-HQZZA   47")
        # pairs known boilerplate columns with a page number that itself
        # varies per page and so isn't caught by the frequency check above.
        # Once we know the line is a boilerplate line at all, treat any
        # leftover bare-number column as that page number and drop it too.
        kept = [c for c in kept if not PAGE_NUMBER_LINE_RE.match(c)]

    if not kept:
        return None

    # Rejoin with a 2-space gap so any accidental heading-like structure in
    # the surviving text stays consistent; body text gets its internal
    # spacing collapsed to single spaces in the final normalization pass.
    return "  ".join(kept)


def _rejoin_hyphenated_wraps(lines: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip()
        if (
            stripped.endswith("-")
            and not stripped.endswith("--")
            and i + 1 < len(lines)
            and lines[i + 1].strip()[:1].islower()
            and not is_heading_line(line)
            and not is_heading_line(lines[i + 1])
        ):
            next_line = lines[i + 1]
            # Keep the hyphen: in this document a line-end hyphen is
            # consistently part of a real compound word (e.g. "half-" /
            # "height" -> "half-height"), not a soft wrap-hyphen to discard.
            merged_body = stripped + next_line.strip()
            out.append(merged_body)
            i += 2
            continue
        out.append(line)
        i += 1
    return out



def _looks_like_header_fragment(line: str) -> bool:
    """A running-header echo is short and doesn't end like a sentence (no
    terminal '.', ':', ';', ','), unlike real body prose. Length is measured
    on whitespace-collapsed content, since `-layout` pads header columns
    with wide gaps that inflate the raw stripped length."""
    stripped = line.strip()
    if not stripped:
        return False
    collapsed = re.sub(r"\s+", " ", stripped)
    if len(collapsed) > 80:
        return False
    return not stripped.endswith((".", ":", ";", ","))


def _strip_leading_header_block(lines: list[str]) -> list[str]:
    """Drops the 1-3 running-header lines at the very top of a page (the
    breadcrumb of whichever heading is currently open, echoed on every page
    of its section — see module docstring). Stops at the first blank line,
    the first genuine heading, or the first line that reads like real body
    prose, so a page with no header (or one starting directly on a heading)
    is left untouched.
    """
    i = 0
    while i < len(lines) and i < 3:
        line = lines[i]
        if not line.strip() or is_heading_line(line) or not _looks_like_header_fragment(line):
            break
        i += 1
    return lines[i:]

def _clean_page(page_text: str, boilerplate: set[str]) -> str:
    lines = _strip_leading_header_block(page_text.split("\n"))

    # Pass 1: protect headings verbatim; strip boilerplate columns and bare
    # page numbers from everything else.
    pass1: list[str] = []
    for line in lines:
        if is_heading_line(line):
            pass1.append(line.rstrip())
            continue
        cleaned = _clean_non_heading_line(line, boilerplate)
        if cleaned is not None:
            pass1.append(cleaned)

    pass2 = _rejoin_hyphenated_wraps(pass1)

    # Pass 3: collapse internal whitespace on body lines only; headings keep
    # their original spacing so later heading detection still matches.
    pass3: list[str] = []
    for line in pass2:
        if is_heading_line(line):
            pass3.append(line)
        else:
            pass3.append(re.sub(r"[ \t]{2,}", " ", line).rstrip())

    # Collapse runs of blank lines down to at most one.
    result: list[str] = []
    prev_blank = False
    for line in pass3:
        is_blank = line.strip() == ""
        if is_blank and prev_blank:
            continue
        result.append(line)
        prev_blank = is_blank

    return "\n".join(result).strip("\n")


def extract_pages(
    pdf_path: Path,
    first_page: int = FIRST_PAGE,
    last_page: int = LAST_PAGE,
) -> list[tuple[int, str]]:
    """Returns a list of (page_number, cleaned_text) tuples."""
    raw = _run_pdftotext(pdf_path, first_page, last_page)
    pages = raw.split("\f")
    # pdftotext emits one form-feed-terminated chunk per page, plus a
    # trailing empty chunk after the last form feed.
    if pages and pages[-1].strip() == "":
        pages = pages[:-1]

    boilerplate = _find_boilerplate_columns(pages)

    out: list[tuple[int, str]] = []
    for i, page_text in enumerate(pages):
        page_number = first_page + i
        cleaned = _clean_page(page_text, boilerplate)
        out.append((page_number, cleaned))
    return out


def write_interim_markdown(
    pdf_path: Path,
    output_path: Path = INTERIM_MARKDOWN,
    first_page: int = FIRST_PAGE,
    last_page: int = LAST_PAGE,
) -> list[tuple[int, str]]:
    pages = extract_pages(pdf_path, first_page, last_page)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for page_number, text in pages:
            f.write(f"<!-- page: {page_number} -->\n")
            f.write(text)
            f.write("\n\n")
    return pages


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=None)
    parser.add_argument("--first-page", type=int, default=FIRST_PAGE)
    parser.add_argument("--last-page", type=int, default=LAST_PAGE)
    parser.add_argument("--out", type=Path, default=INTERIM_MARKDOWN)
    args = parser.parse_args()

    from rag.config import DEFAULT_PDF_PATH

    pdf = args.pdf or DEFAULT_PDF_PATH
    pages = write_interim_markdown(pdf, args.out, args.first_page, args.last_page)
    total_words = sum(len(t.split()) for _, t in pages)
    print(f"Extracted {len(pages)} pages ({total_words} words) -> {args.out}")
