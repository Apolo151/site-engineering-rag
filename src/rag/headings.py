"""Shared heading-detection logic used by both the extraction step (to keep
heading lines intact through whitespace normalization) and the chunker (to
build the section hierarchy).

Headings in the source guide look like:
    1.2.2       SWDM NE
    2.18         PSS-32 Fan Units (FAN and FAN32H)

Table-of-contents echoes look almost identical but end in a page number:
    1.2 System configuration                                                48

and figure/table captions are excluded because they don't start with a bare
chapter-scoped number at all:
    Figure 2-4   1830 PSS-8 shelf
"""

from __future__ import annotations

import re

# Numbered heading prefix (1-2 top-level chapters, up to 3 more levels),
# followed by a run of 2+ spaces (as produced by `pdftotext -layout`) and a
# title. Sub-level headings ("1.2", "2.18.1", ...) are laid out this way.
HEADING_RE = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,3}){0,3})\s{2,}(\S.*?)\s*$")

# The two bare top-level chapter headings ("1 System concept",
# "2 Shelves and common equipment/cards") are laid out with only a single
# space after the number, unlike every sub-level heading. Matched separately
# and more strictly (number must be exactly "1" or "2", nothing else) so a
# single-space gap elsewhere in body prose can't be mistaken for one.
CHAPTER_HEADING_RE = re.compile(r"^\s*([12])\s+(\S.*?)\s*$")

_MAX_LEVELS = 4
_TITLE_MIN_CHARS = 3
_TITLE_MAX_CHARS = 80
_EXCLUDED_FIRST_WORDS = {"Figure", "Table", "Note"}

# The only two bare top-level headings ("1", "2" with no sub-level dots) in
# this page range. A bare digit followed by a wide gap and a short caption
# also happens throughout the doc as numbered figure-legend callouts (e.g.
# "1        Status LED    2        Fan handle ..."), which would otherwise
# collide with both HEADING_RE and CHAPTER_HEADING_RE. Since only these two
# chapter headings are valid at level 1 in this range, requiring an exact
# (whitespace-collapsed) title match closes that hole precisely.
_KNOWN_CHAPTER_TITLES = {
    "1": "System concept",
    "2": "Shelves and common equipment/cards",
}


def parse_heading(line: str) -> tuple[str, str] | None:
    """Returns (number, title) if `line` is a genuine section heading, else
    None. Guards against table-of-contents echoes and figure/table captions.
    """
    match = HEADING_RE.match(line)
    if match:
        number, title = match.group(1), match.group(2)
    else:
        chapter_match = CHAPTER_HEADING_RE.match(line)
        if not chapter_match:
            return None
        number, title = chapter_match.group(1), chapter_match.group(2)

    levels = number.split(".")
    if levels[0] not in _KNOWN_CHAPTER_TITLES:
        return None
    if len(levels) > _MAX_LEVELS:
        return None

    if len(levels) == 1:
        # Bare top-level number: only valid if it's exactly one of the two
        # known chapter headings (see _KNOWN_CHAPTER_TITLES above).
        collapsed_title = re.sub(r"\s+", " ", title).strip()
        if _KNOWN_CHAPTER_TITLES[levels[0]] != collapsed_title:
            return None
        return number, collapsed_title

    if not (_TITLE_MIN_CHARS <= len(title) <= _TITLE_MAX_CHARS):
        return None
    if title[-1].isdigit():
        # Catches TOC lines: "1.2 System configuration    48"
        return None
    first_word = title.split()[0] if title.split() else ""
    if first_word in _EXCLUDED_FIRST_WORDS:
        return None

    return number, title


def heading_level(number: str) -> int:
    return number.count(".") + 1


def is_heading_line(line: str) -> bool:
    return parse_heading(line) is not None
