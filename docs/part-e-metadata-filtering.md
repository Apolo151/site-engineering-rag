# Part E — Metadata Filtering

Stretch challenge, metadata filtering option: let a query optionally restrict retrieval to a
specific shelf using the `shelf` metadata tag attached at chunking time.

## What was built

Every chunk already carries a best-effort `shelf` tag, parsed out of its heading path at chunking
time (`src/rag/chunk.py::_detect_shelf`, e.g. `PSS-8`, `PSS-16II`, `PSS-32`, `PSI-4L`, ...). Real
distribution across the 161 indexed chunks: `PSS-8`=19, `PSS-16II`=19, `PSS-16`=9, `PSS-32`=7,
`PSI-4L`=4, `PSI-8L`=3, and 100 chunks with no clear single-shelf tag (front matter, cross-shelf
overview sections, etc.).

`retrieve()` (`src/rag/retrieve.py`) turns an optional `shelf_filter` argument into a boolean mask
over the chunk array, applied *before* top-k selection:

```python
mask = np.array([c.get("shelf") == shelf_filter for c in index.chunks])
results = search(qvec, index.embeddings, k, mask=mask)
```

Non-matching chunks are scored `-inf` and can never win a rank slot, regardless of how similar
they are to the query (see `test_retrieve_shelf_filter_excludes_higher_scoring_other_shelf` in
`tests/test_rag.py`, which deliberately makes the excluded chunk the objectively best match to
prove the filter overrides ranking rather than just agreeing with it). The filter is exposed at
every layer:

- `answer_question(..., shelf_filter=...)` — `src/rag/pipeline.py`
- `scripts/ask.py --shelf PSS-32`
- `POST /api/ask` (`{"shelf_filter": "PSS-32", ...}`) — `src/rag/web.py`
- a live `<select>` shelf dropdown in the web GUI (`src/rag/static/index.html`), populated from
  `GET /api/config`'s `shelves` list — this dropdown *is* the Part E feature, not a separate
  add-on.

## Before / after

**Query:** *"Which fan units are supported on the 1830 PSS-32 shelf?"* (evaluation Q4), `k=4`.

Nokia's fan-unit sections for every shelf are near-identical in wording ("The 1830 PSS-X fan unit
contains N powerful FAN modules, each individually monitored...", "When replacing the fan unit in
an 1830 PSS-X shelf..."). At k=4 unfiltered, PSS-32's own fan-unit section is outranked by three
*other* shelves' fan sections plus the PSS-32 shelf's own (unrelated) introduction — the correct
section is retrieved 0/4 times, and the effect reaches all the way to the final answer.

### Without filtering

```
$ uv run scripts/ask.py "Which fan units are supported on the 1830 PSS-32 shelf?" -k 4

Retrieved chunks:
  [0.753] c0124 | 2.15.2 Fan unit replacement | p.142-142   <- PSS-8
  [0.721] c0132 | 2.16.2 Fan unit replacement | p.147-147   <- PSS-16
  [0.720] c0135 | 2.17.3 Front view          | p.149-149   <- PSS-16II
  [0.715] c0070 | 2.5.1 Introduction          | p.96-97    <- PSS-32, but the intro, not the fan section

Answer (refused_by=prompt):
Not found in the provided document. The passages describe fan unit locations and replacement
procedures for PSS-8, PSS-16, and PSS-16II, and give a general description of the PSS-32 shelf.
```

The prompt's grounding rules do their job — the model refuses rather than answering from the
wrong shelf's fan chunks — but this is a **false negative**: the answer genuinely exists in the
indexed pages, it just never reached the model's context.

### With `shelf_filter=PSS-32`

```
$ uv run scripts/ask.py "Which fan units are supported on the 1830 PSS-32 shelf?" -k 4 --shelf PSS-32

Retrieved chunks:
  [0.715] c0070 | 2.5.1 Introduction              | p.96-97
  [0.697] c0139 | 2.18.2 Fan unit replacement      | p.151-151   <- section 2.18 "PSS-32 Fan Units (FAN and FAN32H)"
  [0.654] c0071 | 2.5.3 Slot numbering             | p.97-99
  [0.648] c0141 | 2.18.5 High capacity fan requirement | pp.152-153   <- also section 2.18

Answer (refused_by=None):
FAN (standard fan unit) and FAN32H (8DG59243AB) high-output fan unit are supported on the
1830 PSS-32 shelf. [Section 2.18.2 Fan unit replacement, p. 151]
[Section 2.18.5 High capacity fan requirement, pp. 152-153]
```

With the other three shelves' near-duplicate sections removed from competition, two of the four
PSS-8/16/16II fan chunks are simply excluded from consideration, and both of section 2.18's chunks
(which the heading-injection chunking strategy makes carry "FAN and FAN32H" in their embedded
text — see [README.md](../README.md#chunking-strategy)) now fit inside the same k=4 budget. The
answer is now correct, complete (both fan unit variants, including the FAN32H part number), and
cited.

Both commands above are exactly reproducible (`--no-llm` reproduces the retrieved-chunk lists
without needing an API key); the retrieval-level claim (unfiltered top-4 misses section 2.18,
filtered top-4 includes it) is additionally pinned as an automated regression test:
`test_answer_question_shelf_filter_recovers_a_retrieval_miss` in `tests/test_rag.py`.

## What this does and doesn't fix

Metadata filtering only helps when **the caller already knows which shelf the question is
about** — it's a targeting improvement, not a ranking improvement. It doesn't help a genuinely
ambiguous query ("which shelves support redundant power?"), and it doesn't fix the underlying
cause (near-duplicate cross-shelf template text diluting similarity scores) for callers who don't
supply a filter — unfiltered queries still rely on `k=5` and the prompt's SPECIFICITY rule, as
described in the main README. Re-ranking or hybrid keyword search (the other two Part E options)
would address the ranking problem directly; only metadata filtering was implemented here.
