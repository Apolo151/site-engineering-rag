# Site-Engineering RAG Assistant

A small Retrieval-Augmented Generation pipeline grounded in the **Nokia 1830 PSS Product
Information & Planning Guide, Release 23.6** — Chapter 1 (System concept, pp. 47-72) and Chapter
2 through Power filters (pp. 73-166). It answers site-engineering questions strictly from those
pages and says **"Not found in the provided document"** when the answer isn't there, instead of
guessing.

See [`docs/problem-statement.md`](docs/problem-statement.md) for the full problem statement,
[`docs/diagrams.md`](docs/diagrams.md) for the system architecture, data flow, query sequence,
and chunking diagrams, and [`docs/part-e-metadata-filtering.md`](docs/part-e-metadata-filtering.md)
for the metadata-filtering write-up.

## Quickstart

```bash
uv sync                                   # installs Python 3.12 + CPU-only torch + deps
uv run scripts/build_index.py             # extract -> chunk -> embed (~15s on CPU)
cp .env.example .env                      # then fill in OPENAI_API_KEY / GROQ_API_KEY
uv run scripts/ask.py "How many slots does the 1830 PSS-8 shelf provide?"
uv run scripts/evaluate.py                # runs all 8 fixed questions -> eval/results.md
uv run pytest -q
uv run rag-web                            # web GUI -> http://127.0.0.1:8000
```

`scripts/ask.py --no-llm` runs retrieval only and prints the assembled prompt — useful for
grading the retrieval half without an API key.

## Web GUI

A single lightweight page over the same `answer_question()` pipeline — question box, a shelf
dropdown (Part E metadata filtering — see
[`docs/part-e-metadata-filtering.md`](docs/part-e-metadata-filtering.md)), the 8 evaluation
questions as one-click samples, the retrieved passages with their cosine scores and heading
paths, the abstain/answered badge, and a collapsible view of the exact assembled prompt.

```bash
uv run rag-web            # then open http://127.0.0.1:8000
```

It is a FastAPI app (`src/rag/web.py`) serving one self-contained `static/index.html`. 
The persisted index is loaded once at startup and the embedding model is pre-warmed off the request path. Without an
`OPENAI_API_KEY`/`GROQ_API_KEY` it runs in **retrieval-only** mode: sources and the assembled
prompt still render, only the generation step is skipped. `RAG_WEB_HOST` / `RAG_WEB_PORT`
override the bind address.

### Docker

The whole app ships as one image. The prebuilt index is committed, so no PDF and no index build
happen at container build time, and `poppler-utils` is not installed; the MiniLM model is baked
into the image, so the first query is instant and the container needs no Hugging Face access at
runtime.

```bash
cp .env.example .env      # optional - fill in a key for generation; skip for retrieval-only
docker compose up --build # -> http://localhost:8000
```

Or without compose:

```bash
docker build -t rag-web .
docker run -p 8000:8000 --env-file .env rag-web
```

## Pipeline

```
data/interim/1830_pp47-166.md   <- src/rag/extract.py   (Part A.1: pdftotext -f 47 -l 166 -layout)
data/chunks.jsonl               <- src/rag/chunk.py      (Part A.2-3: structure-aware chunking)
index/{embeddings.npy,          <- src/rag/embed.py      (Part B: MiniLM embeddings + manifest)
       chunks.jsonl,
       manifest.json}
                                 src/rag/retrieve.py      (Part B/C: manual top-k cosine search)
                                 src/rag/prompt.py         (Part C: system prompt + context assembly)
                                 src/rag/llm.py             (Part C.7: generation call, Groq/OpenAI-compatible)
                                 src/rag/pipeline.py        (ties retrieval + prompt + generation + abstain gate)
eval/questions.py + scripts/evaluate.py                    (Part D: fixed 8-question evaluation)
```

## Chunking strategy

The chunker (`src/rag/chunk.py`) never splits across a numbered section boundary. It walks the
extracted text, detects heading lines (`2.18    PSS-32 Fan Units (FAN and FAN32H)`), and
groups body text under the deepest currently-open heading. Sections shorter than 60 words are
merged forward into the next section (absorbing one-line stubs like `2.17.5 Location`); sections
longer than ~220 words are windowed on paragraph boundaries with a 40-word overlap, falling back
to a raw word-count cut only when a single paragraph itself exceeds the target.

**The critical decision: every chunk's embedding text is prefixed with its full heading path**
(`text_for_embedding = section_path + "\n\n" + body`), not just the raw paragraph. This was not
optional cosmetics — inspecting the source pages during planning showed that three of the eight
evaluation answers exist **only in a section heading**, never in a body sentence:

- `2.18 PSS-32 Fan Units (FAN and FAN32H)`
- `2.17 PSS-16II Fan Unit (16FAN2 and 16FAN2C)`
- `2.20 PSS-8 Power filter cards (8DC30, 8DC30T, 8DC30T2, 8AC7)`

A chunk built from the body paragraph alone would never surface these card codes to either the
embedding model or the LLM. Since the heading is injected into *every descendant chunk's*
embedding text, any chunk under `2.18` — not just the exact heading line — carries "FAN and
FAN32H" into its vector, so retrieval can find the answer regardless of which sub-section
happens to rank highest.

A second, less obvious payoff of carrying the full path (and the same information in the
"Path:" line of the prompt's context block, see below): several shelves' rack-mounting
paragraphs are **byte-identical across shelves** (the 450.85mm/444.5mm rack-aperture note is the
same sentence under PSS-8, PSS-16, PSS-16II and PSS-32). Only the heading path tells us which
shelf a given passage is actually about — the model needs it to answer Q7 correctly.

Metadata carried per chunk: `chunk_id`, `chapter`, `section_number`, `section_title`,
`section_path`, `page_start`/`page_end`, `word_count`, and a best-effort `shelf` tag (regex over
`section_path`, e.g. `PSS-32`). This `shelf` tag is what Part E's metadata filtering restricts
retrieval by — see [`docs/part-e-metadata-filtering.md`](docs/part-e-metadata-filtering.md).

**Extraction** (`src/rag/extract.py`) de-boilerplates the running header/footer without a
hand-maintained content list, aside from documenting eight fixed template strings that repeat
verbatim on every page of this document (copyright line, doc part number, the two-line
product-family breadcrumb). Everything else is column-frequency-based: `pdftotext -layout` lays
running-header text out in fixed columns (2+ space gaps); any non-heading column of text
repeating on 15+ pages is dropped. Heading lines are matched and protected *before* this pass
runs — a heading's own title is also echoed as a running header on every later page of its
section, so without that protection the one true heading occurrence would look exactly as
"boilerplate" as its echoes and get stripped too. The 1-3 line running-header block at the very
top of each page is stripped positionally (stopping at the first blank line, real heading, or
sentence-like prose), since its left-hand column changes per page (whatever section is
"currently open") and doesn't reliably clear a frequency threshold on its own.

Result: **161 chunks** from 120 pages / ~22,100 words. Word-count distribution: min 21 (the final
section in the range, which can't merge forward into anything), p50 145, p90 218, max 220
(capped — MiniLM truncates at 256 word-pieces, roughly 190 words, so chunks are kept under that
even though the assignment's target band is 100-300 words).

## Embedding & indexing

`all-MiniLM-L6-v2` (384-dim), `normalize_embeddings=True`. Because vectors are L2-normalized,
cosine similarity between a query and every chunk reduces to a single matrix-vector dot product —
this is the manual top-k implementation in `src/rag/retrieve.py::search`, cross-checked in
`tests/test_rag.py` against an independent `cosine_similarity()` that does the full
`a·b / (|a||b|)` computation. No FAISS/Chroma is used (at ~160 vectors, an ANN index buys
nothing — it exists to skip an O(N) scan that would otherwise cost microseconds).

The index is persisted (`index/embeddings.npy` + `index/chunks.jsonl` + `index/manifest.json`)
and versioned: the manifest stores a SHA-256 of `data/chunks.jsonl` at build time, and
`load_index()` refuses to run if the chunks on disk have since changed, rather than silently
serving stale vectors. `scripts/build_index.py` rebuilds everything (extraction through indexing)
in about 15 seconds on CPU.

## Retrieval + prompt engineering

**k = 5.** The assignment's range is k=3-5. At k=4, two of the eight questions (Q2, Q4) missed
their answer-bearing chunk by one rank — Nokia's shelf sections are templated prose ("The 1830
PSS-X shelf provides an N-slot SWDM platform in Y-RU footprint...") so cross-shelf semantic
similarity is high, and the correct chunk for a shelf-specific question routinely ranks 3rd-5th
behind near-duplicate paragraphs about *other* shelves. k=5 was the minimum that got all 8
questions' answer-bearing section into context (see `eval/results.md`, hit@k column and per-row
notes) without stretching outside the assignment's allowed range.

**Abstain gate:** before calling the LLM, if the top-1 cosine score is below `0.25`, the pipeline
refuses immediately without a model call (`refused_by: "threshold"`). This threshold is
deliberately low — for the assignment's trick question (Q8, optical reach), the top retrieved
score is 0.486, comfortably above 0.25, so the floor does *not* fire. Refusal for Q8 comes
entirely from the prompt, which is exactly the point: retrieval floors catch queries with no
plausible match at all (e.g. "what's the CEO's phone number"), but a *plausible-looking but wrong*
retrieval — the actual failure mode this assignment is testing — has to be caught by the model
reading the passages and noticing they don't answer the question.

### Exact system prompt

```text
You are a Nokia 1830 PSS site-engineering assistant. You answer questions about the Nokia
1830 PSS Product Information and Planning Guide (Release 23.6), and you answer ONLY from the
numbered CONTEXT passages supplied in the user message.

Rules — follow all of them, in order:

1. GROUNDING. Use only facts stated literally in the CONTEXT passages. You have no other
   knowledge of Nokia hardware. Do not use prior knowledge, and do not infer, estimate,
   interpolate, or calculate any value that is not written in the context.

2. REFUSAL. If the context does not contain the answer, reply with exactly:
       Not found in the provided document.
   followed by one short sentence naming what the passages do cover instead. A partially
   relevant passage is NOT an answer. Never guess a number, a card name, or a specification.
   Refusing is always better than being plausible and wrong.

3. CITATION. Follow every factual claim with a citation of the form
       [Section <number> <title>, p. <page>]
   copied verbatim from the header of the passage you used. Cite only passages you actually used.

4. SPECIFICITY. Passages for different shelves are often near-identical. Before answering,
   confirm the passage's Path refers to the exact shelf named in the question. If the question
   asks about the 1830 PSS-8 and the only matching passage sits under the 1830 PSS-32, that is
   not an answer — refuse under rule 2.

5. FORM. Be concise: 1-4 sentences or a short bullet list. Reproduce figures, card codes and
   units exactly as written. No preamble, no restating the question. Use plain ASCII characters
   only: a straight hyphen "-", straight square brackets "[" "]" for citations, straight quotes,
   and a normal space — never smart quotes, en/em dashes, fullwidth/CJK brackets, or non-breaking
   spaces.
```

User message template: `CONTEXT:\n{numbered passages, each with Section/Page/Path header}\n\n
QUESTION: {question}\n\nAnswer using only the CONTEXT above, following all rules.` —
`temperature=0`.

Rule 4 (SPECIFICITY) is the one doing the real work on Q1, Q7 and Q2: the retrieved top-5 for
each of these routinely includes 2-3 near-duplicate paragraphs from *other* shelves ranked above
the correct one (see `eval/results.md` notes) — the model has to read the Path line of each
passage and discriminate, not just answer from whatever ranked #1. Rule 2 is what makes Q8 return
a refusal instead of a hallucinated distance.

Generation calls Groq's OpenAI-compatible endpoint (`openai/gpt-oss-120b` by default, configurable
via `RAG_MODEL`/`OPENAI_BASE_URL`/`OPENAI_API_KEY` in `.env` — see `.env.example`; any
OpenAI-compatible provider works). Model output is also passed through a small ASCII-normalization
step (`src/rag/llm.py::_normalize_ascii`) as a deterministic backstop, since some models emit
smart hyphens/quotes/spaces even when explicitly told not to — this keeps the citation format
reliably machine-parseable.

## Evaluation (Part D)

All 8 fixed questions, full detail in [`eval/results.json`](eval/results.json) and the graded
table in [`eval/results.md`](eval/results.md). **Score: 8/8 correct**, including the deliberate
refusal on Q8. A terminal transcript of Q1, Q7 and Q8 running live is in
[`eval/transcript.txt`](eval/transcript.txt).

Run it yourself with `uv run scripts/evaluate.py` (needs an LLM key) or
`uv run scripts/evaluate.py --retrieval-only` (checks hit@k with no API call).

## Known limitations

1. **Figures and tables are invisible.** `pdftotext` extracts text only; any spec that lives
   solely in a figure or a rendered table image is unreachable by this pipeline.
2. **MiniLM truncates at 256 word-pieces** (~190 words). Chunks are capped near 220 words and the
   heading path is placed first in the embedding text specifically so it's never the part that
   gets truncated — but a very dense chunk could still lose its tail.
3. **No lexical/keyword guarantee for unfiltered queries.** Dense retrieval alone has no mechanism
   to force an exact card code like `8DC30T2` to matter more than semantically-similar-but-wrong
   nearby text. Metadata filtering (Part E, implemented) fixes this only when the caller already
   knows which shelf to restrict to; a general fix — re-ranking or hybrid BM25/embedding search,
   Part E's other two options — was not implemented.
4. **Scope is pages 47-166 only.** A question with a real answer elsewhere in the 1,568-page guide
   is correctly refused by this pipeline — the refusal means "not in the indexed pages," not "not
   in the document family."
5. **The abstain threshold (0.25) and k (5) were tuned against these same 8 questions.** That's a
   real overfitting risk on a set this small; they are reasonable defaults, not calibrated values.
6. **Line-wrap rejoining is heuristic.** A line ending in `-` is rejoined with the next line by
   keeping the hyphen (correct for the guide's genuine compound words, e.g. "half-height"); a
   line wrapped at a non-hyphen character (e.g. mid-slash in a shelf code list) is rejoined with a
   single space, which can occasionally introduce a spurious space inside what was one token in
   the original PDF. This doesn't affect any of the 8 evaluation answers.
7. **Part E: only metadata filtering was attempted**, not re-ranking or hybrid search. See
   [`docs/part-e-metadata-filtering.md`](docs/part-e-metadata-filtering.md) for what was built and
   a verified before/after example (pinned by an automated regression test).
