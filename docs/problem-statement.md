# Problem Statement

## Background

A Nokia site or field engineer working on a 1830 PSS deployment needs fast, exact answers from
hardware planning guides — which fan unit fits a given shelf, what a rack aperture requirement
is, how many slots a shelf supports. The source document, the *1830 Photonic Service Switch
Product Information and Planning Guide (Release 23.6)*, is 1,568 pages. Searching it by hand
under time pressure is slow. Asking a general-purpose chatbot is worse: without access to the
actual document, it will produce a fluent, confident, plausible-sounding answer that may simply
be wrong — a wrong rack-unit footprint or fan unit code is a real installation error, not a typo.

## The problem

Build a small Retrieval-Augmented Generation (RAG) assistant that answers site-engineering
questions **strictly from the real 1830 PSS Planning Guide**, and that says so clearly — rather
than guessing — when a question isn't answerable from the indexed material. The system must
demonstrably prefer refusing over fabricating: a RAG pipeline that never says "I don't know" is
considered *more* dangerous than no RAG pipeline at all, because it launders a hallucination
behind the appearance of a grounded citation.

This is deliberately not a "call an API and wire it to the PDF" exercise. It requires
demonstrating each stage of the RAG pattern from first principles:

- **Chunking** a real, messy, multi-column technical PDF into retrieval-friendly passages that
  respect the document's own structure, rather than splitting blindly on a fixed character count.
- **Embedding** those passages with a sentence-transformer model and implementing **top-k
  similarity search manually** — not delegated to a vector-database library — so that the
  mechanics of retrieval (cosine similarity, normalization, ranking) are demonstrably understood.
- **Prompt engineering** that keeps a generative model grounded: answering only from retrieved
  context, citing where each fact came from, and refusing cleanly when the retrieved passages
  don't actually contain the answer.
- **Evaluation** against a fixed, checkable set of questions, including one deliberately designed
  as a trap — a question whose answer is *not* in the indexed pages, to verify the system refuses
  instead of inventing a number.

## Scope and constraints

| Constraint | Value |
|---|---|
| Source document | `1830_Technical_Description.pdf` — 1830 PSS Product Information & Planning Guide, Release 23.6 |
| Indexed page range | pp. 47–166 only: Chapter 1 (System concept) and Chapter 2 through Power filters — **not** the full 1,568-page document |
| Embedding model | `sentence-transformers`, e.g. `all-MiniLM-L6-v2` |
| Similarity search | Must be implemented manually (cosine/dot-product top-k); a vector library (FAISS/Chroma) is allowed only as a *second*, compared implementation |
| Generation model | Any LLM the pipeline can call — retrieval and grounding logic must be original code, not delegated to a "chat with your PDF" library |
| Retrieval depth | k = 3–5 chunks per query |

## Success criteria

The pipeline is run against a fixed set of 8 questions with known, checkable answers drawn from
the indexed pages (shelf slot counts and RU footprints, software load-lines, fan unit and power
filter card codes, rack aperture requirements). Seven of the eight have a verifiable answer in
the source text; the eighth (maximum optical reach without amplification) does **not** — its
purpose is to verify the system recognizes the limits of what it was given and refuses rather
than guesses. Correctness on this last question is graded on the refusal itself, not on producing
any number.

## Deliverables

1. Source code for extraction, chunking, embedding/indexing, and retrieval + prompting.
2. A persisted vector index (or a script that rebuilds it quickly).
3. A short README covering the chunking strategy, the chosen k, the exact system prompt used, and
   known limitations.
4. An evaluation table recording, for each of the 8 questions, the retrieved chunk(s), the
   generated answer, and whether it is correct.
5. A terminal transcript or short recording of the pipeline answering three of the questions live.

## Stretch challenge status

The assignment offers an optional stretch challenge — metadata filtering, re-ranking, or hybrid
(embedding + keyword) search — worth bonus credit. **Metadata filtering was implemented**: every
chunk carries a `shelf` tag from chunking, and `retrieve()`/`answer_question()` can restrict
results to one shelf, exposed via the `--shelf` CLI flag and a live dropdown in the web GUI. See
[`docs/part-e-metadata-filtering.md`](part-e-metadata-filtering.md) for what was built and a
verified before/after example. Re-ranking and hybrid search were **not** implemented — out of
scope for the working time budget. See [`README.md`](../README.md#known-limitations) for the full
list of known limitations.
