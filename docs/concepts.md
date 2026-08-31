## Concepts (RAG & LLM background)

**Embeddings & bi-encoders.** `all-MiniLM-L6-v2` is a 6-layer distilled BERT, mean-pooled into a
384-dim vector per input, contrastively trained so paraphrases land close together in that space.
It's a *bi-encoder*: the question and every passage are encoded independently, which is exactly
what makes a precomputed, disk-persisted index possible — passages are embedded once at index
time, and only the query is embedded at ask time.

**Cosine similarity as a dot product.** `cosine(a,b) = (a·b)/(|a||b|)`, i.e. the angle between two
vectors, ignoring magnitude (so chunk length doesn't bias the score). If every vector is
pre-normalized to unit length, `|a|=|b|=1` and the formula collapses to a plain dot product —
scoring the whole corpus against one query becomes a single `matrix @ query` operation. That's
the "from first principles" search this assignment asks for; a vector database (FAISS/Chroma)
only earns its keep once an exact O(N) scan gets too slow, which doesn't happen at ~160 vectors.

**Why chunking is the highest-leverage step.** A single embedding vector has fixed capacity: pack
five unrelated topics into one chunk and its vector becomes a blurry average of all of them,
useless for precise retrieval. Split too aggressively, on the other hand, and you sever a fact
from the heading that identifies which shelf it's about. Fixed-character-count chunking is the
common default and is provably wrong here — three answers exist only in headings, and the
byte-identical rack-aperture sentences across four shelves are only disambiguated by their
heading path. Structural chunking (split on the document's own numbered sections) plus heading
injection is what makes both classes of question answerable at all.

**RAG's core idea.** An LLM's weights are a lossy, frozen compression of its training data; asked
for a spec it half-remembers, it produces the most probable-sounding continuation — a fluent,
confident, wrong number. RAG splits the job in two: retrieval is non-parametric and auditable (you
can point at the exact page), generation only paraphrases and formats what retrieval found. The
model stops being asked to *know* Nokia hardware specs and is instead asked to *read
comprehension* over text it's handed. The direct consequence: **retrieval failure is
unrecoverable** — if the right chunk never makes the top-k, no prompt can save the answer. That's
why `eval/results.md` reports hit@k (did retrieval find the right section?) separately from answer
correctness (did generation use it correctly?): a wrong answer with hit@k=True is a prompting
problem; hit@k=False would be a retrieval/chunking problem.

**Choosing k is a precision/recall trade.** Too low risks missing the answer entirely; too high
dilutes the prompt with distractors — concretely here, feeding the model *other shelves'* rack
notes right alongside the correct one. k=5 was chosen empirically (see Retrieval section above)
as the smallest value, within the assignment's allowed range, that got every question's answer
into context.

**Prompt engineering as the grounding mechanism.** Four techniques compound in the system prompt
above: (1) *instruction scoping* — telling the model it has "no other knowledge of Nokia
hardware" explicitly revokes its parametric prior rather than merely asking it to prefer the
context; (2) *a concrete refusal affordance* — instruction-tuned models are biased toward being
"helpful," which in practice means guessing; giving an exact, low-effort string to emit
("Not found in the provided document.") makes abstention as easy as answering; (3) *citation
forcing* — requiring a citation copied verbatim from a passage header makes fabrication
structurally harder, since a wrong or missing citation is visibly wrong; (4) *disambiguation* —
telling the model that near-duplicate passages across shelves exist, and to check the Path before
answering, is what actually saves Q1, Q2 and Q7 in this evaluation, where the correctly-cited
answer chunk is *not* the top-ranked retrieval result. `temperature=0` removes sampling
randomness, which is pure variance for a task whose job is to copy facts precisely, not to be
creative.

**Defense in depth.** The similarity floor and the prompt's refusal rule catch different failure
modes: the floor catches *no plausible match at all* (nothing scores near the query); the prompt
catches *plausible-looking but wrong* — passages that are topically related but don't actually
contain the answer, which is exactly Q8's failure mode (top score 0.486, well above the floor, yet
none of the five retrieved passages mention optical reach at all).

**Cache & version your embeddings.** A vector is only meaningful relative to the model that
produced it and the exact text it encodes. The index manifest hashes the chunk file it was built
from; loading the index re-hashes and refuses to proceed on a mismatch, rather than silently
answering questions against embeddings that no longer correspond to what's on disk.