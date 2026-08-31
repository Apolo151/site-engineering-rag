# Evaluation results (Part D)

Pipeline: `all-MiniLM-L6-v2` embeddings, manual cosine-similarity top-k search (k=5), grounded
prompt (see [README.md](../README.md#exact-system-prompt)), generation via Groq
(`openai/gpt-oss-120b`), `temperature=0`. Full detail (retrieved chunks, full prompt, raw
answer) for every question is in [results.json](results.json); this table was filled in by hand
after reading each answer against the source pages.

| # | Question | Retrieved (top-1) | Answer | Correct? | Notes |
|---|---|---|---|---|---|
| 1 | How many slots does the 1830 PSS-8 shelf provide, and what is its rack-unit (RU) footprint? | §2.3.1 Introduction (PSS-16, p.86) — distractor; §2.2.1 (PSS-8, p.82) at rank 3 | "8 slots; 3-RU footprint [Section 2.2.1 Introduction, p. 82]" | ✅ Yes | Top-1 retrieval was the near-identical PSS-16 intro; the model correctly picked the PSS-8 chunk from further down the context and cited it (prompt rule 4 working as intended). |
| 2 | What rack-unit footprint does the 1830 PSS-32 shelf have, and how many slots does it provide? | §2.5.3 Slot numbering (PSS-32, p.97) | "14-RU footprint; 32 slots [Section 2.5.1 Introduction, pp. 96-97]" | ✅ Yes | Correct numbers and correct shelf, citation resolves to the right sub-section even though the ranked top-1 chunk wasn't the exact one used. |
| 3 | What are the two software load-lines supported by the 1830 PSS system? | §1.1.1 Purpose (p.47) | "SWDM software and OCS software. [Section 1.1.1 Purpose, p. 47]" | ✅ Yes | Exact match to source text. |
| 4 | Which fan units are supported on the 1830 PSS-32 shelf? | §2.15.2 (PSS-8 Fan Unit, p.142) — distractor; correct §2.18 family retrieved at rank 5 | "FAN and FAN32H [Section 2.18.2 Fan unit replacement, p. 151]" | ✅ Yes | Card codes only exist in the §2.18 **heading itself** ("PSS-32 Fan Units (FAN and FAN32H)"), not in any body paragraph — retrievable only because heading text is prepended to every child chunk's embedding (see README chunking strategy). Citation points at a child sub-section rather than §2.18 itself; page is still correct. |
| 5 | Which fan unit(s) are used on the 1830 PSS-16II shelf? | §2.17.3 Front view (PSS-16II, p.149) | "16FAN2 and 16FAN2C [Section 2.17.3 Front view, p. 149]" | ✅ Yes | Same heading-injection mechanism as Q4. |
| 6 | Name the power filter cards supported on the 1830 PSS-8 shelf. | §2.20.5 Location (PSS-8, p.160) | "8DC30, 8DC30T, 8DC30T2, 8AC7 [Section 2.20.5 Location, p. 160]" | ✅ Yes | All four card codes correct and complete. |
| 7 | What is the required horizontal rack aperture for mounting a 1830 PSS-8 shelf, and which common aperture size is explicitly NOT supported? | §2.2.2 Rack mounting options (PSS-8, p.82) | "Required: 450.85 mm (17.75 in) [Section 2.2.2, p. 82]. Not supported: 444.5 mm (17.5 in)." | ✅ Yes | This exact sentence is repeated verbatim under all four shelves (PSS-8/16/16II/32); three near-duplicate distractor chunks were also retrieved (PSS-16II, PSS-32, PSS-16) and the model correctly picked the PSS-8 one (prompt rule 4). This was the deliberate stress test for the SPECIFICITY rule. |
| 8 | What is the maximum optical reach, in kilometers, of the 1830 PSS-8 shelf without amplification? (trick question) | §2.2.2 Rack mounting options (p.82) — top score only 0.486, well above the 0.25 abstain floor | "Not found in the provided document. The passages cover rack mounting options, cluster configurations, backplane specifications, and slot numbering." | ✅ Yes (correct refusal) | "optical reach" occurs 0 times in pages 47-166. The similarity floor did **not** trigger (score was too high); the refusal came entirely from prompt rule 2, confirming the prompt — not the retrieval score — is what catches this case. `refused_by: "prompt"` in results.json. |

**Score: 8/8 correct**, including the deliberate trick question (Q8).

## Retrieval hit@k

At k=5, all 8 questions retrieved at least one chunk from the section that actually contains the
answer (`hit@k = True` for every question in `results.json`, including Q8 where no relevant
section exists by design). No question failed at the retrieval stage; every case above where a
"distractor" is listed as top-1 is a **generation-stage** success (the prompt correctly
discriminated among several retrieved shelves), not a retrieval failure.
