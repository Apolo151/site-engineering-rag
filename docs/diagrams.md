# Architecture & Data Flow

Diagrams render natively on GitHub (Mermaid). Module paths refer to [`src/rag/`](../src/rag).

## 1. System architecture

Two independent pipelines share one persisted index: an **offline** build step that runs once
(or whenever the source pages change), and an **online** query step that runs per question.

```mermaid
flowchart TB
    subgraph OFFLINE["Offline — index build (scripts/build_index.py)"]
        direction TB
        PDF[("1830_Technical_Description.pdf<br/>pages 47-166")]
        EXTRACT["extract.py<br/>pdftotext -layout<br/>+ de-boilerplating"]
        MD[("data/interim/<br/>1830_pp47-166.md")]
        CHUNK["chunk.py<br/>heading-aware chunking<br/>+ heading injection"]
        CHUNKS[("data/chunks.jsonl<br/>161 chunks")]
        EMBED["embed.py<br/>all-MiniLM-L6-v2<br/>normalize_embeddings=True"]
        IDX[("index/<br/>embeddings.npy<br/>chunks.jsonl<br/>manifest.json")]

        PDF --> EXTRACT --> MD --> CHUNK --> CHUNKS --> EMBED --> IDX
    end

    subgraph ONLINE["Online — per query (scripts/ask.py / evaluate.py)"]
        direction TB
        Q(["User question"])
        QEMB["retrieve.py<br/>embed_query()"]
        SEARCH["retrieve.py<br/>search()<br/>matrix @ query"]
        GATE{"top-1 score<br/>>= 0.25 ?"}
        PROMPT["prompt.py<br/>build_user_message()<br/>SYSTEM_PROMPT"]
        LLM["llm.py<br/>Groq / OpenAI-compatible<br/>temperature=0"]
        NORM["llm.py<br/>ASCII normalization"]
        ANSWER(["Answer + citation<br/>or refusal"])

        Q --> QEMB --> SEARCH --> GATE
        GATE -->|"no"| REFUSE(["Refuse<br/>(refused_by=threshold)<br/>no LLM call"])
        GATE -->|"yes"| PROMPT --> LLM --> NORM --> ANSWER
    end

    IDX -.->|"loaded once<br/>load_index()"| SEARCH

    style OFFLINE fill:#eef4fb,stroke:#6b8fb5
    style ONLINE fill:#fdf3e6,stroke:#c99a4d
```

**Key property:** retrieval and generation are separate concerns. `retrieve.py` never calls an
LLM; `llm.py` never touches the index. `pipeline.py` (not shown as a box above — it's the glue
that wires GATE → PROMPT → LLM) is the only module that knows about both.

## 2. Data flow — artifact by artifact

What each stage consumes and produces, independent of which script drives it.

```mermaid
flowchart LR
    A[("Source PDF<br/>1568 pages")] -->|"pdftotext -f 47 -l 166 -layout"| B["Raw layout text<br/>(120 pages)"]
    B -->|"strip boilerplate<br/>rejoin hyphen-wraps<br/>protect heading lines"| C[("Clean markdown<br/>&lt;!-- page: N --&gt; markers")]
    C -->|"parse_heading()<br/>build section stack"| D["Leaf sections<br/>(heading path + body tokens)"]
    D -->|"merge sections &lt; 60 words<br/>window sections &gt; 220 words"| E[("chunks.jsonl<br/>161 chunks + metadata")]
    E -->|"text_for_embedding =<br/>section_path + body"| F["all-MiniLM-L6-v2<br/>encode()"]
    F -->|"L2-normalize"| G[("embeddings.npy<br/>(161, 384) float32")]
    G -->|"sha256(chunks.jsonl)"| H[("manifest.json<br/>model, dim, hash")]

    Q(["Question text"]) -->|"same model<br/>encode()"| QV["query vector<br/>(384,)"]
    QV -->|"matrix @ query"| G
    G -->|"top-5 by score"| R["Retrieved chunks<br/>+ scores"]
    R -->|"numbered context block<br/>Section / Page / Path / text"| P["Assembled prompt"]
    P -->|"chat.completions.create<br/>temperature=0"| L["LLM answer"]
    L -->|"ASCII normalize"| Out(["Final answer<br/>with citation, or refusal"])

    style A fill:#f5f5f5
    style C fill:#eef4fb
    style E fill:#eef4fb
    style G fill:#eef4fb
    style H fill:#eef4fb
    style Out fill:#e9f7ef
```

## 3. Query sequence (single question, end to end)

```mermaid
sequenceDiagram
    actor User
    participant Ask as scripts/ask.py
    participant Pipe as pipeline.answer_question
    participant Ret as retrieve.py
    participant Idx as index (in-memory)
    participant Prompt as prompt.py
    participant LLM as llm.py (Groq)

    User->>Ask: question text
    Ask->>Pipe: answer_question(question, index, k=5)
    Pipe->>Ret: retrieve(question, index, k=5)
    Ret->>Ret: embed_query(question)
    Ret->>Idx: search(query_vec, embeddings, k=5)
    Idx-->>Ret: top-5 (chunk_index, score)
    Ret-->>Pipe: [RetrievedChunk, ...]

    alt top-1 score < 0.25
        Pipe-->>Ask: refusal (refused_by="threshold")<br/>no LLM call
    else top-1 score >= 0.25
        Pipe->>Prompt: build_user_message(question, retrieved)
        Prompt-->>Pipe: "CONTEXT:<br/>[1] Section...<br/><br/>QUESTION: ..."
        Pipe->>LLM: generate(user_message)
        LLM->>LLM: SYSTEM_PROMPT + user_message<br/>(rules 1-5, temperature=0)
        LLM->>LLM: ASCII-normalize output
        LLM-->>Pipe: answer text
        Pipe->>Pipe: refused_by = "prompt" if answer<br/>starts with refusal text
        Pipe-->>Ask: AnswerResult(answer, retrieved, refused_by)
    end
    Ask-->>User: retrieved chunks + final answer
```

The two refusal paths (`threshold` vs `prompt`) are logged separately in every evaluation record
— see [`eval/results.json`](../eval/results.json) — because they catch different failure modes
(see next section, and [README.md](../README.md#retrieval--prompt-engineering)).

## 4. Why heading injection matters (the core chunking decision)

Three of the eight evaluation questions have their answer **only in a section heading**, never in
a body sentence. Every chunk's embedding text is prefixed with its full ancestor heading path, so
the answer-bearing words propagate to every descendant chunk — not just to a chunk built from the
heading line itself (there usually isn't one; headings aren't chunked separately, they're
prepended to whatever body text follows them).

```mermaid
flowchart TD
    ROOT["2 Shelves and common equipment/cards"]
    L2["2.18  PSS-32 Fan Units (FAN and FAN32H)"]
    ROOT --> L2

    L2 --> S1["2.18.1 Introduction"]
    L2 --> S2["2.18.2 Fan unit replacement"]
    L2 --> S3["2.18.3 Air filter"]

    S1 --> C1["chunk c0139-a<br/>text_for_embedding =<br/>'2 Shelves... > 2.18 PSS-32 Fan Units<br/>(FAN and FAN32H) > 2.18.1 Introduction<br/><br/>In the 1830 PSS-32 the fan unit is<br/>located directly above...'"]
    S2 --> C2["chunk c0139<br/>text_for_embedding =<br/>'2 Shelves... > 2.18 PSS-32 Fan Units<br/>(FAN and FAN32H) > 2.18.2 Fan unit<br/>replacement<br/><br/>When replacing the fan unit...'"]
    S3 --> C3["chunk c0140<br/>text_for_embedding =<br/>'2 Shelves... > 2.18 PSS-32 Fan Units<br/>(FAN and FAN32H) > 2.18.3 Air filter<br/><br/>Air for cooling the 1830 PSS-32...'"]

    C1 -.->|"'FAN and FAN32H'<br/>present in embedding"| ANSWER(["Retrievable answer<br/>to Q4"])
    C2 -.->|"'FAN and FAN32H'<br/>present in embedding"| ANSWER
    C3 -.->|"'FAN and FAN32H'<br/>present in embedding"| ANSWER

    style L2 fill:#fdf3e6,stroke:#c99a4d
    style ANSWER fill:#e9f7ef,stroke:#4d9a6a
```

Without this, only a chunk built from the `2.18` heading line in isolation (which doesn't exist —
headings are attached to the body text that follows them, not chunked on their own) could ever
surface "FAN and FAN32H" to the embedding model. With it, retrieval succeeds regardless of which
specific sub-section (`2.18.1`, `2.18.2`, or `2.18.3`) happens to rank highest for a given query.
