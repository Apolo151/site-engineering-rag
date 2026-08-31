"""Part B/C — manual top-k similarity search over the persisted index (no
FAISS/Chroma), plus the query-time embedding step.

Embeddings are stored L2-normalized, so cosine similarity between a query
vector and every chunk vector reduces to a single matrix-vector dot product:
cosine(a, b) = (a . b) / (|a| |b|) = a . b when |a| = |b| = 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rag.config import EMBEDDING_MODEL_NAME
from rag.embed import LoadedIndex, load_index


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Explicit cosine similarity, independent of any normalization
    assumption — used to cross-check the fast dot-product path in tests."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def search(
    query_vector: np.ndarray, matrix: np.ndarray, k: int, mask: np.ndarray | None = None
) -> list[tuple[int, float]]:
    """Returns the top-k (row_index, score) pairs for `query_vector` against
    `matrix`, highest score first. Assumes both are L2-normalized, so the
    dot product IS the cosine similarity.

    `mask`, if given, is a boolean array over rows; rows where mask is False
    are excluded (used for the shelf metadata filter).
    """
    scores = matrix @ query_vector
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)

    k = min(k, matrix.shape[0])
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [(int(i), float(scores[i])) for i in top_idx if np.isfinite(scores[i])]


_model_cache: dict[str, object] = {}


def _get_model(model_name: str = EMBEDDING_MODEL_NAME):
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer

        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def embed_query(query: str, model_name: str = EMBEDDING_MODEL_NAME) -> np.ndarray:
    model = _get_model(model_name)
    vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    return vec.astype("float32")


@dataclass
class RetrievedChunk:
    chunk: dict
    score: float


def retrieve(
    query: str,
    index: LoadedIndex,
    k: int,
    shelf_filter: str | None = None,
) -> list[RetrievedChunk]:
    """Embeds `query` and returns the top-k chunks from `index`.

    `shelf_filter`, if given, restricts retrieval to chunks whose `shelf`
    metadata field matches exactly (Part E metadata filtering hook).
    """
    qvec = embed_query(query, index.manifest["model_name"])

    mask = None
    if shelf_filter is not None:
        mask = np.array([c.get("shelf") == shelf_filter for c in index.chunks])
        if not mask.any():
            return []

    results = search(qvec, index.embeddings, k, mask=mask)
    return [RetrievedChunk(chunk=index.chunks[i], score=score) for i, score in results]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ad-hoc retrieval check (no LLM call).")
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=4)
    parser.add_argument("--shelf", default=None)
    args = parser.parse_args()

    idx = load_index()
    for r in retrieve(args.query, idx, args.k, shelf_filter=args.shelf):
        c = r.chunk
        print(f"[{r.score:.3f}] {c['chunk_id']} p.{c['page_start']}-{c['page_end']}")
        print(f"    {c['section_path']}")
        print(f"    {c['text'][:160]}...")
        print()
