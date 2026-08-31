"""Part B — embed every chunk and persist the index to disk.

Cache & version: the manifest records a hash of the chunks that produced the
embeddings. Loading refuses to serve a stale index if the chunks on disk no
longer match what was actually embedded.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from rag.config import (
    CHUNKS_PATH,
    EMBEDDING_MODEL_NAME,
    INDEX_CHUNKS,
    INDEX_EMBEDDINGS,
    INDEX_MANIFEST,
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _load_model(model_name: str = EMBEDDING_MODEL_NAME):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def build_index(
    chunks_path: Path = CHUNKS_PATH,
    index_dir: Path = INDEX_EMBEDDINGS.parent,
    model_name: str = EMBEDDING_MODEL_NAME,
) -> None:
    chunks = [json.loads(line) for line in chunks_path.open(encoding="utf-8") if line.strip()]
    texts = [c["text_for_embedding"] for c in chunks]

    model = _load_model(model_name)
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_EMBEDDINGS, embeddings)

    with INDEX_CHUNKS.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    import sentence_transformers

    manifest = {
        "model_name": model_name,
        "dim": int(embeddings.shape[1]),
        "n_chunks": int(embeddings.shape[0]),
        "normalized": True,
        "sentence_transformers_version": sentence_transformers.__version__,
        "chunks_sha256": _sha256_file(chunks_path),
        "built_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
    }
    INDEX_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Embedded {len(chunks)} chunks -> {embeddings.shape} ({model_name})")
    print(f"-> {INDEX_EMBEDDINGS}, {INDEX_CHUNKS}, {INDEX_MANIFEST}")


class LoadedIndex:
    def __init__(self, embeddings: np.ndarray, chunks: list[dict], manifest: dict):
        self.embeddings = embeddings
        self.chunks = chunks
        self.manifest = manifest


def load_index(
    chunks_path: Path = CHUNKS_PATH,
    index_dir: Path = INDEX_EMBEDDINGS.parent,
) -> LoadedIndex:
    if not INDEX_EMBEDDINGS.exists() or not INDEX_MANIFEST.exists():
        raise FileNotFoundError(
            "No persisted index found. Run `python -m rag.embed` (or "
            "scripts/build_index.py) first."
        )

    manifest = json.loads(INDEX_MANIFEST.read_text(encoding="utf-8"))
    current_hash = _sha256_file(chunks_path)
    if manifest.get("chunks_sha256") != current_hash:
        raise RuntimeError(
            "data/chunks.jsonl has changed since the index was built "
            f"(expected sha256={manifest.get('chunks_sha256')!r}, got {current_hash!r}). "
            "Re-run `python -m rag.embed` to rebuild the index before querying it."
        )

    embeddings = np.load(INDEX_EMBEDDINGS)
    chunks = [json.loads(line) for line in INDEX_CHUNKS.open(encoding="utf-8") if line.strip()]
    if embeddings.shape[0] != len(chunks):
        raise RuntimeError(
            f"Index/chunks size mismatch: {embeddings.shape[0]} vectors vs {len(chunks)} chunks."
        )
    return LoadedIndex(embeddings=embeddings, chunks=chunks, manifest=manifest)


if __name__ == "__main__":
    build_index()
