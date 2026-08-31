# Single-stage image for the RAG web GUI.
# The prebuilt index (index/, data/chunks.jsonl) is committed, so no PDF and no
# index build are needed at serve time; poppler-utils is therefore not installed.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    RAG_WEB_HOST=0.0.0.0 \
    RAG_WEB_PORT=8000 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# 1. Dependencies (cached layer). torch resolves from the CPU wheel index
#    configured in pyproject.toml.
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen --no-install-project

# 2. Bake the embedding model into the image -> instant first query, fully
#    offline at runtime.
RUN uv run --no-sync python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# From here on the model is on disk; force offline so runtime never reaches HF.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# 3. Application code + the committed index.
COPY src/ src/
COPY index/ index/
COPY data/chunks.jsonl data/chunks.jsonl
COPY eval/ eval/
COPY README.md ./
RUN uv sync --no-dev --frozen

# 4. Drop privileges.
RUN useradd -m app && chown -R app:app /app /opt/hf-cache
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/config').status==200 else 1)"

CMD ["uv", "run", "--no-sync", "rag-web"]
