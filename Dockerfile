FROM python:3.12-slim

WORKDIR /app

# System deps for psycopg binary
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    psycopg[binary]>=3.2 \
    psycopg_pool>=3.2 \
    pydantic>=2.10 \
    python-dateutil>=2.9 \
    boto3 \
    fastapi \
    uvicorn[standard] \
    requests \
    rdkit \
    pdbfixer

# Claude API client (optional — reasoning daemon degrades gracefully without it)
RUN pip install --no-cache-dir anthropic || echo "WARNING: anthropic install failed, Claude reasoning disabled"

# Copy application
COPY scripts/ ./scripts/
COPY data/ ./data/

ENV PYTHONPATH=/app/scripts
ENV PYTHONUNBUFFERED=1

# Railway provides $PORT
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
