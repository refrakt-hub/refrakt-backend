# Dockerfile for Refrakt Backend

FROM python:3.11-slim AS builder

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir uv

COPY pyproject.toml /app/
COPY uv.lock /app/
COPY backend/requirements.txt /app/backend/requirements.txt
COPY backend/pyproject.toml /app/backend/pyproject.toml

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    uv pip install --system --no-cache -r backend/requirements.txt

COPY backend/ /app/backend/

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    cd /app && uv pip install --system --no-cache -e .

FROM python:3.11-slim AS runtime

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY backend/ /app/backend/
COPY pyproject.toml /app/

RUN groupadd -r -g 1000 refrakt && \
    useradd -r -u 1000 -g refrakt -d /app -s /bin/bash refrakt && \
    chown -R refrakt:refrakt /app

ENV PYTHONPATH=/app:/app/backend
ENV ENVIRONMENT=production
ENV PORT=8000
ENV UVICORN_WORKERS=4
ENV UVICORN_TIMEOUT_KEEP_ALIVE=5
ENV UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN=30
ENV UVICORN_ACCESS_LOG=false
ENV UVICORN_LOG_LEVEL=info

USER refrakt

EXPOSE 8000

# Use uvicorn CLI for production with worker support
# For single worker, use: python -m backend.main
# For multiple workers, use: uvicorn backend.main:app --workers $UVICORN_WORKERS
CMD ["sh", "-c", "if [ \"$UVICORN_WORKERS\" = \"1\" ]; then python -m backend.main; else uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${UVICORN_WORKERS:-4} --timeout-keep-alive ${UVICORN_TIMEOUT_KEEP_ALIVE:-5} --timeout-graceful-shutdown ${UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN:-30} --log-level ${UVICORN_LOG_LEVEL:-info} --no-access-log; fi"]

