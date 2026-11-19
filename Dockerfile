FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_CACHE_DIR=/app/.cache/models \
    CHROMA_DIR=/app/vectorstore/chroma_db \
    PDF_DIR=/app/data

# Tesseract + Bengali language pack for scanned PDFs; curl for the health check.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-ben curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/.cache/models /app/vectorstore /app/data \
    && chown -R app:app /app
COPY --chown=app:app . .
USER app

# Bake the models into the image (set --build-arg PRELOAD_MODELS=false to skip).
ARG PRELOAD_MODELS=true
RUN if [ "$PRELOAD_MODELS" = "true" ]; then python scripts/download_models.py; fi

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-8501}/_stcore/health" || exit 1

ENTRYPOINT ["sh", "./scripts/docker-entrypoint.sh"]
