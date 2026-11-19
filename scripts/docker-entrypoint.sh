#!/bin/sh
# Build the index on first start if PDFs are mounted, then launch Streamlit.
set -e

if [ "${AUTO_INGEST:-true}" = "true" ]; then
  if ! python scripts/check_index.py; then
    echo "No index found — running ingest (PDFs from ${PDF_DIR:-/app/data})…"
    python ingest.py || echo "Ingest failed or no PDFs mounted; the app will show setup instructions."
  fi
fi

exec streamlit run app.py \
  --server.address=0.0.0.0 \
  --server.port="${PORT:-8501}" \
  --server.headless=true
