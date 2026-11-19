"""Exit 0 if the vector index exists and is non-empty, else 1 (used by Docker)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from legalbot.config import load_settings
from legalbot.vectorstore import index_status


def main() -> int:
    status = index_status(load_settings())
    print(f"index exists={status.exists} chunks={status.count} model={status.embedding_model or '-'}")
    return 0 if status.exists else 1


if __name__ == "__main__":
    sys.exit(main())
