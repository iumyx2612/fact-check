"""Shared configuration constants for the project."""

import os

# Override via environment (e.g. `.env` loaded before imports in entrypoints).
MILVUS_URI = os.environ.get("MILVUS_URI", "https://milvus.vm.trungtd.work:19530")
MILVUS_TOKEN = os.environ.get("MILVUS_TOKEN", "root:Milvus")

FEVEROUS_MILVUS_COLLECTION = os.environ.get("FEVEROUS_MILVUS_COLLECTION", "feverous_bm25")
EXFEVER_MILVUS_COLLECTION = os.environ.get("EXFEVER_MILVUS_COLLECTION", "exfever_bm25")
