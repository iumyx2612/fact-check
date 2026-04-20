"""
Shared OpenAI + Milvus wiring for ExFever GraphCheck entrypoints (benchmark, single-claim script).

Keeps defaults in one place so ``benchmark_exfever_graphcheck`` and ``run_graphcheck_claim`` stay aligned.

Supports OpenAI, vLLM, and custom OpenAI-compatible APIs via environment variables:
- LLM_PROVIDER: "openai" (default), "vllm", or "custom"
- OPENAI_API_KEY, OPENAI_MODEL: For OpenAI provider
- LLM_API_BASE_URL, LLM_API_KEY, LLM_MODEL_NAME: For vLLM/custom provider
"""
from __future__ import annotations

from llama_index.core.retrievers import BaseRetriever

# Re-export from llm_config for backward compatibility
from .llm_config import (
    create_llm,
    openai_llm_for_graphcheck,
    DEFAULT_MAX_TOKENS as OPENAI_MAX_TOKENS,
    DEFAULT_TEMPERATURE as OPENAI_TEMPERATURE,
    DEFAULT_TOP_P as OPENAI_TOP_P,
    DEFAULT_TIMEOUT as LLM_TIMEOUT_S,
    DEFAULT_MAX_RETRIES as LLM_MAX_RETRIES,
)

# Defaults aligned with ``benchmark_exfever_graphcheck`` / ``run_graphcheck_claim``
DEFAULT_PATH_LIMIT = 5
DEFAULT_SIMILARITY_TOP_K = 10
DEFAULT_MAX_CONCURRENT = 10


def milvus_retriever_for_graphcheck(similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K) -> BaseRetriever:
    """Sparse BM25 retriever over the ExFever Milvus collection."""
    from src.modules.retrievers.exfever import build_exfever_milvus_retriever

    return build_exfever_milvus_retriever(similarity_top_k=similarity_top_k)
