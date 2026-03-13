from typing import Optional

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction

from src.utils.constants import (
    MILVUS_URI,
    MILVUS_TOKEN,
    EXFEVER_MILVUS_COLLECTION,
)


def build_exfever_milvus_retriever(
    uri: str = MILVUS_URI,
    token: str = MILVUS_TOKEN,
    collection_name: str = EXFEVER_MILVUS_COLLECTION,
    similarity_top_k: int = 10,
) -> BaseRetriever:
    """
    Build a Milvus-backed BM25 retriever for the ExFever dataset.

    This assumes that the corresponding Milvus collection has already been
    populated by `scripts/graph_check/build_exfever_index.py`.
    """
    vector_store = MilvusVectorStore(
        uri=uri,
        collection_name=collection_name,
        token=token,
        enable_dense=False,
        enable_sparse=True,
        sparse_embedding_function=BM25BuiltInFunction(),
        overwrite=False,  # Reuse existing collection; do not drop data
        use_async_client=False,
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )
    retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    return retriever


def build_exfever_retriever(
    db_path: Optional[str] = None,
    similarity_top_k: int = 10,
) -> BaseRetriever:
    """
    Backwards-compatible entry point used throughout the codebase.

    The `db_path` argument is ignored; retrieval is performed against the
    Milvus collection built by `build_exfever_index.py`.
    """
    _ = db_path  # kept for signature compatibility
    return build_exfever_milvus_retriever(similarity_top_k=similarity_top_k)
