from typing import Optional

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction
from llama_index.vector_stores.milvus.utils import BaseSparseEmbeddingFunction
from llama_index.vector_stores.milvus.utils import BaseMilvusBuiltInFunction
from src.modules.retrievers.sparse_milvus_retriever import SparseMilvusRetriever

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
    bm25_k1: float = 0.9,
    bm25_b: float = 0.4,
) -> BaseRetriever:
    """
    Build a Milvus-backed BM25 retriever for the ExFever dataset.

    This assumes that the corresponding Milvus collection has already been
    populated by `scripts/graph_check/build_exfever_index.py`.

    Args:
        uri: Milvus server URI
        token: Milvus authentication token
        collection_name: Name of the Milvus collection
        similarity_top_k: Number of top results to retrieve
        bm25_k1: BM25 k1 parameter (default: 0.9 to match GraphCheck)
        bm25_b: BM25 b parameter (default: 0.4 to match GraphCheck)

    Note:
        The BM25 parameters (k1, b) are set at collection creation time.
        If the collection was created with different parameters, it must be
        rebuilt for these parameters to take effect.
    """
    vector_store = MilvusVectorStore(
        uri=uri,
        collection_name=collection_name,
        token=token,
        enable_dense=False,
        enable_sparse=True,
        sparse_embedding_function=BM25BuiltInFunction(
            function_params={"k1": bm25_k1, "b": bm25_b}
        ),
        overwrite=False,  # Reuse existing collection; do not drop data
        use_async_client=False,
    )

    # Use our custom sparse retriever to force sparse mode
    retriever = SparseMilvusRetriever(
        vector_store=vector_store,
        similarity_top_k=similarity_top_k,
    )
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
