"""
Sparse-only Milvus retriever that forces sparse search mode
to avoid embedding field lookup errors in sparse-only collections.
"""
from typing import List, Optional
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction


class SparseMilvusRetriever(BaseRetriever):
    """
    A retriever that ensures sparse-only search mode for Milvus collections
    configured with enable_dense=False and enable_sparse=True.
    
    This avoids the embedding field lookup error when querying sparse-only
    Milvus collections by forcing the query mode to SPARSE.
    """

    def __init__(
        self,
        vector_store: MilvusVectorStore,
        similarity_top_k: int = 10,
    ) -> None:
        """Initialize with params."""
        self._vector_store = vector_store
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle, **kwargs: any) -> List[NodeWithScore]:
        """Retrieve nodes given query."""
        # Create vector store query with sparse mode forced
        vector_store_query = VectorStoreQuery(
            query_str=query_bundle.query_str,
            similarity_top_k=self._similarity_top_k,
            mode=VectorStoreQueryMode.SPARSE,  # Force sparse mode
        )
        
        # Query the vector store
        vector_store_result = self._vector_store.query(vector_store_query)
        
        # Convert to NodeWithScore format expected by BaseRetriever
        nodes_with_scores = []
        for i, node in enumerate(vector_store_result.nodes):
            score = vector_store_result.similarities[i] if i < len(vector_store_result.similarities) else 0.0
            nodes_with_scores.append(NodeWithScore(node=node, score=score))
        
        return nodes_with_scores

    async def _aretrieve(self, query_bundle: QueryBundle, **kwargs: any) -> List[NodeWithScore]:
        """Asynchronously retrieve nodes given query."""
        # Create vector store query with sparse mode forced
        vector_store_query = VectorStoreQuery(
            query_str=query_bundle.query_str,
            similarity_top_k=self._similarity_top_k,
            mode=VectorStoreQueryMode.SPARSE,  # Force sparse mode
        )
        
        # Query the vector store asynchronously
        vector_store_result = await self._vector_store.aquery(vector_store_query)
        
        # Convert to NodeWithScore format expected by BaseRetriever
        nodes_with_scores = []
        for i, node in enumerate(vector_store_result.nodes):
            score = vector_store_result.similarities[i] if i < len(vector_store_result.similarities) else 0.0
            nodes_with_scores.append(NodeWithScore(node=node, score=score))
        
        return nodes_with_scores