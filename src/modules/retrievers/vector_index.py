from typing import Any, Dict, List, Optional

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.retrievers import VectorIndexRetriever as LIVectorIndexRetriever
from llama_index.core.schema import BaseNode, ObjectType
from llama_index.core.storage.docstore.types import BaseDocumentStore
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)


class VectorIndexRetriever(LIVectorIndexRetriever):
    def __init__(
        self,
        vector_store: BasePydanticVectorStore,
        doc_store: BaseDocumentStore,
        embed_model: BaseEmbedding,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        vector_store_query_mode: VectorStoreQueryMode = VectorStoreQueryMode.DEFAULT,
        filters: Optional[MetadataFilters] = None,
        alpha: Optional[float] = None,
        node_ids: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
        sparse_top_k: Optional[int] = None,
        hybrid_top_k: Optional[int] = None,
        callback_manager: Optional[CallbackManager] = None,
        object_map: Optional[dict] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        self._vector_store = vector_store
        self._docstore = doc_store
        self._embed_model = embed_model

        self._similarity_top_k = similarity_top_k
        self._vector_store_query_mode = VectorStoreQueryMode(vector_store_query_mode)
        self._alpha = alpha
        self._node_ids = node_ids
        self._doc_ids = doc_ids
        self._filters = filters
        self._sparse_top_k = sparse_top_k
        self._hybrid_top_k = hybrid_top_k
        self._kwargs: Dict[str, Any] = kwargs.get("vector_store_kwargs", {})

        callback_manager = callback_manager or CallbackManager()
        BaseRetriever.__init__(
            self,
            callback_manager=callback_manager,
            object_map=object_map,
            verbose=verbose,
        )

    def _determine_nodes_to_fetch(
        self, query_result: VectorStoreQueryResult
    ) -> list[str]:
        if query_result.nodes:
            return [
                node.node_id
                for node in query_result.nodes
                if node.as_related_node_info().node_type != ObjectType.TEXT
            ]
        elif query_result.ids:
            return list(query_result.ids)
        else:
            return []

    def _insert_fetched_nodes_into_query_result(
        self, query_result: VectorStoreQueryResult, fetched_nodes: List[BaseNode]
    ):
        fetched_nodes_by_id: Dict[str, BaseNode] = {
            str(node.node_id): node for node in fetched_nodes
        }
        new_nodes: List[BaseNode] = []

        if query_result.nodes:
            for node in list(query_result.nodes):
                node_id_str = str(node.node_id)
                if node_id_str in fetched_nodes_by_id:
                    new_nodes.append(fetched_nodes_by_id[node_id_str])
                else:
                    new_nodes.append(node)
        elif query_result.ids:
            for node_id in query_result.ids:
                node_id_str = str(node_id)
                if node_id_str in fetched_nodes_by_id:
                    new_nodes.append(fetched_nodes_by_id[node_id_str])
                else:
                    raise KeyError(
                        f"Node ID {node_id_str} not found in fetched nodes. "
                    )
        elif query_result.ids is None and query_result.nodes is None:
            raise ValueError(
                "Vector store query result should return at least one of nodes or ids."
            )
        return new_nodes