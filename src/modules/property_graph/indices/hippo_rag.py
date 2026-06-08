import asyncio
import json
import os
from typing import Any, List, Optional, Sequence

import fsspec
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core.llms import LLM
from llama_index.core.schema import BaseNode, TransformComponent, TextNode, NodeRelationship
from llama_index.core.ingestion.pipeline import (
    run_transformations,
    arun_transformations,
)
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.graph_stores.types import (
    LabelledNode,
    Relation,
    PropertyGraphStore,
    TRIPLET_SOURCE_KEY,
)
from llama_index.core.graph_stores.types import (
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
)
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.vector_stores.simple import DEFAULT_PERSIST_DIR, DEFAULT_PERSIST_FNAME, SimpleVectorStoreData
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.core.embeddings.utils import EmbedType
from llama_index.core.callbacks import CallbackManager
from llama_index.core.storage.storage_context import StorageContext

from ..transformation.hippo_rag import HippoRAGExtractor


class UTF8SimpleVectorStore(SimpleVectorStore):
    def persist(
        self,
        persist_path: str = os.path.join(DEFAULT_PERSIST_DIR, DEFAULT_PERSIST_FNAME),
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> None:
        """Persist the SimpleVectorStore to a directory with UTF-8 encoding."""
        fs = fs or self._fs
        dirpath = os.path.dirname(persist_path)
        if dirpath and not fs.exists(dirpath):
            fs.makedirs(dirpath)

        with fs.open(persist_path, "w", encoding="utf-8") as f:
            json.dump(self.data.to_dict(), f, ensure_ascii=False)

    @classmethod
    def from_persist_path(
        cls,
        persist_path: str,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> "UTF8SimpleVectorStore":
        """Load from persist path with UTF-8 encoding."""
        fs = fs or fsspec.filesystem("file")
        if not fs.exists(persist_path):
            raise ValueError(
                f"No existing {__name__} found at {persist_path}, skipping load."
            )

        with fs.open(persist_path, "r", encoding="utf-8") as f:
            data_dict = json.load(f)
            data = SimpleVectorStoreData.from_dict(data_dict)
        return cls(data=data)


class UTF8SimplePropertyGraphStore(SimplePropertyGraphStore):
    def persist(
        self,
        persist_path: str,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> None:
        """Persist the graph store to a file with UTF-8 encoding."""
        if fs is None:
            fs = fsspec.filesystem("file")
        with fs.open(persist_path, "w", encoding="utf-8") as f:
            f.write(self.graph.model_dump_json())

    @classmethod
    def from_persist_path(
        cls,
        persist_path: str,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> "UTF8SimplePropertyGraphStore":
        """Load from persist path with UTF-8 encoding."""
        if fs is None:
            fs = fsspec.filesystem("file")
        with fs.open(persist_path, "r", encoding="utf-8") as f:
            data = json.loads(f.read())
        return cls.from_dict(data)


class HippoRAGGraphIndex(PropertyGraphIndex):
    def __init__(
        self,
        nodes: Optional[Sequence[BaseNode]] = None,
        llm: Optional[LLM] = None,
        kg_extractors: List[TransformComponent] = None,
        property_graph_store: Optional[PropertyGraphStore] = None,
        # vector related params
        vector_store: BasePydanticVectorStore = SimpleVectorStore(),
        use_async: bool = True,
        embed_model: Optional[EmbedType] = None,
        embed_kg_nodes: bool = True,
        # parent class params
        callback_manager: Optional[CallbackManager] = None,
        transformations: Optional[List[TransformComponent]] = None,
        storage_context: Optional[StorageContext] = None,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> None:
        if not kg_extractors:
            kg_extractors = [
                HippoRAGExtractor(
                    llm=llm
                )
            ]

        super().__init__(
            nodes=nodes,
            llm=llm,
            kg_extractors=kg_extractors,
            property_graph_store=property_graph_store,
            vector_store=vector_store,
            use_async=use_async,
            embed_model=embed_model,
            embed_kg_nodes=embed_kg_nodes,
            callback_manager=callback_manager,
            transformations=transformations,
            storage_context=storage_context,
            show_progress=show_progress,
            **kwargs
        )

    def _insert_nodes(self, nodes: Sequence[BaseNode]) -> Sequence[BaseNode]:
        """Insert nodes to the index struct."""
        if len(nodes) == 0:
            return nodes

        # run transformations on nodes to extract triplets
        if self._use_async:
            nodes = asyncio.run(
                arun_transformations(
                    nodes, self._kg_extractors, show_progress=self._show_progress
                )
            )
        else:
            nodes = run_transformations(
                nodes, self._kg_extractors, show_progress=self._show_progress
            )

        # ensure all nodes have nodes and/or relations in metadata
        assert all(
            node.metadata.get(KG_NODES_KEY) is not None
            or node.metadata.get(KG_RELATIONS_KEY) is not None
            for node in nodes
        )

        kg_nodes_to_insert: List[LabelledNode] = []
        kg_rels_to_insert: List[Relation] = []
        for node in nodes:
            # remove nodes and relations from metadata
            kg_nodes = node.metadata.pop(KG_NODES_KEY, [])
            kg_rels = node.metadata.pop(KG_RELATIONS_KEY, [])

            # add source id to properties
            for kg_node in kg_nodes:
                kg_node.properties[TRIPLET_SOURCE_KEY] = node.id_
            for kg_rel in kg_rels:
                kg_rel.properties[TRIPLET_SOURCE_KEY] = node.id_

            # add nodes and relations to insert lists
            kg_nodes_to_insert.extend(kg_nodes)
            kg_rels_to_insert.extend(kg_rels)

        # filter out duplicate kg nodes
        kg_node_ids = {node.id for node in kg_nodes_to_insert}
        existing_kg_nodes = self.property_graph_store.get(ids=list(kg_node_ids))
        existing_kg_node_ids = {node.id for node in existing_kg_nodes}
        kg_nodes_to_insert = [
            node for node in kg_nodes_to_insert if node.id not in existing_kg_node_ids
        ]

        # filter out duplicate llama nodes
        existing_nodes = self.property_graph_store.get_llama_nodes(
            [node.id_ for node in nodes]
        )
        existing_node_hashes = {node.hash for node in existing_nodes}
        nodes = [node for node in nodes if node.hash not in existing_node_hashes]

        # embed nodes (if needed)
        # here according to HippoRAG, we embed the triple not the Entity
        emb_triples_to_insert: List[TextNode] = []
        if self._embed_kg_nodes:
            # embed kg nodes
            triple_node_texts = [
                f"{str(rel.source_id)} {rel.label} {str(rel.target_id)}"
                for rel in kg_rels_to_insert
            ]

            if self._use_async:
                triples_embeddings = asyncio.run(
                    self._embed_model.aget_text_embedding_batch(
                        triple_node_texts, show_progress=self._show_progress
                    )
                )
            else:
                triples_embeddings = self._embed_model.get_text_embedding_batch(
                    triple_node_texts,
                    show_progress=self._show_progress,
                )

            for triple_text, rel, embedding in zip(triple_node_texts, kg_rels_to_insert, triples_embeddings):
                metadata = rel.properties.copy()
                emb_triples_to_insert.append(
                    TextNode(
                        text=triple_text,
                        embedding=embedding,
                        metadata=metadata
                    )
                )

        # Add the triples to vector store for retrieval
        if len(emb_triples_to_insert) > 0:
            new_ids = self.vector_store.add(emb_triples_to_insert)
            if not self.vector_store.stores_text:
                nodes_without_embedding = []
                for node, new_id in zip(emb_triples_to_insert, new_ids):
                    # NOTE: remove embedding from node to avoid duplication
                    node_without_embedding = node.model_copy()
                    node_without_embedding.embedding = None
                    nodes_without_embedding.append(node_without_embedding)
                self.docstore.add_documents(nodes_without_embedding)

        # The original chunk
        if len(nodes) > 0:
            self.property_graph_store.upsert_llama_nodes(nodes)

        # The Entities
        if len(kg_nodes_to_insert) > 0:
            self.property_graph_store.upsert_nodes(kg_nodes_to_insert)

        # important: upsert relations after nodes
        if len(kg_rels_to_insert) > 0:
            self.property_graph_store.upsert_relations(kg_rels_to_insert)

        # refresh schema if needed
        if self.property_graph_store.supports_structured_queries:
            self.property_graph_store.get_schema(refresh=False) # False for faster performance

        return nodes
