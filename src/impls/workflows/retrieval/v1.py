from typing import Optional

from workflows import Workflow, step, Context
from llama_index.core.llms import LLM
from llama_index.core.prompts import ChatMessage
from llama_index.core.storage.docstore import BaseDocumentStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.core.graph_stores import SimplePropertyGraphStore

from ...events.retrieval.v1 import (
    RetrievalStartEvent,
    RetrieveEntityEvent,
    RetrieveTripletsEvent,
    RetrievalAggregateEvent,
    RetrievalStopEvent
)
from src.modules.property_graph.transformation.hippo_rag import (
    NER_SYSTEM,
    NER_EXAMPLE_INPUT,
    NER_EXAMPLE_OUTPUT,
    NER_USER,
    fix_broken_generated_json,
    _parse_ner_response
)
from src.modules.retrievers.vector_index import VectorIndexRetriever


class RetrievalWorkflow(Workflow):
    def __init__(self,
                 llm: LLM,
                 vector_retriever: VectorIndexRetriever,
                 graph_store: SimplePropertyGraphStore,
                 **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.graph_store = graph_store
        self.vector_retriever = vector_retriever

    @step
    async def retrieval_init(self, ctx: Context, ev: RetrievalStartEvent) -> RetrieveEntityEvent | RetrieveTripletsEvent:
        claim = ev.claim
        subclaims = ev.subclaims

        ctx.send_event(RetrieveEntityEvent(claim=claim))
        ctx.send_event(RetrieveTripletsEvent(subclaims=subclaims))

    @step
    async def retrieve_entity(self, ctx: Context, ev: RetrieveEntityEvent) -> RetrievalAggregateEvent:
        claim = ev.claim

        # Extract entity from claim
        messages = [
            ChatMessage(role="system", content=NER_SYSTEM),
            ChatMessage(role="user", content=NER_EXAMPLE_INPUT),
            ChatMessage(role="assistant", content=NER_EXAMPLE_OUTPUT),
            ChatMessage(role="user", content=NER_USER.format(content=claim)),
        ]
        response = await self.llm.achat(messages)
        raw = response.message.content or ""
        raw = fix_broken_generated_json(raw)
        entities = _parse_ner_response(raw)

        entity_nodes = self.graph_store.get(ids=entities)
        doc_ids = [entity_node.properties["doc_id"] for entity_node in entity_nodes]
        doc_ids = list(set(doc_ids))

        return RetrievalAggregateEvent(doc_ids=doc_ids)

    @step
    async def retrieve_triplets(self, ctx: Context, ev: RetrieveTripletsEvent) -> RetrievalAggregateEvent:
        subclaims = ev.subclaims

        results = []
        for subclaim in subclaims:
            nodes = self.vector_retriever.retrieve(subclaim)
            results.extend(nodes)

        doc_ids = [result.node.metadata['doc_id'] for result in results]
        doc_ids = list(set(doc_ids))

        return RetrievalAggregateEvent(doc_ids=doc_ids)

    @step
    async def finalize(self, ctx: Context, ev: RetrievalAggregateEvent) -> RetrievalStopEvent:
        ready = ctx.collect_events(ev, [RetrievalAggregateEvent] * 2)

        if not ready:
            return None

        doc_ids = []
        for r in ready:
            doc_ids.extend(r.doc_ids)
        doc_ids = list(set(doc_ids))

        return RetrievalStopEvent(doc_ids=doc_ids)