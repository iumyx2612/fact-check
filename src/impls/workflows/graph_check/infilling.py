import re
from typing import Optional

from workflows import Workflow, step, Context
from llama_index.core import Document
from llama_index.core.llms import LLM
from llama_index.core.prompts import ChatMessage
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.indices import SummaryIndex
from llama_index.retrievers.bm25 import BM25Retriever

from src.modules.schema.graph_check.graph import Graph
from src.modules.datasets.feverous.database.feverous_db import FeverousDB
from src.modules.datasets.feverous.utils.wiki_page import WikiPage
from ...events.graph_check.infilling import (
    InfillingStartEvent,
    InfillingLoopInitialize,
    MakeInfillingRetrievalQuery,
    RetrieveEvidenceEvent,
    MakeInfillingQuery,
    InfillEvent,
    HandleLoopInfo,
    InfillingStopEvent
)
from ...events.graph_check.context import SynthesisContext


def build_retriever(document_path: str) -> BM25Retriever:
    db = FeverousDB(document_path)
    doc_ids = db.get_doc_ids()

    documents = []
    for doc_id in doc_ids:
        page_json = db.get_doc_json(doc_id)
        wiki_page = WikiPage(doc_id, page_json)
        document = Document(text=str(wiki_page))
        documents.append(document)

    index = SummaryIndex(nodes=documents)
    retriever = BM25Retriever.from_defaults(index, similarity_top_k=10)

    return retriever


class InfillingWorkflow(Workflow):
    def __init__(self,
                 llm: LLM,
                 retriever: Optional[BaseRetriever] = None,
                 document_path: str = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        if sum(bool(val) for val in [document_path, retriever]) != 1:
            raise ValueError("Please pass exactly one of document_path or retriever.")

        if document_path:
            retriever = build_retriever(document_path)

        self.retriever = retriever

    @step
    async def initialize(
            self, ctx: Context[SynthesisContext], ev: InfillingStartEvent
    ) -> InfillingLoopInitialize:
        path = ev.path
        graph: Graph = await ctx.store.get("graph")

        infilled_def_triple_texts = [def_triple.triplet_text for def_triple in graph.def_triples]
        infilled_triple_texts = [triple.triplet_text for triple in graph.triples]

        async with ctx.store.edit_state() as ctx_state:
            ctx_state.infilled_def_triplets_texts = infilled_def_triple_texts
            ctx_state.infilled_triple_texts = infilled_triple_texts
            ctx_state.path = path

        return InfillingLoopInitialize()

    @step
    async def loop_init(
            self, ctx: Context[SynthesisContext], ev: InfillingLoopInitialize
    ) -> MakeInfillingRetrievalQuery | MakeInfillingQuery | InfillingStopEvent:
        infilling_index = await ctx.store.get("infilling_index")
        path = await ctx.store.get("path")
        num_loops = len(path)

        if infilling_index >= num_loops:
            return InfillingStopEvent(
                graph=await ctx.store.get("graph")
            )

        async with ctx.store.edit_state() as ctx_state:
            latent_entity = path[infilling_index]
            ctx_state.current_latent_entity = latent_entity

        ctx.send_event(MakeInfillingQuery())
        ctx.send_event(MakeInfillingRetrievalQuery())

    @step
    async def make_retrieval_query(
            self, ctx: Context[SynthesisContext], ev: MakeInfillingRetrievalQuery
    ) -> RetrieveEvidenceEvent:
        """
        Construct a retrieval query for latent entity infilling.

        The retrieval query is formed by concatenating all triples (except the definition triple)
        that include the target latent entity and exclude any unidentified latent entities.
        Each latent entity in the query is replaced with its corresponding reference mapped in the definition triples,
        forming a nearly complete sentence.

        ----------
        Example:

        [Graph]
        # Latent Entities:
        (ENT1) [SEP] is [SEP] a musician
        (ENT2) [SEP] is [SEP] a band
        # Triples:
        (ENT1) [SEP] is part of [SEP] Tall Birds
        (ENT1) [SEP] is a percussionist for [SEP] (ENT2)
        (ENT2) [SEP] formed in [SEP] Issaquah, Washington

        [Infilling retrieval query for (ENT1)]
        a musician is part of Tall Birds.
        """
        graph: Graph = await ctx.store.get("graph")
        latent_entity = await ctx.store.get("current_latent_entity")

        sub_graph = [f"{triplet.sentence}." for triplet in graph.la_ent_2_sub_triples[latent_entity]]
        query = " ".join(
            [triple_sent for triple_sent in sub_graph if set(re.findall(r"\(ENT\d+\)", triple_sent)) == {latent_entity}]
        )

        # Handle edge case where no relevant triples exist
        if query == "":
            query = f"{graph.la_ent_2_def_triple[latent_entity].sentence}."

        while re.search(r"\(ENT\d+\)", query):
            for la_ent, definition in graph.la_ent_2_def.items():
                query = query.replace(la_ent, definition)
            if graph.has_la_ent_w_no_def == 1:  # Edge case
                break

        return RetrieveEvidenceEvent(query=query)

    @step
    async def make_infilling_query(
            self, ctx: Context[SynthesisContext], ev: MakeInfillingQuery
    ) -> InfillEvent:
        """
        Construct an infilling query for latent entity infilling.

        The infilling query is formed by concatenating all triples that include the target latent entity exclude any other unidentified latent entities.
        The target latent entity is replaced with the special token to indicate that it should be infilled.

        ----------
        Example:

        [Graph]
        # Latent Entities:
        (ENT1) [SEP] is [SEP] a musician
        (ENT2) [SEP] is [SEP] a band
        # Triples:
        (ENT1) [SEP] is part of [SEP] Tall Birds
        (ENT1) [SEP] is a percussionist for [SEP] (ENT2)
        (ENT2) [SEP] formed in [SEP] Issaquah, Washington

        [Infilling query for (ENT1)]
        <extra_id_0> is part of Tall Birds. <extra_id_0> is a musician.
        """
        graph: Graph = await ctx.store.get("graph")
        latent_entity = await ctx.store.get("current_latent_entity")

        sub_graph = [f"{triple.sentence}." for triple in graph.la_ent_2_sub_triples[latent_entity]]
        sub_graph.append(f"{graph.la_ent_2_def_triple[latent_entity].sentence}.")

        query = " ".join(
            [triple_sent for triple_sent in sub_graph if set(re.findall(r"\(ENT\d+\)", triple_sent)) == {latent_entity}]
        )

        # Handle edge case where no relevant triples exist
        if query == "":
            query = f"{graph.la_ent_2_def_triple[latent_entity].sentence}."

        variable_name = "<extra_id_0>"
        query = query.strip().replace(latent_entity, variable_name)

        while re.search(r"\(ENT\d+\)", query):
            for la_ent, definition in graph.la_ent_2_def.items():
                query = query.replace(la_ent, definition)
            if graph.has_la_ent_w_no_def == 1:  # Edge case
                break

        return InfillEvent(infill_query=query)

    @step
    async def retrieve_evidence(
            self, ev: RetrieveEvidenceEvent
    ) -> InfillEvent:
        nodes = self.retriever.retrieve(ev.query)

        evidence = "\n".join([node.text for node in nodes])

        return InfillEvent(evidence=evidence)

    @step
    async def infill(self, ctx: Context[SynthesisContext], ev: InfillEvent) -> HandleLoopInfo:
        ready = ctx.collect_events(ev, [InfillEvent] * 2)
        if ready is None:
            return None

        query = ready[0].infill_query if ready[0].infill_query else ready[1].infill_query
        evidence = ready[0].evidence if ready[0].evidence else ready[1].evidence

        prompt = ChatMessage(
            content=f"{evidence}\nBased on the above information, fill in the blank "
                    f"with the correct entity: {query}\nAnswer:",
            role="user"
        )
        response = await self.llm.achat([prompt])
        answer = response.message.content

        if answer.lower().startswith("blank is "):
            answer = answer[len("blank is "):]

        return HandleLoopInfo(infill=answer, qeury=query)

    @step
    async def handle_loop_info(
            self, ctx: Context[SynthesisContext], ev: HandleLoopInfo
    ) -> InfillingLoopInitialize:
        answer = ev.infill
        index = await ctx.store.get("infilling_index")
        graph = await ctx.store.get("graph")
        infilled_def_triplets_texts = await ctx.store.get("infilled_def_triplets_texts")
        infilled_triplets_texts = await ctx.store.get("infilled_triplets_texts")
        current_latent_entity = await ctx.store.get("current_latent_entity")

        if not answer.strip():
            answer = graph.la_ent_2_def[current_latent_entity]
        else:
            answer = answer.split("\n")[0].strip()

        infilled_def_triplets_texts = [
            text.replace(current_latent_entity, answer) for text in infilled_def_triplets_texts
        ]
        infilled_triplets_texts = [
            text.replace(current_latent_entity, answer) for text in infilled_triplets_texts
        ]
        remained_def_triplet_texts = [
            text for text in infilled_def_triplets_texts if re.search(r"\(ENT\d+\)", text.split()[0])
        ]

        if remained_def_triplet_texts:
            graph = Graph(remained_def_triplet_texts, infilled_triplets_texts)
        infilling_log = {
            "infilling_index": index,
            "target_latent_entity": current_latent_entity,
            "infilling_query": ev.query,
            "infilling_answer": answer
        }

        # Update context
        async with ctx.store.edit_state() as ctx_state:
            ctx_state.graph = graph
            ctx_state.infilling_log.append(infilling_log)
            ctx_state.infilled_def_triplets_texts = infilled_def_triplets_texts
            ctx_state.infilled_triplets_texts = infilled_triplets_texts
            ctx_state.infilling_index += 1

        return InfillingLoopInitialize()