import re
import logging
from typing import Optional

from workflows import Workflow, step, Context
from llama_index.core import Document
from llama_index.core.llms import LLM
from llama_index.core.prompts import ChatMessage
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.indices import SummaryIndex
from llama_index.retrievers.bm25 import BM25Retriever

from src.modules.graph_check.graph import Graph
from src.modules.datasets.feverous.database.feverous_db import FeverousDB
from src.modules.datasets.feverous.utils.wiki_page import WikiPage
from ...events.graph_check.infilling import (
    InfillingStartEvent,
    InfillingLoopInitialize,
    RetrieveEvidenceEvent,
    InfillQueryEvent,
    InfillEvent,
    HandleLoopInfo,
    InfillingStopEvent
)
from ...events.graph_check.context import SynthesisContext
from .debug_utils import log_retrieved_nodes
from .trace_sink import (
    GraphCheckTraceSink,
    NullGraphCheckTrace,
    nodes_to_previews,
    trace_uses_logger_fallback,
)

logger = logging.getLogger(__name__)

# Same cap as ``TracedGraphCheck.retrieve_evidence`` before infill prompt (then [:3000] in prompt)
_INFILL_RETRIEVAL_EVIDENCE_CAP = 4000


def build_infilling_retrieval_query(graph: Graph, target_la_ent: str) -> str:
    """Match graphcheck.GraphCheck.get_infilling_retrieval_query (sentence-based)."""
    sub_graph = [f"{triplet.sentence}." for triplet in graph.la_ent_2_sub_triples[target_la_ent]]
    query = " ".join(
        [
            triple_sent
            for triple_sent in sub_graph
            if set(re.findall(r"\(ENT\d+\)", triple_sent)) == {target_la_ent}
        ]
    )
    if query == "":
        query = f"{graph.la_ent_2_def_triple[target_la_ent].sentence}."
    while re.search(r"\(ENT\d+\)", query):
        for la_ent, definition in graph.la_ent_2_def.items():
            query = query.replace(la_ent, definition)
        if graph.has_la_ent_w_no_def == 1:
            break
    return query


def build_infilling_query(graph: Graph, target_la_ent: str) -> str:
    """Match graphcheck.GraphCheck.get_infilling_query (sentence-based, <extra_id_0> placeholder)."""
    sub_graph = [f"{triple.sentence}." for triple in graph.la_ent_2_sub_triples[target_la_ent]]
    sub_graph.append(f"{graph.la_ent_2_def_triple[target_la_ent].sentence}.")
    infilling_query = " ".join(
        [
            triple_sent
            for triple_sent in sub_graph
            if set(re.findall(r"\(ENT\d+\)", triple_sent)) == {target_la_ent}
        ]
    )
    if infilling_query == "":
        infilling_query = f"{graph.la_ent_2_def_triple[target_la_ent].sentence}."
    variable_name = "<extra_id_0>"
    infilling_query = infilling_query.strip().replace(target_la_ent, variable_name)
    while re.search(r"\(ENT\d+\)", infilling_query):
        for la_ent, definition in graph.la_ent_2_def.items():
            infilling_query = infilling_query.replace(la_ent, definition)
        if graph.has_la_ent_w_no_def == 1:
            break
    return infilling_query


def build_feverous_retriever(document_path: str, similarity_top_k: int = 10) -> BM25Retriever:
    """Build a BM25 retriever for Feverous dataset."""
    db = FeverousDB(document_path)
    doc_ids = db.get_doc_ids()

    documents = []
    for doc_id in doc_ids:
        page_json = db.get_doc_json(doc_id)
        wiki_page = WikiPage(doc_id, page_json)
        document = Document(text=str(wiki_page))
        documents.append(document)

    index = SummaryIndex(nodes=documents)
    retriever = BM25Retriever.from_defaults(index, similarity_top_k=similarity_top_k)

    return retriever


def build_exfever_retriever(db_path: str, similarity_top_k: int = 10) -> BaseRetriever:
    """Build an ExFever retriever; db_path is kept for compatibility."""
    from src.modules.retrievers.exfever import build_exfever_retriever as build_retriever
    return build_retriever(db_path, similarity_top_k)


class InfillingWorkflow(Workflow):
    def __init__(self,
                 llm: LLM,
                 retriever: Optional[BaseRetriever] = None,
                 document_path: str = None,
                 dataset_type: str = "feverous",
                 similarity_top_k: int = 10,
                 trace: GraphCheckTraceSink | None = None,
                 **kwargs):
        """
        Initialize infilling workflow.
        
        Args:
            llm: Language model for entity infilling
            retriever: Pre-built retriever (optional)
            document_path: Path to wiki database
            dataset_type: "feverous" or "exfever"
            similarity_top_k: Number of documents to retrieve
        """
        super().__init__(**kwargs)
        self.llm = llm
        
        if retriever is not None:
            self.retriever = retriever
        elif document_path is not None:
            if dataset_type == "exfever":
                self.retriever = build_exfever_retriever(document_path, similarity_top_k)
            else:
                self.retriever = build_feverous_retriever(document_path, similarity_top_k)
        else:
            raise ValueError("Either retriever or document_path must be provided")
        self.trace: GraphCheckTraceSink = trace or NullGraphCheckTrace()
        self._trace_uses_logger_fallback = trace_uses_logger_fallback(self.trace)

    @step
    async def initialize(
            self, ctx: Context[SynthesisContext], ev: InfillingStartEvent
    ) -> InfillingLoopInitialize:
        path = ev.path
        graph = ev.graph

        infilled_def_triple_texts = [def_triple.triplet_text for def_triple in graph.def_triples]
        infilled_triplets_texts = [triple.triplet_text for triple in graph.triples]

        async with ctx.store.edit_state() as ctx_state:
            ctx_state.infilling_index = 0
            ctx_state.infilling_log = []
            ctx_state.infilled_def_triplets_texts = infilled_def_triple_texts
            ctx_state.infilled_triplets_texts = infilled_triplets_texts
            ctx_state.path = path
            ctx_state.path_index_for_trace = ev.path_index
            ctx_state.graph = graph
            ctx_state.claim = ev.claim

        return InfillingLoopInitialize()

    @step
    async def loop_init(
            self, ctx: Context[SynthesisContext], ev: InfillingLoopInitialize
    ) -> InfillingStopEvent | RetrieveEvidenceEvent:
        infilling_index = await ctx.store.get("infilling_index")
        path = await ctx.store.get("path")
        num_loops = len(path)

        if infilling_index >= num_loops:
            infilled_def = await ctx.store.get("infilled_def_triplets_texts")
            infilled_tri = await ctx.store.get("infilled_triplets_texts")
            # Match graphcheck.GraphCheck.infill_graph return: final graph from full infilled lists.
            return InfillingStopEvent(
                graph=Graph(infilled_def, infilled_tri),
                infilling_log=await ctx.store.get("infilling_log"),
            )

        async with ctx.store.edit_state() as ctx_state:
            latent_entity = path[infilling_index]
            ctx_state.current_latent_entity = latent_entity

        graph: Graph = await ctx.store.get("graph")
        latent_entity = await ctx.store.get("current_latent_entity")
        query = build_infilling_retrieval_query(graph, latent_entity)
        if self._trace_uses_logger_fallback:
            logger.debug("[infill] %s  retrieval_query: %s", latent_entity, query)
        return RetrieveEvidenceEvent(query=query)

    @step
    async def make_infilling_query(
            self, ctx: Context[SynthesisContext], ev: RetrieveEvidenceEvent
    ) -> InfillQueryEvent:
        """Build infilling query for the current latent entity (see graphcheck.get_infilling_query)."""
        graph: Graph = await ctx.store.get("graph")
        latent_entity = await ctx.store.get("current_latent_entity")
        infilling_query = build_infilling_query(graph, latent_entity)
        if self._trace_uses_logger_fallback:
            logger.debug("[infill] %s  infill_query: %s", latent_entity, infilling_query)
        return InfillQueryEvent(infill_query=infilling_query, retrieval_query=ev.query)

    @step
    async def retrieve_evidence(
            self, ctx: Context[SynthesisContext], ev: InfillQueryEvent
    ) -> InfillEvent:
        try:
            nodes = self.retriever.retrieve(ev.retrieval_query)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            nodes = []
        evidence = "\n".join([node.text for node in nodes])
        if len(evidence) > _INFILL_RETRIEVAL_EVIDENCE_CAP:
            evidence = evidence[:_INFILL_RETRIEVAL_EVIDENCE_CAP]
        if self._trace_uses_logger_fallback:
            log_retrieved_nodes(logger, "[infill] top-k docs", nodes)

        path = await ctx.store.get("path")
        infilling_index = await ctx.store.get("infilling_index")
        path_idx = await ctx.store.get("path_index_for_trace")
        latent_entity = await ctx.store.get("current_latent_entity")
        doc_previews = (
            [] if self._trace_uses_logger_fallback else nodes_to_previews(nodes)
        )
        self.trace.infill_retrieval_and_query(
            int(path_idx),
            int(infilling_index) + 1,
            len(path),
            str(latent_entity),
            ev.retrieval_query or "",
            doc_previews,
            ev.infill_query or "",
        )

        return InfillEvent(
            infill_query=ev.infill_query,
            retrieval_query=ev.retrieval_query,
            evidence=evidence,
        )

    @step
    async def infill(self, ctx: Context[SynthesisContext], ev: InfillEvent) -> HandleLoopInfo:
        query = ev.infill_query
        evidence = ev.evidence

        if not query:
            raise ValueError("No infilling query available")

        # Match ``OpenAIBaseModel.infill``: ``if not evidence`` uses no-evidence prompt (only ``""`` is falsy)
        if evidence:
            evidence_truncated = evidence[:3000]
            prompt = ChatMessage(
                content=(
                    f"Based on this evidence:\n{evidence_truncated}\n\n"
                    f"Fill in the blank marked with <extra_id_0>: {query}\n"
                    f"Answer with ONLY the entity name, nothing else:"
                ),
                role="user",
            )
        else:
            prompt = ChatMessage(
                content=(
                    f"Fill in the blank marked with <extra_id_0>: {query}\n"
                    f"Answer with ONLY the entity name, nothing else:"
                ),
                role="user",
            )
        try:
            response = await self.llm.achat([prompt])
            answer = response.message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            answer = ""

        if answer is None:
            answer = ""

        path_idx = await ctx.store.get("path_index_for_trace")
        infilling_index = await ctx.store.get("infilling_index")
        current_latent_entity = await ctx.store.get("current_latent_entity")
        self.trace.infill_llm_answer(
            int(path_idx),
            int(infilling_index),
            str(current_latent_entity),
            answer,
        )

        return HandleLoopInfo(infill=answer, query=query)

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

        answer = answer.strip()
        if not answer:
            logger.error("No answer for %s, using definition", current_latent_entity)
            answer = graph.la_ent_2_def[current_latent_entity]
        else:
            answer = answer.split("\n")[0].strip()
            for prefix in ["blank is ", "the answer is ", "the ", "a ", "answer: ", "entity: "]:
                if answer.lower().startswith(prefix):
                    answer = answer[len(prefix):].strip()
                    break

        infilled_def_triplets_texts = [
            text.replace(current_latent_entity, answer) for text in infilled_def_triplets_texts
        ]
        infilled_triplets_texts = [
            text.replace(current_latent_entity, answer) for text in infilled_triplets_texts
        ]

        remained = [
            sent
            for sent in infilled_def_triplets_texts
            if sent.split() and re.search(r"\(ENT\d+\)", sent.split()[0])
        ]
        if remained:
            graph = Graph(remained, infilled_triplets_texts)

        infilling_log = {
            "index": index,
            "target": current_latent_entity,
            "query": ev.query,
            "answer": answer
        }

        async with ctx.store.edit_state() as ctx_state:
            if remained:
                ctx_state.graph = graph
            ctx_state.infilling_log.append(infilling_log)
            ctx_state.infilled_def_triplets_texts = infilled_def_triplets_texts
            ctx_state.infilled_triplets_texts = infilled_triplets_texts
            ctx_state.infilling_index += 1

        return InfillingLoopInitialize()