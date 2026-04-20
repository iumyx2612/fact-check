import logging

from workflows import Workflow, step
from llama_index.core.llms import LLM
from llama_index.core.prompts import ChatMessage

from ...events.graph_check.construct_graph import (
    ConstructGraphStartEvent,
    ParseGraphEvent,
    ConstructGraphStopEvent
)
from src.modules.prompts.graph_check.construct_graph import GRAPH_CONSTRUCT_USER
from src.modules.graph_check.graph import Graph

from .trace_sink import GraphCheckTraceSink, NullGraphCheckTrace, trace_uses_logger_fallback

logger = logging.getLogger(__name__)


def parse_constructed_graph_text(content: str) -> tuple[list[str], list[str]]:
    """Parse LLM graph output; matches graphcheck ConstructModel.parse_graph."""

    first_section, second_section = [], []
    flag = 0

    lines = [line.strip() for line in content.split("\n")]
    for line in lines:
        if not line:
            continue
        if "no latent entities identified" in line.lower():
            continue
        if "(no latent entities needed)" in line.lower():
            continue
        if line.lower().strip() == "none":
            continue
        # Skip section headers (with or without # prefix)
        if line.startswith("# Latent Entities") or line.lower() == "latent entities:":
            continue
        if line.startswith("# Triples") or line.lower() == "triples:":
            flag = 1
            continue
        if not line.startswith("(ENT"):
            flag = 1

        if flag == 0:
            first_section.append(line)
        elif flag == 1:
            second_section.append(line)

    def_triplets: list[str] = []
    for idx, line in enumerate(first_section.copy()):
        expected_prefix = f"(ENT{idx + 1}) [SEP] is [SEP]"
        if line.startswith(expected_prefix):
            def_triplets.append(line)
            first_section.remove(line)

    triplets = first_section + second_section
    return def_triplets, triplets


class GraphConstructWorkflow(Workflow):
    """
    This workflow construct graph based on input claim
    It:
    - Detect latent entities using placeholders (ENT1), (ENT2)
    - Decompose the claim into fact triplets (subject [SEP] relation [SEP] object)
    The generated graph contains 2 sections:
    - # Latent Entities: Triplets that link latent entities to their implicit references in the claim
    - # Triplets: Triplets that capture relationships between entities
    """
    def __init__(
        self,
        llm: LLM,
        trace: GraphCheckTraceSink | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm = llm
        self.trace: GraphCheckTraceSink = trace or NullGraphCheckTrace()
        self._trace_uses_logger_fallback = trace_uses_logger_fallback(self.trace)

    @step
    async def get_response(
            self, start_ev: ConstructGraphStartEvent
    ) -> ParseGraphEvent:
        prompt = ChatMessage(
            content=GRAPH_CONSTRUCT_USER.replace(
                "<<target_claim>>", start_ev.claim
            ),
            role="user"
        )
        response = await self.llm.achat([prompt])
        content = response.message.content

        if content is None:
            raise ValueError("LLM returned None content for graph construction")

        return ParseGraphEvent(
            content=content
        )

    @step
    async def parse_graph(self, ev: ParseGraphEvent) -> ConstructGraphStopEvent:
        raw = ev.content
        self.trace.construct_raw_llm(raw)

        def_triplets, triplets = parse_constructed_graph_text(raw)
        self.trace.construct_parsed(def_triplets, triplets)
        g = Graph(def_triplets, triplets)
        if self._trace_uses_logger_fallback:
            logger.debug(
                "[construct] graph: %s latent entities %s, %s triples",
                g.num_la_ent,
                g.la_ent_list,
                len(g.total_triples),
            )

        return ConstructGraphStopEvent(graph=g)