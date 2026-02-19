from workflows import Workflow, step
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import ChatMessage

from ...events.graph_check.construct_graph import (
    ConstructGraphStartEvent,
    ParseGraphEvent,
    ConstructGraphStopEvent
)
from src.modules.prompts.graph_check.construct_graph import GRAPH_CONSTRUCT_USER
from src.modules.schema.graph_check.graph import Graph


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
    def __init__(self,
                 llm: OpenAI,
                 **kwargs):
        super().__init__(**kwargs)
        self.llm = llm

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

        return ParseGraphEvent(
            content=content
        )

    @step
    async def parse_graph(self, ev: ParseGraphEvent) -> ConstructGraphStopEvent:
        content = ev.content
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
            if line.startswith("# Latent Entities"):
                continue
            if line.startswith("# Triples"):
                flag = 1
                continue
            if not line.startswith("(ENT"):
                flag = 1

            if flag == 0:
                first_section.append(line)
            elif flag == 1:
                second_section.append(line)

        def_triplets = []
        for idx, line in enumerate(first_section.copy()):
            expected_prefix = f"(ENT{idx + 1}) [SEP] is [SEP]"
            if line.startswith(expected_prefix):
                def_triplets.append(line)
                first_section.remove(line)

        triplets = first_section + second_section

        return ConstructGraphStopEvent(
            graph=Graph(
                def_triplets,
                triplets
            )
        )